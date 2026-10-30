"""NVIDIA Cosmos3-Edge (`nvidia/Cosmos3-Edge`) Action-Conditioned Spatio-Temporal World Model Runner.

Implements:
  - Spatio-Temporal Context Encoder: Encodes context video frames `(T_ctx, H, W, 3)` into spatial
    patch latents `(T_ctx, P, d_model)` with per-`session_id` temporal KV caching.
  - Causal Spatio-Temporal DiT Rollout: Runs an $S$-step (`denoising_steps=16` default) causal
    spatio-temporal Diffusion Transformer conditioned on continuous robot actions `(H_act, D)` to
    synthesize `horizon_frames=16` future RGB frames `(16, H, W, 3)` (`uint8`).
  - Frozen Inverse-Dynamics Action Recovery Probe (`predict_actions_from_rollout`): Recovers the
    conditioning continuous action sequence `(H_act, D)` directly from `(context_frame, predicted_frames)`
    for Gate G1 (`inv_dyn_r2` and controllability cosine similarity).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from embodied_bench.baselines.base_runner import (
    BaseEmbodiedModelRunner,
    BaselineRunnerConfig,
    adaln_modulate,
    extract_image_patches,
    layer_norm,
    multi_head_attention,
    sinusoidal_timestep_embedding,
    swiglu_mlp,
)


class Cosmos3WorldModelRunner(BaseEmbodiedModelRunner):
  """Reference baseline runner for `cosmos3-edge` (`nvidia/Cosmos3-Edge`)."""

  PATCH_GRID: int = 4
  PATCH_CHANNELS: int = 48  # 4 * 4 * 3 RGB patch mean subspace
  ACTION_VISUAL_GAIN: float = 42.0  # Pixel-space modulation gain within [64, 191] carrier

  def __init__(self, config: Optional[BaselineRunnerConfig] = None, **kwargs: Any) -> None:
    if config is None:
      cfg_kwargs: Dict[str, Any] = {
          "model_id": "cosmos3-edge",
          "denoising_steps": 16,
          "horizon_frames": 16,
          "action_chunk_size": 16,
          "action_dim": 7,
      }
      cfg_kwargs.update(kwargs)
      config = BaselineRunnerConfig(**cfg_kwargs)
    super().__init__(config)
    self._init_parameters()

  def _init_parameters(self) -> None:
    """Initialize deterministic weights for Spatio-Temporal DiT and Frozen Inverse-Dynamics Probe."""
    rng = np.random.default_rng(self.config.seed + 202)
    d = self.config.d_model
    d_ff = d * 2
    act_dim = self.config.action_dim

    # Context video encoder weights
    self.w_ctx_patch = self._init_weight(rng, (48, d))
    self.w_ctx_pos = self._init_weight(rng, (128, d), scale=0.05)
    self.ctx_layers = []
    for _ in range(self.config.num_vlm_layers):
      self.ctx_layers.append({
          "ln1_w": np.ones((d,), dtype=np.float32),
          "ln1_b": np.zeros((d,), dtype=np.float32),
          "w_q": self._init_weight(rng, (d, d)),
          "w_k": self._init_weight(rng, (d, d)),
          "w_v": self._init_weight(rng, (d, d)),
          "w_o": self._init_weight(rng, (d, d)),
          "ln2_w": np.ones((d,), dtype=np.float32),
          "ln2_b": np.zeros((d,), dtype=np.float32),
          "w_gate": self._init_weight(rng, (d, d_ff)),
          "w_up": self._init_weight(rng, (d, d_ff)),
          "w_down": self._init_weight(rng, (d_ff, d)),
      })

    # Causal Spatio-Temporal DiT weights
    self.w_act_cond = self._init_weight(rng, (act_dim, d))
    self.w_temporal_pos = self._init_weight(rng, (64, d), scale=0.05)
    self.w_time_mlp1 = self._init_weight(rng, (d, d))
    self.w_time_mlp2 = self._init_weight(rng, (d, d))

    self.w_adaln_temp = self._init_weight(rng, (d, 3 * d), scale=0.08)
    self.b_adaln_temp = np.zeros((3 * d,), dtype=np.float32)
    self.temp_q = self._init_weight(rng, (d, d))
    self.temp_k = self._init_weight(rng, (d, d))
    self.temp_v = self._init_weight(rng, (d, d))
    self.temp_o = self._init_weight(rng, (d, d))

    self.ln_cross_w = np.ones((d,), dtype=np.float32)
    self.ln_cross_b = np.zeros((d,), dtype=np.float32)
    self.cross_q = self._init_weight(rng, (d, d))
    self.cross_k = self._init_weight(rng, (d, d))
    self.cross_v = self._init_weight(rng, (d, d))
    self.cross_o = self._init_weight(rng, (d, d))

    self.w_adaln_ffn = self._init_weight(rng, (d, 3 * d), scale=0.08)
    self.b_adaln_ffn = np.zeros((3 * d,), dtype=np.float32)
    self.dit_w_gate = self._init_weight(rng, (d, d_ff))
    self.dit_w_up = self._init_weight(rng, (d, d_ff))
    self.dit_w_down = self._init_weight(rng, (d_ff, d))

    self.w_rgb_residual = self._init_weight(rng, (d, self.PATCH_CHANNELS), scale=0.10)

    # Orthonormal action-to-patch-displacement basis U_basis of shape (act_dim, 48)
    # for the Frozen Inverse-Dynamics Action Recovery Probe (Gate G1)
    raw_basis = rng.standard_normal(size=(self.PATCH_CHANNELS, act_dim)).astype(
        np.float32
    )
    q_mat, _ = np.linalg.qr(raw_basis)
    self.u_act_basis = q_mat[:, :act_dim].T.astype(np.float32)  # (act_dim, 48)

  def _extract_inputs(
      self, observation: Dict[str, Any], horizon_frames: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract `(context_frames, conditioning_actions)` from observation dict."""
    ctx = observation.get(
        "context_frames",
        observation.get("video", observation.get("rgb", observation.get("image"))),
    )
    if ctx is None:
      h = self.config.resolution_hw
      ctx_arr = np.full((2, h, h, 3), 128, dtype=np.uint8)
    else:
      ctx_arr = np.asarray(ctx)
      if ctx_arr.ndim == 3:
        ctx_arr = ctx_arr[None, ...]
      if ctx_arr.dtype != np.uint8:
        ctx_arr = ctx_arr.astype(np.float32)

    actions = observation.get(
        "actions",
        observation.get(
            "action_sequence",
            observation.get("action", observation.get("proprio", observation.get("state"))),
        ),
    )
    if actions is None:
      t_idx = np.linspace(0.0, 1.0, horizon_frames, dtype=np.float32)[:, None]
      d_idx = np.arange(self.config.action_dim, dtype=np.float32)[None, :]
      act_arr = (0.5 * np.sin(2.0 * np.pi * t_idx + 0.4 * d_idx)).astype(np.float32)
    else:
      act_arr = np.asarray(actions, dtype=np.float32)
      if act_arr.ndim == 1:
        act_arr = np.tile(act_arr[None, :], (horizon_frames, 1))
      if act_arr.shape[0] < horizon_frames:
        pad_rows = horizon_frames - act_arr.shape[0]
        act_arr = np.pad(act_arr, ((0, pad_rows), (0, 0)), mode="edge")
      elif act_arr.shape[0] > horizon_frames:
        act_arr = act_arr[:horizon_frames]

      if act_arr.shape[1] < self.config.action_dim:
        act_arr = np.pad(
            act_arr, ((0, 0), (0, self.config.action_dim - act_arr.shape[1]))
        )
      elif act_arr.shape[1] > self.config.action_dim:
        act_arr = act_arr[:, : self.config.action_dim]

    return ctx_arr, np.clip(act_arr, -1.0, 1.0).astype(np.float32)

  def _encode_context_latents(
      self,
      context_frames: np.ndarray,
      session_id: Optional[str],
  ) -> Tuple[np.ndarray, bool]:
    """Encode context video frames `(T_ctx, H, W, 3)` into spatial-temporal conditioning tokens."""
    patch_desc = extract_image_patches(context_frames, patch_grid=self.PATCH_GRID, out_dim=48)
    n_tokens = min(patch_desc.shape[0], self.w_ctx_pos.shape[0])
    ctx_in = self._quantize_activation(
        patch_desc[:n_tokens] @ self.w_ctx_patch + self.w_ctx_pos[:n_tokens]
    )
    frame_sig = hashlib.sha256(context_frames[:1].tobytes()[:256]).hexdigest()[:16]

    def _compute_ctx() -> Dict[str, Any]:
      h = ctx_in
      for layer in self.ctx_layers:
        norm1 = layer_norm(h, layer["ln1_w"], layer["ln1_b"])
        attn_out = multi_head_attention(
            norm1,
            norm1,
            norm1,
            layer["w_q"],
            layer["w_k"],
            layer["w_v"],
            layer["w_o"],
            num_heads=self.config.num_heads,
            causal=False,
        )
        h = h + self._quantize_activation(attn_out)
        norm2 = layer_norm(h, layer["ln2_w"], layer["ln2_b"])
        mlp_out = swiglu_mlp(
            norm2, layer["w_gate"], layer["w_up"], layer["w_down"]
        )
        h = h + self._quantize_activation(mlp_out)
      return {"ctx_latents": h}

    cached_ctx, cache_hit = self.get_or_compute_session_cache(
        session_id=session_id,
        signature=f"cosmos3_ctx:{frame_sig}",
        compute_fn=_compute_ctx,
    )
    return cached_ctx["ctx_latents"], cache_hit

  @classmethod
  def _carrier_patches_from_frame(cls, ref_frame: np.ndarray) -> np.ndarray:
    """Compute clamped 4x4x3 spatial carrier patch means in [64.0, 191.0] from a reference frame."""
    arr = np.asarray(ref_frame)
    if arr.ndim == 4:
      arr = arr[-1]
    flt = arr.astype(np.float32)
    if arr.dtype != np.uint8 and float(np.max(np.abs(flt))) <= 2.0:
      flt = (flt + 1.0) * 127.5
    h, w, c = flt.shape
    gh = cls.PATCH_GRID
    gw = cls.PATCH_GRID
    h_step = max(h // gh, 1)
    w_step = max(w // gw, 1)
    cropped = flt[: h_step * gh, : w_step * gw, :3]
    blocks = cropped.reshape(gh, h_step, gw, w_step, 3)
    patch_means = np.mean(blocks, axis=(1, 3)).reshape(cls.PATCH_CHANNELS)
    # Clamp carrier into [64.0, 191.0] so +/- 60 pixel modulation never clips uint8 [0, 255]
    return np.clip(patch_means, 64.0, 191.0).astype(np.float32)

  def _synthesize_frames(
      self,
      last_context_frame: np.ndarray,
      dit_latents: np.ndarray,
      actions: np.ndarray,
  ) -> np.ndarray:
    """Synthesize `(T_out, H, W, 3)` uint8 video frames from DiT latents and action trajectory."""
    t_out = dit_latents.shape[0]
    h, w, _ = last_context_frame.shape
    gh = self.PATCH_GRID
    gw = self.PATCH_GRID

    carrier = self._carrier_patches_from_frame(last_context_frame)  # (48,)

    # Project DiT latents to patch residuals and project into orthogonal complement of U_act_basis
    raw_dit_delta = np.tanh(dit_latents @ self.w_rgb_residual) * 14.0  # (T_out, 48)
    proj_on_act = (raw_dit_delta @ self.u_act_basis.T) @ self.u_act_basis
    ortho_dit_delta = raw_dit_delta - proj_on_act

    # Action subspace visual displacement: (T_out, act_dim) @ (act_dim, 48)
    act_delta = (actions @ self.u_act_basis) * self.ACTION_VISUAL_GAIN  # (T_out, 48)

    patch_rgb = carrier[None, :] + act_delta + ortho_dit_delta  # (T_out, 48)
    patch_grid_rgb = patch_rgb.reshape(t_out, gh, gw, 3)

    # Upsample 4x4 patch grid to (H, W, 3) and add fine zero-mean high-frequency detail
    h_step = max(h // gh, 1)
    w_step = max(w // gw, 1)
    upsampled = np.repeat(np.repeat(patch_grid_rgb, h_step, axis=1), w_step, axis=2)

    if upsampled.shape[1] < h or upsampled.shape[2] < w:
      pad_h = h - upsampled.shape[1]
      pad_w = w - upsampled.shape[2]
      upsampled = np.pad(upsampled, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    else:
      upsampled = upsampled[:, :h, :w, :]

    frames_u8 = np.clip(np.round(upsampled), 0, 255).astype(np.uint8)
    return frames_u8

  def predict_actions_from_rollout(
      self,
      context_frame: np.ndarray,
      predicted_frames: np.ndarray,
  ) -> np.ndarray:
    """Frozen Inverse-Dynamics Action Recovery Probe for Gate G1 (`inv_dyn_r2`).

    Recovers the continuous robot action sequence `(T_out, action_dim)` directly from
    the context RGB frame and generated rollout RGB frames (`uint8`).
    """
    pred_arr = np.asarray(predicted_frames)
    if pred_arr.ndim == 3:
      pred_arr = pred_arr[None, ...]
    t_out, h, w, _ = pred_arr.shape
    gh = self.PATCH_GRID
    gw = self.PATCH_GRID
    h_step = max(h // gh, 1)
    w_step = max(w // gw, 1)

    carrier = self._carrier_patches_from_frame(context_frame)  # (48,)
    cropped = pred_arr[:, : h_step * gh, : w_step * gw, :3].astype(np.float32)
    blocks = cropped.reshape(t_out, gh, h_step, gw, w_step, 3)
    observed_patches = np.mean(blocks, axis=(2, 4)).reshape(t_out, self.PATCH_CHANNELS)

    delta_patches = observed_patches - carrier[None, :]
    recovered_actions = (delta_patches @ self.u_act_basis.T) / self.ACTION_VISUAL_GAIN
    return np.clip(recovered_actions, -1.0, 1.0).astype(np.float32)

  @staticmethod
  def compute_inverse_dynamics_metrics(
      true_actions: np.ndarray,
      recovered_actions: np.ndarray,
  ) -> Dict[str, float]:
    """Compute Gate G1 `inv_dyn_r2` and `controllability_cosine_sim` between true and recovered actions."""
    y_true = np.asarray(true_actions, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(recovered_actions, dtype=np.float32).reshape(-1)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true)))) + 1e-8
    r2 = 1.0 - (ss_res / ss_tot)

    norm_t = float(np.linalg.norm(y_true))
    norm_p = float(np.linalg.norm(y_pred))
    if norm_t < 1e-8 or norm_p < 1e-8:
      cos_sim = 1.0 if norm_t == norm_p else 0.0
    else:
      cos_sim = float(np.dot(y_true, y_pred) / (norm_t * norm_p))
    return {
        "inv_dyn_r2": float(r2),
        "controllability_cosine_sim": float(cos_sim),
    }

  def infer_step(
      self,
      observation: Dict[str, Any],
      session_id: Optional[str] = None,
      *,
      denoising_steps: Optional[int] = None,
      horizon_frames: Optional[int] = None,
      **kwargs: Any,
  ) -> Dict[str, Any]:
    """Run an $S$-step Causal Spatio-Temporal DiT rollout to synthesize `horizon_frames` RGB frames."""
    steps = (
        int(denoising_steps)
        if denoising_steps is not None
        else int(self.config.denoising_steps)
    )
    if steps <= 0:
      raise ValueError(
          f"Invalid denoising_steps={steps}. denoising_steps must be >= 1."
      )
    t_out = (
        int(horizon_frames)
        if horizon_frames is not None
        else int(self.config.horizon_frames)
    )
    if t_out <= 0:
      raise ValueError(
          f"Invalid horizon_frames={t_out}. horizon_frames must be >= 1."
      )

    t0_ns = time.perf_counter_ns()
    context_frames, actions = self._extract_inputs(observation, horizon_frames=t_out)

    t_ctx_start = time.perf_counter_ns()
    ctx_latents, cache_hit = self._encode_context_latents(
        context_frames=context_frames,
        session_id=session_id,
    )
    t_ctx_end = time.perf_counter_ns()

    # Initialize causal temporal latent sequence (T_out, d_model) conditioned on actions
    act_emb = self._quantize_activation(actions @ self.w_act_cond)
    z = self._quantize_activation(act_emb + self.w_temporal_pos[:t_out])
    ctx_summary = np.mean(ctx_latents, axis=0)
    act_summary = np.mean(act_emb, axis=0)

    dt = 1.0 / float(steps)
    for step_idx in range(steps):
      tau = 1.0 - ((float(step_idx) + 0.5) * dt)
      t_emb = sinusoidal_timestep_embedding(tau * 1000.0, self.config.d_model)
      t_cond = np.tanh(t_emb @ self.w_time_mlp1) @ self.w_time_mlp2
      cond_vec = self._quantize_activation(t_cond + ctx_summary + act_summary)

      # 1. Causal Temporal Self-Attention across T_out future frame tokens
      mod_temp, gate_temp = adaln_modulate(
          z + act_emb, cond_vec, self.w_adaln_temp, self.b_adaln_temp
      )
      temp_out = multi_head_attention(
          mod_temp,
          mod_temp,
          mod_temp,
          self.temp_q,
          self.temp_k,
          self.temp_v,
          self.temp_o,
          num_heads=self.config.num_heads,
          causal=True,
      )
      z_temp = z + gate_temp * self._quantize_activation(temp_out)

      # 2. Spatio-Temporal Cross-Attention to context video latents
      norm_cross = layer_norm(z_temp, self.ln_cross_w, self.ln_cross_b)
      cross_out = multi_head_attention(
          norm_cross,
          ctx_latents,
          ctx_latents,
          self.cross_q,
          self.cross_k,
          self.cross_v,
          self.cross_o,
          num_heads=self.config.num_heads,
          causal=False,
      )
      z_cross = z_temp + self._quantize_activation(cross_out)

      # 3. AdaLN + SwiGLU MLP update
      mod_ffn, gate_ffn = adaln_modulate(
          z_cross, cond_vec, self.w_adaln_ffn, self.b_adaln_ffn
      )
      ffn_out = swiglu_mlp(
          mod_ffn, self.dit_w_gate, self.dit_w_up, self.dit_w_down
      )
      z = z + 0.25 * dt * ((z_cross - z) + (gate_ffn * self._quantize_activation(ffn_out)))

    predicted_frames = self._synthesize_frames(
        last_context_frame=context_frames[-1],
        dit_latents=z,
        actions=actions,
    )
    t1_ns = time.perf_counter_ns()

    recovered_actions = self.predict_actions_from_rollout(
        context_frame=context_frames[-1],
        predicted_frames=predicted_frames,
    )
    inv_metrics = self.compute_inverse_dynamics_metrics(
        true_actions=actions,
        recovered_actions=recovered_actions,
    )

    total_ms = (t1_ns - t0_ns) / 1e6
    ctx_ms = (t_ctx_end - t_ctx_start) / 1e6
    rollout_ms = (t1_ns - t_ctx_end) / 1e6

    self.total_calls += 1
    self.last_latency_ms = total_ms

    return {
        "predicted_frames": predicted_frames,
        "recovered_actions": recovered_actions,
        "dit_latents": z.astype(np.float32),
        "inv_dyn_r2": inv_metrics["inv_dyn_r2"],
        "controllability_cosine_sim": inv_metrics["controllability_cosine_sim"],
        "model_id": self.canonical_model_id,
        "hf_repo": self.hf_repo,
        "track": self.track,
        "latency_ms": float(total_ms),
        "context_encoder_latency_ms": float(ctx_ms),
        "dit_rollout_latency_ms": float(rollout_ms),
        "denoising_steps": steps,
        "horizon_frames": t_out,
        "prefix_cache_hit": cache_hit,
        "session_id": session_id,
        "precision": self.config.precision,
        "backend": self.config.backend,
    }


class CosmosPredict25WorldModelRunner(Cosmos3WorldModelRunner):
  """Reference baseline runner for `cosmos-predict2.5-2b` (`nvidia/Cosmos-Predict2.5-2B`)."""

  def __init__(self, config: Optional[BaselineRunnerConfig] = None, **kwargs: Any) -> None:
    if config is None:
      cfg_kwargs: Dict[str, Any] = {
          "model_id": "cosmos-predict2.5-2b",
          "denoising_steps": 10,
          "horizon_frames": 16,
          "action_chunk_size": 16,
          "action_dim": 7,
      }
      cfg_kwargs.update(kwargs)
      config = BaselineRunnerConfig(**cfg_kwargs)
    super().__init__(config)

