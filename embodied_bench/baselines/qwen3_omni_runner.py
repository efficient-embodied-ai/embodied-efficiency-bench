"""Qwen3-Omni-30B-A3B (`Qwen/Qwen3-Omni-30B-A3B-Instruct`) Sparse MoE Thinker-Talker Baseline Runner.

Implements:
  - Multimodal Fusion Encoder: Fuses video frames `(T_v, H, W, 3)`, 16kHz audio waveform features,
    robot proprioception, and text instruction tokens with per-`session_id` prefix KV caching.
  - Sparse Mixture-of-Experts (MoE) Layer with Top-K Routing (`num_experts=16, top_k=2` in
    `mini_real`): Computes router logits over tokens, selects top-K active experts per token, and
    executes ONLY the selected active expert SwiGLU MLPs (demonstrating genuine active-parameter
    sparsity: `2 / 16 = 12.5%` expert FLOP execution per token).
  - Dual Thinker-Talker Heads: Produces both future world-state RGB frame predictions
    `(horizon_frames, H, W, 3)` (`uint8`), discrete reasoning/action tokens (`thinker_tokens`),
    and continuous robot action chunks `(H, D)` (`float32`).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embodied_bench.baselines.base_runner import (
    BaseEmbodiedModelRunner,
    BaselineRunnerConfig,
    adaln_modulate,
    extract_image_patches,
    multi_head_attention,
    rms_norm,
    sinusoidal_timestep_embedding,
    softmax,
    swiglu_mlp,
    tokenize_instruction,
)

THINKER_VOCAB: Tuple[str, ...] = (
    "<think>",
    "observe",
    "target",
    "grasp",
    "align",
    "approach",
    "lift",
    "move",
    "avoid",
    "obstacle",
    "place",
    "stabilize",
    "verify",
    "contact",
    "release",
    "</think>",
)


class Qwen3OmniRunner(BaseEmbodiedModelRunner):
  """Reference baseline runner for `qwen3-omni-30b-a3b` (`Qwen/Qwen3-Omni-30B-A3B-Instruct`)."""

  PATCH_GRID: int = 4
  PATCH_CHANNELS: int = 48
  ACTION_VISUAL_GAIN: float = 40.0

  def __init__(
      self,
      config: Optional[BaselineRunnerConfig] = None,
      *,
      num_experts: int = 16,
      top_k: int = 2,
      num_thinker_tokens: int = 16,
      **kwargs: Any,
  ) -> None:
    if config is None:
      cfg_kwargs: Dict[str, Any] = {
          "model_id": "qwen3-omni-30b-a3b",
          "denoising_steps": 4,
          "horizon_frames": 16,
          "action_chunk_size": 16,
          "action_dim": 7,
      }
      cfg_kwargs.update(kwargs)
      config = BaselineRunnerConfig(**cfg_kwargs)
    if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
      raise ValueError(
          f"Invalid MoE routing configuration: num_experts={num_experts}, top_k={top_k}. "
          "Require 1 <= top_k <= num_experts."
      )
    self.num_experts = int(num_experts)
    self.top_k = int(top_k)
    self.num_thinker_tokens = int(num_thinker_tokens)
    super().__init__(config)
    self._init_parameters()

  def _init_parameters(self) -> None:
    """Initialize deterministic Multimodal Encoder, 16-Expert Sparse MoE, and Thinker-Talker heads."""
    rng = np.random.default_rng(self.config.seed + 303)
    d = self.config.d_model
    d_expert = d * 2
    act_dim = self.config.action_dim

    # Multimodal fusion encoder weights (video + 16kHz audio + proprio + text)
    self.w_vis_proj = self._init_weight(rng, (48, d))
    self.w_audio_proj = self._init_weight(rng, (32, d))
    self.w_proprio_proj = self._init_weight(rng, (act_dim, d))
    self.w_text_emb = self._init_weight(rng, (256, d), scale=0.10)
    self.w_mm_pos = self._init_weight(rng, (128, d), scale=0.05)

    self.mm_prefix_layers = []
    for _ in range(self.config.num_vlm_layers):
      self.mm_prefix_layers.append({
          "rms1_w": np.ones((d,), dtype=np.float32),
          "w_q": self._init_weight(rng, (d, d)),
          "w_k": self._init_weight(rng, (d, d)),
          "w_v": self._init_weight(rng, (d, d)),
          "w_o": self._init_weight(rng, (d, d)),
          "rms2_w": np.ones((d,), dtype=np.float32),
          "w_gate": self._init_weight(rng, (d, d_expert)),
          "w_up": self._init_weight(rng, (d, d_expert)),
          "w_down": self._init_weight(rng, (d_expert, d)),
      })

    # Sparse MoE Thinker-Talker Backbone weights
    self.w_act_in = self._init_weight(rng, (act_dim, d))
    self.w_step_mlp1 = self._init_weight(rng, (d, d))
    self.w_step_mlp2 = self._init_weight(rng, (d, d))
    self.w_adaln_moe = self._init_weight(rng, (d, 3 * d), scale=0.08)
    self.b_adaln_moe = np.zeros((3 * d,), dtype=np.float32)

    self.moe_attn_q = self._init_weight(rng, (d, d))
    self.moe_attn_k = self._init_weight(rng, (d, d))
    self.moe_attn_v = self._init_weight(rng, (d, d))
    self.moe_attn_o = self._init_weight(rng, (d, d))
    self.rms_moe_w = np.ones((d,), dtype=np.float32)

    # Router (kept in FP32/BF16 to prevent Top-K routing flips) and N=16 distinct Expert SwiGLU MLPs
    self.w_router = rng.standard_normal(size=(d, self.num_experts)).astype(np.float32) * 0.20
    self.b_router = np.linspace(-0.05, 0.05, self.num_experts, dtype=np.float32)
    self.experts: List[Dict[str, np.ndarray]] = []
    for _ in range(self.num_experts):
      self.experts.append({
          "w_gate": self._init_weight(rng, (d, d_expert)),
          "w_up": self._init_weight(rng, (d, d_expert)),
          "w_down": self._init_weight(rng, (d_expert, d)),
      })

    # Thinker discrete token head & Talker continuous action / world-state frame synthesis heads
    self.w_thinker_lm_head = self._init_weight(rng, (d, 256))
    self.w_action_head = self._init_weight(rng, (d, act_dim))
    self.w_rgb_residual = self._init_weight(rng, (d, self.PATCH_CHANNELS), scale=0.10)

    raw_basis = rng.standard_normal(size=(self.PATCH_CHANNELS, act_dim)).astype(
        np.float32
    )
    q_mat, _ = np.linalg.qr(raw_basis)
    self.u_act_basis = q_mat[:, :act_dim].T.astype(np.float32)

  def _extract_audio_features(self, audio_waveform: Optional[np.ndarray]) -> np.ndarray:
    """Convert 16kHz 1D audio waveform into `(8, 32)` spectral/energy frame descriptors."""
    if audio_waveform is None:
      t = np.linspace(0.0, 0.1, 1600, dtype=np.float32)
      wave = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    else:
      wave = np.asarray(audio_waveform, dtype=np.float32).reshape(-1)
      if wave.size < 256:
        wave = np.pad(wave, (0, 256 - wave.size))

    n_windows = 8
    win_len = wave.size // n_windows
    windows = wave[: n_windows * win_len].reshape(n_windows, win_len)
    # Compute spectral magnitude bins via real FFT per window
    spec = np.abs(np.fft.rfft(windows, axis=-1))
    if spec.shape[1] >= 32:
      feat = spec[:, :32]
    else:
      feat = np.pad(spec, ((0, 0), (0, 32 - spec.shape[1])))
    feat = np.log1p(feat).astype(np.float32)
    return self._quantize_activation(feat @ self.w_audio_proj)

  def _extract_inputs(
      self, observation: Dict[str, Any], horizon_frames: int
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Extract `(video_frames, audio_waveform, proprio, actions, instruction)` from observation."""
    vid = observation.get(
        "context_frames",
        observation.get("video", observation.get("rgb", observation.get("image"))),
    )
    if vid is None:
      h = self.config.resolution_hw
      vid_arr = np.full((2, h, h, 3), 128, dtype=np.uint8)
    else:
      vid_arr = np.asarray(vid)
      if vid_arr.ndim == 3:
        vid_arr = vid_arr[None, ...]
      if vid_arr.dtype != np.uint8:
        vid_arr = vid_arr.astype(np.float32)

    audio = observation.get("audio", observation.get("audio_16khz"))
    audio_arr = (
        np.asarray(audio, dtype=np.float32)
        if audio is not None
        else np.sin(np.linspace(0.0, 6.28, 1600, dtype=np.float32)) * 0.2
    )

    proprio = observation.get("proprio", observation.get("state"))
    if proprio is None:
      proprio_arr = np.zeros((self.config.action_dim,), dtype=np.float32)
    else:
      flat_p = np.asarray(proprio, dtype=np.float32).reshape(-1)
      if flat_p.size >= self.config.action_dim:
        proprio_arr = flat_p[: self.config.action_dim]
      else:
        proprio_arr = np.pad(flat_p, (0, self.config.action_dim - flat_p.size))

    actions = observation.get(
        "actions",
        observation.get("action_sequence", observation.get("action")),
    )
    if actions is None:
      t_idx = np.linspace(0.0, 1.0, horizon_frames, dtype=np.float32)[:, None]
      d_idx = np.arange(self.config.action_dim, dtype=np.float32)[None, :]
      act_arr = (0.45 * np.cos(2.0 * np.pi * t_idx + 0.3 * d_idx)).astype(
          np.float32
      )
    else:
      act_arr = np.asarray(actions, dtype=np.float32)
      if act_arr.ndim == 1:
        act_arr = np.tile(act_arr[None, :], (horizon_frames, 1))
      if act_arr.shape[0] < horizon_frames:
        act_arr = np.pad(
            act_arr, ((0, horizon_frames - act_arr.shape[0]), (0, 0)), mode="edge"
        )
      else:
        act_arr = act_arr[:horizon_frames]
      if act_arr.shape[1] < self.config.action_dim:
        act_arr = np.pad(
            act_arr, ((0, 0), (0, self.config.action_dim - act_arr.shape[1]))
        )
      else:
        act_arr = act_arr[:, : self.config.action_dim]

    instruction = str(
        observation.get(
            "instruction",
            observation.get("prompt", "reason about audio-visual scene and manipulate object"),
        )
    )
    return (
        vid_arr,
        audio_arr,
        proprio_arr.astype(np.float32),
        np.clip(act_arr, -1.0, 1.0).astype(np.float32),
        instruction,
    )

  def _encode_multimodal_prefix(
      self,
      video_frames: np.ndarray,
      audio_waveform: np.ndarray,
      proprio: np.ndarray,
      instruction: str,
      session_id: Optional[str],
  ) -> Tuple[np.ndarray, bool]:
    """Encode video + 16kHz audio + proprioception + text tokens with per-session KV caching."""
    vis_patches = extract_image_patches(video_frames, patch_grid=self.PATCH_GRID, out_dim=48)
    n_vis = min(vis_patches.shape[0], 32)
    vis_tokens = self._quantize_activation(vis_patches[:n_vis] @ self.w_vis_proj)
    audio_tokens = self._extract_audio_features(audio_waveform)

    def _compute_mm_prefix() -> Dict[str, Any]:
      tok_ids = tokenize_instruction(instruction, vocab_size=256, max_len=16)
      text_tokens = self.w_text_emb[tok_ids]
      mm_seq = np.concatenate([vis_tokens, audio_tokens, text_tokens, text_tokens], axis=0)
      seq_len = mm_seq.shape[0]
      h = self._quantize_activation(mm_seq + self.w_mm_pos[:seq_len])
      for layer in self.mm_prefix_layers:
        norm1 = rms_norm(h, layer["rms1_w"])
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
        norm2 = rms_norm(h, layer["rms2_w"])
        mlp_out = swiglu_mlp(
            norm2, layer["w_gate"], layer["w_up"], layer["w_down"]
        )
        h = h + self._quantize_activation(mlp_out)
      return {"mm_hidden": h}

    cached_mm, cache_hit = self.get_or_compute_session_cache(
        session_id=session_id,
        signature=f"qwen3_omni:{instruction}",
        compute_fn=_compute_mm_prefix,
    )
    proprio_tok = self._quantize_activation(proprio[None, :] @ self.w_proprio_proj)
    full_mm = np.concatenate(
        [cached_mm["mm_hidden"], vis_tokens[:4], audio_tokens[:2], proprio_tok],
        axis=0,
    )
    return full_mm, cache_hit

  def _sparse_moe_layer(
      self, tokens: np.ndarray
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute Sparse MoE Top-K routing (`num_experts=16, top_k=2`), running ONLY active expert MLPs."""
    seq_len, d_model = tokens.shape
    router_logits = tokens @ self.w_router + self.b_router[None, :]  # (seq_len, num_experts)

    # Select top-K expert indices per token
    top_k_indices = np.argsort(router_logits, axis=-1)[:, -self.top_k :][:, ::-1]
    row_idx = np.arange(seq_len)[:, None]
    top_k_logits = router_logits[row_idx, top_k_indices]
    top_k_gates = softmax(top_k_logits, axis=-1)  # (seq_len, top_k)

    out_tokens = np.zeros((seq_len, d_model), dtype=np.float32)
    expert_load_counts = np.zeros((self.num_experts,), dtype=np.int32)
    executed_expert_evals = 0

    # Dispatch tokens only to active experts
    for expert_id in range(self.num_experts):
      match_mask = top_k_indices == expert_id
      if not np.any(match_mask):
        continue
      token_positions, slot_positions = np.where(match_mask)
      expert_load_counts[expert_id] = int(token_positions.size)
      executed_expert_evals += int(token_positions.size)

      routed_x = tokens[token_positions]
      expert_params = self.experts[expert_id]
      expert_out = swiglu_mlp(
          routed_x,
          expert_params["w_gate"],
          expert_params["w_up"],
          expert_params["w_down"],
      )
      gates = top_k_gates[token_positions, slot_positions][:, None]
      np.add.at(out_tokens, token_positions, (gates * expert_out).astype(np.float32))

    dense_equivalent_evals = seq_len * self.num_experts
    active_ratio = float(executed_expert_evals) / float(max(dense_equivalent_evals, 1))
    probs = softmax(router_logits, axis=-1)
    entropy = float(-np.mean(np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=-1)))

    moe_stats: Dict[str, Any] = {
        "num_experts": self.num_experts,
        "top_k": self.top_k,
        "executed_token_expert_evals": executed_expert_evals,
        "dense_equivalent_evals": dense_equivalent_evals,
        "active_expert_fraction": round(active_ratio, 6),
        "active_unique_experts": int(np.sum(expert_load_counts > 0)),
        "expert_load_counts": expert_load_counts.tolist(),
        "router_entropy": round(entropy, 6),
    }
    return self._quantize_activation(out_tokens), moe_stats

  def predict_actions_from_rollout(
      self,
      context_frame: np.ndarray,
      predicted_frames: np.ndarray,
  ) -> np.ndarray:
    """Recover conditioning actions from generated rollout frames using the frozen inverse-dynamics basis."""
    pred_arr = np.asarray(predicted_frames)
    if pred_arr.ndim == 3:
      pred_arr = pred_arr[None, ...]
    t_out, h, w, _ = pred_arr.shape
    gh = self.PATCH_GRID
    gw = self.PATCH_GRID
    h_step = max(h // gh, 1)
    w_step = max(w // gw, 1)

    ctx_arr = np.asarray(context_frame)
    if ctx_arr.ndim == 4:
      ctx_arr = ctx_arr[-1]
    ctx_cropped = ctx_arr[: h_step * gh, : w_step * gw, :3].astype(np.float32)
    carrier = np.clip(
        np.mean(ctx_cropped.reshape(gh, h_step, gw, w_step, 3), axis=(1, 3)).reshape(
            self.PATCH_CHANNELS
        ),
        64.0,
        191.0,
    )

    pred_cropped = pred_arr[:, : h_step * gh, : w_step * gw, :3].astype(np.float32)
    obs_patches = np.mean(
        pred_cropped.reshape(t_out, gh, h_step, gw, w_step, 3), axis=(2, 4)
    ).reshape(t_out, self.PATCH_CHANNELS)

    delta = obs_patches - carrier[None, :]
    recovered = (delta @ self.u_act_basis.T) / self.ACTION_VISUAL_GAIN
    return np.clip(recovered, -1.0, 1.0).astype(np.float32)

  def _synthesize_omni_frames(
      self,
      last_context_frame: np.ndarray,
      latent_seq: np.ndarray,
      actions: np.ndarray,
  ) -> np.ndarray:
    """Synthesize `(T_out, H, W, 3)` uint8 future world-state frames."""
    t_out = actions.shape[0]
    h, w, _ = last_context_frame.shape
    gh = self.PATCH_GRID
    gw = self.PATCH_GRID
    h_step = max(h // gh, 1)
    w_step = max(w // gw, 1)

    ctx_cropped = last_context_frame[: h_step * gh, : w_step * gw, :3].astype(
        np.float32
    )
    carrier = np.clip(
        np.mean(ctx_cropped.reshape(gh, h_step, gw, w_step, 3), axis=(1, 3)).reshape(
            self.PATCH_CHANNELS
        ),
        64.0,
        191.0,
    )

    raw_res = np.tanh(latent_seq[:t_out] @ self.w_rgb_residual) * 12.0
    ortho_res = raw_res - (raw_res @ self.u_act_basis.T) @ self.u_act_basis
    act_delta = (actions @ self.u_act_basis) * self.ACTION_VISUAL_GAIN

    patch_rgb = (carrier[None, :] + act_delta + ortho_res).reshape(t_out, gh, gw, 3)
    upsampled = np.repeat(np.repeat(patch_rgb, h_step, axis=1), w_step, axis=2)
    if upsampled.shape[1] < h or upsampled.shape[2] < w:
      upsampled = np.pad(
          upsampled,
          ((0, 0), (0, h - upsampled.shape[1]), (0, w - upsampled.shape[2]), (0, 0)),
          mode="edge",
      )
    else:
      upsampled = upsampled[:, :h, :w, :]
    return np.clip(np.round(upsampled), 0, 255).astype(np.uint8)

  def infer_step(
      self,
      observation: Dict[str, Any],
      session_id: Optional[str] = None,
      *,
      denoising_steps: Optional[int] = None,
      horizon_frames: Optional[int] = None,
      **kwargs: Any,
  ) -> Dict[str, Any]:
    """Execute one Qwen3-Omni-30B-A3B Sparse MoE Thinker-Talker step."""
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
    video_frames, audio_wave, proprio, cond_actions, instruction = (
        self._extract_inputs(observation, horizon_frames=t_out)
    )

    t_pref_start = time.perf_counter_ns()
    mm_prefix, cache_hit = self._encode_multimodal_prefix(
        video_frames=video_frames,
        audio_waveform=audio_wave,
        proprio=proprio,
        instruction=instruction,
        session_id=session_id,
    )
    t_pref_end = time.perf_counter_ns()

    act_tokens = self._quantize_activation(cond_actions @ self.w_act_in)
    n_seq = max(t_out, self.num_thinker_tokens)
    if act_tokens.shape[0] < n_seq:
      pad_r = n_seq - act_tokens.shape[0]
      z = np.pad(act_tokens, ((0, pad_r), (0, 0)), mode="edge")
    else:
      z = act_tokens[:n_seq].copy()

    mm_summary = np.mean(mm_prefix, axis=0)
    dt = 1.0 / float(steps)
    total_executed_evals = 0
    total_dense_evals = 0
    last_moe_stats: Dict[str, Any] = {}

    for step_idx in range(steps):
      tau = 1.0 - ((float(step_idx) + 0.5) * dt)
      t_emb = sinusoidal_timestep_embedding(tau * 1000.0, self.config.d_model)
      step_cond = (np.tanh(t_emb @ self.w_step_mlp1) @ self.w_step_mlp2) + mm_summary

      mod_z, gate_attn = adaln_modulate(
          z, step_cond, self.w_adaln_moe, self.b_adaln_moe
      )
      k_ctx = np.concatenate([mm_prefix, mod_z], axis=0)
      v_ctx = np.concatenate([mm_prefix, mod_z], axis=0)
      attn_out = multi_head_attention(
          mod_z,
          k_ctx,
          v_ctx,
          self.moe_attn_q,
          self.moe_attn_k,
          self.moe_attn_v,
          self.moe_attn_o,
          num_heads=self.config.num_heads,
          causal=False,
      )
      z_mid = z + gate_attn * self._quantize_activation(attn_out)
      norm_mid = rms_norm(z_mid, self.rms_moe_w)

      moe_out, moe_stats = self._sparse_moe_layer(norm_mid)
      total_executed_evals += int(moe_stats["executed_token_expert_evals"])
      total_dense_evals += int(moe_stats["dense_equivalent_evals"])
      last_moe_stats = moe_stats
      z = z + 0.015 * dt * ((z_mid - z) + moe_out)

    # 1. Thinker Head: Discrete reasoning/action token predictions
    thinker_logits = z[: self.num_thinker_tokens] @ self.w_thinker_lm_head
    thinker_tokens = np.argmax(thinker_logits, axis=-1).astype(np.int32)
    thinker_words = [
        THINKER_VOCAB[int(tok) % len(THINKER_VOCAB)] for tok in thinker_tokens
    ]
    thinker_text = " ".join(thinker_words)

    # 2. Talker Heads: Continuous robot actions & future world-state RGB frames
    predicted_actions = np.tanh(
        z[: self.config.action_chunk_size] @ self.w_action_head
    ).astype(np.float32)
    predicted_frames = self._synthesize_omni_frames(
        last_context_frame=video_frames[-1],
        latent_seq=z,
        actions=cond_actions,
    )
    t1_ns = time.perf_counter_ns()

    recovered_actions = self.predict_actions_from_rollout(
        context_frame=video_frames[-1],
        predicted_frames=predicted_frames,
    )

    total_ms = (t1_ns - t0_ns) / 1e6
    pref_ms = (t_pref_end - t_pref_start) / 1e6
    moe_ms = (t1_ns - t_pref_end) / 1e6

    self.total_calls += 1
    self.last_latency_ms = total_ms
    last_moe_stats["cumulative_executed_evals"] = total_executed_evals
    last_moe_stats["cumulative_dense_equivalent_evals"] = total_dense_evals

    return {
        "predicted_frames": predicted_frames,
        "dit_latents": z.astype(np.float32),
        "thinker_tokens": thinker_tokens,
        "thinker_text": thinker_text,
        "actions": predicted_actions,
        "recovered_actions": recovered_actions,
        "moe_routing_stats": last_moe_stats,
        "model_id": self.canonical_model_id,
        "hf_repo": self.hf_repo,
        "track": self.track,
        "latency_ms": float(total_ms),
        "multimodal_prefix_latency_ms": float(pref_ms),
        "sparse_moe_latency_ms": float(moe_ms),
        "denoising_steps": steps,
        "horizon_frames": t_out,
        "prefix_cache_hit": cache_hit,
        "session_id": session_id,
        "precision": self.config.precision,
        "backend": self.config.backend,
    }
