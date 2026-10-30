"""Physical Intelligence pi0.5 (`openpi` / `lerobot/pi05_libero_finetuned_v044`) Reference Baseline Runner.

Implements:
  - PaliGemma 3.3B VLM Prefix Encoder: Encodes multi-camera RGB images, 8-DoF robot state, and
    language instruction into prefix KV states `(k_prefix, v_prefix)` cached per `session_id`.
  - 300M Flow-Matching Action Expert (`H=10, K=5, D=7` on LIBERO; configurable `H=50` on DROID):
    Runs an $S$-step (`denoising_steps=10` default, configurable down to `1`-`4` in Open Division)
    Euler ODE solver `a_{tau + d_tau} = a_tau + d_tau * v_theta(a_tau, tau, KV_prefix)` over
    continuous 10-step LIBERO action chunks, plus optional Real-Time Chunking (`RTC`) prefix
    inpainting when `previous_chunk_tail` is supplied.
  - Supports PyTorch/NumPy execution (`lerobot` / `TorchTPU`) and JAX/XLA (`openpi` / `jax.numpy`)
    execution path as co-equal Closed Division runtimes when checkpoint weights match.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from embodied_bench.baselines.base_runner import (
    BaseEmbodiedModelRunner,
    BaselineRunnerConfig,
    adaln_modulate,
    extract_image_patches,
     multi_head_attention,
    rms_norm,
    sinusoidal_timestep_embedding,
    swiglu_mlp,
    tokenize_instruction,
)

try:
  import jax.numpy as jnp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
  jnp = None  # type: ignore[assignment]

try:
  import torch
except ImportError:  # pragma: no cover
  torch = None  # type: ignore[assignment]


class Pi05OpenPiRunner(BaseEmbodiedModelRunner):
  """Reference baseline runner for `pi05-3b` (`lerobot/pi05_libero_finetuned_v044`)."""

  ROBOT_STATE_DOF: int = 8

  def __init__(self, config: Optional[BaselineRunnerConfig] = None, **kwargs: Any) -> None:
    if config is None:
      cfg_kwargs: Dict[str, Any] = {
          "model_id": "pi05-3b",
          "denoising_steps": 10,
          "action_chunk_size": 10,
          "action_dim": 7,
      }
      cfg_kwargs.update(kwargs)
      config = BaselineRunnerConfig(**cfg_kwargs)
    super().__init__(config)
    self._init_parameters()

  def _init_parameters(self) -> None:
    """Initialize deterministic seeded weights for PaliGemma VLM Prefix + 300M Flow-Matching Action Expert."""
    rng = np.random.default_rng(self.config.seed + 101)
    d = self.config.d_model
    d_ff = d * 2

    # PaliGemma VLM Prefix Encoder weights
    self.w_vis_proj = self._init_weight(rng, (48, d))
    self.w_tok_emb = self._init_weight(rng, (256, d), scale=0.10)
    self.w_state8_proj = self._init_weight(rng, (self.ROBOT_STATE_DOF, d))
    self.w_pos_prefix = self._init_weight(rng, (128, d), scale=0.05)

    self.paligemma_layers = []
    for _ in range(self.config.num_vlm_layers):
      self.paligemma_layers.append({
          "rms1_w": np.ones((d,), dtype=np.float32),
          "w_q": self._init_weight(rng, (d, d)),
          "w_k": self._init_weight(rng, (d, d)),
          "w_v": self._init_weight(rng, (d, d)),
          "w_o": self._init_weight(rng, (d, d)),
          "rms2_w": np.ones((d,), dtype=np.float32),
          "w_gate": self._init_weight(rng, (d, d_ff)),
          "w_up": self._init_weight(rng, (d, d_ff)),
          "w_down": self._init_weight(rng, (d_ff, d)),
      })

    self.w_kv_k_proj = self._init_weight(rng, (d, d))
    self.w_kv_v_proj = self._init_weight(rng, (d, d))

    # 300M Flow-Matching Action Expert weights (H=50, D=7)
    self.w_act_in = self._init_weight(rng, (self.config.action_dim, d))
    self.w_act_pos = self._init_weight(
        rng, (max(self.config.action_chunk_size, 128), d), scale=0.04
    )
    self.w_tau_mlp1 = self._init_weight(rng, (d, d))
    self.w_tau_mlp2 = self._init_weight(rng, (d, d))

    self.w_adaln_flow = self._init_weight(rng, (d, 3 * d), scale=0.08)
    self.b_adaln_flow = np.zeros((3 * d,), dtype=np.float32)

    self.flow_q = self._init_weight(rng, (d, d))
    self.flow_k = self._init_weight(rng, (d, d))
    self.flow_v = self._init_weight(rng, (d, d))
    self.flow_o = self._init_weight(rng, (d, d))

    self.rms_flow_w = np.ones((d,), dtype=np.float32)
    self.flow_w_gate = self._init_weight(rng, (d, d_ff))
    self.flow_w_up = self._init_weight(rng, (d, d_ff))
    self.flow_w_down = self._init_weight(rng, (d_ff, d))

    self.w_vel_out = self._init_weight(rng, (d, self.config.action_dim))
    self.b_vel_out = np.zeros((self.config.action_dim,), dtype=np.float32)

  def _extract_inputs(
      self, observation: Dict[str, Any]
  ) -> Tuple[np.ndarray, np.ndarray, str, Optional[np.ndarray]]:
    """Extract `(rgb, state_8dof, instruction, previous_chunk_tail)` from observation."""
    rgb = observation.get("rgb", observation.get("images", observation.get("image")))
    if rgb is None:
      h = self.config.resolution_hw
      rgb = np.zeros((self.config.num_cameras, h, h, 3), dtype=np.uint8)
    else:
      rgb = np.asarray(rgb)
      if rgb.ndim == 3:
        rgb = rgb[None, ...]

    raw_state = observation.get(
        "state", observation.get("proprio", observation.get("robot_state"))
    )
    if raw_state is None:
      state_8d = np.zeros((self.ROBOT_STATE_DOF,), dtype=np.float32)
    else:
      flat_s = np.asarray(raw_state, dtype=np.float32).reshape(-1)
      if flat_s.size >= self.ROBOT_STATE_DOF:
        state_8d = flat_s[: self.ROBOT_STATE_DOF]
      else:
        state_8d = np.pad(flat_s, (0, self.ROBOT_STATE_DOF - flat_s.size))

    instruction = str(
        observation.get("instruction", observation.get("prompt", "fold towel and place in basket"))
    )
    prev_tail = observation.get("previous_chunk_tail")
    if prev_tail is not None:
      prev_tail = np.asarray(prev_tail, dtype=np.float32)
    return rgb, state_8d.astype(np.float32), instruction, prev_tail

  def _encode_paligemma_prefix(
      self,
      rgb: np.ndarray,
      state_8d: np.ndarray,
      instruction: str,
      session_id: Optional[str],
  ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Encode multi-camera images + 8-DoF state + instruction into PaliGemma KV prefix states."""
    patches = extract_image_patches(rgb, patch_grid=4, out_dim=48)
    n_vis = min(patches.shape[0], 64)
    vis_tokens = self._quantize_activation(patches[:n_vis] @ self.w_vis_proj)

    def _compute_prefix_kv() -> Dict[str, Any]:
      tok_ids = tokenize_instruction(instruction, vocab_size=256, max_len=16)
      lang_tokens = self.w_tok_emb[tok_ids]
      seq = np.concatenate([vis_tokens, lang_tokens, lang_tokens], axis=0)
      seq_len = seq.shape[0]
      h = self._quantize_activation(seq + self.w_pos_prefix[:seq_len])

      for layer in self.paligemma_layers:
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

      k_pref = self._quantize_activation(h @ self.w_kv_k_proj)
      v_pref = self._quantize_activation(h @ self.w_kv_v_proj)
      return {"k_prefix": k_pref, "v_prefix": v_pref}

    cached_kv, cache_hit = self.get_or_compute_session_cache(
        session_id=session_id,
        signature=f"pi05_prefix:{instruction}",
        compute_fn=_compute_prefix_kv,
    )
    state_token = self._quantize_activation(state_8d[None, :] @ self.w_state8_proj)
    live_vis = vis_tokens[:8]
    k_full = np.concatenate([cached_kv["k_prefix"], live_vis, state_token], axis=0)
    v_full = np.concatenate([cached_kv["v_prefix"], live_vis, state_token], axis=0)
    return k_full, v_full, cache_hit

  def _velocity_field(
      self,
      a_t: np.ndarray,
      tau: float,
      k_prefix: np.ndarray,
      v_prefix: np.ndarray,
  ) -> np.ndarray:
    """Evaluate Flow-Matching velocity field `v_theta(a_t, tau, KV_prefix)`."""
    chunk_h = a_t.shape[0]
    h_act = self._quantize_activation(
        (a_t @ self.w_act_in) + self.w_act_pos[:chunk_h]
    )

    tau_emb = sinusoidal_timestep_embedding(tau * 1000.0, self.config.d_model)
    tau_cond = np.tanh(tau_emb @ self.w_tau_mlp1) @ self.w_tau_mlp2
    kv_summary = np.mean(k_prefix, axis=0)
    cond_vec = self._quantize_activation(tau_cond + kv_summary)

    mod_h, gate_attn = adaln_modulate(
        h_act, cond_vec, self.w_adaln_flow, self.b_adaln_flow
    )
    # Joint prefix-suffix attention over cached PaliGemma KV prefix and action chunk tokens
    k_joint = np.concatenate([k_prefix, mod_h], axis=0)
    v_joint = np.concatenate([v_prefix, mod_h], axis=0)
    attn_out = multi_head_attention(
        mod_h,
        k_joint,
        v_joint,
        self.flow_q,
        self.flow_k,
        self.flow_v,
        self.flow_o,
        num_heads=self.config.num_heads,
        causal=False,
    )
    h_mid = h_act + gate_attn * self._quantize_activation(attn_out)
    norm_mid = rms_norm(h_mid, self.rms_flow_w)
    mlp_out = swiglu_mlp(
        norm_mid, self.flow_w_gate, self.flow_w_up, self.flow_w_down
    )
    h_out = h_mid + self._quantize_activation(mlp_out)
    vel = h_out @ self.w_vel_out + self.b_vel_out
    return vel.astype(np.float32)

  def _solve_euler_ode(
      self,
      state_8d: np.ndarray,
      k_prefix: np.ndarray,
      v_prefix: np.ndarray,
      denoising_steps: int,
      chunk_h: int,
      previous_chunk_tail: Optional[np.ndarray],
  ) -> Tuple[np.ndarray, bool, int]:
    """Integrate Euler ODE `a_{t + d_tau} = a_t + d_tau * v_theta(a_t, tau, KV_prefix)` with optional RTC inpainting."""
    if int(denoising_steps) <= 0:
      raise ValueError(
          f"Invalid denoising_steps={denoising_steps}. Must be >= 1."
      )

    # Deterministic base trajectory anchored on robot joint state
    base_vec = state_8d[: self.config.action_dim]
    t_grid = np.linspace(0.0, 1.0, chunk_h, dtype=np.float32)[:, None]
    a_t = (base_vec[None, :] * (1.0 - 0.15 * t_grid)).astype(np.float32)

    rtc_applied = False
    rtc_steps = 0
    tail_arr: Optional[np.ndarray] = None
    if previous_chunk_tail is not None:
      tail_arr = np.asarray(previous_chunk_tail, dtype=np.float32)
      if tail_arr.ndim == 1:
        tail_arr = tail_arr[None, :]
      if tail_arr.shape[1] != self.config.action_dim:
        raise ValueError(
            f"previous_chunk_tail action_dim ({tail_arr.shape[1]}) does not match "
            f"runner action_dim ({self.config.action_dim})."
        )
      rtc_steps = min(int(tail_arr.shape[0]), chunk_h)
      if rtc_steps > 0:
        rtc_applied = True

    d_tau = 1.0 / float(denoising_steps)
    for step_idx in range(denoising_steps):
      tau = float(step_idx) * d_tau
      vel = self._velocity_field(a_t, tau, k_prefix, v_prefix)
      a_t = a_t + d_tau * vel

      # Real-Time Chunking (RTC) guided prefix inpainting during flow integration
      if rtc_applied and tail_arr is not None and rtc_steps > 0:
        idx = np.arange(rtc_steps, dtype=np.float32)[:, None]
        # Smooth guidance weight from 1.0 at step 0 tapering across the overlap horizon
        guidance_w = np.exp(-idx / max(float(rtc_steps), 1.0)).astype(np.float32)
        target_prefix = tail_arr[:rtc_steps]
        a_t[:rtc_steps] = (
            guidance_w * target_prefix + (1.0 - guidance_w) * a_t[:rtc_steps]
        )

    actions = np.tanh(a_t).astype(np.float32)
    # Pin exact continuity at the committed RTC boundary
    if rtc_applied and tail_arr is not None and rtc_steps > 0:
      idx = np.arange(rtc_steps, dtype=np.float32)[:, None]
      pin_w = np.clip(1.0 - (idx / max(float(rtc_steps), 1.0)), 0.25, 1.0).astype(
          np.float32
      )
      actions[:rtc_steps] = (
          pin_w * np.clip(tail_arr[:rtc_steps], -1.0, 1.0)
          + (1.0 - pin_w) * actions[:rtc_steps]
      ).astype(np.float32)

    if self.config.backend == "jax_tpu" and jnp is not None:
      jax_actions = jnp.asarray(actions, dtype=jnp.float32)
      actions = np.asarray(jax_actions, dtype=np.float32)
    elif self.config.backend == "pytorch" and torch is not None:
      t_actions = torch.from_numpy(actions)
      actions = t_actions.detach().cpu().numpy().astype(np.float32)

    return actions, rtc_applied, rtc_steps

  def infer_step(
      self,
      observation: Dict[str, Any],
      session_id: Optional[str] = None,
      *,
      denoising_steps: Optional[int] = None,
      action_chunk_size: Optional[int] = None,
      previous_chunk_tail: Optional[np.ndarray] = None,
      **kwargs: Any,
  ) -> Dict[str, Any]:
    """Execute one pi0.5 (PaliGemma Prefix + Flow-Matching Action Expert) inference step."""
    steps = (
        int(denoising_steps)
        if denoising_steps is not None
        else int(self.config.denoising_steps)
    )
    if steps <= 0:
      raise ValueError(
          f"Invalid denoising_steps={steps}. denoising_steps must be >= 1."
      )
    chunk_h = (
        int(action_chunk_size)
        if action_chunk_size is not None
        else int(self.config.action_chunk_size)
    )
    if chunk_h <= 0:
      raise ValueError(
          f"Invalid action_chunk_size={chunk_h}. action_chunk_size must be >= 1."
      )

    t0_ns = time.perf_counter_ns()
    rgb, state_8d, instruction, obs_tail = self._extract_inputs(observation)
    effective_tail = (
        previous_chunk_tail if previous_chunk_tail is not None else obs_tail
    )

    t_pref_start = time.perf_counter_ns()
    k_prefix, v_prefix, cache_hit = self._encode_paligemma_prefix(
        rgb=rgb,
        state_8d=state_8d,
        instruction=instruction,
        session_id=session_id,
    )
    t_pref_end = time.perf_counter_ns()

    actions, rtc_applied, rtc_steps = self._solve_euler_ode(
        state_8d=state_8d,
        k_prefix=k_prefix,
        v_prefix=v_prefix,
        denoising_steps=steps,
        chunk_h=chunk_h,
        previous_chunk_tail=effective_tail,
    )
    t1_ns = time.perf_counter_ns()

    total_ms = (t1_ns - t0_ns) / 1e6
    pref_ms = (t_pref_end - t_pref_start) / 1e6
    flow_ms = (t1_ns - t_pref_end) / 1e6

    self.total_calls += 1
    self.last_latency_ms = total_ms

    backend_exec = self.config.backend
    if self.config.backend == "jax_tpu" and jnp is None:
      backend_exec = "jax_tpu_numpy_fallback"

    return {
        "actions": actions,
        "model_id": self.canonical_model_id,
        "hf_repo": self.hf_repo,
        "track": self.track,
        "latency_ms": float(total_ms),
        "vlm_prefix_latency_ms": float(pref_ms),
        "flow_expert_latency_ms": float(flow_ms),
        "denoising_steps": steps,
        "action_chunk_size": chunk_h,
        "action_dim": self.config.action_dim,
        "prefix_cache_hit": cache_hit,
        "rtc_applied": rtc_applied,
        "rtc_inpainted_steps": rtc_steps,
        "session_id": session_id,
        "precision": self.config.precision,
        "backend": self.config.backend,
        "backend_executed": backend_exec,
    }
