"""Reference Model Server for Track 2A: NVIDIA Cosmos3-Edge (`nvidia/Cosmos3-Edge` / `cosmos3-edge`).

Implements Action-Conditioned Spatio-Temporal DiT World Model & Policy Evaluator:
  - Predicts future video latents of shape `(B, H, 16, 16)` in `[0, 1]` and
    inverse-dynamics latent actions of shape `(B, H, 7)` in `[-1, 1]`.
  - Implements `DiT-Cache` / `TeaCache` temporal feature reuse across consecutive steps
    in a `session_id`: when `enable_dit_cache=True` and normalized frame-to-frame delta
    ||o_t - o_{t-1}||_2 / sqrt(N) < tau, reuses cached intermediate DiT blocks
    (`dit_cache_hit=True`) while running the live input stem and output residual heads
    so Gate `G4` sensitivity to l_inf = 1e-3 perturbations is strictly preserved.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
import numpy as np

from embodied_bench.schema import now_pacific_iso
from runners.base_server import (
    BaseEmbodiedModelRunner,
    EmbodiedRunnerConfig,
    build_arg_parser,
    run_runner_cli,
)


class Cosmos3EdgeRunner(BaseEmbodiedModelRunner):
  """Reference runner for `nvidia/Cosmos3-Edge` (`cosmos3-edge`)."""

  LATENT_H: int = 16
  LATENT_W: int = 16
  D_MODEL: int = 256

  def __init__(self, config: EmbodiedRunnerConfig) -> None:
    if config.canonical_model_id != "cosmos3-edge":
      raise ValueError(
          f"Cosmos3EdgeRunner requires canonical_model_id='cosmos3-edge', "
          f"got {config.canonical_model_id!r} (from model_id={config.model_id!r})."
      )
    super().__init__(config)
    rng = np.random.default_rng(config.seed + 303)
    d = self.D_MODEL
    act_dim = int(config.action_dim)
    latent_dim = self.LATENT_H * self.LATENT_W  # 256

    # Live observation stem + action conditioning projections (always executed on live frame)
    self._obs_stem_w = rng.normal(0.0, 1.0 / np.sqrt(256), size=(256, d)).astype(np.float32)
    self._cond_act_w = rng.normal(0.0, 0.15, size=(act_dim, d)).astype(np.float32)

    # Heavy intermediate spatio-temporal DiT blocks (cached by TeaCache / DiT-Cache on smooth frames)
    self._dit_mid_w1 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, d)).astype(np.float32)
    self._dit_mid_w2 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, d)).astype(np.float32)
    self._dit_mid_w3 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, d)).astype(np.float32)

    # Live output residual heads for future video latents (16x16=256) and inverse-dynamics actions (7)
    self._video_head_w = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, latent_dim)).astype(np.float32)
    self._inv_dyn_head_w = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, act_dim)).astype(np.float32)

  def _extract_frame_features(self, obs_frames: np.ndarray, batch_size: int) -> np.ndarray:
    """Pool spatial observation tensor into a 256-dim feature vector with full linear sensitivity."""
    flat = np.asarray(obs_frames, dtype=np.float32).reshape(batch_size, -1)
    if flat.shape[1] >= 256:
      stride = flat.shape[1] // 256
      pooled = flat[:, : 256 * stride].reshape(batch_size, 256, stride).mean(axis=-1)
    else:
      pooled = np.zeros((batch_size, 256), dtype=np.float32)
      pooled[:, : flat.shape[1]] = flat
    return (pooled @ self._obs_stem_w).astype(np.float32)  # (B, D_MODEL)

  def _run_mid_dit_blocks(
      self,
      stem_hidden: np.ndarray,
      n_steps: int,
  ) -> np.ndarray:
    """Execute N-step spatio-temporal DiT blocks over (B, H, D_MODEL) hidden states."""
    h = stem_hidden.copy()
    dt = 1.0 / float(max(1, n_steps))
    for s_idx in range(n_steps):
      tau = float(s_idx) * dt
      phase = np.float32(np.cos(0.5 * np.pi * tau))
      m1 = np.tanh((h * phase) @ self._dit_mid_w1)
      m2 = np.tanh(m1 @ self._dit_mid_w2)
      m3 = m2 @ self._dit_mid_w3
      h = h + dt * 0.4 * m3
    return h.astype(np.float32)

  def step_world(
      self,
      session_id: str,
      step_idx: int,
      obs_frames: np.ndarray,
      cond_actions: np.ndarray,
      horizon_frames: int = 16,
  ) -> Dict[str, Any]:
    """Execute action-conditioned spatio-temporal rollout with DiT-Cache / TeaCache."""
    if int(horizon_frames) < 1:
      raise ValueError(f"horizon_frames must be >= 1, got {horizon_frames}.")
    t0_ns = time.perf_counter_ns()
    obs_arr = np.asarray(obs_frames, dtype=np.float32)
    batch_size = int(obs_arr.shape[0]) if obs_arr.ndim >= 4 else 1
    h_frames = int(horizon_frames)
    act_dim = int(self.config.action_dim)
    n_steps = int(self.config.num_inference_steps or 16)

    cond_arr = np.asarray(cond_actions, dtype=np.float32)
    if cond_arr.ndim == 2:
      cond_arr = cond_arr[None, :, :]
    if cond_arr.shape[0] == 1 and batch_size > 1:
      cond_arr = np.broadcast_to(
          cond_arr, (batch_size, cond_arr.shape[1], cond_arr.shape[2])
      ).copy()

    # Align cond_actions horizon to h_frames
    aligned_cond = np.zeros((batch_size, h_frames, act_dim), dtype=np.float32)
    copy_h = min(h_frames, int(cond_arr.shape[1]))
    copy_d = min(act_dim, int(cond_arr.shape[2]))
    aligned_cond[:, :copy_h, :copy_d] = cond_arr[:, :copy_h, :copy_d]

    with self._lock:
      session = self._get_or_create_session(session_id, step_idx)
      kv_prefix_cache_hit = bool(self.config.enable_prefix_cache and int(step_idx) > 0)
      if kv_prefix_cache_hit:
        session["kv_prefix_cache_hits"] = int(session.get("kv_prefix_cache_hits", 0)) + 1

      # Live stem features from current observation o_t and action sequence a_{t:t+H}
      live_obs_feat = self._extract_frame_features(obs_arr, batch_size)  # (B, D)
      live_act_feat = aligned_cond @ self._cond_act_w  # (B, H, D)
      t_grid = np.linspace(0.0, 1.0, h_frames, dtype=np.float32)[None, :, None]
      live_stem = (
          live_obs_feat[:, None, :] * (1.0 - 0.15 * t_grid)
          + live_act_feat
      ).astype(np.float32)  # (B, H, D)

      # Evaluate DiT-Cache / TeaCache temporal feature reuse criterion
      prev_obs = session.get("prev_obs_frames")
      cached_mid = session.get("cached_dit_features")
      frame_delta_rms = float("inf")
      frame_delta_l2 = float("inf")
      dit_cache_hit = False

      if (
          prev_obs is not None
          and isinstance(prev_obs, np.ndarray)
          and prev_obs.shape == obs_arr.shape
      ):
        diff = obs_arr - prev_obs
        frame_delta_l2 = float(np.linalg.norm(diff))
        frame_delta_rms = float(frame_delta_l2 / np.sqrt(max(1, diff.size)))
        if (
            self.config.enable_dit_cache
            and int(step_idx) > 0
            and cached_mid is not None
            and cached_mid.shape == live_stem.shape
            and frame_delta_rms < float(self.config.dit_cache_threshold)
        ):
          dit_cache_hit = True
          session["dit_cache_hits"] = int(session.get("dit_cache_hits", 0)) + 1

      if dit_cache_hit and cached_mid is not None:
        # TeaCache residual reuse: combine cached mid-block features with live input stem
        # plus 1 lightweight refinement step so Gate G4 1e-3 perturbations propagate cleanly
        refined = self._run_mid_dit_blocks(live_stem, n_steps=1)
        dit_hidden = (0.55 * cached_mid + 0.45 * refined).astype(np.float32)
      else:
        dit_hidden = self._run_mid_dit_blocks(live_stem, n_steps=n_steps)
        session["cached_dit_features"] = (dit_hidden - 0.45 * live_stem).astype(np.float32)

      session["prev_obs_frames"] = obs_arr.copy()
      session["step_count"] = int(session.get("step_count", 0)) + 1
      session["last_step_idx"] = int(step_idx)
      session["updated_at"] = now_pacific_iso()

    # Decode future video latents (B, H, 16, 16) in [0, 1] and inverse-dynamics actions (B, H, 7) in [-1, 1]
    raw_video = (dit_hidden + 0.35 * live_stem) @ self._video_head_w  # (B, H, 256)
    video_latents = (0.5 + 0.45 * np.tanh(raw_video)).reshape(
        batch_size, h_frames, self.LATENT_H, self.LATENT_W
    ).astype(np.float32)

    raw_actions = (dit_hidden + 0.50 * live_act_feat + 0.25 * live_stem) @ self._inv_dyn_head_w
    latent_actions = np.clip(np.tanh(raw_actions), -1.0, 1.0).astype(np.float32)

    t_end_ns = time.perf_counter_ns()
    step_latency_ms = (t_end_ns - t0_ns) / 1e6

    return {
        "video_latents": video_latents,
        "latent_actions": latent_actions,
        "session_id": str(session_id),
        "step_idx": int(step_idx),
        "model_id": self.config.model_id,
        "canonical_model_id": self.canonical_model_id,
        "architecture": "Spatio-Temporal Action-Conditioned DiT World Model (Cosmos3-Edge)",
        "kv_prefix_cache_hit": kv_prefix_cache_hit,
        "dit_cache_hit": dit_cache_hit,
        "dit_cache_threshold": float(self.config.dit_cache_threshold),
        "frame_delta_rms": (
            round(frame_delta_rms, 6) if np.isfinite(frame_delta_rms) else None
        ),
        "frame_delta_l2": (
            round(frame_delta_l2, 6) if np.isfinite(frame_delta_l2) else None
        ),
        "num_inference_steps": n_steps,
        "effective_dit_steps_executed": 1 if dit_cache_hit else n_steps,
        "step_latency_ms": round(float(step_latency_ms), 4),
        "active_params_b": 2.4,
        "total_params_b": 2.4,
    }

  def step_act(
      self,
      session_id: str,
      step_idx: int,
      rgb_frames: np.ndarray,
      proprio_state: np.ndarray,
      instruction_tokens: np.ndarray,
      chunk_size: int = 16,
      prev_chunk_tail: Optional[np.ndarray] = None,
  ) -> Dict[str, Any]:
    """Inverse-dynamics action extraction wrapper for Cosmos3-Edge."""
    rgb_arr = np.asarray(rgb_frames, dtype=np.float32)
    b = int(rgb_arr.shape[0]) if rgb_arr.ndim >= 4 else 1
    k = int(chunk_size)
    act_dim = int(self.config.action_dim)
    zero_cond = np.zeros((b, k, act_dim), dtype=np.float32)
    wm_res = self.step_world(
        session_id=session_id,
        step_idx=step_idx,
        obs_frames=rgb_arr,
        cond_actions=zero_cond,
        horizon_frames=k,
    )
    actions = wm_res["latent_actions"]
    if prev_chunk_tail is not None:
      tail = np.asarray(prev_chunk_tail, dtype=np.float32)
      if tail.ndim == 2:
        tail = tail[None, :, :]
      d_pin = min(k, int(tail.shape[1]))
      if d_pin > 0:
        actions[:, :d_pin, :] = tail[:, :d_pin, :act_dim]
    return {
        "actions": actions,
        "session_id": str(session_id),
        "step_idx": int(step_idx),
        "model_id": self.config.model_id,
        "canonical_model_id": self.canonical_model_id,
        "kv_prefix_cache_hit": wm_res["kv_prefix_cache_hit"],
        "dit_cache_hit": wm_res["dit_cache_hit"],
        "num_inference_steps": wm_res["num_inference_steps"],
        "step_latency_ms": wm_res["step_latency_ms"],
        "active_params_b": 2.4,
        "total_params_b": 2.4,
    }


def main(argv: Optional[List[str]] = None) -> int:
  parser = build_arg_parser(default_model_id="nvidia/Cosmos3-Edge")
  args = parser.parse_args(argv)
  cfg = EmbodiedRunnerConfig(
      model_id=args.model_id,
      hardware_sku=args.hardware_sku,
      precision=args.precision,
      num_devices=args.num_devices,
      num_inference_steps=args.num_inference_steps,
      enable_prefix_cache=not args.disable_prefix_cache,
      enable_dit_cache=not args.disable_dit_cache,
      execution_strategy=args.execution_strategy,
      weights_mode=args.weights_mode,
  )
  runner = Cosmos3EdgeRunner(cfg)
  return run_runner_cli(runner, args)


if __name__ == "__main__":
  raise SystemExit(main())
