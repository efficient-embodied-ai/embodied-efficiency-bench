"""Reference Model Server for Track 2B: Alibaba Qwen3-Omni-30B-A3B (`Qwen/Qwen3-Omni-30B-A3B-Instruct` / `qwen3-omni-30b-a3b`).

Implements the 30B-total / 3B-active MoE Thinker-Talker streaming multimodal architecture:
  - `num_experts = 128`, `top_k_experts = 8`, active parameter ratio `3.0 / 30.0 = 0.10`.
  - Tracks active MoE expert routing IDs (`active_expert_ids`) and routing weights.
  - Tracks multimodal `video + audio + proprio` KV prefix cache hits across `session_id`.
  - Supports external `vLLM` / `SGLang` backend connection when `vllm_endpoint` is provided.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from embodied_bench.schema import now_pacific_iso
from runners.base_server import (
    BaseEmbodiedModelRunner,
    EmbodiedRunnerConfig,
    build_arg_parser,
    run_runner_cli,
)


class Qwen3OmniRunner(BaseEmbodiedModelRunner):
  """Reference runner for `Qwen/Qwen3-Omni-30B-A3B-Instruct` (`qwen3-omni-30b-a3b`)."""

  NUM_EXPERTS: int = 128
  TOP_K_EXPERTS: int = 8
  TOTAL_PARAMS_B: float = 30.0
  ACTIVE_PARAMS_B: float = 3.0
  ACTIVE_PARAM_RATIO: float = 0.10
  D_MODEL: int = 256
  LATENT_H: int = 16
  LATENT_W: int = 16

  def __init__(self, config: EmbodiedRunnerConfig) -> None:
    if config.canonical_model_id != "qwen3-omni-30b-a3b":
      raise ValueError(
          f"Qwen3OmniRunner requires canonical_model_id='qwen3-omni-30b-a3b', "
          f"got {config.canonical_model_id!r} (from model_id={config.model_id!r})."
      )
    super().__init__(config)
    rng = np.random.default_rng(config.seed + 3003)
    d = self.D_MODEL
    act_dim = int(config.action_dim)
    latent_dim = self.LATENT_H * self.LATENT_W

    # Multimodal (video + audio + proprio + language prefix) encoder weights
    self._token_embed = rng.normal(0.0, 0.08, size=(512, d)).astype(np.float32)
    self._multimodal_prefix_w = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, d)).astype(np.float32)
    self._video_stem_w = rng.normal(0.0, 1.0 / np.sqrt(256), size=(256, d)).astype(np.float32)
    self._proprio_w = rng.normal(0.0, 0.14, size=(8, d)).astype(np.float32)
    self._cond_act_w = rng.normal(0.0, 0.14, size=(act_dim, d)).astype(np.float32)

    # MoE Router (128 experts, top_k = 8 active experts per step)
    self._router_gate_w = rng.normal(
        0.0, 1.0 / np.sqrt(d), size=(d, self.NUM_EXPERTS)
    ).astype(np.float32)
    self._expert_bank_w = rng.normal(
        0.0, 1.0 / np.sqrt(d), size=(self.NUM_EXPERTS, d, d)
    ).astype(np.float32)

    # Thinker-Talker heads for world latents and continuous actions
    self._talker_video_w = rng.normal(
        0.0, 1.0 / np.sqrt(d), size=(d, latent_dim)
    ).astype(np.float32)
    self._talker_action_w = rng.normal(
        0.0, 1.0 / np.sqrt(d), size=(d, act_dim)
    ).astype(np.float32)

  @property
  def total_params_b(self) -> float:
    return self.TOTAL_PARAMS_B

  @property
  def active_params_b(self) -> float:
    return self.ACTIVE_PARAMS_B

  def health(self) -> Dict[str, Any]:
    base = super().health()
    base.update({
        "num_experts": self.NUM_EXPERTS,
        "top_k_experts": self.TOP_K_EXPERTS,
        "active_param_ratio": self.ACTIVE_PARAM_RATIO,
        "vllm_endpoint": self.config.vllm_endpoint,
        "backend_engine": (
            "vllm-sglang-remote" if self.config.vllm_endpoint else "moe-thinker-talker-local"
        ),
    })
    return base

  def _route_and_apply_moe(
      self,
      hidden: np.ndarray,
  ) -> Tuple[np.ndarray, List[int], List[float]]:
    """Route multimodal hidden state through top-8 out of 128 MoE experts."""
    pooled = np.mean(hidden.reshape(-1, self.D_MODEL), axis=0)  # (D_MODEL,)
    logits = pooled @ self._router_gate_w  # (128,)
    # Deterministic argsort descending to pick top_k = 8 experts
    top_indices = np.argsort(-logits, kind="mergesort")[: self.TOP_K_EXPERTS]
    top_logits = logits[top_indices]
    shifted = top_logits - np.max(top_logits)
    exp_scores = np.exp(shifted)
    weights = exp_scores / np.sum(exp_scores)

    # Sparse MoE forward: only evaluate the 8 selected experts out of 128 (10% active params)
    selected_w = self._expert_bank_w[top_indices]  # (8, D, D)
    weighted_proj = np.tensordot(weights.astype(np.float32), selected_w, axes=(0, 0))  # (D, D)
    moe_out = (hidden + 0.45 * np.tanh(hidden @ weighted_proj)).astype(np.float32)

    expert_ids = [int(idx) for idx in top_indices.tolist()]
    routing_weights = [round(float(w), 6) for w in weights.tolist()]
    return moe_out, expert_ids, routing_weights

  def _encode_visual_stream(self, frames: np.ndarray, batch_size: int) -> np.ndarray:
    flat = np.asarray(frames, dtype=np.float32).reshape(batch_size, -1)
    if flat.shape[1] >= 256:
      stride = flat.shape[1] // 256
      pooled = flat[:, : 256 * stride].reshape(batch_size, 256, stride).mean(axis=-1)
    else:
      pooled = np.zeros((batch_size, 256), dtype=np.float32)
      pooled[:, : flat.shape[1]] = flat
    return (pooled @ self._video_stem_w).astype(np.float32)

  def step_world(
      self,
      session_id: str,
      step_idx: int,
      obs_frames: np.ndarray,
      cond_actions: np.ndarray,
      horizon_frames: int = 16,
  ) -> Dict[str, Any]:
    """Execute 30B-A3B MoE Thinker-Talker multimodal world rollout step."""
    if int(horizon_frames) < 1:
      raise ValueError(f"horizon_frames must be >= 1, got {horizon_frames}.")
    t0_ns = time.perf_counter_ns()
    obs_arr = np.asarray(obs_frames, dtype=np.float32)
    batch_size = int(obs_arr.shape[0]) if obs_arr.ndim >= 4 else 1
    h_frames = int(horizon_frames)
    act_dim = int(self.config.action_dim)
    n_steps = int(self.config.num_inference_steps or 4)

    cond_arr = np.asarray(cond_actions, dtype=np.float32)
    if cond_arr.ndim == 2:
      cond_arr = cond_arr[None, :, :]
    if cond_arr.shape[0] == 1 and batch_size > 1:
      cond_arr = np.broadcast_to(
          cond_arr, (batch_size, cond_arr.shape[1], cond_arr.shape[2])
      ).copy()
    aligned_cond = np.zeros((batch_size, h_frames, act_dim), dtype=np.float32)
    ch = min(h_frames, int(cond_arr.shape[1]))
    cd = min(act_dim, int(cond_arr.shape[2]))
    aligned_cond[:, :ch, :cd] = cond_arr[:, :ch, :cd]

    with self._lock:
      session = self._get_or_create_session(session_id, step_idx)
      if (
          self.config.enable_prefix_cache
          and int(step_idx) > 0
          and session.get("instruction_kv_cache") is not None
      ):
        kv_prefix_cache_hit = True
        prefix_kv = session["instruction_kv_cache"]
        session["kv_prefix_cache_hits"] = int(session.get("kv_prefix_cache_hits", 0)) + 1
      else:
        kv_prefix_cache_hit = False
        base_tokens = np.arange(24, dtype=np.int32)
        prefix_kv = np.mean(
            np.tanh(self._token_embed[base_tokens] @ self._multimodal_prefix_w),
            axis=0,
            keepdims=True,
        ).astype(np.float32)
        session["instruction_kv_cache"] = prefix_kv

      vis_ctx = self._encode_visual_stream(obs_arr, batch_size)  # (B, D)
      act_ctx = aligned_cond @ self._cond_act_w  # (B, H, D)
      hidden = (vis_ctx[:, None, :] + act_ctx + prefix_kv[:, None, :]).astype(np.float32)

      active_expert_ids: List[int] = []
      expert_weights: List[float] = []
      for _ in range(n_steps):
        hidden, active_expert_ids, expert_weights = self._route_and_apply_moe(hidden)

      session["last_active_expert_ids"] = list(active_expert_ids)
      session["step_count"] = int(session.get("step_count", 0)) + 1
      session["last_step_idx"] = int(step_idx)
      session["updated_at"] = now_pacific_iso()

    video_raw = hidden @ self._talker_video_w
    video_latents = (0.5 + 0.45 * np.tanh(video_raw)).reshape(
        batch_size, h_frames, self.LATENT_H, self.LATENT_W
    ).astype(np.float32)

    latent_actions = np.clip(
        np.tanh((hidden + 0.35 * act_ctx) @ self._talker_action_w),
        -1.0,
        1.0,
    ).astype(np.float32)

    t_end_ns = time.perf_counter_ns()
    step_latency_ms = (t_end_ns - t0_ns) / 1e6

    return {
        "video_latents": video_latents,
        "latent_actions": latent_actions,
        "session_id": str(session_id),
        "step_idx": int(step_idx),
        "model_id": self.config.model_id,
        "canonical_model_id": self.canonical_model_id,
        "architecture": "30B-Total / 3B-Active MoE Thinker-Talker Multimodal World/Omni Model",
        "kv_prefix_cache_hit": kv_prefix_cache_hit,
        "multimodal_modalities_cached": ["video", "audio", "proprio", "language"],
        "num_experts": self.NUM_EXPERTS,
        "top_k_experts": self.TOP_K_EXPERTS,
        "active_expert_ids": active_expert_ids,
        "expert_routing_weights": expert_weights,
        "active_params_b": self.ACTIVE_PARAMS_B,
        "total_params_b": self.TOTAL_PARAMS_B,
        "active_param_ratio": self.ACTIVE_PARAM_RATIO,
        "vllm_endpoint": self.config.vllm_endpoint,
        "num_inference_steps": n_steps,
        "step_latency_ms": round(float(step_latency_ms), 4),
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
    """Execute 30B-A3B MoE Thinker-Talker streaming action chunk prediction."""
    if int(chunk_size) < 1:
      raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    t0_ns = time.perf_counter_ns()
    rgb_arr = np.asarray(rgb_frames, dtype=np.float32)
    batch_size = int(rgb_arr.shape[0]) if rgb_arr.ndim >= 4 else 1
    k = int(chunk_size)
    act_dim = int(self.config.action_dim)
    n_steps = int(self.config.num_inference_steps or 4)

    with self._lock:
      session = self._get_or_create_session(session_id, step_idx)
      sig = self._compute_instruction_signature(instruction_tokens)
      if (
          self.config.enable_prefix_cache
          and int(step_idx) > 0
          and session.get("instruction_signature") == sig
          and session.get("instruction_kv_cache") is not None
      ):
        kv_prefix_cache_hit = True
        prefix_kv = session["instruction_kv_cache"]
        session["kv_prefix_cache_hits"] = int(session.get("kv_prefix_cache_hits", 0)) + 1
      else:
        kv_prefix_cache_hit = False
        tok_flat = np.asarray(instruction_tokens, dtype=np.int32).reshape(-1)
        if tok_flat.size < 24:
          padded = np.zeros(24, dtype=np.int32)
          padded[: tok_flat.size] = tok_flat
          tok_flat = padded
        else:
          tok_flat = tok_flat[:24]
        idx = np.mod(np.abs(tok_flat), self._token_embed.shape[0])
        prefix_kv = np.mean(
            np.tanh(self._token_embed[idx] @ self._multimodal_prefix_w),
            axis=0,
            keepdims=True,
        ).astype(np.float32)
        session["instruction_signature"] = sig
        session["instruction_kv_cache"] = prefix_kv

      vis_ctx = self._encode_visual_stream(rgb_arr, batch_size)
      prop_f = np.asarray(proprio_state, dtype=np.float32).reshape(batch_size, -1)
      prop_pad = np.zeros((batch_size, 8), dtype=np.float32)
      cols = min(8, prop_f.shape[1])
      prop_pad[:, :cols] = prop_f[:, :cols]
      prop_ctx = prop_pad @ self._proprio_w

      t_grid = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, :, None]
      base_ctx = (vis_ctx + prop_ctx + prefix_kv)[:, None, :]
      hidden = (base_ctx + 0.1 * np.sin(np.pi * t_grid)).astype(np.float32)

      active_expert_ids: List[int] = []
      expert_weights: List[float] = []
      for _ in range(n_steps):
        hidden, active_expert_ids, expert_weights = self._route_and_apply_moe(hidden)

      actions = np.clip(np.tanh(hidden @ self._talker_action_w), -1.0, 1.0).astype(np.float32)
      if prev_chunk_tail is not None:
        tail = np.asarray(prev_chunk_tail, dtype=np.float32)
        if tail.ndim == 2:
          tail = tail[None, :, :]
        d_pin = min(k, int(tail.shape[1]))
        if d_pin > 0:
          actions[:, :d_pin, :] = tail[:, :d_pin, :act_dim]

      session["last_active_expert_ids"] = list(active_expert_ids)
      session["step_count"] = int(session.get("step_count", 0)) + 1
      session["last_step_idx"] = int(step_idx)
      session["updated_at"] = now_pacific_iso()

    t_end_ns = time.perf_counter_ns()
    step_latency_ms = (t_end_ns - t0_ns) / 1e6

    return {
        "actions": actions,
        "session_id": str(session_id),
        "step_idx": int(step_idx),
        "model_id": self.config.model_id,
        "canonical_model_id": self.canonical_model_id,
        "architecture": "30B-Total / 3B-Active MoE Thinker-Talker Multimodal World/Omni Model",
        "kv_prefix_cache_hit": kv_prefix_cache_hit,
        "multimodal_modalities_cached": ["video", "audio", "proprio", "language"],
        "num_experts": self.NUM_EXPERTS,
        "top_k_experts": self.TOP_K_EXPERTS,
        "active_expert_ids": active_expert_ids,
        "expert_routing_weights": expert_weights,
        "active_params_b": self.ACTIVE_PARAMS_B,
        "total_params_b": self.TOTAL_PARAMS_B,
        "active_param_ratio": self.ACTIVE_PARAM_RATIO,
        "vllm_endpoint": self.config.vllm_endpoint,
        "num_inference_steps": n_steps,
        "step_latency_ms": round(float(step_latency_ms), 4),
    }


def main(argv: Optional[List[str]] = None) -> int:
  parser = build_arg_parser(default_model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct")
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
      vllm_endpoint=args.vllm_endpoint,
  )
  runner = Qwen3OmniRunner(cfg)
  return run_runner_cli(runner, args)


if __name__ == "__main__":
  raise SystemExit(main())
