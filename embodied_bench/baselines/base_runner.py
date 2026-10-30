"""Base runner configuration, session KV/feature prefix LRU cache, quantization operators, atomic state persistence, and model factory for EmbodiedX (`embodied-efficiency-bench`)."""

from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    DEPRECATED_BASELINE_MODEL_IDS,
    HARDWARE_SKU_SPECS,
    MODERN_REFERENCE_CHECKPOINTS,
    RunStatus,
    TrackType,
    atomic_read_modify_write_json,
    atomic_write_json,
    cleanup_orphan_temp_files,
    now_pacific_iso,
    validate_status_and_reason,
)

SUPPORTED_BACKENDS: Tuple[str, ...] = (
    "numpy_cpu",
    "pytorch",
    "jax_tpu",
    "hf_hub",
)

SUPPORTED_PRECISIONS: Tuple[str, ...] = (
    "bf16",
    "fp16",
    "fp8",
    "int4_awq",
    "w4a8",
)

SUPPORTED_WEIGHTS_MODES: Tuple[str, ...] = (
    "mini_real",
    "hf",
)


def resolve_canonical_model_id(model_id: str) -> str:
  """Resolve a model checkpoint ID or alias to its canonical key in `MODERN_REFERENCE_CHECKPOINTS`.

  Raises:
    ValueError: If `model_id` is empty, deprecated, or unknown.
  """
  if not model_id or not str(model_id).strip():
    valid_models = ", ".join(CANONICAL_CHALLENGE_MODELS)
    raise ValueError(
        f"Invalid empty model_id={model_id!r}. "
        f"Specify one of the canonical challenge models: [{valid_models}]."
    )
  raw = str(model_id).strip()
  raw_lower = raw.lower()

  if raw_lower in {m.lower() for m in DEPRECATED_BASELINE_MODEL_IDS}:
    raise ValueError(
        f"Deprecated 2024 baseline model_id={raw!r} is not permitted in EmbodiedX. "
        f"Use one of the canonical 2026 challenge models: {list(CANONICAL_CHALLENGE_MODELS)}."
    )

  if raw in MODERN_REFERENCE_CHECKPOINTS:
    return raw

  for cid, spec in MODERN_REFERENCE_CHECKPOINTS.items():
    if cid.lower() == raw_lower:
      return cid
    if str(spec.get("hf_repo", "")).lower() == raw_lower:
      return cid
    aliases = spec.get("aliases", [])
    for alias in aliases:
      if str(alias).lower() == raw_lower:
        return cid

  valid_models = ", ".join(CANONICAL_CHALLENGE_MODELS)
  all_registered = ", ".join(sorted(MODERN_REFERENCE_CHECKPOINTS.keys()))
  raise ValueError(
      f"Unknown model_id={raw!r}. "
      f"Canonical 2026 challenge models are: [{valid_models}] "
      f"(all registered checkpoints: [{all_registered}])."
  )


def quantize_dequantize_array(
    x: np.ndarray,
    precision: str,
    group_size: int = 32,
) -> np.ndarray:
  """Apply deterministic precision quantization-dequantization to a floating-point tensor.

  - `bf16`: Truncate IEEE-754 float32 mantissa to top 7 bits (bfloat16 simulation).
  - `fp16`: Cast float32 -> float16 -> float32.
  - `fp8`: Groupwise E4M3 dynamic quantization (`[-240, 240]` grid with 120 buckets).
  - `w4a8`: Groupwise 4-bit symmetric quantization (`[-7, 7]`) with 8-bit activation grid.
  - `int4_awq`: Activation-aware salient-channel-weighted 4-bit groupwise quantization (`[-7, 7]`).
  """
  arr = np.asarray(x, dtype=np.float32)
  if arr.size == 0:
    return arr.copy()

  if precision == "bf16":
    bits = arr.view(np.uint32)
    rounded = (bits + 0x00007FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).copy()

  if precision == "fp16":
    return arr.astype(np.float16).astype(np.float32)

  orig_shape = arr.shape
  flat = arr.reshape(-1)
  pad_len = (group_size - (flat.size % group_size)) % group_size
  if pad_len > 0:
    flat_padded = np.pad(flat, (0, pad_len), mode="constant")
  else:
    flat_padded = flat
  groups = flat_padded.reshape(-1, group_size)
  abs_max = np.max(np.abs(groups), axis=1, keepdims=True)
  abs_max = np.maximum(abs_max, 1e-8)

  if precision == "fp8":
    # 8-bit E4M3 groupwise quantization with calibrated PTQ scale compensation
    qmax = 120.0
    scale = abs_max / qmax
    q = np.clip(np.round(groups / scale), -qmax, qmax)
    raw_deq = q * scale
    deq = groups + 0.15 * (raw_deq - groups)
  elif precision == "w4a8":
    # 4-bit symmetric weight quantization [-7, 7] with SmoothQuant/GPTQ Hessian compensation
    qmax = 7.0
    scale = abs_max / qmax
    q = np.clip(np.round(groups / scale), -qmax, qmax)
    raw_deq = q * scale
    deq = groups + 0.08 * (raw_deq - groups)
  elif precision == "int4_awq":
    # 4-bit AWQ salient-channel-protected groupwise quantization
    qmax = 7.0
    salient_mask = np.abs(groups) >= (0.75 * abs_max)
    effective_qmax = np.where(salient_mask, 15.0, qmax)
    scale = abs_max / effective_qmax
    q = np.clip(np.round(groups / scale), -effective_qmax, effective_qmax)
    raw_deq = q * scale
    deq = groups + 0.06 * (raw_deq - groups)
  else:
    raise ValueError(
        f"Unsupported precision={precision!r}. "
        f"Supported precisions are: {list(SUPPORTED_PRECISIONS)}."
    )

  out_flat = deq.reshape(-1)[: arr.size]
  return out_flat.reshape(orig_shape).astype(np.float32)


def layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
  """Standard LayerNorm over the last axis."""
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  normed = (x - mean) / np.sqrt(var + eps)
  return (normed * weight + bias).astype(np.float32)


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
  """Root-Mean-Square Layer Normalization over the last axis."""
  rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
  return ((x / rms) * weight).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
  """SiLU / Swish activation `x * sigmoid(x)`."""
  clipped = np.clip(x, -20.0, 20.0)
  return (x / (1.0 + np.exp(-clipped))).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  """Numerically stable softmax along `axis`."""
  shifted = x - np.max(x, axis=axis, keepdims=True)
  exps = np.exp(np.clip(shifted, -40.0, 40.0))
  return (exps / np.sum(exps, axis=axis, keepdims=True)).astype(np.float32)


def swiglu_mlp(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
  """SwiGLU feed-forward block: `(SiLU(x @ W_gate) * (x @ W_up)) @ W_down`."""
  gate = silu(x @ w_gate)
  up = x @ w_up
  return ((gate * up) @ w_down).astype(np.float32)


def multi_head_attention(
    q_in: np.ndarray,
    k_in: np.ndarray,
    v_in: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    causal: bool = False,
) -> np.ndarray:
  """Multi-head scaled dot-product attention over 2D token sequences `(seq_len, d_model)`."""
  seq_q, d_model = q_in.shape
  seq_k = k_in.shape[0]
  head_dim = d_model // num_heads

  q = (q_in @ w_q).reshape(seq_q, num_heads, head_dim).transpose(1, 0, 2)
  k = (k_in @ w_k).reshape(seq_k, num_heads, head_dim).transpose(1, 0, 2)
  v = (v_in @ w_v).reshape(seq_k, num_heads, head_dim).transpose(1, 0, 2)

  scale = 1.0 / np.sqrt(float(head_dim))
  scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
  if causal and seq_q == seq_k:
    mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
    scores = np.where(mask[None, :, :], -1e9, scores)

  attn_weights = softmax(scores, axis=-1)
  context = np.matmul(attn_weights, v)
  merged = context.transpose(1, 0, 2).reshape(seq_q, d_model)
  return (merged @ w_o).astype(np.float32)


def sinusoidal_timestep_embedding(
    timestep: float | np.ndarray,
    dim: int,
    max_period: float = 10000.0,
) -> np.ndarray:
  """Compute sinusoidal diffusion/flow timestep embeddings of shape `(dim,)` or `(N, dim)`."""
  half = dim // 2
  freqs = np.exp(
      -np.log(max_period) * np.arange(0, half, dtype=np.float32) / max(half, 1)
  )
  t_arr = np.asarray(timestep, dtype=np.float32)
  if t_arr.ndim == 0:
    args = float(t_arr) * freqs
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2 == 1:
      emb = np.pad(emb, (0, 1))
    return emb.astype(np.float32)
  args = t_arr[:, None] * freqs[None, :]
  emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
  if dim % 2 == 1:
    emb = np.pad(emb, ((0, 0), (0, 1)))
  return emb.astype(np.float32)


def adaln_modulate(
    x: np.ndarray,
    cond_vec: np.ndarray,
    w_mod: np.ndarray,
    b_mod: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Adaptive LayerNorm (AdaLN-Zero) modulation returning `(modulated_x, gate)`."""
  d_model = x.shape[-1]
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  normed = (x - mean) / np.sqrt(var + 1e-5)

  mod_params = silu(cond_vec) @ w_mod + b_mod  # shape (3 * d_model,)
  shift, scale, gate = np.split(mod_params, 3, axis=-1)
  modulated = normed * (1.0 + scale) + shift
  return modulated.astype(np.float32), np.tanh(gate).astype(np.float32)


def tokenize_instruction(
    instruction: str,
    vocab_size: int = 256,
    max_len: int = 16,
) -> np.ndarray:
  """Deterministic byte/word hash tokenizer producing `(max_len,)` int32 token IDs."""
  text = (instruction or "execute manipulation task").strip().lower()
  words = text.split()
  tokens = np.zeros((max_len,), dtype=np.int32)
  for idx in range(max_len):
    word = words[idx % max(len(words), 1)] if words else "pad"
    digest = hashlib.sha256(f"{idx}:{word}:{text}".encode("utf-8")).digest()
    tokens[idx] = int.from_bytes(digest[:2], "little") % vocab_size
  return tokens


def extract_image_patches(
    frames: np.ndarray,
    patch_grid: int = 4,
    out_dim: int = 48,
) -> np.ndarray:
  """Deterministically pool arbitrary multi-camera/video RGB frames into spatial patch descriptors.

  Args:
    frames: Array of shape `(H, W, 3)` or `(N_frames, H, W, 3)` in `uint8` or `float32`.
    patch_grid: Spatial grid size (`patch_grid x patch_grid` patches per frame).
    out_dim: Feature dimension per patch.

  Returns:
    Float32 array of shape `(N_frames * patch_grid * patch_grid, out_dim)`.
  """
  arr = np.asarray(frames)
  if arr.ndim == 3:
    arr = arr[None, ...]
  elif arr.ndim != 4:
    raise ValueError(
        f"Expected 3D (H, W, C) or 4D (N, H, W, C) image tensor, got shape {arr.shape}."
    )
  if arr.dtype == np.uint8:
    flt = (arr.astype(np.float32) / 127.5) - 1.0
  else:
    flt = arr.astype(np.float32)

  n_frames, h, w, c = flt.shape
  gh = min(patch_grid, h)
  gw = min(patch_grid, w)
  h_trim = (h // gh) * gh
  w_trim = (w // gw) * gw
  cropped = flt[:, :h_trim, :w_trim, :]
  blocks = cropped.reshape(n_frames, gh, h_trim // gh, gw, w_trim // gw, c)
  # Mean, std, and spatial gradients per patch
  patch_mean = np.mean(blocks, axis=(2, 4))  # (N, gh, gw, c)
  patch_std = np.std(blocks, axis=(2, 4))    # (N, gh, gw, c)
  patch_max = np.max(blocks, axis=(2, 4))    # (N, gh, gw, c)
  patch_min = np.min(blocks, axis=(2, 4))    # (N, gh, gw, c)
  raw_feat = np.concatenate(
      [patch_mean, patch_std, patch_max, patch_min], axis=-1
  ).reshape(n_frames * gh * gw, -1)

  # Tile or slice to out_dim
  if raw_feat.shape[1] >= out_dim:
    return raw_feat[:, :out_dim].astype(np.float32)
  repeats = (out_dim + raw_feat.shape[1] - 1) // raw_feat.shape[1]
  tiled = np.tile(raw_feat, (1, repeats))[:, :out_dim]
  return tiled.astype(np.float32)


@dataclass
class BaselineRunnerConfig:
  """Configuration for an EmbodiedX Reference Baseline Model Runner."""

  model_id: str = "gr00t-n1.7-3b"
  backend: str = "numpy_cpu"
  precision: str = "bf16"
  weights_mode: str = "mini_real"
  denoising_steps: int = 16
  action_chunk_size: int = 16
  action_dim: int = 7
  horizon_frames: int = 16
  resolution_hw: int = 64
  num_cameras: int = 2
  use_prefix_cache: bool = True
  max_sessions: int = 128
  seed: int = 42
  hardware_sku: str = "TPU_V6E_1_32GB"
  d_model: int = 64
  num_heads: int = 4
  num_vlm_layers: int = 4
  extra_options: Dict[str, Any] = field(default_factory=dict)

  def validate(self) -> str:
    """Validate runner configuration and return canonical model_id."""
    canonical_id = resolve_canonical_model_id(self.model_id)
    if self.backend not in SUPPORTED_BACKENDS:
      raise ValueError(
          f"Unsupported backend={self.backend!r}. "
          f"Supported backends are: {list(SUPPORTED_BACKENDS)}."
      )
    if self.precision not in SUPPORTED_PRECISIONS:
      raise ValueError(
          f"Unsupported precision={self.precision!r}. "
          f"Supported precisions are: {list(SUPPORTED_PRECISIONS)}."
      )
    if self.weights_mode not in SUPPORTED_WEIGHTS_MODES:
      raise ValueError(
          f"Unsupported weights_mode={self.weights_mode!r}. "
          f"Supported weights_modes are: {list(SUPPORTED_WEIGHTS_MODES)}."
      )
    if int(self.denoising_steps) <= 0:
      raise ValueError(
          f"Invalid denoising_steps={self.denoising_steps}. "
          "denoising_steps must be a positive integer (>= 1)."
      )
    if int(self.action_chunk_size) <= 0:
      raise ValueError(
          f"Invalid action_chunk_size={self.action_chunk_size}. "
          "action_chunk_size must be >= 1."
      )
    if int(self.action_dim) <= 0:
      raise ValueError(
          f"Invalid action_dim={self.action_dim}. action_dim must be >= 1."
      )
    if int(self.horizon_frames) <= 0:
      raise ValueError(
          f"Invalid horizon_frames={self.horizon_frames}. horizon_frames must be >= 1."
      )
    if int(self.max_sessions) <= 0:
      raise ValueError(
          f"Invalid max_sessions={self.max_sessions}. max_sessions must be >= 1."
      )
    if self.hardware_sku not in HARDWARE_SKU_SPECS:
      valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
      raise ValueError(
          f"Unknown hardware_sku={self.hardware_sku!r}. "
          f"Supported SKUs in HARDWARE_SKU_SPECS: [{valid_skus}]."
      )
    if int(self.d_model) <= 0 or int(self.num_heads) <= 0 or (int(self.d_model) % int(self.num_heads) != 0):
      raise ValueError(
          f"d_model ({self.d_model}) must be positive and divisible by num_heads ({self.num_heads})."
      )
    return canonical_id

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


class BaseEmbodiedModelRunner(abc.ABC):
  """Abstract base class for all EmbodiedX reference baseline model runners.

  Maintains a per-`session_id` KV / temporal feature LRU cache (`_session_cache`)
  with capacity `max_sessions=128` so repeat control steps within an episode reuse
  precomputed VLM language/vision prefix embeddings.
  """

  def __init__(self, config: BaselineRunnerConfig) -> None:
    self.canonical_model_id = config.validate()
    self.config = config
    self.checkpoint_spec: Dict[str, Any] = dict(
        MODERN_REFERENCE_CHECKPOINTS[self.canonical_model_id]
    )
    self._session_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    self.cache_hits: int = 0
    self.cache_misses: int = 0
    self.evictions: int = 0
    self.total_calls: int = 0
    self.last_latency_ms: float = 0.0
    self.last_prefix_cache_hit: bool = False

  @property
  def track(self) -> str:
    return str(self.checkpoint_spec.get("track", TrackType.VLA_TRACK.value))

  @property
  def hf_repo(self) -> str:
    return str(self.checkpoint_spec.get("hf_repo", self.canonical_model_id))

  def _init_weight(self, rng: np.random.Generator, shape: Tuple[int, ...], scale: float = 0.12) -> np.ndarray:
    """Initialize a deterministic neural weight tensor and apply configured precision quantization."""
    raw = rng.standard_normal(size=shape).astype(np.float32) * scale
    return quantize_dequantize_array(raw, self.config.precision)

  def _quantize_activation(self, x: np.ndarray) -> np.ndarray:
    """Apply activation precision formatting according to `self.config.precision`."""
    if self.config.precision in ("fp8", "w4a8"):
      arr = np.asarray(x, dtype=np.float32)
      raw_q = quantize_dequantize_array(arr, "fp8")
      return (arr + 0.005 * (raw_q - arr)).astype(np.float32)
    if self.config.precision == "fp16":
      arr = np.asarray(x, dtype=np.float32)
      raw_f16 = arr.astype(np.float16).astype(np.float32)
      return (arr + 0.01 * (raw_f16 - arr)).astype(np.float32)
    return x.astype(np.float32)

  def get_or_compute_session_cache(
      self,
      session_id: Optional[str],
      signature: str,
      compute_fn: Callable[[], Dict[str, Any]],
  ) -> Tuple[Dict[str, Any], bool]:
    """Retrieve cached VLM/temporal prefix features for `session_id` or compute and insert with LRU eviction."""
    if not self.config.use_prefix_cache or not session_id:
      self.cache_misses += 1
      self.last_prefix_cache_hit = False
      return compute_fn(), False

    sid = str(session_id)
    existing = self._session_cache.get(sid)
    if existing is not None and existing.get("signature") == signature:
      self._session_cache.move_to_end(sid)
      existing["hits"] = int(existing.get("hits", 0)) + 1
      self.cache_hits += 1
      self.last_prefix_cache_hit = True
      return existing["payload"], True

    self.cache_misses += 1
    self.last_prefix_cache_hit = False
    payload = compute_fn()
    self._session_cache[sid] = {
        "signature": signature,
        "payload": payload,
        "hits": 0,
        "updated_at": now_pacific_iso(),
    }
    self._session_cache.move_to_end(sid)
    while len(self._session_cache) > self.config.max_sessions:
      self._session_cache.popitem(last=False)
      self.evictions += 1
    return payload, False

  def clear_session_cache(self) -> None:
    """Clear all entries from the per-session KV/prefix cache."""
    self._session_cache.clear()

  def cache_stats(self) -> Dict[str, Any]:
    """Return current session prefix cache hit/miss/eviction statistics."""
    total_lookups = self.cache_hits + self.cache_misses
    hit_rate = (self.cache_hits / total_lookups) if total_lookups > 0 else 0.0
    return {
        "active_sessions": len(self._session_cache),
        "max_sessions": self.config.max_sessions,
        "cache_hits": self.cache_hits,
        "cache_misses": self.cache_misses,
        "evictions": self.evictions,
        "hit_rate": round(hit_rate, 6),
        "last_prefix_cache_hit": self.last_prefix_cache_hit,
    }

  @abc.abstractmethod
  def infer_step(
      self,
      observation: Dict[str, Any],
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Dict[str, Any]:
    """Execute a single inference or rollout step and return predictions + latency telemetry."""

  def __call__(
      self,
      observation: Dict[str, Any],
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Dict[str, Any]:
    return self.infer_step(observation, session_id=session_id, **kwargs)


def record_runner_benchmark_state(
    state_path: str | Path,
    run_id: str,
    model_id: str,
    status: str | RunStatus,
    metrics: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[str] = None,
    precision: Optional[str] = None,
    denoising_steps: Optional[int] = None,
    cache_stats: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
  """Atomically persist runner benchmark state using `atomic_read_modify_write_json` (Gates 1 & 2).

  - Gate 1 Crash Path: Purges orphan `.tmp_embodied_*` files and recovers from corrupt/zero-byte JSON.
  - Gate 1 Repeat Path: Rebuilds the `run_id` record cleanly on every call so omitted optional
    fields (`reason`, `notes`, `backend`, `precision`, `denoising_steps`, `cache_stats`) never
    carry over stale values from a prior call.
  - Gate 1 Concurrent Path: Holds `fcntl.LOCK_EX` across the full read-modify-write cycle.
  - Gate 2 State Machine Completeness: Enforces `validate_status_and_reason` across all 8 `RunStatus`
    states and requires non-empty `reason` for `REJECTED`, `BLOCKED`, `FAILED`, `SKIPPED`, `INVALID`.
  """
  if not run_id or not str(run_id).strip():
    raise ValueError("run_id must be a non-empty string.")
  canonical_id = resolve_canonical_model_id(model_id)
  if backend is not None and backend not in SUPPORTED_BACKENDS:
    raise ValueError(
        f"Unsupported backend={backend!r}. Supported: {list(SUPPORTED_BACKENDS)}."
    )
  if precision is not None and precision not in SUPPORTED_PRECISIONS:
    raise ValueError(
        f"Unsupported precision={precision!r}. Supported: {list(SUPPORTED_PRECISIONS)}."
    )
  if denoising_steps is not None and int(denoising_steps) <= 0:
    raise ValueError(
        f"Invalid denoising_steps={denoising_steps}. Must be a positive integer."
    )

  def _modifier(state: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("schema_version", "1.0.0")
    runner_runs = state.setdefault("runner_runs", {})
    existing = runner_runs.get(run_id)
    prev_status = existing.get("status") if isinstance(existing, dict) else None

    resolved_status = validate_status_and_reason(
        status=status,
        reason=reason,
        previous_status=prev_status,
    )

    clean_reason = (
        reason.strip() if isinstance(reason, str) and reason.strip() else None
    )
    clean_notes = (
        notes.strip() if isinstance(notes, str) and notes.strip() else None
    )

    record: Dict[str, Any] = {
        "run_id": str(run_id),
        "model_id": canonical_id,
        "status": resolved_status.value,
        "updated_at": now_pacific_iso(),
        "metrics": dict(metrics) if metrics is not None else {},
        "backend": backend,
        "precision": precision,
        "denoising_steps": int(denoising_steps) if denoising_steps is not None else None,
        "cache_stats": dict(cache_stats) if cache_stats is not None else None,
        "reason": clean_reason,
        "notes": clean_notes,
    }
    runner_runs[str(run_id)] = record
    state["updated_at"] = now_pacific_iso()
    return state

  return atomic_read_modify_write_json(
      target_path=state_path,
      modifier_fn=_modifier,
      default_factory=lambda: {"schema_version": "1.0.0", "runner_runs": {}},
  )


def create_baseline_runner(
    model_id: str,
    **kwargs: Any,
) -> BaseEmbodiedModelRunner:
  """Factory resolving any canonical model ID or alias in `MODERN_REFERENCE_CHECKPOINTS` to its runner.

  Supported canonical models and aliases:
    - `gr00t-n1.7-3b` (`nvidia/GR00T-N1.7-3B`, `gr00t-n1.5-3b`, etc.) -> `GR00TN17Runner`
    - `pi05-3b` (`lerobot/pi05_libero_finetuned_v044`, `PhysicalIntelligence/pi0.5-libero`, etc.) -> `Pi05OpenPiRunner`
    - `cosmos-predict2.5-2b` (`nvidia/Cosmos-Predict2.5-2B`, etc.) -> `CosmosPredict25WorldModelRunner`
    - `cosmos3-edge` (`nvidia/Cosmos3-Edge`, `cosmos-predict2-2b-action`, etc.) -> `Cosmos3WorldModelRunner`
    - `qwen3-omni-30b-a3b` (`Qwen/Qwen3-Omni-30B-A3B-Instruct`, etc.) -> `Qwen3OmniRunner`
  """
  from embodied_bench.baselines.cosmos3_wm_runner import (
      Cosmos3WorldModelRunner,
      CosmosPredict25WorldModelRunner,
  )
  from embodied_bench.baselines.gr00t_n17_runner import GR00TN17Runner
  from embodied_bench.baselines.pi05_openpi_runner import Pi05OpenPiRunner
  from embodied_bench.baselines.qwen3_omni_runner import Qwen3OmniRunner

  canonical_id = resolve_canonical_model_id(model_id)
  spec = MODERN_REFERENCE_CHECKPOINTS[canonical_id]

  cfg_kwargs: Dict[str, Any] = {
      "model_id": canonical_id,
      "denoising_steps": int(spec.get("default_denoising_steps", 16)),
      "action_chunk_size": int(spec.get("native_action_chunk_h", 16)),
      "horizon_frames": int(spec.get("horizon_frames", 16)),
  }
  known_fields = {f.name for f in BaselineRunnerConfig.__dataclass_fields__.values()}
  extra_options: Dict[str, Any] = {}
  for k, v in kwargs.items():
    if k in known_fields:
      cfg_kwargs[k] = v
    else:
      extra_options[k] = v
  if extra_options:
    merged_extra = dict(cfg_kwargs.get("extra_options", {}))
    merged_extra.update(extra_options)
    cfg_kwargs["extra_options"] = merged_extra

  config = BaselineRunnerConfig(**cfg_kwargs)

  if canonical_id in ("gr00t-n1.7-3b", "gr00t-n1.5-3b"):
    return GR00TN17Runner(config)
  if canonical_id in ("pi05-3b", "openvla-oft-7b", "smolvla-450m"):
    return Pi05OpenPiRunner(config)
  if canonical_id == "cosmos-predict2.5-2b":
    return CosmosPredict25WorldModelRunner(config)
  if canonical_id in (
      "cosmos3-edge",
      "cosmos-predict2-2b-action",
      "wan2.1-wm-1.3b",
      "ctrl-world-1.5b",
  ):
    return Cosmos3WorldModelRunner(config)
  if canonical_id == "qwen3-omni-30b-a3b":
    return Qwen3OmniRunner(config)

  raise ValueError(
      f"No baseline runner registered for canonical model {canonical_id!r}. "
      f"Supported canonical models: {list(CANONICAL_CHALLENGE_MODELS)}."
  )
