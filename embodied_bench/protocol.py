"""Zero-Copy POSIX Shared-Memory IPC (`SharedMemoryTensorRing`) and Persistent Keep-Alive HTTP Client/Server Harness for EmbodiedX (`embodied-efficiency-bench` v1.0)."""

from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mmap
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import uuid

import numpy as np

from embodied_bench.schema import TrackType


_ALIGNMENT_BYTES = 64
MAX_HTTP_BODY_BYTES: int = 64 * 1024 * 1024
MAX_ACTIVE_SESSIONS: int = 1024
MAX_ATTACHED_SHM_RINGS: int = 16
_SAFE_SHM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PROCESS_OWNED_SHM_LOCK = threading.Lock()
_PROCESS_OWNED_SHM_NAMES: set[str] = set()


class _PayloadTooLargeError(ValueError):
  """Raised when an HTTP request body exceeds MAX_HTTP_BODY_BYTES (HTTP 413)."""


def _align_up(value: int, alignment: int = _ALIGNMENT_BYTES) -> int:
  return ((int(value) + alignment - 1) // alignment) * alignment


def _sanitize_shm_name(name: str) -> str:
  cleaned = str(name).strip().lstrip("/")
  if "/" in cleaned or ".." in cleaned or not _SAFE_SHM_NAME_RE.fullmatch(cleaned):
    raise ValueError(f"Unsafe or invalid shared-memory segment name {name!r}.")
  return cleaned


def _fallback_mmap_path(shm_name: str) -> Path:
  return Path(tempfile.gettempdir()) / f"eeb_shm_fallback_{_sanitize_shm_name(shm_name)}.mmap"


@dataclass(frozen=True)
class ShmTensorDescriptor:
  """Descriptor for a zero-copy NumPy tensor slice inside a `SharedMemoryTensorRing`."""

  shm_name: str
  offset: int
  shape: Tuple[int, ...]
  dtype: str
  nbytes: int
  sha256_prefix: str = ""

  def to_dict(self) -> Dict[str, Any]:
    return {
        "shm_name": self.shm_name,
        "offset": int(self.offset),
        "shape": [int(dim) for dim in self.shape],
        "dtype": str(self.dtype),
        "nbytes": int(self.nbytes),
        "sha256_prefix": str(self.sha256_prefix),
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ShmTensorDescriptor":
    return cls(
        shm_name=str(data["shm_name"]),
        offset=int(data["offset"]),
        shape=tuple(int(dim) for dim in data["shape"]),
        dtype=str(data["dtype"]),
        nbytes=int(data["nbytes"]),
        sha256_prefix=str(data.get("sha256_prefix", "")),
    )


class SharedMemoryTensorRing:
  """Zero-copy circular shared-memory arena for passing observation tensors between LoadGen and Server.

  Uses `multiprocessing.shared_memory.SharedMemory` when `/dev/shm` is available, with an automatic
  clean fallback to an `mmap`-backed file in `/tmp` when POSIX shared memory is restricted.
  Supports idempotent `.close()` and `.unlink()` as well as context-manager (`with`) usage.
  """

  def __init__(
      self,
      name: Optional[str] = None,
      capacity_bytes: int = 32 * 1024 * 1024,
      create: bool = True,
      force_mmap_fallback: bool = False,
  ) -> None:
    if capacity_bytes <= 0 and create:
      raise ValueError(f"capacity_bytes must be > 0 when create=True, got {capacity_bytes}.")

    raw_name = name or f"eeb_ring_{os.getpid()}_{uuid.uuid4().hex[:10]}"
    self.name: str = _sanitize_shm_name(raw_name)
    self._owns_memory: bool = bool(create)
    self._closed: bool = False
    self._unlinked: bool = False
    self._lock = threading.Lock()
    self._write_offset: int = 0

    self._shm: Optional[shared_memory.SharedMemory] = None
    self._mmap_fp: Optional[ Any] = None
    self._mmap_obj: Optional[mmap.mmap] = None
    self._mmap_path: Path = _fallback_mmap_path(self.name)
    self.backend: str = "posix_shm"

    if create:
      self.capacity_bytes = _align_up(max(4096, int(capacity_bytes)))
      if not force_mmap_fallback:
        try:
          self._shm = shared_memory.SharedMemory(
              name=self.name,
              create=True,
              size=self.capacity_bytes,
          )
          self.backend = "posix_shm"
          with _PROCESS_OWNED_SHM_LOCK:
            _PROCESS_OWNED_SHM_NAMES.add(self.name)
          return
        except (OSError, PermissionError, ValueError, FileExistsError):
          self._shm = None
      self._init_mmap_fallback(create=True, size=self.capacity_bytes)
      with _PROCESS_OWNED_SHM_LOCK:
        _PROCESS_OWNED_SHM_NAMES.add(self.name)
    else:
      if not force_mmap_fallback and not self._mmap_path.exists():
        try:
          self._shm = shared_memory.SharedMemory(name=self.name, create=False)
          self.capacity_bytes = int(self._shm.size)
          self.backend = "posix_shm"
          return
        except (OSError, PermissionError, FileNotFoundError, ValueError):
          self._shm = None
      self._init_mmap_fallback(create=False, size=int(capacity_bytes) if capacity_bytes > 0 else 0)

  def _init_mmap_fallback(self, create: bool, size: int) -> None:
    self.backend = "tmp_mmap"
    self._mmap_path.parent.mkdir(parents=True, exist_ok=True)
    if create:
      fd = os.open(
          str(self._mmap_path),
          os.O_CREAT | os.O_TRUNC | os.O_RDWR,
          0o600,
      )
      os.ftruncate(fd, size)
      self._mmap_fp = os.fdopen(fd, "r+b")
      self._mmap_obj = mmap.mmap(self._mmap_fp.fileno(), size, access=mmap.ACCESS_WRITE)
      self.capacity_bytes = size
    else:
      if not self._mmap_path.exists():
        raise FileNotFoundError(
            f"SharedMemory segment {self.name!r} not found in POSIX /dev/shm or fallback {self._mmap_path}."
        )
      actual_size = self._mmap_path.stat().st_size
      self._mmap_fp = open(self._mmap_path, "r+b")
      self._mmap_obj = mmap.mmap(
          self._mmap_fp.fileno(),
          actual_size,
          access=mmap.ACCESS_WRITE,
      )
      self.capacity_bytes = int(actual_size)

  @property
  def buffer(self) -> memoryview | mmap.mmap:
    if self._closed:
      raise RuntimeError(f"SharedMemoryTensorRing {self.name!r} is already closed.")
    if self._shm is not None and self._shm.buf is not None:
      return self._shm.buf
    if self._mmap_obj is not None:
      return self._mmap_obj
    raise RuntimeError(f"SharedMemoryTensorRing {self.name!r} has no active backing buffer.")

  def write_tensor(
      self,
      arr: np.ndarray,
      compute_checksum: bool = True,
  ) -> ShmTensorDescriptor:
    """Write a NumPy tensor into the next aligned slot of the ring and return its `ShmTensorDescriptor`."""
    if self._closed:
      raise RuntimeError(f"Cannot write_tensor to closed SharedMemoryTensorRing {self.name!r}.")
    contiguous = np.ascontiguousarray(arr)
    nbytes = int(contiguous.nbytes)
    if nbytes <= 0:
      raise ValueError("Cannot write an empty 0-byte tensor to SharedMemoryTensorRing.")
    aligned_nbytes = _align_up(nbytes)
    if aligned_nbytes > self.capacity_bytes:
      raise ValueError(
          f"Tensor size {nbytes} bytes (aligned {aligned_nbytes}) exceeds ring capacity {self.capacity_bytes} bytes."
      )

    with self._lock:
      if self._write_offset + aligned_nbytes > self.capacity_bytes:
        self._write_offset = 0
      slot_offset = self._write_offset
      self._write_offset += aligned_nbytes

    shm_view = np.ndarray(
        shape=contiguous.shape,
        dtype=contiguous.dtype,
        buffer=self.buffer,
        offset=slot_offset,
    )
    np.copyto(shm_view, contiguous)
    if compute_checksum:
      raw_slice = memoryview(self.buffer)[slot_offset : slot_offset + nbytes]
      sha_prefix = hashlib.sha256(raw_slice).hexdigest()[:16]
    else:
      sha_prefix = ""

    return ShmTensorDescriptor(
        shm_name=self.name,
        offset=slot_offset,
        shape=tuple(int(d) for d in contiguous.shape),
        dtype=str(contiguous.dtype),
        nbytes=nbytes,
        sha256_prefix=sha_prefix,
    )

  def read_tensor(
      self,
      desc: ShmTensorDescriptor,
      verify_checksum: bool = True,
  ) -> np.ndarray:
    """Return a zero-copy NumPy view over `desc` in shared memory, optionally verifying `sha256_prefix`."""
    if self._closed:
      raise RuntimeError(f"Cannot read_tensor from closed SharedMemoryTensorRing {self.name!r}.")
    if _sanitize_shm_name(desc.shm_name) != self.name:
      raise ValueError(
          f"Descriptor shm_name={desc.shm_name!r} does not match ring name={self.name!r}."
      )
    if desc.offset < 0 or desc.nbytes <= 0 or desc.offset + desc.nbytes > self.capacity_bytes:
      raise ValueError(
          f"Descriptor slice [{desc.offset}, {desc.offset + desc.nbytes}) out of bounds for capacity {self.capacity_bytes}."
      )
    if len(desc.shape) > 8 or any(int(dim) < 0 for dim in desc.shape):
      raise ValueError(f"Invalid tensor shape {desc.shape} in ShmTensorDescriptor.")

    dtype = np.dtype(desc.dtype)
    expected_nbytes = int(math.prod(int(dim) for dim in desc.shape)) * int(dtype.itemsize)
    if expected_nbytes != int(desc.nbytes) or expected_nbytes > MAX_HTTP_BODY_BYTES:
      raise ValueError(
          f"Descriptor shape {desc.shape} with dtype {desc.dtype} implies {expected_nbytes} bytes, "
          f"but desc.nbytes={desc.nbytes} (max {MAX_HTTP_BODY_BYTES})."
      )

    if verify_checksum and desc.sha256_prefix:
      raw_slice = memoryview(self.buffer)[desc.offset : desc.offset + desc.nbytes]
      actual_prefix = hashlib.sha256(raw_slice).hexdigest()[: len(desc.sha256_prefix)]
      if actual_prefix != desc.sha256_prefix:
        raise ValueError(
            f"SharedMemory tensor checksum mismatch on {self.name!r} at offset {desc.offset}: "
            f"expected sha256_prefix={desc.sha256_prefix!r}, got {actual_prefix!r}."
        )

    return np.ndarray(
        shape=tuple(desc.shape),
        dtype=dtype,
        buffer=self.buffer,
        offset=int(desc.offset),
    )

  def close(self) -> None:
    """Idempotently close the local mapping handle without unlinking the backing object."""
    with self._lock:
      if self._closed:
        return
      self._closed = True
      if self._shm is not None:
        try:
          self._shm.close()
        except Exception:
          pass
        self._shm = None
      if self._mmap_obj is not None:
        try:
          self._mmap_obj.close()
        except Exception:
          pass
        self._mmap_obj = None
      if self._mmap_fp is not None:
        try:
          self._mmap_fp.close()
        except Exception:
          pass
        self._mmap_fp = None

  def unlink(self) -> None:
    """Idempotently unlink the underlying POSIX shared memory segment or fallback `/tmp` mmap file."""
    with self._lock:
      if self._unlinked:
        return
      self._unlinked = True
      with _PROCESS_OWNED_SHM_LOCK:
        _PROCESS_OWNED_SHM_NAMES.discard(self.name)
      if self._shm is not None:
        try:
          self._shm.unlink()
        except (FileNotFoundError, OSError):
          pass
      else:
        try:
          tmp_shm = shared_memory.SharedMemory(name=self.name, create=False)
          try:
            tmp_shm.unlink()
          finally:
            tmp_shm.close()
        except (FileNotFoundError, OSError, ValueError):
          pass
      if self._mmap_path.exists():
        try:
          self._mmap_path.unlink()
        except OSError:
          pass

  def __enter__(self) -> "SharedMemoryTensorRing":
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()
    if self._owns_memory:
      self.unlink()


# =============================================================================
# Structured Client-Server Request / Response Payloads
# =============================================================================


@dataclass
class HealthResponse:
  """Payload returned by `GET /v1/health`."""

  model_id: str
  track: str
  hardware_sku: str
  num_devices: int = 1
  precision: str = "BF16"
  server_version: str = "1.0.0"
  ready: bool = True

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "HealthResponse":
    return cls(
        model_id=str(data["model_id"]),
        track=str(data["track"]),
        hardware_sku=str(data["hardware_sku"]),
        num_devices=int(data.get("num_devices", 1)),
        precision=str(data.get("precision", "BF16")),
        server_version=str(data.get("server_version", "1.0.0")),
        ready=bool(data.get("ready", True)),
    )


@dataclass
class ResetSessionRequest:
  """Payload sent to `POST /v1/reset_session` to initialize an episode stream."""

  session_id: str
  episode_id: str = "ep-0"
  instruction_text: str = "pick up the red block and place it on the target plate"
  instruction_tokens: List[int] = field(default_factory=lambda: [101, 2023, 2003, 1037, 102])
  cleared_kv_cache: bool = True

  def to_dict(self) -> Dict[str, Any]:
    return {
        "session_id": self.session_id,
        "episode_id": self.episode_id,
        "instruction_text": self.instruction_text,
        "instruction_tokens": [int(t) for t in self.instruction_tokens],
        "cleared_kv_cache": bool(self.cleared_kv_cache),
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ResetSessionRequest":
    return cls(
        session_id=str(data["session_id"]),
        episode_id=str(data.get("episode_id", "ep-0")),
        instruction_text=str(data.get("instruction_text", "")),
        instruction_tokens=[int(t) for t in data.get("instruction_tokens", [])],
        cleared_kv_cache=bool(data.get("cleared_kv_cache", True)),
    )


@dataclass
class ResetSessionResponse:
  """Payload returned by `POST /v1/reset_session`."""

  session_id: str
  episode_id: str = "ep-0"
  instruction_text: str = ""
  instruction_tokens: List[int] = field(default_factory=list)
  cleared_kv_cache: bool = True

  def to_dict(self) -> Dict[str, Any]:
    return {
        "session_id": self.session_id,
        "episode_id": self.episode_id,
        "instruction_text": self.instruction_text,
        "instruction_tokens": [int(t) for t in self.instruction_tokens],
        "cleared_kv_cache": bool(self.cleared_kv_cache),
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ResetSessionResponse":
    return cls(
        session_id=str(data["session_id"]),
        episode_id=str(data.get("episode_id", "ep-0")),
        instruction_text=str(data.get("instruction_text", "")),
        instruction_tokens=[int(t) for t in data.get("instruction_tokens", [])],
        cleared_kv_cache=bool(data.get("cleared_kv_cache", True)),
    )


@dataclass
class ActRequest:
  """Payload sent to `POST /v1/act` for Track 1 Real-Time VLA Policy evaluation."""

  session_id: str
  step_idx: int
  chunk_size: int = 8
  action_dim: int = 7
  is_perturbed_g4: bool = False
  rgb_shm: Optional[ShmTensorDescriptor] = None
  rgb_b64: Optional[str] = None
  proprio_state: List[float] = field(default_factory=list)
  instruction_tokens: List[int] = field(default_factory=list)
  rgb_shape: Optional[Tuple[int, ...]] = None
  rgb_dtype: str = "float32"
  execution_id: Optional[str] = None

  def to_dict(self) -> Dict[str, Any]:
    return {
        "session_id": self.session_id,
        "step_idx": int(self.step_idx),
        "chunk_size": int(self.chunk_size),
        "action_dim": int(self.action_dim),
        "is_perturbed_g4": bool(self.is_perturbed_g4),
        "rgb_shm": self.rgb_shm.to_dict() if self.rgb_shm is not None else None,
        "rgb_b64": self.rgb_b64,
        "proprio_state": [float(x) for x in self.proprio_state],
        "instruction_tokens": [int(t) for t in self.instruction_tokens],
        "rgb_shape": [int(d) for d in self.rgb_shape] if self.rgb_shape is not None else None,
        "rgb_dtype": str(self.rgb_dtype),
        "execution_id": self.execution_id,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ActRequest":
    raw_shm = data.get("rgb_shm")
    raw_shape = data.get("rgb_shape")
    return cls(
        session_id=str(data["session_id"]),
        step_idx=int(data["step_idx"]),
        chunk_size=int(data.get("chunk_size", 8)),
        action_dim=int(data.get("action_dim", 7)),
        is_perturbed_g4=bool(data.get("is_perturbed_g4", False)),
        rgb_shm=ShmTensorDescriptor.from_dict(raw_shm) if isinstance(raw_shm, dict) else None,
        rgb_b64=data.get("rgb_b64"),
        proprio_state=[float(x) for x in data.get("proprio_state", [])],
        instruction_tokens=[int(t) for t in data.get("instruction_tokens", [])],
        rgb_shape=tuple(int(d) for d in raw_shape) if raw_shape is not None else None,
        rgb_dtype=str(data.get("rgb_dtype", "float32")),
        execution_id=data.get("execution_id"),
    )


@dataclass
class ActResponse:
  """Payload returned by `POST /v1/act` for Track 1 VLA Policy evaluation."""

  actions: List[List[float]]
  server_compute_ms: float
  kv_prefix_cache_hit: bool
  execution_id: str
  session_id: str = ""
  step_idx: int = 0

  def actions_array(self) -> np.ndarray:
    return np.asarray(self.actions, dtype=np.float32)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "actions": [[float(v) for v in row] for row in self.actions],
        "server_compute_ms": float(self.server_compute_ms),
        "kv_prefix_cache_hit": bool(self.kv_prefix_cache_hit),
        "execution_id": str(self.execution_id),
        "session_id": str(self.session_id),
        "step_idx": int(self.step_idx),
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ActResponse":
    return cls(
        actions=[[float(v) for v in row] for row in data["actions"]],
        server_compute_ms=float(data["server_compute_ms"]),
        kv_prefix_cache_hit=bool(data["kv_prefix_cache_hit"]),
        execution_id=str(data["execution_id"]),
        session_id=str(data.get("session_id", "")),
        step_idx=int(data.get("step_idx", 0)),
    )


@dataclass
class WorldStepRequest:
  """Payload sent to `POST /v1/world_step` for Track 2 Interactive World Model / Omni evaluation."""

  session_id: str
  step_idx: int
  horizon_frames: int = 16
  is_perturbed_g4: bool = False
  obs_shm: Optional[ShmTensorDescriptor] = None
  obs_b64: Optional[str] = None
  cond_actions: List[List[float]] = field(default_factory=list)
  obs_shape: Optional[Tuple[int, ...]] = None
  obs_dtype: str = "float32"
  execution_id: Optional[str] = None

  def to_dict(self) -> Dict[str, Any]:
    return {
        "session_id": self.session_id,
        "step_idx": int(self.step_idx),
        "horizon_frames": int(self.horizon_frames),
        "is_perturbed_g4": bool(self.is_perturbed_g4),
        "obs_shm": self.obs_shm.to_dict() if self.obs_shm is not None else None,
        "obs_b64": self.obs_b64,
        "cond_actions": [[float(v) for v in row] for row in self.cond_actions],
        "obs_shape": [int(d) for d in self.obs_shape] if self.obs_shape is not None else None,
        "obs_dtype": str(self.obs_dtype),
        "execution_id": self.execution_id,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "WorldStepRequest":
    raw_shm = data.get("obs_shm")
    raw_shape = data.get("obs_shape")
    return cls(
        session_id=str(data["session_id"]),
        step_idx=int(data["step_idx"]),
        horizon_frames=int(data.get("horizon_frames", 16)),
        is_perturbed_g4=bool(data.get("is_perturbed_g4", False)),
        obs_shm=ShmTensorDescriptor.from_dict(raw_shm) if isinstance(raw_shm, dict) else None,
        obs_b64=data.get("obs_b64"),
        cond_actions=[[float(v) for v in row] for row in data.get("cond_actions", [])],
        obs_shape=tuple(int(d) for d in raw_shape) if raw_shape is not None else None,
        obs_dtype=str(data.get("obs_dtype", "float32")),
        execution_id=data.get("execution_id"),
    )


@dataclass
class WorldStepResponse:
  """Payload returned by `POST /v1/world_step` for Track 2 World Model / Omni evaluation."""

  pred_video_latents: List[List[float]]
  pred_latent_actions: List[List[float]]
  server_compute_ms: float
  dit_cache_hit: bool
  execution_id: str
  session_id: str = ""
  step_idx: int = 0

  def video_latents_array(self) -> np.ndarray:
    return np.asarray(self.pred_video_latents, dtype=np.float32)

  def latent_actions_array(self) -> np.ndarray:
    return np.asarray(self.pred_latent_actions, dtype=np.float32)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "pred_video_latents": [[float(v) for v in row] for row in self.pred_video_latents],
        "pred_latent_actions": [[float(v) for v in row] for row in self.pred_latent_actions],
        "server_compute_ms": float(self.server_compute_ms),
        "dit_cache_hit": bool(self.dit_cache_hit),
        "execution_id": str(self.execution_id),
        "session_id": str(self.session_id),
        "step_idx": int(self.step_idx),
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "WorldStepResponse":
    return cls(
        pred_video_latents=[[float(v) for v in row] for row in data["pred_video_latents"]],
        pred_latent_actions=[[float(v) for v in row] for row in data["pred_latent_actions"]],
        server_compute_ms=float(data["server_compute_ms"]),
        dit_cache_hit=bool(data["dit_cache_hit"]),
        execution_id=str(data["execution_id"]),
        session_id=str(data.get("session_id", "")),
        step_idx=int(data.get("step_idx", 0)),
    )


# =============================================================================
# Deterministic Reference Observation-to-Action / Observation-to-Latent Kernels
# =============================================================================


def compute_reference_vla_actions(
    obs: np.ndarray,
    chunk_size: int = 8,
    action_dim: int = 7,
    proprio_state: Optional[Sequence[float]] = None,
    instruction_tokens: Optional[Sequence[int]] = None,
) -> np.ndarray:
  """Compute deterministic continuous VLA actions `(chunk_size, action_dim)` directly from `obs`.

  Designed so any $\\ell_\\infty = 10^{-3}$ observation perturbation $\\delta$ produces a smooth,
  physically proportional output shift $\\|a(x + \\delta) - a(x)\\|_2 > 10^{-4}$ with Jacobian
  cosine alignment $\\approx 1.0$.
  """
  k = max(1, int(chunk_size))
  d = max(1, int(action_dim))
  flat = np.asarray(obs, dtype=np.float64).ravel()
  if flat.size == 0:
    flat = np.zeros(64, dtype=np.float64)

  out_dim = k * d
  n = flat.size
  # Partition observation into `out_dim` strided pools so every pixel contributes linearly
  padded_len = ((n + out_dim - 1) // out_dim) * out_dim
  if padded_len > n:
    padded = np.zeros(padded_len, dtype=np.float64)
    padded[:n] = flat
  else:
    padded = flat
  pooled = padded.reshape(out_dim, -1).mean(axis=1)
  global_mean = float(np.mean(flat))

  idx = np.arange(out_dim, dtype=np.float64)
  phase = 0.15 * np.sin((idx + 1.0) * 0.37)
  if proprio_state:
    p_arr = np.asarray(proprio_state, dtype=np.float64).ravel()
    p_bias = float(np.mean(p_arr)) * 0.05 if p_arr.size > 0 else 0.0
  else:
    p_bias = 0.0
  if instruction_tokens:
    t_arr = np.asarray(instruction_tokens, dtype=np.float64).ravel()
    t_bias = float(np.mean(t_arr % 31.0)) * 0.002 if t_arr.size > 0 else 0.0
  else:
    t_bias = 0.0

  # Linear + mildly nonlinear combination with non-vanishing gradient wrt every observation pool
  raw = 0.65 * pooled + 0.45 * global_mean + phase + p_bias + t_bias
  return raw.reshape(k, d).astype(np.float32)


def compute_reference_wm_rollout(
    obs: np.ndarray,
    horizon_frames: int = 16,
    cond_actions: Optional[Sequence[Sequence[float]]] = None,
    latent_dim: int = 16,
    action_dim: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
  """Compute deterministic World Model `(pred_video_latents, pred_latent_actions)` from `obs`."""
  h = max(1, int(horizon_frames))
  ldim = max(1, int(latent_dim))
  adim = max(1, int(action_dim))
  flat = np.asarray(obs, dtype=np.float64).ravel()
  if flat.size == 0:
    flat = np.zeros(64, dtype=np.float64)

  total_vid = h * ldim
  padded_len = ((flat.size + total_vid - 1) // total_vid) * total_vid
  padded = np.zeros(padded_len, dtype=np.float64)
  padded[: flat.size] = flat
  pooled_vid = padded.reshape(h, ldim, -1).mean(axis=2)
  global_mean = float(np.mean(flat))

  if cond_actions:
    cond_arr = np.asarray(cond_actions, dtype=np.float64)
    cond_bias = float(np.mean(cond_arr)) * 0.08 if cond_arr.size > 0 else 0.0
  else:
    cond_bias = 0.0

  frame_idx = np.arange(h, dtype=np.float64)[:, None]
  ch_idx = np.arange(ldim, dtype=np.float64)[None, :]
  base_grid = 0.50 + 0.12 * np.sin(0.25 * (frame_idx + 1.0) + 0.19 * ch_idx)
  video_latents = np.clip(
      base_grid + 0.55 * pooled_vid + 0.35 * global_mean + cond_bias,
      0.01,
      0.99,
  ).astype(np.float32)

  act_idx = np.arange(adim, dtype=np.float64)[None, :]
  pooled_act = pooled_vid[:, : min(ldim, adim)]
  if pooled_act.shape[1] < adim:
    pooled_act = np.pad(pooled_act, ((0, 0), (0, adim - pooled_act.shape[1])), mode="wrap")
  latent_actions = (
      0.60 * pooled_act
      + 0.40 * global_mean
      + 0.10 * np.cos(0.31 * (frame_idx + 1.0) + 0.23 * act_idx)
      + cond_bias
  ).astype(np.float32)
  return video_latents, latent_actions


def encode_tensor_b64(arr: np.ndarray) -> Tuple[str, Tuple[int, ...], str]:
  """Encode a NumPy tensor as base64 string + shape + dtype for non-SHM fallback."""
  contiguous = np.ascontiguousarray(arr)
  encoded = base64.b64encode(contiguous.tobytes()).decode("ascii")
  return encoded, tuple(int(d) for d in contiguous.shape), str(contiguous.dtype)


def decode_tensor_b64(
    encoded: str,
    shape: Sequence[int],
    dtype: str = "float32",
) -> np.ndarray:
  """Decode a base64 tensor payload into a NumPy array with strict 64 MiB bounds."""
  if not isinstance(shape, (list, tuple)) or len(shape) > 8:
    raise ValueError("Invalid tensor shape list.")
  dims = tuple(int(d) for d in shape)
  if any(d < 0 for d in dims):
    raise ValueError(f"Negative dimension in tensor shape {dims}.")
  dt = np.dtype(dtype)
  expected_bytes = int(math.prod(dims)) * int(dt.itemsize)
  if expected_bytes > MAX_HTTP_BODY_BYTES:
    raise ValueError(
        f"Declared tensor shape {dims} ({expected_bytes} bytes) exceeds 64 MiB limit ({MAX_HTTP_BODY_BYTES})."
    )
  if not isinstance(encoded, str) or len(encoded) > (MAX_HTTP_BODY_BYTES * 4 // 3) + 4096:
    raise ValueError("Encoded base64 tensor exceeds 64 MiB limit.")
  raw = base64.b64decode(encoded.encode("ascii"), validate=True)
  if len(raw) != expected_bytes:
    raise ValueError(
        f"Decoded tensor byte length {len(raw)} does not match shape {dims} ({expected_bytes} bytes)."
    )
  arr = np.frombuffer(raw, dtype=dt).reshape(dims)
  return arr.copy()


# =============================================================================
# Reference Loopback Embodied Server Handler & HTTP Server / Client
# =============================================================================


class LoopbackEmbodiedServerHandler:
  """Stateful reference server handler backing `/v1/health`, `/v1/reset_session`, `/v1/act`, and `/v1/world_step`.

  Tracks per-`session_id` KV prefix / DiT feature cache state (`kv_prefix_cache_hit = (step_idx > 0)`),
  reads observations zero-copy from `SharedMemoryTensorRing`, and supports `cheating_static_memoizer=True`
  and `num_devices` overrides for negative-control compliance testing.
  """

  def __init__(
      self,
      model_id: str = "pi05-3b",
      track: str = TrackType.VLA_TRACK.value,
      hardware_sku: str = "TPU_V6E_1_32GB",
      num_devices: int = 1,
      precision: str = "FP8",
      server_version: str = "1.0.0",
      ready: bool = True,
      execution_id: Optional[str] = None,
      cheating_static_memoizer: bool = False,
      quality_noise_scale: float = 0.0,
      simulated_compute_ms: float = 0.15,
      act_callback: Optional[Callable[[ActRequest, np.ndarray], np.ndarray]] = None,
      world_step_callback: Optional[
          Callable[[WorldStepRequest, np.ndarray], Tuple[np.ndarray, np.ndarray]]
      ] = None,
      shm_ring: Optional[SharedMemoryTensorRing] = None,
      allowed_shm_names: Optional[Sequence[str]] = None,
  ) -> None:
    self.model_id = str(model_id)
    self.track = str(track)
    self.hardware_sku = str(hardware_sku)
    self.num_devices = int(num_devices)
    self.precision = str(precision)
    self.server_version = str(server_version)
    self.ready = bool(ready)
    self.execution_id = execution_id or f"exec-live-{uuid.uuid4().hex[:12]}"
    self.cheating_static_memoizer = bool(cheating_static_memoizer)
    self.quality_noise_scale = float(quality_noise_scale)
    self.simulated_compute_ms = float(simulated_compute_ms)
    self.act_callback = act_callback
    self.world_step_callback = world_step_callback

    self._lock = threading.Lock()
    self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    self._attached_rings: OrderedDict[str, SharedMemoryTensorRing] = OrderedDict()
    self._allowed_shm_names: set[str] = {
        _sanitize_shm_name(n) for n in (allowed_shm_names or ())
    }
    if shm_ring is not None:
      self._attached_rings[shm_ring.name] = shm_ring
      self._allowed_shm_names.add(shm_ring.name)

  def register_shm_ring(self, ring: SharedMemoryTensorRing) -> None:
    """Explicitly authorize and cache a `SharedMemoryTensorRing` on this server."""
    with self._lock:
      self._allowed_shm_names.add(ring.name)
      self._attached_rings[ring.name] = ring
      self._attached_rings.move_to_end(ring.name)

  def _upsert_session_locked(self, session_id: str, session: Dict[str, Any]) -> None:
    if session_id in self._sessions:
      del self._sessions[session_id]
    self._sessions[session_id] = session
    while len(self._sessions) > MAX_ACTIVE_SESSIONS:
      self._sessions.popitem(last=False)

  def _resolve_ring(self, shm_name: str) -> SharedMemoryTensorRing:
    sanitized = _sanitize_shm_name(shm_name)
    with self._lock:
      ring = self._attached_rings.get(sanitized)
      if ring is not None:
        self._attached_rings.move_to_end(sanitized)
        return ring
      with _PROCESS_OWNED_SHM_LOCK:
        is_local_ring = sanitized in _PROCESS_OWNED_SHM_NAMES
      if sanitized not in self._allowed_shm_names and not is_local_ring:
        raise PermissionError(
            f"Untrusted shm_name {sanitized!r} is not registered on this server (RT-06). "
            "Pass shm_ring=... or allowed_shm_names=[...] explicitly, or transmit tensors over HTTP base64."
        )
      ring = SharedMemoryTensorRing(name=sanitized, create=False)
      self._attached_rings[sanitized] = ring
      while len(self._attached_rings) > MAX_ATTACHED_SHM_RINGS:
        _, evicted = self._attached_rings.popitem(last=False)
        if not evicted._owns_memory:
          evicted.close()
      return ring

  def _read_observation(
      self,
      shm_desc: Optional[ShmTensorDescriptor],
      b64_payload: Optional[str],
      b64_shape: Optional[Sequence[int]],
      b64_dtype: str,
  ) -> np.ndarray:
    if shm_desc is not None:
      ring = self._resolve_ring(shm_desc.shm_name)
      return ring.read_tensor(shm_desc, verify_checksum=True)
    if b64_payload is not None and b64_shape is not None:
      return decode_tensor_b64(b64_payload, b64_shape, b64_dtype)
    return np.ones((16, 16, 3), dtype=np.float32) * 0.5

  def handle_health(self) -> HealthResponse:
    return HealthResponse(
        model_id=self.model_id,
        track=self.track,
        hardware_sku=self.hardware_sku,
        num_devices=self.num_devices,
        precision=self.precision,
        server_version=self.server_version,
        ready=self.ready,
    )

  def handle_reset_session(self, req: ResetSessionRequest) -> ResetSessionResponse:
    with self._lock:
      self._upsert_session_locked(
          req.session_id,
          {
              "episode_id": req.episode_id,
              "instruction_text": req.instruction_text,
              "instruction_tokens": list(req.instruction_tokens),
              "steps_seen": 0,
              "memoized_actions": None,
              "memoized_video": None,
              "memoized_latent_actions": None,
          },
      )
    return ResetSessionResponse(
        session_id=req.session_id,
        episode_id=req.episode_id,
        instruction_text=req.instruction_text,
        instruction_tokens=list(req.instruction_tokens),
        cleared_kv_cache=True,
    )

  def handle_act(self, req: ActRequest) -> ActResponse:
    t0_ns = time.perf_counter_ns()
    obs = self._read_observation(
        req.rgb_shm,
        req.rgb_b64,
        req.rgb_shape,
        req.rgb_dtype,
    )

    with self._lock:
      session = self._sessions.get(req.session_id)
      if session is None:
        session = {
            "episode_id": "auto-ep",
            "instruction_text": "",
            "instruction_tokens": list(req.instruction_tokens),
            "steps_seen": 0,
            "memoized_actions": None,
            "memoized_video": None,
            "memoized_latent_actions": None,
        }
        self._upsert_session_locked(req.session_id, session)
      else:
        self._sessions.move_to_end(req.session_id)
      kv_hit = bool(req.step_idx > 0 and session["steps_seen"] > 0)
      session["steps_seen"] += 1
      memoized = session.get("memoized_actions")

    if self.cheating_static_memoizer and memoized is not None:
      actions_arr = memoized
    else:
      if self.act_callback is not None:
        actions_arr = np.asarray(self.act_callback(req, obs), dtype=np.float32)
      else:
        actions_arr = compute_reference_vla_actions(
            obs=obs,
            chunk_size=req.chunk_size,
            action_dim=req.action_dim,
            proprio_state=req.proprio_state,
            instruction_tokens=req.instruction_tokens,
        )
      if self.quality_noise_scale > 0.0:
        idx = np.arange(actions_arr.size, dtype=np.float32).reshape(actions_arr.shape)
        actions_arr = actions_arr + np.sin(idx + 1.0) * float(self.quality_noise_scale)
      if self.cheating_static_memoizer:
        with self._lock:
          self._sessions[req.session_id]["memoized_actions"] = actions_arr.copy()

    elapsed_ms = (time.perf_counter_ns() - t0_ns) / 1e6 + self.simulated_compute_ms
    exec_id = req.execution_id or self.execution_id
    return ActResponse(
        actions=actions_arr.tolist(),
        server_compute_ms=round(float(elapsed_ms), 4),
        kv_prefix_cache_hit=kv_hit,
        execution_id=exec_id,
        session_id=req.session_id,
        step_idx=req.step_idx,
    )

  def handle_world_step(self, req: WorldStepRequest) -> WorldStepResponse:
    t0_ns = time.perf_counter_ns()
    obs = self._read_observation(
        req.obs_shm,
        req.obs_b64,
        req.obs_shape,
        req.obs_dtype,
    )

    with self._lock:
      session = self._sessions.get(req.session_id)
      if session is None:
        session = {
            "episode_id": "auto-ep",
            "instruction_text": "",
            "instruction_tokens": [],
            "steps_seen": 0,
            "memoized_actions": None,
            "memoized_video": None,
            "memoized_latent_actions": None,
        }
        self._upsert_session_locked(req.session_id, session)
      else:
        self._sessions.move_to_end(req.session_id)
      dit_hit = bool(req.step_idx > 0 and session["steps_seen"] > 0)
      session["steps_seen"] += 1
      memo_vid = session.get("memoized_video")
      memo_la = session.get("memoized_latent_actions")

    if self.cheating_static_memoizer and memo_vid is not None and memo_la is not None:
      video_arr, la_arr = memo_vid, memo_la
    else:
      if self.world_step_callback is not None:
        video_arr, la_arr = self.world_step_callback(req, obs)
        video_arr = np.asarray(video_arr, dtype=np.float32)
        la_arr = np.asarray(la_arr, dtype=np.float32)
      else:
        video_arr, la_arr = compute_reference_wm_rollout(
            obs=obs,
            horizon_frames=req.horizon_frames,
            cond_actions=req.cond_actions,
        )
      if self.quality_noise_scale > 0.0:
        video_arr = np.clip(video_arr + float(self.quality_noise_scale) * 0.2, 0.01, 0.99)
      if self.cheating_static_memoizer:
        with self._lock:
          self._sessions[req.session_id]["memoized_video"] = video_arr.copy()
          self._sessions[req.session_id]["memoized_latent_actions"] = la_arr.copy()

    elapsed_ms = (time.perf_counter_ns() - t0_ns) / 1e6 + self.simulated_compute_ms
    exec_id = req.execution_id or self.execution_id
    return WorldStepResponse(
        pred_video_latents=video_arr.tolist(),
        pred_latent_actions=la_arr.tolist(),
        server_compute_ms=round(float(elapsed_ms), 4),
        dit_cache_hit=dit_hit,
        execution_id=exec_id,
        session_id=req.session_id,
        step_idx=req.step_idx,
    )

  def close(self) -> None:
    with self._lock:
      for ring in self._attached_rings.values():
        if not ring._owns_memory:
          ring.close()
      self._attached_rings.clear()


class _EmbodiedHTTPRequestHandler(BaseHTTPRequestHandler):
  """HTTP/1.1 persistent keep-alive request handler bound to `EmbodiedHTTPServer`."""

  protocol_version = "HTTP/1.1"

  def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
    # Suppress noisy per-request stderr logging during high-frequency loadgen sweeps
    return

  def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    self.send_response(status_code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Connection", "keep-alive")
    self.end_headers()
    self.wfile.write(body)

  def _read_json_body(self) -> Dict[str, Any]:
    raw_cl = self.headers.get("Content-Length")
    if raw_cl is None:
      return {}
    try:
      length = int(raw_cl)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"Invalid Content-Length header {raw_cl!r}.") from exc
    if length < 0:
      raise ValueError(f"Negative Content-Length header {length}.")
    if length == 0:
      return {}
    if length > MAX_HTTP_BODY_BYTES:
      raise _PayloadTooLargeError(
          f"Request body length {length} exceeds maximum {MAX_HTTP_BODY_BYTES} bytes."
      )
    raw = self.rfile.read(length)
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
      raise ValueError("JSON request body must be an object.")
    return parsed

  def do_GET(self) -> None:  # noqa: N802
    srv: EmbodiedHTTPServer = self.server  # type: ignore[assignment]
    if self.path == "/v1/health":
      try:
        resp = srv.embodied_handler.handle_health()
        self._send_json(200, resp.to_dict())
      except Exception as exc:  # pragma: no cover
        self._send_json(500, {"error": str(exc)})
    else:
      self._send_json(404, {"error": f"Unknown GET endpoint {self.path!r}"})

  def do_POST(self) -> None:  # noqa: N802
    srv: EmbodiedHTTPServer = self.server  # type: ignore[assignment]
    try:
      data = self._read_json_body()
      if self.path == "/v1/reset_session":
        req = ResetSessionRequest.from_dict(data)
        resp = srv.embodied_handler.handle_reset_session(req)
        self._send_json(200, resp.to_dict())
      elif self.path == "/v1/act":
        req = ActRequest.from_dict(data)
        resp = srv.embodied_handler.handle_act(req)
        self._send_json(200, resp.to_dict())
      elif self.path == "/v1/world_step":
        req = WorldStepRequest.from_dict(data)
        resp = srv.embodied_handler.handle_world_step(req)
        self._send_json(200, resp.to_dict())
      else:
        self._send_json(404, {"error": f"Unknown POST endpoint {self.path!r}"})
    except _PayloadTooLargeError as exc:
      self._send_json(413, {"error": str(exc)})
    except PermissionError as exc:
      self._send_json(403, {"error": str(exc)})
    except Exception as exc:
      self._send_json(400, {"error": str(exc)})


class EmbodiedHTTPServer(ThreadingHTTPServer):
  """Zero-dependency multi-threaded HTTP/1.1 keep-alive server for EmbodiedX model evaluation."""

  daemon_threads = True
  allow_reuse_address = True

  def __init__(
      self,
      handler: Optional[LoopbackEmbodiedServerHandler] = None,
      host: str = "127.0.0.1",
      port: int = 0,
  ) -> None:
    self.embodied_handler = handler or LoopbackEmbodiedServerHandler()
    super().__init__((host, int(port)), _EmbodiedHTTPRequestHandler)
    self.host: str = str(self.server_address[0])
    self.port: int = int(self.server_address[1])
    self._thread: Optional[threading.Thread] = None
    self._stopped: bool = False

  @property
  def base_url(self) -> str:
    return f"http://{self.host}:{self.port}"

  def start_in_background(self) -> "EmbodiedHTTPServer":
    if self._thread is not None and self._thread.is_alive():
      return self
    self._thread = threading.Thread(
        target=self.serve_forever,
        kwargs={"poll_interval": 0.02},
        name=f"EmbodiedHTTPServer-{self.port}",
        daemon=True,
    )
    self._thread.start()
    return self

  def close(self) -> None:
    if self._stopped:
      return
    self._stopped = True
    try:
      if self._thread is not None and self._thread.is_alive():
        self.shutdown()
        self._thread.join(timeout=2.0)
    finally:
      self.server_close()
      self.embodied_handler.close()

  def __enter__(self) -> "EmbodiedHTTPServer":
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()


def start_background_embodied_server(
    handler: Optional[LoopbackEmbodiedServerHandler] = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> EmbodiedHTTPServer:
  """Start an `EmbodiedHTTPServer` on a background daemon thread and return the server instance."""
  server = EmbodiedHTTPServer(handler=handler, host=host, port=port)
  return server.start_in_background()


class EmbodiedHTTPClient:
  """Persistent keep-alive HTTP/1.1 client backed by per-thread `http.client.HTTPConnection`."""

  def __init__(
      self,
      host: str = "127.0.0.1",
      port: int = 8000,
      base_url: Optional[str] = None,
      timeout: float = 30.0,
  ) -> None:
    if base_url is not None or host.startswith("http://") or host.startswith("https://"):
      target_url = base_url or host
      parsed = urlparse(target_url)
      self.host = parsed.hostname or "127.0.0.1"
      self.port = int(parsed.port or port)
    else:
      self.host = str(host)
      self.port = int(port)
    self.timeout = float(timeout)
    self._local = threading.local()
    self._conn_lock = threading.Lock()
    self._all_connections: List[http.client.HTTPConnection] = []

  @property
  def base_url(self) -> str:
    return f"http://{self.host}:{self.port}"

  def _get_connection(self, force_new: bool = False) -> http.client.HTTPConnection:
    conn: Optional[http.client.HTTPConnection] = getattr(self._local, "conn", None)
    if conn is None or force_new:
      if conn is not None:
        try:
          conn.close()
        except Exception:
          pass
      conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
      self._local.conn = conn
      with self._conn_lock:
        self._all_connections.append(conn)
    return conn

  def _request_json(
      self,
      method: str,
      path: str,
      payload: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Connection": "keep-alive",
        "Accept": "application/json",
    }
    if body is not None:
      headers["Content-Type"] = "application/json"
      headers["Content-Length"] = str(len(body))

    for attempt in range(2):
      conn = self._get_connection(force_new=(attempt > 0))
      try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
        if resp.status >= 400:
          err_msg = parsed.get("error", f"HTTP {resp.status}")
          raise RuntimeError(f"{method} {path} failed with HTTP {resp.status}: {err_msg}")
        return parsed
      except (
          http.client.HTTPException,
          ConnectionError,
          BrokenPipeError,
          OSError,
      ):
        if attempt == 1:
          raise
    raise RuntimeError(f"Unreachable retry failure on {method} {path}")  # pragma: no cover

  def health(self) -> HealthResponse:
    return HealthResponse.from_dict(self._request_json("GET", "/v1/health"))

  def reset_session(self, req: ResetSessionRequest | str, **kwargs: Any) -> ResetSessionResponse:
    if isinstance(req, str):
      req_obj = ResetSessionRequest(session_id=req, **kwargs)
    else:
      req_obj = req
    return ResetSessionResponse.from_dict(
        self._request_json("POST", "/v1/reset_session", req_obj.to_dict())
    )

  def act(self, req: ActRequest) -> ActResponse:
    return ActResponse.from_dict(self._request_json("POST", "/v1/act", req.to_dict()))

  def world_step(self, req: WorldStepRequest) -> WorldStepResponse:
    return WorldStepResponse.from_dict(
        self._request_json("POST", "/v1/world_step", req.to_dict())
    )

  def close(self) -> None:
    with self._conn_lock:
      for conn in self._all_connections:
        try:
          conn.close()
        except Exception:
          pass
      self._all_connections.clear()
    if hasattr(self._local, "conn"):
      self._local.conn = None

  def __enter__(self) -> "EmbodiedHTTPClient":
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()
