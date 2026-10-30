"""Base runner configuration, single-chip CAP_SINGLE_CHIP enforcement, abstract server interface, HTTP loopback wrapper, and factory for EmbodiedX v1.0 Reference Model Servers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from embodied_bench.schema import (
    ExecutionStrategy,
    HARDWARE_SKU_SPECS,
    MODERN_REFERENCE_CHECKPOINTS,
    PerformanceCapId,
    now_pacific_iso,
)

MAX_HTTP_BODY_BYTES: int = 64 * 1024 * 1024
MAX_ACTIVE_SESSIONS: int = 1024


# Canonical alias table mapping verified HF repo names and shorthand IDs to challenge models
CANONICAL_MODEL_ALIASES: Dict[str, str] = {
    "gr00t-n1.7-3b": "gr00t-n1.7-3b",
    "gr00t-n1.7": "gr00t-n1.7-3b",
    "gr00t": "gr00t-n1.7-3b",
    "GR00T-N1.7-3B": "gr00t-n1.7-3b",
    "nvidia/GR00T-N1.7-3B": "gr00t-n1.7-3b",
    "nvidia/GR00T-N1.7-LIBERO": "gr00t-n1.7-3b",
    "pi05-3b": "pi05-3b",
    "pi05": "pi05-3b",
    "pi0.5": "pi05-3b",
    "pi0-3b": "pi05-3b",
    "pi0-3.3B": "pi05-3b",
    "openpi-pi05-3b": "pi05-3b",
    "lerobot/pi05_libero_finetuned_v044": "pi05-3b",
    "lerobot/pi05_base": "pi05-3b",
    "PhysicalIntelligence/pi0.5-libero": "pi05-3b",
    "cosmos-predict2.5-2b": "cosmos-predict2.5-2b",
    "cosmos-predict2.5": "cosmos-predict2.5-2b",
    "cosmos": "cosmos-predict2.5-2b",
    "nvidia/Cosmos-Predict2.5-2B": "cosmos-predict2.5-2b",
    "cosmos3-edge": "cosmos3-edge",
    "cosmos3": "cosmos3-edge",
    "Cosmos3-Edge": "cosmos3-edge",
    "qwen3-omni-30b-a3b": "qwen3-omni-30b-a3b",
    "qwen3-omni": "qwen3-omni-30b-a3b",
    "Qwen3-Omni-30B-A3B": "qwen3-omni-30b-a3b",
    "Qwen3-Omni-30B-A3B-Instruct": "qwen3-omni-30b-a3b",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "qwen3-omni-30b-a3b",
    "qwen3-omni-30b-a3b-instruct": "qwen3-omni-30b-a3b",
}

VALID_WEIGHTS_MODES: Tuple[str, ...] = (
    "deterministic-reference",
    "hf-checkpoint",
)

VALID_PRECISIONS: Tuple[str, ...] = (
    "FP32",
    "BF16",
    "FP16",
    "FP8",
    "FP8_E4M3",
    "INT8",
    "INT4",
    "W4A16",
    "W4A8",
    "NVFP4",
)


def resolve_canonical_model_id(model_id: str) -> str:
  """Resolve a model_id or Hugging Face repo path to its canonical challenge model ID."""
  if not model_id or not str(model_id).strip():
    raise ValueError(
        "model_id must be a non-empty string. Choose one of: "
        f"{sorted(set(CANONICAL_MODEL_ALIASES.values()))}."
    )
  raw = str(model_id).strip()
  if raw in CANONICAL_MODEL_ALIASES:
    return CANONICAL_MODEL_ALIASES[raw]
  for cid, spec in MODERN_REFERENCE_CHECKPOINTS.items():
    if raw == cid or raw == spec.get("hf_repo") or raw in spec.get("aliases", []):
      return cid
  valid = sorted(set(CANONICAL_MODEL_ALIASES.values()))
  raise ValueError(
      f"Unsupported model_id={model_id!r} for reference runner. "
      f"Supported canonical models: {valid}."
  )


@dataclass
class EmbodiedRunnerConfig:
  """Configuration for EmbodiedX v1.0 single-chip reference model runners."""

  model_id: str = "gr00t-n1.7-3b"
  hardware_sku: str = "RTX_5090_32GB"
  precision: str = "BF16"
  num_devices: int = 1
  num_inference_steps: Optional[int] = None
  enable_prefix_cache: bool = True
  enable_dit_cache: bool = True
  dit_cache_threshold: float = 0.18
  execution_strategy: str = ExecutionStrategy.RTC_INPAINTING.value
  weights_mode: str = "deterministic-reference"
  vllm_endpoint: Optional[str] = None
  action_dim: int = 7
  seed: int = 20261008

  def __post_init__(self) -> None:
    self.validate()

  @property
  def canonical_model_id(self) -> str:
    return resolve_canonical_model_id(self.model_id)

  def validate(self) -> None:
    """Enforce CAP_SINGLE_CHIP (`num_devices == 1`), SKU lookup, and valid runner settings."""
    if int(self.num_devices) != 1:
      raise ValueError(
          f"CAP_SINGLE_CHIP violation: num_devices={self.num_devices} in EmbodiedRunnerConfig "
          f"for model_id={self.model_id!r}. EmbodiedX strictly requires single-chip execution "
          "(num_devices == 1). Set num_devices=1."
      )
    env_max_devices = os.environ.get("EEB_MAX_DEVICES")
    if env_max_devices is not None and int(env_max_devices) != 1:
      raise ValueError(
          f"CAP_SINGLE_CHIP environment violation: EEB_MAX_DEVICES={env_max_devices!r} != '1'. "
          "Export EEB_MAX_DEVICES=1 inside single-chip container environments."
      )
    canonical = resolve_canonical_model_id(self.model_id)
    if self.hardware_sku not in HARDWARE_SKU_SPECS:
      valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
      raise ValueError(
          f"Unknown hardware_sku={self.hardware_sku!r}. "
          f"Supported SKUs in HARDWARE_SKU_SPECS: [{valid_skus}]."
      )
    if self.weights_mode not in VALID_WEIGHTS_MODES:
      raise ValueError(
          f"Invalid weights_mode={self.weights_mode!r}. "
          f"Choose from {list(VALID_WEIGHTS_MODES)}."
      )
    if self.precision.upper() not in VALID_PRECISIONS:
      raise ValueError(
          f"Invalid precision={self.precision!r}. "
          f"Supported precisions: {list(VALID_PRECISIONS)}."
      )
    self.precision = self.precision.upper()

    valid_strategies = {e.value for e in ExecutionStrategy}
    if self.execution_strategy not in valid_strategies:
      raise ValueError(
          f"Invalid execution_strategy={self.execution_strategy!r}. "
          f"Supported strategies: {sorted(valid_strategies)}."
      )
    if self.num_inference_steps is None:
      spec = MODERN_REFERENCE_CHECKPOINTS.get(canonical, {})
      self.num_inference_steps = int(spec.get("default_denoising_steps", 16))
    elif int(self.num_inference_steps) < 1:
      raise ValueError(
          f"num_inference_steps must be >= 1, got {self.num_inference_steps}."
      )
    else:
      self.num_inference_steps = int(self.num_inference_steps)

    if float(self.dit_cache_threshold) <= 0.0:
      raise ValueError(
          f"dit_cache_threshold must be > 0.0, got {self.dit_cache_threshold}."
      )
    if int(self.action_dim) < 1:
      raise ValueError(f"action_dim must be >= 1, got {self.action_dim}.")

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    d["canonical_model_id"] = self.canonical_model_id
    return d


class BaseEmbodiedModelRunner(ABC):
  """Abstract base class for the 4 canonical EmbodiedX v1.0 model servers."""

  def __init__(self, config: EmbodiedRunnerConfig) -> None:
    config.validate()
    self.config = config
    self.canonical_model_id = config.canonical_model_id
    self.model_spec = MODERN_REFERENCE_CHECKPOINTS.get(self.canonical_model_id, {})
    self.hw_spec = HARDWARE_SKU_SPECS[config.hardware_sku]
    self._lock = threading.RLock()
    self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()

  @property
  def total_params_b(self) -> float:
    return float(self.model_spec.get("params_b", 3.0))

  @property
  def active_params_b(self) -> float:
    return float(self.model_spec.get("active_params_b", self.total_params_b))

  def health(self) -> Dict[str, Any]:
    """Return server health and CAP_SINGLE_CHIP hardware/model configuration metadata."""
    with self._lock:
      active_sessions = len(self._sessions)
    return {
        "status": "healthy",
        "model_id": self.config.model_id,
        "canonical_model_id": self.canonical_model_id,
        "hardware_sku": self.config.hardware_sku,
        "vendor": str(self.hw_spec.get("vendor", "UNKNOWN")),
        "precision": self.config.precision,
        "num_devices": self.config.num_devices,
        "perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "num_inference_steps": int(self.config.num_inference_steps or 1),
        "enable_prefix_cache": bool(self.config.enable_prefix_cache),
        "enable_dit_cache": bool(self.config.enable_dit_cache),
        "execution_strategy": self.config.execution_strategy,
        "weights_mode": self.config.weights_mode,
        "total_params_b": self.total_params_b,
        "active_params_b": self.active_params_b,
        "active_sessions": active_sessions,
        "updated_at": now_pacific_iso(),
    }

  def reset_session(
      self,
      session_id: str,
      instruction_tokens: Optional[np.ndarray] = None,
  ) -> Dict[str, Any]:
    """Reset or initialize stateful KV/DiT cache for `session_id` without stale field carryover."""
    if not session_id or not str(session_id).strip():
      raise ValueError(
          "session_id must be a non-empty string when calling reset_session."
      )
    sid = str(session_id).strip()
    with self._lock:
      state: Dict[str, Any] = {
          "session_id": sid,
          "step_count": 0,
          "last_step_idx": -1,
          "instruction_signature": None,
          "instruction_kv_cache": None,
          "prev_obs_frames": None,
          "cached_dit_features": None,
          "cached_latent_actions_base": None,
          "kv_prefix_cache_hits": 0,
          "dit_cache_hits": 0,
          "updated_at": now_pacific_iso(),
      }
      if instruction_tokens is not None:
        tok_arr = np.asarray(instruction_tokens, dtype=np.int32)
        state["prewarmed_instruction_tokens"] = tok_arr.copy()
      if sid in self._sessions:
        del self._sessions[sid]
      self._sessions[sid] = state
      while len(self._sessions) > MAX_ACTIVE_SESSIONS:
        self._sessions.popitem(last=False)
    return {
        "session_id": sid,
        "reset": True,
        "kv_prefix_cached": False,
        "step_count": 0,
        "updated_at": now_pacific_iso(),
    }

  def _get_or_create_session(self, session_id: str, step_idx: int) -> Dict[str, Any]:
    """Retrieve session state under lock; auto-reset if step_idx == 0 or session is new."""
    if not session_id or not str(session_id).strip():
      raise ValueError("session_id must be a non-empty string.")
    if int(step_idx) < 0:
      raise ValueError(f"step_idx must be >= 0, got {step_idx}.")
    sid = str(session_id).strip()
    with self._lock:
      if sid not in self._sessions or int(step_idx) == 0:
        self.reset_session(sid)
      else:
        self._sessions.move_to_end(sid)
      return self._sessions[sid]

  @staticmethod
  def _compute_instruction_signature(instruction_tokens: np.ndarray) -> bytes:
    """Compute a fast byte signature of the 24-token language instruction prefix."""
    tok_arr = np.ascontiguousarray(np.asarray(instruction_tokens, dtype=np.int32))
    prefix = tok_arr.reshape(-1)[:24]
    return prefix.tobytes()

  @abstractmethod
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
    """Execute one stateful VLA action chunk prediction step."""

  @abstractmethod
  def step_world(
      self,
      session_id: str,
      step_idx: int,
      obs_frames: np.ndarray,
      cond_actions: np.ndarray,
      horizon_frames: int = 16,
  ) -> Dict[str, Any]:
    """Execute one stateful World Model / Omni Model H-frame rollout step."""


def create_model_runner(config: EmbodiedRunnerConfig) -> BaseEmbodiedModelRunner:
  """Factory routing `EmbodiedRunnerConfig` to the 4 canonical model runners."""
  config.validate()
  canonical = config.canonical_model_id
  if canonical == "gr00t-n1.7-3b":
    from runners.gr00t_n17_server import GR00TN17Runner
    return GR00TN17Runner(config)
  if canonical == "pi05-3b":
    from runners.pi05_server import Pi05Runner
    return Pi05Runner(config)
  if canonical == "cosmos3-edge":
    from runners.cosmos3_edge_server import Cosmos3EdgeRunner
    return Cosmos3EdgeRunner(config)
  if canonical == "qwen3-omni-30b-a3b":
    from runners.qwen3_omni_server import Qwen3OmniRunner
    return Qwen3OmniRunner(config)
  raise ValueError(
      f"No reference model runner registered for canonical_model_id={canonical!r}."
  )


def _decode_array(payload: Dict[str, Any], key: str, default_shape: Tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
  """Decode a JSON list or base64-encoded array from an HTTP request payload with 64 MiB cap."""
  if key not in payload or payload[key] is None:
    return np.zeros(default_shape, dtype=dtype)
  val = payload[key]
  if isinstance(val, dict) and "b64" in val:
    arr_dtype = np.dtype(val.get("dtype", dtype))
    raw_shape = val.get("shape", default_shape)
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) > 8:
      raise ValueError("Invalid array shape definition.")
    shape = tuple(int(dim) for dim in raw_shape)
    if any(dim < 0 for dim in shape):
      raise ValueError(f"Negative dimension in array shape {shape}.")
    expected_bytes = int(math.prod(shape)) * int(arr_dtype.itemsize)
    if expected_bytes > MAX_HTTP_BODY_BYTES:
      raise ValueError(
          f"Declared array shape {shape} ({expected_bytes} bytes) exceeds 64 MiB limit ({MAX_HTTP_BODY_BYTES})."
      )
    b64_str = val["b64"]
    if not isinstance(b64_str, str) or len(b64_str) > (MAX_HTTP_BODY_BYTES * 4 // 3) + 4096:
      raise ValueError("Encoded base64 array exceeds 64 MiB limit.")
    raw = base64.b64decode(b64_str.encode("ascii"), validate=True)
    if len(raw) != expected_bytes:
      raise ValueError(f"Decoded array size {len(raw)} bytes does not match shape {shape} ({expected_bytes} bytes).")
    return np.frombuffer(raw, dtype=arr_dtype).reshape(shape).copy()
  arr = np.asarray(val, dtype=dtype)
  if arr.nbytes > MAX_HTTP_BODY_BYTES:
    raise ValueError(f"Array byte size {arr.nbytes} exceeds 64 MiB limit.")
  return arr


def _serialize_response(resp: Dict[str, Any]) -> Dict[str, Any]:
  """Convert numpy arrays in runner response to JSON-serializable lists + shape metadata."""
  out: Dict[str, Any] = {}
  for k, v in resp.items():
    if isinstance(v, np.ndarray):
      out[k] = v.tolist()
      out[f"{k}_shape"] = list(v.shape)
    elif isinstance(v, np.floating):
      out[k] = float(v)
    elif isinstance(v, np.integer):
      out[k] = int(v)
    else:
      out[k] = v
  return out


def make_runner_http_server(
    runner: BaseEmbodiedModelRunner,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
  """Create a ThreadingHTTPServer exposing `/health`, `/v1/health`, `/v1/reset_session`, `/v1/act`, `/v1/vla/act`, `/v1/world_step`, and `/v1/wm/step`."""
  execution_id = f"exec_runner_{runner.canonical_model_id}_{int(time.time() * 1000)}"

  class _Handler(BaseHTTPRequestHandler):

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
      return

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
      body = json.dumps(_serialize_response(data)).encode("utf-8")
      self.send_response(status_code)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
      if self.path.startswith("/v1/health") or self.path == "/health":
        health_payload = dict(runner.health())
        health_payload.setdefault("status", "ok")
        health_payload.setdefault("execution_id", execution_id)
        self._send_json(200, health_payload)
      else:
        self._send_json(404, {"error": f"Unknown endpoint {self.path!r}"})

    def do_POST(self) -> None:  # noqa: N802
      raw_cl = self.headers.get("Content-Length")
      if raw_cl is None:
        length = 0
      else:
        try:
          length = int(raw_cl)
        except (TypeError, ValueError):
          self._send_json(400, {"error": f"Invalid Content-Length header: {raw_cl!r}"})
          return
        if length < 0:
          self._send_json(400, {"error": f"Negative Content-Length header: {length}"})
          return
        if length > MAX_HTTP_BODY_BYTES:
          self._send_json(413, {"error": f"Payload length {length} exceeds maximum {MAX_HTTP_BODY_BYTES} bytes."})
          return
      raw = self.rfile.read(length) if length > 0 else b"{}"
      try:
        payload = json.loads(raw.decode("utf-8"))
        obs_src = (
            payload["observations"]
            if isinstance(payload.get("observations"), dict)
            else payload
        )
        if self.path == "/v1/reset_session":
          inst_raw = obs_src.get("instruction_tokens", payload.get("instruction_tokens"))
          inst = (
              np.asarray(inst_raw, dtype=np.int32)
              if inst_raw is not None
              else None
          )
          res = runner.reset_session(
              session_id=str(payload.get("session_id", "default")),
              instruction_tokens=inst,
          )
          res["status"] = "ok"
          res["execution_id"] = execution_id
          self._send_json(200, res)
        elif self.path in ("/v1/act", "/v1/vla/act"):
          rgb_source = obs_src if "rgb_frames" in obs_src else payload
          prop_source = obs_src if "proprio_state" in obs_src else payload
          inst_source = obs_src if "instruction_tokens" in obs_src else payload
          rgb = _decode_array(rgb_source, "rgb_frames", (1, 2, 3, 32, 32), np.float32)
          proprio = _decode_array(prop_source, "proprio_state", (1, 8), np.float32)
          inst = _decode_array(inst_source, "instruction_tokens", (1, 24), np.int32)
          prev_tail_raw = payload.get("prev_chunk_tail", obs_src.get("prev_chunk_tail"))
          prev_tail = (
              np.asarray(prev_tail_raw, dtype=np.float32)
              if prev_tail_raw is not None
              else None
          )
          chunk_size = int(payload.get("chunk_size", payload.get("chunk_size_k", 16)))
          res = runner.step_act(
              session_id=str(payload.get("session_id", "default")),
              step_idx=int(payload.get("step_idx", 0)),
              rgb_frames=rgb,
              proprio_state=proprio,
              instruction_tokens=inst,
              chunk_size=chunk_size,
              prev_chunk_tail=prev_tail,
          )
          res["status"] = "ok"
          res["execution_id"] = execution_id
          res["server_compute_ms"] = float(res.get("step_latency_ms", 0.0))
          res["cache_hit"] = bool(
              res.get("kv_prefix_cache_hit", False) or res.get("dit_cache_hit", False)
          )
          self._send_json(200, res)
        elif self.path in ("/v1/world_step", "/v1/wm/step"):
          obs_key = "obs_frames" if "obs_frames" in obs_src else (
              "rgb_frames" if "rgb_frames" in obs_src else "obs_frames"
          )
          obs_holder = obs_src if obs_key in obs_src else payload
          cond_key = "cond_actions" if "cond_actions" in obs_src else (
              "action_conditioning" if "action_conditioning" in payload else "cond_actions"
          )
          cond_holder = obs_src if "cond_actions" in obs_src else payload
          obs = _decode_array(obs_holder, obs_key, (1, 2, 3, 32, 32), np.float32)
          cond = _decode_array(cond_holder, cond_key, (1, 16, 7), np.float32)
          if cond.ndim == 2:
            cond = cond[None, :, :]
          horizon_frames = int(
              payload.get("horizon_frames", payload.get("chunk_size_k", 16))
          )
          res = runner.step_world(
              session_id=str(payload.get("session_id", "default")),
              step_idx=int(payload.get("step_idx", 0)),
              obs_frames=obs,
              cond_actions=cond,
              horizon_frames=horizon_frames,
          )
          res["status"] = "ok"
          res["execution_id"] = execution_id
          res["server_compute_ms"] = float(res.get("step_latency_ms", 0.0))
          res["cache_hit"] = bool(
              res.get("dit_cache_hit", False) or res.get("kv_prefix_cache_hit", False)
          )
          if "video_latents" in res and "predicted_latents" not in res:
            res["predicted_latents"] = res["video_latents"]
          self._send_json(200, res)
        else:
          self._send_json(404, {"error": f"Unknown POST endpoint {self.path!r}"})
      except Exception as exc:  # pylint: disable=broad-except
        self._send_json(400, {"error": str(exc)})

  return ThreadingHTTPServer((host, port), _Handler)


def build_arg_parser(default_model_id: str = "gr00t-n1.7-3b") -> argparse.ArgumentParser:
  """Build standard CLI argument parser for EmbodiedX v1.0 reference model servers."""
  parser = argparse.ArgumentParser(
      description="EmbodiedX v1.0 Single-Chip Reference Model Server"
  )
  parser.add_argument("--model-id", type=str, default=default_model_id)
  parser.add_argument("--hardware-sku", type=str, default="RTX_5090_32GB")
  parser.add_argument("--precision", type=str, default="BF16")
  parser.add_argument("--num-devices", type=int, default=1)
  parser.add_argument("--num-inference-steps", type=int, default=None)
  parser.add_argument(
      "--execution-strategy",
      type=str,
      default=ExecutionStrategy.RTC_INPAINTING.value,
  )
  parser.add_argument(
      "--weights-mode",
      type=str,
      default="deterministic-reference",
      choices=list(VALID_WEIGHTS_MODES),
  )
  parser.add_argument("--vllm-endpoint", type=str, default=None)
  parser.add_argument("--disable-prefix-cache", action="store_true")
  parser.add_argument("--disable-dit-cache", action="store_true")
  parser.add_argument("--self-test", action="store_true")
  parser.add_argument(
      "--host",
      type=str,
      default="127.0.0.1",
      help="Bind host when starting HTTP server via --serve or --port.",
  )
  parser.add_argument(
      "--port",
      type=int,
      default=None,
      help="Bind port (starts HTTP server when specified; pass 0 for ephemeral port).",
  )
  parser.add_argument(
      "--serve",
      action="store_true",
      help="Start HTTP server exposing /health, /v1/act, /v1/vla/act, /v1/world_step, /v1/wm/step.",
  )
  return parser


def run_runner_cli(
    runner: BaseEmbodiedModelRunner,
    args: argparse.Namespace,
) -> int:
  """Execute standard CLI behavior (health check, --self-test, and/or --serve / --port)."""
  info = dict(runner.health())
  if getattr(args, "self_test", False):
    rng = np.random.default_rng(runner.config.seed)
    rgb = rng.uniform(0.0, 1.0, size=(1, 2, 3, 32, 32)).astype(np.float32)
    proprio = rng.normal(0.0, 0.5, size=(1, 8)).astype(np.float32)
    inst = np.arange(24, dtype=np.int32).reshape(1, 24)
    cond = rng.uniform(-0.5, 0.5, size=(1, 16, 7)).astype(np.float32)
    step0 = runner.step_act("cli_selftest", 0, rgb, proprio, inst, chunk_size=8)
    step1 = runner.step_act("cli_selftest", 1, rgb, proprio, inst, chunk_size=8)
    wm0 = runner.step_world("cli_selftest_wm", 0, rgb, cond, horizon_frames=16)
    info["self_test"] = {
        "step0_kv_hit": bool(step0["kv_prefix_cache_hit"]),
        "step1_kv_hit": bool(step1["kv_prefix_cache_hit"]),
        "actions_shape": list(step1["actions"].shape),
        "video_latents_shape": list(wm0["video_latents"].shape),
    }

  should_serve = bool(getattr(args, "serve", False) or getattr(args, "port", None) is not None)
  if should_serve:
    bind_host = str(getattr(args, "host", "127.0.0.1"))
    bind_port = int(args.port) if getattr(args, "port", None) is not None else 0
    srv = make_runner_http_server(runner, host=bind_host, port=bind_port)
    actual_host, actual_port = srv.server_address[:2]
    server_url = f"http://{actual_host}:{actual_port}"
    if getattr(args, "self_test", False):
      thread = threading.Thread(target=srv.serve_forever, daemon=True)
      thread.start()
      try:
        import urllib.request
        with urllib.request.urlopen(f"{server_url}/health", timeout=5.0) as resp:
          health_http = json.loads(resp.read().decode("utf-8"))
        info["http_self_test"] = {
            "server_url": server_url,
            "health_status": health_http.get("status"),
            "execution_id": health_http.get("execution_id"),
        }
      finally:
        srv.shutdown()
        srv.server_close()
      print(json.dumps(info, indent=2, sort_keys=True))
      return 0

    info["server_url"] = server_url
    info["serving"] = True
    print(json.dumps(info, indent=2, sort_keys=True), flush=True)
    try:
      srv.serve_forever()
    except KeyboardInterrupt:
      pass
    finally:
      srv.server_close()
    return 0

  print(json.dumps(info, indent=2, sort_keys=True))
  return 0


def main(argv: Optional[List[str]] = None) -> int:
  """CLI entrypoint for testing, inspecting, or serving any reference model runner."""
  parser = build_arg_parser()
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
  runner = create_model_runner(cfg)
  return run_runner_cli(runner, args)


if __name__ == "__main__":
  raise SystemExit(main())
