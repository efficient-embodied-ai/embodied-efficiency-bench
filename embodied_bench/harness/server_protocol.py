"""Zero-lock-in HTTP/JSON+base64 wire protocol and reference server for EmbodiedX (`embodied-efficiency-bench`).

Allows any participant inference stack (`PyTorch`, `JAX/Pallas` on Google Cloud TPU,
`ROCm` on AMD Instinct/Radeon, `TensorRT` on NVIDIA Jetson/Blackwell, or custom C++/Rust
runtimes) to serve Track 1 VLA policies and Track 2 World Models over standard HTTP endpoints:
  - `GET  /health`
  - `POST /v1/vla/act`
  - `POST /v1/wm/step`
  - `POST /v1/reset_session`
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from embodied_bench.schema import (
    ExecutionStrategy,
    now_pacific_iso,
)

MAX_HTTP_BODY_BYTES: int = 64 * 1024 * 1024
MAX_ACTIVE_SESSIONS: int = 1024


class _PayloadTooLargeError(ValueError):
  """Raised when Content-Length exceeds MAX_HTTP_BODY_BYTES (HTTP 413)."""


SUPPORTED_SERVER_ENDPOINTS: Tuple[str, ...] = (
    "/health",
    "/v1/vla/act",
    "/v1/wm/step",
    "/v1/reset_session",
)


def encode_ndarray_b64(arr: np.ndarray) -> Dict[str, Any]:
  """Encode a NumPy ndarray into a JSON-safe base64 dictionary preserving exact dtype and shape.

  Args:
    arr: Input NumPy array of any numeric or boolean dtype and shape.

  Returns:
    Dictionary with keys `__ndarray_b64__`, `dtype`, and `shape`.
  """
  contig = np.ascontiguousarray(np.asarray(arr))
  raw_b64 = base64.b64encode(contig.tobytes()).decode("ascii")
  return {
      "__ndarray_b64__": raw_b64,
      "dtype": str(contig.dtype),
      "shape": list(contig.shape),
  }


def decode_ndarray_b64(payload: Dict[str, Any]) -> np.ndarray:
  """Decode a base64 ndarray dictionary produced by `encode_ndarray_b64` back to an exact NumPy array.

  Args:
    payload: Dictionary containing `__ndarray_b64__`, `dtype`, and `shape`.

  Returns:
    Reconstructed writable NumPy ndarray with exact original `dtype` and `shape`.
  """
  if not isinstance(payload, dict) or "__ndarray_b64__" not in payload:
    raise ValueError(
        "Invalid ndarray payload: expected dict containing '__ndarray_b64__', 'dtype', and 'shape'. "
        "Encode arrays with encode_ndarray_b64(arr) before transmission."
    )
  if "dtype" not in payload or "shape" not in payload:
    raise ValueError(
        "Incomplete ndarray payload: both 'dtype' and 'shape' are required alongside '__ndarray_b64__'."
    )
  dt = np.dtype(payload["dtype"])
  raw_shape = payload["shape"]
  if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) > 8:
    raise ValueError("Invalid ndarray shape: expected a list of at most 8 integers.")
  shape = tuple(int(dim) for dim in raw_shape)
  if any(dim < 0 for dim in shape):
    raise ValueError(f"Negative dimension in ndarray shape {shape}.")
  expected_bytes = int(math.prod(shape)) * int(dt.itemsize)
  if expected_bytes > MAX_HTTP_BODY_BYTES:
    raise ValueError(
        f"Declared ndarray shape {shape} ({expected_bytes} bytes) exceeds 64 MiB limit ({MAX_HTTP_BODY_BYTES})."
    )
  b64_raw = payload["__ndarray_b64__"]
  if not isinstance(b64_raw, str) or len(b64_raw) > (MAX_HTTP_BODY_BYTES * 4 // 3) + 4096:
    raise ValueError("Encoded '__ndarray_b64__' string exceeds 64 MiB limit.")
  raw_bytes = base64.b64decode(b64_raw.encode("ascii"), validate=True)
  if len(raw_bytes) != expected_bytes:
    raise ValueError(
        f"Decoded ndarray byte count {len(raw_bytes)} does not match shape {shape} ({expected_bytes} bytes)."
    )
  flat = np.frombuffer(raw_bytes, dtype=dt)
  return flat.reshape(shape).copy()


def is_encoded_ndarray(obj: Any) -> bool:
  """Return True if `obj` is a dict formatted by `encode_ndarray_b64`."""
  return (
      isinstance(obj, dict)
      and "__ndarray_b64__" in obj
      and "dtype" in obj
      and "shape" in obj
  )


def encode_payload_arrays(obj: Any) -> Any:
  """Recursively encode any `np.ndarray` instances inside a nested dict/list payload."""
  if isinstance(obj, np.ndarray):
    return encode_ndarray_b64(obj)
  if isinstance(obj, dict):
    if is_encoded_ndarray(obj):
      return obj
    return {str(k): encode_payload_arrays(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [encode_payload_arrays(v) for v in obj]
  if isinstance(obj, (np.floating, np.integer)):
    return obj.item()
  return obj


def decode_payload_arrays(obj: Any) -> Any:
  """Recursively decode any base64-encoded ndarray dicts inside a nested dict/list payload."""
  if is_encoded_ndarray(obj):
    return decode_ndarray_b64(obj)
  if isinstance(obj, dict):
    return {str(k): decode_payload_arrays(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [decode_payload_arrays(v) for v in obj]
  return obj


@dataclass
class SessionStateRecord:
  """Per-session state tracked by `EmbodiedBenchmarkServer` for sticky stream affinity and KV cache reuse."""

  session_id: str
  step_count: int = 0
  cache_hits: int = 0
  created_at: str = field(default_factory=now_pacific_iso)
  last_active_at: str = field(default_factory=now_pacific_iso)
  context_tokens_cached: int = 0
  last_action_chunk: Optional[np.ndarray] = None

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    d.pop("last_action_chunk", None)
    return d


class ReferenceEmbodiedRunner:
  """Default deterministic reference runner for Track 1 VLA and Track 2 World Model HTTP serving.

  Can be used directly in unit/integration tests or subclassed by participant model servers.
  """

  def __init__(
      self,
      action_dim: int = 7,
      default_chunk_size: int = 8,
      default_horizon_frames: int = 16,
      simulated_compute_delay_s: float = 0.0,
      kp_gain: float = 0.65,
  ) -> None:
    self.action_dim = int(action_dim)
    self.default_chunk_size = int(default_chunk_size)
    self.default_horizon_frames = int(default_horizon_frames)
    self.simulated_compute_delay_s = float(simulated_compute_delay_s)
    self.kp_gain = float(kp_gain)
    self._session_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    self._lock = threading.Lock()

  def infer_vla(
      self,
      obs_dict: Dict[str, Any],
      session_id: str = "default_session",
  ) -> Dict[str, Any]:
    """Compute a `(chunk_size, action_dim)` action chunk driving `ee_pos` toward `goal_pos` (or `target_pos`)."""
    if self.simulated_compute_delay_s > 0.0:
      time.sleep(self.simulated_compute_delay_s)

    chunk_size = int(obs_dict.get("chunk_size", self.default_chunk_size))
    action_dim = int(obs_dict.get("action_dim", self.action_dim))
    emergency_stop = bool(obs_dict.get("emergency_stop", False))

    with self._lock:
      cached = self._session_cache.get(session_id)
      cache_hit = cached is not None
      step_idx = 0 if cached is None else int(cached.get("step_idx", 0)) + 1
      self._session_cache[session_id] = {
          "step_idx": step_idx,
          "updated_at": time.perf_counter(),
      }
      self._session_cache.move_to_end(session_id)
      while len(self._session_cache) > MAX_ACTIVE_SESSIONS:
        self._session_cache.popitem(last=False)
      self._session_cache.move_to_end(session_id)
      while len(self._session_cache) > MAX_ACTIVE_SESSIONS:
        self._session_cache.popitem(last=False)

    if emergency_stop:
      actions = np.zeros((chunk_size, action_dim), dtype=np.float32)
      return {
          "actions": actions,
          "cache_hit": cache_hit,
          "emergency_stop_engaged": True,
      }

    # If closed-loop state observations (`ee_pos` and `goal_pos` / `object_pos`) are present,
    # generate proportional trajectory steps toward the active target.
    ee_pos = obs_dict.get("ee_pos")
    goal_pos = obs_dict.get("goal_pos", obs_dict.get("target_pos"))
    object_pos = obs_dict.get("object_pos")
    object_grasped = bool(obs_dict.get("object_grasped", False))

    if ee_pos is not None and (goal_pos is not None or object_pos is not None):
      ee = np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3]
      if object_pos is not None and not object_grasped:
        target = np.asarray(object_pos, dtype=np.float32).reshape(-1)[:3]
        gripper_cmd = -1.0 if float(np.linalg.norm(target - ee)) < 0.08 else 1.0
      else:
        target = np.asarray(
            goal_pos if goal_pos is not None else object_pos, dtype=np.float32
        ).reshape(-1)[:3]
        gripper_cmd = -1.0

      actions = np.zeros((chunk_size, action_dim), dtype=np.float32)
      sim_ee = ee.copy()
      for h in range(chunk_size):
        err = target - sim_ee
        dist = float(np.linalg.norm(err))
        if dist > 1e-6:
          step_vec = (err / dist) * min(dist * self.kp_gain, 0.12)
        else:
          step_vec = np.zeros(3, dtype=np.float32)
        actions[h, :3] = step_vec
        if action_dim > 6:
          actions[h, 6] = gripper_cmd
        sim_ee = sim_ee + step_vec * 0.5
      return {
          "actions": actions,
          "cache_hit": cache_hit,
          "emergency_stop_engaged": False,
      }

    # Fallback deterministic feature projection from visual/proprioceptive arrays
    proprio = obs_dict.get("proprio", obs_dict.get("state"))
    if proprio is not None:
      vec = np.asarray(proprio, dtype=np.float32).reshape(-1)
      base_val = float(np.mean(vec)) if vec.size > 0 else 0.1
    else:
      base_val = 0.05 * ((step_idx % 7) + 1)

    grid = np.linspace(0.0, 1.0, chunk_size * action_dim, dtype=np.float32).reshape(
        chunk_size, action_dim
    )
    actions = np.tanh(grid + base_val).astype(np.float32)
    return {
        "actions": actions,
        "cache_hit": cache_hit,
        "emergency_stop_engaged": False,
    }

  def infer_wm(
      self,
      obs_dict: Dict[str, Any],
      session_id: str = "default_session",
  ) -> Dict[str, Any]:
    """Predict future world-model latent frames `(horizon_frames, latent_dim)` and inverse-dynamics actions."""
    if self.simulated_compute_delay_s > 0.0:
      time.sleep(self.simulated_compute_delay_s)

    horizon_frames = int(obs_dict.get("horizon_frames", self.default_horizon_frames))
    latent_dim = int(obs_dict.get("latent_dim", 16))
    with self._lock:
      cached = self._session_cache.get(session_id)
      cache_hit = cached is not None
      step_idx = 0 if cached is None else int(cached.get("step_idx", 0)) + 1
      self._session_cache[session_id] = {
          "step_idx": step_idx,
          "updated_at": time.perf_counter(),
      }
      self._session_cache.move_to_end(session_id)
      while len(self._session_cache) > MAX_ACTIVE_SESSIONS:
        self._session_cache.popitem(last=False)

    cond_action = obs_dict.get("action", obs_dict.get("conditioning_action"))
    if cond_action is not None:
      act_arr = np.asarray(cond_action, dtype=np.float32).reshape(-1)
      seed_offset = float(np.mean(act_arr)) if act_arr.size > 0 else 0.0
    else:
      seed_offset = 0.02 * (step_idx + 1)

    t_axis = np.linspace(0.1, 1.0, horizon_frames, dtype=np.float32)[:, None]
    d_axis = np.linspace(0.0, 0.5, latent_dim, dtype=np.float32)[None, :]
    predicted_latents = np.cos(t_axis + d_axis + seed_offset).astype(np.float32)
    latent_actions = np.sin(
        np.linspace(0.0, 1.0, self.action_dim, dtype=np.float32) + seed_offset
    ).astype(np.float32)
    return {
        "predicted_latents": predicted_latents,
        "latent_actions": latent_actions,
        "cache_hit": cache_hit,
    }

  def reset_session(self, session_id: Optional[str] = None) -> None:
    """Clear cached context for `session_id` (or all sessions when `session_id` is None or `'*'`)."""
    with self._lock:
      if session_id is None or session_id == "*":
        self._session_cache.clear()
      else:
        self._session_cache.pop(session_id, None)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
  """ThreadingHTTPServer with `allow_reuse_address = True` and daemon worker threads."""

  allow_reuse_address = True
  daemon_threads = True


class EmbodiedBenchmarkServer:
  """Zero-lock-in HTTP/JSON+base64 benchmark server wrapping any VLA / World Model runner.

  Endpoints:
    - `GET  /health`: Server liveness, active session count, total requests, and cache hits.
    - `POST /v1/vla/act`: Execute Track 1 VLA policy inference for `session_id`.
    - `POST /v1/wm/step`: Execute Track 2 World Model rollout step for `session_id`.
    - `POST /v1/reset_session`: Reset KV / temporal context cache for `session_id` (or all sessions).
  """

  def __init__(
      self,
      runner: Any = None,
      host: str = "127.0.0.1",
      port: int = 0,
      server_id: str = "embodiedx-reference-server",
  ) -> None:
    self.runner = runner if runner is not None else ReferenceEmbodiedRunner()
    self.host = str(host)
    self._requested_port = int(port)
    self.server_id = str(server_id)
    self._lock = threading.Lock()
    self._sessions: OrderedDict[str, SessionStateRecord] = OrderedDict()
    self._total_requests = 0
    self._total_cache_hits = 0
    self._started_at = now_pacific_iso()
    self._httpd: Optional[_ReusableThreadingHTTPServer] = None
    self._thread: Optional[threading.Thread] = None
    self._running = False

    parent = self

    class _Handler(BaseHTTPRequestHandler):
      protocol_version = "HTTP/1.1"

      def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence per-request stderr noise during high-frequency LoadGen sweeps.
        return

      def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

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
        if self.path.rstrip("/") in ("/health", "/v1/health") or self.path in ("/health", "/v1/health"):
          self._send_json(200, parent.get_health_snapshot())
        else:
          self._send_json(
              404,
              {
                  "status": "error",
                  "error": f"Unknown GET endpoint {self.path!r}. Supported: {list(SUPPORTED_SERVER_ENDPOINTS)}.",
              },
          )

      def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        try:
          body = self._read_json_body()
        except _PayloadTooLargeError as exc:
          self._send_json(413, {"status": "error", "error": str(exc)})
          return
        except Exception as exc:  # pragma: no cover
          self._send_json(
              400,
              {"status": "error", "error": f"Malformed JSON request body: {exc}"},
          )
          return

        try:
          if path in ("/v1/vla/act", "/v1/act"):
            resp = parent.handle_vla_act(body)
            self._send_json(200, resp)
          elif path in ("/v1/wm/step", "/v1/world_step"):
            resp = parent.handle_wm_step(body)
            self._send_json(200, resp)
          elif path == "/v1/reset_session":
            resp = parent.handle_reset_session(body)
            self._send_json(200, resp)
          else:
            self._send_json(
                404,
                {
                    "status": "error",
                    "error": f"Unknown POST endpoint {self.path!r}. Supported: {list(SUPPORTED_SERVER_ENDPOINTS)}.",
                },
            )
        except Exception as exc:
          self._send_json(
              500,
              {"status": "error", "error": str(exc)},
          )

    self._handler_cls = _Handler

  @property
  def port(self) -> int:
    """Return actual bound TCP port (resolves ephemeral `port=0` once started)."""
    if self._httpd is not None:
      return int(self._httpd.server_port)
    return self._requested_port

  @property
  def base_url(self) -> str:
    """Return `http://<host>:<port>` base URL for clients."""
    return f"http://{self.host}:{self.port}"

  def get_health_snapshot(self) -> Dict[str, Any]:
    """Return thread-safe server health and session cache telemetry."""
    with self._lock:
      session_summaries = {
          sid: rec.to_dict() for sid, rec in self._sessions.items()
      }
      total_req = self._total_requests
      total_hits = self._total_cache_hits
    return {
        "status": "ok",
        "server_id": self.server_id,
        "started_at": self._started_at,
        "timestamp": now_pacific_iso(),
        "supported_endpoints": list(SUPPORTED_SERVER_ENDPOINTS),
        "active_sessions": len(session_summaries),
        "total_requests": total_req,
        "total_cache_hits": total_hits,
        "cache_hit_ratio": round(
            float(total_hits) / float(max(1, total_req)), 4
        ),
        "sessions": session_summaries,
    }

  def _record_session_step(
      self,
      session_id: str,
      runner_reported_cache_hit: Optional[bool] = None,
  ) -> Tuple[int, bool]:
    """Update per-session step count and KV/temporal cache hit counters under lock."""
    with self._lock:
      rec = self._sessions.get(session_id)
      if rec is None:
        rec = SessionStateRecord(session_id=session_id, step_count=0, cache_hits=0)
        self._sessions[session_id] = rec
        is_hit = bool(runner_reported_cache_hit) if runner_reported_cache_hit is not None else False
        while len(self._sessions) > MAX_ACTIVE_SESSIONS:
          self._sessions.popitem(last=False)
      else:
        self._sessions.move_to_end(session_id)
        is_hit = (
            bool(runner_reported_cache_hit)
            if runner_reported_cache_hit is not None
            else (rec.step_count > 0)
        )
      rec.step_count += 1
      if is_hit:
        rec.cache_hits += 1
        self._total_cache_hits += 1
      rec.context_tokens_cached += 256
      rec.last_active_at = now_pacific_iso()
      self._total_requests += 1
      return rec.step_count, is_hit

  def handle_vla_act(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a `/v1/vla/act` or `/v1/act` request and return base64-encoded action chunk + metadata."""
    t0_ns = time.perf_counter_ns()
    session_id = str(request_data.get("session_id", "default_session"))
    raw_obs = request_data.get("observations", request_data.get("observation", {}))
    decoded_obs = decode_payload_arrays(raw_obs) if isinstance(raw_obs, dict) else {}
    if "chunk_size" in request_data and "chunk_size" not in decoded_obs:
      decoded_obs["chunk_size"] = int(request_data["chunk_size"])
    if "action_dim" in request_data and "action_dim" not in decoded_obs:
      decoded_obs["action_dim"] = int(request_data["action_dim"])
    if "execution_strategy" in request_data and "execution_strategy" not in decoded_obs:
      decoded_obs["execution_strategy"] = str(request_data["execution_strategy"])
    if "emergency_stop" in request_data and "emergency_stop" not in decoded_obs:
      decoded_obs["emergency_stop"] = bool(request_data["emergency_stop"])

    if hasattr(self.runner, "infer_vla"):
      out = self.runner.infer_vla(decoded_obs, session_id=session_id)
    elif callable(self.runner):
      out = self.runner(decoded_obs, session_id=session_id)
    else:
      raise TypeError(
          f"Runner {type(self.runner).__name__} must implement infer_vla(obs_dict, session_id) or be callable."
      )

    if isinstance(out, np.ndarray):
      actions_arr = out
      runner_hit = None
      extra_meta: Dict[str, Any] = {}
    elif isinstance(out, dict):
      raw_act = out.get(
          "actions",
          out.get("predicted_latents", out.get("predicted_frames", out.get("next_frames"))),
      )
      actions_arr = np.asarray(
          raw_act if raw_act is not None else np.zeros((8, 7), dtype=np.float32),
          dtype=np.float32,
      )
      runner_hit = out.get("cache_hit", out.get("prefix_cache_hit"))
      extra_meta = {
          k: encode_payload_arrays(v)
          for k, v in out.items()
          if k not in ("actions", "cache_hit", "prefix_cache_hit")
      }
    else:
      actions_arr = np.asarray(out, dtype=np.float32)
      runner_hit = None
      extra_meta = {}

    step_count, cache_hit = self._record_session_step(session_id, runner_hit)
    compute_ms = (time.perf_counter_ns() - t0_ns) / 1e6

    return {
      "status": "ok",
      "session_id": session_id,
      "step_index": step_count,
      "cache_hit": cache_hit,
      "server_compute_ms": round(compute_ms, 4),
      "execution_strategy": str(
          decoded_obs.get("execution_strategy", ExecutionStrategy.ASYNC_CHUNK.value)
      ),
      "actions": encode_ndarray_b64(actions_arr),
      "metadata": extra_meta,
    }

  def handle_wm_step(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a `/v1/wm/step` or `/v1/world_step` request and return base64-encoded predicted latents + metadata."""
    t0_ns = time.perf_counter_ns()
    session_id = str(request_data.get("session_id", "default_session"))
    raw_obs = request_data.get("observations", request_data.get("observation", {}))
    decoded_obs = decode_payload_arrays(raw_obs) if isinstance(raw_obs, dict) else {}
    if "horizon_frames" in request_data and "horizon_frames" not in decoded_obs:
      decoded_obs["horizon_frames"] = int(request_data["horizon_frames"])
    if "latent_dim" in request_data and "latent_dim" not in decoded_obs:
      decoded_obs["latent_dim"] = int(request_data["latent_dim"])

    if hasattr(self.runner, "infer_wm"):
      out = self.runner.infer_wm(decoded_obs, session_id=session_id)
    elif callable(self.runner):
      out = self.runner(decoded_obs, session_id=session_id)
    else:
      raise TypeError(
          f"Runner {type(self.runner).__name__} must implement infer_wm(obs_dict, session_id) or be callable."
      )

    if isinstance(out, np.ndarray):
      latents_arr = out
      runner_hit = None
      extra_meta: Dict[str, Any] = {}
    elif isinstance(out, dict):
      raw_pred = out.get(
          "predicted_latents",
          out.get(
              "dit_latents",
              out.get("predicted_frames", out.get("next_frames", out.get("actions"))),
          ),
      )
      latents_arr = np.asarray(
          raw_pred if raw_pred is not None else np.zeros((16, 16), dtype=np.float32),
          dtype=np.float32,
      )
      runner_hit = out.get("cache_hit", out.get("prefix_cache_hit"))
      extra_meta = {
          k: encode_payload_arrays(v)
          for k, v in out.items()
          if k not in ("predicted_latents", "dit_latents", "predicted_frames", "next_frames", "cache_hit", "prefix_cache_hit")
      }
    else:
      latents_arr = np.asarray(out, dtype=np.float32)
      runner_hit = None
      extra_meta = {}

    step_count, cache_hit = self._record_session_step(session_id, runner_hit)
    compute_ms = (time.perf_counter_ns() - t0_ns) / 1e6

    return {
        "status": "ok",
        "session_id": session_id,
        "step_index": step_count,
        "cache_hit": cache_hit,
        "server_compute_ms": round(compute_ms, 4),
        "predicted_latents": encode_ndarray_b64(latents_arr),
        "metadata": extra_meta,
    }

  def handle_reset_session(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Reset a specific `session_id` (or all sessions if `session_id` is omitted or `'*'`)."""
    session_id = request_data.get("session_id")
    with self._lock:
      if session_id is None or session_id == "*":
        cleared = len(self._sessions)
        self._sessions.clear()
      else:
        sid = str(session_id)
        cleared = 1 if sid in self._sessions else 0
        self._sessions.pop(sid, None)

    if hasattr(self.runner, "reset_session"):
      self.runner.reset_session(session_id)
    elif hasattr(self.runner, "clear_session_cache"):
      self.runner.clear_session_cache()

    return {
        "status": "ok",
        "reset_session_id": session_id or "*",
        "sessions_cleared": cleared,
        "timestamp": now_pacific_iso(),
    }

  def start_background(self) -> str:
    """Start the HTTP server on a background daemon thread and return `base_url`."""
    with self._lock:
      if self._running and self._httpd is not None:
        return self.base_url
      self._httpd = _ReusableThreadingHTTPServer(
          (self.host, self._requested_port), self._handler_cls
      )
      self._thread = threading.Thread(
          target=self._httpd.serve_forever,
          kwargs={"poll_interval": 0.01},
          name=f"EmbodiedBenchmarkServer-{self.port}",
          daemon=True,
      )
      self._thread.start()
      self._running = True
    return self.base_url

  def shutdown(self) -> None:
    """Cleanly stop the background HTTP server and close the listening socket."""
    with self._lock:
      if not self._running or self._httpd is None:
        return
      httpd = self._httpd
      thread = self._thread
      self._httpd = None
      self._thread = None
      self._running = False

    try:
      httpd.shutdown()
    finally:
      httpd.server_close()
    if thread is not None and thread.is_alive():
      thread.join(timeout=2.0)

  def __enter__(self) -> "EmbodiedBenchmarkServer":
    self.start_background()
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.shutdown()
