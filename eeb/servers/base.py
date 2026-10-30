"""gRPC and HTTP servicers that expose any in-process `Policy` over UDS, TCP, or HTTP.

Endpoint grammar (shared with `eeb.loadgen.client`)::

    unix:///tmp/eeb.sock      Unix-domain socket (absolute path; preferred, ADR-1)
    unix:/tmp/eeb.sock        same
    tcp://127.0.0.1:50051     TCP gRPC
    127.0.0.1:50051           TCP gRPC (bare host:port)
    http://127.0.0.1:8080     HTTP/JSON server
    inproc:<module:Class>     in-process policy (client side only; never served)

Usage::

    from eeb.servers.base import serve_policy
    server = serve_policy(DummyVLAPolicy(), "unix:///tmp/eeb.sock")
    server.wait_for_termination()
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Generic, Iterator, List, Optional, Sequence, TypeVar

import numpy as np

from eeb.protocol.policy import ActionChunk, Context, Observation, Policy, validate_policy
from eeb.servers.torch_backend import (
    TORCHTPU_INSTALL_HINT,
    maybe_compile_torch_module,
    resolve_torch_device,
    synchronize_torch_device,
)

_LOG = logging.getLogger("eeb.servers")


@dataclass(frozen=True)
class Endpoint:
  """Parsed endpoint."""

  kind: str  # "unix" | "tcp" | "http" | "inproc"
  target: str  # grpc/http target string for unix/tcp/http, policy spec for inproc

  @property
  def grpc_target(self) -> str:
    if self.kind == "unix":
      return f"unix:{self.target}"
    if self.kind == "tcp":
      return self.target
    raise ValueError(f"Endpoint kind {self.kind!r} has no gRPC target.")

  @property
  def http_url(self) -> str:
    if self.kind == "http":
      return f"http://{self.target}"
    raise ValueError(f"Endpoint kind {self.kind!r} is not an HTTP endpoint.")

  @property
  def is_http(self) -> bool:
    return self.kind == "http"

  @property
  def is_inproc(self) -> bool:
    return self.kind == "inproc"


MAX_UNIX_SOCKET_PATH_BYTES = 107


def normalize_and_validate_uds_path(raw_path: str, endpoint_repr: Optional[str] = None) -> str:
  """Resolve relative Unix domain socket paths and enforce the 107-byte `AF_UNIX` kernel limit."""
  cleaned = (raw_path or "").strip()
  if not cleaned:
    raise ValueError(
        f"Unix-domain socket endpoint has empty path: {endpoint_repr or raw_path!r}. "
        "Use unix:///tmp/eeb.sock or --uds ./policy.sock."
    )
  resolved = os.path.abspath(os.path.expanduser(cleaned))
  encoded_len = len(os.fsencode(resolved))
  if encoded_len > MAX_UNIX_SOCKET_PATH_BYTES:
    raise ValueError(
        f"Unix-domain socket path is {encoded_len} bytes, exceeding the {MAX_UNIX_SOCKET_PATH_BYTES}-byte "
        f"AF_UNIX sun_path kernel limit: {resolved!r}. Use a shorter path such as '/tmp/eeb_policy.sock'."
    )
  return resolved


def parse_endpoint(endpoint: str) -> Endpoint:
  """Parse the endpoint grammar documented in the module docstring."""
  ep = (endpoint or "").strip()
  if not ep:
    raise ValueError(
        "Empty endpoint. Use unix:///path.sock, tcp://host:port, http://host:port, host:port, or inproc:<module:Class>."
    )
  if ep.startswith("unix://"):
    path = normalize_and_validate_uds_path(ep[len("unix://"):], endpoint_repr=endpoint)
    return Endpoint("unix", path)
  if ep.startswith("unix:"):
    path = normalize_and_validate_uds_path(ep[len("unix:"):], endpoint_repr=endpoint)
    return Endpoint("unix", path)
  if ep.startswith("inproc:"):
    spec = ep[len("inproc:"):]
    if ":" not in spec and "." not in spec:
      raise ValueError(f"inproc endpoint needs '<module>:<Class>', got {endpoint!r}.")
    return Endpoint("inproc", spec)
  if ep.startswith("http://"):
    rest = ep[len("http://"):].split("/", 1)[0]
    if ":" not in rest:
      raise ValueError(f"HTTP endpoint must be http://host:port, got {endpoint!r}.")
    host, _, port = rest.rpartition(":")
    if not host or not port.isdigit():
      raise ValueError(f"HTTP endpoint must be http://host:port with numeric port, got {endpoint!r}.")
    return Endpoint("http", f"{host}:{port}")
  if ep.startswith("tcp://"):
    ep = ep[len("tcp://"):]
  if ":" not in ep:
    raise ValueError(f"TCP endpoint must be host:port, got {endpoint!r}.")
  host, _, port = ep.rpartition(":")
  if not port.isdigit():
    raise ValueError(f"TCP endpoint port must be numeric, got {endpoint!r}.")
  return Endpoint("tcp", f"{host}:{port}")


def _grpc():
  try:
    import grpc  # noqa: WPS433
  except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "grpcio is required to serve or call a policy over a socket. "
        "Install with `pip install -e '.[grpc]'` or use an inproc:<module:Class> endpoint."
    ) from exc
  return grpc


_TIn = TypeVar("_TIn")
_TOut = TypeVar("_TOut")


@dataclass
class _MicroBatchItem(Generic[_TIn, _TOut]):
  payload: _TIn
  session_key: str = ""
  done: threading.Event = field(default_factory=threading.Event)
  result: Optional[_TOut] = None
  error: Optional[BaseException] = None


class DynamicMicroBatcher(Generic[_TIn, _TOut]):
  """Thread-safe work-conserving dynamic micro-batcher for single-chip concurrent serving.

  Concurrent gRPC server threads submit per-session requests via `submit(payload, session_id=...)`.
  While the accelerator executes batch `k`, arriving requests accumulate in `_pending` and are popped
  together up to `max_batch_size` as soon as batch `k` completes. When the accelerator is idle,
  the batcher tracks live `session_id`s seen within `session_ttl_s` (purged on `on_reset()` or
  expiry) and wakes immediately (`0 ms` dead sleep) as soon as `len(_pending)` reaches
  `min(max_batch_size, active_cohort)`, eliminating low-concurrency batch-window penalties.
  """

  _INITIAL_COHORT_DISCOVERY_MAX_S = 0.0015

  def __init__(
      self,
      batch_fn: Callable[[Sequence[_TIn]], Sequence[_TOut]],
      *,
      max_batch_size: int = 16,
      batch_window_ms: float = 0.5,
      session_ttl_s: float = 2.0,
  ) -> None:
    if int(max_batch_size) < 1:
      raise ValueError(f"DynamicMicroBatcher requires max_batch_size >= 1, got {max_batch_size}.")
    if float(batch_window_ms) < 0.0:
      raise ValueError(f"DynamicMicroBatcher requires batch_window_ms >= 0, got {batch_window_ms}.")
    if float(session_ttl_s) <= 0.0:
      raise ValueError(f"DynamicMicroBatcher requires session_ttl_s > 0, got {session_ttl_s}.")
    self._batch_fn = batch_fn
    self.max_batch_size = int(max_batch_size)
    self.batch_window_s = float(batch_window_ms) / 1000.0
    self.session_ttl_s = float(session_ttl_s)
    self._cond = threading.Condition()
    self._pending: List[_MicroBatchItem[_TIn, _TOut]] = []
    self._active_sessions: dict[str, float] = {}
    self._cohort_warmed = False
    self._last_batch_exec_s = 0.0
    self._gpu_busy = False
    self.num_batches = 0
    self.num_items = 0
    self.max_observed_batch_size = 0

  def _prune_expired_sessions_locked(self, now: float) -> None:
    if not self._active_sessions:
      return
    cutoff = now - self.session_ttl_s
    expired = [sid for sid, ts in self._active_sessions.items() if ts < cutoff]
    for sid in expired:
      self._active_sessions.pop(sid, None)
    if not self._active_sessions:
      self._cohort_warmed = False

  def active_session_count(self) -> int:
    """Return the current count of live non-expired `session_id`s tracked by the batcher."""
    with self._cond:
      self._prune_expired_sessions_locked(time.perf_counter())
      return len(self._active_sessions)

  def register_session(self, session_id: str) -> None:
    """Pre-register a session key on `reset()` so batch assembly anticipates all active workers."""
    key = str(session_id or "").strip()
    if not key:
      return
    with self._cond:
      now = time.perf_counter()
      self._prune_expired_sessions_locked(now)
      self._active_sessions[key] = now
      self._cond.notify_all()

  def on_reset(self, session_id: Optional[str] = None) -> None:
    """Purge a reset/disconnected session (or all sessions if None) and wake any waiting batch."""
    with self._cond:
      self._prune_expired_sessions_locked(time.perf_counter())
      if session_id is None:
        self._active_sessions.clear()
        self._cohort_warmed = False
      else:
        key = str(session_id).strip()
        if key:
          self._active_sessions.pop(key, None)
        if not self._active_sessions:
          self._cohort_warmed = False
      self._cond.notify_all()

  def close(self) -> None:
    """Wake any waiting threads and reset tracked sessions on server shutdown."""
    self.on_reset()

  def submit(self, payload: _TIn, *, session_id: Optional[str] = None) -> _TOut:
    raw_sid = str(session_id).strip() if session_id is not None else ""
    skey = raw_sid if raw_sid else f"thread:{threading.get_ident()}"
    item: _MicroBatchItem[_TIn, _TOut] = _MicroBatchItem(payload=payload, session_key=skey)
    is_leader = False
    with self._cond:
      now = time.perf_counter()
      self._prune_expired_sessions_locked(now)
      self._active_sessions[skey] = now
      self._pending.append(item)
      self._cond.notify_all()
      if not self._gpu_busy:
        self._gpu_busy = True
        is_leader = True

    if is_leader:
      self._drain_loop()

    item.done.wait()
    if item.error is not None:
      raise item.error
    return item.result  # type: ignore[return-value]

  def _drain_loop(self) -> None:
    while True:
      with self._cond:
        now = time.perf_counter()
        self._prune_expired_sessions_locked(now)
        if not self._pending:
          self._gpu_busy = False
          return
        if self.batch_window_s > 0.0 and len(self._pending) < self.max_batch_size:
          wait_start = now
          if not self._cohort_warmed:
            eff_window_s = min(self.batch_window_s, self._INITIAL_COHORT_DISCOVERY_MAX_S)
            deadline = now + eff_window_s
            while len(self._pending) < self.max_batch_size:
              rem = deadline - time.perf_counter()
              if rem <= 0.0:
                break
              self._cond.wait(timeout=rem)
              self._prune_expired_sessions_locked(time.perf_counter())
          else:
            deadline = now + self.batch_window_s
            timed_out = False
            while True:
              self._prune_expired_sessions_locked(time.perf_counter())
              target = min(self.max_batch_size, max(1, len(self._active_sessions)))
              if len(self._pending) >= target:
                break
              rem = deadline - time.perf_counter()
              if rem <= 0.0:
                timed_out = True
                break
              self._cond.wait(timeout=rem)
            if timed_out:
              present_keys = {it.session_key for it in self._pending}
              stale_cutoff = wait_start - max(self.batch_window_s, self._last_batch_exec_s * 2.0)
              for sid in list(self._active_sessions.keys()):
                if sid not in present_keys and self._active_sessions[sid] < stale_cutoff:
                  self._active_sessions.pop(sid, None)
        self._cohort_warmed = True
        batch = self._pending[: self.max_batch_size]
        del self._pending[: len(batch)]

      t_exec0 = time.perf_counter()
      try:
        outputs = list(self._batch_fn([it.payload for it in batch]))
        if len(outputs) != len(batch):
          raise RuntimeError(
              f"DynamicMicroBatcher batch_fn returned {len(outputs)} outputs for batch of size {len(batch)}."
          )
      except BaseException as exc:  # noqa: BLE001
        exec_s = max(0.0, time.perf_counter() - t_exec0)
        for it in batch:
          it.error = exc
          it.done.set()
        with self._cond:
          self._last_batch_exec_s = exec_s
          self.num_batches += 1
          self.num_items += len(batch)
          if len(batch) > self.max_observed_batch_size:
            self.max_observed_batch_size = len(batch)
          if not self._pending:
            self._gpu_busy = False
            return
      else:
        exec_s = max(0.0, time.perf_counter() - t_exec0)
        for it, out in zip(batch, outputs):
          it.result = out
          it.done.set()
        with self._cond:
          self._last_batch_exec_s = exec_s
          self.num_batches += 1
          self.num_items += len(batch)
          if len(batch) > self.max_observed_batch_size:
            self.max_observed_batch_size = len(batch)
          if not self._pending:
            self._gpu_busy = False
            return


class PolicyServicer:
  """Adapts a `Policy` to the `EmbodiedPolicy` gRPC service.

  `serialize_calls=True` wraps `predict`/`rollout` in a lock so that a policy that is not
  thread-safe can be served under concurrency (requests queue, which is what a single
  accelerator does anyway). Policies that batch internally should set it to False.
  """

  def __init__(self, policy: Policy, serialize_calls: bool = False) -> None:
    validate_policy(policy)
    self.policy = policy
    self._lock: Optional[threading.Lock] = threading.Lock() if serialize_calls else None
    self.num_predict = 0
    self.num_rollout = 0
    self.num_reset = 0
    self._counter_lock = threading.Lock()

  # The generated base class is bound lazily (see build_servicer) so this module imports
  # without grpc. Method names/signatures match eeb_pb2_grpc.EmbodiedPolicyServicer.

  def Capabilities(self, request: Any, context: Any) -> Any:  # noqa: N802 - gRPC naming
    from eeb.protocol.codec import caps_to_proto

    return caps_to_proto(self.policy.capabilities())

  def Predict(self, request: Any, context: Any) -> Any:  # noqa: N802
    from eeb.protocol.codec import action_chunk_to_proto, proto_to_observation

    t_recv = time.perf_counter()
    try:
      obs = proto_to_observation(request)
    except ValueError as exc:
      _abort_invalid(context, str(exc))
      raise  # pragma: no cover - abort raises
    try:
      if self._lock is not None:
        with self._lock:
          chunk = self.policy.predict(obs)
      else:
        chunk = self.policy.predict(obs)
    except NotImplementedError as exc:
      _abort_unimplemented(context, str(exc))
      raise  # pragma: no cover
    except Exception as exc:  # pragma: no cover - surfaced to the client with details
      _LOG.exception("policy.predict failed")
      _abort_internal(context, f"policy.predict raised {type(exc).__name__}: {exc}")
      raise
    if not isinstance(chunk, ActionChunk):
      chunk = ActionChunk(actions=np.asarray(chunk, dtype=np.float32))
    chunk.t_recv = t_recv
    chunk.t_done = time.perf_counter()
    with self._counter_lock:
      self.num_predict += 1
    return action_chunk_to_proto(chunk)

  def Rollout(self, request: Any, context: Any) -> Iterator[Any]:  # noqa: N802
    from eeb.protocol.codec import frame_to_proto, proto_to_context

    try:
      ctx, actions = proto_to_context(request)
    except ValueError as exc:
      _abort_invalid(context, str(exc))
      raise  # pragma: no cover
    try:
      if self._lock is not None:
        with self._lock:
          for frame in self.policy.rollout(ctx, actions):
            frame.t_done = frame.t_done or time.perf_counter()
            yield frame_to_proto(frame)
      else:
        for frame in self.policy.rollout(ctx, actions):
          frame.t_done = frame.t_done or time.perf_counter()
          yield frame_to_proto(frame)
    except NotImplementedError as exc:
      _abort_unimplemented(context, str(exc))
      raise  # pragma: no cover
    with self._counter_lock:
      self.num_rollout += 1

  def Reset(self, request: Any, context: Any) -> Any:  # noqa: N802
    from eeb.protocol import eeb_pb2

    if self._lock is not None:
      with self._lock:
        self.policy.reset(request.session_id)
    else:
      self.policy.reset(request.session_id)
    with self._counter_lock:
      self.num_reset += 1
    return eeb_pb2.Empty()


def _abort_invalid(context: Any, message: str) -> None:
  grpc = _grpc()
  context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)


def _abort_unimplemented(context: Any, message: str) -> None:
  grpc = _grpc()
  context.abort(grpc.StatusCode.UNIMPLEMENTED, message)


def _abort_internal(context: Any, message: str) -> None:
  grpc = _grpc()
  context.abort(grpc.StatusCode.INTERNAL, message)


def build_servicer(policy: Policy, serialize_calls: bool = False) -> Any:
  """Create a servicer instance that also derives from the generated base class."""
  from eeb.protocol import eeb_pb2_grpc

  class _BoundServicer(PolicyServicer, eeb_pb2_grpc.EmbodiedPolicyServicer):
    pass

  return _BoundServicer(policy, serialize_calls=serialize_calls)


class HttpPolicyServerHandle:
  """Lifecycle wrapper matching `grpc.Server.stop` / `wait_for_termination` for HTTP policies."""

  def __init__(self, httpd: ThreadingHTTPServer, thread: threading.Thread, ep: Endpoint, policy: Policy) -> None:
    self._httpd = httpd
    self._thread = thread
    self.eeb_endpoint = ep
    self.eeb_bound_port = int(httpd.server_address[1])
    self.eeb_servicer = PolicyServicer(policy, serialize_calls=False)

  def stop(self, grace: Optional[float] = None) -> None:
    self._httpd.shutdown()
    self._httpd.server_close()
    self._thread.join(timeout=2.0 if grace is None else max(0.1, float(grace)))

  def wait_for_termination(self, timeout: Optional[float] = None) -> None:
    self._thread.join(timeout=timeout)


def _decode_wire_tensor(spec: Any, *, max_bytes: int, expected_dtype: np.dtype) -> np.ndarray:
  if not isinstance(spec, dict) or "b64" not in spec or "shape" not in spec:
    raise ValueError("Wire tensor requires {'b64': ..., 'shape': [...]}.")
  shape_raw = spec["shape"]
  if not isinstance(shape_raw, list) or len(shape_raw) > 5:
    raise ValueError("Invalid wire tensor shape list.")
  shape = []
  total_elems = 1
  for d in shape_raw:
    di = int(d)
    if di < 0 or di > 65536:
      raise ValueError(f"Wire tensor dimension {di} out of bounds.")
    shape.append(di)
    total_elems *= di
    if total_elems * expected_dtype.itemsize > max_bytes:
      raise ValueError(f"Declared wire tensor shape exceeds {max_bytes} bytes.")
  b64_str = spec["b64"]
  if not isinstance(b64_str, str) or len(b64_str) > (max_bytes * 4 // 3) + 64:
    raise ValueError("Wire tensor payload exceeds maximum base64 length.")
  raw = base64.b64decode(b64_str, validate=True)
  expected_bytes = total_elems * expected_dtype.itemsize
  if len(raw) != expected_bytes:
    raise ValueError(f"Wire tensor byte count {len(raw)} does not match shape {shape} ({expected_bytes} bytes).")
  return np.frombuffer(raw, dtype=expected_dtype).reshape(tuple(shape)).copy()


def serve_http_policy(
    policy: Policy,
    endpoint: str | Endpoint,
    *,
    serialize_calls: bool = False,
    max_message_mb: int = 64,
) -> HttpPolicyServerHandle:
  """Start a threaded HTTP/JSON server for `policy` on `http://host:port`."""
  validate_policy(policy)
  ep = parse_endpoint(endpoint) if isinstance(endpoint, str) else endpoint
  if not ep.is_http:
    raise ValueError(f"serve_http_policy requires an http://host:port endpoint, got {ep!r}.")
  host, _, port_s = ep.target.rpartition(":")
  port = int(port_s)
  max_bytes = max(1, int(max_message_mb)) * 1024 * 1024
  lock: Optional[threading.Lock] = threading.Lock() if serialize_calls else None

  class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
      return

    def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
      body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
      raw_cl = self.headers.get("Content-Length")
      if raw_cl is None:
        self._send_json(400, {"error": "Missing Content-Length header."})
        return None
      try:
        length = int(raw_cl)
      except (TypeError, ValueError):
        self._send_json(400, {"error": "Invalid Content-Length header."})
        return None
      if length < 0:
        self._send_json(400, {"error": "Negative Content-Length header."})
        return None
      if length > max_bytes:
        self._send_json(413, {"error": f"Request body length {length} exceeds maximum {max_bytes} bytes."})
        return None
      body = self.rfile.read(length) if length > 0 else b"{}"
      try:
        data = json.loads(body.decode("utf-8"))
      except Exception as exc:  # noqa: BLE001
        self._send_json(400, {"error": f"Invalid JSON payload: {exc}"})
        return None
      if not isinstance(data, dict):
        self._send_json(400, {"error": "JSON body must be an object."})
        return None
      return data

    def do_GET(self) -> None:  # noqa: N802
      path = self.path.split("?", 1)[0]
      if path == "/health":
        caps = policy.capabilities()
        self._send_json(200, {"status": "ok", "model_id": caps.model_id, "track": caps.track})
        return
      if path == "/capabilities":
        self._send_json(200, policy.capabilities().to_dict())
        return
      self._send_json(404, {"error": f"Unknown path {path!r}"})

    def do_POST(self) -> None:  # noqa: N802
      path = self.path.split("?", 1)[0]
      data = self._read_json_body()
      if data is None:
        return
      try:
        if path == "/reset":
          sid = str(data.get("session_id", ""))
          if lock is not None:
            with lock:
              policy.reset(sid)
          else:
            policy.reset(sid)
          self._send_json(200, {"ok": True})
          return
        if path == "/predict":
          t_recv = time.perf_counter()
          images_in: dict[str, np.ndarray] = {}
          raw_imgs = data.get("images", {})
          if not isinstance(raw_imgs, dict) or len(raw_imgs) > 16:
            raise ValueError("Invalid images mapping.")
          for cam, spec in raw_imgs.items():
            images_in[str(cam)] = _decode_wire_tensor(spec, max_bytes=max_bytes, expected_dtype=np.dtype("uint8"))
          proprio = _decode_wire_tensor(data["proprio"], max_bytes=max_bytes, expected_dtype=np.dtype("<f4"))
          obs = Observation(
              session_id=str(data.get("session_id", "")),
              step=int(data.get("step", 0)),
              images=images_in,
              proprio=proprio,
              instruction=str(data.get("instruction", "")),
              t_send=float(data.get("t_send", 0.0)),
          )
          if lock is not None:
            with lock:
              chunk = policy.predict(obs)
          else:
            chunk = policy.predict(obs)
          if not isinstance(chunk, ActionChunk):
            chunk = ActionChunk(actions=np.asarray(chunk, dtype=np.float32))
          t_done = time.perf_counter()
          act_arr = np.ascontiguousarray(chunk.actions, dtype="<f4")
          self._send_json(
              200,
              {
                  "actions_b64": base64.b64encode(act_arr.tobytes()).decode("ascii"),
                  "shape": list(act_arr.shape),
                  "t_recv": float(t_recv),
                  "t_done": float(t_done),
              },
          )
          return
        if path == "/rollout":
          raw_frames = data.get("frames", [])
          if not isinstance(raw_frames, list) or len(raw_frames) > 64:
            raise ValueError("Invalid frames list.")
          frames_in = [
              _decode_wire_tensor(f_spec, max_bytes=max_bytes, expected_dtype=np.dtype("uint8"))
              for f_spec in raw_frames
          ]
          proprio = _decode_wire_tensor(data["proprio"], max_bytes=max_bytes, expected_dtype=np.dtype("<f4"))
          actions = _decode_wire_tensor(data["actions"], max_bytes=max_bytes, expected_dtype=np.dtype("<f4"))
          ctx = Context(
              session_id=str(data.get("session_id", "")),
              frames=frames_in,
              proprio=proprio,
              instruction=str(data.get("instruction", "")),
              num_frames=int(data.get("num_frames", 0)),
              t_send=float(data.get("t_send", 0.0)),
          )
          if lock is not None:
            with lock:
              frames_out = list(policy.rollout(ctx, actions))
          else:
            frames_out = list(policy.rollout(ctx, actions))
          wire_out = []
          for fr in frames_out:
            arr = np.ascontiguousarray(fr.image, dtype=np.uint8)
            wire_out.append({
                "index": int(fr.index),
                "camera": str(fr.camera),
                "t_done": float(fr.t_done or time.perf_counter()),
                "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
                "shape": list(arr.shape),
            })
          self._send_json(200, {"frames": wire_out})
          return
      except ValueError as exc:
        self._send_json(400, {"error": str(exc)})
        return
      except NotImplementedError as exc:
        self._send_json(501, {"error": str(exc)})
        return
      except Exception as exc:  # noqa: BLE001
        _LOG.exception("HTTP policy server failed on %s", path)
        self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
        return
      self._send_json(404, {"error": f"Unknown path {path!r}"})

  httpd = ThreadingHTTPServer((host, port), _Handler)
  bound_ep = Endpoint("http", f"{host}:{httpd.server_address[1]}")
  th = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
  th.start()
  _LOG.info("eeb HTTP policy server listening on %s (%s)", bound_ep.http_url, type(policy).__name__)
  return HttpPolicyServerHandle(httpd, th, bound_ep, policy)


def serve_policy(
    policy: Policy,
    endpoint: str,
    max_workers: int = 16,
    serialize_calls: bool = False,
    max_message_mb: int = 256,
) -> Any:
  """Start a gRPC or HTTP server for `policy` on `endpoint` and return the started server.

  The caller owns the server lifecycle (`server.stop(grace)` / `wait_for_termination()`).
  For Unix-domain sockets a stale socket file is removed before binding.
  """
  ep = parse_endpoint(endpoint)
  if ep.is_inproc:
    raise ValueError("Cannot serve an inproc: endpoint; pass unix:///path.sock, tcp://host:port, or http://host:port.")
  if ep.is_http:
    return serve_http_policy(
        policy,
        ep,
        serialize_calls=serialize_calls,
        max_message_mb=min(int(max_message_mb), 64),
    )

  grpc = _grpc()
  from concurrent import futures

  from eeb.protocol import eeb_pb2_grpc

  if ep.kind == "unix":
    sock_dir = os.path.dirname(ep.target)
    if sock_dir:
      os.makedirs(sock_dir, exist_ok=True)
    if os.path.exists(ep.target):
      os.unlink(ep.target)

  options = [
      ("grpc.max_send_message_length", max_message_mb * 1024 * 1024),
      ("grpc.max_receive_message_length", max_message_mb * 1024 * 1024),
  ]
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
  servicer = build_servicer(policy, serialize_calls=serialize_calls)
  eeb_pb2_grpc.add_EmbodiedPolicyServicer_to_server(servicer, server)
  bound_port = server.add_insecure_port(ep.grpc_target)
  if bound_port == 0:
    hint = (
        "use a shorter socket path under /tmp/ (<=107 bytes) and check directory permissions"
        if ep.kind == "unix"
        else "check whether the TCP port is already in use"
    )
    raise OSError(f"Failed to bind gRPC server to {ep.grpc_target}; {hint}.")
  server.start()
  server.eeb_servicer = servicer  # type: ignore[attr-defined]
  server.eeb_endpoint = ep  # type: ignore[attr-defined]
  server.eeb_bound_port = bound_port  # type: ignore[attr-defined]
  _LOG.info("eeb policy server listening on %s (%s)", ep.grpc_target, type(policy).__name__)
  return server


__all__ = [
    "DynamicMicroBatcher",
    "Endpoint",
    "HttpPolicyServerHandle",
    "MAX_UNIX_SOCKET_PATH_BYTES",
    "PolicyServicer",
    "TORCHTPU_INSTALL_HINT",
    "build_servicer",
    "maybe_compile_torch_module",
    "normalize_and_validate_uds_path",
    "parse_endpoint",
    "resolve_torch_device",
    "serve_http_policy",
    "serve_policy",
    "synchronize_torch_device",
]
