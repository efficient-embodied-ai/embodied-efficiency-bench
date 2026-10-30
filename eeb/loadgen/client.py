"""LoadGen client: one interface over gRPC (UDS / TCP), HTTP, and in-process policies.

All implementations return the same `PredictResult` so the sweep code never branches on
transport. Latency is measured client-side with `time.perf_counter()` around the whole call
(serialize -> transport -> server -> deserialize), which is the obs->action latency a robot
runtime would see. Server timestamps (`t_recv`, `t_done`) use `time.perf_counter()` and are
only used for the same-host overhead decomposition (`server_compute_ms`).
"""

from __future__ import annotations

import abc
import base64
from dataclasses import dataclass, field
import http.client
import json
import threading
import time
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

import numpy as np

from eeb.protocol.policy import ActionChunk, Caps, Context, Frame, Observation, Policy, instantiate_policy, validate_policy
from eeb.servers.base import Endpoint, parse_endpoint


@dataclass
class PredictResult:
  """One completed Predict call with client-side timing."""

  chunk: ActionChunk
  latency_ms: float
  t_send_wall: float
  t_recv_wall: float
  server_compute_ms: Optional[float] = None
  request_bytes: int = 0
  response_bytes: int = 0


@dataclass
class RolloutResult:
  """One completed streaming Rollout with per-frame arrival times (ms since send)."""

  frames: List[Frame]
  frame_arrival_ms: List[float] = field(default_factory=list)
  total_ms: float = 0.0

  @property
  def ttff_ms(self) -> Optional[float]:
    """Time to first frame."""
    return self.frame_arrival_ms[0] if self.frame_arrival_ms else None


class PolicyClient(abc.ABC):
  """Transport-agnostic client interface used by the LoadGen."""

  transport: str = "abstract"

  @abc.abstractmethod
  def capabilities(self) -> Caps: ...

  @abc.abstractmethod
  def reset(self, session_id: str) -> float:
    """Reset a session; returns the client-side latency in ms."""

  @abc.abstractmethod
  def predict(self, obs: Observation) -> PredictResult: ...

  @abc.abstractmethod
  def rollout(self, ctx: Context, actions: np.ndarray) -> RolloutResult: ...

  def close(self) -> None:
    return None

  def __enter__(self) -> "PolicyClient":
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()


class InProcessPolicyClient(PolicyClient):
  """Calls a `Policy` directly (dev mode). The policy instance may be shared by many clients."""

  transport = "inproc"

  def __init__(self, policy: Policy) -> None:
    validate_policy(policy)
    self.policy = policy

  def capabilities(self) -> Caps:
    return self.policy.capabilities()

  def reset(self, session_id: str) -> float:
    t0 = time.perf_counter()
    self.policy.reset(session_id)
    return (time.perf_counter() - t0) * 1000.0

  def predict(self, obs: Observation) -> PredictResult:
    obs.t_send = time.time()
    t0 = time.perf_counter()
    t_recv_srv = time.perf_counter()
    chunk = self.policy.predict(obs)
    if not isinstance(chunk, ActionChunk):
      chunk = ActionChunk(actions=np.asarray(chunk, dtype=np.float32))
    chunk.t_recv = chunk.t_recv or t_recv_srv
    chunk.t_done = chunk.t_done or time.perf_counter()
    t1 = time.perf_counter()
    return PredictResult(
        chunk=chunk,
        latency_ms=(t1 - t0) * 1000.0,
        t_send_wall=obs.t_send,
        t_recv_wall=time.time(),
        server_compute_ms=chunk.server_compute_ms,
    )

  def rollout(self, ctx: Context, actions: np.ndarray) -> RolloutResult:
    ctx.t_send = time.time()
    t0 = time.perf_counter()
    frames: List[Frame] = []
    arrivals: List[float] = []
    for frame in self.policy.rollout(ctx, actions):
      arrivals.append((time.perf_counter() - t0) * 1000.0)
      frames.append(frame)
    return RolloutResult(frames=frames, frame_arrival_ms=arrivals, total_ms=(time.perf_counter() - t0) * 1000.0)


class GrpcPolicyClient(PolicyClient):
  """gRPC client over a Unix-domain socket or TCP."""

  transport = "grpc"

  def __init__(self, endpoint: str | Endpoint, timeout_s: float = 60.0, max_message_mb: int = 256, wait_for_ready_s: float = 10.0) -> None:
    try:
      import grpc  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
      raise ImportError("grpcio is required for socket endpoints; `pip install -e '.[grpc]'`.") from exc
    from eeb.protocol import eeb_pb2, eeb_pb2_grpc

    self._pb2 = eeb_pb2
    self._grpc = grpc
    self.endpoint = parse_endpoint(endpoint) if isinstance(endpoint, str) else endpoint
    if self.endpoint.is_inproc:
      raise ValueError("GrpcPolicyClient cannot open an inproc: endpoint; use InProcessPolicyClient.")
    self.timeout_s = float(timeout_s)
    options = [
        ("grpc.max_send_message_length", max_message_mb * 1024 * 1024),
        ("grpc.max_receive_message_length", max_message_mb * 1024 * 1024),
    ]
    self.channel = grpc.insecure_channel(self.endpoint.grpc_target, options=options)
    if wait_for_ready_s > 0:
      try:
        grpc.channel_ready_future(self.channel).result(timeout=wait_for_ready_s)
      except grpc.FutureTimeoutError as exc:
        self.channel.close()
        raise ConnectionError(
            f"No EEB policy server reachable at {self.endpoint.grpc_target} within {wait_for_ready_s:.0f}s. "
            "Start one with `eeb serve --policy <module:Class> --uds /path.sock` (or --tcp host:port)."
        ) from exc
    self.stub = eeb_pb2_grpc.EmbodiedPolicyStub(self.channel)

  def capabilities(self) -> Caps:
    from eeb.protocol.codec import proto_to_caps

    return proto_to_caps(self.stub.Capabilities(self._pb2.Empty(), timeout=self.timeout_s))

  def reset(self, session_id: str) -> float:
    t0 = time.perf_counter()
    self.stub.Reset(self._pb2.Session(session_id=session_id), timeout=self.timeout_s)
    return (time.perf_counter() - t0) * 1000.0

  def predict(self, obs: Observation) -> PredictResult:
    from eeb.protocol.codec import observation_to_proto, proto_to_action_chunk

    obs.t_send = time.time()
    t0 = time.perf_counter()
    request = observation_to_proto(obs)
    response = self.stub.Predict(request, timeout=self.timeout_s)
    chunk = proto_to_action_chunk(response)
    t1 = time.perf_counter()
    return PredictResult(
        chunk=chunk,
        latency_ms=(t1 - t0) * 1000.0,
        t_send_wall=obs.t_send,
        t_recv_wall=time.time(),
        server_compute_ms=chunk.server_compute_ms,
        request_bytes=request.ByteSize(),
        response_bytes=response.ByteSize(),
    )

  def rollout(self, ctx: Context, actions: np.ndarray) -> RolloutResult:
    from eeb.protocol.codec import context_to_proto, proto_to_frame

    ctx.t_send = time.time()
    t0 = time.perf_counter()
    frames: List[Frame] = []
    arrivals: List[float] = []
    for msg in self.stub.Rollout(context_to_proto(ctx, actions), timeout=self.timeout_s):
      arrivals.append((time.perf_counter() - t0) * 1000.0)
      frames.append(proto_to_frame(msg))
    return RolloutResult(frames=frames, frame_arrival_ms=arrivals, total_ms=(time.perf_counter() - t0) * 1000.0)

  def close(self) -> None:
    try:
      self.channel.close()
    except Exception:  # pragma: no cover
      pass


class HttpPolicyClient(PolicyClient):
  """HTTP/JSON client for `http://host:port` endpoints."""

  transport = "http"

  def __init__(
      self,
      endpoint: str | Endpoint,
      timeout_s: float = 60.0,
      max_message_mb: int = 256,
      wait_for_ready_s: float = 10.0,
  ) -> None:
    self.endpoint = parse_endpoint(endpoint) if isinstance(endpoint, str) else endpoint
    if not self.endpoint.is_http:
      raise ValueError(f"HttpPolicyClient requires an http://host:port endpoint, got {endpoint!r}.")
    parsed = urlparse(self.endpoint.http_url)
    self._host = parsed.hostname or "127.0.0.1"
    self._port = int(parsed.port or 8080)
    self.timeout_s = float(timeout_s)
    self._max_bytes = int(max_message_mb) * 1024 * 1024
    self._local = threading.local()
    self._conns: List[http.client.HTTPConnection] = []
    self._conns_lock = threading.Lock()
    if wait_for_ready_s > 0:
      deadline = time.perf_counter() + float(wait_for_ready_s)
      last_err: Optional[Exception] = None
      while True:
        try:
          self._request_json("GET", "/health", None, timeout_s=min(2.0, float(wait_for_ready_s)))
          break
        except Exception as exc:  # noqa: BLE001
          last_err = exc
          self._drop_thread_conn()
          if time.perf_counter() >= deadline:
            self.close()
            raise ConnectionError(
                f"No EEB HTTP policy server reachable at {self.endpoint.http_url} within {wait_for_ready_s:.0f}s "
                f"(last error: {last_err}). Start one with `eeb serve --model pi05 --port {self._port}`."
            ) from exc
          time.sleep(0.05)

  def _get_conn(self, timeout_s: Optional[float] = None) -> http.client.HTTPConnection:
    conn = getattr(self._local, "conn", None)
    eff_timeout = self.timeout_s if timeout_s is None else float(timeout_s)
    if conn is None:
      conn = http.client.HTTPConnection(self._host, self._port, timeout=eff_timeout)
      self._local.conn = conn
      with self._conns_lock:
        self._conns.append(conn)
    else:
      conn.timeout = eff_timeout
    return conn

  def _drop_thread_conn(self) -> None:
    conn = getattr(self._local, "conn", None)
    if conn is not None:
      try:
        conn.close()
      except Exception:  # noqa: BLE001
        pass
      self._local.conn = None

  def _request_json(
      self,
      method: str,
      path: str,
      payload: Optional[Dict[str, Any]],
      *,
      timeout_s: Optional[float] = None,
  ) -> tuple[Dict[str, Any], int, int]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else b""
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"} if payload is not None else {"Connection": "keep-alive"}
    for attempt in range(2):
      conn = self._get_conn(timeout_s=timeout_s)
      try:
        conn.request(method, path, body=body if payload is not None else None, headers=headers)
        resp = conn.getresponse()
        raw = resp.read(self._max_bytes + 1)
        if len(raw) > self._max_bytes:
          raise ValueError(f"HTTP response exceeded {self._max_bytes} bytes on {path}.")
        status = int(resp.status)
        if status >= 400:
          raise RuntimeError(f"HTTP {status} from {self.endpoint.http_url}{path}: {raw.decode('utf-8', errors='replace')}")
        return json.loads(raw.decode("utf-8")), len(body), len(raw)
      except (http.client.NotConnected, http.client.BadStatusLine, BrokenPipeError, ConnectionResetError):
        self._drop_thread_conn()
        if attempt == 1:
          raise
    raise RuntimeError(f"Unreachable retry loop on {path}")  # pragma: no cover

  def capabilities(self) -> Caps:
    data, _, _ = self._request_json("GET", "/capabilities", None)
    return Caps.from_dict(data)

  def reset(self, session_id: str) -> float:
    t0 = time.perf_counter()
    self._request_json("POST", "/reset", {"session_id": str(session_id)})
    return (time.perf_counter() - t0) * 1000.0

  def predict(self, obs: Observation) -> PredictResult:
    obs.t_send = time.time()
    t0 = time.perf_counter()
    images_wire: Dict[str, Dict[str, Any]] = {}
    for cam, img in obs.images.items():
      arr = np.ascontiguousarray(img, dtype=np.uint8)
      images_wire[cam] = {
          "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
          "shape": list(arr.shape),
          "dtype": "uint8",
      }
    prop = np.ascontiguousarray(obs.proprio, dtype=np.float32)
    req = {
        "session_id": obs.session_id,
        "step": int(obs.step),
        "instruction": obs.instruction,
        "images": images_wire,
        "proprio": {
            "b64": base64.b64encode(prop.tobytes()).decode("ascii"),
            "shape": list(prop.shape),
            "dtype": "float32",
        },
        "t_send": float(obs.t_send),
    }
    resp, req_bytes, resp_bytes = self._request_json("POST", "/predict", req)
    act_shape = tuple(int(d) for d in resp["shape"])
    act_raw = base64.b64decode(resp["actions_b64"])
    actions = np.frombuffer(act_raw, dtype=np.float32).reshape(act_shape).copy()
    chunk = ActionChunk(
        actions=actions,
        t_recv=float(resp.get("t_recv", 0.0)),
        t_done=float(resp.get("t_done", 0.0)),
    )
    t1 = time.perf_counter()
    return PredictResult(
        chunk=chunk,
        latency_ms=(t1 - t0) * 1000.0,
        t_send_wall=obs.t_send,
        t_recv_wall=time.time(),
        server_compute_ms=chunk.server_compute_ms,
        request_bytes=req_bytes,
        response_bytes=resp_bytes,
    )

  def rollout(self, ctx: Context, actions: np.ndarray) -> RolloutResult:
    ctx.t_send = time.time()
    t0 = time.perf_counter()
    frames_wire = []
    for frm in ctx.frames:
      arr = np.ascontiguousarray(frm, dtype=np.uint8)
      frames_wire.append({
          "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
          "shape": list(arr.shape),
          "dtype": "uint8",
      })
    prop = np.ascontiguousarray(ctx.proprio, dtype=np.float32)
    act = np.ascontiguousarray(actions, dtype=np.float32)
    req = {
        "session_id": ctx.session_id,
        "instruction": ctx.instruction,
        "num_frames": int(ctx.num_frames),
        "frames": frames_wire,
        "proprio": {
            "b64": base64.b64encode(prop.tobytes()).decode("ascii"),
            "shape": list(prop.shape),
            "dtype": "float32",
        },
        "actions": {
            "b64": base64.b64encode(act.tobytes()).decode("ascii"),
            "shape": list(act.shape),
            "dtype": "float32",
        },
        "t_send": float(ctx.t_send),
    }
    resp, _, _ = self._request_json("POST", "/rollout", req)
    done_ms = (time.perf_counter() - t0) * 1000.0
    frames_out: List[Frame] = []
    arrivals: List[float] = []
    for item in resp.get("frames", []):
      f_shape = tuple(int(d) for d in item["shape"])
      f_raw = base64.b64decode(item["b64"])
      img = np.frombuffer(f_raw, dtype=np.uint8).reshape(f_shape).copy()
      frames_out.append(
          Frame(
              index=int(item["index"]),
              image=img,
              t_done=float(item.get("t_done", 0.0)),
              camera=str(item.get("camera", "wm")),
          )
      )
      arrivals.append(done_ms)
    return RolloutResult(frames=frames_out, frame_arrival_ms=arrivals, total_ms=done_ms)

  def close(self) -> None:
    with self._conns_lock:
      for conn in self._conns:
        try:
          conn.close()
        except Exception:  # noqa: BLE001
          pass
      self._conns.clear()
    self._local = threading.local()


def open_client(endpoint: str, policy: Optional[Policy] = None, policy_kwargs: Optional[Dict[str, Any]] = None, **grpc_kwargs: Any) -> PolicyClient:
  """Factory: `inproc:` -> `InProcessPolicyClient`, `http://` -> `HttpPolicyClient`, else `GrpcPolicyClient`.

  Pass an already-constructed `policy` to share one in-process instance across many clients
  (the sweep does this so every "robot" talks to the same policy object).
  """
  ep = parse_endpoint(endpoint)
  if ep.is_inproc:
    return InProcessPolicyClient(policy if policy is not None else instantiate_policy(ep.target, policy_kwargs))
  if ep.is_http:
    return HttpPolicyClient(ep, **grpc_kwargs)
  return GrpcPolicyClient(ep, **grpc_kwargs)


make_client = open_client


__all__ = [
    "GrpcPolicyClient",
    "HttpPolicyClient",
    "InProcessPolicyClient",
    "PolicyClient",
    "PredictResult",
    "RolloutResult",
    "make_client",
    "open_client",
]
