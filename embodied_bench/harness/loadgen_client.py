"""MLPerf-style LoadGen client (`EmbodiedLoadGenClient`) and crash-safe state registry for EmbodiedX (`embodied-efficiency-bench`).

Supports:
  1. Single-Stream Real-Time Mode (`concurrency=1`): Sequential closed-loop requests
     measuring real wall-clock `time.perf_counter_ns()` latency (`ttfa_ms` / `ttff_ms`,
     `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `jitter_ms`, `hz_react` / `hz_plan`).
  2. Multi-Stream Concurrency Sweep (`concurrency in [1, 2, 4, 8, 16, 32, 64]`):
     `concurrent.futures.ThreadPoolExecutor` driving `C` concurrent robot streams with
     sticky `session_id` affinity (`robot_stream_0` .. `robot_stream_{C-1}`), measuring
     real aggregate throughput (`actions_per_sec` or `fps`), tail `p99` latency, and
     KV/temporal cache hit ratio.
  3. Crash-safe, concurrent-safe LoadGen session state registry (`record_loadgen_session_state`)
     built on `atomic_read_modify_write_json` and `validate_status_and_reason` (Gates 1 & 2).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import http.client
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import numpy as np

from embodied_bench.harness.server_protocol import (
    EmbodiedBenchmarkServer,
    decode_ndarray_b64,
    encode_payload_arrays,
)
from embodied_bench.schema import (
    ExecutionStrategy,
    RunStatus,
    TrackType,
    atomic_read_modify_write_json,
    is_wm_track,
    now_pacific_iso,
    validate_status_and_reason,
)


DEFAULT_LOADGEN_CONCURRENCY_GRID: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class LoadGenStreamResult:
  """Latency, reactivity, throughput, and cache telemetry for a LoadGen execution."""

  track: str
  concurrency: int
  num_streams: int
  steps_per_stream: int
  total_requests: int
  warmup_steps: int
  wall_elapsed_s: float
  ttfa_ms: Optional[float]
  ttff_ms: Optional[float]
  mean_ms: float
  p50_ms: float
  p95_ms: float
  p99_ms: float
  jitter_ms: float
  hz_react: Optional[float]
  hz_plan: Optional[float]
  actions_per_sec: Optional[float]
  chunks_per_sec: Optional[float]
  fps: Optional[float]
  cache_hits: int
  cache_hit_ratio: float
  slo_target_ms: float
  slo_miss_rate_pct: float
  session_ids: List[str] = field(default_factory=list)
  latencies_ms: List[float] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def record_loadgen_session_state(
    registry_path: str | Path,
    session_run_id: str,
    status: str | RunStatus,
    concurrency: int = 1,
    metrics: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    notes: Optional[str] = None,
    slo_preset: Optional[str] = None,
    error_code: Optional[str] = None,
    enforce_transitions: bool = True,
) -> Dict[str, Any]:
  """Atomically record LoadGen session state without stale optional field carryover (Gates 1 & 2).

  Args:
    registry_path: Target JSON file path for the LoadGen state registry.
    session_run_id: Unique identifier for the LoadGen run or concurrency slice.
    status: Target `RunStatus` (`PENDING`, `RUNNING`, `CERTIFIED`, `REJECTED`, `BLOCKED`, `FAILED`, `SKIPPED`, `INVALID`).
    concurrency: Active stream concurrency `C >= 1`.
    metrics: Measured latency/throughput metrics dict for this run.
    reason: Mandatory non-empty explanation when `status` is `REJECTED`, `BLOCKED`, `FAILED`, `SKIPPED`, or `INVALID`.
    notes: Optional human-readable operator notes (omitted/cleared on repeat calls if not passed).
    slo_preset: Optional SLO preset label (omitted/cleared on repeat calls if not passed).
    error_code: Optional machine-readable error code (omitted/cleared on repeat calls if not passed).
    enforce_transitions: Whether to validate state transitions against `VALID_STATE_TRANSITIONS`.

  Returns:
    Updated registry state dictionary after atomic fsync + replace.
  """
  if not session_run_id or not str(session_run_id).strip():
    raise ValueError("session_run_id must be a non-empty string.")
  if int(concurrency) < 1:
    raise ValueError(f"concurrency must be >= 1, got {concurrency}.")

  def _modify(state: Dict[str, Any]) -> Dict[str, Any]:
    sessions = state.setdefault("loadgen_sessions", {})
    existing = sessions.get(session_run_id)
    prev_status = (
        existing.get("status")
        if (enforce_transitions and isinstance(existing, dict))
        else None
    )
    resolved_status = validate_status_and_reason(
        status=status,
        reason=reason,
        previous_status=prev_status,
    )
    # Build a fresh record dictionary so omitted optional fields (`notes`, `slo_preset`,
    # `error_code`, `reason`) NEVER carry over stale values from an earlier call (Gate 1 Repeat Path).
    fresh_record: Dict[str, Any] = {
        "session_run_id": str(session_run_id),
        "status": resolved_status.value,
        "concurrency": int(concurrency),
        "updated_at": now_pacific_iso(),
        "metrics": dict(metrics) if metrics is not None else {},
        "reason": (
            reason.strip()
            if isinstance(reason, str) and reason.strip()
            else None
        ),
        "notes": (
            notes.strip()
            if isinstance(notes, str) and notes.strip()
            else None
        ),
        "slo_preset": (
            slo_preset.strip()
            if isinstance(slo_preset, str) and slo_preset.strip()
            else None
        ),
        "error_code": (
            error_code.strip()
            if isinstance(error_code, str) and error_code.strip()
            else None
        ),
    }
    sessions[str(session_run_id)] = fresh_record
    state["updated_at"] = now_pacific_iso()
    return state

  return atomic_read_modify_write_json(
      target_path=registry_path,
      modifier_fn=_modify,
      default_factory=lambda: {
          "schema_version": "1.0.0",
          "loadgen_sessions": {},
      },
  )


class EmbodiedLoadGenClient:
  """Client driving single-stream real-time (`C=1`) and multi-stream (`C in [1..64]`) workloads against `EmbodiedBenchmarkServer`."""

  def __init__(
      self,
      base_url: Optional[str] = None,
      *,
      host: str = "127.0.0.1",
      port: int = 8000,
      in_process_server: Optional[EmbodiedBenchmarkServer] = None,
      timeout_s: float = 15.0,
  ) -> None:
    self.in_process_server = in_process_server
    self.timeout_s = float(timeout_s)
    if base_url is not None:
      parsed = urlparse(base_url)
      self.host = parsed.hostname or host
      self.port = int(parsed.port or port)
      self.base_url = f"http://{self.host}:{self.port}"
    elif in_process_server is not None and in_process_server.port > 0:
      self.host = in_process_server.host
      self.port = in_process_server.port
      self.base_url = in_process_server.base_url
    else:
      self.host = str(host)
      self.port = int(port)
      self.base_url = f"http://{self.host}:{self.port}"
    self._tls = threading.local()

  def _get_connection(self) -> http.client.HTTPConnection:
    conn: Optional[http.client.HTTPConnection] = getattr(
        self._tls, "conn", None
    )
    if conn is None or conn.host != self.host or conn.port != self.port:
      conn = http.client.HTTPConnection(
          self.host, self.port, timeout=self.timeout_s
      )
      self._tls.conn = conn
    return conn

  def _reset_connection(self) -> None:
    conn: Optional[http.client.HTTPConnection] = getattr(
        self._tls, "conn", None
    )
    if conn is not None:
      try:
        conn.close()
      except Exception:
        pass
    self._tls.conn = None

  def _request_json(
      self,
      method: str,
      path: str,
      payload: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Execute an HTTP request (or in-process fallback when no socket is bound) and return parsed JSON."""
    if self.in_process_server is not None and self.port <= 0:
      if method == "GET" and path == "/health":
        return self.in_process_server.get_health_snapshot()
      if method == "POST" and path == "/v1/vla/act":
        return self.in_process_server.handle_vla_act(payload or {})
      if method == "POST" and path == "/v1/wm/step":
        return self.in_process_server.handle_wm_step(payload or {})
      if method == "POST" and path == "/v1/reset_session":
        return self.in_process_server.handle_reset_session(payload or {})
      raise ValueError(f"Unsupported in-process endpoint: {method} {path}")

    body_bytes = (
        json.dumps(payload).encode("utf-8") if payload is not None else b""
    )
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "Connection": "close",
    }
    conn = http.client.HTTPConnection(
        self.host, self.port, timeout=self.timeout_s
    )
    try:
      conn.request(
          method,
          path,
          body=body_bytes if method == "POST" else None,
          headers=headers,
      )
      resp = conn.getresponse()
      raw_resp = resp.read().decode("utf-8")
      data = json.loads(raw_resp) if raw_resp else {}
      if resp.status >= 400:
        err_msg = data.get("error", f"HTTP {resp.status}")
        raise RuntimeError(
            f"Server returned HTTP {resp.status} on {method} {path}: {err_msg}"
        )
      return data
    finally:
      conn.close()

  def check_health(self) -> Dict[str, Any]:
    """Query `GET /health` on the benchmark server."""
    return self._request_json("GET", "/health")

  def reset_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Reset sticky session state on the server via `POST /v1/reset_session`."""
    return self._request_json(
        "POST", "/v1/reset_session", {"session_id": session_id}
    )

  def send_vla_act(
      self,
      observations: Dict[str, Any],
      session_id: str = "robot_stream_0",
      chunk_size: int = 8,
      action_dim: int = 7,
      execution_strategy: str = ExecutionStrategy.ASYNC_CHUNK.value,
      emergency_stop: bool = False,
  ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Send a Track 1 VLA policy inference request and measure wall-clock latency via `time.perf_counter_ns()`."""
    req_payload = {
        "session_id": session_id,
        "chunk_size": int(chunk_size),
        "action_dim": int(action_dim),
        "execution_strategy": str(execution_strategy),
        "emergency_stop": bool(emergency_stop),
        "observations": encode_payload_arrays(observations),
    }
    t0_ns = time.perf_counter_ns()
    resp = self._request_json("POST", "/v1/vla/act", req_payload)
    latency_ms = (time.perf_counter_ns() - t0_ns) / 1e6
    actions = decode_ndarray_b64(resp["actions"])
    return actions, float(latency_ms), resp

  def send_wm_step(
      self,
      observations: Dict[str, Any],
      session_id: str = "wm_stream_0",
      horizon_frames: int = 16,
      latent_dim: int = 16,
  ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Send a Track 2 World Model rollout request and measure wall-clock latency via `time.perf_counter_ns()`."""
    req_payload = {
        "session_id": session_id,
        "horizon_frames": int(horizon_frames),
        "latent_dim": int(latent_dim),
        "observations": encode_payload_arrays(observations),
    }
    t0_ns = time.perf_counter_ns()
    resp = self._request_json("POST", "/v1/wm/step", req_payload)
    latency_ms = (time.perf_counter_ns() - t0_ns) / 1e6
    predicted_latents = decode_ndarray_b64(resp["predicted_latents"])
    return predicted_latents, float(latency_ms), resp

  @staticmethod
  def _default_observation_for_step(step_idx: int, wm_mode: bool = False) -> Dict[str, Any]:
    if wm_mode:
      return {
          "action": np.full(
              (7,), 0.05 * ((step_idx % 5) + 1), dtype=np.float32
          ),
          "frame_latent": np.ones((4, 8), dtype=np.float32) * (0.1 + 0.01 * step_idx),
      }
    return {
        "proprio": np.linspace(
            0.0, 0.5 + 0.02 * step_idx, 7, dtype=np.float32
        ),
        "ee_pos": np.array([0.05 * (step_idx % 4), 0.0, 0.25], dtype=np.float32),
        "goal_pos": np.array([0.35, 0.20, 0.25], dtype=np.float32),
    }

  def _summarize_latencies(
      self,
      track: str,
      concurrency: int,
      steps_per_stream: int,
      warmup_steps: int,
      wall_elapsed_s: float,
      latencies_ms: Sequence[float],
      cache_hits: int,
      session_ids: Sequence[str],
      chunk_size: int,
      horizon_frames: int,
      target_slo_ms: float,
  ) -> LoadGenStreamResult:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    total_reqs = int(arr.size)
    mean_ms = round(float(np.mean(arr)), 4) if total_reqs > 0 else 0.0
    p50_ms = round(float(np.percentile(arr, 50)), 4) if total_reqs > 0 else 0.0
    p95_ms = round(float(np.percentile(arr, 95)), 4) if total_reqs > 0 else 0.0
    p99_ms = round(float(np.percentile(arr, 99)), 4) if total_reqs > 0 else 0.0
    jitter_ms = round(float(np.std(arr)), 4) if total_reqs > 0 else 0.0
    slo_misses = int(np.sum(arr > float(target_slo_ms))) if total_reqs > 0 else 0
    slo_miss_pct = round(
        (float(slo_misses) / float(max(1, total_reqs))) * 100.0, 3
    )
    cache_ratio = round(
        float(cache_hits) / float(max(1, total_reqs)), 4
    )
    safe_wall_s = max(float(wall_elapsed_s), 1e-9)
    safe_p99_ms = max(float(p99_ms), 1e-6)

    wm_mode = is_wm_track(track)
    if wm_mode:
      ttfa_ms = None
      ttff_ms = mean_ms
      hz_react = None
      hz_plan = round(1000.0 / safe_p99_ms, 4)
      actions_per_sec = None
      chunks_per_sec = None
      fps = round(float(total_reqs * horizon_frames) / safe_wall_s, 4)
    else:
      ttfa_ms = mean_ms
      ttff_ms = None
      hz_react = round(1000.0 / safe_p99_ms, 4)
      hz_plan = None
      chunks_per_sec = round(float(total_reqs) / safe_wall_s, 4)
      actions_per_sec = round(float(total_reqs * chunk_size) / safe_wall_s, 4)
      fps = None

    return LoadGenStreamResult(
        track=track,
        concurrency=int(concurrency),
        num_streams=int(concurrency),
        steps_per_stream=int(steps_per_stream),
        total_requests=total_reqs,
        warmup_steps=int(warmup_steps),
        wall_elapsed_s=round(float(wall_elapsed_s), 6),
        ttfa_ms=ttfa_ms,
        ttff_ms=ttff_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        jitter_ms=jitter_ms,
        hz_react=hz_react,
        hz_plan=hz_plan,
        actions_per_sec=actions_per_sec,
        chunks_per_sec=chunks_per_sec,
        fps=fps,
        cache_hits=int(cache_hits),
        cache_hit_ratio=cache_ratio,
        slo_target_ms=float(target_slo_ms),
        slo_miss_rate_pct=slo_miss_pct,
        session_ids=list(session_ids),
        latencies_ms=[round(float(x), 4) for x in latencies_ms],
    )

  def run_single_stream(
      self,
      track: str = TrackType.VLA_TRACK.value,
      num_steps: int = 16,
      warmup_steps: int = 2,
      session_id: str = "robot_stream_0",
      obs_generator: Optional[Callable[[int], Dict[str, Any]]] = None,
      chunk_size: int = 8,
      action_dim: int = 7,
      horizon_frames: int = 16,
      target_slo_ms: float = 50.0,
  ) -> LoadGenStreamResult:
    """Execute Single-Stream Real-Time Mode (`concurrency=1`) sequential closed-loop evaluation."""
    if int(num_steps) < 1:
      raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    wm_mode = is_wm_track(track)
    self.reset_session(session_id)

    for w in range(max(0, int(warmup_steps))):
      obs = (
          obs_generator(w)
          if obs_generator is not None
          else self._default_observation_for_step(w, wm_mode=wm_mode)
      )
      if wm_mode:
        self.send_wm_step(obs, session_id=session_id, horizon_frames=horizon_frames)
      else:
        self.send_vla_act(
            obs, session_id=session_id, chunk_size=chunk_size, action_dim=action_dim
        )

    latencies_ms: List[float] = []
    cache_hits = 0
    wall_t0_ns = time.perf_counter_ns()
    for s in range(int(num_steps)):
      obs = (
          obs_generator(s)
          if obs_generator is not None
          else self._default_observation_for_step(s, wm_mode=wm_mode)
      )
      if wm_mode:
        _, lat_ms, resp = self.send_wm_step(
            obs, session_id=session_id, horizon_frames=horizon_frames
        )
      else:
        _, lat_ms, resp = self.send_vla_act(
            obs, session_id=session_id, chunk_size=chunk_size, action_dim=action_dim
        )
      latencies_ms.append(lat_ms)
      if resp.get("cache_hit"):
        cache_hits += 1
    wall_elapsed_s = (time.perf_counter_ns() - wall_t0_ns) / 1e9

    return self._summarize_latencies(
        track=track,
        concurrency=1,
        steps_per_stream=num_steps,
        warmup_steps=warmup_steps,
        wall_elapsed_s=wall_elapsed_s,
        latencies_ms=latencies_ms,
        cache_hits=cache_hits,
        session_ids=[session_id],
        chunk_size=chunk_size,
        horizon_frames=horizon_frames,
        target_slo_ms=target_slo_ms,
    )

  def run_concurrency_level(
      self,
      concurrency: int,
      steps_per_stream: int = 10,
      warmup_steps: int = 1,
      track: str = TrackType.VLA_TRACK.value,
      chunk_size: int = 8,
      action_dim: int = 7,
      horizon_frames: int = 16,
      target_slo_ms: float = 50.0,
      session_prefix: str = "robot_stream",
      obs_generator: Optional[Callable[[int], Dict[str, Any]]] = None,
  ) -> LoadGenStreamResult:
    """Drive `C` concurrent robot streams with sticky `session_id` affinity (`robot_stream_0` .. `robot_stream_{C-1}`)."""
    c = int(concurrency)
    if c < 1:
      raise ValueError(f"concurrency must be >= 1, got {concurrency}.")
    if int(steps_per_stream) < 1:
      raise ValueError(f"steps_per_stream must be >= 1, got {steps_per_stream}.")

    wm_mode = is_wm_track(track)
    session_ids = [f"{session_prefix}_{i}" for i in range(c)]
    for sid in session_ids:
      self.reset_session(sid)

    start_barrier = threading.Barrier(c)
    obs_lock = threading.Lock()

    def _get_obs(step_idx: int) -> Dict[str, Any]:
      if obs_generator is not None:
        with obs_lock:
          return obs_generator(step_idx)
      return self._default_observation_for_step(step_idx, wm_mode=wm_mode)

    def _worker(stream_idx: int) -> Tuple[List[float], int]:
      sid = session_ids[stream_idx]
      for w in range(max(0, int(warmup_steps))):
        obs = _get_obs(w)
        if wm_mode:
          self.send_wm_step(obs, session_id=sid, horizon_frames=horizon_frames)
        else:
          self.send_vla_act(
              obs, session_id=sid, chunk_size=chunk_size, action_dim=action_dim
          )

      start_barrier.wait(timeout=self.timeout_s)
      stream_lats: List[float] = []
      stream_hits = 0
      for s in range(int(steps_per_stream)):
        obs = _get_obs(s + stream_idx)
        if wm_mode:
          _, lat_ms, resp = self.send_wm_step(
              obs, session_id=sid, horizon_frames=horizon_frames
          )
        else:
          _, lat_ms, resp = self.send_vla_act(
              obs, session_id=sid, chunk_size=chunk_size, action_dim=action_dim
          )
        stream_lats.append(lat_ms)
        if resp.get("cache_hit"):
          stream_hits += 1
      return stream_lats, stream_hits

    all_latencies: List[float] = []
    total_hits = 0
    wall_t0_ns = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=c) as pool:
      futures = [pool.submit(_worker, idx) for idx in range(c)]
      for fut in as_completed(futures):
        lats, hits = fut.result()
        all_latencies.extend(lats)
        total_hits += hits
    wall_elapsed_s = (time.perf_counter_ns() - wall_t0_ns) / 1e9

    return self._summarize_latencies(
        track=track,
        concurrency=c,
        steps_per_stream=steps_per_stream,
        warmup_steps=warmup_steps,
        wall_elapsed_s=wall_elapsed_s,
        latencies_ms=all_latencies,
        cache_hits=total_hits,
        session_ids=session_ids,
        chunk_size=chunk_size,
        horizon_frames=horizon_frames,
        target_slo_ms=target_slo_ms,
    )

  def run_multi_stream_sweep(
      self,
      concurrency_levels: Sequence[int] = DEFAULT_LOADGEN_CONCURRENCY_GRID,
      steps_per_stream: int = 8,
      warmup_steps: int = 1,
      track: str = TrackType.VLA_TRACK.value,
      chunk_size: int = 8,
      action_dim: int = 7,
      horizon_frames: int = 16,
      target_slo_ms: float = 50.0,
      registry_path: Optional[str | Path] = None,
      run_id_prefix: str = "loadgen",
      obs_generator: Optional[Callable[[int], Dict[str, Any]]] = None,
  ) -> Dict[int, LoadGenStreamResult]:
    """Execute a multi-stream concurrency sweep across `concurrency_levels` (e.g. `[1, 2, 4, 8, 16, 32, 64]`)."""
    results: Dict[int, LoadGenStreamResult] = {}
    for c in concurrency_levels:
      c_int = int(c)
      run_id = f"{run_id_prefix}_c{c_int}"
      if registry_path is not None:
        record_loadgen_session_state(
            registry_path=registry_path,
            session_run_id=run_id,
            status=RunStatus.PENDING,
            concurrency=c_int,
            metrics={},
        )
        record_loadgen_session_state(
            registry_path=registry_path,
            session_run_id=run_id,
            status=RunStatus.RUNNING,
            concurrency=c_int,
            metrics={},
        )
      res = self.run_concurrency_level(
          concurrency=c_int,
          steps_per_stream=steps_per_stream,
          warmup_steps=warmup_steps,
          track=track,
          chunk_size=chunk_size,
          action_dim=action_dim,
          horizon_frames=horizon_frames,
          target_slo_ms=target_slo_ms,
          obs_generator=obs_generator,
      )
      results[c_int] = res
      if registry_path is not None:
        record_loadgen_session_state(
            registry_path=registry_path,
            session_run_id=run_id,
            status=RunStatus.CERTIFIED,
            concurrency=c_int,
            metrics=res.to_dict(),
        )
    return results
