"""Pacing (think-time) and session-sticky routing for the EEB LoadGen.

Pacing model (plan §4 M1: "think-time = chunk execution at 20 Hz"):

  A robot that receives an action chunk at wall time `t_recv` executes `execute_horizon`
  actions, one per control period `dt_ms`, before it has a new observation worth sending.
  The next request for that session is therefore released at::

      t_next_send = t_recv + execute_horizon * dt_ms / 1000

  This is the SYNCHRONOUS_HOLD execution strategy of `embodied_bench.schema.ExecutionStrategy`:
  the robot holds while inference runs. The per-call diagnostic `deadline_missed` flags calls
  whose obs->action latency exceeded the chunk execution time (a policy slower than the chunk
  it produces would stall even an ASYNC_CHUNK controller).

Session-sticky routing:

  Step t+1 of session k must always be issued by the same LoadGen slot (thread + connection)
  as step t, so that any server-side per-session state (KV cache, chunk buffer) is exercised
  the way a real robot would. `SessionRouter` hands out sticky slot assignments and records a
  `(session_id, step, slot)` trace; `verify_sticky_trace` is the pure check used by the
  report and by the tests' negative control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PacingConfig:
  """Think-time configuration derived from the server's `Caps` (or overridden by the recipe)."""

  dt_ms: float
  horizon: int
  execute_horizon: Optional[int] = None
  enabled: bool = True
  mode: str = "SYNCHRONOUS_HOLD"

  def __post_init__(self) -> None:
    if self.dt_ms <= 0.0:
      raise ValueError(f"PacingConfig.dt_ms must be > 0 ms, got {self.dt_ms}.")
    if self.horizon <= 0:
      raise ValueError(f"PacingConfig.horizon must be > 0, got {self.horizon}.")
    k = self.effective_execute_horizon
    if k <= 0 or k > self.horizon:
      raise ValueError(
          f"PacingConfig.execute_horizon={self.execute_horizon} must be in [1, horizon={self.horizon}]."
      )
    if self.mode not in ("SYNCHRONOUS_HOLD", "NONE"):
      raise ValueError(f"PacingConfig.mode must be SYNCHRONOUS_HOLD or NONE, got {self.mode!r}.")

  @property
  def effective_execute_horizon(self) -> int:
    return int(self.execute_horizon) if self.execute_horizon is not None else int(self.horizon)

  @property
  def think_time_s(self) -> float:
    """Chunk execution time: executed actions x control period (0 when pacing is disabled)."""
    if not self.enabled or self.mode == "NONE":
      return 0.0
    return self.effective_execute_horizon * float(self.dt_ms) / 1000.0

  @property
  def think_time_ms(self) -> float:
    return self.think_time_s * 1000.0

  def next_send_time(self, t_recv: float) -> float:
    """Monotonic timestamp at which the next observation of this session should be sent."""
    return t_recv + self.think_time_s

  def deadline_missed(self, latency_ms: float) -> bool:
    """True when inference exceeded the chunk execution replan deadline (`execute_horizon * dt_ms`)."""
    if not self.enabled or self.mode == "NONE":
      return False
    return float(latency_ms) > self.think_time_ms

  def single_tick_cutoff_exceeded(self, latency_ms: float) -> bool:
    """True when single-call inference exceeded one real-time control tick (`dt_ms`, e.g., 50 ms at 20 Hz)."""
    return float(latency_ms) > float(self.dt_ms)

  def to_dict(self) -> Dict[str, object]:
    return {
        "mode": self.mode if self.enabled else "NONE",
        "enabled": bool(self.enabled),
        "dt_ms": float(self.dt_ms),
        "single_tick_cutoff_ms": float(self.dt_ms),
        "control_freq_hz": round(1000.0 / float(self.dt_ms), 4),
        "horizon": int(self.horizon),
        "execute_horizon": self.effective_execute_horizon,
        "think_time_ms": round(self.think_time_ms, 4),
    }


def sleep_until(t_monotonic: float) -> float:
  """Sleep until a `time.perf_counter()` timestamp; returns the actual oversleep in ms (>= 0)."""
  while True:
    remaining = t_monotonic - time.perf_counter()
    if remaining <= 0.0:
      return -remaining * 1000.0
    time.sleep(min(remaining, 0.05))


@dataclass(frozen=True)
class RouteRecord:
  session_id: str
  step: int
  slot: int


class StickinessViolation(ValueError):
  """Raised when a session's step was issued by a different slot than its previous steps."""


class SessionRouter:
  """Sticky session -> slot assignment with a verifiable routing trace.

  New sessions are assigned to the least-loaded slot (ties -> lowest index); once assigned a
  session never moves. `record()` appends to the trace and fails fast on a violation.
  """

  def __init__(self, num_slots: int) -> None:
    if int(num_slots) <= 0:
      raise ValueError(f"SessionRouter needs num_slots >= 1, got {num_slots}.")
    self.num_slots = int(num_slots)
    self._assign: Dict[str, int] = {}
    self._load: List[int] = [0] * self.num_slots
    self._trace: List[RouteRecord] = []
    self._lock = threading.Lock()

  def route(self, session_id: str, preferred_slot: Optional[int] = None) -> int:
    with self._lock:
      slot = self._assign.get(session_id)
      if slot is None:
        if preferred_slot is not None and 0 <= int(preferred_slot) < self.num_slots:
          slot = int(preferred_slot)
        else:
          slot = min(range(self.num_slots), key=lambda i: (self._load[i], i))
        self._assign[session_id] = slot
        self._load[slot] += 1
      return slot

  def record(self, session_id: str, step: int, slot: int) -> None:
    with self._lock:
      expected = self._assign.get(session_id)
      if expected is None:
        self._assign[session_id] = int(slot)
        self._load[int(slot)] += 1
        expected = int(slot)
      self._trace.append(RouteRecord(session_id=session_id, step=int(step), slot=int(slot)))
      if int(slot) != expected:
        raise StickinessViolation(
            f"Session {session_id!r} step {step} was issued by slot {slot} but the session is pinned to slot {expected}. "
            "Each LoadGen worker must own its sessions for the whole run."
        )

  @property
  def trace(self) -> List[RouteRecord]:
    with self._lock:
      return list(self._trace)

  @property
  def assignments(self) -> Dict[str, int]:
    with self._lock:
      return dict(self._assign)

  def verify(self) -> Tuple[bool, List[str]]:
    return verify_sticky_trace(self.trace)


def verify_sticky_trace(trace: Sequence[RouteRecord | Tuple[str, int, int]]) -> Tuple[bool, List[str]]:
  """Pure check: every (session_id) maps to exactly one slot across the whole trace.

  Legitimate lookalikes that must NOT fire: one slot serving several sessions interleaved;
  a session re-sending the same step number (retry) on its own slot.
  """
  seen: Dict[str, int] = {}
  violations: List[str] = []
  for rec in trace:
    sid, step, slot = (rec.session_id, rec.step, rec.slot) if isinstance(rec, RouteRecord) else (rec[0], int(rec[1]), int(rec[2]))
    prev = seen.setdefault(sid, slot)
    if prev != slot:
      violations.append(f"session {sid!r} step {step}: slot {slot} != pinned slot {prev}")
  return (len(violations) == 0, violations)


@dataclass
class SessionSchedule:
  """Mutable per-session pacing state owned by exactly one LoadGen slot."""

  session_id: str
  slot: int
  next_send_monotonic: float = 0.0
  step: int = 0
  oversleep_ms: List[float] = field(default_factory=list)

  def arm(self, pacing: PacingConfig, t_recv_monotonic: float) -> None:
    self.next_send_monotonic = pacing.next_send_time(t_recv_monotonic)

  def wait(self) -> float:
    if self.next_send_monotonic <= 0.0:
      return 0.0
    over = sleep_until(self.next_send_monotonic)
    self.oversleep_ms.append(over)
    return over


__all__ = [
    "PacingConfig",
    "RouteRecord",
    "SessionRouter",
    "SessionSchedule",
    "StickinessViolation",
    "sleep_until",
    "verify_sticky_trace",
]
