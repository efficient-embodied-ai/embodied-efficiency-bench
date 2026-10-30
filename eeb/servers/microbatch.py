"""Dynamic micro-batching helper for Cloud TPU static power-of-2 shapes.

Coalesces concurrent `predict()` / `rollout()` calls across active client sessions
into a single padded TPU batch `B_pad in buckets` (typically `(1, 2, 4, 8, 16)`)
so XLA / TorchTPU compiles a fixed finite set of shapes once at warm-start and executes
each concurrency wave in a single Vector/MXU pass instead of serializing `B=1` calls.
"""

from __future__ import annotations

from concurrent.futures import Future
import threading
import time
from typing import Callable, Dict, Generic, List, Sequence, Tuple, TypeVar

TReq = TypeVar("TReq")
TRes = TypeVar("TRes")


def select_power_of_two_bucket(batch_size: int, buckets: Sequence[int]) -> int:
  for b in buckets:
    if b >= batch_size:
      return int(b)
  return int(buckets[-1])


class DynamicMicroBatcher(Generic[TReq, TRes]):
  """Thread-safe dynamic micro-batching scheduler targeting static XLA bucket shapes."""

  def __init__(
      self,
      execute_batch_fn: Callable[[List[TReq], int], List[TRes]],
      *,
      buckets: Sequence[int] = (1, 2, 4, 8, 16),
      max_batch_delay_ms: float = 15.0,
      session_ttl_s: float = 0.6,
      reset_cohort_gap_s: float = 0.10,
      name: str = "eeb-microbatcher",
  ) -> None:
    self._execute_batch_fn = execute_batch_fn
    self.buckets: Tuple[int, ...] = tuple(sorted({int(b) for b in buckets if int(b) > 0}))
    if not self.buckets:
      raise ValueError("DynamicMicroBatcher requires at least one positive bucket size.")
    self.max_batch_size = int(self.buckets[-1])
    self.max_batch_delay_s = max(0.0, float(max_batch_delay_ms) / 1000.0)
    self.session_ttl_s = max(0.1, float(session_ttl_s))
    self.reset_cohort_gap_s = max(0.02, float(reset_cohort_gap_s))

    self._cv = threading.Condition()
    self._queue: List[Tuple[str, TReq, Future[TRes]]] = []
    self._active_sessions: Dict[str, float] = {}
    self._last_exec_s: float = 0.05
    self._last_reset_ts: float = 0.0
    self._stopped = False
    self.batches_executed = 0
    self.items_executed = 0
    self.bucket_histogram: Dict[int, int] = {b: 0 for b in self.buckets}

    self._worker = threading.Thread(target=self._loop, name=name, daemon=True)
    self._worker.start()

  def _prune_locked(self, now: float) -> int:
    ttl = max(self.session_ttl_s, self._last_exec_s * 2.5)
    cutoff = now - ttl
    stale = [sid for sid, ts in self._active_sessions.items() if ts < cutoff]
    for sid in stale:
      del self._active_sessions[sid]
    return max(1, min(len(self._active_sessions), self.max_batch_size))

  def register_session(self, session_id: str) -> None:
    now = time.monotonic()
    sid = session_id or "default"
    with self._cv:
      # When a new sweep trial or gate probe starts after a completed wave cluster,
      # clear retired session IDs from prior trials while keeping concurrent threads
      # of the current startup burst together.
      if (now - self._last_reset_ts) > self.reset_cohort_gap_s and not self._queue:
        self._active_sessions.clear()
      self._last_reset_ts = now
      self._active_sessions[sid] = now
      self._prune_locked(now)
      self._cv.notify_all()

  def submit(self, session_id: str, item: TReq) -> TRes:
    fut: Future[TRes] = Future()
    sid = session_id or "default"
    now = time.monotonic()
    with self._cv:
      if self._stopped:
        raise RuntimeError("DynamicMicroBatcher is stopped.")
      self._active_sessions[sid] = now
      self._queue.append((sid, item, fut))
      self._cv.notify_all()
    return fut.result()

  def _loop(self) -> None:
    while True:
      with self._cv:
        while not self._queue and not self._stopped:
          self._cv.wait()
        if self._stopped and not self._queue:
          return

        t_first = time.monotonic()
        while True:
          now = time.monotonic()
          target = self._prune_locked(now)
          qlen = len(self._queue)
          if qlen >= self.max_batch_size or qlen >= target or self._stopped:
            break
          remaining = (t_first + self.max_batch_delay_s) - now
          if remaining <= 0.0:
            break
          self._cv.wait(timeout=remaining)

        n_take = min(len(self._queue), self.max_batch_size)
        batch = self._queue[:n_take]
        del self._queue[:n_take]

      sids = [b[0] for b in batch]
      items = [b[1] for b in batch]
      futs = [b[2] for b in batch]
      padded_size = select_power_of_two_bucket(len(items), self.buckets)
      t0 = time.monotonic()
      try:
        outputs = self._execute_batch_fn(items, padded_size)
        if len(outputs) != len(items):
          raise RuntimeError(
              f"Batch function returned {len(outputs)} outputs for {len(items)} inputs (bucket={padded_size})."
          )
      except BaseException as exc:  # noqa: BLE001
        t_done = time.monotonic()
        with self._cv:
          self._last_exec_s = max(0.005, t_done - t0)
          for sid in sids:
            if sid in self._active_sessions:
              self._active_sessions[sid] = t_done
        for fut in futs:
          if not fut.done():
            fut.set_exception(exc)
        continue

      t_done = time.monotonic()
      with self._cv:
        self._last_exec_s = max(0.005, t_done - t0)
        self.batches_executed += 1
        self.items_executed += len(items)
        self.bucket_histogram[padded_size] = self.bucket_histogram.get(padded_size, 0) + 1
        for sid in sids:
          if sid in self._active_sessions:
            self._active_sessions[sid] = t_done
      for fut, out in zip(futs, outputs):
        if not fut.done():
          fut.set_result(out)

  def close(self, timeout_s: float = 5.0) -> None:
    with self._cv:
      self._stopped = True
      self._cv.notify_all()
    self._worker.join(timeout=timeout_s)


__all__ = ["DynamicMicroBatcher", "select_power_of_two_bucket"]
