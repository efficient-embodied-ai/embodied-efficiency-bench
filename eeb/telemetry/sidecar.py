"""Background 10 Hz telemetry sampler with pluggable vendor sources and honest provenance.

Usage::

    sidecar = TelemetrySidecar.auto(hz=10.0)     # picks nvml -> tegrastats -> amd_smi -> tpu_info -> none
    sidecar.start()
    ... run the timed window, remember (t0, t1) = time.time() before/after ...
    sidecar.stop()
    summary = sidecar.summarize(t0, t1)          # mean/peak W, peak MB, energy J, provenance
    jpa = joules_per_action(summary, num_actions)

Energy is integrated with the trapezoid rule over the MEASURED power samples inside the
window. If the source is unavailable or the window has fewer than two power samples, the
summary's `provenance` is `UNAVAILABLE` and every power/energy field is None - the harness
never substitutes TDP tables for a missing reading (that is `ESTIMATED` and must be explicit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from eeb import PROVENANCE_ESTIMATED, PROVENANCE_MEASURED, PROVENANCE_UNAVAILABLE
from eeb.telemetry.source import NullSource, TelemetrySource

MIN_POWER_COVERAGE_PCT = 90.0


def detect_sources(device_index: int = 0) -> List[TelemetrySource]:
  """Instantiate every vendor source (cheap probes only); order = preference."""
  from eeb.telemetry.amd_smi import AmdSmiSource
  from eeb.telemetry.nvml import NvmlSource
  from eeb.telemetry.tegrastats import TegrastatsSource
  from eeb.telemetry.tpu_info import TpuInfoSource

  return [NvmlSource(device_index), TegrastatsSource(), AmdSmiSource(device_index), TpuInfoSource()]


def select_source(device_index: int = 0, prefer: Optional[str] = None) -> TelemetrySource:
  """First available source in preference order (or the one named by `prefer`)."""
  sources = detect_sources(device_index)
  if prefer:
    for src in sources:
      if src.name == prefer:
        return src if src.available() else NullSource(f"requested source {prefer!r} unavailable: {src.unavailable_reason()}")
    raise ValueError(f"Unknown telemetry source {prefer!r}; choose from {[s.name for s in sources]}.")
  for src in sources:
    if src.available():
      return src
  reasons = "; ".join(f"{s.name}: {s.unavailable_reason()}" for s in sources)
  return NullSource(f"no supported telemetry tool found ({reasons})")


@dataclass
class TelemetrySummary:
  source: str
  provenance: str
  num_samples: int
  window_s: float
  mean_power_w: Optional[float]
  peak_power_w: Optional[float]
  energy_j: Optional[float]
  peak_mem_used_mb: Optional[float]
  mean_util_pct: Optional[float]
  sample_hz_actual: float
  power_coverage_pct: float
  reason: Optional[str] = None
  extra: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    d = {
        "source": self.source,
        "provenance": self.provenance,
        "num_samples": self.num_samples,
        "window_s": round(self.window_s, 4),
        "mean_power_w": None if self.mean_power_w is None else round(self.mean_power_w, 3),
        "peak_power_w": None if self.peak_power_w is None else round(self.peak_power_w, 3),
        "energy_j": None if self.energy_j is None else round(self.energy_j, 4),
        "peak_mem_used_mb": None if self.peak_mem_used_mb is None else round(self.peak_mem_used_mb, 2),
        "mean_util_pct": None if self.mean_util_pct is None else round(self.mean_util_pct, 2),
        "sample_hz_actual": round(self.sample_hz_actual, 3),
        "power_coverage_pct": round(self.power_coverage_pct, 2),
        "reason": self.reason,
    }
    d.update(self.extra)
    return d


def summarize_samples(samples: Sequence[Dict[str, Any]], t0: float, t1: float, source_name: str = "") -> TelemetrySummary:
  """Pure aggregation over samples with `ts` in [t0, t1] (testable without hardware)."""
  window = [s for s in samples if t0 <= float(s.get("ts", -1)) <= t1]
  n = len(window)
  window_s = max(t1 - t0, 1e-9)
  hz = n / window_s if n else 0.0
  src = source_name or (window[0].get("source", "") if window else "")
  mems = [float(s["mem_used_mb"]) for s in window if s.get("mem_used_mb") is not None]
  utils = [float(s["util_pct"]) for s in window if s.get("util_pct") is not None]
  powered = [s for s in window if s.get("power_w") is not None and s.get("provenance") in (PROVENANCE_MEASURED, PROVENANCE_ESTIMATED)]
  coverage = (len(powered) / n * 100.0) if n else 0.0
  base = dict(
      source=src,
      num_samples=n,
      window_s=window_s,
      peak_mem_used_mb=max(mems) if mems else None,
      mean_util_pct=float(np.mean(utils)) if utils else None,
      sample_hz_actual=hz,
      power_coverage_pct=coverage,
  )
  if n == 0:
    return TelemetrySummary(provenance=PROVENANCE_UNAVAILABLE, mean_power_w=None, peak_power_w=None, energy_j=None, reason="no telemetry samples in window", **base)
  if len(powered) < 2 or coverage < MIN_POWER_COVERAGE_PCT:
    reasons = {str(s.get("reason")) for s in window if s.get("reason")}
    why = "; ".join(sorted(reasons)) if reasons else f"only {len(powered)} power samples ({coverage:.0f}% coverage) in {window_s:.2f}s window"
    return TelemetrySummary(provenance=PROVENANCE_UNAVAILABLE, mean_power_w=None, peak_power_w=None, energy_j=None, reason=why, **base)
  ts = np.asarray([float(s["ts"]) for s in powered], dtype=np.float64)
  pw = np.asarray([float(s["power_w"]) for s in powered], dtype=np.float64)
  order = np.argsort(ts)
  ts, pw = ts[order], pw[order]
  energy_counter_pairs = sorted(
      [
          (float(s["ts"]), float(s["energy_mj"]))
          for s in powered
          if isinstance(s.get("energy_mj"), (int, float))
          and not isinstance(s.get("energy_mj"), bool)
          and np.isfinite(float(s["energy_mj"]))
      ],
      key=lambda item: item[0],
  )
  energy_method = "trapezoid_polling"
  counter_used = False
  if len(energy_counter_pairs) >= 2:
    t_first, e_first_mj = energy_counter_pairs[0]
    t_last, e_last_mj = energy_counter_pairs[-1]
    if e_last_mj >= e_first_mj and (t_last - t_first) > 1e-6:
      energy = (e_last_mj - e_first_mj) / 1000.0
      energy += float(pw[0]) * max(0.0, t_first - t0) + float(pw[-1]) * max(0.0, t1 - t_last)
      energy_method = "hardware_energy_counter_mj"
      counter_used = True
  if not counter_used:
    energy = float(np.trapezoid(pw, ts)) if hasattr(np, "trapezoid") else float(np.trapz(pw, ts))  # noqa: NPY201
    # Extend the integral to the full window edges with the nearest sample (<= one period each side).
    energy += float(pw[0]) * max(0.0, float(ts[0]) - t0) + float(pw[-1]) * max(0.0, t1 - float(ts[-1]))
  mean_pw = float(energy / window_s) if counter_used and window_s > 0.0 else float(pw.mean())
  provenance = PROVENANCE_MEASURED if all(s.get("provenance") == PROVENANCE_MEASURED for s in powered) else PROVENANCE_ESTIMATED
  return TelemetrySummary(
      provenance=provenance,
      mean_power_w=mean_pw,
      peak_power_w=float(pw.max()),
      energy_j=energy,
      reason=None,
      extra={"energy_method": energy_method},
      **base,
  )


def joules_per_action(summary: TelemetrySummary, num_actions: int) -> Optional[float]:
  """Energy per produced action over the window, or None when energy is not MEASURED/ESTIMATED."""
  if summary.energy_j is None or num_actions <= 0:
    return None
  return round(summary.energy_j / float(num_actions), 6)


class TelemetrySidecar:
  """Samples one `TelemetrySource` from a daemon thread at `hz` and keeps every sample in memory."""

  def __init__(self, source: Optional[TelemetrySource] = None, hz: float = 10.0) -> None:
    if hz <= 0.0:
      raise ValueError("TelemetrySidecar hz must be > 0.")
    self.source = source or NullSource()
    self.hz = float(hz)
    self._samples: List[Dict[str, Any]] = []
    self._marks: List[Dict[str, Any]] = []
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._thread: Optional[threading.Thread] = None
    self.started_at: Optional[float] = None
    self.stopped_at: Optional[float] = None

  @classmethod
  def auto(cls, hz: float = 10.0, device_index: int = 0, prefer: Optional[str] = None) -> "TelemetrySidecar":
    return cls(select_source(device_index=device_index, prefer=prefer), hz=hz)

  @property
  def available(self) -> bool:
    return self.source.available()

  def start(self) -> "TelemetrySidecar":
    if self._thread is not None:
      return self
    self._stop.clear()
    self.source.start()
    self.started_at = time.time()
    self._thread = threading.Thread(target=self._loop, name="eeb-telemetry-sidecar", daemon=True)
    self._thread.start()
    return self

  def _loop(self) -> None:
    period = 1.0 / self.hz
    next_t = time.perf_counter()
    while not self._stop.is_set():
      sample = self.source.sample()
      with self._lock:
        self._samples.append(sample)
      next_t += period
      delay = next_t - time.perf_counter()
      if delay > 0:
        self._stop.wait(delay)
      else:
        next_t = time.perf_counter()

  def stop(self) -> "TelemetrySidecar":
    if self._thread is None:
      return self
    self._stop.set()
    self._thread.join(timeout=max(2.0, 3.0 / self.hz))
    self._thread = None
    self.source.stop()
    self.stopped_at = time.time()
    return self

  def mark(self, label: str, **info: Any) -> None:
    with self._lock:
      self._marks.append({"label": label, "ts": time.time(), **info})

  @property
  def samples(self) -> List[Dict[str, Any]]:
    with self._lock:
      return list(self._samples)

  @property
  def marks(self) -> List[Dict[str, Any]]:
    with self._lock:
      return list(self._marks)

  def latest(self) -> Optional[Dict[str, Any]]:
    with self._lock:
      return dict(self._samples[-1]) if self._samples else None

  def summarize(self, t0: Optional[float] = None, t1: Optional[float] = None) -> TelemetrySummary:
    t_start = t0 if t0 is not None else (self.started_at or 0.0)
    t_end = t1 if t1 is not None else time.time()
    return summarize_samples(self.samples, t_start, t_end, source_name=self.source.name)

  def describe(self) -> Dict[str, Any]:
    return {"source": self.source.describe(), "hz": self.hz, "num_samples": len(self.samples)}

  def __enter__(self) -> "TelemetrySidecar":
    return self.start()

  def __exit__(self, *exc: Any) -> None:
    self.stop()


__all__ = [
    "MIN_POWER_COVERAGE_PCT",
    "TelemetrySidecar",
    "TelemetrySummary",
    "detect_sources",
    "joules_per_action",
    "select_source",
    "summarize_samples",
]
