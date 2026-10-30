"""NVIDIA Jetson (Orin NX / AGX Orin / Thor) telemetry via a long-running `tegrastats` subprocess.

Reuses `embodied_bench.telemetry.parse_tegrastats_line` for the INA3221 rail / RAM fields and adds
GPU utilization from the `GR3D_FREQ <pct>%` token. Module power = `VDD_IN` (total board input);
`VDD_CPU_GPU_CV` is exported separately as `power_cpu_gpu_cv_w`.
"""

from __future__ import annotations

import re
import subprocess
import threading
from typing import Any, Dict, Optional

from eeb.telemetry.source import TelemetrySource, make_sample, unavailable_sample, which

_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%", re.IGNORECASE)


def parse_tegrastats_sample(line: str) -> Dict[str, Any]:
  """Parse one tegrastats line into the normalized sample dict (raises ValueError if no power rails)."""
  from embodied_bench.telemetry import parse_tegrastats_line

  parsed = parse_tegrastats_line(line)
  util_m = _GR3D_RE.search(line)
  return make_sample(
      "tegrastats",
      power_w=parsed.vdd_in_watts,
      mem_used_mb=parsed.ram_used_mb,
      util_pct=float(util_m.group(1)) if util_m else None,
      power_cpu_gpu_cv_w=parsed.vdd_cpu_gpu_cv_watts,
      power_soc_w=parsed.vdd_soc_watts,
      mem_total_mb=parsed.ram_total_mb,
  )


class TegrastatsSource(TelemetrySource):
  """Streams `tegrastats --interval <ms>` in a reader thread; `sample()` returns the latest line."""

  name = "tegrastats"

  def __init__(self, interval_ms: int = 100, binary: str = "tegrastats") -> None:
    self.interval_ms = int(interval_ms)
    self.binary = binary
    self._path = which(binary)
    self._proc: Optional[subprocess.Popen[str]] = None
    self._thread: Optional[threading.Thread] = None
    self._latest: Optional[Dict[str, Any]] = None
    self._latest_error: Optional[str] = None
    self._lock = threading.Lock()

  def available(self) -> bool:
    return self._path is not None

  def unavailable_reason(self) -> str:
    return f"{self.binary} not on PATH (not a Jetson / L4T host)"

  def start(self) -> None:
    if not self.available() or self._proc is not None:
      return
    self._proc = subprocess.Popen(
        [self._path or self.binary, "--interval", str(self.interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    self._thread = threading.Thread(target=self._reader, name="eeb-tegrastats", daemon=True)
    self._thread.start()

  def _reader(self) -> None:
    assert self._proc is not None and self._proc.stdout is not None
    for line in self._proc.stdout:
      try:
        sample = parse_tegrastats_sample(line)
        with self._lock:
          self._latest = sample
      except ValueError as exc:
        with self._lock:
          self._latest_error = str(exc)

  def stop(self) -> None:
    if self._proc is not None:
      try:
        self._proc.terminate()
        self._proc.wait(timeout=2.0)
      except Exception:  # pragma: no cover
        try:
          self._proc.kill()
        except Exception:
          pass
      self._proc = None

  def sample(self) -> Dict[str, Any]:
    if not self.available():
      return unavailable_sample(self.name, self.unavailable_reason())
    with self._lock:
      if self._latest is not None:
        return dict(self._latest)
      err = self._latest_error
    return unavailable_sample(self.name, err or "tegrastats started but no line parsed yet")


__all__ = ["TegrastatsSource", "parse_tegrastats_sample"]
