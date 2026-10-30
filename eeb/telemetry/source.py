"""Telemetry source interface shared by nvml / tegrastats / amd_smi / tpu_info.

Every `sample()` returns a dict with exactly these keys::

    {"power_w": float|None, "mem_used_mb": float|None, "util_pct": float|None,
     "provenance": "MEASURED" | "ESTIMATED" | "UNAVAILABLE", "source": str, "ts": float}

Rule: a source never invents a number. If the vendor tool does not expose a value, the
field is `None` and `provenance` is `UNAVAILABLE`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence

from eeb import PROVENANCE_ESTIMATED, PROVENANCE_MEASURED, PROVENANCE_UNAVAILABLE

SAMPLE_KEYS = ("power_w", "mem_used_mb", "util_pct", "provenance", "source", "ts")


def make_sample(
    source: str,
    power_w: Optional[float] = None,
    mem_used_mb: Optional[float] = None,
    util_pct: Optional[float] = None,
    provenance: Optional[str] = None,
    ts: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
  """Build a normalized sample; provenance defaults to MEASURED iff power is a number."""
  if provenance is None:
    provenance = PROVENANCE_MEASURED if power_w is not None else PROVENANCE_UNAVAILABLE
  if provenance not in (PROVENANCE_MEASURED, PROVENANCE_ESTIMATED, PROVENANCE_UNAVAILABLE):
    raise ValueError(f"Unknown provenance {provenance!r}.")
  if provenance == PROVENANCE_UNAVAILABLE and power_w is not None:
    raise ValueError("A sample cannot carry power_w with provenance UNAVAILABLE.")
  sample: Dict[str, Any] = {
      "power_w": None if power_w is None else float(power_w),
      "mem_used_mb": None if mem_used_mb is None else float(mem_used_mb),
      "util_pct": None if util_pct is None else float(util_pct),
      "provenance": provenance,
      "source": source,
      "ts": float(ts if ts is not None else time.time()),
  }
  sample.update(extra)
  return sample


def unavailable_sample(source: str, reason: str) -> Dict[str, Any]:
  return make_sample(source=source, provenance=PROVENANCE_UNAVAILABLE, reason=reason)


class TelemetrySource:
  """Base class. Subclasses override `available()` and `sample()`; `start()/stop()` are optional."""

  name = "abstract"

  def available(self) -> bool:
    return False

  def unavailable_reason(self) -> str:
    return "not available"

  def start(self) -> None:
    return None

  def stop(self) -> None:
    return None

  def sample(self) -> Dict[str, Any]:
    return unavailable_sample(self.name, self.unavailable_reason())

  def describe(self) -> Dict[str, Any]:
    return {"name": self.name, "available": self.available(), "reason": None if self.available() else self.unavailable_reason()}


class NullSource(TelemetrySource):
  """Used when no vendor tool is present: every sample is UNAVAILABLE (never fabricated)."""

  name = "none"

  def __init__(self, reason: str = "no supported telemetry tool found (pynvml/nvidia-smi, tegrastats, amd-smi/rocm-smi, tpu-info)") -> None:
    self._reason = reason

  def unavailable_reason(self) -> str:
    return self._reason


def which(binary: str) -> Optional[str]:
  return shutil.which(binary)


def run_cli(args: Sequence[str], timeout_s: float = 2.0) -> Optional[str]:
  """Run a vendor CLI and return stdout, or None if it is missing / fails / times out."""
  try:
    proc = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout_s, check=False)
  except (OSError, subprocess.TimeoutExpired):
    return None
  if proc.returncode != 0:
    return None
  return proc.stdout


__all__ = [
    "NullSource",
    "SAMPLE_KEYS",
    "TelemetrySource",
    "make_sample",
    "run_cli",
    "unavailable_sample",
    "which",
]
