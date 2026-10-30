"""NVIDIA discrete-GPU telemetry: `pynvml` if importable, else `nvidia-smi --query-gpu` subprocess."""

from __future__ import annotations

from typing import Any, Dict, Optional

from eeb.telemetry.source import TelemetrySource, make_sample, run_cli, unavailable_sample, which

NVIDIA_SMI_QUERY = "power.draw,memory.used,memory.total,utilization.gpu,name"


def parse_nvidia_smi_csv_line(line: str) -> Dict[str, Any]:
  """Parse one `nvidia-smi --query-gpu=power.draw,memory.used,memory.total,utilization.gpu,name --format=csv,noheader,nounits` row.

  Example: ``285.43, 20480, 32607, 97, NVIDIA GeForce RTX 5090``. A field of ``[N/A]`` (e.g. power
  on some laptops / MIG slices) yields None for that field rather than a fabricated number.
  """
  parts = [p.strip() for p in line.strip().split(",")]
  if len(parts) < 4:
    raise ValueError(f"Unexpected nvidia-smi row (need >= 4 comma-separated fields): {line!r}")

  def _num(s: str) -> Optional[float]:
    if not s or s.upper().startswith("[N/A]") or s.upper() == "N/A":
      return None
    try:
      return float(s)
    except ValueError:
      return None

  return {
      "power_w": _num(parts[0]),
      "mem_used_mb": _num(parts[1]),
      "mem_total_mb": _num(parts[2]),
      "util_pct": _num(parts[3]),
      "name": parts[4] if len(parts) > 4 else "",
  }


class NvmlSource(TelemetrySource):
  """Power / memory / utilization of one NVIDIA GPU (index `device_index`)."""

  name = "nvml"

  def __init__(self, device_index: int = 0) -> None:
    self.device_index = int(device_index)
    self._nvml: Any = None
    self._handle: Any = None
    self._mode = "unavailable"
    self._reason = "pynvml not importable and nvidia-smi not on PATH"
    self._init()

  def _init(self) -> None:
    try:
      import pynvml  # type: ignore[import-not-found]  # noqa: WPS433

      pynvml.nvmlInit()
      self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
      self._nvml = pynvml
      self._mode = "pynvml"
      return
    except Exception as exc:  # ImportError or NVMLError
      self._reason = f"pynvml unavailable ({type(exc).__name__}: {exc})"
    if which("nvidia-smi"):
      probe = run_cli(["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits", "-i", str(self.device_index)])
      if probe and probe.strip():
        self._mode = "nvidia-smi"
        return
      self._reason = "nvidia-smi present but returned no GPU row (driver not loaded?)"
    else:
      self._reason = self._reason + "; nvidia-smi not on PATH"

  def available(self) -> bool:
    return self._mode in ("pynvml", "nvidia-smi")

  def unavailable_reason(self) -> str:
    return self._reason

  def stop(self) -> None:
    if self._nvml is not None:
      try:
        self._nvml.nvmlShutdown()
      except Exception:  # pragma: no cover
        pass
      self._nvml = None

  def sample(self) -> Dict[str, Any]:
    if self._mode == "pynvml":
      nv = self._nvml
      try:
        power_w: Optional[float] = nv.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
      except Exception:
        power_w = None
      try:
        mem = nv.nvmlDeviceGetMemoryInfo(self._handle)
        mem_used_mb: Optional[float] = float(mem.used) / (1024.0 * 1024.0)
      except Exception:
        mem_used_mb = None
      try:
        util_pct: Optional[float] = float(nv.nvmlDeviceGetUtilizationRates(self._handle).gpu)
      except Exception:
        util_pct = None
      extra: Dict[str, Any] = {}
      energy_fn = getattr(nv, "nvmlDeviceGetTotalEnergyConsumption", None)
      if callable(energy_fn):
        try:
          energy_mj = energy_fn(self._handle)
          if energy_mj is not None:
            extra["energy_mj"] = float(energy_mj)
        except Exception:
          pass
      return make_sample(f"{self.name}:pynvml", power_w=power_w, mem_used_mb=mem_used_mb, util_pct=util_pct, extra=extra or None)
    if self._mode == "nvidia-smi":
      out = run_cli(["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits", "-i", str(self.device_index)])
      if not out or not out.strip():
        return unavailable_sample(f"{self.name}:nvidia-smi", "nvidia-smi returned no data")
      try:
        row = parse_nvidia_smi_csv_line(out.strip().splitlines()[0])
      except ValueError as exc:
        return unavailable_sample(f"{self.name}:nvidia-smi", str(exc))
      return make_sample(f"{self.name}:nvidia-smi", power_w=row["power_w"], mem_used_mb=row["mem_used_mb"], util_pct=row["util_pct"], device_name=row["name"])
    return unavailable_sample(self.name, self._reason)


__all__ = ["NVIDIA_SMI_QUERY", "NvmlSource", "parse_nvidia_smi_csv_line"]
