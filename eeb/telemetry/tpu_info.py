"""Google Cloud TPU telemetry via in-process `tpu_info` libtpu gRPC metrics or `tpu-info` CLI (ADR-4).

Exposes per-chip HBM usage and TensorCore utilization / duty cycle; per-chip power is UNAVAILABLE
unless a hardware power sensor field is present (never uses TDP defaults unless `--assume-power-watts`
is explicitly passed at report time).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from eeb.telemetry.source import TelemetrySource, make_sample, run_cli, unavailable_sample, which

_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*GiB\s*/\s*(\d+(?:\.\d+)?)\s*GiB", re.IGNORECASE)
_ROW_MEM_DUTY_RE = re.compile(
    r"\|\s*\d+\s*\|\s*(\d+(?:\.\d+)?)\s*GiB\s*/\s*(\d+(?:\.\d+)?)\s*GiB\s*\|\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_TC_UTIL_RE = re.compile(
    r"TensorCore\s+Utilization.*?\|\s*\d+\s*\|\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE | re.DOTALL,
)
_PWR_RE = re.compile(r"Power[^0-9]*(\d+(?:\.\d+)?)\s*W\b", re.IGNORECASE)
_DUTY_RE = re.compile(r"Duty\s*Cycle[^0-9\n]*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_NO_TPU_RE = re.compile(r"no tpu chips found|libtpu not found", re.IGNORECASE)


def parse_tpu_info_text(text: str) -> Dict[str, Any]:
  """Parse `tpu-info` stdout for chip 0. Raises ValueError when no active chip/HBM table is present."""
  if _NO_TPU_RE.search(text) and not _MEM_RE.search(text):
    raise ValueError("tpu-info reports no TPU chips on this host")
  row_m = _ROW_MEM_DUTY_RE.search(text)
  mem_m = _MEM_RE.search(text)
  if not row_m and not mem_m:
    raise ValueError("tpu-info output has no '<used> GiB / <total> GiB' HBM field")
  if row_m:
    used_mb = float(row_m.group(1)) * 1024.0
    total_mb = float(row_m.group(2)) * 1024.0
    duty_val: Optional[float] = float(row_m.group(3))
  else:
    assert mem_m is not None
    used_mb = float(mem_m.group(1)) * 1024.0
    total_mb = float(mem_m.group(2)) * 1024.0
    duty_m = _DUTY_RE.search(text)
    duty_val = float(duty_m.group(1)) if duty_m else None

  tc_m = _TC_UTIL_RE.search(text)
  if tc_m:
    tc_val = float(tc_m.group(1))
    if duty_val is None or (duty_val == 0.0 and tc_val > 0.0):
      duty_val = tc_val

  pwr_m = _PWR_RE.search(text)
  return make_sample(
      "tpu-info",
      power_w=float(pwr_m.group(1)) if pwr_m else None,
      mem_used_mb=used_mb,
      util_pct=duty_val,
      mem_total_mb=total_mb,
      power_note=None if pwr_m else "tpu-info exposes no per-chip power on this generation (ADR-4: TPU power ESTIMATED/UNAVAILABLE)",
  )


class TpuInfoSource(TelemetrySource):
  name = "tpu_info"

  def __init__(self, runtime_addr: str = "localhost:8431") -> None:
    self._path = which("tpu-info")
    self._runtime_addr = runtime_addr
    self._reason = "tpu-info not installed and not on PATH"
    self._ok = False
    self._use_lib = False
    self._chip_type: Any = None
    self._channel: Any = None
    self._stub: Any = None
    self._prev_idle_sample: Optional[Tuple[float, float]] = None

    # Prefer fast in-process `tpu_info` Python library (<2 ms/sample over localhost:8431 gRPC,
    # allowing accurate 10 Hz sampling without ~900 ms subprocess spawn overhead).
    try:
      from tpu_info import device as _tpu_device  # noqa: WPS433

      chip_type, count = _tpu_device.get_local_chips()
      if chip_type is not None and count > 0:
        self._chip_type = chip_type
        self._use_lib = True
        self._ok = True
        self._reason = ""
        return
    except Exception as exc:
      self._reason = f"tpu_info.device probe failed: {exc}"

    if self._path:
      out = run_cli([self._path], timeout_s=10.0)
      if out is None:
        self._reason = "tpu-info failed to run"
      else:
        try:
          parse_tpu_info_text(out)
          self._ok = True
          self._reason = ""
        except ValueError as exc:
          self._reason = str(exc)

  def available(self) -> bool:
    return self._ok

  def unavailable_reason(self) -> str:
    return self._reason

  def _ensure_stub(self) -> Tuple[Any, Any]:
    import grpc  # noqa: WPS433
    from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics_pb2  # noqa: WPS433
    from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc  # noqa: WPS433

    if self._stub is None or self._channel is None:
      self._channel = grpc.secure_channel(self._runtime_addr, grpc.local_channel_credentials())
      self._stub = tpu_metrics_grpc.RuntimeMetricServiceStub(self._channel)
    return self._stub, tpu_metrics_pb2

  def _reset_stub(self) -> None:
    if self._channel is not None:
      try:
        self._channel.close()
      except Exception:
        pass
    self._channel = None
    self._stub = None

  def _sample_via_grpc(self) -> Dict[str, Any]:
    from tpu_info import metrics as _tpu_metrics  # noqa: WPS433

    stub, pb2 = self._ensure_stub()

    def _query(metric_enum: Any) -> List[Any]:
      resp = stub.GetRuntimeMetric(pb2.MetricRequest(metric_name=metric_enum.value), timeout=1.0)
      return sorted(resp.metric.metrics, key=lambda m: m.attribute.value.int_attr)

    totals = _query(_tpu_metrics.MetricName.TOTAL_MEMORY)
    usages = _query(_tpu_metrics.MetricName.MEMORY_USAGE)
    duties = _query(_tpu_metrics.MetricName.DUTY_CYCLE_PCT)
    if not totals or not usages:
      raise ValueError("libtpu runtime reported empty HBM metrics for chip 0")

    used_mb = float(usages[0].gauge.as_int) / (1024.0 * 1024.0)
    total_mb = float(totals[0].gauge.as_int) / (1024.0 * 1024.0)
    duty_pct: Optional[float] = float(duties[0].gauge.as_double) if duties else None

    # Compute exact interval TensorCore utilization from cumulative TENSORCORE_IDLE_DURATION
    # when available, since libtpu DUTY_CYCLE_PCT updates on a coarse multi-second window.
    util_pct = duty_pct
    try:
      idles = _query(_tpu_metrics.MetricName.TENSORCORE_IDLE_DURATION)
      now = time.monotonic()
      if idles:
        idle_s = float(idles[0].gauge.as_double)
        if self._prev_idle_sample is not None:
          prev_t, prev_idle = self._prev_idle_sample
          dt = now - prev_t
          d_idle = idle_s - prev_idle
          if dt >= 0.02 and d_idle >= -1e-3:
            inst_util = max(0.0, min(100.0, (1.0 - max(0.0, d_idle) / dt) * 100.0))
            util_pct = round(inst_util, 3)
        self._prev_idle_sample = (now, idle_s)
    except Exception:
      pass

    return make_sample(
        "tpu-info",
        power_w=None,
        mem_used_mb=round(used_mb, 3),
        util_pct=util_pct,
        mem_total_mb=round(total_mb, 3),
        power_note="tpu-info exposes no per-chip power on this generation (ADR-4: TPU power ESTIMATED/UNAVAILABLE)",
    )

  def sample(self) -> Dict[str, Any]:
    if not self._ok:
      return unavailable_sample(self.name, self._reason)
    if self._use_lib:
      try:
        return self._sample_via_grpc()
      except Exception as exc:
        self._reset_stub()
        return unavailable_sample(self.name, f"libtpu runtime metrics unavailable: {exc}")
    out = run_cli([self._path or "tpu-info"], timeout_s=10.0)
    if not out:
      return unavailable_sample(self.name, "tpu-info returned no output")
    try:
      return parse_tpu_info_text(out)
    except ValueError as exc:
      return unavailable_sample(self.name, str(exc))


__all__ = ["TpuInfoSource", "parse_tpu_info_text"]
