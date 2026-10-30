"""Single-chip hardware verifier (`SingleChipHardwareVerifier`), live 100 ms background telemetry sampler, `PerformanceCapEnvelope` eligibility checker (`verify_single_chip_envelope`), and multi-chip bypass command-line auditor (`audit_command_line_for_multi_chip_bypass`) for EmbodiedX (`embodied-efficiency-bench`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import glob
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from embodied_bench.schema import (
    HARDWARE_SKU_SPECS,
    PERFORMANCE_CAP_ENVELOPES,
    PerformanceCapId,
    now_pacific_iso,
    resolve_perf_cap_id,
)
from embodied_bench.telemetry import (
    compute_hardware_fingerprint,
    parse_rocm_smi_csv,
    parse_tegrastats_line,
    parse_tpu_info_output,
)


@dataclass
class SingleChipVerificationResult:
  """Result of verifying a single-chip SKU and active device count against a `PerformanceCapEnvelope`."""

  passed: bool
  hardware_sku: str
  perf_cap: str
  detected_device_count: int
  native_vram_gb: float
  measured_vram_gb: Optional[float]
  max_vram_gb: float
  fp16_tflops: float
  max_fp16_tflops: float
  fp8_tops: float
  max_fp8_tops: float
  mem_bw_gbps: float
  max_mem_bw_gbps: float
  violations: List[str] = field(default_factory=list)
  reason: str = "Single-chip envelope verification passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class MultiChipAuditResult:
  """Result of auditing a command line or environment configuration for multi-chip bypass flags."""

  passed: bool
  inspected_lines: int
  violations: List[str] = field(default_factory=list)
  reason: str = "No multi-chip bypass flags or environment variables detected."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class TelemetrySamplePoint:
  """Single 100 ms background telemetry sample from `SingleChipHardwareVerifier`."""

  timestamp_ns: int
  vram_used_mb: float
  vram_total_mb: float
  power_watts: float
  detected_device_count: int
  source: str


@dataclass
class SingleChipTelemetrySummary:
  """Aggregated 100 ms background telemetry and single-chip verification summary."""

  passed: bool
  hardware_sku: str
  vendor: str
  device_name: str
  perf_cap: str
  detected_device_count: int
  sample_count: int
  poll_interval_ms: float
  peak_vram_mb: float
  total_vram_mb: float
  mean_power_watts: float
  peak_power_watts: float
  telemetry_source: str
  hardware_fingerprint: str
  envelope_verification: SingleChipVerificationResult
  violations: List[str] = field(default_factory=list)
  reason: str = "Single-chip live telemetry and envelope verification passed."

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    d["envelope_verification"] = self.envelope_verification.to_dict()
    return d


def verify_single_chip_envelope(
    hardware_sku: str,
    detected_device_count: int = 1,
    perf_cap: str | PerformanceCapId = PerformanceCapId.CAP_SINGLE_CHIP,
    measured_vram_gb: Optional[float] = None,
) -> SingleChipVerificationResult:
  """Verify `detected_device_count == 1` and native SKU specs against `PERFORMANCE_CAP_ENVELOPES[perf_cap]`.

  Enforces (`CAP_SINGLE_CHIP` defaults):
    - `detected_device_count == 1`
    - Native & measured HBM/VRAM <= `envelope.max_vram_gb` (64.0 GB for `CAP_SINGLE_CHIP`)
    - Native BF16/FP16 compute <= `envelope.max_fp16_tflops` (1,000.0 TFLOPS for `CAP_SINGLE_CHIP`)
    - Native FP8/INT8 compute <= `envelope.max_int8_fp8_tops` (2,000.0 TOPS for `CAP_SINGLE_CHIP`)
    - Native memory bandwidth <= `envelope.max_mem_bw_gbps` (2,000.0 GB/s for `CAP_SINGLE_CHIP`)
  """
  resolved_cap = resolve_perf_cap_id(perf_cap)
  if resolved_cap not in PERFORMANCE_CAP_ENVELOPES:
    valid_caps = ", ".join(sorted(PERFORMANCE_CAP_ENVELOPES.keys()))
    raise ValueError(
        f"Unknown perf_cap={perf_cap!r}. Supported performance cap envelopes: [{valid_caps}]."
    )
  envelope = PERFORMANCE_CAP_ENVELOPES[resolved_cap]

  if hardware_sku not in HARDWARE_SKU_SPECS:
    valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
    raise ValueError(
        f"Unknown hardware_sku={hardware_sku!r}. Supported SKUs in HARDWARE_SKU_SPECS: [{valid_skus}]."
    )
  sku_spec = HARDWARE_SKU_SPECS[hardware_sku]

  # Use physical_vram_mb if present to catch datacenter cards (e.g. 80GB/192GB) when evaluated against native single-chip caps
  native_vram_mb = float(
      sku_spec.get("physical_vram_mb", sku_spec.get("vram_mb", 0.0))
  )
  native_vram_gb = round(native_vram_mb / 1024.0, 3)
  fp16_tflops = float(sku_spec.get("fp16_tflops", 0.0))
  fp8_tops = float(sku_spec.get("fp8_tflops", fp16_tflops * 2.0))
  mem_bw_gbps = float(sku_spec.get("mem_bw_gbps", 0.0))
  sku_num_devices = int(sku_spec.get("num_devices", 1))

  violations: List[str] = []
  dev_count = int(detected_device_count)
  if dev_count != 1 or sku_num_devices != 1:
    violations.append(
        f"Single-chip device count violation: detected_device_count={dev_count} (sku_num_devices={sku_num_devices}), "
        "expected exactly 1 accelerator chip/package. Restrict execution to 1 chip via "
        "`CUDA_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`, or `TPU_VISIBLE_DEVICES=0`."
    )

  if native_vram_gb > envelope.max_vram_gb + 1e-6:
    violations.append(
        f"Native HBM/VRAM {native_vram_gb:.1f} GB on {hardware_sku} exceeds {resolved_cap} ceiling "
        f"({envelope.max_vram_gb:.1f} GB). Select a qualifying single-chip SKU (e.g. TPU_V6E_1_32GB, "
        "RTX_5090_32GB, JETSON_AGX_THOR_64GB, AMD_MI210_64GB) or evaluate under CAP_DATACENTER_1000T / CAP_UNCONSTRAINED."
    )

  if measured_vram_gb is not None and float(measured_vram_gb) > envelope.max_vram_gb + 1e-6:
    violations.append(
        f"Measured active HBM/VRAM {float(measured_vram_gb):.2f} GB exceeds {resolved_cap} ceiling "
        f"({envelope.max_vram_gb:.1f} GB). Reduce batch/concurrency or enable weight quantization."
    )

  if fp16_tflops > envelope.max_fp16_tflops + 1e-6:
    violations.append(
        f"Peak BF16/FP16 compute {fp16_tflops:.1f} TFLOPS on {hardware_sku} exceeds {resolved_cap} ceiling "
        f"({envelope.max_fp16_tflops:.1f} TFLOPS). Use a single-chip SKU within <= {envelope.max_fp16_tflops:.0f} BF16 TFLOPS."
    )

  if fp8_tops > envelope.max_int8_fp8_tops + 1e-6:
    violations.append(
        f"Peak FP8/INT8 compute {fp8_tops:.1f} TOPS on {hardware_sku} exceeds {resolved_cap} ceiling "
        f"({envelope.max_int8_fp8_tops:.1f} TOPS). Use a single-chip SKU within <= {envelope.max_int8_fp8_tops:.0f} TOPS."
    )

  if mem_bw_gbps > envelope.max_mem_bw_gbps + 1e-6:
    violations.append(
        f"Peak memory bandwidth {mem_bw_gbps:.1f} GB/s on {hardware_sku} exceeds {resolved_cap} ceiling "
        f"({envelope.max_mem_bw_gbps:.1f} GB/s). Use a single-chip SKU within <= {envelope.max_mem_bw_gbps:.0f} GB/s."
    )

  passed = len(violations) == 0
  reason = (
      f"{hardware_sku} satisfies {resolved_cap} single-chip envelope "
      f"(devices={dev_count}, VRAM={native_vram_gb:.1f}/{envelope.max_vram_gb:.1f} GB, "
      f"BF16={fp16_tflops:.1f}/{envelope.max_fp16_tflops:.1f} TFLOPS, "
      f"TOPS={fp8_tops:.1f}/{envelope.max_int8_fp8_tops:.1f}, "
      f"BW={mem_bw_gbps:.1f}/{envelope.max_mem_bw_gbps:.1f} GB/s)."
      if passed
      else f"Single-chip envelope check failed for {hardware_sku} under {resolved_cap}: "
      + " ".join(violations)
  )

  return SingleChipVerificationResult(
      passed=passed,
      hardware_sku=hardware_sku,
      perf_cap=resolved_cap,
      detected_device_count=dev_count,
      native_vram_gb=native_vram_gb,
      measured_vram_gb=(
          round(float(measured_vram_gb), 3)
          if measured_vram_gb is not None
          else None
      ),
      max_vram_gb=envelope.max_vram_gb,
      fp16_tflops=fp16_tflops,
      max_fp16_tflops=envelope.max_fp16_tflops,
      fp8_tops=fp8_tops,
      max_fp8_tops=envelope.max_int8_fp8_tops,
      mem_bw_gbps=mem_bw_gbps,
      max_mem_bw_gbps=envelope.max_mem_bw_gbps,
      violations=violations,
      reason=reason,
  )


_VISIBLE_DEV_ENV_RE = re.compile(
    r"\b(CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|TPU_VISIBLE_DEVICES|ASCEND_RT_VISIBLE_DEVICES|ZE_AFFINITY_MASK)\s*=\s*(['\"]?)([^'\"\s#]+)\2"
)
_PARALLEL_FLAG_RE = re.compile(
    r"(?:^|\s)(--tensor-parallel-size|-tp|--pipeline-parallel-size|-pp|--data-parallel-size|-dp|--num-devices|--num-gpus|--nproc_per_node|--nproc-per-node|--num_processes|--num-processes)(?:\s+|=)(\d+)\b"
)
_TPU_BOUNDS_RE = re.compile(
    r"\bTPU_CHIPS_PER_HOST_BOUNDS\s*=\s*(['\"]?)(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\1"
)
_TPU_TOPOLOGY_RE = re.compile(
    r"\b(?:TPU_TOPOLOGY|--topology|--tpu-topology)(?:\s+|=)(['\"]?)(\d+)x(\d+)(?:x(\d+))?\1"
)
_MULTI_CHIP_TPU_ACCEL_RE = re.compile(
    r"\b(v5e|v5p|v6e|v7x)-(4|8|16|32|64|128|256)\b", re.IGNORECASE
)


def _strip_shell_comment(line: str) -> str:
  """Strip `# ...` comments from a shell command line while preserving quoted strings."""
  stripped = line.lstrip()
  if not stripped or stripped.startswith("#"):
    return ""
  in_single = False
  in_double = False
  out_chars: List[str] = []
  for idx, ch in enumerate(stripped):
    if ch == "'" and not in_double:
      in_single = not in_single
    elif ch == '"' and not in_single:
      in_double = not in_double
    elif ch == "#" and not in_single and not in_double:
      if idx == 0 or stripped[idx - 1].isspace():
        break
    out_chars.append(ch)
  return "".join(out_chars).strip()


def audit_command_line_for_multi_chip_bypass(
    cmd_or_env: str,
) -> MultiChipAuditResult:
  """Audit a launch command or environment block for multi-chip parallelism bypasses (Gate 4).

  Detects positive violations:
    - Multi-device visibility env vars (`CUDA_VISIBLE_DEVICES=0,1`, `HIP_VISIBLE_DEVICES=0,1`, `TPU_VISIBLE_DEVICES=0,1,2,3`).
    - Multi-device parallelism flags (`--tensor-parallel-size 2`, `--pipeline-parallel-size 2`, `--nproc_per_node=4`).
    - Multi-chip TPU host bounds or topologies (`TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`, `TPU_TOPOLOGY=2x2x1`, `v6e-4`, `v6e-8`).

  Negative-control immunity:
    - Ignores `# ...` full-line and inline comments.
    - Stays silent on single-device settings (`CUDA_VISIBLE_DEVICES=0`, `--tensor-parallel-size 1`, `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1`).
  """
  violations: List[str] = []
  raw_lines = str(cmd_or_env).splitlines()
  inspected = 0

  for raw_line in raw_lines:
    active = _strip_shell_comment(raw_line)
    if not active:
      continue
    inspected += 1

    # 1. Check *_VISIBLE_DEVICES=0,1
    for m in _VISIBLE_DEV_ENV_RE.finditer(active):
      var_name = m.group(1)
      val_str = m.group(3).strip()
      devices = [tok.strip() for tok in val_str.split(",") if tok.strip()]
      if len(devices) > 1:
        violations.append(
            f"Multi-device environment variable {var_name}={val_str!r} exposes {len(devices)} devices. "
            f"Set {var_name}={devices[0]} to enforce single-chip execution."
        )

    # 2. Check --tensor-parallel-size / --pipeline-parallel-size / --nproc_per_node > 1
    for m in _PARALLEL_FLAG_RE.finditer(active):
      flag_name = m.group(1)
      count_val = int(m.group(2))
      if count_val > 1:
        violations.append(
            f"Multi-device parallelism flag '{flag_name} {count_val}' (> 1) violates single-chip rules. "
            f"Set '{flag_name} 1'."
        )

    # 3. Check TPU_CHIPS_PER_HOST_BOUNDS=x,y,z where x*y*z > 1
    for m in _TPU_BOUNDS_RE.finditer(active):
      bx, by, bz = int(m.group(2)), int(m.group(3)), int(m.group(4))
      prod = bx * by * bz
      if prod > 1:
        violations.append(
            f"Multi-chip TPU bounds TPU_CHIPS_PER_HOST_BOUNDS={bx},{by},{bz} ({prod} chips) violates single-chip rules. "
            "Use TPU_CHIPS_PER_HOST_BOUNDS=1,1,1."
        )

    # 4. Check TPU_TOPOLOGY=2x2x1
    for m in _TPU_TOPOLOGY_RE.finditer(active):
      tx = int(m.group(2))
      ty = int(m.group(3))
      tz = int(m.group(4)) if m.group(4) is not None else 1
      if tx * ty * tz > 1:
        topo_str = f"{tx}x{ty}" + (f"x{tz}" if m.group(4) is not None else "")
        violations.append(
            f"Multi-chip TPU topology {topo_str!r} ({tx * ty * tz} chips) violates single-chip rules. "
            "Use 1x1x1 (single-chip)."
        )

    # 5. Check multi-chip TPU accelerator slices (e.g. v6e-4, v6e-8)
    for m in _MULTI_CHIP_TPU_ACCEL_RE.finditer(active):
      gen = m.group(1)
      chips = int(m.group(2))
      if chips > 1:
        violations.append(
            f"Multi-chip TPU slice '{gen}-{chips}' ({chips} chips) violates single-chip rules. "
            f"Use '{gen}-1' (single-chip)."
        )

  passed = len(violations) == 0
  reason = (
      "No multi-chip bypass flags or environment variables detected."
      if passed
      else "Multi-chip bypass audit failed: " + " ".join(violations)
  )
  return MultiChipAuditResult(
      passed=passed,
      inspected_lines=inspected,
      violations=violations,
      reason=reason,
  )


def parse_rocm_smi_json(json_text: str) -> Dict[str, Any]:
  """Parse AMD `rocm-smi --showpower --showmeminfo vram --json` output.

  Example JSON:
    {
      "card0": {
        "Card series": "AMD Instinct MI210",
        "VRAM Total Memory (B)": "68719476736",
        "VRAM Total Used Memory (B)": "17179869184",
        "Average Graphics Package Power (W)": "215.4"
      }
    }
  """
  data = json.loads(json_text)
  if not isinstance(data, dict) or not data:
    raise ValueError(f"Expected non-empty JSON object from rocm-smi --json, got: {json_text!r}.")
  card_keys = [k for k in sorted(data.keys()) if k.lower().startswith("card")]
  if not card_keys:
    card_keys = [sorted(data.keys())[0]]
  detected_count = len(card_keys)
  card = data[card_keys[0]]
  if not isinstance(card, dict):
    raise ValueError(f"Expected object for {card_keys[0]} in rocm-smi JSON.")

  lower_map = {str(k).lower(): v for k, v in card.items()}
  name = str(
      lower_map.get("card series")
      or lower_map.get("card model")
      or "AMD Instinct / Radeon GPU"
  )
  total_b = float(lower_map.get("vram total memory (b)", 0.0))
  used_b = float(lower_map.get("vram total used memory (b)", 0.0))
  power_w = float(
      lower_map.get("average graphics package power (w)")
      or lower_map.get("current socket graphics package power (w)")
      or 0.0
  )
  if total_b <= 0.0 or power_w <= 0.0:
    raise ValueError(
        f"Incomplete VRAM or power fields in rocm-smi --json output: {json_text!r}."
    )
  return {
      "device_name": name,
      "vendor": "AMD_ROCM",
      "detected_device_count": detected_count,
      "peak_vram_mb": round(used_b / (1024.0 * 1024.0), 2),
      "total_vram_mb": round(total_b / (1024.0 * 1024.0), 2),
      "mean_power_watts": round(power_w, 2),
  }


class SingleChipHardwareVerifier:
  """Live 100 ms background telemetry collector and single-chip eligibility verifier across NVIDIA, AMD, and Google Cloud TPU."""

  def __init__(
      self,
      hardware_sku: str = "TPU_V6E_1_32GB",
      perf_cap: str | PerformanceCapId = PerformanceCapId.CAP_SINGLE_CHIP,
      poll_interval_s: float = 0.10,
      allow_live_probe: bool = False,
      custom_sampler: Optional[Callable[[], TelemetrySamplePoint]] = None,
  ) -> None:
    if hardware_sku not in HARDWARE_SKU_SPECS:
      valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
      raise ValueError(
          f"Unsupported hardware_sku={hardware_sku!r}. Choose from: [{valid_skus}]."
      )
    self.hardware_sku = hardware_sku
    self.spec = HARDWARE_SKU_SPECS[hardware_sku]
    self.perf_cap = resolve_perf_cap_id(perf_cap)
    self.poll_interval_s = max(0.01, float(poll_interval_s))
    self.allow_live_probe = bool(allow_live_probe)
    self.custom_sampler = custom_sampler

    self._samples: List[TelemetrySamplePoint] = []
    self._lock = threading.Lock()
    self._stop_event = threading.Event()
    self._thread: Optional[threading.Thread] = None

  def detect_physical_device_count(self) -> int:
    """Detect visible accelerator device count across NVIDIA, AMD ROCm, and Google Cloud TPU."""
    if not self.allow_live_probe:
      return int(self.spec.get("num_devices", 1))

    vendor = str(self.spec.get("vendor", "NVIDIA"))
    if vendor == "GOOGLE_TPU":
      try:
        import importlib
        import importlib.util
        import sys

        torch_present = (
            sys.modules.get("torch") is not None
            or importlib.util.find_spec("torch") is not None
        )
        torchtpu_in_sys = sys.modules.get("torch_tpu", False)
        torchtpu_present = torchtpu_in_sys is not None and (
            torchtpu_in_sys is not False
            or importlib.util.find_spec("torch_tpu") is not None
        )
        if torch_present:
          torch_mod = importlib.import_module("torch")
          if getattr(torch_mod, "tpu", None) is None and torchtpu_present:
            importlib.import_module("torch_tpu")
          tpu_ns = getattr(torch_mod, "tpu", None)
          if tpu_ns is not None and callable(getattr(tpu_ns, "device_count", None)):
            cnt = int(tpu_ns.device_count())
            if cnt > 0:
              return cnt
      except Exception:
        pass
      try:
        import jax  # type: ignore[import-not-found]

        return len(jax.devices())
      except Exception:
        pass
      accel_devs = glob.glob("/dev/accel*")
      if accel_devs:
        return len(accel_devs)
      vfio_devs = [
          p for p in glob.glob("/dev/vfio/*") if os.path.basename(p).isdigit()
      ]
      if vfio_devs:
        return len(vfio_devs)
    elif vendor == "AMD_ROCM":
      try:
        proc = subprocess.run(
            ["rocm-smi", "--showpower", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
        )
        parsed = parse_rocm_smi_json(proc.stdout)
        return int(parsed.get("detected_device_count", 1))
      except Exception:
        pass
    else:
      try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if lines:
          return len(lines)
      except Exception:
        pass
    return int(self.spec.get("num_devices", 1))

  def sample_once(self) -> TelemetrySamplePoint:
    """Collect a single telemetry sample from live hardware CLI/API or calibrated SKU fallback."""
    if self.custom_sampler is not None:
      pt = self.custom_sampler()
      with self._lock:
        self._samples.append(pt)
      return pt

    vendor = str(self.spec.get("vendor", "NVIDIA"))
    sample: Optional[TelemetrySamplePoint] = None
    if self.allow_live_probe:
      if vendor == "GOOGLE_TPU":
        sample = self._probe_tpu()
      elif vendor == "AMD_ROCM":
        sample = self._probe_amd_rocm()
      elif vendor == "NVIDIA_TEGRA":
        sample = self._probe_tegrastats() or self._probe_nvidia_smi()
      else:
        sample = self._probe_nvidia_smi()

    if sample is None:
      total_mb = float(self.spec["vram_mb"])
      idle_w = float(self.spec["idle_watts"])
      tdp_w = float(self.spec["tdp_watts"])
      step_idx = len(self._samples)
      util = 0.62 + 0.04 * math.sin(step_idx * 0.5)
      pwr_w = round(idle_w + (tdp_w - idle_w) * util, 2)
      used_mb = round(total_mb * 0.48, 2)
      sample = TelemetrySamplePoint(
          timestamp_ns=time.perf_counter_ns(),
          vram_used_mb=used_mb,
          vram_total_mb=total_mb,
          power_watts=pwr_w,
          detected_device_count=int(self.spec.get("num_devices", 1)),
          source="sku_calibrated_100ms_sampler",
      )

    with self._lock:
      self._samples.append(sample)
    return sample

  def _probe_nvidia_smi(self) -> Optional[TelemetrySamplePoint]:
    try:
      proc = subprocess.run(
          [
              "nvidia-smi",
              "--query-gpu=memory.used,memory.total,power.draw",
              "--format=csv,noheader,nounits",
          ],
          capture_output=True,
          text=True,
          timeout=1.5,
          check=True,
      )
      lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
      if not lines:
        return None
      parts = [p.strip() for p in lines[0].split(",")]
      return TelemetrySamplePoint(
          timestamp_ns=time.perf_counter_ns(),
          vram_used_mb=float(parts[0]),
          vram_total_mb=float(parts[1]),
          power_watts=float(parts[2]),
          detected_device_count=len(lines),
          source="nvidia_smi_live",
      )
    except Exception:
      return None

  def _probe_tegrastats(self) -> Optional[TelemetrySamplePoint]:
    try:
      proc = subprocess.run(
          ["tegrastats", "--interval", "100"],
          capture_output=True,
          text=True,
          timeout=0.35,
          check=False,
      )
      lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
      if not lines:
        return None
      parsed = parse_tegrastats_line(lines[-1])
      return TelemetrySamplePoint(
          timestamp_ns=time.perf_counter_ns(),
          vram_used_mb=parsed.ram_used_mb,
          vram_total_mb=parsed.ram_total_mb,
          power_watts=parsed.vdd_in_watts,
          detected_device_count=1,
          source="tegrastats_ina3221_live",
      )
    except Exception:
      return None

  def _probe_amd_rocm(self) -> Optional[TelemetrySamplePoint]:
    try:
      proc = subprocess.run(
          ["rocm-smi", "--showpower", "--showmeminfo", "vram", "--json"],
          capture_output=True,
          text=True,
          timeout=1.5,
          check=True,
      )
      parsed = parse_rocm_smi_json(proc.stdout)
      return TelemetrySamplePoint(
          timestamp_ns=time.perf_counter_ns(),
          vram_used_mb=float(parsed["peak_vram_mb"]),
          vram_total_mb=float(parsed["total_vram_mb"]),
          power_watts=float(parsed["mean_power_watts"]),
          detected_device_count=int(parsed["detected_device_count"]),
          source="rocm_smi_json_live",
      )
    except Exception:
      return None

  def _probe_tpu(self) -> Optional[TelemetrySamplePoint]:
    try:
      proc = subprocess.run(
          ["tpu-info"],
          capture_output=True,
          text=True,
          timeout=1.5,
          check=True,
      )
      parsed = parse_tpu_info_output(proc.stdout)
      dev_count = self.detect_physical_device_count()
      return TelemetrySamplePoint(
          timestamp_ns=time.perf_counter_ns(),
          vram_used_mb=float(parsed["peak_vram_mb"]),
          vram_total_mb=float(parsed["total_vram_mb"]),
          power_watts=float(parsed["mean_power_watts"]),
          detected_device_count=dev_count,
          source="tpu_info_live",
      )
    except Exception:
      return None

  def start(self) -> None:
    """Start the background 100 ms telemetry sampling loop."""
    if self._thread is not None and self._thread.is_alive():
      return
    self._stop_event.clear()
    self.sample_once()

    def _loop() -> None:
      while not self._stop_event.wait(timeout=self.poll_interval_s):
        self.sample_once()

    self._thread = threading.Thread(
        target=_loop,
        name=f"SingleChipHardwareVerifier-{self.hardware_sku}",
        daemon=True,
    )
    self._thread.start()

  def stop(self) -> SingleChipTelemetrySummary:
    """Stop the background telemetry thread and return a `SingleChipTelemetrySummary`."""
    self._stop_event.set()
    if self._thread is not None and self._thread.is_alive():
      self._thread.join(timeout=2.0)
    self._thread = None
    return self.summarize()

  def summarize(self) -> SingleChipTelemetrySummary:
    """Summarize collected 100 ms telemetry samples and verify single-chip envelope compliance."""
    with self._lock:
      if not self._samples:
        self.sample_once()
      pts = list(self._samples)

    peak_vram_mb = round(max(p.vram_used_mb for p in pts), 2)
    total_vram_mb = round(max(p.vram_total_mb for p in pts), 2)
    mean_power_w = round(float(np.mean([p.power_watts for p in pts])), 2)
    peak_power_w = round(max(p.power_watts for p in pts), 2)
    detected_devs = max(p.detected_device_count for p in pts)
    source = pts[-1].source
    device_name = str(self.spec["display_name"])
    vendor = str(self.spec["vendor"])

    env_res = verify_single_chip_envelope(
        hardware_sku=self.hardware_sku,
        detected_device_count=detected_devs,
        perf_cap=self.perf_cap,
        measured_vram_gb=round(peak_vram_mb / 1024.0, 3),
    )

    return SingleChipTelemetrySummary(
        passed=env_res.passed,
        hardware_sku=self.hardware_sku,
        vendor=vendor,
        device_name=device_name,
        perf_cap=self.perf_cap,
        detected_device_count=detected_devs,
        sample_count=len(pts),
        poll_interval_ms=round(self.poll_interval_s * 1000.0, 1),
        peak_vram_mb=peak_vram_mb,
        total_vram_mb=total_vram_mb,
        mean_power_watts=mean_power_w,
        peak_power_watts=peak_power_w,
        telemetry_source=source,
        hardware_fingerprint=compute_hardware_fingerprint(
            self.hardware_sku, device_name, total_vram_mb
        ),
        envelope_verification=env_res,
        violations=list(env_res.violations),
        reason=env_res.reason,
    )

  def __enter__(self) -> "SingleChipHardwareVerifier":
    self.start()
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.stop()
