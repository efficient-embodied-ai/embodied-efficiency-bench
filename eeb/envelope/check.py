"""Live single-chip envelope check (plan §1: one envelope, one chip, `num_devices == 1`).

Enumerates accelerators with the vendor tools actually present on the host
(`nvidia-smi -L` + `--query-gpu`, `rocm-smi --showid`/`amd-smi list`, `tpu_info` / `torch_tpu` /
`jax.devices()`, Jetson `/etc/nv_tegra_release`), enforces exactly one visible device,
reads its total memory, and looks up spec TOPS / bandwidth from `embodied_bench.schema.HARDWARE_SKU_SPECS`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import re
import sys
from typing import Any, Dict, List, Optional

from eeb import PROVENANCE_MEASURED, PROVENANCE_UNAVAILABLE
from eeb.telemetry.source import run_cli, which

PROVENANCE_SPEC_TABLE = "SPEC_TABLE"

ENVELOPE_MAX_MEMORY_GB = 64.0
ENVELOPE_MAX_BF16_TFLOPS = 1000.0
ENVELOPE_MAX_FP8_INT8_TOPS = 2000.0
ENVELOPE_MAX_MEM_BW_TBPS = 2.0


@dataclass
class DeviceInfo:
  index: int
  name: str
  vendor: str
  total_memory_mb: Optional[float]
  source: str
  uuid: str = ""

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class DeviceInventory:
  devices: List[DeviceInfo] = field(default_factory=list)
  tools_tried: List[str] = field(default_factory=list)
  notes: List[str] = field(default_factory=list)
  visibility_env: Dict[str, str] = field(default_factory=dict)

  @property
  def num_devices(self) -> int:
    return len(self.devices)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "num_devices": self.num_devices,
        "devices": [d.to_dict() for d in self.devices],
        "tools_tried": list(self.tools_tried),
        "notes": list(self.notes),
        "visibility_env": dict(self.visibility_env),
    }


def _visible_indices(env_var: str) -> Optional[List[int]]:
  raw = os.environ.get(env_var)
  if raw is None or raw.strip() == "":
    return None
  out: List[int] = []
  for tok in raw.split(","):
    tok = tok.strip()
    if tok.isdigit():
      out.append(int(tok))
  return out


def _enumerate_nvidia(inv: DeviceInventory) -> None:
  if not which("nvidia-smi"):
    return
  inv.tools_tried.append("nvidia-smi")
  out = run_cli(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader,nounits"], timeout_s=5.0)
  if out is None:
    inv.notes.append("nvidia-smi present but failed (driver not loaded?)")
    return
  rows: List[DeviceInfo] = []
  for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
      continue
    try:
      mem = float(parts[2])
    except ValueError:
      mem = None
    rows.append(DeviceInfo(index=int(parts[0]) if parts[0].isdigit() else len(rows), name=parts[1], vendor="NVIDIA", total_memory_mb=mem, source="nvidia-smi", uuid=parts[3] if len(parts) > 3 else ""))
  visible = _visible_indices("CUDA_VISIBLE_DEVICES")
  if visible is not None:
    inv.visibility_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    kept = [d for d in rows if d.index in visible]
    if len(rows) > len(kept):
      inv.notes.append(f"host exposes {len(rows)} NVIDIA GPUs; CUDA_VISIBLE_DEVICES restricts the process to {len(kept)}")
    rows = kept
  inv.devices.extend(rows)


def _enumerate_jetson(inv: DeviceInventory) -> None:
  release = "/etc/nv_tegra_release"
  if not os.path.exists(release):
    return
  inv.tools_tried.append(release)
  if any(d.vendor.startswith("NVIDIA") for d in inv.devices):
    return
  total_mb: Optional[float] = None
  try:
    with open("/proc/meminfo", encoding="utf-8") as fp:
      for line in fp:
        if line.startswith("MemTotal:"):
          total_mb = float(line.split()[1]) / 1024.0
          break
  except OSError:
    pass
  name = "NVIDIA Jetson (L4T)"
  try:
    with open("/proc/device-tree/model", encoding="utf-8", errors="ignore") as fp:
      name = fp.read().strip("\x00\n ") or name
  except OSError:
    pass
  inv.devices.append(DeviceInfo(index=0, name=name, vendor="NVIDIA_TEGRA", total_memory_mb=total_mb, source="nv_tegra_release+/proc/meminfo"))
  inv.notes.append("Jetson unified memory: total_memory_mb is system RAM shared by CPU and GPU")


def _enumerate_amd(inv: DeviceInventory) -> None:
  if which("amd-smi"):
    inv.tools_tried.append("amd-smi")
    out = run_cli(["amd-smi", "list", "--json"], timeout_s=5.0)
    if out:
      try:
        import json

        data = json.loads(out)
        entries = data if isinstance(data, list) else data.get("gpus", data.get("gpu", []))
        for i, item in enumerate(entries if isinstance(entries, list) else []):
          inv.devices.append(DeviceInfo(index=int(item.get("gpu", i)), name=str(item.get("product_name", item.get("name", "AMD GPU"))), vendor="AMD_ROCM", total_memory_mb=None, source="amd-smi", uuid=str(item.get("uuid", "")))
          )
      except (ValueError, AttributeError):
        inv.notes.append("amd-smi list --json could not be parsed")
  if not any(d.vendor == "AMD_ROCM" for d in inv.devices) and which("rocm-smi"):
    inv.tools_tried.append("rocm-smi")
    out = run_cli(["rocm-smi", "--showid", "--showproductname", "--showmeminfo", "vram", "--csv"], timeout_s=5.0)
    if out:
      lines = [ln for ln in out.strip().splitlines() if ln.strip() and not ln.startswith("=")]
      if len(lines) >= 2:
        headers = [h.strip().lower() for h in lines[0].split(",")]
        for i, line in enumerate(lines[1:]):
          row = dict(zip(headers, [v.strip() for v in line.split(",")]))
          name = row.get("card series") or row.get("card model") or "AMD GPU"
          total_b = row.get("vram total memory (b)")
          inv.devices.append(DeviceInfo(index=i, name=name, vendor="AMD_ROCM", total_memory_mb=float(total_b) / (1024.0 * 1024.0) if total_b and total_b.replace(".", "", 1).isdigit() else None, source="rocm-smi", uuid=row.get("unique id", "")))
  visible = _visible_indices("HIP_VISIBLE_DEVICES") or _visible_indices("ROCR_VISIBLE_DEVICES")
  if visible is not None:
    inv.visibility_env["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", os.environ.get("ROCR_VISIBLE_DEVICES", ""))
    amd = [d for d in inv.devices if d.vendor == "AMD_ROCM"]
    kept = [d for d in amd if d.index in visible]
    if len(amd) > len(kept):
      inv.notes.append(f"host exposes {len(amd)} AMD GPUs; HIP/ROCR_VISIBLE_DEVICES restricts the process to {len(kept)}")
      inv.devices = [d for d in inv.devices if d.vendor != "AMD_ROCM"] + kept


_TPU_CHIP_RE = re.compile(r"(TPU\s*v\d+\w*|TPU\d+\w*)", re.IGNORECASE)
_TPU_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*GiB\s*/\s*(\d+(?:\.\d+)?)\s*GiB", re.IGNORECASE)
_TPU_DEV_ROW_RE = re.compile(r"\|\s*/dev/(?:vfio/\d+|accel\d+)\s*\|\s*([^|]+?)\s*\|", re.IGNORECASE)


def _apply_tpu_visibility(inv: DeviceInventory, rows: List[DeviceInfo]) -> List[DeviceInfo]:
  visible = _visible_indices("TPU_VISIBLE_DEVICES") or _visible_indices("TPU_VISIBLE_CHIPS")
  if visible is not None and len(rows) > 1:
    kept = [d for d in rows if d.index in visible]
    if len(rows) > len(kept):
      inv.notes.append(
          f"host exposes {len(rows)} Cloud TPU chips; TPU_VISIBLE_DEVICES restricts the process to {len(kept)}"
      )
    return kept
  return rows


def _enumerate_tpu_via_tpu_info_lib(inv: DeviceInventory) -> Optional[List[DeviceInfo]]:
  """Non-invasive sysfs + localhost:8431 TPU enumeration via `tpu_info` without locking `/dev/vfio/*`."""
  try:
    from tpu_info import device as _tpu_device  # noqa: WPS433
  except Exception:
    return None
  try:
    chip_type, count = _tpu_device.get_local_chips()
  except Exception:
    return None
  if chip_type is None or count <= 0:
    return None

  inv.tools_tried.append("tpu_info.device")
  per_chip_mem_mb: List[Optional[float]] = [float(chip_type.value.hbm_gib) * 1024.0] * int(count)
  src = "tpu_info.device"
  try:
    from tpu_info import metrics as _tpu_metrics  # noqa: WPS433

    usages = _tpu_metrics.get_chip_usage(chip_type)
    if usages:
      devs_per_chip = max(1, int(chip_type.value.devices_per_chip))
      measured: List[Optional[float]] = []
      for idx in range(int(count)):
        slice_u = usages[idx * devs_per_chip : (idx + 1) * devs_per_chip]
        if slice_u:
          tot_bytes = sum(int(u.total_memory) for u in slice_u)
          measured.append(float(tot_bytes) / (1024.0 * 1024.0))
        else:
          measured.append(per_chip_mem_mb[idx])
      per_chip_mem_mb = measured
      src = "tpu_info.metrics"
  except Exception:
    pass

  rows = [
      DeviceInfo(
          index=i,
          name=str(chip_type),
          vendor="GOOGLE_TPU",
          total_memory_mb=per_chip_mem_mb[i],
          source=src,
      )
      for i in range(int(count))
  ]
  return _apply_tpu_visibility(inv, rows)


def _enumerate_tpu(inv: DeviceInventory, use_torchtpu: bool = True, use_jax: bool = True) -> None:
  tpu_vis_raw = os.environ.get("TPU_VISIBLE_DEVICES") or os.environ.get("TPU_VISIBLE_CHIPS")
  if tpu_vis_raw:
    env_key = "TPU_VISIBLE_DEVICES" if os.environ.get("TPU_VISIBLE_DEVICES") else "TPU_VISIBLE_CHIPS"
    inv.visibility_env[env_key] = tpu_vis_raw

  # When another process (such as `eeb serve`) already holds `/dev/vfio/*`, avoid initializing a
  # conflicting PjRt runtime inside the client process and enumerate via `tpu_info` directly
  # (unless unit tests have explicitly preloaded `torch_tpu` or `jax` in `sys.modules`).
  if sys.modules.get("torch_tpu") is None and sys.modules.get("jax") is None:
    try:
      from tpu_info import device as _tpu_device  # noqa: WPS433

      owners = _tpu_device.get_chip_owners()
      other_owners = {p: pid for p, pid in owners.items() if pid != os.getpid()}
      if other_owners:
        rows_busy = _enumerate_tpu_via_tpu_info_lib(inv)
        if rows_busy:
          inv.devices.extend(rows_busy)
          return
    except Exception:
      pass

  if use_torchtpu:
    try:
      import importlib
      import importlib.util

      torch_present = sys.modules.get("torch") is not None or importlib.util.find_spec("torch") is not None
      torchtpu_in_sys = sys.modules.get("torch_tpu", False)
      torchtpu_present = torchtpu_in_sys is not None and (
          torchtpu_in_sys is not False or importlib.util.find_spec("torch_tpu") is not None
      )
      if torch_present:
        torch_mod = importlib.import_module("torch")
        tpu_ns = getattr(torch_mod, "tpu", None)
        if tpu_ns is None and torchtpu_present:
          importlib.import_module("torch_tpu")
          tpu_ns = getattr(torch_mod, "tpu", None)
        if tpu_ns is not None:
          inv.tools_tried.append("torch.tpu")
          count = 0
          dev_count_fn = getattr(tpu_ns, "device_count", None)
          if callable(dev_count_fn):
            try:
              count = int(dev_count_fn())
            except RuntimeError as exc:
              if "internal to TorchTPU" not in str(exc):
                raise
              acc = getattr(torch_mod, "accelerator", None)
              if acc is not None and callable(getattr(acc, "is_available", None)) and bool(acc.is_available()):
                cur = acc.current_accelerator() if callable(getattr(acc, "current_accelerator", None)) else None
                if str(getattr(cur, "type", cur)) == "tpu" and callable(getattr(acc, "device_count", None)):
                  count = int(acc.device_count())
          rows: List[DeviceInfo] = []
          name_fn = getattr(tpu_ns, "get_device_name", None)
          props_fn = getattr(tpu_ns, "get_device_properties", None)
          for i in range(max(0, count)):
            dev_name = "Google Cloud TPU"
            if callable(name_fn):
              try:
                raw_name = name_fn(i)
                if raw_name:
                  dev_name = str(raw_name)
              except Exception:
                pass
            mem_mb: Optional[float] = None
            if callable(props_fn):
              try:
                props = props_fn(i)
                total_mem = getattr(props, "total_memory", None)
                if total_mem is None and isinstance(props, dict):
                  total_mem = props.get("total_memory")
                if total_mem:
                  mem_mb = float(total_mem) / (1024.0 * 1024.0)
              except Exception:
                pass
            rows.append(
                DeviceInfo(
                    index=i,
                    name=dev_name,
                    vendor="GOOGLE_TPU",
                    total_memory_mb=mem_mb,
                    source="torch.tpu",
                )
            )
          if rows:
            inv.devices.extend(_apply_tpu_visibility(inv, rows))
            return
    except Exception as exc:  # pragma: no cover - torch_tpu present but broken
      inv.notes.append(f"torch.tpu failed: {type(exc).__name__}: {exc}")

  if use_jax:
    try:
      import importlib.util

      jax_in_sys = sys.modules.get("jax", False)
      jax_present = jax_in_sys is not None and (
          jax_in_sys is not False or importlib.util.find_spec("jax") is not None
      )
      if jax_present:
        inv.tools_tried.append("jax.devices()")
        import jax  # noqa: WPS433

        devs = [d for d in jax.devices() if getattr(d, "platform", "") == "tpu"]
        jax_rows: List[DeviceInfo] = []
        for d in devs:
          stats = {}
          try:
            stats = d.memory_stats() or {}
          except Exception:
            pass
          total = stats.get("bytes_limit")
          jax_rows.append(
              DeviceInfo(
                  index=int(getattr(d, "id", 0)),
                  name=str(getattr(d, "device_kind", "TPU")),
                  vendor="GOOGLE_TPU",
                  total_memory_mb=float(total) / (1024.0 * 1024.0) if total else None,
                  source="jax.devices()",
              )
          )
        if jax_rows:
          inv.devices.extend(_apply_tpu_visibility(inv, jax_rows))
          return
    except Exception as exc:  # pragma: no cover - jax present but broken
      inv.notes.append(f"jax.devices() failed: {type(exc).__name__}: {exc}")

  lib_rows = _enumerate_tpu_via_tpu_info_lib(inv)
  if lib_rows:
    inv.devices.extend(lib_rows)
    return

  if which("tpu-info"):
    inv.tools_tried.append("tpu-info")
    out = run_cli(["tpu-info"], timeout_s=15.0)
    if out and "No TPU chips found" not in out:
      chips = _TPU_MEM_RE.findall(out)
      kind = _TPU_CHIP_RE.search(out)
      if chips:
        rows = [
            DeviceInfo(
                index=i,
                name=kind.group(1) if kind else "Google Cloud TPU",
                vendor="GOOGLE_TPU",
                total_memory_mb=float(total) * 1024.0,
                source="tpu-info",
            )
            for i, (_used, total) in enumerate(chips)
        ]
      else:
        dev_rows = _TPU_DEV_ROW_RE.findall(out)
        rows = [
            DeviceInfo(
                index=i,
                name=chip_name.strip() or (kind.group(1) if kind else "Google Cloud TPU"),
                vendor="GOOGLE_TPU",
                total_memory_mb=32768.0 if "v6e" in chip_name.lower() or "v4" in chip_name.lower() else (16384.0 if "v5e" in chip_name.lower() else None),
                source="tpu-info",
            )
            for i, chip_name in enumerate(dev_rows)
        ]
      inv.devices.extend(_apply_tpu_visibility(inv, rows))


def enumerate_devices(use_jax: bool = True, use_torchtpu: bool = True) -> DeviceInventory:
  """Enumerate accelerators visible to this process with whatever vendor tools exist."""
  inv = DeviceInventory()
  _enumerate_nvidia(inv)
  _enumerate_jetson(inv)
  _enumerate_amd(inv)
  _enumerate_tpu(inv, use_torchtpu=use_torchtpu, use_jax=use_jax)
  if not inv.tools_tried:
    inv.notes.append(
        "no vendor enumeration tool found (nvidia-smi, /etc/nv_tegra_release, amd-smi, rocm-smi, torch.tpu, jax, tpu-info)"
    )
  return inv


@dataclass
class EnvelopeCheckResult:
  passed: bool
  num_devices: int
  reason: str
  hardware_sku: Optional[str]
  device: Optional[Dict[str, Any]]
  measured_total_memory_mb: Optional[float]
  memory_provenance: str
  spec: Dict[str, Any]
  spec_provenance: str
  envelope_limits: Dict[str, float]
  violations: List[str]
  inventory: Dict[str, Any]

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def sku_spec_summary(hardware_sku: Optional[str]) -> Dict[str, Any]:
  from embodied_bench.schema import HARDWARE_SKU_SPECS, resolve_hardware_sku

  if not hardware_sku:
    return {}
  resolved = resolve_hardware_sku(hardware_sku)
  if resolved not in HARDWARE_SKU_SPECS:
    return {}
  s = HARDWARE_SKU_SPECS[resolved]
  return {
      "display_name": s.get("display_name"),
      "vendor": s.get("vendor"),
      "vram_mb": s.get("vram_mb"),
      "fp16_tflops": s.get("fp16_tflops"),
      "fp8_int8_tops": s.get("fp8_tflops"),
      "mem_bw_gbps": s.get("mem_bw_gbps"),
      "tdp_watts": s.get("tdp_watts"),
      "hourly_cost_usd": s.get("hourly_cost_usd"),
  }


def check_envelope(
    hardware_sku: Optional[str] = None,
    inventory: Optional[DeviceInventory] = None,
    allow_no_device: bool = False,
) -> EnvelopeCheckResult:
  """Enforce `num_devices == 1`, memory <= 64 GB, and spec TOPS/BW within the envelope."""
  from embodied_bench.schema import resolve_hardware_sku

  if hardware_sku:
    hardware_sku = resolve_hardware_sku(hardware_sku)
  inv = inventory if inventory is not None else enumerate_devices()
  violations: List[str] = []
  spec = sku_spec_summary(hardware_sku)
  spec_prov = PROVENANCE_SPEC_TABLE if spec else PROVENANCE_UNAVAILABLE
  if hardware_sku and not spec:
    violations.append(f"hardware_sku {hardware_sku!r} is not in embodied_bench.schema.HARDWARE_SKU_SPECS")

  device: Optional[DeviceInfo] = inv.devices[0] if inv.num_devices == 1 else None
  mem_mb = device.total_memory_mb if device else None
  mem_prov = PROVENANCE_MEASURED if mem_mb is not None else PROVENANCE_UNAVAILABLE

  if inv.num_devices == 0:
    msg = "no accelerator device enumerated on this host (CPU-only); EEB requires exactly one chip"
    if allow_no_device:
      inv.notes.append("allow_no_device=True: CPU-only host accepted for harness self-test (never admissible)")
    else:
      violations.append(msg)
  elif inv.num_devices > 1:
    violations.append(
        f"{inv.num_devices} accelerator devices visible; EEB requires num_devices == 1. "
        "Pin one device (CUDA_VISIBLE_DEVICES=0 / HIP_VISIBLE_DEVICES=0 / TPU_VISIBLE_DEVICES=0 on a 1-chip TPU host)."
    )

  if mem_mb is not None and mem_mb / 1024.0 > ENVELOPE_MAX_MEMORY_GB + 0.5:
    violations.append(f"device memory {mem_mb / 1024.0:.1f} GB exceeds the {ENVELOPE_MAX_MEMORY_GB:.0f} GB single-chip envelope")
  if spec:
    if float(spec.get("fp16_tflops") or 0.0) > ENVELOPE_MAX_BF16_TFLOPS:
      violations.append(f"SKU spec {spec['fp16_tflops']} BF16 TFLOPS exceeds {ENVELOPE_MAX_BF16_TFLOPS:.0f}")
    if float(spec.get("fp8_int8_tops") or 0.0) > ENVELOPE_MAX_FP8_INT8_TOPS:
      violations.append(f"SKU spec {spec['fp8_int8_tops']} FP8/INT8 TOPS exceeds {ENVELOPE_MAX_FP8_INT8_TOPS:.0f}")
    if float(spec.get("mem_bw_gbps") or 0.0) > ENVELOPE_MAX_MEM_BW_TBPS * 1000.0:
      violations.append(f"SKU spec {spec['mem_bw_gbps']} GB/s exceeds {ENVELOPE_MAX_MEM_BW_TBPS:.1f} TB/s")
    if mem_mb is not None and spec.get("vram_mb") and abs(float(spec["vram_mb"]) - mem_mb) / float(spec["vram_mb"]) > 0.15:
      violations.append(
          f"measured device memory {mem_mb:.0f} MB differs >15% from SKU {hardware_sku} spec {spec['vram_mb']} MB; "
          "the recipe hardware_sku probably does not match the host"
      )
    if device and spec.get("vendor") and str(spec["vendor"]).split("_")[0] != device.vendor.split("_")[0]:
      violations.append(f"device vendor {device.vendor} does not match SKU vendor {spec['vendor']}")

  passed = not violations
  if passed:
    reason = (
        f"single device {device.name!r} ({device.vendor}, {mem_mb:.0f} MB) within envelope"
        if device and mem_mb is not None
        else ("CPU-only host accepted with allow_no_device=True (harness self-test only)" if inv.num_devices == 0 else f"single device {device.name!r} within envelope")
    )
  else:
    reason = "; ".join(violations)
  return EnvelopeCheckResult(
      passed=passed,
      num_devices=inv.num_devices,
      reason=reason,
      hardware_sku=hardware_sku,
      device=device.to_dict() if device else None,
      measured_total_memory_mb=mem_mb,
      memory_provenance=mem_prov,
      spec=spec,
      spec_provenance=spec_prov,
      envelope_limits={
          "max_memory_gb": ENVELOPE_MAX_MEMORY_GB,
          "max_bf16_tflops": ENVELOPE_MAX_BF16_TFLOPS,
          "max_fp8_int8_tops": ENVELOPE_MAX_FP8_INT8_TOPS,
          "max_mem_bw_tbps": ENVELOPE_MAX_MEM_BW_TBPS,
      },
      violations=violations,
      inventory=inv.to_dict(),
  )


__all__ = [
    "DeviceInfo",
    "DeviceInventory",
    "ENVELOPE_MAX_BF16_TFLOPS",
    "ENVELOPE_MAX_FP8_INT8_TOPS",
    "ENVELOPE_MAX_MEMORY_GB",
    "ENVELOPE_MAX_MEM_BW_TBPS",
    "EnvelopeCheckResult",
    "PROVENANCE_SPEC_TABLE",
    "check_envelope",
    "enumerate_devices",
    "sku_spec_summary",
]
