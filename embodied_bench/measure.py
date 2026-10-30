"""Device-agnostic Performance Cap Envelopes (`PerformanceCapEnvelope`), 3-mode cap satisfaction (`native_qualifying_device`, `hardware_enforced_slice_clock_cap`, `roofline_normalized_latency`), Roofline-Normalized Effective Latency ($L_{\\text{norm}}$), and hardware-independent efficiency profiling (`FLOPs/action`, `FLOPs/frame`, `Bytes-moved/step`, `Peak VRAM (GB)`) for `eeb measure`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from embodied_bench.schema import (
    HARDWARE_SKU_SPECS,
    PERFORMANCE_CAP_ENVELOPES,
    PerfCapSatisfactionMode,
    PerformanceCapEnvelope,
    PerformanceCapId,
    RecipeConfig,
    is_wm_track,
    resolve_perf_cap_id,
)


BYTES_PER_PARAM_TABLE: Dict[str, float] = {
    "FP32": 4.0,
    "BF16": 2.0,
    "FP16": 2.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "W4A8": 0.65,
    "W4A16": 0.55,
    "INT4_AWQ": 0.55,
    "INT4": 0.55,
    "NVFP4": 0.52,
}


@dataclass(frozen=True)
class HardwareIndependentProfile:
  """Hardware-independent algorithmic efficiency metrics recorded per step/action/frame."""

  track: str
  model_params_b: float
  precision: str
  bytes_per_param: float
  batch_or_concurrency: int
  horizon_or_chunk: int
  denoising_steps: int
  flops_per_step_gflops: float
  flops_per_action_gflops: Optional[float]
  flops_per_frame_gflops: Optional[float]
  bytes_moved_per_step_mb: float
  bytes_moved_per_step_bytes: int
  arithmetic_intensity_flops_per_byte: float
  ai_op: float
  peak_vram_gb: float

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class PerformanceCapEvaluation:
  """Result of evaluating a single-device execution against a device-agnostic PerformanceCapEnvelope."""

  envelope_id: str
  tier_alias: str
  envelope: Dict[str, Any]
  hardware_sku: str
  num_devices: int
  single_device_compliant: bool
  satisfaction_mode: str
  satisfied_natively: bool
  satisfied_via_slice_or_clock_cap: bool
  satisfied_via_roofline_normalization: bool
  vram_budget_compliant: bool
  passed: bool
  device_fp16_tflops: float
  device_fp8_int8_tops: float
  device_mem_bw_gbps: float
  device_total_vram_gb: float
  effective_tflops: float
  effective_fp8_int8_tops: float
  effective_mem_bw_gbps: float
  effective_vram_cap_gb: float
  peak_vram_gb: float
  mps_active_thread_pct: float
  clock_scale_factor: float
  vram_fraction_cap: float
  ai_op: float
  roofline_normalization_factor: float
  physical_latency_mean_ms: float
  physical_latency_p50_ms: float
  physical_latency_p99_ms: float
  normalized_latency_mean_ms: float
  normalized_latency_p50_ms: float
  normalized_latency_p99_ms: float
  normalized_reactivity_hz: float
  raw_single_device_throughput: float
  raw_per_chip_throughput: float
  power_efficiency_per_watt: float
  system_power_watts: float
  throughput_per_100_tops: float
  throughput_per_tbps_bw: float
  throughput_per_100_gbps_bw: float
  same_chip_speedup_vs_bf16_ref: float
  throughput_per_dollar_hr: float
  tco_efficiency_per_dollar: float
  iso_compute_and_bw_efficiency: Dict[str, Any]
  semianalysis_normalizations: Dict[str, Any]
  hardware_independent_profile: Dict[str, Any]
  throughput_per_1k_usd: float = 0.0
  enforcement_commands: Dict[str, str] = field(default_factory=dict)
  violations: List[str] = field(default_factory=list)
  reason: str = "Performance cap envelope satisfied."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def get_performance_cap_envelope(envelope_id: str | PerformanceCapId) -> PerformanceCapEnvelope:
  """Look up a `PerformanceCapEnvelope` by ID or `SINGLE-*` alias with actionable error messaging."""
  key = resolve_perf_cap_id(envelope_id)
  if key not in PERFORMANCE_CAP_ENVELOPES:
    valid = ", ".join(sorted(PERFORMANCE_CAP_ENVELOPES.keys()))
    raise ValueError(
        f"Unknown performance cap envelope {envelope_id!r}. Supported envelopes: [{valid}]."
    )
  return PERFORMANCE_CAP_ENVELOPES[key]


def resolve_device_exec_tflops(sku_spec: Dict[str, Any], precision: str) -> float:
  """Return the non-sparse dense peak silicon compute (TFLOPS/TOPS) for a given execution precision."""
  prec = str(precision or "BF16").upper()
  fp16 = float(sku_spec.get("fp16_tflops", 330.0))
  if prec in ("FP32", "BF16", "FP16", "W4A16", "INT4_AWQ"):
    return fp16
  if prec == "FP8":
    if bool(sku_spec.get("supports_native_fp8", True)):
      return float(sku_spec.get("fp8_tflops", fp16 * 2.0))
    return fp16
  if prec in ("INT8", "W4A8", "INT4", "NVFP4"):
    return float(sku_spec.get("dense_int8_tops", sku_spec.get("fp8_tflops", fp16 * 2.0)))
  return fp16


def compute_iso_tops_and_bw_efficiency(
    raw_single_device_throughput: float,
    effective_bf16_tflops: Optional[float] = None,
    effective_mem_bw_gbps: float = 1000.0,
    *,
    effective_fp8_int8_tops: Optional[float] = None,
) -> Tuple[float, float]:
  """Compute Iso-TOPS Normalized Throughput (`throughput_per_100_tops` against Dense BF16 TFLOPS / 100) and Iso-Bandwidth Efficiency (`throughput_per_100_gbps_bw`)."""
  resolved_tflops = (
      float(effective_bf16_tflops)
      if effective_bf16_tflops is not None
      else float(effective_fp8_int8_tops if effective_fp8_int8_tops is not None else 500.0)
  )
  tops_denom = max(resolved_tflops / 100.0, 1e-6)
  bw_denom = max(float(effective_mem_bw_gbps) / 100.0, 1e-6)
  iso_tops = round(float(raw_single_device_throughput) / tops_denom, 4)
  iso_bw = round(float(raw_single_device_throughput) / bw_denom, 4)
  return iso_tops, iso_bw


def compute_semianalysis_normalizations(
    raw_per_chip_throughput: Optional[float] = None,
    system_power_watts: Optional[float] = None,
    peak_fp8_int8_tops: Optional[float] = None,
    peak_mem_bw_gbps: Optional[float] = None,
    same_chip_speedup_vs_bf16_ref: float = 1.0,
    hourly_cost_usd: Optional[float] = None,
    *,
    capex_usd: Optional[float] = None,
    raw_throughput: Optional[float] = None,
    hardware_sku: Optional[str] = None,
    accelerator_power_watts: Optional[float] = None,
    device_power_watts: Optional[float] = None,
    system_overhead_watts: Optional[float] = None,
    peak_bf16_tflops: Optional[float] = None,
    effective_bf16_tflops: Optional[float] = None,
    peak_tops: Optional[float] = None,
    effective_fp8_int8_tops: Optional[float] = None,
    peak_bw_gbps: Optional[float] = None,
    effective_mem_bw_gbps: Optional[float] = None,
    peak_bw_tbps: Optional[float] = None,
    bf16_ref_throughput_same_chip: Optional[float] = None,
    same_chip_bf16_ref_throughput: Optional[float] = None,
) -> Dict[str, Any]:
  """Compute SemiAnalysis-Style 4-Way Performance Normalizations for an operating point:

  1. Y1 `raw_per_chip_throughput`: Raw `Executed Actions/s/chip` or `Generated Frames/s/chip`.
  2. Y2 `power_efficiency_per_watt` (`Actions/Joule` or `Frames/Joule`):
       Throughput / P_active_watts (measured active device board/chip power only; zero invented host overhead).
  3. Y3 `tco_efficiency_per_dollar` (`(Output/s) / ($/hr)`):
       - `tco_efficiency_per_dollar` / `throughput_per_dollar_hr`: Throughput / Reference_Hourly_Rate_USD (`$/hr`).
       - `units_per_dollar`: (Throughput * 3600) / Reference_Hourly_Rate_USD (`Output / $`).
       - `throughput_per_1k_usd`: Auxiliary CapEx reference `Throughput / (CapEx_USD / 1000)`.
  4. Y4 `iso_compute_and_bw_efficiency`:
       - `throughput_per_100_tops`: Throughput / (Dense BF16 TFLOPS / 100) so quantization speedups increase Y4.
       - `throughput_per_tbps_bw`: Throughput / Peak BW (TB/s).
       - `same_chip_speedup_vs_bf16_ref`: Speedup vs. unoptimized native BF16 reference on the exact same chip.
  """
  del system_overhead_watts  # Strict Y2 invariant: zero invented system overhead watts.
  sku_spec = HARDWARE_SKU_SPECS.get(hardware_sku, {}) if hardware_sku else {}
  resolved_raw = raw_per_chip_throughput if raw_per_chip_throughput is not None else (raw_throughput or 0.0)
  raw_tput = round(float(resolved_raw), 4)

  accel_w = accelerator_power_watts if accelerator_power_watts is not None else device_power_watts
  if accel_w is not None:
    active_watts = float(accel_w)
  elif system_power_watts is not None:
    active_watts = float(system_power_watts)
  else:
    tdp_w = float(sku_spec.get("tdp_watts", 250.0))
    active_watts = tdp_w * 0.75

  power_eff_per_watt = round(raw_tput / max(active_watts, 1e-6), 4)

  if effective_bf16_tflops is not None:
    resolved_bf16_tflops = float(effective_bf16_tflops)
  elif peak_bf16_tflops is not None:
    resolved_bf16_tflops = float(peak_bf16_tflops)
  elif "fp16_tflops" in sku_spec:
    resolved_bf16_tflops = float(sku_spec["fp16_tflops"])
  elif peak_fp8_int8_tops is not None:
    resolved_bf16_tflops = float(peak_fp8_int8_tops) / 2.0
  elif effective_fp8_int8_tops is not None:
    resolved_bf16_tflops = float(effective_fp8_int8_tops) / 2.0
  elif peak_tops is not None:
    resolved_bf16_tflops = float(peak_tops) / 2.0
  else:
    resolved_bf16_tflops = 330.0

  if peak_mem_bw_gbps is not None:
    resolved_bw_gbps = float(peak_mem_bw_gbps)
  elif effective_mem_bw_gbps is not None:
    resolved_bw_gbps = float(effective_mem_bw_gbps)
  elif peak_bw_gbps is not None:
    resolved_bw_gbps = float(peak_bw_gbps)
  elif peak_bw_tbps is not None:
    resolved_bw_gbps = float(peak_bw_tbps) * 1000.0
  else:
    resolved_bw_gbps = float(sku_spec.get("mem_bw_gbps", 1000.0))

  resolved_bw_tbps = resolved_bw_gbps / 1000.0
  tput_per_100_tops = round(raw_tput / max(resolved_bf16_tflops / 100.0, 1e-6), 4)
  tput_per_tbps_bw = round(raw_tput / max(resolved_bw_tbps, 1e-6), 4)
  tput_per_100_gbps_bw = round(raw_tput / max(resolved_bw_gbps / 100.0, 1e-6), 4)

  ref_tput = bf16_ref_throughput_same_chip if bf16_ref_throughput_same_chip is not None else same_chip_bf16_ref_throughput
  if ref_tput is not None and float(ref_tput) > 0.0:
    resolved_speedup = round(raw_tput / float(ref_tput), 4)
  else:
    resolved_speedup = round(float(same_chip_speedup_vs_bf16_ref), 4)

  resolved_cost = float(
      hourly_cost_usd
      if hourly_cost_usd is not None
      else sku_spec.get("hourly_cost_usd", 1.0)
  )
  resolved_capex = float(
      capex_usd
      if (capex_usd is not None and float(capex_usd) > 0.0)
      else sku_spec.get("capex_usd", 2500.0)
  )
  tco_per_dollar_hr_rate = round(raw_tput / max(resolved_cost, 1e-6), 4)
  units_per_dollar = round((raw_tput * 3600.0) / max(resolved_cost, 1e-6), 4)
  tput_per_1k_usd = round(raw_tput / max(resolved_capex / 1000.0, 1e-6), 4)

  return {
      "raw_per_chip_throughput": raw_tput,
      "power_efficiency_per_watt": power_eff_per_watt,
      "active_device_power_watts": round(active_watts, 2),
      "system_power_watts": round(active_watts, 2),
      "iso_compute_and_bw_efficiency": {
          "throughput_per_100_tops": tput_per_100_tops,
          "throughput_per_tbps_bw": tput_per_tbps_bw,
          "throughput_per_100_gbps_bw": tput_per_100_gbps_bw,
          "same_chip_speedup_vs_bf16_ref": resolved_speedup,
          "dense_bf16_tflops": round(resolved_bf16_tflops, 3),
      },
      "throughput_per_100_tops": tput_per_100_tops,
      "throughput_per_tbps_bw": tput_per_tbps_bw,
      "throughput_per_100_gbps_bw": tput_per_100_gbps_bw,
      "same_chip_speedup_vs_bf16_ref": resolved_speedup,
      "tco_efficiency_per_dollar": tco_per_dollar_hr_rate,
      "throughput_per_dollar_hr": tco_per_dollar_hr_rate,
      "units_per_dollar": units_per_dollar,
      "throughput_per_1k_usd": tput_per_1k_usd,
      "capex_usd": round(resolved_capex, 2),
      "hourly_cost_usd": round(resolved_cost, 4),
  }


def compute_roofline_normalized_latency(
    physical_latency_ms: float,
    tflops_eff: float,
    tflops_cap: float,
    bw_eff_gbps: float,
    bw_cap_gbps: float,
    ai_op: float = 1.0,
) -> Tuple[float, float]:
  """Compute Roofline-Normalized Effective Latency ($L_{\\text{norm}}$) and normalization factor.

  Formula:
    L_norm = L_phys * max(1.0, min(TFLOPS_eff / TFLOPS_cap, AI_op * (BW_eff / BW_cap)))

  When the device's effective compute (`tflops_eff`) or bandwidth (`bw_eff_gbps`) is already
  within the envelope (`<= tflops_cap` and `<= bw_cap_gbps`), the multiplier clamps to `1.0`
  so native qualifying devices and hardware-enforced slices are never penalized.

  Returns:
    (normalized_latency_ms, normalization_factor)
  """
  if physical_latency_ms < 0.0:
    raise ValueError(f"physical_latency_ms must be >= 0.0, got {physical_latency_ms}.")
  if tflops_cap <= 0.0 or bw_cap_gbps <= 0.0:
    raise ValueError(
        f"tflops_cap ({tflops_cap}) and bw_cap_gbps ({bw_cap_gbps}) must be > 0.0."
    )
  compute_ratio = float(tflops_eff) / float(tflops_cap)
  bw_ratio = max(0.05, float(ai_op)) * (float(bw_eff_gbps) / float(bw_cap_gbps))
  norm_factor = max(1.0, min(compute_ratio, bw_ratio))
  l_norm = float(physical_latency_ms) * norm_factor
  return round(l_norm, 3), round(norm_factor, 4)


def compute_hardware_independent_profile(
    recipe: RecipeConfig,
    model_params_b: float,
    batch_or_concurrency: int,
    horizon_or_chunk: int,
    peak_vram_mb: float,
    envelope: Optional[PerformanceCapEnvelope] = None,
) -> HardwareIndependentProfile:
  """Compute hardware-independent `FLOPs/action` (or `FLOPs/frame`), `Bytes-moved/step`, and `Peak VRAM (GB)`."""
  prec_upper = recipe.precision.upper()
  bytes_per_param = BYTES_PER_PARAM_TABLE.get(prec_upper, 2.0)
  b = max(1, int(batch_or_concurrency))
  k_or_h = max(1, int(horizon_or_chunk))
  steps = max(1, int(recipe.denoising_steps))

  opt_text = " ".join(recipe.optimizations).lower()
  kv_reuse = any(tag in opt_text for tag in ("kv_cache", "prefix_cache", "dual_rate", "system1_system2", "ditcache"))

  if is_wm_track(recipe.track):
    # Action-conditioned World Model / Omni rollout: e.g. Cosmos-Predict2.5-2B canonical 256x256,
    # H_frames=16 at 24 FPS, 10 steps, CFG=1.0 -> 60,000 GFLOPs/rollout (3,750 GFLOPs/frame) incl. 3D-VAE decode.
    res_hw = float(getattr(recipe, "resolution_hw", 256) or 256)
    res_scale = (res_hw / 256.0) ** 2
    p_scale = float(model_params_b) / 2.0
    h_scale = float(k_or_h) / 16.0
    s_scale = float(steps) / 10.0
    cfg_mult = 2.0 if float(getattr(recipe, "guidance_scale", 1.0) or 1.0) > 1.05 else 1.0
    cache_factor = 0.94 if kv_reuse else 1.0
    vae_gflops = 7200.0 * p_scale * float(b) * h_scale * res_scale
    dit_gflops = 52800.0 * p_scale * float(b) * h_scale * s_scale * res_scale * cfg_mult * cache_factor
    flops_per_step_gflops = round(vae_gflops + dit_gflops, 3)
    flops_per_frame = round(flops_per_step_gflops / float(b * k_or_h), 3)
    flops_per_action = None
    weight_footprint_mb = float(model_params_b) * 1024.0 * bytes_per_param
    effective_weight_passes = max(1.0, float(steps) * (0.90 if kv_reuse else 1.0) + 0.25)
    activation_traffic_mb = float(b * k_or_h * steps) * 5.5 * res_scale
    bytes_moved_mb = round(weight_footprint_mb * effective_weight_passes + activation_traffic_mb, 3)
  else:
    # Real-Time VLA Policy: VLM prefix encoding over multi-view 224x224 patch tokens + Action Expert (Flow/DiT)
    res_hw = float(getattr(recipe, "resolution_hw", 224) or 224)
    res_scale = (res_hw / 224.0) ** 2
    cam_scale = max(1, int(getattr(recipe, "num_cameras", 2) or 2)) / 2.0
    vlm_params_b = float(model_params_b) * 0.85
    expert_params_b = max(0.15, float(model_params_b) * 0.15)
    vlm_reuse_factor = 0.82 if kv_reuse else 1.0
    vlm_flops_gflops = vlm_params_b * 520.0 * float(b) * res_scale * cam_scale * vlm_reuse_factor
    expert_flops_gflops = expert_params_b * 55.0 * float(b) * float(steps) * (1.0 + 0.02 * k_or_h)
    flops_per_step_gflops = round(vlm_flops_gflops + expert_flops_gflops, 3)
    flops_per_action = round(flops_per_step_gflops / float(b * k_or_h), 3)
    flops_per_frame = None

    vlm_weights_mb = vlm_params_b * 1024.0 * bytes_per_param * vlm_reuse_factor
    expert_passes = min(float(steps), 4.0) if "compile" in opt_text or "graph" in opt_text else float(steps) * 0.75
    expert_weights_mb = expert_params_b * 1024.0 * bytes_per_param * max(1.0, expert_passes)
    kv_and_act_mb = float(b) * (64.0 * cam_scale * res_scale + 4.0 * k_or_h)
    bytes_moved_mb = round(vlm_weights_mb + expert_weights_mb + kv_and_act_mb, 3)

  bytes_moved_bytes = int(round(bytes_moved_mb * 1024.0 * 1024.0))
  ai_flops_per_byte = round(
      (flops_per_step_gflops * 1e9) / max(float(bytes_moved_bytes), 1.0),
      3,
  )

  cap = envelope or get_performance_cap_envelope(recipe.resolved_perf_cap())
  ridge_flops_per_byte = (cap.max_fp16_tflops * 1000.0) / max(cap.max_mem_bw_gbps, 1e-6)
  ai_op = round(max(0.5, min(1.5, ai_flops_per_byte / max(ridge_flops_per_byte, 1e-6))), 4)
  peak_vram_gb = round(float(peak_vram_mb) / 1024.0, 4)

  return HardwareIndependentProfile(
      track=recipe.track,
      model_params_b=round(float(model_params_b), 3),
      precision=recipe.precision,
      bytes_per_param=bytes_per_param,
      batch_or_concurrency=b,
      horizon_or_chunk=k_or_h,
      denoising_steps=steps,
      flops_per_step_gflops=flops_per_step_gflops,
      flops_per_action_gflops=flops_per_action,
      flops_per_frame_gflops=flops_per_frame,
      bytes_moved_per_step_mb=bytes_moved_mb,
      bytes_moved_per_step_bytes=bytes_moved_bytes,
      arithmetic_intensity_flops_per_byte=ai_flops_per_byte,
      ai_op=ai_op,
      peak_vram_gb=peak_vram_gb,
  )


def generate_slice_clock_cap_commands(
    hardware_sku: str,
    envelope_id: str | PerformanceCapId,
) -> Dict[str, str]:
  """Generate single-device hardware-enforced slice, clock-lock, and HBM/VRAM-cap commands across NVIDIA, AMD, Google Cloud TPU, and Custom ASICs."""
  sku_spec = HARDWARE_SKU_SPECS.get(hardware_sku, HARDWARE_SKU_SPECS["H100_SXM5"])
  cap = get_performance_cap_envelope(envelope_id)

  dev_tflops = float(sku_spec["fp16_tflops"])
  dev_bw = float(sku_spec["mem_bw_gbps"])
  dev_vram_gb = float(sku_spec["vram_mb"]) / 1024.0
  vendor = str(sku_spec.get("vendor", "NVIDIA"))

  target_compute_frac = min(1.0, cap.max_fp16_tflops / max(dev_tflops, 1e-6))
  target_bw_frac = min(1.0, cap.max_mem_bw_gbps / max(dev_bw, 1e-6))
  target_vram_frac = min(1.0, cap.max_vram_gb / max(dev_vram_gb, 1e-6))

  mps_pct = max(5, min(100, int(round(min(target_compute_frac, target_bw_frac) * 100.0))))
  sm_clock_mhz = max(600, int(round(1800 * min(1.0, target_compute_frac * 1.15))))
  mem_clock_mhz = max(800, int(round(2600 * min(1.0, target_bw_frac))))

  return {
      "single_device_env": (
          "export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0"
          if vendor == "GOOGLE_TPU"
          else (
              "export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0"
              if vendor == "AMD_ROCM"
              else "export CUDA_VISIBLE_DEVICES=0"
          )
      ),
      "cuda_mps_env": f"export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={mps_pct}",
      "nvidia_smi_lock_clocks": f"sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc {sm_clock_mhz},{sm_clock_mhz} && sudo nvidia-smi -lmc {mem_clock_mhz},{mem_clock_mhz}",
      "rocm_smi_lock_clocks": f"sudo rocm-smi --setsclk 3 && sudo rocm-smi --setmclk 2",
      "tpu_single_chip_cap": f"export XLA_PYTHON_CLIENT_MEM_FRACTION={target_vram_frac:.4f}",
      "pytorch_vram_cap": f"torch.cuda.set_per_process_memory_fraction({target_vram_frac:.4f})",
      "eeb_cli_flags": (
          f"--perf-cap {cap.envelope_id} --perf-cap-mode hardware_enforced_slice_clock_cap "
          f"--mps-active-thread-pct {mps_pct} --clock-scale-factor 1.0 --vram-fraction-cap {target_vram_frac:.4f}"
      ),
  }


def evaluate_performance_cap_envelope(
    recipe: RecipeConfig,
    model_params_b: float,
    batch_or_concurrency: int,
    horizon_or_chunk: int,
    physical_latency_mean_ms: float,
    physical_latency_p50_ms: float,
    physical_latency_p99_ms: float,
    peak_vram_mb: float,
    perf_cap_override: Optional[str] = None,
    perf_cap_mode_override: Optional[str] = None,
    mps_active_thread_pct_override: Optional[float] = None,
    clock_scale_factor_override: Optional[float] = None,
    vram_fraction_cap_override: Optional[float] = None,
    ai_op_override: Optional[float] = None,
    device_power_watts: Optional[float] = None,
    system_power_watts: Optional[float] = None,
    same_chip_bf16_ref_throughput: Optional[float] = None,
    hourly_cost_usd: Optional[float] = None,
) -> PerformanceCapEvaluation:
  """Evaluate a recipe operating point against a single-device `(TOPS, HBM)` `PerformanceCapEnvelope`.

  Enforces `num_devices == 1` and supports three ways to satisfy `PerformanceCapEnvelope`:
    1. `native_qualifying_device`: Single-device SKU natively fits within `(max_fp16_tflops, max_int8_fp8_tops, max_mem_bw_gbps, max_vram_gb)`.
    2. `hardware_enforced_slice_clock_cap`: Hardware slice (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` / MIG / 1-chip TPU slice) +
       locked clocks (`clock_scale_factor`) + `vram_fraction_cap` bound effective specs within the envelope.
    3. `roofline_normalized_latency`: Physical wall-clock latency $L_{\\text{phys}}$ is recorded alongside
       Roofline-Normalized Effective Latency $L_{\\text{norm}}$, Raw Single-Device Throughput,
       Iso-TOPS Normalized Throughput (`throughput_per_100_tops`), and Iso-Bandwidth Efficiency (`throughput_per_100_gbps_bw`).
  """
  env_id = perf_cap_override or recipe.resolved_perf_cap()
  cap = get_performance_cap_envelope(env_id)
  sku_spec = HARDWARE_SKU_SPECS[recipe.hardware_sku]

  num_devices = int(getattr(recipe, "num_devices", 1))
  sku_devices = int(sku_spec.get("num_devices", 1))
  single_device_compliant = (num_devices == 1 and sku_devices == 1 and num_devices <= cap.max_num_devices)

  dev_tflops = float(sku_spec["fp16_tflops"])
  dev_tops = float(sku_spec.get("fp8_tflops", dev_tflops * 2.0))
  dev_bw = float(sku_spec["mem_bw_gbps"])
  dev_vram_gb = round(float(sku_spec["vram_mb"]) / 1024.0, 4)

  mps_pct = (
      float(mps_active_thread_pct_override)
      if mps_active_thread_pct_override is not None
      else (float(recipe.mps_active_thread_pct) if recipe.mps_active_thread_pct is not None else 100.0)
  )
  clk_scale = (
      float(clock_scale_factor_override)
      if clock_scale_factor_override is not None
      else (float(recipe.clock_scale_factor) if recipe.clock_scale_factor is not None else 1.0)
  )
  vram_frac = (
      float(vram_fraction_cap_override)
      if vram_fraction_cap_override is not None
      else (float(recipe.vram_fraction_cap) if recipe.vram_fraction_cap is not None else 1.0)
  )

  if not (0.0 < mps_pct <= 100.0):
    raise ValueError(f"mps_active_thread_pct must be in (0.0, 100.0], got {mps_pct}.")
  if not (0.0 < clk_scale <= 1.0):
    raise ValueError(f"clock_scale_factor must be in (0.0, 1.0], got {clk_scale}.")
  if not (0.0 < vram_frac <= 1.0):
    raise ValueError(f"vram_fraction_cap must be in (0.0, 1.0], got {vram_frac}.")

  hw_ind_profile = compute_hardware_independent_profile(
      recipe=recipe,
      model_params_b=model_params_b,
      batch_or_concurrency=batch_or_concurrency,
      horizon_or_chunk=horizon_or_chunk,
      peak_vram_mb=peak_vram_mb,
      envelope=cap,
  )
  ai_op = float(ai_op_override) if ai_op_override is not None else hw_ind_profile.ai_op

  eff_tflops = round(dev_tflops * (mps_pct / 100.0) * clk_scale, 3)
  eff_tops = round(dev_tops * (mps_pct / 100.0) * clk_scale, 3)
  eff_bw = round(dev_bw * (mps_pct / 100.0) * clk_scale, 3)
  eff_vram_cap_gb = round(dev_vram_gb * vram_frac, 4)
  peak_vram_gb = hw_ind_profile.peak_vram_gb

  native_compute_ok = (
      single_device_compliant
      and dev_tflops <= cap.max_fp16_tflops + 1e-6
      and dev_tops <= cap.max_int8_fp8_tops + 1e-6
      and dev_bw <= cap.max_mem_bw_gbps + 1e-6
      and dev_vram_gb <= cap.max_vram_gb + 1e-6
  )
  slice_compute_ok = (
      single_device_compliant
      and eff_tflops <= cap.max_fp16_tflops + 1e-6
      and eff_tops <= cap.max_int8_fp8_tops + 1e-6
      and eff_bw <= cap.max_mem_bw_gbps + 1e-6
      and eff_vram_cap_gb <= cap.max_vram_gb + 1e-6
  )
  vram_budget_compliant = (
      peak_vram_gb <= cap.max_vram_gb + 1e-6
      and peak_vram_gb <= eff_vram_cap_gb + 1e-6
  )

  requested_mode = perf_cap_mode_override or recipe.perf_cap_mode
  satisfied_natively = False
  satisfied_via_slice = False
  satisfied_via_roofline = False
  violations: List[str] = []

  if not single_device_compliant:
    violations.append(
        f"Single-device violation: num_devices={num_devices} (sku_devices={sku_devices}) exceeds "
        f"{cap.envelope_id} max_num_devices={cap.max_num_devices}. Only single-device (1 chip / 1 package) submissions are permitted."
    )

  if not vram_budget_compliant:
    violations.append(
        f"Active peak_vram_gb={peak_vram_gb:.2f} GB exceeds {cap.envelope_id} budget "
        f"(max_vram_gb={cap.max_vram_gb:.1f} GB, effective_vram_cap_gb={eff_vram_cap_gb:.2f} GB). "
        "Apply W4/FP8 quantization or set --vram-fraction-cap."
    )

  if requested_mode == PerfCapSatisfactionMode.NATIVE_QUALIFYING_DEVICE.value:
    if native_compute_ok and vram_budget_compliant:
      satisfied_natively = True
      resolved_mode = PerfCapSatisfactionMode.NATIVE_QUALIFYING_DEVICE.value
    else:
      resolved_mode = PerfCapSatisfactionMode.NATIVE_QUALIFYING_DEVICE.value
      violations.append(
          f"Hardware SKU {recipe.hardware_sku} ({dev_tops:.1f} TOPS / {dev_tflops:.1f} TFLOPS, {dev_bw:.1f} GB/s, {dev_vram_gb:.1f} GB) "
          f"exceeds native {cap.envelope_id} ({cap.tier_alias}) limits ({cap.max_int8_fp8_tops:.1f} TOPS / {cap.max_fp16_tflops:.1f} TFLOPS, "
          f"{cap.max_mem_bw_gbps:.1f} GB/s, {cap.max_vram_gb:.1f} GB). "
          "Use --perf-cap-mode hardware_enforced_slice_clock_cap or roofline_normalized_latency."
      )
  elif requested_mode == PerfCapSatisfactionMode.HARDWARE_ENFORCED_SLICE_CLOCK_CAP.value:
    if slice_compute_ok and vram_budget_compliant:
      satisfied_via_slice = True
      resolved_mode = PerfCapSatisfactionMode.HARDWARE_ENFORCED_SLICE_CLOCK_CAP.value
    else:
      resolved_mode = PerfCapSatisfactionMode.HARDWARE_ENFORCED_SLICE_CLOCK_CAP.value
      violations.append(
          f"Effective slice/clock specs ({eff_tops:.1f} TOPS / {eff_tflops:.1f} TFLOPS, {eff_bw:.1f} GB/s, {eff_vram_cap_gb:.1f} GB) "
          f"exceed {cap.envelope_id} ({cap.tier_alias}) limits ({cap.max_int8_fp8_tops:.1f} TOPS / {cap.max_fp16_tflops:.1f} TFLOPS, "
          f"{cap.max_mem_bw_gbps:.1f} GB/s, {cap.max_vram_gb:.1f} GB). "
          "Lower --mps-active-thread-pct, --clock-scale-factor, or --vram-fraction-cap."
      )
  elif requested_mode == PerfCapSatisfactionMode.ROOFLINE_NORMALIZED_LATENCY.value:
    resolved_mode = PerfCapSatisfactionMode.ROOFLINE_NORMALIZED_LATENCY.value
    satisfied_via_roofline = single_device_compliant and vram_budget_compliant
  else:
    if native_compute_ok and mps_pct == 100.0 and clk_scale == 1.0:
      satisfied_natively = vram_budget_compliant
      resolved_mode = PerfCapSatisfactionMode.NATIVE_QUALIFYING_DEVICE.value
    elif slice_compute_ok and (mps_pct < 100.0 or clk_scale < 1.0 or vram_frac < 1.0):
      satisfied_via_slice = vram_budget_compliant
      resolved_mode = PerfCapSatisfactionMode.HARDWARE_ENFORCED_SLICE_CLOCK_CAP.value
    else:
      satisfied_via_roofline = single_device_compliant and vram_budget_compliant
      resolved_mode = PerfCapSatisfactionMode.ROOFLINE_NORMALIZED_LATENCY.value

  norm_mean_ms, norm_factor = compute_roofline_normalized_latency(
      physical_latency_ms=physical_latency_mean_ms,
      tflops_eff=eff_tflops,
      tflops_cap=cap.max_fp16_tflops,
      bw_eff_gbps=eff_bw,
      bw_cap_gbps=cap.max_mem_bw_gbps,
      ai_op=ai_op,
  )
  norm_p50_ms, _ = compute_roofline_normalized_latency(
      physical_latency_ms=physical_latency_p50_ms,
      tflops_eff=eff_tflops,
      tflops_cap=cap.max_fp16_tflops,
      bw_eff_gbps=eff_bw,
      bw_cap_gbps=cap.max_mem_bw_gbps,
      ai_op=ai_op,
  )
  norm_p99_ms, _ = compute_roofline_normalized_latency(
      physical_latency_ms=physical_latency_p99_ms,
      tflops_eff=eff_tflops,
      tflops_cap=cap.max_fp16_tflops,
      bw_eff_gbps=eff_bw,
      bw_cap_gbps=cap.max_mem_bw_gbps,
      ai_op=ai_op,
  )
  norm_hz = round(1000.0 / max(norm_p99_ms, 1e-6), 4)

  units_per_call = (
      max(1, int(batch_or_concurrency))
      if is_wm_track(recipe.track)
      else max(1, int(batch_or_concurrency) * int(horizon_or_chunk))
  )
  raw_throughput = round(
      float(units_per_call) / max(float(physical_latency_mean_ms) / 1000.0, 1e-6),
      4,
  )
  iso_tops_tput, iso_bw_tput = compute_iso_tops_and_bw_efficiency(
      raw_single_device_throughput=raw_throughput,
      effective_bf16_tflops=eff_tflops,
      effective_mem_bw_gbps=eff_bw,
  )
  sa_norm = compute_semianalysis_normalizations(
      raw_throughput=raw_throughput,
      hardware_sku=recipe.hardware_sku,
      device_power_watts=device_power_watts,
      system_power_watts=system_power_watts,
      effective_bf16_tflops=eff_tflops,
      effective_mem_bw_gbps=eff_bw,
      same_chip_bf16_ref_throughput=same_chip_bf16_ref_throughput,
      hourly_cost_usd=hourly_cost_usd,
  )

  passed = len(violations) == 0 and (
      satisfied_natively or satisfied_via_slice or satisfied_via_roofline
  )
  reason = (
      f"Performance cap {cap.envelope_id} ({cap.tier_alias}) satisfied via '{resolved_mode}' "
      f"(num_devices={num_devices}, L_phys_p99={physical_latency_p99_ms:.2f}ms, L_norm_p99={norm_p99_ms:.2f}ms, "
      f"norm_factor={norm_factor:.3f}x, peak_vram={peak_vram_gb:.2f}/{cap.max_vram_gb:.1f}GB, "
      f"iso_tops={iso_tops_tput:.2f}/100TOPS, perf_per_watt={sa_norm['power_efficiency_per_watt']:.4f}/W, "
      f"perf_per_dollar_hr={sa_norm['throughput_per_dollar_hr']:.1f}/$)."
      if passed
      else f"Performance cap {cap.envelope_id} ({cap.tier_alias}) check failed: " + "; ".join(violations)
  )

  return PerformanceCapEvaluation(
      envelope_id=cap.envelope_id,
      tier_alias=cap.tier_alias,
      envelope=cap.to_dict(),
      hardware_sku=recipe.hardware_sku,
      num_devices=num_devices,
      single_device_compliant=single_device_compliant,
      satisfaction_mode=resolved_mode,
      satisfied_natively=satisfied_natively,
      satisfied_via_slice_or_clock_cap=satisfied_via_slice,
      satisfied_via_roofline_normalization=satisfied_via_roofline,
      vram_budget_compliant=vram_budget_compliant,
      passed=passed,
      device_fp16_tflops=dev_tflops,
      device_fp8_int8_tops=dev_tops,
      device_mem_bw_gbps=dev_bw,
      device_total_vram_gb=dev_vram_gb,
      effective_tflops=eff_tflops,
      effective_fp8_int8_tops=eff_tops,
      effective_mem_bw_gbps=eff_bw,
      effective_vram_cap_gb=eff_vram_cap_gb,
      peak_vram_gb=peak_vram_gb,
      mps_active_thread_pct=round(mps_pct, 2),
      clock_scale_factor=round(clk_scale, 4),
      vram_fraction_cap=round(vram_frac, 4),
      ai_op=ai_op,
      roofline_normalization_factor=norm_factor,
      physical_latency_mean_ms=round(float(physical_latency_mean_ms), 3),
      physical_latency_p50_ms=round(float(physical_latency_p50_ms), 3),
      physical_latency_p99_ms=round(float(physical_latency_p99_ms), 3),
      normalized_latency_mean_ms=norm_mean_ms,
      normalized_latency_p50_ms=norm_p50_ms,
      normalized_latency_p99_ms=norm_p99_ms,
      normalized_reactivity_hz=norm_hz,
      raw_single_device_throughput=raw_throughput,
      raw_per_chip_throughput=sa_norm["raw_per_chip_throughput"],
      power_efficiency_per_watt=sa_norm["power_efficiency_per_watt"],
      system_power_watts=sa_norm["system_power_watts"],
      throughput_per_100_tops=iso_tops_tput,
      throughput_per_tbps_bw=sa_norm["throughput_per_tbps_bw"],
      throughput_per_100_gbps_bw=iso_bw_tput,
      same_chip_speedup_vs_bf16_ref=sa_norm["same_chip_speedup_vs_bf16_ref"],
      throughput_per_dollar_hr=sa_norm["throughput_per_dollar_hr"],
      throughput_per_1k_usd=sa_norm["throughput_per_1k_usd"],
      tco_efficiency_per_dollar=sa_norm["tco_efficiency_per_dollar"],
      iso_compute_and_bw_efficiency=sa_norm["iso_compute_and_bw_efficiency"],
      semianalysis_normalizations=sa_norm,
      hardware_independent_profile=hw_ind_profile.to_dict(),
      enforcement_commands=generate_slice_clock_cap_commands(recipe.hardware_sku, cap.envelope_id),
      violations=violations,
      reason=reason,
  )


def run_eeb_measure(
    recipe_path: str | Path,
    output_path: Optional[str | Path] = None,
    include_stage2: bool = True,
    warmup_calls: int = 30,
    timed_calls: int = 320,
    num_runs: int = 3,
    perf_cap: Optional[str] = None,
    perf_cap_mode: Optional[str] = None,
    mps_active_thread_pct: Optional[float] = None,
    clock_scale_factor: Optional[float] = None,
    vram_fraction_cap: Optional[float] = None,
    live_runner: bool = False,
    server_url: Optional[str] = None,
) -> Dict[str, Any]:
  """Run the full `eeb measure` protocol with Device-Agnostic Performance Cap evaluation."""
  from embodied_bench.stage1_golden_replay import run_stage1_from_recipe_path

  return run_stage1_from_recipe_path(
      recipe_path=recipe_path,
      output_path=output_path,
      include_stage2=include_stage2,
      warmup_calls=warmup_calls,
      timed_calls=timed_calls,
      num_runs=num_runs,
      perf_cap=perf_cap,
      perf_cap_mode=perf_cap_mode,
      mps_active_thread_pct=mps_active_thread_pct,
      clock_scale_factor=clock_scale_factor,
      vram_fraction_cap=vram_fraction_cap,
      live_runner=live_runner,
      server_url=server_url,
  )
