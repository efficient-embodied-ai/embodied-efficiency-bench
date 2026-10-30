"""AI Agent Turn-0 Autonomous Optimization Kickoff & Study Workspace Bootstrapper (`agent_kickoff.py`) for EmbodiedX (`embodied-efficiency-bench` / CoRL 2026).

Provides:
  - `detect_host_accelerator(...)`: Auto-detects the host hardware environment (`nvidia-smi`, `tegrastats`,
    `rocm-smi`, `tpu-info`, or CPU development fallback) and maps to a qualifying `CAP_SINGLE_CHIP` SKU
    (`TPU_V6E_1_32GB`, `RTX_5090_32GB`, `JETSON_AGX_THOR_64GB`, `AMD_MI210_64GB`, etc.).
  - `parse_kickoff_prompt(...)`: Extracts target canonical model, hardware SKU, division (`CLOSED` vs `OPEN`),
    requested optimizations, and device-count hints from a natural-language kickoff prompt.
  - `run_agent_kickoff(...)`: Validates `CAP_SINGLE_CHIP` (`num_devices == 1`, <= 64 GB, <= 1,000 BF16 TFLOPS)
    and `CLOSED`/`OPEN` division compliance, atomically writes `<study_dir>/STUDY_PLAN.json` (`fcntl.flock` via
    `atomic_read_modify_write_json` / `atomic_write_json`), generates `<study_dir>/EXPANDED_AGENT_PROMPT.md`
    and `<study_dir>/baseline_recipe.yaml`, and optionally runs a live HTTP `EmbodiedBenchmarkServer` +
    `EmbodiedLoadGenClient` baseline sweep (`--run-baseline`) to record `<study_dir>/baseline_anchor.json`.
  - `update_study_milestone(...)`: Crash-safe, concurrent-safe state machine updater for the 5-milestone DAG
    (`M1_HARDWARE_AND_DATA_PREFLIGHT` -> `M5_SIGNED_RECEIPT_AND_PR_PACKAGE`) enforcing Gates 1 & 2.
  - `main(argv)`: CLI entrypoint (`python3 -m embodied_bench.agent_kickoff` / `eeb agent-kickoff`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
  import yaml
except ImportError:  # pragma: no cover
  yaml = None  # type: ignore[assignment]

from embodied_bench.baselines import (
    create_baseline_runner,
    resolve_canonical_model_id,
)
from embodied_bench.harness import (
    EmbodiedBenchmarkServer,
    EmbodiedLoadGenClient,
    SingleChipHardwareVerifier,
    audit_command_line_for_multi_chip_bypass,
    parse_rocm_smi_json,
    verify_single_chip_envelope,
)
from embodied_bench.measure import compute_semianalysis_normalizations
from embodied_bench.pareto import extract_semianalysis_4way_pareto_frontiers
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    CLOSED_DIVISION_FORBIDDEN_KEYWORDS,
    CompressionTier,
    DivisionType,
    ExecutionStrategy,
    HARDWARE_SKU_SPECS,
    MODERN_REFERENCE_CHECKPOINTS,
    PERFORMANCE_CAP_ENVELOPES,
    PerformanceCapId,
    RecipeConfig,
    RunStatus,
    TrackType,
    atomic_read_modify_write_json,
    atomic_write_json,
    cleanup_orphan_temp_files,
    is_wm_track,
    now_pacific_iso,
    resolve_perf_cap_id,
    validate_division_compliance,
    validate_status_and_reason,
)
from embodied_bench.telemetry import (
    parse_tegrastats_line,
    parse_tpu_info_output,
)


# Ensure common H100_SXM_80GB alias is present in HARDWARE_SKU_SPECS lookup table
if "H100_SXM_80GB" not in HARDWARE_SKU_SPECS and "H100_SXM5_80GB" in HARDWARE_SKU_SPECS:
  HARDWARE_SKU_SPECS["H100_SXM_80GB"] = dict(HARDWARE_SKU_SPECS["H100_SXM5_80GB"])


MILESTONE_IDS: Tuple[str, ...] = (
    "M1_HARDWARE_AND_DATA_PREFLIGHT",
    "M2_LIVE_BASELINE_ANCHOR",
    "M3_CLOSED_DIVISION_KERNEL_KV_OPT",
    "M4_OPEN_DIVISION_STEP_CACHE_OPT",
    "M5_SIGNED_RECEIPT_AND_PR_PACKAGE",
)


DIVISION_OPTIMIZATION_SKELETONS: Dict[str, List[Dict[str, Any]]] = {
    DivisionType.CLOSED.value: [
        {
            "skeleton_id": "closed_vlm_kv_prefix_cache",
            "division": DivisionType.CLOSED.value,
            "name": "Stateful 24-Token VLM Language-Instruction KV-Prefix Cache Reuse",
            "description": (
                "Cache and reuse static language-instruction and system-prompt KV states across "
                "all steps t >= 1 within the same session_id without altering model weights or FLOPs."
            ),
            "target_tracks": [TrackType.VLA_TRACK.value, TrackType.WM_TRACK.value],
            "preserves_frozen_weights": True,
            "preserves_denoising_steps": True,
        },
        {
            "skeleton_id": "closed_zero_copy_posix_shm_ipc",
            "division": DivisionType.CLOSED.value,
            "name": "Zero-Copy POSIX Shared-Memory (/dev/shm) Observation Transport",
            "description": (
                "Map dual-camera RGB and proprioception tensors directly from SharedMemoryTensorRing "
                "via zero-copy buffer views instead of Base64 JSON serialization."
            ),
            "target_tracks": [TrackType.VLA_TRACK.value, TrackType.WM_TRACK.value],
            "preserves_frozen_weights": True,
            "preserves_denoising_steps": True,
        },
        {
            "skeleton_id": "closed_compiler_graph_fusion",
            "division": DivisionType.CLOSED.value,
            "name": "Whole-Graph Compiler Fusion (TorchTPU + torch.compile / Pallas / TensorRT / HIP Graph)",
            "description": (
                "Compile static bucket shapes for the VLM backbone and DiT/Flow action head with "
                "fused RMSNorm/SwiGLU/SDPA kernels while keeping numerical parity within Gate G1."
            ),
            "target_tracks": [TrackType.VLA_TRACK.value, TrackType.WM_TRACK.value],
            "preserves_frozen_weights": True,
            "preserves_denoising_steps": True,
        },
        {
            "skeleton_id": "closed_async_host_dispatch_zero_sync",
            "division": DivisionType.CLOSED.value,
            "name": "Zero Host-Device Sync in Timed Inference Loop",
            "description": (
                "Eliminate all blocking .item(), .cpu(), .numpy(), torch.cuda.synchronize(), and "
                "torch.tpu.synchronize() calls inside the timed forward pass until final action chunk emission."
            ),
            "target_tracks": [TrackType.VLA_TRACK.value, TrackType.WM_TRACK.value],
            "preserves_frozen_weights": True,
            "preserves_denoising_steps": True,
        },
    ],
    DivisionType.OPEN.value: [
        {
            "skeleton_id": "open_dit_flow_step_reduction",
            "division": DivisionType.OPEN.value,
            "name": "DiT / Flow-Matching ODE Step Reduction (16 -> 2..4 or 10 -> 1..4 Steps)",
            "description": (
                "Reduce flow-matching / diffusion denoising steps via consistency distillation or "
                "higher-order ODE solvers while maintaining Gate G1/G2 closed-loop quality."
            ),
            "target_tracks": [TrackType.VLA_TRACK.value, TrackType.WM_TRACK.value],
            "preserves_frozen_weights": False,
            "preserves_denoising_steps": False,
        },
        {
            "skeleton_id": "open_rtc_inpainting_execution",
            "division": DivisionType.OPEN.value,
            "name": "Real-Time Chunking (RTC_INPAINTING) Boundary-Conditioned Execution",
            "description": (
                "Overlap inference latency L with smooth action-chunk prefix inpainting in LIBERO-RT "
                "to eliminate synchronous-hold control stalls at 20 Hz."
            ),
            "target_tracks": [TrackType.VLA_TRACK.value],
            "preserves_frozen_weights": True,
            "preserves_denoising_steps": False,
        },
        {
            "skeleton_id": "open_temporal_dit_feature_cache",
            "division": DivisionType.OPEN.value,
            "name": "Temporal DiT Residual Feature Caching (DiT-Cache / TeaCache)",
            "description": (
                "Exploit >= 0.94 frame-to-frame correlation across consecutive observations o_t -> o_{t+1} "
                "while dynamically invalidating on Gate G4 l_inf=1e-3 perturbations."
            ),
            "target_tracks": [TrackType.WM_TRACK.value],
            "preserves_frozen_weights": True,
            "preserves_denoising_steps": False,
        },
        {
            "skeleton_id": "open_mixed_precision_fp8_w4_quantization",
            "division": DivisionType.OPEN.value,
            "name": "Salient-Channel-Protected FP8 / INT4-AWQ / W4A8 MoE Quantization",
            "description": (
                "Quantize VLM, DiT, or 30B/3B MoE weights to FP8 E4M3, INT4-AWQ, or W4A8 so all 4 "
                "canonical models fit comfortably inside a single 16-64 GB chip."
            ),
            "target_tracks": [TrackType.VLA_TRACK.value, TrackType.WM_TRACK.value],
            "preserves_frozen_weights": False,
            "preserves_denoising_steps": True,
        },
    ],
}


ANTI_CHEAT_INVARIANTS: Dict[str, Any] = {
    "cap_single_chip": {
        "envelope_id": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "tier_alias": PerformanceCapId.SINGLE_CHIP_UNIFIED.value,
        "required_num_devices": 1,
        "max_vram_gb": 64.0,
        "max_bf16_tflops": 1000.0,
        "max_fp8_int8_tops": 2000.0,
        "max_mem_bw_gbps": 2000.0,
        "prohibited_env_vars": [
            "CUDA_VISIBLE_DEVICES=0,1",
            "HIP_VISIBLE_DEVICES=0,1",
            "TPU_VISIBLE_DEVICES=0,1,2,3",
            "TPU_CHIPS_PER_HOST_BOUNDS=2,2,1",
        ],
        "prohibited_flags": [
            "--tensor-parallel-size > 1",
            "--pipeline-parallel-size > 1",
            "--data-parallel-size > 1",
            "--num-devices > 1",
        ],
    },
    "co_located_quality_gates_g1_g4": {
        "G1_teacher_forced_parity": {
            "vla_min_action_cosine_sim": 0.995,
            "vla_max_action_nrmse": 0.025,
            "wm_min_latent_action_cosine_sim": 0.990,
            "wm_max_psnr_drop_db": 1.5,
        },
        "G2_mcnemar_non_inferiority": {
            "closed_min_delta_pts": -2.0,
            "closed_min_sr_ratio": 0.98,
            "open_min_delta_pts": -5.0,
            "mcnemar_alpha": 0.05,
        },
        "G3_same_execution_sha256": {
            "require_identical_execution_id": True,
            "require_bit_identical_replay_audit": True,
        },
        "G4_perturbed_input_anti_memoization": {
            "perturbation_linf_eps": 1e-3,
            "min_l2_output_shift": 1e-5,
            "min_responsive_sample_fraction": 0.99,
        },
    },
    "hot_loop_ast_rules": {
        "zero_device_sync_in_hot_loop": True,
        "prohibited_hot_loop_calls": [
            "tensor.item()",
            "tensor.cpu()",
            "tensor.numpy()",
            "torch.cuda.synchronize()",
        ],
        "prohibited_timer_monkeypatching": [
            "time.perf_counter",
            "time.perf_counter_ns",
            "time.monotonic",
            "time.monotonic_ns",
        ],
        "prohibited_raw_bytes_memoization": [
            "obs.tobytes()",
            "hashlib.sha256(obs)",
            "pickle.dumps(obs)",
        ],
    },
}


HARDWARE_SKU_ALIASES: Dict[str, str] = {
    "tpu_v6e_1_32gb": "TPU_V6E_1_32GB",
    "tpu_v6e_1": "TPU_V6E_1_32GB",
    "tpu-v6e-1": "TPU_V6E_1_32GB",
    "v6e-1": "TPU_V6E_1_32GB",
    "v6e_1": "TPU_V6E_1_32GB",
    "tpu v6e-1": "TPU_V6E_1_32GB",
    "tpu v6e": "TPU_V6E_1_32GB",
    "trillium": "TPU_V6E_1_32GB",
    "tpu_v5e_1_16gb": "TPU_V5E_1_16GB",
    "tpu_v5e_1": "TPU_V5E_1_16GB",
    "tpu-v5e-1": "TPU_V5E_1_16GB",
    "v5e-1": "TPU_V5E_1_16GB",
    "tpu v5e-1": "TPU_V5E_1_16GB",
    "rtx_5090_32gb": "RTX_5090_32GB",
    "rtx_5090": "RTX_5090_32GB",
    "rtx-5090": "RTX_5090_32GB",
    "rtx 5090": "RTX_5090_32GB",
    "5090": "RTX_5090_32GB",
    "rtx_4090_24gb": "RTX_4090_24GB",
    "rtx_4090": "RTX_4090_24GB",
    "rtx-4090": "RTX_4090_24GB",
    "rtx 4090": "RTX_4090_24GB",
    "4090": "RTX_4090_24GB",
    "l40s_48gb": "L40S_48GB",
    "l40s": "L40S_48GB",
    "l4_24gb": "L4_24GB",
    "l4 24gb": "L4_24GB",
    "nvidia l4": "L4_24GB",
    "l4": "L4_24GB",
    "rtx_4060_ti_16gb": "RTX_4060_TI_16GB",
    "rtx 4060 ti": "RTX_4060_TI_16GB",
    "4060 ti": "RTX_4060_TI_16GB",
    "jetson_agx_thor_64gb": "JETSON_AGX_THOR_64GB",
    "jetson agx thor": "JETSON_AGX_THOR_64GB",
    "jetson thor": "JETSON_AGX_THOR_64GB",
    "agx thor": "JETSON_AGX_THOR_64GB",
    "thor 64gb": "JETSON_AGX_THOR_64GB",
    "jetson_orin_nx_16gb": "JETSON_ORIN_NX_16GB",
    "jetson orin nx": "JETSON_ORIN_NX_16GB",
    "orin nx": "JETSON_ORIN_NX_16GB",
    "jetson_orin_agx_64gb": "JETSON_ORIN_AGX_64GB",
    "jetson agx orin": "JETSON_ORIN_AGX_64GB",
    "agx orin": "JETSON_ORIN_AGX_64GB",
    "amd_mi210_64gb": "AMD_MI210_64GB",
    "amd_mi210": "AMD_MI210_64GB",
    "instinct mi210": "AMD_MI210_64GB",
    "mi210": "AMD_MI210_64GB",
    "amd_w7900_48gb": "AMD_W7900_48GB",
    "w7900": "AMD_W7900_48GB",
    "amd_rx_7900_xtx_24gb": "AMD_RX_7900_XTX_24GB",
    "7900 xtx": "AMD_RX_7900_XTX_24GB",
    "rx 7900 xtx": "AMD_RX_7900_XTX_24GB",
    "amd_mi250_1gcd_64gb": "AMD_MI250_1GCD_64GB",
    "mi250": "AMD_MI250_1GCD_64GB",
    "apple_m4_pro_16gb": "APPLE_M4_PRO_16GB",
    "m4 pro": "APPLE_M4_PRO_16GB",
    "custom_asic_single_device": "CUSTOM_ASIC_SINGLE_DEVICE",
    "h100_sxm_80gb": "H100_SXM_80GB",
    "h100_sxm5_80gb": "H100_SXM5_80GB",
    "h100_sxm5": "H100_SXM5",
    "h100 80gb": "H100_SXM_80GB",
    "h100": "H100_SXM_80GB",
    "b200_sxm6_192gb": "B200_SXM6_192GB",
    "b200_sxm6": "B200_SXM6",
    "b200": "B200_SXM6",
    "tpu_v7x_1": "TPU_V7X_1",
    "tpu_v7x_1chip": "TPU_V7X_1CHIP",
    "amd_mi300x_192gb": "AMD_MI300X_192GB",
    "amd_mi300x_80gb_cap": "AMD_MI300X_80GB_CAP",
    "mi300x": "AMD_MI300X_192GB",
}


def resolve_hardware_sku(raw_sku: str) -> str:
  """Resolve a hardware SKU name or alias to a canonical key in `HARDWARE_SKU_SPECS`."""
  if not raw_sku or not str(raw_sku).strip():
    raise ValueError(
        "Empty hardware_sku provided. Specify a valid SKU from HARDWARE_SKU_SPECS "
        "(e.g. 'TPU_V6E_1_32GB', 'RTX_5090_32GB', 'JETSON_AGX_THOR_64GB', 'AMD_MI210_64GB')."
    )
  cleaned = str(raw_sku).strip()
  if cleaned in HARDWARE_SKU_SPECS:
    return cleaned
  upper = cleaned.upper()
  if upper in HARDWARE_SKU_SPECS:
    return upper
  lower = cleaned.lower()
  if lower in HARDWARE_SKU_ALIASES:
    return HARDWARE_SKU_ALIASES[lower]
  valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
  raise ValueError(
      f"Unknown hardware_sku={raw_sku!r}. Supported SKUs in HARDWARE_SKU_SPECS: [{valid_skus}]."
  )


@dataclass
class HostAcceleratorInfo:
  """Detected host accelerator metadata mapped to `HARDWARE_SKU_SPECS` and `CAP_SINGLE_CHIP`."""

  hardware_sku: str
  vendor: str
  device_name: str
  detected_device_count: int
  vram_gb: float
  bf16_tflops: float
  fp8_tops: float
  mem_bw_gbps: float
  detection_source: str
  qualifies_cap_single_chip: bool
  envelope_reason: str

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  def __getitem__(self, key: str) -> Any:
    return self.to_dict()[key]

  def get(self, key: str, default: Any = None) -> Any:
    return self.to_dict().get(key, default)


def _map_gpu_name_to_sku(name: str, vram_mb: float = 0.0) -> str:
  """Map a raw GPU/TPU/SoC product string to a canonical `HARDWARE_SKU_SPECS` key."""
  lower = name.lower()
  if "b200" in lower:
    return "B200_SXM6"
  if "h100" in lower:
    return "H100_SXM_80GB"
  if "5090" in lower:
    return "RTX_5090_32GB"
  if "l40s" in lower:
    return "L40S_48GB"
  if re.search(r"\bl4\b", lower):
    return "L4_24GB"
  if "4090" in lower:
    return "RTX_4090_24GB"
  if "4060" in lower:
    return "RTX_4060_TI_16GB"
  if "thor" in lower:
    return "JETSON_AGX_THOR_64GB" if vram_mb <= 70000 else "JETSON_THOR_128GB"
  if "orin" in lower:
    return "JETSON_ORIN_NX_16GB" if 0 < vram_mb <= 20000 else "JETSON_ORIN_AGX_64GB"
  if "mi300x" in lower:
    return "AMD_MI300X_192GB"
  if "mi300a" in lower:
    return "AMD_MI300A_128GB"
  if "mi250" in lower:
    return "AMD_MI250_1GCD_64GB"
  if "mi210" in lower:
    return "AMD_MI210_64GB"
  if "w7900" in lower:
    return "AMD_W7900_48GB"
  if "7900" in lower:
    return "AMD_RX_7900_XTX_24GB"
  if "v7x" in lower or "ironwood" in lower:
    return "TPU_V7X_1"
  if "v5e" in lower:
    return "TPU_V5E_1_16GB"
  if "v6e" in lower or "trillium" in lower or "tpu" in lower:
    return "TPU_V6E_1_32GB"
  if vram_mb >= 70000:
    return "H100_SXM_80GB"
  if vram_mb >= 40000:
    return "L40S_48GB"
  if vram_mb >= 28000:
    return "RTX_5090_32GB"
  if vram_mb >= 20000:
    return "RTX_4090_24GB"
  return "TPU_V6E_1_32GB"


def _build_host_accelerator_info(
    sku: str,
    detected_device_count: int,
    detection_source: str,
    perf_cap: str = PerformanceCapId.CAP_SINGLE_CHIP.value,
) -> HostAcceleratorInfo:
  resolved_sku = resolve_hardware_sku(sku)
  spec = HARDWARE_SKU_SPECS[resolved_sku]
  env_check = verify_single_chip_envelope(
      hardware_sku=resolved_sku,
      detected_device_count=detected_device_count,
      perf_cap=perf_cap,
  )
  return HostAcceleratorInfo(
      hardware_sku=resolved_sku,
      vendor=str(spec.get("vendor", "UNKNOWN")),
      device_name=str(spec.get("display_name", resolved_sku)),
      detected_device_count=int(detected_device_count),
      vram_gb=env_check.native_vram_gb,
      bf16_tflops=env_check.fp16_tflops,
      fp8_tops=env_check.fp8_tops,
      mem_bw_gbps=env_check.mem_bw_gbps,
      detection_source=detection_source,
      qualifies_cap_single_chip=env_check.passed,
      envelope_reason=env_check.reason,
  )


def detect_host_accelerator(
    probe_live: bool = True,
    fallback_sku: str = "TPU_V6E_1_32GB",
    perf_cap: str = PerformanceCapId.CAP_SINGLE_CHIP.value,
    nvidia_smi_output: Optional[str] = None,
    tegrastats_output: Optional[str] = None,
    rocm_smi_output: Optional[str] = None,
    tpu_info_output: Optional[str] = None,
    override_device_count: Optional[int] = None,
) -> HostAcceleratorInfo:
  """Auto-detect the host's accelerator environment (`nvidia-smi`, `tegrastats`, `rocm-smi`, `tpu-info`, or CPU fallback).

  Checks in order:
    1. Injected CLI outputs (`nvidia_smi_output`, `tegrastats_output`, `rocm_smi_output`, `tpu_info_output`) for deterministic tests.
    2. `EEB_HARDWARE_SKU` environment variable override.
    3. Live `tpu-info` / `/dev/accel*` / `/dev/vfio/*` for Google Cloud TPU hosts.
    4. Live `tegrastats` / `/etc/nv_tegra_release` for NVIDIA Jetson Thor / Orin edge SoCs.
    5. Live `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` for NVIDIA GPUs.
    6. Live `rocm-smi --showpower --showmeminfo vram --json` for AMD Instinct / Radeon GPUs.
    7. CPU development fallback mapped to `fallback_sku` (`TPU_V6E_1_32GB` by default).
  """
  # 1. Injected mock outputs for deterministic testing
  if tpu_info_output is not None and tpu_info_output.strip():
    parsed_tpu = parse_tpu_info_output(tpu_info_output)
    total_mb = float(parsed_tpu.get("total_vram_mb", 32768.0))
    sku = "TPU_V5E_1_16GB" if total_mb <= 18000.0 else (
        "TPU_V7X_1" if total_mb >= 100000.0 else "TPU_V6E_1_32GB"
    )
    dev_cnt = int(override_device_count if override_device_count is not None else 1)
    return _build_host_accelerator_info(sku, dev_cnt, "tpu_info_injected", perf_cap=perf_cap)

  if tegrastats_output is not None and tegrastats_output.strip():
    lines = [ln.strip() for ln in tegrastats_output.splitlines() if ln.strip()]
    parsed_teg = parse_tegrastats_line(lines[-1])
    sku = "JETSON_ORIN_NX_16GB" if parsed_teg.ram_total_mb <= 20000.0 else "JETSON_AGX_THOR_64GB"
    dev_cnt = int(override_device_count if override_device_count is not None else 1)
    return _build_host_accelerator_info(sku, dev_cnt, "tegrastats_injected", perf_cap=perf_cap)

  if nvidia_smi_output is not None and nvidia_smi_output.strip():
    lines = [ln.strip() for ln in nvidia_smi_output.splitlines() if ln.strip()]
    parts = [p.strip() for p in lines[0].split(",")]
    gpu_name = parts[0]
    vram_mb = float(parts[1]) if len(parts) > 1 and parts[1].replace(".", "", 1).isdigit() else 32768.0
    sku = _map_gpu_name_to_sku(gpu_name, vram_mb)
    dev_cnt = int(override_device_count if override_device_count is not None else len(lines))
    return _build_host_accelerator_info(sku, dev_cnt, "nvidia_smi_injected", perf_cap=perf_cap)

  if rocm_smi_output is not None and rocm_smi_output.strip():
    parsed_rocm = parse_rocm_smi_json(rocm_smi_output)
    sku = _map_gpu_name_to_sku(
        str(parsed_rocm.get("device_name", "AMD Instinct MI210")),
        float(parsed_rocm.get("total_vram_mb", 65536.0)),
    )
    dev_cnt = int(
        override_device_count
        if override_device_count is not None
        else parsed_rocm.get("detected_device_count", 1)
    )
    return _build_host_accelerator_info(sku, dev_cnt, "rocm_smi_injected", perf_cap=perf_cap)

  # 2. Environment variable override
  env_sku = os.environ.get("EEB_HARDWARE_SKU", "").strip()
  if env_sku:
    dev_cnt = int(override_device_count if override_device_count is not None else 1)
    return _build_host_accelerator_info(env_sku, dev_cnt, "env_EEB_HARDWARE_SKU", perf_cap=perf_cap)

  if probe_live:
    # 3. Probe Google Cloud TPU (TorchTPU `torch.tpu` first, then `tpu-info` / `/dev/accel*`)
    try:
      import importlib
      import importlib.util

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
            total_mb = 32768.0
            dev_name = ""
            name_fn = getattr(tpu_ns, "get_device_name", None)
            if callable(name_fn):
              try:
                dev_name = str(name_fn(0) or "")
              except Exception:
                dev_name = ""
            props_fn = getattr(tpu_ns, "get_device_properties", None)
            if callable(props_fn):
              props = props_fn(0)
              total_mem = getattr(props, "total_memory", None)
              if total_mem is None and isinstance(props, dict):
                total_mem = props.get("total_memory")
              if total_mem:
                total_mb = float(total_mem) / (1024.0 * 1024.0)
            if "v5e" in dev_name.lower() or total_mb <= 18000.0:
              sku = "TPU_V5E_1_16GB"
            elif total_mb >= 100000.0:
              sku = "TPU_V7X_1"
            else:
              sku = "TPU_V6E_1_32GB"
            dev_cnt = int(override_device_count if override_device_count is not None else cnt)
            return _build_host_accelerator_info(sku, dev_cnt, "torchtpu_live", perf_cap=perf_cap)
    except Exception:
      pass

    try:
      proc_tpu = subprocess.run(
          ["tpu-info"],
          capture_output=True,
          text=True,
          timeout=1.2,
          check=True,
      )
      if proc_tpu.stdout.strip():
        parsed_tpu = parse_tpu_info_output(proc_tpu.stdout)
        total_mb = float(parsed_tpu.get("total_vram_mb", 32768.0))
        sku = "TPU_V5E_1_16GB" if total_mb <= 18000.0 else (
            "TPU_V7X_1" if total_mb >= 100000.0 else "TPU_V6E_1_32GB"
        )
        accel_devs = glob.glob("/dev/accel*")
        dev_cnt = int(
            override_device_count
            if override_device_count is not None
            else (len(accel_devs) if accel_devs else 1)
        )
        return _build_host_accelerator_info(sku, dev_cnt, "tpu_info_live", perf_cap=perf_cap)
    except Exception:
      pass

    # 4. Probe NVIDIA Jetson tegrastats
    if Path("/etc/nv_tegra_release").exists():
      try:
        proc_teg = subprocess.run(
            ["tegrastats", "--interval", "50"],
            capture_output=True,
            text=True,
            timeout=0.25,
            check=False,
        )
        lines = [ln.strip() for ln in proc_teg.stdout.splitlines() if ln.strip()]
        if lines:
          parsed_teg = parse_tegrastats_line(lines[-1])
          sku = (
              "JETSON_ORIN_NX_16GB"
              if parsed_teg.ram_total_mb <= 20000.0
              else "JETSON_AGX_THOR_64GB"
          )
          dev_cnt = int(override_device_count if override_device_count is not None else 1)
          return _build_host_accelerator_info(sku, dev_cnt, "tegrastats_live", perf_cap=perf_cap)
      except Exception:
        pass

    # 5. Probe NVIDIA nvidia-smi
    try:
      proc_nv = subprocess.run(
          [
              "nvidia-smi",
              "--query-gpu=name,memory.total",
              "--format=csv,noheader,nounits",
          ],
          capture_output=True,
          text=True,
          timeout=1.2,
          check=True,
      )
      lines = [ln.strip() for ln in proc_nv.stdout.splitlines() if ln.strip()]
      if lines:
        parts = [p.strip() for p in lines[0].split(",")]
        gpu_name = parts[0]
        vram_mb = (
            float(parts[1])
            if len(parts) > 1 and parts[1].replace(".", "", 1).isdigit()
            else 32768.0
        )
        sku = _map_gpu_name_to_sku(gpu_name, vram_mb)
        dev_cnt = int(override_device_count if override_device_count is not None else len(lines))
        return _build_host_accelerator_info(sku, dev_cnt, "nvidia_smi_live", perf_cap=perf_cap)
    except Exception:
      pass

    # 6. Probe AMD rocm-smi
    try:
      proc_rocm = subprocess.run(
          ["rocm-smi", "--showpower", "--showmeminfo", "vram", "--json"],
          capture_output=True,
          text=True,
          timeout=1.2,
          check=True,
      )
      if proc_rocm.stdout.strip():
        parsed_rocm = parse_rocm_smi_json(proc_rocm.stdout)
        sku = _map_gpu_name_to_sku(
            str(parsed_rocm.get("device_name", "AMD Instinct MI210")),
            float(parsed_rocm.get("total_vram_mb", 65536.0)),
        )
        dev_cnt = int(
            override_device_count
            if override_device_count is not None
            else parsed_rocm.get("detected_device_count", 1)
        )
        return _build_host_accelerator_info(sku, dev_cnt, "rocm_smi_live", perf_cap=perf_cap)
    except Exception:
      pass

  # 7. CPU development fallback
  dev_cnt = int(override_device_count if override_device_count is not None else 1)
  return _build_host_accelerator_info(fallback_sku, dev_cnt, "cpu_dev_fallback", perf_cap=perf_cap)


_PROMPT_MODEL_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:pi0\.5|pi05|pi0_5|openpi|pi0-3b|pi0\.5-3b|pi05-3b)\b", re.IGNORECASE), "pi05-3b"),
    (re.compile(r"\b(?:gr00t[-\s_]?n1\.7|gr00t[-\s_]?1\.7|gr00t-n1\.7-3b|gr00t)\b", re.IGNORECASE), "gr00t-n1.7-3b"),
    (
        re.compile(
            r"\b(?:cosmos[-\s_]?predict[-\s_]?2\.5(?:[-\s_]?2b)?|cosmos[-\s_]?predict|cosmos[-\s_]?2\.5)\b",
            re.IGNORECASE,
        ),
        "cosmos-predict2.5-2b",
    ),
    (re.compile(r"\b(?:cosmos3[-\s_]?edge|cosmos[-\s_]?3|cosmos3)\b", re.IGNORECASE), "cosmos3-edge"),
    (re.compile(r"\bcosmos\b", re.IGNORECASE), "cosmos-predict2.5-2b"),
    (re.compile(r"\b(?:qwen3[-\s_]?omni|qwen3-omni-30b-a3b|qwen[-\s_]?omni)\b", re.IGNORECASE), "qwen3-omni-30b-a3b"),
    (re.compile(r"\b(?:openvla[-\s_]?oft|openvla-oft-7b)\b", re.IGNORECASE), "openvla-oft-7b"),
    (re.compile(r"\b(?:smolvla|smolvla-450m)\b", re.IGNORECASE), "smolvla-450m"),
)

_PROMPT_SKU_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:h100[_\-\s]?sxm5?[_\-\s]?80gb|h100[_\-\s]?80gb|h100)\b", re.IGNORECASE), "H100_SXM_80GB"),
    (re.compile(r"\b(?:b200[_\-\s]?sxm6?|b200)\b", re.IGNORECASE), "B200_SXM6"),
    (re.compile(r"\b(?:tpu[_\-\s]*v7x|v7x)\b", re.IGNORECASE), "TPU_V7X_1"),
    (re.compile(r"\b(?:tpu[_\-\s]*v6e[_\-\s]*1[_\-\s]*32gb|tpu[_\-\s]*v6e[_\-\s]*1|v6e[_\-\s]*1|tpu[_\-\s]*v6e|trillium)\b", re.IGNORECASE), "TPU_V6E_1_32GB"),
    (re.compile(r"\b(?:tpu[_\-\s]*v5e[_\-\s]*1[_\-\s]*16gb|tpu[_\-\s]*v5e[_\-\s]*1|v5e[_\-\s]*1|tpu[_\-\s]*v5e)\b", re.IGNORECASE), "TPU_V5E_1_16GB"),
    (re.compile(r"\b(?:rtx[_\-\s]*5090[_\-\s]*32gb|rtx[_\-\s]*5090|5090)\b", re.IGNORECASE), "RTX_5090_32GB"),
    (re.compile(r"\b(?:rtx[_\-\s]*4090[_\-\s]*24gb|rtx[_\-\s]*4090|4090)\b", re.IGNORECASE), "RTX_4090_24GB"),
    (re.compile(r"\b(?:l40s[_\-\s]*48gb|l40s)\b", re.IGNORECASE), "L40S_48GB"),
    (re.compile(r"\b(?:nvidia[_\-\s]*l4(?:[_\-\s]*24gb)?|l4[_\-\s]*24gb|l4)\b", re.IGNORECASE), "L4_24GB"),
    (re.compile(r"\b(?:rtx[_\-\s]*4060[_\-\s]*ti|4060[_\-\s]*ti)\b", re.IGNORECASE), "RTX_4060_TI_16GB"),
    (re.compile(r"\b(?:jetson[_\-\s]*agx[_\-\s]*thor[_\-\s]*64gb|jetson[_\-\s]*agx[_\-\s]*thor|jetson[_\-\s]*thor|agx[_\-\s]*thor|thor)\b", re.IGNORECASE), "JETSON_AGX_THOR_64GB"),
    (re.compile(r"\b(?:jetson[_\-\s]*orin[_\-\s]*nx|orin[_\-\s]*nx)\b", re.IGNORECASE), "JETSON_ORIN_NX_16GB"),
    (re.compile(r"\b(?:jetson[_\-\s]*agx[_\-\s]*orin|jetson[_\-\s]*orin[_\-\s]*agx|agx[_\-\s]*orin)\b", re.IGNORECASE), "JETSON_ORIN_AGX_64GB"),
    (re.compile(r"\b(?:amd[_\-\s]*mi300x|mi300x)\b", re.IGNORECASE), "AMD_MI300X_192GB"),
    (re.compile(r"\b(?:amd[_\-\s]*mi210[_\-\s]*64gb|amd[_\-\s]*mi210|instinct[_\-\s]*mi210|mi210)\b", re.IGNORECASE), "AMD_MI210_64GB"),
    (re.compile(r"\b(?:amd[_\-\s]*w7900|w7900)\b", re.IGNORECASE), "AMD_W7900_48GB"),
    (re.compile(r"\b(?:rx[_\-\s]*7900[_\-\s]*xtx|7900[_\-\s]*xtx)\b", re.IGNORECASE), "AMD_RX_7900_XTX_24GB"),
)


def parse_kickoff_prompt(prompt: Optional[str]) -> Dict[str, Any]:
  """Parse a natural-language kickoff prompt into structured model, SKU, division, and optimization hints."""
  if not prompt or not str(prompt).strip():
    return {
        "model_id": None,
        "hardware_sku": None,
        "division": None,
        "num_devices": None,
        "optimizations": [],
    }
  text = str(prompt).strip()
  detected_model: Optional[str] = None
  for pattern, canonical_model in _PROMPT_MODEL_PATTERNS:
    if pattern.search(text):
      detected_model = canonical_model
      break

  detected_sku: Optional[str] = None
  for pattern, canonical_sku in _PROMPT_SKU_PATTERNS:
    if pattern.search(text):
      detected_sku = canonical_sku
      break

  detected_division: Optional[str] = None
  if re.search(r"\bclosed(?:\s+division)?\b", text, re.IGNORECASE):
    detected_division = DivisionType.CLOSED.value
  elif re.search(r"\bopen(?:\s+division)?\b", text, re.IGNORECASE):
    detected_division = DivisionType.OPEN.value

  detected_num_devices: Optional[int] = None
  multi_dev_match = re.search(
      r"\b(?:(\d+)\s*x\s*(?:gpu|tpu|chip|device)|num[_\-]devices\s*[=:]?\s*(\d+)|(\d+)\s+(?:gpus|tpus|chips|devices))\b",
      text,
      re.IGNORECASE,
  )
  if multi_dev_match:
    for grp in multi_dev_match.groups():
      if grp is not None and grp.isdigit():
        detected_num_devices = int(grp)
        break

  opts: List[str] = []
  for kw in CLOSED_DIVISION_FORBIDDEN_KEYWORDS:
    if re.search(rf"\b{re.escape(kw)}[a-z0-9_]*\b", text, re.IGNORECASE):
      opts.append(kw)
  if re.search(r"\bstep[_\-\s]+distillation\b", text, re.IGNORECASE) and "step_distillation" not in opts:
    opts.append("step_distillation")
  if re.search(r"\brtc[_\-\s]*inpainting\b", text, re.IGNORECASE):
    opts.append("rtc_inpainting")
  if re.search(r"\bkv[_\-\s]*prefix[_\-\s]*cache\b", text, re.IGNORECASE):
    opts.append("vlm_prefix_kv_cache_reuse")

  return {
      "model_id": detected_model,
      "hardware_sku": detected_sku,
      "division": detected_division,
      "num_devices": detected_num_devices,
      "optimizations": opts,
  }


def _build_initial_milestones(
    model_id: str,
    hardware_sku: str,
    division: str,
    study_dir: Path,
    recipe_path: Path,
) -> Dict[str, Dict[str, Any]]:
  """Construct the 5-milestone DAG (`M1` -> `M5`) with copy-pasteable `eeb` CLI commands."""
  ts = now_pacific_iso()
  m4_initial_status = (
      RunStatus.SKIPPED.value
      if division == DivisionType.CLOSED.value
      else RunStatus.PENDING.value
  )
  m4_initial_reason = (
      "Skipped in CLOSED division: weights, denoising steps, and resolution are frozen; "
      "step reduction and feature caching belong to OPEN division."
      if division == DivisionType.CLOSED.value
      else None
  )

  policy_map = {
      "pi05-3b": "eeb.servers.pi05:Pi05Policy",
      "gr00t-n1.7-3b": "eeb.servers.gr00t:Gr00tN17Policy",
      "cosmos-predict2.5-2b": "eeb.servers.cosmos_predict25:CosmosPredict25Policy",
  }
  tpu_flag = hardware_sku.startswith("TPU_")
  policy_spec = policy_map.get(model_id, "eeb.servers.pi05:Pi05Policy")
  device_kw = '{"device": "tpu"}' if tpu_flag else '{"device": "cuda"}'
  native_hw = 256 if model_id in ("gr00t-n1.7-3b", "cosmos-predict2.5-2b") else 224
  eval_traces_dir = study_dir / "eval_traces"
  golden_npz = study_dir / "golden_ref.npz"
  champ_dir = study_dir / "runs" / "00_champion"
  cand_dir = study_dir / "runs" / "cand_01"

  return {
      "M1_HARDWARE_AND_DATA_PREFLIGHT": {
          "id": "M1_HARDWARE_AND_DATA_PREFLIGHT",
          "index": 1,
          "title": "Single-Chip Hardware Envelope, Telemetry & EmbodiedX-Traces-v1 Preflight",
          "depends_on": [],
          "status": RunStatus.PENDING.value,
          "reason": None,
          "deliverables": [
              str(study_dir / "STUDY_PLAN.json"),
              str(eval_traces_dir / "manifest.json"),
              str(golden_npz),
          ],
          "cli_commands": [
              f"eeb envelope check --sku {hardware_sku}",
              "eeb telemetry probe --seconds 2 --hz 10",
              "eeb data inspect data/embodiedx_traces_v1_mini",
              f"eeb data prepare-traces --out {eval_traces_dir} --image-hw {native_hw}x{native_hw} --split public",
              f"eeb data perturb {eval_traces_dir} --split public",
          ],
          "updated_at": ts,
      },
      "M2_LIVE_BASELINE_ANCHOR": {
          "id": "M2_LIVE_BASELINE_ANCHOR",
          "index": 2,
          "title": "Live UDS gRPC Server + Frozen BF16 Golden Generation + Baseline Sweep Anchor",
          "depends_on": ["M1_HARDWARE_AND_DATA_PREFLIGHT"],
          "status": RunStatus.PENDING.value,
          "reason": None,
          "deliverables": [
              str(champ_dir / "report.json"),
              str(champ_dir / "receipt.json"),
          ],
          "cli_commands": [
              f"eeb serve --policy {policy_spec} --uds /tmp/eeb_policy.sock --policy-kwargs '{device_kw}' &",
              f"eeb data generate-golden {eval_traces_dir} --endpoint unix:///tmp/eeb_policy.sock --out {golden_npz} --split public",
              (
                  f"eeb measure --recipe {recipe_path} --endpoint unix:///tmp/eeb_policy.sock "
                  f"--traces {eval_traces_dir} --golden {golden_npz} --strict --out {champ_dir}"
              ),
          ],
          "updated_at": ts,
      },
      "M3_CLOSED_DIVISION_KERNEL_KV_OPT": {
          "id": "M3_CLOSED_DIVISION_KERNEL_KV_OPT",
          "index": 3,
          "title": "CLOSED Division Systems Optimization & Mechanical Comparison (`eeb compare`)",
          "depends_on": ["M2_LIVE_BASELINE_ANCHOR"],
          "status": RunStatus.PENDING.value,
          "reason": None,
          "deliverables": [
              str(cand_dir / "report.json"),
              str(study_dir / "compare_cand_01.json"),
          ],
          "cli_commands": [
              (
                  f"eeb measure --recipe {recipe_path} --endpoint unix:///tmp/eeb_policy.sock "
                  f"--traces {eval_traces_dir} --golden {golden_npz} --strict --out {cand_dir}"
              ),
              (
                  f"eeb compare --baseline {champ_dir} --candidate {cand_dir} "
                  f"--out {study_dir / 'compare_cand_01.json'}"
              ),
          ],
          "updated_at": ts,
      },
      "M4_OPEN_DIVISION_STEP_CACHE_OPT": {
          "id": "M4_OPEN_DIVISION_STEP_CACHE_OPT",
          "index": 4,
          "title": "OPEN Division Algorithmic Co-Design (Step Distillation, RTC_INPAINTING, DiT-Cache, FP8/W4)",
          "depends_on": ["M3_CLOSED_DIVISION_KERNEL_KV_OPT"],
          "status": m4_initial_status,
          "reason": m4_initial_reason,
          "deliverables": [
              str(study_dir / "runs" / "m4_open" / "report.json"),
              str(study_dir / "libero_rt_eval.json"),
          ],
          "cli_commands": [
              (
                  f"eeb eval-libero-rt --recipe {recipe_path} "
                  f"--sweep-report {cand_dir / 'report.json'} --output {study_dir / 'libero_rt_eval.json'}"
              ),
              (
                  f"eeb measure --recipe {recipe_path} --endpoint unix:///tmp/eeb_policy.sock "
                  f"--traces {eval_traces_dir} --golden {golden_npz} --strict "
                  f"--out {study_dir / 'runs' / 'm4_open'}"
              ),
          ],
          "updated_at": ts,
      },
      "M5_SIGNED_RECEIPT_AND_PR_PACKAGE": {
          "id": "M5_SIGNED_RECEIPT_AND_PR_PACKAGE",
          "index": 5,
          "title": "Cryptographic SHA-256 Receipt Verification, AST Anti-Cheat Audit & PR Leaderboard Package",
          "depends_on": ["M4_OPEN_DIVISION_STEP_CACHE_OPT"],
          "status": RunStatus.PENDING.value,
          "reason": None,
          "deliverables": [
              str(cand_dir / "report.json"),
              str(cand_dir / "receipt.json"),
              str(study_dir / "validation_report.json"),
          ],
          "cli_commands": [
              f"eeb verify-receipt --receipt {cand_dir}",
              (
                  f"eeb validate-submission --submission {cand_dir} "
                  f"--output {study_dir / 'validation_report.json'}"
              ),
          ],
          "updated_at": ts,
      },
  }


def render_expanded_agent_prompt(study_plan: Dict[str, Any]) -> str:
  """Render `<study_dir>/EXPANDED_AGENT_PROMPT.md` startup contract for AI coding agents."""
  study_id = study_plan["study_id"]
  study_dir = study_plan["study_dir"]
  model_id = study_plan["model_id"]
  hf_repo = study_plan["hf_repo"]
  track = study_plan["track"]
  division = study_plan["division"]
  hardware_sku = study_plan["hardware_sku"]
  perf_cap = study_plan["perf_cap"]
  recipe_path = study_plan["recipe_path"]
  milestones = study_plan["milestones"]
  active_skeletons = study_plan["allowed_optimization_skeletons"]["active_division_skeletons"]

  skeleton_lines = "\n".join(
      f"- **`{sk['skeleton_id']}`** ({sk['division']}): *{sk['name']}* : {sk['description']}"
      for sk in active_skeletons
  )

  milestone_sections: List[str] = []
  for mid in MILESTONE_IDS:
    m = milestones[mid]
    cmds = "\n".join(m["cli_commands"])
    deps = ", ".join(m["depends_on"]) if m["depends_on"] else "None"
    milestone_sections.append(
        f"### {m['index']}. `{mid}` ({m['title']})\n"
        f"- **Status**: `{m['status']}` | **Depends On**: `{deps}`\n"
        f"```bash\n{cmds}\n```"
    )
  milestones_md = "\n\n".join(milestone_sections)

  return f"""# EmbodiedX (`embodied-efficiency-bench` v1.0) Agent & Harness Startup Contract

## 1. Study Identity & Target Envelope
- **Study ID**: `{study_id}`
- **Study Workspace**: `{study_dir}`
- **Canonical Model**: `{model_id}` (`{hf_repo}`, Track: `{track}`)
- **Division**: `{division}`
- **Target Hardware SKU**: `{hardware_sku}` (`num_devices == 1`)
- **Performance Cap Envelope**: `{perf_cap}` (`<= 1,000 BF16 TFLOPS`, `<= 2,000 FP8/INT8 TOPS`, `<= 64 GB HBM/VRAM`, `<= 2,000 GB/s`)
- **Generated Recipe**: `{recipe_path}`

## 2. SemiAnalysis Normalization Views
Every candidate run is evaluated against the real-time reactivity X-axis (`p99 Chunk Rate (Chunks/s) [1000 / L_p99, ms]` for `VLA_TRACK` or `p99 Prediction Rate (Predictions/s) [1000 / L_p99, ms]` for `WM_TRACK`, scored from pooled steady-state $p_{{99}}(C)$ across active slots with a mandatory `<= 15%` per-slot completion fairness fence):
1. **`raw_per_chip_throughput`**: Raw executed `actions/sec/chip` (Track 1 VLA) or generated `FPS/chip` (Track 2 World Model) on `1` physical device.
2. **`power_efficiency_per_watt`**: Throughput per active device Watt (`actions/J` or `frames/J`, telemetry-only, zero synthetic host overhead; official `Per Watt` award eligibility requires live hardware telemetry such as `nvmlDeviceGetTotalEnergyConsumption()` / `tegrastats`).
3. **`throughput_per_100_tops` / `throughput_per_tbps_bw`**: Iso-compute (`Dense BF16 TFLOPS / 100`) and iso-bandwidth (`TB/s`) utilization.
4. **`tco_efficiency_per_dollar` (`throughput_per_dollar_hr`)**: Throughput per `$/hr` reference TCO rate (`(Output/s)/($/hr)`).

## 3. Allowed Optimization Skeletons (`division={division}`)
{skeleton_lines}

## 4. Immutable Harness Invariants (`G1`-`G4` & `CAP_SINGLE_CHIP`)
- **`CAP_SINGLE_CHIP` (`num_devices == 1`)**: Multi-chip execution (`CUDA_VISIBLE_DEVICES=0,1`, `--tensor-parallel-size > 1`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`) and >64 GB datacenter SKUs (`H100_SXM_80GB`, `B200_SXM6`) are strictly rejected. Single-chip Cloud TPU runs MUST export `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0`. **Host-CPU Neural Offload Ban (`D1`)**: In `CLOSED` division, 100% of neural network forward-pass operators inside timed `act()` / `rollout()` calls must execute on the single target accelerator (`WM_TRACK` `stream_chunk_frames > 0` and timed host-CPU neural offload are prohibited; pre-caching frozen episode instruction text embeddings on the accelerator during episode `reset()` is permitted).
- **`Gate G1` (Golden Replay Parity)**: VLA action Cosine Similarity `>= 0.995` & NRMSE `<= 0.025`; World Model frame/latent Cosine Similarity `>= 0.990`, PSNR drop `<= 1.5 dB` (`PSNR >= 28.0 dB` vs frozen BF16 golden), & temporal L1 ratio `<= 1.15`.
- **`Gate G2` (LIBERO-RT / Policy Ranking)**: Closed-loop `LIBERO-RT` degradation `<= 2.0 pts` (`SR >= 0.98 * SR_ref` in `CLOSED`) or Track 2 Kendall's tau `>= 0.80`.
- **`Gate G3` (Co-Located Same-Execution Guarantee)**: Timing and quality metrics share the exact same `execution_id` and SHA-256 signed receipt (`>= 320` timed VLA calls / `>= 200` timed WM calls x 3 runs, native `>= 224x224` wire RGB resolution).
- **`Gate G4` (Live Perturbation Anti-Memoization)**: Injects BF16-safe state/action (`1e-2` > 1 BF16 ULP) AND vision-only (`lsb_delta=4` across 25% of coherent `4x4` macro-blocks) observation perturbations inside `eeb measure` and verifies `l2_output_shift >= 1e-5`.
- **Immutable Evaluator Rule**: External harnesses and coding agents MUST treat `eeb/loadgen/`, `eeb/telemetry/`, `eeb/envelope/`, and `embodied_bench/attestation.py` as read-only, MUST preserve accelerator sync barriers (`synchronize_torch_device()`, `torch.cuda.synchronize()`, `torch.tpu.synchronize()`, `jax.block_until_ready()`), and MUST pass `--golden` on every scored run.

## 5. Five-Step Agent & Harness Execution Flow (`M1` -> `M5`)

{milestones_md}
"""


def update_study_milestone(
    study_plan_path: str | Path,
    milestone_id: str,
    status: str | RunStatus,
    reason: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    error_code: Optional[str] = None,
    enforce_transitions: bool = True,
) -> Dict[str, Any]:
  """Atomically update a milestone's state inside `STUDY_PLAN.json` under `fcntl.flock` (Gates 1 & 2).

  - Gate 1: Cleans up orphan `.tmp_embodied_*` files, recovers from corrupt JSON, prevents stale
    optional field carryover (`reason`, `metrics`, `notes`, `error_code` reset to `None` when not passed),
    and serializes concurrent thread/subprocess updates via `atomic_read_modify_write_json`.
  - Gate 2: Enforces `VALID_STATE_TRANSITIONS` and requires a non-empty `reason` whenever `status` is
    `REJECTED`, `BLOCKED`, `FAILED`, `SKIPPED`, or `INVALID`.
  """
  if milestone_id not in MILESTONE_IDS:
    raise ValueError(
        f"Unknown milestone_id={milestone_id!r}. Valid milestone IDs are: {list(MILESTONE_IDS)}."
    )

  # Pre-validate status and mandatory reason before taking lock
  target_status = validate_status_and_reason(status, reason=reason)
  plan_path = Path(study_plan_path)

  def _default_plan() -> Dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "study_id": plan_path.parent.name or "recovered_study",
        "status": RunStatus.PENDING.value,
        "reason": None,
        "milestone_ids": list(MILESTONE_IDS),
        "milestones": {
            mid: {
                "id": mid,
                "index": idx + 1,
                "title": mid,
                "depends_on": [MILESTONE_IDS[idx - 1]] if idx > 0 else [],
                "status": RunStatus.PENDING.value,
                "reason": None,
                "metrics": None,
                "notes": None,
                "error_code": None,
                "updated_at": now_pacific_iso(),
            }
            for idx, mid in enumerate(MILESTONE_IDS)
        },
        "updated_at": now_pacific_iso(),
    }

  def _modifier(current: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(current)
    milestones = dict(state.get("milestones", {}))
    if milestone_id not in milestones or not isinstance(milestones[milestone_id], dict):
      milestones[milestone_id] = {
          "id": milestone_id,
          "index": MILESTONE_IDS.index(milestone_id) + 1,
          "title": milestone_id,
          "depends_on": (
              [MILESTONE_IDS[MILESTONE_IDS.index(milestone_id) - 1]]
              if MILESTONE_IDS.index(milestone_id) > 0
              else []
          ),
          "status": RunStatus.PENDING.value,
          "reason": None,
      }

    m_entry = dict(milestones[milestone_id])
    prev_status_str = m_entry.get("status")
    if enforce_transitions and prev_status_str:
      # Allow idempotent repeat-call of the same status without transition error
      if str(prev_status_str) != target_status.value:
        validate_status_and_reason(
            target_status,
            reason=reason,
            previous_status=str(prev_status_str),
        )

    # Gate 1 Repeat-Call Hygiene: explicitly overwrite optional fields so stale values never persist
    m_entry["status"] = target_status.value
    m_entry["reason"] = str(reason).strip() if reason and str(reason).strip() else None
    m_entry["metrics"] = dict(metrics) if metrics is not None else None
    m_entry["notes"] = str(notes) if notes is not None else None
    m_entry["error_code"] = str(error_code) if error_code is not None else None
    m_entry["updated_at"] = now_pacific_iso()

    milestones[milestone_id] = m_entry
    state["milestones"] = milestones
    state["milestone_list"] = [milestones[m] for m in MILESTONE_IDS if m in milestones]
    state["updated_at"] = now_pacific_iso()
    return state

  return atomic_read_modify_write_json(
      target_path=plan_path,
      modifier_fn=_modifier,
      default_factory=_default_plan,
  )


def _execute_live_baseline_anchor(
    model_id: str,
    hardware_sku: str,
    perf_cap: str,
    precision: str,
    denoising_steps: int,
    output_path: Path,
    concurrencies: Sequence[int] = (1, 2),
    steps_per_stream: int = 4,
    warmup_steps: int = 1,
) -> Dict[str, Any]:
  """Run a live HTTP `EmbodiedBenchmarkServer` + `EmbodiedLoadGenClient` sweep and persist `baseline_anchor.json`."""
  runner = create_baseline_runner(
      model_id=model_id,
      precision=precision,
      denoising_steps=denoising_steps,
      weights_mode="mini_real",
      hardware_sku=hardware_sku,
  )
  hw_ver = verify_single_chip_envelope(
      hardware_sku=hardware_sku,
      detected_device_count=1,
      perf_cap=perf_cap,
  )
  c_list = [max(1, int(c)) for c in concurrencies] if concurrencies else [1, 2]

  with SingleChipHardwareVerifier(
      hardware_sku=hardware_sku,
      perf_cap=perf_cap,
      poll_interval_s=0.05,
  ) as live_verifier:
    with EmbodiedBenchmarkServer(
        runner=runner,
        host="127.0.0.1",
        port=0,
        server_id=f"eeb-kickoff-baseline-{runner.canonical_model_id}",
    ) as srv:
      client = EmbodiedLoadGenClient(base_url=srv.base_url)
      sweep_map = client.run_multi_stream_sweep(
          concurrency_levels=c_list,
          steps_per_stream=max(1, int(steps_per_stream)),
          warmup_steps=max(0, int(warmup_steps)),
          track=runner.track,
          chunk_size=runner.config.action_chunk_size,
          horizon_frames=runner.config.horizon_frames,
      )

  tel_summary = live_verifier.summarize()
  wm_flag = is_wm_track(runner.track)
  x_key = "hz_plan" if wm_flag else "hz_react"
  y_key = "fps" if wm_flag else "actions_per_sec"
  lat_key = "rollout_latency_p99_ms" if wm_flag else "ttfa_p99_ms"

  operating_points: List[Dict[str, Any]] = []
  for c_val, s_res in sorted(sweep_map.items()):
    raw_tput = float(s_res.fps if wm_flag else s_res.actions_per_sec or 0.0)
    sa = compute_semianalysis_normalizations(
        raw_throughput=raw_tput,
        hardware_sku=hardware_sku,
        device_power_watts=tel_summary.mean_power_watts,
        system_power_watts=None,
        same_chip_bf16_ref_throughput=raw_tput,
    )
    operating_points.append({
        "concurrency": int(c_val),
        x_key: float(s_res.hz_plan if wm_flag else s_res.hz_react or 0.0),
        y_key: raw_tput,
        lat_key: float(s_res.p99_ms),
        "mean_latency_ms": float(s_res.mean_ms),
        "p50_latency_ms": float(s_res.p50_ms),
        "p99_latency_ms": float(s_res.p99_ms),
        "cache_hit_ratio": float(s_res.cache_hit_ratio),
        "gate_a_passed": True,
        "hardware_sku": hardware_sku,
        "raw_per_chip_throughput": sa["raw_per_chip_throughput"],
        "power_efficiency_per_watt": sa["power_efficiency_per_watt"],
        "throughput_per_100_tops": sa["throughput_per_100_tops"],
        "throughput_per_tbps_bw": sa["throughput_per_tbps_bw"],
        "throughput_per_dollar_hr": sa["throughput_per_dollar_hr"],
        "semianalysis_normalizations": sa,
    })

  frontiers_4way = extract_semianalysis_4way_pareto_frontiers(
      operating_points=operating_points,
      x_key=x_key,
      require_gate_a=True,
  )
  peak_tput = max((float(p["raw_per_chip_throughput"]) for p in operating_points), default=0.0)
  peak_hz = max((float(p[x_key]) for p in operating_points), default=0.0)
  min_p99 = min((float(p["p99_latency_ms"]) for p in operating_points), default=0.0)

  payload: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "generated_at": now_pacific_iso(),
      "model_id": runner.canonical_model_id,
      "hf_repo": runner.hf_repo,
      "track": runner.track,
      "precision": precision,
      "denoising_steps": int(runner.config.denoising_steps),
      "hardware_sku": hardware_sku,
      "perf_cap": perf_cap,
      "status": RunStatus.CERTIFIED.value if hw_ver.passed else RunStatus.REJECTED.value,
      "reason": None if hw_ver.passed else hw_ver.reason,
      "x_metric_name": x_key,
      "y_metric_name": y_key,
      "latency_metric_name": lat_key,
      "peak_raw_per_chip_throughput": round(peak_tput, 4),
      "peak_reactivity_hz": round(peak_hz, 4),
      "best_p99_latency_ms": round(min_p99, 4),
      "single_chip_verification": hw_ver.to_dict(),
      "hardware_telemetry": tel_summary.to_dict(),
      "operating_points": operating_points,
      "semianalysis_4way_pareto_frontiers": frontiers_4way,
  }
  atomic_write_json(output_path, payload)
  return payload


def run_agent_kickoff(
    prompt: Optional[str] = None,
    model_id: Optional[str] = None,
    hardware_sku: Optional[str] = None,
    division: Optional[str] = None,
    study_dir: Optional[str | Path] = None,
    perf_cap: str = PerformanceCapId.CAP_SINGLE_CHIP.value,
    num_devices: Optional[int] = None,
    precision: Optional[str] = None,
    denoising_steps: Optional[int] = None,
    optimizations: Optional[Sequence[str]] = None,
    weights_frozen: Optional[bool] = None,
    run_baseline: bool = False,
    baseline_concurrencies: Sequence[int] = (1, 2),
    baseline_steps: int = 4,
    baseline_warmup: int = 1,
    probe_live_hardware: bool = False,
    notes: Optional[str] = None,
    raise_on_violation: bool = False,
) -> Dict[str, Any]:
  """Bootstrap an autonomous single-chip optimization study workspace (`STUDY_PLAN.json` + `EXPANDED_AGENT_PROMPT.md`).

  Enforces:
    1. Gate 1 (Crash recovery from orphan `.tmp_embodied_*` & corrupt JSON, repeat-call optional field
       hygiene, and `fcntl.flock` concurrent safety).
    2. Gate 2 (State machine transitions and mandatory non-empty `reason` on `REJECTED` / `BLOCKED` / `FAILED`).
    3. Gate 4 (Preflight rejection of multi-chip `num_devices != 1`, >64GB SKUs like `H100_SXM_80GB` under
       `CAP_SINGLE_CHIP`, and forbidden `OPEN`-division levers like `step_distillation` in `CLOSED` division).
  """
  parsed_prompt = parse_kickoff_prompt(prompt)

  # 1. Resolve canonical model
  raw_model = model_id or parsed_prompt["model_id"] or "pi05-3b"
  canonical_model = resolve_canonical_model_id(raw_model)
  ckpt_spec = MODERN_REFERENCE_CHECKPOINTS[canonical_model]
  track = str(ckpt_spec["track"])
  hf_repo = str(ckpt_spec["hf_repo"])
  default_steps = int(ckpt_spec.get("default_denoising_steps", 4))

  # 2. Resolve division
  resolved_division = str(division or parsed_prompt["division"] or DivisionType.CLOSED.value).upper()
  if resolved_division not in (DivisionType.CLOSED.value, DivisionType.OPEN.value):
    raise ValueError(
        f"Invalid division={resolved_division!r}. Supported divisions are ['CLOSED', 'OPEN']."
    )

  # 3. Resolve device count & hardware SKU (auto-detecting host if not explicitly provided)
  resolved_perf_cap = resolve_perf_cap_id(perf_cap)
  effective_num_devices = (
      int(num_devices)
      if num_devices is not None
      else (
          int(parsed_prompt["num_devices"])
          if parsed_prompt["num_devices"] is not None
          else 1
      )
  )

  raw_sku = hardware_sku or parsed_prompt["hardware_sku"]
  if raw_sku is not None:
    resolved_sku = resolve_hardware_sku(raw_sku)
    host_accel = detect_host_accelerator(
        probe_live=False,
        fallback_sku=resolved_sku,
        perf_cap=resolved_perf_cap,
        override_device_count=effective_num_devices,
    )
  else:
    host_accel = detect_host_accelerator(
        probe_live=probe_live_hardware,
        fallback_sku="TPU_V6E_1_32GB",
        perf_cap=resolved_perf_cap,
        override_device_count=effective_num_devices,
    )
    resolved_sku = host_accel.hardware_sku

  # 4. Resolve study directory
  sku_slug = resolved_sku.lower()
  model_slug = canonical_model.replace(".", "").replace("-", "_")
  div_slug = resolved_division.lower()
  study_id = f"study_{model_slug}_{sku_slug}_{div_slug}"
  target_dir = (
      Path(study_dir)
      if study_dir is not None
      else Path("studies") / study_id
  )
  target_dir.mkdir(parents=True, exist_ok=True)
  cleanup_orphan_temp_files(target_dir)

  # 5. Resolve precision, denoising steps, and optimizations
  resolved_precision = (
      str(precision).lower()
      if precision is not None
      else ("bf16" if resolved_division == DivisionType.CLOSED.value else "fp8")
  )
  resolved_steps = (
      int(denoising_steps)
      if denoising_steps is not None
      else (
          default_steps
          if resolved_division == DivisionType.CLOSED.value
          else min(4, default_steps)
      )
  )
  resolved_weights_frozen = (
      bool(weights_frozen)
      if weights_frozen is not None
      else (resolved_division == DivisionType.CLOSED.value)
  )

  merged_opts: List[str] = []
  if optimizations:
    merged_opts.extend(str(o) for o in optimizations)
  for p_opt in parsed_prompt["optimizations"]:
    if p_opt not in merged_opts:
      merged_opts.append(p_opt)
  if not merged_opts:
    if resolved_division == DivisionType.CLOSED.value:
      merged_opts = ["vlm_prefix_kv_cache_reuse", "zero_copy_posix_shm_ipc"]
    else:
      merged_opts = list(ckpt_spec.get("rewarded_optimizations", ["vlm_prefix_kv_cache_reuse"]))

  # 6. Preflight compliance checks (CAP_SINGLE_CHIP + Command Audit + Division Compliance)
  preflight_violations: List[str] = []

  if prompt:
    cmd_audit = audit_command_line_for_multi_chip_bypass(prompt)
    if not cmd_audit.passed:
      preflight_violations.extend(cmd_audit.violations)

  env_ver = verify_single_chip_envelope(
      hardware_sku=resolved_sku,
      detected_device_count=effective_num_devices,
      perf_cap=resolved_perf_cap,
  )
  if not env_ver.passed:
    preflight_violations.extend(env_ver.violations)

  # Construct candidate RecipeConfig and check division compliance
  recipe_id = f"{model_slug}_{sku_slug}_{div_slug}"
  recipe_path = target_dir / "baseline_recipe.yaml"
  native_res_hw = 256 if (is_wm_track(track) or canonical_model == "gr00t-n1.7-3b") else 224
  candidate_recipe = RecipeConfig(
      recipe_id=recipe_id,
      track=track,
      model_family=canonical_model,
      model_name=hf_repo,
      baseline_model_id=canonical_model,
      precision=resolved_precision,
      hardware_sku=resolved_sku,
      compression_tier=(
          CompressionTier.STRICT_LOSSLESS.value
          if resolved_division == DivisionType.CLOSED.value
          else CompressionTier.EXTREME_COMPRESSION.value
      ),
      division=resolved_division,
      perf_cap=resolved_perf_cap,
      num_devices=effective_num_devices,
      execution_strategy=(
          ExecutionStrategy.ASYNC_CHUNK.value
          if resolved_division == DivisionType.CLOSED.value
          else ExecutionStrategy.RTC_INPAINTING.value
      ),
      denoising_steps=resolved_steps,
      reference_denoising_steps=default_steps,
      resolution_hw=native_res_hw,
      reference_resolution_hw=native_res_hw,
      weights_frozen=resolved_weights_frozen,
      optimizations=merged_opts,
      notes=notes,
  )

  # Check CLOSED division denoising_steps parity for VLA as well as WM when explicitly overridden
  if (
      resolved_division == DivisionType.CLOSED.value
      and denoising_steps is not None
      and int(denoising_steps) != default_steps
  ):
    preflight_violations.append(
        f"CLOSED division requires denoising_steps ({denoising_steps}) == "
        f"reference_denoising_steps ({default_steps}) for {canonical_model}. "
        "Use --division OPEN for step reduction / distillation."
    )

  try:
    validate_division_compliance(candidate_recipe)
  except ValueError as exc:
    preflight_violations.append(str(exc))

  preflight_passed = len(preflight_violations) == 0
  overall_status = RunStatus.CERTIFIED if preflight_passed else RunStatus.REJECTED
  overall_reason: Optional[str] = (
      None
      if preflight_passed
      else "Kickoff preflight rejected: " + " ".join(preflight_violations)
  )
  validate_status_and_reason(overall_status, reason=overall_reason)

  # Write baseline_recipe.yaml (or JSON fallback) if preflight passed
  recipe_dict = candidate_recipe.to_dict()
  if yaml is not None:
    recipe_path.write_text(yaml.safe_dump(recipe_dict, sort_keys=False), encoding="utf-8")
  else:
    recipe_path.write_text(json.dumps(recipe_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  initial_milestones = _build_initial_milestones(
      model_id=canonical_model,
      hardware_sku=resolved_sku,
      division=resolved_division,
      study_dir=target_dir,
      recipe_path=recipe_path,
  )
  if preflight_passed:
    initial_milestones["M1_HARDWARE_AND_DATA_PREFLIGHT"]["status"] = RunStatus.CERTIFIED.value
    initial_milestones["M1_HARDWARE_AND_DATA_PREFLIGHT"]["reason"] = None
  else:
    initial_milestones["M1_HARDWARE_AND_DATA_PREFLIGHT"]["status"] = RunStatus.REJECTED.value
    initial_milestones["M1_HARDWARE_AND_DATA_PREFLIGHT"]["reason"] = overall_reason

  active_skeletons = (
      list(DIVISION_OPTIMIZATION_SKELETONS[DivisionType.CLOSED.value])
      if resolved_division == DivisionType.CLOSED.value
      else (
          list(DIVISION_OPTIMIZATION_SKELETONS[DivisionType.CLOSED.value])
          + list(DIVISION_OPTIMIZATION_SKELETONS[DivisionType.OPEN.value])
      )
  )

  study_plan_path = target_dir / "STUDY_PLAN.json"
  expanded_prompt_path = target_dir / "EXPANDED_AGENT_PROMPT.md"
  baseline_anchor_path = target_dir / "baseline_anchor.json"

  def _upsert_study_plan(existing: Dict[str, Any]) -> Dict[str, Any]:
    recovered = bool(existing.get("recovered_from_corrupt_state", False))
    plan: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "study_id": study_id,
        "study_dir": str(target_dir),
        "created_at": existing.get("created_at", now_pacific_iso()),
        "updated_at": now_pacific_iso(),
        "prompt": str(prompt) if prompt is not None else None,
        "notes": str(notes) if notes is not None else None,
        "model_id": canonical_model,
        "hf_repo": hf_repo,
        "track": track,
        "division": resolved_division,
        "hardware_sku": resolved_sku,
        "perf_cap": resolved_perf_cap,
        "num_devices": effective_num_devices,
        "single_device_env": (
            "TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0"
            if "TPU" in resolved_sku.upper()
            else (
                "HIP_VISIBLE_DEVICES=0"
                if any(tok in resolved_sku.upper() for tok in ("AMD", "MI210", "W7900"))
                else "CUDA_VISIBLE_DEVICES=0"
            )
        ),
        "precision": resolved_precision,
        "denoising_steps": resolved_steps,
        "reference_denoising_steps": default_steps,
        "weights_frozen": resolved_weights_frozen,
        "optimizations": list(merged_opts),
        "passed": preflight_passed,
        "status": overall_status.value,
        "reason": overall_reason,
        "violations": list(preflight_violations),
        "host_accelerator": host_accel.to_dict(),
        "single_chip_verification": env_ver.to_dict(),
        "recipe_path": str(recipe_path),
        "expanded_prompt_path": str(expanded_prompt_path),
        "baseline_anchor_path": None,
        "baseline_metrics": None,
        "milestone_ids": list(MILESTONE_IDS),
        "milestones": initial_milestones,
        "milestone_list": [initial_milestones[m] for m in MILESTONE_IDS],
        "allowed_optimization_skeletons": {
            DivisionType.CLOSED.value: list(DIVISION_OPTIMIZATION_SKELETONS[DivisionType.CLOSED.value]),
            DivisionType.OPEN.value: (
                list(DIVISION_OPTIMIZATION_SKELETONS[DivisionType.CLOSED.value])
                + list(DIVISION_OPTIMIZATION_SKELETONS[DivisionType.OPEN.value])
            ),
            "active_division_skeletons": active_skeletons,
        },
        "anti_cheat_invariants": dict(ANTI_CHEAT_INVARIANTS),
    }
    if recovered:
      plan["recovered_from_corrupt_state"] = True
    return plan

  study_plan = atomic_read_modify_write_json(
      target_path=study_plan_path,
      modifier_fn=_upsert_study_plan,
      default_factory=dict,
  )

  # Render EXPANDED_AGENT_PROMPT.md
  expanded_md = render_expanded_agent_prompt(study_plan)
  expanded_prompt_path.write_text(expanded_md, encoding="utf-8")

  # Optionally execute live HTTP server baseline sweep (`--run-baseline`) if preflight passed
  if run_baseline and preflight_passed:
    update_study_milestone(
        study_plan_path=study_plan_path,
        milestone_id="M2_LIVE_BASELINE_ANCHOR",
        status=RunStatus.RUNNING,
    )
    baseline_report = _execute_live_baseline_anchor(
        model_id=canonical_model,
        hardware_sku=resolved_sku,
        perf_cap=resolved_perf_cap,
        precision=resolved_precision,
        denoising_steps=resolved_steps,
        output_path=baseline_anchor_path,
        concurrencies=baseline_concurrencies,
        steps_per_stream=baseline_steps,
        warmup_steps=baseline_warmup,
    )
    b_metrics = {
        "peak_raw_per_chip_throughput": baseline_report["peak_raw_per_chip_throughput"],
        "peak_reactivity_hz": baseline_report["peak_reactivity_hz"],
        "best_p99_latency_ms": baseline_report["best_p99_latency_ms"],
        "num_operating_points": len(baseline_report["operating_points"]),
    }
    update_study_milestone(
        study_plan_path=study_plan_path,
        milestone_id="M2_LIVE_BASELINE_ANCHOR",
        status=RunStatus.CERTIFIED,
        metrics=b_metrics,
    )

    def _attach_baseline(curr: Dict[str, Any]) -> Dict[str, Any]:
      updated = dict(curr)
      updated["baseline_anchor_path"] = str(baseline_anchor_path)
      updated["baseline_metrics"] = b_metrics
      updated["updated_at"] = now_pacific_iso()
      return updated

    study_plan = atomic_read_modify_write_json(
        target_path=study_plan_path,
        modifier_fn=_attach_baseline,
        default_factory=dict,
    )
    expanded_prompt_path.write_text(render_expanded_agent_prompt(study_plan), encoding="utf-8")

  if not preflight_passed and raise_on_violation:
    raise ValueError(overall_reason or "Kickoff preflight rejected.")

  return study_plan


def build_arg_parser() -> argparse.ArgumentParser:
  """Build argument parser for `python3 -m embodied_bench.agent_kickoff`."""
  parser = argparse.ArgumentParser(
      prog="eeb-agent-kickoff",
      description=(
          "EmbodiedX (`embodied-efficiency-bench` v1.0) Turn-0 AI Agent Autonomous Optimization "
          "Kickoff & Study Workspace Bootstrapper"
      ),
  )
  parser.add_argument(
      "--prompt",
      default=None,
      help="Natural-language startup prompt (e.g. 'Start pi0.5 on TPU v6e-1 in CLOSED division').",
  )
  parser.add_argument(
      "--model",
      default=None,
      help="Canonical model ID or HF alias ('pi05-3b', 'gr00t-n1.7-3b', 'cosmos-predict2.5-2b').",
  )
  parser.add_argument(
      "--hardware-sku",
      default=None,
      help="Target single-chip hardware SKU (auto-detected if omitted; e.g. 'TPU_V6E_1_32GB', 'RTX_5090_32GB').",
  )
  parser.add_argument(
      "--division",
      default=None,
      help="Benchmark division ('CLOSED' or 'OPEN').",
  )
  parser.add_argument(
      "--study-dir",
      default=None,
      help="Directory to bootstrap STUDY_PLAN.json and EXPANDED_AGENT_PROMPT.md.",
  )
  parser.add_argument(
      "--perf-cap",
      default=PerformanceCapId.CAP_SINGLE_CHIP.value,
      help="Performance cap envelope (default: CAP_SINGLE_CHIP).",
  )
  parser.add_argument(
      "--num-devices",
      type=int,
      default=None,
      help="Number of accelerator devices (must be 1 under CAP_SINGLE_CHIP).",
  )
  parser.add_argument(
      "--precision",
      default=None,
      help="Execution precision ('bf16', 'fp16', 'fp8', 'int4_awq', 'w4a8').",
  )
  parser.add_argument(
      "--denoising-steps",
      type=int,
      default=None,
      help="Optional denoising / ODE step count override.",
  )
  parser.add_argument(
      "--optimizations",
      nargs="*",
      action="extend",
      default=[],
      help="Optional list of optimization levers to declare in recipe.",
  )
  parser.add_argument(
      "--run-baseline",
      action="store_true",
      help="Immediately run a live HTTP EmbodiedBenchmarkServer + EmbodiedLoadGenClient baseline sweep.",
  )
  parser.add_argument(
      "--baseline-concurrencies",
      nargs="+",
      action="extend",
      type=int,
      default=[],
      help="Concurrency levels for --run-baseline (default: 1 2).",
  )
  parser.add_argument(
      "--baseline-steps",
      type=int,
      default=4,
      help="Timed steps per stream for --run-baseline (default: 4).",
  )
  parser.add_argument(
      "--baseline-warmup",
      type=int,
      default=1,
      help="Warmup steps per stream for --run-baseline (default: 1).",
  )
  parser.add_argument(
      "--probe-live-hardware",
      action="store_true",
      help="Probe live host CLI tools (nvidia-smi, tegrastats, rocm-smi, tpu-info) when --hardware-sku is omitted.",
  )
  parser.add_argument(
      "--notes",
      default=None,
      help="Optional human-readable notes to store in STUDY_PLAN.json.",
  )
  parser.add_argument(
      "--update-milestone",
      default=None,
      help="Optional milestone ID (M1..M5) to update in an existing STUDY_PLAN.json.",
  )
  parser.add_argument(
      "--milestone-status",
      default=None,
      help="Target RunStatus when using --update-milestone.",
  )
  parser.add_argument(
      "--milestone-reason",
      default=None,
      help="Reason string when transitioning a milestone to REJECTED/BLOCKED/FAILED/SKIPPED/INVALID.",
  )
  parser.add_argument(
      "--output",
      default=None,
      help="Optional path to write CLI summary JSON.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """CLI entrypoint for `python3 -m embodied_bench.agent_kickoff`."""
  parser = build_arg_parser()
  args = parser.parse_args(argv)

  if args.update_milestone:
    if not args.study_dir:
      raise ValueError("--study-dir is required when using --update-milestone.")
    if not args.milestone_status:
      raise ValueError("--milestone-status is required when using --update-milestone.")
    plan_file = Path(args.study_dir) / "STUDY_PLAN.json"
    updated_plan = update_study_milestone(
        study_plan_path=plan_file,
        milestone_id=args.update_milestone,
        status=args.milestone_status,
        reason=args.milestone_reason,
    )
    m_obj = updated_plan["milestones"][args.update_milestone]
    summary = {
        "command": "agent-kickoff",
        "action": "update-milestone",
        "study_dir": str(args.study_dir),
        "study_plan_path": str(plan_file),
        "milestone_id": args.update_milestone,
        "status": m_obj["status"],
        "reason": m_obj["reason"],
    }
    if args.output:
      atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

  concurrencies = [int(c) for c in args.baseline_concurrencies] if args.baseline_concurrencies else [1, 2]
  try:
    plan = run_agent_kickoff(
        prompt=args.prompt,
        model_id=args.model,
        hardware_sku=args.hardware_sku,
        division=args.division,
        study_dir=args.study_dir,
        perf_cap=args.perf_cap,
        num_devices=args.num_devices,
        precision=args.precision,
        denoising_steps=args.denoising_steps,
        optimizations=args.optimizations if args.optimizations else None,
        run_baseline=args.run_baseline,
        baseline_concurrencies=concurrencies,
        baseline_steps=args.baseline_steps,
        baseline_warmup=args.baseline_warmup,
        probe_live_hardware=args.probe_live_hardware,
        notes=args.notes,
        raise_on_violation=False,
    )
  except ValueError as exc:
    err_summary = {
        "command": "agent-kickoff",
        "passed": False,
        "status": RunStatus.REJECTED.value,
        "reason": str(exc),
    }
    if args.output:
      atomic_write_json(args.output, err_summary)
    print(json.dumps(err_summary, indent=2, sort_keys=True))
    return 2

  summary = {
      "command": "agent-kickoff",
      "study_id": plan["study_id"],
      "study_dir": plan["study_dir"],
      "study_plan_path": str(Path(plan["study_dir"]) / "STUDY_PLAN.json"),
      "expanded_prompt_path": plan["expanded_prompt_path"],
      "recipe_path": plan["recipe_path"],
      "baseline_anchor_path": plan["baseline_anchor_path"],
      "model_id": plan["model_id"],
      "track": plan["track"],
      "division": plan["division"],
      "hardware_sku": plan["hardware_sku"],
      "perf_cap": plan["perf_cap"],
      "num_devices": plan["num_devices"],
      "passed": plan["passed"],
      "status": plan["status"],
      "reason": plan["reason"],
      "milestone_ids": plan["milestone_ids"],
      "baseline_metrics": plan["baseline_metrics"],
  }
  if args.output:
    atomic_write_json(args.output, summary)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if plan["passed"] else 2


if __name__ == "__main__":
  sys.exit(main())
