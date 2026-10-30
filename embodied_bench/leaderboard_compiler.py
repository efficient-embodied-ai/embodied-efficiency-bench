"""Multi-submission Pareto curve compiler, reference baseline generator, CAP_SINGLE_CHIP & G1-G4 receipt auditor, and GitHub PR / Website leaderboard exporter for EmbodiedX (`embodied-efficiency-bench` v1.0)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from embodied_bench.attestation import (
    compute_attestation_sha256,
    create_submission_receipt,
    verify_submission_receipt,
)
from embodied_bench.measure import (
    compute_semianalysis_normalizations,
    run_eeb_measure,
)
from embodied_bench.pareto import (
    compute_iso_reactivity_slices,
    extract_pareto_frontier,
    extract_semianalysis_4way_pareto_frontiers,
)
from embodied_bench.schema import (
    ANALYTIC_REFERENCE_PROVENANCE,
    BenchmarkClock,
    CANONICAL_CHALLENGE_MODELS,
    CompressionTier,
    DivisionType,
    ExecutionStrategy,
    HARDWARE_SKU_SPECS,
    LIVE_EMPIRICAL_PROVENANCE,
    MODERN_REFERENCE_CHECKPOINTS,
    PERFORMANCE_CAP_ENVELOPES,
    PerformanceCapId,
    RecipeConfig,
    RunStatus,
    TrackType,
    atomic_write_json,
    is_vla_track,
    is_wm_track,
    load_yaml_or_json,
    now_pacific_iso,
    validate_submission_meta,
)


CANONICAL_CONCURRENCY_GRID: Tuple[int, ...] = (1, 2, 4, 8, 16)
SEASON1_OFFICIAL_HARDWARE_TARGETS: Dict[str, str] = {
    "NVIDIA_GPU": "L40S_48GB",
    "GOOGLE_CLOUD_TPU": "TPU_V6E_1_32GB",
}
MODEL_LOCKED_K_EVAL: Dict[str, int] = {
    "pi05-3b": 5,
    "gr00t-n1.7-3b": 8,
    "cosmos-predict2.5-2b": 1,
}

GAPLESS_SLA_VLA_MAX_P99_MS: float = 250.0
GAPLESS_SLA_VLA_MIN_RATE_HZ: float = 4.0
GAPLESS_SLA_GR00T_MAX_P99_MS: float = 400.0
GAPLESS_SLA_GR00T_MIN_RATE_HZ: float = 2.5
GAPLESS_SLA_WM_MAX_P99_MS: float = 1000.0
GAPLESS_SLA_WM_MIN_RATE_HZ: float = 1.0
AWARD_A_MAX_TAIL_RATIO_C1: float = 1.5
MAX_BIMODAL_LATENCY_RATIO: float = 4.0
HYPERVOLUME_U_REF: float = 0.5


def get_gapless_sla_for_model(track: str, model_id: Optional[str] = None) -> Tuple[float, float, int]:
  """Return `(sla_max_p99_ms, sla_min_rate_hz, locked_k_eval)` for a track and canonical model."""
  tok = str(model_id or "").strip().lower()
  if is_wm_track(track) or "cosmos" in tok:
    return GAPLESS_SLA_WM_MAX_P99_MS, GAPLESS_SLA_WM_MIN_RATE_HZ, 1
  if "gr00t" in tok or "groot" in tok:
    return GAPLESS_SLA_GR00T_MAX_P99_MS, GAPLESS_SLA_GR00T_MIN_RATE_HZ, 8
  return GAPLESS_SLA_VLA_MAX_P99_MS, GAPLESS_SLA_VLA_MIN_RATE_HZ, 5


GAPLESS_REALTIME_SLA_SPEC: Dict[str, Dict[str, Any]] = {
    "VLA_TRACK_1A_PI05": {
        "track": TrackType.VLA_TRACK.value,
        "model_id": "pi05-3b",
        "locked_k_eval": 5,
        "control_freq_hz": 20.0,
        "max_p99_latency_ms": GAPLESS_SLA_VLA_MAX_P99_MS,
        "min_control_rate_hz": GAPLESS_SLA_VLA_MIN_RATE_HZ,
        "rate_unit": "Chunks/s",
        "throughput_unit": "Actions/s",
        "efficiency_unit": "Actions/Joule",
        "description": (
            "Gapless Real-Time Control SLA for Track 1A (pi05-3b): p99 <= 250.0 ms (>= 4.0 Chunks/s), "
            "anchored to locked K_eval=5 at 20 Hz control (5 * 50 ms = 250 ms)."
        ),
    },
    "VLA_TRACK_1B_GR00T": {
        "track": TrackType.VLA_TRACK.value,
        "model_id": "gr00t-n1.7-3b",
        "locked_k_eval": 8,
        "control_freq_hz": 20.0,
        "max_p99_latency_ms": GAPLESS_SLA_GR00T_MAX_P99_MS,
        "min_control_rate_hz": GAPLESS_SLA_GR00T_MIN_RATE_HZ,
        "rate_unit": "Chunks/s",
        "throughput_unit": "Actions/s",
        "efficiency_unit": "Actions/Joule",
        "description": (
            "Gapless Real-Time Control SLA for Track 1B (gr00t-n1.7-3b): p99 <= 400.0 ms (>= 2.5 Chunks/s), "
            "anchored to locked K_eval=8 at 20 Hz control (8 * 50 ms = 400 ms)."
        ),
    },
    TrackType.VLA_TRACK.value: {
        "track": TrackType.VLA_TRACK.value,
        "max_p99_latency_ms_by_model": {"pi05-3b": 250.0, "gr00t-n1.7-3b": 400.0},
        "min_control_rate_hz_by_model": {"pi05-3b": 4.0, "gr00t-n1.7-3b": 2.5},
        "locked_k_eval_by_model": {"pi05-3b": 5, "gr00t-n1.7-3b": 8},
        "rate_unit": "Chunks/s",
        "throughput_unit": "Actions/s",
        "efficiency_unit": "Actions/Joule",
        "description": (
            "Gapless Real-Time Control SLA for Track 1 VLA: pi05-3b requires p99 <= 250.0 ms (>= 4.0 Chunks/s, K_eval=5); "
            "gr00t-n1.7-3b requires p99 <= 400.0 ms (>= 2.5 Chunks/s, K_eval=8)."
        ),
    },
    TrackType.WM_TRACK.value: {
        "track": TrackType.WM_TRACK.value,
        "model_id": "cosmos-predict2.5-2b",
        "locked_k_eval": 1,
        "horizon_frames": 16,
        "max_p99_latency_ms": GAPLESS_SLA_WM_MAX_P99_MS,
        "min_control_rate_hz": GAPLESS_SLA_WM_MIN_RATE_HZ,
        "rate_unit": "Predictions/s",
        "throughput_unit": "Predictions/s",
        "efficiency_unit": "Predictions/Joule",
        "description": (
            "Gapless Real-Time Control SLA for Track 2 World Models (cosmos-predict2.5-2b): p99 <= 1000.0 ms (>= 1.0 Pred/s), "
            "anchored to interactive 1 Hz world-model rollouts (16 frames per prediction)."
        ),
    },
}


SUBMISSION_PROTOCOL_SPEC: Dict[str, Any] = {
    "mode": "TWO_PHASE_COMMIT_REVEAL",
    "submission_folder_convention": "submissions/<track>/<team>_<sku>_<engine>_<date>/",
    "season1_official_hardware_targets": dict(SEASON1_OFFICIAL_HARDWARE_TARGETS),
    "canonical_concurrency_grid": list(CANONICAL_CONCURRENCY_GRID),
    "phase1_contest_pr": {
        "required_files": ["bundle_summary.json", "submission_meta.json"],
        "commitment_field": "recipe_commit_sha256",
        "commitment_definition": "SHA256(canonical_reveal_manifest.json) pinning OCI image @sha256 digest, HF checkpoint shard SHA-256s, lockfile SHA-256, exact server/client CLI flags, and Phase-1 attestation_sha256",
        "exposes_kernel_source_or_flags": False,
    },
    "phase2_snapshot_reveal": {
        "window_after_snapshot_hours": 24,
        "max_finalists_per_lab_per_track_division": 1,
        "evaluation_split": "private_organizer_held_out_traces",
        "ranking_authority": "PHASE2_ORGANIZER_CLEAN_HARDWARE_SCORE",
        "fraud_disqualification_tolerance_pct": 8.0,
        "reproducibility_tolerance_pct": 8.0,
        "open_winner_badge": "VERIFIED_OPEN_RECIPE",
        "rolling_proprietary_badge": "PROPRIETARY_AUDIT_ONLY",
    },
    "season1_schedule": {
        "public_launch": "2026-10-30",
        "checkpoint1_live_spotlight_freeze": "2026-11-10T23:59:00Z",
        "corl_workshop_live_showcase": "2026-11-12",
        "checkpoint2_season1_awards_freeze": "2026-12-18T23:59:00-12:00",
        "continuous_rolling_leaderboard": "2027+",
    },
}


@dataclass(frozen=True)
class ReferenceSeedSpec:
  """Specification for a seeded multi-vendor single-chip reference submission."""

  recipe_id: str
  model_id: str
  model_family: str
  model_name: str
  track: str
  division: str
  hardware_sku: str
  precision: str
  denoising_steps: int
  reference_denoising_steps: int
  speedup_factor_hint: float
  execution_strategy: str
  optimizations: Tuple[str, ...]
  quality_noise_scale: float = 0.0018


def _canonicalize_model_id(raw_model_id: str) -> str:
  """Resolve model identifier or HF repo alias to canonical checkpoint_id."""
  cleaned = str(raw_model_id).strip()
  if cleaned in MODERN_REFERENCE_CHECKPOINTS:
    return cleaned
  lower = cleaned.lower()
  for cid, spec in MODERN_REFERENCE_CHECKPOINTS.items():
    if cid.lower() == lower:
      return cid
    for alias in spec.get("aliases", []):
      if str(alias).lower() == lower:
        return cid
  return cleaned


def _canonicalize_track(raw_track: str, model_id: str) -> str:
  """Normalize track identifier to canonical VLA_TRACK or WM_TRACK."""
  if is_wm_track(raw_track):
    return TrackType.WM_TRACK.value
  if is_vla_track(raw_track):
    return TrackType.VLA_TRACK.value
  ckpt = MODERN_REFERENCE_CHECKPOINTS.get(model_id, {})
  if is_wm_track(str(ckpt.get("track", ""))):
    return TrackType.WM_TRACK.value
  return TrackType.VLA_TRACK.value


def _vendor_family(vendor_raw: str) -> str:
  """Map SKU vendor string to canonical 3-vendor family (GOOGLE_TPU, NVIDIA, AMD)."""
  v = str(vendor_raw).upper()
  if "TPU" in v or "GOOGLE" in v:
    return "GOOGLE_TPU"
  if "AMD" in v or "ROCM" in v:
    return "AMD"
  if "NVIDIA" in v or "TEGRA" in v:
    return "NVIDIA"
  return v


def build_reference_seed_matrix() -> List[ReferenceSeedSpec]:
  """Build the EEB v1.0 Closed Division single-chip reference matrix across the 3 canonical models (`pi05-3b`, `gr00t-n1.7-3b`, `cosmos-predict2.5-2b`), centered on NVIDIA and Google Cloud TPU with single-chip AMD compatibility coverage."""
  specs: List[ReferenceSeedSpec] = [
      # =====================================================================
      # 1. Track 1A: gr00t-n1.7-3b (VLA_TRACK, CLOSED, locked execute_horizon K=8)
      # =====================================================================
      ReferenceSeedSpec(
          recipe_id="ref_gr00t_n17_3b_tpu_v6e_1_closed_bf16",
          model_id="gr00t-n1.7-3b",
          model_family="gr00t-n1.7",
          model_name="GR00T-N1.7-3B-Native-BF16-TorchTPU",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="TPU_V6E_1_32GB",
          precision="BF16",
          denoising_steps=16,
          reference_denoising_steps=16,
          speedup_factor_hint=1.15,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torchtpu_torch_compile",
              "pallas_bf16_systolic_fusion",
              "vlm_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_gr00t_n17_3b_tpu_v5e_1_closed_bf16",
          model_id="gr00t-n1.7-3b",
          model_family="gr00t-n1.7",
          model_name="GR00T-N1.7-3B-Native-BF16-TorchTPU-TPUv5e",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="TPU_V5E_1_16GB",
          precision="BF16",
          denoising_steps=16,
          reference_denoising_steps=16,
          speedup_factor_hint=1.10,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torchtpu_torch_compile",
              "vlm_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_gr00t_n17_3b_rtx_5090_closed_bf16",
          model_id="gr00t-n1.7-3b",
          model_family="gr00t-n1.7",
          model_name="GR00T-N1.7-3B-Native-BF16-CUDAGraph",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="RTX_5090_32GB",
          precision="BF16",
          denoising_steps=16,
          reference_denoising_steps=16,
          speedup_factor_hint=1.12,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torch_compile_inductor_bf16",
              "cuda_graph_static_capture",
              "vlm_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_gr00t_n17_3b_jetson_thor_closed_bf16",
          model_id="gr00t-n1.7-3b",
          model_family="gr00t-n1.7",
          model_name="GR00T-N1.7-3B-Native-BF16-Thor",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="JETSON_AGX_THOR_64GB",
          precision="BF16",
          denoising_steps=16,
          reference_denoising_steps=16,
          speedup_factor_hint=1.10,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "cuda_graph_static_capture",
              "vlm_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_gr00t_n17_3b_amd_mi210_closed_bf16",
          model_id="gr00t-n1.7-3b",
          model_family="gr00t-n1.7",
          model_name="GR00T-N1.7-3B-Native-BF16-ROCm",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="AMD_MI210_64GB",
          precision="BF16",
          denoising_steps=16,
          reference_denoising_steps=16,
          speedup_factor_hint=1.08,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "rocm_composable_kernel_bf16",
              "hip_graph_static_capture",
          ),
      ),
      # =====================================================================
      # 2. Track 1B: pi05-3b (VLA_TRACK, CLOSED, locked execute_horizon K=5)
      # =====================================================================
      ReferenceSeedSpec(
          recipe_id="ref_pi05_3b_tpu_v6e_1_closed_bf16",
          model_id="pi05-3b",
          model_family="pi05",
          model_name="pi0.5-3B-OpenPi-TorchTPU-BF16-TPUv6e",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="TPU_V6E_1_32GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.15,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torchtpu_torch_compile",
              "pallas_bf16_systolic_fusion",
              "static_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_pi05_3b_tpu_v5e_1_closed_bf16",
          model_id="pi05-3b",
          model_family="pi05",
          model_name="pi0.5-3B-OpenPi-TorchTPU-BF16-TPUv5e",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="TPU_V5E_1_16GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.10,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torchtpu_torch_compile",
              "static_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_pi05_3b_rtx_5090_closed_bf16",
          model_id="pi05-3b",
          model_family="pi05",
          model_name="pi0.5-3B-OpenPi-Native-BF16-RTX5090",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="RTX_5090_32GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.12,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torch_compile_inductor_bf16",
              "cuda_graph_static_capture",
              "static_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_pi05_3b_l40s_closed_bf16",
          model_id="pi05-3b",
          model_family="pi05",
          model_name="pi0.5-3B-OpenPi-Native-BF16-L40S",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="L40S_48GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.10,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torch_compile_inductor_bf16",
              "static_kv_cache_prefix_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_pi05_3b_amd_w7900_closed_bf16",
          model_id="pi05-3b",
          model_family="pi05",
          model_name="pi0.5-3B-OpenPi-Native-BF16-W7900",
          track=TrackType.VLA_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="AMD_W7900_48GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.06,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "rocm_composable_kernel_bf16",
              "static_kv_cache_prefix_reuse",
          ),
      ),
      # =====================================================================
      # 3. Track 2: cosmos-predict2.5-2b (WM_TRACK, CLOSED, 256x256, H=16 @ 24 FPS, 10 steps, CFG=1.0)
      # =====================================================================
      ReferenceSeedSpec(
          recipe_id="ref_cosmos_predict25_2b_tpu_v6e_1_closed_bf16",
          model_id="cosmos-predict2.5-2b",
          model_family="cosmos-predict2.5",
          model_name="Cosmos-Predict2.5-2B-TorchTPU-BF16-TPUv6e",
          track=TrackType.WM_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="TPU_V6E_1_32GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.15,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torchtpu_torch_compile",
              "pallas_spatiotemporal_attention_fusion",
              "cross_frame_kv_cache_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_cosmos_predict25_2b_tpu_v5e_1_closed_bf16",
          model_id="cosmos-predict2.5-2b",
          model_family="cosmos-predict2.5",
          model_name="Cosmos-Predict2.5-2B-TorchTPU-BF16-TPUv5e",
          track=TrackType.WM_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="TPU_V5E_1_16GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.10,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torchtpu_torch_compile",
              "cross_frame_kv_cache_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_cosmos_predict25_2b_rtx_5090_closed_bf16",
          model_id="cosmos-predict2.5-2b",
          model_family="cosmos-predict2.5",
          model_name="Cosmos-Predict2.5-2B-Native-BF16-RTX5090",
          track=TrackType.WM_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="RTX_5090_32GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.12,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "torch_compile_inductor_bf16",
              "cuda_graph_rollout_capture",
              "cross_frame_kv_cache_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_cosmos_predict25_2b_jetson_thor_closed_bf16",
          model_id="cosmos-predict2.5-2b",
          model_family="cosmos-predict2.5",
          model_name="Cosmos-Predict2.5-2B-Native-BF16-Thor",
          track=TrackType.WM_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="JETSON_AGX_THOR_64GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.10,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "cuda_graph_rollout_capture",
              "cross_frame_kv_cache_reuse",
          ),
      ),
      ReferenceSeedSpec(
          recipe_id="ref_cosmos_predict25_2b_amd_mi210_closed_bf16",
          model_id="cosmos-predict2.5-2b",
          model_family="cosmos-predict2.5",
          model_name="Cosmos-Predict2.5-2B-Native-BF16-MI210",
          track=TrackType.WM_TRACK.value,
          division=DivisionType.CLOSED.value,
          hardware_sku="AMD_MI210_64GB",
          precision="BF16",
          denoising_steps=10,
          reference_denoising_steps=10,
          speedup_factor_hint=1.08,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
          optimizations=(
              "rocm_flash_attention_bf16",
              "cross_frame_kv_cache_reuse",
          ),
      ),
  ]
  return specs


def _seed_spec_to_recipe_config(spec: ReferenceSeedSpec) -> RecipeConfig:
  """Convert a ReferenceSeedSpec into a validated RecipeConfig under CAP_SINGLE_CHIP."""
  is_open = spec.division == DivisionType.OPEN.value
  wm = is_wm_track(spec.track)
  locked_k = 8 if spec.model_id == "gr00t-n1.7-3b" else 5
  return RecipeConfig(
      recipe_id=spec.recipe_id,
      clock=BenchmarkClock.EEB_V1.value,
      division=spec.division,
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP.value,
      num_devices=1,
      execution_strategy=spec.execution_strategy,
      track=spec.track,
      model_family=spec.model_family,
      model_name=spec.model_name,
      baseline_model_id=spec.model_id,
      precision=spec.precision,
      hardware_sku=spec.hardware_sku,
      compression_tier=CompressionTier.STRICT_LOSSLESS.value,
      horizon_frames=16,
      denoising_steps=spec.denoising_steps,
      reference_denoising_steps=spec.reference_denoising_steps,
      resolution_hw=256 if (wm or spec.model_id == "gr00t-n1.7-3b") else 224,
      reference_resolution_hw=256 if (wm or spec.model_id == "gr00t-n1.7-3b") else 224,
      batch_sizes=[1, 2, 4, 8, 16],
      action_chunk_sizes=[locked_k],
      native_action_chunk_size=16 if spec.model_id == "gr00t-n1.7-3b" else 10,
      execute_horizon=locked_k,
      control_freq_hz=24.0 if wm else 20.0,
      chunk_horizon_s=round(16.0 / 24.0, 4) if wm else round(locked_k / 20.0, 4),
      guidance_scale=1.0,
      concurrency_levels=[1, 2, 4, 8, 16],
      weights_frozen=not is_open,
      declared_training_compute_gpu_hours=24.0 if is_open else 0.0,
      declared_inference_flops_gflops=60000.0 if wm else 1760.0,
      osi_license="NVIDIA-Open-Model-License" if spec.model_id != "pi05-3b" else "Apache-2.0",
      optimizations=list(spec.optimizations),
      speedup_factor_hint=spec.speedup_factor_hint,
      quality_noise_scale=spec.quality_noise_scale,
      stage2_task_suite="WM-MPC-All" if wm else "LIBERO-All",
      stage2_delta_success_pct=-0.35,
      notes=(
          f"Illustrative roofline seed submission for {spec.model_id} ({spec.division}) on 1x {spec.hardware_sku} "
          "under CAP_SINGLE_CHIP (ANALYTIC_REFERENCE_PROJECTION; pending physical hardware runner measurement)."
      ),
  )


def generate_reference_submissions(submissions_root: Path | str) -> List[Path]:
  """Run `run_eeb_measure` + `create_submission_receipt` across all 3 canonical EEB v1.0 models under `CAP_SINGLE_CHIP`.

  Writes each submission into `<target_dir>/<recipe_id>/{recipe.json,sweep_results.json,receipt.json}`
  and returns the list of generated `receipt.json` paths.
  """
  root = Path(submissions_root)
  base_dir = root
  base_dir.mkdir(parents=True, exist_ok=True)

  specs = build_reference_seed_matrix()
  expected_ids = {s.recipe_id for s in specs}
  for child in sorted(base_dir.iterdir()):
    if child.is_dir() and child.name.startswith("ref_") and child.name not in expected_ids:
      shutil.rmtree(child)

  receipt_paths: List[Path] = []
  for spec in specs:
    sub_dir = base_dir / spec.recipe_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    recipe_cfg = _seed_spec_to_recipe_config(spec)
    recipe_path = sub_dir / "recipe.json"
    sweep_path = sub_dir / "sweep_results.json"
    receipt_path = sub_dir / "receipt.json"

    atomic_write_json(recipe_path, recipe_cfg.to_dict())
    report = run_eeb_measure(
        recipe_path=recipe_path,
        output_path=sweep_path,
        include_stage2=True,
        warmup_calls=30,
        timed_calls=200 if is_wm_track(spec.track) else 320,
        num_runs=3,
        perf_cap=PerformanceCapId.CAP_SINGLE_CHIP.value,
    )
    seed_prov = {
        **dict(ANALYTIC_REFERENCE_PROVENANCE),
        "status": "ILLUSTRATIVE_SEED",
        "provenance_tier": "ILLUSTRATIVE_SEED_PENDING_HARDWARE",
        "hardware_measurement_status": "PENDING",
        "physical_run": False,
        "disclaimer": (
            "ILLUSTRATIVE_SEED / PENDING physical hardware measurement. "
            "EEB v1.0 Closed Division roofline reference baseline seed prior to organizer hardware runs."
        ),
    }
    report["execution_provenance"] = seed_prov
    atomic_write_json(sweep_path, report)
    receipt = create_submission_receipt(report=report, output_path=None)
    # Enrich top-level metadata while keeping attested_body 100% canonical for verify_submission_receipt
    receipt["division"] = report["division"]
    receipt["scenario"] = report["scenario"]
    receipt["clock"] = report["clock"]
    receipt["perf_cap"] = report["perf_cap"]
    receipt["num_devices"] = report["num_devices"]
    receipt["single_device_compliant"] = report["single_device_compliant"]
    receipt["execution_strategy"] = report["execution_strategy"]
    receipt["model_family"] = report["model_family"]
    receipt["model_name"] = report["model_name"]
    receipt["baseline_model_id"] = report["baseline_model_id"]
    receipt["precision"] = report["precision"]
    receipt["x_metric_name"] = report["x_metric_name"]
    receipt["y_metric_name"] = report["y_metric_name"]
    receipt["latency_metric_name"] = report["latency_metric_name"]
    receipt["summary_metrics"] = report["summary_metrics"]
    receipt["semianalysis_normalizations"] = report["semianalysis_normalizations"]
    receipt["performance_cap_evaluation"] = report["performance_cap_evaluation"]
    receipt["compliance_gates_g1_g4"] = report["compliance_gates_g1_g4"]
    receipt["operating_points"] = report["operating_points"]
    receipt["pareto_frontier"] = report["pareto_frontier"]
    receipt["execution_provenance"] = seed_prov
    atomic_write_json(receipt_path, receipt)
    receipt_paths.append(receipt_path)

  return receipt_paths


def _resolve_hardware_sku(
    receipt: Dict[str, Any],
    body: Dict[str, Any],
    sweep_data: Dict[str, Any],
) -> str:
  """Resolve canonical hardware_sku across receipt, attested_body, envelope_check, and telemetry."""
  raw_fallback = ""
  env_sweep = sweep_data.get("envelope_check") if isinstance(sweep_data.get("envelope_check"), dict) else {}
  env_receipt = receipt.get("envelope_check") if isinstance(receipt.get("envelope_check"), dict) else {}
  env_body = body.get("envelope_check") if isinstance(body.get("envelope_check"), dict) else {}

  for cand in (
      receipt.get("hardware_sku"),
      body.get("hardware_sku"),
      sweep_data.get("hardware_sku"),
      env_sweep.get("hardware_sku"),
      env_receipt.get("hardware_sku"),
      env_body.get("hardware_sku"),
  ):
    if cand is None:
      continue
    s = str(cand).strip()
    if not raw_fallback and s:
      raw_fallback = s
    if s and s not in ("UNKNOWN_SKU", "None"):
      return s

  device_names: List[str] = []
  for env_obj in (env_sweep, env_receipt, env_body):
    dev = env_obj.get("device")
    if isinstance(dev, dict) and dev.get("name"):
      device_names.append(str(dev["name"]))
  for pts_src in (receipt.get("operating_points"), sweep_data.get("operating_points")):
    if isinstance(pts_src, list):
      for pt in pts_src:
        if isinstance(pt, dict):
          ht = pt.get("hardware_telemetry")
          if isinstance(ht, dict) and ht.get("device_name"):
            device_names.append(str(ht["device_name"]))
            break

  for dname in device_names:
    low = dname.lower()
    if "tpu v6e" in low or "v6e-1" in low:
      return "TPU_V6E_1"
    if "tpu v5e" in low or "v5e-1" in low:
      return "TPU_V5E_1"
    if "l40s" in low:
      return "L40S_48GB"
    if "nvidia l4" in low or re.search(r"\bl4\b", low):
      return "L4_24GB"
    if "rtx 5090" in low:
      return "RTX_5090_32GB"
    if "rtx 4090" in low:
      return "RTX_4090_24GB"
    if "jetson agx thor" in low or "agx thor" in low:
      return "JETSON_AGX_THOR_64GB"
    if "orin nx" in low:
      return "JETSON_ORIN_NX_16GB"
    if "mi210" in low:
      return "AMD_MI210_64GB"
    if "w7900" in low:
      return "AMD_W7900_48GB"

  return raw_fallback


def _audit_single_receipt(
    receipt_path: Path,
) -> Tuple[bool, List[str], Dict[str, Any]]:
  """Verify cryptographic SHA-256 signature, CAP_SINGLE_CHIP bounds, and G1-G4 compliance for one receipt.json."""
  violations: List[str] = []
  try:
    receipt = load_yaml_or_json(receipt_path)
  except Exception as exc:
    return False, [f"Failed to parse receipt JSON at {receipt_path}: {exc}"], {}

  ok_sig, sig_msg = verify_submission_receipt(receipt_path)
  if not ok_sig:
    violations.append(sig_msg)

  body = receipt.get("attested_body") if isinstance(receipt.get("attested_body"), dict) else {}

  # Optional sibling sweep_results.json or report.json fallback
  sweep_data: Dict[str, Any] = {}
  for sibling_name in ("sweep_results.json", "report.json"):
    sibling_sweep = receipt_path.parent / sibling_name
    if sibling_sweep.exists():
      try:
        sweep_data = load_yaml_or_json(sibling_sweep)
      except Exception:
        sweep_data = {}
      break

  raw_recipe_id = str(
      receipt.get("recipe_id")
      or body.get("recipe_id")
      or sweep_data.get("recipe_id")
      or receipt_path.parent.name
  )
  recipe_id = (
      receipt_path.parent.name
      if (raw_recipe_id.startswith("adhoc-") and receipt_path.parent.name)
      else raw_recipe_id
  )
  hardware_sku = _resolve_hardware_sku(receipt, body, sweep_data)
  status = str(receipt.get("status") or body.get("status") or "")

  frontier_pts = body.get("frontier_points") if isinstance(body.get("frontier_points"), list) else []
  exec_prov_pre = (
      body.get("execution_provenance")
      if isinstance(body.get("execution_provenance"), dict)
      else (
          receipt.get("execution_provenance")
          if isinstance(receipt.get("execution_provenance"), dict)
          else (
              sweep_data.get("execution_provenance")
              if isinstance(sweep_data.get("execution_provenance"), dict)
              else {}
          )
      )
  )
  env_chk_pre = (
      sweep_data.get("envelope_check")
      if isinstance(sweep_data.get("envelope_check"), dict)
      else (
          receipt.get("envelope_check")
          if isinstance(receipt.get("envelope_check"), dict)
          else (body.get("envelope_check") if isinstance(body.get("envelope_check"), dict) else None)
      )
  )
  sum_pre = (
      receipt.get("summary_metrics")
      if isinstance(receipt.get("summary_metrics"), dict)
      else (sweep_data.get("summary_metrics") if isinstance(sweep_data.get("summary_metrics"), dict) else {})
  )
  reason_str = str(receipt.get("reason") or body.get("reason") or "")
  is_real_admissible_pending = (
      status == RunStatus.PENDING.value
      and exec_prov_pre.get("loadgen") == "REAL_WALLCLOCK"
      and exec_prov_pre.get("policy_is_synthetic") is False
      and bool(frontier_pts)
      and not bool(sweep_data.get("admissibility_findings"))
      and (env_chk_pre is None or bool(env_chk_pre.get("passed", False)))
      and sum_pre.get("all_variance_passed", True) is True
      and not reason_str.lower().startswith("not admissible")
  )

  if status != RunStatus.CERTIFIED.value and not is_real_admissible_pending:
    violations.append(
        f"Submission status must be CERTIFIED (or admissible ADR-5 real-hardware PENDING) for leaderboard inclusion, "
        f"got status={status!r} (reason={receipt.get('reason') or body.get('reason')!r})."
    )

  # Cross-check top-level throughput/frontier against signed attested_body.frontier_points to catch post-signing edits outside attested_body
  if frontier_pts:
    attested_max_tput = max(float(pt.get("y_throughput", 0.0)) for pt in frontier_pts)
    top_summary = receipt.get("summary_metrics")
    if isinstance(top_summary, dict) and "peak_throughput" in top_summary:
      claimed_peak = float(top_summary["peak_throughput"])
      if abs(claimed_peak - attested_max_tput) > 1e-2:
        violations.append(
            f"Tampered summary_metrics.peak_throughput={claimed_peak} != signed attested_body frontier peak={attested_max_tput}."
        )
    top_frontier = receipt.get("pareto_frontier")
    if isinstance(top_frontier, list) and len(top_frontier) == len(frontier_pts):
      y_key_check = str(receipt.get("y_metric_name") or sweep_data.get("y_metric_name") or "")
      for idx, (t_pt, a_pt) in enumerate(zip(top_frontier, frontier_pts)):
        if isinstance(t_pt, dict):
          val = t_pt.get(y_key_check, t_pt.get("raw_per_chip_throughput"))
          if val is not None and abs(float(val) - float(a_pt.get("y_throughput", 0.0))) > 1e-2:
            violations.append(
                f"Tampered pareto_frontier[{idx}] throughput={val} != signed attested_body.frontier_points[{idx}].y_throughput={a_pt.get('y_throughput')}."
            )

  # Enforce CAP_SINGLE_CHIP: num_devices == 1, vram_gb <= 64.0, bf16_tflops <= 1000.0
  sku_spec = HARDWARE_SKU_SPECS.get(hardware_sku)
  if sku_spec is None:
    violations.append(
        f"Unknown hardware_sku={hardware_sku!r} in receipt {receipt_path}. Must be registered in HARDWARE_SKU_SPECS."
    )
    sku_spec = {}

  pcap = (
      receipt.get("performance_cap_evaluation")
      if isinstance(receipt.get("performance_cap_evaluation"), dict)
      else (
          sweep_data.get("performance_cap_evaluation")
          if isinstance(sweep_data.get("performance_cap_evaluation"), dict)
          else {}
      )
  )
  recipe_dict = (
      receipt.get("recipe")
      if isinstance(receipt.get("recipe"), dict)
      else (sweep_data.get("recipe") if isinstance(sweep_data.get("recipe"), dict) else {})
  )

  candidates_num_devices: List[int] = []
  for src in (
      receipt.get("num_devices"),
      body.get("num_devices"),
      pcap.get("num_devices"),
      recipe_dict.get("num_devices"),
      sku_spec.get("num_devices"),
  ):
    if src is not None:
      candidates_num_devices.append(int(src))
  if any(d == 0 for d in candidates_num_devices):
    num_devices = 0
    violations.append(
        f"CAP_SINGLE_CHIP physical accelerator requirement: CPU-only / allow-no-device self-test run "
        f"(num_devices=0) cannot enter the official hardware leaderboard for recipe_id={recipe_id!r} "
        f"(hardware_sku={hardware_sku!r}); re-run on 1 physical qualifying accelerator."
    )
  else:
    num_devices = max(candidates_num_devices) if candidates_num_devices else 1
  if any(d > 1 or d < 0 for d in candidates_num_devices):
    bad_dev = max(candidates_num_devices) if max(candidates_num_devices) > 1 else min(candidates_num_devices)
    num_devices = bad_dev
    violations.append(
        f"CAP_SINGLE_CHIP single-device violation: num_devices={bad_dev} != 1 for recipe_id={recipe_id!r} "
        f"(hardware_sku={hardware_sku!r}). Multi-GPU / multi-chip submissions are prohibited."
    )

  sku_vram_gb = float(sku_spec.get("vram_mb", 0.0)) / 1024.0 if sku_spec else 0.0
  peak_vram_gb = float(
      receipt.get(
          "vram_gb",
          body.get(
              "vram_gb",
              pcap.get("peak_vram_gb", pcap.get("device_total_vram_gb", sku_vram_gb)),
          ),
      )
      or sku_vram_gb
  )
  effective_vram_gb = max(sku_vram_gb, peak_vram_gb)
  if effective_vram_gb > 64.0 + 1e-6:
    violations.append(
        f"CAP_SINGLE_CHIP VRAM violation: vram_gb={effective_vram_gb:.2f} GB > 64.0 GB limit "
        f"for recipe_id={recipe_id!r} (hardware_sku={hardware_sku!r})."
    )

  sku_bf16_tflops = float(sku_spec.get("fp16_tflops", 0.0)) if sku_spec else 0.0
  claimed_bf16_tflops = float(
      receipt.get(
          "bf16_tflops",
          body.get(
              "bf16_tflops",
              pcap.get("effective_tflops", pcap.get("device_fp16_tflops", sku_bf16_tflops)),
          ),
      )
      or sku_bf16_tflops
  )
  effective_bf16_tflops = max(sku_bf16_tflops, claimed_bf16_tflops)
  if effective_bf16_tflops > 1000.0 + 1e-6:
    violations.append(
        f"CAP_SINGLE_CHIP compute violation: bf16_tflops={effective_bf16_tflops:.1f} TFLOPS > 1000.0 TFLOPS limit "
        f"for recipe_id={recipe_id!r} (hardware_sku={hardware_sku!r})."
    )

  if pcap and not bool(pcap.get("passed", True)):
    violations.append(
        f"PerformanceCapEnvelope evaluation failed: {pcap.get('reason', 'unspecified cap violation')}"
    )

  # Enforce G1-G4 compliance suite
  g1_g4 = (
      receipt.get("compliance_gates_g1_g4")
      if isinstance(receipt.get("compliance_gates_g1_g4"), dict)
      else (
          body.get("compliance_gates_g1_g4")
          if isinstance(body.get("compliance_gates_g1_g4"), dict)
          else (
              sweep_data.get("compliance_gates_g1_g4")
              if isinstance(sweep_data.get("compliance_gates_g1_g4"), dict)
              else {}
          )
      )
  )
  if g1_g4:
    if not bool(g1_g4.get("passed", True)):
      violations.append(
          f"Compliance gates G1-G4 failed for recipe_id={recipe_id!r}: {g1_g4.get('reason', 'gate failure')}"
      )
    for gate_key in ("g1_parity", "g2_non_inferiority", "g3_same_execution", "g4_cache_memoization"):
      gate_obj = g1_g4.get(gate_key)
      if isinstance(gate_obj, dict) and not bool(gate_obj.get("passed", True)):
        violations.append(
            f"Compliance gate {gate_key} failed for recipe_id={recipe_id!r}: {gate_obj.get('reason', 'failed')}"
        )

  # Enforce observation wire-resolution minimum (>= 224x224, >= 150,528 B/req) for non-synthetic hardware runs
  srv_prov = (
      sweep_data.get("server_provenance")
      if isinstance(sweep_data.get("server_provenance"), dict)
      else (body.get("server_provenance") if isinstance(body.get("server_provenance"), dict) else {})
  )
  exec_prov_top = (
      receipt.get("execution_provenance")
      if isinstance(receipt.get("execution_provenance"), dict)
      else (
          sweep_data.get("execution_provenance")
          if isinstance(sweep_data.get("execution_provenance"), dict)
          else {}
      )
  )
  caps_obj = (
      sweep_data.get("capabilities")
      if isinstance(sweep_data.get("capabilities"), dict)
      else (
          sweep_data.get("caps")
          if isinstance(sweep_data.get("caps"), dict)
          else (receipt.get("caps") if isinstance(receipt.get("caps"), dict) else {})
      )
  )
  srv_impl = str(caps_obj.get("server_impl", ""))
  is_synthetic_run = (
      bool(srv_prov.get("is_synthetic", False))
      or bool(exec_prov_top.get("policy_is_synthetic", False))
      or srv_impl == "dummy-synthetic-v1"
      or srv_impl.upper().startswith("SYNTHETIC_DUMMY")
  )
  obs_src = (
      sweep_data.get("observation_source")
      if isinstance(sweep_data.get("observation_source"), dict)
      else (
          receipt.get("observation_source")
          if isinstance(receipt.get("observation_source"), dict)
          else (body.get("observation_source") if isinstance(body.get("observation_source"), dict) else None)
      )
  )
  if isinstance(obs_src, dict) and not is_synthetic_run:
    if obs_src.get("wire_resolution_admissible") is False:
      violations.append(
          f"Observation wire resolution violation for recipe_id={recipe_id!r}: "
          f"wire_bytes_per_request_images={obs_src.get('wire_bytes_per_request_images')} "
          f"(image_hw={obs_src.get('image_hw')}) is below protocol minimum 224x224 (150,528 B/req)."
      )
    else:
      img_hw = obs_src.get("image_hw")
      w_bytes = obs_src.get("bytes_per_request_images", obs_src.get("wire_bytes_per_request_images"))
      if isinstance(img_hw, list) and len(img_hw) >= 2:
        try:
          if min(int(img_hw[0]), int(img_hw[1])) < 224:
            violations.append(
                f"Observation wire resolution violation for recipe_id={recipe_id!r}: "
                f"image_hw={img_hw} is below protocol minimum 224x224."
            )
        except (TypeError, ValueError):
          pass
      if isinstance(w_bytes, (int, float)) and not isinstance(w_bytes, bool) and int(w_bytes) < 150528:
        violations.append(
            f"Observation wire resolution violation for recipe_id={recipe_id!r}: "
            f"bytes_per_request_images={int(w_bytes)} B is below protocol minimum 150,528 B (224x224x3 RGB)."
        )

  # Validate optional or required sibling submission_meta.json (Two-Phase Commit-Reveal protocol)
  meta_file = receipt_path.parent / "submission_meta.json"
  has_open_recipe = any(
      (receipt_path.parent / rname).is_file()
      for rname in ("recipe.yaml", "recipe.yml", "recipe.json")
  ) or bool(receipt.get("recipe") or sweep_data.get("recipe"))
  is_bundle_summary_only = (
      receipt_path.name == "bundle_summary.json"
      and not (receipt_path.parent / "receipt.json").is_file()
      and not (receipt_path.parent / "submission_receipt.json").is_file()
      and not (receipt_path.parent / "report.json").is_file()
      and not (receipt_path.parent / "sweep_results.json").is_file()
      and not has_open_recipe
  )
  normalized_meta: Dict[str, Any] = {}
  if meta_file.is_file():
    try:
      raw_meta = load_yaml_or_json(meta_file)
      meta_ok, meta_errs, normalized_meta = validate_submission_meta(
          raw_meta, require_commit_hash=(not has_open_recipe)
      )
      if not meta_ok:
        violations.extend(meta_errs)
      else:
        meta_sku = normalized_meta.get("hardware_sku")
        if meta_sku and hardware_sku and meta_sku != hardware_sku:
          violations.append(
              f"submission_meta.json hardware_sku={meta_sku!r} != signed bundle hardware_sku={hardware_sku!r}."
          )
        bundle_div = str(
            receipt.get("division") or body.get("division") or sweep_data.get("division") or ""
        ).strip().upper()
        meta_div = normalized_meta.get("division")
        if meta_div and bundle_div and meta_div != bundle_div:
          violations.append(
              f"submission_meta.json division={meta_div!r} != signed bundle division={bundle_div!r}."
          )
    except Exception as exc:
      violations.append(f"Failed to parse {meta_file}: {exc}")
  elif is_bundle_summary_only:
    violations.append(
        f"Phase 1 Commit-Reveal bundle at {receipt_path.parent} contains bundle_summary.json without an open recipe "
        "or submission_meta.json; provide submission_meta.json with hardware_sku, division, and 64-hex recipe_commit_sha256."
    )

  if normalized_meta.get("submission_phase"):
    submission_phase = str(normalized_meta["submission_phase"])
  elif has_open_recipe:
    submission_phase = "OPEN_RECIPE"
  else:
    submission_phase = "COMMIT_ONLY" if normalized_meta.get("recipe_commit_sha256") else "OPEN_RECIPE"

  # RT Fix 7: Cryptographically bind Phase-1 recipe_commit_sha256 inside attested_body
  meta_commit_hash = normalized_meta.get("recipe_commit_sha256")
  body_commit_hash = body.get("recipe_commit_sha256")
  if submission_phase == "COMMIT_ONLY":
    if not body_commit_hash:
      violations.append(
          f"Phase 1 COMMIT_ONLY bundle {receipt_path} must cryptographically sign recipe_commit_sha256 inside "
          "attested_body (package with 'eeb prepare-commit-bundle')."
      )
    elif meta_commit_hash and str(body_commit_hash) != str(meta_commit_hash):
      violations.append(
          f"Commit-reveal hash mismatch for {receipt_path}: submission_meta.recipe_commit_sha256={meta_commit_hash!r} "
          f"!= signed attested_body.recipe_commit_sha256={body_commit_hash!r}."
      )
  elif body_commit_hash and meta_commit_hash and str(body_commit_hash) != str(meta_commit_hash):
    violations.append(
        f"Commit-reveal hash mismatch for {receipt_path}: submission_meta.recipe_commit_sha256={meta_commit_hash!r} "
        f"!= signed attested_body.recipe_commit_sha256={body_commit_hash!r}."
    )

  # Resolve canonical track, model_id, and division for protocol anti-gaming audits (RT Fixes 1-6)
  raw_model_audit = str(
      receipt.get("baseline_model_id")
      or receipt.get("model_id")
      or body.get("baseline_model_id")
      or body.get("model_id")
      or sweep_data.get("baseline_model_id")
      or sweep_data.get("model_id")
      or caps_obj.get("model_id")
      or ""
  )
  model_id_audit = _canonicalize_model_id(raw_model_audit)
  raw_track_audit = str(receipt.get("track") or body.get("track") or sweep_data.get("track") or "")
  track_audit = _canonicalize_track(raw_track_audit, model_id_audit)
  wm_audit = is_wm_track(track_audit)
  prec_audit = str(receipt.get("precision") or body.get("precision") or sweep_data.get("precision") or "BF16")
  div_audit = str(
      receipt.get("division")
      or body.get("division")
      or sweep_data.get("division")
      or normalized_meta.get("division")
      or (
          DivisionType.OPEN.value
          if ("open" in recipe_id.lower() or prec_audit.upper() not in ("BF16", "FP16", "FP32"))
          else DivisionType.CLOSED.value
      )
  ).upper()

  audit_ops = (
      receipt.get("operating_points")
      if isinstance(receipt.get("operating_points"), list) and receipt.get("operating_points")
      else (
          sweep_data.get("operating_points")
          if isinstance(sweep_data.get("operating_points"), list) and sweep_data.get("operating_points")
          else []
      )
  )

  # RT Fix 3: Enforce exact ordered canonical concurrency grid prefix [1, 2, 4, 8, 16][:k] starting at C=1
  if audit_ops:
    obs_cs = [_extract_point_concurrency(p) for p in audit_ops if isinstance(p, dict)]
    if obs_cs and all(c is not None for c in obs_cs):
      cs_tuple = tuple(int(c) for c in obs_cs)  # type: ignore[arg-type]
      expected_prefix = CANONICAL_CONCURRENCY_GRID[: len(cs_tuple)]
      if len(cs_tuple) > len(CANONICAL_CONCURRENCY_GRID) or cs_tuple != expected_prefix:
        violations.append(
            f"Concurrency grid violation for recipe_id={recipe_id!r}: reported concurrencies={list(cs_tuple)} "
            f"must be an exact ordered prefix of canonical grid {list(CANONICAL_CONCURRENCY_GRID)} starting at C=1."
        )
  elif frontier_pts:
    f_cs = [_extract_point_concurrency(p) for p in frontier_pts if isinstance(p, dict)]
    if f_cs and all(c is not None for c in f_cs):
      if 1 not in f_cs or any(int(c) not in CANONICAL_CONCURRENCY_GRID for c in f_cs):  # type: ignore[arg-type]
        violations.append(
            f"Concurrency grid violation for recipe_id={recipe_id!r}: signed frontier concurrencies={f_cs} "
            f"must include C=1 and lie exclusively on canonical grid {list(CANONICAL_CONCURRENCY_GRID)}."
        )

  # RT Fix 4: Enforce locked K_eval across BOTH CLOSED and OPEN divisions
  _sla_p99, _sla_hz, locked_k_eval = get_gapless_sla_for_model(track_audit, model_id_audit)
  if model_id_audit in MODEL_LOCKED_K_EVAL:
    if wm_audit:
      hf_rec = recipe_dict.get("horizon_frames")
      if hf_rec is not None and int(hf_rec) != 16:
        violations.append(
            f"Locked horizon violation for {model_id_audit!r} ({div_audit}): horizon_frames={int(hf_rec)} != 16."
        )
      for p in audit_ops:
        if isinstance(p, dict):
          hf_pt = p.get("horizon_frames", p.get("horizon"))
          if hf_pt is not None and int(hf_pt) != 16:
            violations.append(
                f"Locked horizon violation for {model_id_audit!r} ({div_audit}) at point {p.get('point_id')!r}: "
                f"horizon_frames={int(hf_pt)} != 16."
            )
            break
    else:
      eh_rec = recipe_dict.get("execute_horizon")
      if eh_rec is not None and int(eh_rec) != locked_k_eval:
        violations.append(
            f"Locked K_eval violation for {model_id_audit!r} ({div_audit}): recipe execute_horizon={int(eh_rec)} "
            f"!= canonical K_eval={locked_k_eval} (evaluation horizon is locked across both CLOSED and OPEN)."
        )
      for p in audit_ops:
        if isinstance(p, dict) and p.get("execute_horizon") is not None:
          if int(p["execute_horizon"]) != locked_k_eval:
            violations.append(
                f"Locked K_eval violation for {model_id_audit!r} ({div_audit}) at point {p.get('point_id')!r}: "
                f"execute_horizon={int(p['execute_horizon'])} != canonical K_eval={locked_k_eval}."
            )
            break
      for fp in frontier_pts:
        if isinstance(fp, dict) and fp.get("execute_horizon") is not None:
          if int(fp["execute_horizon"]) != locked_k_eval:
            violations.append(
                f"Locked K_eval violation in signed frontier for {model_id_audit!r} ({div_audit}): "
                f"execute_horizon={int(fp['execute_horizon'])} != canonical K_eval={locked_k_eval}."
            )
            break

  # RT Fix 2 / RT-02 / RT-03: Reject synthetic observation payloads, smoke goldens, sub-protocol warmups/slots, and bimodal tails
  from eeb.data.schema import SYNTHETIC_CI_SMOKE_TRACES_KIND, is_synthetic_trace_manifest

  if isinstance(obs_src, dict):
    obs_kind = str(obs_src.get("kind") or "")
    if body.get("observation_source_kind") and obs_kind and str(body["observation_source_kind"]) != obs_kind:
      violations.append(
          f"Tampered observation_source.kind={obs_kind!r} != signed attested_body.observation_source_kind={body['observation_source_kind']!r}."
      )
    if not is_synthetic_run and (
        obs_kind.upper() in ("SYNTHETIC_RANDOM_TENSORS", SYNTHETIC_CI_SMOKE_TRACES_KIND)
        or is_synthetic_trace_manifest(obs_src)
    ):
      violations.append(
          f"Observation source violation for recipe_id={recipe_id!r}: {obs_kind or SYNTHETIC_CI_SMOKE_TRACES_KIND!r} "
          "cannot enter the leaderboard; run on real EmbodiedX-Traces-v1 observations (RT-02)."
      )
  elif not is_synthetic_run and str(body.get("observation_source_kind") or "").upper() in (
      "SYNTHETIC_RANDOM_TENSORS",
      SYNTHETIC_CI_SMOKE_TRACES_KIND,
  ):
    violations.append(
        f"Observation source violation for recipe_id={recipe_id!r}: "
        f"{body.get('observation_source_kind')!r} cannot enter the leaderboard (RT-02)."
    )

  qg_obj = (
      receipt.get("quality_gate")
      if isinstance(receipt.get("quality_gate"), dict)
      else (sweep_data.get("quality_gate") if isinstance(sweep_data.get("quality_gate"), dict) else {})
  )
  top_golden_sha = qg_obj.get("golden_sha256")
  body_golden_sha = body.get("golden_sha256")
  if top_golden_sha and body_golden_sha and str(top_golden_sha) != str(body_golden_sha):
    violations.append(
        f"Tampered quality_gate.golden_sha256={top_golden_sha!r} != signed attested_body.golden_sha256={body_golden_sha!r}."
    )

  if not is_synthetic_run:
    sweep_cfg_obj = (
        receipt.get("sweep_config")
        if isinstance(receipt.get("sweep_config"), dict)
        else (sweep_data.get("sweep_config") if isinstance(sweep_data.get("sweep_config"), dict) else None)
    )
    if isinstance(sweep_cfg_obj, dict):
      sc_w = sweep_cfg_obj.get("warmup_calls")
      if isinstance(sc_w, int) and not isinstance(sc_w, bool) and sc_w < 30:
        violations.append(
            f"Protocol warmup violation for recipe_id={recipe_id!r}: sweep_config.warmup_calls={sc_w} < 30 minimum (RT-03)."
        )
      sc_m = sweep_cfg_obj.get("measured_calls")
      min_m_calls = 200 if wm_audit else 320
      if isinstance(sc_m, int) and not isinstance(sc_m, bool) and sc_m < min_m_calls:
        violations.append(
            f"Protocol sample count violation for recipe_id={recipe_id!r}: "
            f"sweep_config.measured_calls={sc_m} < {min_m_calls} minimum (RT-03)."
        )
      sc_r = sweep_cfg_obj.get("num_runs")
      if isinstance(sc_r, int) and not isinstance(sc_r, bool) and sc_r < 3:
        violations.append(
            f"Protocol repetition violation for recipe_id={recipe_id!r}: sweep_config.num_runs={sc_r} < 3 minimum (RT-03)."
        )
    mp_obj = (
        receipt.get("measurement_protocol")
        if isinstance(receipt.get("measurement_protocol"), dict)
        else (sweep_data.get("measurement_protocol") if isinstance(sweep_data.get("measurement_protocol"), dict) else None)
    )
    if isinstance(mp_obj, dict):
      mp_w = mp_obj.get("warmup_calls")
      if isinstance(mp_w, int) and not isinstance(mp_w, bool) and mp_w < 30:
        violations.append(
            f"Protocol warmup violation for recipe_id={recipe_id!r}: measurement_protocol.warmup_calls={mp_w} < 30 minimum (RT-03)."
        )
    if isinstance(qg_obj, dict):
      if bool(qg_obj.get("allow_smoke_golden")):
        violations.append(
            f"Smoke golden violation for recipe_id={recipe_id!r}: allow_smoke_golden=True (<100 samples) cannot enter leaderboard (RT-03)."
        )
      qg_ns = qg_obj.get("num_samples")
      if isinstance(qg_ns, int) and not isinstance(qg_ns, bool) and qg_ns < 100:
        violations.append(
            f"Golden sample count violation for recipe_id={recipe_id!r}: quality_gate.num_samples={qg_ns} < 100 minimum (RT-03)."
        )
      per_c_qg = qg_obj.get("per_concurrency")
      if isinstance(per_c_qg, dict):
        for c_tag, g_entry in per_c_qg.items():
          if not isinstance(g_entry, dict):
            continue
          g_meta_e = g_entry.get("golden") if isinstance(g_entry.get("golden"), dict) else {}
          g_chk_e = g_entry.get("checks") if isinstance(g_entry.get("checks"), dict) else {}
          if bool(g_meta_e.get("allow_smoke_golden") or g_chk_e.get("allow_smoke_golden") or g_entry.get("allow_smoke_golden")):
            violations.append(
                f"Smoke golden violation at {c_tag} for recipe_id={recipe_id!r}: allow_smoke_golden=True cannot enter leaderboard (RT-03)."
            )
            break
          ns_e = g_meta_e.get("num_samples", g_entry.get("num_samples", g_chk_e.get("num_samples")))
          if isinstance(ns_e, int) and not isinstance(ns_e, bool) and ns_e < 100:
            violations.append(
                f"Golden sample count violation at {c_tag} for recipe_id={recipe_id!r}: num_samples={ns_e} < 100 minimum (RT-03)."
            )
            break

    obs_plimits: List[float] = []
    for p in audit_ops:
      if not isinstance(p, dict):
        continue
      if p.get("slot_starvation_passed") is False or (
          isinstance(p.get("max_slot_count_spread"), (int, float))
          and not isinstance(p.get("max_slot_count_spread"), bool)
          and float(p["max_slot_count_spread"]) > 0.15 + 1e-9
      ):
        violations.append(
            f"Slot-starvation violation at {p.get('point_id', 'point')!r} for recipe_id={recipe_id!r}: "
            f"max_slot_count_spread={p.get('max_slot_count_spread')!r} > 0.15 (RT-03)."
        )
        break
      if p.get("c1_tail_ratio_passed") is False:
        violations.append(
            f"C=1 tail-latency stability violation at {p.get('point_id', 'point')!r} for recipe_id={recipe_id!r}: "
            "c1_tail_ratio_passed=False (p99/p50 > 1.5; RT-03)."
        )
        break
      ht_a = p.get("hardware_telemetry") if isinstance(p.get("hardware_telemetry"), dict) else {}
      plim_v = ht_a.get("power_limit_w", p.get("power_limit_w"))
      if isinstance(plim_v, (int, float)) and not isinstance(plim_v, bool) and float(plim_v) > 0.0:
        obs_plimits.append(float(plim_v))
      p50_v = p.get("rollout_latency_p50_ms" if wm_audit else "obs_to_action_p50_ms", p.get("p50_latency_ms"))
      max_v = p.get("rollout_latency_max_ms" if wm_audit else "obs_to_action_max_ms", p.get("max_latency_ms"))
      if p50_v is not None and max_v is not None and float(p50_v) > 0.0:
        ratio_mp = float(max_v) / float(p50_v)
        if ratio_mp > MAX_BIMODAL_LATENCY_RATIO + 1e-9:
          violations.append(
              f"Bimodal tail latency violation at {p.get('point_id', 'point')!r} for recipe_id={recipe_id!r}: "
              f"max/p50={ratio_mp:.2f} > {MAX_BIMODAL_LATENCY_RATIO:.1f} (max={float(max_v):.2f} ms, p50={float(p50_v):.2f} ms)."
          )
          break
    if len(obs_plimits) >= 2 and (max(obs_plimits) - min(obs_plimits)) > 1.0:
      violations.append(
          f"Dynamic power-limit capping violation for recipe_id={recipe_id!r}: power_limit_w varied across concurrency points "
          f"(min={min(obs_plimits):.1f} W, max={max(obs_plimits):.1f} W); constant power_limit_w is required across the sweep."
      )

  # RT Fix 6: Mandatory Stage 2 Closed-Loop / 3-Part Quality Gate B receipt for OPEN division submissions
  if div_audit == DivisionType.OPEN.value and not is_synthetic_run:
    gate_b_obj = (
        body.get("stage2_gate_b")
        if isinstance(body.get("stage2_gate_b"), dict)
        else (
            receipt.get("stage2_gate_b")
            if isinstance(receipt.get("stage2_gate_b"), dict)
            else (sweep_data.get("stage2_gate_b") if isinstance(sweep_data.get("stage2_gate_b"), dict) else None)
        )
    )
    if not isinstance(gate_b_obj, dict) or not bool(gate_b_obj.get("passed", False)):
      violations.append(
          f"OPEN division quality gate violation for recipe_id={recipe_id!r}: OPEN submissions must carry a passing "
          "stage2_gate_b evaluation (Track 1: LIBERO closed-loop SR >= ref - 5.0pp; Track 2: 3-part WM gate with Kendall tau >= 0.80)."
      )
    elif wm_audit:
      tau_val = gate_b_obj.get("s2_policy_ranking_kendall_tau")
      if tau_val is None or float(tau_val) < 0.80 - 1e-9:
        violations.append(
            f"OPEN division Track 2 quality gate violation for recipe_id={recipe_id!r}: "
            f"stage2_gate_b.s2_policy_ranking_kendall_tau={tau_val!r} must be present and >= 0.80."
        )
      if gate_b_obj.get("psnr_db") is not None and float(gate_b_obj["psnr_db"]) < 28.0 - 1e-9:
        violations.append(f"OPEN division Track 2 PSNR violation: psnr_db={gate_b_obj['psnr_db']} < 28.0 dB.")
      if gate_b_obj.get("lpips") is not None and float(gate_b_obj["lpips"]) > 0.12 + 1e-9:
        violations.append(f"OPEN division Track 2 LPIPS violation: lpips={gate_b_obj['lpips']} > 0.12.")
      if gate_b_obj.get("fvd") is not None and float(gate_b_obj["fvd"]) > 120.0 + 1e-9:
        violations.append(f"OPEN division Track 2 FVD violation: fvd={gate_b_obj['fvd']} > 120.0.")
      if (
          gate_b_obj.get("inverse_dynamics_action_cosine_sim") is not None
          and float(gate_b_obj["inverse_dynamics_action_cosine_sim"]) < 0.990 - 1e-9
      ):
        violations.append(
            f"OPEN division Track 2 controllability violation: inverse_dynamics_action_cosine_sim="
            f"{gate_b_obj['inverse_dynamics_action_cosine_sim']} < 0.990."
        )
    else:
      delta_sr = gate_b_obj.get("delta_success_rate_pct")
      if delta_sr is None or float(delta_sr) < -5.0 - 1e-9:
        violations.append(
            f"OPEN division Track 1 closed-loop Gate B violation for recipe_id={recipe_id!r}: "
            f"delta_success_rate_pct={delta_sr!r} < -5.0 percentage points."
        )

  if submission_phase == "PROPRIETARY_AUDIT_ONLY":
    recipe_badge = "PROPRIETARY_AUDIT_ONLY"
  elif submission_phase == "COMMIT_ONLY" and not has_open_recipe:
    recipe_badge = "COMMIT_REVEAL_PENDING"
  elif status == RunStatus.CERTIFIED.value and has_open_recipe:
    recipe_badge = "VERIFIED_OPEN_RECIPE"
  else:
    recipe_badge = "OPEN_RECIPE_INCLUDED" if has_open_recipe else "UNVERIFIED_BUNDLE"

  merged_context = {
      "receipt": receipt,
      "body": body,
      "sweep_data": sweep_data,
      "sku_spec": sku_spec,
      "recipe_id": recipe_id,
      "reported_recipe_id": raw_recipe_id,
      "receipt_status": status,
      "hardware_sku": hardware_sku,
      "num_devices": num_devices,
      "vram_gb": round(effective_vram_gb, 2),
      "bf16_tflops": round(effective_bf16_tflops, 2),
      "submission_meta": normalized_meta,
      "submission_phase": submission_phase,
      "recipe_badge": recipe_badge,
      "recipe_commit_sha256": normalized_meta.get("recipe_commit_sha256"),
      "engine_tag": normalized_meta.get("engine_tag"),
      "submitter_team": normalized_meta.get("team"),
  }
  return len(violations) == 0, violations, merged_context


def _reconstruct_operating_points(
    ctx: Dict[str, Any],
    track: str,
    model_id: str,
    division: str,
) -> Tuple[str, str, str, List[Dict[str, Any]]]:
  """Extract or reconstruct enriched operating points with all 4 SemiAnalysis normalizations."""
  receipt = ctx["receipt"]
  body = ctx["body"]
  sweep_data = ctx["sweep_data"]
  hardware_sku = ctx["hardware_sku"]
  recipe_id = ctx["recipe_id"]

  wm = is_wm_track(track)
  x_key = "hz_plan" if wm else "hz_react"
  y_key = "predictions_per_sec" if wm else "actions_per_sec"
  latency_key = "rollout_latency_p99_ms" if wm else "obs_to_action_p99_ms"

  raw_points = (
      receipt.get("operating_points")
      if isinstance(receipt.get("operating_points"), list) and receipt.get("operating_points")
      else (
          sweep_data.get("operating_points")
          if isinstance(sweep_data.get("operating_points"), list) and sweep_data.get("operating_points")
          else []
      )
  )

  points: List[Dict[str, Any]] = []
  _sla_p99_rec, _sla_hz_rec, locked_k_rec = get_gapless_sla_for_model(track, model_id)
  if raw_points:
    exec_prov_r = (
        receipt.get("execution_provenance")
        or body.get("execution_provenance")
        or sweep_data.get("execution_provenance")
        or {}
    )
    top_power_prov_r = exec_prov_r.get("power") if isinstance(exec_prov_r, dict) else None
    for pt in raw_points:
      item = dict(pt)
      item.pop("performance_cap_evaluation", None)
      item.pop("co_located_gate_a", None)
      item.pop("runs", None)
      item["recipe_id"] = recipe_id
      item["track"] = track
      item["model_id"] = model_id
      item["division"] = division
      item["hardware_sku"] = hardware_sku
      if x_key not in item or item.get(x_key) is None:
        x_val = item.get("hz_plan" if wm else "hz_react")
        if x_val is None:
          x_val = item.get("hz_react", item.get("x_reactivity_hz"))
        if x_val is not None:
          item[x_key] = float(x_val)
      wm_frame_origin = False
      if wm:
        h_f = max(float(item.get("horizon_frames", 16) or 16), 1.0)
        fps_v = item.get("generated_frames_per_sec", item.get("frames_per_sec", item.get("fps")))
        rec_ym = str(
            receipt.get("y_metric_name")
            or body.get("y_metric_name")
            or sweep_data.get("y_metric_name")
            or ""
        )
        if item.get("predictions_per_sec") is not None:
          eff_y = float(item["predictions_per_sec"])
        elif item.get("chunks_per_sec") is not None:
          eff_y = float(item["chunks_per_sec"])
          wm_frame_origin = True
        elif fps_v is not None:
          eff_y = round(float(fps_v) / h_f, 4)
          wm_frame_origin = True
        elif item.get("raw_per_chip_throughput") is not None:
          raw_v = float(item["raw_per_chip_throughput"])
          if rec_ym in ("fps", "generated_frames_per_sec", "frames_per_sec"):
            eff_y = round(raw_v / h_f, 4)
            wm_frame_origin = True
          else:
            eff_y = raw_v
        elif item.get("y_throughput") is not None:
          raw_v = float(item["y_throughput"])
          if rec_ym in ("fps", "generated_frames_per_sec", "frames_per_sec"):
            eff_y = round(raw_v / h_f, 4)
            wm_frame_origin = True
          else:
            eff_y = raw_v
        else:
          eff_y = None
        if eff_y is not None:
          gen_fps = float(fps_v) if fps_v is not None else round(float(eff_y) * h_f, 4)
          item["predictions_per_sec"] = round(float(eff_y), 4)
          item["generated_frames_per_sec"] = round(gen_fps, 4)
          item["fps"] = round(gen_fps, 4)
          item["raw_per_chip_throughput"] = round(float(eff_y), 4)
        eff_lat = item.get(
            "rollout_latency_p99_ms",
            item.get("obs_to_action_p99_ms", item.get("p99_latency_ms", item.get("ttff_p99_ms", item.get("ttfa_p99_ms")))),
        )
        if eff_lat is not None:
          item["rollout_latency_p99_ms"] = float(eff_lat)
          item.setdefault("ttff_p99_ms", round(float(eff_lat), 3))
          h_frames = max(int(item.get("horizon_frames", 16) or 16) - 1, 0)
          item.setdefault("observation_staleness_ms", round(float(eff_lat) + h_frames * (1000.0 / 24.0), 3))
      else:
        # RT Fix 4: Recompute executed_actions_per_sec from chunks_per_sec * locked K_eval when present
        cps_val = item.get("chunks_per_sec")
        if cps_val is not None and float(cps_val) > 0.0 and model_id in MODEL_LOCKED_K_EVAL:
          eff_y = round(float(cps_val) * float(locked_k_rec), 4)
          item["execute_horizon"] = locked_k_rec
          item["executed_actions_per_sec"] = eff_y
          item["actions_per_sec"] = float(item.get("actions_per_sec", eff_y))
          item["raw_per_chip_throughput"] = eff_y
        else:
          eff_y = item.get(
              "executed_actions_per_sec",
              item.get("actions_per_sec", item.get("raw_per_chip_throughput", item.get("y_throughput"))),
          )
          if eff_y is not None:
            item.setdefault("executed_actions_per_sec", float(eff_y))
            item.setdefault("actions_per_sec", float(eff_y))
            if item.get("raw_per_chip_throughput") is None:
              item["raw_per_chip_throughput"] = float(eff_y)
        eff_lat = item.get(
            "obs_to_action_p99_ms",
            item.get("ttfa_p99_ms", item.get("p99_latency_ms")),
        )
        if eff_lat is not None:
          item["obs_to_action_p99_ms"] = float(eff_lat)
          item.setdefault("ttfa_p99_ms", float(eff_lat))
          k_chunk = max(int(item.get("execute_horizon", item.get("chunk_size", locked_k_rec)) or locked_k_rec) - 1, 0)
          item.setdefault("observation_staleness_ms", round(float(eff_lat) + k_chunk * 50.0, 3))
      if eff_y is not None and hardware_sku in HARDWARE_SKU_SPECS:
        sa = compute_semianalysis_normalizations(
            raw_throughput=float(eff_y),
            hardware_sku=hardware_sku,
        )
        for sa_key in (
            "throughput_per_100_tops",
            "throughput_per_tbps_bw",
            "same_chip_speedup_vs_bf16_ref",
            "tco_efficiency_per_dollar",
            "throughput_per_dollar_hr",
            "units_per_dollar",
            "throughput_per_1k_usd",
        ):
          if (item.get(sa_key) is None or (wm_frame_origin and sa_key != "same_chip_speedup_vs_bf16_ref")) and sa.get(sa_key) is not None:
            item[sa_key] = sa[sa_key]
        ht_item = item.get("hardware_telemetry") if isinstance(item.get("hardware_telemetry"), dict) else {}
        sa_item = item.get("semianalysis_normalizations") if isinstance(item.get("semianalysis_normalizations"), dict) else {}
        pt_power_prov = (
            ht_item.get("power_provenance")
            or item.get("power_provenance")
            or sa_item.get("power_provenance")
            or top_power_prov_r
        )
        # RT Fix 1: Only MEASURED board telemetry may populate power_efficiency_per_watt
        if pt_power_prov != "MEASURED":
          item["power_efficiency_per_watt"] = None
          item["predictions_per_watt"] = None
          item["actions_per_watt"] = None
          item["frames_per_watt"] = None
          item["fps_per_watt"] = None
        elif wm:
          h_f_pw = max(float(item.get("horizon_frames", 16) or 16), 1.0)
          sys_w_m = ht_item.get("system_power_watts", ht_item.get("mean_power_watts", ht_item.get("mean_power_w")))
          if item.get("predictions_per_watt") is not None:
            item["power_efficiency_per_watt"] = float(item["predictions_per_watt"])
          elif not wm_frame_origin and item.get("power_efficiency_per_watt") is not None:
            item["power_efficiency_per_watt"] = float(item["power_efficiency_per_watt"])
            item["predictions_per_watt"] = float(item["power_efficiency_per_watt"])
          elif sys_w_m is not None and float(sys_w_m) > 0.0:
            pw_pred = round(float(eff_y) / float(sys_w_m), 4)
            item["power_efficiency_per_watt"] = pw_pred
            item["predictions_per_watt"] = pw_pred
          else:
            eff_watt_fps = item.get("fps_per_watt", item.get("frames_per_watt", item.get("power_efficiency_per_watt")))
            if eff_watt_fps is not None:
              pw_pred = round(float(eff_watt_fps) / h_f_pw, 4)
              item["power_efficiency_per_watt"] = pw_pred
              item["predictions_per_watt"] = pw_pred
        elif item.get("power_efficiency_per_watt") is None:
          eff_watt = item.get("actions_per_watt")
          if eff_watt is not None:
            item["power_efficiency_per_watt"] = float(eff_watt)
          else:
            sys_w_m = ht_item.get("system_power_watts", ht_item.get("mean_power_watts", ht_item.get("mean_power_w")))
            if sys_w_m is not None and float(sys_w_m) > 0.0:
              item["power_efficiency_per_watt"] = round(float(eff_y) / float(sys_w_m), 4)
      points.append(item)
  else:
    exec_prov = receipt.get("execution_provenance") or body.get("execution_provenance") or {}
    power_is_measured = (
        isinstance(exec_prov, dict) and exec_prov.get("power") == "MEASURED"
    )
    body_ym = str(body.get("y_metric_name") or receipt.get("y_metric_name") or "")
    for fpt in body.get("frontier_points", []):
      raw_tput_val = float(fpt.get("y_throughput", 0.0))
      if wm and body_ym in ("fps", "generated_frames_per_sec", "frames_per_sec"):
        raw_tput = round(raw_tput_val / 16.0, 4)
      else:
        raw_tput = raw_tput_val
      sa = compute_semianalysis_normalizations(
          raw_throughput=raw_tput,
          hardware_sku=hardware_sku,
      )
      if not power_is_measured:
        sa["power_efficiency_per_watt"] = None
        sa["system_power_watts"] = None
      lat_val = float(fpt.get("p99_latency_ms", 0.0))
      pt_dict: Dict[str, Any] = {
          "point_id": fpt.get("point_id", "P0"),
          "execution_id": fpt.get("execution_id", ""),
          "recipe_id": recipe_id,
          "track": track,
          "model_id": model_id,
          "division": division,
          "hardware_sku": hardware_sku,
          "gate_a_passed": bool(fpt.get("gate_a_passed", True)),
          x_key: float(fpt.get("x_reactivity_hz", 0.0)),
          y_key: raw_tput,
          latency_key: lat_val,
          "raw_per_chip_throughput": sa["raw_per_chip_throughput"],
          "power_efficiency_per_watt": sa["power_efficiency_per_watt"],
          "throughput_per_100_tops": sa["throughput_per_100_tops"],
          "throughput_per_tbps_bw": sa["throughput_per_tbps_bw"],
          "same_chip_speedup_vs_bf16_ref": sa["same_chip_speedup_vs_bf16_ref"],
          "throughput_per_1k_usd": sa["throughput_per_1k_usd"],
          "tco_efficiency_per_dollar": sa["tco_efficiency_per_dollar"],
          "throughput_per_dollar_hr": sa["throughput_per_dollar_hr"],
          "semianalysis_normalizations": sa,
      }
      if fpt.get("concurrency") is not None:
        pt_dict["concurrency"] = int(fpt["concurrency"])
      if fpt.get("p50_latency_ms") is not None:
        pt_dict["rollout_latency_p50_ms" if wm else "obs_to_action_p50_ms"] = float(fpt["p50_latency_ms"])
      if wm:
        pt_dict["predictions_per_sec"] = raw_tput
        pt_dict["generated_frames_per_sec"] = round(raw_tput * 16.0, 4)
        pt_dict["fps"] = round(raw_tput * 16.0, 4)
        pt_dict["ttff_p99_ms"] = round(lat_val, 3)
        pt_dict["observation_staleness_ms"] = round(lat_val + 15.0 * (1000.0 / 24.0), 3)
      else:
        pt_dict["executed_actions_per_sec"] = raw_tput
        pt_dict["ttfa_p99_ms"] = lat_val
        pt_dict["observation_staleness_ms"] = round(lat_val + max(locked_k_rec - 1, 0) * 50.0, 3)
      points.append(pt_dict)

  return x_key, y_key, latency_key, points


def _compute_4way_frontiers_and_slices(
    operating_points: Sequence[Dict[str, Any]],
    x_key: str,
    latency_key: str,
    track: str = "",
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
  """Compute all 4 SemiAnalysis Pareto frontiers and log-linear Iso-Reactivity slices."""
  frontiers_4way = extract_semianalysis_4way_pareto_frontiers(
      operating_points=operating_points,
      x_key=x_key,
      require_gate_a=True,
  )
  y_key_map = {
      "raw_per_chip_throughput": "raw_per_chip_throughput",
      "power_efficiency_per_watt": "power_efficiency_per_watt",
      "iso_compute_and_bw_efficiency": "throughput_per_100_tops",
      "tco_efficiency_per_dollar": "tco_efficiency_per_dollar",
  }
  slices_by_norm: Dict[str, Dict[str, Optional[float]]] = {}
  for norm_name, f_pts in frontiers_4way.items():
    t_key = y_key_map[norm_name]
    slices_by_norm[norm_name] = compute_iso_reactivity_slices(
        frontier_points=f_pts,
        latency_key=latency_key,
        throughput_key=t_key,
        track=track,
    )

  raw_slices = dict(slices_by_norm.get("raw_per_chip_throughput", {}))
  iso_summary: Dict[str, Any] = dict(raw_slices)
  if is_wm_track(track):
    iso_summary["1Hz"] = raw_slices.get("Iso-1Hz")
    iso_summary["1.5Hz"] = raw_slices.get("Iso-1.5Hz")
    iso_summary["2Hz"] = raw_slices.get("Iso-2Hz")
    iso_summary["3Hz"] = raw_slices.get("Iso-3Hz")
    iso_summary["5Hz"] = raw_slices.get("Iso-5Hz")
    iso_summary["10Hz"] = raw_slices.get("Iso-10Hz")
  else:
    iso_summary["10Hz"] = raw_slices.get("Iso-10Hz")
    iso_summary["20Hz"] = raw_slices.get("Iso-20Hz")
    iso_summary["30Hz"] = raw_slices.get("Iso-30Hz")
    iso_summary["50Hz"] = raw_slices.get("Iso-50Hz")
  iso_summary["by_normalization"] = slices_by_norm
  return frontiers_4way, iso_summary


def _extract_point_concurrency(pt: Dict[str, Any]) -> Optional[int]:
  for k in ("concurrency", "batch_size"):
    val = pt.get(k)
    if isinstance(val, (int, float)) and not isinstance(val, bool) and int(val) > 0:
      return int(val)
  pid = str(pt.get("point_id", "") or "").strip()
  m = re.fullmatch(r"[CBcb](\d+)", pid)
  if m:
    return int(m.group(1))
  return None


def compute_realtime_dominated_hypervolume(
    rate_tput_points: Sequence[Tuple[float, float]],
    sla_min_rate_hz: float,
    *,
    u_ref: float = HYPERVOLUME_U_REF,
) -> float:
  """Compute exact 2D dominated hypervolume over normalized reactivity `u = rate_hz / sla_min_rate_hz >= 1.0`.

  Evaluated strictly over the SLA-admissible discrete point set `Q = {p : r >= sla_min_rate_hz, y > 0}`
  against reference point `(u_ref = 0.5, y_ref = 0.0)` so even a single admissible point placed
  exactly on the SLA boundary (`u = 1.0`) earns positive area `(1.0 - 0.5) * y > 0`.
  """
  sla = float(sla_min_rate_hz)
  if sla <= 0.0:
    return 0.0
  filtered = [
      (float(r) / sla, float(t))
      for r, t in rate_tput_points
      if float(r) >= sla - 1e-9 and float(t) > 0.0
  ]
  if not filtered:
    return 0.0

  # Sort by strictly descending normalized reactivity u >= 1.0, breaking ties by descending throughput
  filtered.sort(key=lambda item: (-item[0], -item[1]))
  pareto_desc: List[Tuple[float, float]] = []
  best_tput = 0.0
  for u_raw, t_val in filtered:
    u_val = max(1.0, u_raw)
    if pareto_desc and abs(u_val - pareto_desc[-1][0]) <= 1e-9:
      continue
    if t_val > best_tput + 1e-12:
      pareto_desc.append((u_val, t_val))
      best_tput = t_val

  hv = 0.0
  m = len(pareto_desc)
  for idx, (u_i, y_i) in enumerate(pareto_desc):
    next_u = pareto_desc[idx + 1][0] if (idx + 1) < m else float(u_ref)
    width = u_i - max(next_u, float(u_ref))
    if width > 0.0:
      hv += width * y_i
  return round(hv, 4)


def compute_submission_award_metrics(
    operating_points: Sequence[Dict[str, Any]],
    *,
    track: str,
    model_id: Optional[str] = None,
    x_key: str,
    latency_key: str,
    exec_prov: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Compute the 3 deterministic SLA-anchored scalar award metrics + Real-Time Dominated Hypervolume tie-breaker."""
  wm = is_wm_track(track)
  sla_max_p99, sla_min_rate, locked_k = get_gapless_sla_for_model(track, model_id)
  rate_unit = "Predictions/s" if wm else "Chunks/s"
  tput_unit = "Predictions/s" if wm else "Actions/s"
  eff_unit = "Predictions/Joule" if wm else "Actions/Joule"

  raw_admissible = [
      p
      for p in operating_points
      if isinstance(p, dict)
      and bool(p.get("gate_a_passed", True))
      and p.get("variance_passed", True) is not False
  ]
  # RT Fix 3: Restrict scoring strictly to discrete canonical concurrency grid points {1, 2, 4, 8, 16}
  has_any_concurrency = any(_extract_point_concurrency(p) is not None for p in raw_admissible)
  if has_any_concurrency:
    admissible_pts = [
        p for p in raw_admissible if _extract_point_concurrency(p) in CANONICAL_CONCURRENCY_GRID
    ]
  else:
    admissible_pts = list(raw_admissible)

  def _pt_p99(p: Dict[str, Any]) -> Optional[float]:
    raw = p.get(latency_key, p.get("p99_latency_ms"))
    if raw is None:
      return None
    val = float(raw)
    return val if val > 0.0 else None

  def _pt_p50(p: Dict[str, Any], fallback_p99: float) -> float:
    for pk in (
        ("rollout_latency_p50_ms", "obs_to_action_p50_ms", "p50_latency_ms")
        if wm
        else ("obs_to_action_p50_ms", "p50_latency_ms")
    ):
      raw = p.get(pk)
      if raw is not None and float(raw) > 0.0:
        return float(raw)
    return fallback_p99

  def _pt_tput(p: Dict[str, Any]) -> float:
    if wm:
      if p.get("predictions_per_sec") is not None:
        return float(p["predictions_per_sec"])
      if p.get("chunks_per_sec") is not None:
        return float(p["chunks_per_sec"])
      h_f = max(float(p.get("horizon_frames", 16) or 16), 1.0)
      fps_v = float(
          p.get("generated_frames_per_sec", p.get("fps", p.get("raw_per_chip_throughput", 0.0))) or 0.0
      )
      return fps_v / h_f
    cps = p.get("chunks_per_sec")
    if cps is not None and float(cps) > 0.0 and model_id and _canonicalize_model_id(model_id) in MODEL_LOCKED_K_EVAL:
      return float(cps) * float(locked_k)
    return float(
        p.get("executed_actions_per_sec", p.get("actions_per_sec", p.get("raw_per_chip_throughput", 0.0))) or 0.0
    )

  # (a) Award A: Interactive Reactivity Rank at C=1 (score = 1000 / p99_ms, bimodal gate p99/p50 <= 1.5)
  c1_candidates: List[Tuple[float, float, Dict[str, Any]]] = []
  for p in admissible_pts:
    c_val = _extract_point_concurrency(p)
    if c_val != 1:
      continue
    p99 = _pt_p99(p)
    if p99 is None:
      continue
    p50 = _pt_p50(p, p99)
    c1_candidates.append((p99, p50, p))
  if not c1_candidates and admissible_pts and not has_any_concurrency:
    p0 = admissible_pts[0]
    p99_0 = _pt_p99(p0)
    if p99_0 is not None:
      c1_candidates.append((p99_0, _pt_p50(p0, p99_0), p0))

  if c1_candidates:
    best_p99, best_p50, _best_c1 = min(c1_candidates, key=lambda item: (item[0], item[1]))
    tail_ratio = best_p99 / max(best_p50, 1e-9)
    tail_ok = tail_ratio <= (AWARD_A_MAX_TAIL_RATIO_C1 + 1e-9)
    interactive_obj: Dict[str, Any] = {
        "eligible": bool(tail_ok),
        "c1_interactive_rate": round(1000.0 / best_p99, 4),
        "c1_p99_rate_hz": round(1000.0 / best_p99, 4),
        "c1_p50_rate_hz": round(1000.0 / best_p50, 4),
        "c1_p50_ms": round(best_p50, 3),
        "c1_p99_ms": round(best_p99, 3),
        "c1_tail_ratio_p99_over_p50": round(tail_ratio, 4),
        "tail_ratio_gate_passed": bool(tail_ok),
    }
  else:
    interactive_obj = {
        "eligible": False,
        "c1_interactive_rate": None,
        "c1_p99_rate_hz": None,
        "c1_p50_rate_hz": None,
        "c1_p50_ms": None,
        "c1_p99_ms": None,
        "c1_tail_ratio_p99_over_p50": None,
        "tail_ratio_gate_passed": False,
    }

  # SLA-admissible discrete points: strictly p99 <= sla_max_p99_ms
  sla_pts: List[Tuple[float, float, float, Dict[str, Any]]] = []
  for p in admissible_pts:
    p99 = _pt_p99(p)
    if p99 is None or p99 > sla_max_p99 + 1e-9:
      continue
    rate_hz = 1000.0 / p99
    tp = _pt_tput(p)
    if tp > 0.0:
      sla_pts.append((tp, rate_hz, p99, p))

  # (b) Award B: Real-Time Fleet Capacity Rank (SLA-Bounded Max Throughput on discrete {1,2,4,8,16})
  if sla_pts:
    best_tp, best_rate, best_lat, best_tp_pt = max(
        sla_pts, key=lambda item: (item[0], item[1], -item[2])
    )
    fleet_obj: Dict[str, Any] = {
        "eligible": True,
        "sla_max_throughput": round(best_tp, 4),
        "concurrency": _extract_point_concurrency(best_tp_pt) or 1,
        "p99_latency_ms": round(best_lat, 3),
        "rate_hz": round(best_rate, 4),
    }
    if wm:
      h_f = max(float(best_tp_pt.get("horizon_frames", 16) or 16), 1.0)
      fleet_obj["sla_max_frames_per_sec"] = round(best_tp * h_f, 4)
  else:
    fleet_obj = {
        "eligible": False,
        "sla_max_throughput": None,
        "concurrency": None,
        "p99_latency_ms": None,
        "rate_hz": None,
    }

  # (c) Award C: Green Embodied Efficiency Rank (SLA-Bounded Peak Efficiency, MEASURED telemetry + constant power limit only)
  top_power_prov = (exec_prov or {}).get("power") if isinstance(exec_prov, dict) else None
  sweep_power_limits: List[float] = []
  for p in admissible_pts:
    ht_l = p.get("hardware_telemetry") if isinstance(p.get("hardware_telemetry"), dict) else {}
    lim = ht_l.get("power_limit_w", p.get("power_limit_w"))
    if isinstance(lim, (int, float)) and not isinstance(lim, bool) and float(lim) > 0.0:
      sweep_power_limits.append(float(lim))
  power_limit_constant_ok = (
      (max(sweep_power_limits) - min(sweep_power_limits)) <= 1.0
      if len(sweep_power_limits) >= 2
      else True
  )

  green_candidates: List[Tuple[float, float, Optional[float], Dict[str, Any]]] = []
  if power_limit_constant_ok:
    for tp, _rate, p99, p in sla_pts:
      ht = p.get("hardware_telemetry") if isinstance(p.get("hardware_telemetry"), dict) else {}
      sa_p = p.get("semianalysis_normalizations") if isinstance(p.get("semianalysis_normalizations"), dict) else {}
      pt_prov = (
          ht.get("power_provenance")
          or p.get("power_provenance")
          or sa_p.get("power_provenance")
          or top_power_prov
      )
      if pt_prov != "MEASURED":
        continue
      sys_w = ht.get("system_power_watts", ht.get("mean_power_watts", ht.get("mean_power_w", sa_p.get("system_power_watts"))))
      raw_pw = p.get("power_efficiency_per_watt")
      if sys_w is not None and float(sys_w) > 0.0:
        eff_per_j = float(tp) / float(sys_w)
        sys_w_f: Optional[float] = float(sys_w)
      elif p.get("predictions_per_watt") is not None and float(p["predictions_per_watt"]) > 0.0:
        eff_per_j = float(p["predictions_per_watt"])
        sys_w_f = None
      elif raw_pw is not None and float(raw_pw) > 0.0:
        pw_val = float(raw_pw)
        if wm and p.get("predictions_per_sec") is None:
          eff_per_j = pw_val / max(float(p.get("horizon_frames", 16) or 16), 1.0)
        else:
          eff_per_j = pw_val
        sys_w_f = None
      else:
        continue
      green_candidates.append((eff_per_j, p99, sys_w_f, p))

  if green_candidates:
    best_eff, best_eff_p99, best_sys_w, best_eff_pt = max(
        green_candidates, key=lambda item: (item[0], -item[1])
    )
    green_obj: Dict[str, Any] = {
        "eligible": True,
        "sla_peak_efficiency_per_joule": round(best_eff, 6),
        "concurrency": _extract_point_concurrency(best_eff_pt) or 1,
        "p99_latency_ms": round(best_eff_p99, 3),
        "system_power_watts": round(best_sys_w, 2) if best_sys_w is not None else None,
        "power_provenance": "MEASURED",
        "power_limit_constant_ok": True,
    }
    if wm:
      h_f = max(float(best_eff_pt.get("horizon_frames", 16) or 16), 1.0)
      green_obj["sla_peak_frames_per_joule"] = round(best_eff * h_f, 6)
  else:
    green_obj = {
        "eligible": False,
        "sla_peak_efficiency_per_joule": None,
        "concurrency": None,
        "p99_latency_ms": None,
        "system_power_watts": None,
        "power_provenance": str(top_power_prov or "UNAVAILABLE"),
        "power_limit_constant_ok": bool(power_limit_constant_ok),
    }

  # (d) Real-Time Dominated Hypervolume Tie-Breaker (normalized u_ref = 0.5)
  hv = compute_realtime_dominated_hypervolume(
      [(rate_hz, tp) for tp, rate_hz, _lat, _p in sla_pts],
      sla_min_rate_hz=sla_min_rate,
      u_ref=HYPERVOLUME_U_REF,
  )

  return {
      "locked_k_eval": locked_k,
      "sla_max_p99_ms": sla_max_p99,
      "sla_min_rate_hz": sla_min_rate,
      "hypervolume_u_ref": HYPERVOLUME_U_REF,
      "rate_unit": rate_unit,
      "throughput_unit": tput_unit,
      "efficiency_unit": eff_unit,
      "hypervolume_unit": f"norm_sla_rate*({tput_unit})",
      "interactive_reactivity_c1": interactive_obj,
      "realtime_fleet_capacity_sla": fleet_obj,
      "green_embodied_efficiency_sla": green_obj,
      "realtime_dominated_hypervolume": hv,
  }


def compute_track_award_rankings(
    verified_submissions: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
  """Compute deterministic 1-indexed award rankings per `(track, model_id, division)` group."""
  by_group: Dict[str, List[Dict[str, Any]]] = {}
  for sub in verified_submissions:
    m_id = str(sub.get("model_id") or sub.get("baseline_model_id") or "")
    gid = f"{sub['track']}__{m_id}__{sub['division']}"
    by_group.setdefault(gid, []).append(sub)

  rankings: Dict[str, Dict[str, Any]] = {}
  for gid, subs in sorted(by_group.items()):
    sample_am = subs[0].get("award_metrics", {})

    # 1. Award A: Interactive Reactivity Rank at C=1 (lowest p99 / highest 1000/p99, tie p50, tie HV)
    react_pool = [
        s for s in subs if s.get("award_metrics", {}).get("interactive_reactivity_c1", {}).get("eligible")
    ]
    react_pool.sort(
        key=lambda s: (
            float(s["award_metrics"]["interactive_reactivity_c1"]["c1_p99_ms"]),
            float(s["award_metrics"]["interactive_reactivity_c1"]["c1_p50_ms"]),
            -float(s["award_metrics"].get("realtime_dominated_hypervolume", 0.0)),
            str(s.get("recipe_id", "")),
        )
    )
    react_ranked = [
        {
            "rank": idx + 1,
            "recipe_id": s["recipe_id"],
            "hardware_sku": s["hardware_sku"],
            "submission_phase": s.get("submission_phase", "OPEN_RECIPE"),
            "recipe_badge": s.get("recipe_badge", ""),
            "c1_interactive_rate": s["award_metrics"]["interactive_reactivity_c1"]["c1_interactive_rate"],
            "c1_p99_rate_hz": s["award_metrics"]["interactive_reactivity_c1"]["c1_p99_rate_hz"],
            "c1_p50_ms": s["award_metrics"]["interactive_reactivity_c1"]["c1_p50_ms"],
            "c1_p99_ms": s["award_metrics"]["interactive_reactivity_c1"]["c1_p99_ms"],
            "c1_tail_ratio_p99_over_p50": s["award_metrics"]["interactive_reactivity_c1"]["c1_tail_ratio_p99_over_p50"],
            "realtime_dominated_hypervolume": s["award_metrics"]["realtime_dominated_hypervolume"],
        }
        for idx, s in enumerate(react_pool)
    ]

    # 2. Real-Time Fleet Capacity Rank (SLA-Bounded Max Throughput, tie HV, tie p99)
    fleet_pool = [
        s for s in subs if s.get("award_metrics", {}).get("realtime_fleet_capacity_sla", {}).get("eligible")
    ]
    fleet_pool.sort(
        key=lambda s: (
            -float(s["award_metrics"]["realtime_fleet_capacity_sla"]["sla_max_throughput"]),
            -float(s["award_metrics"].get("realtime_dominated_hypervolume", 0.0)),
            float(s["award_metrics"]["realtime_fleet_capacity_sla"]["p99_latency_ms"]),
            str(s.get("recipe_id", "")),
        )
    )
    fleet_ranked = [
        {
            "rank": idx + 1,
            "recipe_id": s["recipe_id"],
            "hardware_sku": s["hardware_sku"],
            "submission_phase": s.get("submission_phase", "OPEN_RECIPE"),
            "recipe_badge": s.get("recipe_badge", ""),
            "sla_max_throughput": s["award_metrics"]["realtime_fleet_capacity_sla"]["sla_max_throughput"],
            "concurrency": s["award_metrics"]["realtime_fleet_capacity_sla"]["concurrency"],
            "p99_latency_ms": s["award_metrics"]["realtime_fleet_capacity_sla"]["p99_latency_ms"],
            "realtime_dominated_hypervolume": s["award_metrics"]["realtime_dominated_hypervolume"],
        }
        for idx, s in enumerate(fleet_pool)
    ]

    # 3. Green Embodied Efficiency Rank (SLA-Bounded Peak Efficiency with MEASURED telemetry, tie HV, tie p99)
    green_pool = [
        s for s in subs if s.get("award_metrics", {}).get("green_embodied_efficiency_sla", {}).get("eligible")
    ]
    green_pool.sort(
        key=lambda s: (
            -float(s["award_metrics"]["green_embodied_efficiency_sla"]["sla_peak_efficiency_per_joule"]),
            -float(s["award_metrics"].get("realtime_dominated_hypervolume", 0.0)),
            float(s["award_metrics"]["green_embodied_efficiency_sla"]["p99_latency_ms"]),
            str(s.get("recipe_id", "")),
        )
    )
    green_ranked = [
        {
            "rank": idx + 1,
            "recipe_id": s["recipe_id"],
            "hardware_sku": s["hardware_sku"],
            "submission_phase": s.get("submission_phase", "OPEN_RECIPE"),
            "recipe_badge": s.get("recipe_badge", ""),
            "sla_peak_efficiency_per_joule": s["award_metrics"]["green_embodied_efficiency_sla"]["sla_peak_efficiency_per_joule"],
            "concurrency": s["award_metrics"]["green_embodied_efficiency_sla"]["concurrency"],
            "p99_latency_ms": s["award_metrics"]["green_embodied_efficiency_sla"]["p99_latency_ms"],
            "system_power_watts": s["award_metrics"]["green_embodied_efficiency_sla"]["system_power_watts"],
            "realtime_dominated_hypervolume": s["award_metrics"]["realtime_dominated_hypervolume"],
        }
        for idx, s in enumerate(green_pool)
    ]

    rankings[gid] = {
        "group_id": gid,
        "track": subs[0]["track"],
        "model_id": str(subs[0].get("model_id") or subs[0].get("baseline_model_id") or ""),
        "division": subs[0]["division"],
        "sla_max_p99_ms": sample_am.get("sla_max_p99_ms"),
        "sla_min_rate_hz": sample_am.get("sla_min_rate_hz"),
        "rate_unit": sample_am.get("rate_unit"),
        "throughput_unit": sample_am.get("throughput_unit"),
        "efficiency_unit": sample_am.get("efficiency_unit"),
        "interactive_reactivity_rank": react_ranked,
        "realtime_fleet_capacity_rank": fleet_ranked,
        "green_embodied_efficiency_rank": green_ranked,
    }
  return rankings


def _discover_submission_manifests(sub_root: Path) -> List[Path]:
  """Discover one signed manifest per measured submission folder (`receipt.json` > `submission_receipt.json` > `bundle_summary.json`)."""
  if not sub_root.exists():
    return []
  dirs_seen: Dict[Path, Path] = {}
  priority = ("receipt.json", "submission_receipt.json", "bundle_summary.json")
  for fname in priority:
    for p in sorted(sub_root.rglob(fname)):
      if not p.is_file():
        continue
      parent = p.parent.resolve()
      if parent not in dirs_seen:
        dirs_seen[parent] = p
  return sorted(dirs_seen.values(), key=lambda p: p.as_posix())


def compile_submissions_to_leaderboard(
    submissions_dir: Path | str,
    output_json_path: Path | str,
    strict_verify: bool = True,
) -> Dict[str, Any]:
  """Recursively discover `**/receipt.json` inside `submissions_dir`, verify cryptographic integrity + `CAP_SINGLE_CHIP` + `G1`-`G4`, and compile 4-way SemiAnalysis Pareto frontiers."""
  sub_root = Path(submissions_dir)
  out_path = Path(output_json_path)
  repo_root = Path(__file__).resolve().parents[1]
  sub_root_resolved = sub_root.resolve()

  def _portable_path(p: Path) -> str:
    pr = p.resolve()
    try:
      return pr.relative_to(repo_root).as_posix()
    except ValueError:
      pass
    if pr == sub_root_resolved:
      return sub_root.name or "submissions"
    try:
      return (Path(sub_root.name or "submissions") / pr.relative_to(sub_root_resolved)).as_posix()
    except ValueError:
      return p.name or "external"

  receipt_paths = _discover_submission_manifests(sub_root)

  verified_submissions: List[Dict[str, Any]] = []
  rejected_submissions: List[Dict[str, Any]] = []

  for rpath in receipt_paths:
    passed, violations, ctx = _audit_single_receipt(rpath)
    if not passed:
      reason_str = "; ".join(violations)
      rejected_submissions.append({
          "receipt_path": _portable_path(rpath),
          "recipe_id": ctx.get("recipe_id", rpath.parent.name),
          "hardware_sku": ctx.get("hardware_sku", "UNKNOWN"),
          "status": RunStatus.REJECTED.value,
          "reason": reason_str,
          "violations": violations,
      })
      continue

    receipt = ctx["receipt"]
    body = ctx["body"]
    sweep_data = ctx["sweep_data"]
    sku_spec = ctx["sku_spec"]
    recipe_id = ctx["recipe_id"]
    hardware_sku = ctx["hardware_sku"]

    raw_model_id = str(
        receipt.get("baseline_model_id")
        or receipt.get("model_id")
        or body.get("baseline_model_id")
        or body.get("model_id")
        or sweep_data.get("baseline_model_id")
        or sweep_data.get("model_id")
        or (sweep_data.get("caps") or {}).get("model_id")
        or ""
    )
    model_id = _canonicalize_model_id(raw_model_id)
    raw_track = str(receipt.get("track") or body.get("track") or sweep_data.get("track") or "")
    track = _canonicalize_track(raw_track, model_id)

    precision = str(receipt.get("precision") or body.get("precision") or sweep_data.get("precision") or "BF16")
    division = str(
        receipt.get("division")
        or body.get("division")
        or sweep_data.get("division")
        or (ctx.get("submission_meta") or {}).get("division")
        or (
            DivisionType.OPEN.value
            if ("open" in recipe_id.lower() or precision.upper() not in ("BF16", "FP16", "FP32"))
            else DivisionType.CLOSED.value
        )
    ).upper()

    x_key, y_key, latency_key, op_points = _reconstruct_operating_points(
        ctx=ctx,
        track=track,
        model_id=model_id,
        division=division,
    )
    frontiers_4way, iso_slices = _compute_4way_frontiers_and_slices(
        operating_points=op_points,
        x_key=x_key,
        latency_key=latency_key,
        track=track,
    )
    raw_frontier = frontiers_4way["raw_per_chip_throughput"]

    anchor_pt = raw_frontier[0] if raw_frontier else op_points[0]
    p99_latency_ms = round(float(anchor_pt[latency_key]), 3)
    anchor_hz = round(float(anchor_pt[x_key]), 4)
    anchor_raw = round(float(anchor_pt["raw_per_chip_throughput"]), 4)
    peak_hz = round(max(float(p[x_key]) for p in raw_frontier), 4) if raw_frontier else 0.0
    peak_pt = (
        max(raw_frontier, key=lambda p: (float(p["raw_per_chip_throughput"]), float(p[x_key])))
        if raw_frontier
        else op_points[0]
    )
    peak_raw = round(float(peak_pt["raw_per_chip_throughput"]), 4)
    peak_pt_c = int(peak_pt.get("concurrency", peak_pt.get("batch_size", 1)))
    peak_pt_hz = round(float(peak_pt[x_key]), 4)
    peak_pt_p99 = round(float(peak_pt[latency_key]), 3)

    wm = is_wm_track(track)
    _sla_p99_cut, cutoff_hz, _lk_cut = get_gapless_sla_for_model(track, model_id)
    qualifying_pool = [
        p for p in (raw_frontier or op_points)
        if p.get("gate_a_passed", True) and float(p.get(x_key, 0.0)) + 1e-9 >= cutoff_hz
    ]
    if qualifying_pool:
      best_at_cutoff = max(
          qualifying_pool,
          key=lambda p: (float(p["raw_per_chip_throughput"]), float(p[x_key])),
      )
      meets_cutoff = True
      cutoff_c: Optional[int] = int(best_at_cutoff.get("concurrency", best_at_cutoff.get("batch_size", 1)))
      cutoff_p99: Optional[float] = round(float(best_at_cutoff[latency_key]), 3)
      cutoff_x: Optional[float] = round(float(best_at_cutoff[x_key]), 4)
      cutoff_y: Optional[float] = round(float(best_at_cutoff["raw_per_chip_throughput"]), 4)
    else:
      meets_cutoff = False
      cutoff_c = None
      cutoff_p99 = None
      cutoff_x = None
      cutoff_y = None

    watt_vals = [
        float(p["power_efficiency_per_watt"])
        for p in raw_frontier
        if p.get("power_efficiency_per_watt") is not None
    ] or [
        float(p["power_efficiency_per_watt"])
        for p in op_points
        if p.get("power_efficiency_per_watt") is not None
    ]
    peak_watt: Optional[float] = round(max(watt_vals), 4) if watt_vals else None
    tops_vals = [float(p["throughput_per_100_tops"]) for p in op_points if p.get("throughput_per_100_tops") is not None]
    peak_tops = round(max(tops_vals), 4) if tops_vals else 0.0
    tbps_vals = [float(p["throughput_per_tbps_bw"]) for p in op_points if p.get("throughput_per_tbps_bw") is not None]
    peak_tbps = round(max(tbps_vals), 4) if tbps_vals else 0.0
    speedup_vals = [
        float(p["same_chip_speedup_vs_bf16_ref"])
        for p in op_points
        if p.get("same_chip_speedup_vs_bf16_ref") is not None
    ]
    peak_speedup = round(max(speedup_vals), 4) if speedup_vals else 1.0
    tco_rate_vals = [
        float(p["tco_efficiency_per_dollar"])
        for p in op_points
        if p.get("tco_efficiency_per_dollar") is not None
    ]
    peak_tco_rate = round(max(tco_rate_vals), 4) if tco_rate_vals else 0.0
    units_per_dollar_vals = [
        float(p.get("units_per_dollar", p.get("throughput_per_dollar_hr")))
        for p in op_points
        if p.get("units_per_dollar", p.get("throughput_per_dollar_hr")) is not None
    ]
    peak_units_per_dollar = round(max(units_per_dollar_vals), 2) if units_per_dollar_vals else 0.0
    tco_1k_vals = [
        float(p["throughput_per_1k_usd"])
        for p in op_points
        if p.get("throughput_per_1k_usd") is not None
    ]
    peak_tco_1k = round(max(tco_1k_vals), 4) if tco_1k_vals else 0.0

    vendor_raw = str(sku_spec.get("vendor", "UNKNOWN"))
    vendor_fam = _vendor_family(vendor_raw)
    raw_exec_prov = (
        receipt.get("execution_provenance")
        or body.get("execution_provenance")
        or sweep_data.get("execution_provenance")
        or dict(ANALYTIC_REFERENCE_PROVENANCE)
    )
    exec_prov = dict(raw_exec_prov) if isinstance(raw_exec_prov, dict) else dict(ANALYTIC_REFERENCE_PROVENANCE)
    if (
        exec_prov.get("loadgen") == "REAL_WALLCLOCK"
        and exec_prov.get("policy_is_synthetic") is False
    ):
      exec_prov.setdefault("is_live_empirical", True)

    award_metrics = compute_submission_award_metrics(
        operating_points=op_points,
        track=track,
        model_id=model_id,
        x_key=x_key,
        latency_key=latency_key,
        exec_prov=exec_prov,
    )

    verified_submissions.append({
        "recipe_id": recipe_id,
        "reported_recipe_id": ctx.get("reported_recipe_id", recipe_id),
        "receipt_path": _portable_path(rpath),
        "attestation_sha256": str(receipt.get("attestation_sha256", "")),
        "status": RunStatus.CERTIFIED.value,
        "receipt_status": ctx.get("receipt_status", RunStatus.CERTIFIED.value),
        "submission_phase": ctx.get("submission_phase", "OPEN_RECIPE"),
        "recipe_badge": ctx.get("recipe_badge", "OPEN_RECIPE_INCLUDED"),
        "recipe_commit_sha256": ctx.get("recipe_commit_sha256"),
        "engine_tag": ctx.get("engine_tag"),
        "submitter_team": ctx.get("submitter_team"),
        "execution_provenance": exec_prov,
        "track": track,
        "model_id": model_id,
        "baseline_model_id": raw_model_id,
        "model_name": str(receipt.get("model_name") or body.get("model_name") or recipe_id),
        "division": division,
        "hardware_sku": hardware_sku,
        "hardware_display_name": str(sku_spec.get("display_name", hardware_sku)),
        "vendor": vendor_raw,
        "vendor_family": vendor_fam,
        "precision": precision,
        "perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "num_devices": ctx["num_devices"],
        "vram_gb": ctx["vram_gb"],
        "bf16_tflops": ctx["bf16_tflops"],
        "x_metric_name": x_key,
        "y_metric_name": y_key,
        "latency_metric_name": latency_key,
        "p99_latency_ms": p99_latency_ms,
        "single_stream_reactivity_hz": anchor_hz,
        "single_stream_throughput": anchor_raw,
        "peak_reactivity_hz": peak_hz,
        "qualifying_cutoff_hz": cutoff_hz,
        "meets_realtime_cutoff": meets_cutoff,
        "concurrency_at_cutoff": cutoff_c,
        "p99_latency_at_cutoff_ms": cutoff_p99,
        "reactivity_at_cutoff_hz": cutoff_x,
        "output_at_cutoff": cutoff_y,
        "raw_per_chip_throughput": peak_raw,
        "peak_point_concurrency": peak_pt_c,
        "peak_point_reactivity_hz": peak_pt_hz,
        "peak_point_p99_latency_ms": peak_pt_p99,
        "power_efficiency_per_watt": peak_watt,
        "throughput_per_100_tops": peak_tops,
        "throughput_per_tbps_bw": peak_tbps,
        "same_chip_speedup_vs_bf16_ref": peak_speedup,
        "tco_efficiency_per_dollar": peak_tco_rate,
        "throughput_per_dollar_hr": peak_tco_rate,
        "units_per_dollar": peak_units_per_dollar,
        "throughput_per_1k_usd": peak_tco_1k,
        "award_metrics": award_metrics,
        "iso_reactivity_slices": iso_slices,
        "semianalysis_4way_pareto_frontiers": frontiers_4way,
        "pareto_frontier": raw_frontier,
        "operating_points": op_points,
    })

  if strict_verify and rejected_submissions:
    details = " | ".join(
        f"[{item['recipe_id']} ({item['receipt_path']})]: {item['reason']}"
        for item in rejected_submissions
    )
    raise ValueError(
        f"Leaderboard compilation aborted (strict_verify=True): {len(rejected_submissions)} submission receipt(s) "
        f"failed verification: {details}"
    )

  # Group operating points by (track, model_id, division)
  grouped_points: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
  grouped_subs: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
  model_points: Dict[str, List[Dict[str, Any]]] = {}
  model_subs: Dict[str, List[Dict[str, Any]]] = {}

  for sub in verified_submissions:
    gkey = (sub["track"], sub["model_id"], sub["division"])
    grouped_points.setdefault(gkey, []).extend(sub["operating_points"])
    grouped_subs.setdefault(gkey, []).append(sub)
    model_points.setdefault(sub["model_id"], []).extend(sub["operating_points"])
    model_subs.setdefault(sub["model_id"], []).append(sub)

  grouped_frontiers: Dict[str, Dict[str, Any]] = {}
  for (track, model_id, division), pts in sorted(grouped_points.items()):
    subs_in_group = grouped_subs[(track, model_id, division)]
    x_key = subs_in_group[0]["x_metric_name"]
    y_key = subs_in_group[0]["y_metric_name"]
    latency_key = subs_in_group[0]["latency_metric_name"]
    f4, slices = _compute_4way_frontiers_and_slices(
        pts, x_key=x_key, latency_key=latency_key, track=track
    )
    group_id = f"{track}__{model_id}__{division}"
    grouped_frontiers[group_id] = {
        "group_id": group_id,
        "track": track,
        "model_id": model_id,
        "division": division,
        "x_metric_name": x_key,
        "y_metric_name": y_key,
        "latency_metric_name": latency_key,
        "num_submissions": len(subs_in_group),
        "num_operating_points": len(pts),
        "hardware_skus": sorted({s["hardware_sku"] for s in subs_in_group}),
        "vendors": sorted({s["vendor_family"] for s in subs_in_group}),
        "semianalysis_4way_pareto_frontiers": f4,
        "pareto_frontier": f4["raw_per_chip_throughput"],
        "iso_reactivity_slices": slices,
        "operating_points": pts,
    }

  # Cross-chip combined view per model_id
  cross_chip_by_model: Dict[str, Dict[str, Any]] = {}
  for model_id, pts in sorted(model_points.items()):
    subs_for_model = model_subs[model_id]
    track = subs_for_model[0]["track"]
    x_key = subs_for_model[0]["x_metric_name"]
    y_key = subs_for_model[0]["y_metric_name"]
    latency_key = subs_for_model[0]["latency_metric_name"]
    f4, slices = _compute_4way_frontiers_and_slices(
        pts, x_key=x_key, latency_key=latency_key, track=track
    )

    div_breakdown: Dict[str, Any] = {}
    for div in (DivisionType.CLOSED.value, DivisionType.OPEN.value):
      gid = f"{track}__{model_id}__{div}"
      if gid in grouped_frontiers:
        div_breakdown[div] = {
            "num_submissions": grouped_frontiers[gid]["num_submissions"],
            "hardware_skus": grouped_frontiers[gid]["hardware_skus"],
            "vendors": grouped_frontiers[gid]["vendors"],
            "semianalysis_4way_pareto_frontiers": grouped_frontiers[gid]["semianalysis_4way_pareto_frontiers"],
            "iso_reactivity_slices": grouped_frontiers[gid]["iso_reactivity_slices"],
        }

    cross_chip_by_model[model_id] = {
        "model_id": model_id,
        "track": track,
        "x_metric_name": x_key,
        "y_metric_name": y_key,
        "latency_metric_name": latency_key,
        "num_submissions": len(subs_for_model),
        "num_operating_points": len(pts),
        "hardware_skus": sorted({s["hardware_sku"] for s in subs_for_model}),
        "vendors": sorted({s["vendor_family"] for s in subs_for_model}),
        "divisions": sorted({s["division"] for s in subs_for_model}),
        "semianalysis_4way_pareto_frontiers": f4,
        "pareto_frontier": f4["raw_per_chip_throughput"],
        "iso_reactivity_slices": slices,
        "divisions_breakdown": div_breakdown,
    }

  award_rankings = compute_track_award_rankings(verified_submissions)

  num_live = sum(
      1
      for s in verified_submissions
      if isinstance(s.get("execution_provenance"), dict)
      and bool(s["execution_provenance"].get("is_live_empirical", False))
  )
  num_analytic = len(verified_submissions) - num_live
  if len(verified_submissions) == 0:
    prov_banner = (
        "FOUNDING_SEASON_OPEN_AWAITING_SUBMISSIONS "
        "(0 PRE-POPULATED BASELINES; READY FOR PHASE-1 CONTESTANT SUBMISSIONS)"
    )
  elif num_live > 0 and num_analytic > 0:
    prov_banner = "MIXED_LIVE_AND_ANALYTIC_REFERENCE"
  elif num_live > 0:
    prov_banner = "LIVE_EMPIRICAL_RUNNER_EXECUTION"
  else:
    prov_banner = (
        "ANALYTIC_ROOFLINE_REFERENCE_PROJECTION "
        "(SEEDED REFERENCE BASELINES ARE ANALYTIC PROJECTIONS, NOT LIVE HARDWARE RUNS)"
    )
  prov_summary = {
      "num_live_empirical_submissions": num_live,
      "num_analytic_reference_submissions": num_analytic,
      "provenance_banner": prov_banner,
  }

  leaderboard_data: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "benchmark_suite": (
          "EmbodiedX / embodied-efficiency-bench v1.0 "
          "(CoRL 2026 Efficient Foundation Models for Real-Time Embodied AI)"
      ),
      "generated_at": now_pacific_iso(),
      "season1_official_hardware_targets": copy.deepcopy(SEASON1_OFFICIAL_HARDWARE_TARGETS),
      "submission_protocol": copy.deepcopy(SUBMISSION_PROTOCOL_SPEC),
      "gapless_realtime_sla": copy.deepcopy(GAPLESS_REALTIME_SLA_SPEC),
      "execution_provenance_summary": prov_summary,
      "performance_cap_envelope": PERFORMANCE_CAP_ENVELOPES[PerformanceCapId.CAP_SINGLE_CHIP.value].to_dict(),
      "submissions_dir": _portable_path(sub_root),
      "num_discovered_receipts": len(receipt_paths),
      "num_verified_submissions": len(verified_submissions),
      "num_certified_submissions": len(verified_submissions),
      "num_submissions": len(verified_submissions),
      "num_rejected_submissions": len(rejected_submissions),
      "canonical_models": list(CANONICAL_CHALLENGE_MODELS),
      "models_represented": sorted(cross_chip_by_model.keys()),
      "vendors_represented": sorted({s["vendor_family"] for s in verified_submissions}),
      "skus_represented": sorted({s["hardware_sku"] for s in verified_submissions}),
      "award_rankings": award_rankings,
      "grouped_frontiers": grouped_frontiers,
      "by_track_model_division": grouped_frontiers,
      "cross_chip_by_model": cross_chip_by_model,
      "models": cross_chip_by_model,
      "entries": verified_submissions,
      "submissions": verified_submissions,
      "rejected_submissions": rejected_submissions,
  }

  atomic_write_json(out_path, leaderboard_data)
  return leaderboard_data


def escape_markdown_cell(val: Any) -> str:
  """Escape untrusted values before interpolating into GFM Markdown table cells (RT-07)."""
  s = "" if val is None else str(val)
  return (
      s.replace("\\", "\\\\")
      .replace("|", "\\|")
      .replace("`", "'")
      .replace("\r", " ")
      .replace("\n", " ")
      .replace("<", "&lt;")
      .replace(">", "&gt;")
  )


def render_pr_comment_markdown(leaderboard_data: Dict[str, Any]) -> str:
  """Format a GitHub-Flavored Markdown PR comment and leaderboard summary table."""
  lines: List[str] = []
  lines.append("# EmbodiedX (`embodied-efficiency-bench` v1.0): Official Season 1 Leaderboard")
  lines.append("")
  prov = leaderboard_data.get("execution_provenance_summary", {})
  num_analytic = int(prov.get("num_analytic_reference_submissions", 0))
  num_total = int(leaderboard_data.get("num_verified_submissions", 0))
  if num_total == 0:
    lines.append(
        "> **FOUNDING SEASON v1.0 OPEN FOR SUBMISSIONS (ZERO PRE-POPULATED CURVES):** "
        "EmbodiedX Season 1 launches with zero organizer-populated GPU or TPU curves. "
        "All rankings and Pareto frontiers are populated strictly from verified community "
        "and partner Phase 1 Commit-Reveal bundles (`bundle_summary.json` + `submission_meta.json`) "
        "and ranked for Official Season 1 Awards strictly by Phase 2 Organizer Clean-Hardware Re-Measurement."
    )
    lines.append("")
  elif num_analytic > 0:
    lines.append(
        f"> **PROVENANCE DISCLAIMER (ILLUSTRATIVE ROOFLINE SEED - PENDING PHYSICAL HARDWARE RUNNER MEASUREMENT):** "
        f"`{num_analytic}` of `{num_total}` submission(s) below have execution provenance "
        "`ANALYTIC_REFERENCE_PROJECTION` / `ILLUSTRATIVE_SEED` (`is_live_empirical: false`). "
        "These entries are deterministic roofline capacity projections anchored to non-sparse dense datasheet specs "
        "and canonical forward-pass FLOPs/bytes; they are **NOT** empirical GPU/TPU hardware runs. "
        "Official CoRL 2026 rankings accept only physically executed `eeb measure --golden ... --strict` bundles "
        "(`is_live_empirical: true`) on 1 qualifying accelerator."
    )
    lines.append("")
  lines.append(
      f"- **Generated At:** `{leaderboard_data.get('generated_at', '')}`"
  )
  lines.append(
      "- **Official Season 1 Competition Hardware (`CAP_SINGLE_CHIP`, `num_devices == 1`):** "
      "**NVIDIA GPU Target:** `NVIDIA L40S (48 GB GDDR6, Ada SM89, 350 W TDP, native FP8)` | "
      "**Cloud TPU Target:** `Google Cloud TPU v6e-1 (32 GB HBM2e, Trillium)`"
  )
  lines.append(
      "- **Season 1 Checkpoints & Awards:** "
      "**Oct 30, 2026** (v1.0 launch) -> **Nov 10, 2026 (23:59 UTC)** (Checkpoint 1: CoRL 2026 Live Spotlight Freeze) -> "
      "**Nov 12, 2026** (CoRL 2026 EEAI Workshop Live Showcase, Austin, Texas) -> "
      "**Dec 18, 2026 (23:59 AoE)** (Checkpoint 2: Official Season 1 Awards Freeze + 24h Phase 2 Reveal). "
      "Recognitions include official CoRL 2026 EEAI Workshop Award Certificates, monetary prizes, "
      "an invitation to apply for Google Cloud credits, and an invitation for verified Open-Recipe winners to contribute to the EmbodiedX Season 1 Benchmark Report."
  )
  lines.append(
      "- **Gapless Real-Time Control SLAs (Locked `K_eval` Across Both `CLOSED` and `OPEN`):** "
      "Track 1A `pi05-3b` (`K_eval = 5`, 50 ms step): `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`); "
      "Track 1B `gr00t-n1.7-3b` (`K_eval = 8`, 50 ms step): `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`); "
      "Track 2 `cosmos-predict2.5-2b` (`16 frames/pred`): `p99 <= 1000.0 ms` (`>= 1.0 Pred/s`)."
  )
  lines.append(
      "- **Track Award Metrics (per `(Track, Model, Division)`):** "
      "**Award A (`Interactive Reactivity`)** highest `1000.0 / p99_ms` at `C == 1` with `p99_ms / p50_ms <= 1.5`; "
      "**Award B (`Real-Time Fleet Throughput`)** max locked-`K_eval` `Actions/s` or `Predictions/s` under SLA over `{1, 2, 4, 8, 16}`; "
      "**Award C (`Green Embodied Efficiency`)** max locked-`K_eval` `Actions/J` or `Predictions/J` under SLA (`MEASURED` telemetry only, constant `power_limit_w`); "
      "tie-broken by normalized Real-Time Dominated Hypervolume over fixed `(u_ref = 0.5, y_ref = 0.0)`."
  )
  lines.append(
      f"- **Verified Submissions:** `{leaderboard_data.get('num_verified_submissions', 0)}` "
      f"| **Quarantined/Rejected:** `{leaderboard_data.get('num_rejected_submissions', 0)}`"
  )
  models_rep = [escape_markdown_cell(x) for x in leaderboard_data.get("models_represented", [])]
  vendors_rep = [escape_markdown_cell(x) for x in leaderboard_data.get("vendors_represented", [])]
  skus_rep = [escape_markdown_cell(x) for x in leaderboard_data.get("skus_represented", [])]
  lines.append(
      f"- **Models Covered:** `{', '.join(models_rep) if models_rep else 'Awaiting Founding Submissions (pi05-3b, gr00t-n1.7-3b, cosmos-predict2.5-2b)'}`"
  )
  lines.append(
      f"- **Vendor Families:** `{', '.join(vendors_rep) if vendors_rep else 'Awaiting Founding Submissions'}` "
      f"(`{', '.join(skus_rep) if skus_rep else 'L40S_48GB, TPU_V6E_1_32GB'}`)"
  )
  lines.append("")

  if num_total == 0:
    lines.append("## 1. Season 1 Leaderboard Status (Awaiting Founding Submissions)")
    lines.append("")
    lines.append(
        "| Track | Canonical Model | Locked `K_eval` / Horizon | Gapless Real-Time SLA | Official GPU Target | Official TPU Target | Status |"
    )
    lines.append("| :--- | :--- | :---: | :---: | :--- | :--- | :--- |")
    lines.append(
        "| `VLA_TRACK` (1A) | `pi05-3b` | `K_eval = 5` (250 ms chunk, 50 ms step) | `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`) | `L40S_48GB` | `TPU_V6E_1_32GB` | Open for Phase 1 PRs |"
    )
    lines.append(
        "| `VLA_TRACK` (1B) | `gr00t-n1.7-3b` | `K_eval = 8` (400 ms chunk, 50 ms step) | `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`) | `L40S_48GB` | `TPU_V6E_1_32GB` | Open for Phase 1 PRs |"
    )
    lines.append(
        "| `WM_TRACK` (2) | `cosmos-predict2.5-2b` | `16 frames/pred` (1.0 s horizon) | `p99 <= 1000.0 ms` (`>= 1.0 Rollouts/s`) | `L40S_48GB` | `TPU_V6E_1_32GB` | Open for Phase 1 PRs |"
    )
    lines.append("")
    lines.append("## 2. Submit a Phase 1 Commit-Reveal Entry")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "eeb measure --recipe recipes/track1_vla_pi05_l40s_bf16.yaml --endpoint grpc://127.0.0.1:50051 "
        "--golden /path/to/golden.npz --strict --out /tmp/my_run"
    )
    lines.append(
        "eeb prepare-commit-bundle --source /tmp/my_run "
        "--output submissions/track1_vla/<team>_l40s_trt_20261108 "
        "--recipe-commit-sha256 <sha256_of_canonical_reveal_manifest_json> "
        "--team '<Team Name>' --engine-tag tensorrt-fp8"
    )
    lines.append("eeb verify-bundle --submission submissions/track1_vla/<team>_l40s_trt_20261108")
    lines.append("```")
  else:
    lines.append("## 1. Verified Single-Chip Submissions (4-Way SemiAnalysis Normalizations & SLA Awards)")
    lines.append("")
    lines.append(
        "| Submission (`recipe_id`) | Track | Model | Division | Hardware SKU | Vendor | Precision | C=1 p99 Latency (ms) | C=1 p99 Rate (`Chunks/s` or `Rollouts/s`) | SLA Fleet Output (`Actions/s` or `Pred/s`) | Peak Unconstrained Raw (`Actions/s` or `Pred/s`) | Peak Perf/Watt (`Actions/J` or `Pred/J`) | Peak Per 100 TOPS (`Dense BF16 TFLOPS / 100`) | Peak Perf/$/hr (`(Output/s)/($/hr)`) | Real-Time HV (`u_ref=0.5`) |"
    )
    lines.append(
        "| :--- | :--- | :--- | :---: | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    subs = sorted(
        leaderboard_data.get("submissions", []),
        key=lambda s: (
            str(s.get("track", "")),
            str(s.get("model_id", "")),
            str(s.get("division", "")),
            -(
                float(
                    s.get("award_metrics", {})
                    .get("realtime_fleet_capacity_sla", {})
                    .get("sla_max_throughput")
                    or -1.0
                )
                if isinstance(s.get("award_metrics"), dict)
                else -1.0
            ),
            -float(s.get("raw_per_chip_throughput", 0.0)),
        ),
    )
    for sub in subs:
      is_wm = is_wm_track(str(sub.get("track", "")))
      rate_unit = "Rollouts/s" if is_wm else "Chunks/s"
      watt_val = sub.get("power_efficiency_per_watt")
      watt_col = f"{float(watt_val):.4f}" if watt_val is not None else "N/A"
      c1_hz = float(sub.get("single_stream_reactivity_hz", sub.get("peak_reactivity_hz", 0.0)))
      am = sub.get("award_metrics", {}) if isinstance(sub.get("award_metrics"), dict) else {}
      sla_ms = float(am.get("sla_max_p99_ms", 1000.0 if is_wm else 250.0))
      fleet_sla = am.get("realtime_fleet_capacity_sla", {}) if isinstance(am.get("realtime_fleet_capacity_sla"), dict) else {}
      fleet_val = fleet_sla.get("sla_max_throughput")
      if fleet_sla.get("eligible") and fleet_val is not None:
        sla_col = (
            f"{float(fleet_val):.2f} "
            f"(C={fleet_sla.get('concurrency', 1)}, {float(fleet_sla.get('p99_latency_ms', 0.0)):.1f} ms)"
        )
      else:
        sla_col = f"EXCEEDS {sla_ms:g} ms SLA"
      peak_c = sub.get("peak_point_concurrency")
      peak_pt_hz = sub.get("peak_point_reactivity_hz")
      raw_tput = float(
          sub.get("raw_per_chip_throughput")
          or (sub.get("summary_metrics") or {}).get("peak_throughput")
          or 0.0
      )
      peak_raw_col = (
          f"{raw_tput:.2f} (C={peak_c}, {float(peak_pt_hz):.2f} {rate_unit})"
          if peak_c is not None and peak_pt_hz is not None
          else f"{raw_tput:.2f}"
      )
      tco_rate = float(sub.get("tco_efficiency_per_dollar", sub.get("throughput_per_dollar_hr", 0.0)) or 0.0)
      hv_val = float(am.get("realtime_dominated_hypervolume", 0.0) or 0.0)
      model_cell = sub.get("model_id", sub.get("baseline_model_id", ""))
      p99_ms_val = float(sub.get("p99_latency_ms") or 0.0)
      tops_val = float(sub.get("throughput_per_100_tops") or 0.0)
      lines.append(
          f"| `{escape_markdown_cell(sub.get('recipe_id', ''))}` | `{escape_markdown_cell(sub.get('track', ''))}` | "
          f"`{escape_markdown_cell(model_cell)}` | `{escape_markdown_cell(sub.get('division', ''))}` | "
          f"`{escape_markdown_cell(sub.get('hardware_sku', ''))}` | `{escape_markdown_cell(sub.get('vendor_family', ''))}` | "
          f"`{escape_markdown_cell(sub.get('precision', ''))}` | "
          f"{p99_ms_val:.2f} ms | {c1_hz:.2f} {rate_unit} | "
          f"{sla_col} | {peak_raw_col} | {watt_col} | "
          f"{tops_val:.2f} | {tco_rate:.2f} | {hv_val:.2f} |"
      )

    lines.append("")
    lines.append("## 2. Cross-Chip Pareto Frontier Leaders & Iso-Reactivity Slices by Model")
    lines.append("")
    lines.append(
        "| Model ID | Track | Best Raw (`SKU`) | Best Perf/Watt (`SKU`) | Best Per 100 Dense BF16 TFLOPS (`SKU`) | Best Perf/$/hr (`SKU`) | Slice 1 (`10 Chunks/s` VLA \\| `1.0 Rollouts/s` WM) | Slice 2 (`20 Chunks/s` VLA \\| `3.0 Rollouts/s` WM) | Slice 3 (`30 Chunks/s` VLA \\| `5.0 Rollouts/s` WM) | Slice 4 (`50 Chunks/s` VLA \\| `10.0 Rollouts/s` WM) |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |"
    )

    cross_chip = leaderboard_data.get("cross_chip_by_model", {})
    for model_id, mdata in sorted(cross_chip.items()):
      f4 = mdata.get("semianalysis_4way_pareto_frontiers", {})

      def _top_str(frontier_list: Sequence[Dict[str, Any]], metric_key: str, fmt: str) -> str:
        valid_pts = [p for p in frontier_list if p.get(metric_key) is not None]
        if not valid_pts:
          return "N/A"
        best = max(valid_pts, key=lambda p: float(p[metric_key]))
        val = float(best[metric_key])
        return f"{format(val, fmt)} (`{escape_markdown_cell(best.get('hardware_sku', 'UNKNOWN'))}`)"

      raw_str = _top_str(f4.get("raw_per_chip_throughput", []), "raw_per_chip_throughput", ".2f")
      watt_str = _top_str(f4.get("power_efficiency_per_watt", []), "power_efficiency_per_watt", ".4f")
      tops_str = _top_str(f4.get("iso_compute_and_bw_efficiency", []), "throughput_per_100_tops", ".2f")
      cost_str = _top_str(f4.get("tco_efficiency_per_dollar", []), "tco_efficiency_per_dollar", ".2f")

      slices = mdata.get("iso_reactivity_slices", {})

      def _fmt_slice(val: Optional[float]) -> str:
        return f"{float(val):.2f}" if val is not None else "N/A"

      if is_wm_track(str(mdata.get("track", ""))):
        s0 = slices.get("Iso-1Hz") if slices.get("Iso-1Hz") is not None else slices.get("Iso-1.5Hz")
        slice_vals = (s0, slices.get("Iso-3Hz"), slices.get("Iso-5Hz"), slices.get("Iso-10Hz"))
      else:
        slice_vals = (slices.get("Iso-10Hz"), slices.get("Iso-20Hz"), slices.get("Iso-30Hz"), slices.get("Iso-50Hz"))

      lines.append(
          f"| `{escape_markdown_cell(model_id)}` | `{escape_markdown_cell(mdata.get('track', ''))}` | "
          f"{raw_str} | {watt_str} | {tops_str} | {cost_str} | "
          f"{_fmt_slice(slice_vals[0])} | {_fmt_slice(slice_vals[1])} | "
          f"{_fmt_slice(slice_vals[2])} | {_fmt_slice(slice_vals[3])} |"
      )

    award_rankings = leaderboard_data.get("award_rankings", {})
    if award_rankings:
      lines.append("")
      lines.append("## 3. Season 1 Track Award Standings (Gapless Real-Time SLAs)")
      lines.append("")
      lines.append(
          "| Group (`track__model__division`) | Award A Leader (`Interactive Reactivity`, `C=1`) | Award B Leader (`Real-Time Fleet Capacity`) | Award C Leader (`Green Embodied Efficiency`, `MEASURED`) |"
      )
      lines.append("| :--- | :--- | :--- | :--- |")
      for gid, grp in sorted(award_rankings.items()):
        r_a = grp.get("interactive_reactivity_rank", [])
        r_b = grp.get("realtime_fleet_capacity_rank", [])
        r_c = grp.get("green_embodied_efficiency_rank", [])
        col_a = (
            f"`{escape_markdown_cell(r_a[0]['recipe_id'])}` "
            f"({float(r_a[0]['c1_interactive_rate']):.2f} {escape_markdown_cell(grp.get('rate_unit', 'Chunks/s'))})"
            if r_a
            else "N/A"
        )
        col_b = (
            f"`{escape_markdown_cell(r_b[0]['recipe_id'])}` "
            f"({float(r_b[0]['sla_max_throughput']):.2f} {escape_markdown_cell(grp.get('throughput_unit', 'Actions/s'))})"
            if r_b
            else "N/A"
        )
        col_c = (
            f"`{escape_markdown_cell(r_c[0]['recipe_id'])}` "
            f"({float(r_c[0]['sla_peak_efficiency_per_joule']):.4f} {escape_markdown_cell(grp.get('efficiency_unit', 'Actions/Joule'))})"
            if r_c
            else "N/A"
        )
        lines.append(f"| `{escape_markdown_cell(gid)}` | {col_a} | {col_b} | {col_c} |")

  rejected = leaderboard_data.get("rejected_submissions", [])
  if rejected:
    lines.append("")
    lines.append("## Quarantined / Rejected Submissions (`strict_verify=False`)")
    lines.append("")
    lines.append("| Recipe ID | Hardware SKU | Receipt Path | Rejection Reason |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for item in rejected:
      lines.append(
          f"| `{escape_markdown_cell(item.get('recipe_id', ''))}` | `{escape_markdown_cell(item.get('hardware_sku', ''))}` | "
          f"`{escape_markdown_cell(item.get('receipt_path', ''))}` | {escape_markdown_cell(item.get('reason', ''))} |"
      )

  lines.append("")
  return "\n".join(lines)
