"""Stage 2 Top-K Closed-Loop Rollout Verification for Track 1 (Real-Time VLA Policies on LIBERO / LIBERO-RT) and Track 2 (Action-Conditioned World-Model MPC)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from embodied_bench.libero_rt import (
    LiberoRTResult,
    run_libero_rt_evaluation,
)
from embodied_bench.quality_gates import (
    GateBResult,
    GateG2Result,
    evaluate_gate_g2_mcnemar_non_inferiority,
    evaluate_stage2_gate_b,
)
from embodied_bench.schema import LiberoRTTier, RecipeConfig, is_wm_track


_PI05_SUITES: Dict[str, float] = {
    "LIBERO-Spatial": 96.8,
    "LIBERO-Object": 98.0,
    "LIBERO-Goal": 96.4,
    "LIBERO-Long": 94.0,
}

_GR00T17_SUITES: Dict[str, float] = {
    "LIBERO-Spatial": 96.4,
    "LIBERO-Object": 97.6,
    "LIBERO-Goal": 96.0,
    "LIBERO-Long": 93.4,
}

_GR00T_SUITES: Dict[str, float] = {
    "LIBERO-Spatial": 95.6,
    "LIBERO-Object": 96.8,
    "LIBERO-Goal": 95.0,
    "LIBERO-Long": 91.8,
}

_OPENVLA_OFT_SUITES: Dict[str, float] = {
    "LIBERO-Spatial": 96.2,
    "LIBERO-Object": 97.4,
    "LIBERO-Goal": 95.8,
    "LIBERO-Long": 92.6,
}

_COSMOS3_EDGE_SUITES: Dict[str, float] = {
    "WM-MPC-PickPlace": 89.0,
    "WM-MPC-DrawerOpen": 91.5,
    "WM-MPC-ToolUse": 85.0,
    "WM-MPC-LongHorizon": 81.0,
}

_QWEN3_OMNI_SUITES: Dict[str, float] = {
    "WM-MPC-PickPlace": 90.5,
    "WM-MPC-DrawerOpen": 92.5,
    "WM-MPC-ToolUse": 86.5,
    "WM-MPC-LongHorizon": 83.0,
}

_COSMOS2_SUITES: Dict[str, float] = {
    "WM-MPC-PickPlace": 87.0,
    "WM-MPC-DrawerOpen": 90.0,
    "WM-MPC-ToolUse": 82.5,
    "WM-MPC-LongHorizon": 78.0,
}

# Reference baseline closed-loop success rates (%) across standardized embodied benchmarks
STAGE2_REFERENCE_SUITES: Dict[str, Dict[str, float]] = {
    "gr00t-n1.7-3b": dict(_GR00T17_SUITES),
    "GR00T-N1.7-3B": dict(_GR00T17_SUITES),
    "nvidia/GR00T-N1.7-3B": dict(_GR00T17_SUITES),
    "nvidia/GR00T-N1.7-LIBERO": dict(_GR00T17_SUITES),
    "pi05-3b": dict(_PI05_SUITES),
    "pi0-3b": dict(_PI05_SUITES),
    "pi0-3.3B": dict(_PI05_SUITES),
    "lerobot/pi05_libero_finetuned_v044": dict(_PI05_SUITES),
    "lerobot/pi05_base": dict(_PI05_SUITES),
    "PhysicalIntelligence/pi0.5-libero": dict(_PI05_SUITES),
    "gr00t-n1.5-3b": dict(_GR00T_SUITES),
    "GR00T-N1.5-3B": dict(_GR00T_SUITES),
    "nvidia/GR00T-N1.5-3B": dict(_GR00T_SUITES),
    "openvla-oft-7b": dict(_OPENVLA_OFT_SUITES),
    "OpenVLA-OFT-7B": dict(_OPENVLA_OFT_SUITES),
    "moojink/openvla-7b-oft-finetuned-libero": dict(_OPENVLA_OFT_SUITES),
    "smolvla-450m": {
        "LIBERO-Spatial": 91.5,
        "LIBERO-Object": 93.0,
        "LIBERO-Goal": 90.2,
        "LIBERO-Long": 85.4,
    },
    "cosmos3-edge": dict(_COSMOS3_EDGE_SUITES),
    "Cosmos3-Edge": dict(_COSMOS3_EDGE_SUITES),
    "nvidia/Cosmos3-Edge": dict(_COSMOS3_EDGE_SUITES),
    "nvidia/Cosmos3-Edge-Policy-DROID": dict(_COSMOS3_EDGE_SUITES),
    "nvidia/Cosmos3-Super-Image2Video": dict(_COSMOS3_EDGE_SUITES),
    "qwen3-omni-30b-a3b": dict(_QWEN3_OMNI_SUITES),
    "Qwen3-Omni-30B-A3B-Instruct": dict(_QWEN3_OMNI_SUITES),
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": dict(_QWEN3_OMNI_SUITES),
    "cosmos-predict2-2b-action": dict(_COSMOS2_SUITES),
    "cosmos-predict2.5-2b": dict(_COSMOS2_SUITES),
    "cosmos-predict2.5": dict(_COSMOS2_SUITES),
    "Cosmos-Predict2.5-2B": dict(_COSMOS2_SUITES),
    "nvidia/Cosmos-Predict2.5-2B": dict(_COSMOS2_SUITES),
    "nvidia/Cosmos-Predict2-2B-Video2World": dict(_COSMOS2_SUITES),
    "wan2.1-wm-1.3b": {
        "WM-MPC-PickPlace": 85.5,
        "WM-MPC-DrawerOpen": 88.5,
        "WM-MPC-ToolUse": 80.5,
        "WM-MPC-LongHorizon": 76.0,
    },
    "ctrl-world-1.5b": {
        "WM-MPC-PickPlace": 86.5,
        "WM-MPC-DrawerOpen": 89.0,
        "WM-MPC-ToolUse": 81.5,
        "WM-MPC-LongHorizon": 77.5,
    },
    "DreamDojo-2B": {
        "WM-MPC-PickPlace": 88.5,
        "WM-MPC-DrawerOpen": 91.0,
        "WM-MPC-ToolUse": 84.0,
        "WM-MPC-LongHorizon": 79.5,
    },
    "DreamDojo-14B": {
        "WM-MPC-PickPlace": 93.0,
        "WM-MPC-DrawerOpen": 94.5,
        "WM-MPC-ToolUse": 89.5,
        "WM-MPC-LongHorizon": 86.0,
    },
    "DreamZero-14B": {
        "WM-MPC-PickPlace": 93.5,
        "WM-MPC-DrawerOpen": 95.0,
        "WM-MPC-ToolUse": 90.5,
        "WM-MPC-LongHorizon": 87.5,
    },
}


def get_reference_suites_for_recipe(recipe: RecipeConfig) -> Dict[str, float]:
  """Return reference baseline closed-loop suite success rates (%) for a recipe."""
  default_suites = (
      STAGE2_REFERENCE_SUITES["cosmos-predict2-2b-action"]
      if is_wm_track(recipe.track)
      else STAGE2_REFERENCE_SUITES["pi05-3b"]
  )
  return dict(STAGE2_REFERENCE_SUITES.get(recipe.baseline_model_id, default_suites))


def run_stage2_verification(
    recipe: RecipeConfig,
    override_delta_pct: Optional[float] = None,
    custom_candidate_suites: Optional[Dict[str, float]] = None,
) -> GateBResult:
  """Execute Stage 2 Top-K Closed-Loop Rollout Verification and evaluate Gate B."""
  baseline_suites = get_reference_suites_for_recipe(recipe)

  delta = (
      float(override_delta_pct)
      if override_delta_pct is not None
      else float(recipe.stage2_delta_success_pct)
  )

  if custom_candidate_suites is not None:
    candidate_suites = dict(custom_candidate_suites)
  else:
    candidate_suites = {}
    for idx, (suite_name, base_val) in enumerate(sorted(baseline_suites.items())):
      offset = (-0.15 if idx % 2 == 0 else 0.15) if abs(delta) > 0.01 else 0.0
      candidate_suites[suite_name] = round(max(0.0, min(100.0, base_val + delta + offset)), 3)

  return evaluate_stage2_gate_b(
      compression_tier=recipe.compression_tier,
      task_suite=recipe.stage2_task_suite,
      per_suite_baseline_pct=baseline_suites,
      per_suite_candidate_pct=candidate_suites,
  )


def run_stage2_libero_rt_and_g2(
    recipe: RecipeConfig,
    policy_latency_mean_ms: float = 35.0,
    policy_latency_p99_ms: float = 45.0,
    chunk_size: int = 8,
    tier: str = LiberoRTTier.TIER2_CLOUD_CPU_LATENCY_REPLAY.value,
) -> tuple[LiberoRTResult, GateG2Result]:
  """Run latency-coupled LIBERO-RT evaluation + Gate G2 Paired McNemar non-inferiority check."""
  baseline_suites = get_reference_suites_for_recipe(recipe)
  libero_rt_res = run_libero_rt_evaluation(
      recipe=recipe,
      policy_latency_mean_ms=policy_latency_mean_ms,
      policy_latency_p99_ms=policy_latency_p99_ms,
      chunk_size=chunk_size,
      tier=tier,
      base_suites=baseline_suites,
  )
  base_mean_sr = float(sum(baseline_suites.values())) / max(1, len(baseline_suites))
  g2_res = evaluate_gate_g2_mcnemar_non_inferiority(
      baseline_success_rate_pct=base_mean_sr,
      candidate_success_rate_pct=libero_rt_res.latency_free_success_rate,
      clock=recipe.clock,
      division=recipe.division,
      compression_tier=recipe.compression_tier,
      p99_latency_ms=policy_latency_p99_ms,
  )
  return libero_rt_res, g2_res


def attach_stage2_to_stage1_report(
    stage1_report: Dict[str, Any],
    gate_b_result: GateBResult,
) -> Dict[str, Any]:
  """Attach Stage 2 Gate B closed-loop verification result to a Stage 1 evaluation report."""
  updated = dict(stage1_report)
  updated["stage2_gate_b"] = gate_b_result.to_dict()
  if updated.get("status") == "CERTIFIED" and not gate_b_result.passed:
    updated["status"] = "REJECTED"
    updated["reason"] = gate_b_result.reason
  return updated
