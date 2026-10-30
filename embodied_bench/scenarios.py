"""Unified Scenario Evaluators (`P1_ONBOARD`, `P2_FLEET`, `P3_EDGE_CLOUD`, `S1_INTERACTIVE_WM`, `S2_WM_EVALUATOR`) and Kendall's Tau Ranking Metric for EmbodiedX (`embodied-efficiency-bench` v1.0)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from embodied_bench.adapters import build_adapter
from embodied_bench.golden_dataset import (
    generate_track1_golden_batch,
    generate_track2_golden_batch,
)
from embodied_bench.schema import (
    HARDWARE_SKU_SPECS,
    P2_CONCURRENCY_GRID,
    P2_SLO_PRESETS,
    POWER_CLASS_SPECS,
    QUALITY_GATE_THRESHOLDS,
    RecipeConfig,
    S1_FPS_TARGETS_MS,
    S1_STREAM_CONCURRENCY_GRID,
    ScenarioType,
)
from embodied_bench.telemetry import TelemetryMonitor, compute_joules_per_decision


# Default 12-policy evaluation pool for S2_WM_EVALUATOR (ground-truth simulator success % vs WM-predicted score)
DEFAULT_S2_POLICY_POOL: List[Dict[str, Any]] = [
    {"policy_id": "pol_01_openvla_oft_bf16", "gt_sim_success_pct": 96.5, "ref_wm_score_pct": 95.8},
    {"policy_id": "pol_02_pi05_flow10_bf16", "gt_sim_success_pct": 95.2, "ref_wm_score_pct": 94.6},
    {"policy_id": "pol_03_pi05_flow2_rtc_fp8", "gt_sim_success_pct": 93.8, "ref_wm_score_pct": 93.1},
    {"policy_id": "pol_04_openvla_oft_int4_awq", "gt_sim_success_pct": 92.1, "ref_wm_score_pct": 91.5},
    {"policy_id": "pol_05_gr00t_n15_3b_bf16", "gt_sim_success_pct": 89.4, "ref_wm_score_pct": 88.9},
    {"policy_id": "pol_06_smolvla_450m_fp16", "gt_sim_success_pct": 85.0, "ref_wm_score_pct": 84.2},
    {"policy_id": "pol_07_diffusion_policy_unet", "gt_sim_success_pct": 81.5, "ref_wm_score_pct": 80.9},
    {"policy_id": "pol_08_act_chunk16_fp16", "gt_sim_success_pct": 77.2, "ref_wm_score_pct": 76.8},
    {"policy_id": "pol_09_octo_base_fp16", "gt_sim_success_pct": 73.6, "ref_wm_score_pct": 72.9},
    {"policy_id": "pol_10_rt1_tokenized_int8", "gt_sim_success_pct": 66.8, "ref_wm_score_pct": 66.1},
    {"policy_id": "pol_11_bc_resnet50_mlp", "gt_sim_success_pct": 54.0, "ref_wm_score_pct": 53.5},
    {"policy_id": "pol_12_undertrained_int4_pruned", "gt_sim_success_pct": 38.5, "ref_wm_score_pct": 39.2},
]


@dataclass
class S2PolicyEvalResult:
  """Result of S2_WM_EVALUATOR offline policy ranking across a 12-policy pool."""

  recipe_id: str
  hardware_sku: str
  num_policies_evaluated: int
  episodes_per_policy: int
  kendall_tau: float
  min_required_kendall_tau: float
  concordant_pairs: int
  discordant_pairs: int
  tied_pairs: int
  z_score: float
  p_value: float
  total_eval_wall_time_s: float
  dollars_per_policy_evaluated: float
  joules_per_policy_evaluated: float
  policy_rankings: List[Dict[str, Any]] = field(default_factory=list)
  passed: bool = True
  violations: List[str] = field(default_factory=list)
  reason: str = "S2 World Model Policy Evaluator passed (Kendall's tau >= 0.80)."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def compute_kendall_tau(
    x_scores: Sequence[float],
    y_scores: Sequence[float],
) -> Tuple[float, int, int, int, float, float]:
  """Compute Kendall's rank correlation coefficient tau-b, pair counts, z-score, and two-sided p-value.

  Returns:
    (tau, concordant_pairs, discordant_pairs, tied_pairs, z_score, p_value)
  """
  if len(x_scores) != len(y_scores) or len(x_scores) < 2:
    raise ValueError(
        f"compute_kendall_tau requires two equal-length sequences with n >= 2 (got {len(x_scores)} and {len(y_scores)})."
    )
  n = len(x_scores)
  concordant = 0
  discordant = 0
  tied = 0
  ties_x = 0
  ties_y = 0

  for i in range(n):
    for j in range(i + 1, n):
      dx = float(x_scores[i]) - float(x_scores[j])
      dy = float(y_scores[i]) - float(y_scores[j])
      if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        tied += 1
        ties_x += 1
        ties_y += 1
      elif abs(dx) <= 1e-12:
        tied += 1
        ties_x += 1
      elif abs(dy) <= 1e-12:
        tied += 1
        ties_y += 1
      elif dx * dy > 0.0:
        concordant += 1
      else:
        discordant += 1

  total_pairs = n * (n - 1) // 2
  denom_x = max(1, total_pairs - ties_x)
  denom_y = max(1, total_pairs - ties_y)
  tau = float(concordant - discordant) / math.sqrt(float(denom_x) * float(denom_y))

  # Normal approximation for significance of Kendall's tau
  var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
  z_score = float(concordant - discordant) / math.sqrt(max(var_s, 1e-9))
  p_val = float(math.erfc(abs(z_score) / math.sqrt(2.0)))
  return round(tau, 4), concordant, discordant, tied, round(z_score, 4), round(p_val, 6)


def evaluate_p1_onboard(recipe: RecipeConfig) -> Dict[str, Any]:
  """Evaluate Scenario P1_ONBOARD: ConstantStream (C=1) with power classes E1/E2/E3/E_OPEN."""
  adapter = build_adapter(recipe)
  monitor = TelemetryMonitor(recipe.hardware_sku, allow_live_probe=False)
  pclass = recipe.resolved_power_class()
  pclass_spec = POWER_CLASS_SPECS[pclass]
  max_power_w = float(pclass_spec["max_power_watts"])

  chunk_k = recipe.action_chunk_sizes[0] if recipe.action_chunk_sizes else 8
  golden = generate_track2_golden_batch(concurrency=1, chunk_size=chunk_k, action_dim=recipe.action_dim)
  _, timing = adapter.run_track2_actions(golden)
  hw = monitor.measure_operating_point(
      model_params_b=adapter.params_b,
      precision=recipe.precision,
      batch_or_concurrency=1,
      horizon_or_chunk=chunk_k,
      decision_latency_ms=timing.chunk_latency_mean_ms,
  )

  j_active, j_paced = compute_joules_per_decision(
      active_power_watts=hw.mean_power_watts,
      idle_power_watts=hw.idle_power_watts,
      decision_latency_ms=timing.chunk_latency_mean_ms,
      paced_freq_hz=10.0,
  )
  power_compliant = hw.mean_power_watts <= max_power_w + 1e-6
  hz_react = round(1000.0 / timing.ttfa_p99_ms, 4)

  return {
      "scenario": ScenarioType.P1_ONBOARD.value,
      "stream_mode": "ConstantStream",
      "concurrency": 1,
      "power_class": pclass,
      "power_class_max_watts": max_power_w,
      "mean_power_watts": hw.mean_power_watts,
      "idle_power_watts": hw.idle_power_watts,
      "power_class_compliant": power_compliant,
      "ttfa_p50_ms": timing.ttfa_p50_ms,
      "ttfa_p99_ms": timing.ttfa_p99_ms,
      "chunk_latency_mean_ms": timing.chunk_latency_mean_ms,
      "hz_react": hz_react,
      "joules_per_decision_active": j_active,
      "joules_per_decision_10hz_paced": j_paced,
      "best_efficiency_award_eligible": float(recipe.stage2_delta_success_pct) >= -5.0 and power_compliant,
      "realtime_award_eligible": (
          float(recipe.stage2_delta_success_pct) >= -15.0
          and timing.ttfa_p99_ms <= 100.0
          and power_compliant
      ),
  }


def evaluate_p2_fleet(
    recipe: RecipeConfig,
    concurrency_grid: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
  """Evaluate Scenario P2_FLEET: Server (C in [1..128]) across SLO presets (10Hz/50ms, 5Hz/100ms, 2Hz/250ms)."""
  adapter = build_adapter(recipe)
  monitor = TelemetryMonitor(recipe.hardware_sku, allow_live_probe=False)
  sku_spec = HARDWARE_SKU_SPECS[recipe.hardware_sku]
  hourly_cost_usd = float(sku_spec.get("hourly_cost_usd", 2.50))

  c_grid = list(concurrency_grid) if concurrency_grid is not None else list(P2_CONCURRENCY_GRID)
  chunk_k = recipe.action_chunk_sizes[0] if recipe.action_chunk_sizes else 8

  sweep_points: List[Dict[str, Any]] = []
  for c in c_grid:
    golden = generate_track2_golden_batch(concurrency=c, chunk_size=chunk_k, action_dim=recipe.action_dim)
    _, timing = adapter.run_track2_actions(golden)
    hw = monitor.measure_operating_point(
        model_params_b=adapter.params_b,
        precision=recipe.precision,
        batch_or_concurrency=c,
        horizon_or_chunk=chunk_k,
        decision_latency_ms=timing.chunk_latency_mean_ms,
    )
    policy_steps_per_sec = round(float(c) / (timing.chunk_latency_mean_ms / 1000.0), 4)
    actions_per_sec = round(float(c * chunk_k) / (timing.chunk_latency_mean_ms / 1000.0), 4)
    sweep_points.append({
        "concurrency": c,
        "chunk_size": chunk_k,
        "ttfa_p99_ms": timing.ttfa_p99_ms,
        "chunk_latency_mean_ms": timing.chunk_latency_mean_ms,
        "chunk_latency_p99_ms": timing.chunk_latency_p99_ms,
        "policy_steps_per_sec_per_chip": policy_steps_per_sec,
        "actions_per_sec_per_chip": actions_per_sec,
        "mean_power_watts": hw.mean_power_watts,
    })

  max_robots_by_slo: Dict[str, int] = {}
  slo_breakdown: Dict[str, Dict[str, Any]] = {}

  for slo_name, (freq_hz, budget_ms) in P2_SLO_PRESETS.items():
    admissible = [pt for pt in sweep_points if float(pt["ttfa_p99_ms"]) <= budget_ms]
    if admissible:
      best_pt = max(admissible, key=lambda item: int(item["concurrency"]))
      # Max concurrent robots served per chip is concurrency C * (achieved step rate / required robot query rate)
      achieved_step_rate_per_stream = 1000.0 / float(best_pt["chunk_latency_mean_ms"])
      mux_factor = max(1.0, min(4.0, achieved_step_rate_per_stream / max(freq_hz, 1e-6)))
      robots_supported = int(round(int(best_pt["concurrency"]) * mux_factor))
      power_kw = float(best_pt["mean_power_watts"]) / 1000.0
      robots_per_kw = round(robots_supported / max(power_kw, 1e-6), 2)
      dollars_per_robot_hr = round(hourly_cost_usd / max(robots_supported, 1), 5)
      step_rate = float(best_pt["policy_steps_per_sec_per_chip"])
    else:
      robots_supported = 0
      robots_per_kw = 0.0
      dollars_per_robot_hr = None
      step_rate = 0.0

    max_robots_by_slo[slo_name] = robots_supported
    slo_breakdown[slo_name] = {
        "control_freq_hz": freq_hz,
        "max_p99_latency_ms": budget_ms,
        "max_robots_per_chip_at_slo": robots_supported,
        "policy_steps_per_sec_per_chip": step_rate,
        "robots_per_kw": robots_per_kw,
        "dollars_per_robot_hour": dollars_per_robot_hr,
    }

  # Primary fleet metric anchored to standard 5Hz_100ms SLO (or highest non-zero SLO if 100ms is 0)
  primary_slo = "5Hz_100ms"
  if max_robots_by_slo.get(primary_slo, 0) == 0 and max_robots_by_slo.get("2Hz_250ms", 0) > 0:
    primary_slo = "2Hz_250ms"
  primary_info = slo_breakdown[primary_slo]
  peak_policy_steps_per_sec = max(
      (float(pt["policy_steps_per_sec_per_chip"]) for pt in sweep_points),
      default=0.0,
  )

  return {
      "scenario": ScenarioType.P2_FLEET.value,
      "stream_mode": "Server",
      "concurrency_grid": c_grid,
      "primary_slo_preset": primary_slo,
      "max_robots_per_chip_at_slo": max_robots_by_slo,
      "robots_per_chip_at_primary_slo": primary_info["max_robots_per_chip_at_slo"],
      "policy_steps_per_sec_per_chip": round(peak_policy_steps_per_sec, 4),
      "robots_per_kw": primary_info["robots_per_kw"],
      "dollars_per_robot_hour": primary_info["dollars_per_robot_hour"],
      "hourly_chip_cost_usd": hourly_cost_usd,
      "slo_presets": slo_breakdown,
      "concurrency_sweep": sweep_points,
  }


def evaluate_p3_edge_cloud(recipe: RecipeConfig) -> Dict[str, Any]:
  """Evaluate Scenario P3_EDGE_CLOUD: SingleStream with emulated network (rtt_ms, jitter_ms, drop_rate)."""
  adapter = build_adapter(recipe)
  chunk_k = recipe.action_chunk_sizes[0] if recipe.action_chunk_sizes else 8
  golden = generate_track2_golden_batch(concurrency=1, chunk_size=chunk_k, action_dim=recipe.action_dim)
  _, timing = adapter.run_track2_actions(golden)

  rtt_ms = max(0.0, float(recipe.network_rtt_ms))
  jitter_ms = max(0.0, float(recipe.network_jitter_ms))
  drop_rate = max(0.0, min(0.50, float(recipe.network_drop_rate)))

  # p99 network transport adds RTT + 2.33 * jitter + packet-drop retransmit/hold tail penalty
  drop_tail_penalty_ms = drop_rate * (rtt_ms * 2.0 + 1000.0 / max(recipe.control_freq_hz, 1.0))
  eff_mean_ms = round(timing.ttfa_mean_ms + rtt_ms + 0.5 * jitter_ms + drop_tail_penalty_ms * 0.4, 3)
  eff_p99_ms = round(timing.ttfa_p99_ms + rtt_ms + 2.33 * jitter_ms + drop_tail_penalty_ms, 3)
  eff_hz_react = round(1000.0 / max(eff_p99_ms, 1e-6), 4)

  return {
      "scenario": ScenarioType.P3_EDGE_CLOUD.value,
      "stream_mode": "SingleStream_EdgeCloudSplit",
      "network_rtt_ms": rtt_ms,
      "network_jitter_ms": jitter_ms,
      "network_drop_rate": drop_rate,
      "device_ttfa_mean_ms": timing.ttfa_mean_ms,
      "device_ttfa_p99_ms": timing.ttfa_p99_ms,
      "effective_edge_cloud_ttfa_mean_ms": eff_mean_ms,
      "effective_edge_cloud_ttfa_p99_ms": eff_p99_ms,
      "effective_hz_react": eff_hz_react,
      "packet_drop_fallback_rate": round(drop_rate, 4),
      "edge_cloud_100ms_slo_met": eff_p99_ms <= 100.0,
      "edge_cloud_250ms_slo_met": eff_p99_ms <= 250.0,
  }


def evaluate_s1_interactive_wm(
    recipe: RecipeConfig,
    stream_grid: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
  """Evaluate Scenario S1_INTERACTIVE_WM: SingleStream / MultiStream (S in [1..32]) over 30-60s sessions at 10fps & 24fps."""
  adapter = build_adapter(recipe)
  monitor = TelemetryMonitor(recipe.hardware_sku, allow_live_probe=False)
  s_grid = list(stream_grid) if stream_grid is not None else list(S1_STREAM_CONCURRENCY_GRID)
  h = max(1, int(recipe.horizon_frames))
  session_s = max(30.0, min(60.0, float(recipe.session_duration_s)))

  stream_points: List[Dict[str, Any]] = []
  for s in s_grid:
    golden = generate_track1_golden_batch(batch_size=s, horizon_frames=h)
    _, _, timing = adapter.run_track1_rollout(golden)
    hw = monitor.measure_operating_point(
        model_params_b=adapter.params_b,
        precision=recipe.precision,
        batch_or_concurrency=s,
        horizon_or_chunk=h,
        decision_latency_ms=timing.rollout_latency_mean_ms,
    )
    per_frame_p99_ms = round(timing.rollout_latency_p99_ms / float(h), 3)
    per_frame_mean_ms = round(timing.rollout_latency_mean_ms / float(h), 3)
    total_fps = round((s * h) / (timing.rollout_latency_mean_ms / 1000.0), 4)
    fps_per_stream = round(total_fps / float(s), 4)
    stream_points.append({
        "streams": s,
        "horizon_frames": h,
        "rollout_latency_p99_ms": timing.rollout_latency_p99_ms,
        "per_frame_latency_mean_ms": per_frame_mean_ms,
        "per_frame_latency_p99_ms": per_frame_p99_ms,
        "fps_per_stream": fps_per_stream,
        "total_fps_per_chip": total_fps,
        "mean_power_watts": hw.mean_power_watts,
        "meets_10fps_slo": per_frame_p99_ms <= S1_FPS_TARGETS_MS["10fps"],
        "meets_24fps_slo": per_frame_p99_ms <= S1_FPS_TARGETS_MS["24fps"],
    })

  streams_10fps = max(
      (int(pt["streams"]) for pt in stream_points if bool(pt["meets_10fps_slo"])),
      default=0,
  )
  streams_24fps = max(
      (int(pt["streams"]) for pt in stream_points if bool(pt["meets_24fps_slo"])),
      default=0,
  )

  return {
      "scenario": ScenarioType.S1_INTERACTIVE_WM.value,
      "stream_mode": "MultiStream_InteractiveWorldModel",
      "session_duration_s": session_s,
      "stream_concurrency_grid": s_grid,
      "streams_per_chip_at_10fps": streams_10fps,
      "streams_per_chip_at_24fps": streams_24fps,
      "stream_sweep": stream_points,
  }


def evaluate_s2_wm_evaluator(
    recipe: RecipeConfig,
    policy_pool: Optional[Sequence[Dict[str, Any]]] = None,
    episodes_per_policy: int = 50,
    override_wm_scores: Optional[Sequence[float]] = None,
) -> S2PolicyEvalResult:
  """Evaluate Scenario S2_WM_EVALUATOR: Offline ranking of a 12-policy pool against ground-truth simulator success.

  Computes Kendall's rank correlation tau, `dollars_per_policy_evaluated`, and enforces `kendall_tau >= 0.80`.
  """
  s2_cfg = QUALITY_GATE_THRESHOLDS["s2_wm_evaluator"]
  min_tau = float(s2_cfg["min_kendall_tau"])
  req_pool_size = int(s2_cfg["required_policy_pool_size"])

  pool = [dict(p) for p in (policy_pool or DEFAULT_S2_POLICY_POOL)]
  if len(pool) < req_pool_size:
    raise ValueError(
        f"S2_WM_EVALUATOR requires a pool of at least {req_pool_size} policies, got {len(pool)}."
    )

  adapter = build_adapter(recipe)
  monitor = TelemetryMonitor(recipe.hardware_sku, allow_live_probe=False)
  sku_spec = HARDWARE_SKU_SPECS[recipe.hardware_sku]
  hourly_cost_usd = float(sku_spec.get("hourly_cost_usd", 2.95))

  # Measure batch rollout speed at B=16 to price offline policy evaluation
  eval_batch = 16
  golden = generate_track1_golden_batch(batch_size=eval_batch, horizon_frames=recipe.horizon_frames)
  _, _, timing = adapter.run_track1_rollout(golden)
  hw = monitor.measure_operating_point(
      model_params_b=adapter.params_b,
      precision=recipe.precision,
      batch_or_concurrency=eval_batch,
      horizon_or_chunk=recipe.horizon_frames,
      decision_latency_ms=timing.rollout_latency_mean_ms,
  )

  # Each episode requires ~4 horizon rollouts; batched across `eval_batch`
  rollouts_per_policy = math.ceil((episodes_per_policy * 4) / float(eval_batch))
  sec_per_policy = rollouts_per_policy * (timing.rollout_latency_mean_ms / 1000.0)
  total_wall_s = round(sec_per_policy * len(pool), 3)
  dollars_per_policy = round((sec_per_policy / 3600.0) * hourly_cost_usd, 6)
  joules_per_policy = round(sec_per_policy * hw.mean_power_watts, 3)

  gt_scores = [float(p["gt_sim_success_pct"]) for p in pool]
  if override_wm_scores is not None:
    if len(override_wm_scores) != len(pool):
      raise ValueError(
          f"override_wm_scores length ({len(override_wm_scores)}) != policy pool size ({len(pool)})."
      )
    wm_scores = [float(s) for s in override_wm_scores]
  else:
    rng = np.random.default_rng(20261007 + int(recipe.denoising_steps) * 31)
    # Candidate WM ranking noise scales with recipe quality_noise_scale
    rank_noise_std = float(recipe.quality_noise_scale) * 280.0
    wm_scores = [
        round(
            max(0.0, min(100.0, float(p["ref_wm_score_pct"]) + float(rng.normal(0.0, rank_noise_std)))),
            3,
        )
        for p in pool
    ]

  tau, conc, disc, tied, z_score, p_val = compute_kendall_tau(gt_scores, wm_scores)

  # Compute ground-truth and WM rank orders (1 = highest success rate)
  gt_order = np.argsort(-np.asarray(gt_scores))
  wm_order = np.argsort(-np.asarray(wm_scores))
  gt_ranks = {int(idx): rank + 1 for rank, idx in enumerate(gt_order)}
  wm_ranks = {int(idx): rank + 1 for rank, idx in enumerate(wm_order)}

  rankings: List[Dict[str, Any]] = []
  for i, p in enumerate(pool):
    rankings.append({
        "policy_id": p["policy_id"],
        "gt_sim_success_pct": round(gt_scores[i], 3),
        "wm_predicted_success_pct": round(wm_scores[i], 3),
        "gt_rank": gt_ranks[i],
        "wm_rank": wm_ranks[i],
        "rank_displacement": abs(gt_ranks[i] - wm_ranks[i]),
    })

  violations: List[str] = []
  if tau < min_tau:
    violations.append(
        f"S2 Kendall's tau={tau:.4f} < min_required={min_tau:.2f} "
        f"(concordant={conc}, discordant={disc}). Increase WM denoising steps or reduce quantization error."
    )

  passed = len(violations) == 0
  reason = (
      f"S2_WM_EVALUATOR passed (kendall_tau={tau:.4f} >= {min_tau:.2f}, "
      f"${dollars_per_policy:.5f}/policy on {recipe.hardware_sku})."
      if passed
      else "S2_WM_EVALUATOR rejected candidate: " + "; ".join(violations)
  )
  return S2PolicyEvalResult(
      recipe_id=recipe.recipe_id,
      hardware_sku=recipe.hardware_sku,
      num_policies_evaluated=len(pool),
      episodes_per_policy=episodes_per_policy,
      kendall_tau=tau,
      min_required_kendall_tau=min_tau,
      concordant_pairs=conc,
      discordant_pairs=disc,
      tied_pairs=tied,
      z_score=z_score,
      p_value=p_val,
      total_eval_wall_time_s=total_wall_s,
      dollars_per_policy_evaluated=dollars_per_policy,
      joules_per_policy_evaluated=joules_per_policy,
      policy_rankings=rankings,
      passed=passed,
      violations=violations,
      reason=reason,
  )


def evaluate_recipe_scenario(recipe: RecipeConfig) -> Dict[str, Any]:
  """Dispatch scenario evaluation for a recipe's resolved scenario (`P1`, `P2`, `P3`, `S1`, or `S2`)."""
  scen = recipe.resolved_scenario()
  if scen == ScenarioType.P1_ONBOARD.value:
    return evaluate_p1_onboard(recipe)
  if scen == ScenarioType.P2_FLEET.value:
    return evaluate_p2_fleet(recipe)
  if scen == ScenarioType.P3_EDGE_CLOUD.value:
    return evaluate_p3_edge_cloud(recipe)
  if scen == ScenarioType.S1_INTERACTIVE_WM.value:
    return evaluate_s1_interactive_wm(recipe)
  if scen == ScenarioType.S2_WM_EVALUATOR.value:
    return evaluate_s2_wm_evaluator(recipe).to_dict()
  raise ValueError(f"Unsupported scenario={scen!r}.")
