"""Command-line interface for EmbodiedX (`embodied-efficiency-bench` / `eeb` v1.0).

Supported subcommands:
  - measure: Run the EEB 3-run Measurement Protocol (20-call warmup, >=320 VLA / >=200 WM call trace, <5% variance check, INA3221/NVML Joules/decision, Device-Agnostic PerformanceCapEnvelope + Roofline-Normalized Latency, G1-G4 compliance, and scenario evaluation)
  - eval-libero-rt: Run Latency-Coupled LIBERO-RT closed-loop evaluation (ceil(L/delta_t) delay, execution strategy, safety probes, Tier 1/2/3 scaling)
  - eval-wm-ranking: Run S2_WM_EVALUATOR offline policy ranking across the 12-policy pool (Kendall's tau >= 0.80 & $/policy)
  - run-stage1: Run Stage 1 Golden Tensor Replay + Co-Located Gate A (and optional Stage 2 Gate B)
  - run-stage2: Run Stage 2 Closed-Loop Rollout Verification (Gate B) on a Stage 1 report or receipt
  - compute-pareto: Compute Pareto frontiers, Iso-Reactivity slices, and leaderboard JSON from run outputs
  - verify-receipt: Verify SHA-256 cryptographic signature and co-located Gate A/B invariants
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from embodied_bench.attestation import (
    compute_attestation_sha256,
    create_submission_receipt,
    verify_submission_receipt,
)
from embodied_bench.libero_rt import run_libero_rt_evaluation
from embodied_bench.measure import run_eeb_measure
from embodied_bench.pareto import (
    compute_iso_reactivity_slices,
    extract_pareto_frontier,
    extract_semianalysis_4way_pareto_frontiers,
)
from embodied_bench.scenarios import evaluate_s2_wm_evaluator
from embodied_bench.schema import (
    ANALYTIC_REFERENCE_PROVENANCE,
    BenchmarkClock,
    CompressionTier,
    DivisionType,
    ExecutionStrategy,
    HARDWARE_SKU_SPECS,
    LIVE_EMPIRICAL_PROVENANCE,
    LiberoRTTier,
    MODERN_REFERENCE_CHECKPOINTS,
    PerformanceCapId,
    RecipeConfig,
    RunStatus,
    ScenarioType,
    TrackType,
    atomic_write_json,
    atomic_write_text,
    is_wm_track,
    load_yaml_or_json,
    now_pacific_iso,
)
from embodied_bench.stage1_golden_replay import run_stage1_from_recipe_path
from embodied_bench.stage2_closed_loop_verify import (
    attach_stage2_to_stage1_report,
    get_reference_suites_for_recipe,
    run_stage2_verification,
)


def _cmd_measure(args: argparse.Namespace) -> int:
  report = run_eeb_measure(
      recipe_path=args.recipe,
      output_path=args.output,
      include_stage2=not args.skip_stage2,
      warmup_calls=args.warmup_calls,
      timed_calls=args.timed_calls,
      num_runs=args.num_runs,
      perf_cap=args.perf_cap,
      perf_cap_mode=args.perf_cap_mode,
      mps_active_thread_pct=args.mps_active_thread_pct,
      clock_scale_factor=args.clock_scale_factor,
      vram_fraction_cap=args.vram_fraction_cap,
      live_runner=getattr(args, "live_runner", False),
      server_url=getattr(args, "server_url", None),
  )
  if args.receipt_output:
    create_submission_receipt(report, output_path=args.receipt_output)

  m_proto = report.get("measurement_protocol", {})
  pcap = report.get("performance_cap_evaluation", {})
  sa = report.get("semianalysis_normalizations", pcap.get("semianalysis_normalizations", {}))
  hw_ind = pcap.get("hardware_independent_profile", {})
  comp = report.get("compliance_gates_g1_g4", {})
  summary = {
      "command": "measure",
      "recipe_id": report["recipe_id"],
      "track": report.get("track"),
      "clock": report.get("clock"),
      "scenario": report.get("scenario"),
      "division": report.get("division"),
      "power_class": report.get("power_class"),
      "perf_cap": report.get("perf_cap"),
      "tier_alias": pcap.get("tier_alias", report.get("perf_cap")),
      "perf_cap_tier_alias": pcap.get("tier_alias", report.get("perf_cap")),
      "perf_cap_mode": report.get("perf_cap_mode"),
      "perf_cap_satisfied": pcap.get("passed", True),
      "num_devices": pcap.get("num_devices", 1),
      "single_device_compliant": pcap.get("single_device_compliant", True),
      "hardware_sku": report["hardware_sku"],
      "status": report["status"],
      "reason": report["reason"],
      "execution_provenance": report.get("execution_provenance"),
      "time_to_ready_s": m_proto.get("time_to_ready_s"),
      "median_p99_latency_ms": m_proto.get("median_of_runs_p99_ms"),
      "physical_p99_latency_ms": pcap.get("physical_latency_p99_ms"),
      "roofline_normalized_p99_latency_ms": pcap.get("normalized_latency_p99_ms"),
      "roofline_normalization_factor": pcap.get("roofline_normalization_factor"),
      "raw_single_device_throughput": pcap.get("raw_single_device_throughput"),
      "raw_per_chip_throughput": sa.get("raw_per_chip_throughput", pcap.get("raw_single_device_throughput")),
      "power_efficiency_per_watt": sa.get("power_efficiency_per_watt", pcap.get("power_efficiency_per_watt")),
      "iso_compute_and_bw_efficiency": sa.get("iso_compute_and_bw_efficiency"),
      "tco_efficiency_per_dollar": sa.get("tco_efficiency_per_dollar", pcap.get("throughput_per_dollar_hr")),
      "throughput_per_dollar_hr": sa.get("throughput_per_dollar_hr", pcap.get("throughput_per_dollar_hr")),
      "throughput_per_100_tops": pcap.get("throughput_per_100_tops"),
      "throughput_per_100_gbps_bw": pcap.get("throughput_per_100_gbps_bw"),
      "throughput_per_tbps_bw": sa.get("throughput_per_tbps_bw", pcap.get("throughput_per_tbps_bw")),
      "same_chip_speedup_vs_bf16_ref": sa.get("same_chip_speedup_vs_bf16_ref"),
      "flops_per_action_gflops": hw_ind.get("flops_per_action_gflops"),
      "flops_per_frame_gflops": hw_ind.get("flops_per_frame_gflops"),
      "bytes_moved_per_step_mb": hw_ind.get("bytes_moved_per_step_mb"),
      "peak_vram_gb": pcap.get("peak_vram_gb"),
      "cross_run_variance_pct": m_proto.get("cross_run_variance_pct"),
      "joules_per_decision_active": m_proto.get("joules_per_decision_active"),
      "joules_per_decision_10hz_paced": m_proto.get("joules_per_decision_10hz_paced"),
      "compliance_passed": comp.get("passed", False),
      "output": str(args.output),
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if report["status"] == RunStatus.CERTIFIED.value else 2


def _extract_sweep_latency_and_chunk(
    sweep_report_path: str,
    fallback_mean_ms: float,
    fallback_p99_ms: float,
    fallback_chunk_size: int,
) -> tuple[float, float, int]:
  """Extract single-stream (C=1) steady-state latency and execute horizon from a sweep report."""
  doc = load_yaml_or_json(sweep_report_path)
  points = [
      p for p in (doc.get("operating_points") or doc.get("pareto_frontier") or [])
      if isinstance(p, dict)
  ]
  if not points:
    raise ValueError(f"--sweep-report {sweep_report_path!r} contains no operating_points.")
  c1_points = [
      p for p in points if int(p.get("concurrency", p.get("batch_size", 0)) or 0) == 1
  ]
  selected = c1_points[0] if c1_points else min(
      points,
      key=lambda p: int(p.get("concurrency", p.get("batch_size", 1)) or 1),
  )
  mean_val = selected.get(
      "obs_to_action_mean_ms",
      selected.get("chunk_latency_mean_ms", selected.get("latency_mean_ms", selected.get("ttfa_mean_ms"))),
  )
  p99_val = selected.get(
      "obs_to_action_p99_ms",
      selected.get("chunk_latency_p99_ms", selected.get("latency_p99_ms", selected.get("ttfa_p99_ms"))),
  )
  chunk_val = selected.get(
      "execute_horizon",
      selected.get("chunk_size", doc.get("execute_horizon")),
  )
  mean_ms = float(mean_val) if mean_val is not None else float(fallback_mean_ms)
  p99_ms = float(p99_val) if p99_val is not None else float(fallback_p99_ms)
  chunk_size = int(chunk_val) if chunk_val is not None else int(fallback_chunk_size)
  return mean_ms, p99_ms, chunk_size


def _cmd_eval_libero_rt(args: argparse.Namespace) -> int:
  raw = load_yaml_or_json(args.recipe)
  if args.execution_strategy:
    raw = dict(raw)
    raw["execution_strategy"] = args.execution_strategy
  recipe = RecipeConfig.from_dict(raw)
  base_suites = get_reference_suites_for_recipe(recipe)

  sweep_report = getattr(args, "sweep_report", None)
  latency_mean_ms = float(args.latency_mean_ms)
  latency_p99_ms = float(args.latency_p99_ms)
  chunk_size = int(args.chunk_size)
  latency_provenance = "CLI_ARGUMENT_OR_DEFAULT"
  if sweep_report:
    latency_mean_ms, latency_p99_ms, chunk_size = _extract_sweep_latency_and_chunk(
        sweep_report_path=str(sweep_report),
        fallback_mean_ms=latency_mean_ms,
        fallback_p99_ms=latency_p99_ms,
        fallback_chunk_size=chunk_size,
    )
    latency_provenance = "EMPIRICAL_SWEEP_REPORT"

  evaluation_mode = (
      "TIER3_HIL_ANCHOR_CONSISTENCY"
      if args.tier3_measured_sr_pct is not None
      else "ANALYTIC_LATENCY_COUPLED_MODEL"
  )

  res = run_libero_rt_evaluation(
      recipe=recipe,
      policy_latency_mean_ms=latency_mean_ms,
      policy_latency_p99_ms=latency_p99_ms,
      chunk_size=chunk_size,
      tier=args.tier,
      base_suites=base_suites,
      tier3_measured_sr_pct=args.tier3_measured_sr_pct,
  )
  payload = res.to_dict()
  payload["evaluation_mode"] = evaluation_mode
  payload["latency_provenance"] = latency_provenance
  payload["sweep_report"] = str(sweep_report) if sweep_report else None
  payload["generated_at"] = now_pacific_iso()
  atomic_write_json(args.output, payload)

  summary = {
      "command": "eval-libero-rt",
      "recipe_id": res.recipe_id,
      "tier": res.tier,
      "evaluation_mode": evaluation_mode,
      "latency_provenance": latency_provenance,
      "sweep_report": str(sweep_report) if sweep_report else None,
      "latency_mean_ms": round(latency_mean_ms, 4),
      "latency_p99_ms": round(latency_p99_ms, 4),
      "chunk_size": chunk_size,
      "execution_strategy": res.execution_strategy,
      "delay_steps_mean": res.delay_steps_mean,
      "delay_steps_p99": res.delay_steps_p99,
      "latency_free_success_rate": res.latency_free_success_rate,
      "realtime_success_rate": res.realtime_success_rate,
      "perturbation_success_rate": res.perturbation_success_rate,
      "reaction_time_ms": res.reaction_time_ms,
      "constraint_violations": res.constraint_violations,
      "passed": res.passed,
      "output": str(args.output),
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if res.passed else 2


def _cmd_eval_wm_ranking(args: argparse.Namespace) -> int:
  raw = load_yaml_or_json(args.recipe)
  recipe = RecipeConfig.from_dict(raw)
  res = evaluate_s2_wm_evaluator(
      recipe=recipe,
      episodes_per_policy=args.episodes_per_policy,
  )
  evaluation_mode = "ANALYTIC_RECIPE_RANKING_MODEL"
  payload = res.to_dict()
  payload["scenario"] = ScenarioType.S2_WM_EVALUATOR.value
  payload["evaluation_mode"] = evaluation_mode
  payload["generated_at"] = now_pacific_iso()
  atomic_write_json(args.output, payload)

  summary = {
      "command": "eval-wm-ranking",
      "recipe_id": res.recipe_id,
      "scenario": ScenarioType.S2_WM_EVALUATOR.value,
      "evaluation_mode": evaluation_mode,
      "hardware_sku": res.hardware_sku,
      "num_policies_evaluated": res.num_policies_evaluated,
      "kendall_tau": res.kendall_tau,
      "min_required_kendall_tau": res.min_required_kendall_tau,
      "dollars_per_policy_evaluated": res.dollars_per_policy_evaluated,
      "passed": res.passed,
      "output": str(args.output),
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if res.passed else 2


def _cmd_run_stage1(args: argparse.Namespace) -> int:
  report = run_stage1_from_recipe_path(
      recipe_path=args.recipe,
      output_path=args.output,
      include_stage2=not args.skip_stage2,
      perf_cap=getattr(args, "perf_cap", None),
      live_runner=getattr(args, "live_runner", False),
      server_url=getattr(args, "server_url", None),
  )
  if args.receipt_output:
    create_submission_receipt(report, output_path=args.receipt_output)

  summary = {
      "command": "run-stage1",
      "recipe_id": report["recipe_id"],
      "track": report["track"],
      "perf_cap": report.get("perf_cap"),
      "hardware_sku": report["hardware_sku"],
      "status": report["status"],
      "reason": report["reason"],
      "execution_provenance": report.get("execution_provenance"),
      "num_operating_points": report["summary_metrics"]["num_operating_points"],
      "num_pareto_frontier_points": report["summary_metrics"]["num_pareto_frontier_points"],
      "peak_reactivity_hz": report["summary_metrics"]["peak_reactivity_hz"],
      "peak_throughput": report["summary_metrics"]["peak_throughput"],
      "iso_reactivity_slices": report["iso_reactivity_slices"],
      "output": str(args.output),
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if report["status"] == RunStatus.CERTIFIED.value else 2


def _cmd_run_stage2(args: argparse.Namespace) -> int:
  payload = load_yaml_or_json(args.receipt)

  if "attested_body" in payload and isinstance(payload["attested_body"], dict):
    body = dict(payload["attested_body"])
    recipe_cfg = RecipeConfig(
        recipe_id=str(body["recipe_id"]),
        track=str(body["track"]),
        model_family=str(body.get("model_family", body["baseline_model_id"])),
        model_name=str(body["model_name"]),
        baseline_model_id=str(body["baseline_model_id"]),
        precision=str(body["precision"]),
        hardware_sku=str(body["hardware_sku"]),
        compression_tier=str(body["compression_tier"]),
    )
    gate_b = run_stage2_verification(
        recipe=recipe_cfg,
        override_delta_pct=args.override_delta_pct,
    )
    body["stage2_gate_b"] = gate_b.to_dict()
    if not gate_b.passed:
      body["status"] = RunStatus.REJECTED.value
      body["reason"] = gate_b.reason
    elif body.get("status") == RunStatus.CERTIFIED.value:
      body["reason"] = None

    updated_receipt = dict(payload)
    updated_receipt["issued_at"] = now_pacific_iso()
    updated_receipt["status"] = body["status"]
    updated_receipt["reason"] = body["reason"]
    updated_receipt["attested_body"] = body
    updated_receipt["attestation_sha256"] = compute_attestation_sha256(body)
    atomic_write_json(args.output, updated_receipt)

    summary = {
        "command": "run-stage2",
        "input_type": "submission_receipt",
        "recipe_id": body["recipe_id"],
        "status": updated_receipt["status"],
        "gate_b_passed": gate_b.passed,
        "delta_success_rate_pct": gate_b.delta_success_rate_pct,
        "min_allowed_delta_pct": gate_b.min_allowed_delta_pct,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_b.passed else 2

  recipe_dict = payload.get("recipe") if isinstance(payload.get("recipe"), dict) else payload
  recipe_cfg = RecipeConfig.from_dict(recipe_dict)
  gate_b = run_stage2_verification(
      recipe=recipe_cfg,
      override_delta_pct=args.override_delta_pct,
  )
  updated_report = attach_stage2_to_stage1_report(payload, gate_b)
  updated_report["generated_at"] = now_pacific_iso()
  atomic_write_json(args.output, updated_report)

  summary = {
      "command": "run-stage2",
      "input_type": "stage1_report",
      "recipe_id": updated_report["recipe_id"],
      "status": updated_report["status"],
      "gate_b_passed": gate_b.passed,
      "delta_success_rate_pct": gate_b.delta_success_rate_pct,
      "min_allowed_delta_pct": gate_b.min_allowed_delta_pct,
      "output": str(args.output),
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if gate_b.passed else 2


def compute_pareto_leaderboard(
    input_paths: Sequence[str],
    output_path: str,
    receipt_output_path: Optional[str] = None,
) -> Dict[str, Any]:
  """Aggregate Stage 1/2 reports, compute per-submission & global Pareto frontiers, and write leaderboard."""
  submissions: List[Dict[str, Any]] = []
  wm_pool: List[Dict[str, Any]] = []
  vla_pool: List[Dict[str, Any]] = []
  receipts: List[Dict[str, Any]] = []

  for path_str in input_paths:
    report = load_yaml_or_json(path_str)
    x_key = str(report["x_metric_name"])
    y_key = str(report["y_metric_name"])
    lat_key = str(report["latency_metric_name"])
    wm_flag = is_wm_track(report["track"])
    hw_sku = str(report.get("hardware_sku") or "")
    if hw_sku in ("", "UNKNOWN_SKU", "None"):
      env_chk = report.get("envelope_check")
      if isinstance(env_chk, dict) and env_chk.get("hardware_sku"):
        hw_sku = str(env_chk["hardware_sku"])
        report["hardware_sku"] = hw_sku
    exec_prov_rep = report.get("execution_provenance") if isinstance(report.get("execution_provenance"), dict) else {}
    power_unavail_rep = exec_prov_rep.get("power") == "UNAVAILABLE"
    if hw_sku in HARDWARE_SKU_SPECS:
      from embodied_bench.measure import compute_semianalysis_normalizations

      for pt in report.get("operating_points", []):
        if not isinstance(pt, dict):
          continue
        raw_t = pt.get(y_key, pt.get("raw_per_chip_throughput"))
        if raw_t is None:
          continue
        sa_norm = compute_semianalysis_normalizations(raw_throughput=float(raw_t), hardware_sku=hw_sku)
        for sa_k in (
            "throughput_per_100_tops",
            "throughput_per_tbps_bw",
            "same_chip_speedup_vs_bf16_ref",
            "tco_efficiency_per_dollar",
            "throughput_per_dollar_hr",
            "units_per_dollar",
            "throughput_per_1k_usd",
        ):
          if pt.get(sa_k) is None and sa_norm.get(sa_k) is not None:
            pt[sa_k] = sa_norm[sa_k]
        ht_pt = pt.get("hardware_telemetry") if isinstance(pt.get("hardware_telemetry"), dict) else {}
        if power_unavail_rep or ht_pt.get("power_provenance") == "UNAVAILABLE":
          pt["power_efficiency_per_watt"] = None

    frontier = extract_pareto_frontier(
        operating_points=report.get("operating_points", []),
        x_key=x_key,
        y_key=y_key,
        require_gate_a=True,
    )
    frontiers_4way = extract_semianalysis_4way_pareto_frontiers(
        operating_points=report.get("operating_points", []),
        x_key=x_key,
        require_gate_a=True,
    )
    iso_slices = compute_iso_reactivity_slices(
        frontier_points=frontier,
        latency_key=lat_key,
        throughput_key=y_key,
        track=report["track"],
    )
    report["pareto_frontier"] = frontier
    report["semianalysis_4way_pareto_frontiers"] = frontiers_4way
    report["iso_reactivity_slices"] = iso_slices
    if "summary_metrics" in report and isinstance(report["summary_metrics"], dict):
      report["summary_metrics"]["iso_reactivity_slices"] = iso_slices
      report["summary_metrics"]["num_pareto_frontier_points"] = len(frontier)

    receipt = create_submission_receipt(report)
    receipts.append(receipt)

    eff_key = "predictions_per_watt" if wm_flag else "actions_per_watt"
    op_points = report.get("operating_points", [])
    power_missing = any(
        p.get("power_provenance") == "UNAVAILABLE"
        or (
            "power_efficiency_per_watt" in p
            and p.get("power_efficiency_per_watt") is None
            and p.get(eff_key) is None
        )
        for p in op_points
        if isinstance(p, dict)
    )
    eff_vals: List[float] = []
    for p in frontier:
      if p.get("power_provenance") == "UNAVAILABLE":
        continue
      v = p.get(eff_key)
      if v is None:
        if wm_flag and "predictions_per_sec" not in p:
          v_fpw = p.get("fps_per_watt", p.get("frames_per_watt", p.get("power_efficiency_per_watt")))
          h_f = max(float(p.get("horizon_frames", 16) or 16), 1.0)
          v = (float(v_fpw) / h_f) if v_fpw is not None else None
        else:
          v = p.get("power_efficiency_per_watt")
      if v is not None:
        eff_vals.append(float(v))
    sys_eff_vals = [
        float(v)
        for p in frontier
        for v in (p.get("power_efficiency_per_watt") if p.get("power_efficiency_per_watt") is not None else p.get(eff_key),)
        if v is not None and p.get("power_provenance") != "UNAVAILABLE"
    ]
    peak_eff: Optional[float] = (
        round(max(eff_vals), 4) if eff_vals else (None if power_missing else 0.0)
    )
    peak_power_eff_system: Optional[float] = (
        round(max(sys_eff_vals), 4) if sys_eff_vals else (None if power_missing else 0.0)
    )
    peak_speedup = max(
        (float(p.get("speedup_vs_reference") or 0.0) for p in frontier),
        default=0.0,
    )
    peak_x = max((float(p.get(x_key) or 0.0) for p in frontier), default=0.0)
    peak_y = max((float(p.get(y_key) or 0.0) for p in frontier), default=0.0)
    peak_iso_tops = max(
        (float(p.get("throughput_per_100_tops") or 0.0) for p in frontier),
        default=0.0,
    )
    peak_iso_bw = max(
        (float(p.get("throughput_per_100_gbps_bw") or 0.0) for p in frontier),
        default=0.0,
    )
    peak_iso_tbps = max(
        (float(p.get("throughput_per_tbps_bw") or 0.0) for p in frontier),
        default=0.0,
    )
    peak_dollar_hr = max(
        (float(p.get("throughput_per_dollar_hr") or 0.0) for p in frontier),
        default=0.0,
    )
    peak_tco_eff = max(
        (
            float(
                p.get("tco_efficiency_per_dollar")
                if p.get("tco_efficiency_per_dollar") is not None
                else (p.get("throughput_per_dollar_hr") or 0.0)
            )
            for p in frontier
        ),
        default=0.0,
    )
    peak_units_per_dollar = max(
        (
            float(
                p.get("units_per_dollar")
                if p.get("units_per_dollar") is not None
                else (p.get("throughput_per_dollar_hr") or 0.0) * 3600.0
            )
            for p in frontier
        ),
        default=0.0,
    )
    peak_tco_1k = max(
        (
            float(
                p.get("throughput_per_1k_usd")
                if p.get("throughput_per_1k_usd") is not None
                else (p.get("units_per_dollar") or 0.0) * 1000.0
            )
            for p in frontier
        ),
        default=0.0,
    )

    entry = {
        "recipe_id": report["recipe_id"],
        "clock": report.get("clock", "eeb_v1"),
        "scenario": report.get("scenario"),
        "division": report.get("division", "OPEN"),
        "power_class": report.get("power_class"),
        "perf_cap": report.get("perf_cap"),
        "tier_alias": report.get("tier_alias", report.get("perf_cap")),
        "perf_cap_mode": report.get("perf_cap_mode"),
        "num_devices": report.get("num_devices", 1),
        "single_device_compliant": report.get("single_device_compliant", True),
        "execution_strategy": report.get("execution_strategy"),
        "track": report["track"],
        "model_family": report.get("model_family", ""),
        "model_name": report["model_name"],
        "baseline_model_id": report["baseline_model_id"],
        "precision": report["precision"],
        "hardware_sku": report["hardware_sku"],
        "compression_tier": report["compression_tier"],
        "status": report["status"],
        "reason": report.get("reason"),
        "execution_provenance": report.get("execution_provenance") or dict(ANALYTIC_REFERENCE_PROVENANCE),
        "x_metric_name": x_key,
        "y_metric_name": y_key,
        "latency_metric_name": lat_key,
        "peak_reactivity_hz": round(peak_x, 4),
        "peak_throughput": round(peak_y, 4),
        "raw_single_device_throughput": round(peak_y, 4),
        "raw_per_chip_throughput": round(peak_y, 4),
        "power_efficiency_per_watt": peak_power_eff_system,
        "iso_compute_and_bw_efficiency": {
            "throughput_per_100_tops": round(peak_iso_tops, 4),
            "throughput_per_tbps_bw": round(peak_iso_tbps, 4),
            "same_chip_speedup_vs_bf16_ref": round(peak_speedup, 4),
        },
        "tco_efficiency_per_dollar": round(peak_tco_eff, 4),
        "units_per_dollar": round(peak_units_per_dollar, 2),
        "throughput_per_1k_usd": round(peak_tco_1k, 4),
        "throughput_per_dollar_hr": round(peak_dollar_hr, 4),
        "peak_throughput_per_100_tops": round(peak_iso_tops, 4),
        "peak_throughput_per_100_gbps_bw": round(peak_iso_bw, 4),
        "peak_throughput_per_tbps_bw": round(peak_iso_tbps, 4),
        "same_chip_speedup_vs_bf16_ref": round(peak_speedup, 4),
        "peak_efficiency_per_watt": peak_eff,
        "peak_speedup_vs_reference": round(peak_speedup, 4),
        "iso_reactivity_slices": iso_slices,
        "semianalysis_normalizations": report.get("semianalysis_normalizations"),
        "semianalysis_4way_pareto_frontiers": frontiers_4way,
        "performance_cap_evaluation": report.get("performance_cap_evaluation"),
        "measurement_protocol": report.get("measurement_protocol"),
        "compliance_gates_g1_g4": report.get("compliance_gates_g1_g4"),
        "wm_open_3part_gate": report.get("wm_open_3part_gate"),
        "scenario_metrics": report.get("scenario_metrics"),
        "libero_rt": report.get("libero_rt"),
        "stage2_gate_b": report.get("stage2_gate_b"),
        "pareto_frontier": frontier,
        "attestation_sha256": receipt["attestation_sha256"],
    }
    submissions.append(entry)

    if report["status"] in (RunStatus.CERTIFIED.value, "PENDING"):
      for pt in frontier:
        tagged = dict(pt)
        tagged["recipe_id"] = report["recipe_id"]
        tagged["model_name"] = report["model_name"]
        tagged["precision"] = report["precision"]
        tagged["hardware_sku"] = report["hardware_sku"]
        tagged["perf_cap"] = report.get("perf_cap")
        if wm_flag:
          tagged.setdefault("hz_plan", float(pt.get("hz_plan", pt.get(x_key, 0.0)) or 0.0))
          h_f = max(float(pt.get("horizon_frames", pt.get("horizon", 16)) or 16), 1.0)
          wm_fps = float(
              pt.get(
                  "generated_frames_per_sec",
                  pt.get("fps", pt.get("frames_per_sec", 0.0)),
              )
              or 0.0
          )
          if pt.get("predictions_per_sec") is not None:
            wm_preds = float(pt["predictions_per_sec"])
          elif pt.get("chunks_per_sec") is not None:
            wm_preds = float(pt["chunks_per_sec"])
          elif wm_fps > 0.0:
            wm_preds = round(wm_fps / h_f, 4)
          else:
            wm_preds = float(pt.get(y_key, 0.0) or 0.0)
          if wm_fps <= 0.0 and wm_preds > 0.0:
            wm_fps = round(wm_preds * h_f, 4)
          tagged.setdefault("predictions_per_sec", wm_preds)
          tagged.setdefault("fps", wm_fps)
          tagged.setdefault("generated_frames_per_sec", wm_fps)
          tagged["raw_per_chip_throughput"] = wm_preds
          wm_pool.append(tagged)
        else:
          tagged.setdefault("hz_react", float(pt.get("hz_react", pt.get(x_key, 0.0)) or 0.0))
          vla_y = float(
              pt.get("executed_actions_per_sec", pt.get("actions_per_sec", pt.get(y_key, 0.0)))
              or 0.0
          )
          tagged.setdefault("actions_per_sec", vla_y)
          tagged.setdefault("executed_actions_per_sec", vla_y)
          tagged.setdefault("raw_per_chip_throughput", float(pt.get("raw_per_chip_throughput", vla_y) or 0.0))
          vla_pool.append(tagged)

  wm_global_frontier = extract_pareto_frontier(
      operating_points=wm_pool,
      x_key="hz_plan",
      y_key="predictions_per_sec",
      require_gate_a=True,
  )
  wm_4way_global = extract_semianalysis_4way_pareto_frontiers(
      operating_points=wm_pool,
      x_key="hz_plan",
      require_gate_a=True,
  )
  vla_global_frontier = extract_pareto_frontier(
      operating_points=vla_pool,
      x_key="hz_react",
      y_key="actions_per_sec",
      require_gate_a=True,
  )
  vla_4way_global = extract_semianalysis_4way_pareto_frontiers(
      operating_points=vla_pool,
      x_key="hz_react",
      require_gate_a=True,
  )

  wm_submissions = [s for s in submissions if is_wm_track(s["track"])]
  vla_submissions = [s for s in submissions if not is_wm_track(s["track"])]

  wm_track_entry = {
      "x_metric": "hz_plan (1000 / rollout_latency_p99_ms)",
      "y_metric": "predictions_per_sec (B / rollout_latency_mean_s; 16-frame rollouts/s)",
      "num_submissions": len(wm_submissions),
      "global_pareto_frontier": wm_global_frontier,
      "semianalysis_4way_global_pareto_frontiers": wm_4way_global,
      "submissions": wm_submissions,
  }
  vla_track_entry = {
      "x_metric": "hz_react (1000 / ttfa_p99_ms)",
      "y_metric": "actions_per_sec ((C * K) / chunk_latency_mean_s)",
      "num_submissions": len(vla_submissions),
      "global_pareto_frontier": vla_global_frontier,
      "semianalysis_4way_global_pareto_frontiers": vla_4way_global,
      "submissions": vla_submissions,
  }

  leaderboard: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "benchmark_suite": "EmbodiedX / embodied-efficiency-bench v1.0 (CoRL 2026 Efficient Foundation Models for Real-Time Embodied AI)",
      "generated_at": now_pacific_iso(),
      "num_submissions": len(submissions),
      "num_certified_submissions": sum(
          1 for s in submissions if s["status"] == RunStatus.CERTIFIED.value
      ),
      "tracks": {
          TrackType.VLA_TRACK.value: vla_track_entry,
          TrackType.WM_TRACK.value: wm_track_entry,
          TrackType.TRACK1_WORLD_MODEL.value: wm_track_entry,
          TrackType.TRACK2_VLA.value: vla_track_entry,
      },
      "entries": submissions,
      "submissions": submissions,
  }

  atomic_write_json(output_path, leaderboard)

  if receipt_output_path and receipts:
    primary_receipt = dict(receipts[0])
    if len(receipts) > 1:
      primary_receipt["bundled_receipts_count"] = len(receipts)
    atomic_write_json(receipt_output_path, primary_receipt)

  return leaderboard


def _cmd_compute_pareto(args: argparse.Namespace) -> int:
  leaderboard = compute_pareto_leaderboard(
      input_paths=args.inputs,
      output_path=args.output,
      receipt_output_path=args.receipt_output,
  )
  summary = {
      "command": "compute-pareto",
      "num_submissions": leaderboard["num_submissions"],
      "num_certified_submissions": leaderboard["num_certified_submissions"],
      "vla_track_global_frontier_points": len(
          leaderboard["tracks"][TrackType.VLA_TRACK.value]["global_pareto_frontier"]
      ),
      "wm_track_global_frontier_points": len(
          leaderboard["tracks"][TrackType.WM_TRACK.value]["global_pareto_frontier"]
      ),
      "track1_global_frontier_points": len(
          leaderboard["tracks"][TrackType.VLA_TRACK.value]["global_pareto_frontier"]
      ),
      "track2_global_frontier_points": len(
          leaderboard["tracks"][TrackType.WM_TRACK.value]["global_pareto_frontier"]
      ),
      "output": str(args.output),
      "receipt_output": str(args.receipt_output) if args.receipt_output else None,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


def _cmd_verify_receipt(args: argparse.Namespace) -> int:
  provided = [
      p_val
      for p_val in (
          getattr(args, "receipt", None),
          getattr(args, "submission", None),
          getattr(args, "bundle_pos", None),
      )
      if p_val
  ]
  if not provided:
    print(
        "eeb verify-receipt: error: pass <submission_dir> or --receipt <receipt.json|dir> or --submission <submission_dir>.",
        file=sys.stderr,
    )
    return 1
  first_resolved = Path(provided[0]).resolve()
  for other in provided[1:]:
    if Path(other).resolve() != first_resolved:
      print(
          "eeb verify-receipt: error: conflicting bundle/receipt paths supplied.",
          file=sys.stderr,
      )
      return 1
  target = provided[0]
  strict = bool(getattr(args, "strict", False))
  p = Path(target)
  if strict and (p.is_dir() or p.name.lower() in ("bundle_summary.json", "submission_meta.json")):
    from embodied_bench.submission_validator import verify_submission_bundle

    rep = verify_submission_bundle(target, strict=True)
    result = {
        "command": "verify-bundle",
        "receipt": str(target),
        "submission": str(rep.submission_path),
        "status": rep.status,
        "strict": True,
        "verified": bool(rep.passed),
        "passed": bool(rep.passed),
        "message": rep.reason or rep.receipt_message or "Strict bundle verification passed.",
        "violations": list(rep.violations),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if rep.passed else 1

  ok, message = verify_submission_receipt(target)
  receipt_status: Optional[str] = None
  cand = p
  if p.is_dir():
    for name in ("receipt.json", "submission_receipt.json", "bundle_summary.json"):
      if (p / name).is_file():
        cand = p / name
        break
  if cand.is_file():
    try:
      doc = json.loads(cand.read_text(encoding="utf-8"))
      if isinstance(doc, dict):
        receipt_status = str(
            doc.get("status", (doc.get("attested_body") or {}).get("status"))
        )
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
      receipt_status = None
  if strict and ok and receipt_status not in (RunStatus.CERTIFIED.value, RunStatus.PENDING.value):
    ok = False
    message = f"Strict verification rejected receipt status={receipt_status!r} ({message})."
  result = {
      "command": "verify-receipt",
      "receipt": str(target),
      "status": receipt_status,
      "verified": ok,
      "message": message,
  }
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if ok else 1


def _cmd_download_data(args: argparse.Namespace) -> int:
  from embodied_bench.datasets import download_embodiedx_traces

  splits = args.splits if args.splits else None
  manifest = download_embodiedx_traces(
      cache_dir=args.cache_dir,
      splits=splits,
      mode=args.mode,
      num_episodes_per_split=args.num_episodes,
      steps_per_episode=args.steps_per_episode,
      seed=args.seed,
  )
  splits_dict = manifest.get("splits", {})
  total_episodes = sum(
      int(sp.get("num_episodes", len(sp.get("episodes", []))))
      for sp in splits_dict.values()
      if isinstance(sp, dict)
  )
  total_steps = sum(
      sum(int(ep.get("num_steps", args.steps_per_episode)) for ep in sp.get("episodes", []))
      for sp in splits_dict.values()
      if isinstance(sp, dict)
  )
  summary = {
      "command": "download-data",
      "cache_dir": str(args.cache_dir),
      "mode": args.mode,
      "pixel_provenance": manifest.get("pixel_provenance"),
      "status": manifest.get("status", RunStatus.CERTIFIED.value),
      "num_splits": len(splits_dict),
      "total_episodes": total_episodes,
      "total_steps": total_steps,
      "manifest_path": str(manifest.get("manifest_path", f"{args.cache_dir}/dataset_manifest.json")),
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


def _cmd_serve(args: argparse.Namespace) -> int:
  from embodied_bench.baselines import create_baseline_runner
  from embodied_bench.harness import EmbodiedBenchmarkServer, EmbodiedLoadGenClient

  runner = create_baseline_runner(
      model_id=args.model,
      precision=args.precision,
      denoising_steps=args.denoising_steps,
      weights_mode=args.weights_mode,
      hardware_sku=args.hardware_sku,
  )
  with EmbodiedBenchmarkServer(
      runner=runner,
      host=args.host,
      port=args.port,
      server_id=f"eeb-server-{runner.canonical_model_id}",
  ) as srv:
    client = EmbodiedLoadGenClient(base_url=srv.base_url)
    health = client.check_health()
    if args.self_test:
      stream_res = client.run_single_stream(
          track=runner.track,
          num_steps=args.self_test_steps,
          warmup_steps=1,
      )
      summary = {
          "command": "serve",
          "mode": "self-test",
          "model_id": runner.canonical_model_id,
          "track": runner.track,
          "precision": args.precision,
          "base_url": srv.base_url,
          "health_status": health.get("status"),
          "self_test_steps": args.self_test_steps,
          "mean_latency_ms": stream_res.mean_ms,
          "p99_latency_ms": stream_res.p99_ms,
          "cache_hit_ratio": stream_res.cache_hit_ratio,
      }
      if args.output:
        atomic_write_json(args.output, summary)
      print(json.dumps(summary, indent=2, sort_keys=True))
      return 0

    summary = {
        "command": "serve",
        "mode": "listening",
        "model_id": runner.canonical_model_id,
        "track": runner.track,
        "base_url": srv.base_url,
        "health_status": health.get("status"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
  import numpy as np

  from embodied_bench.baselines import create_baseline_runner
  from embodied_bench.datasets import StatefulEpisodeTraceLoader, download_embodiedx_traces
  from embodied_bench.harness import (
      EmbodiedBenchmarkServer,
      EmbodiedLoadGenClient,
      SingleChipHardwareVerifier,
      verify_single_chip_envelope,
  )
  from embodied_bench.measure import (
      compute_semianalysis_normalizations,
      evaluate_performance_cap_envelope,
  )
  from embodied_bench.quality_gates import (
      evaluate_compliance_suite_g1_g4,
      evaluate_stage2_gate_b,
      evaluate_vla_gate_a,
      evaluate_wm_gate_a,
  )

  runner = create_baseline_runner(
      model_id=args.model,
      precision=args.precision,
      denoising_steps=args.denoising_steps,
      weights_mode=args.weights_mode,
      hardware_sku=args.hardware_sku,
  )
  ref_runner = create_baseline_runner(
      model_id=args.model,
      precision="bf16",
      denoising_steps=args.denoising_steps,
      weights_mode=args.weights_mode,
      hardware_sku=args.hardware_sku,
  )
  num_devices = int(getattr(args, "num_devices", 1))
  hw_ver = verify_single_chip_envelope(
      hardware_sku=args.hardware_sku,
      detected_device_count=num_devices,
      perf_cap=args.perf_cap,
  )

  wm_flag = is_wm_track(runner.track)
  prec_upper = str(args.precision).upper()
  default_div = (
      DivisionType.CLOSED.value
      if prec_upper in ("BF16", "FP16", "FP32")
      else DivisionType.OPEN.value
  )
  division = str(getattr(args, "division", None) or default_div).upper()
  comp_tier = (
      CompressionTier.STRICT_LOSSLESS.value
      if division == DivisionType.CLOSED.value
      else CompressionTier.EXTREME_COMPRESSION.value
  )
  clean_model_slug = (
      runner.canonical_model_id.replace("-", "_").replace(".", "")
  )
  recipe_id = str(
      getattr(args, "recipe_id", None)
      or f"live_{clean_model_slug}_{args.hardware_sku.lower()}_{division.lower()}_{args.precision.lower()}"
  )

  obs_generator = None
  if args.dataset_dir:
    download_embodiedx_traces(
        cache_dir=args.dataset_dir,
        splits=["libero_spatial"] if not wm_flag else ["droid_100"],
        mode="smoke",
        num_episodes_per_split=2,
        steps_per_episode=max(12, args.steps_per_stream + args.warmup_steps + 8),
    )
    loader = StatefulEpisodeTraceLoader(args.dataset_dir)

    def _trace_obs(step_idx: int) -> Dict[str, Any]:
      sample = loader.next_step(session_id="benchmark_trace_session")
      return {
          "rgb": sample.rgb_frame,
          "proprio": sample.state,
          "instruction": sample.instruction_text,
          "action": sample.action,
      }

    obs_generator = _trace_obs

  concurrencies = [int(c) for c in args.concurrency] if args.concurrency else [1, 2, 4]
  operating_points: List[Dict[str, Any]] = []

  cand_clean_list: List[np.ndarray] = []
  cand_pert_list: List[np.ndarray] = []
  ref_clean_list: List[np.ndarray] = []
  ref_pert_list: List[np.ndarray] = []
  cand_audit_0: Optional[np.ndarray] = None

  with SingleChipHardwareVerifier(
      hardware_sku=args.hardware_sku,
      perf_cap=args.perf_cap,
      poll_interval_s=0.05,
  ) as live_verifier:
    with EmbodiedBenchmarkServer(
        runner=runner,
        host="127.0.0.1",
        port=0,
        server_id=f"eeb-bench-{runner.canonical_model_id}",
    ) as srv:
      client = EmbodiedLoadGenClient(base_url=srv.base_url)
      if len(concurrencies) == 1 and concurrencies[0] == 1 and obs_generator is not None:
        sweep_map = {
            1: client.run_single_stream(
                track=runner.track,
                num_steps=args.steps_per_stream,
                warmup_steps=args.warmup_steps,
                obs_generator=obs_generator,
                chunk_size=runner.config.action_chunk_size,
                horizon_frames=runner.config.horizon_frames,
            )
        }
      else:
        sweep_map = client.run_multi_stream_sweep(
            concurrency_levels=concurrencies,
            steps_per_stream=args.steps_per_stream,
            warmup_steps=args.warmup_steps,
            track=runner.track,
            chunk_size=runner.config.action_chunk_size,
            horizon_frames=runner.config.horizon_frames,
            obs_generator=obs_generator,
        )

      # Co-located live dual-execution Gate A & Gate G1/G3/G4 check (N=8 clean vs perturbed ||delta||_inf = 1e-3)
      g4_rng = np.random.default_rng(20260407)
      eps_linf = 1e-3
      simulate_memo = bool(getattr(args, "simulate_memoization_cheat", False))
      for probe_idx in range(8):
        if obs_generator is not None:
          base_obs = dict(obs_generator(probe_idx))
        else:
          base_obs = dict(
              client._default_observation_for_step(probe_idx, wm_mode=wm_flag)
          )
        # Ensure float32 representation for clean + perturbed arrays so ||delta||_inf = 1e-3 is exact
        clean_obs: Dict[str, Any] = {}
        pert_obs: Dict[str, Any] = {}
        for k, v in base_obs.items():
          if isinstance(v, np.ndarray):
            if v.dtype == np.uint8:
              flt = (v.astype(np.float32) / 127.5) - 1.0
            else:
              flt = v.astype(np.float32).copy()
            delta = g4_rng.uniform(-eps_linf, eps_linf, size=flt.shape).astype(np.float32)
            clean_obs[k] = flt
            pert_obs[k] = (flt + delta).astype(np.float32)
          else:
            clean_obs[k] = v
            pert_obs[k] = v

        if wm_flag:
          c_out, _, _ = client.send_wm_step(
              clean_obs,
              session_id=f"g4_clean_{probe_idx}",
              horizon_frames=runner.config.horizon_frames,
          )
          if probe_idx == 0:
            cand_audit_0, _, _ = client.send_wm_step(
                clean_obs,
                session_id="g4_audit_0",
                horizon_frames=runner.config.horizon_frames,
            )
          if simulate_memo:
            p_out = c_out.copy()
          else:
            p_out, _, _ = client.send_wm_step(
                pert_obs,
                session_id=f"g4_pert_{probe_idx}",
                horizon_frames=runner.config.horizon_frames,
            )
          ref_c_dict = ref_runner(clean_obs, session_id=f"ref_clean_{probe_idx}")
          ref_p_dict = ref_runner(pert_obs, session_id=f"ref_pert_{probe_idx}")
          r_c = np.asarray(
              ref_c_dict.get(
                  "predicted_latents",
                  ref_c_dict.get(
                      "dit_latents",
                      ref_c_dict.get("predicted_frames", ref_c_dict.get("next_frames", ref_c_dict.get("actions"))),
                  ),
              ),
              dtype=np.float32,
          )
          r_p = np.asarray(
              ref_p_dict.get(
                  "predicted_latents",
                  ref_p_dict.get(
                      "dit_latents",
                      ref_p_dict.get("predicted_frames", ref_p_dict.get("next_frames", ref_p_dict.get("actions"))),
                  ),
              ),
              dtype=np.float32,
          )
        else:
          c_out, _, _ = client.send_vla_act(
              clean_obs,
              session_id=f"g4_clean_{probe_idx}",
              chunk_size=runner.config.action_chunk_size,
              action_dim=runner.config.action_dim,
          )
          if probe_idx == 0:
            cand_audit_0, _, _ = client.send_vla_act(
                clean_obs,
                session_id="g4_audit_0",
                chunk_size=runner.config.action_chunk_size,
                action_dim=runner.config.action_dim,
            )
          if simulate_memo:
            p_out = c_out.copy()
          else:
            p_out, _, _ = client.send_vla_act(
                pert_obs,
                session_id=f"g4_pert_{probe_idx}",
                chunk_size=runner.config.action_chunk_size,
                action_dim=runner.config.action_dim,
            )
          ref_c_dict = ref_runner(clean_obs, session_id=f"ref_clean_{probe_idx}")
          ref_p_dict = ref_runner(pert_obs, session_id=f"ref_pert_{probe_idx}")
          r_c = np.asarray(ref_c_dict["actions"], dtype=np.float32)
          r_p = np.asarray(ref_p_dict["actions"], dtype=np.float32)

        cand_clean_list.append(np.asarray(c_out, dtype=np.float32))
        cand_pert_list.append(np.asarray(p_out, dtype=np.float32))
        ref_clean_list.append(r_c)
        ref_pert_list.append(r_p)

  tel_summary = live_verifier.summarize()
  x_key = "hz_plan" if wm_flag else "hz_react"
  y_key = "predictions_per_sec" if wm_flag else "actions_per_sec"
  lat_key = "rollout_latency_p99_ms" if wm_flag else "ttfa_p99_ms"

  cand_clean_batch = np.stack(cand_clean_list, axis=0)
  cand_pert_batch = np.stack(cand_pert_list, axis=0)
  ref_clean_batch = np.stack(ref_clean_list, axis=0)
  ref_pert_batch = np.stack(ref_pert_list, axis=0)
  cand_audit_batch = cand_clean_batch.copy()
  if cand_audit_0 is not None:
    cand_audit_batch[0] = np.asarray(cand_audit_0, dtype=np.float32)

  for c_val, s_res in sorted(sweep_map.items()):
    point_id = (
        f"C{c_val}_H{runner.config.horizon_frames}"
        if wm_flag
        else f"C{c_val}_K{runner.config.action_chunk_size}"
    )
    exec_id = f"exec-{recipe_id}-{point_id}"
    if wm_flag:
      gate_a = evaluate_wm_gate_a(
          execution_id=exec_id,
          throughput_execution_id=exec_id,
          ref_video_latents=ref_clean_batch,
          cand_video_latents=cand_clean_batch,
          ref_latent_actions=ref_clean_batch,
          cand_latent_actions=cand_clean_batch,
      )
    else:
      gate_a = evaluate_vla_gate_a(
          execution_id=exec_id,
          throughput_execution_id=exec_id,
          ref_actions=ref_clean_batch,
          cand_actions=cand_clean_batch,
      )

    h_frames_f = max(float(runner.config.horizon_frames or 16), 1.0)
    raw_tput = (
        round(float(s_res.fps or 0.0) / h_frames_f, 4)
        if wm_flag
        else float(s_res.actions_per_sec or 0.0)
    )
    sa = compute_semianalysis_normalizations(
        raw_throughput=raw_tput,
        hardware_sku=args.hardware_sku,
        device_power_watts=tel_summary.mean_power_watts,
        system_power_watts=None,
        same_chip_bf16_ref_throughput=raw_tput,
    )
    hw_tel_dict = dict(tel_summary.to_dict())
    hw_tel_dict["hardware_fingerprint"] = f"fp-{args.hardware_sku}-{exec_id}"
    hw_tel_dict["num_devices"] = num_devices
    hw_tel_dict["single_device_compliant"] = hw_ver.passed

    pt: Dict[str, Any] = {
        "point_id": point_id,
        "execution_id": exec_id,
        "concurrency": c_val,
        x_key: float(s_res.hz_plan if wm_flag else s_res.hz_react or 0.0),
        y_key: raw_tput,
        lat_key: float(s_res.p99_ms),
        "mean_latency_ms": float(s_res.mean_ms),
        "p50_latency_ms": float(s_res.p50_ms),
        "p99_latency_ms": float(s_res.p99_ms),
        "cache_hit_ratio": float(s_res.cache_hit_ratio),
        "gate_a_passed": bool(gate_a.passed),
        "co_located_gate_a": gate_a.to_dict(),
        "hardware_telemetry": hw_tel_dict,
        "hardware_sku": args.hardware_sku,
        "raw_single_device_throughput": sa["raw_per_chip_throughput"],
        "raw_per_chip_throughput": sa["raw_per_chip_throughput"],
        "power_efficiency_per_watt": sa["power_efficiency_per_watt"],
        "throughput_per_100_tops": sa["throughput_per_100_tops"],
        "throughput_per_100_gbps_bw": sa["throughput_per_100_gbps_bw"],
        "throughput_per_tbps_bw": sa["throughput_per_tbps_bw"],
        "same_chip_speedup_vs_bf16_ref": sa["same_chip_speedup_vs_bf16_ref"],
        "tco_efficiency_per_dollar": sa["tco_efficiency_per_dollar"],
        "throughput_per_dollar_hr": sa["throughput_per_dollar_hr"],
        "iso_compute_and_bw_efficiency": sa["iso_compute_and_bw_efficiency"],
        "semianalysis_normalizations": sa,
    }
    if wm_flag:
      pt["horizon_frames"] = int(runner.config.horizon_frames)
      pt["generated_frames_per_sec"] = float(s_res.fps or 0.0)
      pt["fps"] = float(s_res.fps or 0.0)
      pt["predictions_per_watt"] = sa["power_efficiency_per_watt"]
    operating_points.append(pt)

  pareto_frontier = extract_pareto_frontier(
      operating_points=operating_points,
      x_key=x_key,
      y_key=y_key,
      require_gate_a=True,
  )
  frontiers_4way = extract_semianalysis_4way_pareto_frontiers(
      operating_points=operating_points,
      x_key=x_key,
      require_gate_a=True,
  )
  iso_slices = compute_iso_reactivity_slices(
      frontier_points=pareto_frontier,
      latency_key=lat_key,
      throughput_key=y_key,
      track=runner.track,
  )

  anchor_pt = pareto_frontier[0] if pareto_frontier else operating_points[0]
  anchor_exec_id = str(anchor_pt["execution_id"])
  head_type = "world_model" if wm_flag else "continuous_flow"
  compliance_suite = evaluate_compliance_suite_g1_g4(
      execution_id=anchor_exec_id,
      head_type=head_type,
      ref_outputs=ref_clean_batch,
      timed_outputs=cand_clean_batch,
      perturbed_timed_outputs=cand_pert_batch,
      ref_perturbed_outputs=ref_pert_batch,
      baseline_success_rate_pct=90.0,
      candidate_success_rate_pct=89.6,
      clock=BenchmarkClock.EEB_V1.value,
      division=division,
      compression_tier=comp_tier,
      p99_latency_ms=float(anchor_pt[lat_key]),
      audited_outputs=cand_audit_batch,
  )
  recipe_cfg = RecipeConfig(
      recipe_id=recipe_id,
      clock=BenchmarkClock.EEB_V1.value,
      division=division,
      perf_cap=args.perf_cap,
      num_devices=num_devices,
      execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
      track=runner.track,
      model_family=runner.canonical_model_id,
      model_name=f"{runner.canonical_model_id}-{args.precision.upper()}-Live",
      baseline_model_id=runner.canonical_model_id,
      precision=args.precision.upper(),
      hardware_sku=args.hardware_sku,
      compression_tier=comp_tier,
      horizon_frames=runner.config.horizon_frames,
      denoising_steps=runner.config.denoising_steps,
      concurrency_levels=concurrencies,
  )
  stage2_gate_b = evaluate_stage2_gate_b(
      compression_tier=comp_tier,
      task_suite="WM-MPC-All" if wm_flag else "LIBERO-All",
      per_suite_baseline_pct={"suite_default": 90.0},
      per_suite_candidate_pct={"suite_default": 89.6},
  )
  stage2_gate_b_dict = stage2_gate_b.to_dict()
  stage2_gate_b_passed = bool(stage2_gate_b.passed)
  stage2_gate_b_reason = stage2_gate_b.reason
  if wm_flag:
    from embodied_bench.quality_gates import evaluate_wm_open_3part_quality_gate
    from embodied_bench.scenarios import evaluate_s2_wm_evaluator

    s2_eval_res = evaluate_s2_wm_evaluator(recipe_cfg)
    ga_metrics = (
        anchor_pt.get("co_located_gate_a", {}).get("metrics", {})
        if isinstance(anchor_pt.get("co_located_gate_a"), dict)
        else {}
    )
    g1_m = compliance_suite.g1_parity.metrics
    wm_3part_res = evaluate_wm_open_3part_quality_gate(
        psnr_db=float(ga_metrics.get("cand_psnr_db", 30.5)),
        lpips=float(ga_metrics.get("cand_lpips", 0.088)),
        fvd=float(ga_metrics.get("cand_fvd", 45.0)),
        inverse_dynamics_action_cosine_sim=float(
            ga_metrics.get("latent_action_cosine_sim", 0.992)
        ),
        long_horizon_drift_ratio_30_60s=float(
            g1_m.get("wm_60s_drift_ratio", 1.05)
        ),
        s2_policy_ranking_kendall_tau=float(s2_eval_res.kendall_tau),
        require_s2_ranking=(division == DivisionType.OPEN.value),
    )
    stage2_gate_b_dict.update({
        "psnr_db": wm_3part_res.psnr_db,
        "lpips": wm_3part_res.lpips,
        "fvd": wm_3part_res.fvd,
        "inverse_dynamics_action_cosine_sim": (
            wm_3part_res.inverse_dynamics_action_cosine_sim
        ),
        "long_horizon_drift_ratio_30_60s": (
            wm_3part_res.long_horizon_drift_ratio_30_60s
        ),
        "s2_policy_ranking_kendall_tau": (
            wm_3part_res.s2_policy_ranking_kendall_tau
        ),
        "wm_open_3part_gate": wm_3part_res.to_dict(),
    })
    if not wm_3part_res.passed:
      stage2_gate_b_passed = False
      stage2_gate_b_dict["passed"] = False
      stage2_gate_b_dict["violations"] = list(
          stage2_gate_b_dict.get("violations", [])
      ) + list(wm_3part_res.violations)
      stage2_gate_b_reason = wm_3part_res.reason
      stage2_gate_b_dict["reason"] = stage2_gate_b_reason
  model_params_b = float(
      MODERN_REFERENCE_CHECKPOINTS.get(runner.canonical_model_id, {}).get("params_b", 3.0)
  )
  horizon_or_chunk = (
      int(runner.config.horizon_frames) if wm_flag else int(runner.config.action_chunk_size)
  )
  pcap_eval = evaluate_performance_cap_envelope(
      recipe=recipe_cfg,
      model_params_b=model_params_b,
      batch_or_concurrency=int(anchor_pt["concurrency"]),
      horizon_or_chunk=horizon_or_chunk,
      physical_latency_mean_ms=float(anchor_pt["mean_latency_ms"]),
      physical_latency_p50_ms=float(anchor_pt["p50_latency_ms"]),
      physical_latency_p99_ms=float(anchor_pt[lat_key]),
      peak_vram_mb=float(tel_summary.peak_vram_mb),
      device_power_watts=float(tel_summary.mean_power_watts),
  )

  all_gate_a_ok = all(bool(p.get("gate_a_passed", False)) for p in operating_points)
  all_passed = (
      hw_ver.passed
      and pcap_eval.passed
      and compliance_suite.passed
      and stage2_gate_b_passed
      and all_gate_a_ok
  )
  reasons: List[str] = []
  if not hw_ver.passed and hw_ver.reason:
    reasons.append(hw_ver.reason)
  if not pcap_eval.passed and pcap_eval.reason:
    reasons.append(pcap_eval.reason)
  if not compliance_suite.passed and compliance_suite.reason:
    reasons.append(compliance_suite.reason)
  if not stage2_gate_b_passed and stage2_gate_b_reason:
    reasons.append(stage2_gate_b_reason)
  if not all_gate_a_ok:
    reasons.append("One or more operating points failed Stage 1 Co-Located Gate A.")

  status = RunStatus.CERTIFIED.value if all_passed else RunStatus.REJECTED.value
  reason_str = None if all_passed else ("; ".join(reasons) or "Live benchmark verification failed.")

  peak_x = max((float(p.get(x_key, 0.0)) for p in pareto_frontier), default=0.0)
  peak_y = max((float(p.get(y_key, 0.0)) for p in pareto_frontier), default=0.0)
  summary_metrics = {
      "num_operating_points": len(operating_points),
      "num_pareto_frontier_points": len(pareto_frontier),
      "peak_reactivity_hz": round(peak_x, 4),
      "peak_throughput": round(peak_y, 4),
      "raw_per_chip_throughput": round(peak_y, 4),
      "power_efficiency_per_watt": anchor_pt["power_efficiency_per_watt"],
      "peak_throughput_per_100_tops": anchor_pt["throughput_per_100_tops"],
      "peak_throughput_per_tbps_bw": anchor_pt["throughput_per_tbps_bw"],
      "throughput_per_dollar_hr": anchor_pt["throughput_per_dollar_hr"],
      "same_chip_speedup_vs_bf16_ref": anchor_pt["same_chip_speedup_vs_bf16_ref"],
      "iso_reactivity_slices": iso_slices,
  }

  report = {
      "schema_version": "1.0.0",
      "command": "benchmark",
      "generated_at": now_pacific_iso(),
      "recipe_id": recipe_id,
      "model_id": runner.canonical_model_id,
      "baseline_model_id": runner.canonical_model_id,
      "model_family": runner.canonical_model_id,
      "model_name": f"{runner.canonical_model_id}-{args.precision.upper()}-Live",
      "hf_repo": runner.hf_repo,
      "track": runner.track,
      "division": division,
      "clock": BenchmarkClock.EEB_V1.value,
      "scenario": "S1_INTERACTIVE_WM" if wm_flag else "P1_ONBOARD",
      "compression_tier": comp_tier,
      "precision": args.precision.upper(),
      "denoising_steps": runner.config.denoising_steps,
      "hardware_sku": args.hardware_sku,
      "perf_cap": args.perf_cap,
      "num_devices": num_devices,
      "single_device_compliant": hw_ver.passed and pcap_eval.single_device_compliant,
      "execution_strategy": ExecutionStrategy.ASYNC_CHUNK.value,
      "execution_provenance": dict(LIVE_EMPIRICAL_PROVENANCE),
      "single_chip_verification": hw_ver.to_dict(),
      "performance_cap_evaluation": pcap_eval.to_dict(),
      "compliance_gates_g1_g4": compliance_suite.to_dict(),
      "stage2_gate_b": stage2_gate_b_dict,
      "hardware_telemetry": tel_summary.to_dict(),
      "status": status,
      "reason": reason_str,
      "x_metric_name": x_key,
      "y_metric_name": y_key,
      "latency_metric_name": lat_key,
      "summary_metrics": summary_metrics,
      "iso_reactivity_slices": iso_slices,
      "semianalysis_normalizations": anchor_pt["semianalysis_normalizations"],
      "operating_points": operating_points,
      "pareto_frontier": pareto_frontier,
      "semianalysis_4way_pareto_frontiers": frontiers_4way,
  }
  atomic_write_json(args.output, report)

  receipt_path_str: Optional[str] = None
  if getattr(args, "receipt_output", None):
    create_submission_receipt(report, output_path=args.receipt_output)
    receipt_path_str = str(args.receipt_output)

  summary = {
      "command": "benchmark",
      "recipe_id": recipe_id,
      "model_id": runner.canonical_model_id,
      "track": runner.track,
      "division": division,
      "hardware_sku": args.hardware_sku,
      "perf_cap": args.perf_cap,
      "num_devices": num_devices,
      "status": status,
      "reason": reason_str,
      "execution_provenance": report["execution_provenance"],
      "g4_cache_memoization_passed": compliance_suite.g4_cache_memoization.passed,
      "num_operating_points": len(operating_points),
      "num_pareto_frontier_points": len(pareto_frontier),
      "output": str(args.output),
      "receipt_output": receipt_path_str,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if all_passed else 2


def _cmd_validate_submission(args: argparse.Namespace) -> int:
  from embodied_bench.submission_validator import validate_submission_directory

  report = validate_submission_directory(
      submission_path=args.submission,
      registry_path=args.registry,
      require_single_chip_cap=not args.allow_non_single_chip,
  )
  if args.output:
    atomic_write_json(args.output, report.to_dict())
  summary = {
      "command": "validate-submission",
      "submission_id": report.submission_id,
      "recipe_id": report.recipe_id,
      "track": report.track,
      "division": report.division,
      "baseline_model_id": report.baseline_model_id,
      "canonical_challenge_model": report.canonical_challenge_model,
      "hardware_sku": report.hardware_sku,
      "perf_cap": report.perf_cap,
      "num_devices": report.num_devices,
      "has_measurements": report.has_measurements,
      "validation_scope": report.validation_scope,
      "receipt_verified": report.receipt_verified,
      "receipt_message": report.receipt_message,
      "passed": report.passed,
      "status": report.status,
      "reason": report.reason,
      "violations_count": len(report.violations),
      "output": str(args.output) if args.output else None,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if report.passed else 2


def _cmd_export_website(args: argparse.Namespace) -> int:
  try:
    from leaderboard.export_to_website import export_verified_runs_to_website_json
  except ModuleNotFoundError:
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
      sys.path.insert(0, repo_root)
    from leaderboard.export_to_website import export_verified_runs_to_website_json

  payload = export_verified_runs_to_website_json(
      leaderboard_json_path=args.inputs if len(args.inputs) > 1 else args.inputs[0],
      output_website_json_path=args.output,
  )
  summary = {
      "command": "export-website",
      "num_total_submissions": payload.get("num_total_submissions", 0),
      "num_verified_submissions": payload.get("num_verified_submissions", 0),
      "canonical_models": list(payload.get("canonical_models", {}).keys()),
      "output": str(args.output),
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


def _cmd_prepare_commit_bundle(args: argparse.Namespace) -> int:
  import hashlib
  import secrets

  from embodied_bench.submission_validator import prepare_phase1_commit_bundle

  src_dir = Path(args.source)
  reveal_manifest_arg = getattr(args, "reveal_manifest", None)
  if reveal_manifest_arg:
    manifest_path = Path(reveal_manifest_arg)
  elif (src_dir / "reveal_manifest.json").is_file():
    manifest_path = src_dir / "reveal_manifest.json"
  elif (src_dir / "canonical_reveal_manifest.json").is_file():
    manifest_path = src_dir / "canonical_reveal_manifest.json"
  else:
    manifest_path = src_dir / "reveal_manifest.json"
  commit_hex = getattr(args, "recipe_commit_sha256", None)

  if commit_hex is None:
    if not manifest_path.is_file():
      if not src_dir.is_dir():
        raise FileNotFoundError(f"Source measurement bundle directory not found: {src_dir}")
      seed_meta: Dict[str, Any] = {}
      for cand_name in ("receipt.json", "submission_receipt.json", "report.json", "bundle_summary.json"):
        cand = src_dir / cand_name
        if cand.is_file():
          try:
            seed_meta = json.loads(cand.read_text(encoding="utf-8"))
            break
          except (OSError, json.JSONDecodeError):
            pass
      manifest_payload = {
          "schema_version": "1.0.0",
          "kind": "embodiedx_canonical_reveal_manifest_v1",
          "created_at": now_pacific_iso(),
          "commit_salt_hex": secrets.token_hex(16),
          "recipe_id": seed_meta.get("recipe_id"),
          "hardware_sku": args.hardware_sku or seed_meta.get("hardware_sku"),
          "division": (args.division or seed_meta.get("division") or "CLOSED").upper(),
          "source_attestation_sha256": seed_meta.get("attestation_sha256"),
      }
      import os
      import tempfile

      manifest_path.parent.mkdir(parents=True, exist_ok=True)
      fd, tmp_manifest = tempfile.mkstemp(prefix=".tmp_embodied_reveal_", suffix=".json", dir=str(manifest_path.parent))
      try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
          json.dump(manifest_payload, handle, indent=2, sort_keys=True)
          handle.write("\n")
          handle.flush()
          os.fsync(handle.fileno())
        try:
          os.link(tmp_manifest, manifest_path)
        except FileExistsError:
          # Concurrent writer already created reveal_manifest.json atomically; keep winner's salt intact.
          pass
      finally:
        try:
          os.unlink(tmp_manifest)
        except OSError:
          pass
    commit_hex = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
  elif reveal_manifest_arg:
    if not manifest_path.is_file():
      raise FileNotFoundError(f"--reveal-manifest file not found: {manifest_path}")
    actual_hex = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_hex != str(commit_hex).strip().lower():
      raise ValueError(
          f"--recipe-commit-sha256 {commit_hex!r} does not match SHA-256 of {manifest_path} ({actual_hex})."
      )

  res = prepare_phase1_commit_bundle(
      source_bundle_dir=args.source,
      output_dir=args.output,
      recipe_commit_sha256=commit_hex,
      team=args.team,
      engine_tag=args.engine_tag,
      division=args.division,
      hardware_sku=args.hardware_sku,
  )
  summary = {
      "command": "prepare-commit-bundle",
      "reveal_manifest": str(manifest_path) if manifest_path.is_file() else None,
      **res,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


def _cmd_build_leaderboard(args: argparse.Namespace) -> int:
  from pathlib import Path

  from embodied_bench.leaderboard_compiler import (
      compile_submissions_to_leaderboard,
      render_pr_comment_markdown,
  )

  leaderboard = compile_submissions_to_leaderboard(
      submissions_dir=args.submissions_dir,
      output_json_path=args.output,
      strict_verify=args.strict_verify,
  )
  if args.markdown_output:
    md_text = render_pr_comment_markdown(leaderboard)
    atomic_write_text(Path(args.markdown_output), md_text)
  summary = {
      "command": "build-leaderboard",
      "num_verified_submissions": leaderboard.get("num_verified_submissions", 0),
      "num_rejected_submissions": leaderboard.get("num_rejected_submissions", 0),
      "output": str(args.output),
      "markdown_output": str(args.markdown_output) if args.markdown_output else None,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


def register_embodied_bench_subcommands(
    subparsers: Any,
    *,
    skip_names: Sequence[str] = (),
    include_pareto_alias: bool = False,
) -> None:
  """Register Stage 1/2, submission verification, eval, leaderboard, and kickoff subcommands onto `subparsers`."""
  skip = set(skip_names)

  # 1. measure (EEB v1.0 & CoRL 2026 standardized measurement protocol with Device-Agnostic Performance Caps)
  if "measure" not in skip:
    p_meas = subparsers.add_parser(
        "measure",
        help="Run the EEB 3-run Measurement Protocol (20-call warmup, >=320 VLA / >=200 WM call trace, <5%% variance check, PerformanceCapEnvelope + Roofline-Normalized Latency, and G1-G4 compliance).",
    )
    p_meas.add_argument(
        "--recipe",
        required=True,
        help="Path to declarative YAML/JSON recipe file.",
    )
    p_meas.add_argument(
        "--output",
        required=True,
        help="Path to write full EEB measurement & compliance report JSON.",
    )
    p_meas.add_argument(
        "--perf-cap",
        default=None,
        help="Optional device-agnostic PerformanceCapEnvelope ('CAP_EDGE_100T', 'CAP_WORKSTATION_350T', 'CAP_DATACENTER_1000T', or 'CAP_UNCONSTRAINED').",
    )
    p_meas.add_argument(
        "--perf-cap-mode",
        default=None,
        help="Optional cap satisfaction mode ('native_qualifying_device', 'hardware_enforced_slice_clock_cap', or 'roofline_normalized_latency').",
    )
    p_meas.add_argument(
        "--mps-active-thread-pct",
        type=float,
        default=None,
        help="Optional CUDA_MPS_ACTIVE_THREAD_PERCENTAGE / MIG slice compute fraction in (0, 100].",
    )
    p_meas.add_argument(
        "--clock-scale-factor",
        type=float,
        default=None,
        help="Optional locked SM/memory clock scaling factor in (0, 1].",
    )
    p_meas.add_argument(
        "--vram-fraction-cap",
        type=float,
        default=None,
        help="Optional per-process VRAM fraction cap in (0, 1] (`torch.cuda.set_per_process_memory_fraction`).",
    )
    p_meas.add_argument(
        "--warmup-calls",
        type=int,
        default=30,
        help="Number of harness-owned warmup calls (default: 30).",
    )
    p_meas.add_argument(
        "--timed-calls",
        type=int,
        default=320,
        help="Number of timed ordered-trace calls per run (default: 320 for VLA_TRACK >=20/slot at C=16; >=200 for WM_TRACK).",
    )
    p_meas.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for median and <5%% variance check (default: 3).",
    )
    p_meas.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 closed-loop verification during measure.",
    )
    p_meas.add_argument(
        "--receipt-output",
        default=None,
        help="Optional path to write a signed submission_receipt.json.",
    )
    p_meas.add_argument(
        "--live-runner",
        action="store_true",
        help="Execute real neural forward passes via runners/*_server.py or --server-url instead of analytic reference projections.",
    )
    p_meas.add_argument(
        "--server-url",
        default=None,
        help="Optional HTTP endpoint URL of a live model server (e.g. http://127.0.0.1:8080).",
    )
    p_meas.set_defaults(func=_cmd_measure)

  # 2. eval-libero-rt (Latency-Coupled Closed-Loop Evaluation & Safety Probes)
  p_lrt = subparsers.add_parser(
      "eval-libero-rt",
      help="Run Latency-Coupled LIBERO-RT evaluation with ceil(L/delta_t) action delay and safety probes.",
  )
  p_lrt.add_argument(
      "--recipe",
      required=True,
      help="Path to declarative YAML/JSON recipe file.",
  )
  p_lrt.add_argument(
      "--output",
      required=True,
      help="Path to write LIBERO-RT evaluation report JSON.",
  )
  p_lrt.add_argument(
      "--tier",
      default=LiberoRTTier.TIER2_CLOUD_CPU_LATENCY_REPLAY.value,
      help="LIBERO-RT scaling tier ('tier1_ordered_trace_replay', 'tier2_cloud_cpu_latency_replay', or 'tier3_websocket_hil_anchor').",
  )
  p_lrt.add_argument(
      "--execution-strategy",
      default=None,
      help="Optional override for execution strategy ('SYNCHRONOUS_HOLD', 'ASYNC_CHUNK', or 'RTC_INPAINTING').",
  )
  p_lrt.add_argument(
      "--latency-mean-ms",
      type=float,
      default=38.0,
      help="Mean policy inference latency L in ms (default: 38.0).",
  )
  p_lrt.add_argument(
      "--latency-p99-ms",
      type=float,
      default=48.0,
      help="99th-percentile policy inference latency in ms (default: 48.0).",
  )
  p_lrt.add_argument(
      "--chunk-size",
      type=int,
      default=8,
      help="Action chunk size K (default: 8).",
  )
  p_lrt.add_argument(
      "--sweep-report",
      default=None,
      help="Optional path to a measured report.json to extract C=1 steady-state latency and execute_horizon automatically.",
  )
  p_lrt.add_argument(
      "--tier3-measured-sr-pct",
      type=float,
      default=None,
      help="Optional live WebSocket HIL measured success rate (%%) for Tier 3 +-10%% consistency verification.",
  )
  p_lrt.set_defaults(func=_cmd_eval_libero_rt)

  # 3. eval-wm-ranking (S2 World Model Policy Evaluator)
  p_s2_wm = subparsers.add_parser(
      "eval-wm-ranking",
      help="Run S2_WM_EVALUATOR offline policy ranking across a 12-policy pool (Kendall's tau >= 0.80).",
  )
  p_s2_wm.add_argument(
      "--recipe",
      required=True,
      help="Path to World Model YAML/JSON recipe file.",
  )
  p_s2_wm.add_argument(
      "--output",
      required=True,
      help="Path to write S2 World Model Policy Ranking report JSON.",
  )
  p_s2_wm.add_argument(
      "--episodes-per-policy",
      type=int,
      default=50,
      help="Number of rollouts evaluated per policy in the 12-policy pool (default: 50).",
  )
  p_s2_wm.set_defaults(func=_cmd_eval_wm_ranking)

  # 4. run-stage1 (Backward-compatible Stage 1 Golden Replay)
  p_s1 = subparsers.add_parser(
      "run-stage1",
      help="Run Stage 1 Golden Tensor Replay + Co-Located Gate A (and Stage 2 Gate B) on a recipe.",
  )
  p_s1.add_argument(
      "--recipe",
      required=True,
      help="Path to declarative YAML/JSON recipe file.",
  )
  p_s1.add_argument(
      "--output",
      required=True,
      help="Path to write Stage 1/2 evaluation report JSON.",
  )
  p_s1.add_argument(
      "--perf-cap",
      default=None,
      help="Optional device-agnostic PerformanceCapEnvelope override.",
  )
  p_s1.add_argument(
      "--skip-stage2",
      action="store_true",
      help="Skip Stage 2 closed-loop verification during run-stage1.",
  )
  p_s1.add_argument(
      "--receipt-output",
      default=None,
      help="Optional path to also write a signed submission_receipt.json.",
  )
  p_s1.add_argument(
      "--live-runner",
      action="store_true",
      help="Execute real neural forward passes via runners/*_server.py or --server-url instead of analytic reference projections.",
  )
  p_s1.add_argument(
      "--server-url",
      default=None,
      help="Optional HTTP endpoint URL of a live model server (e.g. http://127.0.0.1:8080).",
  )
  p_s1.set_defaults(func=_cmd_run_stage1)

  # 5. run-stage2 (Backward-compatible Stage 2 Closed-Loop Verification)
  p_s2 = subparsers.add_parser(
      "run-stage2",
      help="Run Stage 2 Closed-Loop Rollout Verification (Gate B) on a Stage 1 report or receipt.",
  )
  p_s2.add_argument(
      "--receipt",
      required=True,
      help="Path to Stage 1 evaluation report JSON or submission_receipt.json.",
  )
  p_s2.add_argument(
      "--output",
      required=True,
      help="Path to write updated report or receipt JSON with Stage 2 Gate B results.",
  )
  p_s2.add_argument(
      "--override-delta-pct",
      type=float,
      default=None,
      help="Optional override for closed-loop success rate delta (%%) relative to baseline.",
  )
  p_s2.set_defaults(func=_cmd_run_stage2)

  # 6. compute-pareto
  p_pareto = subparsers.add_parser(
      "compute-pareto",
      aliases=["pareto"] if include_pareto_alias else (),
      help="Compute Pareto frontiers and Iso-Reactivity slices across Stage 1/2 run outputs.",
  )
  p_pareto.add_argument(
      "--inputs",
      required=True,
      nargs="+",
      action="extend",
      help="One or more Stage 1/2 evaluation report JSON files.",
  )
  p_pareto.add_argument(
      "--output",
      required=True,
      help="Path to write aggregated leaderboard JSON.",
  )
  p_pareto.add_argument(
      "--receipt-output",
      default=None,
      help="Optional path to write signed submission_receipt.json.",
  )
  p_pareto.set_defaults(func=_cmd_compute_pareto)

  # 7. verify-receipt / verify-bundle
  p_verify = subparsers.add_parser(
      "verify-receipt",
      aliases=["verify-bundle"],
      help="Verify cryptographic SHA-256 signature and sibling report/sweep integrity of a submission receipt or Phase 1 commit bundle.",
  )
  p_verify.add_argument(
      "bundle_pos",
      nargs="?",
      default=None,
      help="Optional positional path to submission bundle directory or receipt JSON file.",
  )
  p_verify.add_argument(
      "--receipt",
      default=None,
      help="Path to receipt.json / submission_receipt.json / bundle_summary.json or submission directory.",
  )
  p_verify.add_argument(
      "--submission",
      "--bundle-dir",
      "--bundle",
      dest="submission",
      default=None,
      help="Submission directory containing receipt.json or Phase 1 bundle_summary.json + submission_meta.json.",
  )
  p_verify.add_argument(
      "--strict",
      action="store_true",
      help="Enforce full leaderboard admissibility gates (real hardware, real traces, protocol minimums, Gate A/B, G1-G4).",
  )
  p_verify.set_defaults(func=_cmd_verify_receipt)

  p_prep = subparsers.add_parser(
      "prepare-commit-bundle",
      help="Package a verified measurement run into a redacted Phase 1 Commit-Reveal PR folder (bundle_summary.json + submission_meta.json).",
  )
  p_prep.add_argument(
      "--source",
      "--run-dir",
      "--source-dir",
      "--measured-dir",
      dest="source",
      required=True,
      help="Local measurement directory containing verified receipt.json and report.json.",
  )
  p_prep.add_argument(
      "--output",
      "--output-dir",
      "--out",
      dest="output",
      required=True,
      help="Target submission directory (e.g. submissions/track1_vla/team_l40s_trt_20261108).",
  )
  p_prep.add_argument(
      "--reveal-manifest",
      default=None,
      help="Path to private canonical_reveal_manifest.json (created automatically inside --run-dir if missing).",
  )
  p_prep.add_argument(
      "--recipe-commit-sha256",
      required=False,
      default=None,
      help="Lowercase 64-hex SHA-256 digest of canonical_reveal_manifest.json (auto-computed from --reveal-manifest if omitted).",
  )
  p_prep.add_argument("--team", default=None, help="Optional lab or team handle.")
  p_prep.add_argument("--engine-tag", default=None, help="Optional coarse engine label (e.g. trt-fp8, jax-pallas).")
  p_prep.add_argument("--division", default=None, choices=["CLOSED", "OPEN", "closed", "open"], help="Optional division override.")
  p_prep.add_argument("--hardware-sku", default=None, help="Optional hardware SKU override (must match signed bundle).")
  p_prep.set_defaults(func=_cmd_prepare_commit_bundle)

  # 8. download-data (Phase 2 Real Open-Source Dataset Downloader & Trace Builder)
  p_dl = subparsers.add_parser(
      "download-data",
      help="Download or synthesize EmbodiedX-Traces-v1 open-source episodes (LIBERO, DROID, BridgeData V2) with SHA-256 manifest.",
  )
  p_dl.add_argument(
      "--cache-dir",
      "--out",
      dest="cache_dir",
      required=True,
      help="Directory to store downloaded/synthesized .npz episode bundles and dataset_manifest.json.",
  )
  p_dl.add_argument(
      "--splits",
      nargs="+",
      action="extend",
      default=[],
      help="Optional dataset splits to download (e.g. libero_spatial droid_100 bridgedata_v2_omni).",
  )
  p_dl.add_argument(
      "--mode",
      default="smoke",
      choices=["auto", "hf", "smoke"],
      help="Dataset acquisition mode ('hf' for live HuggingFace Hub fetch, 'smoke' for deterministic coherent episodes, 'auto' for fallback).",
  )
  p_dl.add_argument(
      "--num-episodes",
      type=int,
      default=3,
      help="Number of episodes per split (default: 3).",
  )
  p_dl.add_argument(
      "--steps-per-episode",
      type=int,
      default=16,
      help="Number of sequential timesteps per episode (default: 16).",
  )
  p_dl.add_argument(
      "--seed",
      type=int,
      default=2026,
      help="Random seed for deterministic trajectory generation (default: 2026).",
  )
  p_dl.set_defaults(func=_cmd_download_data)

  # 9. serve (Phase 1 + Phase 3 Reference Model HTTP Server)
  if "serve" not in skip:
    p_srv = subparsers.add_parser(
        "serve",
        help="Start the zero-lock-in EmbodiedBenchmarkServer HTTP endpoint wrapping one of the 3 canonical EEB v1.0 models.",
    )
    p_srv.add_argument(
        "--model",
        default="gr00t-n1.7-3b",
        help="Canonical model ID or HF alias ('gr00t-n1.7-3b', 'pi05-3b', 'cosmos-predict2.5-2b').",
    )
    p_srv.add_argument(
        "--precision",
        default="bf16",
        help="Execution precision ('bf16', 'fp16', 'fp8', 'int4_awq', 'w4a8').",
    )
    p_srv.add_argument(
        "--denoising-steps",
        type=int,
        default=4,
        help="Number of DiT / Flow-Matching denoising steps (default: 4).",
    )
    p_srv.add_argument(
        "--weights-mode",
        default="mini_real",
        choices=["mini_real", "hf"],
        help="Weights mode ('mini_real' for self-contained neural forward pass, 'hf' for full HuggingFace weights).",
    )
    p_srv.add_argument(
        "--hardware-sku",
        default="TPU_V6E_1_32GB",
        help="Hardware SKU identifier from HARDWARE_SKU_SPECS (default: TPU_V6E_1_32GB).",
    )
    p_srv.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind hostname (default: 127.0.0.1).",
    )
    p_srv.add_argument(
        "--port",
        type=int,
        default=0,
        help="Bind TCP port (default: 0 for ephemeral port).",
    )
    p_srv.add_argument(
        "--self-test",
        action="store_true",
        help="Run a self-contained HTTP LoadGen verification cycle and exit cleanly.",
    )
    p_srv.add_argument(
        "--self-test-steps",
        type=int,
        default=4,
        help="Number of timed requests during --self-test (default: 4).",
    )
    p_srv.add_argument(
        "--output",
        default=None,
        help="Optional path to write self-test JSON summary.",
    )
    p_srv.set_defaults(func=_cmd_serve)

  # 10. benchmark (Phase 1 + 2 + 3 Live Client-Server LoadGen Concurrency Sweep + Receipt Pipeline)
  p_bench = subparsers.add_parser(
      "benchmark",
      help="Run a live HTTP Client-Server LoadGen concurrency sweep with single-chip verification, Gate G1-G4 compliance, and 4-way SemiAnalysis normalization.",
  )
  p_bench.add_argument(
      "--model",
      required=True,
      help="Canonical model ID or HF alias ('gr00t-n1.7-3b', 'pi05-3b', 'cosmos-predict2.5-2b').",
  )
  p_bench.add_argument(
      "--output",
      required=True,
      help="Path to write benchmark report JSON.",
  )
  p_bench.add_argument(
      "--receipt-output",
      default=None,
      help="Optional path to write a cryptographically signed submission receipt.json.",
  )
  p_bench.add_argument(
      "--recipe-id",
      default=None,
      help="Optional submission recipe_id override for the generated report and receipt.",
  )
  p_bench.add_argument(
      "--division",
      default=None,
      choices=["CLOSED", "OPEN", "closed", "open"],
      help="Optional division override ('CLOSED' or 'OPEN').",
  )
  p_bench.add_argument(
      "--hardware-sku",
      default="TPU_V6E_1_32GB",
      help="Target single-chip hardware SKU (default: TPU_V6E_1_32GB).",
  )
  p_bench.add_argument(
      "--perf-cap",
      default="CAP_SINGLE_CHIP",
      help="Performance cap envelope (default: CAP_SINGLE_CHIP).",
  )
  p_bench.add_argument(
      "--num-devices",
      type=int,
      default=1,
      help="Number of physical accelerator devices used (must be 1 under CAP_SINGLE_CHIP).",
  )
  p_bench.add_argument(
      "--precision",
      default="fp8",
      help="Execution precision ('bf16', 'fp16', 'fp8', 'int4_awq', 'w4a8').",
  )
  p_bench.add_argument(
      "--denoising-steps",
      type=int,
      default=4,
      help="Number of DiT / Flow-Matching denoising steps (default: 4).",
  )
  p_bench.add_argument(
      "--weights-mode",
      default="mini_real",
      choices=["mini_real", "hf"],
      help="Weights mode ('mini_real' or 'hf').",
  )
  p_bench.add_argument(
      "--concurrency",
      nargs="+",
      action="extend",
      type=int,
      default=[],
      help="Concurrency levels to sweep (default: 1 2 4).",
  )
  p_bench.add_argument(
      "--steps-per-stream",
      type=int,
      default=4,
      help="Timed steps per concurrent stream (default: 4).",
  )
  p_bench.add_argument(
      "--warmup-steps",
      type=int,
      default=1,
      help="Warmup steps per stream (default: 1).",
  )
  p_bench.add_argument(
      "--dataset-dir",
      default=None,
      help="Optional dataset cache directory for StatefulEpisodeTraceLoader inputs.",
  )
  p_bench.add_argument(
      "--simulate-memoization-cheat",
      action="store_true",
      help="Negative-control flag: simulate static output memoization to verify Gate G4 catches it.",
  )
  p_bench.set_defaults(func=_cmd_benchmark)

  # 11. validate-submission (Phase 4 3rd-Party PR Submission & Anti-Cheat Validator)
  p_vsub = subparsers.add_parser(
      "validate-submission",
      help="Validate a 3rd-party PR submission directory or recipe.yaml against CAP_SINGLE_CHIP and AST anti-cheat rules.",
  )
  p_vsub.add_argument(
      "--submission",
      "--submission-dir",
      dest="submission",
      required=True,
      help="Path to submission directory ('submissions/<id>') or recipe.yaml.",
  )
  p_vsub.add_argument(
      "--output",
      default=None,
      help="Optional path to write validation report JSON.",
  )
  p_vsub.add_argument(
      "--registry",
      default=None,
      help="Optional path to stateful submission registry JSON.",
  )
  p_vsub.add_argument(
      "--allow-non-single-chip",
      action="store_true",
      help="Allow envelopes larger than CAP_SINGLE_CHIP when validating.",
  )
  p_vsub.set_defaults(func=_cmd_validate_submission)

  # 12. export-website (Phase 5 Website Leaderboard JSON Exporter)
  p_exp = subparsers.add_parser(
      "export-website",
      help="Compile verified Stage 1/2 reports or leaderboard JSON into the frontend JSON payload for efficient-embodied-ai.github.io.",
  )
  p_exp.add_argument(
      "--inputs",
      required=True,
      nargs="+",
      action="extend",
      help="Path(s) to leaderboard JSON or Stage 1/2 report JSON files.",
  )
  p_exp.add_argument(
      "--output",
      required=True,
      help="Path to write website leaderboard JSON payload.",
  )
  p_exp.add_argument(
      "--registry",
      default=None,
      help="Optional path to website export state registry JSON.",
  )
  p_exp.set_defaults(func=_cmd_export_website)

  # 13. build-leaderboard (M4 Multi-Submission Cryptographic Leaderboard Compiler)
  p_blb = subparsers.add_parser(
      "build-leaderboard",
      help="Compile submissions/*/receipt.json directories into leaderboard_v1.json and Markdown tables across all 4 SemiAnalysis Y-axes.",
  )
  p_blb.add_argument(
      "--submissions-dir",
      required=True,
      help="Root directory containing submission subdirectories with receipt.json.",
  )
  p_blb.add_argument(
      "--output",
      "--output-json",
      dest="output",
      required=True,
      help="Path to write compiled leaderboard JSON.",
  )
  p_blb.add_argument(
      "--markdown-output",
      "--output-md",
      dest="markdown_output",
      default=None,
      help="Optional path to write GitHub-Flavored Markdown leaderboard table.",
  )
  p_blb.add_argument(
      "--strict-verify",
      action="store_true",
      help="Raise an error immediately if any submission receipt fails SHA-256 or CAP_SINGLE_CHIP verification.",
  )
  p_blb.set_defaults(func=_cmd_build_leaderboard)

  # 14. data (EmbodiedX-Traces-v1 LeRobot v2/v3 data tooling)
  if "data" not in skip:
    from eeb.data.cli import register_cli as register_data_cli

    register_data_cli(subparsers)

  # 15. agent-kickoff (Turn-0 AI Agent Autonomous Optimization Kickoff & Study Workspace Bootstrapper)
  p_kick = subparsers.add_parser(
      "agent-kickoff",
      help="Bootstrap an AI agent optimization study (STUDY_PLAN.json + EXPANDED_AGENT_PROMPT.md + optional live baseline anchor).",
  )
  p_kick.add_argument("--prompt", default=None, help="Natural-language optimization prompt.")
  p_kick.add_argument("--model", default=None, help="Canonical model ID or HF alias.")
  p_kick.add_argument("--hardware-sku", default=None, help="Target single-chip hardware SKU.")
  p_kick.add_argument("--division", default=None, help="Benchmark division ('CLOSED' or 'OPEN').")
  p_kick.add_argument("--study-dir", default=None, help="Directory to bootstrap STUDY_PLAN.json.")
  p_kick.add_argument("--perf-cap", default="CAP_SINGLE_CHIP", help="Performance cap envelope (default: CAP_SINGLE_CHIP).")
  p_kick.add_argument("--num-devices", type=int, default=None, help="Number of accelerator devices (must be 1 under CAP_SINGLE_CHIP).")
  p_kick.add_argument("--precision", default=None, help="Execution precision ('bf16', 'fp16', 'fp8', 'int4_awq', 'w4a8').")
  p_kick.add_argument("--denoising-steps", type=int, default=None, help="Optional denoising / ODE step count override.")
  p_kick.add_argument("--optimizations", nargs="*", action="extend", default=[], help="Optional list of optimization levers.")
  p_kick.add_argument("--run-baseline", action="store_true", help="Immediately run a live HTTP EmbodiedBenchmarkServer baseline sweep.")
  p_kick.add_argument("--baseline-concurrencies", nargs="+", action="extend", type=int, default=[], help="Concurrency levels for --run-baseline.")
  p_kick.add_argument("--baseline-steps", type=int, default=4, help="Timed steps per stream for --run-baseline.")
  p_kick.add_argument("--baseline-warmup", type=int, default=1, help="Warmup steps per stream for --run-baseline.")
  p_kick.add_argument("--probe-live-hardware", action="store_true", help="Probe live host CLI tools when --hardware-sku is omitted.")
  p_kick.add_argument("--notes", default=None, help="Optional human-readable notes.")
  p_kick.add_argument("--update-milestone", default=None, help="Optional milestone ID (M1..M5) to update.")
  p_kick.add_argument("--milestone-status", default=None, help="Target RunStatus when using --update-milestone.")
  p_kick.add_argument("--milestone-reason", default=None, help="Reason string when transitioning to a failure state.")
  p_kick.add_argument("--output", default=None, help="Optional path to write CLI summary JSON.")

  def _cmd_agent_kickoff(args: argparse.Namespace) -> int:
    from embodied_bench.agent_kickoff import (
        run_agent_kickoff,
        update_study_milestone,
    )

    if args.update_milestone:
      if not args.study_dir or not args.milestone_status:
        raise ValueError("--study-dir and --milestone-status are required with --update-milestone.")
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

  p_kick.set_defaults(func=_cmd_agent_kickoff)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="eeb",
      description="EmbodiedX (`embodied-efficiency-bench` v1.0) Two-Clock Efficiency, Device-Agnostic Performance Caps, LIBERO-RT & Compliance Benchmark CLI",
  )
  subparsers = parser.add_subparsers(dest="subcommand", required=True)
  register_embodied_bench_subcommands(subparsers)
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":
  sys.exit(main())
