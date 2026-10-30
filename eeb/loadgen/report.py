"""JSON report + SHA-256 receipt for `eeb measure`, reusing `embodied_bench` schema/attestation/measure.

Honesty rules baked into the writer:

* `execution_provenance.loadgen` is always `REAL_WALLCLOCK` (every latency is a measured
  `perf_counter` delta); `execution_provenance.policy` is the server's own `Caps.server_impl`
  tag, so a `SYNTHETIC_DUMMY` run can never masquerade as a model measurement.
* Power / Joules fields are `None` unless the telemetry window was `MEASURED` (or the operator
  explicitly passed `--assume-power-watts`, which is labelled `ESTIMATED`). We never fill TDP
  tables into a measured report.
* `status` is `PENDING` for an admissible self-measured run (ADR-5: `CERTIFIED` only after an
  organizer re-run) and `INVALID` with a concrete `reason` otherwise (synthetic policy, envelope
  failure, cross-run variance, errors, failed gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Tuple
import uuid

from eeb import LOADGEN_PROVENANCE_REAL_WALLCLOCK, PROVENANCE_ESTIMATED, PROVENANCE_MEASURED, PROVENANCE_UNAVAILABLE, __version__
from eeb.envelope.check import EnvelopeCheckResult
from eeb.loadgen.gate import CoLocatedGateResult
from eeb.loadgen.sweep import ConcurrencyResult, SweepResult
from eeb.telemetry.sidecar import TelemetrySummary

REPORT_SCHEMA_VERSION = "1.0.0"
REPORT_KIND = "eeb_loadgen_measure"
X_METRIC = "hz_react"
Y_METRIC = "executed_actions_per_sec"
Y_METRIC_RAW = "actions_per_sec"
LATENCY_METRIC = "obs_to_action_p99_ms"
WM_X_METRIC = "hz_plan"
WM_Y_METRIC = "predictions_per_sec"
WM_Y_METRIC_FRAMES = "generated_frames_per_sec"
WM_LATENCY_METRIC = "rollout_latency_p99_ms"

CANONICAL_LOCKED_EXECUTE_HORIZONS: Dict[str, int] = {
    "pi05": 5,
    "pi0.5": 5,
    "pi05-3b": 5,
    "pi0.5-3b": 5,
    "gr00t": 8,
    "groot": 8,
    "gr00t-n1.7": 8,
    "gr00t-n1.7-3b": 8,
    "cosmos-predict2.5": 16,
    "cosmos-predict2.5-2b": 16,
    "cosmos_predict25": 16,
}


def canonical_locked_execute_horizon(recipe: Optional[Dict[str, Any]], caps: Any) -> Optional[int]:
  """Return the locked `execute_horizon` (or `horizon_frames`) for known canonical v1.0 models."""
  rec = recipe or {}
  candidates = [
      rec.get("model_family"),
      rec.get("model_name"),
      rec.get("baseline_model_id"),
      getattr(caps, "model_id", None),
  ]
  for cand in candidates:
    if not cand:
      continue
    tok = str(cand).strip().lower()
    if tok in CANONICAL_LOCKED_EXECUTE_HORIZONS:
      return CANONICAL_LOCKED_EXECUTE_HORIZONS[tok]
    if "pi05" in tok or "pi0.5" in tok:
      return 5
    if "gr00t" in tok or "groot" in tok:
      return 8
    if "cosmos-predict2.5" in tok or "cosmos_predict25" in tok or "cosmos-predict25" in tok:
      return 16
  return None


def extract_iso_reactivity_slices(
    frontier_points: List[Dict[str, Any]],
    latency_key: str = LATENCY_METRIC,
    throughput_key: str = Y_METRIC,
    track: Optional[str] = None,
) -> Dict[str, Any]:
  """Extract iso-reactivity throughput slices using `executed_actions_per_sec` (when present) as canonical Y."""
  from embodied_bench.pareto import compute_iso_reactivity_slices

  if not frontier_points:
    return {}
  eff_y = (
      throughput_key
      if any(pt.get(throughput_key) is not None for pt in frontier_points)
      else Y_METRIC_RAW
  )
  return compute_iso_reactivity_slices(
      frontier_points=frontier_points,
      latency_key=latency_key,
      throughput_key=eff_y,
      track=track,
  )


@dataclass
class ReportInputs:
  sweep: SweepResult
  recipe: Optional[Dict[str, Any]] = None  # RecipeConfig.to_dict() or None for ad-hoc runs
  recipe_path: Optional[str] = None
  envelope: Optional[EnvelopeCheckResult] = None
  telemetry_by_run: Dict[Tuple[int, int], TelemetrySummary] = field(default_factory=dict)
  telemetry_source: Optional[Dict[str, Any]] = None
  gates: Dict[int, CoLocatedGateResult] = field(default_factory=dict)
  gate: Optional[CoLocatedGateResult] = None
  allow_sanity_only: bool = False
  assumed_power_w: Optional[float] = None
  notes: List[str] = field(default_factory=list)


def _median(vals: List[float]) -> Optional[float]:
  clean = [float(v) for v in vals if v is not None and v == v]
  return round(float(statistics.median(clean)), 4) if clean else None


def _point_telemetry(
    point: ConcurrencyResult,
    inputs: ReportInputs,
    sku: Optional[str],
    device_name: str,
    total_mem_mb: float,
    *,
    is_wm: bool = False,
    effective_tput: float = 0.0,
) -> Dict[str, Any]:
  from embodied_bench.schema import HARDWARE_SKU_SPECS
  from embodied_bench.telemetry import compute_hardware_fingerprint

  overhead_w = 0.0

  summaries = [inputs.telemetry_by_run.get((point.concurrency, r.run_idx)) for r in point.runs]
  present = [s for s in summaries if s is not None]
  provs = {s.provenance for s in present}
  chip_energy_j_total = 0.0
  window_s_total = 0.0
  predicted_steps_total = 0
  executed_units_total = 0
  chunks_total = 0
  means: List[float] = []
  peaks: List[float] = []
  mems: List[float] = []
  for run, summ in zip(point.runs, summaries):
    if summ is None:
      continue
    if summ.peak_mem_used_mb is not None:
      mems.append(summ.peak_mem_used_mb)
    if summ.energy_j is not None and summ.mean_power_w is not None:
      chip_energy_j_total += summ.energy_j
      window_s_total += float(run.window_s)
      predicted_steps_total += run.num_timed_calls * run.horizon
      executed_units_total += run.num_timed_calls * (1 if is_wm else run.execute_horizon)
      chunks_total += run.num_timed_calls
      means.append(summ.mean_power_w)
      peaks.append(summ.peak_power_w or summ.mean_power_w)
  if present and provs == {PROVENANCE_MEASURED} and means:
    provenance = PROVENANCE_MEASURED
  elif present and means and PROVENANCE_UNAVAILABLE not in provs:
    provenance = PROVENANCE_ESTIMATED
  else:
    provenance = PROVENANCE_UNAVAILABLE
  source = inputs.telemetry_source or {}
  reason = None
  if provenance == PROVENANCE_UNAVAILABLE:
    reasons = {s.reason for s in present if s.reason}
    reason = "; ".join(sorted(reasons)) if reasons else "no telemetry summaries recorded for this point"
  mean_w = round(statistics.median(means), 3) if means else None
  peak_w = round(max(peaks), 3) if peaks else None

  if provenance == PROVENANCE_UNAVAILABLE and inputs.assumed_power_w is not None:
    provenance = PROVENANCE_ESTIMATED
    mean_w = round(float(inputs.assumed_power_w), 3)
    peak_w = mean_w
    window_s_total = sum(float(r.window_s) for r in point.runs)
    chunks_total = sum(r.num_timed_calls for r in point.runs)
    predicted_steps_total = sum(r.num_timed_calls * r.horizon for r in point.runs)
    executed_units_total = sum(
        r.num_timed_calls * (1 if is_wm else r.execute_horizon) for r in point.runs
    )
    chip_energy_j_total = mean_w * window_s_total
    reason = f"power ESTIMATED from operator-supplied --assume-power-watts={inputs.assumed_power_w} (no live telemetry)"

  is_measured_pwr = provenance == PROVENANCE_MEASURED
  sys_w = round(mean_w + overhead_w, 3) if mean_w is not None else None
  sys_energy_j_total = (chip_energy_j_total + overhead_w * window_s_total) if (is_measured_pwr and mean_w is not None) else None
  chip_jps = (
      round(chip_energy_j_total / predicted_steps_total, 6)
      if is_measured_pwr and mean_w is not None and predicted_steps_total > 0 and chip_energy_j_total > 0
      else None
  )
  unit_sys_j: Optional[float] = None
  if is_measured_pwr and sys_w is not None and sys_w > 0:
    if effective_tput and effective_tput > 0:
      unit_sys_j = round(sys_w / float(effective_tput), 6)
    elif sys_energy_j_total is not None and executed_units_total > 0 and sys_energy_j_total > 0:
      unit_sys_j = round(sys_energy_j_total / executed_units_total, 6)
  jpc = (
      round(sys_energy_j_total / chunks_total, 6)
      if is_measured_pwr and sys_energy_j_total is not None and chunks_total > 0 and sys_energy_j_total > 0
      else None
  )

  telemetry_src_name = (
      source.get("name") or source.get("source", {}).get("name")
      if isinstance(source, dict)
      else None
  )
  if provenance != PROVENANCE_MEASURED:
    telemetry_scope = "UNAVAILABLE"
  elif telemetry_src_name == "nvml":
    telemetry_scope = "DISCRETE_BOARD_TBP_NVML"
  elif telemetry_src_name == "tegrastats":
    telemetry_scope = "EDGE_SOM_VDD_IN_TEGRASTATS"
  elif telemetry_src_name == "amd_smi":
    telemetry_scope = "DISCRETE_BOARD_TBP_AMD_SMI"
  elif telemetry_src_name == "tpu_info":
    telemetry_scope = "ACCELERATOR_BOARD_TPU_INFO"
  else:
    telemetry_scope = "MEASURED_BOARD_TELEMETRY"

  limit_vals = [
      float(s.extra["power_limit_w"])
      for s in present
      if isinstance(getattr(s, "extra", None), dict)
      and isinstance(s.extra.get("power_limit_w"), (int, float))
      and not isinstance(s.extra.get("power_limit_w"), bool)
  ]
  if not limit_vals and isinstance(source, dict) and isinstance(source.get("power_limit_w"), (int, float)):
    limit_vals = [float(source["power_limit_w"])]
  power_limit_w = round(float(statistics.median(limit_vals)), 2) if limit_vals else None

  return {
      "hardware_sku": sku,
      "device_name": device_name,
      "total_vram_mb": total_mem_mb,
      "hardware_fingerprint": compute_hardware_fingerprint(sku or "UNKNOWN_SKU", device_name, total_mem_mb),
      "telemetry_source": telemetry_src_name,
      "power_telemetry_scope": telemetry_scope,
      "power_limit_w": power_limit_w,
      "power_provenance": provenance,
      "power_reason": reason,
      "mean_power_watts": mean_w,
      "peak_power_watts": peak_w,
      "system_overhead_watts": overhead_w if mean_w is not None else None,
      "system_power_watts": sys_w,
      "chip_energy_j": round(chip_energy_j_total, 4) if mean_w is not None else None,
      "system_energy_j": round(sys_energy_j_total, 4) if sys_energy_j_total is not None else None,
      "energy_j": round(sys_energy_j_total, 4) if sys_energy_j_total is not None else None,
      "chip_joules_per_predicted_step": chip_jps,
      "joules_per_action": (None if is_wm else unit_sys_j),
      "joules_per_prediction": (unit_sys_j if is_wm else None),
      "joules_per_frame": (
          round(unit_sys_j / max(int(point.runs[0].horizon), 1), 6)
          if (is_wm and unit_sys_j is not None and point.runs)
          else None
      ),
      "joules_per_chunk": jpc,
      "peak_mem_used_mb": round(max(mems), 2) if mems else None,
      "peak_mem_used_gb": round(max(mems) / 1024.0, 4) if mems else None,
      "peak_mem_provenance": PROVENANCE_MEASURED if mems else PROVENANCE_UNAVAILABLE,
      "num_runs_with_power": len(means),
  }


def _normalizations(actions_per_sec: float, sku: Optional[str], hw: Dict[str, Any]) -> Dict[str, Any]:
  from embodied_bench.measure import compute_semianalysis_normalizations
  from embodied_bench.schema import HARDWARE_SKU_SPECS

  sys_w = hw.get("system_power_watts")
  is_measured_power = hw.get("power_provenance") == PROVENANCE_MEASURED
  if not sku or sku not in HARDWARE_SKU_SPECS:
    pwr_eff_no_sku = (
        round(float(actions_per_sec) / float(sys_w), 4)
        if is_measured_power and sys_w is not None and float(sys_w) > 0.0
        else None
    )
    return {
        "raw_per_chip_throughput": round(float(actions_per_sec), 4),
        "power_efficiency_per_watt": pwr_eff_no_sku,
        "system_power_watts": sys_w if is_measured_power else None,
        "power_provenance": hw.get("power_provenance"),
        "power_telemetry_scope": hw.get("power_telemetry_scope"),
        "provenance": PROVENANCE_UNAVAILABLE,
        "reason": f"hardware_sku {sku!r} not in HARDWARE_SKU_SPECS; iso-TOPS / per-$ normalizations need a known SKU",
    }
  sa = compute_semianalysis_normalizations(
      raw_per_chip_throughput=float(actions_per_sec),
      system_power_watts=float(sys_w) if (is_measured_power and sys_w is not None) else None,
      hardware_sku=sku,
  )
  sa["spec_provenance"] = "SPEC_TABLE"
  sa["power_provenance"] = hw.get("power_provenance")
  sa["power_telemetry_scope"] = hw.get("power_telemetry_scope")
  if not is_measured_power or sys_w is None:
    # Never report a per-Watt number derived from a TDP table or operator --assume-power-watts estimate.
    sa["power_efficiency_per_watt"] = None
    sa["system_power_watts"] = None
    sa["power_reason"] = hw.get("power_reason") or "power telemetry not MEASURED"
  return sa


def build_report(inputs: ReportInputs) -> Dict[str, Any]:
  from embodied_bench import __version__ as eb_version
  from embodied_bench.pareto import extract_pareto_frontier
  from embodied_bench.schema import RunStatus, now_pacific_iso

  sweep = inputs.sweep
  caps = sweep.caps
  recipe = dict(inputs.recipe or {})
  sku = recipe.get("hardware_sku")
  track_val = str(recipe.get("track") or caps.track or "VLA_TRACK")
  is_wm = track_val == "WM_TRACK"
  env = inputs.envelope
  device_name = (env.device or {}).get("name") if env and env.device else "NO_ACCELERATOR_DEVICE"
  total_mem = float((env.device or {}).get("total_memory_mb") or 0.0) if env and env.device else 0.0

  admissibility: List[str] = []
  if caps.is_synthetic:
    admissibility.append(f"policy server_impl={caps.server_impl!r} is synthetic: harness validation only, never a model result")
  if env is None:
    admissibility.append("envelope check not executed")
  elif not env.passed:
    admissibility.append(f"envelope check failed: {env.reason}")
  elif env.num_devices == 0:
    admissibility.append("envelope check passed only with allow_no_device (CPU-only host)")
  if not sweep.config.protocol_compliant_for_track(track_val):
    admissibility.append("sweep below protocol minimums: " + "; ".join(sweep.config.protocol_notes_for_track(track_val)))

  from eeb.data.schema import SYNTHETIC_CI_SMOKE_TRACES_KIND, is_synthetic_trace_manifest
  from eeb.loadgen.gate import evaluate_wire_observation_resolution

  if is_synthetic_trace_manifest(sweep.observation_source) and not inputs.allow_sanity_only:
    admissibility.append(
        f"observation source uses non-certifiable synthetic smoke traces ({SYNTHETIC_CI_SMOKE_TRACES_KIND}); "
        "rebuild traces from real pinned episodes without --synthetic-ci-smoke (RT-02)"
    )

  wire_ok, wire_bytes, wire_hw, wire_reason = evaluate_wire_observation_resolution(
      sweep.observation_source,
      is_wm=is_wm,
  )
  if not wire_ok:
    admissibility.append(wire_reason)

  division = str(recipe.get("division", "CLOSED") or "CLOSED").upper()
  locked_h = canonical_locked_execute_horizon(recipe, caps)
  eff_sweep_h = int(sweep.pacing.effective_execute_horizon)
  if locked_h is not None and eff_sweep_h != locked_h:
    admissibility.append(
        f"{division} division locked execute_horizon violation: canonical model requires execute_horizon={locked_h}, "
        f"got effective_execute_horizon={eff_sweep_h} (evaluation horizon K_eval is strictly locked across both CLOSED and OPEN divisions)"
    )
  if is_wm and division == "CLOSED":
    sc_frames = recipe.get("stream_chunk_frames")
    if isinstance(sc_frames, (int, float)) and not isinstance(sc_frames, bool) and int(sc_frames) > 0:
      admissibility.append(
          f"CLOSED division stream_chunk_frames={int(sc_frames)} violation: pseudo-causal sub-clip chunking "
          "is disabled on full-clip bidirectional 3D-DiTs in Closed Division"
      )

  effective_gates: Dict[int, CoLocatedGateResult] = dict(inputs.gates)
  if inputs.gate is not None:
    for pt in sweep.points:
      effective_gates.setdefault(pt.concurrency, inputs.gate)

  operating_points: List[Dict[str, Any]] = []
  for point in sweep.points:
    gate = effective_gates.get(point.concurrency)
    exec_id = gate.execution_id if gate else f"{recipe.get('recipe_id', 'adhoc')}-C{point.concurrency}-{uuid.uuid4().hex[:8]}"
    effective_tput = (
        point.median_chunks_per_sec
        if is_wm
        else point.median_executed_actions_per_sec
    )
    hw = _point_telemetry(
        point,
        inputs,
        sku,
        device_name or "NO_ACCELERATOR_DEVICE",
        total_mem,
        is_wm=is_wm,
        effective_tput=float(effective_tput),
    )
    p50 = point.median_latency_ms.get("p50")
    p95 = point.median_latency_ms.get("p95")
    p99 = point.median_latency_ms.get("p99")
    lat_mean = point.median_latency_ms.get("mean")
    lat_max = point.median_latency_ms.get("max")
    hz_react = round(1000.0 / p99, 4) if p99 and p99 > 0 else 0.0
    exec_h = int(sweep.pacing.effective_execute_horizon)
    obs_staleness_ms = (
        round(float(p99) + float(max(exec_h - 1, 0)) * float(sweep.pacing.dt_ms), 4)
        if p99 is not None and p99 == p99
        else None
    )
    raw_ttff_p50 = point.median_ttff_ms.get("p50") if point.median_ttff_ms else None
    raw_ttff_p99 = point.median_ttff_ms.get("p99") if point.median_ttff_ms else None
    ttff_p50 = (
        round(float(raw_ttff_p50), 4)
        if raw_ttff_p50 is not None and raw_ttff_p50 == raw_ttff_p50
        else (point.median_ttfa_ms.get("p50") if is_wm else None)
    )
    ttff_p99 = (
        round(float(raw_ttff_p99), 4)
        if raw_ttff_p99 is not None and raw_ttff_p99 == raw_ttff_p99
        else (point.median_ttfa_ms.get("p99") if is_wm else None)
    )
    sa = _normalizations(effective_tput, sku, hw)
    gate_passed = bool(gate.passed) if gate else False
    op_jerk = (
        gate.normalized_action_jerk
        if (gate is not None and gate.normalized_action_jerk is not None)
        else point.median_normalized_action_jerk
    )
    if (
        not caps.is_synthetic
        and p50 is not None
        and lat_max is not None
        and float(p50) > 0.0
        and (float(lat_max) / float(p50)) > 4.0
    ):
      admissibility.append(
          f"C={point.concurrency}: bimodal latency tail violation max/p50={float(lat_max) / float(p50):.2f} > 4.0 "
          f"(max={float(lat_max):.2f} ms, p50={float(p50):.2f} ms; hybrid cache/fallback spike or severe scheduler stall)"
      )
    slot_ok = bool(getattr(point, "slot_starvation_passed", True))
    slot_reason = str(getattr(point, "slot_starvation_reason", "ok"))
    max_slot_spread = float(getattr(point, "max_slot_count_spread", 0.0) or 0.0)
    c1_tail_ok = bool(getattr(point, "c1_tail_ratio_passed", True))
    c1_tail_reason = str(getattr(point, "c1_tail_ratio_reason", "ok"))
    if not slot_ok:
      admissibility.append(f"C={point.concurrency}: {slot_reason}")
    if not caps.is_synthetic and point.concurrency == 1 and not c1_tail_ok:
      admissibility.append(f"C={point.concurrency}: {c1_tail_reason}")
    if not point.variance_passed:
      admissibility.append(f"C={point.concurrency}: {point.variance_reason}")
    if not point.all_runs_passed:
      admissibility.append(f"C={point.concurrency}: a run had errors / sticky-routing violations / inline canary failures / malformed chunks")
    if gate is None:
      admissibility.append(f"C={point.concurrency}: no co-located gate executed")
    elif not gate.passed:
      admissibility.append(f"C={point.concurrency}: gate {gate.mode} failed: {gate.reason}")
    elif not gate.quality_certified and not inputs.allow_sanity_only:
      admissibility.append(
          f"C={point.concurrency}: gate {gate.mode} is not quality_certified (SANITY_ONLY without --golden)"
      )
    if gate is not None and gate.mode == "GOLDEN_REPLAY" and not inputs.allow_sanity_only:
      g_meta = gate.golden if isinstance(gate.golden, dict) else {}
      g_samples = g_meta.get("num_samples", gate.checks.get("num_samples"))
      g_smoke = bool(g_meta.get("allow_smoke_golden") or gate.checks.get("allow_smoke_golden"))
      if g_smoke:
        admissibility.append(
            f"C={point.concurrency}: GOLDEN_REPLAY bundle has allow_smoke_golden=True (<100 samples; RT-03)"
        )
      elif isinstance(g_samples, int) and g_samples < 100:
        admissibility.append(
            f"C={point.concurrency}: GOLDEN_REPLAY num_samples={g_samples} < 100 minimum (RT-03)"
        )
    cold_ttfa_p50 = point.median_ttfa_ms.get("p50")
    cold_ttfa_p99 = point.median_ttfa_ms.get("p99")
    op: Dict[str, Any] = {
        "point_id": f"C{point.concurrency}",
        "execution_id": exec_id,
        "hardware_sku": sku,
        "concurrency": point.concurrency,
        "chunk_size": int(caps.horizon),
        "horizon": int(caps.horizon),
        "action_dim": int(caps.action_dim),
        "execute_horizon": exec_h,
        "obs_to_action_p50_ms": p50,
        "obs_to_action_p95_ms": p95,
        "obs_to_action_p99_ms": p99,
        "obs_to_action_mean_ms": lat_mean,
        "obs_to_action_max_ms": lat_max,
        "ttfa_p50_ms": cold_ttfa_p50,
        "ttfa_p99_ms": cold_ttfa_p99,
        "cold_session_ttfa_p50_ms": cold_ttfa_p50,
        "cold_session_ttfa_p99_ms": cold_ttfa_p99,
        "ttff_p50_ms": ttff_p50,
        "ttff_p99_ms": ttff_p99,
        "observation_staleness_ms": obs_staleness_ms,
        "peak_mem_used_gb": hw.get("peak_mem_used_gb"),
        "hz_react": hz_react,
        "chunks_per_sec": point.median_chunks_per_sec,
        "actions_per_sec": None if is_wm else point.median_actions_per_sec,
        "executed_actions_per_sec": None if is_wm else point.median_executed_actions_per_sec,
        "pacing_enabled": bool(sweep.pacing.enabled),
        "paced_wall_chunks_per_sec": (
            _median([r.paced_wall_chunks_per_sec for r in point.runs if r.paced_wall_chunks_per_sec is not None])
            if sweep.pacing.enabled
            else None
        ),
        "paced_wall_executed_actions_per_sec": (
            _median([r.paced_wall_executed_actions_per_sec for r in point.runs if r.paced_wall_executed_actions_per_sec is not None])
            if sweep.pacing.enabled
            else None
        ),
        "active_service_chunks_per_sec": _median(
            [r.active_service_chunks_per_sec for r in point.runs if r.active_service_chunks_per_sec is not None]
        ),
        "active_service_executed_actions_per_sec": (
            round(
                float(
                    _median(
                        [r.active_service_chunks_per_sec for r in point.runs if r.active_service_chunks_per_sec is not None]
                    )
                    or 0.0
                )
                * exec_h,
                4,
            )
            if any(r.active_service_chunks_per_sec is not None for r in point.runs)
            else None
        ),
        "frames_per_sec": point.median_actions_per_sec if is_wm else None,
        "normalized_action_jerk": op_jerk,
        "deadline_miss_rate": point.median_deadline_miss_rate,
        "transport_overhead_p50_ms": _median([r.transport_overhead_p50_ms for r in point.runs if r.transport_overhead_p50_ms is not None]),
        "cross_run_spread_pct": point.cross_run_spread_pct,
        "variance_passed": point.variance_passed,
        "variance_reason": point.variance_reason,
        "max_slot_count_spread": max_slot_spread,
        "slot_starvation_passed": slot_ok,
        "c1_tail_ratio_passed": c1_tail_ok,
        "raw_per_chip_throughput": sa.get("raw_per_chip_throughput"),
        "throughput_per_100_tops": sa.get("throughput_per_100_tops"),
        "throughput_per_tbps_bw": sa.get("throughput_per_tbps_bw"),
        "throughput_per_1k_usd": sa.get("throughput_per_1k_usd"),
        "tco_efficiency_per_dollar": sa.get("tco_efficiency_per_dollar"),
        "units_per_dollar": sa.get("units_per_dollar"),
        "throughput_per_dollar_hr": sa.get("throughput_per_dollar_hr"),
        "joules_per_action": hw.get("joules_per_action"),
        "joules_per_prediction": hw.get("joules_per_prediction"),
        "joules_per_frame": hw.get("joules_per_frame"),
        "joules_per_chunk": hw.get("joules_per_chunk"),
        "semianalysis_normalizations": sa,
        "gate_a_passed": gate_passed,
        "gate_a_mode": gate.mode if gate else None,
        "gate_mode": gate.mode if gate else None,
        "quality_certified": bool(gate.quality_certified) if gate else False,
        "co_located_gate_a": gate.to_dict() if gate else {"execution_id": exec_id, "passed": False, "reason": "gate not executed"},
        "hardware_telemetry": hw,
        "runs": [r.to_dict(include_samples=sweep.config.keep_samples) for r in point.runs],
    }
    if is_wm:
      op["batch_size"] = point.concurrency
      op["horizon_frames"] = int(caps.horizon)
      op["hz_plan"] = hz_react
      op["rollout_latency_p50_ms"] = p50
      op["rollout_latency_p95_ms"] = p95
      op["rollout_latency_p99_ms"] = p99
      op["rollout_latency_mean_ms"] = lat_mean
      op["rollout_latency_max_ms"] = lat_max
      op["predictions_per_sec"] = point.median_chunks_per_sec
      op["generated_frames_per_sec"] = point.median_actions_per_sec
      op["fps"] = point.median_actions_per_sec
    pwr_eff = (
        sa.get("power_efficiency_per_watt")
        if hw.get("power_provenance") == PROVENANCE_MEASURED
        else None
    )
    op["power_efficiency_per_watt"] = pwr_eff
    if is_wm:
      op["predictions_per_watt"] = pwr_eff
      frames_pw = round(float(pwr_eff) * float(max(int(caps.horizon), 1)), 4) if pwr_eff is not None else None
      op["frames_per_watt"] = frames_pw
      op["fps_per_watt"] = frames_pw
    else:
      op["actions_per_watt"] = pwr_eff
    operating_points.append(op)

  if not sweep.points and inputs.gate is not None and not inputs.gate.quality_certified and not inputs.allow_sanity_only:
    admissibility.append(f"gate {inputs.gate.mode} is not quality_certified (SANITY_ONLY without --golden)")

  x_key_effective = WM_X_METRIC if is_wm else X_METRIC
  lat_key_effective = WM_LATENCY_METRIC if is_wm else LATENCY_METRIC
  if is_wm:
    y_key_effective = WM_Y_METRIC
  else:
    y_key_effective = (
        Y_METRIC
        if any(op.get(Y_METRIC) is not None for op in operating_points)
        else Y_METRIC_RAW
    )
  admissible_ops = [
      op for op in operating_points
      if op.get("variance_passed", True)
      and op.get("slot_starvation_passed", True)
      and op.get("gate_a_passed", False)
  ]
  frontier = extract_pareto_frontier(admissible_ops, x_key=x_key_effective, y_key=y_key_effective, require_gate_a=True)
  frontier = [{k: v for k, v in pt.items() if k != "runs"} for pt in frontier]
  iso_slices = extract_iso_reactivity_slices(
      frontier_points=frontier,
      latency_key=lat_key_effective,
      throughput_key=y_key_effective,
      track=caps.track,
  ) if frontier else {}

  power_provs = {op["hardware_telemetry"]["power_provenance"] for op in operating_points} or {PROVENANCE_UNAVAILABLE}
  if power_provs == {PROVENANCE_MEASURED}:
    power_prov = PROVENANCE_MEASURED
  elif PROVENANCE_UNAVAILABLE in power_provs:
    power_prov = PROVENANCE_UNAVAILABLE
  else:
    power_prov = PROVENANCE_ESTIMATED

  if admissibility:
    status, reason = RunStatus.INVALID.value, "not admissible: " + " | ".join(admissibility)
  else:
    status, reason = RunStatus.PENDING.value, "self-measured real run; awaiting organizer re-run (ADR-5) before CERTIFIED"

  best_exec = max(operating_points, key=lambda p: p.get(y_key_effective) or 0.0) if operating_points else None
  best_raw = max(operating_points, key=lambda p: p.get(Y_METRIC_RAW) or 0.0) if operating_points else None
  best_lat = min((p[lat_key_effective] for p in operating_points if p.get(lat_key_effective)), default=None)

  all_gates_list = [effective_gates.get(pt.concurrency) for pt in sweep.points] if sweep.points else ([inputs.gate] if inputs.gate else [])
  quality_certified_all = bool(all_gates_list) and all(g is not None and bool(g.quality_certified) for g in all_gates_list)
  gate_passed_all = bool(all_gates_list) and all(g is not None and bool(g.passed) for g in all_gates_list)
  gate_modes = sorted({g.mode for g in all_gates_list if g is not None})
  gate_mode_val = gate_modes[0] if len(gate_modes) == 1 else ("MIXED" if gate_modes else None)
  jerk_vals = [op["normalized_action_jerk"] for op in operating_points if op.get("normalized_action_jerk") is not None]
  if not jerk_vals and inputs.gate is not None and inputs.gate.normalized_action_jerk is not None:
    jerk_vals = [inputs.gate.normalized_action_jerk]
  golden_shas = sorted({
      str((g.golden or {}).get("golden_sha256") or g.checks.get("golden_sha256"))
      for g in all_gates_list
      if g is not None and ((isinstance(g.golden, dict) and g.golden.get("golden_sha256")) or g.checks.get("golden_sha256"))
  })
  quality_gate_dict: Dict[str, Any] = {
      "passed": gate_passed_all,
      "quality_certified": quality_certified_all,
      "mode": gate_mode_val,
      "gate_mode": gate_mode_val,
      "modes": gate_modes,
      "golden_sha256": golden_shas[0] if len(golden_shas) == 1 else None,
      "normalized_action_jerk": _median(jerk_vals) if jerk_vals else None,
      "per_concurrency": {
          f"C{c}": g.to_dict() for c, g in sorted(effective_gates.items()) if g is not None
      },
  }

  summary_dict: Dict[str, Any] = {
      "num_operating_points": len(operating_points),
      "num_pareto_frontier_points": len(frontier),
      "peak_executed_actions_per_sec": (
          None if is_wm else (best_exec[y_key_effective] if best_exec else None)
      ),
      "peak_executed_actions_per_sec_concurrency": (
          None if is_wm else (best_exec["concurrency"] if best_exec else None)
      ),
      "peak_actions_per_sec": (
          None if is_wm else (best_raw[Y_METRIC_RAW] if best_raw else None)
      ),
      "peak_actions_per_sec_concurrency": (
          None if is_wm else (best_raw["concurrency"] if best_raw else None)
      ),
      "best_obs_to_action_p99_ms": None if is_wm else best_lat,
      "all_variance_passed": all(p.variance_passed for p in sweep.points),
      "iso_reactivity_slices": iso_slices,
  }
  if is_wm:
    best_frames = max(operating_points, key=lambda p: p.get(WM_Y_METRIC_FRAMES) or 0.0) if operating_points else None
    summary_dict["peak_predictions_per_sec"] = best_exec[y_key_effective] if best_exec else None
    summary_dict["peak_predictions_per_sec_concurrency"] = best_exec["concurrency"] if best_exec else None
    summary_dict["peak_generated_frames_per_sec"] = best_frames[WM_Y_METRIC_FRAMES] if best_frames else None
    summary_dict["peak_generated_frames_per_sec_concurrency"] = best_frames["concurrency"] if best_frames else None
    summary_dict["best_rollout_latency_p99_ms"] = best_lat

  report: Dict[str, Any] = {
      "schema_version": REPORT_SCHEMA_VERSION,
      "report_kind": REPORT_KIND,
      "generated_at": now_pacific_iso(),
      "harness": {"eeb_version": __version__, "embodied_bench_version": eb_version},
      "recipe_id": recipe.get("recipe_id") or f"adhoc-{caps.model_id or caps.server_impl}".lower(),
      "recipe_path": inputs.recipe_path,
      "recipe": dict(inputs.recipe) if inputs.recipe else None,
      "track": track_val,
      "model_family": recipe.get("model_family", ""),
      "model_name": recipe.get("model_name") or caps.model_id or caps.server_impl,
      "baseline_model_id": recipe.get("baseline_model_id") or caps.model_id or caps.server_impl,
      "precision": recipe.get("precision") or caps.precision,
      "hardware_sku": sku or "UNKNOWN_SKU",
      "compression_tier": recipe.get("compression_tier", "strict_lossless"),
      "clock": recipe.get("clock"),
      "scenario": recipe.get("scenario"),
      "division": recipe.get("division", "CLOSED"),
      "perf_cap": recipe.get("perf_cap"),
      "num_devices": (env.num_devices if env else None),
      "execution_strategy": sweep.pacing.mode if sweep.pacing.enabled else "NONE",
      "status": status,
      "reason": reason,
      "admissibility_findings": admissibility,
      "execution_provenance": {
          "loadgen": LOADGEN_PROVENANCE_REAL_WALLCLOCK,
          "policy": caps.server_impl,
          "policy_is_synthetic": caps.is_synthetic,
          "power": power_prov,
          "observations": sweep.observation_source.get("kind"),
          "gate": gate_modes,
      },
      "x_metric_name": x_key_effective,
      "x_axis_label": (
          "p99 Rollout Rate (Rollouts/s | 16x = Stream FPS)"
          if is_wm
          else "p99 Chunk Rate (Chunks/s) [1000 / L_p99, ms]"
      ),
      "y_metric_name": y_key_effective,
      "y_axis_label": (
          "Predictions/s/chip (16-Frame Rollouts/s)"
          if is_wm
          else "Executed Actions/s/chip"
      ),
      "latency_metric_name": lat_key_effective,
      "endpoint": sweep.endpoint,
      "transport": sweep.transport,
      "caps": caps.to_dict(),
      "sweep_config": sweep.config.to_dict(track=track_val),
      "pacing": sweep.pacing.to_dict(),
      "observation_source": {
          **dict(sweep.observation_source or {}),
          "wire_bytes_per_request_images": wire_bytes,
          "wire_resolution_admissible": wire_ok,
      },
      "host": sweep.host,
      "time_to_ready_s": sweep.time_to_ready_s,
      "wall_duration_s": round(sweep.finished_at_wall - sweep.started_at_wall, 3),
      "envelope_check": env.to_dict() if env else None,
      "quality_gate": quality_gate_dict,
      "quality_certified": quality_certified_all,
      "gate_mode": gate_mode_val,
      "telemetry": {
          "source": inputs.telemetry_source,
          "power_provenance": power_prov,
          "assumed_power_w": inputs.assumed_power_w,
          "per_run": {f"C{c}_r{r}": s.to_dict() for (c, r), s in sorted(inputs.telemetry_by_run.items())},
      },
      "summary": summary_dict,
      "summary_metrics": summary_dict,
      "iso_reactivity_slices": iso_slices,
      "operating_points": operating_points,
      "pareto_frontier": frontier,
      "stage2_gate_b": None,
      "notes": list(inputs.notes),
  }
  return report


_GENERATED_RECIPE_TAG = "eeb.loadgen.report.write_report_bundle"


def write_report_bundle(report: Dict[str, Any], out_dir: str | Path, basename: str = "report") -> Dict[str, str]:
  """Atomically write `<out>/<basename>.json`, `<out>/receipt.json`, and self-contained `<out>/recipe.json`."""
  import fcntl
  from embodied_bench.attestation import create_submission_receipt
  from embodied_bench.schema import _lock_path_for, atomic_write_json, cleanup_orphan_temp_files

  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  bundle_lock = _lock_path_for(out / ".bundle_meta")

  with open(bundle_lock, "a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    try:
      cleanup_orphan_temp_files(out)
      for item in out.iterdir():
        if item.is_file() and item.name.startswith(".") and item.name.endswith(".tmp"):
          try:
            item.unlink()
          except OSError:
            pass
      report_path = out / f"{basename}.json"
      receipt_path = out / "receipt.json"
      recipe_json_path = out / "recipe.json"
      recipe_yaml_path = out / "recipe.yaml"

      recipe_payload = report.get("recipe")
      out_paths: Dict[str, str] = {}
      if isinstance(recipe_payload, dict) and recipe_payload.get("recipe_id") and not recipe_payload.get("_adhoc_cli"):
        if not recipe_yaml_path.exists():
          can_write = True
          if recipe_json_path.exists():
            try:
              existing = json.loads(recipe_json_path.read_text(encoding="utf-8"))
              can_write = isinstance(existing, dict) and existing.get("_generated_by") == _GENERATED_RECIPE_TAG
            except (OSError, json.JSONDecodeError, ValueError):
              can_write = True
          if can_write:
            bundled = dict(recipe_payload)
            bundled["_generated_by"] = _GENERATED_RECIPE_TAG
            atomic_write_json(recipe_json_path, bundled)
          out_paths["recipe"] = str(recipe_json_path)
        else:
          out_paths["recipe"] = str(recipe_yaml_path)
      elif recipe_json_path.exists():
        # Gate 1 Repeat Path: clear stale auto-generated recipe.json if re-running ad-hoc without a recipe.
        try:
          existing = json.loads(recipe_json_path.read_text(encoding="utf-8"))
          if isinstance(existing, dict) and existing.get("_generated_by") == _GENERATED_RECIPE_TAG:
            recipe_json_path.unlink()
        except (OSError, json.JSONDecodeError, ValueError):
          pass

      atomic_write_json(report_path, report)
      receipt = create_submission_receipt(report, receipt_path)
      out_paths.update({
          "report": str(report_path),
          "receipt": str(receipt_path),
          "attestation_sha256": receipt["attestation_sha256"],
      })
      return out_paths
    finally:
      fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def validate_report(report: Dict[str, Any]) -> List[str]:
  """Structural validation used by tests / CI: returns a list of problems (empty == valid)."""
  from embodied_bench.schema import validate_status_and_reason

  problems: List[str] = []
  required = [
      "schema_version", "report_kind", "recipe_id", "track", "model_name", "baseline_model_id", "precision",
      "hardware_sku", "compression_tier", "status", "reason", "execution_provenance", "x_metric_name",
      "y_metric_name", "latency_metric_name", "operating_points", "pareto_frontier", "caps", "sweep_config",
  ]
  for key in required:
    if key not in report:
      problems.append(f"missing top-level key {key!r}")
  try:
    validate_status_and_reason(report.get("status", ""), report.get("reason"))
  except ValueError as exc:
    problems.append(str(exc))
  prov = report.get("execution_provenance", {})
  if prov.get("loadgen") != LOADGEN_PROVENANCE_REAL_WALLCLOCK:
    problems.append("execution_provenance.loadgen must be REAL_WALLCLOCK")
  if not prov.get("policy"):
    problems.append("execution_provenance.policy missing")
  if prov.get("power") not in (PROVENANCE_MEASURED, PROVENANCE_ESTIMATED, PROVENANCE_UNAVAILABLE):
    problems.append(f"execution_provenance.power invalid: {prov.get('power')!r}")
  is_wm = report.get("track") == "WM_TRACK"
  req_op_keys: Tuple[str, ...] = (
      ("point_id", "execution_id", WM_X_METRIC, WM_Y_METRIC, WM_LATENCY_METRIC, "gate_a_passed", "co_located_gate_a", "hardware_telemetry")
      if is_wm
      else ("point_id", "execution_id", X_METRIC, Y_METRIC, LATENCY_METRIC, "gate_a_passed", "co_located_gate_a", "hardware_telemetry")
  )
  for op in report.get("operating_points", []):
    for key in req_op_keys:
      if key not in op:
        problems.append(f"operating point {op.get('point_id')!r} missing {key!r}")
    gate = op.get("co_located_gate_a", {})
    if gate.get("execution_id") != op.get("execution_id"):
      problems.append(f"operating point {op.get('point_id')!r}: gate execution_id != point execution_id (G3)")
    hw = op.get("hardware_telemetry", {})
    if hw.get("power_provenance") == PROVENANCE_UNAVAILABLE and (
        hw.get("mean_power_watts") is not None
        or op.get("joules_per_action") is not None
        or op.get("joules_per_prediction") is not None
        or op.get("joules_per_frame") is not None
    ):
      problems.append(f"operating point {op.get('point_id')!r}: power UNAVAILABLE but watts/Joules present (fabrication)")
  if prov.get("policy_is_synthetic") and report.get("status") != "INVALID":
    problems.append("synthetic policy report must have status INVALID")
  return problems


def _md_safe(val: Any) -> str:
  s = "" if val is None else str(val)
  return (
      s.replace("\\", "\\\\")
      .replace("|", "\\|")
      .replace("\r", " ")
      .replace("\n", " ")
      .replace("<", "&lt;")
      .replace(">", "&gt;")
  )


def _fmt_opt(val: Any, fmt: str = ".2f", na: str = "N/A") -> str:
  if val is None or (isinstance(val, float) and val != val):
    return na
  try:
    return format(float(val), fmt)
  except (TypeError, ValueError):
    return _md_safe(val)


def report_to_markdown_summary(report: Dict[str, Any]) -> str:
  is_wm = report.get("track") == "WM_TRACK"
  ops = report.get("operating_points", [])
  c1_ttfa = next(
      (op.get("cold_session_ttfa_p50_ms", op.get("ttfa_p50_ms")) for op in ops if op.get("concurrency") == 1),
      ops[0].get("cold_session_ttfa_p50_ms", ops[0].get("ttfa_p50_ms")) if ops else None,
  )
  if is_wm:
    rows = [
        "| B | p50 ms | p99 ms | TTFF p99 ms | rollout/s | pred/s | gen FPS | pred/J | per $/hr | per 100T | stale ms | spread % | var | gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for op in ops:
      spread = (op.get("cross_run_spread_pct") or {}).get("latency_p50_ms", 0.0)
      gate_label = (
          ("CERT" if op.get("quality_certified") else "SANITY")
          if op.get("gate_a_passed")
          else "FAIL"
      )
      rows.append(
          f"| {op['concurrency']} "
          f"| {_fmt_opt(op.get('rollout_latency_p50_ms', op.get('obs_to_action_p50_ms')))} "
          f"| {_fmt_opt(op.get('rollout_latency_p99_ms', op.get('obs_to_action_p99_ms')))} "
          f"| {_fmt_opt(op.get('ttff_p99_ms'))} "
          f"| {_fmt_opt(op.get('hz_plan', op.get('hz_react')))} "
          f"| {_fmt_opt(op.get('predictions_per_sec', op.get('chunks_per_sec')))} "
          f"| {_fmt_opt(op.get('generated_frames_per_sec', op.get('frames_per_sec')), '.1f')} "
          f"| {_fmt_opt(op.get('predictions_per_watt', op.get('power_efficiency_per_watt')), '.3f')} "
          f"| {_fmt_opt(op.get('tco_efficiency_per_dollar', op.get('throughput_per_dollar_hr')), '.2f')} "
          f"| {_fmt_opt(op.get('throughput_per_100_tops'), '.2f')} "
          f"| {_fmt_opt(op.get('observation_staleness_ms'), '.1f')} "
          f"| {_fmt_opt(spread)} "
          f"| {'ok' if op.get('variance_passed') else 'FAIL'} "
          f"| {gate_label} |"
      )
  else:
    rows = [
        "| C | p50 ms | p99 ms | chunk/s | exec act/s | act/J | per $/hr | per 100T | stale ms | jerk | spread % | var | gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for op in ops:
      spread = (op.get("cross_run_spread_pct") or {}).get("latency_p50_ms", 0.0)
      gate_label = (
          ("CERT" if op.get("quality_certified") else "SANITY")
          if op.get("gate_a_passed")
          else "FAIL"
      )
      rows.append(
          f"| {op['concurrency']} "
          f"| {_fmt_opt(op.get('obs_to_action_p50_ms'))} "
          f"| {_fmt_opt(op.get('obs_to_action_p99_ms'))} "
          f"| {_fmt_opt(op.get('hz_react'))} "
          f"| {_fmt_opt(op.get('executed_actions_per_sec', op.get('actions_per_sec')), '.1f')} "
          f"| {_fmt_opt(op.get('power_efficiency_per_watt'), '.3f')} "
          f"| {_fmt_opt(op.get('tco_efficiency_per_dollar', op.get('throughput_per_dollar_hr')), '.2f')} "
          f"| {_fmt_opt(op.get('throughput_per_100_tops'), '.2f')} "
          f"| {_fmt_opt(op.get('observation_staleness_ms'), '.1f')} "
          f"| {_fmt_opt(op.get('normalized_action_jerk'), '.4f')} "
          f"| {_fmt_opt(spread)} "
          f"| {'ok' if op.get('variance_passed') else 'FAIL'} "
          f"| {gate_label} |"
      )
  c1_label = "B=1" if is_wm else "C=1"
  head = (
      f"status={_md_safe(report['status'])} track={_md_safe(report.get('track'))} "
      f"policy={_md_safe(report['execution_provenance']['policy'])} "
      f"loadgen={_md_safe(report['execution_provenance']['loadgen'])} "
      f"power={_md_safe(report['execution_provenance']['power'])} "
      f"transport={_md_safe(report.get('transport'))} "
      f"cold_session_ttfa_p50_ms({c1_label})={_fmt_opt(c1_ttfa)}\n"
  )
  return head + "\n".join(rows) + f"\nreason: {_md_safe(report['reason'])}\n"


__all__ = [
    "CANONICAL_LOCKED_EXECUTE_HORIZONS",
    "LATENCY_METRIC",
    "REPORT_KIND",
    "REPORT_SCHEMA_VERSION",
    "ReportInputs",
    "WM_LATENCY_METRIC",
    "WM_X_METRIC",
    "WM_Y_METRIC",
    "WM_Y_METRIC_FRAMES",
    "X_METRIC",
    "Y_METRIC",
    "Y_METRIC_RAW",
    "build_report",
    "canonical_locked_execute_horizon",
    "extract_iso_reactivity_slices",
    "report_to_markdown_summary",
    "validate_report",
    "write_report_bundle",
]
