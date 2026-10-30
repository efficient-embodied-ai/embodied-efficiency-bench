"""Mechanical baseline-vs-candidate evaluator for `eeb compare`.

Provides a deterministic fitness function for both human contestants and
external GPU/TPU hill-climbing harnesses:

  * Enforces Closed Division apples-to-apples invariants (matching track,
    canonical model identity, locked execution horizon `K_exec` / `H_frames`,
    single-chip `num_devices`, non-synthetic policy provenance, co-located
    `GOLDEN_REPLAY` quality gate, and cross-run variance stability).
  * Renders a side-by-side per-concurrency Markdown table and summary KPIs
    (`best_output_at_cutoff`, `peak_pareto_output`, `c1_reactivity_hz`).
  * Returns an unmistakable verdict (`ACCEPT` -> exit code `0`, `REJECT` ->
    exit code `2`) with an explicit `rejection_code` and human-readable
    `reason`.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

from eeb.data.schema import is_synthetic_trace_manifest
from embodied_bench.baselines import resolve_canonical_model_id
from embodied_bench.schema import (
    MODERN_REFERENCE_CHECKPOINTS,
    TrackType,
    is_wm_track,
    now_pacific_iso,
)

COMPARE_SCHEMA_VERSION = "1.0.0"

DEFAULT_CUTOFF_HZ_BY_TRACK: Dict[str, float] = {
    TrackType.VLA_TRACK.value: 4.0,
    TrackType.WM_TRACK.value: 1.0,
}

DEFAULT_CUTOFF_HZ_BY_MODEL: Dict[str, float] = {
    "pi05-3b": 4.0,
    "gr00t-n1.7-3b": 2.5,
    "cosmos-predict2.5-2b": 1.0,
}

LOCKED_CLOSED_EXEC_HORIZON_BY_MODEL: Dict[str, int] = {
    "pi05-3b": 5,
    "gr00t-n1.7-3b": 8,
    "cosmos-predict2.5-2b": 16,
}


def resolve_report_path(path_or_dir: Union[str, Path]) -> Path:
  """Resolve a report file or bundle directory (`report.json` / `sweep_results.json`)."""
  p = Path(path_or_dir)
  if p.is_dir():
    for candidate in ("report.json", "sweep_results.json", "receipt.json"):
      cand_path = p / candidate
      if cand_path.is_file():
        return cand_path
    raise FileNotFoundError(
        f"No report.json, sweep_results.json, or receipt.json found inside directory '{p}'."
    )
  if not p.is_file():
    raise FileNotFoundError(f"Report file does not exist: '{p}'.")
  return p


def load_report_doc(path_or_dir: Union[str, Path]) -> Tuple[Path, Dict[str, Any]]:
  """Load and sanity-check a measured report JSON dictionary."""
  resolved = resolve_report_path(path_or_dir)
  try:
    doc = json.loads(resolved.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise ValueError(f"Failed to parse JSON report '{resolved}': {exc}") from exc
  if not isinstance(doc, dict):
    raise ValueError(f"Report '{resolved}' must contain a top-level JSON object.")
  if not isinstance(doc.get("operating_points"), list):
    raise ValueError(f"Report '{resolved}' is missing an 'operating_points' list.")
  return resolved, doc


def _normalize_track(doc: Dict[str, Any]) -> str:
  raw = str(doc.get("track") or "").strip()
  if not raw:
    caps = doc.get("caps") if isinstance(doc.get("caps"), dict) else {}
    raw = str(caps.get("track") or "").strip()
  if is_wm_track(raw):
    return TrackType.WM_TRACK.value
  if raw in ("VLA_TRACK", "TRACK_1_SCENARIO_A", "TRACK_2_VLA_REALTIME"):
    return TrackType.VLA_TRACK.value
  return raw


def _resolve_model_identity(doc: Dict[str, Any]) -> str:
  candidates: List[str] = []
  for key in ("baseline_model_id", "model_family", "model_name"):
    val = doc.get(key)
    if isinstance(val, str) and val.strip():
      candidates.append(val.strip())
  recipe = doc.get("recipe") if isinstance(doc.get("recipe"), dict) else {}
  for key in ("baseline_model_id", "model_family", "model_name"):
    val = recipe.get(key)
    if isinstance(val, str) and val.strip():
      candidates.append(val.strip())
  caps = doc.get("caps") if isinstance(doc.get("caps"), dict) else {}
  cap_model = caps.get("model_id")
  if isinstance(cap_model, str) and cap_model.strip():
    candidates.append(cap_model.strip())

  for raw in candidates:
    try:
      return resolve_canonical_model_id(raw)
    except ValueError:
      continue
  return candidates[0].lower() if candidates else ""


def _resolve_execute_horizon(doc: Dict[str, Any], is_wm: bool) -> Optional[int]:
  ops = doc.get("operating_points") or []
  for op in ops:
    if not isinstance(op, dict):
      continue
    keys = ("horizon_frames", "execute_horizon", "horizon", "chunk_size") if is_wm else ("execute_horizon", "chunk_size", "horizon")
    for k in keys:
      v = op.get(k)
      if isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) > 0:
        return int(v)
  sweep_cfg = doc.get("sweep_config") if isinstance(doc.get("sweep_config"), dict) else {}
  if isinstance(sweep_cfg.get("execute_horizon"), int) and sweep_cfg["execute_horizon"] > 0:
    return int(sweep_cfg["execute_horizon"])
  recipe = doc.get("recipe") if isinstance(doc.get("recipe"), dict) else {}
  r_keys = ("horizon_frames", "execute_horizon") if is_wm else ("execute_horizon", "action_horizon")
  for k in r_keys:
    v = recipe.get(k)
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
      return int(v)
  caps = doc.get("caps") if isinstance(doc.get("caps"), dict) else {}
  if isinstance(caps.get("horizon"), int) and caps["horizon"] > 0:
    return int(caps["horizon"])
  return None


def _is_synthetic_report(doc: Dict[str, Any]) -> bool:
  prov = doc.get("execution_provenance") if isinstance(doc.get("execution_provenance"), dict) else {}
  if prov.get("policy_is_synthetic") is True:
    return True
  if str(prov.get("policy") or "").upper() == "SYNTHETIC_DUMMY":
    return True
  caps = doc.get("caps") if isinstance(doc.get("caps"), dict) else {}
  if caps.get("is_synthetic") is True:
    return True
  if str(caps.get("server_impl") or "").upper() == "SYNTHETIC_DUMMY":
    return True
  return False


def _point_concurrency(op: Dict[str, Any]) -> int:
  val = op.get("concurrency", op.get("batch_size", 0))
  return int(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0


def _point_hz(op: Dict[str, Any], is_wm: bool) -> Optional[float]:
  keys = ("hz_plan", "hz_react") if is_wm else ("hz_react",)
  for k in keys:
    v = op.get(k)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      return float(v)
  return None


def _point_output(op: Dict[str, Any], is_wm: bool) -> Optional[float]:
  if is_wm:
    for k in ("predictions_per_sec", "chunks_per_sec"):
      v = op.get(k)
      if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    h_raw = op.get("horizon_frames", op.get("horizon", 16))
    h_f = max(float(h_raw) if isinstance(h_raw, (int, float)) and not isinstance(h_raw, bool) else 16.0, 1.0)
    for k in ("generated_frames_per_sec", "fps", "frames_per_sec", "actions_per_sec"):
      v = op.get(k)
      if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) / h_f
    v = op.get("raw_per_chip_throughput")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      return float(v)
    return None
  for k in ("executed_actions_per_sec", "actions_per_sec", "raw_per_chip_throughput"):
    v = op.get(k)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      return float(v)
  return None


def _point_power_eff(op: Dict[str, Any], is_wm: bool) -> Optional[float]:
  hw = op.get("hardware_telemetry") if isinstance(op.get("hardware_telemetry"), dict) else {}
  if hw.get("power_provenance") == "UNAVAILABLE":
    return None
  if is_wm:
    v_pred = op.get("predictions_per_watt")
    if isinstance(v_pred, (int, float)) and not isinstance(v_pred, bool):
      return float(v_pred)
    if "predictions_per_sec" in op:
      v_pw = op.get("power_efficiency_per_watt")
      if isinstance(v_pw, (int, float)) and not isinstance(v_pw, bool):
        return float(v_pw)
    h_raw = op.get("horizon_frames", op.get("horizon", 16))
    h_f = max(float(h_raw) if isinstance(h_raw, (int, float)) and not isinstance(h_raw, bool) else 16.0, 1.0)
    for k in ("frames_per_watt", "fps_per_watt", "power_efficiency_per_watt"):
      v = op.get(k)
      if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) / h_f
    sa = op.get("semianalysis_normalizations") if isinstance(op.get("semianalysis_normalizations"), dict) else {}
    v = sa.get("power_efficiency_per_watt")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      return float(v) if "predictions_per_sec" in op else float(v) / h_f
    return None
  for k in ("power_efficiency_per_watt", "actions_per_watt"):
    v = op.get(k)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      return float(v)
  sa = op.get("semianalysis_normalizations") if isinstance(op.get("semianalysis_normalizations"), dict) else {}
  v = sa.get("power_efficiency_per_watt")
  if isinstance(v, (int, float)) and not isinstance(v, bool):
    return float(v)
  return None


def _point_gate_status(op: Dict[str, Any], allow_sanity_only: bool) -> Tuple[bool, str]:
  """Return (admissible_point, short_label) for a single operating point."""
  var_ok = bool(op.get("variance_passed", True))
  gate_ok = bool(op.get("gate_a_passed", False))
  mode = str(op.get("gate_a_mode") or op.get("gate_mode") or "").upper()
  has_explicit_qc = "quality_certified" in op
  qc = bool(op.get("quality_certified", True)) if has_explicit_qc else (mode != "SANITY_ONLY")

  if not gate_ok:
    return False, "GATE_FAIL"
  if not var_ok:
    return False, "VAR_FAIL"
  if not qc and not allow_sanity_only:
    return False, "SANITY"
  return True, ("SANITY_OK" if (not qc and allow_sanity_only) else "CERT")


def _summarize_side(
    ops: List[Dict[str, Any]],
    *,
    is_wm: bool,
    cutoff_hz: float,
    allow_sanity_only: bool,
) -> Dict[str, Any]:
  admissible: List[Dict[str, Any]] = []
  for op in ops:
    if not isinstance(op, dict):
      continue
    ok, _label = _point_gate_status(op, allow_sanity_only=allow_sanity_only)
    hz = _point_hz(op, is_wm)
    out = _point_output(op, is_wm)
    if ok and hz is not None and out is not None:
      admissible.append({
          "concurrency": _point_concurrency(op),
          "hz": hz,
          "output": out,
          "power_eff": _point_power_eff(op, is_wm),
          "ttff_p99_ms": op.get("ttff_p99_ms"),
      })
  admissible.sort(key=lambda item: item["concurrency"])
  c1_pt = next((p for p in admissible if p["concurrency"] == 1), (admissible[0] if admissible else None))
  peak_pt = max(admissible, key=lambda p: p["output"]) if admissible else None
  qualifying = [p for p in admissible if p["hz"] >= cutoff_hz]
  best_cutoff_pt = max(qualifying, key=lambda p: p["output"]) if qualifying else None
  max_hz_pt = max(admissible, key=lambda p: p["hz"]) if admissible else None
  return {
      "num_total_points": len(ops),
      "num_admissible_points": len(admissible),
      "num_qualifying_at_cutoff": len(qualifying),
      "c1_reactivity_hz": round(c1_pt["hz"], 4) if c1_pt else None,
      "c1_concurrency": c1_pt["concurrency"] if c1_pt else None,
      "max_admissible_hz": round(max_hz_pt["hz"], 4) if max_hz_pt else None,
      "peak_pareto_output": round(peak_pt["output"], 4) if peak_pt else 0.0,
      "peak_pareto_concurrency": peak_pt["concurrency"] if peak_pt else None,
      "best_output_at_cutoff": round(best_cutoff_pt["output"], 4) if best_cutoff_pt else 0.0,
      "best_concurrency_at_cutoff": best_cutoff_pt["concurrency"] if best_cutoff_pt else None,
      "best_hz_at_cutoff": round(best_cutoff_pt["hz"], 4) if best_cutoff_pt else None,
      "best_power_eff_at_cutoff": (
          round(best_cutoff_pt["power_eff"], 4)
          if (best_cutoff_pt and best_cutoff_pt["power_eff"] is not None)
          else None
      ),
  }


def _pct_delta(base_val: Optional[float], cand_val: Optional[float]) -> Optional[float]:
  if base_val is None or cand_val is None or base_val <= 0.0:
    return None
  return round(((cand_val - base_val) / base_val) * 100.0, 3)


def compare_reports(
    baseline_path: Union[str, Path],
    candidate_path: Union[str, Path],
    *,
    cutoff_hz: Optional[float] = None,
    min_gain_pct: float = 2.0,
    allow_sanity_only: bool = False,
) -> Dict[str, Any]:
  """Compare a candidate report against a baseline report under Closed Division rules."""
  if min_gain_pct < 0.0:
    raise ValueError(f"--min-gain-pct must be >= 0.0, got {min_gain_pct!r}.")
  if cutoff_hz is not None and cutoff_hz <= 0.0:
    raise ValueError(f"--cutoff-hz must be > 0.0, got {cutoff_hz!r}.")

  b_file, b_doc = load_report_doc(baseline_path)
  c_file, c_doc = load_report_doc(candidate_path)

  b_track = _normalize_track(b_doc)
  c_track = _normalize_track(c_doc)
  is_wm = b_track == TrackType.WM_TRACK.value

  b_model = _resolve_model_identity(b_doc)
  c_model = _resolve_model_identity(c_doc)
  default_cutoff = DEFAULT_CUTOFF_HZ_BY_MODEL.get(
      c_model or b_model,
      DEFAULT_CUTOFF_HZ_BY_TRACK.get(b_track, 4.0),
  )
  eff_cutoff = float(cutoff_hz) if cutoff_hz is not None else float(default_cutoff)
  b_div = str(b_doc.get("division") or "CLOSED").upper()
  c_div = str(c_doc.get("division") or "CLOSED").upper()
  b_horizon = _resolve_execute_horizon(b_doc, is_wm=is_wm)
  c_horizon = _resolve_execute_horizon(c_doc, is_wm=(c_track == TrackType.WM_TRACK.value))
  b_syn = _is_synthetic_report(b_doc)
  c_syn = _is_synthetic_report(c_doc)

  b_ops = [op for op in b_doc.get("operating_points", []) if isinstance(op, dict)]
  c_ops = [op for op in c_doc.get("operating_points", []) if isinstance(op, dict)]
  b_by_c = {_point_concurrency(op): op for op in b_ops if _point_concurrency(op) > 0}
  c_by_c = {_point_concurrency(op): op for op in c_ops if _point_concurrency(op) > 0}
  all_c = sorted(set(b_by_c) | set(c_by_c))

  rows: List[Dict[str, Any]] = []
  for c in all_c:
    b_op = b_by_c.get(c)
    c_op = c_by_c.get(c)
    b_hz = _point_hz(b_op, is_wm) if b_op else None
    c_hz = _point_hz(c_op, is_wm) if c_op else None
    b_out = _point_output(b_op, is_wm) if b_op else None
    c_out = _point_output(c_op, is_wm) if c_op else None
    b_pe = _point_power_eff(b_op, is_wm) if b_op else None
    c_pe = _point_power_eff(c_op, is_wm) if c_op else None
    b_ok, b_lbl = _point_gate_status(b_op, allow_sanity_only=True) if b_op else (False, "MISSING")
    c_ok, c_lbl = _point_gate_status(c_op, allow_sanity_only=allow_sanity_only) if c_op else (False, "MISSING")
    rows.append({
        "concurrency": c,
        "baseline_hz": round(b_hz, 4) if b_hz is not None else None,
        "candidate_hz": round(c_hz, 4) if c_hz is not None else None,
        "hz_delta_pct": _pct_delta(b_hz, c_hz),
        "baseline_output": round(b_out, 4) if b_out is not None else None,
        "candidate_output": round(c_out, 4) if c_out is not None else None,
        "output_delta_pct": _pct_delta(b_out, c_out),
        "baseline_power_eff": round(b_pe, 4) if b_pe is not None else None,
        "candidate_power_eff": round(c_pe, 4) if c_pe is not None else None,
        "power_eff_delta_pct": _pct_delta(b_pe, c_pe),
        "baseline_admissible": b_ok,
        "baseline_gate": b_lbl,
        "candidate_admissible": c_ok,
        "candidate_gate": c_lbl,
    })

  b_sum = _summarize_side(b_ops, is_wm=is_wm, cutoff_hz=eff_cutoff, allow_sanity_only=True)
  c_sum = _summarize_side(c_ops, is_wm=(c_track == TrackType.WM_TRACK.value), cutoff_hz=eff_cutoff, allow_sanity_only=allow_sanity_only)

  verdict = "REJECT"
  decision_code = "REJECT_UNKNOWN"
  reason: str = ""

  # 1. Track check
  if not b_track or not c_track or b_track != c_track:
    decision_code = "REJECT_TRACK_MISMATCH"
    reason = f"Baseline track='{b_track}' does not match candidate track='{c_track}'."
  # 2. Model family identity check
  elif b_model and c_model and b_model != c_model:
    decision_code = "REJECT_MODEL_MISMATCH"
    reason = f"Baseline model='{b_model}' does not match candidate model='{c_model}'."
  # 3. Division check (cannot claim CLOSED champion win with an OPEN division recipe)
  elif b_div == "CLOSED" and c_div != "CLOSED":
    decision_code = "REJECT_DIVISION_MISMATCH"
    reason = f"Baseline division='CLOSED' cannot be superseded by candidate division='{c_div}'."
  # 4. Locked execution horizon check
  elif b_horizon is not None and c_horizon is not None and b_horizon != c_horizon:
    decision_code = "REJECT_HORIZON_MISMATCH"
    reason = (
        f"Candidate execute_horizon={c_horizon} does not match baseline execute_horizon={b_horizon}. "
        "Keep the execution horizon locked across A/B runs."
    )
  elif (
      c_div == "CLOSED"
      and c_model in LOCKED_CLOSED_EXEC_HORIZON_BY_MODEL
      and c_horizon is not None
      and c_horizon != LOCKED_CLOSED_EXEC_HORIZON_BY_MODEL[c_model]
  ):
    expected_h = LOCKED_CLOSED_EXEC_HORIZON_BY_MODEL[c_model]
    decision_code = "REJECT_HORIZON_MISMATCH"
    reason = (
        f"Candidate execute_horizon={c_horizon} violates Closed Division locked horizon "
        f"{expected_h} for canonical model '{c_model}'."
    )
  # 5. Synthetic candidate policy / trace checks
  elif c_syn and not b_syn:
    decision_code = "REJECT_SYNTHETIC_CANDIDATE"
    reason = "Candidate uses SYNTHETIC_DUMMY policy against a non-synthetic baseline report."
  elif (
      not allow_sanity_only
      and is_synthetic_trace_manifest(c_doc.get("observation_source"))
      and not is_synthetic_trace_manifest(b_doc.get("observation_source"))
  ):
    decision_code = "REJECT_SYNTHETIC_TRACES"
    reason = "Candidate uses non-certifiable SYNTHETIC_CI_SMOKE_TRACES against a real-trace baseline report (RT-02)."
  # 6. Single-chip device count check
  elif isinstance(c_doc.get("num_devices"), int) and not isinstance(c_doc.get("num_devices"), bool) and int(c_doc["num_devices"]) > 1:
    decision_code = "REJECT_MULTI_DEVICE"
    reason = f"Candidate recorded num_devices={c_doc['num_devices']} > 1 (CAP_SINGLE_CHIP requires num_devices == 1)."
  elif (
      isinstance(b_doc.get("num_devices"), int)
      and int(b_doc["num_devices"]) == 1
      and isinstance(c_doc.get("num_devices"), int)
      and int(c_doc["num_devices"]) == 0
  ):
    decision_code = "REJECT_NO_DEVICE_CANDIDATE"
    reason = "Candidate ran with num_devices=0 (--allow-no-device CPU host) against a single-accelerator baseline (num_devices=1)."
  else:
    # 7. Closed Division stream_chunk_frames check, quality gate, & variance checks.
    # Note: In v1.0 Closed Division (Regime 3 bidirectional full-clip 3D-DiT), TTFF == L_clip is a
    # co-reported D7 diagnostic metric; real-time admissibility is gated on hz_plan >= 1.5 rollouts/s (24.0 FPS).
    c_recipe = c_doc.get("recipe") if isinstance(c_doc.get("recipe"), dict) else {}
    sc_val = c_recipe.get("stream_chunk_frames", c_doc.get("stream_chunk_frames"))
    qg = c_doc.get("quality_gate") if isinstance(c_doc.get("quality_gate"), dict) else None
    any_gate_passed = any(bool(op.get("gate_a_passed", False)) for op in c_ops)
    any_qc = any(
        bool(op.get("quality_certified", str(op.get("gate_a_mode") or op.get("gate_mode") or "").upper() != "SANITY_ONLY"))
        for op in c_ops
        if op.get("gate_a_passed", False)
    )
    if (
        is_wm
        and c_div == "CLOSED"
        and isinstance(sc_val, (int, float))
        and not isinstance(sc_val, bool)
        and int(sc_val) > 0
    ):
      decision_code = "REJECT_CLOSED_STREAM_CHUNKING"
      reason = (
          f"Candidate WM_TRACK Closed Division sets stream_chunk_frames={int(sc_val)} > 0; "
          "pseudo-causal sub-clip chunking is forbidden on frozen bidirectional 3D-DiTs in Closed Division."
      )
    elif (qg is not None and qg.get("passed") is False) or not any_gate_passed:
      decision_code = "REJECT_QUALITY_GATE"
      reason = "Candidate failed the co-located quality gate (quality_gate.passed=False or all operating points failed Gate A/G4)."
    elif not allow_sanity_only and ((qg is not None and qg.get("quality_certified") is False) or not any_qc):
      mode_str = (qg or {}).get("mode") or "SANITY_ONLY"
      decision_code = "REJECT_QUALITY_GATE"
      reason = (
          f"Candidate quality_gate.quality_certified=False (mode='{mode_str}'). "
          "Re-run `eeb measure --golden <golden_ref.npz>` or pass `--allow-sanity-only` for smoke checks."
      )
    elif c_sum["num_admissible_points"] == 0:
      decision_code = "REJECT_VARIANCE_UNSTABLE"
      reason = "Candidate has zero variance-stable, gate-passing operating points (all gate-passing points had variance_passed=False)."
    else:
      b_cut = b_sum["best_output_at_cutoff"]
      c_cut = c_sum["best_output_at_cutoff"]
      if b_cut > 0.0 and c_cut <= 0.0:
        decision_code = "REJECT_REACTIVITY_REGRESSION"
        reason = (
            f"Candidate has no admissible operating point meeting cutoff >= {eff_cutoff:.2f} Hz "
            f"(max candidate admissible Hz={c_sum['max_admissible_hz']}), whereas baseline achieved "
            f"{b_cut:.2f} at C={b_sum['best_concurrency_at_cutoff']} ({b_sum['best_hz_at_cutoff']} Hz)."
        )
      elif b_cut <= 0.0 and c_cut <= 0.0:
        decision_code = "REJECT_REACTIVITY_CUTOFF_UNMET"
        reason = (
            f"Neither baseline nor candidate has an admissible point meeting cutoff >= {eff_cutoff:.2f} Hz "
            f"(candidate max admissible Hz={c_sum['max_admissible_hz']})."
        )
      elif b_cut <= 0.0 and c_cut > 0.0:
        verdict = "ACCEPT"
        decision_code = "ACCEPT_FIRST_REALTIME_POINT"
        reason = (
            f"Candidate achieves the first admissible real-time operating point at cutoff >= {eff_cutoff:.2f} Hz: "
            f"{c_cut:.2f} at C={c_sum['best_concurrency_at_cutoff']} ({c_sum['best_hz_at_cutoff']} Hz)."
        )
      else:
        gain_pct = ((c_cut - b_cut) / b_cut) * 100.0
        if gain_pct >= min_gain_pct:
          verdict = "ACCEPT"
          decision_code = "ACCEPT_IMPROVED_AT_CUTOFF"
          reason = (
              f"Candidate improves best qualifying output at >= {eff_cutoff:.2f} Hz by "
              f"+{gain_pct:.2f}% ({b_cut:.2f} -> {c_cut:.2f}, required >= +{min_gain_pct:.2f}%)."
          )
        elif gain_pct < 0.0:
          decision_code = "REJECT_THROUGHPUT_REGRESSION"
          reason = (
              f"Candidate regresses best qualifying output at >= {eff_cutoff:.2f} Hz by "
              f"{gain_pct:.2f}% ({b_cut:.2f} -> {c_cut:.2f})."
          )
        else:
          decision_code = "REJECT_BELOW_NOISE_FLOOR"
          reason = (
              f"Candidate gain at >= {eff_cutoff:.2f} Hz is +{gain_pct:.2f}% "
              f"({b_cut:.2f} -> {c_cut:.2f}), below minimum required +{min_gain_pct:.2f}% noise floor."
          )

  b_cut_val = b_sum["best_output_at_cutoff"]
  c_cut_val = c_sum["best_output_at_cutoff"]
  gain_at_cutoff_pct = _pct_delta(b_cut_val if b_cut_val > 0.0 else None, c_cut_val if c_cut_val > 0.0 else None)
  peak_gain_pct = _pct_delta(
      b_sum["peak_pareto_output"] if b_sum["peak_pareto_output"] > 0.0 else None,
      c_sum["peak_pareto_output"] if c_sum["peak_pareto_output"] > 0.0 else None,
  )
  c1_hz_delta_pct = _pct_delta(b_sum["c1_reactivity_hz"], c_sum["c1_reactivity_hz"])
  model_info = MODERN_REFERENCE_CHECKPOINTS.get(c_model or b_model, {})

  return {
      "schema_version": COMPARE_SCHEMA_VERSION,
      "command": "compare",
      "generated_at": now_pacific_iso(),
      "verdict": verdict,
      "accepted": verdict == "ACCEPT",
      "exit_code": 0 if verdict == "ACCEPT" else 2,
      "decision_code": decision_code,
      "reason": reason,
      "track": b_track,
      "model_id": c_model or b_model,
      "model_display_name": model_info.get("display_name", c_model or b_model),
      "division": c_div,
      "hardware_sku": str(c_doc.get("hardware_sku") or b_doc.get("hardware_sku") or "UNKNOWN_SKU"),
      "baseline_report": str(b_file),
      "candidate_report": str(c_file),
      "baseline_recipe_id": b_doc.get("recipe_id"),
      "candidate_recipe_id": c_doc.get("recipe_id"),
      "baseline_execute_horizon": b_horizon,
      "candidate_execute_horizon": c_horizon,
      "cutoff_hz": eff_cutoff,
      "min_gain_pct": float(min_gain_pct),
      "allow_sanity_only": bool(allow_sanity_only),
      "metrics": {
          "best_output_at_cutoff": {
              "baseline": b_cut_val,
              "candidate": c_cut_val,
              "delta_pct": gain_at_cutoff_pct,
              "baseline_concurrency": b_sum["best_concurrency_at_cutoff"],
              "candidate_concurrency": c_sum["best_concurrency_at_cutoff"],
          },
          "peak_pareto_output": {
              "baseline": b_sum["peak_pareto_output"],
              "candidate": c_sum["peak_pareto_output"],
              "delta_pct": peak_gain_pct,
              "baseline_concurrency": b_sum["peak_pareto_concurrency"],
              "candidate_concurrency": c_sum["peak_pareto_concurrency"],
          },
          "c1_reactivity_hz": {
              "baseline": b_sum["c1_reactivity_hz"],
              "candidate": c_sum["c1_reactivity_hz"],
              "delta_pct": c1_hz_delta_pct,
          },
          "best_power_eff_at_cutoff": {
              "baseline": b_sum["best_power_eff_at_cutoff"],
              "candidate": c_sum["best_power_eff_at_cutoff"],
              "delta_pct": _pct_delta(b_sum["best_power_eff_at_cutoff"], c_sum["best_power_eff_at_cutoff"]),
          },
      },
      "baseline_summary": b_sum,
      "candidate_summary": c_sum,
      "per_concurrency": rows,
  }


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


def format_comparison_markdown(result: Dict[str, Any]) -> str:
  """Render a deterministic Markdown comparison table without em-dashes."""
  is_wm = result.get("track") == TrackType.WM_TRACK.value
  c_hdr = "B" if is_wm else "C"
  hz_hdr = "Base rollout/s -> Cand" if is_wm else "Base chunk/s -> Cand"
  out_hdr = "Base Pred/s -> Cand Pred/s" if is_wm else "Base act/s -> Cand act/s"
  eff_hdr = "pred/J Delta %" if is_wm else "act/J Delta %"

  def _pair(a: Optional[float], b: Optional[float], nd: int = 2) -> str:
    sa = "N/A" if a is None else f"{a:.{nd}f}"
    sb = "N/A" if b is None else f"{b:.{nd}f}"
    return f"{sa} -> {sb}"

  def _delta(d: Optional[float]) -> str:
    if d is None:
      return "N/A"
    sign = "+" if d >= 0.0 else ""
    return f"{sign}{d:.2f}%"

  lines = [
      (
          f"verdict={_md_safe(result['verdict'])} code={_md_safe(result['decision_code'])} "
          f"track={_md_safe(result['track'])} model={_md_safe(result['model_id'])} "
          f"sku={_md_safe(result['hardware_sku'])} horizon={_md_safe(result['candidate_execute_horizon'])} "
          f"cutoff_hz={result['cutoff_hz']:.2f} min_gain_pct={result['min_gain_pct']:.2f}"
      ),
      f"| {c_hdr} | {hz_hdr} | {out_hdr} | Out Delta % | {eff_hdr} | Gate |",
      "|---|---|---|---|---|---|",
  ]
  for r in result.get("per_concurrency", []):
    lines.append(
        "| {c} | {hz} | {out} | {od} | {pd} | {g} |".format(
            c=r["concurrency"],
            hz=_pair(r["baseline_hz"], r["candidate_hz"], 2),
            out=_pair(r["baseline_output"], r["candidate_output"], 2 if is_wm else 1),
            od=_delta(r["output_delta_pct"]),
            pd=_delta(r["power_eff_delta_pct"]),
            g=_md_safe(r["candidate_gate"]),
        )
    )
  m = result["metrics"]
  cut_m = m["best_output_at_cutoff"]
  peak_m = m["peak_pareto_output"]
  c1_m = m["c1_reactivity_hz"]
  lines.append(
      "summary: "
      f"best_output_at_cutoff={_pair(cut_m['baseline'], cut_m['candidate'], 2)} ({_delta(cut_m['delta_pct'])}) | "
      f"peak_pareto_output={_pair(peak_m['baseline'], peak_m['candidate'], 2)} ({_delta(peak_m['delta_pct'])}) | "
      f"c1_reactivity_hz={_pair(c1_m['baseline'], c1_m['candidate'], 2)} ({_delta(c1_m['delta_pct'])})"
  )
  lines.append(f"reason: {_md_safe(result['reason'])}")
  return "\n".join(lines)


def write_comparison_json(result: Dict[str, Any], out_path: Union[str, Path]) -> Path:
  """Crash-safe, lock-protected atomic write for `eeb compare --out <file>`."""
  target = Path(out_path)
  target.parent.mkdir(parents=True, exist_ok=True)
  lock_digest = hashlib.sha256(str(target.resolve()).encode("utf-8")).hexdigest()[:24]
  lock_path = Path(tempfile.gettempdir()) / f"eeb_compare_{lock_digest}.lock"
  payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")

  lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
  try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    for orphan in target.parent.glob(f".{target.name}.*.tmp"):
      try:
        orphan.unlink()
      except OSError:
        pass
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
      with os.fdopen(fd, "wb") as fp:
        fp.write(payload)
        fp.flush()
        os.fsync(fp.fileno())
      os.replace(tmp, target)
    except BaseException:
      tmp.unlink(missing_ok=True)
      raise
    dir_fd = os.open(str(target.parent), os.O_RDONLY)
    try:
      os.fsync(dir_fd)
    except OSError as exc:
      if exc.errno not in (errno.EINVAL, errno.EBADF):
        raise
    finally:
      os.close(dir_fd)
  finally:
    try:
      fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
      os.close(lock_fd)
  return target
