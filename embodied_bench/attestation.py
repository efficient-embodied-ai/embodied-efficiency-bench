"""Cryptographic SHA-256 submission receipt generator and verifier for EmbodiedX."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from embodied_bench.schema import (
    RunStatus,
    atomic_write_json,
    load_yaml_or_json,
    now_pacific_iso,
    validate_status_and_reason,
)


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
  """Serialize a dictionary into canonical sorted-key UTF-8 JSON bytes for SHA-256 hashing."""
  return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_canonical_attestation_body(report: Dict[str, Any]) -> Dict[str, Any]:
  """Extract the security-critical fields covered by the SHA-256 submission attestation."""
  frontier_digest_items: List[Dict[str, Any]] = []
  x_key = str(report["x_metric_name"])
  y_key = str(report["y_metric_name"])
  lat_key = str(report["latency_metric_name"])

  is_wm = str(report.get("track", "")) == "WM_TRACK"
  for pt in report.get("pareto_frontier", []):
    gate_a = pt.get("co_located_gate_a", {})
    hw = pt.get("hardware_telemetry", {})
    item: Dict[str, Any] = {
        "point_id": pt.get("point_id"),
        "execution_id": pt.get("execution_id"),
        "gate_a_execution_id": gate_a.get("execution_id"),
        "gate_a_passed": bool(pt.get("gate_a_passed", False)),
        "x_reactivity_hz": round(float(pt[x_key]), 4),
        "y_throughput": round(float(pt[y_key]), 4),
        "p99_latency_ms": round(float(pt[lat_key]), 3),
        "hardware_fingerprint": hw.get("hardware_fingerprint"),
    }
    c_val = pt.get("concurrency", pt.get("batch_size"))
    if c_val is not None:
      item["concurrency"] = int(c_val)
    if pt.get("execute_horizon") is not None:
      item["execute_horizon"] = int(pt["execute_horizon"])
    h_val = pt.get("horizon_frames" if is_wm else "horizon")
    if h_val is not None:
      item["horizon"] = int(h_val)
    p50_val = pt.get("rollout_latency_p50_ms" if is_wm else "obs_to_action_p50_ms", pt.get("p50_latency_ms"))
    if p50_val is not None:
      item["p50_latency_ms"] = round(float(p50_val), 3)
    max_val = pt.get("rollout_latency_max_ms" if is_wm else "obs_to_action_max_ms", pt.get("max_latency_ms"))
    if max_val is not None:
      item["max_latency_ms"] = round(float(max_val), 3)
    frontier_digest_items.append(item)

  body: Dict[str, Any] = {
      "schema_version": str(report.get("schema_version", "0.2.0")),
      "recipe_id": str(report["recipe_id"]),
      "track": str(report["track"]),
      "model_name": str(report["model_name"]),
      "baseline_model_id": str(report["baseline_model_id"]),
      "precision": str(report["precision"]),
      "hardware_sku": str(report["hardware_sku"]),
      "compression_tier": str(report["compression_tier"]),
      "status": str(report["status"]),
      "reason": report.get("reason"),
      "iso_reactivity_slices": report.get("iso_reactivity_slices", {}),
      "frontier_points": frontier_digest_items,
      "stage2_gate_b": report.get("stage2_gate_b"),
  }
  if "division" in report and report["division"] is not None:
    body["division"] = str(report["division"])
  if "execution_provenance" in report and report["execution_provenance"] is not None:
    body["execution_provenance"] = report["execution_provenance"]
  obs_src = report.get("observation_source")
  if isinstance(obs_src, dict) and obs_src.get("kind") is not None:
    body["observation_source_kind"] = str(obs_src["kind"])
  q_gate = report.get("quality_gate")
  if isinstance(q_gate, dict) and q_gate.get("golden_sha256"):
    body["golden_sha256"] = str(q_gate["golden_sha256"])
  if report.get("recipe_commit_sha256"):
    body["recipe_commit_sha256"] = str(report["recipe_commit_sha256"])
  if report.get("phase1_source_attestation_sha256"):
    body["phase1_source_attestation_sha256"] = str(report["phase1_source_attestation_sha256"])
  return body


def compute_attestation_sha256(canonical_body: Dict[str, Any]) -> str:
  """Compute hex SHA-256 digest over canonical attestation body."""
  return hashlib.sha256(_canonical_json_bytes(canonical_body)).hexdigest()


def create_submission_receipt(
    report: Dict[str, Any],
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
  """Create a signed submission_receipt.json from a Stage 1/2 or live benchmark evaluation report."""
  validate_status_and_reason(
      status=report.get("status", RunStatus.INVALID.value),
      reason=report.get("reason"),
  )
  body = build_canonical_attestation_body(report)
  digest = compute_attestation_sha256(body)
  receipt: Dict[str, Any] = {
      "receipt_version": "0.2.0",
      "issued_at": now_pacific_iso(),
      "recipe_id": body["recipe_id"],
      "track": body["track"],
      "hardware_sku": body["hardware_sku"],
      "status": body["status"],
      "reason": body["reason"],
      "attestation_algorithm": "SHA-256",
      "attestation_sha256": digest,
      "attested_body": body,
  }
  for opt_key in (
      "division",
      "scenario",
      "clock",
      "perf_cap",
      "num_devices",
      "single_device_compliant",
      "execution_strategy",
      "model_family",
      "model_name",
      "baseline_model_id",
      "precision",
      "x_metric_name",
      "y_metric_name",
      "latency_metric_name",
      "summary_metrics",
      "iso_reactivity_slices",
      "semianalysis_normalizations",
      "performance_cap_evaluation",
      "compliance_gates_g1_g4",
      "envelope_check",
      "observation_source",
      "quality_gate",
      "quality_certified",
      "gate_mode",
      "operating_points",
      "pareto_frontier",
      "execution_provenance",
  ):
    if opt_key in report and report[opt_key] is not None:
      receipt[opt_key] = report[opt_key]
  if output_path is not None:
    atomic_write_json(output_path, receipt)
  return receipt


def _resolve_metric_val(
    pt: Dict[str, Any], primary_key: str, *fallbacks: str
) -> Optional[float]:
  """Return first non-None numeric metric value from `pt` among `(primary_key, *fallbacks)`."""
  for k in (primary_key, *fallbacks):
    if k and k in pt and pt[k] is not None:
      return float(pt[k])
  return None


def _verify_container_against_attested_body(
    container: Dict[str, Any],
    body: Dict[str, Any],
    *,
    label: str,
) -> Optional[str]:
  """Verify that top-level mirrored metadata, frontier, operating points, and summary agree with signed `body`."""
  for shared_key in (
      "recipe_id",
      "track",
      "model_name",
      "baseline_model_id",
      "precision",
      "hardware_sku",
      "compression_tier",
      "division",
      "recipe_commit_sha256",
      "status",
      "reason",
      "iso_reactivity_slices",
      "execution_provenance",
  ):
    if (
        shared_key in body
        and shared_key in container
        and container[shared_key] is not None
        and container[shared_key] != body[shared_key]
    ):
      return (
          f"Tampered {label} field {shared_key!r}={container[shared_key]!r} != "
          f"signed attested_body[{shared_key!r}]={body[shared_key]!r}."
      )

  frontier_pts = body.get("frontier_points")
  if not isinstance(frontier_pts, list):
    return f"Invalid signed attested_body.frontier_points in {label}."

  is_wm = str(body.get("track", "")) == "WM_TRACK"
  x_key = str(container.get("x_metric_name") or ("hz_plan" if is_wm else "hz_react"))
  y_key = str(
      container.get("y_metric_name")
      or ("predictions_per_sec" if is_wm else "executed_actions_per_sec")
  )
  lat_key = str(
      container.get("latency_metric_name")
      or ("rollout_latency_p99_ms" if is_wm else "obs_to_action_p99_ms")
  )
  x_fallbacks = ("hz_react", "hz_plan")
  y_fallbacks = (
      "predictions_per_sec",
      "generated_frames_per_sec",
      "frames_per_sec",
      "fps",
      "executed_actions_per_sec",
      "actions_per_sec",
      "raw_per_chip_throughput",
  )
  lat_fallbacks = ("rollout_latency_p99_ms", "obs_to_action_p99_ms", "ttfa_p99_ms")

  top_frontier = container.get("pareto_frontier")
  if isinstance(top_frontier, list):
    if len(top_frontier) != len(frontier_pts):
      return (
          f"Tampered {label}.pareto_frontier length={len(top_frontier)} != "
          f"signed attested_body.frontier_points length={len(frontier_pts)}."
      )
    for idx, (t_pt, a_pt) in enumerate(zip(top_frontier, frontier_pts)):
      if not isinstance(t_pt, dict):
        return f"Invalid entry {label}.pareto_frontier[{idx}]: expected object."
      if t_pt.get("point_id") != a_pt.get("point_id") or t_pt.get("execution_id") != a_pt.get("execution_id"):
        return (
            f"Tampered {label}.pareto_frontier[{idx}] identity "
            f"(point_id={t_pt.get('point_id')!r}, execution_id={t_pt.get('execution_id')!r}) != "
            f"signed ({a_pt.get('point_id')!r}, {a_pt.get('execution_id')!r})."
        )
      tx = _resolve_metric_val(t_pt, x_key, *x_fallbacks)
      ty = _resolve_metric_val(t_pt, y_key, *y_fallbacks)
      tlat = _resolve_metric_val(t_pt, lat_key, *lat_fallbacks)
      traw = _resolve_metric_val(t_pt, "raw_per_chip_throughput")
      ax = float(a_pt.get("x_reactivity_hz", 0.0))
      ay = float(a_pt.get("y_throughput", 0.0))
      alat = float(a_pt.get("p99_latency_ms", 0.0))
      if tx is not None and abs(tx - ax) > 1e-2:
        return f"Tampered {label}.pareto_frontier[{idx}] reactivity={tx} != signed x_reactivity_hz={ax}."
      if ty is not None and abs(ty - ay) > 1e-2:
        return f"Tampered {label}.pareto_frontier[{idx}] throughput={ty} != signed y_throughput={ay}."
      if traw is not None and abs(traw - ay) > 1e-2:
        return f"Tampered {label}.pareto_frontier[{idx}] raw_per_chip_throughput={traw} != signed y_throughput={ay}."
      if tlat is not None and abs(tlat - alat) > 1e-2:
        return f"Tampered {label}.pareto_frontier[{idx}] latency={tlat} != signed p99_latency_ms={alat}."

  top_ops = container.get("operating_points")
  if isinstance(top_ops, list):
    ops_by_id: Dict[Any, Dict[str, Any]] = {}
    for op in top_ops:
      if isinstance(op, dict):
        ops_by_id[op.get("point_id")] = op
    for a_pt in frontier_pts:
      pid = a_pt.get("point_id")
      op = ops_by_id.get(pid)
      if op is None:
        return (
            f"Tampered {label}.operating_points: missing signed frontier point_id={pid!r}."
        )
      if op.get("execution_id") != a_pt.get("execution_id"):
        return (
            f"Tampered {label}.operating_points[{pid!r}].execution_id={op.get('execution_id')!r} != "
            f"signed execution_id={a_pt.get('execution_id')!r}."
        )
      ox = _resolve_metric_val(op, x_key, *x_fallbacks)
      oy = _resolve_metric_val(op, y_key, *y_fallbacks)
      olat = _resolve_metric_val(op, lat_key, *lat_fallbacks)
      oraw = _resolve_metric_val(op, "raw_per_chip_throughput")
      ax = float(a_pt.get("x_reactivity_hz", 0.0))
      ay = float(a_pt.get("y_throughput", 0.0))
      alat = float(a_pt.get("p99_latency_ms", 0.0))
      if ox is not None and abs(ox - ax) > 1e-2:
        return f"Tampered {label}.operating_points[{pid!r}] reactivity={ox} != signed x_reactivity_hz={ax}."
      if oy is not None and abs(oy - ay) > 1e-2:
        return f"Tampered {label}.operating_points[{pid!r}] throughput={oy} != signed y_throughput={ay}."
      if oraw is not None and abs(oraw - ay) > 1e-2:
        return f"Tampered {label}.operating_points[{pid!r}] raw_per_chip_throughput={oraw} != signed y_throughput={ay}."
      if olat is not None and abs(olat - alat) > 1e-2:
        return f"Tampered {label}.operating_points[{pid!r}] latency={olat} != signed p99_latency_ms={alat}."

    for idx, op in enumerate(top_ops):
      if not isinstance(op, dict):
        return f"Invalid entry {label}.operating_points[{idx}]: expected object."
      pid = op.get("point_id", f"idx_{idx}")
      oy = _resolve_metric_val(op, y_key, *y_fallbacks)
      oraw = _resolve_metric_val(op, "raw_per_chip_throughput")
      if oy is not None and oraw is not None and abs(oraw - oy) > 1e-2:
        return (
            f"Tampered {label}.operating_points[{pid!r}]: raw_per_chip_throughput={oraw} != "
            f"{y_key}={oy}."
        )
      if bool(op.get("gate_a_passed", False)) and bool(op.get("variance_passed", True)):
        ox = _resolve_metric_val(op, x_key, *x_fallbacks)
        if ox is not None and oy is not None:
          dominated_or_matched = any(
              float(a_pt.get("x_reactivity_hz", 0.0)) >= ox - 1e-2
              and float(a_pt.get("y_throughput", 0.0)) >= oy - 1e-2
              for a_pt in frontier_pts
          )
          if not dominated_or_matched:
            return (
                f"Tampered {label}.operating_points[{pid!r}]: admissible point "
                f"(x={ox}, y={oy}) exceeds every signed attested_body.frontier_points entry."
            )

  attested_max_tput = (
      max(float(pt.get("y_throughput", 0.0)) for pt in frontier_pts)
      if frontier_pts
      else 0.0
  )
  for sum_field in ("summary_metrics", "summary"):
    top_sum = container.get(sum_field)
    if not isinstance(top_sum, dict):
      continue
    if "num_pareto_frontier_points" in top_sum and top_sum["num_pareto_frontier_points"] is not None:
      if int(top_sum["num_pareto_frontier_points"]) != len(frontier_pts):
        return (
            f"Tampered {label}.{sum_field}.num_pareto_frontier_points="
            f"{top_sum['num_pareto_frontier_points']} != signed count={len(frontier_pts)}."
        )
    if frontier_pts and "peak_throughput" in top_sum and top_sum["peak_throughput"] is not None:
      claimed_peak = float(top_sum["peak_throughput"])
      if abs(claimed_peak - attested_max_tput) > 1e-2:
        return (
            f"Tampered {label}.{sum_field}.peak_throughput={claimed_peak} != "
            f"signed attested_body frontier peak={attested_max_tput}."
        )
    if isinstance(top_ops, list) and top_ops:
      peak_field_specs = (
          ("peak_executed_actions_per_sec", "executed_actions_per_sec", ("actions_per_sec", "raw_per_chip_throughput")),
          ("peak_predictions_per_sec", "predictions_per_sec", ("chunks_per_sec", "raw_per_chip_throughput")),
          ("peak_generated_frames_per_sec", "generated_frames_per_sec", ("frames_per_sec", "fps")),
      )
      for peak_k, pk_primary, pk_fallbacks in peak_field_specs:
        if peak_k in top_sum and top_sum[peak_k] is not None:
          pk_vals = [
              _resolve_metric_val(op, pk_primary, *pk_fallbacks)
              for op in top_ops
              if isinstance(op, dict)
          ]
          pk_num = [v for v in pk_vals if v is not None]
          if pk_num:
            expected_peak = max(pk_num)
            if abs(float(top_sum[peak_k]) - expected_peak) > 1e-2:
              return (
                  f"Tampered {label}.{sum_field}.{peak_k}={top_sum[peak_k]} != "
                  f"operating_points peak={expected_peak}."
              )
  return None


def verify_submission_receipt(
    receipt_or_path: Dict[str, Any] | str | Path,
) -> Tuple[bool, str]:
  """Verify cryptographic SHA-256 integrity, co-located Gate A/B invariants, and sibling bundle parity."""
  bundle_dir: Optional[Path] = None
  check_sibling_reports = False
  if isinstance(receipt_or_path, (str, Path)):
    p = Path(receipt_or_path)
    if not p.exists():
      return (False, f"Submission receipt path does not exist: {p}")
    if p.is_dir():
      bundle_dir = p
      check_sibling_reports = True
      candidates = [p / "receipt.json", p / "submission_receipt.json", p / "bundle_summary.json"]
      found = next((c for c in candidates if c.is_file()), None)
      if found is None:
        return (
            False,
            f"No 'receipt.json', 'submission_receipt.json', or 'bundle_summary.json' found inside directory {p}.",
        )
      receipt_file = found
    else:
      receipt_file = p
      bundle_dir = p.parent
      check_sibling_reports = p.name in (
          "receipt.json",
          "submission_receipt.json",
          "bundle_summary.json",
      )
    try:
      receipt = load_yaml_or_json(receipt_file)
    except Exception as exc:
      return (False, f"Failed to parse receipt file {receipt_file}: {exc}")
  else:
    receipt = dict(receipt_or_path)

  claimed_sha = receipt.get("attestation_sha256")
  body = receipt.get("attested_body")
  if not claimed_sha or not isinstance(body, dict):
    return (
        False,
        "Invalid submission receipt structure: missing 'attestation_sha256' or 'attested_body'. "
        "Re-generate the bundle using 'eeb measure' or 'embodied-bench compute-pareto --receipt-output <path>'.",
    )

  expected_sha = compute_attestation_sha256(body)
  if claimed_sha != expected_sha:
    return (
        False,
        f"Cryptographic SHA-256 mismatch: claimed={claimed_sha} != recomputed={expected_sha}. "
        "The submission receipt payload was modified after signing.",
    )

  mirror_err = _verify_container_against_attested_body(receipt, body, label="receipt")
  if mirror_err is not None:
    return (False, mirror_err)

  try:
    resolved_status = validate_status_and_reason(
        status=body.get("status", ""),
        reason=body.get("reason"),
    )
  except ValueError as exc:
    return (False, f"Receipt status validation failed: {exc}")

  frontier_pts = body.get("frontier_points", [])
  for pt in frontier_pts:
    exec_id = pt.get("execution_id")
    gate_exec_id = pt.get("gate_a_execution_id")
    if not exec_id or exec_id != gate_exec_id:
      return (
          False,
          f"Co-located execution_id mismatch on point {pt.get('point_id')!r}: "
          f"execution_id={exec_id!r} != gate_a_execution_id={gate_exec_id!r}.",
      )
    if not pt.get("gate_a_passed", False):
      return (
          False,
          f"Frontier point {pt.get('point_id')!r} has gate_a_passed=False; "
          "dominated or inadmissible points cannot appear on a certified frontier.",
      )

  if resolved_status == RunStatus.CERTIFIED:
    if not frontier_pts:
      return (
          False,
          "Receipt claims status=CERTIFIED but frontier_points is empty.",
      )
    stage2 = body.get("stage2_gate_b")
    if isinstance(stage2, dict) and not stage2.get("passed", False):
      return (
          False,
          "Receipt claims status=CERTIFIED but stage2_gate_b.passed is False.",
      )
    g1_g4 = receipt.get("compliance_gates_g1_g4") or body.get("compliance_gates_g1_g4")
    if isinstance(g1_g4, dict) and not g1_g4.get("passed", True):
      return (
          False,
          f"Receipt claims status=CERTIFIED but compliance_gates_g1_g4 failed: {g1_g4.get('reason')}",
      )

  if check_sibling_reports and bundle_dir is not None:
    for sibling_name in ("report.json", "sweep_results.json", "bundle_summary.json"):
      if sibling_name == receipt_file.name:
        continue
      sibling_path = bundle_dir / sibling_name
      if not sibling_path.is_file():
        continue
      try:
        sibling_report = load_yaml_or_json(sibling_path)
      except Exception as exc:
        return (False, f"Failed to parse sibling {sibling_name} in {bundle_dir}: {exc}")
      if sibling_name == "bundle_summary.json" and "attested_body" in sibling_report:
        sib_claimed = sibling_report.get("attestation_sha256")
        sib_body = sibling_report.get("attested_body")
        if not isinstance(sib_body, dict) or not sib_claimed:
          return (False, f"Invalid sibling bundle_summary.json in {bundle_dir}: missing attested_body or attestation_sha256.")
        sib_recomp = compute_attestation_sha256(sib_body)
        if sib_claimed != sib_recomp or sib_recomp != expected_sha:
          return (
              False,
              f"Cryptographic SHA-256 mismatch against sibling {sibling_name}: "
              f"receipt={expected_sha} != bundle_summary={sib_recomp} (claimed={sib_claimed}).",
          )
      else:
        try:
          sibling_body = build_canonical_attestation_body(sibling_report)
        except Exception as exc:
          return (
              False,
              f"Sibling {sibling_name} in {bundle_dir} failed canonical attestation reconstruction: {exc}",
          )
        sibling_sha = compute_attestation_sha256(sibling_body)
        if sibling_sha != expected_sha:
          return (
              False,
              f"Cryptographic SHA-256 mismatch against sibling {sibling_name}: "
              f"receipt={expected_sha} != recomputed({sibling_name})={sibling_sha}. "
              f"{sibling_name} was modified after signing.",
          )
      sib_err = _verify_container_against_attested_body(
          sibling_report, body, label=sibling_name
      )
      if sib_err is not None:
        return (False, sib_err)

  return (
      True,
      f"Submission receipt verified for recipe_id={body['recipe_id']!r} "
      f"(status={body['status']}, sha256={expected_sha[:16]}...).",
  )
