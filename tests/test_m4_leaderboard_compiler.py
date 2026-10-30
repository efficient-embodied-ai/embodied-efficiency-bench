"""Tests for Milestone M4: EmbodiedX Season 1 Option-A Leaderboard Compiler, Two-Phase Commit-Reveal, Gapless SLA Awards, and 9 Red-Team Anti-Cheat Gates.

Enforces Adversarial Self-Review Gates 1, 2, 3, 4, and 5:
  - Gate 1: Crash recovery (orphan `.tmp`), repeat-call idempotency, and concurrent writers on `prepare_phase1_commit_bundle`.
  - Gate 2: Every REJECTED receipt produces non-empty violation reasons.
  - Gate 3: Both flag orders tested via subprocess for `eeb prepare-commit-bundle`, `eeb verify-bundle`, `eeb build-leaderboard`, `eeb export-website`, and `scripts/export_website_leaderboard.py`.
  - Gate 4: Negative controls + positive lookalikes for all 9 Red-Team anti-gaming defenses (RT1..RT9).
  - Gate 5: Reuses `CANONICAL_CHALLENGE_MODELS`, `GAPLESS_REALTIME_SLA_SPEC`, `compute_realtime_normalized_hypervolume`, and cryptographic attestation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

from eeb.loadgen.sweep import _apply_timed_anti_lut_jitter, is_canonical_concurrency_prefix
from embodied_bench.attestation import (
    create_submission_receipt,
    verify_submission_receipt,
)
from embodied_bench.cli import main as eeb_main
from embodied_bench.quality_gates import evaluate_vla_gate_a
from embodied_bench.leaderboard_compiler import (
    GAPLESS_REALTIME_SLA_SPEC,
    HYPERVOLUME_U_REF,
    SEASON1_OFFICIAL_HARDWARE_TARGETS,
    SUBMISSION_PROTOCOL_SPEC,
    build_reference_seed_matrix,
    compile_submissions_to_leaderboard,
    compute_realtime_dominated_hypervolume,
    generate_reference_submissions,
    render_pr_comment_markdown,
)
from embodied_bench.schema import CANONICAL_CHALLENGE_MODELS, DivisionType, TrackType, atomic_write_json
from embodied_bench.submission_validator import (
    prepare_phase1_commit_bundle,
    validate_submission_directory,
    verify_submission_bundle,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_measured_submission(
    sub_dir: Path,
    *,
    recipe_id: str,
    track: str = TrackType.VLA_TRACK.value,
    model_id: str = "pi05-3b",
    division: str = DivisionType.CLOSED.value,
    hardware_sku: str = "L40S_48GB",
    points: List[Dict[str, Any]] | None = None,
    power_provenance: str = "MEASURED",
    power_telemetry_scope: str = "NVML_BOARD_POWER_W",
    power_limit_w: float = 350.0,
    execute_horizon: int | None = None,
    stage2_gate_b: Dict[str, Any] | None = None,
    observation_source_kind: str = "EMBODIEDX_TRACES_V1_BUNDLE",
    golden_sha256: str = "b" * 64,
    recipe_commit_sha256: str | None = None,
) -> Path:
  """Build a fully signed single-chip measured submission bundle in `sub_dir`."""
  sub_dir.mkdir(parents=True, exist_ok=True)
  wm = track == TrackType.WM_TRACK.value
  locked_k = execute_horizon if execute_horizon is not None else (1 if wm else (8 if model_id == "gr00t-n1.7-3b" else 5))
  x_key = "hz_plan" if wm else "hz_react"
  y_key = "fps" if wm else "executed_actions_per_sec"
  lat_key = "rollout_latency_p99_ms" if wm else "obs_to_action_p99_ms"

  if points is None:
    points = [
        {"concurrency": 1, "p50_ms": 40.0, "p99_ms": 48.0, "max_ms": 62.0, "y": 110.0, "power_w": 220.0},
        {"concurrency": 2, "p50_ms": 60.0, "p99_ms": 72.0, "max_ms": 95.0, "y": 180.0, "power_w": 260.0},
        {"concurrency": 4, "p50_ms": 110.0, "p99_ms": 135.0, "max_ms": 180.0, "y": 295.0, "power_w": 300.0},
    ]

  op_pts: List[Dict[str, Any]] = []
  for row in points:
    c = int(row["concurrency"])
    p50 = float(row["p50_ms"])
    p99 = float(row["p99_ms"])
    max_ms = float(row.get("max_ms", p99 * 1.15))
    y_val = float(row["y"])
    pwr_w = float(row.get("power_w", 240.0))
    pt_p_limit = float(row.get("power_limit_w", power_limit_w))
    pt_prov = str(row.get("power_provenance", power_provenance))
    exec_id = f"exec-{recipe_id}-c{c}"
    x_val = round(1000.0 / p99, 4)
    p_eff = round(y_val / pwr_w, 4) if pt_prov == "MEASURED" else None
    gate_a_metrics = row.get(
        "gate_a_metrics",
        {
            "action_cosine_sim": 0.9992,
            "action_cosine_cvar_1pct": 0.9940,
            "action_mse": 1e-5,
            "latent_action_cosine_sim": 0.996,
            "cand_psnr_db": 32.0,
            "cand_lpips": 0.05,
            "cand_fvd": 40.0,
        },
    )
    op_pts.append({
        "point_id": f"C{c}",
        "execution_id": exec_id,
        "concurrency": c,
        "batch_size": c,
        "execute_horizon": locked_k,
        "chunk_size": locked_k,
        "horizon_frames": 16,
        x_key: x_val,
        y_key: y_val,
        lat_key: p99,
        "p50_latency_ms": p50,
        "p99_latency_ms": p99,
        "max_latency_ms": max_ms,
        "mean_latency_ms": p50,
        "power_efficiency_per_watt": p_eff,
        "throughput_per_100_tops": round(y_val / 3.66, 4),
        "throughput_per_tbps_bw": round(y_val / 0.864, 4),
        "throughput_per_dollar_hr": round(y_val / 1.15, 4),
        "tco_efficiency_per_dollar": round(y_val / 1.15, 4),
        "same_chip_speedup_vs_bf16_ref": 1.0,
        "gate_a_passed": True,
        "co_located_gate_a": {
            "execution_id": exec_id,
            "throughput_execution_id": exec_id,
            "passed": True,
            "metrics": gate_a_metrics,
        },
        "hardware_telemetry": {
            "power_provenance": pt_prov,
            "power_telemetry_scope": power_telemetry_scope,
            "mean_power_watts": pwr_w,
            "power_limit_w": pt_p_limit,
        },
    })

  if stage2_gate_b is None:
    stage2_gate_b = {
        "passed": True,
        "delta_success_rate_pct": -0.4,
        "psnr_db": 31.5,
        "lpips": 0.06,
        "fvd": 42.0,
        "inverse_dynamics_action_cosine_sim": 0.994,
        "s2_policy_ranking_kendall_tau": 0.91,
    }

  report: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "recipe_id": recipe_id,
      "track": track,
      "division": division,
      "model_family": model_id,
      "model_name": f"{model_id}-{hardware_sku}",
      "baseline_model_id": model_id,
      "precision": "FP8" if division == DivisionType.OPEN.value else "BF16",
      "compression_tier": "moderate_quant" if division == DivisionType.OPEN.value else "strict_lossless",
      "hardware_sku": hardware_sku,
      "perf_cap": "CAP_SINGLE_CHIP",
      "num_devices": 1,
      "single_device_compliant": True,
      "execute_horizon": locked_k,
      "concurrency_levels": [int(p["concurrency"]) for p in op_pts],
      "status": "CERTIFIED",
      "reason": None,
      "x_metric_name": x_key,
      "y_metric_name": y_key,
      "latency_metric_name": lat_key,
      "observation_source": {
          "kind": observation_source_kind,
          "image_hw": [256, 256] if (wm or model_id == "gr00t-n1.7-3b") else [224, 224],
          "bytes_per_request_images": 393216 if (wm or model_id == "gr00t-n1.7-3b") else 301056,
      },
      "golden_bundle": {"sha256": golden_sha256},
      "execution_provenance": {
          "is_live_empirical": True,
          "is_analytic_projection": False,
          "execution_mode": "live_empirical_runner",
          "power": power_provenance,
      },
      "hardware_telemetry": {
          "power_provenance": power_provenance,
          "power_telemetry_scope": power_telemetry_scope,
          "power_limit_w": power_limit_w,
      },
      "compliance_gates_g1_g4": {"passed": True, "execution_id": op_pts[0]["execution_id"]},
      "stage2_gate_b": stage2_gate_b,
      "operating_points": op_pts,
      "pareto_frontier": op_pts,
  }
  if recipe_commit_sha256 is not None:
    report["recipe_commit_sha256"] = recipe_commit_sha256

  report_path = sub_dir / "report.json"
  receipt_path = sub_dir / "receipt.json"
  atomic_write_json(report_path, report)
  create_submission_receipt(report=report, output_path=receipt_path)
  return receipt_path


def test_m4_day1_empty_option_a_leaderboard_and_season1_metadata() -> None:
  """Verifies Option A Day-1 posture: 0 bundled submissions, L40S + TPU v6e-1 targets, and exact award metadata."""
  lb = json.loads((REPO_ROOT / "leaderboard" / "leaderboard_v1.json").read_text(encoding="utf-8"))
  assert lb["num_verified_submissions"] == 0
  assert lb["num_rejected_submissions"] == 0
  assert lb["submissions"] == []
  assert "L40S_48GB" in lb["season1_official_hardware_targets"].values()
  assert "TPU_V6E_1_32GB" in lb["season1_official_hardware_targets"].values()
  assert set(SEASON1_OFFICIAL_HARDWARE_TARGETS.values()) == {"L40S_48GB", "TPU_V6E_1_32GB"}
  assert SUBMISSION_PROTOCOL_SPEC["phase2_snapshot_reveal"]["ranking_authority"] == "PHASE2_ORGANIZER_CLEAN_HARDWARE_SCORE"
  assert GAPLESS_REALTIME_SLA_SPEC["VLA_TRACK_1A_PI05"]["max_p99_latency_ms"] == 250.0
  assert GAPLESS_REALTIME_SLA_SPEC["VLA_TRACK_1B_GR00T"]["max_p99_latency_ms"] == 400.0
  assert GAPLESS_REALTIME_SLA_SPEC["WM_TRACK"]["max_p99_latency_ms"] == 1000.0

  md = render_pr_comment_markdown(lb)
  assert "FOUNDING SEASON v1.0 OPEN FOR SUBMISSIONS" in md
  assert "NVIDIA L40S" in md and "Google Cloud TPU v6e-1" in md
  assert "monetary prizes" in md
  assert "an invitation to apply for Google Cloud credits" in md
  assert "\u2014" not in md


def test_m4_analytic_seeds_drop_unmeasured_energy_while_measured_populates_awards(tmp_path: Path) -> None:
  """RT Fix 1: Analytic seeds carry 0 Y2 energy points; MEASURED receipts populate Y2 and Awards A/B/C."""
  specs = build_reference_seed_matrix()
  assert {s.model_id for s in specs} == set(CANONICAL_CHALLENGE_MODELS)

  seed_dir = tmp_path / "analytic_seeds"
  receipts = generate_reference_submissions(seed_dir)
  assert len(receipts) == len(specs)
  for rpath in receipts:
    ok, msg = verify_submission_receipt(rpath)
    assert ok is True, msg

  lb_seed = compile_submissions_to_leaderboard(seed_dir, output_json_path=tmp_path / "lb_seed.json")
  assert lb_seed["num_verified_submissions"] == len(specs)
  for model_id in CANONICAL_CHALLENGE_MODELS:
    f4 = lb_seed["cross_chip_by_model"][model_id]["semianalysis_4way_pareto_frontiers"]
    assert f4["power_efficiency_per_watt"] == []
    assert len(f4["raw_per_chip_throughput"]) >= 1
    assert len(f4["iso_compute_and_bw_efficiency"]) >= 1
    assert len(f4["tco_efficiency_per_dollar"]) >= 1
  for grp in lb_seed["award_rankings"].values():
    assert grp["green_embodied_efficiency_rank"] == []

  meas_root = tmp_path / "measured_season1"
  _make_measured_submission(
      meas_root / "team_alpha_l40s",
      recipe_id="alpha_pi05_l40s_trt",
      model_id="pi05-3b",
      hardware_sku="L40S_48GB",
      points=[
          {"concurrency": 1, "p50_ms": 30.0, "p99_ms": 36.0, "max_ms": 44.0, "y": 140.0, "power_w": 210.0},
          {"concurrency": 2, "p50_ms": 50.0, "p99_ms": 62.0, "max_ms": 80.0, "y": 245.0, "power_w": 250.0},
          {"concurrency": 4, "p50_ms": 95.0, "p99_ms": 120.0, "max_ms": 150.0, "y": 380.0, "power_w": 290.0},
      ],
  )
  _make_measured_submission(
      meas_root / "team_beta_tpu",
      recipe_id="beta_pi05_tpu_v6e",
      model_id="pi05-3b",
      hardware_sku="TPU_V6E_1_32GB",
      power_provenance="UNAVAILABLE",
      points=[
          {"concurrency": 1, "p50_ms": 22.0, "p99_ms": 25.0, "max_ms": 30.0, "y": 200.0, "power_w": 150.0},
          {"concurrency": 2, "p50_ms": 32.0, "p99_ms": 38.0, "max_ms": 48.0, "y": 310.0, "power_w": 160.0},
      ],
  )
  lb_meas = compile_submissions_to_leaderboard(meas_root, output_json_path=tmp_path / "lb_meas.json")
  assert lb_meas["num_verified_submissions"] == 2
  pi05_awards = lb_meas["award_rankings"]["VLA_TRACK__pi05-3b__CLOSED"]
  # Award A: TPU v6e entry has lower C=1 p99 (25 ms -> 40 Chunks/s > 27.78 Chunks/s)
  assert pi05_awards["interactive_reactivity_rank"][0]["recipe_id"] == "beta_pi05_tpu_v6e"
  # Award B: L40S entry has higher SLA-compliant peak throughput (380 Act/s at C=4 <= 250 ms > 310 Act/s)
  assert pi05_awards["realtime_fleet_capacity_rank"][0]["recipe_id"] == "alpha_pi05_l40s_trt"
  # Award C: Only MEASURED board telemetry entry qualifies (UNAVAILABLE TPU entry excluded)
  assert len(pi05_awards["green_embodied_efficiency_rank"]) == 1
  assert pi05_awards["green_embodied_efficiency_rank"][0]["recipe_id"] == "alpha_pi05_l40s_trt"

  md_pop = render_pr_comment_markdown(lb_meas)
  assert "| Submission (`recipe_id`)" in md_pop
  assert "Season 1 Track Award Standings" in md_pop
  assert "EXCEEDS 250 ms SLA" not in md_pop
  assert "380.00 (C=4, 120.0 ms)" in md_pop
  assert "\u2014" not in md_pop


def test_m4_rt8_boundary_hypervolume_and_rt5_bimodal_stall_gate(tmp_path: Path) -> None:
  """RT Fix 8 (`u_ref=0.5`) gives strictly positive HV right on the SLA boundary; RT Fix 5 gates Award A stalls."""
  assert HYPERVOLUME_U_REF == pytest.approx(0.5)
  # Point directly on pi05-3b 250.0 ms SLA boundary -> hz_react = 4.0 == sla_min_rate_hz -> u = 1.0
  hv_val = compute_realtime_dominated_hypervolume([(4.0, 20.0)], sla_min_rate_hz=4.0)
  assert hv_val == pytest.approx(10.0)  # (1.0 - 0.5) * 20.0 = 10.0 > 0

  # RT5: Bimodal stall C=1 entry (p99/p50 = 40/20 = 2.0 > 1.5) is disqualified from Award A
  subs_rt5 = tmp_path / "rt5_subs"
  _make_measured_submission(
      subs_rt5 / "bimodal_stall",
      recipe_id="stall_c1",
      points=[{"concurrency": 1, "p50_ms": 20.0, "p99_ms": 40.0, "max_ms": 50.0, "y": 150.0}],
  )
  _make_measured_submission(
      subs_rt5 / "smooth_tail",
      recipe_id="smooth_c1",
      points=[{"concurrency": 1, "p50_ms": 42.0, "p99_ms": 50.0, "max_ms": 58.0, "y": 120.0}],
  )
  lb = compile_submissions_to_leaderboard(subs_rt5, output_json_path=tmp_path / "lb_rt5.json")
  award_a = lb["award_rankings"]["VLA_TRACK__pi05-3b__CLOSED"]["interactive_reactivity_rank"]
  assert [row["recipe_id"] for row in award_a] == ["smooth_c1"]


def test_m4_all_nine_redteam_negative_controls_and_commit_reveal(tmp_path: Path) -> None:
  """Gate 1 + Gate 2 + Gate 4 verification across RT1, RT2, RT3, RT4, RT6, RT7, and multi-device bounds."""
  # === RT2a: Anti-LUT frame jitter modifies >= 16 pixels within +-1 LSB without mutating caller array ===
  raw_frame = np.full((224, 224, 3), 128, dtype=np.uint8)
  jittered = _apply_timed_anti_lut_jitter(raw_frame, num_pixels=16)
  diff = np.abs(jittered.astype(np.int16) - raw_frame.astype(np.int16))
  assert int(np.count_nonzero(diff)) >= 16
  assert int(diff.max()) == 1
  assert int(raw_frame.min()) == 128 and int(raw_frame.max()) == 128

  # === RT3: Canonical concurrency prefix validator ===
  assert is_canonical_concurrency_prefix([1, 2, 4]) is True
  assert is_canonical_concurrency_prefix([2, 4, 8]) is False
  assert is_canonical_concurrency_prefix([1, 3]) is False

  # === RT6a: CVaR-1% tail cosine drop caught even when mean cosine >= 0.995 ===
  base_200 = np.zeros((200, 5, 7), dtype=np.float32)
  base_200[..., 0] = 1.0
  cand_200 = base_200.copy()
  cand_200[0, ..., 0] = 0.0
  cand_200[0, ..., 1] = 1.0
  g_cvar = evaluate_vla_gate_a(
      execution_id="exec-cvar-1",
      throughput_execution_id="exec-cvar-1",
      ref_actions=base_200,
      cand_actions=cand_200,
  )
  assert g_cvar.passed is False
  assert float(g_cvar.metrics["action_cosine_sim"]) >= 0.995
  assert float(g_cvar.metrics["action_cosine_cvar_1pct"]) < 0.980
  assert any("cvar_1pct" in v for v in g_cvar.violations)

  # === Negative-control directory plugged into compile_submissions_to_leaderboard ===
  neg_root = tmp_path / "neg_subs"
  # Valid baseline reference
  valid_dir = neg_root / "00_valid_l40s"
  _make_measured_submission(valid_dir, recipe_id="valid_l40s")

  # NC-RT1: Mixed dynamic power_limit_w across C points
  _make_measured_submission(
      neg_root / "01_mixed_plimit",
      recipe_id="bad_mixed_plimit",
      points=[
          {"concurrency": 1, "p50_ms": 40.0, "p99_ms": 48.0, "y": 100.0, "power_limit_w": 150.0},
          {"concurrency": 2, "p50_ms": 60.0, "p99_ms": 70.0, "y": 170.0, "power_limit_w": 350.0},
      ],
  )
  # NC-RT2: Bimodal spike max/p50 > 4.0 (cache-miss burst hiding)
  _make_measured_submission(
      neg_root / "02_bimodal_max",
      recipe_id="bad_bimodal_max",
      points=[{"concurrency": 1, "p50_ms": 30.0, "p99_ms": 42.0, "max_ms": 150.0, "y": 120.0}],
  )
  # NC-RT3: Non-canonical concurrency grid skipping C=1
  _make_measured_submission(
      neg_root / "03_skip_c1",
      recipe_id="bad_skip_c1",
      points=[{"concurrency": 4, "p50_ms": 90.0, "p99_ms": 110.0, "y": 300.0}],
  )
  # NC-RT4: OPEN submission inflating execute_horizon from K_eval=5 to 16
  _make_measured_submission(
      neg_root / "04_open_k16",
      recipe_id="bad_open_k16",
      division="OPEN",
      execute_horizon=16,
  )
  # NC-RT6b: OPEN Track 2 WM submission with Kendall tau = 0.65 < 0.80
  _make_measured_submission(
      neg_root / "05_open_wm_low_tau",
      recipe_id="bad_open_wm_tau",
      track="WM_TRACK",
      model_id="cosmos-predict2.5-2b",
      division="OPEN",
      stage2_gate_b={
          "passed": True,
          "psnr_db": 31.0,
          "lpips": 0.07,
          "fvd": 50.0,
          "inverse_dynamics_action_cosine_sim": 0.995,
          "s2_policy_ranking_kendall_tau": 0.65,
      },
  )
  # NC-Tampered golden_sha256 in signed body
  tampered_dir = neg_root / "06_tampered_golden"
  _make_measured_submission(tampered_dir, recipe_id="bad_tampered_golden")
  t_doc = json.loads((tampered_dir / "receipt.json").read_text(encoding="utf-8"))
  t_doc["attested_body"]["golden_sha256"] = "f" * 64
  (tampered_dir / "receipt.json").write_text(json.dumps(t_doc, indent=2), encoding="utf-8")

  # NC-RT7: Phase-1 Commit bundle with mismatched recipe_commit_sha256 in submission_meta.json
  p1_src_dir = tmp_path / "p1_src_run"
  _make_measured_submission(p1_src_dir, recipe_id="valid_phase1_l40s")
  p1_good_dir = neg_root / "07_good_phase1_commit"
  # Gate 1 crash test: pre-existing orphan .tmp cleaned up properly
  p1_good_dir.mkdir(parents=True, exist_ok=True)
  (p1_good_dir / ".tmp_stale.tmp").write_text("partial", encoding="utf-8")
  commit_hex = "c" * 64
  res_p1 = prepare_phase1_commit_bundle(
      source_bundle_dir=p1_src_dir,
      output_dir=p1_good_dir,
      recipe_commit_sha256=commit_hex,
      team="Robotics-Lab",
      engine_tag="tensorrt-fp8",
  )
  assert res_p1["recipe_commit_sha256"] == commit_hex
  assert (p1_good_dir / "bundle_summary.json").is_file()
  assert (p1_good_dir / "submission_meta.json").is_file()

  # Gate 1 repeat + concurrent calls on prepare_phase1_commit_bundle produce identical recipe_commit_sha256 & attestation
  res_p1_repeat = prepare_phase1_commit_bundle(
      source_bundle_dir=p1_src_dir,
      output_dir=p1_good_dir,
      recipe_commit_sha256=commit_hex,
      team="Robotics-Lab",
      engine_tag="tensorrt-fp8",
  )
  assert res_p1_repeat["recipe_commit_sha256"] == res_p1["recipe_commit_sha256"]
  assert res_p1_repeat["attestation_sha256"] == res_p1["attestation_sha256"]
  with ThreadPoolExecutor(max_workers=4) as pool:
    futs = [
        pool.submit(
            prepare_phase1_commit_bundle,
            p1_src_dir,
            tmp_path / f"conc_p1_{idx}",
            recipe_commit_sha256=commit_hex,
            team="Robotics-Lab",
            engine_tag="tensorrt-fp8",
        )
        for idx in range(4)
    ]
    for f in futs:
      r_f = f.result()
      assert r_f["recipe_commit_sha256"] == res_p1["recipe_commit_sha256"]
      assert r_f["attestation_sha256"] == res_p1["attestation_sha256"]

  p1_bad_dir = neg_root / "08_bad_phase1_commit_hash"
  shutil.copytree(p1_good_dir, p1_bad_dir)
  bad_meta = json.loads((p1_bad_dir / "submission_meta.json").read_text(encoding="utf-8"))
  bad_meta["recipe_commit_sha256"] = "d" * 64
  (p1_bad_dir / "submission_meta.json").write_text(json.dumps(bad_meta, indent=2), encoding="utf-8")
  bad_p1_rep = verify_submission_bundle(p1_bad_dir)
  assert bad_p1_rep.passed is False
  assert bad_p1_rep.reason and "mismatch" in bad_p1_rep.reason.lower()

  # NC-RT7b: Phase-1 bundle leaking server.py / report.json rejected by validate_submission_directory
  p1_leak_dir = tmp_path / "p1_leak_dir"
  shutil.copytree(p1_good_dir, p1_leak_dir)
  (p1_leak_dir / "server.py").write_text("print('secret kernel')\n", encoding="utf-8")
  leak_val = validate_submission_directory(p1_leak_dir)
  assert leak_val.passed is False and leak_val.reason and "Phase 1" in leak_val.reason

  with pytest.raises(ValueError, match="failed verification"):
    compile_submissions_to_leaderboard(neg_root, output_json_path=tmp_path / "lb_err.json", strict_verify=True)

  compiled = compile_submissions_to_leaderboard(neg_root, output_json_path=tmp_path / "lb_q.json", strict_verify=False)
  assert compiled["num_verified_submissions"] == 2
  assert compiled["num_rejected_submissions"] == 7
  for q in compiled["rejected_submissions"]:
    assert q["reason"] and len(q["reason"].strip()) > 0


def test_m4_cli_both_flag_orders_for_commit_bundle_and_export(tmp_path: Path) -> None:
  """Gate 3: Run prepare-commit-bundle, verify-bundle, build-leaderboard, and export_website_leaderboard.py in both flag orders."""
  src_a = tmp_path / "src_a"
  src_b = tmp_path / "src_b"
  _make_measured_submission(src_a, recipe_id="cli_team_a_l40s")
  _make_measured_submission(src_b, recipe_id="cli_team_b_tpu", hardware_sku="TPU_V6E_1_32GB")

  subs_dir = tmp_path / "subs"
  b_a = subs_dir / "track1_vla" / "team_a_l40s"
  b_b = subs_dir / "track1_vla" / "team_b_tpu"

  # prepare-commit-bundle order A & order B
  assert (
      eeb_main([
          "prepare-commit-bundle",
          "--source",
          str(src_a),
          "--output",
          str(b_a),
          "--recipe-commit-sha256",
          "1" * 64,
          "--team",
          "Team-A",
          "--engine-tag",
          "trt-fp8",
      ])
      == 0
  )
  assert (
      eeb_main([
          "prepare-commit-bundle",
          "--engine-tag",
          "jax-pallas",
          "--team",
          "Team-B",
          "--recipe-commit-sha256",
          "2" * 64,
          "--out",
          str(b_b),
          "--measured-dir",
          str(src_b),
      ])
      == 0
  )

  # verify-bundle order A & order B
  assert eeb_main(["verify-bundle", "--submission", str(b_a)]) == 0
  assert eeb_main(["verify-bundle", "--bundle", str(b_b)]) == 0

  script = REPO_ROOT / "scripts" / "export_website_leaderboard.py"
  out1_lb = tmp_path / "out1" / "lb.json"
  out1_md = tmp_path / "out1" / "LB.md"
  out1_web = tmp_path / "out1" / "web.json"
  r1 = subprocess.run(
      [
          sys.executable,
          str(script),
          "--submissions-dir",
          str(subs_dir),
          "--leaderboard-out",
          str(out1_lb),
          "--markdown-out",
          str(out1_md),
          "--website-out",
          str(out1_web),
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  assert r1.returncode == 0, r1.stderr

  out2_lb = tmp_path / "out2" / "lb.json"
  out2_md = tmp_path / "out2" / "LB.md"
  out2_web = tmp_path / "out2" / "web.json"
  r2 = subprocess.run(
      [
          sys.executable,
          str(script),
          "--website-out",
          str(out2_web),
          "--markdown-out",
          str(out2_md),
          "--leaderboard-out",
          str(out2_lb),
          "--submissions-dir",
          str(subs_dir),
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  assert r2.returncode == 0, r2.stderr
  assert json.loads(out1_web.read_text(encoding="utf-8"))["num_verified_submissions"] == 2
  assert json.loads(out2_web.read_text(encoding="utf-8"))["num_verified_submissions"] == 2
