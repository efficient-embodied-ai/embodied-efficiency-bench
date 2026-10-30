"""Regression suite for all 16 First-Time Contestant User Journey seam fixes (P0-1..P2-6).

Covers all 8 Adversarial Self-Review gates:
  - Gate 1 (Three Paths): Crash (.tmp_* orphan recovery), Repeat (no stale recipe.json / episode carryover),
    and Concurrent (multi-thread + multi-process writers under fcntl.flock without in-tree .lock pollution).
  - Gate 2 (State Machine Completeness): CERTIFIED / PENDING / REJECTED / INVALID states with non-empty
    reason enforcement and explicit validation_scope (RECIPE_ONLY_NO_MEASUREMENTS vs FULL_MEASURED_BUNDLE).
  - Gate 3 (Verbatim CLI in Both Flag Orders): eeb measure, eeb verify-receipt (--receipt file/dir,
    --submission dir), eeb validate-submission, eeb compute-pareto, eeb eval-libero-rt --sweep-report,
    eeb data perturb, and eeb data inspect.
  - Gate 4 (Negative Controls & Legitimate Lookalikes):
    * Live Gate G4 catches static memoizer VLA, constant-fallback VLA, and memoizer WM policies while
      accepting deterministic physical-proxy VLA and WM policies.
    * Bundle tamper verifier catches tampered sibling report.json, tampered receipt pareto_frontier,
      tampered non-frontier operating_points, tampered summary_metrics, and orphan report.json without
      receipt.json while accepting untampered bundles and recipe-only folders.
    * Missing power telemetry produces power_efficiency_per_watt=None without crashing compute-pareto /
      leaderboard_compiler / export_to_website, while --assume-power-watts produces ESTIMATED with exact
      joules_per_action == round(1 / power_efficiency_per_watt, 6) consistency.
    * Variance-failed concurrency points are excluded from pareto_frontier while variance-passing points
      stay on pareto_frontier.
    * num_devices=0 emits CPU-only rejection message while num_devices=2 emits multi-GPU rejection message.
    * Relative UDS ./rel.sock normalizes cleanly while >107-byte socket path raises clean actionable error.
    * DummyWMPolicy(horizon=8) works while unknown policy kwargs produce actionable signature error.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterator
from unittest import mock

import numpy as np
import pytest

from eeb.cli import build_parser as build_eeb_parser
from eeb.cli import main as eeb_main
from eeb.data.perturb import PerturbationConfig, verify_perturbation_manifest, write_perturbed_split
from eeb.loadgen.client import InProcessPolicyClient
from eeb.loadgen.gate import load_golden_npz, run_gate, run_live_g4_perturbation_probe
from eeb.loadgen.report import (
    ReportInputs,
    WM_LATENCY_METRIC,
    WM_X_METRIC,
    WM_Y_METRIC,
    build_report,
    report_to_markdown_summary,
    write_report_bundle,
)
from eeb.loadgen.sweep import ConcurrencySweep, SweepConfig
from eeb.protocol.policy import (
    SYNTHETIC_DUMMY_SERVER_IMPL,
    ActionChunk,
    BasePolicy,
    Caps,
    Context,
    Frame,
    Observation,
    instantiate_policy,
)
from eeb.servers.base import MAX_UNIX_SOCKET_PATH_BYTES, normalize_and_validate_uds_path, serve_policy
from eeb.servers.dummy import DummyVLAPolicy, DummyWMPolicy
from embodied_bench.attestation import verify_submission_receipt
from embodied_bench.leaderboard_compiler import compile_submissions_to_leaderboard, render_pr_comment_markdown
from embodied_bench.measure import compute_semianalysis_normalizations
from embodied_bench.quality_gates import evaluate_track1_gate_a, evaluate_track2_gate_a
from embodied_bench.schema import HARDWARE_SKU_SPECS, TrackType, load_yaml_or_json
from embodied_bench.submission_validator import validate_submission_directory
from leaderboard.export_to_website import export_verified_runs_to_website_json

REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_TRACES_DIR = REPO_ROOT / "data" / "embodiedx_traces_v1_mini"


class MemoizingCheatVLAPolicy(BasePolicy):
  """Negative-control Track 1 policy that memoizes the first chunk across all future calls."""

  def __init__(self) -> None:
    self._cached: ActionChunk | None = None

  def capabilities(self) -> Caps:
    return Caps(
        track="VLA_TRACK",
        horizon=8,
        dt_ms=50.0,
        precision="FP32",
        model_id="memo-vla",
        server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
        action_dim=7,
        proprio_dim=8,
        cameras=("agentview", "wrist"),
        image_height=64,
        image_width=64,
    )

  def reset(self, session_id: str) -> None:
    return None

  def predict(self, obs: Observation) -> ActionChunk:
    if self._cached is None:
      base = np.full((8, 7), 0.25, dtype=np.float32)
      self._cached = ActionChunk(actions=base)
    return ActionChunk(actions=self._cached.actions.copy())


class ConstantMissFallbackVLAPolicy(BasePolicy):
  """Negative-control Track 1 policy that recognizes 1 clean signature and returns a constant fallback on every perturbed input."""

  def __init__(self) -> None:
    self._first_sig: bytes | None = None

  def capabilities(self) -> Caps:
    return Caps(
        track="VLA_TRACK",
        horizon=8,
        dt_ms=50.0,
        precision="FP32",
        model_id="const-miss-vla",
        server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
        action_dim=7,
        proprio_dim=8,
        cameras=("agentview", "wrist"),
        image_height=64,
        image_width=64,
    )

  def reset(self, session_id: str) -> None:
    return None

  def predict(self, obs: Observation) -> ActionChunk:
    sig = obs.proprio.tobytes()
    if self._first_sig is None:
      self._first_sig = sig
    if sig == self._first_sig:
      return ActionChunk(actions=np.full((8, 7), 0.1, dtype=np.float32))
    # Constant fallback regardless of perturbation delta -> passes naive (clean, pert_a) only check,
    # caught by secondary (pert_a, pert_b) probe.
    return ActionChunk(actions=np.full((8, 7), 0.9, dtype=np.float32))


class MemoizingCheatWMPolicy(BasePolicy):
  """Negative-control Track 2 world model that streams constant cached frames ignoring proprio/action perturbations."""

  def capabilities(self) -> Caps:
    return Caps(
        track="WM_TRACK",
        horizon=4,
        dt_ms=1000.0 / 24.0,
        modalities=("rgb", "proprio", "language", "actions"),
        precision="FP32",
        model_id="memo-wm",
        server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
        action_dim=7,
        proprio_dim=8,
        image_height=32,
        image_width=32,
    )

  def reset(self, session_id: str) -> None:
    return None

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    n = int(ctx.num_frames) if int(ctx.num_frames) > 0 else 4
    static = np.full((32, 32, 3), 127, dtype=np.uint8)
    for idx in range(n):
      yield Frame(index=idx, image=static.copy())


def _run_quick_measure(
    out_dir: Path,
    *,
    recipe_path: Path | None = None,
    endpoint: str = "inproc:eeb.servers.dummy:DummyVLAPolicy",
    policy_kwargs: str | None = '{"compute_delay_ms": 0.5}',
    assume_power_watts: float | None = None,
) -> int:
  argv = [
      "measure",
      "--endpoint",
      endpoint,
      "--sweep",
      "1,2",
      "--warmup",
      "1",
      "--calls",
      "4",
      "--runs",
      "2",
      "--max-variance-pct",
      "500.0",
      "--no-pacing",
      "--allow-no-device",
      "--out",
      str(out_dir),
      "--quiet",
  ]
  if recipe_path is not None:
    argv.extend(["--recipe", str(recipe_path)])
  else:
    argv.extend(["--sku", "RTX_5090_32GB"])
  if policy_kwargs is not None:
    argv.extend(["--policy-kwargs", policy_kwargs])
  if assume_power_watts is not None:
    argv.extend(["--assume-power-watts", str(assume_power_watts)])
  return eeb_main(argv)


def _promote_bundle_to_single_chip_certified(sub_dir: Path, status: str = "CERTIFIED") -> None:
  """Produce a cleanly re-signed bundle simulating a physical 1-chip run for validator/compiler tests."""
  from embodied_bench.attestation import compute_attestation_sha256, create_submission_receipt

  rep_path = sub_dir / "report.json"
  rec_path = sub_dir / "receipt.json"
  rep = json.loads(rep_path.read_text(encoding="utf-8"))
  rep["status"] = status
  rep["reason"] = "admissible physical single-chip run awaiting organizer Phase-2 check" if status == "PENDING" else None
  rep["num_devices"] = 1
  rep["single_device_compliant"] = True
  rep["admissibility_findings"] = []
  rep.setdefault("execution_provenance", {}).update({
      "policy": "TORCH_CUDA",
      "policy_is_synthetic": False,
      "loadgen": "REAL_WALLCLOCK",
      "observations": "EMBODIEDX_TRACES_V1_BUNDLE",
  })
  rep.setdefault("server_provenance", {})["is_synthetic"] = False
  rep.setdefault("caps", {})["server_impl"] = "pi05-torch-v1"
  rep["observation_source"] = {
      "kind": "REAL_ROBOT_TRACES",
      "pixel_provenance": "REAL_HF_LEROBOT_CAMERA_PIXELS",
      "is_synthetic_smoke": False,
      "certifiable": True,
      "image_hw": [224, 224],
      "bytes_per_request_images": 301056,
      "wire_bytes_per_request_images": 301056,
      "wire_resolution_admissible": True,
  }
  rep.setdefault("sweep_config", {}).update({
      "warmup_calls": 30,
      "measured_calls": 320,
      "num_runs": 3,
  })
  rep.setdefault("measurement_protocol", {})["warmup_calls"] = 30
  qg = rep.setdefault("quality_gate", {})
  qg["passed"] = True
  qg["quality_certified"] = True
  qg["allow_smoke_golden"] = False
  qg["num_samples"] = 100
  if isinstance(qg.get("per_concurrency"), dict):
    for g_ent in qg["per_concurrency"].values():
      if isinstance(g_ent, dict):
        g_ent["allow_smoke_golden"] = False
        g_ent["num_samples"] = 100
  rep.setdefault("summary_metrics", {})["all_variance_passed"] = True
  rep.setdefault("envelope_check", {})["num_devices"] = 1
  rep["envelope_check"]["passed"] = True
  rep.setdefault("performance_cap_evaluation", {})["num_devices"] = 1
  rep["performance_cap_evaluation"]["single_device_compliant"] = True
  rep["performance_cap_evaluation"]["passed"] = True
  for pt in rep.get("operating_points", []):
    pt["gate_a_passed"] = True
    pt["variance_passed"] = True
    pt["slot_starvation_passed"] = True
    pt["max_slot_count_spread"] = 0.0
    pt["c1_tail_ratio_passed"] = True
  for pt in rep.get("pareto_frontier", []):
    pt["gate_a_passed"] = True
  rep_path.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")

  receipt = create_submission_receipt(rep)
  for key in (
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
      "semianalysis_normalizations",
      "execution_provenance",
      "envelope_check",
      "observation_source",
      "sweep_config",
      "measurement_protocol",
      "quality_gate",
      "performance_cap_evaluation",
      "compliance_gates_g1_g4",
      "operating_points",
      "pareto_frontier",
  ):
    if key in rep:
      receipt[key] = copy.deepcopy(rep[key])
  assert receipt["attestation_sha256"] == compute_attestation_sha256(receipt["attested_body"])
  rec_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


# -----------------------------------------------------------------------------
# Gate 1: Crash, Repeat, and Concurrent Writer Paths (+ P0-2a, P0-4, P2-6)
# -----------------------------------------------------------------------------
def test_gate1_report_bundle_crash_repeat_concurrent_and_no_repo_locks(tmp_path: Path) -> None:
  bundle_dir = tmp_path / "bundle_stateful"
  bundle_dir.mkdir(parents=True, exist_ok=True)
  orphan_a = bundle_dir / ".report.json.12345.tmp"
  orphan_b = bundle_dir / ".tmp_embodied_report.json_stale.json"
  orphan_a.write_text("{corrupt", encoding="utf-8")
  orphan_b.write_text("{corrupt2", encoding="utf-8")

  # Call 1 with --recipe writes recipe.json and cleans up crash orphans
  recipe_yaml = REPO_ROOT / "recipes" / "track1_vla_pi05_rtx5090.yaml"
  assert _run_quick_measure(bundle_dir, recipe_path=recipe_yaml) == 0
  assert not orphan_a.exists()
  assert not orphan_b.exists()
  assert (bundle_dir / "recipe.json").is_file()
  assert (bundle_dir / "report.json").is_file()
  assert (bundle_dir / "receipt.json").is_file()
  assert list(bundle_dir.glob("*.lock")) == []
  assert list(bundle_dir.glob(".*.lock")) == []

  # Repeat path: Call 2 WITHOUT --recipe must NOT retain stale recipe.json from Call 1
  assert _run_quick_measure(bundle_dir, recipe_path=None) == 0
  assert not (bundle_dir / "recipe.json").exists(), "Stale recipe.json carried over onto ad-hoc measurement run"
  ok_rep, msg_rep = verify_submission_receipt(bundle_dir)
  assert ok_rep, msg_rep

  # Concurrent threads + subprocesses targeting the same output bundle directory
  report_doc = json.loads((bundle_dir / "report.json").read_text(encoding="utf-8"))

  def _writer(idx: int) -> str:
    clone = copy.deepcopy(report_doc)
    clone["recipe_id"] = f"concurrent_thread_{idx}"
    paths = write_report_bundle(clone, bundle_dir)
    return paths["attestation_sha256"]

  with ThreadPoolExecutor(max_workers=4) as pool:
    digests = list(pool.map(_writer, range(8)))
  assert len(digests) == 8
  ok_thr, msg_thr = verify_submission_receipt(bundle_dir)
  assert ok_thr, msg_thr

  worker_code = (
      "import json, sys\n"
      "from eeb.loadgen.report import write_report_bundle\n"
      "d = sys.argv[1]; tag = sys.argv[2]; src = sys.argv[3]\n"
      "rep = json.loads(open(src, encoding='utf-8').read())\n"
      "rep['recipe_id'] = tag\n"
      "write_report_bundle(rep, d)\n"
  )
  src_copy = tmp_path / "base_report.json"
  src_copy.write_text(json.dumps(report_doc), encoding="utf-8")
  procs = [
      subprocess.Popen(
          [sys.executable, "-c", worker_code, str(bundle_dir), f"proc_{i}", str(src_copy)],
          cwd=str(REPO_ROOT),
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      for i in range(4)
  ]
  for p in procs:
    _out, err = p.communicate(timeout=30)
    assert p.returncode == 0, err
  ok_proc, msg_proc = verify_submission_receipt(bundle_dir)
  assert ok_proc, msg_proc
  assert list(tmp_path.rglob("*.lock")) == []
  assert list(tmp_path.rglob(".*.lock")) == []


def test_gate1_perturb_shard_bundle_crash_repeat_and_concurrent(tmp_path: Path) -> None:
  out_dir = tmp_path / "mini_traces_copy"
  shutil.copytree(MINI_TRACES_DIR, out_dir)
  pert_split = out_dir / "perturbed" / "public"
  pert_split.mkdir(parents=True, exist_ok=True)
  orphan_tmp = pert_split / ".stale_orphan.npz.tmp"
  orphan_tmp.write_bytes(b"partial")

  # Crash recovery + full write across 4 episodes from shard bundle
  m1 = write_perturbed_split(out_dir, out_dir, cfg=PerturbationConfig(seed=101), split="public", max_episodes=4)
  assert m1.is_file()
  assert not orphan_tmp.exists()
  ok1, probs1 = verify_perturbation_manifest(m1)
  assert ok1, probs1
  assert len(list(pert_split.glob("episode_*.npz"))) == 4

  # Repeat call with fewer episodes (max_episodes=2) purges stale episode_000002/000003.npz
  m2 = write_perturbed_split(out_dir, out_dir, cfg=PerturbationConfig(seed=101), split="public", max_episodes=2)
  ok2, probs2 = verify_perturbation_manifest(m2)
  assert ok2, probs2
  eps_after = sorted(p.name for p in pert_split.glob("episode_*.npz"))
  assert eps_after == ["episode_0000.npz", "episode_0001.npz"]
  doc2 = json.loads(m2.read_text(encoding="utf-8"))
  assert doc2["episode_count"] == 2

  # Concurrent thread writers on the same perturbation split leave a 100% consistent manifest + files
  def _perturb_worker(seed_val: int) -> Path:
    return write_perturbed_split(
        out_dir,
        out_dir,
        cfg=PerturbationConfig(seed=seed_val),
        split="public",
        max_episodes=3,
    )

  with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(_perturb_worker, [201, 202, 203, 204]))
  ok_conc, probs_conc = verify_perturbation_manifest(pert_split / "perturbation_manifest.json")
  assert ok_conc, probs_conc
  assert list(out_dir.rglob("*.lock")) == []


# -----------------------------------------------------------------------------
# Gate 2 + P0-2: Submission Validation Scopes, States, and Tamper Resistance
# -----------------------------------------------------------------------------
def test_gate2_and_gate4_submission_validator_scopes_and_tamper_detection(tmp_path: Path) -> None:
  # 1. Recipe-only directory -> passed=True, validation_scope=RECIPE_ONLY_NO_MEASUREMENTS
  recipe_only = tmp_path / "recipe_only_sub"
  recipe_only.mkdir()
  shutil.copy2(REPO_ROOT / "recipes" / "track1_vla_pi05_rtx5090.yaml", recipe_only / "recipe.yaml")
  r_only = validate_submission_directory(recipe_only)
  assert r_only.passed is True
  assert r_only.status == "CERTIFIED"
  assert r_only.reason is None
  assert r_only.has_measurements is False
  assert r_only.validation_scope == "RECIPE_ONLY_NO_MEASUREMENTS"
  assert "RECIPE_ONLY_NO_MEASUREMENTS" in (r_only.receipt_message or "")

  # 2. CPU-only allow-no-device bundle (status=INVALID, num_devices=0):
  #    - verify-receipt succeeds (0) because the bundle is cryptographically untampered
  #    - validate-submission rejects (2) with non-empty reason because status=INVALID / num_devices=0
  cpu_bundle = tmp_path / "cpu_bundle"
  assert _run_quick_measure(cpu_bundle, recipe_path=REPO_ROOT / "recipes" / "track1_vla_pi05_rtx5090.yaml") == 0
  ok_sig, _msg_sig = verify_submission_receipt(cpu_bundle)
  assert ok_sig is True
  r_cpu = validate_submission_directory(cpu_bundle)
  assert r_cpu.passed is False
  assert r_cpu.status == "REJECTED"
  assert r_cpu.reason and "status='INVALID'" in r_cpu.reason
  assert r_cpu.has_measurements is True
  assert r_cpu.validation_scope == "FULL_MEASURED_BUNDLE"

  # 3. Valid single-chip PENDING bundle (contestant hardware submission per ADR-5) AND CERTIFIED bundle -> pass
  for status_val in ("PENDING", "CERTIFIED"):
    good_bundle = tmp_path / f"good_{status_val.lower()}"
    shutil.copytree(cpu_bundle, good_bundle)
    _promote_bundle_to_single_chip_certified(good_bundle, status=status_val)
    r_good = validate_submission_directory(good_bundle)
    assert r_good.passed is True, r_good.violations
    assert r_good.has_measurements is True
    assert r_good.validation_scope == "FULL_MEASURED_BUNDLE"
    assert r_good.receipt_verified is True

  base_good = tmp_path / "good_certified"

  # 4a. Negative control: Tampered sibling report.json without touching receipt.json -> caught!
  tamp_report_dir = tmp_path / "tampered_sibling_report"
  shutil.copytree(base_good, tamp_report_dir)
  rep_doc = json.loads((tamp_report_dir / "report.json").read_text(encoding="utf-8"))
  rep_doc["pareto_frontier"][0]["executed_actions_per_sec"] = 999999.0
  (tamp_report_dir / "report.json").write_text(json.dumps(rep_doc, indent=2), encoding="utf-8")
  ok_tr, msg_tr = verify_submission_receipt(tamp_report_dir)
  assert ok_tr is False
  assert "sibling report.json" in msg_tr
  r_tr = validate_submission_directory(tamp_report_dir)
  assert r_tr.passed is False and r_tr.reason

  # 4b. Negative control: Tampered top-level pareto_frontier in receipt.json without touching attested_body -> caught!
  tamp_top_frontier = tmp_path / "tampered_top_frontier"
  shutil.copytree(base_good, tamp_top_frontier)
  rec_doc = json.loads((tamp_top_frontier / "receipt.json").read_text(encoding="utf-8"))
  rec_doc["pareto_frontier"][0]["executed_actions_per_sec"] = 888888.0
  (tamp_top_frontier / "receipt.json").write_text(json.dumps(rec_doc, indent=2), encoding="utf-8")
  ok_tf, msg_tf = verify_submission_receipt(tamp_top_frontier)
  assert ok_tf is False
  assert "pareto_frontier" in msg_tf
  assert validate_submission_directory(tamp_top_frontier).passed is False

  # 4c. Negative control: Tampered non-frontier operating_points entry inflated above attested frontier -> caught!
  tamp_op_pts = tmp_path / "tampered_operating_points"
  shutil.copytree(base_good, tamp_op_pts)
  rec_op = json.loads((tamp_op_pts / "receipt.json").read_text(encoding="utf-8"))
  rec_op["operating_points"].append({
      "point_id": "forged_C64",
      "execution_id": "forged-exec",
      "concurrency": 64,
      "gate_a_passed": True,
      "hz_react": 999.0,
      "executed_actions_per_sec": 777777.0,
      "actions_per_sec": 777777.0,
      "raw_per_chip_throughput": 777777.0,
      "obs_to_action_p99_ms": 1.0,
      "ttfa_p99_ms": 1.0,
  })
  (tamp_op_pts / "receipt.json").write_text(json.dumps(rec_op, indent=2), encoding="utf-8")
  ok_op, msg_op = verify_submission_receipt(tamp_op_pts)
  assert ok_op is False
  assert "operating_points" in msg_op

  # 4d. Negative control: Tampered top-level summary_metrics.peak_throughput -> caught!
  tamp_sum = tmp_path / "tampered_summary_metrics"
  shutil.copytree(base_good, tamp_sum)
  rec_sm = json.loads((tamp_sum / "receipt.json").read_text(encoding="utf-8"))
  rec_sm["summary_metrics"]["peak_throughput"] = 555555.0
  (tamp_sum / "receipt.json").write_text(json.dumps(rec_sm, indent=2), encoding="utf-8")
  ok_sm, msg_sm = verify_submission_receipt(tamp_sum)
  assert ok_sm is False
  assert "summary_metrics.peak_throughput" in msg_sm

  # 4e. Negative control: Orphan report.json with deleted receipt.json -> caught!
  orphan_rep = tmp_path / "orphan_report_no_receipt"
  shutil.copytree(base_good, orphan_rep)
  (orphan_rep / "receipt.json").unlink()
  r_orph = validate_submission_directory(orphan_rep)
  assert r_orph.passed is False
  assert r_orph.reason and "without a cryptographic 'receipt.json'" in r_orph.reason


# -----------------------------------------------------------------------------
# P0-1, P0-3, P1-1, P1-2, P1-3, P1-4, P1-6, P2-1, P2-5: Schema, Metrics & Normalizations
# -----------------------------------------------------------------------------
def test_wm_and_vla_metric_names_power_provenance_tco_capex_and_variance_exclusion(tmp_path: Path) -> None:
  # 1. Track 2 (WM_TRACK) emits canonical WM metric names and columns without em-dashes
  wm_dir = tmp_path / "wm_no_power"
  assert (
      _run_quick_measure(
          wm_dir,
          recipe_path=REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_rtx5090.yaml",
          endpoint="inproc:eeb.servers.dummy:DummyWMPolicy",
          policy_kwargs='{"frame_delay_ms": 0.05, "horizon": 16, "frame_hw": [32, 32]}',
      )
      == 0
  )
  wm_rep = json.loads((wm_dir / "report.json").read_text(encoding="utf-8"))
  assert wm_rep["x_metric_name"] == WM_X_METRIC == "hz_plan"
  assert wm_rep["y_metric_name"] == WM_Y_METRIC == "predictions_per_sec"
  assert wm_rep["latency_metric_name"] == WM_LATENCY_METRIC == "rollout_latency_p99_ms"
  pt0 = wm_rep["operating_points"][0]
  assert pt0["hz_plan"] > 0.0
  assert pt0["predictions_per_sec"] > 0.0
  assert pt0["generated_frames_per_sec"] > 0.0
  assert pt0["fps"] == pt0["generated_frames_per_sec"]
  assert pt0["rollout_latency_p99_ms"] > 0.0
  assert wm_rep["execution_provenance"]["power"] == "UNAVAILABLE"
  assert pt0["hardware_telemetry"]["power_provenance"] == "UNAVAILABLE"
  assert pt0["power_efficiency_per_watt"] is None
  assert pt0["fps_per_watt"] is None
  assert pt0["joules_per_frame"] is None

  md_wm = report_to_markdown_summary(wm_rep)
  assert "rollout/s" in md_wm and "gen FPS" in md_wm and "per $/hr" in md_wm and "per 100T" in md_wm
  assert "p99 Rollout Rate (Rollouts/s | 16x = Stream FPS)" in wm_rep["x_axis_label"]
  assert "cold_session_ttfa_p50_ms(B=1)=" in md_wm
  assert "\u2014" not in md_wm

  # 2. RT Fix 1: Assumed power produces ESTIMATED and sets power_efficiency_per_watt / joules_per_action to None
  vla_pwr_dir = tmp_path / "vla_est_power"
  assert (
      _run_quick_measure(
          vla_pwr_dir,
          recipe_path=REPO_ROOT / "recipes" / "track1_vla_pi05_rtx5090.yaml",
          assume_power_watts=250.0,
      )
      == 0
  )
  vla_rep = json.loads((vla_pwr_dir / "report.json").read_text(encoding="utf-8"))
  assert vla_rep["execution_provenance"]["power"] == "ESTIMATED"
  for pt in vla_rep["operating_points"]:
    assert pt["hardware_telemetry"]["power_provenance"] == "ESTIMATED"
    assert pt["power_efficiency_per_watt"] is None
    assert pt["joules_per_action"] is None
    # D6: Canonical Y3 is Per $/hr Reference TCO Rate; CapEx Per $1k stays auxiliary and distinct
    assert pt["tco_efficiency_per_dollar"] == pt["throughput_per_dollar_hr"]
    assert pt["tco_efficiency_per_dollar"] != pt["throughput_per_1k_usd"]

  # Every HARDWARE_SKU_SPECS entry defines positive capex_usd and MEASURED power populates Y2
  from eeb.loadgen.report import _normalizations as loadgen_sa_norm

  norm_est = loadgen_sa_norm(
      actions_per_sec=300.0,
      sku="L40S_48GB",
      hw={"system_power_watts": 200.0, "power_provenance": "ESTIMATED"},
  )
  assert norm_est["power_efficiency_per_watt"] is None
  norm_meas = loadgen_sa_norm(
      actions_per_sec=300.0,
      sku="L40S_48GB",
      hw={"system_power_watts": 200.0, "power_provenance": "MEASURED"},
  )
  assert norm_meas["power_efficiency_per_watt"] == pytest.approx(1.5, rel=1e-4)

  for sku_id, spec in HARDWARE_SKU_SPECS.items():
    assert float(spec["capex_usd"]) > 0.0, sku_id
  norm = compute_semianalysis_normalizations(
      raw_throughput=300.0,
      hardware_sku="RTX_5090_32GB",
      device_power_watts=200.0,
      same_chip_bf16_ref_throughput=200.0,
  )
  assert norm["tco_efficiency_per_dollar"] == norm["throughput_per_dollar_hr"]
  assert 315.7 <= norm["tco_efficiency_per_dollar"] <= 315.9  # 300 / $0.95/hr
  assert 150.0 <= norm["throughput_per_1k_usd"] <= 150.2  # 300 / ($1999 / $1000)

  # 3. P1-3: Variance-failed operating points are excluded from pareto_frontier (while variance-passing stay)
  vla_var = copy.deepcopy(vla_rep)
  sweep_policy = DummyVLAPolicy(compute_delay_ms=4.0)
  sweep = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      SweepConfig(concurrency=[1, 2], warmup_calls=1, timed_calls=4, runs=2, max_variance_pct=80.0, max_p99_variance_pct=150.0, pacing_enabled=False, server_warmup_calls=0),
      policy=sweep_policy,
  )
  res = sweep.run()
  res.points[0].variance_passed = True
  res.points[1].variance_passed = False  # Mark C=2 variance-failed even if its gate passed
  gates = {
      p.concurrency: run_gate(p, res.caps, f"var-{p.concurrency}", client=InProcessPolicyClient(sweep_policy))
      for p in res.points
  }
  assert all(g.passed for g in gates.values())
  from eeb.envelope.check import check_envelope

  built = build_report(
      ReportInputs(
          sweep=res,
          recipe=vla_var.get("recipe"),
          envelope=check_envelope(hardware_sku="RTX_5090_32GB", allow_no_device=True),
          gates=gates,
          allow_sanity_only=True,
      )
  )
  assert built["operating_points"][0]["gate_a_passed"] is True
  assert built["operating_points"][1]["variance_passed"] is False
  assert built["operating_points"][1]["gate_a_passed"] is True
  assert [p["concurrency"] for p in built["pareto_frontier"]] == [1]

  # 4. P1-4: Track 1 Gate A = VLA actions, Track 2 Gate A = WM frames (with legacy auto-dispatch safety)
  acts_ref = np.ones((2, 8, 7), dtype=np.float32)
  frames_ref = np.full((2, 4, 16, 16, 3), 120.0, dtype=np.float32)
  r_t1 = evaluate_track1_gate_a("e1", "e1", ref_actions=acts_ref, cand_actions=acts_ref * 1.0001)
  r_t2 = evaluate_track2_gate_a("e2", "e2", ref_frames=frames_ref, cand_frames=frames_ref)
  assert r_t1.passed is True and r_t1.track == TrackType.VLA_TRACK.value
  assert r_t2.passed is True and r_t2.track == TrackType.WM_TRACK.value

  # 5. P0-3 + P2-5: compute-pareto, compile_submissions_to_leaderboard, and export_to_website handle None power
  #    AND emit accurate CPU-only (num_devices=0) vs multi-GPU (num_devices=2) messages
  subs_root = tmp_path / "subs_mix"
  wm_cert = subs_root / "wm_cert_nopower"
  vla_cert = subs_root / "vla_cert_estpower"
  cpu_rej = subs_root / "cpu_zero_dev"
  multi_rej = subs_root / "multi_two_dev"
  shutil.copytree(wm_dir, wm_cert)
  shutil.copytree(vla_pwr_dir, vla_cert)
  shutil.copytree(wm_dir, cpu_rej)
  shutil.copytree(wm_dir, multi_rej)
  _promote_bundle_to_single_chip_certified(wm_cert)
  _promote_bundle_to_single_chip_certified(vla_cert)

  multi_rec = json.loads((multi_rej / "receipt.json").read_text(encoding="utf-8"))
  multi_rec["status"] = "CERTIFIED"
  multi_rec["reason"] = None
  multi_rec["num_devices"] = 2
  (multi_rej / "receipt.json").write_text(json.dumps(multi_rec, indent=2), encoding="utf-8")

  lb_json = tmp_path / "compiled_lb.json"
  lb = compile_submissions_to_leaderboard(subs_root, lb_json, strict_verify=False)
  assert lb["num_verified_submissions"] == 2
  assert lb["num_rejected_submissions"] == 2
  rej_by_id = {Path(item["receipt_path"]).parent.name: item["reason"] for item in lb["rejected_submissions"]}
  assert "CPU-only / allow-no-device self-test run (num_devices=0)" in rej_by_id["cpu_zero_dev"]
  assert "num_devices=1 != 1" not in rej_by_id["cpu_zero_dev"]
  assert "num_devices=2 != 1" in rej_by_id["multi_two_dev"]

  wm_entry = next(s for s in lb["submissions"] if s["track"] == "WM_TRACK")
  assert wm_entry["power_efficiency_per_watt"] is None
  md_pr = render_pr_comment_markdown(lb)
  assert "Perf/$/hr" in md_pr and "(Output/s)/($/hr)" in md_pr and "N/A" in md_pr

  web_json = tmp_path / "website_export.json"
  web = export_verified_runs_to_website_json(lb_json, web_json)
  assert web["num_verified_submissions"] == 2


# -----------------------------------------------------------------------------
# Gate 4 + P1-5: Live Same-Execution Gate G4 Anti-Memoization Audit
# -----------------------------------------------------------------------------
def test_gate4_live_g4_probe_catches_memoizers_and_accepts_legitimate_policies(tmp_path: Path) -> None:
  # Legitimate deterministic VLA policy passes live G4
  good_vla = DummyVLAPolicy(compute_delay_ms=0.0)
  g4_good_vla = run_live_g4_perturbation_probe(
      InProcessPolicyClient(good_vla), good_vla.capabilities(), "exec-good-vla"
  )
  assert g4_good_vla["passed"] is True
  assert g4_good_vla["cache_hit_suspected"] is False

  # Legitimate deterministic WM policy passes live G4
  good_wm = DummyWMPolicy(frame_delay_ms=0.0, num_frames=4, frame_hw=(32, 32))
  g4_good_wm = run_live_g4_perturbation_probe(
      InProcessPolicyClient(good_wm), good_wm.capabilities(), "exec-good-wm"
  )
  assert g4_good_wm["passed"] is True

  # Negative control 1: Static memoizer VLA policy fails live G4 AND fails `eeb measure` quality_gate
  bad_memo = MemoizingCheatVLAPolicy()
  g4_bad_memo = run_live_g4_perturbation_probe(
      InProcessPolicyClient(bad_memo), bad_memo.capabilities(), "exec-bad-memo"
  )
  assert g4_bad_memo["passed"] is False
  assert g4_bad_memo["cache_hit_suspected"] is True

  memo_out = tmp_path / "memo_cli_run"
  rc_memo = eeb_main([
      "measure",
      "--endpoint",
      "inproc:tests.test_user_journey_seam_fixes:MemoizingCheatVLAPolicy",
      "--sweep",
      "1",
      "--warmup",
      "1",
      "--calls",
      "2",
      "--runs",
      "1",
      "--no-pacing",
      "--allow-no-device",
      "--allow-sanity-only",
      "--strict",
      "--sku",
      "RTX_5090_32GB",
      "--out",
      str(memo_out),
      "--quiet",
  ])
  assert rc_memo == 2
  memo_rep = json.loads((memo_out / "report.json").read_text(encoding="utf-8"))
  assert memo_rep["quality_gate"]["passed"] is False
  assert "G4 anti-memoization failed" in (memo_rep["reason"] or "")

  # Negative control 2: Constant-fallback cache-miss VLA policy caught by secondary (pert_a, pert_b) probe
  bad_const = ConstantMissFallbackVLAPolicy()
  g4_bad_const = run_live_g4_perturbation_probe(
      InProcessPolicyClient(bad_const), bad_const.capabilities(), "exec-bad-const"
  )
  assert g4_bad_const["passed"] is False

  # Negative control 3: Static memoizer WM policy fails live G4
  bad_wm = MemoizingCheatWMPolicy()
  g4_bad_wm = run_live_g4_perturbation_probe(
      InProcessPolicyClient(bad_wm), bad_wm.capabilities(), "exec-bad-wm"
  )
  assert g4_bad_wm["passed"] is False


# -----------------------------------------------------------------------------
# Gate 3 + P0-2e, P0-4, P1-4, P1-5, P2-2, P2-3, P2-4: CLI Subcommands in Both Flag Orders
# -----------------------------------------------------------------------------
def test_gate3_cli_both_flag_orders_help_uds_and_policy_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  # P2-2: `eeb --help` surfaces both core measurement and submission/leaderboard commands
  help_text = build_eeb_parser().format_help()
  for expected_cmd in (
      "serve",
      "measure",
      "telemetry",
      "envelope",
      "data",
      "verify-receipt",
      "prepare-commit-bundle",
      "verify-bundle",
      "validate-submission",
      "compute-pareto",
      "build-leaderboard",
  ):
    assert expected_cmd in help_text

  # P2-3: DummyWMPolicy accepts horizon alias and bad kwargs raise clean error via eeb CLI (exit 1)
  wm_h = instantiate_policy("eeb.servers.dummy:DummyWMPolicy", {"horizon": 6, "frame_delay_ms": 0.1})
  assert wm_h.capabilities().horizon == 6
  with pytest.raises(TypeError, match="Accepted constructor parameters"):
    instantiate_policy("eeb.servers.dummy:DummyWMPolicy", {"nonexistent_kwarg": 123})
  with mock.patch("sys.stderr", new_callable=io.StringIO) as err_buf:
    rc_bad_kw = eeb_main([
        "measure",
        "--endpoint",
        "inproc:eeb.servers.dummy:DummyWMPolicy",
        "--policy-kwargs",
        '{"nonexistent_kwarg": 1}',
        "--out",
        str(tmp_path / "bad_kw"),
    ])
  assert rc_bad_kw == 1
  assert "Accepted constructor parameters" in err_buf.getvalue()

  # P2-4: Relative UDS path auto-normalized & bound cleanly; >107-byte path rejected cleanly with exit 1
  short_cwd = Path("/tmp") / f"eeb_uds_test_{os.getpid()}"
  short_cwd.mkdir(parents=True, exist_ok=True)
  try:
    monkeypatch.chdir(short_cwd)
    resolved = normalize_and_validate_uds_path("./rel_srv.sock")
    assert Path(resolved).is_absolute()
    srv = serve_policy(DummyVLAPolicy(compute_delay_ms=0.1), "unix://./rel_srv.sock")
    srv.stop(grace=0.5)
  finally:
    shutil.rmtree(short_cwd, ignore_errors=True)

  overlong_sock = "/tmp/" + ("x" * (MAX_UNIX_SOCKET_PATH_BYTES + 10)) + ".sock"
  with pytest.raises(ValueError, match="AF_UNIX sun_path kernel limit"):
    normalize_and_validate_uds_path(overlong_sock)
  with mock.patch("sys.stderr", new_callable=io.StringIO) as uds_err:
    assert eeb_main(["serve", "--policy", "eeb.servers.dummy:DummyVLAPolicy", "--uds", overlong_sock]) == 1
  assert "107-byte" in uds_err.getvalue()

  # Prepare certified VLA + WM bundles for both-flag-order CLI checks
  vla_dir = tmp_path / "cli_vla"
  wm_dir = tmp_path / "cli_wm"
  assert _run_quick_measure(vla_dir, recipe_path=REPO_ROOT / "recipes" / "track1_vla_pi05_rtx5090.yaml") == 0
  assert (
      _run_quick_measure(
          wm_dir,
          recipe_path=REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_rtx5090.yaml",
          endpoint="inproc:eeb.servers.dummy:DummyWMPolicy",
          policy_kwargs='{"frame_delay_ms": 0.05, "horizon": 16, "frame_hw": [32, 32]}',
      )
      == 0
  )
  _promote_bundle_to_single_chip_certified(vla_dir)
  _promote_bundle_to_single_chip_certified(wm_dir)

  # 1. verify-receipt in both flag forms (--receipt file, --receipt dir, --submission dir)
  assert eeb_main(["verify-receipt", "--receipt", str(vla_dir / "receipt.json")]) == 0
  assert eeb_main(["verify-receipt", "--receipt", str(vla_dir)]) == 0
  assert eeb_main(["verify-receipt", "--submission", str(wm_dir)]) == 0

  # 2. validate-submission in both flag orders
  v_out1 = tmp_path / "val1.json"
  v_out2 = tmp_path / "val2.json"
  assert eeb_main(["validate-submission", "--submission", str(vla_dir), "--output", str(v_out1)]) == 0
  assert eeb_main(["validate-submission", "--output", str(v_out2), "--submission", str(wm_dir)]) == 0
  assert json.loads(v_out1.read_text(encoding="utf-8"))["validation_scope"] == "FULL_MEASURED_BUNDLE"
  assert json.loads(v_out2.read_text(encoding="utf-8"))["receipt_verified"] is True

  # 3. compute-pareto in both flag orders + Track 1 = VLA / Track 2 = WM counter parity
  p_out1 = tmp_path / "pareto1.json"
  p_out2 = tmp_path / "pareto2.json"
  with mock.patch("sys.stdout", new_callable=io.StringIO) as p_buf1:
    assert eeb_main(["compute-pareto", "--inputs", str(vla_dir / "report.json"), str(wm_dir / "report.json"), "--output", str(p_out1)]) == 0
  p_sum1 = json.loads(p_buf1.getvalue())
  assert p_sum1["track1_global_frontier_points"] == p_sum1["vla_track_global_frontier_points"] >= 1
  assert p_sum1["track2_global_frontier_points"] == p_sum1["wm_track_global_frontier_points"] >= 1

  with mock.patch("sys.stdout", new_callable=io.StringIO) as p_buf2:
    assert eeb_main(["compute-pareto", "--output", str(p_out2), "--inputs", str(wm_dir / "report.json"), str(vla_dir / "report.json")]) == 0
  p_sum2 = json.loads(p_buf2.getvalue())
  assert p_sum2["num_certified_submissions"] == 2

  # 4. eval-libero-rt --sweep-report in both flag orders
  lrt1 = tmp_path / "lrt1.json"
  lrt2 = tmp_path / "lrt2.json"
  recipe_vla = REPO_ROOT / "recipes" / "track1_vla_pi05_rtx5090.yaml"
  assert eeb_main(["eval-libero-rt", "--recipe", str(recipe_vla), "--sweep-report", str(vla_dir / "report.json"), "--output", str(lrt1)]) in (0, 2)
  assert eeb_main(["eval-libero-rt", "--output", str(lrt2), "--sweep-report", str(vla_dir / "report.json"), "--recipe", str(recipe_vla)]) in (0, 2)
  lrt_doc = json.loads(lrt1.read_text(encoding="utf-8"))
  assert lrt_doc["latency_provenance"] == "EMPIRICAL_SWEEP_REPORT"
  assert lrt_doc["evaluation_mode"] == "ANALYTIC_LATENCY_COUPLED_MODEL"

  # 5. eeb data perturb + eeb data inspect on shard bundle layout in both flag orders (plus --split hidden negative control)
  traces_copy = tmp_path / "traces_cli"
  shutil.copytree(MINI_TRACES_DIR, traces_copy)
  assert eeb_main(["data", "perturb", str(traces_copy), "--split", "public", "--max-episodes", "2", "--quiet"]) == 0
  assert eeb_main(["data", "perturb", "--quiet", "--out", str(tmp_path / "pert_explicit"), "--max-episodes", "2", "--split", "public", str(traces_copy)]) == 0
  assert eeb_main(["data", "perturb", str(traces_copy), "--split", "hidden", "--quiet"]) == 2
  with mock.patch("sys.stdout", new_callable=io.StringIO) as insp_buf:
    assert eeb_main(["data", "inspect", str(traces_copy / "perturbed" / "public")]) == 0
  assert "perturbation manifest OK" in insp_buf.getvalue()
  with mock.patch("sys.stdout", new_callable=io.StringIO) as insp_bundle_buf:
    assert eeb_main(["data", "inspect", str(traces_copy)]) == 0
  assert "manifest verification OK: 8 file(s) checked" in insp_bundle_buf.getvalue()

  # 6. eeb export-website works outside repo root in both flag orders
  web_out1 = tmp_path / "web_exp1.json"
  web_out2 = tmp_path / "web_exp2.json"
  assert eeb_main(["export-website", "--inputs", str(vla_dir / "receipt.json"), str(wm_dir / "receipt.json"), "--output", str(web_out1)]) == 0
  assert eeb_main(["export-website", "--output", str(web_out2), "--inputs", str(wm_dir / "receipt.json"), str(vla_dir / "receipt.json")]) == 0
  assert json.loads(web_out1.read_text(encoding="utf-8"))["num_verified_submissions"] >= 1
  assert json.loads(web_out2.read_text(encoding="utf-8"))["num_verified_submissions"] >= 1


def test_track2_wm_recipes_use_cross_frame_kv_cache_and_zero_repo_lock_files() -> None:
  wm_recipes = sorted((REPO_ROOT / "recipes").glob("track2_wm_cosmos_predict25_*.yaml"))
  assert len(wm_recipes) == 13
  for rp in wm_recipes:
    doc = load_yaml_or_json(rp)
    opts = doc.get("optimizations", [])
    assert "vlm_prefix_kv_cache_reuse" not in opts, f"{rp.name} still lists vlm_prefix_kv_cache_reuse"
    assert "causal_kv_cache_rolling_window" not in opts, f"{rp.name} still lists causal_kv_cache_rolling_window on bidirectional 3D-DiT"
    assert "cross_frame_kv_cache_reuse" in opts, f"{rp.name} missing cross_frame_kv_cache_reuse"

  lock_files = [
      p
      for p in REPO_ROOT.rglob("*.lock")
      if ".git" not in p.parts and ".venv" not in p.parts
  ]
  assert lock_files == [], f"Unexpected in-tree .lock files found: {lock_files}"


class BlindProprioOnlyVLAPolicy(BasePolicy):
  """Negative-control Track 1 policy that uses proprioception but ignores camera images."""

  def capabilities(self) -> Caps:
    return Caps(
        track="VLA_TRACK",
        horizon=8,
        execute_horizon=8,
        dt_ms=50.0,
        precision="BF16",
        model_id="gr00t-n1.7-3b",
        server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
        action_dim=7,
        proprio_dim=8,
        cameras=("agentview", "wrist"),
        image_height=64,
        image_width=64,
    )

  def reset(self, session_id: str) -> None:
    return None

  def predict(self, obs: Observation) -> ActionChunk:
    # Depends heavily on proprio (would have bypassed old proprio-perturbing G4!) but ignores obs.images
    p_sum = float(np.sum(np.asarray(obs.proprio, dtype=np.float32)))
    out = np.full((8, 7), np.tanh(p_sum * 10.0), dtype=np.float32)
    return ActionChunk(actions=out)


class BlindActionsOnlyWMPolicy(BasePolicy):
  """Negative-control Track 2 policy that uses actions/proprio but ignores context video frames."""

  def capabilities(self) -> Caps:
    return Caps(
        track="WM_TRACK",
        horizon=4,
        execute_horizon=4,
        dt_ms=41.67,
        precision="BF16",
        model_id="cosmos-predict2.5-2b",
        server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
        action_dim=7,
        proprio_dim=8,
        cameras=("agentview",),
        image_height=32,
        image_width=32,
    )

  def reset(self, session_id: str) -> None:
    return None

  def predict(self, obs: Observation) -> ActionChunk:
    raise NotImplementedError

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    sig = float(np.sum(np.asarray(actions, dtype=np.float32)) + np.sum(np.asarray(ctx.proprio, dtype=np.float32)))
    val = int(round((np.tanh(sig) * 0.5 + 0.5) * 255.0)) % 256
    nf = int(ctx.num_frames or 4)
    for idx in range(nf):
      yield Frame(index=idx, image=np.full((32, 32, 3), (val + idx) % 256, dtype=np.uint8))


def test_prong1_and_prong2_remediations_g4_vision_wm_ttff_pareto_ties_pacing_batcher_and_cli(
    tmp_path: Path,
) -> None:
  """Comprehensive regression suite covering Prong 1/2 fixes across Gate 1, Gate 3, and Gate 4."""
  import threading
  import time
  from eeb.loadgen.compare import compare_reports
  from eeb.servers.base import DynamicMicroBatcher
  from embodied_bench.pareto import (
      extract_pareto_frontier,
      extract_semianalysis_4way_pareto_frontiers,
  )

  # 1. Gate 4 Live G4 Vision-Only Perturbation: proprio/action-only blind policies REJECTED,
  #    while vision-responsive DummyVLAPolicy / DummyWMPolicy PASS.
  blind_vla = BlindProprioOnlyVLAPolicy()
  g4_blind_vla = run_live_g4_perturbation_probe(
      InProcessPolicyClient(blind_vla), blind_vla.capabilities(), "exec-blind-vla"
  )
  assert g4_blind_vla["passed"] is False
  assert g4_blind_vla["cache_hit_suspected"] is True

  blind_wm = BlindActionsOnlyWMPolicy()
  g4_blind_wm = run_live_g4_perturbation_probe(
      InProcessPolicyClient(blind_wm), blind_wm.capabilities(), "exec-blind-wm"
  )
  assert g4_blind_wm["passed"] is False
  assert g4_blind_wm["cache_hit_suspected"] is True

  # Confirm run_gate rejects a blind VLA policy even when sweep metrics look healthy
  blind_sweep = ConcurrencySweep(
      "inproc:tests.test_user_journey_seam_fixes:BlindProprioOnlyVLAPolicy",
      SweepConfig(concurrency=[1], warmup_calls=1, timed_calls=2, runs=1, server_warmup_calls=0),
      policy=blind_vla,
  ).run()
  g_res_blind = run_gate(
      blind_sweep.points[0],
      blind_sweep.caps,
      "exec-blind-gate",
      client=InProcessPolicyClient(blind_vla),
  )
  assert g_res_blind.passed is False
  assert any("G4" in f for f in g_res_blind.violations)

  # 2. Gate 4 Track 2 Full-Clip 3D-DiT Honest TTFF == L_clip Diagnostic (620 ms ACCEPTED)
  #    + CLOSED stream_chunk_frames > 0 Rejection (REJECT_CLOSED_STREAM_CHUNKING)
  wm_pol = DummyWMPolicy(frame_delay_ms=0.05, num_frames=4, frame_hw=(32, 32))
  wm_sweep = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyWMPolicy",
      SweepConfig(concurrency=[1, 4], warmup_calls=1, timed_calls=3, runs=1, server_warmup_calls=0),
      policy=wm_pol,
  ).run()
  pt_b1_good = copy.deepcopy(wm_sweep.points[0])
  pt_b1_good.median_ttff_ms["p99"] = 85.0
  assert run_gate(pt_b1_good, wm_sweep.caps, "wm-b1-good", client=InProcessPolicyClient(wm_pol)).passed is True

  pt_b1_fullclip = copy.deepcopy(wm_sweep.points[0])
  pt_b1_fullclip.median_ttff_ms["p99"] = 620.0
  gate_b1_fullclip = run_gate(pt_b1_fullclip, wm_sweep.caps, "wm-b1-fullclip", client=InProcessPolicyClient(wm_pol))
  assert gate_b1_fullclip.passed is True
  assert gate_b1_fullclip.checks["b1_ttff_p99_ms_diagnostic"] == 620.0

  # Legitimate lookalike: B=4 also passes with high TTFF_p99 diagnostic
  pt_b4_lookalike = copy.deepcopy(wm_sweep.points[1])
  pt_b4_lookalike.median_ttff_ms["p99"] = 185.0
  assert run_gate(pt_b4_lookalike, wm_sweep.caps, "wm-b4-ok", client=InProcessPolicyClient(wm_pol)).passed is True

  # Also verify `compare_reports` accepts honest B=1 TTFF_p99 = 620 ms while rejecting CLOSED stream_chunk_frames=4
  wm_base_dir = tmp_path / "wm_cmp_base"
  assert (
      _run_quick_measure(
          wm_base_dir,
          recipe_path=REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_rtx5090.yaml",
          endpoint="inproc:eeb.servers.dummy:DummyWMPolicy",
          policy_kwargs='{"frame_delay_ms": 0.05, "horizon": 16, "frame_hw": [32, 32]}',
      )
      == 0
  )
  wm_base_rep = json.loads((wm_base_dir / "report.json").read_text(encoding="utf-8"))
  wm_cand_bad_chunk = copy.deepcopy(wm_base_rep)
  wm_cand_bad_chunk["recipe"]["stream_chunk_frames"] = 4
  wm_cand_bad_path = tmp_path / "wm_cand_bad_chunk.json"
  wm_cand_bad_path.write_text(json.dumps(wm_cand_bad_chunk), encoding="utf-8")
  cmp_bad_chunk = compare_reports(wm_base_dir, wm_cand_bad_path, allow_sanity_only=True)
  assert cmp_bad_chunk["verdict"] == "REJECT"
  assert cmp_bad_chunk["decision_code"] == "REJECT_CLOSED_STREAM_CHUNKING"
  assert cmp_bad_chunk["reason"]

  # 3. Pareto Co-SOTA Tie Preservation across NVIDIA and GOOGLE_TPU (+ strictly dominated exclusion)
  tie_nvd = {
      "submission_id": "sub_nvd",
      "recipe_id": "sub_nvd",
      "hardware_sku": "RTX_5090_32GB",
      "vendor": "NVIDIA",
      "hz_react": 25.0,
      "executed_actions_per_sec": 200.0,
      "gate_a_passed": True,
  }
  tie_tpu = {
      "submission_id": "sub_tpu",
      "recipe_id": "sub_tpu",
      "hardware_sku": "TPU_V6E_1_32GB",
      "vendor": "GOOGLE_TPU",
      "hz_react": 25.0,
      "executed_actions_per_sec": 200.0,
      "gate_a_passed": True,
  }
  strict_dom = {
      "submission_id": "sub_dom",
      "recipe_id": "sub_dom",
      "hardware_sku": "RTX_4090_24GB",
      "vendor": "NVIDIA",
      "hz_react": 20.0,
      "executed_actions_per_sec": 150.0,
      "gate_a_passed": True,
  }
  frontier = extract_pareto_frontier(
      [tie_nvd, tie_tpu, strict_dom],
      x_key="hz_react",
      y_key="executed_actions_per_sec",
  )
  assert {p["submission_id"] for p in frontier} == {"sub_nvd", "sub_tpu"}
  f4_ties = extract_semianalysis_4way_pareto_frontiers(
      [
          {
              **tie_nvd,
              "raw_per_chip_throughput": 200.0,
              "power_efficiency_per_watt": 1.0,
              "throughput_per_100_tops": 95.0,
              "tco_efficiency_per_dollar": 210.0,
          },
          {
              **tie_tpu,
              "raw_per_chip_throughput": 200.0,
              "power_efficiency_per_watt": 1.0,
              "throughput_per_100_tops": 95.0,
              "tco_efficiency_per_dollar": 210.0,
          },
          {
              **strict_dom,
              "raw_per_chip_throughput": 150.0,
              "power_efficiency_per_watt": 0.5,
              "throughput_per_100_tops": 40.0,
              "tco_efficiency_per_dollar": 100.0,
          },
      ],
      x_key="hz_react",
  )
  for y_mode, pts in f4_ties.items():
    assert {p["submission_id"] for p in pts} == {"sub_nvd", "sub_tpu"}, y_mode

  # 4. Unpaced Default (`SweepConfig().pacing_enabled is False`) vs `--paced` Opt-In Separation
  assert SweepConfig().pacing_enabled is False
  unpaced_dir = tmp_path / "meas_default_unpaced"
  paced_dir = tmp_path / "meas_optin_paced"
  assert (
      eeb_main([
          "measure",
          "--endpoint",
          "inproc:eeb.servers.dummy:DummyVLAPolicy",
          "--policy-kwargs",
          '{"compute_delay_ms": 0.5}',
          "--sweep",
          "1",
          "--warmup",
          "1",
          "--timed",
          "3",
          "--runs",
          "1",
          "--allow-no-device",
          "--allow-sanity-only",
          "--sku",
          "RTX_5090_32GB",
          "--out",
          str(unpaced_dir),
          "--quiet",
      ])
      == 0
  )
  unpaced_rep = json.loads((unpaced_dir / "report.json").read_text(encoding="utf-8"))
  assert unpaced_rep["sweep_config"]["pacing_enabled"] is False
  assert unpaced_rep["operating_points"][0]["pacing_enabled"] is False

  assert (
      eeb_main([
          "measure",
          "--quiet",
          "--out",
          str(paced_dir),
          "--sku",
          "RTX_5090_32GB",
          "--allow-sanity-only",
          "--allow-no-device",
          "--paced",
          "--runs",
          "1",
          "--timed",
          "2",
          "--warmup",
          "1",
          "--sweep",
          "1",
          "--policy-kwargs",
          '{"compute_delay_ms": 1.0, "dt_ms": 15.0, "horizon": 2, "execute_horizon": 2}',
          "--endpoint",
          "inproc:eeb.servers.dummy:DummyVLAPolicy",
      ])
      == 0
  )
  paced_rep = json.loads((paced_dir / "report.json").read_text(encoding="utf-8"))
  assert paced_rep["sweep_config"]["pacing_enabled"] is True
  pt_paced = paced_rep["operating_points"][0]
  assert pt_paced["pacing_enabled"] is True
  # Under P0 Fix #3, paced mode reports true steady-state wall-clock throughput on Y (no Little's Law inflation)
  assert pt_paced["chunks_per_sec"] == pt_paced["paced_wall_chunks_per_sec"] > 0.0
  assert pt_paced["executed_actions_per_sec"] == pt_paced["paced_wall_executed_actions_per_sec"] > 0.0
  assert pt_paced["active_service_chunks_per_sec"] > pt_paced["chunks_per_sec"] > 0.0
  assert pt_paced["active_service_executed_actions_per_sec"] > pt_paced["executed_actions_per_sec"] > 0.0

  # 5. Gate 1 DynamicMicroBatcher Three Paths: Crash, Repeat, and Concurrent Coalescing
  observed_batch_sizes: list[int] = []
  fail_flag = {"boom": False}

  def _batch_fn(reqs: list[int]) -> list[tuple[int, int]]:
    observed_batch_sizes.append(len(reqs))
    time.sleep(0.012)  # Encourage concurrent thread coalescing inside queue window
    if fail_flag["boom"]:
      raise RuntimeError("injected batch kernel crash")
    return [(x * 3 + 1, len(reqs)) for x in reqs]

  batcher = DynamicMicroBatcher(
      _batch_fn,
      max_batch_size=8,
      batch_window_ms=15.0,
  )
  # (a) Crash path: exception inside _batch_fn propagates to all coalesced callers with zero hangs
  fail_flag["boom"] = True
  barrier_crash = threading.Barrier(4)

  def _crash_caller(v: int) -> str:
    barrier_crash.wait(timeout=2.0)
    try:
      batcher.submit(v)
      return "unexpected_ok"
    except RuntimeError as exc:
      return str(exc)

  with ThreadPoolExecutor(max_workers=4) as pool:
    crash_msgs = list(pool.map(_crash_caller, [1, 2, 3, 4]))
  assert all("injected batch kernel crash" in m for m in crash_msgs)

  # (b) Repeat path after crash recovery: leader-follower state cleanly resets `_gpu_busy = False`
  fail_flag["boom"] = False
  assert batcher.submit(5)[0] == 16
  assert batcher.submit(7)[0] == 22

  # (c) Concurrent coalescing path across 6 barrier-synchronized callers
  observed_batch_sizes.clear()
  barrier_conc = threading.Barrier(6)

  def _conc_caller(v: int) -> tuple[int, int]:
    barrier_conc.wait(timeout=2.0)
    return batcher.submit(v)

  inputs = [10, 20, 30, 40, 50, 60]
  with ThreadPoolExecutor(max_workers=6) as pool:
    outs = list(pool.map(_conc_caller, inputs))
  for inp, (val, bsz) in zip(inputs, outs):
    assert val == inp * 3 + 1
    assert bsz >= 1
  assert batcher.max_observed_batch_size >= 2
  assert max(observed_batch_sizes) >= 2

  # 6. Gate 3 Verbatim Unified `eeb` CLI Subcommands in Both Flag Orders
  # (a) `eeb agent-kickoff` in both flag orders
  ko1 = tmp_path / "ko1"
  ko2 = tmp_path / "ko2"
  assert eeb_main(["agent-kickoff", "--prompt", "Start pi0.5 on RTX 5090 in CLOSED division", "--study-dir", str(ko1)]) == 0
  assert eeb_main(["agent-kickoff", "--study-dir", str(ko2), "--prompt", "Start Cosmos-Predict2.5 on TPU v6e-1"]) == 0
  assert (ko1 / "STUDY_PLAN.json").exists() and (ko2 / "STUDY_PLAN.json").exists()

  # (b) `eeb data generate-golden` with positional traces_dir vs `--traces` in both flag orders
  g_pos1 = tmp_path / "g_pos1.npz"
  g_pos2 = tmp_path / "g_pos2.npz"
  assert (
      eeb_main([
          "data",
          "generate-golden",
          str(MINI_TRACES_DIR),
          "--endpoint",
          "inproc:eeb.servers.dummy:DummyVLAPolicy",
          "--out",
          str(g_pos1),
          "--num-samples",
          "2",
          "--allow-smoke-golden",
          "--split",
          "public",
          "--quiet",
      ])
      == 0
  )
  assert (
      eeb_main([
          "data",
          "generate-golden",
          "--quiet",
          "--split",
          "public",
          "--allow-smoke-golden",
          "--num-samples",
          "2",
          "--out",
          str(g_pos2),
          "--endpoint",
          "inproc:eeb.servers.dummy:DummyVLAPolicy",
          str(MINI_TRACES_DIR),
      ])
      == 0
  )
  assert g_pos1.exists() and g_pos2.exists()

  # (c) `eeb eval-wm-ranking`, `eeb run-stage1`, `eeb run-stage2`, `eeb benchmark`, `eeb download-data`, and `eeb pareto` alias
  wm_recipe = REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_rtx5090.yaml"
  vla_recipe = REPO_ROOT / "recipes" / "track1_vla_pi05_rtx5090.yaml"
  wmr1, wmr2 = tmp_path / "wmr1.json", tmp_path / "wmr2.json"
  assert eeb_main(["eval-wm-ranking", "--recipe", str(wm_recipe), "--output", str(wmr1)]) in (0, 2)
  assert eeb_main(["eval-wm-ranking", "--output", str(wmr2), "--recipe", str(wm_recipe)]) in (0, 2)

  s1a, s1b = tmp_path / "s1a.json", tmp_path / "s1b.json"
  assert eeb_main(["run-stage1", "--recipe", str(vla_recipe), "--skip-stage2", "--output", str(s1a)]) in (0, 2)
  assert eeb_main(["run-stage1", "--output", str(s1b), "--skip-stage2", "--recipe", str(vla_recipe)]) in (0, 2)

  s2a, s2b = tmp_path / "s2a.json", tmp_path / "s2b.json"
  assert eeb_main(["run-stage2", "--receipt", str(s1a), "--output", str(s2a)]) in (0, 2)
  assert eeb_main(["run-stage2", "--output", str(s2b), "--receipt", str(s1b)]) in (0, 2)

  bma, bmb = tmp_path / "bma.json", tmp_path / "bmb.json"
  assert (
      eeb_main([
          "benchmark",
          "--model",
          "pi05-3b",
          "--weights-mode",
          "mini_real",
          "--concurrency",
          "1",
          "--warmup-steps",
          "1",
          "--steps-per-stream",
          "2",
          "--output",
          str(bma),
      ])
      in (0, 2)
  )
  assert (
      eeb_main([
          "benchmark",
          "--output",
          str(bmb),
          "--steps-per-stream",
          "2",
          "--warmup-steps",
          "1",
          "--concurrency",
          "1",
          "--weights-mode",
          "mini_real",
          "--model",
          "cosmos-predict2.5-2b",
      ])
      in (0, 2)
  )

  dda, ddb = tmp_path / "dda", tmp_path / "ddb"
  assert (
      eeb_main([
          "download-data",
          "--mode",
          "smoke",
          "--num-episodes",
          "2",
          "--steps-per-episode",
          "2",
          "--cache-dir",
          str(dda),
      ])
      == 0
  )
  assert (
      eeb_main([
          "download-data",
          "--out",
          str(ddb),
          "--steps-per-episode",
          "2",
          "--num-episodes",
          "2",
          "--mode",
          "smoke",
      ])
      == 0
  )

  par_alias = tmp_path / "par_alias.json"
  assert eeb_main(["pareto", "--output", str(par_alias), "--inputs", str(s1a)]) == 0
  assert par_alias.exists()


def test_xy_axis_redteam_and_literature_audit_addendum_gates(tmp_path: Path) -> None:
  """Positive and negative controls for all 6 Red-Team & Literature Audit Addendum fixes."""
  import re
  import threading
  import time
  from eeb.envelope.check import check_envelope
  from eeb.loadgen.gate import sanity_gate
  from eeb.servers.cosmos_predict25.policy import CosmosPredict25JaxPolicy
  from eeb.telemetry.sidecar import summarize_samples
  from embodied_bench.agent_kickoff import _map_gpu_name_to_sku, parse_kickoff_prompt, run_agent_kickoff
  from embodied_bench.schema import HARDWARE_SKU_SPECS, RecipeConfig, validate_division_compliance
  from embodied_bench.stage1_golden_replay import run_stage1_from_recipe_path

  # (a) Closed-Division WM stream_chunk_frames ban + Open lookalike + Stage1 honest TTFF == L_clip
  rcfg_bad = RecipeConfig(
      recipe_id="wm_closed_bad_chunk",
      track="WM_TRACK",
      model_family="cosmos-predict2.5",
      model_name="cosmos-predict2.5-2b",
      baseline_model_id="cosmos-predict2.5-2b",
      precision="BF16",
      hardware_sku="RTX_5090_32GB",
      division="CLOSED",
      denoising_steps=10,
      reference_denoising_steps=10,
      stream_chunk_frames=4,
  )
  with pytest.raises(ValueError, match="stream_chunk_frames=4"):
    rcfg_bad.validate()
  with pytest.raises(ValueError, match="stream_chunk_frames=4"):
    validate_division_compliance(rcfg_bad)

  rcfg_open_ok = RecipeConfig(
      recipe_id="wm_open_ok_chunk",
      track="WM_TRACK",
      model_family="cosmos-predict2.5",
      model_name="cosmos-predict2.5-2b",
      baseline_model_id="cosmos-predict2.5-2b",
      precision="BF16",
      hardware_sku="RTX_5090_32GB",
      division="OPEN",
      denoising_steps=4,
      reference_denoising_steps=10,
      stream_chunk_frames=4,
      declared_training_compute_gpu_hours=0.0,
      declared_inference_flops_gflops=1200.0,
      osi_license="Apache-2.0",
  )
  rcfg_open_ok.validate()
  assert validate_division_compliance(rcfg_open_ok) == []

  with pytest.raises(ValueError, match="stream_chunk_frames=2 is prohibited in CLOSED division"):
    CosmosPredict25JaxPolicy(checkpoint_dir="/tmp/nonexistent", stream_chunk_frames=2)

  s1_wm = run_stage1_from_recipe_path(
      recipe_path=REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_rtx5090.yaml",
      output_path=tmp_path / "wm_stage1.json",
      include_stage2=False,
      warmup_calls=2,
      timed_calls=8,
      num_runs=1,
  )
  for op in s1_wm["operating_points"]:
    assert op["ttff_p99_ms"] == op["rollout_latency_p99_ms"]
    assert op["observation_staleness_ms"] >= 625.0

  # (b) Opaque random UUIDv4 session_ids + inline blind perturbation/golden canaries inside ConcurrencySweep
  seen_sessions: list[str] = []
  sess_lock = threading.Lock()

  class PhaseSniffingCheatingVLAPolicy(BasePolicy):
    """Cheats during timed sweep unless session_id leaks phase prefix; ignores 1e-3 camera perturbation."""

    def capabilities(self) -> Caps:
      return Caps(
          track="VLA_TRACK",
          horizon=5,
          execute_horizon=5,
          dt_ms=50.0,
          precision="BF16",
          model_id="pi05-3b",
          server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
          action_dim=7,
          proprio_dim=8,
          cameras=("agentview", "wrist"),
          image_height=64,
          image_width=64,
      )

    def reset(self, session_id: str) -> None:
      return None

    def predict(self, obs: Observation) -> ActionChunk:
      with sess_lock:
        seen_sessions.append(obs.session_id)
      # Only compute real image sensitivity if session_id leaks a phase marker; otherwise quantize coarsely
      if any(tag in obs.session_id for tag in ("golden", "g4-probe", "warmup")):
        img_mean = float(np.mean(obs.images["agentview"].astype(np.float64)))
        return ActionChunk(actions=np.full((5, 7), img_mean * 0.01, dtype=np.float32))
      return ActionChunk(actions=np.zeros((5, 7), dtype=np.float32))

  cheat_pol = PhaseSniffingCheatingVLAPolicy()
  cheat_res = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      SweepConfig(concurrency=[1, 2], warmup_calls=2, timed_calls=8, runs=1, server_warmup_calls=2),
      policy=cheat_pol,
  ).run()
  hex32 = re.compile(r"^[0-9a-f]{32}$")
  assert len(seen_sessions) >= 10
  for sid in seen_sessions:
    assert hex32.match(sid), f"session_id leaked structural info: {sid}"
    for leaked in ("c1-", "c2-", "r0-", "golden", "g4-probe", "warmup"):
      assert leaked not in sid
  assert cheat_res.points[0].runs[0].num_inline_canaries >= 1
  assert cheat_res.points[0].runs[0].num_inline_canary_failures >= 1
  assert cheat_res.points[0].all_runs_passed is False
  gate_cheat = sanity_gate(cheat_res.points[0], cheat_res.caps, "exec-cheat")
  assert gate_cheat.passed is False
  assert any("inline" in v.lower() and ("canary" in v.lower() or "canaries" in v.lower()) for v in gate_cheat.violations)

  # Positive lookalike: honest vision-reactive DummyVLAPolicy passes inline canaries with zero failures
  honest_pol = DummyVLAPolicy(compute_delay_ms=0.05, horizon=5, execute_horizon=5)
  honest_res = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      SweepConfig(concurrency=[2], warmup_calls=2, timed_calls=12, runs=1, server_warmup_calls=1),
      policy=honest_pol,
  ).run()
  assert honest_res.points[0].runs[0].num_inline_canaries >= 1
  assert honest_res.points[0].runs[0].num_inline_canary_failures == 0
  assert honest_res.points[0].all_runs_passed is True
  assert sanity_gate(honest_res.points[0], honest_res.caps, "exec-honest").passed is True

  # (c) Per-slot quota n_slot = ceil(N/C) + pooled p99(C) scoring + diagnostic worst_slot_p99 capture
  call_counter = {"n": 0, "stalls": 0}
  cnt_lock = threading.Lock()
  parked_session: dict[str, Optional[str]] = {"sid": None}

  class SingleRequestLIFOStarvationPolicy(DummyVLAPolicy):
    """Stalls 2 requests on 1 victim stream (scored p99 equals pooled_p99; diagnostic worst_slot_p99 surfaces victim tail)."""

    def predict(self, obs: Observation) -> ActionChunk:
      with cnt_lock:
        call_counter["n"] += 1
        idx = call_counter["n"]
        if idx >= 16 and parked_session["sid"] is None:
          parked_session["sid"] = obs.session_id
        if obs.session_id == parked_session["sid"] and call_counter["stalls"] < 2:
          call_counter["stalls"] += 1
          do_stall = True
        else:
          do_stall = False
      if do_stall:
        time.sleep(0.045)
      return super().predict(obs)

  lifo_pol = SingleRequestLIFOStarvationPolicy(compute_delay_ms=0.1, horizon=5, execute_horizon=5)
  lifo_res = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      SweepConfig(
          concurrency=[4],
          warmup_calls=1,
          timed_calls=400,
          runs=1,
          server_warmup_calls=0,
          inline_canaries_enabled=False,
      ),
      policy=lifo_pol,
  ).run()
  lifo_run = lifo_res.points[0].runs[0]
  # Every one of the 4 concurrent slots completed exactly ceil(400 / 4) = 100 timed calls
  assert lifo_run.num_timed_calls == 400
  assert lifo_run.slot_count_spread == 0.0
  # Scored p99(C) is strictly pooled steady-state p99 across all active slots at C,
  # while diagnostic worst_slot_p99 captures per-slot outliers alongside the <=15% slot fairness fence.
  assert lifo_run.latency_ms["p99"] == lifo_run.latency_ms["pooled_p99"]
  assert lifo_run.latency_ms["worst_slot_p99"] > lifo_run.latency_ms["p99"] * 2.0
  assert lifo_run.latency_ms["worst_slot_p99"] >= 15.0

  # (d) Track-aware WM_TRACK timed_calls=200 compliance + X-axis max_p99_variance_pct (<= 8%) tail gate
  cfg_200 = SweepConfig(warmup_calls=30, timed_calls=200, runs=3)
  assert cfg_200.protocol_compliant_for_track("WM_TRACK") is True
  assert cfg_200.protocol_notes_for_track("WM_TRACK") == []
  assert cfg_200.protocol_compliant_for_track("VLA_TRACK") is False
  assert any("320" in n for n in cfg_200.protocol_notes_for_track("VLA_TRACK"))

  run_phase = {"calls": 0}

  class JitteryTailStableMeanPolicy(DummyVLAPolicy):
    """Run 0 has tight 2 ms tail; Run 1 injects one 25 ms p99 spike while keeping mean throughput within tolerance."""

    def predict(self, obs: Observation) -> ActionChunk:
      with cnt_lock:
        run_phase["calls"] += 1
        k = run_phase["calls"]
      # Inject a single tail spike inside run 2 only
      if k == 35:
        time.sleep(0.025)
      else:
        time.sleep(0.0015)
      return super().predict(obs)

  jit_pol = JitteryTailStableMeanPolicy(compute_delay_ms=0.0, horizon=5, execute_horizon=5)
  jit_res = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      SweepConfig(
          concurrency=[1],
          warmup_calls=0,
          timed_calls=25,
          runs=2,
          max_variance_pct=80.0,
          max_p99_variance_pct=8.0,
          server_warmup_calls=0,
          inline_canaries_enabled=False,
      ),
      policy=jit_pol,
  ).run()
  assert jit_res.points[0].variance_passed is False
  assert "latency_p99_ms=" in jit_res.points[0].variance_reason

  # (e) summarize_samples prioritizes monotonic NVML hardware energy counter (energy_mj) over trapezoid polling
  # Case 1: Between polling instants the GPU burst to higher power so hardware energy_mj integrated 180 J over 1.0 s,
  # while instantaneous polls happened to see only 100 W at the sample instants.
  samples_hw = [
      {
          "ts": 10.0 + 0.25 * i,
          "power_w": 100.0,
          "mem_used_mb": 4096.0,
          "util_pct": 95.0,
          "provenance": "MEASURED",
          "source": "nvml",
          "energy_mj": 500_000.0 + 45_000.0 * i,
      }
      for i in range(5)
  ]
  sum_hw = summarize_samples(samples_hw, t0=10.0, t1=11.0, source_name="nvml")
  assert sum_hw.provenance == "MEASURED"
  assert sum_hw.extra.get("energy_method") == "hardware_energy_counter_mj"
  assert sum_hw.energy_j == pytest.approx(180.0, rel=1e-3)
  assert sum_hw.mean_power_w == pytest.approx(180.0, rel=1e-3)

  # Case 2: Fallback lookalike when energy_mj is absent -> trapezoid polling gives 100.0 J / 100.0 W
  samples_trap = [
      {
          "ts": 10.0 + 0.25 * i,
          "power_w": 100.0,
          "mem_used_mb": 4096.0,
          "util_pct": 95.0,
          "provenance": "MEASURED",
          "source": "nvml",
      }
      for i in range(5)
  ]
  sum_trap = summarize_samples(samples_trap, t0=10.0, t1=11.0, source_name="nvml")
  assert sum_trap.provenance == "MEASURED"
  assert sum_trap.extra.get("energy_method") == "trapezoid_polling"
  assert sum_trap.energy_j == pytest.approx(100.0, rel=1e-3)
  assert sum_trap.mean_power_w == pytest.approx(100.0, rel=1e-3)

  # (f) L4_24GB SKU + agent_kickoff resolution + 3-variable Cloud TPU v6e-1 slicing & host-CPU neural offload ban
  assert "L4_24GB" in HARDWARE_SKU_SPECS
  assert HARDWARE_SKU_SPECS["L4_24GB"]["tdp_watts"] == 72.0
  assert check_envelope(hardware_sku="L4_24GB", allow_no_device=True).passed is True
  assert parse_kickoff_prompt("Start Cosmos-Predict2.5 on NVIDIA L4")["hardware_sku"] == "L4_24GB"
  assert _map_gpu_name_to_sku("NVIDIA L4", 24.0) == "L4_24GB"

  ko_tpu = tmp_path / "kickoff_tpu_v6e_check"
  plan_tpu = run_agent_kickoff(prompt="Start Cosmos-Predict2.5 on TPU v6e-1", study_dir=ko_tpu)
  assert plan_tpu["hardware_sku"] == "TPU_V6E_1_32GB"
  assert "TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0" in plan_tpu["single_device_env"]
  prompt_tpu = (ko_tpu / "EXPANDED_AGENT_PROMPT.md").read_text(encoding="utf-8")
  assert "TPU_CHIPS_PER_HOST_BOUNDS=1,1,1" in prompt_tpu
  assert "Host-CPU Neural Offload Ban" in prompt_tpu
  assert "stream_chunk_frames > 0" in prompt_tpu
  assert "p99 Prediction Rate (Predictions/s)" in prompt_tpu
  assert "pooled steady-state" in prompt_tpu
  assert "\u2014" not in prompt_tpu


def test_b1_standalone_measured_bundle_validation_and_f2_disambiguated_leaderboard_units(
    tmp_path: Path,
) -> None:
  from embodied_bench.leaderboard_compiler import compile_submissions_to_leaderboard, render_pr_comment_markdown
  from embodied_bench.submission_validator import prepare_phase1_commit_bundle, verify_submission_bundle

  # 1. Positive controls (Gate 4 lookalikes): standalone TPU + L40S measured bundles AND Phase-1 Commit bundles pass
  tpu_dir = tmp_path / "standalone_pi05_tpu_v6e"
  l40s_dir = tmp_path / "standalone_gr00t_l40s"
  assert _run_quick_measure(tpu_dir, recipe_path=REPO_ROOT / "recipes" / "track1_vla_pi05_tpu_v6e.yaml") == 0
  assert (
      _run_quick_measure(
          l40s_dir,
          recipe_path=REPO_ROOT / "recipes" / "track1_vla_gr00t_n17_l40s_bf16.yaml",
          policy_kwargs='{"compute_delay_ms": 0.5, "horizon": 16, "execute_horizon": 8}',
      )
      == 0
  )
  _promote_bundle_to_single_chip_certified(tpu_dir)
  _promote_bundle_to_single_chip_certified(l40s_dir)

  reg_path = tmp_path / "validation_registry.json"
  for ref_dir, expected_sku in ((tpu_dir, "TPU_V6E_1_32GB"), (l40s_dir, "L40S_48GB")):
    assert not (ref_dir / "recipe.yaml").is_file()
    rep = validate_submission_directory(ref_dir, registry_path=reg_path)
    assert rep.passed is True, f"{ref_dir.name} failed validation: {rep.reason}"
    assert rep.status == "CERTIFIED"
    assert rep.reason is None
    assert rep.has_measurements is True
    assert rep.validation_scope == "FULL_MEASURED_BUNDLE"
    assert rep.receipt_verified is True
    assert rep.num_devices == 1
    assert rep.canonical_challenge_model is True
    assert rep.hardware_sku == expected_sku

  # Gate 1 repeat-call idempotency on standalone bundle + Phase-1 Commit bundle + template recipe bundle
  rep_repeat = validate_submission_directory(tpu_dir, registry_path=reg_path)
  assert rep_repeat.passed is True and rep_repeat.hardware_sku == "TPU_V6E_1_32GB"

  p1_dir = tmp_path / "phase1_l40s_bundle"
  p1_res = prepare_phase1_commit_bundle(
      source_bundle_dir=l40s_dir,
      output_dir=p1_dir,
      recipe_commit_sha256="a" * 64,
      team="Lab-Alpha",
      engine_tag="tensorrt-bf16",
  )
  assert p1_res["recipe_commit_sha256"] == "a" * 64
  p1_ver = verify_submission_bundle(p1_dir)
  assert p1_ver.passed is True, p1_ver.reason
  p1_val = validate_submission_directory(p1_dir, registry_path=reg_path)
  assert p1_val.passed is True and p1_val.validation_scope == "PHASE1_COMMIT_REVEAL_BUNDLE"

  tmpl_dir = tmp_path / "template_l40s_recipe_only"
  tmpl_dir.mkdir()
  shutil.copy2(REPO_ROOT / "recipes" / "track1_vla_pi05_l40s_bf16.yaml", tmpl_dir / "recipe.yaml")
  t_rep = validate_submission_directory(tmpl_dir)
  assert t_rep.passed is True
  assert t_rep.validation_scope == "RECIPE_ONLY_NO_MEASUREMENTS"

  # 2. Negative controls (Gate 4 violations & Gate 2 non-empty reasons):
  # (a) Empty folder lacking both recipe and report+receipt raises FileNotFoundError
  empty_sub = tmp_path / "empty_submission"
  empty_sub.mkdir()
  with pytest.raises(FileNotFoundError, match="No 'recipe.yaml'"):
    validate_submission_directory(empty_sub)

  # (b) Standalone orphan report.json without receipt.json -> REJECTED with actionable reason
  standalone_orphan_rep = tmp_path / "standalone_orphan_report"
  standalone_orphan_rep.mkdir()
  shutil.copy2(tpu_dir / "report.json", standalone_orphan_rep / "report.json")
  r_no_rcpt = validate_submission_directory(standalone_orphan_rep)
  assert r_no_rcpt.passed is False and r_no_rcpt.status == "REJECTED"
  assert r_no_rcpt.reason and "without a cryptographic 'receipt.json'" in r_no_rcpt.reason

  # (c) Standalone orphan receipt.json without sibling report.json -> REJECTED with actionable reason
  standalone_orphan_rcpt = tmp_path / "standalone_orphan_receipt"
  standalone_orphan_rcpt.mkdir()
  shutil.copy2(tpu_dir / "receipt.json", standalone_orphan_rcpt / "receipt.json")
  r_no_rep = validate_submission_directory(standalone_orphan_rcpt)
  assert r_no_rep.passed is False and r_no_rep.status == "REJECTED"
  assert r_no_rep.reason and "missing 'report.json'" in r_no_rep.reason

  # (d) Unpromoted INVALID diagnostic bundle is rejected with non-empty reason
  diag_dir = tmp_path / "invalid_diag_bundle"
  assert _run_quick_measure(diag_dir, recipe_path=REPO_ROOT / "recipes" / "track1_vla_pi05_l40s_bf16.yaml") == 0
  d_rep = validate_submission_directory(diag_dir)
  assert d_rep.passed is False and d_rep.status == "REJECTED"
  assert d_rep.reason and "INVALID" in d_rep.reason

  # 3. Empty Day-1 leaderboard + populated fixture both render disambiguated units and Season 1 prizes
  lb_empty = json.loads((REPO_ROOT / "leaderboard" / "leaderboard_v1.json").read_text(encoding="utf-8"))
  assert lb_empty["num_verified_submissions"] == 0
  md_empty = render_pr_comment_markdown(lb_empty)
  assert "monetary prizes" in md_empty
  assert "an invitation to apply for Google Cloud credits" in md_empty
  assert "\u2014" not in md_empty

  subs_fixture = tmp_path / "subs_for_md"
  shutil.copytree(p1_dir, subs_fixture / "phase1_l40s_bundle")
  lb_pop = compile_submissions_to_leaderboard(subs_fixture, output_json_path=tmp_path / "lb_pop.json")
  md_text = render_pr_comment_markdown(lb_pop)
  assert "Chunks/s" in md_text and "Rollouts/s" in md_text
  assert " Hz |" not in md_text and " Hz)" not in md_text


