"""Unit and integration test suite for EmbodiedX (`embodied-efficiency-bench` v1.0).

Covers:
  - Gate 1: Crash path (orphan temp cleanup & corrupt JSON recovery), repeat-call path (zero stale field carryover),
    and concurrent-writer path (multi-thread + multi-process atomic read-modify-write under fcntl.flock).
  - Gate 2: Complete 8-state RunStatus state machine reachability, transition legality, and mandatory non-empty reasons.
  - Gate 3: Verbatim CLI execution of `measure`, `eval-libero-rt`, `eval-wm-ranking`, `run-stage1`, `run-stage2`,
    `compute-pareto`, and `verify-receipt` in both flag orders.
  - Gate 4: Negative controls & legitimate lookalikes for Track 1/2 Gate A, Stage 2 Gate B, MLPerf Compliance Gates
    G1-G4 (TEST01 parity, McNemar G2, Same-Execution G3, TEST04 anti-memoization G4), CLOSED/OPEN division rules,
    LIBERO-RT horizon/sim-time caps, AST hot-loop sync auditor, and SHA-256 cryptographic receipt tampering detection.
  - LIBERO-RT latency coupling (ceil(L / delta_t) action delay, SYNCHRONOUS_HOLD vs ASYNC_CHUNK vs RTC_INPAINTING,
    OBJECT_DISPLACEMENT / GOAL_SWITCH / EMERGENCY_STOP safety probes, and Tier 1/2/3 scaling).
  - Unified Scenarios P1_ONBOARD, P2_FLEET, P3_EDGE_CLOUD, S1_INTERACTIVE_WM, and S2_WM_EVALUATOR (Kendall's tau).
  - Pareto frontier domination stripping & log-linear Iso-10Hz / Iso-20Hz / Iso-30Hz / Iso-50Hz interpolation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from embodied_bench.adapters import (
    ADAPTER_REGISTRY,
    ModelAdapter,
    build_adapter,
    register_model_adapter,
)
from embodied_bench.attestation import (
    create_submission_receipt,
    verify_submission_receipt,
)
from embodied_bench.cli import main as cli_main
from embodied_bench.golden_dataset import (
    EMBODIEDX_TRACES_V1_SOURCES,
    generate_track1_golden_batch,
    generate_track2_golden_batch,
    load_golden_archive_npz,
    save_golden_archive_npz,
)
from embodied_bench.libero_rt import (
    compute_delay_steps,
    run_libero_rt_evaluation,
    validate_chunk_horizon,
    verify_tier3_hil_consistency,
)
from embodied_bench.measure import (
    compute_hardware_independent_profile,
    compute_iso_tops_and_bw_efficiency,
    compute_roofline_normalized_latency,
    compute_semianalysis_normalizations,
    evaluate_performance_cap_envelope,
    generate_slice_clock_cap_commands,
    get_performance_cap_envelope,
    run_eeb_measure,
)
from embodied_bench.pareto import (
    compute_iso_reactivity_slices,
    extract_pareto_frontier,
    extract_semianalysis_4way_pareto_frontiers,
    interpolate_log_linear_at_latency,
    is_pareto_efficient,
)
from embodied_bench.quality_gates import (
    audit_hot_loop_device_sync,
    evaluate_compliance_suite_g1_g4,
    evaluate_gate_g1_parity,
    evaluate_gate_g2_mcnemar_non_inferiority,
    evaluate_gate_g3_same_execution,
    evaluate_gate_g4_cache_memoization,
    evaluate_stage2_gate_b,
    evaluate_track1_gate_a,
    evaluate_track2_gate_a,
    evaluate_wm_open_3part_quality_gate,
    verify_same_execution_id,
)
from embodied_bench.scenarios import (
    DEFAULT_S2_POLICY_POOL,
    compute_kendall_tau,
    evaluate_p1_onboard,
    evaluate_p2_fleet,
    evaluate_p3_edge_cloud,
    evaluate_s1_interactive_wm,
    evaluate_s2_wm_evaluator,
)
from embodied_bench.schema import (
    BenchmarkClock,
    CANONICAL_CHALLENGE_MODELS,
    CompressionTier,
    DEPRECATED_BASELINE_MODEL_IDS,
    DivisionType,
    ExecutionStrategy,
    HARDWARE_SKU_SPECS,
    LiberoRTTier,
    MODERN_REFERENCE_CHECKPOINTS,
    PERF_CAP_ALIASES,
    PERFORMANCE_CAP_ENVELOPES,
    PerfCapSatisfactionMode,
    PerformanceCapId,
    PowerClass,
    REASON_REQUIRED_STATUSES,
    RecipeConfig,
    RunStatus,
    SCHEMA_FIELD_EXAMPLES,
    SafetyProbeType,
    ScenarioType,
    TrackType,
    VALID_STATE_TRANSITIONS,
    atomic_read_modify_write_json,
    cleanup_orphan_temp_files,
    is_vla_track,
    is_wm_track,
    load_yaml_or_json,
    now_pacific_iso,
    register_model_checkpoint,
    resolve_perf_cap_id,
    upsert_run_record,
    validate_division_compliance,
    validate_status_and_reason,
)
from embodied_bench.stage1_golden_replay import run_stage1_from_recipe_path
from embodied_bench.telemetry import (
    compute_joules_per_decision,
    parse_custom_asic_telemetry_json,
    parse_rocm_smi_csv,
    parse_tegrastats_line,
    parse_tpu_info_output,
    run_eeb_measurement_protocol,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "recipes"


def _subprocess_worker_increment(args: tuple[str, int]) -> int:
  """Top-level helper for multi-process Gate 1 atomic RMW test."""
  target_path, worker_id = args

  def _mod(state: dict) -> dict:
    state["counter"] = int(state.get("counter", 0)) + 1
    writers = list(state.get("writers", []))
    writers.append(worker_id)
    state["writers"] = writers
    return state

  atomic_read_modify_write_json(
      target_path,
      modifier_fn=_mod,
      default_factory=lambda: {"counter": 0, "writers": []},
  )
  return worker_id


class TestEmbodiedBenchSuite(unittest.TestCase):
  """Comprehensive Gate 1-8, LIBERO-RT, G1-G4 Compliance, and P1/P2/P3/S1/S2 Scenario tests."""

  # -------------------------------------------------------------------------
  # Gate 1: Crash Path, Repeat-Call Path, and Concurrent-Writer Path
  # -------------------------------------------------------------------------
  def test_gate1_crash_path_orphan_cleanup_and_corrupt_recovery(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = Path(tmpdir)
      registry_file = tmp_path / "registry.json"

      orphan1 = tmp_path / f".tmp_embodied_{registry_file.name}_abc123.json"
      orphan2 = tmp_path / f".tmp_embodied_{registry_file.name}_def456.json"
      orphan1.write_text("{corrupt partial write", encoding="utf-8")
      orphan2.write_text("", encoding="utf-8")
      registry_file.write_text("{truncated_json: [1, 2", encoding="utf-8")

      state = upsert_run_record(
          registry_path=registry_file,
          run_id="run-crash-recovery-01",
          status=RunStatus.PENDING,
          metrics={"fps": 0.0},
      )
      self.assertTrue(state.get("recovered_from_corrupt_state"))
      self.assertIn("run-crash-recovery-01", state["runs"])
      self.assertFalse(orphan1.exists())
      self.assertFalse(orphan2.exists())
      self.assertEqual(cleanup_orphan_temp_files(tmp_path), 0)

  def test_gate1_repeat_call_path_no_stale_optional_field_carryover(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      registry_file = Path(tmpdir) / "registry.json"

      upsert_run_record(
          registry_path=registry_file,
          run_id="run-repeat-01",
          status=RunStatus.PENDING,
          metrics={"step": 0},
          notes="initial queue note",
      )
      s_blocked = upsert_run_record(
          registry_path=registry_file,
          run_id="run-repeat-01",
          status=RunStatus.BLOCKED,
          metrics={"step": 1},
          reason="Waiting on H100_SXM5 exclusive node allocation.",
          notes="temporary cluster lock",
      )
      rec_blocked = s_blocked["runs"]["run-repeat-01"]
      self.assertEqual(rec_blocked["status"], "BLOCKED")
      self.assertIsNotNone(rec_blocked["reason"])
      self.assertEqual(rec_blocked["notes"], "temporary cluster lock")

      upsert_run_record(
          registry_path=registry_file,
          run_id="run-repeat-01",
          status=RunStatus.RUNNING,
          metrics={"step": 2},
      )
      s_certified = upsert_run_record(
          registry_path=registry_file,
          run_id="run-repeat-01",
          status=RunStatus.CERTIFIED,
          metrics={"fps": 148.5, "hz_plan": 19.2},
      )
      rec_cert = s_certified["runs"]["run-repeat-01"]
      self.assertEqual(rec_cert["status"], "CERTIFIED")
      self.assertIsNone(rec_cert["reason"])
      self.assertIsNone(rec_cert["notes"])
      self.assertEqual(rec_cert["metrics"], {"fps": 148.5, "hz_plan": 19.2})

  def test_gate1_concurrent_writers_threads_and_processes(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      shared_json = Path(tmpdir) / "concurrent_state.json"
      num_threads = 16
      num_procs = 8

      def _thread_worker(tid: int) -> None:
        upsert_run_record(
            registry_path=shared_json,
            run_id=f"thread-run-{tid:02d}",
            status=RunStatus.PENDING,
            metrics={"tid": tid},
        )
        upsert_run_record(
            registry_path=shared_json,
            run_id=f"thread-run-{tid:02d}",
            status=RunStatus.RUNNING,
            metrics={"tid": tid},
        )
        upsert_run_record(
            registry_path=shared_json,
            run_id=f"thread-run-{tid:02d}",
            status=RunStatus.CERTIFIED,
            metrics={"tid": tid, "done": True},
        )

      with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_thread_worker, range(num_threads)))

      with ProcessPoolExecutor(max_workers=4) as proc_pool:
        list(
            proc_pool.map(
                _subprocess_worker_increment,
                [(str(shared_json), pid) for pid in range(num_procs)],
            )
        )

      final_payload = json.loads(shared_json.read_text(encoding="utf-8"))
      self.assertEqual(len(final_payload["runs"]), num_threads)
      for tid in range(num_threads):
        r = final_payload["runs"][f"thread-run-{tid:02d}"]
        self.assertEqual(r["status"], "CERTIFIED")
        self.assertTrue(r["metrics"]["done"])
      self.assertEqual(final_payload["counter"], num_procs)
      self.assertEqual(sorted(final_payload["writers"]), list(range(num_procs)))

  # -------------------------------------------------------------------------
  # Gate 2: State Machine Completeness & Mandatory Non-Empty Failure Reasons
  # -------------------------------------------------------------------------
  def test_gate2_all_states_reachable_and_reason_enforced(self) -> None:
    visited: set[RunStatus] = set()

    visited.add(validate_status_and_reason(RunStatus.PENDING))
    visited.add(
        validate_status_and_reason(
            RunStatus.BLOCKED,
            reason="Hardware node busy",
            previous_status=RunStatus.PENDING,
        )
    )
    visited.add(
        validate_status_and_reason(
            RunStatus.RUNNING,
            previous_status=RunStatus.BLOCKED,
        )
    )
    visited.add(
        validate_status_and_reason(
            RunStatus.CERTIFIED,
            previous_status=RunStatus.RUNNING,
        )
    )
    visited.add(
        validate_status_and_reason(
            RunStatus.REJECTED,
            reason="Gate A failed cosine similarity threshold",
            previous_status=RunStatus.RUNNING,
        )
    )
    visited.add(
        validate_status_and_reason(
            RunStatus.FAILED,
            reason="Adapter threw OOM exception",
            previous_status=RunStatus.RUNNING,
        )
    )
    visited.add(
        validate_status_and_reason(
            RunStatus.SKIPPED,
            reason="Optional Stage 2 skipped by flag",
            previous_status=RunStatus.PENDING,
        )
    )
    visited.add(
        validate_status_and_reason(
            RunStatus.INVALID,
            reason="Mismatched co-located execution_id",
            previous_status=RunStatus.RUNNING,
        )
    )
    self.assertEqual(visited, set(RunStatus))
    self.assertEqual(set(VALID_STATE_TRANSITIONS.keys()), set(RunStatus))

    for st in REASON_REQUIRED_STATUSES:
      with self.assertRaises(ValueError):
        validate_status_and_reason(st, reason=None)
      with self.assertRaises(ValueError):
        validate_status_and_reason(st, reason="   ")

    with self.assertRaises(ValueError):
      validate_status_and_reason(
          RunStatus.CERTIFIED,
          previous_status=RunStatus.PENDING,
      )

  # -------------------------------------------------------------------------
  # Gate 3: Verbatim CLI Execution (`measure`, `eval-libero-rt`, `eval-wm-ranking`, `run-stage1`, `run-stage2`, `compute-pareto`, `verify-receipt`)
  # -------------------------------------------------------------------------
  def test_gate3_cli_commands_verbatim_and_both_flag_orders(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      r1_yaml = str(RECIPES_DIR / "track2_wm_cosmos2b_workstation350t.yaml")
      r2_yaml = str(RECIPES_DIR / "track1_vla_pi05_edge100t.yaml")
      r0_yaml = str(RECIPES_DIR / "track1_vla_pi05_orin_nx.yaml")
      out1 = str(tmp / "track1_out.json")
      out2 = str(tmp / "track2_out.json")
      meas_out = str(tmp / "round0_measure.json")
      lrt_out = str(tmp / "libero_rt_out.json")
      wm_rank_out = str(tmp / "wm_ranking_out.json")
      stage2_out = str(tmp / "track1_stage2.json")
      leaderboard_out = str(tmp / "leaderboard.json")
      receipt_out = str(tmp / "submission_receipt.json")

      # 1. measure (both flag orders)
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_m1:
        rc_m1 = cli_main(["measure", "--recipe", r0_yaml, "--output", meas_out])
      self.assertEqual(rc_m1, 0)
      m1_json = json.loads(buf_m1.getvalue())
      self.assertEqual(m1_json["status"], "CERTIFIED")
      self.assertIn(m1_json["clock"], ("eeb_v1", "round0_orin_nx_sprint"))
      self.assertEqual(m1_json["scenario"], "P1_ONBOARD")
      self.assertEqual(m1_json["power_class"], "E1")
      self.assertTrue(m1_json["compliance_passed"])

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_m2:
        rc_m2 = cli_main(["measure", "--output", meas_out, "--recipe", r0_yaml, "--warmup-calls", "30", "--timed-calls", "1000"])
      self.assertEqual(rc_m2, 0)
      self.assertLess(json.loads(buf_m2.getvalue())["cross_run_variance_pct"], 5.0)

      # 2. eval-libero-rt (both flag orders)
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_lrt1:
        rc_lrt1 = cli_main([
            "eval-libero-rt",
            "--recipe",
            r2_yaml,
            "--output",
            lrt_out,
            "--execution-strategy",
            "RTC_INPAINTING",
            "--latency-mean-ms",
            "42.0",
        ])
      self.assertEqual(rc_lrt1, 0)
      lrt_json = json.loads(buf_lrt1.getvalue())
      self.assertTrue(lrt_json["passed"])
      self.assertEqual(lrt_json["delay_steps_mean"], 1)

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_lrt2:
        rc_lrt2 = cli_main([
            "eval-libero-rt",
            "--output",
            lrt_out,
            "--recipe",
            r2_yaml,
            "--tier",
            "tier3_websocket_hil_anchor",
        ])
      self.assertEqual(rc_lrt2, 0)
      self.assertTrue(json.loads(buf_lrt2.getvalue())["passed"])

      # 3. eval-wm-ranking (both flag orders)
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_wm1:
        rc_wm1 = cli_main(["eval-wm-ranking", "--recipe", r1_yaml, "--output", wm_rank_out])
      self.assertEqual(rc_wm1, 0)
      wm_json = json.loads(buf_wm1.getvalue())
      self.assertTrue(wm_json["passed"])
      self.assertGreaterEqual(wm_json["kendall_tau"], 0.80)
      self.assertEqual(wm_json["num_policies_evaluated"], 12)

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_wm2:
        rc_wm2 = cli_main(["eval-wm-ranking", "--output", wm_rank_out, "--episodes-per-policy", "40", "--recipe", r1_yaml])
      self.assertEqual(rc_wm2, 0)
      self.assertTrue(json.loads(buf_wm2.getvalue())["passed"])

      # 4. run-stage1 (normal flag order + reverse flag order)
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf1:
        rc1 = cli_main(["run-stage1", "--recipe", r1_yaml, "--output", out1])
      self.assertEqual(rc1, 0)
      self.assertEqual(json.loads(buf1.getvalue())["status"], "CERTIFIED")

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf2:
        rc2 = cli_main(["run-stage1", "--output", out2, "--recipe", r2_yaml])
      self.assertEqual(rc2, 0)
      self.assertEqual(json.loads(buf2.getvalue())["status"], "CERTIFIED")

      # 5. run-stage2 (normal flag order + reverse flag order)
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_s2:
        rc_s2 = cli_main(["run-stage2", "--receipt", out1, "--output", stage2_out])
      self.assertEqual(rc_s2, 0)
      self.assertTrue(json.loads(buf_s2.getvalue())["gate_b_passed"])

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_s2_rev:
        rc_s2_rev = cli_main(["run-stage2", "--output", stage2_out, "--receipt", out1])
      self.assertEqual(rc_s2_rev, 0)
      self.assertTrue(json.loads(buf_s2_rev.getvalue())["gate_b_passed"])

      # 6. compute-pareto (normal flag order + reverse flag order)
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_p1:
        rc_p1 = cli_main([
            "compute-pareto",
            "--inputs",
            out1,
            out2,
            meas_out,
            "--output",
            leaderboard_out,
            "--receipt-output",
            receipt_out,
        ])
      self.assertEqual(rc_p1, 0)
      self.assertEqual(json.loads(buf_p1.getvalue())["num_certified_submissions"], 3)

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_p2:
        rc_p2 = cli_main([
            "compute-pareto",
            "--output",
            leaderboard_out,
            "--receipt-output",
            receipt_out,
            "--inputs",
            out1,
            out2,
        ])
      self.assertEqual(rc_p2, 0)
      self.assertEqual(json.loads(buf_p2.getvalue())["num_submissions"], 2)

      # 7. verify-receipt via subprocess verbatim CLI invocation
      proc = subprocess.run(
          [
              sys.executable,
              "-m",
              "embodied_bench.cli",
              "verify-receipt",
              "--receipt",
              receipt_out,
          ],
          cwd=str(REPO_ROOT),
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stderr)
      verify_payload = json.loads(proc.stdout)
      self.assertTrue(verify_payload["verified"])

  # -------------------------------------------------------------------------
  # Gate 4: Negative Controls & Legitimate Lookalikes (Including G1-G4 & Divisions)
  # -------------------------------------------------------------------------
  def test_gate4_negative_controls_and_legitimate_lookalikes(self) -> None:
    # A. Co-located Same-Execution ID validator
    verify_same_execution_id("exec-same-123", "exec-same-123")
    with self.assertRaises(ValueError):
      verify_same_execution_id("exec-throughput-A", "exec-quality-B")
    with self.assertRaises(ValueError):
      verify_same_execution_id("", "exec-quality-B")

    # B. Track 1 Gate A & Track 2 Gate A
    golden_t1 = generate_track1_golden_batch(batch_size=2, horizon_frames=16)
    good_t1 = evaluate_track1_gate_a(
        execution_id="exec-t1",
        throughput_execution_id="exec-t1",
        ref_video_latents=golden_t1.ref_video_latents,
        cand_video_latents=np.clip(golden_t1.ref_video_latents + 0.002, 0.0, 1.0),
        ref_latent_actions=golden_t1.ref_latent_actions,
        cand_latent_actions=golden_t1.ref_latent_actions,
    )
    self.assertTrue(good_t1.passed)

    bad_t1 = evaluate_track1_gate_a(
        execution_id="exec-t1-bad",
        throughput_execution_id="exec-t1-bad",
        ref_video_latents=golden_t1.ref_video_latents,
        cand_video_latents=1.0 - golden_t1.ref_video_latents,
        ref_latent_actions=golden_t1.ref_latent_actions,
        cand_latent_actions=-golden_t1.ref_latent_actions,
    )
    self.assertFalse(bad_t1.passed)

    golden_t2 = generate_track2_golden_batch(concurrency=2, chunk_size=8)
    good_t2 = evaluate_track2_gate_a(
        execution_id="exec-t2",
        throughput_execution_id="exec-t2",
        ref_actions=golden_t2.ref_actions,
        cand_actions=golden_t2.ref_actions + 0.001,
    )
    self.assertTrue(good_t2.passed)

    bad_t2 = evaluate_track2_gate_a(
        execution_id="exec-t2-bad",
        throughput_execution_id="exec-t2-bad",
        ref_actions=golden_t2.ref_actions,
        cand_actions=-golden_t2.ref_actions,
    )
    self.assertFalse(bad_t2.passed)

    # C. MLPerf Compliance Gate G1 (ar_discrete, continuous_flow, world_model)
    g1_ar_ok = evaluate_gate_g1_parity(
        execution_id="exec-g1",
        head_type="ar_discrete",
        ref_outputs=golden_t2.ref_actions,
        timed_outputs=golden_t2.ref_actions + 0.001,
    )
    self.assertTrue(g1_ar_ok.passed)

    g1_ar_bad = evaluate_gate_g1_parity(
        execution_id="exec-g1",
        head_type="ar_discrete",
        ref_outputs=golden_t2.ref_actions,
        timed_outputs=golden_t2.ref_actions + 0.25,
    )
    self.assertFalse(g1_ar_bad.passed)

    g1_wm_bad_drift = evaluate_gate_g1_parity(
        execution_id="exec-g1-wm",
        head_type="world_model",
        ref_outputs=golden_t1.ref_video_latents,
        timed_outputs=golden_t1.ref_video_latents,
        ref_latent_actions=golden_t1.ref_latent_actions,
        timed_latent_actions=golden_t1.ref_latent_actions,
        long_horizon_60s_drift_ratio=1.35,  # exceeds 1.20 bound
    )
    self.assertFalse(g1_wm_bad_drift.passed)

    # D. Gate G2 (Paired-Seed McNemar Non-Inferiority & Round 0 / EEB v1 Awards)
    # Round 0: -4.0 pts & p99=82ms -> eligible for leaderboard, Best Efficiency Award, AND Real-Time Award
    g2_r0_both = evaluate_gate_g2_mcnemar_non_inferiority(
        baseline_success_rate_pct=96.0,
        candidate_success_rate_pct=92.0,
        clock=BenchmarkClock.ROUND0_ORIN_NX_SPRINT.value,
        division=DivisionType.CLOSED.value,
        p99_latency_ms=82.0,
    )
    self.assertTrue(g2_r0_both.passed)
    self.assertTrue(g2_r0_both.best_efficiency_award_eligible)
    self.assertTrue(g2_r0_both.realtime_award_eligible)

    # Round 0: -8.0 pts & p99=75ms -> eligible for leaderboard & Real-Time Award, NOT Best Efficiency Award (-5 pts)
    g2_r0_rt_only = evaluate_gate_g2_mcnemar_non_inferiority(
        baseline_success_rate_pct=96.0,
        candidate_success_rate_pct=88.0,
        clock=BenchmarkClock.ROUND0_ORIN_NX_SPRINT.value,
        division=DivisionType.OPEN.value,
        p99_latency_ms=75.0,
    )
    self.assertTrue(g2_r0_rt_only.passed)
    self.assertFalse(g2_r0_rt_only.best_efficiency_award_eligible)
    self.assertTrue(g2_r0_rt_only.realtime_award_eligible)

    # EEB v1 CLOSED: -4.0 pts fails -3.0 pts / 97% relative requirement
    g2_v1_closed_fail = evaluate_gate_g2_mcnemar_non_inferiority(
        baseline_success_rate_pct=96.0,
        candidate_success_rate_pct=92.0,
        clock=BenchmarkClock.EEB_V1.value,
        division=DivisionType.CLOSED.value,
    )
    self.assertFalse(g2_v1_closed_fail.passed)

    # E. Gate G3 (Same-Execution & Bit-Identical Replay Audit)
    g3_ok = evaluate_gate_g3_same_execution(
        timing_execution_id="exec-1",
        quality_execution_id="exec-1",
        telemetry_execution_id="exec-1",
        timed_outputs=golden_t2.ref_actions,
        audited_outputs=golden_t2.ref_actions.copy(),
    )
    self.assertTrue(g3_ok.passed)

    g3_tampered = evaluate_gate_g3_same_execution(
        timing_execution_id="exec-1",
        quality_execution_id="exec-1",
        telemetry_execution_id="exec-1",
        timed_outputs=golden_t2.ref_actions,
        audited_outputs=golden_t2.ref_actions + 0.01,
    )
    self.assertFalse(g3_tampered.passed)

    # F. Gate G4 (TEST04 Perturbed-Input Cache & Memoization Detection)
    ref_base = golden_t2.ref_actions
    ref_pert = np.clip(ref_base + 1e-3, -1.0, 1.0)
    cand_base = ref_base + 0.0005
    cand_pert_good = np.clip(cand_base + 0.95e-3, -1.0, 1.0)
    g4_good = evaluate_gate_g4_cache_memoization(
        base_outputs=cand_base,
        perturbed_outputs=cand_pert_good,
        ref_base_outputs=ref_base,
        ref_perturbed_outputs=ref_pert,
    )
    self.assertTrue(g4_good.passed)
    self.assertFalse(g4_good.cache_hit_suspected)

    # Positive violation: memoized cache cheat returns identical output on micro-perturbed input
    g4_cached_cheat = evaluate_gate_g4_cache_memoization(
        base_outputs=cand_base,
        perturbed_outputs=cand_base.copy(),
        ref_base_outputs=ref_base,
        ref_perturbed_outputs=ref_pert,
    )
    self.assertFalse(g4_cached_cheat.passed)
    self.assertTrue(g4_cached_cheat.cache_hit_suspected)

    # G. CLOSED vs OPEN Division Validator
    closed_valid = RecipeConfig(
        recipe_id="closed_valid_vla",
        track=TrackType.TRACK2_VLA.value,
        model_family="OpenVLA-OFT",
        model_name="OpenVLA-OFT-7B-FP16",
        baseline_model_id="OpenVLA-OFT-7B",
        precision="FP16",
        hardware_sku="JETSON_ORIN_NX_16GB",
        division=DivisionType.CLOSED.value,
        optimizations=["tensorrt_fp16_kernel_fusion", "cudagraph_capture"],
    )
    self.assertEqual(validate_division_compliance(closed_valid), [])

    with self.assertRaises(ValueError):
      RecipeConfig(
          recipe_id="closed_illegal_distill",
          track=TrackType.TRACK1_WORLD_MODEL.value,
          model_family="DreamDojo",
          model_name="DreamDojo-2B-4Step",
          baseline_model_id="DreamDojo-2B",
          precision="FP8",
          hardware_sku="H100_SXM5",
          division=DivisionType.CLOSED.value,
          denoising_steps=4,
          reference_denoising_steps=25,
          optimizations=["consistency_distillation_4step"],
      ).validate()

    # H. Static AST Hot-Loop Device Sync Auditor & Receipt Tampering
    bad_loop_code = (
        "def rollout(steps, x):\n"
        "  for _ in range(steps):\n"
        "    val = x.item()\n"
        "  return val\n"
    )
    good_nonloop_code = (
        "def summarize(x):\n"
        "  final_scalar = x.item()\n"
        "  return final_scalar\n"
    )
    self.assertEqual(len(audit_hot_loop_device_sync(bad_loop_code)), 1)
    self.assertEqual(len(audit_hot_loop_device_sync(good_nonloop_code)), 0)

    report = run_stage1_from_recipe_path(RECIPES_DIR / "track1_vla_pi05_edge100t.yaml")
    receipt = create_submission_receipt(report)
    ok_untampered, _ = verify_submission_receipt(receipt)
    self.assertTrue(ok_untampered)

    tampered = json.loads(json.dumps(receipt))
    tampered["attested_body"]["frontier_points"][0]["y_throughput"] = 999999.0
    ok_tampered, msg_tampered = verify_submission_receipt(tampered)
    self.assertFalse(ok_tampered)
    self.assertIn("SHA-256 mismatch", msg_tampered)

  # -------------------------------------------------------------------------
  # LIBERO-RT Latency Coupling, Execution Strategies & Safety Probes
  # -------------------------------------------------------------------------
  def test_libero_rt_latency_coupling_and_safety_probes(self) -> None:
    # 1. Discrete step delay d = ceil(L / delta_t) at delta_t = 50.0 ms (20 Hz)
    self.assertEqual(compute_delay_steps(0.0, 50.0), 0)
    self.assertEqual(compute_delay_steps(42.0, 50.0), 1)
    self.assertEqual(compute_delay_steps(50.0, 50.0), 1)
    self.assertEqual(compute_delay_steps(51.0, 50.0), 2)
    self.assertEqual(compute_delay_steps(185.0, 50.0), 4)

    # 2. Enforce chunk_horizon_s <= 1.0 cap
    self.assertAlmostEqual(validate_chunk_horizon(chunk_size=8, delta_t_ms=50.0), 0.40)
    with self.assertRaises(ValueError):
      validate_chunk_horizon(chunk_size=24, delta_t_ms=50.0)  # 1.20s > 1.0s cap

    # 3. Compare SYNCHRONOUS_HOLD vs ASYNC_CHUNK vs RTC_INPAINTING at L=160ms (d=4 steps)
    raw_recipe = load_yaml_or_json(RECIPES_DIR / "track1_vla_pi05_edge100t.yaml")

    r_sync = RecipeConfig.from_dict({**raw_recipe, "execution_strategy": ExecutionStrategy.SYNCHRONOUS_HOLD.value})
    r_async = RecipeConfig.from_dict({**raw_recipe, "execution_strategy": ExecutionStrategy.ASYNC_CHUNK.value})
    r_rtc = RecipeConfig.from_dict({**raw_recipe, "execution_strategy": ExecutionStrategy.RTC_INPAINTING.value})

    res_sync = run_libero_rt_evaluation(r_sync, policy_latency_mean_ms=160.0, policy_latency_p99_ms=190.0, chunk_size=8)
    res_async = run_libero_rt_evaluation(r_async, policy_latency_mean_ms=160.0, policy_latency_p99_ms=190.0, chunk_size=8)
    res_rtc = run_libero_rt_evaluation(r_rtc, policy_latency_mean_ms=160.0, policy_latency_p99_ms=190.0, chunk_size=8)

    self.assertEqual(res_sync.delay_steps_mean, 4)
    # RTC_INPAINTING > ASYNC_CHUNK > SYNCHRONOUS_HOLD on closed-loop realtime success & perturbation success
    self.assertGreater(res_rtc.realtime_success_rate, res_async.realtime_success_rate)
    self.assertGreater(res_async.realtime_success_rate, res_sync.realtime_success_rate)
    self.assertGreater(res_rtc.perturbation_success_rate, res_sync.perturbation_success_rate)
    self.assertLess(res_rtc.reaction_time_ms, res_sync.reaction_time_ms)

    # Verify all 3 safety probes are populated
    for probe_key in (
        SafetyProbeType.OBJECT_DISPLACEMENT.value,
        SafetyProbeType.GOAL_SWITCH.value,
        SafetyProbeType.EMERGENCY_STOP.value,
    ):
      self.assertIn(probe_key, res_rtc.safety_probes)

    # 4. Simulator-time budget (1.5x p90 demo duration) exceeded when synchronous latency is extreme
    res_timeout = run_libero_rt_evaluation(
        r_sync,
        policy_latency_mean_ms=320.0,
        policy_latency_p99_ms=380.0,
        chunk_size=4,
    )
    self.assertTrue(res_timeout.sim_time_budget_exceeded)
    self.assertFalse(res_timeout.passed)

    # 5. Tier 3 WebSocket HIL consistency check (+-10% tolerance)
    t3_ok = verify_tier3_hil_consistency(tier12_realtime_sr_pct=91.0, tier3_hil_realtime_sr_pct=88.5)
    self.assertTrue(t3_ok.consistent)
    t3_fail = verify_tier3_hil_consistency(tier12_realtime_sr_pct=91.0, tier3_hil_realtime_sr_pct=74.0)
    self.assertFalse(t3_fail.consistent)

  # -------------------------------------------------------------------------
  # Unified Scenarios (P1, P2, P3, S1, S2) & EEB Measurement Protocol
  # -------------------------------------------------------------------------
  def test_unified_scenarios_p1_p2_p3_s1_s2_and_measurement_protocol(self) -> None:
    # 1. Tegrastats INA3221 rail parser & Joules/decision
    sample = parse_tegrastats_line(
        "RAM 8420/15656MB (lfb 124x4MB)VDD_IN 18450mW/17920mW VDD_CPU_GPU_CV 13200mW/12800mW VDD_SOC 3100mW/3050mW"
    )
    self.assertAlmostEqual(sample.vdd_in_watts, 18.45, places=2)
    self.assertAlmostEqual(sample.vdd_cpu_gpu_cv_watts, 13.20, places=2)
    j_act, j_paced = compute_joules_per_decision(
        active_power_watts=18.45,
        idle_power_watts=6.5,
        decision_latency_ms=40.0,
        paced_freq_hz=10.0,
    )
    # Active: 18.45W * 0.04s = 0.738J; Paced 10Hz (100ms): 0.738J + 6.5W * 0.06s (0.39J) = 1.128J
    self.assertAlmostEqual(j_act, 0.738, places=3)
    self.assertAlmostEqual(j_paced, 1.128, places=3)

    # 2. EEB 3-run measurement protocol (<5% cross-run variance check)
    proto_ok = run_eeb_measurement_protocol(
        hardware_sku="JETSON_ORIN_NX_16GB",
        base_latency_mean_ms=42.0,
        base_latency_p50_ms=41.0,
        base_latency_p99_ms=46.0,
        active_power_watts=19.5,
        power_class=PowerClass.E1.value,
    )
    self.assertTrue(proto_ok.passed)
    self.assertTrue(proto_ok.power_class_compliant)
    self.assertLess(proto_ok.cross_run_variance_pct, 5.0)

    # Positive violation: thermal throttling >5% cross-run spread or E1 >25W power violation
    proto_bad = run_eeb_measurement_protocol(
        hardware_sku="JETSON_ORIN_NX_16GB",
        base_latency_mean_ms=42.0,
        base_latency_p50_ms=41.0,
        base_latency_p99_ms=46.0,
        active_power_watts=28.5,  # exceeds 25W E1 ceiling
        power_class=PowerClass.E1.value,
        per_run_mean_overrides_ms=[40.0, 42.0, 48.0],  # >19% spread
    )
    self.assertFalse(proto_bad.passed)
    self.assertFalse(proto_bad.cross_run_variance_passed)
    self.assertFalse(proto_bad.power_class_compliant)

    # 3. P1_ONBOARD (Jetson Orin NX 16GB E1)
    r0_cfg = RecipeConfig.from_dict(load_yaml_or_json(RECIPES_DIR / "track1_vla_pi05_orin_nx.yaml"))
    p1_metrics = evaluate_p1_onboard(r0_cfg)
    self.assertEqual(p1_metrics["scenario"], ScenarioType.P1_ONBOARD.value)
    self.assertEqual(p1_metrics["power_class"], PowerClass.E1.value)
    self.assertTrue(p1_metrics["power_class_compliant"])
    self.assertTrue(p1_metrics["best_efficiency_award_eligible"])

    # 4. P2_FLEET (Server C in [1..128])
    p2_cfg = RecipeConfig.from_dict({
        **load_yaml_or_json(RECIPES_DIR / "track1_vla_pi05_tpu_v6e.yaml"),
        "scenario": ScenarioType.P2_FLEET.value,
    })
    p2_metrics = evaluate_p2_fleet(p2_cfg)
    self.assertEqual(p2_metrics["scenario"], ScenarioType.P2_FLEET.value)
    self.assertIn("10Hz_50ms", p2_metrics["max_robots_per_chip_at_slo"])
    self.assertIn("5Hz_100ms", p2_metrics["max_robots_per_chip_at_slo"])
    self.assertIn("2Hz_250ms", p2_metrics["max_robots_per_chip_at_slo"])
    self.assertGreater(p2_metrics["robots_per_chip_at_primary_slo"], 0)
    self.assertGreater(p2_metrics["robots_per_kw"], 0.0)
    self.assertIsNotNone(p2_metrics["dollars_per_robot_hour"])

    # 5. P3_EDGE_CLOUD (SingleStream with network RTT/jitter/drop)
    p3_cfg = RecipeConfig.from_dict({
        **load_yaml_or_json(RECIPES_DIR / "track1_vla_pi05_edge100t.yaml"),
        "scenario": ScenarioType.P3_EDGE_CLOUD.value,
        "network_rtt_ms": 22.0,
        "network_jitter_ms": 4.0,
        "network_drop_rate": 0.01,
    })
    p3_metrics = evaluate_p3_edge_cloud(p3_cfg)
    self.assertEqual(p3_metrics["scenario"], ScenarioType.P3_EDGE_CLOUD.value)
    self.assertGreater(p3_metrics["effective_edge_cloud_ttfa_p99_ms"], p3_metrics["device_ttfa_p99_ms"])

    # 6. S1_INTERACTIVE_WM (Streams per chip at 10fps & 24fps over 30-60s sessions)
    s1_cfg = RecipeConfig.from_dict(load_yaml_or_json(RECIPES_DIR / "track2_wm_cosmos2b_workstation350t.yaml"))
    s1_metrics = evaluate_s1_interactive_wm(s1_cfg)
    self.assertEqual(s1_metrics["scenario"], ScenarioType.S1_INTERACTIVE_WM.value)
    self.assertGreaterEqual(s1_metrics["streams_per_chip_at_10fps"], s1_metrics["streams_per_chip_at_24fps"])
    self.assertGreater(s1_metrics["streams_per_chip_at_10fps"], 0)

    # 7. S2_WM_EVALUATOR (Offline ranking of 12 policies via Kendall's tau >= 0.80)
    s2_good = evaluate_s2_wm_evaluator(s1_cfg)
    self.assertTrue(s2_good.passed)
    self.assertEqual(s2_good.num_policies_evaluated, len(DEFAULT_S2_POLICY_POOL))
    self.assertGreaterEqual(s2_good.kendall_tau, 0.80)
    self.assertGreater(s2_good.dollars_per_policy_evaluated, 0.0)

    # Positive violation: reversed/noisy ranking fails kendall_tau >= 0.80 gate
    reversed_scores = [float(p["gt_sim_success_pct"]) for p in reversed(DEFAULT_S2_POLICY_POOL)]
    s2_bad = evaluate_s2_wm_evaluator(s1_cfg, override_wm_scores=reversed_scores)
    self.assertFalse(s2_bad.passed)
    self.assertLess(s2_bad.kendall_tau, 0.0)

    tau_perfect, conc, disc, _, _, _ = compute_kendall_tau([1, 2, 3, 4], [10, 20, 30, 40])
    self.assertAlmostEqual(tau_perfect, 1.0, places=4)
    self.assertEqual(conc, 6)
    self.assertEqual(disc, 0)

  # -------------------------------------------------------------------------
  # Pareto Frontier Domination Stripping & Log-Linear Iso-Reactivity Slicing
  # -------------------------------------------------------------------------
  def test_pareto_frontier_and_iso_reactivity_interpolation(self) -> None:
    points = [
        {"point_id": "p1", "hz_plan": 50.0, "fps": 100.0, "rollout_latency_p99_ms": 20.0, "gate_a_passed": True},
        {"point_id": "p_dom1", "hz_plan": 40.0, "fps": 90.0, "rollout_latency_p99_ms": 25.0, "gate_a_passed": True},
        {"point_id": "p2", "hz_plan": 20.0, "fps": 400.0, "rollout_latency_p99_ms": 50.0, "gate_a_passed": True},
        {"point_id": "p_inadmissible", "hz_plan": 60.0, "fps": 900.0, "rollout_latency_p99_ms": 16.6, "gate_a_passed": False},
        {"point_id": "p3", "hz_plan": 10.0, "fps": 900.0, "rollout_latency_p99_ms": 100.0, "gate_a_passed": True},
    ]
    frontier = extract_pareto_frontier(
        operating_points=points,
        x_key="hz_plan",
        y_key="fps",
        require_gate_a=True,
    )
    self.assertEqual([p["point_id"] for p in frontier], ["p1", "p2", "p3"])

    slices = compute_iso_reactivity_slices(
        frontier_points=frontier,
        latency_key="rollout_latency_p99_ms",
        throughput_key="fps",
    )
    self.assertAlmostEqual(slices["Iso-50Hz"] or 0.0, 100.0, places=3)
    self.assertAlmostEqual(slices["Iso-20Hz"] or 0.0, 400.0, places=3)
    self.assertAlmostEqual(slices["Iso-10Hz"] or 0.0, 900.0, places=3)
    self.assertIsNotNone(slices["Iso-30Hz"])
    self.assertGreater(slices["Iso-30Hz"], 100.0)
    self.assertLess(slices["Iso-30Hz"], 400.0)

    alpha = (math.log(1000.0 / 30.0) - math.log(20.0)) / (math.log(50.0) - math.log(20.0))
    expected_30hz = round(math.exp((1.0 - alpha) * math.log(100.0) + alpha * math.log(400.0)), 4)
    self.assertAlmostEqual(slices["Iso-30Hz"], expected_30hz, places=4)

    self.assertIsNone(
        interpolate_log_linear_at_latency(
            latencies_ms=[40.0, 80.0],
            throughputs=[120.0, 240.0],
            target_latency_ms=20.0,
        )
    )

  # -------------------------------------------------------------------------
  # All 10 Seed Recipes (Including CoRL 2026 VLA_TRACK & WM_TRACK), Golden Archive I/O & Schema Metadata
  # -------------------------------------------------------------------------
  def test_all_six_seed_recipes_and_golden_archive_io(self) -> None:
    expected_recipes = [
        "track1_vla_pi05_orin_nx.yaml",
        "track1_vla_pi05_edge100t.yaml",
        "track1_vla_gr00t_edge100t.yaml",
        "track2_wm_cosmos2b_workstation350t.yaml",
        "track1_vla_pi05_edge.yaml",
        "track1_vla_pi05_tpu_v6e.yaml",
        "track1_vla_gr00t_n17_edge.yaml",
        "track1_vla_gr00t_n17_tpu_v6e.yaml",
        "track2_wm_cosmos_predict25_edge.yaml",
        "track2_wm_cosmos_predict25_tpu_v6e.yaml",
    ]
    for fname in expected_recipes:
      rpath = RECIPES_DIR / fname
      self.assertTrue(rpath.exists(), msg=f"Missing recipe {rpath}")
      report = run_stage1_from_recipe_path(rpath)
      self.assertEqual(report["status"], "CERTIFIED", msg=f"Recipe {fname} failed: {report.get('reason')}")
      self.assertGreater(len(report["pareto_frontier"]), 0)
      self.assertTrue(report["stage2_gate_b"]["passed"])
      self.assertTrue(report["compliance_gates_g1_g4"]["passed"])
      self.assertTrue(report["measurement_protocol"]["passed"])
      self.assertTrue(report["performance_cap_evaluation"]["passed"])

    with tempfile.TemporaryDirectory() as tmpdir:
      npz_file = Path(tmpdir) / "golden_batches.npz"
      save_golden_archive_npz(npz_file)
      loaded = load_golden_archive_npz(npz_file)
      self.assertIn("t1_obs_frames", loaded)
      self.assertIn("t2_ref_actions", loaded)

    self.assertGreaterEqual(len(HARDWARE_SKU_SPECS), 11)
    self.assertIn("JETSON_ORIN_NX_16GB", HARDWARE_SKU_SPECS)
    self.assertIn("RTX_4060_TI_16GB", HARDWARE_SKU_SPECS)
    self.assertGreaterEqual(len(SCHEMA_FIELD_EXAMPLES), 8)
    ts = now_pacific_iso()
    self.assertTrue(ts.endswith("(PDT)") or ts.endswith("(PST)") or ts.endswith("(PT)"))
    mask = is_pareto_efficient(np.array([[10.0, 100.0], [20.0, 200.0]]))
    self.assertEqual(mask.tolist(), [False, True])

  # -------------------------------------------------------------------------
  # Modern 2025/2026 VLA & World Model Baselines + Device-Agnostic PerformanceCapEnvelopes
  # -------------------------------------------------------------------------
  def test_modern_vla_wm_baselines_and_performance_cap_envelopes(self) -> None:
    # 1. Verify modern 2025/2026 reference checkpoints exist for VLA_TRACK and WM_TRACK
    for ckpt in (
        "pi05-3b",
        "gr00t-n1.5-3b",
        "openvla-oft-7b",
        "smolvla-450m",
        "cosmos-predict2-2b-action",
        "wan2.1-wm-1.3b",
        "ctrl-world-1.5b",
    ):
      self.assertIn(ckpt, MODERN_REFERENCE_CHECKPOINTS)

    self.assertTrue(is_vla_track(TrackType.VLA_TRACK.value))
    self.assertTrue(is_vla_track(TrackType.TRACK1_VLA.value))
    self.assertTrue(is_vla_track(TrackType.TRACK2_VLA.value))
    self.assertTrue(is_wm_track(TrackType.WM_TRACK.value))
    self.assertTrue(is_wm_track(TrackType.TRACK2_WM.value))
    self.assertTrue(is_wm_track(TrackType.TRACK1_WORLD_MODEL.value))

    # Negative control: Deprecated 2024 7-token-AR vanilla OpenVLA must be rejected
    for deprecated_id in DEPRECATED_BASELINE_MODEL_IDS:
      with self.assertRaises(ValueError):
        RecipeConfig(
            recipe_id="stale_vanilla_openvla",
            track=TrackType.VLA_TRACK.value,
            model_family="OpenVLA",
            model_name="OpenVLA-7B-2024",
            baseline_model_id=deprecated_id,
            precision="BF16",
            hardware_sku="RTX_4090",
        ).validate()

    # 2. Verify 3-Part World Model Open-Division Quality Gate (positive & negative controls)
    wm3_good = evaluate_wm_open_3part_quality_gate(
        psnr_db=28.5,
        lpips=0.085,
        fvd=62.0,
        inverse_dynamics_action_cosine_sim=0.994,
        long_horizon_drift_ratio_30_60s=1.12,
        s2_policy_ranking_kendall_tau=0.86,
    )
    self.assertTrue(wm3_good.passed)
    self.assertTrue(wm3_good.visual_fidelity_passed)
    self.assertTrue(wm3_good.action_controllability_passed)
    self.assertTrue(wm3_good.long_horizon_and_ranking_passed)

    wm3_bad = evaluate_wm_open_3part_quality_gate(
        psnr_db=21.0,  # < 24.0 dB
        lpips=0.15,    # > 0.12
        fvd=140.0,     # > 120.0
        inverse_dynamics_action_cosine_sim=0.88,  # < 0.95
        long_horizon_drift_ratio_30_60s=1.35,     # > 1.25
        s2_policy_ranking_kendall_tau=0.65,       # < 0.80
    )
    self.assertFalse(wm3_bad.passed)
    self.assertFalse(wm3_bad.visual_fidelity_passed)
    self.assertFalse(wm3_bad.action_controllability_passed)
    self.assertFalse(wm3_bad.long_horizon_and_ranking_passed)
    self.assertGreaterEqual(len(wm3_bad.violations), 5)

    # 3. Verify PerformanceCapEnvelopes (CAP_EDGE_100T / SINGLE-EDGE, CAP_WORKSTATION_350T / SINGLE-ACCELERATOR-STD, CAP_DATACENTER_1000T / SINGLE-ACCELERATOR-MAX)
    edge_cap = get_performance_cap_envelope(PerformanceCapId.CAP_EDGE_100T)
    self.assertEqual(edge_cap.max_fp16_tflops, 100.0)
    self.assertEqual(edge_cap.max_int8_fp8_tops, 200.0)
    self.assertEqual(edge_cap.max_mem_bw_gbps, 250.0)
    self.assertEqual(edge_cap.max_vram_gb, 24.0)
    self.assertEqual(edge_cap.max_num_devices, 1)
    self.assertEqual(edge_cap.tier_alias, "SINGLE-EDGE")

    ws_cap = get_performance_cap_envelope(PerformanceCapId.CAP_WORKSTATION_350T)
    self.assertEqual(ws_cap.max_fp16_tflops, 400.0)
    self.assertEqual(ws_cap.max_int8_fp8_tops, 800.0)
    self.assertEqual(ws_cap.max_mem_bw_gbps, 1000.0)
    self.assertEqual(ws_cap.max_vram_gb, 48.0)
    self.assertEqual(ws_cap.max_num_devices, 1)
    self.assertEqual(ws_cap.tier_alias, "SINGLE-ACCELERATOR-STD")

    dc_cap = get_performance_cap_envelope(PerformanceCapId.CAP_DATACENTER_1000T)
    self.assertEqual(dc_cap.max_fp16_tflops, 1500.0)
    self.assertEqual(dc_cap.max_int8_fp8_tops, 3000.0)
    self.assertEqual(dc_cap.max_mem_bw_gbps, 5300.0)
    self.assertEqual(dc_cap.max_vram_gb, 192.0)
    self.assertEqual(dc_cap.max_num_devices, 1)
    self.assertEqual(dc_cap.tier_alias, "SINGLE-ACCELERATOR-MAX")

    # 4. Verify Roofline-Normalized Effective Latency formula:
    # L_norm = L_phys * max(1.0, min(TFLOPS_eff / TFLOPS_cap, AI_op * (BW_eff / BW_cap)))
    # Case A: Native qualifying device (88 TFLOPS <= 100, 240 GB/s <= 250) -> norm_factor = 1.0
    l_norm_native, factor_native = compute_roofline_normalized_latency(
        physical_latency_ms=30.0,
        tflops_eff=88.0,
        tflops_cap=100.0,
        bw_eff_gbps=240.0,
        bw_cap_gbps=250.0,
        ai_op=1.0,
    )
    self.assertAlmostEqual(factor_native, 1.0, places=4)
    self.assertAlmostEqual(l_norm_native, 30.0, places=3)

    # Case B: Larger uncapped GPU (RTX 4090: 330 TFLOPS, 1000 GB/s) evaluated against CAP_EDGE_100T (100 TFLOPS, 250 GB/s)
    # min(330/100 = 3.30, 1.0 * 1000/250 = 4.00) = 3.30 -> L_norm = 20.0 * 3.30 = 66.0 ms
    l_norm_scaled, factor_scaled = compute_roofline_normalized_latency(
        physical_latency_ms=20.0,
        tflops_eff=330.0,
        tflops_cap=100.0,
        bw_eff_gbps=1000.0,
        bw_cap_gbps=250.0,
        ai_op=1.0,
    )
    self.assertAlmostEqual(factor_scaled, 3.30, places=4)
    self.assertAlmostEqual(l_norm_scaled, 66.0, places=3)

    # 5. Test all 3 PerformanceCapEnvelope satisfaction modes via evaluate_performance_cap_envelope & CLI
    pi05_recipe_path = RECIPES_DIR / "track1_vla_pi05_edge100t.yaml"
    pi05_cfg = RecipeConfig.from_dict(load_yaml_or_json(pi05_recipe_path))

    # Mode 1: Native Qualifying Device (RTX_4060_TI_16GB has 88 TFLOPS, 240 GB/s, 16 GB)
    eval_m1 = evaluate_performance_cap_envelope(
        recipe=pi05_cfg,
        model_params_b=3.3,
        batch_or_concurrency=1,
        horizon_or_chunk=8,
        physical_latency_mean_ms=24.0,
        physical_latency_p50_ms=23.5,
        physical_latency_p99_ms=26.0,
        peak_vram_mb=4200.0,
        perf_cap_override="CAP_EDGE_100T",
        perf_cap_mode_override="native_qualifying_device",
    )
    self.assertTrue(eval_m1.passed)
    self.assertTrue(eval_m1.satisfied_natively)
    self.assertEqual(eval_m1.roofline_normalization_factor, 1.0)
    self.assertIsNotNone(eval_m1.hardware_independent_profile["flops_per_action_gflops"])
    self.assertGreater(eval_m1.hardware_independent_profile["bytes_moved_per_step_mb"], 0.0)

    # Mode 1 Negative Control: H100_SXM5 fails native_qualifying_device under CAP_EDGE_100T
    h100_cfg = RecipeConfig.from_dict({**load_yaml_or_json(pi05_recipe_path), "hardware_sku": "H100_SXM5"})
    eval_m1_fail = evaluate_performance_cap_envelope(
        recipe=h100_cfg,
        model_params_b=3.3,
        batch_or_concurrency=1,
        horizon_or_chunk=8,
        physical_latency_mean_ms=10.0,
        physical_latency_p50_ms=9.8,
        physical_latency_p99_ms=11.0,
        peak_vram_mb=4200.0,
        perf_cap_override="CAP_EDGE_100T",
        perf_cap_mode_override="native_qualifying_device",
    )
    self.assertFalse(eval_m1_fail.passed)

    # Mode 2: Hardware-Enforced Slice / Clock Cap on H100_SXM5 (7% MPS + 0.20 VRAM cap fits inside CAP_EDGE_100T)
    eval_m2 = evaluate_performance_cap_envelope(
        recipe=h100_cfg,
        model_params_b=3.3,
        batch_or_concurrency=1,
        horizon_or_chunk=8,
        physical_latency_mean_ms=28.0,
        physical_latency_p50_ms=27.5,
        physical_latency_p99_ms=30.0,
        peak_vram_mb=4200.0,
        perf_cap_override="CAP_EDGE_100T",
        perf_cap_mode_override="hardware_enforced_slice_clock_cap",
        mps_active_thread_pct_override=7.0,
        clock_scale_factor_override=1.0,
        vram_fraction_cap_override=0.20,
    )
    self.assertTrue(eval_m2.passed)
    self.assertTrue(eval_m2.satisfied_via_slice_or_clock_cap)
    self.assertEqual(eval_m2.roofline_normalization_factor, 1.0)
    cmds = generate_slice_clock_cap_commands("H100_SXM5", "CAP_EDGE_100T")
    self.assertIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", cmds["cuda_mps_env"])

    # Mode 3: Roofline-Normalized Latency on uncapped H100_SXM5 (with active peak VRAM <= 16GB)
    eval_m3 = evaluate_performance_cap_envelope(
        recipe=h100_cfg,
        model_params_b=3.3,
        batch_or_concurrency=1,
        horizon_or_chunk=8,
        physical_latency_mean_ms=9.5,
        physical_latency_p50_ms=9.2,
        physical_latency_p99_ms=10.5,
        peak_vram_mb=4200.0,
        perf_cap_override="CAP_EDGE_100T",
        perf_cap_mode_override="roofline_normalized_latency",
    )
    self.assertTrue(eval_m3.passed)
    self.assertTrue(eval_m3.satisfied_via_roofline_normalization)
    self.assertGreater(eval_m3.roofline_normalization_factor, 1.0)
    self.assertGreater(eval_m3.normalized_latency_p99_ms, eval_m3.physical_latency_p99_ms)

    # 6. Verbatim CLI execution of `eeb measure --perf-cap ...` in both flag orders
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      out_pi05 = str(tmp / "pi05_edge100t.json")
      out_cosmos = str(tmp / "cosmos_ws350t.json")
      cosmos_recipe_path = str(RECIPES_DIR / "track2_wm_cosmos2b_workstation350t.yaml")

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_c1:
        rc_c1 = cli_main([
            "measure",
            "--recipe",
            str(pi05_recipe_path),
            "--perf-cap",
            "CAP_EDGE_100T",
            "--output",
            out_pi05,
        ])
      self.assertEqual(rc_c1, 0)
      c1_data = json.loads(buf_c1.getvalue())
      self.assertEqual(c1_data["perf_cap"], "CAP_EDGE_100T")
      self.assertEqual(c1_data["perf_cap_mode"], "native_qualifying_device")
      self.assertTrue(c1_data["perf_cap_satisfied"])
      self.assertIsNotNone(c1_data["flops_per_action_gflops"])

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_c2:
        rc_c2 = cli_main([
            "measure",
            "--output",
            out_cosmos,
            "--perf-cap",
            "CAP_WORKSTATION_350T",
            "--recipe",
            cosmos_recipe_path,
        ])
      self.assertEqual(rc_c2, 0)
      c2_data = json.loads(buf_c2.getvalue())
      self.assertEqual(c2_data["perf_cap"], "CAP_WORKSTATION_350T")
      self.assertTrue(c2_data["perf_cap_satisfied"])
      self.assertIsNotNone(c2_data["flops_per_frame_gflops"])

  def test_canonical_4model_lineup_single_device_envelopes_and_embodiedx_traces(self) -> None:
    """Verify the 3 Canonical EEB v1.0 Challenge Models, Single-Device (TOPS, HBM) envelopes across NVIDIA/Google Cloud TPU/AMD/Custom ASICs, Cross-Platform Telemetry, and EmbodiedX-Traces-v1."""
    # 1. Verify the 3 Canonical EEB v1.0 Challenge Models (VLA + Action-Conditioned WM)
    self.assertEqual(
        CANONICAL_CHALLENGE_MODELS,
        ("pi05-3b", "gr00t-n1.7-3b", "cosmos-predict2.5-2b"),
    )
    for model_id in CANONICAL_CHALLENGE_MODELS:
      self.assertIn(model_id, MODERN_REFERENCE_CHECKPOINTS)
      self.assertIn(model_id, ADAPTER_REGISTRY)

    # Verify extensible model registration without editing harness internals
    register_model_checkpoint(
        model_id="custom-vla-2b-test",
        hf_id="org/custom-vla-2b-test",
        track="VLA_TRACK",
        architecture="2B Custom Flow-Matching VLA",
        reference_steps=8,
        native_chunk_or_horizon=16,
        license_id="Apache-2.0",
    )
    self.assertIn("custom-vla-2b-test", MODERN_REFERENCE_CHECKPOINTS)
    register_model_adapter("custom-vla-2b-test", ModelAdapter, params_b=2.0)
    self.assertIn("custom-vla-2b-test", ADAPTER_REGISTRY)

    # 2. Verify Single-Device Rule (`num_devices == 1`) positive & negative controls
    valid_single_dev = RecipeConfig(
        recipe_id="valid_1dev",
        track="VLA_TRACK",
        model_family="gr00t-n1.7",
        model_name="GR00T-N1.7-3B-FP8",
        baseline_model_id="gr00t-n1.7-3b",
        precision="FP8",
        hardware_sku="TPU_V6E_1_32GB",
        perf_cap="SINGLE-EDGE",
        num_devices=1,
    )
    valid_single_dev.validate()
    self.assertEqual(valid_single_dev.resolved_perf_cap(), "CAP_EDGE_100T")

    with self.assertRaises(ValueError):
      RecipeConfig(
          recipe_id="invalid_multidev",
          track="VLA_TRACK",
          model_family="gr00t-n1.7",
          model_name="GR00T-N1.7-3B-MultiGPU",
          baseline_model_id="gr00t-n1.7-3b",
          precision="FP8",
          hardware_sku="H100_SXM5_80GB",
          perf_cap="SINGLE-ACCELERATOR-MAX",
          num_devices=2,  # Multi-device (>1 chip) is forbidden
      ).validate()

    # 3. Verify Unified Single-Device Tier (`CAP_SINGLE_CHIP` / `SINGLE_CHIP_UNIFIED`), 32GB Parity Sub-Cap (`CAP_SINGLE_CHIP_32GB`), Legacy Aliases & Cross-Platform SKUs
    self.assertEqual(resolve_perf_cap_id("SINGLE_CHIP_UNIFIED"), "CAP_SINGLE_CHIP")
    self.assertEqual(resolve_perf_cap_id("SINGLE-CHIP"), "CAP_SINGLE_CHIP")
    self.assertEqual(resolve_perf_cap_id("SINGLE_CHIP_32GB"), "CAP_SINGLE_CHIP_32GB")
    self.assertEqual(resolve_perf_cap_id("SINGLE-EDGE"), "CAP_EDGE_100T")
    self.assertEqual(resolve_perf_cap_id("SINGLE-ACCELERATOR-STD"), "CAP_WORKSTATION_350T")
    self.assertEqual(resolve_perf_cap_id("SINGLE-ACCELERATOR-MAX"), "CAP_DATACENTER_1000T")

    unified_cap = get_performance_cap_envelope(PerformanceCapId.CAP_SINGLE_CHIP)
    self.assertEqual(unified_cap.tier_alias, "SINGLE_CHIP_UNIFIED")
    self.assertEqual(unified_cap.max_fp16_tflops, 1000.0)
    self.assertEqual(unified_cap.max_int8_fp8_tops, 2000.0)
    self.assertEqual(unified_cap.max_mem_bw_gbps, 2000.0)
    self.assertEqual(unified_cap.max_vram_gb, 64.0)
    self.assertEqual(unified_cap.max_num_devices, 1)

    subcap_32gb = get_performance_cap_envelope(PerformanceCapId.CAP_SINGLE_CHIP_32GB)
    self.assertEqual(subcap_32gb.tier_alias, "SINGLE_CHIP_32GB")
    self.assertEqual(subcap_32gb.max_vram_gb, 32.0)
    self.assertEqual(subcap_32gb.max_num_devices, 1)

    for sku in (
        "JETSON_ORIN_NX_16GB",
        "JETSON_AGX_THOR_64GB",
        "RTX_4090_24GB",
        "RTX_5090_32GB",
        "L40S_48GB",
        "AMD_MI210_64GB",
        "AMD_W7900_48GB",
        "AMD_RX_7900_XTX_24GB",
        "AMD_MI250_1GCD_64GB",
        "TPU_V6E_1_32GB",
        "TPU_V5E_1_16GB",
        "CUSTOM_ASIC_SINGLE_DEVICE",
    ):
      self.assertIn(sku, HARDWARE_SKU_SPECS)

    # Verify that right-sized Physical-AI chips across NVIDIA, Google Cloud TPU, AMD, and Custom ASIC natively qualify under CAP_SINGLE_CHIP
    for matched_sku in (
        "JETSON_AGX_THOR_64GB",
        "RTX_5090_32GB",
        "L40S_48GB",
        "RTX_4090_24GB",
        "TPU_V6E_1_32GB",
        "TPU_V5E_1_16GB",
        "AMD_MI210_64GB",
        "AMD_W7900_48GB",
        "AMD_RX_7900_XTX_24GB",
        "AMD_MI250_1GCD_64GB",
        "CUSTOM_ASIC_SINGLE_DEVICE",
    ):
      matched_recipe = RecipeConfig(
          recipe_id=f"unified_{matched_sku.lower()}",
          track="VLA_TRACK",
          model_family="gr00t-n1.7",
          model_name="GR00T-N1.7-3B-FP8",
          baseline_model_id="gr00t-n1.7-3b",
          precision="FP8",
          hardware_sku=matched_sku,
          perf_cap="CAP_SINGLE_CHIP",
          num_devices=1,
      )
      cap_eval = evaluate_performance_cap_envelope(
          recipe=matched_recipe,
          model_params_b=3.0,
          batch_or_concurrency=1,
          horizon_or_chunk=16,
          physical_latency_mean_ms=15.0,
          physical_latency_p50_ms=14.5,
          physical_latency_p99_ms=16.5,
          peak_vram_mb=12000.0,
      )
      self.assertTrue(cap_eval.passed, f"{matched_sku} failed CAP_SINGLE_CHIP: {cap_eval.reason}")
      self.assertTrue(cap_eval.satisfied_natively, f"{matched_sku} did not satisfy CAP_SINGLE_CHIP natively")
      self.assertGreater(cap_eval.power_efficiency_per_watt, 0.0)
      self.assertGreater(cap_eval.tco_efficiency_per_dollar, 0.0)
      self.assertIn("throughput_per_100_tops", cap_eval.iso_compute_and_bw_efficiency)

    # 4. Verify Iso-TOPS (`throughput_per_100_tops`) and Iso-Bandwidth (`throughput_per_100_gbps_bw`) math
    iso_tops, iso_bw = compute_iso_tops_and_bw_efficiency(
        raw_single_device_throughput=320.0,
        effective_fp8_int8_tops=800.0,
        effective_mem_bw_gbps=1000.0,
    )
    self.assertAlmostEqual(iso_tops, 40.0, places=4)
    self.assertAlmostEqual(iso_bw, 32.0, places=4)

    # 5. Verify Cross-Platform Telemetry Parsers (AMD rocm-smi, Google Cloud TPU tpu-info, Custom ASIC JSON)
    rocm_csv = (
        "device,Card series,VRAM Total Memory (B),VRAM Total Used Memory (B),Average Graphics Package Power (W)\n"
        "card0,AMD Instinct MI300X,206158430208,17179869184,285.5\n"
    )
    rocm_parsed = parse_rocm_smi_csv(rocm_csv)
    self.assertEqual(rocm_parsed["vendor"], "AMD_ROCM")
    self.assertAlmostEqual(rocm_parsed["mean_power_watts"], 285.5, places=2)
    self.assertAlmostEqual(rocm_parsed["peak_vram_mb"], 16384.0, places=1)
    with self.assertRaises(ValueError):
      parse_rocm_smi_csv("invalid_header_only")

    tpu_out = "Chip 0 | TPU v6e | 14.50 GiB / 32.00 GiB | Duty Cycle: 91.2% | Power: 168.0 W"
    tpu_parsed = parse_tpu_info_output(tpu_out)
    self.assertEqual(tpu_parsed["vendor"], "GOOGLE_TPU")
    self.assertAlmostEqual(tpu_parsed["mean_power_watts"], 168.0, places=2)
    self.assertAlmostEqual(tpu_parsed["peak_vram_mb"], 14.50 * 1024.0, places=1)
    with self.assertRaises(ValueError):
      parse_tpu_info_output("no metrics here")

    asic_json = json.dumps({"power_watts": 62.5, "used_hbm_mb": 6144.0, "total_hbm_mb": 32768.0})
    asic_parsed = parse_custom_asic_telemetry_json(asic_json)
    self.assertEqual(asic_parsed["vendor"], "CUSTOM_ASIC")
    self.assertAlmostEqual(asic_parsed["mean_power_watts"], 62.5, places=2)
    self.assertAlmostEqual(asic_parsed["peak_vram_mb"], 6144.0, places=1)
    with self.assertRaises(Exception):
      parse_custom_asic_telemetry_json("{bad json")

    # 6. Verify Open-Source Dataset Trace Metadata (`EmbodiedX-Traces-v1`)
    for ds_name in ("LIBERO", "DROID", "BridgeData-V2", "AgiBot-World"):
      self.assertIn(ds_name, EMBODIEDX_TRACES_V1_SOURCES)
    t1_batch = generate_track1_golden_batch(batch_size=2, horizon_frames=16)
    self.assertEqual(t1_batch.dataset_version, "EmbodiedX-Traces-v1")
    self.assertGreaterEqual(t1_batch.temporal_frame_coherence, 0.90)
    # Consecutive frames in temporally coherent rollout are correlated
    frame_diff = float(np.mean(np.abs(t1_batch.ref_video_latents[:, 1] - t1_batch.ref_video_latents[:, 0])))
    self.assertLess(frame_diff, 0.25)

    t2_batch = generate_track2_golden_batch(concurrency=4, chunk_size=8)
    self.assertEqual(t2_batch.dataset_version, "EmbodiedX-Traces-v1")
    self.assertEqual(t2_batch.instruction_prefix_reuse_ratio, 1.0)
    # First 24 instruction tokens are constant across concurrent calls in the episode
    self.assertTrue(np.array_equal(t2_batch.instruction_tokens[0, :24], t2_batch.instruction_tokens[1, :24]))

    # 7. Run `eeb measure` on all 4 canonical CAP_SINGLE_CHIP recipes and verify CERTIFIED status + CAP_SINGLE_CHIP + Iso-TOPS metrics
    canonical_recipes = (
        "track1_vla_gr00t_n17_edge.yaml",
        "track1_vla_pi05_edge.yaml",
        "track2_wm_cosmos_predict25_edge.yaml",
        "track2_wm_cosmos_predict25_tpu_v6e.yaml",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      for rec_name in canonical_recipes:
        rec_path = RECIPES_DIR / rec_name
        self.assertTrue(rec_path.exists(), f"Missing canonical recipe: {rec_path}")
        out_json = tmp / f"{rec_name}.json"
        with mock.patch("sys.stdout", new_callable=io.StringIO) as buf:
          rc = cli_main(["measure", "--recipe", str(rec_path), "--output", str(out_json)])
        self.assertEqual(rc, 0, f"Recipe {rec_name} failed: {buf.getvalue()}")
        summary = json.loads(buf.getvalue())
        self.assertEqual(summary["status"], "CERTIFIED")
        self.assertEqual(summary["perf_cap"], "CAP_SINGLE_CHIP")
        self.assertEqual(summary["perf_cap_tier_alias"], "SINGLE_CHIP_UNIFIED")
        self.assertTrue(summary["single_device_compliant"])
        self.assertEqual(summary["num_devices"], 1)
        self.assertGreater(summary["raw_single_device_throughput"], 0.0)
        self.assertGreater(summary["throughput_per_100_tops"], 0.0)
        self.assertGreater(summary["throughput_per_100_gbps_bw"], 0.0)

  def test_phase1_to_5_cli_subcommands_in_both_flag_orders(self) -> None:
    """Gate 3: Test `download-data`, `serve`, `benchmark`, `validate-submission`, and `export-website` CLI subcommands in both flag orders."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      cache_dir_a = str(tmp / "traces_a")
      cache_dir_b = str(tmp / "traces_b")

      # 1. download-data (Order A vs Order B)
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_dl_a:
        rc = cli_main([
            "download-data",
            "--cache-dir",
            cache_dir_a,
            "--mode",
            "smoke",
            "--num-episodes",
            "2",
            "--steps-per-episode",
            "6",
        ])
      self.assertEqual(rc, 0)
      dl_a = json.loads(buf_dl_a.getvalue())
      self.assertEqual(dl_a["status"], "CERTIFIED")
      self.assertGreater(dl_a["total_episodes"], 0)

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_dl_b:
        rc = cli_main([
            "download-data",
            "--steps-per-episode",
            "6",
            "--num-episodes",
            "2",
            "--mode",
            "smoke",
            "--cache-dir",
            cache_dir_b,
        ])
      self.assertEqual(rc, 0)
      dl_b = json.loads(buf_dl_b.getvalue())
      self.assertEqual(dl_b["total_episodes"], dl_a["total_episodes"])

      # 2. serve --self-test (Order A vs Order B)
      srv_out_a = str(tmp / "srv_a.json")
      srv_out_b = str(tmp / "srv_b.json")
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_srv_a:
        rc = cli_main([
            "serve",
            "--model",
            "gr00t-n1.7-3b",
            "--precision",
            "fp8",
            "--denoising-steps",
            "2",
            "--self-test",
            "--self-test-steps",
            "3",
            "--output",
            srv_out_a,
        ])
      self.assertEqual(rc, 0)
      srv_a = json.loads(buf_srv_a.getvalue())
      self.assertEqual(srv_a["model_id"], "gr00t-n1.7-3b")
      self.assertEqual(srv_a["health_status"], "ok")

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_srv_b:
        rc = cli_main([
            "serve",
            "--output",
            srv_out_b,
            "--self-test-steps",
            "3",
            "--self-test",
            "--denoising-steps",
            "2",
            "--precision",
            "fp8",
            "--model",
            "cosmos3-edge",
        ])
      self.assertEqual(rc, 0)
      srv_b = json.loads(buf_srv_b.getvalue())
      self.assertEqual(srv_b["model_id"], "cosmos3-edge")

      # 3. benchmark (Order A vs Order B)
      bench_out_a = str(tmp / "bench_a.json")
      bench_out_b = str(tmp / "bench_b.json")
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_b_a:
        rc = cli_main([
            "benchmark",
            "--model",
            "pi05-3b",
            "--hardware-sku",
            "TPU_V6E_1_32GB",
            "--concurrency",
            "1",
            "2",
            "--steps-per-stream",
            "3",
            "--denoising-steps",
            "2",
            "--output",
            bench_out_a,
        ])
      self.assertEqual(rc, 0)
      b_a = json.loads(buf_b_a.getvalue())
      self.assertEqual(b_a["status"], "CERTIFIED")
      self.assertEqual(b_a["num_operating_points"], 2)

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_b_b:
        rc = cli_main([
            "benchmark",
            "--output",
            bench_out_b,
            "--denoising-steps",
            "2",
            "--steps-per-stream",
            "3",
            "--hardware-sku",
            "AMD_MI210_64GB",
            "--model",
            "qwen3-omni-30b-a3b",
            "--concurrency",
            "1",
            "2",
        ])
      self.assertEqual(rc, 0)
      b_b = json.loads(buf_b_b.getvalue())
      self.assertEqual(b_b["status"], "CERTIFIED")

      # 4. validate-submission (Order A vs Order B)
      tmpl_sub_dir = tmp / "google_tpu_v6e_pi05_bf16"
      tmpl_sub_dir.mkdir(parents=True, exist_ok=True)
      (tmpl_sub_dir / "recipe.yaml").write_text(
          (REPO_ROOT / "recipes" / "track1_vla_pi05_tpu_v6e.yaml").read_text(encoding="utf-8"),
          encoding="utf-8",
      )
      sub_dir = str(tmpl_sub_dir)
      val_out_a = str(tmp / "val_a.json")
      val_out_b = str(tmp / "val_b.json")
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_v_a:
        rc = cli_main([
            "validate-submission",
            "--submission",
            sub_dir,
            "--output",
            val_out_a,
        ])
      self.assertEqual(rc, 0)
      v_a = json.loads(buf_v_a.getvalue())
      self.assertTrue(v_a["passed"])

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_v_b:
        rc = cli_main([
            "validate-submission",
            "--output",
            val_out_b,
            "--submission",
            sub_dir,
        ])
      self.assertEqual(rc, 0)
      v_b = json.loads(buf_v_b.getvalue())
      self.assertTrue(v_b["passed"])

      # 5. export-website (Order A vs Order B)
      seed_lb = str(REPO_ROOT / "tests" / "fixtures" / "seed_leaderboard.json")
      web_out_a = str(tmp / "web_a.json")
      web_out_b = str(tmp / "web_b.json")
      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_w_a:
        rc = cli_main([
            "export-website",
            "--inputs",
            seed_lb,
            "--output",
            web_out_a,
        ])
      self.assertEqual(rc, 0)
      w_a = json.loads(buf_w_a.getvalue())
      self.assertEqual(len(w_a["canonical_models"]), 3)

      with mock.patch("sys.stdout", new_callable=io.StringIO) as buf_w_b:
        rc = cli_main([
            "export-website",
            "--output",
            web_out_b,
            "--inputs",
            seed_lb,
        ])
      self.assertEqual(rc, 0)
      w_b = json.loads(buf_w_b.getvalue())
      self.assertEqual(len(w_b["canonical_models"]), 3)


if __name__ == "__main__":
  unittest.main()


