"""Comprehensive Red-Team Regression Suite for RT-02 through RT-14 (Adversarial Self-Review Gates 1-8).

Covers positive integration paths, Gate 4 negative controls + valid lookalikes, Gate 1 stateful
crash/repeat/concurrent paths, Gate 2 state-machine reason invariants, and Gate 3 both-flag-order CLI checks.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest

from eeb.cli import build_parser, main as eeb_main
from eeb.data.golden import (
    PROTOCOL_MIN_GOLDEN_SAMPLES as MIN_CERTIFIABLE_GOLDEN_SAMPLES,
    generate_golden_reference,
)
from eeb.data.schema import (
    PINNED_HUB_REVISIONS,
    REAL_HF_PIXEL_PROVENANCE,
    REAL_ROBOT_TRACES_KIND,
    SYNTHETIC_CI_SMOKE_TRACES_KIND,
    inspect_synthetic_trace_manifest,
    is_synthetic_trace_manifest,
)
from eeb.loadgen.compare import compare_reports, format_comparison_markdown
from eeb.loadgen.gate import load_golden_npz, run_gate
from eeb.loadgen.report import (
    WM_Y_METRIC,
    WM_Y_METRIC_FRAMES,
    ReportInputs,
    build_report,
    report_to_markdown_summary,
    write_report_bundle,
)
from eeb.loadgen.sweep import (
    PROTOCOL_MAX_C1_P99_P50_RATIO as MAX_C1_P99_P50_RATIO,
    PROTOCOL_MAX_SLOT_COUNT_SPREAD as MAX_SLOT_COUNT_SPREAD,
    PROTOCOL_MIN_WARMUP_CALLS as MIN_HARNESS_WARMUP_CALLS,
    ConcurrencySweep,
    SweepConfig,
    TraceBundleObservationSource,
)
from eeb.envelope.check import DeviceInfo, EnvelopeCheckResult
from eeb.servers.base import serve_policy
from eeb.servers.cosmos_predict25.policy import (
    DEFAULT_COSMOS_ACTION_COND_WEIGHTS as COSMOS_ACTION_WEIGHTS,
    DEFAULT_COSMOS_DIFFUSERS_REPO as COSMOS_HELPER_ID,
    DEFAULT_COSMOS_DIFFUSERS_REVISION as COSMOS_HELPER_SHA,
    DEFAULT_COSMOS_REPO as COSMOS_CKPT,
    DEFAULT_COSMOS_REVISION as COSMOS_SHA,
)
from eeb.servers.dummy import DummyVLAPolicy, DummyWMPolicy
from eeb.servers.gr00t.policy import (
    CANONICAL_GR00T_N17_REPO as GR00T_CKPT,
    CANONICAL_GR00T_N17_REVISION as GR00T_SHA,
    CANONICAL_GR00T_N17_SUBDIR as GR00T_SUBDIR,
    DEFAULT_QWEN3_VL_CONFIG_FALLBACK as GR00T_HELPER_ID,
    DEFAULT_QWEN3_VL_CONFIG_REVISION as GR00T_HELPER_SHA,
)
from eeb.servers.pi05.policy import (
    DEFAULT_GEMMA_TOKENIZER_REPO as PI05_TOK_ID,
    DEFAULT_GEMMA_TOKENIZER_REVISION as PI05_TOK_SHA,
    DEFAULT_LEROBOT_PI05_REPO as PI05_CKPT,
    DEFAULT_LEROBOT_PI05_REVISION as PI05_SHA,
)
from embodied_bench.attestation import create_submission_receipt, verify_submission_receipt
from embodied_bench.dataset_loader import DEFAULT_MINI_V1_DIR, EmbodiedXDatasetLoader
from embodied_bench.datasets.download_traces import update_dataset_manifest
from embodied_bench.leaderboard_compiler import (
    _audit_single_receipt,
    escape_markdown_cell,
    get_gapless_sla_for_model,
    render_pr_comment_markdown,
    validate_submission_meta,
)
from embodied_bench.measure import compute_semianalysis_normalizations
from embodied_bench.protocol import (
    LoopbackEmbodiedServerHandler,
    SharedMemoryTensorRing,
    _PROCESS_OWNED_SHM_LOCK,
    _PROCESS_OWNED_SHM_NAMES,
)
from embodied_bench.quality_gates import evaluate_track1_gate_a, evaluate_track2_gate_a
from embodied_bench.schema import (
    HARDWARE_SKU_SPECS,
    MODERN_REFERENCE_CHECKPOINTS,
    load_yaml_or_json,
)
from embodied_bench.submission_validator import (
    validate_submission_directory,
    verify_submission_bundle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_SHAS = {
    "pi05_ckpt": ("lerobot/pi05_libero_finetuned_v044", "7152f13e34c6c115d808aea0e5d08aca74eac14b"),
    "pi05_tok": ("unsloth/gemma-2b", "7ac9d201a57c8cdc6f939069fc0f044c60197a4a"),
    "gr00t_ckpt": ("nvidia/GR00T-N1.7-LIBERO", "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"),
    "gr00t_helper": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
    "cosmos_ckpt": ("nvidia/Cosmos-Predict2.5-2B", "85f8ae7bfe8f5525c8d103429524dcf12f98bf7b"),
    "cosmos_helper": ("KyleShao/Cosmos-Predict2.5-2B-Diffusers", "1791139b148954cff76830ae8eefbdc1741a6cc0"),
    "libero_data": ("HuggingFaceVLA/libero", "86958911c0f959db2bbbdb107eb3e17c5f9c798e"),
    "droid_data": ("lerobot/droid_100", "87301a2d2e99340e2010c9ef0f1d8e780b08aaf9"),
}


class _NonSyntheticVLAPolicy(DummyVLAPolicy):
  """Deterministic test double carrying non-synthetic server_impl to exercise leaderboard gates."""

  def capabilities(self):
    caps = super().capabilities()
    return type(caps)(
        **{
            **caps.to_dict(),
            "model_id": "pi05-3b",
            "server_impl": "pi05-torch-test-v1",
            "horizon": 10,
            "execute_horizon": 5,
            "image_height": self.image_hw[0],
            "image_width": self.image_hw[1],
        }
    )


def _single_chip_envelope(sku: str = "L40S_48GB") -> EnvelopeCheckResult:
  spec = HARDWARE_SKU_SPECS[sku]
  dev = DeviceInfo(
      index=0,
      name=str(spec["display_name"]),
      vendor=str(spec["vendor"]),
      total_memory_mb=float(spec["vram_mb"]),
      source="test_fixture",
  )
  return EnvelopeCheckResult(
      passed=True,
      num_devices=1,
      reason="single device within envelope",
      hardware_sku=sku,
      device=dev.to_dict(),
      measured_total_memory_mb=float(spec["vram_mb"]),
      memory_provenance="MEASURED",
      spec={
          "display_name": spec["display_name"],
          "vendor": spec["vendor"],
          "vram_mb": spec["vram_mb"],
          "fp16_tflops": spec["fp16_tflops"],
          "fp8_int8_tops": spec["fp8_tflops"],
          "mem_bw_gbps": spec["mem_bw_gbps"],
          "tdp_watts": spec["tdp_watts"],
          "hourly_cost_usd": spec["hourly_cost_usd"],
      },
      spec_provenance="SPEC_TABLE",
      envelope_limits={
          "max_memory_gb": 64.0,
          "max_bf16_tflops": 1000.0,
          "max_fp8_int8_tops": 2000.0,
          "max_mem_bw_tbps": 2.0,
      },
      violations=[],
      inventory={"num_devices": 1, "devices": [dev.to_dict()], "tools_tried": ["test_fixture"], "notes": [], "visibility_env": {}},
  )


# -----------------------------------------------------------------------------
# RT-02: Real EmbodiedX-Traces-v1 Provenance & Synthetic Smoke Rejection
# -----------------------------------------------------------------------------
def test_rt02_real_traces_default_and_synthetic_smoke_rejection(tmp_path: Path) -> None:
  loader = EmbodiedXDatasetLoader(dataset_dir=DEFAULT_MINI_V1_DIR, split="mini_v1")
  ok, errs = loader.verify_manifest()
  assert ok is True, errs
  manifest = json.loads(loader.manifest_path.read_text(encoding="utf-8"))
  assert manifest["kind"] == REAL_ROBOT_TRACES_KIND
  assert manifest["pixel_provenance"] == REAL_HF_PIXEL_PROVENANCE
  assert manifest["is_synthetic_smoke"] is False
  assert manifest["certifiable"] is True
  shard_revs = {s["hf_repo"]: s["hf_revision"] for s in manifest["shards"]}
  assert shard_revs == dict(PINNED_HUB_REVISIONS)

  # Positive lookalike: is_synthetic_trace_manifest returns bool False on real mini bundle
  assert is_synthetic_trace_manifest(DEFAULT_MINI_V1_DIR) is False
  assert is_synthetic_trace_manifest(manifest) is False
  is_s, rsn_s = inspect_synthetic_trace_manifest(DEFAULT_MINI_V1_DIR)
  assert is_s is False and rsn_s == ""

  # Negative control: --synthetic-ci-smoke explicitly tags trace bundle as non-certifiable smoke
  smoke_dir = tmp_path / "smoke_224"
  rc_smoke = eeb_main([
      "data",
      "prepare-traces",
      "--synthetic-ci-smoke",
      "--image-hw",
      "224x224",
      "--episodes",
      "2",
      "--steps",
      "4",
      "--out",
      str(smoke_dir),
      "--quiet",
  ])
  assert rc_smoke == 0
  assert is_synthetic_trace_manifest(smoke_dir) is True
  is_sm, rsn_sm = inspect_synthetic_trace_manifest(smoke_dir)
  assert is_sm is True and rsn_sm

  # Feeding synthetic smoke traces into a non-synthetic run marks report INVALID and rejects in compare/leaderboard
  pol = _NonSyntheticVLAPolicy(compute_delay_ms=0.0, horizon=10, action_dim=7, image_hw=(224, 224))
  src_smoke = TraceBundleObservationSource(str(smoke_dir), split="public")
  cfg = SweepConfig(
      concurrency=[1],
      warmup_calls=30,
      timed_calls=320,
      runs=3,
      pacing_enabled=False,
      max_variance_pct=500.0,
      max_p99_variance_pct=500.0,
      inline_canaries_enabled=False,
      keep_samples=False,
  )
  sweep_res = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      cfg,
      policy=pol,
      observation_source=src_smoke,
  ).run()
  gate = run_gate(sweep_res.points[0], pol.capabilities(), "exec-rt02-smoke")
  rec_dict = load_yaml_or_json(REPO_ROOT / "recipes" / "track1_vla_pi05_l40s.yaml")
  rep_bad = build_report(
      ReportInputs(
          sweep=sweep_res,
          recipe=rec_dict,
          envelope=_single_chip_envelope("L40S_48GB"),
          gates={1: gate},
          allow_sanity_only=False,
      )
  )
  assert rep_bad["status"] == "INVALID"
  assert rep_bad["reason"] and "SYNTHETIC_CI_SMOKE_TRACES" in rep_bad["reason"]

  # Compare rejects candidate using synthetic smoke traces against real baseline
  real_base_doc = json.loads(json.dumps(rep_bad))
  real_base_doc["status"] = "CERTIFIED"
  real_base_doc["reason"] = None
  real_base_doc["quality_gate"]["passed"] = True
  real_base_doc["quality_gate"]["quality_certified"] = True
  real_base_doc["quality_gate"]["mode"] = "GOLDEN_REPLAY"
  for op in real_base_doc["operating_points"]:
    op["quality_certified"] = True
    op["gate_a_mode"] = "GOLDEN_REPLAY"
  cand_smoke_doc = json.loads(json.dumps(real_base_doc))
  real_base_doc["observation_source"] = {
      "kind": REAL_ROBOT_TRACES_KIND,
      "pixel_provenance": REAL_HF_PIXEL_PROVENANCE,
      "is_synthetic_smoke": False,
      "certifiable": True,
  }
  b_path = tmp_path / "b_real" / "report.json"
  c_path = tmp_path / "c_smoke" / "report.json"
  b_path.parent.mkdir()
  c_path.parent.mkdir()
  b_path.write_text(json.dumps(real_base_doc), encoding="utf-8")
  c_path.write_text(json.dumps(cand_smoke_doc), encoding="utf-8")
  cmp_res = compare_reports(b_path, c_path)
  assert cmp_res["verdict"] == "REJECT"
  assert cmp_res["decision_code"] == "REJECT_SYNTHETIC_TRACES"
  assert cmp_res["reason"]


# -----------------------------------------------------------------------------
# RT-03 & RT-04: Warmup>=30, Golden>=100, Pred/s, Fairness, Tail Ratio, L40S Spec
# -----------------------------------------------------------------------------
def test_rt03_rt04_protocol_floors_wm_predictions_per_sec_fairness_and_l40s_spec(tmp_path: Path) -> None:
  assert MIN_HARNESS_WARMUP_CALLS == 30
  assert MIN_CERTIFIABLE_GOLDEN_SAMPLES == 100
  assert WM_Y_METRIC == "predictions_per_sec"
  assert WM_Y_METRIC_FRAMES == "generated_frames_per_sec"
  assert MAX_SLOT_COUNT_SPREAD == pytest.approx(0.15)
  assert MAX_C1_P99_P50_RATIO == pytest.approx(1.5)

  # Gapless SLAs (RT-03)
  assert get_gapless_sla_for_model("VLA_TRACK", "pi05-3b") == (250.0, 4.0, 5)
  assert get_gapless_sla_for_model("VLA_TRACK", "gr00t-n1.7-3b") == (400.0, 2.5, 8)
  assert get_gapless_sla_for_model("WM_TRACK", "cosmos-predict2.5-2b") == (1000.0, 1.0, 1)
  assert MODERN_REFERENCE_CHECKPOINTS["cosmos-predict2.5-2b"]["horizon_frames"] == 16

  # RT-04: Dense FP16 Tensor Core spec for L40S = 362.05 TFLOPS (not 2:4 sparse 733)
  assert HARDWARE_SKU_SPECS["L40S_48GB"]["fp16_tflops"] == pytest.approx(362.05)
  assert HARDWARE_SKU_SPECS["L40S_48GB"]["mem_bw_gbps"] == pytest.approx(864.0)
  norms = compute_semianalysis_normalizations(raw_throughput=100.0, hardware_sku="L40S_48GB", device_power_watts=300.0)
  assert norms["throughput_per_100_tops"] == pytest.approx(100.0 / 3.6205, rel=1e-3)
  assert norms["throughput_per_tbps_bw"] == pytest.approx(100.0 / 0.864, rel=1e-3)

  # Negative control vs lookalike on warmup_calls >= 30
  assert SweepConfig(concurrency=[1], warmup_calls=29, timed_calls=320, runs=3).protocol_compliant_for_track("VLA_TRACK") is False
  assert SweepConfig(concurrency=[1], warmup_calls=30, timed_calls=320, runs=3).protocol_compliant_for_track("VLA_TRACK") is True

  # Negative control vs lookalike on golden num_samples >= 100
  with pytest.raises(ValueError, match="100"):
    generate_golden_reference(
        endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy",
        out_path=tmp_path / "too_small.npz",
        num_samples=8,
        allow_smoke_golden=False,
    )
  smoke_g = tmp_path / "smoke_8.npz"
  generate_golden_reference(
      endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy",
      out_path=smoke_g,
      num_samples=8,
      allow_smoke_golden=True,
  )
  loaded_smoke = load_golden_npz(smoke_g)
  assert loaded_smoke["metadata"]["allow_smoke_golden"] is True
  assert loaded_smoke["metadata"]["certifiable"] is False

  cert_g = tmp_path / "cert_100.npz"
  generate_golden_reference(
      endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy",
      out_path=cert_g,
      traces=str(DEFAULT_MINI_V1_DIR),
      num_samples=100,
  )
  loaded_cert = load_golden_npz(cert_g)
  assert loaded_cert["metadata"]["allow_smoke_golden"] is False
  assert loaded_cert["metadata"]["certifiable"] is True
  assert loaded_cert["n"] == 100

  # WM Track primary Y-axis is predictions_per_sec (no 16x frame inflation)
  wm_out = tmp_path / "wm_bundle"
  assert (
      eeb_main([
          "measure",
          "--endpoint",
          "inproc:eeb.servers.dummy:DummyWMPolicy",
          "--sweep",
          "1",
          "--warmup",
          "1",
          "--calls",
          "4",
          "--runs",
          "2",
          "--no-pacing",
          "--allow-no-device",
          "--sku",
          "L40S_48GB",
          "--out",
          str(wm_out),
          "--quiet",
      ])
      == 0
  )
  wm_rep = json.loads((wm_out / "report.json").read_text(encoding="utf-8"))
  assert wm_rep["y_metric_name"] == "predictions_per_sec"
  pt = wm_rep["operating_points"][0]
  assert pt["generated_frames_per_sec"] == pytest.approx(pt["predictions_per_sec"] * 16.0, rel=1e-2)
  assert pt["raw_per_chip_throughput"] == pytest.approx(pt["predictions_per_sec"], rel=1e-3)
  ok_rcpt, msg_rcpt = verify_submission_receipt(wm_out)
  assert ok_rcpt is True, msg_rcpt

  # Leaderboard auditor negative controls for slot starvation (>0.15) and C=1 p99/p50 > 1.5
  base_sub = {
      "schema_version": "1.0.0",
      "recipe_id": "test_pi05_l40s_closed",
      "track": "VLA_TRACK",
      "model_name": "pi0.5-3B",
      "baseline_model_id": "pi05-3b",
      "precision": "BF16",
      "compression_tier": "strict_lossless",
      "division": "CLOSED",
      "hardware_sku": "L40S_48GB",
      "num_devices": 1,
      "status": "CERTIFIED",
      "reason": None,
      "x_metric_name": "hz_react",
      "y_metric_name": "executed_actions_per_sec",
      "latency_metric_name": "obs_to_action_p99_ms",
      "execution_provenance": {
          "policy": "TORCH_CUDA",
          "policy_is_synthetic": False,
          "loadgen": "REAL_WALLCLOCK",
          "observations": "EMBODIEDX_TRACES_V1_BUNDLE",
          "power": "UNAVAILABLE",
      },
      "observation_source": {
          "kind": REAL_ROBOT_TRACES_KIND,
          "pixel_provenance": REAL_HF_PIXEL_PROVENANCE,
          "is_synthetic_smoke": False,
          "certifiable": True,
          "image_hw": [224, 224],
          "bytes_per_request_images": 301056,
          "wire_resolution_admissible": True,
      },
      "sweep_config": {"warmup_calls": 30, "measured_calls": 320, "num_runs": 3, "execute_horizon": 5},
      "quality_gate": {"passed": True, "quality_certified": True, "allow_smoke_golden": False, "num_samples": 100},
      "operating_points": [
          {
              "point_id": "C1",
              "concurrency": 1,
              "execution_id": "exec-ok-1",
              "execute_horizon": 5,
              "hz_react": 20.0,
              "executed_actions_per_sec": 100.0,
              "raw_per_chip_throughput": 100.0,
              "obs_to_action_p50_ms": 45.0,
              "obs_to_action_p99_ms": 50.0,
              "obs_to_action_max_ms": 55.0,
              "gate_a_passed": True,
              "variance_passed": True,
              "slot_starvation_passed": True,
              "max_slot_count_spread": 0.0,
              "c1_tail_ratio_passed": True,
              "co_located_gate_a": {"execution_id": "exec-ok-1", "passed": True},
          }
      ],
      "pareto_frontier": [
          {
              "point_id": "C1",
              "concurrency": 1,
              "execution_id": "exec-ok-1",
              "execute_horizon": 5,
              "hz_react": 20.0,
              "executed_actions_per_sec": 100.0,
              "raw_per_chip_throughput": 100.0,
              "obs_to_action_p99_ms": 50.0,
              "gate_a_passed": True,
              "co_located_gate_a": {"execution_id": "exec-ok-1", "passed": True},
          }
      ],
  }

  def _write_sub(folder: Path, doc: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    rcpt = create_submission_receipt(report=doc, output_path=None)
    rcpt["num_devices"] = 1
    rcpt["operating_points"] = doc["operating_points"]
    rcpt["sweep_config"] = doc["sweep_config"]
    rcpt["quality_gate"] = doc["quality_gate"]
    rcpt["observation_source"] = doc["observation_source"]
    rcpt["execution_provenance"] = doc["execution_provenance"]
    out_p = folder / "receipt.json"
    out_p.write_text(json.dumps(rcpt, indent=2, sort_keys=True), encoding="utf-8")
    return out_p

  # Lookalike PASS
  ok_clean, viol_clean, _ = _audit_single_receipt(_write_sub(tmp_path / "lb_ok", base_sub))
  assert ok_clean is True, viol_clean

  # Negative control 1: slot starvation spread 0.25 > 0.15
  starved_sub = json.loads(json.dumps(base_sub))
  starved_sub["operating_points"][0]["max_slot_count_spread"] = 0.25
  starved_sub["operating_points"][0]["slot_starvation_passed"] = False
  ok_st, viol_st, _ = _audit_single_receipt(_write_sub(tmp_path / "lb_starved", starved_sub))
  assert ok_st is False and any("Slot-starvation violation" in v for v in viol_st)

  # Negative control 2: C=1 p99/p50 > 1.5
  tail_sub = json.loads(json.dumps(base_sub))
  tail_sub["operating_points"][0]["c1_tail_ratio_passed"] = False
  ok_tl, viol_tl, _ = _audit_single_receipt(_write_sub(tmp_path / "lb_tail", tail_sub))
  assert ok_tl is False and any("C=1 tail-latency stability violation" in v for v in viol_tl)


# -----------------------------------------------------------------------------
# RT-05 CLI: Day-1 Quickstart Subcommands & Live HTTP Server in Both Flag Orders
# -----------------------------------------------------------------------------
def test_rt05_day1_quickstart_cli_and_live_http_both_flag_orders(tmp_path: Path) -> None:
  # Verify `eeb serve` parser flags (--model, --port HTTP default, --grpc)
  serve_parser = build_parser()
  ns_a = serve_parser.parse_args(["serve", "--model", "lerobot/pi05_libero_finetuned_v044", "--port", "8000"])
  ns_b = serve_parser.parse_args(["serve", "--port", "8000", "--model", "lerobot/pi05_libero_finetuned_v044", "--grpc"])
  assert ns_a.model == "lerobot/pi05_libero_finetuned_v044" and ns_a.port == 8000 and ns_a.grpc is False
  assert ns_b.model == "lerobot/pi05_libero_finetuned_v044" and ns_b.port == 8000 and ns_b.grpc is True

  # Spin up live HTTP Policy Server on ephemeral loopback port
  srv = serve_policy(
      DummyVLAPolicy(compute_delay_ms=0.05, horizon=10, execute_horizon=5, action_dim=7),
      "http://127.0.0.1:0",
  )
  try:
    server_url = f"http://127.0.0.1:{srv.eeb_bound_port}"

    # 1. generate-golden via --server in both flag orders
    g1 = tmp_path / "golden_a.npz"
    g2 = tmp_path / "golden_b.npz"
    assert (
        eeb_main([
            "data",
            "generate-golden",
            "--server",
            server_url,
            "--traces",
            str(DEFAULT_MINI_V1_DIR),
            "--num-samples",
            "100",
            "--out",
            str(g1),
            "--quiet",
        ])
        == 0
    )
    assert (
        eeb_main([
            "data",
            "generate-golden",
            "--quiet",
            "--out",
            str(g2),
            "--num-samples",
            "4",
            "--allow-smoke-golden",
            "--traces",
            str(DEFAULT_MINI_V1_DIR),
            "--model",
            "inproc:eeb.servers.dummy:DummyVLAPolicy",
        ])
        == 0
    )

    # 2. eeb measure via --track, --division, --server, --hardware, --out in both flag orders
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    assert (
        eeb_main([
            "measure",
            "--track",
            "vla",
            "--division",
            "closed",
            "--server",
            server_url,
            "--traces",
            str(DEFAULT_MINI_V1_DIR),
            "--golden",
            str(g1),
            "--hardware",
            "L40S_48GB",
            "--sweep",
            "1,2",
            "--warmup",
            "1",
            "--calls",
            "4",
            "--runs",
            "2",
            "--no-pacing",
            "--allow-no-device",
            "--out",
            str(run_a),
            "--quiet",
        ])
        == 0
    )
    assert (
        eeb_main([
            "measure",
            "--quiet",
            "--out",
            str(run_b),
            "--allow-no-device",
            "--no-pacing",
            "--runs",
            "2",
            "--calls",
            "4",
            "--warmup",
            "1",
            "--sweep",
            "1",
            "--hardware",
            "TPU_V6E_1_32GB",
            "--golden",
            str(g1),
            "--traces",
            str(DEFAULT_MINI_V1_DIR),
            "--server",
            server_url,
            "--division",
            "closed",
            "--track",
            "vla",
        ])
        == 0
    )
    assert not (run_a / "recipe.json").exists()

    # 3. eeb prepare-commit-bundle with --run-dir and auto-generated --reveal-manifest in both flag orders
    p1_a = tmp_path / "phase1_a"
    rev_a = tmp_path / "reveal_a.json"
    p1_b = tmp_path / "phase1_b"
    rev_b = tmp_path / "reveal_b.json"
    assert (
        eeb_main([
            "prepare-commit-bundle",
            "--run-dir",
            str(run_a),
            "--team",
            "team_alpha",
            "--engine-tag",
            "torch-compile-v1",
            "--out",
            str(p1_a),
            "--reveal-manifest",
            str(rev_a),
        ])
        == 0
    )
    assert (
        eeb_main([
            "prepare-commit-bundle",
            "--reveal-manifest",
            str(rev_b),
            "--out",
            str(p1_b),
            "--engine-tag",
            "torch-compile-v1",
            "--team",
            "team_alpha",
            "--run-dir",
            str(run_b),
        ])
        == 0
    )
    assert rev_a.is_file() and rev_b.is_file()

    # 4. eeb verify-bundle positional dir + --strict in both flag orders (non-strict passes signature;
    #    strict rejects allow-no-device CPU status=INVALID with non-zero exit code 1 as designed)
    assert eeb_main(["verify-bundle", str(p1_a)]) == 0
    assert eeb_main(["verify-bundle", str(p1_a), "--strict"]) == 1
    assert eeb_main(["verify-bundle", "--strict", str(p1_b)]) == 1
  finally:
    srv.stop(0)


# -----------------------------------------------------------------------------
# RT-06: HTTP DoS Bounds, Session LRU Cap, & POSIX Shared-Memory Guard
# -----------------------------------------------------------------------------
def test_rt06_http_dos_bounds_session_lru_and_foreign_shm_rejection() -> None:
  max_http_body = 64 * 1024 * 1024
  max_active_sessions = 1024

  # 1. Session LRU bound caps memory cleanly at 1024
  wm_pol = DummyWMPolicy(frame_delay_ms=0.0, horizon=4, frame_hw=(16, 16))
  vla_pol = DummyVLAPolicy(compute_delay_ms=0.0, horizon=4, action_dim=7)
  for i in range(max_active_sessions + 80):
    wm_pol.reset(f"sess-{i}")
    vla_pol.reset(f"sess-{i}")
  assert len(wm_pol.sessions) == max_active_sessions
  assert len(vla_pol.sessions) == max_active_sessions

  # 2. POSIX Shared Memory privilege guard: owned SHM works; foreign SHM rejected on server handler
  ring = SharedMemoryTensorRing(capacity_bytes=8192, create=True)
  handler = LoopbackEmbodiedServerHandler()
  try:
    attached_ok = handler._resolve_ring(ring.name)  # noqa: SLF001
    assert attached_ok.name == ring.name
    handler._attached_rings.clear()  # noqa: SLF001
    with _PROCESS_OWNED_SHM_LOCK:
      _PROCESS_OWNED_SHM_NAMES.discard(ring.name)
    with pytest.raises(PermissionError, match="Untrusted shm_name"):
      handler._resolve_ring(ring.name)  # noqa: SLF001
  finally:
    for r_cached in list(handler._attached_rings.values()):  # noqa: SLF001
      r_cached.close()
    ring.close()
    ring.unlink()

  # 3. Live HTTP server rejects negative/invalid Content-Length (400), >64MiB (413), and oversized tensors (400)
  srv = serve_policy(DummyVLAPolicy(), "http://127.0.0.1:0")
  try:
    port = srv.eeb_bound_port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    conn.putrequest("POST", "/predict")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "-12")
    conn.endheaders()
    resp_neg = conn.getresponse()
    resp_neg.read()
    assert resp_neg.status == 400
    conn.close()

    conn2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    conn2.putrequest("POST", "/predict")
    conn2.putheader("Content-Type", "application/json")
    conn2.putheader("Content-Length", "not-an-int")
    conn2.endheaders()
    resp_nan = conn2.getresponse()
    resp_nan.read()
    assert resp_nan.status == 400
    conn2.close()

    conn3 = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    conn3.putrequest("POST", "/predict")
    conn3.putheader("Content-Type", "application/json")
    conn3.putheader("Content-Length", str(max_http_body + 1024))
    conn3.endheaders()
    resp_big = conn3.getresponse()
    resp_big.read()
    assert resp_big.status == 413
    conn3.close()

    bad_shape_payload = json.dumps({
        "session_id": "s1",
        "step": 0,
        "instruction": "pick",
        "proprio": {"dtype": "float32", "shape": [8], "b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="},
        "images": {
            "agentview": {"dtype": "uint8", "shape": [8192, 8192, 3], "b64": "AAAA"},
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/predict",
        data=bad_shape_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
      urllib.request.urlopen(req, timeout=5.0)
    assert exc_info.value.code == 400
  finally:
    srv.stop(0)


# -----------------------------------------------------------------------------
# RT-07: Slug Sanitization, Symlink Escape Rejection, Markdown Escaping & CI YAML
# -----------------------------------------------------------------------------
def test_rt07_slug_sanitization_symlinks_markdown_escaping_and_ci_workflows(tmp_path: Path) -> None:
  # 1. SAFE_SLUG_PATTERN blocks shell metacharacters, pipes, HTML, and path traversal
  for bad_team in ("../escape", "team|table_inject", "team;rm -rf /", "<script>alert(1)</script>"):
    ok_m, errs_m, _ = validate_submission_meta(
        {
            "team": bad_team,
            "engine_tag": "clean-engine-1",
            "hardware_sku": "L40S_48GB",
            "division": "CLOSED",
        },
        require_commit_hash=False,
    )
    assert ok_m is False and errs_m

  # Valid lookalike slug passes
  ok_good, errs_good, _ = validate_submission_meta(
      {
          "team": "Lab_Robotics-2026.v1",
          "engine_tag": "torch-compile_fp8",
          "hardware_sku": "L40S_48GB",
          "division": "CLOSED",
      },
      require_commit_hash=False,
  )
  assert ok_good is True, errs_good

  # 2. Symlink escape inside submission directory is rejected cleanly
  outside_secret = tmp_path / "outside_secret.txt"
  outside_secret.write_text("secret", encoding="utf-8")
  sub_sym = tmp_path / "sub_with_symlink"
  sub_sym.mkdir()
  shutil.copy2(REPO_ROOT / "recipes" / "track1_vla_pi05_l40s.yaml", sub_sym / "recipe.yaml")
  os.symlink(outside_secret, sub_sym / "sneaky_link.json")
  rep_sym = validate_submission_directory(sub_sym)
  assert rep_sym.passed is False
  assert any("symbolic link" in v.lower() or "symlink" in v.lower() for v in rep_sym.violations)
  rep_vb = verify_submission_bundle(sub_sym, strict=True)
  assert rep_vb.passed is False and any(
      "symbolic link" in e.lower() or "symlink" in e.lower() for e in rep_vb.violations
  )

  # 3. Markdown escaping neutralizes pipe injection, newlines, and HTML tags
  malicious_lb = {
      "updated_at": "2026-10-30T00:00:00-07:00",
      "num_verified_submissions": 1,
      "num_rejected_submissions": 1,
      "submissions": [
          {
              "recipe_id": "evil|pipe\n<script>x</script>",
              "track": "VLA_TRACK",
              "baseline_model_id": "pi05-3b",
              "division": "CLOSED",
              "hardware_sku": "L40S_48GB",
              "precision": "BF16",
              "submission_phase": "OPEN_RECIPE",
              "recipe_badge": "VERIFIED_OPEN_RECIPE",
              "summary_metrics": {"peak_throughput": 100.0},
              "award_metrics": {
                  "interactive_reactivity_c1": {"c1_p99_ms": 50.0, "c1_p50_ms": 40.0, "c1_interactive_rate": 20.0},
                  "realtime_fleet_capacity_sla": {"sla_max_throughput": 100.0, "concurrency": 2},
                  "green_embodied_efficiency_sla": {"sla_peak_efficiency_per_joule": None},
                  "realtime_dominated_hypervolume": 50.0,
              },
          }
      ],
      "rejected_submissions": [
          {
              "recipe_id": "bad|rec",
              "hardware_sku": "L40S_48GB",
              "reason": "line1\nline2 | <img src=x onerror=1>",
          }
      ],
      "award_rankings": {},
      "pareto_frontiers": {},
  }
  md_txt = render_pr_comment_markdown(malicious_lb)
  assert "<script>" not in md_txt
  assert "<img" not in md_txt
  assert "evil\\|pipe" in escape_markdown_cell("evil|pipe\n<script>x</script>")
  assert "<script>" not in escape_markdown_cell("evil|pipe\n<script>x</script>")
  assert "\u2014" not in md_txt

  # 4. CI workflow least-privilege & shell-injection audit
  for wf_name in ("ci_preflight.yml", "publish_leaderboard.yml"):
    wf_txt = (REPO_ROOT / "ci" / "workflows" / wf_name).read_text(encoding="utf-8")
    assert "contents: read" in wf_txt
    assert "for d in $(" not in wf_txt

  ci_txt = (REPO_ROOT / "ci" / "workflows" / "ci_preflight.yml").read_text(encoding="utf-8")
  assert "BASE_REF:" in ci_txt
  assert "embodiedx_leaderboard_data.json" in ci_txt


# -----------------------------------------------------------------------------
# RT-08, RT-09, RT-13, RT-14: Packaging, Docker, Pinned SHAs, SPDX, & Safety Scan
# -----------------------------------------------------------------------------
def test_rt08_rt09_rt13_rt14_packaging_docker_shas_spdx_and_ast_safety() -> None:
  # RT-08: pyproject.toml extras & package isolation + .dockerignore + all 7 Dockerfiles
  pyproj = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
  for extra in ("server =", "tpu =", "data =", "sim ="):
    assert extra in pyproj
  assert '"tests*"' in pyproj and '"runs*"' in pyproj and '"submissions*"' in pyproj

  dignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
  for pat in (".git", ".venv", "runs", "submissions"):
    assert pat in dignore

  dockerfiles = sorted((REPO_ROOT / "docker").glob("Dockerfile.*"))
  assert len(dockerfiles) == 7
  for df in dockerfiles:
    txt = df.read_text(encoding="utf-8")
    assert "eeb" in txt and "embodied_bench" in txt and "recipes" in txt
    assert "\nUSER " in txt, f"{df.name} missing non-root USER"
    assert "HEALTHCHECK " in txt, f"{df.name} missing HEALTHCHECK"

  # RT-09 + RT-13: Authoritative 40-char HF commit SHAs across policies, schema, datasets, and all 43 recipes
  hex40 = re.compile(r"^[0-9a-f]{40}$")
  for repo_id, sha in EXPECTED_SHAS.values():
    assert hex40.match(sha), f"Invalid SHA for {repo_id}: {sha}"

  assert (PI05_CKPT, PI05_SHA) == EXPECTED_SHAS["pi05_ckpt"]
  assert (PI05_TOK_ID, PI05_TOK_SHA) == EXPECTED_SHAS["pi05_tok"]
  assert (GR00T_CKPT, GR00T_SHA) == EXPECTED_SHAS["gr00t_ckpt"]
  assert GR00T_SUBDIR == "libero_10"
  assert (GR00T_HELPER_ID, GR00T_HELPER_SHA) == EXPECTED_SHAS["gr00t_helper"]
  assert (COSMOS_CKPT, COSMOS_SHA) == EXPECTED_SHAS["cosmos_ckpt"]
  assert "robot/action-cond" in COSMOS_ACTION_WEIGHTS
  assert (COSMOS_HELPER_ID, COSMOS_HELPER_SHA) == EXPECTED_SHAS["cosmos_helper"]
  assert PINNED_HUB_REVISIONS["HuggingFaceVLA/libero"] == EXPECTED_SHAS["libero_data"][1]
  assert PINNED_HUB_REVISIONS["lerobot/droid_100"] == EXPECTED_SHAS["droid_data"][1]

  assert MODERN_REFERENCE_CHECKPOINTS["pi05-3b"]["hf_revision"] == PI05_SHA
  assert MODERN_REFERENCE_CHECKPOINTS["gr00t-n1.7-3b"]["hf_revision"] == GR00T_SHA
  assert MODERN_REFERENCE_CHECKPOINTS["gr00t-n1.7-3b"]["hf_subdir"] == "libero_10"
  assert MODERN_REFERENCE_CHECKPOINTS["cosmos-predict2.5-2b"]["hf_revision"] == COSMOS_SHA
  assert MODERN_REFERENCE_CHECKPOINTS["cosmos-predict2.5-2b"]["hf_subdir"] == "robot/action-cond"

  recipes = sorted((REPO_ROOT / "recipes").glob("*.yaml"))
  assert len(recipes) == 43
  valid_spdx = {"Apache-2.0", "MIT", "BSD-3-Clause"}
  for rp in recipes:
    rtxt = rp.read_text(encoding="utf-8")
    assert "gr00t17-lerobot-libero_" not in rtxt, f"Dead HF ID in {rp.name}"
    assert "vllm_torchtpu" not in rtxt, f"Fictional backend in {rp.name}"
    assert "\u2014" not in rtxt, f"Em-dash in {rp.name}"
    rdoc = load_yaml_or_json(rp)
    mid = rdoc.get("baseline_model_id")
    if mid in ("pi05-3b", "gr00t-n1.7-3b", "cosmos-predict2.5-2b"):
      assert hex40.match(str(rdoc.get("hf_revision", ""))), f"{rp.name} missing 40-hex hf_revision"
    if rdoc.get("osi_license"):
      assert rdoc["osi_license"] in valid_spdx, f"{rp.name} non-standard SPDX {rdoc['osi_license']!r}"

  # RT-14 + TRAP40: AST scan enforcing weights_only=True on every torch.load and zero personal home paths
  personal_path_re = re.compile(r"/(?:usr/local/google/home|home|Users)/[a-zA-Z0-9_.-]+/")
  for folder in ("eeb", "embodied_bench", "runners", "leaderboard"):
    for py_path in sorted((REPO_ROOT / folder).rglob("*.py")):
      src = py_path.read_text(encoding="utf-8")
      assert "gr00t17-lerobot-libero_" not in src, f"Dead GR00T alias in {py_path}"
      assert not personal_path_re.search(src), f"Hardcoded personal path in {py_path}"
      tree = ast.parse(src, filename=str(py_path))
      for node in ast.walk(tree):
        if isinstance(node, ast.Call):
          fn = node.func
          is_torch_load = (
              isinstance(fn, ast.Attribute)
              and fn.attr == "load"
              and isinstance(fn.value, ast.Name)
              and fn.value.id == "torch"
          )
          if is_torch_load:
            kw_map = {kw.arg: kw.value for kw in node.keywords}
            assert "weights_only" in kw_map, f"torch.load without weights_only=True in {py_path}:{node.lineno}"
            val = kw_map["weights_only"]
            assert isinstance(val, ast.Constant) and val.value is True, (
                f"torch.load must pass literal weights_only=True in {py_path}:{node.lineno}"
            )


# -----------------------------------------------------------------------------
# RT-11: Deterministic RTC Continuity & Multi-Step Cumulative AR Drift Gates
# -----------------------------------------------------------------------------
def test_rt11_deterministic_rtc_continuity_and_wm_multistep_drift_gates() -> None:
  rng = np.random.default_rng(7)
  t_axis = np.linspace(0.0, 1.0, 16, dtype=np.float32)[None, :, None]
  smooth_ref = np.broadcast_to(0.25 * np.sin(2.0 * np.pi * t_axis) + 0.35, (8, 16, 7)).copy()

  # Track 1 Lookalike PASS: smooth candidate chunk tracks reference cleanly with deterministic continuity
  res_t1_ok = evaluate_track1_gate_a(
      execution_id="exec-rt11-ok",
      throughput_execution_id="exec-rt11-ok",
      ref_actions=smooth_ref,
      cand_actions=smooth_ref.copy(),
  )
  r2_t1_ok = evaluate_track1_gate_a(
      execution_id="exec-rt11-ok",
      throughput_execution_id="exec-rt11-ok",
      ref_actions=smooth_ref,
      cand_actions=smooth_ref.copy(),
  )
  assert res_t1_ok.passed is True
  assert res_t1_ok.rtc_prefix_continuity_l2 == r2_t1_ok.rtc_prefix_continuity_l2
  assert res_t1_ok.rtc_prefix_continuity_l2 <= 0.08

  # Track 1 Negative control: high-frequency boundary discontinuity jump trips jerk/continuity check
  sawtooth = smooth_ref.copy()
  sawtooth[:, ::2, :] += 0.35
  sawtooth[:, 1::2, :] -= 0.35
  res_t1_bad = evaluate_track1_gate_a(
      execution_id="exec-rt11-bad",
      throughput_execution_id="exec-rt11-bad",
      ref_actions=smooth_ref,
      cand_actions=sawtooth,
  )
  assert res_t1_bad.passed is False
  assert res_t1_bad.reason

  # Track 2 Lookalike PASS vs Negative control: late-horizon autoregressive drift caught explicitly
  ref_lat = rng.normal(0.0, 1.0, size=(4, 16, 16, 16)).astype(np.float32)
  ref_la = np.clip(rng.normal(0.0, 0.3, size=(4, 16, 7)), -1.0, 1.0).astype(np.float32)
  res_t2_ok = evaluate_track2_gate_a(
      execution_id="exec-rt11-wm-ok",
      throughput_execution_id="exec-rt11-wm-ok",
      ref_video_latents=ref_lat,
      cand_video_latents=ref_lat.copy(),
      ref_latent_actions=ref_la,
      cand_latent_actions=ref_la.copy(),
  )
  assert res_t2_ok.passed is True

  drifted_lat = ref_lat.copy()
  # Keep first 12 frames exact so mean looks mild, then inject severe AR rollout collapse on final frames
  drifted_lat[:, 12:, :, :] += 1.5
  res_t2_drift = evaluate_track2_gate_a(
      execution_id="exec-rt11-wm-drift",
      throughput_execution_id="exec-rt11-wm-drift",
      ref_video_latents=ref_lat,
      cand_video_latents=drifted_lat,
      ref_latent_actions=ref_la,
      cand_latent_actions=ref_la.copy(),
  )
  assert res_t2_drift.passed is False
  assert res_t2_drift.reason


# -----------------------------------------------------------------------------
# Gate 1: Stateful Crash, Repeat, and Concurrent-Writer Verification
# -----------------------------------------------------------------------------
def test_gate1_dataset_manifest_crash_repeat_and_concurrent_writers(tmp_path: Path) -> None:
  manifest_path = tmp_path / "dataset_manifest.json"
  orphan = tmp_path / ".tmp_embodied_dataset_manifest.json_crash.json"
  orphan.write_text("{corrupt_orphan", encoding="utf-8")

  # Call 1 cleans crash orphan and records notes="first_note"
  m1 = update_dataset_manifest(
      manifest_path,
      split_id="libero_10",
      status="CERTIFIED",
      source_mode="hf_lerobot",
      pixel_provenance=REAL_HF_PIXEL_PROVENANCE,
      episodes=[{"episode_id": "ep0", "steps": 16}],
      notes="first_note",
  )
  assert not orphan.exists()
  assert m1["splits"]["libero_10"]["certifiable"] is True
  assert m1["splits"]["libero_10"]["notes"] == "first_note"

  # Repeat path: Call 2 omitting notes does NOT retain stale "first_note"
  m2 = update_dataset_manifest(
      manifest_path,
      split_id="libero_10",
      status="CERTIFIED",
      source_mode="smoke_offline",
      pixel_provenance="SYNTHETIC_PROCEDURAL_FALLBACK",
      episodes=[{"episode_id": "ep0", "steps": 8}],
  )
  assert m2["splits"]["libero_10"]["notes"] is None
  assert m2["splits"]["libero_10"]["is_synthetic_smoke"] is True
  assert m2["splits"]["libero_10"]["certifiable"] is False

  # Concurrent thread + subprocess updates preserve valid JSON and all distinct split keys
  split_ids = ["libero_spatial", "libero_object", "libero_goal", "droid_100"]

  def _thread_update(sid: str) -> str:
    update_dataset_manifest(
        manifest_path,
        split_id=sid,
        status="CERTIFIED",
        source_mode="hf_lerobot",
        pixel_provenance=REAL_HF_PIXEL_PROVENANCE,
        episodes=[{"episode_id": f"{sid}_0", "steps": 16}],
    )
    return sid

  with ThreadPoolExecutor(max_workers=4) as pool:
    assert list(pool.map(_thread_update, split_ids)) == split_ids

  proc_code = (
      "import sys\n"
      "from embodied_bench.datasets.download_traces import update_dataset_manifest\n"
      "update_dataset_manifest(sys.argv[1], split_id=sys.argv[2], status='CERTIFIED', "
      "source_mode='hf_lerobot', episodes=[{'episode_id': sys.argv[2], 'steps': 16}])\n"
  )
  procs = [
      subprocess.Popen(
          [sys.executable, "-c", proc_code, str(manifest_path), sid],
          cwd=str(REPO_ROOT),
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      for sid in split_ids
  ]
  for p in procs:
    _o, err = p.communicate(timeout=30)
    assert p.returncode == 0, err

  final_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
  for sid in split_ids:
    assert sid in final_doc["splits"]
    assert final_doc["splits"][sid]["num_episodes"] == 1


# -----------------------------------------------------------------------------
# Cross-Subagent Integration Deltas A, B, C (Quickstart Parity, Pooled p99, CI Split)
# -----------------------------------------------------------------------------
def test_cross_subagent_integration_deltas_a_b_c(tmp_path: Path) -> None:
  import pytest
  import yaml

  from eeb.cli import (
      _POLICY_MODEL_SHORTHANDS,
      build_parser,
      inject_checkpoint_kwarg,
      resolve_policy_spec,
  )
  from eeb.protocol.policy import load_policy_class

  # Delta A.1: pyproject.toml exposes [vla,gpu] Quickstart extras & /challenge/ homepage
  pyproj = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
  assert "\nvla = [" in pyproj
  assert "\ngpu = [" in pyproj
  assert 'Homepage = "https://efficient-embodied-ai.github.io/challenge/"' in pyproj

  # Delta A.2: Every registered model shorthand resolves cleanly & maps --checkpoint appropriately
  for alias in _POLICY_MODEL_SHORTHANDS:
    spec = resolve_policy_spec(alias)
    cls = load_policy_class(spec)
    assert cls is not None, f"Failed resolving shorthand {alias!r} -> {spec!r}"

  assert inject_checkpoint_kwarg(resolve_policy_spec("pi05"), "lerobot/pi05_libero_finetuned_v044", {}) == {
      "checkpoint_dir": "lerobot/pi05_libero_finetuned_v044"
  }
  assert inject_checkpoint_kwarg(resolve_policy_spec("gr00t"), "nvidia/GR00T-N1.7-LIBERO", {}) == {
      "model_path": "nvidia/GR00T-N1.7-LIBERO"
  }
  assert inject_checkpoint_kwarg(resolve_policy_spec("cosmos"), "nvidia/Cosmos-Predict2.5-2B", {}) == {
      "model_id": "nvidia/Cosmos-Predict2.5-2B"
  }
  assert inject_checkpoint_kwarg(resolve_policy_spec("dummy-vla"), "ignored/ckpt", {}) == {}
  # Explicit caller kwarg takes precedence over CLI --checkpoint shorthand
  assert inject_checkpoint_kwarg(
      resolve_policy_spec("pi05"), "cli/ckpt", {"checkpoint_dir": "explicit/ckpt"}
  ) == {"checkpoint_dir": "explicit/ckpt"}

  # Delta A.3: Exact Quickstart CLI flag parsing (forward & reversed order + negative lookalike)
  parser = build_parser()
  qs_fwd = parser.parse_args([
      "measure",
      "--track",
      "track1a",
      "--division",
      "closed",
      "--hardware",
      "nvidia_l40s_48gb",
      "--server",
      "http://127.0.0.1:8080",
      "--traces",
      "./data/traces",
      "--golden",
      "./data/pi05_golden.npz",
      "--sweep",
      "1,2,4,8,16",
      "--warmup",
      "30",
      "--calls",
      "320",
      "--runs",
      "3",
      "--strict",
      "--out",
      "./runs/my_pi05_l40s",
  ])
  qs_rev = parser.parse_args([
      "measure",
      "--out",
      "./runs/my_pi05_l40s",
      "--strict",
      "--runs",
      "3",
      "--calls",
      "320",
      "--warmup",
      "30",
      "--sweep",
      "1,2,4,8,16",
      "--golden",
      "./data/pi05_golden.npz",
      "--traces",
      "./data/traces",
      "--server",
      "http://127.0.0.1:8080",
      "--hardware",
      "nvidia_l40s_48gb",
      "--division",
      "closed",
      "--track",
      "track1b",
  ])
  assert qs_fwd.track == "track1a" and qs_fwd.sku == "nvidia_l40s_48gb"
  assert qs_rev.track == "track1b" and qs_rev.endpoint == "http://127.0.0.1:8080"
  with pytest.raises(SystemExit):
    parser.parse_args(["measure", "--track", "track3", "--server", "http://127.0.0.1:8080", "--out", "/tmp/x"])

  # Delta B: Pooled steady-state p99(C) + slot fairness fence negative control
  from dataclasses import replace

  from eeb.loadgen.sweep import ObservationSource, ObservationSpec

  pol = _NonSyntheticVLAPolicy(compute_delay_ms=0.1, image_hw=(224, 224))
  golden_p = tmp_path / "g_pi05.npz"
  generate_golden_reference(
      endpoint="inproc:tests.test_red_team_fixes_rt02_rt14:_NonSyntheticVLAPolicy",
      out_path=str(golden_p),
      traces=None,
      num_samples=100,
      image_hw=(224, 224),
      policy_kwargs=json.dumps({"compute_delay_ms": 0.1, "image_hw": [224, 224]}),
  )
  g_npz = load_golden_npz(str(golden_p))
  sweep_obj = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      SweepConfig(concurrency=[1, 2], warmup_calls=30, timed_calls=320, runs=3, server_warmup_calls=1),
      policy=pol,
      observation_source=ObservationSource(ObservationSpec.from_caps(pol.capabilities(), image_hw=(224, 224))),
      golden=g_npz,
  )
  sweep_res = sweep_obj.run()
  c2_run0 = sweep_res.points[1].runs[0]
  assert c2_run0.latency_ms["p99"] == c2_run0.latency_ms["pooled_p99"]
  assert "worst_slot_p99" in c2_run0.latency_ms and "worst_slot_p50" in c2_run0.latency_ms
  assert c2_run0.slot_count_spread <= 0.15

  from eeb.loadgen.client import open_client

  gate_cli = open_client("inproc:eeb.servers.dummy:DummyVLAPolicy", policy=pol)
  try:
    gates = {
        p.concurrency: run_gate(p, sweep_res.caps, f"exec-db-{p.concurrency}", client=gate_cli, golden=g_npz)
        for p in sweep_res.points
    }
  finally:
    gate_cli.close()
  rep_ok = build_report(
      ReportInputs(
          sweep=sweep_res,
          recipe={"recipe_id": "pi05_db_ok", "track": "VLA_TRACK", "division": "CLOSED", "hardware_sku": "L40S_48GB"},
          recipe_path=None,
          envelope=_single_chip_envelope("L40S_48GB"),
          telemetry_by_run={},
          telemetry_source="test",
          gates=gates,
          assumed_power_w=300.0,
      )
  )
  assert rep_ok["operating_points"][1]["slot_starvation_passed"] is True

  # Negative control: inject slot_count_spread = 0.22 (> 0.15) and verify slot_starvation_passed flips False
  # and excludes C=2 from pareto_frontier while marking the report INVALID.
  bad_runs = [replace(r, slot_count_spread=0.22) for r in sweep_res.points[1].runs]
  bad_pt2 = sweep_obj._aggregate(2, bad_runs)  # noqa: SLF001
  bad_sweep = replace(sweep_res, points=[sweep_res.points[0], bad_pt2])
  rep_starved = build_report(
      ReportInputs(
          sweep=bad_sweep,
          recipe={"recipe_id": "pi05_db_bad", "track": "VLA_TRACK", "division": "CLOSED", "hardware_sku": "L40S_48GB"},
          recipe_path=None,
          envelope=_single_chip_envelope("L40S_48GB"),
          telemetry_by_run={},
          telemetry_source="test",
          gates=gates,
          assumed_power_w=300.0,
      )
  )
  assert rep_starved["operating_points"][1]["slot_starvation_passed"] is False
  assert rep_starved["status"] == "INVALID"
  assert "Slot sample imbalance" in rep_starved["reason"]
  assert all(pt["concurrency"] != 2 for pt in rep_starved["pareto_frontier"])

  # Delta A.5: prepare-commit-bundle writes reveal_manifest.json & verifies cleanly without --team/--engine-tag
  run_dir = tmp_path / "run_ok"
  write_report_bundle(rep_ok, str(run_dir))
  sub_out = tmp_path / "submissions" / "track1_vla" / "lab_alpha_pi05"
  assert eeb_main(["prepare-commit-bundle", "--run-dir", str(run_dir), "--output", str(sub_out)]) == 0
  assert (run_dir / "reveal_manifest.json").is_file()
  assert sorted(p.name for p in sub_out.iterdir()) == ["bundle_summary.json", "submission_meta.json"]
  # Cryptographic signature verify succeeds (0); strict hardware gate rejects synthetic ObservationSource (1)
  assert eeb_main(["verify-bundle", str(sub_out)]) == 0
  assert eeb_main(["verify-bundle", str(sub_out), "--strict"]) == 1

  # Positive lookalike for verify-bundle --strict: real-trace CERTIFIED report passes strictly without --team/--engine-tag
  cert_doc = {
      **rep_ok,
      "recipe_id": "track1_vla_pi05_l40s",
      "baseline_model_id": "pi05-3b",
      "model_name": "pi05-3b",
      "precision": "BF16",
      "status": "CERTIFIED",
      "reason": "all protocol and co-located gates passed",
      "admissibility_findings": [],
      "observation_source": {
          "kind": REAL_ROBOT_TRACES_KIND,
          "pixel_provenance": REAL_HF_PIXEL_PROVENANCE,
          "is_synthetic_smoke": False,
          "certifiable": True,
          "image_hw": [224, 224],
          "bytes_per_request_images": 301056,
      },
      "execution_provenance": {
          "policy_server": "pi05-torch-v1",
          "policy_is_synthetic": False,
          "loadgen": "REAL_WALLCLOCK",
          "telemetry_source": "nvml",
          "power_provenance": "MEASURED",
          "spec_provenance": "SPEC_TABLE",
      },
      "quality_gate": {
          "passed": True,
          "quality_certified": True,
          "mode": "GOLDEN_REPLAY",
          "checks": {"num_samples": 100, "allow_smoke_golden": False},
          "per_concurrency": {
              "1": {"passed": True, "mode": "GOLDEN_REPLAY", "checks": {"num_samples": 100, "allow_smoke_golden": False}}
          },
      },
      "operating_points": [
          {
              "point_id": "C1",
              "concurrency": 1,
              "execution_id": "exec-qs-1",
              "execute_horizon": 5,
              "hz_react": 20.0,
              "executed_actions_per_sec": 100.0,
              "raw_per_chip_throughput": 100.0,
              "obs_to_action_p50_ms": 45.0,
              "obs_to_action_p99_ms": 50.0,
              "obs_to_action_max_ms": 55.0,
              "gate_a_passed": True,
              "variance_passed": True,
              "slot_starvation_passed": True,
              "max_slot_count_spread": 0.0,
              "c1_tail_ratio_passed": True,
              "co_located_gate_a": {"execution_id": "exec-qs-1", "passed": True},
          }
      ],
      "pareto_frontier": [
          {
              "point_id": "C1",
              "concurrency": 1,
              "execution_id": "exec-qs-1",
              "execute_horizon": 5,
              "hz_react": 20.0,
              "executed_actions_per_sec": 100.0,
              "raw_per_chip_throughput": 100.0,
              "obs_to_action_p99_ms": 50.0,
              "gate_a_passed": True,
              "co_located_gate_a": {"execution_id": "exec-qs-1", "passed": True},
          }
      ],
      "summary": {
          "num_operating_points": 1,
          "num_pareto_frontier_points": 1,
          "peak_reactivity_hz": 20.0,
          "peak_executed_actions_per_sec": 100.0,
          "peak_chunks_per_sec": 20.0,
      },
      "summary_metrics": {"best_hz_react": 20.0, "best_raw_per_chip_throughput": 100.0},
  }
  cert_run = tmp_path / "cert_run"
  write_report_bundle(cert_doc, str(cert_run))
  # Gate 1 (Crash path): stale .tmp_embodied_reveal_*.json residue never poisons reveal_manifest.json creation
  (cert_run / ".tmp_embodied_reveal_orphan.json").write_text("{corrupt", encoding="utf-8")
  # Gate 1 (Concurrent path): 4 concurrent threads initialize reveal_manifest.json atomically with identical commit_sha256
  from concurrent.futures import ThreadPoolExecutor

  def _concurrent_prep(idx: int) -> int:
    out_i = tmp_path / "submissions" / "track1_vla" / f"lab_cert_concurrent_{idx}"
    return eeb_main(
        ["prepare-commit-bundle", "--run-dir", str(cert_run), "--output", str(out_i), "--team", f"team_{idx}", "--engine-tag", f"eng_{idx}"]
    )

  with ThreadPoolExecutor(max_workers=4) as pool:
    rc_list = list(pool.map(_concurrent_prep, range(4)))
  assert rc_list == [0, 0, 0, 0]
  assert (cert_run / "reveal_manifest.json").is_file()
  on_disk_sha = __import__("hashlib").sha256((cert_run / "reveal_manifest.json").read_bytes()).hexdigest()
  for idx in range(4):
    meta_i = json.loads((tmp_path / "submissions" / "track1_vla" / f"lab_cert_concurrent_{idx}" / "submission_meta.json").read_text(encoding="utf-8"))
    assert meta_i["recipe_commit_sha256"] == on_disk_sha
  # Gate 1 (Repeat path with omitted optional flags on same output dir + clean dir): no stale --team/--engine-tag carryover
  repeat_same_dir = tmp_path / "submissions" / "track1_vla" / "lab_cert_concurrent_0"
  assert eeb_main(["prepare-commit-bundle", "--run-dir", str(cert_run), "--output", str(repeat_same_dir)]) == 0
  meta_same = json.loads((repeat_same_dir / "submission_meta.json").read_text(encoding="utf-8"))
  assert meta_same["team"] is None and meta_same["engine_tag"] is None
  cert_sub = tmp_path / "submissions" / "track1_vla" / "lab_cert_pi05"
  assert eeb_main(["prepare-commit-bundle", "--run-dir", str(cert_run), "--output", str(cert_sub)]) == 0
  meta_repeat = json.loads((cert_sub / "submission_meta.json").read_text(encoding="utf-8"))
  assert meta_repeat["recipe_commit_sha256"] == on_disk_sha
  assert meta_repeat["team"] is None and meta_repeat["engine_tag"] is None
  assert eeb_main(["verify-bundle", str(cert_sub), "--strict"]) == 0

  # Delta C: Workflow 2-job privilege isolation & pre-install PR boundary enforcement
  pub_wf = yaml.safe_load((REPO_ROOT / "ci" / "workflows" / "publish_leaderboard.yml").read_text(encoding="utf-8"))
  assert set(pub_wf["jobs"].keys()) == {"compile_leaderboard", "deploy_website"}
  comp_job_txt = json.dumps(pub_wf["jobs"]["compile_leaderboard"])
  dep_job_txt = json.dumps(pub_wf["jobs"]["deploy_website"])
  assert "secrets." not in comp_job_txt
  assert pub_wf["jobs"]["deploy_website"]["needs"] == "compile_leaderboard"
  assert "pip install" not in dep_job_txt and "eeb " not in dep_job_txt and "pytest" not in dep_job_txt
  assert "website_repo/challenge/data/leaderboard_v1.json" in dep_job_txt
  assert "website_repo/data/leaderboard_v1.json" not in dep_job_txt

  ci_wf = yaml.safe_load((REPO_ROOT / "ci" / "workflows" / "ci_preflight.yml").read_text(encoding="utf-8"))
  ci_steps = ci_wf["jobs"]["preflight-and-validate"]["steps"]
  guard_idx = next(i for i, s in enumerate(ci_steps) if "git diff" in s.get("run", "") and "ls-tree" in s.get("run", ""))
  install_idx = next(i for i, s in enumerate(ci_steps) if 'pip install -e ".[data,dev]"' in s.get("run", ""))
  assert guard_idx < install_idx, "Pre-install PR file/symlink guard must execute strictly BEFORE pip install -e"
