"""End-to-end tests for the live EmbodiedX (`embodied-efficiency-bench`) pipeline, provenance separation, and red-team remediations."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import request as urllib_request

import numpy as np
import pytest

from embodied_bench.attestation import verify_submission_receipt
from embodied_bench.baselines import create_baseline_runner
from embodied_bench.cli import main as eeb_main
from embodied_bench.harness import (
    EmbodiedBenchmarkServer,
    encode_payload_arrays,
    verify_single_chip_envelope,
)
from embodied_bench.schema import (
    ANALYTIC_REFERENCE_PROVENANCE,
    HARDWARE_SKU_SPECS,
    LIVE_EMPIRICAL_PROVENANCE,
)
from embodied_bench.submission_validator import validate_submission_directory
from runners import cosmos3_edge_server, gr00t_n17_server, pi05_server, qwen3_omni_server


def test_five_step_live_cli_pipeline_and_flag_orders(tmp_path: Path) -> None:
  """Verifies download-data -> benchmark --receipt-output -> verify-receipt -> build-leaderboard -> export-website in both flag orders (Gate 3)."""
  data_dir_a = tmp_path / "data_a"
  data_dir_b = tmp_path / "data_b"

  # Step 1A: eeb download-data with --out (flag order A)
  rc = eeb_main([
      "download-data",
      "--out",
      str(data_dir_a),
      "--mode",
      "smoke",
      "--splits",
      "libero_spatial",
      "bridgedata_v2_omni",
      "--num-episodes",
      "2",
      "--steps-per-episode",
      "6",
  ])
  assert rc == 0
  manifest_a = json.loads((data_dir_a / "dataset_manifest.json").read_text(encoding="utf-8"))
  assert manifest_a["pixel_provenance"] == "SYNTHETIC_PROCEDURAL_FALLBACK"
  assert manifest_a["is_real_camera_pixels"] is False

  # Step 1B: eeb download-data with reversed flag order B
  rc = eeb_main([
      "download-data",
      "--steps-per-episode",
      "6",
      "--num-episodes",
      "2",
      "--splits",
      "libero_spatial",
      "--mode",
      "smoke",
      "--out",
      str(data_dir_b),
  ])
  assert rc == 0
  assert (data_dir_b / "dataset_manifest.json").exists()

  subs_dir = tmp_path / "submissions"
  sub_vla_dir = subs_dir / "gr00t_fp8_tpu_v6e_live"
  sub_wm_dir = subs_dir / "cosmos3_fp8_rtx4090_live"
  sub_vla_dir.mkdir(parents=True, exist_ok=True)
  sub_wm_dir.mkdir(parents=True, exist_ok=True)

  vla_report_path = sub_vla_dir / "benchmark_report.json"
  vla_receipt_path = sub_vla_dir / "receipt.json"
  wm_report_path = sub_wm_dir / "benchmark_report.json"
  wm_receipt_path = sub_wm_dir / "receipt.json"

  # Step 2A: eeb benchmark --receipt-output (VLA track, flag order A)
  rc = eeb_main([
      "benchmark",
      "--model",
      "gr00t-n1.7-3b",
      "--hardware-sku",
      "TPU_V6E_1_32GB",
      "--precision",
      "fp8",
      "--concurrency",
      "1",
      "2",
      "--steps-per-stream",
      "3",
      "--warmup-steps",
      "1",
      "--dataset-dir",
      str(data_dir_a),
      "--recipe-id",
      "gr00t_fp8_tpu_v6e_live",
      "--output",
      str(vla_report_path),
      "--receipt-output",
      str(vla_receipt_path),
  ])
  assert rc == 0
  vla_report = json.loads(vla_report_path.read_text(encoding="utf-8"))
  assert vla_report["status"] == "CERTIFIED"
  assert vla_report["execution_provenance"] == LIVE_EMPIRICAL_PROVENANCE
  assert vla_report["compliance_gates_g1_g4"]["passed"] is True
  assert vla_report["compliance_gates_g1_g4"]["g4_cache_memoization"]["passed"] is True

  # Step 2B: eeb benchmark --receipt-output (WM track, reversed flag order B)
  rc = eeb_main([
      "benchmark",
      "--receipt-output",
      str(wm_receipt_path),
      "--output",
      str(wm_report_path),
      "--recipe-id",
      "cosmos3_fp8_rtx4090_live",
      "--dataset-dir",
      str(data_dir_a),
      "--warmup-steps",
      "1",
      "--steps-per-stream",
      "3",
      "--concurrency",
      "1",
      "2",
      "--precision",
      "fp8",
      "--hardware-sku",
      "RTX_4090_24GB",
      "--model",
      "cosmos3-edge",
  ])
  assert rc == 0
  wm_report = json.loads(wm_report_path.read_text(encoding="utf-8"))
  assert wm_report["status"] == "CERTIFIED"
  assert wm_report["execution_provenance"] == LIVE_EMPIRICAL_PROVENANCE
  assert wm_report["compliance_gates_g1_g4"]["passed"] is True
  assert wm_report["compliance_gates_g1_g4"]["g4_cache_memoization"]["passed"] is True

  # Step 3: eeb verify-receipt on both live receipts
  assert eeb_main(["verify-receipt", "--receipt", str(vla_receipt_path)]) == 0
  assert eeb_main(["verify-receipt", "--receipt", str(wm_receipt_path)]) == 0

  # Step 4: eeb build-leaderboard in both flag orders
  lb_path_a = tmp_path / "leaderboard_a.json"
  md_path_a = tmp_path / "LEADERBOARD_A.md"
  lb_path_b = tmp_path / "leaderboard_b.json"
  md_path_b = tmp_path / "LEADERBOARD_B.md"

  assert (
      eeb_main([
          "build-leaderboard",
          "--submissions-dir",
          str(subs_dir),
          "--output",
          str(lb_path_a),
          "--markdown-output",
          str(md_path_a),
          "--strict-verify",
      ])
      == 0
  )
  assert (
      eeb_main([
          "build-leaderboard",
          "--strict-verify",
          "--markdown-output",
          str(md_path_b),
          "--output",
          str(lb_path_b),
          "--submissions-dir",
          str(subs_dir),
      ])
      == 0
  )

  lb_data = json.loads(lb_path_a.read_text(encoding="utf-8"))
  assert lb_data["num_verified_submissions"] == 2
  assert lb_data["execution_provenance_summary"]["num_live_empirical_submissions"] == 2
  assert lb_data["execution_provenance_summary"]["num_analytic_reference_submissions"] == 0
  for sub in lb_data["submissions"]:
    assert sub["execution_provenance"]["execution_mode"] == "live_empirical_runner"
    assert sub["execution_provenance"]["is_analytic_projection"] is False

  # Step 5: eeb export-website in both flag orders
  web_path_a = tmp_path / "website_a.json"
  web_path_b = tmp_path / "website_b.json"
  assert (
      eeb_main([
          "export-website",
          "--inputs",
          str(lb_path_a),
          "--output",
          str(web_path_a),
      ])
      == 0
  )
  assert (
      eeb_main([
          "export-website",
          "--output",
          str(web_path_b),
          "--inputs",
          str(lb_path_b),
      ])
      == 0
  )
  web_data = json.loads(web_path_a.read_text(encoding="utf-8"))
  assert web_data["num_verified_submissions"] == 2
  for sub in web_data["submissions"]:
    assert sub["execution_provenance"]["execution_mode"] == "live_empirical_runner"
    assert sub["execution_provenance"]["is_analytic_projection"] is False


def test_measure_and_stage1_live_vs_analytic_provenance(tmp_path: Path) -> None:
  """Verifies CRITICAL-1: eeb measure and run-stage1 distinguish analytic reference mode from --live-runner and --server-url."""
  recipe_path = Path("recipes/track1_vla_gr00t_n17_edge.yaml")
  assert recipe_path.exists()

  # 1. Analytic mode (default without --live-runner)
  analytic_out = tmp_path / "measure_analytic.json"
  analytic_receipt = tmp_path / "receipt_analytic.json"
  rc = eeb_main([
      "measure",
      "--recipe",
      str(recipe_path),
      "--output",
      str(analytic_out),
      "--receipt-output",
      str(analytic_receipt),
      "--warmup-calls",
      "30",
      "--timed-calls",
      "6",
      "--num-runs",
      "3",
  ])
  assert rc == 0
  analytic_rep = json.loads(analytic_out.read_text(encoding="utf-8"))
  assert analytic_rep["execution_provenance"] == ANALYTIC_REFERENCE_PROVENANCE
  assert analytic_rep["execution_provenance"]["is_analytic_projection"] is True
  ok, _ = verify_submission_receipt(analytic_receipt)
  assert ok is True

  # 2. Live in-process runner mode (--live-runner)
  live_out = tmp_path / "measure_live.json"
  live_receipt = tmp_path / "receipt_live.json"
  rc = eeb_main([
      "measure",
      "--live-runner",
      "--warmup-calls",
      "30",
      "--timed-calls",
      "6",
      "--num-runs",
      "3",
      "--receipt-output",
      str(live_receipt),
      "--output",
      str(live_out),
      "--recipe",
      str(recipe_path),
  ])
  assert rc == 0
  live_rep = json.loads(live_out.read_text(encoding="utf-8"))
  assert live_rep["execution_provenance"] == LIVE_EMPIRICAL_PROVENANCE
  assert live_rep["execution_provenance"]["is_analytic_projection"] is False
  assert live_rep["compliance_gates_g1_g4"]["g4_cache_memoization"]["passed"] is True
  ok, _ = verify_submission_receipt(live_receipt)
  assert ok is True

  # 3. Live HTTP server mode (--server-url) on both `measure` and `run-stage1`
  runner = create_baseline_runner(
      model_id="gr00t-n1.7-3b",
      precision="fp8",
      denoising_steps=4,
      weights_mode="mini_real",
      hardware_sku="RTX_4090_24GB",
  )
  with EmbodiedBenchmarkServer(runner=runner, host="127.0.0.1", port=0) as srv:
    srv_meas_out = tmp_path / "measure_srv.json"
    rc = eeb_main([
        "measure",
        "--recipe",
        str(recipe_path),
        "--server-url",
        srv.base_url,
        "--warmup-calls",
        "30",
        "--timed-calls",
        "6",
        "--num-runs",
        "3",
        "--output",
        str(srv_meas_out),
    ])
    assert rc == 0
    srv_meas_rep = json.loads(srv_meas_out.read_text(encoding="utf-8"))
    assert srv_meas_rep["execution_provenance"] == LIVE_EMPIRICAL_PROVENANCE

    srv_s1_out = tmp_path / "stage1_srv.json"
    rc = eeb_main([
        "run-stage1",
        "--server-url",
        srv.base_url,
        "--output",
        str(srv_s1_out),
        "--recipe",
        str(recipe_path),
    ])
    assert rc == 0
    srv_s1_rep = json.loads(srv_s1_out.read_text(encoding="utf-8"))
    assert srv_s1_rep["execution_provenance"] == LIVE_EMPIRICAL_PROVENANCE


def test_unified_runner_servers_cli_and_endpoints(tmp_path: Path) -> None:
  """Verifies HIGH-2: runners/*_server.py support --serve/--port/--host and all unified HTTP endpoints."""
  for mod in [
      gr00t_n17_server,
      pi05_server,
      cosmos3_edge_server,
      qwen3_omni_server,
  ]:
    assert (
        mod.main([
            "--serve",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--self-test",
        ])
        == 0
    )
    assert (
        mod.main([
            "--self-test",
            "--port",
            "0",
            "--host",
            "127.0.0.1",
            "--serve",
        ])
        == 0
    )

  vla_runner = create_baseline_runner("gr00t-n1.7-3b", precision="fp8", denoising_steps=2)
  with EmbodiedBenchmarkServer(runner=vla_runner, host="127.0.0.1", port=0) as srv:
    for health_path in ("/health", "/v1/health"):
      with urllib_request.urlopen(f"{srv.base_url}{health_path}", timeout=5.0) as resp:
        assert resp.status == 200
        h_body = json.loads(resp.read().decode("utf-8"))
        assert h_body["status"] == "ok"

    obs_payload = encode_payload_arrays({
        "rgb": np.zeros((1, 64, 64, 3), dtype=np.uint8),
        "proprio": np.zeros((1, 14), dtype=np.float32),
    })
    req_bytes = json.dumps({"session_id": "test_ep", "observation": obs_payload}).encode("utf-8")
    for post_path in ("/v1/act", "/v1/vla/act", "/v1/world_step", "/v1/wm/step"):
      req = urllib_request.Request(
          f"{srv.base_url}{post_path}",
          data=req_bytes,
          headers={"Content-Type": "application/json"},
          method="POST",
      )
      with urllib_request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 200
        body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"


def test_datacenter_sku_aliases_and_single_chip_cap() -> None:
  """Verifies MEDIUM-1: H100_SXM_80GB, A100_SXM_80GB, and B200_180GB are registered and enforced."""
  for sku in ("H100_SXM_80GB", "A100_SXM_80GB", "B200_180GB"):
    assert sku in HARDWARE_SKU_SPECS

  # Negative control: H100_SXM_80GB (80 GB VRAM) violates CAP_SINGLE_CHIP (<= 64 GB VRAM)
  res_single = verify_single_chip_envelope(
      hardware_sku="H100_SXM_80GB",
      detected_device_count=1,
      perf_cap="CAP_SINGLE_CHIP",
  )
  assert res_single.passed is False
  assert "exceeds CAP_SINGLE_CHIP ceiling" in (res_single.reason or "")

  # Positive lookalike: H100_SXM_80GB passes under CAP_DATACENTER_1000T
  res_dc = verify_single_chip_envelope(
      hardware_sku="H100_SXM_80GB",
      detected_device_count=1,
      perf_cap="CAP_DATACENTER_1000T",
  )
  assert res_dc.passed is True


def test_negative_controls_tampered_receipt_memoization_multidevice_and_hot_loop_sync(
    tmp_path: Path,
) -> None:
  """Verifies Gate 4 negative controls across receipt tampering, G4 memoization, multi-device, and CRITICAL-2 hot-loop sync."""
  # 1. Memoized runner (--simulate-memoization-cheat) fails Gate G4 and produces REJECTED
  memo_report_path = tmp_path / "memo_report.json"
  memo_receipt_path = tmp_path / "memo_receipt.json"
  rc = eeb_main([
      "benchmark",
      "--model",
      "gr00t-n1.7-3b",
      "--hardware-sku",
      "TPU_V6E_1_32GB",
      "--precision",
      "fp8",
      "--concurrency",
      "1",
      "--steps-per-stream",
      "2",
      "--warmup-steps",
      "1",
      "--simulate-memoization-cheat",
      "--output",
      str(memo_report_path),
      "--receipt-output",
      str(memo_receipt_path),
  ])
  assert rc == 2
  memo_rep = json.loads(memo_report_path.read_text(encoding="utf-8"))
  assert memo_rep["status"] == "REJECTED"
  assert memo_rep["compliance_gates_g1_g4"]["g4_cache_memoization"]["passed"] is False

  # 2. Multi-device (--num-devices 2 under CAP_SINGLE_CHIP) produces REJECTED
  mdev_report_path = tmp_path / "mdev_report.json"
  rc_mdev = eeb_main([
      "benchmark",
      "--model",
      "gr00t-n1.7-3b",
      "--hardware-sku",
      "TPU_V6E_1_32GB",
      "--num-devices",
      "2",
      "--perf-cap",
      "CAP_SINGLE_CHIP",
      "--concurrency",
      "1",
      "--steps-per-stream",
      "2",
      "--warmup-steps",
      "1",
      "--output",
      str(mdev_report_path),
  ])
  assert rc_mdev == 2
  mdev_rep = json.loads(mdev_report_path.read_text(encoding="utf-8"))
  assert mdev_rep["status"] == "REJECTED"
  assert mdev_rep["single_device_compliant"] is False

  # 3. Tampered receipt is rejected by verify-receipt and build-leaderboard
  clean_sub_dir = tmp_path / "subs" / "clean_sub"
  tampered_sub_dir = tmp_path / "subs" / "tampered_sub"
  clean_sub_dir.mkdir(parents=True, exist_ok=True)
  tampered_sub_dir.mkdir(parents=True, exist_ok=True)

  clean_receipt_path = clean_sub_dir / "receipt.json"
  rc_clean = eeb_main([
      "benchmark",
      "--model",
      "pi05-3b",
      "--hardware-sku",
      "TPU_V6E_1_32GB",
      "--precision",
      "fp8",
      "--concurrency",
      "1",
      "--steps-per-stream",
      "2",
      "--warmup-steps",
      "1",
      "--recipe-id",
      "pi05_clean_live",
      "--output",
      str(clean_sub_dir / "report.json"),
      "--receipt-output",
      str(clean_receipt_path),
  ])
  assert rc_clean == 0
  assert eeb_main(["verify-receipt", "--receipt", str(clean_receipt_path)]) == 0

  tampered_payload = json.loads(clean_receipt_path.read_text(encoding="utf-8"))
  tampered_payload["attested_body"]["frontier_points"][0]["y_throughput"] = 999999.0
  tampered_receipt_path = tampered_sub_dir / "receipt.json"
  tampered_receipt_path.write_text(json.dumps(tampered_payload, indent=2), encoding="utf-8")
  assert eeb_main(["verify-receipt", "--receipt", str(tampered_receipt_path)]) == 1

  lb_out = tmp_path / "lb_quarantine.json"
  assert (
      eeb_main([
          "build-leaderboard",
          "--submissions-dir",
          str(tmp_path / "subs"),
          "--output",
          str(lb_out),
      ])
      == 0
  )
  lb_q = json.loads(lb_out.read_text(encoding="utf-8"))
  assert lb_q["num_verified_submissions"] == 1
  assert lb_q["num_rejected_submissions"] == 1

  # 4. CRITICAL-2: Hot-loop device sync (`CHEAT_HOT_LOOP_DEVICE_SYNC`) in validate-submission
  bad_sub_dir = tmp_path / "bad_hot_loop_sub"
  bad_sub_dir.mkdir(parents=True, exist_ok=True)
  (bad_sub_dir / "recipe.yaml").write_text(
      Path("recipes/track1_vla_gr00t_n17_edge.yaml").read_text(encoding="utf-8"),
      encoding="utf-8",
  )
  (bad_sub_dir / "runner.py").write_text(
      "def run_timed_loop(steps, tensor):\n"
      "    for i in range(steps):\n"
      "        val = tensor.item()\n"
      "    return val\n",
      encoding="utf-8",
  )
  bad_report = validate_submission_directory(bad_sub_dir)
  assert bad_report.passed is False
  assert any("CHEAT_HOT_LOOP_DEVICE_SYNC" in str(v) for v in bad_report.violations)
  assert eeb_main(["validate-submission", "--submission", str(bad_sub_dir)]) == 2

  # Positive lookalike: `.item()` outside loop (post-run summary) passes
  good_sub_dir = tmp_path / "good_post_loop_sub"
  good_sub_dir.mkdir(parents=True, exist_ok=True)
  (good_sub_dir / "recipe.yaml").write_text(
      Path("recipes/track1_vla_gr00t_n17_edge.yaml").read_text(encoding="utf-8"),
      encoding="utf-8",
  )
  (good_sub_dir / "runner.py").write_text(
      "def run_timed_loop(steps, tensor):\n"
      "    acc = tensor\n"
      "    for i in range(steps):\n"
      "        acc = acc + 1\n"
      "    summary_scalar = acc.item()\n"
      "    return summary_scalar\n",
      encoding="utf-8",
  )
  good_report = validate_submission_directory(good_sub_dir)
  assert good_report.passed is True
  assert not any("CHEAT_HOT_LOOP_DEVICE_SYNC" in str(v) for v in good_report.violations)
  assert eeb_main(["validate-submission", "--submission", str(good_sub_dir)]) == 0
