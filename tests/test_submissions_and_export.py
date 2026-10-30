"""Unit and integration tests for Phase 4-5 Submission Validator, Static AST Anti-Cheat Auditor (`Gate 4`), Stateful Submission Registry (`Gates 1 & 2`), Reference Submissions, and Website Exporter (`export_to_website.py`)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import List
import pytest

from embodied_bench.cli import compute_pareto_leaderboard
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    CANONICAL_V1_MODELS,
    REASON_REQUIRED_STATUSES,
    RunStatus,
)
from embodied_bench.stage1_golden_replay import run_stage1_from_recipe_path
from embodied_bench.submission_validator import (
    audit_submission_python_sources,
    check_single_chip_hardware_eligibility,
    record_submission_validation,
    validate_submission_directory,
)
from leaderboard.export_to_website import (
    CANONICAL_WEBSITE_SINGLE_CHIP_SKUS,
    SEMIANALYSIS_Y_AXIS_NORMALIZATIONS,
    export_verified_runs_to_website_json,
    main as export_cli_main,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

REFERENCE_SUBMISSION_RECIPES = {
    "google_tpu_v6e_pi05_bf16": "track1_vla_pi05_tpu_v6e.yaml",
    "nvidia_l40s_gr00t_n17_fp8": "track1_vla_gr00t_n17_l40s.yaml",
}
REFERENCE_SUBMISSION_IDS = tuple(REFERENCE_SUBMISSION_RECIPES.keys())


def _materialize_template_submission(tmp_path: Path, sub_id: str) -> Path:
  sub_path = tmp_path / sub_id
  sub_path.mkdir(parents=True, exist_ok=True)
  src_recipe = REPO_ROOT / "recipes" / REFERENCE_SUBMISSION_RECIPES[sub_id]
  (sub_path / "recipe.yaml").write_text(src_recipe.read_text(encoding="utf-8"), encoding="utf-8")
  return sub_path


# ==============================================================================
# 1. Reference Submission Packages & Single-Chip Envelope Validation
# ==============================================================================


@pytest.mark.parametrize("sub_id", REFERENCE_SUBMISSION_IDS)
def test_validate_all_four_reference_submissions(sub_id: str, tmp_path: Path) -> None:
  sub_path = _materialize_template_submission(tmp_path, sub_id)
  reg_path = tmp_path / "submission_registry.json"

  report = validate_submission_directory(sub_path, registry_path=reg_path)
  assert report.passed is True, f"Reference submission {sub_id} failed: {report.reason}"
  assert report.status == RunStatus.CERTIFIED.value
  assert report.reason is None
  assert report.schema_valid is True
  assert report.division_compliant is True
  assert report.canonical_challenge_model is True
  assert report.num_devices == 1
  assert report.hardware_envelope_check["passed"] is True
  assert report.source_audit["passed"] is True

  reg_data = json.loads(reg_path.read_text(encoding="utf-8"))
  assert reg_data["submissions"][sub_id]["status"] == RunStatus.CERTIFIED.value


def test_reference_submissions_cover_all_canonical_models_and_vendors(tmp_path: Path) -> None:
  models_seen = set()
  vendors_seen = set()
  for sub_id in REFERENCE_SUBMISSION_IDS:
    report = validate_submission_directory(_materialize_template_submission(tmp_path, sub_id))
    models_seen.add(report.baseline_model_id)
    vendors_seen.add(report.hardware_envelope_check["vendor"])

  assert models_seen == {"pi05-3b", "gr00t-n1.7-3b"}
  assert models_seen.issubset(set(CANONICAL_V1_MODELS))
  assert {"GOOGLE_TPU", "NVIDIA"}.issubset(vendors_seen)

  # Option A Day-1 invariant: submissions/ has ZERO pre-populated bundles on launch
  bundled_subs = [
      p
      for p in SUBMISSIONS_DIR.rglob("*")
      if p.is_file() and p.name not in ("README.md", ".gitkeep")
  ]
  assert bundled_subs == []

  # Verify all 3 canonical v1.0 models have Official Season 1 L40S_48GB and TPU_V6E_1 starter recipes,
  # while Day-1 submissions/ launches with zero pre-populated reference_baselines/diagnostics curves (Option A).
  assert not (SUBMISSIONS_DIR / "reference_baselines").exists()
  assert not (SUBMISSIONS_DIR / "diagnostics").exists()
  l40s_models = set()
  tpu_models = set()
  for rec_path in sorted((REPO_ROOT / "recipes").glob("*.yaml")):
    txt = rec_path.read_text(encoding="utf-8")
    sku_hit = (
        "L40S_48GB"
        if "hardware_sku: L40S_48GB" in txt
        else ("TPU_V6E_1" if ("hardware_sku: TPU_V6E_1" in txt) else None)
    )
    if sku_hit is None:
      continue
    model_hit = (
        "pi05-3b"
        if "baseline_model_id: pi05-3b" in txt
        else (
            "gr00t-n1.7-3b"
            if "baseline_model_id: gr00t-n1.7-3b" in txt
            else ("cosmos-predict2.5-2b" if "baseline_model_id: cosmos-predict2.5-2b" in txt else None)
        )
    )
    if model_hit:
      if sku_hit == "L40S_48GB":
        l40s_models.add(model_hit)
      else:
        tpu_models.add(model_hit)
  assert l40s_models == set(CANONICAL_V1_MODELS)
  assert tpu_models == set(CANONICAL_V1_MODELS)


def test_single_chip_hardware_eligibility_rejects_datacenter_80gb_and_multidevice(
    tmp_path: Path,
) -> None:
  # H100_SXM5 has 80GB VRAM and 3350 GB/s bandwidth -> exceeds CAP_SINGLE_CHIP (64GB, 2000 GB/s)
  h100_check = check_single_chip_hardware_eligibility("H100_SXM5")
  assert h100_check.passed is False
  assert any("vram_gb" in v for v in h100_check.violations)
  assert any("mem_bw_gbps" in v for v in h100_check.violations)

  # Submission with num_devices=2 is rejected
  bad_dir = tmp_path / "bad_multidevice_sub"
  bad_dir.mkdir(parents=True)
  bad_recipe = bad_dir / "recipe.yaml"
  bad_recipe.write_text(
      "\n".join([
          "recipe_id: bad_multigpu",
          "track: VLA_TRACK",
          "division: OPEN",
          "model_family: pi05",
          "model_name: pi05-2gpu",
          "baseline_model_id: pi05-3b",
          "precision: FP8",
          "hardware_sku: RTX_5090_32GB",
          "num_devices: 2",
          "declared_training_compute_gpu_hours: 10.0",
          "declared_inference_flops_gflops: 60.0",
          "osi_license: Apache-2.0",
      ])
      + "\n",
      encoding="utf-8",
  )
  rep = validate_submission_directory(bad_dir)
  assert rep.passed is False
  assert rep.status == RunStatus.REJECTED.value
  assert rep.reason is not None and "num_devices" in rep.reason


# ==============================================================================
# 2. Gate 1: Crash Path, Repeat-Call Path, & Concurrent-Writer Path
# ==============================================================================


def test_gate1_crash_path_orphan_cleanup_and_corrupt_json_recovery(tmp_path: Path) -> None:
  reg_path = tmp_path / "submissions_state.json"
  orphan_tmp = tmp_path / f".tmp_embodied_{reg_path.name}_crashed123.json"
  orphan_tmp.write_text("{partial_crash_payload", encoding="utf-8")
  reg_path.write_text("{corrupt_truncated_json", encoding="utf-8")

  state = record_submission_validation(
      registry_path=reg_path,
      submission_id="sub_after_crash",
      status=RunStatus.CERTIFIED,
      recipe_id="rec_1",
      hardware_sku="TPU_V6E_1_32GB",
  )
  assert not orphan_tmp.exists(), "Orphan temp file was not cleaned up"
  assert state.get("recovered_from_corrupt_state") is True
  assert state["submissions"]["sub_after_crash"]["status"] == RunStatus.CERTIFIED.value


def test_gate1_repeat_call_path_no_stale_optional_field_carryover(tmp_path: Path) -> None:
  reg_path = tmp_path / "repeat_registry.json"

  # Call 1: populate all optional fields
  record_submission_validation(
      registry_path=reg_path,
      submission_id="sub_repeat",
      status=RunStatus.REJECTED,
      reason="Initial Gate A failure.",
      recipe_id="rec_old",
      hardware_sku="RTX_5090_32GB",
      division="CLOSED",
      track="VLA_TRACK",
      violations=["violation_1", "violation_2"],
      notes="Temporary reviewer note",
      receipt_sha256="deadbeef" * 8,
  )

  # Call 2: update same submission_id omitting optional fields
  updated = record_submission_validation(
      registry_path=reg_path,
      submission_id="sub_repeat",
      status=RunStatus.CERTIFIED,
      recipe_id="rec_new",
      hardware_sku="TPU_V6E_1_32GB",
  )
  rec = updated["submissions"]["sub_repeat"]
  assert rec["status"] == RunStatus.CERTIFIED.value
  assert rec["recipe_id"] == "rec_new"
  assert rec["hardware_sku"] == "TPU_V6E_1_32GB"
  assert rec["reason"] is None
  assert rec["notes"] is None
  assert rec["violations"] == []
  assert rec["receipt_sha256"] is None
  assert rec["division"] is None
  assert rec["track"] is None


def test_gate1_concurrent_threads_and_subprocesses(tmp_path: Path) -> None:
  reg_path = tmp_path / "concurrent_registry.json"

  def _thread_worker(idx: int) -> None:
    record_submission_validation(
        registry_path=reg_path,
        submission_id=f"thread_sub_{idx}",
        status=RunStatus.CERTIFIED,
        recipe_id=f"recipe_thread_{idx}",
        hardware_sku="TPU_V6E_1_32GB",
    )

  with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(_thread_worker, range(16)))

  # Also spawn concurrent subprocess writers
  procs: List[subprocess.Popen[str]] = []
  for p_idx in range(6):
    code = (
        "from embodied_bench.submission_validator import record_submission_validation; "
        f"record_submission_validation({str(reg_path)!r}, 'proc_sub_{p_idx}', 'CERTIFIED', "
        f"recipe_id='recipe_proc_{p_idx}', hardware_sku='AMD_MI210_64GB')"
    )
    procs.append(
        subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    )
  for proc in procs:
    _, stderr = proc.communicate(timeout=30)
    assert proc.returncode == 0, f"Subprocess writer failed: {stderr}"

  final_state = json.loads(reg_path.read_text(encoding="utf-8"))
  assert final_state["num_submissions"] == 22
  assert final_state["num_certified"] == 22
  for i in range(16):
    assert f"thread_sub_{i}" in final_state["submissions"]
  for j in range(6):
    assert f"proc_sub_{j}" in final_state["submissions"]


# ==============================================================================
# 3. Gate 2: State Machine Completeness & Mandatory Non-Empty Reason
# ==============================================================================


def test_gate2_all_eight_run_statuses_reachable_and_reason_enforced(tmp_path: Path) -> None:
  reg_path = tmp_path / "state_machine_registry.json"

  for status in RunStatus:
    sub_id = f"sub_{status.value.lower()}"
    if status in REASON_REQUIRED_STATUSES:
      # Missing or whitespace-only reason MUST raise ValueError
      with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
        record_submission_validation(reg_path, sub_id, status=status, reason=None)
      with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
        record_submission_validation(reg_path, sub_id, status=status, reason="   ")

      # Valid non-empty reason succeeds
      state = record_submission_validation(
          reg_path,
          sub_id,
          status=status,
          reason=f"Actionable diagnostic reason for {status.value}.",
      )
      assert state["submissions"][sub_id]["status"] == status.value
      assert state["submissions"][sub_id]["reason"] == f"Actionable diagnostic reason for {status.value}."
    else:
      state = record_submission_validation(reg_path, sub_id, status=status)
      assert state["submissions"][sub_id]["status"] == status.value
      assert state["submissions"][sub_id]["reason"] is None

  # Verify illegal transition rejection when enforce_transitions=True
  record_submission_validation(
      reg_path, "sub_strict_fsm", status=RunStatus.CERTIFIED, enforce_transitions=True
  )
  with pytest.raises(ValueError, match="Illegal state transition"):
    record_submission_validation(
        reg_path, "sub_strict_fsm", status=RunStatus.PENDING, enforce_transitions=True
    )


# ==============================================================================
# 4. Gate 4: Static AST Anti-Cheat Auditor (Positive Violations & Negative Controls)
# ==============================================================================


def test_gate4_positive_violations_detected_by_ast_auditor() -> None:
  # 1. Raw-input byte memoization
  res_memo = audit_submission_python_sources(
      source_code=(
          "class CheatingPolicy:\n"
          "    def __init__(self):\n"
          "        self._memo_cache = {}\n"
          "    def forward(self, obs):\n"
          "        key = obs.tobytes()\n"
          "        if key not in self._memo_cache:\n"
          "            self._memo_cache[obs.tobytes()] = self._compute(obs)\n"
          "        return self._memo_cache[obs.tobytes()]\n"
      )
  )
  assert res_memo.passed is False
  assert any(v.rule_id == "CHEAT_RAW_INPUT_MEMOIZATION" for v in res_memo.violations)

  # 2. Mocking or overriding time.perf_counter_ns / time.perf_counter
  res_timer_assign = audit_submission_python_sources(
      source_code=(
          "import time\n"
          "time.perf_counter_ns = lambda: 1000\n"
      )
  )
  assert res_timer_assign.passed is False
  assert any(v.rule_id == "CHEAT_TIMER_OVERRIDE_OR_MOCK" for v in res_timer_assign.violations)

  res_timer_mock = audit_submission_python_sources(
      source_code=(
          "from unittest import mock\n"
          "with mock.patch('time.perf_counter_ns', return_value=42):\n"
          "    pass\n"
      )
  )
  assert res_timer_mock.passed is False
  assert any(v.rule_id == "CHEAT_TIMER_OVERRIDE_OR_MOCK" for v in res_timer_mock.violations)

  # 3. Hardcoding gate_a_passed = True or overriding verify_submission_receipt
  res_gate = audit_submission_python_sources(
      source_code=(
          "def run_eval():\n"
          "    gate_a_passed = True\n"
          "    return gate_a_passed\n"
      )
  )
  assert res_gate.passed is False
  assert any(v.rule_id == "CHEAT_GATE_OR_RECEIPT_BYPASS" for v in res_gate.violations)

  res_receipt = audit_submission_python_sources(
      source_code=(
          "def verify_submission_receipt(path):\n"
          "    return True, 'bypassed'\n"
      )
  )
  assert res_receipt.passed is False
  assert any(v.rule_id == "CHEAT_GATE_OR_RECEIPT_BYPASS" for v in res_receipt.violations)

  # 4. Multi-device environment bypass (`CUDA_VISIBLE_DEVICES="0,1"`, `tensor_parallel_size=2`)
  res_multidev = audit_submission_python_sources(
      source_code=(
          "import os\n"
          "os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'\n"
          "engine = build_engine(tensor_parallel_size=2)\n"
      )
  )
  assert res_multidev.passed is False
  multidev_rules = [v for v in res_multidev.violations if v.rule_id == "CHEAT_MULTI_DEVICE_BYPASS"]
  assert len(multidev_rules) >= 2


def test_gate4_negative_controls_stay_completely_silent() -> None:
  legitimate_code = '''"""Module docstring mentioning prohibited patterns for documentation only:
  - Do NOT set CUDA_VISIBLE_DEVICES=0,1 or tensor_parallel_size=2.
  - Do NOT mock time.perf_counter_ns or use _memo_cache[obs.tobytes()].
  """
import os
import time

# Comment mentioning CUDA_VISIBLE_DEVICES=0,1 and _memo_cache[obs.tobytes()] and time.perf_counter_ns = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"


class LegitimateVLARunner:
    """Runner docstring mentioning CUDA_VISIBLE_DEVICES=0,1 and time.perf_counter_ns."""

    def __init__(self) -> None:
        self.session_kv_cache = {}
        self.tensor_parallel_size = 1
        self.num_devices = 1

    def step(self, session_id: str, instruction: str, image_tensor):
        """Step docstring with time.perf_counter_ns and _memo_cache[obs.tobytes()]."""
        t0 = time.perf_counter_ns()
        t_start = time.perf_counter()
        if session_id not in self.session_kv_cache:
            self.session_kv_cache[session_id] = self.encode_prompt(instruction, tensor_parallel_size=1)
        kv_prefix = self.session_kv_cache.get(session_id)
        action = self.decode_action(image_tensor, kv_prefix, num_devices=1)
        elapsed_ns = time.perf_counter_ns() - t0
        elapsed_s = time.perf_counter() - t_start
        return action, elapsed_ns, elapsed_s
'''
  res = audit_submission_python_sources(source_code=legitimate_code)
  assert res.passed is True, f"False positive on legitimate code: {res.reason}"
  assert res.violations == []


# ==============================================================================
# 5. Website JSON Exporter & CLI Flag-Order Verification (`Gate 3`)
# ==============================================================================


def test_export_verified_runs_to_website_json_and_cli_both_flag_orders(
    tmp_path: Path,
) -> None:
  report_paths: List[str] = []
  stage1_recipes = [
      (sub_id, REPO_ROOT / "recipes" / rec_name)
      for sub_id, rec_name in REFERENCE_SUBMISSION_RECIPES.items()
  ] + [
      ("track2_wm_cosmos_tpu_v6e", REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_tpu_v6e.yaml"),
      ("track2_wm_cosmos_l40s_bf16", REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_l40s_bf16.yaml"),
  ]
  for label, recipe_path in stage1_recipes:
    out_report = tmp_path / f"{label}_report.json"
    rep = run_stage1_from_recipe_path(
        recipe_path=recipe_path,
        output_path=out_report,
    )
    assert rep["status"] == RunStatus.CERTIFIED.value, f"{label} failed stage1: {rep.get('reason')}"
    report_paths.append(str(out_report))

  leaderboard_path = tmp_path / "leaderboard.json"
  compute_pareto_leaderboard(
      input_paths=report_paths,
      output_path=str(leaderboard_path),
  )

  website_json_path = tmp_path / "website_leaderboard.json"
  payload = export_verified_runs_to_website_json(
      leaderboard_json_path=leaderboard_path,
      output_website_json_path=website_json_path,
  )

  assert website_json_path.is_file()
  assert payload["site"] == "efficient-embodied-ai.github.io"
  assert payload["num_verified_submissions"] == 4

  # Verify all 3 canonical v1.0 challenge models are present and populated
  assert set(payload["canonical_models"].keys()) == set(CANONICAL_V1_MODELS)
  for model_id in CANONICAL_V1_MODELS:
    assert payload["canonical_models"][model_id]["num_verified_runs"] >= 1

  # Verify all 9 cross-vendor single-chip SKUs are present with exact labels
  assert set(payload["cross_vendor_single_chip_skus"].keys()) == set(
      CANONICAL_WEBSITE_SINGLE_CHIP_SKUS.keys()
  )
  expected_labels = {
      "Google Cloud TPU v6e-1 32GB",
      "TPU v5e-1 16GB",
      "NVIDIA Jetson AGX Thor 64GB",
      "RTX 5090 32GB",
      "L40S 48GB",
      "RTX 4090 24GB",
      "NVIDIA L4 24GB",
      "AMD Instinct MI210 64GB",
      "Radeon PRO W7900 48GB",
  }
  assert set(payload["cross_vendor_sku_labels"]) == expected_labels

  # Verify all 4 SemiAnalysis Y-axis normalizations are present on frontiers and submissions
  expected_y_keys = {
      "raw_per_chip_throughput",
      "power_efficiency_per_watt",
      "throughput_per_100_tops",
      "throughput_per_dollar_hr",
  }
  assert set(payload["y_axis_normalizations"].keys()) == expected_y_keys
  for norm_key in expected_y_keys:
    norm_entry = payload["y_axis_normalizations"][norm_key]
    assert len(norm_entry["vla_global_frontier"]) > 0
    assert len(norm_entry["wm_global_frontier"]) > 0

  for sub in payload["submissions"]:
    for norm_key in expected_y_keys:
      assert float(sub[norm_key]) > 0.0

  # Gate 3: Test CLI main(argv) in both flag orders (--input first vs --output first)
  cli_out_1 = tmp_path / "cli_website_1.json"
  rc1 = export_cli_main(["--input", str(leaderboard_path), "--output", str(cli_out_1)])
  assert rc1 == 0 and cli_out_1.is_file()

  cli_out_2 = tmp_path / "cli_website_2.json"
  rc2 = export_cli_main(["--output", str(cli_out_2), "--input", *report_paths])
  assert rc2 == 0 and cli_out_2.is_file()
  loaded_2 = json.loads(cli_out_2.read_text(encoding="utf-8"))
  assert loaded_2["num_verified_submissions"] == 4
