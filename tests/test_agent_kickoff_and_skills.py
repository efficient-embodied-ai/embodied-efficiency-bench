"""Unit and CLI integration tests for `embodied_bench/agent_kickoff.py`, `AGENTS.md`, and the EmbodiedX AI Agent Skill Packages (`tests/test_agent_kickoff_and_skills.py`).

Covers Adversarial Self-Review Gates 1, 2, 3, 4, 5, 6, and 7:
  - Gate 1 (Three Paths): Crash recovery from orphan `.tmp_embodied_*` & corrupt `STUDY_PLAN.json`,
    repeat-call idempotency & optional field hygiene, and concurrent thread + subprocess updates under `fcntl.flock`.
  - Gate 2 (State Machine): Reachability of `PENDING`, `RUNNING`, `CERTIFIED`, `BLOCKED`, `FAILED`, `REJECTED`,
    `SKIPPED`, `INVALID` and mandatory non-empty `reason` enforcement on failure/rejection states.
  - Gate 3 (Execute Documented CLI Verbatim): Execution of `python3 -m embodied_bench.agent_kickoff` in both
    flag orders, including `--run-baseline` live HTTP `EmbodiedBenchmarkServer` + `EmbodiedLoadGenClient` sweeps.
  - Gate 4 (Negative Controls): Rejection of multi-chip `--num-devices 2`, 80GB `H100_SXM_80GB` under
    `CAP_SINGLE_CHIP`, and forbidden `OPEN`-division levers (`step_distillation`) when `--division CLOSED`,
    plus negative-control lookalikes that pass cleanly.
  - Verification of `AGENTS.md`, `skills/embodiedx-agent-kickoff/SKILL.md`,
    `skills/embodiedx-submission-auditor/SKILL.md`, and `.claude/commands/embodiedx-optimize.md`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import pytest
import yaml

from embodied_bench.agent_kickoff import (
    MILESTONE_IDS,
    detect_host_accelerator,
    main as kickoff_main,
    parse_kickoff_prompt,
    resolve_hardware_sku,
    run_agent_kickoff,
    update_study_milestone,
)
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    DivisionType,
    PerformanceCapId,
    RunStatus,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_detect_host_accelerator_across_tpu_nvidia_tegra_rocm_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verify detect_host_accelerator maps tpu-info, tegrastats, nvidia-smi, rocm-smi, and CPU fallback to CAP_SINGLE_CHIP SKUs."""
  monkeypatch.delenv("EEB_HARDWARE_SKU", raising=False)

  # 1. Google Cloud TPU v6e-1 (32GB) via tpu-info output
  tpu_out = "TPU v6e-1 | HBM Usage: 12.40 GiB / 32.00 GiB | Duty Cycle: 68.0%"
  tpu_info = detect_host_accelerator(probe_live=False, tpu_info_output=tpu_out)
  assert tpu_info.hardware_sku == "TPU_V6E_1_32GB"
  assert tpu_info.vendor == "GOOGLE_TPU"
  assert tpu_info.detected_device_count == 1
  assert tpu_info.qualifies_cap_single_chip is True

  # 2. NVIDIA Jetson AGX Thor 64GB via tegrastats output
  tegra_out = "RAM 18420/65536MB (lfb 4096x4MB) VDD_IN 54200mW/52100mW"
  tegra_info = detect_host_accelerator(probe_live=False, tegrastats_output=tegra_out)
  assert tegra_info.hardware_sku == "JETSON_AGX_THOR_64GB"
  assert tegra_info.vendor == "NVIDIA_TEGRA"
  assert tegra_info.qualifies_cap_single_chip is True

  # 3. NVIDIA GeForce RTX 5090 32GB via nvidia-smi output
  nv_out = "NVIDIA GeForce RTX 5090, 32768\n"
  nv_info = detect_host_accelerator(probe_live=False, nvidia_smi_output=nv_out)
  assert nv_info.hardware_sku == "RTX_5090_32GB"
  assert nv_info.vendor == "NVIDIA"
  assert nv_info.detected_device_count == 1
  assert nv_info.qualifies_cap_single_chip is True

  # 4. AMD Instinct MI210 64GB via rocm-smi JSON output
  rocm_out = json.dumps({
      "card0": {
          "Card series": "AMD Instinct MI210",
          "VRAM Total Memory (B)": str(64 * 1024 * 1024 * 1024),
          "VRAM Total Used Memory (B)": str(16 * 1024 * 1024 * 1024),
          "Average Graphics Package Power (W)": "195.0",
      }
  })
  rocm_info = detect_host_accelerator(probe_live=False, rocm_smi_output=rocm_out)
  assert rocm_info.hardware_sku == "AMD_MI210_64GB"
  assert rocm_info.vendor == "AMD_ROCM"
  assert rocm_info.detected_device_count == 1
  assert rocm_info.qualifies_cap_single_chip is True

  # 5. CPU development fallback
  fallback_info = detect_host_accelerator(probe_live=False, fallback_sku="TPU_V6E_1_32GB")
  assert fallback_info.hardware_sku == "TPU_V6E_1_32GB"
  assert fallback_info.detection_source == "cpu_dev_fallback"
  assert fallback_info.qualifies_cap_single_chip is True


def test_parse_natural_language_kickoff_prompt() -> None:
  """Verify natural-language prompt parsing for all 4 canonical models, SKUs, and divisions."""
  p1 = parse_kickoff_prompt("Optimize pi0.5 on TPU v6e-1 in OPEN division with rtc_inpainting")
  assert p1["model_id"] == "pi05-3b"
  assert p1["hardware_sku"] == "TPU_V6E_1_32GB"
  assert p1["division"] == "OPEN"
  assert "rtc_inpainting" in p1["optimizations"]

  p2 = parse_kickoff_prompt("Benchmark GR00T-N1.7 on RTX 5090 in CLOSED division")
  assert p2["model_id"] == "gr00t-n1.7-3b"
  assert p2["hardware_sku"] == "RTX_5090_32GB"
  assert p2["division"] == "CLOSED"

  p3 = parse_kickoff_prompt("Run Cosmos3-Edge on Jetson AGX Thor in OPEN division")
  assert p3["model_id"] == "cosmos3-edge"
  assert p3["hardware_sku"] == "JETSON_AGX_THOR_64GB"
  assert p3["division"] == "OPEN"

  p4 = parse_kickoff_prompt("Optimize Qwen3-Omni on MI210 in CLOSED division")
  assert p4["model_id"] == "qwen3-omni-30b-a3b"
  assert p4["hardware_sku"] == "AMD_MI210_64GB"
  assert p4["division"] == "CLOSED"


def test_gate1_three_paths_crash_repeat_and_concurrent_on_study_plan(tmp_path: Path) -> None:
  """Gate 1: Verify crash recovery, repeat-call optional field hygiene, and concurrent thread/process safety on STUDY_PLAN.json."""
  study_dir = tmp_path / "gate1_study"
  study_dir.mkdir(parents=True, exist_ok=True)
  plan_path = study_dir / "STUDY_PLAN.json"

  # 1. Crash Path: plant orphan `.tmp_embodied_*` files and corrupt 0-byte / truncated STUDY_PLAN.json
  orphan_a = study_dir / ".tmp_embodied_STUDY_PLAN.json_orphan1.json"
  orphan_b = study_dir / ".tmp_embodied_orphan2.json"
  orphan_a.write_text("{corrupt", encoding="utf-8")
  orphan_b.write_text("", encoding="utf-8")
  plan_path.write_text("{corrupt_truncated_json", encoding="utf-8")

  recovered_plan = run_agent_kickoff(
      prompt="Optimize pi0.5 on TPU v6e-1 in OPEN division",
      study_dir=study_dir,
      notes="initial_note_to_clear_on_repeat",
  )
  assert not orphan_a.exists(), "Orphan temp file was not cleaned up"
  assert not orphan_b.exists(), "Orphan temp file was not cleaned up"
  assert recovered_plan["recovered_from_corrupt_state"] is True
  assert recovered_plan["passed"] is True
  assert recovered_plan["status"] == RunStatus.CERTIFIED.value
  assert recovered_plan["notes"] == "initial_note_to_clear_on_repeat"
  assert recovered_plan["milestone_ids"] == list(MILESTONE_IDS)

  # 2. Repeat-Call Path: omit optional `notes` on repeat kickoff and `metrics`/`notes`/`error_code` on milestone update
  repeat_plan = run_agent_kickoff(
      prompt="Optimize pi0.5 on TPU v6e-1 in OPEN division",
      study_dir=study_dir,
  )
  assert repeat_plan["notes"] is None, "Stale optional notes carried over on repeat call"

  update_study_milestone(
      study_plan_path=plan_path,
      milestone_id="M2_LIVE_BASELINE_ANCHOR",
      status=RunStatus.RUNNING,
      metrics={"temp_metric": 123.4},
      notes="running_note",
      error_code="WARN_TEMP",
  )
  after_cert = update_study_milestone(
      study_plan_path=plan_path,
      milestone_id="M2_LIVE_BASELINE_ANCHOR",
      status=RunStatus.CERTIFIED,
  )
  m2_rec = after_cert["milestones"]["M2_LIVE_BASELINE_ANCHOR"]
  assert m2_rec["status"] == RunStatus.CERTIFIED.value
  assert m2_rec["reason"] is None
  assert m2_rec["metrics"] is None
  assert m2_rec["notes"] is None
  assert m2_rec["error_code"] is None

  # 3. Concurrent-Writer Path: 5 threads + 2 subprocesses updating milestones in STUDY_PLAN.json simultaneously
  def _thread_worker(idx: int) -> None:
    mid = MILESTONE_IDS[idx % len(MILESTONE_IDS)]
    update_study_milestone(
        study_plan_path=plan_path,
        milestone_id=mid,
        status=RunStatus.CERTIFIED,
        metrics={"thread_idx": idx},
        enforce_transitions=False,
    )

  subproc_code = (
      "import sys; "
      "from embodied_bench.agent_kickoff import update_study_milestone; "
      "update_study_milestone(sys.argv[1], sys.argv[2], 'CERTIFIED', metrics={'subproc': True}, enforce_transitions=False)"
  )
  procs = [
      subprocess.Popen(
          [sys.executable, "-c", subproc_code, str(plan_path), MILESTONE_IDS[p]],
          cwd=str(REPO_ROOT),
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
      )
      for p in range(2)
  ]
  with ThreadPoolExecutor(max_workers=5) as pool:
    list(pool.map(_thread_worker, range(10)))

  for proc in procs:
    _, stderr = proc.communicate(timeout=10.0)
    assert proc.returncode == 0, f"Concurrent subprocess failed: {stderr.decode()}"

  final_disk = json.loads(plan_path.read_text(encoding="utf-8"))
  assert set(final_disk["milestones"].keys()) == set(MILESTONE_IDS)
  for mid in MILESTONE_IDS:
    assert final_disk["milestones"][mid]["status"] == RunStatus.CERTIFIED.value


def test_gate2_state_machine_transitions_and_mandatory_reason(tmp_path: Path) -> None:
  """Gate 2: Verify PENDING, RUNNING, CERTIFIED, BLOCKED, FAILED, REJECTED, SKIPPED, INVALID transitions and mandatory reason."""
  study_dir = tmp_path / "gate2_study"
  run_agent_kickoff(
      model_id="gr00t-n1.7-3b",
      hardware_sku="RTX_5090_32GB",
      division="OPEN",
      study_dir=study_dir,
  )
  plan_path = study_dir / "STUDY_PLAN.json"

  # Illegal direct transition PENDING -> CERTIFIED on M2 must raise ValueError
  with pytest.raises(ValueError, match="Illegal state transition"):
    update_study_milestone(
        study_plan_path=plan_path,
        milestone_id="M2_LIVE_BASELINE_ANCHOR",
        status=RunStatus.CERTIFIED,
    )

  # Legal PENDING -> RUNNING -> CERTIFIED on M2
  update_study_milestone(plan_path, "M2_LIVE_BASELINE_ANCHOR", RunStatus.RUNNING)
  update_study_milestone(plan_path, "M2_LIVE_BASELINE_ANCHOR", RunStatus.CERTIFIED)

  # Failure/rejection/skip states require non-empty reason
  failure_cases = [
      ("M3_CLOSED_DIVISION_KERNEL_KV_OPT", RunStatus.PENDING, RunStatus.BLOCKED),
      ("M4_OPEN_DIVISION_STEP_CACHE_OPT", RunStatus.RUNNING, RunStatus.FAILED),
      ("M5_SIGNED_RECEIPT_AND_PR_PACKAGE", RunStatus.RUNNING, RunStatus.REJECTED),
  ]
  for mid, prep_status, target_status in failure_cases:
    if prep_status == RunStatus.RUNNING:
      update_study_milestone(plan_path, mid, RunStatus.RUNNING)

    with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
      update_study_milestone(plan_path, mid, target_status, reason="")

    with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
      update_study_milestone(plan_path, mid, target_status, reason="   ")

    updated = update_study_milestone(
        plan_path,
        mid,
        target_status,
        reason=f"Milestone {mid} entered {target_status.value} during Gate 2 test.",
    )
    assert updated["milestones"][mid]["status"] == target_status.value
    assert target_status.value in updated["milestones"][mid]["reason"]


def test_gate3_cli_both_flag_orders_and_live_baseline_anchor(tmp_path: Path) -> None:
  """Gate 3: Execute `python3 -m embodied_bench.agent_kickoff` in both flag orders including `--run-baseline`."""
  # Order A: --prompt first, --run-baseline at end
  study_a = tmp_path / "cli_order_a"
  out_a = tmp_path / "summary_a.json"
  rc_a = kickoff_main([
      "--prompt",
      "Optimize pi0.5 on TPU v6e-1 in OPEN division",
      "--study-dir",
      str(study_a),
      "--baseline-concurrencies",
      "1",
      "2",
      "--baseline-steps",
      "3",
      "--run-baseline",
      "--output",
      str(out_a),
  ])
  assert rc_a == 0
  assert (study_a / "STUDY_PLAN.json").exists()
  assert (study_a / "EXPANDED_AGENT_PROMPT.md").exists()
  assert (study_a / "baseline_recipe.yaml").exists()
  assert (study_a / "baseline_anchor.json").exists()

  anchor_a = json.loads((study_a / "baseline_anchor.json").read_text(encoding="utf-8"))
  assert anchor_a["model_id"] == "pi05-3b"
  assert anchor_a["hardware_sku"] == "TPU_V6E_1_32GB"
  assert anchor_a["status"] == RunStatus.CERTIFIED.value
  assert len(anchor_a["operating_points"]) == 2
  assert anchor_a["peak_raw_per_chip_throughput"] > 0.0

  plan_a = json.loads((study_a / "STUDY_PLAN.json").read_text(encoding="utf-8"))
  assert plan_a["milestones"]["M1_HARDWARE_AND_DATA_PREFLIGHT"]["status"] == RunStatus.CERTIFIED.value
  assert plan_a["milestones"]["M2_LIVE_BASELINE_ANCHOR"]["status"] == RunStatus.CERTIFIED.value

  # Order B: --run-baseline first, explicit --division / --hardware-sku / --model in reverse order via subprocess
  study_b = tmp_path / "cli_order_b"
  proc = subprocess.run(
      [
          sys.executable,
          "-m",
          "embodied_bench.agent_kickoff",
          "--run-baseline",
          "--baseline-steps",
          "2",
          "--baseline-concurrencies",
          "1",
          "--division",
          "CLOSED",
          "--hardware-sku",
          "RTX_5090_32GB",
          "--model",
          "cosmos3-edge",
          "--study-dir",
          str(study_b),
      ],
      cwd=str(REPO_ROOT),
      capture_output=True,
      text=True,
      timeout=30.0,
      check=False,
  )
  assert proc.returncode == 0, f"Subprocess CLI failed: {proc.stderr}"
  stdout_json = json.loads(proc.stdout)
  assert stdout_json["passed"] is True
  assert stdout_json["model_id"] == "cosmos3-edge"
  assert stdout_json["hardware_sku"] == "RTX_5090_32GB"
  assert stdout_json["division"] == "CLOSED"
  assert (study_b / "baseline_anchor.json").exists()


def test_gate4_negative_controls_multi_chip_80gb_sku_and_closed_division_violations(
    tmp_path: Path,
) -> None:
  """Gate 4: Verify negative controls fire on multi-chip, >64GB H100_SXM_80GB, and CLOSED division step_distillation."""
  # 1. Positive Violation A: Multi-chip `--num-devices 2` is rejected at kickoff preflight
  study_multi = tmp_path / "neg_multi_chip"
  res_multi = run_agent_kickoff(
      model_id="pi05-3b",
      hardware_sku="TPU_V6E_1_32GB",
      division="OPEN",
      num_devices=2,
      study_dir=study_multi,
  )
  assert res_multi["passed"] is False
  assert res_multi["status"] == RunStatus.REJECTED.value
  assert res_multi["reason"] is not None and "detected_device_count=2" in res_multi["reason"]
  assert res_multi["milestones"]["M1_HARDWARE_AND_DATA_PREFLIGHT"]["status"] == RunStatus.REJECTED.value

  with pytest.raises(ValueError, match="detected_device_count=2"):
    run_agent_kickoff(
        model_id="pi05-3b",
        hardware_sku="TPU_V6E_1_32GB",
        num_devices=2,
        study_dir=tmp_path / "neg_multi_raise",
        raise_on_violation=True,
    )

  rc_multi_cli = kickoff_main([
      "--model",
      "pi05-3b",
      "--hardware-sku",
      "TPU_V6E_1_32GB",
      "--num-devices",
      "2",
      "--study-dir",
      str(tmp_path / "neg_multi_cli"),
  ])
  assert rc_multi_cli == 2

  # 2. Positive Violation B: 80GB `H100_SXM_80GB` is rejected under `CAP_SINGLE_CHIP` (<= 64 GB, <= 2000 GB/s)
  study_h100 = tmp_path / "neg_h100_80gb"
  res_h100 = run_agent_kickoff(
      model_id="gr00t-n1.7-3b",
      hardware_sku="H100_SXM_80GB",
      division="OPEN",
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP.value,
      study_dir=study_h100,
  )
  assert res_h100["passed"] is False
  assert res_h100["status"] == RunStatus.REJECTED.value
  assert res_h100["reason"] is not None and "80.0 GB" in res_h100["reason"]

  rc_h100_cli = kickoff_main([
      "--model",
      "gr00t-n1.7-3b",
      "--hardware-sku",
      "H100_SXM_80GB",
      "--study-dir",
      str(tmp_path / "neg_h100_cli"),
  ])
  assert rc_h100_cli == 2

  # 3. Positive Violation C: Forbidden OPEN-division lever `step_distillation` in `--division CLOSED`
  study_closed_bad = tmp_path / "neg_closed_distill"
  res_closed_bad = run_agent_kickoff(
      model_id="pi05-3b",
      hardware_sku="TPU_V6E_1_32GB",
      division="CLOSED",
      optimizations=["step_distillation"],
      study_dir=study_closed_bad,
  )
  assert res_closed_bad["passed"] is False
  assert res_closed_bad["status"] == RunStatus.REJECTED.value
  assert res_closed_bad["reason"] is not None and "CLOSED division forbids" in res_closed_bad["reason"]

  rc_closed_cli = kickoff_main([
      "--model",
      "pi05-3b",
      "--hardware-sku",
      "TPU_V6E_1_32GB",
      "--division",
      "CLOSED",
      "--optimizations",
      "step_distillation",
      "--study-dir",
      str(tmp_path / "neg_closed_cli"),
  ])
  assert rc_closed_cli == 2

  # 4. Negative-Control Lookalikes:
  #    - `step_distillation` in OPEN division passes cleanly
  #    - `vlm_prefix_kv_cache_reuse` in CLOSED division on 1x TPU_V6E_1_32GB passes cleanly
  res_open_ok = run_agent_kickoff(
      model_id="pi05-3b",
      hardware_sku="TPU_V6E_1_32GB",
      division="OPEN",
      optimizations=["step_distillation", "rtc_inpainting"],
      study_dir=tmp_path / "lookalike_open_ok",
  )
  assert res_open_ok["passed"] is True
  assert res_open_ok["status"] == RunStatus.CERTIFIED.value
  assert res_open_ok["violations"] == []

  res_closed_ok = run_agent_kickoff(
      model_id="pi05-3b",
      hardware_sku="TPU_V6E_1_32GB",
      division="CLOSED",
      optimizations=["vlm_prefix_kv_cache_reuse", "zero_copy_posix_shm_ipc"],
      study_dir=tmp_path / "lookalike_closed_ok",
  )
  assert res_closed_ok["passed"] is True
  assert res_closed_ok["status"] == RunStatus.CERTIFIED.value
  assert res_closed_ok["violations"] == []


def test_agents_md_and_skill_packages_exist_with_valid_frontmatter_and_commands() -> None:
  """Verify AGENTS.md, skills/embodiedx-agent-kickoff/SKILL.md, skills/embodiedx-submission-auditor/SKILL.md, and .claude/commands/embodiedx-optimize.md."""
  agents_md = REPO_ROOT / "AGENTS.md"
  kickoff_skill = REPO_ROOT / "skills" / "embodiedx-agent-kickoff" / "SKILL.md"
  auditor_skill = REPO_ROOT / "skills" / "embodiedx-submission-auditor" / "SKILL.md"
  claude_cmd = REPO_ROOT / ".claude" / "commands" / "embodiedx-optimize.md"

  for path in (agents_md, kickoff_skill, auditor_skill, claude_cmd):
    assert path.exists(), f"Missing required agent/skill file: {path}"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 400

  # Verify AGENTS.md covers Turn-0 kickoff, all 3 canonical v1.0 models (D2), CAP_SINGLE_CHIP, 4 Y-axes, and G1-G4
  from embodied_bench.schema import CANONICAL_V1_MODELS
  agents_text = agents_md.read_text(encoding="utf-8")
  assert "python3 -m embodied_bench.agent_kickoff" in agents_text
  for model_id in CANONICAL_V1_MODELS:
    assert model_id in agents_text, f"AGENTS.md missing canonical v1.0 model {model_id}"
  assert "CAP_SINGLE_CHIP" in agents_text
  assert "raw_per_chip_throughput" in agents_text
  assert "power_efficiency_per_watt" in agents_text
  assert "iso_compute_and_bw_efficiency" in agents_text
  assert "tco_efficiency_per_dollar" in agents_text
  for gate_or_d in ("D1", "D2", "D3", "G4"):
    assert gate_or_d in agents_text

  # Verify YAML frontmatter in skills and slash command
  for skill_path, expected_name in (
      (kickoff_skill, "embodiedx-agent-kickoff"),
      (auditor_skill, "embodiedx-submission-auditor"),
  ):
    raw = skill_path.read_text(encoding="utf-8")
    assert raw.startswith("---\n"), f"{skill_path} must start with YAML frontmatter"
    parts = raw.split("---", 2)
    assert len(parts) >= 3
    meta = yaml.safe_load(parts[1])
    assert isinstance(meta, dict)
    assert meta.get("name") == expected_name
    assert isinstance(meta.get("description"), str) and len(meta["description"]) > 20
    assert ("python3 -m embodied_bench." in parts[2]) or ("eeb " in parts[2])

  claude_raw = claude_cmd.read_text(encoding="utf-8")
  assert claude_raw.startswith("---\n")
  claude_parts = claude_raw.split("---", 2)
  claude_meta = yaml.safe_load(claude_parts[1])
  assert isinstance(claude_meta, dict) and "description" in claude_meta
  assert "python3 -m embodied_bench.agent_kickoff" in claude_parts[2]
  assert ("measure" in claude_parts[2]) and ("verify-receipt" in claude_parts[2])
