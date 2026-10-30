"""Tests for `eeb compare` (`eeb/loadgen/compare.py`) and the Agent & External Harness Startup Skill.

Covers Adversarial Self-Review Gates 1, 2, 3, and 4:
  - Gate 1: Crash-orphan cleanup, repeat-call overwrites, and concurrent thread/process safety on `write_comparison_json`.
  - Gate 2: Reachability and mandatory non-empty `reason` across `ACCEPT` and every `REJECT_*` decision code.
  - Gate 3: Execution of `eeb compare` CLI in both flag orders (file paths and bundle directories, table and `--json`).
  - Gate 4: Positive violation checks and passing lookalikes for quality gates, horizon locks, reactivity cutoffs,
    variance exclusion, synthetic policy provenance, single-chip device count, and Track 2 Closed stream_chunk_frames ban + TTFF diagnostic reporting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List
from unittest import mock

import pytest

from eeb.cli import build_parser, main as eeb_main
from eeb.loadgen.compare import (
    compare_reports,
    format_comparison_markdown,
    write_comparison_json,
)
from embodied_bench.agent_kickoff import parse_kickoff_prompt, run_agent_kickoff

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_vla_report(
    *,
    points: List[Dict[str, Any]],
    model_id: str = "pi05-3b",
    division: str = "CLOSED",
    execute_horizon: int = 5,
    synthetic: bool = False,
    num_devices: int = 1,
    qg_passed: bool = True,
    quality_certified: bool = True,
    gate_mode: str = "GOLDEN_REPLAY",
) -> Dict[str, Any]:
  ops = []
  for p in points:
    c = int(p["concurrency"])
    ops.append({
        "point_id": f"C{c}",
        "concurrency": c,
        "execute_horizon": p.get("execute_horizon", execute_horizon),
        "hz_react": float(p["hz_react"]),
        "executed_actions_per_sec": float(p["executed_actions_per_sec"]),
        "actions_per_sec": float(p["executed_actions_per_sec"]) * 10.0,
        "power_efficiency_per_watt": p.get("power_efficiency_per_watt", 0.5),
        "variance_passed": p.get("variance_passed", True),
        "gate_a_passed": p.get("gate_a_passed", qg_passed),
        "gate_a_mode": p.get("gate_a_mode", gate_mode),
        "quality_certified": p.get("quality_certified", quality_certified),
        "hardware_telemetry": {"power_provenance": p.get("power_provenance", "MEASURED")},
    })
  return {
      "schema_version": "1.0.0",
      "report_kind": "eeb_measurement_report",
      "recipe_id": f"track1_vla_{model_id}_rtx5090",
      "track": "VLA_TRACK",
      "baseline_model_id": model_id,
      "model_family": model_id,
      "division": division,
      "hardware_sku": "RTX_5090_32GB",
      "num_devices": num_devices,
      "sweep_config": {"execute_horizon": execute_horizon},
      "execution_provenance": {
          "policy": "SYNTHETIC_DUMMY" if synthetic else "TORCH_CUDA",
          "policy_is_synthetic": synthetic,
          "power": "MEASURED",
      },
      "quality_gate": {
          "passed": qg_passed,
          "quality_certified": quality_certified,
          "mode": gate_mode,
      },
      "operating_points": ops,
  }


def _make_wm_report(
    *,
    points: List[Dict[str, Any]],
    horizon_frames: int = 16,
    ttff_b1_ms: float = 42.0,
) -> Dict[str, Any]:
  ops = []
  for p in points:
    b = int(p["concurrency"])
    ops.append({
        "point_id": f"C{b}",
        "concurrency": b,
        "batch_size": b,
        "horizon_frames": horizon_frames,
        "execute_horizon": horizon_frames,
        "hz_plan": float(p["hz_plan"]),
        "hz_react": float(p["hz_plan"]),
        "predictions_per_sec": float(p.get("predictions_per_sec", float(p["generated_frames_per_sec"]) / max(horizon_frames, 1))),
        "generated_frames_per_sec": float(p["generated_frames_per_sec"]),
        "fps": float(p["generated_frames_per_sec"]),
        "ttff_p99_ms": ttff_b1_ms if b == 1 else 65.0,
        "power_efficiency_per_watt": p.get("power_efficiency_per_watt"),
        "variance_passed": p.get("variance_passed", True),
        "gate_a_passed": p.get("gate_a_passed", True),
        "gate_a_mode": "GOLDEN_REPLAY",
        "quality_certified": True,
        "hardware_telemetry": {
            "power_provenance": "UNAVAILABLE" if p.get("power_efficiency_per_watt") is None else "MEASURED"
        },
    })
  return {
      "schema_version": "1.0.0",
      "report_kind": "eeb_measurement_report",
      "recipe_id": "track2_wm_cosmos_predict25_rtx5090",
      "track": "WM_TRACK",
      "baseline_model_id": "cosmos-predict2.5-2b",
      "model_family": "cosmos-predict2.5-2b",
      "division": "CLOSED",
      "hardware_sku": "RTX_5090_32GB",
      "num_devices": 1,
      "sweep_config": {"execute_horizon": horizon_frames},
      "execution_provenance": {
          "policy": "TORCH_CUDA",
          "policy_is_synthetic": False,
          "power": "MEASURED",
      },
      "quality_gate": {"passed": True, "quality_certified": True, "mode": "GOLDEN_REPLAY"},
      "operating_points": ops,
  }


def _write_bundle(dir_path: Path, report_doc: Dict[str, Any], filename: str = "report.json") -> Path:
  dir_path.mkdir(parents=True, exist_ok=True)
  target = dir_path / filename
  target.write_text(json.dumps(report_doc, indent=2, sort_keys=True), encoding="utf-8")
  return target


def test_gate2_and_gate4_compare_accept_and_all_rejection_codes(tmp_path: Path) -> None:
  base = _make_vla_report(
      points=[
          {"concurrency": 1, "hz_react": 34.0, "executed_actions_per_sec": 170.0, "power_efficiency_per_watt": 0.60},
          {"concurrency": 4, "hz_react": 22.5, "executed_actions_per_sec": 420.0, "power_efficiency_per_watt": 1.35},
          {"concurrency": 16, "hz_react": 12.0, "executed_actions_per_sec": 780.0, "power_efficiency_per_watt": 2.40},
      ]
  )
  b_dir = tmp_path / "base_vla"
  _write_bundle(b_dir, base)

  # 1. Genuine >= 2.0% win at cutoff >= 20 Hz -> ACCEPT (exit 0)
  cand_win = _make_vla_report(
      points=[
          {"concurrency": 1, "hz_react": 38.0, "executed_actions_per_sec": 190.0, "power_efficiency_per_watt": 0.68},
          {"concurrency": 4, "hz_react": 24.0, "executed_actions_per_sec": 462.0, "power_efficiency_per_watt": 1.50},
          {"concurrency": 16, "hz_react": 13.5, "executed_actions_per_sec": 850.0, "power_efficiency_per_watt": 2.65},
      ]
  )
  c_win_path = _write_bundle(tmp_path / "cand_win", cand_win)
  res_win = compare_reports(b_dir, c_win_path)
  assert res_win["verdict"] == "ACCEPT"
  assert res_win["accepted"] is True
  assert res_win["exit_code"] == 0
  assert res_win["decision_code"] == "ACCEPT_IMPROVED_AT_CUTOFF"
  assert res_win["cutoff_hz"] == 4.0
  assert res_win["metrics"]["best_output_at_cutoff"]["baseline"] == 780.0
  assert res_win["metrics"]["best_output_at_cutoff"]["candidate"] == 850.0
  assert res_win["metrics"]["best_output_at_cutoff"]["delta_pct"] == pytest.approx((850.0 - 780.0) / 780.0 * 100.0, rel=1e-3)

  # 2. First real-time qualifying point when baseline had none at >= 4.0 Hz -> ACCEPT_FIRST_REALTIME_POINT
  slow_base = _make_vla_report(
      points=[{"concurrency": 1, "hz_react": 3.5, "executed_actions_per_sec": 82.0}]
  )
  res_first = compare_reports(_write_bundle(tmp_path / "slow_base", slow_base), c_win_path)
  assert res_first["verdict"] == "ACCEPT"
  assert res_first["decision_code"] == "ACCEPT_FIRST_REALTIME_POINT"

  # 3. Noise-floor rejection (+0.95% < 2.0%) -> REJECT_BELOW_NOISE_FLOOR
  cand_noise = _make_vla_report(
      points=[
          {"concurrency": 1, "hz_react": 34.5, "executed_actions_per_sec": 172.0},
          {"concurrency": 16, "hz_react": 12.1, "executed_actions_per_sec": 786.0},
      ]
  )
  res_noise = compare_reports(b_dir, _write_bundle(tmp_path / "cand_noise", cand_noise))
  assert res_noise["verdict"] == "REJECT" and res_noise["exit_code"] == 2
  assert res_noise["decision_code"] == "REJECT_BELOW_NOISE_FLOOR" and res_noise["reason"]

  # 4. Throughput regression at cutoff -> REJECT_THROUGHPUT_REGRESSION
  cand_reg = _make_vla_report(
      points=[{"concurrency": 1, "hz_react": 30.0, "executed_actions_per_sec": 380.0}]
  )
  res_reg = compare_reports(b_dir, _write_bundle(tmp_path / "cand_reg", cand_reg))
  assert res_reg["decision_code"] == "REJECT_THROUGHPUT_REGRESSION" and res_reg["exit_code"] == 2

  # 5. Reactivity regression: huge batch output at 3.0 Hz, zero points >= 4.0 Hz -> REJECT_REACTIVITY_REGRESSION
  cand_drop_hz = _make_vla_report(
      points=[{"concurrency": 16, "hz_react": 3.0, "executed_actions_per_sec": 1500.0}]
  )
  res_drop_hz = compare_reports(b_dir, _write_bundle(tmp_path / "cand_drop_hz", cand_drop_hz))
  assert res_drop_hz["decision_code"] == "REJECT_REACTIVITY_REGRESSION" and res_drop_hz["reason"]

  # 6. Neither baseline nor candidate meets cutoff -> REJECT_REACTIVITY_CUTOFF_UNMET
  res_unmet = compare_reports(
      _write_bundle(tmp_path / "slow_base2", slow_base),
      _write_bundle(tmp_path / "slow_cand2", copy.deepcopy(slow_base)),
  )
  assert res_unmet["decision_code"] == "REJECT_REACTIVITY_CUTOFF_UNMET"

  # 7. Quality gate failures: passed=False AND SANITY_ONLY without --allow-sanity-only (plus passing lookalike with flag)
  cand_bad_gate = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0}],
      qg_passed=False,
  )
  res_bg = compare_reports(b_dir, _write_bundle(tmp_path / "cand_bg", cand_bad_gate))
  assert res_bg["decision_code"] == "REJECT_QUALITY_GATE"

  cand_sanity = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 850.0}],
      quality_certified=False,
      gate_mode="SANITY_ONLY",
  )
  p_sanity = _write_bundle(tmp_path / "cand_sanity", cand_sanity)
  assert compare_reports(b_dir, p_sanity)["decision_code"] == "REJECT_QUALITY_GATE"
  assert compare_reports(b_dir, p_sanity, allow_sanity_only=True)["verdict"] == "ACCEPT"

  # 8. Variance instability: C=4 fast point has variance_passed=False so excluded -> REJECT
  cand_var = _make_vla_report(
      points=[
          {"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0, "variance_passed": False},
      ]
  )
  res_var = compare_reports(b_dir, _write_bundle(tmp_path / "cand_var", cand_var))
  assert res_var["decision_code"] == "REJECT_VARIANCE_UNSTABLE"

  # 9. Horizon, Model, Track, Division, Synthetic, Multi-device, and CPU no-device mismatches
  cand_h8 = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0}],
      execute_horizon=8,
  )
  assert compare_reports(b_dir, _write_bundle(tmp_path / "cand_h8", cand_h8))["decision_code"] == "REJECT_HORIZON_MISMATCH"

  # Both reports tampered to horizon=10 for pi05-3b in CLOSED still rejected by canonical locked horizon K=5
  base_h10 = _make_vla_report(points=[{"concurrency": 4, "hz_react": 22.0, "executed_actions_per_sec": 400.0}], execute_horizon=10)
  cand_h10 = _make_vla_report(points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 500.0}], execute_horizon=10)
  assert (
      compare_reports(
          _write_bundle(tmp_path / "b_h10", base_h10),
          _write_bundle(tmp_path / "c_h10", cand_h10),
      )["decision_code"]
      == "REJECT_HORIZON_MISMATCH"
  )

  cand_wrong_model = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0}],
      model_id="gr00t-n1.7-3b",
      execute_horizon=8,
  )
  assert compare_reports(b_dir, _write_bundle(tmp_path / "cand_model", cand_wrong_model))["decision_code"] == "REJECT_MODEL_MISMATCH"

  cand_open = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0}],
      division="OPEN",
  )
  assert compare_reports(b_dir, _write_bundle(tmp_path / "cand_open", cand_open))["decision_code"] == "REJECT_DIVISION_MISMATCH"

  cand_syn = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0}],
      synthetic=True,
  )
  assert compare_reports(b_dir, _write_bundle(tmp_path / "cand_syn", cand_syn))["decision_code"] == "REJECT_SYNTHETIC_CANDIDATE"

  cand_2gpu = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0}],
      num_devices=2,
  )
  assert compare_reports(b_dir, _write_bundle(tmp_path / "cand_2gpu", cand_2gpu))["decision_code"] == "REJECT_MULTI_DEVICE"

  cand_cpu = _make_vla_report(
      points=[{"concurrency": 4, "hz_react": 25.0, "executed_actions_per_sec": 600.0}],
      num_devices=0,
  )
  assert compare_reports(b_dir, _write_bundle(tmp_path / "cand_cpu", cand_cpu))["decision_code"] == "REJECT_NO_DEVICE_CANDIDATE"

  # 10. Track 2 WM comparison (default cutoff_hz == 1.5 Rollouts/s [24 FPS, L_p99 <= 666.7 ms];
  #     TTFF_p99 == L_clip is a co-reported D7 diagnostic on full-clip bidirectional 3D-DiTs, while
  #     stream_chunk_frames > 0 is banned in CLOSED division)
  wm_base = _make_wm_report(
      points=[
          {"concurrency": 1, "hz_plan": 2.4, "generated_frames_per_sec": 38.4, "power_efficiency_per_watt": None},
          {"concurrency": 2, "hz_plan": 1.6, "generated_frames_per_sec": 51.2, "power_efficiency_per_watt": None},
      ],
      ttff_b1_ms=416.7,
  )
  wm_cand_ok = _make_wm_report(
      points=[
          {"concurrency": 1, "hz_plan": 2.8, "generated_frames_per_sec": 44.8, "power_efficiency_per_watt": None},
          {"concurrency": 2, "hz_plan": 1.8, "generated_frames_per_sec": 57.6, "power_efficiency_per_watt": None},
      ],
      ttff_b1_ms=357.1,
  )
  wm_b_path = _write_bundle(tmp_path / "wm_base", wm_base, filename="sweep_results.json")
  wm_c_path = _write_bundle(tmp_path / "wm_cand_ok", wm_cand_ok)
  assert compare_reports(b_dir, wm_c_path)["decision_code"] == "REJECT_TRACK_MISMATCH"
  res_wm = compare_reports(wm_b_path.parent, wm_c_path.parent)
  assert res_wm["verdict"] == "ACCEPT"
  assert res_wm["cutoff_hz"] == 1.0
  md_wm = format_comparison_markdown(res_wm)
  assert "Base Pred/s -> Cand Pred/s" in md_wm and "N/A" in md_wm and "\u2014" not in md_wm and "---\n" not in md_wm

  # Full-clip 3D-DiT at B=1 with L_p99 = TTFF_p99 = 620 ms (1.61 Rollouts/s >= 1.5) MUST be accepted
  wm_cand_fullclip_ttff = _make_wm_report(
      points=[{"concurrency": 1, "hz_plan": 1.6129, "generated_frames_per_sec": 60.0}],
      ttff_b1_ms=620.0,
  )
  assert (
      compare_reports(wm_b_path, _write_bundle(tmp_path / "wm_fullclip_ttff", wm_cand_fullclip_ttff))["verdict"]
      == "ACCEPT"
  )

  # Negative control: Closed division pseudo-causal stream_chunk_frames=4 is rejected
  wm_cand_chunked = _make_wm_report(
      points=[{"concurrency": 1, "hz_plan": 2.8, "generated_frames_per_sec": 60.0}],
      ttff_b1_ms=90.0,
  )
  wm_cand_chunked["recipe"] = {"stream_chunk_frames": 4}
  res_chunked = compare_reports(wm_b_path, _write_bundle(tmp_path / "wm_chunked", wm_cand_chunked))
  assert res_chunked["decision_code"] == "REJECT_CLOSED_STREAM_CHUNKING"
  assert res_chunked["reason"]


def test_gate1_and_gate3_compare_cli_flag_orders_and_atomic_output(tmp_path: Path) -> None:
  assert "compare" in build_parser().format_help()

  b_dir = tmp_path / "bundle_base"
  c_dir = tmp_path / "bundle_cand"
  _write_bundle(b_dir, _make_vla_report(points=[{"concurrency": 1, "hz_react": 30.0, "executed_actions_per_sec": 150.0}]))
  _write_bundle(c_dir, _make_vla_report(points=[{"concurrency": 1, "hz_react": 36.0, "executed_actions_per_sec": 180.0}]))

  # Crash recovery + both flag orders via CLI
  out_json = tmp_path / "out_dir" / "compare.json"
  out_json.parent.mkdir(parents=True, exist_ok=True)
  stale_tmp = out_json.parent / ".compare.json.999.tmp"
  stale_tmp.write_text("{corrupt", encoding="utf-8")

  with mock.patch("sys.stdout", new_callable=io.StringIO) as buf1:
    rc1 = eeb_main(["compare", "--baseline", str(b_dir), "--candidate", str(c_dir), "--out", str(out_json)])
  assert rc1 == 0
  assert not stale_tmp.exists()
  assert "verdict=ACCEPT" in buf1.getvalue()
  assert "\u2014" not in buf1.getvalue()

  with mock.patch("sys.stdout", new_callable=io.StringIO) as buf2:
    rc2 = eeb_main([
        "compare",
        "--json",
        "--min-gain-pct",
        "5.0",
        "--cutoff-hz",
        "25.0",
        "--out",
        str(out_json),
        "--candidate",
        str(c_dir / "report.json"),
        "--baseline",
        str(b_dir / "report.json"),
    ])
  assert rc2 == 0
  doc2 = json.loads(buf2.getvalue())
  assert doc2["accepted"] is True and doc2["min_gain_pct"] == 5.0 and doc2["cutoff_hz"] == 25.0

  # Reversed slow vs fast -> exit code 2
  assert eeb_main(["compare", "--baseline", str(c_dir), "--candidate", str(b_dir), "--json"]) == 2

  # Concurrent thread + subprocess writes on out_json leave valid JSON and zero in-tree .lock files
  def _t_worker(i: int) -> None:
    d = dict(doc2)
    d["worker"] = i
    write_comparison_json(d, out_json)

  with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(_t_worker, range(8)))

  sub_code = (
      "import json, sys\n"
      "from eeb.loadgen.compare import write_comparison_json\n"
      "write_comparison_json({'verdict': 'ACCEPT', 'proc': int(sys.argv[2])}, sys.argv[1])\n"
  )
  procs = [
      subprocess.Popen(
          [sys.executable, "-c", sub_code, str(out_json), str(i)],
          cwd=str(REPO_ROOT),
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      for i in range(3)
  ]
  for p in procs:
    _o, err = p.communicate(timeout=20)
    assert p.returncode == 0, err
  assert isinstance(json.loads(out_json.read_text(encoding="utf-8")), dict)
  assert list(tmp_path.rglob("*.lock")) == []
  assert list(tmp_path.rglob(".*.lock")) == []


def test_agent_kickoff_v1_cosmos_predict25_closed_default_and_framework_docs(tmp_path: Path) -> None:
  parsed = parse_kickoff_prompt("Start Cosmos-Predict2.5 on TPU v6e-1")
  assert parsed["model_id"] == "cosmos-predict2.5-2b"
  assert parsed["hardware_sku"] == "TPU_V6E_1_32GB"

  plan = run_agent_kickoff(
      prompt="Start pi0.5 on RTX 5090",
      study_dir=tmp_path / "kickoff_v1_default",
  )
  assert plan["model_id"] == "pi05-3b"
  assert plan["division"] == "CLOSED"
  assert plan["passed"] is True
  expanded = (tmp_path / "kickoff_v1_default" / "EXPANDED_AGENT_PROMPT.md").read_text(encoding="utf-8")
  assert "eeb compare" in expanded
  assert "eeb measure" in expanded
  assert "\u2014" not in expanded

  skill_text = (REPO_ROOT / "skills" / "embodiedx-agent-kickoff" / "SKILL.md").read_text(encoding="utf-8")
  readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
  for doc_name, text in (("SKILL.md", skill_text), ("README.md", readme_text)):
    assert "\u2014" not in text, f"{doc_name} contains Unicode em-dash"
    assert "eeb compare" in text, f"{doc_name} missing eeb compare"
    for fw in (
        "google-pytorch/tpu-agent-suite",
        "AI-Hypercomputer/accelerator-agents",
        "algorithmicsuperintelligence/openevolve",
        "karpathy/autoresearch",
        "pub.sakana.ai/ai-cuda-engineer",
        "ScalingIntelligence/KernelBench",
    ):
      assert fw in text, f"{doc_name} missing framework reference {fw}"
    # TPU Agent Suite must appear first among Cloud TPU agent frameworks
    assert text.index("google-pytorch/tpu-agent-suite") < text.index("AI-Hypercomputer/accelerator-agents")
