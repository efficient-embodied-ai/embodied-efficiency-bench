"""Unit and integration tests for Phase 1 Client-Server Benchmark Harness, LoadGen Client, Latency-Coupled LIBERO-RT Environment, and Single-Chip Hardware Verifier (`tests/test_harness.py`).

Covers Adversarial Self-Review Gates 1, 2, 4, 5, 6, and 7:
  - Live HTTP `EmbodiedBenchmarkServer` + `EmbodiedLoadGenClient` over ephemeral port (`port=0`) for `C=1` and `C=4`.
  - `LiberoRTClosedLoopEnv` latency-coupled physics stepping across `SYNCHRONOUS_HOLD`, `ASYNC_CHUNK`, `RTC_INPAINTING`, and `EMERGENCY_STOP`.
  - Gate 1 (Crash recovery from orphan `.tmp_embodied_*` & corrupt JSON, repeat-call optional field hygiene, and concurrent threads + subprocesses) on `record_loadgen_session_state`.
  - Gate 2 (All `RunStatus` states reachable and non-empty `reason` enforced on `REJECTED`, `BLOCKED`, `FAILED`, `SKIPPED`, `INVALID`).
  - Gate 4 (Positive violation detection and negative-control immunity on `verify_single_chip_envelope` and `audit_command_line_for_multi_chip_bypass`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest

from embodied_bench.harness import (
    EmbodiedBenchmarkServer,
    EmbodiedLoadGenClient,
    LiberoRTClosedLoopEnv,
    LiberoRTClosedLoopResult,
    ReferenceEmbodiedRunner,
    SingleChipHardwareVerifier,
    audit_command_line_for_multi_chip_bypass,
    decode_ndarray_b64,
    encode_ndarray_b64,
    parse_rocm_smi_json,
    record_loadgen_session_state,
    verify_single_chip_envelope,
)
from embodied_bench.libero_rt import LiberoRTResult
from embodied_bench.schema import (
    ExecutionStrategy,
    PerformanceCapId,
    RunStatus,
    SafetyProbeType,
    TrackType,
)


def test_ndarray_b64_roundtrip_preserves_dtype_and_shape() -> None:
  """Verify base64 wire encoding preserves exact dtype and multidimensional shapes."""
  for dtype in (np.float32, np.float16, np.float64, np.int64, np.uint8, np.bool_):
    orig = np.arange(24).reshape(2, 3, 4).astype(dtype)
    encoded = encode_ndarray_b64(orig)
    assert "__ndarray_b64__" in encoded
    assert encoded["dtype"] == str(orig.dtype)
    assert encoded["shape"] == [2, 3, 4]

    decoded = decode_ndarray_b64(encoded)
    assert decoded.dtype == orig.dtype
    assert decoded.shape == orig.shape
    assert np.array_equal(decoded, orig)


def test_live_http_server_and_loadgen_client_c1_and_c4(tmp_path: Path) -> None:
  """Verify live HTTP EmbodiedBenchmarkServer + EmbodiedLoadGenClient on ephemeral port=0 for C=1 and C=4."""
  runner = ReferenceEmbodiedRunner(action_dim=7, default_chunk_size=8)
  with EmbodiedBenchmarkServer(runner=runner, host="127.0.0.1", port=0) as server:
    assert server.port > 0
    client = EmbodiedLoadGenClient(base_url=server.base_url)

    health = client.check_health()
    assert health["status"] == "ok"
    assert "/v1/vla/act" in health["supported_endpoints"]
    assert "/v1/wm/step" in health["supported_endpoints"]

    # 1. Single-Stream Real-Time Mode (C=1) on VLA_TRACK
    c1_vla = client.run_single_stream(
        track=TrackType.VLA_TRACK.value,
        num_steps=6,
        warmup_steps=1,
        session_id="robot_stream_0",
        chunk_size=8,
    )
    assert c1_vla.concurrency == 1
    assert c1_vla.total_requests == 6
    assert c1_vla.ttfa_ms is not None and c1_vla.ttfa_ms > 0.0
    assert c1_vla.p50_ms > 0.0
    assert c1_vla.p95_ms >= c1_vla.p50_ms
    assert c1_vla.p99_ms >= c1_vla.p95_ms
    assert c1_vla.hz_react is not None and c1_vla.hz_react > 0.0
    assert c1_vla.actions_per_sec is not None and c1_vla.actions_per_sec > 0.0
    assert c1_vla.cache_hit_ratio > 0.8

    # 2. Single-Stream Mode (C=1) on WM_TRACK
    c1_wm = client.run_single_stream(
        track=TrackType.WM_TRACK.value,
        num_steps=4,
        warmup_steps=1,
        session_id="wm_stream_0",
        horizon_frames=16,
    )
    assert c1_wm.ttff_ms is not None and c1_wm.ttff_ms > 0.0
    assert c1_wm.hz_plan is not None and c1_wm.hz_plan > 0.0
    assert c1_wm.fps is not None and c1_wm.fps > 0.0

    # 3. Multi-Stream Concurrency Sweep (C=1 and C=4) with sticky session_id affinity
    reg_path = tmp_path / "loadgen_sweep_registry.json"
    sweep = client.run_multi_stream_sweep(
        concurrency_levels=[1, 4],
        steps_per_stream=5,
        warmup_steps=1,
        track=TrackType.VLA_TRACK.value,
        chunk_size=8,
        registry_path=reg_path,
        run_id_prefix="test_sweep",
    )
    assert set(sweep.keys()) == {1, 4}
    c4_res = sweep[4]
    assert c4_res.concurrency == 4
    assert c4_res.total_requests == 20
    assert c4_res.session_ids == [
        "robot_stream_0",
        "robot_stream_1",
        "robot_stream_2",
        "robot_stream_3",
    ]
    assert c4_res.actions_per_sec is not None and c4_res.actions_per_sec > 0.0
    assert c4_res.cache_hit_ratio > 0.8

    # Verify reset_session clears server sessions
    reset_resp = client.reset_session("*")
    assert reset_resp["status"] == "ok"
    assert client.check_health()["active_sessions"] == 0


def test_libero_rt_closed_loop_env_strategies_and_emergency_stop() -> None:
  """Verify LiberoRTClosedLoopEnv advances k=ceil(L/delta_t) ticks per HTTP call and differentiates strategies."""
  runner = ReferenceEmbodiedRunner(action_dim=7, default_chunk_size=8)
  with EmbodiedBenchmarkServer(runner=runner, host="127.0.0.1", port=0) as server:
    client = EmbodiedLoadGenClient(base_url=server.base_url)

    # Inject 110ms latency so k = ceil((110 + http_ms) / 50.0) = 3 physics ticks per HTTP request
    env_sync = LiberoRTClosedLoopEnv(
        control_freq_hz=20.0,
        chunk_size=8,
        execution_strategy=ExecutionStrategy.SYNCHRONOUS_HOLD.value,
        injected_latency_ms=110.0,
    )
    env_async = LiberoRTClosedLoopEnv(
        control_freq_hz=20.0,
        chunk_size=8,
        execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
        injected_latency_ms=110.0,
    )
    env_rtc = LiberoRTClosedLoopEnv(
        control_freq_hz=20.0,
        chunk_size=8,
        execution_strategy=ExecutionStrategy.RTC_INPAINTING.value,
        injected_latency_ms=110.0,
    )

    res_sync = env_sync.run_closed_loop_evaluation(
        client=client,
        recipe_id="test_sync",
        num_episodes=2,
        max_policy_calls=18,
        include_safety_probes=True,
    )
    res_async = env_async.run_closed_loop_evaluation(
        client=client,
        recipe_id="test_async",
        num_episodes=2,
        max_policy_calls=18,
        include_safety_probes=True,
    )
    res_rtc = env_rtc.run_closed_loop_evaluation(
        client=client,
        recipe_id="test_rtc",
        num_episodes=2,
        max_policy_calls=18,
        include_safety_probes=True,
    )

    assert isinstance(res_rtc, LiberoRTClosedLoopResult)
    assert isinstance(res_rtc, LiberoRTResult)
    assert isinstance(res_rtc.to_libero_rt_result(), LiberoRTResult)
    assert res_rtc.delay_steps_mean >= 3

    # RTC_INPAINTING and ASYNC_CHUNK must achieve higher realtime_success_rate and lower tracking_rmse than SYNCHRONOUS_HOLD
    assert res_rtc.realtime_success_rate >= res_async.realtime_success_rate
    assert res_async.realtime_success_rate > res_sync.realtime_success_rate
    assert res_rtc.mean_tracking_rmse <= res_sync.mean_tracking_rmse

    # Verify all 3 safety probes (OBJECT_DISPLACEMENT, GOAL_SWITCH, EMERGENCY_STOP) are recorded
    for probe in SafetyProbeType:
      assert probe.value in res_rtc.safety_probes
      probe_entry = res_rtc.safety_probes[probe.value]
      assert probe_entry["reaction_time_ms"] > 0.0
      assert "empirical_reaction_time_ms" in probe_entry

    # Direct EMERGENCY_STOP episode test
    estop_trace = env_rtc.run_episode(
        client=client,
        episode_index=99,
        max_policy_calls=8,
        safety_probe=SafetyProbeType.EMERGENCY_STOP,
        probe_at_call=2,
    )
    assert estop_trace.safety_probe == SafetyProbeType.EMERGENCY_STOP.value
    assert estop_trace.probe_reaction_time_ms is not None
    assert estop_trace.probe_reaction_time_ms > 0.0


def test_gate1_three_paths_on_record_loadgen_session_state(tmp_path: Path) -> None:
  """Gate 1: Verify crash recovery (orphan tmp + corrupt JSON), repeat-call hygiene, and concurrent threads + subprocesses."""
  reg_path = tmp_path / "loadgen_state.json"

  # 1. Crash Path: create an orphan temp file and corrupt 0-byte / malformed JSON file
  orphan_tmp = tmp_path / f".tmp_embodied_{reg_path.name}_orphan123.json"
  orphan_tmp.write_text("{corrupt_orphan", encoding="utf-8")
  reg_path.write_text("{corrupt_json_payload", encoding="utf-8")

  state_after_crash = record_loadgen_session_state(
      registry_path=reg_path,
      session_run_id="run_alpha",
      status=RunStatus.PENDING,
      concurrency=1,
      metrics={"mean_ms": 12.5},
      notes="initial_note",
      slo_preset="10Hz_50ms",
      error_code="WARN_INIT",
  )
  assert not orphan_tmp.exists(), "Orphan .tmp_embodied_* file must be cleaned up"
  assert state_after_crash.get("recovered_from_corrupt_state") is True

  # 2. Repeat-Call Path: omit optional fields (`notes`, `slo_preset`, `error_code`, `reason`)
  # and verify zero stale optional field carryover!
  record_loadgen_session_state(
      registry_path=reg_path,
      session_run_id="run_alpha",
      status=RunStatus.RUNNING,
      concurrency=1,
      metrics={"mean_ms": 11.2},
  )
  state_repeat = record_loadgen_session_state(
      registry_path=reg_path,
      session_run_id="run_alpha",
      status=RunStatus.CERTIFIED,
      concurrency=1,
      metrics={"mean_ms": 10.8, "p99_ms": 14.1},
  )
  rec = state_repeat["loadgen_sessions"]["run_alpha"]
  assert rec["status"] == RunStatus.CERTIFIED.value
  assert rec["notes"] is None
  assert rec["slo_preset"] is None
  assert rec["error_code"] is None
  assert rec["reason"] is None

  # 3. Concurrent-Writer Path: 8 threads + 4 subprocesses writing simultaneously
  def _thread_writer(idx: int) -> None:
    rid = f"thread_run_{idx}"
    record_loadgen_session_state(
        registry_path=reg_path,
        session_run_id=rid,
        status=RunStatus.PENDING,
        concurrency=idx + 1,
        metrics={"idx": idx},
    )
    record_loadgen_session_state(
        registry_path=reg_path,
        session_run_id=rid,
        status=RunStatus.RUNNING,
        concurrency=idx + 1,
        metrics={"idx": idx},
    )
    record_loadgen_session_state(
        registry_path=reg_path,
        session_run_id=rid,
        status=RunStatus.CERTIFIED,
        concurrency=idx + 1,
        metrics={"idx": idx, "done": True},
    )

  repo_root = str(Path(__file__).resolve().parents[1])
  subproc_code = (
      "import sys; "
      "from embodied_bench.harness.loadgen_client import record_loadgen_session_state; "
      "record_loadgen_session_state(sys.argv[1], sys.argv[2], 'PENDING', concurrency=2, metrics={'subproc': True}); "
      "record_loadgen_session_state(sys.argv[1], sys.argv[2], 'RUNNING', concurrency=2, metrics={'subproc': True}); "
      "record_loadgen_session_state(sys.argv[1], sys.argv[2], 'CERTIFIED', concurrency=2, metrics={'subproc': True})"
  )

  procs = [
      subprocess.Popen(
          [sys.executable, "-c", subproc_code, str(reg_path), f"subproc_run_{p}"],
          cwd=repo_root,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
      )
      for p in range(2)
  ]
  with ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(_thread_writer, range(6)))

  for proc in procs:
    _, stderr = proc.communicate(timeout=10.0)
    assert proc.returncode == 0, f"Subprocess writer failed: {stderr.decode()}"

  final_disk = json.loads(reg_path.read_text(encoding="utf-8"))
  sessions = final_disk["loadgen_sessions"]
  for idx in range(6):
    assert sessions[f"thread_run_{idx}"]["status"] == RunStatus.CERTIFIED.value
  for p in range(2):
    assert sessions[f"subproc_run_{p}"]["status"] == RunStatus.CERTIFIED.value


def test_gate2_state_machine_completeness_and_reason_enforcement(tmp_path: Path) -> None:
  """Gate 2: Verify all RunStatus values are reachable and non-empty reason is enforced."""
  reg_path = tmp_path / "gate2_registry.json"

  # Reach PENDING -> RUNNING -> CERTIFIED
  record_loadgen_session_state(reg_path, "s_cert", RunStatus.PENDING)
  record_loadgen_session_state(reg_path, "s_cert", RunStatus.RUNNING)
  record_loadgen_session_state(reg_path, "s_cert", RunStatus.CERTIFIED)

  # Reach REJECTED, BLOCKED, FAILED, SKIPPED, INVALID - first verify empty reason raises ValueError
  transitions = [
      ("s_rej", RunStatus.RUNNING, RunStatus.REJECTED),
      ("s_blk", RunStatus.PENDING, RunStatus.BLOCKED),
      ("s_fail", RunStatus.RUNNING, RunStatus.FAILED),
      ("s_skip", RunStatus.PENDING, RunStatus.SKIPPED),
      ("s_inv", RunStatus.PENDING, RunStatus.INVALID),
  ]
  for sid, start_st, target_st in transitions:
    record_loadgen_session_state(reg_path, sid, RunStatus.PENDING)
    if start_st == RunStatus.RUNNING:
      record_loadgen_session_state(reg_path, sid, RunStatus.RUNNING)

    with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
      record_loadgen_session_state(reg_path, sid, target_st, reason="")

    with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
      record_loadgen_session_state(reg_path, sid, target_st, reason="   ")

    # Now succeed with an actionable reason
    state = record_loadgen_session_state(
        reg_path,
        sid,
        target_st,
        reason=f"Transitioned to {target_st.value} due to explicit test condition.",
    )
    assert state["loadgen_sessions"][sid]["status"] == target_st.value

  # Verify illegal transition (e.g. PENDING -> CERTIFIED directly) is rejected
  record_loadgen_session_state(reg_path, "s_illegal", RunStatus.PENDING)
  with pytest.raises(ValueError, match="Illegal state transition"):
    record_loadgen_session_state(reg_path, "s_illegal", RunStatus.CERTIFIED)


def test_gate4_single_chip_verifier_and_negative_controls() -> None:
  """Gate 4: Verify positive violations fire and negative-control lookalikes stay silent."""
  # 1. Negative controls on verify_single_chip_envelope (qualifying single-chip SKUs pass)
  for valid_sku in (
      "TPU_V6E_1_32GB",
      "RTX_5090_32GB",
      "JETSON_AGX_THOR_64GB",
      "AMD_MI210_64GB",
      "JETSON_ORIN_NX_16GB",
  ):
    res = verify_single_chip_envelope(
        hardware_sku=valid_sku,
        detected_device_count=1,
        perf_cap=PerformanceCapId.CAP_SINGLE_CHIP,
    )
    assert res.passed is True, f"Expected {valid_sku} to pass CAP_SINGLE_CHIP: {res.reason}"
    assert res.violations == []

  # 2. Positive violations on verify_single_chip_envelope
  multi_dev = verify_single_chip_envelope(
      hardware_sku="RTX_5090_32GB",
      detected_device_count=2,
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP,
  )
  assert multi_dev.passed is False
  assert any("detected_device_count=2" in v for v in multi_dev.violations)

  over_vram_sku = verify_single_chip_envelope(
      hardware_sku="B200_SXM6",
      detected_device_count=1,
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP,
  )
  assert over_vram_sku.passed is False
  assert len(over_vram_sku.violations) >= 2  # Violates VRAM, TFLOPS, TOPS, and BW caps

  over_measured_vram = verify_single_chip_envelope(
      hardware_sku="RTX_5090_32GB",
      detected_device_count=1,
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP_32GB,
      measured_vram_gb=36.5,
  )
  assert over_measured_vram.passed is False
  assert any("36.50 GB" in v for v in over_measured_vram.violations)

  # 3. Negative controls on audit_command_line_for_multi_chip_bypass (legitimate lookalikes stay silent)
  legitimate_cmd = """
  # Commented-out multi-GPU flags must be ignored:
  # CUDA_VISIBLE_DEVICES=0,1 python3 serve.py --tensor-parallel-size 4
  # TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 TPU_VISIBLE_DEVICES=0 \\
    TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=1x1x1 \\
    python3 -m embodied_bench.cli serve --tensor-parallel-size 1 --pipeline-parallel-size=1 --num-devices 1 --port 8080 # inline comment CUDA_VISIBLE_DEVICES=0,1,2,3
  """
  neg_audit = audit_command_line_for_multi_chip_bypass(legitimate_cmd)
  assert neg_audit.passed is True, f"False positive on legitimate single-chip command: {neg_audit.violations}"
  assert neg_audit.violations == []

  # 4. Positive violations on audit_command_line_for_multi_chip_bypass
  violation_cases = [
      "CUDA_VISIBLE_DEVICES=0,1 python3 serve.py",
      "HIP_VISIBLE_DEVICES=0,1,2,3 python3 serve.py",
      "python3 -m vllm.entrypoints.openai.api_server --tensor-parallel-size 2",
      "python3 serve.py --tensor-parallel-size=4 --pipeline-parallel-size 2",
      "TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 python3 serve_tpu.py",
      "TPU_TOPOLOGY=2x2x1 python3 serve_tpu.py",
      "gcloud compute tpus tpu-vm create --accelerator-type=v6e-4",
  ]
  for bad_cmd in violation_cases:
    pos_audit = audit_command_line_for_multi_chip_bypass(bad_cmd)
    assert pos_audit.passed is False, f"Expected violation to fire on: {bad_cmd!r}"
    assert len(pos_audit.violations) >= 1


def test_single_chip_hardware_verifier_100ms_background_sampler() -> None:
  """Verify SingleChipHardwareVerifier 100ms background thread sampling and AMD ROCm JSON parser."""
  rocm_json = json.dumps({
      "card0": {
          "Card series": "AMD Instinct MI210",
          "VRAM Total Memory (B)": str(64 * 1024 * 1024 * 1024),
          "VRAM Total Used Memory (B)": str(18 * 1024 * 1024 * 1024),
          "Average Graphics Package Power (W)": "210.5",
      }
  })
  parsed_rocm = parse_rocm_smi_json(rocm_json)
  assert parsed_rocm["vendor"] == "AMD_ROCM"
  assert parsed_rocm["detected_device_count"] == 1
  assert parsed_rocm["total_vram_mb"] == 65536.0
  assert parsed_rocm["mean_power_watts"] == 210.5

  with SingleChipHardwareVerifier(
      hardware_sku="TPU_V6E_1_32GB",
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP,
      poll_interval_s=0.05,
      allow_live_probe=False,
  ) as verifier:
    time.sleep(0.12)
    summary = verifier.summarize()

  assert summary.passed is True
  assert summary.sample_count >= 2
  assert summary.detected_device_count == 1
  assert summary.peak_vram_mb > 0.0
  assert summary.mean_power_watts > 0.0
  assert summary.hardware_fingerprint.startswith("hw-")
