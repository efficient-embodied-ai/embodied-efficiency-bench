"""Comprehensive unit and adversarial self-review tests (Gates 1-7) for EmbodiedX Phase 3 Reference Baseline Model Runners."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import statistics
from typing import Any, Dict

import numpy as np
import pytest

from embodied_bench.baselines import (
    SUPPORTED_BACKENDS,
    SUPPORTED_PRECISIONS,
    BaselineRunnerConfig,
    Cosmos3WorldModelRunner,
    CosmosPredict25WorldModelRunner,
    GR00TN17Runner,
    Pi05OpenPiRunner,
    Qwen3OmniRunner,
    create_baseline_runner,
    quantize_dequantize_array,
    record_runner_benchmark_state,
    resolve_canonical_model_id,
)
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    REASON_REQUIRED_STATUSES,
    RunStatus,
    TrackType,
)


def _make_sample_observation(height: int = 32, width: int = 32) -> Dict[str, Any]:
  rng = np.random.default_rng(1234)
  rgb = rng.integers(80, 175, size=(2, height, width, 3), dtype=np.uint8)
  proprio = np.array([0.15, -0.25, 0.35, -0.10, 0.40, -0.30, 0.20, 0.05], dtype=np.float32)
  t_grid = np.linspace(0.0, 1.0, 16, dtype=np.float32)[:, None]
  d_grid = np.arange(7, dtype=np.float32)[None, :]
  actions = (0.6 * np.sin(2.0 * np.pi * t_grid + 0.5 * d_grid)).astype(np.float32)
  audio = (0.3 * np.sin(np.linspace(0.0, 12.56, 1600, dtype=np.float32))).astype(
      np.float32
  )
  return {
      "rgb": rgb,
      "context_frames": rgb,
      "proprio": proprio,
      "state": proprio,
      "actions": actions,
      "audio": audio,
      "instruction": "pick up the red block and place it on the blue plate",
  }


def _concurrent_subprocess_worker(args: tuple[str, str, str]) -> str:
  state_path_str, run_id, model_id = args
  record_runner_benchmark_state(
      state_path=state_path_str,
      run_id=run_id,
      model_id=model_id,
      status=RunStatus.PENDING,
      metrics={"step": 0},
  )
  record_runner_benchmark_state(
      state_path=state_path_str,
      run_id=run_id,
      model_id=model_id,
      status=RunStatus.RUNNING,
      metrics={"step": 1},
      backend="numpy_cpu",
      precision="bf16",
      denoising_steps=8,
  )
  record_runner_benchmark_state(
      state_path=state_path_str,
      run_id=run_id,
      model_id=model_id,
      status=RunStatus.CERTIFIED,
      metrics={"latency_ms": 12.5},
  )
  return run_id


class TestFactoryAndCanonicalModelResolution:
  """Verify factory resolution across all 3 canonical EEB v1.0 challenge models and aliases."""

  def test_all_four_canonical_models_instantiate(self) -> None:
    expected_classes = {
        "gr00t-n1.7-3b": GR00TN17Runner,
        "pi05-3b": Pi05OpenPiRunner,
        "cosmos-predict2.5-2b": CosmosPredict25WorldModelRunner,
        "cosmos3-edge": Cosmos3WorldModelRunner,
        "qwen3-omni-30b-a3b": Qwen3OmniRunner,
    }
    for model_id in CANONICAL_CHALLENGE_MODELS:
      runner = create_baseline_runner(model_id, resolution_hw=32)
      assert isinstance(runner, expected_classes[model_id])
      assert runner.canonical_model_id == model_id

  @pytest.mark.parametrize(
      ("alias", "expected_canonical"),
      [
          ("nvidia/GR00T-N1.7-3B", "gr00t-n1.7-3b"),
          ("GR00T-N1.7-3B", "gr00t-n1.7-3b"),
          ("nvidia/GR00T-N1.7-LIBERO", "gr00t-n1.7-3b"),
          ("lerobot/pi05_libero_finetuned_v044", "pi05-3b"),
          ("PhysicalIntelligence/pi0.5-libero", "pi05-3b"),
          ("openpi-pi05-3b", "pi05-3b"),
          ("nvidia/Cosmos-Predict2.5-2B", "cosmos-predict2.5-2b"),
          ("Cosmos-Predict2.5-2B", "cosmos-predict2.5-2b"),
          ("nvidia/Cosmos3-Edge", "cosmos3-edge"),
          ("nvidia/Cosmos3-Edge-Policy-DROID", "cosmos3-edge"),
          ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen3-omni-30b-a3b"),
          ("Qwen3-Omni-30B-A3B", "qwen3-omni-30b-a3b"),
      ],
  )
  def test_checkpoint_aliases_resolve_correctly(
      self, alias: str, expected_canonical: str
  ) -> None:
    assert resolve_canonical_model_id(alias) == expected_canonical
    runner = create_baseline_runner(alias, resolution_hw=32)
    assert runner.canonical_model_id == expected_canonical


class TestCanonicalBaselineRunnersExecution:
  """Verify shapes, dtypes, determinism, RTC inpainting, Inverse-Dynamics probe, and Sparse MoE routing."""

  def test_gr00t_n17_runner_shapes_and_determinism(self) -> None:
    obs = _make_sample_observation()
    r1 = create_baseline_runner("gr00t-n1.7-3b", seed=42, denoising_steps=8)
    r2 = create_baseline_runner("gr00t-n1.7-3b", seed=42, denoising_steps=8)
    r_diff = create_baseline_runner("gr00t-n1.7-3b", seed=99, denoising_steps=8)

    out1 = r1.infer_step(obs, session_id="s-gr00t")
    out2 = r2.infer_step(obs, session_id="s-gr00t")
    out_diff = r_diff.infer_step(obs, session_id="s-gr00t")

    actions = out1["actions"]
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (16, 7)
    assert actions.dtype == np.float32
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
    assert out1["track"] == TrackType.VLA_TRACK.value
    assert out1["latency_ms"] > 0.0
    assert np.allclose(actions, out2["actions"], atol=1e-6)
    assert not np.allclose(actions, out_diff["actions"], atol=1e-3)

  def test_pi05_openpi_runner_euler_ode_and_rtc_inpainting(self) -> None:
    obs = _make_sample_observation()
    runner = create_baseline_runner(
        "pi05-3b",
        seed=42,
        denoising_steps=6,
        backend="jax_tpu",
    )
    out_no_rtc = runner.infer_step(obs, session_id="s-pi05")
    actions = out_no_rtc["actions"]
    assert actions.shape == (10, 7)
    assert actions.dtype == np.float32
    assert out_no_rtc["rtc_applied"] is False
    assert out_no_rtc["rtc_inpainted_steps"] == 0

    # Supply previous_chunk_tail for Real-Time Chunking (RTC) prefix inpainting
    target_tail = np.full((5, 7), 0.75, dtype=np.float32)
    out_rtc = runner.infer_step(
        obs,
        session_id="s-pi05",
        previous_chunk_tail=target_tail,
    )
    assert out_rtc["rtc_applied"] is True
    assert out_rtc["rtc_inpainted_steps"] == 5
    assert out_rtc["actions"].shape == (10, 7)
    # First action with RTC inpainting must be pinned closely to target_tail[0]
    err_with_rtc = float(np.linalg.norm(out_rtc["actions"][0] - target_tail[0]))
    err_without_rtc = float(np.linalg.norm(actions[0] - target_tail[0]))
    assert err_with_rtc < err_without_rtc
    assert np.allclose(out_rtc["actions"][0], target_tail[0], atol=1e-4)

    # Explicit action_chunk_size=50 override remains supported for DROID-horizon experiments
    droid_runner = create_baseline_runner(
        "pi05-3b",
        seed=42,
        action_chunk_size=50,
        denoising_steps=6,
        backend="jax_tpu",
    )
    assert droid_runner.infer_step(obs, session_id="s-pi05-50")["actions"].shape == (50, 7)

  def test_cosmos3_wm_runner_rollout_and_inverse_dynamics_probe(self) -> None:
    obs = _make_sample_observation(height=32, width=32)
    runner = create_baseline_runner(
        "cosmos3-edge",
        seed=42,
        denoising_steps=8,
        horizon_frames=16,
    )
    assert isinstance(runner, Cosmos3WorldModelRunner)
    out = runner.infer_step(obs, session_id="s-cosmos")
    frames = out["predicted_frames"]
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (16, 32, 32, 3)
    assert frames.dtype == np.uint8
    assert out["track"] == TrackType.WM_TRACK.value

    recovered = runner.predict_actions_from_rollout(
        context_frame=obs["context_frames"][-1],
        predicted_frames=frames,
    )
    assert recovered.shape == (16, 7)
    assert recovered.dtype == np.float32
    assert out["inv_dyn_r2"] > 0.92
    assert out["controllability_cosine_sim"] > 0.98

    # Negative control for Inverse-Dynamics Probe: negated action trajectory must fail R2 gate
    wrong_metrics = runner.compute_inverse_dynamics_metrics(
        true_actions=-obs["actions"],
        recovered_actions=recovered,
    )
    assert wrong_metrics["inv_dyn_r2"] < 0.0
    assert wrong_metrics["controllability_cosine_sim"] < -0.90

  def test_qwen3_omni_runner_sparse_moe_routing_and_thinker_talker(self) -> None:
    obs = _make_sample_observation(height=32, width=32)
    runner = create_baseline_runner(
        "qwen3-omni-30b-a3b",
        seed=42,
        denoising_steps=4,
        horizon_frames=16,
    )
    assert isinstance(runner, Qwen3OmniRunner)
    out = runner.infer_step(obs, session_id="s-qwen")

    assert out["predicted_frames"].shape == (16, 32, 32, 3)
    assert out["predicted_frames"].dtype == np.uint8
    assert out["thinker_tokens"].shape == (16,)
    assert out["thinker_tokens"].dtype == np.int32
    assert isinstance(out["thinker_text"], str) and len(out["thinker_text"]) > 0
    assert out["actions"].shape == (16, 7)
    assert out["actions"].dtype == np.float32

    moe_stats = out["moe_routing_stats"]
    assert moe_stats["num_experts"] == 16
    assert moe_stats["top_k"] == 2
    assert pytest.approx(moe_stats["active_expert_fraction"], rel=1e-5) == 2.0 / 16.0
    assert (
        moe_stats["executed_token_expert_evals"] * 8
        == moe_stats["dense_equivalent_evals"]
    )
    assert sum(moe_stats["expert_load_counts"]) == moe_stats["executed_token_expert_evals"]


class TestSessionPrefixCacheAndStepScaling:
  """Verify per-session KV prefix caching, LRU eviction, and step-reduction wall-clock scaling."""

  @pytest.mark.parametrize("model_id", list(CANONICAL_CHALLENGE_MODELS))
  def test_session_prefix_cache_hits_and_lru_eviction(self, model_id: str) -> None:
    obs = _make_sample_observation(height=32, width=32)
    runner = create_baseline_runner(
        model_id,
        seed=42,
        denoising_steps=4,
        max_sessions=3,
        use_prefix_cache=True,
    )

    out_miss = runner.infer_step(obs, session_id="sess-A")
    assert out_miss["prefix_cache_hit"] is False
    stats1 = runner.cache_stats()
    assert stats1["cache_misses"] == 1
    assert stats1["cache_hits"] == 0

    out_hit = runner.infer_step(obs, session_id="sess-A")
    assert out_hit["prefix_cache_hit"] is True
    stats2 = runner.cache_stats()
    assert stats2["cache_misses"] == 1
    assert stats2["cache_hits"] == 1
    assert stats2["hit_rate"] == 0.5

    # Verify LRU eviction when exceeding max_sessions=3
    runner.infer_step(obs, session_id="sess-B")
    runner.infer_step(obs, session_id="sess-C")
    # Touch sess-A so sess-B becomes oldest
    runner.infer_step(obs, session_id="sess-A")
    # Insert sess-D -> should evict sess-B
    runner.infer_step(obs, session_id="sess-D")

    stats_lru = runner.cache_stats()
    assert stats_lru["active_sessions"] == 3
    assert stats_lru["evictions"] == 1
    # sess-A should still be a cache hit, while sess-B was evicted and is now a miss
    assert runner.infer_step(obs, session_id="sess-A")["prefix_cache_hit"] is True
    assert runner.infer_step(obs, session_id="sess-B")["prefix_cache_hit"] is False

  @pytest.mark.parametrize(
      ("model_id", "step_latency_key"),
      [
          ("gr00t-n1.7-3b", "system1_dit_latency_ms"),
          ("pi05-3b", "flow_expert_latency_ms"),
          ("cosmos3-edge", "dit_rollout_latency_ms"),
          ("qwen3-omni-30b-a3b", "sparse_moe_latency_ms"),
      ],
  )
  def test_denoising_step_reduction_16_vs_4_wall_clock_scaling(
      self, model_id: str, step_latency_key: str
  ) -> None:
    obs = _make_sample_observation(height=32, width=32)
    runner = create_baseline_runner(model_id, seed=42)
    # Warm up session prefix cache so we measure pure denoising/ODE rollout scaling
    runner.infer_step(obs, session_id="bench-sess", denoising_steps=4)

    lat_16 = [
        runner.infer_step(obs, session_id="bench-sess", denoising_steps=16)[
            step_latency_key
        ]
        for _ in range(5)
    ]
    lat_4 = [
        runner.infer_step(obs, session_id="bench-sess", denoising_steps=4)[
            step_latency_key
        ]
        for _ in range(5)
    ]
    med_16 = statistics.median(lat_16)
    med_4 = statistics.median(lat_4)
    assert med_4 < med_16, (
        f"Expected 4-step latency ({med_4:.3f} ms) < 16-step latency ({med_16:.3f} ms) "
        f"for {model_id}."
    )


class TestAdversarialGates1To4:
  """Verify Gate 1 (Crash, Repeat, Concurrent), Gate 2 (State Machine), and Gate 4 (Negative Controls)."""

  def test_gate1_crash_path_orphan_temp_cleanup_and_corrupt_json_recovery(
      self, tmp_path: Path
  ) -> None:
    state_file = tmp_path / "runner_state.json"
    orphan_tmp = tmp_path / f".tmp_embodied_{state_file.name}_crashed123.json"
    orphan_tmp.write_text("{corrupt partial write", encoding="utf-8")
    state_file.write_text("{broken json payload", encoding="utf-8")

    state = record_runner_benchmark_state(
        state_path=state_file,
        run_id="run-crash-01",
        model_id="gr00t-n1.7-3b",
        status=RunStatus.PENDING,
        metrics={"latency_ms": 15.2},
    )
    assert not orphan_tmp.exists()
    assert state.get("recovered_from_corrupt_state") is True
    assert state["runner_runs"]["run-crash-01"]["status"] == "PENDING"

  def test_gate1_repeat_path_no_stale_optional_field_carryover(
      self, tmp_path: Path
  ) -> None:
    state_file = tmp_path / "runner_state.json"
    record_runner_benchmark_state(
        state_path=state_file,
        run_id="run-repeat-01",
        model_id="pi05-3b",
        status=RunStatus.BLOCKED,
        metrics={"attempt": 1},
        backend="jax_tpu",
        precision="fp8",
        denoising_steps=10,
        cache_stats={"cache_hits": 4},
        reason="Waiting on single-chip TPU v6e allocation.",
        notes="Initial blocked note.",
    )
    # Second call omits optional fields: backend, precision, denoising_steps, cache_stats, reason, notes
    state2 = record_runner_benchmark_state(
        state_path=state_file,
        run_id="run-repeat-01",
        model_id="pi05-3b",
        status=RunStatus.RUNNING,
        metrics={"attempt": 2},
    )
    rec = state2["runner_runs"]["run-repeat-01"]
    assert rec["status"] == "RUNNING"
    assert rec["metrics"] == {"attempt": 2}
    assert rec["backend"] is None
    assert rec["precision"] is None
    assert rec["denoising_steps"] is None
    assert rec["cache_stats"] is None
    assert rec["reason"] is None
    assert rec["notes"] is None

  def test_gate1_concurrent_writers_threads_and_subprocesses(
      self, tmp_path: Path
  ) -> None:
    state_file = tmp_path / "concurrent_runner_state.json"

    def _thread_worker(idx: int) -> str:
      rid = f"thread-run-{idx:02d}"
      model = CANONICAL_CHALLENGE_MODELS[idx % len(CANONICAL_CHALLENGE_MODELS)]
      record_runner_benchmark_state(
          state_path=state_file,
          run_id=rid,
          model_id=model,
          status=RunStatus.PENDING,
          metrics={"idx": idx},
      )
      record_runner_benchmark_state(
          state_path=state_file,
          run_id=rid,
          model_id=model,
          status=RunStatus.RUNNING,
          metrics={"idx": idx},
      )
      record_runner_benchmark_state(
          state_path=state_file,
          run_id=rid,
          model_id=model,
          status=RunStatus.CERTIFIED,
          metrics={"idx": idx, "certified": True},
      )
      return rid

    with ThreadPoolExecutor(max_workers=8) as pool:
      thread_ids = list(pool.map(_thread_worker, range(12)))

    proc_args = [
        (
            str(state_file),
            f"proc-run-{i:02d}",
            CANONICAL_CHALLENGE_MODELS[i % len(CANONICAL_CHALLENGE_MODELS)],
        )
        for i in range(6)
    ]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as proc_pool:
      proc_ids = list(proc_pool.map(_concurrent_subprocess_worker, proc_args))

    final_state = record_runner_benchmark_state(
        state_path=state_file,
        run_id="sentinel-run",
        model_id="gr00t-n1.7-3b",
        status=RunStatus.PENDING,
    )
    runs = final_state["runner_runs"]
    for rid in thread_ids + proc_ids:
      assert rid in runs
      assert runs[rid]["status"] == "CERTIFIED"

  def test_gate2_state_machine_completeness_and_mandatory_reasons(
      self, tmp_path: Path
  ) -> None:
    state_file = tmp_path / "sm_state.json"

    # Reach all 8 permitted RunStatus states via valid transitions
    transitions = [
        ("r-pending", [RunStatus.PENDING]),
        ("r-running", [RunStatus.PENDING, RunStatus.RUNNING]),
        ("r-certified", [RunStatus.PENDING, RunStatus.RUNNING, RunStatus.CERTIFIED]),
        ("r-rejected", [RunStatus.PENDING, RunStatus.RUNNING, RunStatus.REJECTED]),
        ("r-blocked", [RunStatus.PENDING, RunStatus.BLOCKED]),
        ("r-failed", [RunStatus.PENDING, RunStatus.RUNNING, RunStatus.FAILED]),
        ("r-skipped", [RunStatus.PENDING, RunStatus.SKIPPED]),
        ("r-invalid", [RunStatus.PENDING, RunStatus.INVALID]),
    ]
    reached = set()
    for run_id, path in transitions:
      for st in path:
        if st in REASON_REQUIRED_STATUSES:
          # Must reject missing or whitespace-only reason
          with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
            record_runner_benchmark_state(
                state_path=state_file,
                run_id=run_id,
                model_id="gr00t-n1.7-3b",
                status=st,
                reason="   ",
            )
          state = record_runner_benchmark_state(
              state_path=state_file,
              run_id=run_id,
              model_id="gr00t-n1.7-3b",
              status=st,
              reason=f"Actionable reason for {st.value}",
          )
        else:
          state = record_runner_benchmark_state(
              state_path=state_file,
              run_id=run_id,
              model_id="gr00t-n1.7-3b",
              status=st,
          )
      reached.add(RunStatus(state["runner_runs"][run_id]["status"]))

    assert reached == set(RunStatus)

    # Illegal transition CERTIFIED -> RUNNING must fail
    with pytest.raises(ValueError, match="Illegal state transition"):
      record_runner_benchmark_state(
          state_path=state_file,
          run_id="r-certified",
          model_id="gr00t-n1.7-3b",
          status=RunStatus.RUNNING,
      )

  def test_gate4_negative_controls_and_quantization_perturbation_bounds(self) -> None:
    # 1. Unknown and deprecated model IDs must be rejected
    with pytest.raises(ValueError, match="Unknown model_id"):
      create_baseline_runner("nonexistent-model-99b")

    with pytest.raises(ValueError, match="Deprecated 2024 baseline"):
      create_baseline_runner("openvla-7b-vanilla")

    # 2. Invalid denoising_steps <= 0 must be rejected at config init and step call
    with pytest.raises(ValueError, match="denoising_steps"):
      create_baseline_runner("gr00t-n1.7-3b", denoising_steps=0)

    with pytest.raises(ValueError, match="denoising_steps"):
      create_baseline_runner("pi05-3b", denoising_steps=-2)

    runner_bf16 = create_baseline_runner(
        "gr00t-n1.7-3b", precision="bf16", seed=42, denoising_steps=6
    )
    obs = _make_sample_observation()
    with pytest.raises(ValueError, match="denoising_steps"):
      runner_bf16.infer_step(obs, denoising_steps=0)

    # 3. Quantized precision perturbation bounds (bf16 vs fp8, w4a8, int4_awq)
    act_bf16 = runner_bf16.infer_step(obs, session_id="q-test")["actions"]

    for q_prec in ("fp8", "w4a8", "int4_awq"):
      runner_q = create_baseline_runner(
          "gr00t-n1.7-3b", precision=q_prec, seed=42, denoising_steps=6
      )
      act_q = runner_q.infer_step(obs, session_id="q-test")["actions"]
      l2_diff = float(np.linalg.norm(act_bf16 - act_q))
      max_abs_diff = float(np.max(np.abs(act_bf16 - act_q)))
      cos_sim = float(
          np.dot(act_bf16.reshape(-1), act_q.reshape(-1))
          / (np.linalg.norm(act_bf16) * np.linalg.norm(act_q) + 1e-8)
      )
      # Must actually quantize (non-zero perturbation vs bf16) while remaining bounded
      assert l2_diff > 1e-5, f"Expected non-zero quantization diff for {q_prec}"
      assert max_abs_diff < 0.35, f"Perturbation exceeded bound for {q_prec}: {max_abs_diff}"
      assert cos_sim > 0.92, f"Cosine similarity too low for {q_prec}: {cos_sim}"
