"""Milestone 3 (M3) test suite for the 4 Canonical Reference Model Runners, CAP_SINGLE_CHIP enforcement, Multi-Vendor Single-Chip Dockerfiles, HTTP server wrapper, and Gate G4 perturbation responsiveness."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import urllib.request
import numpy as np
import pytest

from embodied_bench.quality_gates import (
    audit_hot_loop_device_sync,
    evaluate_gate_g4_cache_memoization,
)
from embodied_bench.schema import ExecutionStrategy
from runners import (
    Cosmos3EdgeRunner,
    EmbodiedRunnerConfig,
    GR00TN17Runner,
    Pi05Runner,
    Qwen3OmniRunner,
    create_model_runner,
    make_runner_http_server,
)
from runners import base_server, cosmos3_edge_server, gr00t_n17_server, pi05_server, qwen3_omni_server


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cap_single_chip_enforcement_and_negative_controls(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify EmbodiedRunnerConfig enforces num_devices == 1 (CAP_SINGLE_CHIP) and rejects invalid configs."""
  # Positive control: single chip (num_devices=1) succeeds across NVIDIA, AMD, and Google TPU SKUs
  for sku in ("RTX_5090_32GB", "JETSON_AGX_THOR_64GB", "AMD_MI210_64GB", "TPU_V6E_1_32GB", "TPU_V5E_1_16GB"):
    cfg = EmbodiedRunnerConfig(model_id="gr00t-n1.7-3b", hardware_sku=sku, num_devices=1)
    assert cfg.num_devices == 1
    assert cfg.canonical_model_id == "gr00t-n1.7-3b"

  # Negative control 1: num_devices=2 is rejected
  with pytest.raises(ValueError, match="CAP_SINGLE_CHIP violation"):
    EmbodiedRunnerConfig(model_id="gr00t-n1.7-3b", hardware_sku="RTX_5090_32GB", num_devices=2)

  # Negative control 2: num_devices=0 is rejected
  with pytest.raises(ValueError, match="CAP_SINGLE_CHIP violation"):
    EmbodiedRunnerConfig(model_id="pi05-3b", hardware_sku="TPU_V6E_1_32GB", num_devices=0)

  # Negative control 3: EEB_MAX_DEVICES=2 in environment is rejected
  monkeypatch.setenv("EEB_MAX_DEVICES", "2")
  with pytest.raises(ValueError, match="EEB_MAX_DEVICES"):
    EmbodiedRunnerConfig(model_id="cosmos3-edge", hardware_sku="RTX_5090_32GB", num_devices=1)
  monkeypatch.setenv("EEB_MAX_DEVICES", "1")
  cfg_ok = EmbodiedRunnerConfig(model_id="cosmos3-edge", hardware_sku="RTX_5090_32GB", num_devices=1)
  assert cfg_ok.num_devices == 1

  # Negative control 4: unknown SKU, weights_mode, precision, model_id
  with pytest.raises(ValueError, match="Unknown hardware_sku"):
    EmbodiedRunnerConfig(model_id="gr00t-n1.7-3b", hardware_sku="NONEXISTENT_GPU_999GB")
  with pytest.raises(ValueError, match="Invalid weights_mode"):
    EmbodiedRunnerConfig(model_id="gr00t-n1.7-3b", weights_mode="invalid-mode")
  with pytest.raises(ValueError, match="Unsupported model_id"):
    EmbodiedRunnerConfig(model_id="unknown-model-xyz")


@pytest.mark.parametrize(
    "model_id,expected_cls,expected_canonical",
    [
        ("nvidia/GR00T-N1.7-3B", GR00TN17Runner, "gr00t-n1.7-3b"),
        ("lerobot/pi05_libero_finetuned_v044", Pi05Runner, "pi05-3b"),
        ("nvidia/Cosmos3-Edge", Cosmos3EdgeRunner, "cosmos3-edge"),
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", Qwen3OmniRunner, "qwen3-omni-30b-a3b"),
    ],
)
def test_factory_and_stateful_kv_prefix_cache_transitions(
    model_id: str,
    expected_cls: type,
    expected_canonical: str,
) -> None:
  """Verify create_model_runner routing and kv_prefix_cache_hit False (step 0) -> True (steps 1, 2)."""
  cfg = EmbodiedRunnerConfig(
      model_id=model_id,
      hardware_sku="TPU_V6E_1_32GB",
      precision="BF16",
      num_devices=1,
      enable_prefix_cache=True,
  )
  runner = create_model_runner(cfg)
  assert isinstance(runner, expected_cls)
  health = runner.health()
  assert health["status"] == "healthy"
  assert health["canonical_model_id"] == expected_canonical
  assert health["num_devices"] == 1

  rng = np.random.default_rng(42)
  rgb = rng.uniform(0.0, 1.0, size=(2, 2, 3, 32, 32)).astype(np.float32)
  proprio = rng.normal(0.0, 0.2, size=(2, 8)).astype(np.float32)
  inst = np.arange(24, dtype=np.int32).reshape(1, 24)
  cond = rng.uniform(-0.5, 0.5, size=(2, 16, 7)).astype(np.float32)

  # Step 0 -> kv_prefix_cache_hit is False; Steps 1 and 2 -> True
  r0 = runner.step_act("sess_alpha", 0, rgb, proprio, inst, chunk_size=16)
  r1 = runner.step_act("sess_alpha", 1, rgb, proprio, inst, chunk_size=16)
  r2 = runner.step_act("sess_alpha", 2, rgb, proprio, inst, chunk_size=16)
  assert r0["kv_prefix_cache_hit"] is False
  assert r1["kv_prefix_cache_hit"] is True
  assert r2["kv_prefix_cache_hit"] is True
  assert r0["actions"].shape == (2, 16, 7)

  # World step stateful session check
  w0 = runner.step_world("sess_wm", 0, rgb, cond, horizon_frames=16)
  w1 = runner.step_world("sess_wm", 1, rgb, cond, horizon_frames=16)
  w2 = runner.step_world("sess_wm", 2, rgb, cond, horizon_frames=16)
  assert w0["kv_prefix_cache_hit"] is False
  assert w1["kv_prefix_cache_hit"] is True
  assert w2["kv_prefix_cache_hit"] is True
  assert w1["video_latents"].shape == (2, 16, 16, 16)
  assert w1["latent_actions"].shape == (2, 16, 7)

  # Repeat-call / reset_session path: resetting clears cached prefix so next step is False
  reset_info = runner.reset_session("sess_alpha")
  assert reset_info["reset"] is True
  assert reset_info["kv_prefix_cached"] is False
  r_after_reset = runner.step_act("sess_alpha", 0, rgb, proprio, inst, chunk_size=16)
  assert r_after_reset["kv_prefix_cache_hit"] is False


def test_gr00t_n17_dual_system_and_step_reduction_latency() -> None:
  """Verify GR00TN17Runner 24-token prefix cache, hf-checkpoint mode, and 16 -> 4 step reduction."""
  rng = np.random.default_rng(107)
  rgb = rng.uniform(0.0, 1.0, size=(2, 2, 3, 32, 32)).astype(np.float32)
  proprio = rng.normal(0.0, 0.2, size=(2, 8)).astype(np.float32)
  inst = np.arange(24, dtype=np.int32).reshape(1, 24)

  runner_16 = GR00TN17Runner(
      EmbodiedRunnerConfig(
          model_id="nvidia/GR00T-N1.7-3B",
          hardware_sku="RTX_5090_32GB",
          num_inference_steps=16,
      )
  )
  runner_4 = GR00TN17Runner(
      EmbodiedRunnerConfig(
          model_id="nvidia/GR00T-N1.7-3B",
          hardware_sku="RTX_5090_32GB",
          num_inference_steps=4,
          weights_mode="hf-checkpoint",
      )
  )

  # Warmup and average over 5 calls to verify 4-step DiT is measurably faster than 16-step DiT
  runner_16.step_act("bench16", 0, rgb, proprio, inst, chunk_size=16)
  runner_4.step_act("bench4", 0, rgb, proprio, inst, chunk_size=16)

  dit_16_times = [
      runner_16.step_act("bench16", idx, rgb, proprio, inst, chunk_size=16)["dit_system1_ms"]
      for idx in range(1, 7)
  ]
  dit_4_times = [
      runner_4.step_act("bench4", idx, rgb, proprio, inst, chunk_size=16)["dit_system1_ms"]
      for idx in range(1, 7)
  ]
  assert float(np.median(dit_4_times)) < float(np.median(dit_16_times))

  res = runner_16.step_act("bench16", 7, rgb, proprio, inst, chunk_size=16)
  assert res["cached_prefix_tokens"] == 24
  assert res["kv_prefix_cache_hit"] is True
  assert res["active_params_b"] == 3.0
  assert res["total_params_b"] == 3.0


def test_pi05_rtc_inpainting_boundary_pinning_and_smooth_blending() -> None:
  """Verify Pi05Runner RTC_INPAINTING pins first d actions and reduces boundary jump vs unconditioned."""
  rng = np.random.default_rng(205)
  rgb = rng.uniform(0.0, 1.0, size=(2, 2, 3, 32, 32)).astype(np.float32)
  proprio = rng.normal(0.0, 0.2, size=(2, 8)).astype(np.float32)
  inst = np.arange(24, dtype=np.int32).reshape(1, 24)

  # Create a distinct prev_chunk_tail of length d = 4 steps
  d = 4
  chunk_size = 16
  prev_chunk_tail = np.full((2, d, 7), 0.65, dtype=np.float32)

  rtc_runner = Pi05Runner(
      EmbodiedRunnerConfig(
          model_id="lerobot/pi05_libero_finetuned_v044",
          hardware_sku="TPU_V6E_1_32GB",
          num_inference_steps=4,
          execution_strategy=ExecutionStrategy.RTC_INPAINTING.value,
      )
  )
  async_runner = Pi05Runner(
      EmbodiedRunnerConfig(
          model_id="lerobot/pi05_libero_finetuned_v044",
          hardware_sku="TPU_V6E_1_32GB",
          num_inference_steps=4,
          execution_strategy=ExecutionStrategy.ASYNC_CHUNK.value,
      )
  )

  res_rtc = rtc_runner.step_act(
      session_id="pi05_rtc",
      step_idx=0,
      rgb_frames=rgb,
      proprio_state=proprio,
      instruction_tokens=inst,
      chunk_size=chunk_size,
      prev_chunk_tail=prev_chunk_tail,
  )
  res_async = async_runner.step_act(
      session_id="pi05_async",
      step_idx=0,
      rgb_frames=rgb,
      proprio_state=proprio,
      instruction_tokens=inst,
      chunk_size=chunk_size,
      prev_chunk_tail=prev_chunk_tail,
  )

  assert res_rtc["rtc_inpainting_applied"] is True
  assert res_rtc["rtc_pinned_steps"] == d
  # Exact pinning on steps 0..d-1
  np.testing.assert_allclose(res_rtc["actions"][:, :d, :], prev_chunk_tail, atol=1e-6)
  # Smooth boundary jump at step d is strictly smaller than unconditioned jump
  assert res_rtc["boundary_jump_l2"] < res_rtc["unconditioned_boundary_jump_l2"]
  assert res_rtc["boundary_jump_l2"] < res_async["boundary_jump_l2"]


def test_cosmos3_edge_dit_cache_and_gate_g4_perturbation_sensitivity() -> None:
  """Verify Cosmos3EdgeRunner DiT-Cache hits on smooth frames, misses on jumps, and passes Gate G4."""
  rng = np.random.default_rng(309)
  obs0 = rng.uniform(0.1, 0.9, size=(4, 2, 3, 32, 32)).astype(np.float32)
  # Smooth consecutive frame (delta << threshold 0.18)
  obs1_smooth = np.clip(obs0 + rng.normal(0.0, 0.005, size=obs0.shape).astype(np.float32), 0.0, 1.0)
  # Large scene change (delta > threshold 0.18)
  obs2_jump = (1.0 - obs0).astype(np.float32)
  cond = rng.uniform(-0.4, 0.4, size=(4, 16, 7)).astype(np.float32)

  runner = Cosmos3EdgeRunner(
      EmbodiedRunnerConfig(
          model_id="nvidia/Cosmos3-Edge",
          hardware_sku="JETSON_AGX_THOR_64GB",
          num_inference_steps=16,
          enable_dit_cache=True,
          dit_cache_threshold=0.18,
      )
  )

  # Step 0: initial frame -> dit_cache_hit is False
  s0 = runner.step_world("cosmos_sess", 0, obs0, cond, horizon_frames=16)
  assert s0["dit_cache_hit"] is False
  assert s0["kv_prefix_cache_hit"] is False
  assert s0["effective_dit_steps_executed"] == 16

  # Step 1: smooth consecutive frame -> dit_cache_hit is True
  s1 = runner.step_world("cosmos_sess", 1, obs1_smooth, cond, horizon_frames=16)
  assert s1["dit_cache_hit"] is True
  assert s1["kv_prefix_cache_hit"] is True
  assert s1["effective_dit_steps_executed"] == 1

  # Step 2: large scene jump -> dit_cache_hit is False (recomputes all 16 steps)
  s2 = runner.step_world("cosmos_sess", 2, obs2_jump, cond, horizon_frames=16)
  assert s2["dit_cache_hit"] is False
  assert s2["effective_dit_steps_executed"] == 16

  # Gate G4 (l_inf = 1e-3 perturbation) verification while DiT-Cache is active!
  delta_pert = rng.choice([-1e-3, 1e-3], size=obs1_smooth.shape).astype(np.float32)
  obs1_perturbed = np.clip(obs1_smooth + delta_pert, 0.0, 1.0)

  # Prime session on obs0 then evaluate base vs perturbed at step 1 (both hit DiT-Cache)
  runner.step_world("g4_base_sess", 0, obs0, cond, horizon_frames=16)
  base_out = runner.step_world("g4_base_sess", 1, obs1_smooth, cond, horizon_frames=16)
  runner.step_world("g4_pert_sess", 0, obs0, cond, horizon_frames=16)
  pert_out = runner.step_world("g4_pert_sess", 1, obs1_perturbed, cond, horizon_frames=16)

  assert base_out["dit_cache_hit"] is True
  assert pert_out["dit_cache_hit"] is True

  g4_video = evaluate_gate_g4_cache_memoization(
      base_outputs=base_out["video_latents"],
      perturbed_outputs=pert_out["video_latents"],
      perturbation_linf=1e-3,
  )
  g4_actions = evaluate_gate_g4_cache_memoization(
      base_outputs=base_out["latent_actions"],
      perturbed_outputs=pert_out["latent_actions"],
      perturbation_linf=1e-3,
  )
  assert g4_video.passed is True, g4_video.reason
  assert g4_video.cache_hit_suspected is False
  assert g4_video.responsive_sample_fraction == 1.0
  assert g4_actions.passed is True, g4_actions.reason


def test_qwen3_omni_moe_routing_and_vllm_endpoint() -> None:
  """Verify Qwen3OmniRunner routes top_k=8 of 128 experts with 3.0B active / 30.0B total params."""
  cfg = EmbodiedRunnerConfig(
      model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
      hardware_sku="AMD_MI210_64GB",
      precision="FP8",
      num_inference_steps=4,
      vllm_endpoint="http://127.0.0.1:8000/v1",
  )
  runner = Qwen3OmniRunner(cfg)
  h = runner.health()
  assert h["total_params_b"] == 30.0
  assert h["active_params_b"] == 3.0
  assert h["active_param_ratio"] == pytest.approx(0.10)
  assert h["num_experts"] == 128
  assert h["top_k_experts"] == 8
  assert h["vllm_endpoint"] == "http://127.0.0.1:8000/v1"
  assert h["backend_engine"] == "vllm-sglang-remote"

  rng = np.random.default_rng(999)
  obs = rng.uniform(0.0, 1.0, size=(2, 2, 3, 32, 32)).astype(np.float32)
  cond = rng.uniform(-0.5, 0.5, size=(2, 16, 7)).astype(np.float32)
  proprio = rng.normal(0.0, 0.1, size=(2, 8)).astype(np.float32)
  inst = np.arange(24, dtype=np.int32).reshape(1, 24)

  w0 = runner.step_world("qwen_sess", 0, obs, cond, horizon_frames=16)
  w1 = runner.step_world("qwen_sess", 1, obs, cond, horizon_frames=16)
  assert w0["kv_prefix_cache_hit"] is False
  assert w1["kv_prefix_cache_hit"] is True
  assert w1["num_experts"] == 128
  assert w1["top_k_experts"] == 8
  assert len(w1["active_expert_ids"]) == 8
  assert len(set(w1["active_expert_ids"])) == 8
  assert all(0 <= eid < 128 for eid in w1["active_expert_ids"])
  assert sum(w1["expert_routing_weights"]) == pytest.approx(1.0, abs=1e-4)
  assert w1["active_params_b"] == 3.0
  assert w1["total_params_b"] == 30.0

  a0 = runner.step_act("qwen_act_sess", 0, obs, proprio, inst, chunk_size=8)
  a1 = runner.step_act("qwen_act_sess", 1, obs, proprio, inst, chunk_size=8)
  assert a0["kv_prefix_cache_hit"] is False
  assert a1["kv_prefix_cache_hit"] is True
  assert len(a1["active_expert_ids"]) == 8


def test_concurrent_multi_session_thread_safety_and_http_loopback() -> None:
  """Verify thread-safe concurrent session stepping and HTTP loopback server endpoints."""
  runner = create_model_runner(
      EmbodiedRunnerConfig(model_id="gr00t-n1.7-3b", hardware_sku="RTX_5090_32GB", num_inference_steps=2)
  )
  rng = np.random.default_rng(77)
  rgb = rng.uniform(0.0, 1.0, size=(1, 2, 3, 16, 16)).astype(np.float32)
  proprio = np.zeros((1, 8), dtype=np.float32)
  inst = np.arange(24, dtype=np.int32).reshape(1, 24)

  def _worker(stream_idx: int) -> tuple[bool, bool]:
    sid = f"concurrent_stream_{stream_idx}"
    s0 = runner.step_act(sid, 0, rgb, proprio, inst, chunk_size=8)
    s1 = runner.step_act(sid, 1, rgb, proprio, inst, chunk_size=8)
    return bool(s0["kv_prefix_cache_hit"]), bool(s1["kv_prefix_cache_hit"])

  with ThreadPoolExecutor(max_workers=8) as pool:
    outcomes = list(pool.map(_worker, range(16)))
  assert all(o == (False, True) for o in outcomes)

  # Verify HTTP server wrapper (/v1/health, /v1/reset_session, /v1/act)
  srv = make_runner_http_server(runner, host="127.0.0.1", port=0)
  port = srv.server_address[1]
  t = threading.Thread(target=srv.serve_forever, daemon=True)
  t.start()
  try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=5.0) as resp:
      health_data = json.loads(resp.read().decode("utf-8"))
    assert health_data["status"] == "healthy"
    assert health_data["num_devices"] == 1

    req_body = json.dumps({
        "session_id": "http_sess",
        "step_idx": 0,
        "chunk_size": 8,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/act",
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
      act_data = json.loads(resp.read().decode("utf-8"))
    assert act_data["actions_shape"] == [1, 8, 7]
    assert act_data["kv_prefix_cache_hit"] is False
  finally:
    srv.shutdown()
    srv.server_close()


def test_three_vendor_single_chip_dockerfiles_and_cli_entrypoints(
    capsys: pytest.CaptureFixture[str],
) -> None:
  """Verify all 3 vendor Dockerfiles enforce single-device visibility env vars and CLI flag orders work."""
  docker_dir = REPO_ROOT / "docker"
  cuda_df = (docker_dir / "Dockerfile.nvidia-cuda").read_text(encoding="utf-8")
  rocm_df = (docker_dir / "Dockerfile.amd-rocm").read_text(encoding="utf-8")
  tpu_df = (docker_dir / "Dockerfile.google-tpu-v6e").read_text(encoding="utf-8")

  # 1. NVIDIA CUDA Dockerfile checks
  for token in (
      "CUDA_VISIBLE_DEVICES=0",
      "EEB_MAX_DEVICES=1",
      "JETSON_AGX_THOR_64GB",
      "RTX_5090_32GB",
      "L40S_48GB",
      "RTX_4090_24GB",
  ):
    assert token in cuda_df, f"Missing {token!r} in Dockerfile.nvidia-cuda"

  # 2. AMD ROCm Dockerfile checks
  for token in (
      "ROCR_VISIBLE_DEVICES=0",
      "HIP_VISIBLE_DEVICES=0",
      "EEB_MAX_DEVICES=1",
      "AMD_MI210_64GB",
      "AMD_W7900_48GB",
      "AMD_RX_7900_XTX_24GB",
  ):
    assert token in rocm_df, f"Missing {token!r} in Dockerfile.amd-rocm"

  # 3. Google Cloud TPU v6e Dockerfile checks
  for token in (
      "TPU_VISIBLE_DEVICES=0",
      "PJRT_DEVICE=TPU",
      "EEB_MAX_DEVICES=1",
      "TPU_V6E_1_32GB",
      "TPU_V5E_1_16GB",
  ):
    assert token in tpu_df, f"Missing {token!r} in Dockerfile.google-tpu-v6e"

  # Gate 3: Execute CLI main(argv) in both flag orders
  rc1 = base_server.main([
      "--model-id", "nvidia/GR00T-N1.7-3B",
      "--hardware-sku", "RTX_5090_32GB",
      "--num-inference-steps", "4",
      "--self-test",
  ])
  out1 = json.loads(capsys.readouterr().out)
  assert rc1 == 0
  assert out1["self_test"]["step0_kv_hit"] is False
  assert out1["self_test"]["step1_kv_hit"] is True

  rc2 = base_server.main([
      "--self-test",
      "--num-inference-steps", "2",
      "--hardware-sku", "TPU_V6E_1_32GB",
      "--model-id", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  ])
  out2 = json.loads(capsys.readouterr().out)
  assert rc2 == 0
  assert out2["canonical_model_id"] == "qwen3-omni-30b-a3b"

  # Individual server CLI entrypoints
  assert gr00t_n17_server.main(["--hardware-sku", "RTX_5090_32GB"]) == 0
  assert pi05_server.main(["--hardware-sku", "TPU_V6E_1_32GB"]) == 0
  assert cosmos3_edge_server.main(["--hardware-sku", "JETSON_AGX_THOR_64GB"]) == 0
  assert qwen3_omni_server.main(["--hardware-sku", "AMD_MI210_64GB"]) == 0
  _ = capsys.readouterr()

  # AST Hot-Loop Device Sync Audit across all M3 runner files
  for fname in (
      "base_server.py",
      "gr00t_n17_server.py",
      "pi05_server.py",
      "cosmos3_edge_server.py",
      "qwen3_omni_server.py",
  ):
    src = (REPO_ROOT / "runners" / fname).read_text(encoding="utf-8")
    assert audit_hot_loop_device_sync(src) == []
