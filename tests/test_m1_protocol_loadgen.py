"""Milestone 1 Tests: Client-Server HTTP Keep-Alive Harness, Zero-Copy POSIX Shared-Memory IPC (`SharedMemoryTensorRing`), and Async LoadGen (`loadgen.py`)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from embodied_bench.loadgen import (
    LiveLoadGenConfig,
    run_live_server_benchmark,
)
from embodied_bench.protocol import (
    EmbodiedHTTPClient,
    LoopbackEmbodiedServerHandler,
    SharedMemoryTensorRing,
    ShmTensorDescriptor,
    start_background_embodied_server,
)
from embodied_bench.quality_gates import audit_hot_loop_device_sync
from embodied_bench.schema import (
    DivisionType,
    PerformanceCapId,
    RecipeConfig,
    RunStatus,
    TrackType,
)


def _make_vla_recipe() -> RecipeConfig:
  return RecipeConfig(
      recipe_id="m1_test_vla_pi05_tpu_v6e",
      track=TrackType.VLA_TRACK.value,
      model_family="pi05",
      model_name="pi05-3b",
      baseline_model_id="pi05-3b",
      precision="FP8",
      hardware_sku="TPU_V6E_1_32GB",
      division=DivisionType.OPEN.value,
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP.value,
      num_devices=1,
      action_chunk_sizes=[8],
      concurrency_levels=[1, 2, 4],
      action_dim=7,
      action_head_type="continuous_flow",
      optimizations=["vlm_prefix_kv_cache_reuse", "w4_fp8_quantization"],
  )


def _make_wm_recipe() -> RecipeConfig:
  return RecipeConfig(
      recipe_id="m1_test_wm_cosmos3_edge",
      track=TrackType.WM_TRACK.value,
      model_family="cosmos3",
      model_name="cosmos3-edge",
      baseline_model_id="cosmos3-edge",
      precision="FP8",
      hardware_sku="JETSON_AGX_THOR_64GB",
      division=DivisionType.OPEN.value,
      perf_cap=PerformanceCapId.CAP_SINGLE_CHIP.value,
      num_devices=1,
      horizon_frames=16,
      denoising_steps=4,
      batch_sizes=[1, 2, 4],
      optimizations=["causal_spatio_temporal_kv_caching", "fp8_nvfp4_quantization"],
  )


def test_m1_shared_memory_tensor_ring_zero_copy_cleanup_and_corruption() -> None:
  """Test 1: `SharedMemoryTensorRing` zero-copy write/read, cleanup on close/unlink, mmap fallback, and corruption detection."""
  rng = np.random.default_rng(42)
  source_tensor = rng.uniform(0.0, 1.0, size=(32, 32, 3)).astype(np.float32)

  with SharedMemoryTensorRing(capacity_bytes=2 * 1024 * 1024, create=True) as writer_ring:
    desc = writer_ring.write_tensor(source_tensor, compute_checksum=True)
    assert isinstance(desc, ShmTensorDescriptor)
    assert desc.shape == (32, 32, 3)
    assert desc.dtype == "float32"
    assert desc.nbytes == source_tensor.nbytes
    assert len(desc.sha256_prefix) == 16

    # Attach a reader ring to the same shared memory segment without copying
    reader_ring = SharedMemoryTensorRing(name=writer_ring.name, create=False)
    try:
      zero_copy_view = reader_ring.read_tensor(desc, verify_checksum=True)
      assert np.allclose(zero_copy_view, source_tensor, atol=0.0)
      assert zero_copy_view.base is not None

      # Verify zero-copy memory sharing: mutating writer buffer mutates reader view in-place
      writer_view = writer_ring.read_tensor(desc, verify_checksum=False)
      original_val = float(writer_view[0, 0, 0])
      writer_view[0, 0, 0] = original_val + 1.25
      assert np.isclose(float(zero_copy_view[0, 0, 0]), original_val + 1.25)

      # Corruption detection: checksum validation must fail when underlying bytes are modified
      with pytest.raises(ValueError, match="checksum mismatch"):
        reader_ring.read_tensor(desc, verify_checksum=True)

      # Restore original byte value and verify checksum passes again
      writer_view[0, 0, 0] = original_val
      restored_view = reader_ring.read_tensor(desc, verify_checksum=True)
      assert np.allclose(restored_view, source_tensor, atol=0.0)
    finally:
      # Idempotent close on reader
      reader_ring.close()
      reader_ring.close()

  # Idempotent close and unlink after context manager exit
  writer_ring.close()
  writer_ring.unlink()
  writer_ring.unlink()

  # Also verify automatic /tmp mmap fallback works identically
  with SharedMemoryTensorRing(
      capacity_bytes=512 * 1024,
      create=True,
      force_mmap_fallback=True,
  ) as fallback_ring:
    assert fallback_ring.backend == "tmp_mmap"
    desc_fb = fallback_ring.write_tensor(source_tensor)
    fb_reader = SharedMemoryTensorRing(
        name=fallback_ring.name,
        create=False,
        force_mmap_fallback=True,
    )
    try:
      arr_fb = fb_reader.read_tensor(desc_fb, verify_checksum=True)
      assert np.allclose(arr_fb, source_tensor)
    finally:
      fb_reader.close()


def test_m1_loopback_http_shm_concurrency_sweep_vla_and_wm() -> None:
  """Test 2: Full loopback live HTTP + SHM server concurrency sweep (C in {1, 2, 4}) for Track 1 (VLA_TRACK) and Track 2 (WM_TRACK)."""
  # Track 1: VLA_TRACK concurrency sweep
  vla_recipe = _make_vla_recipe()
  vla_handler = LoopbackEmbodiedServerHandler(
      model_id=vla_recipe.model_name,
      track=vla_recipe.track,
      hardware_sku=vla_recipe.hardware_sku,
      num_devices=1,
      precision=vla_recipe.precision,
  )
  with start_background_embodied_server(vla_handler) as vla_server:
    with EmbodiedHTTPClient(base_url=vla_server.base_url) as client:
      vla_report = run_live_server_benchmark(
          client=client,
          recipe=vla_recipe,
          concurrencies=(1, 2, 4),
          warmup_steps=4,
          timed_steps=12,
          use_shm=True,
      )

  assert vla_report["passed"] is True
  assert vla_report["status"] == RunStatus.CERTIFIED.value
  assert vla_report["single_device_compliant"] is True
  assert vla_report["gate_g1_passed"] is True
  assert vla_report["gate_g3_passed"] is True
  assert vla_report["gate_g4_passed"] is True
  assert len(vla_report["operating_points"]) == 3

  for expected_c, op in zip((1, 2, 4), vla_report["operating_points"]):
    assert op["concurrency"] == expected_c
    assert op["kv_cache_hit_rate"] > 0.80
    assert 0.0 < op["latency_p50_ms"] <= op["latency_p95_ms"] <= op["latency_p99_ms"]
    assert op["latency_mean_ms"] > 0.0
    assert op["hz_react"] > 0.0
    # Verify all 4 SemiAnalysis normalizations exist and are positive
    sa = op["semianalysis_normalizations"]
    assert sa["raw_per_chip_throughput"] > 0.0
    assert sa["power_efficiency_per_watt"] > 0.0
    assert sa["tco_efficiency_per_dollar"] > 0.0
    iso = sa["iso_compute_and_bw_efficiency"]
    assert iso["throughput_per_100_tops"] > 0.0
    assert iso["throughput_per_tbps_bw"] > 0.0
    assert iso["same_chip_speedup_vs_bf16_ref"] > 0.0
    assert op["gate_g4"]["min_l2_output_shift"] > 1e-4

  # Track 2: WM_TRACK concurrency sweep
  wm_recipe = _make_wm_recipe()
  wm_handler = LoopbackEmbodiedServerHandler(
      model_id=wm_recipe.model_name,
      track=wm_recipe.track,
      hardware_sku=wm_recipe.hardware_sku,
      num_devices=1,
      precision=wm_recipe.precision,
  )
  with start_background_embodied_server(wm_handler) as wm_server:
    with EmbodiedHTTPClient(base_url=wm_server.base_url) as client:
      wm_report = run_live_server_benchmark(
          client=client,
          recipe=wm_recipe,
          concurrencies=(1, 2, 4),
          warmup_steps=4,
          timed_steps=12,
          use_shm=True,
      )

  assert wm_report["passed"] is True
  assert wm_report["status"] == RunStatus.CERTIFIED.value
  assert wm_report["gate_g1_passed"] is True
  assert wm_report["gate_g3_passed"] is True
  assert wm_report["gate_g4_passed"] is True
  assert len(wm_report["operating_points"]) == 3

  for expected_c, op in zip((1, 2, 4), wm_report["operating_points"]):
    assert op["concurrency"] == expected_c
    assert op["kv_cache_hit_rate"] > 0.80
    assert op["dit_cache_hit_rate"] > 0.80
    assert 0.0 < op["latency_p50_ms"] <= op["latency_p95_ms"] <= op["latency_p99_ms"]
    assert op["hz_plan"] > 0.0
    sa = op["semianalysis_normalizations"]
    assert sa["raw_per_chip_throughput"] > 0.0
    assert sa["power_efficiency_per_watt"] > 0.0
    assert sa["tco_efficiency_per_dollar"] > 0.0
    assert sa["iso_compute_and_bw_efficiency"]["throughput_per_100_tops"] > 0.0


def test_m1_negative_controls_gate_g4_cheating_memoizer_and_multi_device_rejection() -> None:
  """Test 3 (Gate 4 Negative Control): Cheating static memoizer is rejected by Gate G4, and num_devices=2 is rejected by CAP_SINGLE_CHIP."""
  vla_recipe = _make_vla_recipe()

  # Negative Control A: Server with cheating_static_memoizer=True ignores pixel perturbations
  cheating_handler = LoopbackEmbodiedServerHandler(
      model_id=vla_recipe.model_name,
      track=vla_recipe.track,
      hardware_sku=vla_recipe.hardware_sku,
      num_devices=1,
      precision=vla_recipe.precision,
      cheating_static_memoizer=True,
  )
  with start_background_embodied_server(cheating_handler) as cheat_server:
    with EmbodiedHTTPClient(base_url=cheat_server.base_url) as client:
      cheat_report = run_live_server_benchmark(
          client=client,
          recipe=vla_recipe,
          concurrencies=(1, 2),
          warmup_steps=2,
          timed_steps=10,
          use_shm=True,
      )

  assert cheat_report["passed"] is False
  assert cheat_report["status"] == RunStatus.REJECTED.value
  assert cheat_report["gate_g4_passed"] is False
  assert any("G4" in v for v in cheat_report["violations"])
  for op in cheat_report["operating_points"]:
    assert op["gate_g4"]["passed"] is False
    assert op["gate_g4"]["cache_hit_suspected"] is True
    assert op["gate_g4"]["min_l2_output_shift"] == 0.0

  # Negative Control B: Server reporting num_devices=2 is rejected by CAP_SINGLE_CHIP
  multi_chip_handler = LoopbackEmbodiedServerHandler(
      model_id=vla_recipe.model_name,
      track=vla_recipe.track,
      hardware_sku=vla_recipe.hardware_sku,
      num_devices=2,
      precision=vla_recipe.precision,
  )
  with start_background_embodied_server(multi_chip_handler) as multi_server:
    with EmbodiedHTTPClient(base_url=multi_server.base_url) as client:
      multi_report = run_live_server_benchmark(
          client=client,
          recipe=vla_recipe,
          concurrencies=(1, 2),
          warmup_steps=2,
          timed_steps=8,
          use_shm=True,
      )
      assert multi_report["passed"] is False
      assert multi_report["status"] == RunStatus.REJECTED.value
      assert multi_report["single_device_compliant"] is False
      assert "CAP_SINGLE_CHIP" in multi_report["reason"]
      assert "num_devices=2" in multi_report["reason"]

      # Also verify raise_on_violation=True raises ValueError mentioning CAP_SINGLE_CHIP
      strict_cfg = LiveLoadGenConfig(
          concurrencies=(1,),
          warmup_steps=1,
          timed_steps=4,
          raise_on_violation=True,
      )
      with pytest.raises(ValueError, match="CAP_SINGLE_CHIP"):
        run_live_server_benchmark(
            client=client,
            recipe=vla_recipe,
            config=strict_cfg,
        )


def test_m1_hot_loop_ast_zero_device_sync_violations() -> None:
  """Verify that neither protocol.py nor loadgen.py has blocking .item()/.cpu()/.numpy() inside loops."""
  repo_root = Path(__file__).resolve().parents[1]
  for rel_path in ("embodied_bench/protocol.py", "embodied_bench/loadgen.py"):
    src = (repo_root / rel_path).read_text(encoding="utf-8")
    findings = audit_hot_loop_device_sync(src)
    assert findings == [], f"Unexpected hot-loop sync violation in {rel_path}: {findings}"
