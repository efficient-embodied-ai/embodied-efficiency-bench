"""Milestone 2 (`EmbodiedX-Traces-v1` Open-Source Data Loader & Bundled `mini_v1` Dataset) test suite."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from embodied_bench.adapters import build_adapter
from embodied_bench.dataset_loader import (
    DEFAULT_MINI_V1_DIR,
    EmbodiedXDatasetLoader,
    build_and_write_manifest,
    download_from_hf_hub,
    load_episode_shard_npz,
    save_episode_shard_npz,
)
from embodied_bench.golden_dataset import (
    EMBODIEDX_TRACES_V1_SOURCES,
    Track1GoldenBatch,
    Track2GoldenBatch,
)
from embodied_bench.quality_gates import (
    evaluate_track1_gate_a,
    evaluate_track2_gate_a,
)
from embodied_bench.schema import (
    CANONICAL_V1_MODELS,
    RecipeConfig,
    cleanup_orphan_temp_files,
)
from scripts.build_embodiedx_traces import main as build_traces_main


def test_m2_mini_v1_manifest_and_8_shards_integrity() -> None:
  """Test 1: Verify manifest.json and all 8 episode shards pass SHA-256, cover all 4 sources, coherence >= 0.92, constant 24-token prefix, and exact L_inf = 1e-3 Gate G4 perturbations."""
  loader = EmbodiedXDatasetLoader(dataset_dir=DEFAULT_MINI_V1_DIR, split="mini_v1")
  verified, errors = loader.verify_manifest()
  assert verified is True, f"Expected mini_v1 manifest verification to pass, got errors: {errors}"
  assert errors == []

  manifest = json.loads(loader.manifest_path.read_text(encoding="utf-8"))
  assert manifest["dataset_version"] == "EmbodiedX-Traces-v1"
  assert manifest["split"] == "mini_v1"
  assert manifest["total_episodes"] == 8
  assert manifest["kind"] == "REAL_ROBOT_TRACES"
  assert manifest["pixel_provenance"] == "REAL_HF_LEROBOT_CAMERA_PIXELS"
  assert manifest["is_synthetic_smoke"] is False
  assert manifest["certifiable"] is True
  assert set(manifest["sources"]) == {"LIBERO", "DROID"}
  assert set(manifest["sources"]).issubset(set(EMBODIEDX_TRACES_V1_SOURCES.keys()))
  assert set(manifest["canonical_models"]) == set(CANONICAL_V1_MODELS)
  assert float(manifest["min_temporal_frame_coherence"]) >= 0.94

  shards = loader.preload_into_pinned_ram(verify=True)
  assert len(shards) == 8

  source_counts = {"LIBERO": 0, "DROID": 0}
  for shard in shards:
    source_counts[shard.source_dataset] += 1
    assert shard.num_steps == 16
    assert shard.instruction_tokens.shape == (16, 32)
    assert shard.instruction_tokens.dtype == np.int32
    # Constant 24-token instruction prefix across all T steps
    assert np.all(shard.instruction_tokens[:, :24] == shard.instruction_tokens[0:1, :24])
    assert shard.instruction_prefix_reuse_ratio() >= 0.75

    # Multi-view RGB frames and temporal coherence >= 0.92 (and >= 0.94)
    assert shard.rgb_frames.shape == (16, 2, 3, 64, 64)
    assert shard.rgb_frames.dtype == np.uint8
    assert shard.temporal_frame_coherence() >= 0.94

    # Pinned RAM buffers are C-contiguous and read-only (zero-I/O & mutation-safe)
    assert shard.rgb_frames.flags.c_contiguous is True
    assert shard.rgb_frames.flags.writeable is False
    with pytest.raises(ValueError):
      shard.rgb_frames[0, 0, 0, 0, 0] = 42

    # Proprioception, expert actions, and world-model conditioning actions
    assert shard.proprio_state.shape == (16, 8)
    assert shard.proprio_state.dtype == np.float32
    assert shard.gt_actions.shape == (16, 16, 7)
    assert shard.gt_actions.dtype == np.float32
    assert float(np.min(shard.gt_actions)) >= -1.0
    assert float(np.max(shard.gt_actions)) <= 1.0
    assert shard.cond_actions.shape == (16, 16, 7)
    assert shard.cond_actions.dtype == np.float32

    # Exact L_inf = 1e-3 Gate G4 micro-perturbations
    clean_norm = shard.rgb_frames[shard.g4_perturbed_indices].astype(np.float32) / np.float32(255.0)
    linf = float(np.max(np.abs(shard.g4_perturbed_rgb - clean_norm)))
    assert abs(linf - 1e-3) <= 1e-7
    assert abs(shard.g4_perturbation_linf() - 1e-3) <= 1e-7

    # Golden BF16 reference outputs for all 3 locked v1.0 canonical models
    for vla_model in ("pi05-3b", "gr00t-n1.7-3b"):
      assert vla_model in shard.golden_bf16_ref_actions
      assert shard.golden_bf16_ref_actions[vla_model].shape == (16, 16, 7)
    for wm_model in ("cosmos-predict2.5-2b",):
      assert wm_model in shard.golden_bf16_ref_latents
      assert shard.golden_bf16_ref_latents[wm_model].shape == (16, 16, 16, 16)

  # 4 episodes per pinned real open-source LeRobot dataset source (LIBERO + DROID)
  assert all(count == 4 for count in source_counts.values())


def test_m2_negative_control_tampered_shard_and_stateful_three_paths(tmp_path: Path) -> None:
  """Test 2 (Gate 1 & Gate 4 Negative Control): Tamper with 1 byte of a shard in a temp copy and verify verify_manifest() catches the SHA-256 mismatch with an actionable error message."""
  # Lookalike control: clean copy of mini_v1 must pass verify_manifest() silently
  clean_copy_dir = tmp_path / "clean_copy"
  shutil.copytree(DEFAULT_MINI_V1_DIR, clean_copy_dir)
  clean_loader = EmbodiedXDatasetLoader(dataset_dir=clean_copy_dir, split="mini_v1")
  ok_clean, err_clean = clean_loader.verify_manifest()
  assert ok_clean is True
  assert err_clean == []

  # Negative control: flip 1 byte inside episode_00_libero_spatial.npz
  tampered_dir = tmp_path / "tampered_copy"
  shutil.copytree(DEFAULT_MINI_V1_DIR, tampered_dir)
  shard_0_path = tampered_dir / "episode_00_libero_spatial.npz"
  raw_bytes = bytearray(shard_0_path.read_bytes())
  mid_idx = len(raw_bytes) // 2
  raw_bytes[mid_idx] ^= 0xFF
  shard_0_path.write_bytes(bytes(raw_bytes))

  tampered_loader = EmbodiedXDatasetLoader(dataset_dir=tampered_dir, split="mini_v1")
  ok_tampered, err_tampered = tampered_loader.verify_manifest()
  assert ok_tampered is False
  assert len(err_tampered) == 1
  assert "SHA-256 mismatch" in err_tampered[0]
  assert "episode_00_libero_spatial.npz" in err_tampered[0]
  assert "build_embodiedx_traces.py" in err_tampered[0]

  # Preloading a tampered dataset with verify=True must raise ValueError
  with pytest.raises(ValueError, match="SHA-256 mismatch"):
    tampered_loader.preload_into_pinned_ram(verify=True)

  # Validator negative control: violating 24-token prefix invariant raises ValueError
  shard0 = load_episode_shard_npz(clean_copy_dir / "episode_00_libero_spatial.npz")
  bad_tokens = shard0.instruction_tokens.copy()
  bad_tokens[1, 0] += 1
  bad_shard = replace(shard0, instruction_tokens=bad_tokens)
  with pytest.raises(ValueError, match="constant 24-token instruction prefix"):
    bad_shard.validate()

  # Gate 1 Three Paths: Crash path (orphan temp file cleanup), Repeat path, and Concurrent writers
  orphan_tmp = clean_copy_dir / ".tmp_embodied_manifest.json_orphan.json"
  orphan_tmp.write_text("{corrupt", encoding="utf-8")
  removed = cleanup_orphan_temp_files(clean_copy_dir, prefix=".tmp_embodied_")
  assert removed == 1
  assert not orphan_tmp.exists()

  shards = clean_loader.preload_into_pinned_ram(verify=True)
  shard_paths = [clean_copy_dir / s["filename"] for s in json.loads(clean_loader.manifest_path.read_text())["shards"]]

  def _concurrent_writer(worker_id: int) -> bool:
    m = build_and_write_manifest(
        dataset_dir=clean_copy_dir,
        shards=shards,
        shard_paths=shard_paths,
        split="mini_v1",
    )
    return m["total_episodes"] == 8 and worker_id >= 0

  with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(_concurrent_writer, range(8)))
  assert all(results)
  ok_after_concurrent, errs_after = clean_loader.verify_manifest()
  assert ok_after_concurrent is True
  assert errs_after == []


def test_m2_to_track1_and_track2_golden_batch_all_4_canonical_models() -> None:
  """Test 3: Verify to_track1_golden_batch and to_track2_golden_batch produce valid Track1GoldenBatch and Track2GoldenBatch objects for all canonical v1.0 models and work with adapters & Gate A."""
  loader = EmbodiedXDatasetLoader(dataset_dir=DEFAULT_MINI_V1_DIR, split="mini_v1")
  loader.preload_into_pinned_ram(verify=True)

  for model_id in CANONICAL_V1_MODELS:
    # 1. Track1GoldenBatch verification
    t1_batch = loader.to_track1_golden_batch(model_id=model_id, batch_size=4, horizon_frames=16)
    assert isinstance(t1_batch, Track1GoldenBatch)
    assert t1_batch.batch_size == 4
    assert t1_batch.horizon_frames == 16
    assert t1_batch.obs_frames.shape == (4, 3, 64, 64)
    assert t1_batch.obs_frames.dtype == np.float32
    assert t1_batch.cond_actions.shape == (4, 16, 7)
    assert t1_batch.ref_video_latents.shape == (4, 16, 16, 16)
    assert t1_batch.ref_latent_actions.shape == (4, 16, 7)
    assert t1_batch.temporal_frame_coherence >= 0.94
    assert set(t1_batch.dataset_sources) == {"LIBERO", "DROID"}

    # 2. Track2GoldenBatch verification
    t2_batch = loader.to_track2_golden_batch(model_id=model_id, concurrency=4, chunk_size=16)
    assert isinstance(t2_batch, Track2GoldenBatch)
    assert t2_batch.concurrency == 4
    assert t2_batch.chunk_size == 16
    assert t2_batch.action_dim == 7
    assert t2_batch.rgb_frames.shape == (4, 2, 3, 64, 64)
    assert t2_batch.rgb_frames.dtype == np.float32
    assert t2_batch.proprio_state.shape == (4, 8)
    assert t2_batch.instruction_tokens.shape == (4, 32)
    assert t2_batch.instruction_tokens.dtype == np.int32
    assert t2_batch.ref_actions.shape == (4, 16, 7)
    assert t2_batch.temporal_frame_coherence >= 0.94
    assert t2_batch.instruction_prefix_reuse_ratio >= 0.75
    assert set(t2_batch.dataset_sources) == {"LIBERO", "DROID"}

  # Seamless end-to-end adapter + Gate A check on real loaded shards
  wm_recipe = RecipeConfig(
      recipe_id="test_wm_cosmos_predict25",
      track="WM_TRACK",
      model_family="cosmos-predict2.5-2b",
      model_name="Cosmos-Predict2.5-2B",
      baseline_model_id="cosmos-predict2.5-2b",
      precision="FP8",
      hardware_sku="TPU_V6E_1_32GB",
      division="OPEN",
  )
  wm_adapter = build_adapter(wm_recipe)
  t1_cosmos = loader.to_track1_golden_batch("cosmos-predict2.5-2b", batch_size=2, horizon_frames=16)
  cand_video, cand_la, _ = wm_adapter.run_track1_rollout(t1_cosmos)
  gate_a_t1 = evaluate_track1_gate_a(
      execution_id="exec-m2-wm",
      throughput_execution_id="exec-m2-wm",
      ref_video_latents=t1_cosmos.ref_video_latents,
      cand_video_latents=cand_video,
      ref_latent_actions=t1_cosmos.ref_latent_actions,
      cand_latent_actions=cand_la,
  )
  assert gate_a_t1.passed is True

  vla_recipe = RecipeConfig(
      recipe_id="test_vla_gr00t",
      track="VLA_TRACK",
      model_family="gr00t-n1.7-3b",
      model_name="GR00T-N1.7-3B",
      baseline_model_id="gr00t-n1.7-3b",
      precision="FP8",
      hardware_sku="TPU_V6E_1_32GB",
      division="OPEN",
  )
  vla_adapter = build_adapter(vla_recipe)
  t2_gr00t = loader.to_track2_golden_batch("gr00t-n1.7-3b", concurrency=2, chunk_size=8)
  cand_act, _ = vla_adapter.run_track2_actions(t2_gr00t)
  gate_a_t2 = evaluate_track2_gate_a(
      execution_id="exec-m2-vla",
      throughput_execution_id="exec-m2-vla",
      ref_actions=t2_gr00t.ref_actions,
      cand_actions=cand_act,
  )
  assert gate_a_t2.passed is True


def test_m2_cli_both_flag_orders_and_hf_hub_offline_fallback(tmp_path: Path) -> None:
  """Gate 3 & HF Hub fallback: Execute CLI in both flag orders and verify download_from_hf_hub offline fallback."""
  out_a = tmp_path / "cli_order_a"
  rc_a = build_traces_main(["--output-dir", str(out_a), "--split", "mini_v1", "--num-steps", "8"])
  assert rc_a == 0
  loader_a = EmbodiedXDatasetLoader(dataset_dir=out_a, split="mini_v1")
  assert loader_a.verify_manifest()[0] is True

  out_b = tmp_path / "cli_order_b"
  rc_b = build_traces_main(["--split", "mini_v1", "--num-steps", "8", "--output-dir", str(out_b)])
  assert rc_b == 0
  loader_b = EmbodiedXDatasetLoader(dataset_dir=out_b, split="mini_v1")
  assert loader_b.verify_manifest()[0] is True

  # Test download_from_hf_hub fallback into a fresh directory
  hf_dir = tmp_path / "hf_fallback"
  hf_loader = download_from_hf_hub(
      repo_id="efficient-embodied-ai/EmbodiedX-Traces-v1",
      split="mini_v1",
      local_dir=hf_dir,
  )
  ok_hf, err_hf = hf_loader.verify_manifest()
  assert ok_hf is True
  assert err_hf == []
  shard_rt = save_episode_shard_npz(
      hf_loader.preload_into_pinned_ram()[0],
      tmp_path / "roundtrip_shard.npz",
  )
  loaded_rt = load_episode_shard_npz(shard_rt, freeze_buffers=True)
  assert loaded_rt.episode_id == "libero_spatial_ep00"
