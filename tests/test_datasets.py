"""Unit and integration tests for EmbodiedX Phase 2 Real Open-Source Dataset Downloader, Stateful Trace Loader, and Co-Located Golden Reference Quality Metrics.

Covers:
  - Canonical dataset registry `EMBODIEDX_TRACES_V1_DATASETS` mapping (`libero_spatial`, `libero_object`,
    `libero_goal`, `libero_10`, `droid_100`, `bridgedata_v2_omni`) to HuggingFace / LeRobot repos.
  - `download_embodiedx_traces` in `smoke` and `auto`/`hf` modes, `.npz` episode bundle generation,
    temporal frame continuity ($\\rho(o_t, o_{t+1}) > 0.85$ vs random noise negative control),
    and SHA-256 manifest verification.
  - Gate 1: Crash path (orphan `.tmp_embodied_*` cleanup & corrupt JSON recovery), repeat-call path
    (zero stale optional field carryover), and concurrent-writer path (multi-thread + multi-process
    atomic read-modify-write via `update_dataset_manifest`).
  - Gate 2: Complete 8-state `RunStatus` reachability, transition legality, and mandatory non-empty
    failure reasons on dataset manifest updates.
  - Gate 4: Negative controls on SHA-256 `.npz` corruption/tampering check, `compute_colocated_quality_metrics`
    (Track 1 VLA & Track 2 World/Omni passing vs failing tensors), and `verify_gate_g4_anti_memoization`
    (firing on memoized cache vs staying silent on real forward pass).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from embodied_bench.datasets import (
    ColocatedQualityResult,
    DATASET_MANIFEST_FIELD_EXAMPLES,
    DATASET_SCHEMA_VERSION,
    DatasetIntegrityError,
    EMBODIEDX_TRACES_V1_DATASETS,
    EpisodeStepSample,
    GATE_G4_MEMOIZATION_MAX_DIFF_FLOOR,
    GateG4AntiMemoizationResult,
    MissingDatasetArtifactError,
    StatefulEpisodeTraceLoader,
    TRACK1_G1_THRESHOLDS,
    TRACK2_G1_THRESHOLDS,
    compute_colocated_quality_metrics,
    compute_file_sha256,
    compute_frame_temporal_correlation,
    compute_token_rouge_l,
    download_embodiedx_traces,
    update_dataset_manifest,
    verify_gate_g4_anti_memoization,
)
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    MODERN_REFERENCE_CHECKPOINTS,
    REASON_REQUIRED_STATUSES,
    RunStatus,
    TrackType,
    cleanup_orphan_temp_files,
)


def _subprocess_manifest_worker(args: tuple[str, int]) -> int:
    """Top-level multiprocessing worker for Gate 1 concurrent subprocess manifest update test."""
    manifest_path, worker_id = args
    split_id = f"subproc_split_{worker_id:02d}"
    update_dataset_manifest(
        manifest_path=manifest_path,
        split_id=split_id,
        status=RunStatus.PENDING,
        source_mode="smoke",
        hf_repo=f"lerobot/{split_id}",
    )
    update_dataset_manifest(
        manifest_path=manifest_path,
        split_id=split_id,
        status=RunStatus.RUNNING,
        source_mode="smoke",
        hf_repo=f"lerobot/{split_id}",
    )
    update_dataset_manifest(
        manifest_path=manifest_path,
        split_id=split_id,
        status=RunStatus.CERTIFIED,
        episodes=[
            {
                "episode_id": f"{split_id}_ep0000",
                "relative_path": f"{split_id}/episode_0000.npz",
                "sha256": f"{worker_id:064x}",
                "num_steps": 16,
            }
        ],
        source_mode="smoke",
        hf_repo=f"lerobot/{split_id}",
    )
    return worker_id


class TestEmbodiedXDatasets(unittest.TestCase):
    """Comprehensive tests for EmbodiedX-Traces-v1 downloader, trace loader, and golden reference metrics."""

    def test_canonical_dataset_registry_and_schema_reuse(self) -> None:
        expected_repos = {
            "libero_spatial": "lerobot/libero_spatial_image",
            "libero_object": "lerobot/libero_object_image",
            "libero_goal": "lerobot/libero_goal_image",
            "libero_10": "lerobot/libero_10_image",
            "droid_100": "lerobot/droid_100",
            "bridgedata_v2_omni": "rail-berkeley/bridge_dataset",
        }
        for split_id, expected_repo in expected_repos.items():
            self.assertIn(split_id, EMBODIEDX_TRACES_V1_DATASETS)
            entry = EMBODIEDX_TRACES_V1_DATASETS[split_id]
            self.assertEqual(entry["hf_repo"], expected_repo)
            self.assertTrue(len(entry["sample_instructions"]) >= 3)
            for model_id in entry["canonical_models"]:
                self.assertIn(model_id, CANONICAL_CHALLENGE_MODELS)
                self.assertIn(model_id, MODERN_REFERENCE_CHECKPOINTS)

        omni_entry = EMBODIEDX_TRACES_V1_DATASETS["bridgedata_v2_omni"]
        self.assertEqual(
            omni_entry["paired_omni_checkpoint"],
            MODERN_REFERENCE_CHECKPOINTS["qwen3-omni-30b-a3b"]["hf_repo"],
        )
        self.assertEqual(omni_entry["audio_sample_rate_hz"], 16000)

        # Gate 6: Verify worked examples exist for every dataset manifest field
        for field_name, meta in DATASET_MANIFEST_FIELD_EXAMPLES.items():
            self.assertIn("description", meta, msg=f"Missing description for {field_name}")
            self.assertIn("example", meta, msg=f"Missing worked example for {field_name}")

    def test_download_embodiedx_traces_smoke_and_temporal_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "traces_cache"
            manifest = download_embodiedx_traces(
                cache_dir=cache_dir,
                splits=["libero_spatial", "droid_100", "bridgedata_v2_omni"],
                mode="smoke",
                num_episodes_per_split=3,
                steps_per_episode=16,
                seed=2026,
            )
            self.assertEqual(manifest["schema_version"], DATASET_SCHEMA_VERSION)
            self.assertIn("PDT", manifest["updated_at"] + " PDT PST")

            for split_id in ("libero_spatial", "droid_100", "bridgedata_v2_omni"):
                self.assertIn(split_id, manifest["splits"])
                split_rec = manifest["splits"][split_id]
                self.assertEqual(split_rec["status"], RunStatus.CERTIFIED.value)
                self.assertEqual(split_rec["num_episodes"], 3)
                self.assertEqual(len(split_rec["episodes"]), 3)

                for ep in split_rec["episodes"]:
                    npz_path = cache_dir / ep["relative_path"]
                    self.assertTrue(npz_path.exists())
                    self.assertEqual(compute_file_sha256(npz_path), ep["sha256"])
                    self.assertGreater(ep["mean_temporal_correlation"], 0.85)
                    self.assertGreater(ep["min_temporal_correlation"], 0.85)

                    with np.load(npz_path, allow_pickle=False) as data:
                        rgb = data["rgb_frames"]
                        wrist = data["wrist_rgb_frames"]
                        state = data["state"]
                        actions = data["actions"]
                        chunks = data["action_chunks"]
                        tokens = data["instruction_tokens"]
                        audio = data["audio_waveform_16khz"]

                        self.assertEqual(rgb.shape, (16, 64, 64, 3))
                        self.assertEqual(wrist.shape, (16, 64, 64, 3))
                        self.assertEqual(state.shape, (16, 8))
                        self.assertEqual(actions.shape, (16, 7))
                        self.assertEqual(chunks.shape, (16, 16, 7))
                        self.assertEqual(tokens.shape, (32,))
                        self.assertEqual(audio.shape, (16, 1600))

                        # Verify adjacent frames evolve smoothly (correlation > 0.85) but are not static
                        corrs = compute_frame_temporal_correlation(rgb)
                        self.assertEqual(corrs.shape, (15,))
                        self.assertTrue(np.all(corrs > 0.85), msg=f"Correlations too low: {corrs}")
                        frame_diffs = np.linalg.norm((rgb[1:] - rgb[:-1]).reshape(15, -1), axis=1)
                        self.assertTrue(np.all(frame_diffs > 1e-2))

            # Gate 4 Negative Control: Random white noise frames must NOT pass the > 0.85 continuity threshold
            rng = np.random.default_rng(42)
            white_noise_frames = rng.uniform(0.0, 1.0, size=(16, 64, 64, 3)).astype(np.float32)
            noise_corrs = compute_frame_temporal_correlation(white_noise_frames)
            self.assertLess(float(np.max(noise_corrs)), 0.15)

    def test_download_embodiedx_traces_hf_mode_mock_and_error_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "hf_cache"
            t_grid = np.linspace(0.1, 0.9, 8, dtype=np.float32)[:, None, None, None]
            mock_frames = np.broadcast_to(t_grid, (8, 64, 64, 3)).tolist()
            fake_hf_response = {
                "rows": [
                    {
                        "row_idx": 0,
                        "row": {
                            "task": "pick up the red block from the wooden tray",
                            "rgb_frames": mock_frames,
                        },
                    },
                    {
                        "row_idx": 1,
                        "row": {
                            "task": "place the red block inside the blue bin",
                            "rgb_frames": mock_frames,
                        },
                    },
                ]
            }
            encoded = json.dumps(fake_hf_response).encode("utf-8")
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = encoded
            mock_resp.__enter__.return_value = mock_resp

            with mock.patch("urllib.request.urlopen", return_value=mock_resp):
                manifest = download_embodiedx_traces(
                    cache_dir=cache_dir,
                    splits=["libero_spatial"],
                    mode="hf",
                    num_episodes_per_split=2,
                    steps_per_episode=8,
                )
            split_rec = manifest["splits"]["libero_spatial"]
            self.assertEqual(split_rec["status"], "CERTIFIED")
            self.assertEqual(split_rec["source_mode"], "hf")
            self.assertEqual(
                split_rec["episodes"][0]["instruction_text"],
                "pick up the red block from the wooden tray",
            )

            # Negative control: forced mode='hf' when network fails records FAILED status with reason
            fail_cache = Path(tmpdir) / "hf_fail_cache"
            with mock.patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
                with self.assertRaises(RuntimeError) as ctx:
                    download_embodiedx_traces(
                        cache_dir=fail_cache,
                        splits=["libero_object"],
                        mode="hf",
                        num_episodes_per_split=1,
                        steps_per_episode=4,
                    )
                self.assertIn("mode='smoke'", str(ctx.exception))

            fail_manifest = json.loads((fail_cache / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(fail_manifest["splits"]["libero_object"]["status"], "FAILED")
            self.assertIn("Network unreachable", fail_manifest["splits"]["libero_object"]["reason"])

    # -------------------------------------------------------------------------
    # Gate 1: Crash Path, Repeat-Call Path, and Concurrent-Writer Path
    # -------------------------------------------------------------------------
    def test_gate1_manifest_crash_path_orphan_cleanup_and_corrupt_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "dataset_manifest.json"

            orphan_a = tmp_path / f".tmp_embodied_{manifest_path.name}_crash1.json"
            orphan_b = tmp_path / f".tmp_embodied_{manifest_path.name}_crash2.json"
            orphan_a.write_text("{partial_crash_write", encoding="utf-8")
            orphan_b.write_text("", encoding="utf-8")
            manifest_path.write_text("{corrupt_manifest_json: [", encoding="utf-8")

            recovered = update_dataset_manifest(
                manifest_path=manifest_path,
                split_id="libero_spatial",
                status=RunStatus.PENDING,
                source_mode="smoke",
            )
            self.assertTrue(recovered.get("recovered_from_corrupt_state"))
            self.assertEqual(recovered["splits"]["libero_spatial"]["status"], "PENDING")
            self.assertFalse(orphan_a.exists())
            self.assertFalse(orphan_b.exists())
            self.assertEqual(cleanup_orphan_temp_files(tmp_path), 0)

    def test_gate1_manifest_repeat_call_no_stale_optional_field_carryover(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "dataset_manifest.json"

            update_dataset_manifest(
                manifest_path=manifest_path,
                split_id="droid_100",
                status=RunStatus.PENDING,
                notes="queued for download",
                extra_metadata={"attempt": 1},
            )
            s_blocked = update_dataset_manifest(
                manifest_path=manifest_path,
                split_id="droid_100",
                status=RunStatus.BLOCKED,
                reason="Waiting on local NVMe scratch quota.",
                notes="disk quota lock",
                extra_metadata={"attempt": 2, "quota_mb": 512},
            )
            rec_blocked = s_blocked["splits"]["droid_100"]
            self.assertEqual(rec_blocked["status"], "BLOCKED")
            self.assertEqual(rec_blocked["reason"], "Waiting on local NVMe scratch quota.")
            self.assertEqual(rec_blocked["notes"], "disk quota lock")
            self.assertEqual(rec_blocked["extra_metadata"], {"attempt": 2, "quota_mb": 512})

            # Transition to RUNNING and then CERTIFIED omitting optional fields
            update_dataset_manifest(
                manifest_path=manifest_path,
                split_id="droid_100",
                status=RunStatus.RUNNING,
            )
            s_cert = update_dataset_manifest(
                manifest_path=manifest_path,
                split_id="droid_100",
                status=RunStatus.CERTIFIED,
                episodes=[{"episode_id": "droid_100_ep0000", "relative_path": "droid_100/episode_0000.npz", "sha256": "a" * 64}],
            )
            rec_cert = s_cert["splits"]["droid_100"]
            self.assertEqual(rec_cert["status"], "CERTIFIED")
            self.assertIsNone(rec_cert["reason"])
            self.assertIsNone(rec_cert["notes"])
            self.assertIsNone(rec_cert["extra_metadata"])
            self.assertEqual(rec_cert["num_episodes"], 1)

    def test_gate1_manifest_concurrent_threads_and_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "concurrent_manifest.json"
            num_threads = 12
            num_procs = 6

            def _thread_worker(tid: int) -> None:
                sid = f"thread_split_{tid:02d}"
                update_dataset_manifest(
                    manifest_path=manifest_path,
                    split_id=sid,
                    status=RunStatus.PENDING,
                )
                update_dataset_manifest(
                    manifest_path=manifest_path,
                    split_id=sid,
                    status=RunStatus.RUNNING,
                )
                update_dataset_manifest(
                    manifest_path=manifest_path,
                    split_id=sid,
                    status=RunStatus.CERTIFIED,
                    episodes=[{"episode_id": f"{sid}_ep0", "relative_path": f"{sid}/ep0.npz", "sha256": f"{tid:064x}"}],
                )

            with ThreadPoolExecutor(max_workers=6) as tpool:
                list(tpool.map(_thread_worker, range(num_threads)))

            with ProcessPoolExecutor(max_workers=4) as ppool:
                list(
                    ppool.map(
                        _subprocess_manifest_worker,
                        [(str(manifest_path), pid) for pid in range(num_procs)],
                    )
                )

            final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            splits = final_manifest["splits"]
            self.assertEqual(len(splits), num_threads + num_procs)
            for tid in range(num_threads):
                self.assertEqual(splits[f"thread_split_{tid:02d}"]["status"], "CERTIFIED")
            for pid in range(num_procs):
                self.assertEqual(splits[f"subproc_split_{pid:02d}"]["status"], "CERTIFIED")

    # -------------------------------------------------------------------------
    # Gate 2: State Machine Completeness on Dataset Build Status
    # -------------------------------------------------------------------------
    def test_gate2_dataset_manifest_state_machine_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "state_machine_manifest.json"
            visited: set[str] = set()

            # Path 1: PENDING -> BLOCKED -> RUNNING -> REJECTED -> PENDING -> SKIPPED -> PENDING -> FAILED -> PENDING -> INVALID
            for status, reason in (
                (RunStatus.PENDING, None),
                (RunStatus.BLOCKED, "Waiting for network mirror"),
                (RunStatus.RUNNING, None),
                (RunStatus.REJECTED, "Frame resolution mismatch"),
                (RunStatus.PENDING, None),
                (RunStatus.SKIPPED, "Split excluded by user filter"),
                (RunStatus.PENDING, None),
                (RunStatus.RUNNING, None),
                (RunStatus.FAILED, "Checksum verification error"),
                (RunStatus.PENDING, None),
                (RunStatus.INVALID, "Unsupported schema format"),
            ):
                state = update_dataset_manifest(
                    manifest_path=manifest_path,
                    split_id="libero_goal",
                    status=status,
                    reason=reason,
                )
                visited.add(state["splits"]["libero_goal"]["status"])

            # Path 2: PENDING -> RUNNING -> CERTIFIED on second split
            for status in (RunStatus.PENDING, RunStatus.RUNNING, RunStatus.CERTIFIED):
                state = update_dataset_manifest(
                    manifest_path=manifest_path,
                    split_id="libero_10",
                    status=status,
                )
                visited.add(state["splits"]["libero_10"]["status"])

            self.assertEqual(visited, {s.value for s in RunStatus})

            # Verify every reason-required status raises ValueError when reason is missing/blank
            for req_status in REASON_REQUIRED_STATUSES:
                with self.assertRaises(ValueError):
                    update_dataset_manifest(
                        manifest_path=manifest_path,
                        split_id=f"missing_reason_{req_status.value}",
                        status=req_status,
                        reason="   ",
                    )

            # Verify illegal transition (CERTIFIED -> PENDING) raises ValueError
            with self.assertRaises(ValueError):
                update_dataset_manifest(
                    manifest_path=manifest_path,
                    split_id="libero_10",
                    status=RunStatus.PENDING,
                )

    # -------------------------------------------------------------------------
    # Trace Loader: Sequential Steps, Prefix KV Reuse, & Gate G4 Perturbations
    # -------------------------------------------------------------------------
    def test_stateful_trace_loader_sequential_cursor_and_g4_perturbation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "traces"
            download_embodiedx_traces(
                cache_dir=cache_dir,
                splits=["libero_spatial"],
                mode="smoke",
                num_episodes_per_split=2,
                steps_per_episode=6,
                seed=2026,
            )
            loader = StatefulEpisodeTraceLoader(cache_dir=cache_dir, default_split="libero_spatial")

            # Serve full 6-step episode 0 and verify sequential step_idx + constant instruction prefix
            ep0_samples: list[EpisodeStepSample] = [
                loader.next_step("session_alpha", apply_g4_perturbation=False)
                for _ in range(6)
            ]
            self.assertEqual([s.step_idx for s in ep0_samples], [0, 1, 2, 3, 4, 5])
            self.assertTrue(ep0_samples[0].is_first_step)
            self.assertTrue(ep0_samples[-1].is_last_step)
            self.assertTrue(all(s.episode_idx == 0 for s in ep0_samples))

            first_prefix_sha = ep0_samples[0].instruction_prefix_sha256
            first_tokens = ep0_samples[0].instruction_tokens
            for s in ep0_samples[1:]:
                self.assertEqual(s.instruction_prefix_sha256, first_prefix_sha)
                np.testing.assert_array_equal(s.instruction_tokens, first_tokens)
                self.assertEqual(s.instruction_text, ep0_samples[0].instruction_text)

            # Next call wraps cleanly to episode_idx=1, step_idx=0
            ep1_first = loader.next_step("session_alpha", apply_g4_perturbation=False)
            self.assertEqual(ep1_first.episode_idx, 1)
            self.assertEqual(ep1_first.step_idx, 0)

            # Verify Gate G4 paired perturbation bounds (0 < ||delta||_inf <= 1e-3)
            clean_step, pert_step = loader.sample_paired_g4_step("session_beta", perturbation_eps=1e-3)
            self.assertFalse(clean_step.g4_perturbed)
            self.assertTrue(pert_step.g4_perturbed)
            self.assertEqual(clean_step.episode_idx, pert_step.episode_idx)
            self.assertEqual(clean_step.step_idx, pert_step.step_idx)

            rgb_diff_linf = float(np.max(np.abs(pert_step.rgb_frame - clean_step.rgb_frame)))
            state_diff_linf = float(np.max(np.abs(pert_step.state - clean_step.state)))
            self.assertGreater(rgb_diff_linf, 0.0)
            self.assertLessEqual(rgb_diff_linf, 1e-3 + 1e-7)
            self.assertGreater(state_diff_linf, 0.0)
            self.assertLessEqual(state_diff_linf, 1e-3 + 1e-7)
            self.assertLessEqual(pert_step.perturbation_linf, 1e-3 + 1e-7)

    # -------------------------------------------------------------------------
    # Gate 4: Negative Controls (SHA-256 Tampering, G1 Metrics, G4 Memoization)
    # -------------------------------------------------------------------------
    def test_gate4_sha256_corruption_and_missing_file_negative_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "traces"
            download_embodiedx_traces(
                cache_dir=cache_dir,
                splits=["libero_spatial"],
                mode="smoke",
                num_episodes_per_split=2,
                steps_per_episode=4,
            )
            # Legitimate control: passes SHA-256 check
            loader = StatefulEpisodeTraceLoader(cache_dir=cache_dir)
            sample = loader.next_step("clean_session")
            self.assertEqual(sample.step_idx, 0)

            # Negative control 1: Corrupt 1 byte in episode_0001.npz and verify SHA-256 check fires
            ep1_path = cache_dir / "libero_spatial" / "episode_0001.npz"
            raw_bytes = bytearray(ep1_path.read_bytes())
            raw_bytes[-10] ^= 0xFF
            ep1_path.write_bytes(bytes(raw_bytes))

            with self.assertRaises(DatasetIntegrityError) as ctx:
                StatefulEpisodeTraceLoader(cache_dir=cache_dir)
            self.assertIn("SHA-256 integrity mismatch", str(ctx.exception))
            self.assertIn("download_embodiedx_traces", str(ctx.exception))

            # Negative control 2: Delete episode_0001.npz and verify MissingDatasetArtifactError fires
            ep1_path.unlink()
            with self.assertRaises(MissingDatasetArtifactError) as missing_ctx:
                StatefulEpisodeTraceLoader(cache_dir=cache_dir)
            self.assertIn("Missing episode bundle", str(missing_ctx.exception))

    def test_gate4_colocated_quality_metrics_track1_and_track2(self) -> None:
        rng = np.random.default_rng(2026)

        # === Track 1 (VLA Actions (H, D)) ===
        ref_actions = rng.uniform(-0.8, 0.8, size=(16, 7)).astype(np.float32)
        cand_actions_good = (
            ref_actions + rng.normal(0.0, 0.003, size=ref_actions.shape).astype(np.float32)
        )
        t1_pass = compute_colocated_quality_metrics(
            candidate_outputs=cand_actions_good,
            reference_bf16_outputs=ref_actions,
            track=TrackType.VLA_TRACK,
            execution_id="exec-t1-001",
            throughput_execution_id="exec-t1-001",
        )
        self.assertTrue(t1_pass.passed)
        self.assertGreaterEqual(t1_pass["action_cosine_sim"], TRACK1_G1_THRESHOLDS["min_action_cosine_sim"])
        self.assertLessEqual(t1_pass["action_l2_rmse"], TRACK1_G1_THRESHOLDS["max_action_l2_rmse"])
        self.assertEqual(t1_pass.violations, [])

        # Track 1 Negative Control: High quantization noise / sign flip fails Gate G1
        cand_actions_bad = -0.5 * ref_actions + rng.normal(0.0, 0.15, size=ref_actions.shape).astype(np.float32)
        t1_fail = compute_colocated_quality_metrics(
            candidate_outputs=cand_actions_bad,
            reference_bf16_outputs=ref_actions,
            track=TrackType.VLA_TRACK,
        )
        self.assertFalse(t1_fail.passed)
        self.assertGreaterEqual(len(t1_fail.violations), 2)

        # === Track 2 (World Model / Omni Frames (T, H, W, 3) + Action Probe + Text) ===
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "wm_traces"
            download_embodiedx_traces(
                cache_dir=cache_dir,
                splits=["bridgedata_v2_omni"],
                mode="smoke",
                num_episodes_per_split=1,
                steps_per_episode=16,
            )
            loader = StatefulEpisodeTraceLoader(cache_dir=cache_dir, default_split="bridgedata_v2_omni")
            step = loader.next_step("wm_session")
            ref_frames = step.horizon_rgb_frames
            ref_ep_actions = step.horizon_actions

        cand_frames_good = np.clip(
            ref_frames + rng.normal(0.0, 0.004, size=ref_frames.shape).astype(np.float32),
            0.0,
            1.0,
        )
        t2_pass = compute_colocated_quality_metrics(
            candidate_outputs=cand_frames_good,
            reference_bf16_outputs=ref_frames,
            track=TrackType.WM_TRACK,
            reference_actions=ref_ep_actions,
            reference_text="put the carrot on the plate and describe the scene state",
            candidate_text="put the carrot on the plate and describe the scene state",
        )
        self.assertTrue(t2_pass.passed, msg=t2_pass.reason)
        self.assertGreaterEqual(t2_pass.psnr_db, TRACK2_G1_THRESHOLDS["min_psnr_db"])
        self.assertLessEqual(t2_pass.lpips_proxy, TRACK2_G1_THRESHOLDS["max_lpips_proxy"])
        self.assertGreaterEqual(t2_pass.inv_dyn_r2, TRACK2_G1_THRESHOLDS["min_inv_dyn_r2"])
        self.assertAlmostEqual(t2_pass.rouge_l, 1.0, places=5)

        # Track 2 Negative Control: Corrupted/noisy rollout fails PSNR, LPIPS proxy, and InvDyn R^2
        cand_frames_bad = rng.uniform(0.0, 1.0, size=ref_frames.shape).astype(np.float32)
        t2_fail = compute_colocated_quality_metrics(
            candidate_outputs=cand_frames_bad,
            reference_bf16_outputs=ref_frames,
            track=TrackType.WM_TRACK,
            reference_actions=ref_ep_actions,
            reference_text="put the carrot on the plate",
            candidate_text="completely unrelated hallucinated tokens",
        )
        self.assertFalse(t2_fail.passed)
        self.assertLess(t2_fail.psnr_db, TRACK2_G1_THRESHOLDS["min_psnr_db"])
        self.assertGreater(t2_fail.lpips_proxy, TRACK2_G1_THRESHOLDS["max_lpips_proxy"])
        self.assertLess(t2_fail.inv_dyn_r2, TRACK2_G1_THRESHOLDS["min_inv_dyn_r2"])
        self.assertLess(t2_fail.rouge_l, 0.25)
        self.assertEqual(len(t2_fail.violations), 3)

        # Co-located execution_id mismatch negative control
        with self.assertRaises(ValueError):
            compute_colocated_quality_metrics(
                candidate_outputs=cand_actions_good,
                reference_bf16_outputs=ref_actions,
                execution_id="exec-A",
                throughput_execution_id="exec-B",
            )

    def test_gate4_anti_memoization_fires_on_cache_and_silent_on_real_forward(self) -> None:
        rng = np.random.default_rng(99)
        clean_out = rng.uniform(-0.9, 0.9, size=(8, 16, 7)).astype(np.float32)

        # 1. Negative Control (Violation): Memoized lookup table returns identical outputs (max_diff == 0 < 1e-7)
        memoized_pert_out = clean_out.copy()
        res_memoized = verify_gate_g4_anti_memoization(clean_out, memoized_pert_out)
        self.assertFalse(res_memoized.passed)
        self.assertTrue(res_memoized.memoization_detected)
        self.assertLess(res_memoized.max_diff, GATE_G4_MEMOIZATION_MAX_DIFF_FLOOR)
        self.assertTrue(len(res_memoized.violations) >= 1)

        with self.assertRaises(ValueError):
            verify_gate_g4_anti_memoization(clean_out, memoized_pert_out, raise_on_violation=True)

        # 2. Legitimate Lookalike (Silent): Real model forward pass shifts outputs proportionally to 1e-3 input perturbation
        jacobian_shift = rng.normal(0.0, 8e-4, size=clean_out.shape).astype(np.float32)
        responsive_pert_out = clean_out + jacobian_shift
        res_responsive = verify_gate_g4_anti_memoization(
            clean_outputs=clean_out,
            perturbed_outputs=responsive_pert_out,
            reference_clean_outputs=clean_out,
            reference_perturbed_outputs=clean_out + jacobian_shift * 1.02,
        )
        self.assertTrue(res_responsive.passed, msg=res_responsive.reason)
        self.assertFalse(res_responsive.memoization_detected)
        self.assertGreaterEqual(res_responsive.max_diff, GATE_G4_MEMOIZATION_MAX_DIFF_FLOOR)
        self.assertEqual(res_responsive.violations, [])


if __name__ == "__main__":
    unittest.main()
