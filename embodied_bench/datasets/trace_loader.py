"""Stateful Multi-Step Episode Trace Loader (`StatefulEpisodeTraceLoader`) for EmbodiedX (`EmbodiedX-Traces-v1`).

Enforces:
  - Cryptographic SHA-256 integrity verification of every `.npz` episode bundle against
    `dataset_manifest.json` on initialization and load (Gate 4 & Gate 6).
  - Per-`session_id` stateful temporal cursor `(split_id, episode_idx, step_idx)` so consecutive
    `next_step(session_id)` calls return sequential camera observations $o_0, o_1, \\dots, o_{T-1}$
    with a constant language instruction prefix (`instruction_tokens` / `instruction_text`),
    enabling genuine VLM prefix KV-cache reuse and World Model causal spatiotemporal KV caching.
  - Seeded micro-perturbation injection (`apply_g4_perturbation=True`, $\\|\\delta\\|_\\infty \\le 10^{-3}$)
    for MLPerf TEST04-style Gate G4 anti-memoization verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embodied_bench.datasets.download_traces import (
    EMBODIEDX_TRACES_V1_DATASETS,
    compute_file_sha256,
)


class DatasetIntegrityError(ValueError, RuntimeError):
    """Raised when a dataset manifest or `.npz` episode bundle fails SHA-256 integrity validation."""


class MissingDatasetArtifactError(FileNotFoundError, DatasetIntegrityError):
    """Raised when `dataset_manifest.json` or an episode `.npz` file is missing from `cache_dir`."""


@dataclass
class SessionCursor:
    """Per-session temporal cursor tracking the active split, episode, and step index."""

    session_id: str
    split_id: str
    episode_idx: int = 0
    step_idx: int = 0
    total_steps_served: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeStepSample:
    """Single step observation, proprioception, instruction prefix, and reference targets from an episode."""

    session_id: str
    split_id: str
    episode_id: str
    episode_idx: int
    step_idx: int
    total_episode_steps: int
    is_first_step: bool
    is_last_step: bool
    rgb_frame: np.ndarray                 # shape: (H, W, 3), float32 in [0, 1]
    wrist_rgb_frame: np.ndarray           # shape: (H, W, 3), float32 in [0, 1]
    multi_camera_rgb: np.ndarray          # shape: (2, H, W, 3), float32 in [0, 1]
    state: np.ndarray                     # shape: (state_dim,), float32
    action: np.ndarray                    # shape: (action_dim,), float32
    reference_action_chunk: np.ndarray    # shape: (chunk_horizon, action_dim), float32
    horizon_rgb_frames: np.ndarray        # shape: (T, H, W, 3), float32 episode frame rollout
    horizon_actions: np.ndarray           # shape: (T, action_dim), float32 episode actions
    instruction_text: str
    instruction_tokens: np.ndarray        # shape: (32,), int32 constant across episode steps
    instruction_prefix_sha256: str
    audio_waveform_16khz: np.ndarray      # shape: (1600,), float32
    g4_perturbed: bool = False
    perturbation_linf: float = 0.0
    state_perturbation_linf: float = 0.0
    rgb_perturbation_linf: float = 0.0
    clean_rgb_frame: Optional[np.ndarray] = None
    clean_state: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable summary of step sample metadata and tensor shapes."""
        return {
            "session_id": self.session_id,
            "split_id": self.split_id,
            "episode_id": self.episode_id,
            "episode_idx": self.episode_idx,
            "step_idx": self.step_idx,
            "total_episode_steps": self.total_episode_steps,
            "is_first_step": self.is_first_step,
            "is_last_step": self.is_last_step,
            "rgb_frame_shape": list(self.rgb_frame.shape),
            "multi_camera_rgb_shape": list(self.multi_camera_rgb.shape),
            "state_shape": list(self.state.shape),
            "action_shape": list(self.action.shape),
            "reference_action_chunk_shape": list(self.reference_action_chunk.shape),
            "horizon_rgb_frames_shape": list(self.horizon_rgb_frames.shape),
            "instruction_text": self.instruction_text,
            "instruction_prefix_sha256": self.instruction_prefix_sha256,
            "g4_perturbed": self.g4_perturbed,
            "perturbation_linf": self.perturbation_linf,
            "state_perturbation_linf": self.state_perturbation_linf,
            "rgb_perturbation_linf": self.rgb_perturbation_linf,
        }


def _apply_bounded_linf_perturbation(
    clean_arr: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1e-3,
    low_bound: Optional[float] = None,
    high_bound: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Apply deterministic non-zero micro-perturbation with $0 < \\|\\delta\\|_\\infty \\le \\text{eps}$."""
    if eps <= 0.0 or eps > 1e-2:
        raise ValueError(
            f"Invalid Gate G4 perturbation_eps={eps}: must be in (0, 1e-2] (canonical default is 1e-3)."
        )
    arr = np.asarray(clean_arr, dtype=np.float32)
    raw_delta = rng.uniform(-0.98 * eps, 0.98 * eps, size=arr.shape).astype(np.float32)
    # Guarantee a well-defined non-zero L_inf norm near eps on the first element
    flat_delta = raw_delta.reshape(-1)
    if flat_delta.size > 0:
        flat_delta[0] = np.float32(0.95 * eps)

    if low_bound is not None and high_bound is not None:
        # Reflect delta inward near boundaries so clipping never exceeds eps or collapses to zero
        near_high = (arr + raw_delta) > high_bound
        near_low = (arr + raw_delta) < low_bound
        raw_delta = np.where(near_high, -np.abs(raw_delta), raw_delta)
        raw_delta = np.where(near_low, np.abs(raw_delta), raw_delta)
        perturbed = np.clip(arr + raw_delta, low_bound, high_bound).astype(np.float32)
    else:
        perturbed = (arr + raw_delta).astype(np.float32)

    actual_diff = np.abs(perturbed.astype(np.float64) - arr.astype(np.float64))
    linf = float(np.max(actual_diff)) if actual_diff.size > 0 else 0.0
    return perturbed, linf


class StatefulEpisodeTraceLoader:
    """Loads verified `EmbodiedX-Traces-v1` `.npz` episode bundles and serves stateful step streams."""

    def __init__(
        self,
        cache_dir: str | Path,
        default_split: Optional[str] = None,
        verify_on_init: bool = True,
        seed: int = 2026,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "dataset_manifest.json"
        self.seed = int(seed)
        self._lock = threading.Lock()
        self._cursors: Dict[str, SessionCursor] = {}

        if not self.manifest_path.exists():
            raise MissingDatasetArtifactError(
                f"Dataset manifest not found at {self.manifest_path}. "
                "Run download_embodiedx_traces(cache_dir=...) first to generate episode bundles and dataset_manifest.json."
            )

        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(
                f"Corrupted JSON in dataset manifest at {self.manifest_path}: {exc}. "
                "Re-run download_embodiedx_traces(cache_dir=...) to repair the manifest."
            ) from exc

        splits_map = self.manifest.get("splits", {})
        if not isinstance(splits_map, dict) or not splits_map:
            raise DatasetIntegrityError(
                f"No dataset splits registered in {self.manifest_path}. "
                "Run download_embodiedx_traces(cache_dir=...) to populate at least one split."
            )

        self.available_splits: List[str] = sorted(splits_map.keys())
        if default_split is not None:
            if default_split not in splits_map:
                raise ValueError(
                    f"Requested default_split={default_split!r} is not present in {self.manifest_path}. "
                    f"Available splits: {self.available_splits}."
                )
            self.default_split = default_split
        else:
            self.default_split = (
                "libero_spatial"
                if "libero_spatial" in splits_map
                else self.available_splits[0]
            )

        if verify_on_init:
            self.verify_manifest_integrity()

    def verify_manifest_integrity(self, split_id: Optional[str] = None) -> Dict[str, Any]:
        """Verify that every `.npz` episode bundle exists on disk and matches its SHA-256 digest."""
        splits_map = self.manifest.get("splits", {})
        target_splits = [split_id] if split_id is not None else sorted(splits_map.keys())
        verified_count = 0

        for sid in target_splits:
            if sid not in splits_map:
                raise ValueError(
                    f"Unknown split_id={sid!r} in verify_manifest_integrity(). "
                    f"Available splits: {sorted(splits_map.keys())}."
                )
            split_rec = splits_map[sid]
            episodes = split_rec.get("episodes", [])
            if not episodes:
                raise DatasetIntegrityError(
                    f"Split {sid!r} in {self.manifest_path} contains 0 episodes. "
                    "Re-run download_embodiedx_traces(cache_dir=...) to materialize episode bundles."
                )
            for ep in episodes:
                rel_path = ep.get("relative_path")
                expected_sha = ep.get("sha256")
                if not rel_path or not expected_sha:
                    raise DatasetIntegrityError(
                        f"Malformed episode entry in split {sid!r} of {self.manifest_path}: "
                        "missing 'relative_path' or 'sha256'. Re-run download_embodiedx_traces()."
                    )
                npz_path = self.cache_dir / rel_path
                if not npz_path.exists():
                    raise MissingDatasetArtifactError(
                        f"Missing episode bundle at {npz_path} (referenced by split {sid!r} in {self.manifest_path}). "
                        f"Re-run download_embodiedx_traces(cache_dir={str(self.cache_dir)!r}, splits=[{sid!r}]) to restore it."
                    )
                actual_sha = compute_file_sha256(npz_path)
                if actual_sha != expected_sha:
                    raise DatasetIntegrityError(
                        f"SHA-256 integrity mismatch for {npz_path}: expected {expected_sha}, "
                        f"got {actual_sha}. The episode bundle was corrupted or modified on disk; "
                        f"re-run download_embodiedx_traces(cache_dir={str(self.cache_dir)!r}, splits=[{sid!r}]) to regenerate."
                    )
                verified_count += 1

        return {
            "verified": True,
            "verified_episodes": verified_count,
            "splits": target_splits,
        }

    def _load_verified_episode(self, split_id: str, episode_idx: int) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Verify SHA-256 and load a single episode `.npz` archive."""
        splits_map = self.manifest.get("splits", {})
        if split_id not in splits_map:
            raise ValueError(
                f"Split {split_id!r} not found in manifest. Available splits: {self.available_splits}."
            )
        episodes = splits_map[split_id].get("episodes", [])
        if not episodes:
            raise DatasetIntegrityError(
                f"Split {split_id!r} has no episodes in {self.manifest_path}."
            )
        ep_meta = episodes[int(episode_idx) % len(episodes)]
        npz_path = self.cache_dir / ep_meta["relative_path"]
        if not npz_path.exists():
            raise MissingDatasetArtifactError(
                f"Episode file {npz_path} is missing on disk. "
                f"Re-run download_embodiedx_traces(cache_dir={str(self.cache_dir)!r}, splits=[{split_id!r}])."
            )
        actual_sha = compute_file_sha256(npz_path)
        expected_sha = ep_meta.get("sha256", "")
        if actual_sha != expected_sha:
            raise DatasetIntegrityError(
                f"SHA-256 checksum mismatch while loading {npz_path}: expected {expected_sha}, "
                f"computed {actual_sha}. Re-run download_embodiedx_traces() to restore clean episode data."
            )
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
        except Exception as exc:
            raise DatasetIntegrityError(
                f"Failed to read numpy archive {npz_path}: {exc}. "
                "Re-run download_embodiedx_traces() to regenerate the episode bundle."
            ) from exc
        return ep_meta, arrays

    def reset_session(
        self,
        session_id: str,
        split_id: Optional[str] = None,
        episode_idx: int = 0,
        step_idx: int = 0,
    ) -> SessionCursor:
        """Reset or initialize the temporal cursor for `session_id`."""
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id must be a non-empty string.")
        resolved_split = split_id or self.default_split
        if resolved_split not in self.manifest.get("splits", {}):
            raise ValueError(
                f"Unknown split_id={resolved_split!r}. Available splits: {self.available_splits}."
            )
        with self._lock:
            cursor = SessionCursor(
                session_id=str(session_id),
                split_id=resolved_split,
                episode_idx=int(episode_idx),
                step_idx=int(step_idx),
                total_steps_served=0,
            )
            self._cursors[str(session_id)] = cursor
            return cursor

    def get_session_cursor(self, session_id: str) -> Dict[str, Any]:
        """Return a snapshot dictionary of the current cursor state for `session_id`."""
        with self._lock:
            if session_id not in self._cursors:
                return SessionCursor(
                    session_id=session_id,
                    split_id=self.default_split,
                    episode_idx=0,
                    step_idx=0,
                    total_steps_served=0,
                ).to_dict()
            return self._cursors[session_id].to_dict()

    def next_step(
        self,
        session_id: str,
        apply_g4_perturbation: bool = False,
        perturbation_eps: float = 1e-3,
        split_id: Optional[str] = None,
    ) -> EpisodeStepSample:
        """Return the next sequential step sample for `session_id` and advance its temporal cursor.

        Consecutive calls with the same `session_id` step through $t = 0, 1, \\dots, T-1$ of the
        active episode before wrapping to `(episode_idx + 1, step_idx = 0)`. All steps within an
        episode carry the exact same `instruction_text`, `instruction_tokens`, and
        `instruction_prefix_sha256` to exercise VLM prefix KV caching.
        """
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id must be a non-empty string.")

        with self._lock:
            if session_id not in self._cursors:
                self._cursors[session_id] = SessionCursor(
                    session_id=session_id,
                    split_id=split_id or self.default_split,
                    episode_idx=0,
                    step_idx=0,
                    total_steps_served=0,
                )
            cursor = self._cursors[session_id]
            if split_id is not None and split_id != cursor.split_id:
                if split_id not in self.manifest.get("splits", {}):
                    raise ValueError(
                        f"Unknown split_id={split_id!r}. Available splits: {self.available_splits}."
                    )
                cursor.split_id = split_id
                cursor.episode_idx = 0
                cursor.step_idx = 0

            active_split = cursor.split_id
            num_episodes = len(self.manifest["splits"][active_split]["episodes"])
            ep_idx = cursor.episode_idx % num_episodes
            step_idx = cursor.step_idx

        ep_meta, arrays = self._load_verified_episode(active_split, ep_idx)
        rgb_seq = np.asarray(arrays["rgb_frames"], dtype=np.float32)
        total_steps = int(rgb_seq.shape[0])

        if step_idx >= total_steps:
            ep_idx = (ep_idx + 1) % num_episodes
            step_idx = 0
            ep_meta, arrays = self._load_verified_episode(active_split, ep_idx)
            rgb_seq = np.asarray(arrays["rgb_frames"], dtype=np.float32)
            total_steps = int(rgb_seq.shape[0])

        with self._lock:
            cursor = self._cursors[session_id]
            if step_idx + 1 < total_steps:
                cursor.episode_idx = ep_idx
                cursor.step_idx = step_idx + 1
            else:
                cursor.episode_idx = (ep_idx + 1) % num_episodes
                cursor.step_idx = 0
            cursor.total_steps_served += 1

        clean_rgb = np.asarray(rgb_seq[step_idx], dtype=np.float32).copy()
        wrist_seq = np.asarray(arrays["wrist_rgb_frames"], dtype=np.float32)
        clean_wrist = np.asarray(wrist_seq[step_idx], dtype=np.float32).copy()
        clean_state = np.asarray(arrays["state"][step_idx], dtype=np.float32).copy()
        action = np.asarray(arrays["actions"][step_idx], dtype=np.float32).copy()
        ref_chunk = np.asarray(arrays["action_chunks"][step_idx], dtype=np.float32).copy()
        instr_tokens = np.asarray(arrays["instruction_tokens"], dtype=np.int32).copy()
        instr_text = str(ep_meta.get("instruction_text", str(arrays["instruction_text"].item())))
        prefix_sha = str(
            ep_meta.get(
                "instruction_prefix_sha256",
                hashlib.sha256(np.ascontiguousarray(instr_tokens).tobytes()).hexdigest(),
            )
        )
        audio_step = np.asarray(arrays["audio_waveform_16khz"][step_idx], dtype=np.float32).copy()
        horizon_rgb = rgb_seq.copy()
        horizon_actions = np.asarray(arrays["actions"], dtype=np.float32).copy()

        if apply_g4_perturbation:
            split_hash = int.from_bytes(hashlib.sha256(active_split.encode("utf-8")).digest()[:4], "little")
            pert_rng = np.random.default_rng(self.seed + split_hash + ep_idx * 997 + step_idx * 31)
            rgb_out, rgb_linf = _apply_bounded_linf_perturbation(
                clean_rgb, pert_rng, eps=float(perturbation_eps), low_bound=0.0, high_bound=1.0
            )
            wrist_out, wrist_linf = _apply_bounded_linf_perturbation(
                clean_wrist, pert_rng, eps=float(perturbation_eps), low_bound=0.0, high_bound=1.0
            )
            state_out, state_linf = _apply_bounded_linf_perturbation(
                clean_state, pert_rng, eps=float(perturbation_eps), low_bound=-1.0, high_bound=1.0
            )
            horizon_rgb = horizon_rgb.copy()
            horizon_rgb[step_idx] = rgb_out
            multi_cam = np.stack([rgb_out, wrist_out], axis=0)
            overall_linf = max(rgb_linf, wrist_linf, state_linf)
        else:
            rgb_out = clean_rgb
            wrist_out = clean_wrist
            state_out = clean_state
            multi_cam = np.stack([rgb_out, wrist_out], axis=0)
            rgb_linf = 0.0
            state_linf = 0.0
            overall_linf = 0.0

        return EpisodeStepSample(
            session_id=session_id,
            split_id=active_split,
            episode_id=str(ep_meta["episode_id"]),
            episode_idx=ep_idx,
            step_idx=step_idx,
            total_episode_steps=total_steps,
            is_first_step=(step_idx == 0),
            is_last_step=(step_idx == total_steps - 1),
            rgb_frame=rgb_out,
            wrist_rgb_frame=wrist_out,
            multi_camera_rgb=multi_cam,
            state=state_out,
            action=action,
            reference_action_chunk=ref_chunk,
            horizon_rgb_frames=horizon_rgb,
            horizon_actions=horizon_actions,
            instruction_text=instr_text,
            instruction_tokens=instr_tokens,
            instruction_prefix_sha256=prefix_sha,
            audio_waveform_16khz=audio_step,
            g4_perturbed=bool(apply_g4_perturbation),
            perturbation_linf=round(float(overall_linf), 8),
            state_perturbation_linf=round(float(state_linf), 8),
            rgb_perturbation_linf=round(float(rgb_linf), 8),
            clean_rgb_frame=clean_rgb,
            clean_state=clean_state,
        )

    def sample_paired_g4_step(
        self,
        session_id: str,
        perturbation_eps: float = 1e-3,
        split_id: Optional[str] = None,
    ) -> Tuple[EpisodeStepSample, EpisodeStepSample]:
        """Return `(clean_sample, perturbed_sample)` for the current step of `session_id` and advance cursor once."""
        cursor_before = self.get_session_cursor(session_id)
        active_split = split_id or cursor_before["split_id"]
        clean_sample = self.next_step(
            session_id=session_id,
            apply_g4_perturbation=False,
            split_id=active_split,
        )
        # Temporarily rewind to produce the exact paired perturbed sample at the same (episode_idx, step_idx)
        self.reset_session(
            session_id=session_id,
            split_id=clean_sample.split_id,
            episode_idx=clean_sample.episode_idx,
            step_idx=clean_sample.step_idx,
        )
        perturbed_sample = self.next_step(
            session_id=session_id,
            apply_g4_perturbation=True,
            perturbation_eps=perturbation_eps,
            split_id=clean_sample.split_id,
        )
        return clean_sample, perturbed_sample
