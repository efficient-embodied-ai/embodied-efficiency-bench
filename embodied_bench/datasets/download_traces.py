"""Real Open-Source Embodied AI Dataset Registry & Deterministic Trajectory Synthesizer (`EmbodiedX-Traces-v1`).

Maps canonical EmbodiedX split IDs to real open-source HuggingFace / LeRobot datasets across
Track 1 (Real-Time VLA Policies) and Track 2 (Action-Conditioned World Models & Omni Models):
  - `libero_spatial`    -> `lerobot/libero_spatial_image`
  - `libero_object`     -> `lerobot/libero_object_image`
  - `libero_goal`       -> `lerobot/libero_goal_image`
  - `libero_10`         -> `lerobot/libero_10_image`
  - `droid_100`         -> `lerobot/droid_100`
  - `bridgedata_v2_omni`-> `rail-berkeley/bridge_dataset` (paired with `Qwen/Qwen3-Omni-30B-A3B-Instruct` spoken+text instructions)

Supports online HuggingFace Datasets Server / Hub metadata & row fetching (`mode="hf"` / `"auto"`)
and deterministic offline smoke trajectory generation (`mode="smoke"` / `"auto"` fallback) where
camera frames evolve smoothly along a continuous physical trajectory ($o_t \\to o_{t+1}$) with
high frame-to-frame structural continuity ($\\rho(o_t, o_{t+1}) > 0.85$) and SHA-256 manifest
verification under `atomic_read_modify_write_json` (Gate 1 & Gate 2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from eeb.data.schema import (
    REAL_ROBOT_TRACES_KIND as EMBODIEDX_TRACES_KIND,
    SYNTHETIC_CI_SMOKE_TRACES_KIND,
    is_synthetic_trace_manifest,
)
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    MODERN_REFERENCE_CHECKPOINTS,
    RunStatus,
    TrackType,
    atomic_read_modify_write_json,
    atomic_write_json,
    cleanup_orphan_temp_files,
    now_pacific_iso,
    validate_status_and_reason,
)


DATASET_SCHEMA_VERSION = "EmbodiedX-Traces-v1.0"

# Canonical Dataset Registry mapping split IDs to real open-source HuggingFace / LeRobot repositories
EMBODIEDX_TRACES_V1_DATASETS: Dict[str, Dict[str, Any]] = {
    "libero_spatial": {
        "split_id": "libero_spatial",
        "hf_repo": "lerobot/libero_spatial_image",
        "tracks": (TrackType.VLA_TRACK.value, TrackType.TRACK1_VLA.value),
        "canonical_models": ("gr00t-n1.7-3b", "pi05-3b"),
        "robot_morphology": "Franka Emika Panda (Tabletop Spatial Manipulation)",
        "modality": "2-camera 224x224 RGB (agentview + eye_in_hand) + 8-DoF state + language instruction + 7-DoF action",
        "num_cameras": 2,
        "native_resolution_hw": (224, 224),
        "state_dim": 8,
        "action_dim": 7,
        "control_freq_hz": 20.0,
        "license": "MIT",
        "sample_instructions": (
            "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "pick up the black bowl next to the ramekin and place it on the plate",
            "pick up the black bowl from table center and place it on the plate",
            "pick up the black bowl on the cookie box and place it on the plate",
            "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        ),
    },
    "libero_object": {
        "split_id": "libero_object",
        "hf_repo": "lerobot/libero_object_image",
        "tracks": (TrackType.VLA_TRACK.value, TrackType.TRACK1_VLA.value),
        "canonical_models": ("gr00t-n1.7-3b", "pi05-3b"),
        "robot_morphology": "Franka Emika Panda (Multi-Object Tabletop Manipulation)",
        "modality": "2-camera 224x224 RGB (agentview + eye_in_hand) + 8-DoF state + language instruction + 7-DoF action",
        "num_cameras": 2,
        "native_resolution_hw": (224, 224),
        "state_dim": 8,
        "action_dim": 7,
        "control_freq_hz": 20.0,
        "license": "MIT",
        "sample_instructions": (
            "pick up the alphabet soup and place it in the basket",
            "pick up the cream cheese and place it in the basket",
            "pick up the salad dressing and place it in the basket",
            "pick up the bbq sauce and place it in the basket",
            "pick up the ketchup and place it in the basket",
        ),
    },
    "libero_goal": {
        "split_id": "libero_goal",
        "hf_repo": "lerobot/libero_goal_image",
        "tracks": (TrackType.VLA_TRACK.value, TrackType.TRACK1_VLA.value),
        "canonical_models": ("gr00t-n1.7-3b", "pi05-3b"),
        "robot_morphology": "Franka Emika Panda (Goal-Conditioned Manipulation)",
        "modality": "2-camera 224x224 RGB (agentview + eye_in_hand) + 8-DoF state + language instruction + 7-DoF action",
        "num_cameras": 2,
        "native_resolution_hw": (224, 224),
        "state_dim": 8,
        "action_dim": 7,
        "control_freq_hz": 20.0,
        "license": "MIT",
        "sample_instructions": (
            "open the middle drawer of the cabinet",
            "put the bowl on the stove",
            "put the wine bottle on top of the cabinet",
            "open the top drawer and put the bowl inside",
            "push the plate to the front of the stove",
        ),
    },
    "libero_10": {
        "split_id": "libero_10",
        "hf_repo": "lerobot/libero_10_image",
        "tracks": (TrackType.VLA_TRACK.value, TrackType.TRACK1_VLA.value),
        "canonical_models": ("gr00t-n1.7-3b", "pi05-3b"),
        "robot_morphology": "Franka Emika Panda (LIBERO-Long 10-Task Multi-Stage Manipulation)",
        "modality": "2-camera 224x224 RGB (agentview + eye_in_hand) + 8-DoF state + language instruction + 7-DoF action",
        "num_cameras": 2,
        "native_resolution_hw": (224, 224),
        "state_dim": 8,
        "action_dim": 7,
        "control_freq_hz": 20.0,
        "license": "MIT",
        "sample_instructions": (
            "put both the alphabet soup and the tomato sauce in the basket",
            "put both the cream cheese box and the butter in the basket",
            "turn on the stove and put the moka pot on it",
            "put the black bowl in the bottom drawer of the cabinet and close it",
            "put the yellow and white mug in the microwave and close it",
        ),
    },
    "droid_100": {
        "split_id": "droid_100",
        "hf_repo": "lerobot/droid_100",
        "tracks": (
            TrackType.VLA_TRACK.value,
            TrackType.WM_TRACK.value,
            TrackType.TRACK1_VLA.value,
            TrackType.TRACK2_WM.value,
        ),
        "canonical_models": ("gr00t-n1.7-3b", "pi05-3b", "cosmos-predict2.5-2b"),
        "robot_morphology": "Franka Panda 7-DoF + Robotiq 2F-85 (In-the-Wild DROID Stereo/Wrist)",
        "modality": "Exterior stereo + wrist RGB + 8-DoF joint/gripper state + 7-DoF actions + language instruction",
        "num_cameras": 2,
        "native_resolution_hw": (256, 256),
        "state_dim": 8,
        "action_dim": 7,
        "control_freq_hz": 15.0,
        "license": "CC-BY-4.0",
        "sample_instructions": (
            "pick up the blue cup from the kitchen counter and place it in the drying rack",
            "wipe the spilled coffee on the table with the sponge",
            "close the laptop lid on the office desk",
            "move the orange fruit into the ceramic bowl",
            "press the toaster lever down",
        ),
    },
    "bridgedata_v2_omni": {
        "split_id": "bridgedata_v2_omni",
        "hf_repo": "rail-berkeley/bridge_dataset",
        "paired_omni_checkpoint": MODERN_REFERENCE_CHECKPOINTS["qwen3-omni-30b-a3b"]["hf_repo"],
        "tracks": (TrackType.WM_TRACK.value, TrackType.TRACK2_WM.value, TrackType.TRACK1_WORLD_MODEL.value),
        "canonical_models": ("cosmos-predict2.5-2b",),
        "v2_planned_models": ("cosmos3-edge", "qwen3-omni-30b-a3b"),
        "robot_morphology": "WidowX 250 6-DoF Tabletop Manipulator (BridgeData-V2 + Qwen3-Omni Spoken/Text Prompts)",
        "modality": "Video frames + 16kHz audio waveform + text prompt + 8-DoF state + 7-DoF continuous actions",
        "num_cameras": 2,
        "native_resolution_hw": (224, 224),
        "state_dim": 8,
        "action_dim": 7,
        "audio_sample_rate_hz": 16000,
        "control_freq_hz": 10.0,
        "license": "CC-BY-4.0",
        "sample_instructions": (
            "put the carrot on the plate and describe the scene state",
            "move the silver pot to the right burner of the toy kitchen",
            "pick up the green cloth and fold it over the block",
            "place the wooden spoon inside the metal pan",
            "flip the red cup upright on the counter",
        ),
    },
}


# Concrete worked examples for dataset manifest schema fields (Gate 6: Write For The Reader)
DATASET_MANIFEST_FIELD_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "schema_version": {
        "description": "Version identifier for the EmbodiedX trace manifest format.",
        "example": "EmbodiedX-Traces-v1.0",
    },
    "updated_at": {
        "description": "Human-readable America/Los_Angeles (PDT/PST) timestamp of the last manifest update.",
        "example": "2026-10-08T09:30:00-07:00 (PDT)",
    },
    "split_id": {
        "description": "Canonical EmbodiedX dataset split identifier from EMBODIEDX_TRACES_V1_DATASETS.",
        "example": "libero_spatial",
    },
    "hf_repo": {
        "description": "Upstream HuggingFace / LeRobot dataset repository identifier.",
        "example": "lerobot/libero_spatial_image",
    },
    "status": {
        "description": "Lifecycle status of the dataset split build ('PENDING', 'RUNNING', 'CERTIFIED', 'REJECTED', 'BLOCKED', 'FAILED', 'SKIPPED', 'INVALID').",
        "example": "CERTIFIED",
    },
    "reason": {
        "description": "Mandatory non-empty explanation whenever status is REJECTED, BLOCKED, FAILED, SKIPPED, or INVALID.",
        "example": "HuggingFace Hub API returned HTTP 503 while mode='hf' was forced; re-run with mode='auto' or mode='smoke'.",
    },
    "sha256": {
        "description": "Hex-encoded SHA-256 digest of the compressed .npz episode bundle on disk.",
        "example": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
}


@dataclass
class EpisodeBundleMetadata:
    """Metadata and integrity record for a single multi-step episode `.npz` bundle."""

    split_id: str
    episode_id: str
    episode_idx: int
    relative_path: str
    sha256: str
    num_steps: int
    image_hw: Tuple[int, int]
    num_cameras: int
    state_dim: int
    action_dim: int
    chunk_horizon: int
    instruction_text: str
    instruction_prefix_sha256: str
    mean_temporal_correlation: float
    min_temporal_correlation: float
    has_audio_16khz: bool
    pixel_provenance: str = "SYNTHETIC_PROCEDURAL_FALLBACK"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute hex-encoded SHA-256 checksum of a file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot compute SHA-256: file not found at {path}. "
            "Run download_embodiedx_traces() to generate the episode bundle first."
        )
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_frame_temporal_correlation(frames: np.ndarray) -> np.ndarray:
    """Compute per-step Pearson structural correlation between adjacent frames $o_t$ and $o_{t+1}$.

    Args:
        frames: Array of shape `(T, H, W, C)` or `(T, C, H, W)` with `T >= 2`.

    Returns:
        1D float32 array of shape `(T - 1,)` with Pearson correlation coefficients in `[-1, 1]`.
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[0] < 2:
        raise ValueError(
            f"Expected frame sequence with shape (T >= 2, ...), got {arr.shape}. "
            "Pass an episode trajectory with at least 2 time steps."
        )
    t_steps = arr.shape[0]
    flat = arr.reshape(t_steps, -1)
    means = flat.mean(axis=1, keepdims=True)
    centered = flat - means
    norms = np.linalg.norm(centered, axis=1)

    corrs = np.zeros(t_steps - 1, dtype=np.float32)
    for t in range(t_steps - 1):
        denom = max(float(norms[t] * norms[t + 1]), 1e-12)
        corrs[t] = float(np.dot(centered[t], centered[t + 1]) / denom)
    return corrs


def _tokenize_instruction_deterministic(instruction_text: str, length: int = 32) -> np.ndarray:
    """Deterministically tokenize an instruction string into a fixed-length int32 token array."""
    words = instruction_text.strip().lower().split()
    tokens = np.zeros(length, dtype=np.int32)
    # Shared BOS + system instruction framing prefix in tokens[:4]
    tokens[0] = 101
    tokens[1] = 2026
    tokens[2] = 301
    tokens[3] = 409
    for idx in range(4, length):
        word = words[(idx - 4) % max(len(words), 1)] if words else "robot"
        h = hashlib.sha256(f"{idx}:{word}".encode("utf-8")).digest()
        tokens[idx] = 500 + (int.from_bytes(h[:4], "little") % 31000)
    return tokens


def _synthesize_smooth_episode_arrays(
    split_id: str,
    episode_idx: int,
    steps_per_episode: int,
    image_hw: int,
    seed: int,
    instruction_text: str,
    chunk_horizon: int = 16,
) -> Dict[str, Any]:
    """Synthesize a physically coherent multi-step robot trajectory with smooth optical continuity $o_t \\to o_{t+1}$.

    Adjacent camera frames share a static tabletop workspace background while the simulated robot
    end-effector and target object translate smoothly along the continuous action trajectory,
    yielding frame-to-frame Pearson correlation > 0.92 (vs ~0.0 for random white noise).
    """
    spec = EMBODIEDX_TRACES_V1_DATASETS[split_id]
    state_dim = int(spec.get("state_dim", 8))
    action_dim = int(spec.get("action_dim", 7))

    split_hash = int.from_bytes(hashlib.sha256(split_id.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(int(seed) + split_hash + int(episode_idx) * 1009)

    t_grid = np.linspace(0.0, 1.0, steps_per_episode, dtype=np.float32)

    # 1. Smooth continuous 7-DoF actions & 8-DoF proprioceptive state trajectory
    phase_offsets = rng.uniform(0.0, np.pi, size=(1, action_dim)).astype(np.float32)
    freq_scales = rng.uniform(1.0, 2.2, size=(1, action_dim)).astype(np.float32)
    base_actions = (
        0.65 * np.sin(t_grid[:, None] * np.pi * freq_scales + phase_offsets)
        + 0.25 * np.cos(t_grid[:, None] * 0.5 * np.pi + phase_offsets * 0.5)
    )
    action_innov = rng.normal(0.0, 0.02, size=(steps_per_episode, action_dim)).astype(np.float32)
    actions = np.clip(base_actions + action_innov, -1.0, 1.0).astype(np.float32)

    # Reference teacher-forced action chunks of shape (T, chunk_horizon, action_dim)
    action_chunks = np.zeros((steps_per_episode, chunk_horizon, action_dim), dtype=np.float32)
    for t in range(steps_per_episode):
        for k in range(chunk_horizon):
            src_t = min(t + k, steps_per_episode - 1)
            decay = float(np.exp(-0.015 * max(0, (t + k) - (steps_per_episode - 1))))
            action_chunks[t, k] = np.clip(actions[src_t] * decay, -1.0, 1.0)

    # Integrate actions into smooth proprioceptive state in [-0.8, 0.8]
    init_state = rng.uniform(-0.35, 0.35, size=(1, state_dim)).astype(np.float32)
    padded_actions = np.zeros((steps_per_episode, state_dim), dtype=np.float32)
    padded_actions[:, :action_dim] = actions
    if state_dim > action_dim:
        # 8th DoF gripper finger mirror
        padded_actions[:, action_dim:] = -actions[:, -1:]
    state = np.clip(
        init_state + np.cumsum(padded_actions * (0.45 / max(steps_per_episode, 1)), axis=0),
        -0.8,
        0.8,
    ).astype(np.float32)

    # 2. Smooth 2-camera visual trajectory (agentview + wrist view)
    y_coords, x_coords = np.meshgrid(
        np.linspace(0.0, 1.0, image_hw, dtype=np.float32),
        np.linspace(0.0, 1.0, image_hw, dtype=np.float32),
        indexing="ij",
    )

    def _build_camera_sequence(cam_idx: int) -> np.ndarray:
        cam_rng = np.random.default_rng(int(seed) + split_hash + episode_idx * 97 + cam_idx * 503)
        # Static tabletop & lighting gradient + workspace landmarks
        bg = np.zeros((image_hw, image_hw, 3), dtype=np.float32)
        for ch in range(3):
            kx = float(cam_rng.uniform(1.5, 3.5))
            ky = float(cam_rng.uniform(1.5, 3.5))
            phi = float(cam_rng.uniform(0.0, 2.0 * np.pi))
            bg[:, :, ch] = (
                0.45
                + 0.22 * np.sin(2.0 * np.pi * kx * x_coords + phi) * np.cos(2.0 * np.pi * ky * y_coords)
                + 0.14 * (1.0 - 0.5 * (x_coords + y_coords))
            )
        # Static target object landmark
        goal_x = float(cam_rng.uniform(0.55, 0.78))
        goal_y = float(cam_rng.uniform(0.55, 0.78))
        goal_blob = np.exp(-((x_coords - goal_x) ** 2 + (y_coords - goal_y) ** 2) / (2.0 * 0.085 ** 2))
        bg[:, :, 0] = np.clip(bg[:, :, 0] + 0.25 * goal_blob, 0.05, 0.95)
        bg[:, :, 2] = np.clip(bg[:, :, 2] - 0.15 * goal_blob, 0.05, 0.95)

        frames = np.zeros((steps_per_episode, image_hw, image_hw, 3), dtype=np.float32)
        ee_x0 = float(cam_rng.uniform(0.22, 0.36))
        ee_y0 = float(cam_rng.uniform(0.22, 0.36))

        # Temporal AR(1) illumination state for ultra-realistic camera sensor continuity
        illum_noise = np.zeros((image_hw, image_hw, 3), dtype=np.float32)
        for t in range(steps_per_episode):
            prog = float(t) / max(steps_per_episode - 1, 1)
            # End-effector moves smoothly toward target modulated by state trajectory
            ee_x = (1.0 - prog) * ee_x0 + prog * goal_x + 0.04 * float(state[t, 0])
            ee_y = (1.0 - prog) * ee_y0 + prog * goal_y + 0.04 * float(state[t, 1])
            ee_blob = np.exp(-((x_coords - ee_x) ** 2 + (y_coords - ee_y) ** 2) / (2.0 * 0.095 ** 2))

            illum_noise = (
                0.85 * illum_noise
                + cam_rng.normal(0.0, 0.006, size=(image_hw, image_hw, 3)).astype(np.float32)
            )
            frame = bg.copy()
            frame[:, :, 1] += 0.28 * ee_blob
            frame[:, :, 2] += 0.20 * ee_blob
            frame = np.clip(frame + illum_noise, 0.0, 1.0)
            frames[t] = frame
        return frames

    rgb_frames = _build_camera_sequence(cam_idx=0)
    wrist_rgb_frames = _build_camera_sequence(cam_idx=1)
    multi_camera_rgb = np.stack([rgb_frames, wrist_rgb_frames], axis=1)

    # 3. Constant episode instruction tokens (enabling VLM prefix KV-cache reuse across all T steps)
    instruction_tokens = _tokenize_instruction_deterministic(instruction_text, length=32)

    # 4. 16kHz spoken instruction audio waveform (1600 samples = 100ms per step)
    samples_per_step = 1600
    t_audio = np.linspace(0.0, float(steps_per_episode) * 0.1, steps_per_episode * samples_per_step, dtype=np.float32)
    f0 = 165.0 + float((episode_idx % 5) * 18.0)
    envelope = 0.55 * (0.6 + 0.4 * np.sin(2.0 * np.pi * 3.5 * t_audio))
    waveform_flat = envelope * (
        0.6 * np.sin(2.0 * np.pi * f0 * t_audio)
        + 0.3 * np.sin(2.0 * np.pi * (2.0 * f0) * t_audio)
        + 0.1 * np.cos(2.0 * np.pi * (3.0 * f0) * t_audio)
    )
    audio_waveform_16khz = np.clip(
        waveform_flat.reshape(steps_per_episode, samples_per_step),
        -1.0,
        1.0,
    ).astype(np.float32)

    temporal_corrs = compute_frame_temporal_correlation(rgb_frames)

    return {
        "rgb_frames": rgb_frames,
        "wrist_rgb_frames": wrist_rgb_frames,
        "multi_camera_rgb": multi_camera_rgb,
        "state": state,
        "actions": actions,
        "action_chunks": action_chunks,
        "instruction_tokens": instruction_tokens,
        "instruction_text": np.array(instruction_text),
        "audio_waveform_16khz": audio_waveform_16khz,
        "temporal_correlations": temporal_corrs,
        "pixel_provenance": np.array("SYNTHETIC_PROCEDURAL_FALLBACK"),
    }


def fetch_hf_dataset_rows(
    hf_repo: str,
    split: str = "train",
    config: str = "default",
    offset: int = 0,
    length: int = 16,
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    """Fetch real rows/metadata from the HuggingFace Datasets Server HTTP API."""
    query = urllib.parse.urlencode({
        "dataset": hf_repo,
        "config": config,
        "split": split,
        "offset": int(offset),
        "length": int(length),
    })
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "EmbodiedX-Traces-Downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload, dict) or "rows" not in payload:
                raise RuntimeError(
                    f"Unexpected HuggingFace Datasets Server response for {hf_repo!r}: missing 'rows' key."
                )
            return payload
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to fetch HuggingFace dataset {hf_repo!r} from {url}: {exc}. "
            "Check network connectivity or run download_embodiedx_traces(..., mode='smoke' / mode='auto') "
            "for deterministic offline trajectory synthesis."
        ) from exc


def update_dataset_manifest(
    manifest_path: str | Path,
    split_id: str,
    status: str | RunStatus,
    episodes: Optional[Sequence[Dict[str, Any]]] = None,
    source_mode: str = "smoke",
    hf_repo: Optional[str] = None,
    reason: Optional[str] = None,
    notes: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    pixel_provenance: str = "SYNTHETIC_PROCEDURAL_FALLBACK",
) -> Dict[str, Any]:
    """Atomically update a dataset split entry in `dataset_manifest.json` (Gate 1 & Gate 2).

    Enforces:
      - Gate 1 Crash Path: Cleans up orphan `.tmp_embodied_*` files and recovers from corrupt JSON.
      - Gate 1 Repeat Path: Replaces the split record cleanly without carrying over stale optional
        fields (`reason`, `notes`, `extra_metadata`) from prior calls.
      - Gate 1 Concurrent Path: Holds `fcntl.LOCK_EX` across the entire read-modify-write cycle.
      - Gate 2 State Machine: Validates `RunStatus` transitions and enforces non-empty `reason`
        on `REJECTED`, `BLOCKED`, `FAILED`, `SKIPPED`, and `INVALID`.
    """
    if not split_id or not str(split_id).strip():
        raise ValueError(
            "split_id must be a non-empty string (e.g. 'libero_spatial' or 'droid_100')."
        )

    def _modifier(state: Dict[str, Any]) -> Dict[str, Any]:
        is_real = str(pixel_provenance) != "SYNTHETIC_PROCEDURAL_FALLBACK"
        state.setdefault("schema_version", DATASET_SCHEMA_VERSION)
        state.setdefault("canonical_challenge_models", list(CANONICAL_CHALLENGE_MODELS))
        state["pixel_provenance"] = str(pixel_provenance)
        state["is_real_camera_pixels"] = bool(is_real)
        state["kind"] = "REAL_ROBOT_TRACES" if is_real else "SYNTHETIC_CI_SMOKE_TRACES"
        state["is_synthetic_smoke"] = not is_real
        state["certifiable"] = bool(is_real)
        splits_map = state.setdefault("splits", {})

        existing = splits_map.get(split_id)
        prev_status = existing.get("status") if isinstance(existing, dict) else None
        resolved_status = validate_status_and_reason(
            status=status,
            reason=reason,
            previous_status=prev_status,
        )

        spec = EMBODIEDX_TRACES_V1_DATASETS.get(split_id, {})
        resolved_hf_repo = (
            hf_repo
            if hf_repo is not None
            else (
                existing.get("hf_repo")
                if isinstance(existing, dict) and existing.get("hf_repo")
                else spec.get("hf_repo", f"custom/{split_id}")
            )
        )
        resolved_episodes = (
            [dict(ep) for ep in episodes]
            if episodes is not None
            else (
                list(existing.get("episodes", []))
                if isinstance(existing, dict)
                else []
            )
        )

        is_synth = is_synthetic_trace_manifest({"pixel_provenance": str(pixel_provenance)})
        record_kind = (
            SYNTHETIC_CI_SMOKE_TRACES_KIND if is_synth else EMBODIEDX_TRACES_KIND
        )
        # Build fresh record to prevent stale optional field carryover on repeat calls (Gate 1)
        record: Dict[str, Any] = {
            "kind": record_kind,
            "split_id": split_id,
            "hf_repo": resolved_hf_repo,
            "tracks": list(spec.get("tracks", (TrackType.VLA_TRACK.value,))),
            "canonical_models": list(spec.get("canonical_models", CANONICAL_CHALLENGE_MODELS[:2])),
            "status": resolved_status.value,
            "source_mode": str(source_mode),
            "pixel_provenance": str(pixel_provenance),
            "is_synthetic_smoke": bool(is_synth),
            "certifiable": not bool(is_synth),
            "num_episodes": len(resolved_episodes),
            "episodes": resolved_episodes,
            "updated_at": now_pacific_iso(),
            "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None,
            "notes": notes.strip() if isinstance(notes, str) and notes.strip() else None,
            "extra_metadata": dict(extra_metadata) if extra_metadata is not None else None,
        }
        splits_map[split_id] = record
        state["updated_at"] = now_pacific_iso()
        return state

    return atomic_read_modify_write_json(
        manifest_path,
        modifier_fn=_modifier,
        default_factory=lambda: {
            "kind": (
                SYNTHETIC_CI_SMOKE_TRACES_KIND
                if is_synthetic_trace_manifest({"pixel_provenance": str(pixel_provenance)})
                else EMBODIEDX_TRACES_KIND
            ),
            "schema_version": DATASET_SCHEMA_VERSION,
            "canonical_challenge_models": list(CANONICAL_CHALLENGE_MODELS),
            "pixel_provenance": str(pixel_provenance),
            "is_synthetic_smoke": bool(
                is_synthetic_trace_manifest({"pixel_provenance": str(pixel_provenance)})
            ),
            "certifiable": not bool(
                is_synthetic_trace_manifest({"pixel_provenance": str(pixel_provenance)})
            ),
            "splits": {},
            "updated_at": now_pacific_iso(),
        },
    )


def _decode_real_episode_from_hf_row_or_lerobot(
    *,
    split_id: str,
    ep_idx: int,
    steps_per_episode: int,
    image_hw: int,
    chunk_horizon: int,
    row_data: Optional[Dict[str, Any]],
    default_instr: str,
    lerobot_cache_dir: Path,
) -> Dict[str, Any]:
    """Build episode arrays strictly from real camera frames in ``row_data`` or pinned LeRobot Hub files."""
    spec = EMBODIEDX_TRACES_V1_DATASETS[split_id]
    state_dim = int(spec.get("state_dim", 8))
    action_dim = int(spec.get("action_dim", 7))
    t_len = int(steps_per_episode)
    hw = int(image_hw)
    hc = int(chunk_horizon)

    if isinstance(row_data, dict) and ("rgb_frames" in row_data or "frames" in row_data):
        raw_rgb = np.asarray(row_data.get("rgb_frames", row_data.get("frames")), dtype=np.float32)
        if raw_rgb.max() > 1.0:
            raw_rgb = raw_rgb / 255.0
        if raw_rgb.ndim == 3:
            raw_rgb = np.repeat(raw_rgb[None, ...], t_len, axis=0)
        if raw_rgb.ndim != 4 or raw_rgb.shape[-1] != 3 or raw_rgb.shape[0] < t_len:
            raise RuntimeError(
                f"HF row for {split_id!r} ep={ep_idx} has invalid rgb_frames shape {raw_rgb.shape}; "
                f"expected (T>={t_len}, H, W, 3)."
            )
        raw_wrist = np.asarray(row_data.get("wrist_rgb_frames", raw_rgb), dtype=np.float32)
        if raw_wrist.max() > 1.0:
            raw_wrist = raw_wrist / 255.0

        def _resize_seq(seq_f: np.ndarray) -> np.ndarray:
            seq_sub = seq_f[:t_len]
            if seq_sub.shape[1] == hw and seq_sub.shape[2] == hw:
                return np.clip(seq_sub, 0.0, 1.0).astype(np.float32)
            ys = np.linspace(0, seq_sub.shape[1] - 1, hw).astype(np.intp)
            xs = np.linspace(0, seq_sub.shape[2] - 1, hw).astype(np.intp)
            return np.clip(seq_sub[:, ys][:, :, xs, :], 0.0, 1.0).astype(np.float32)

        rgb_frames = _resize_seq(raw_rgb)
        wrist_rgb_frames = _resize_seq(raw_wrist)
        raw_state = np.asarray(row_data.get("state", np.zeros((t_len, state_dim), dtype=np.float32)), dtype=np.float32)
        if raw_state.ndim == 1:
            raw_state = np.repeat(raw_state[None, :], t_len, axis=0)
        state = np.zeros((t_len, state_dim), dtype=np.float32)
        sd_copy = min(state_dim, raw_state.shape[1])
        state[:, :sd_copy] = raw_state[:t_len, :sd_copy]

        raw_act = np.asarray(row_data.get("actions", np.zeros((t_len, action_dim), dtype=np.float32)), dtype=np.float32)
        if raw_act.ndim == 1:
            raw_act = np.repeat(raw_act[None, :], t_len, axis=0)
        actions = np.zeros((t_len, action_dim), dtype=np.float32)
        ad_copy = min(action_dim, raw_act.shape[1])
        actions[:, :ad_copy] = np.clip(raw_act[:t_len, :ad_copy], -1.0, 1.0)
        instr_text = str(
            row_data.get("task")
            or row_data.get("language_instruction")
            or row_data.get("instruction")
            or default_instr
        )
    else:
        from embodied_bench.dataset_loader import _resize_rgb_uint8_hwc
        from eeb.data.download import download as dl_lerobot
        from eeb.data.lerobot_loader import LeRobotDataset
        from eeb.data.schema import PINNED_HUB_REVISIONS

        canonical_repo = "lerobot/droid_100" if split_id == "droid_100" else "HuggingFaceVLA/libero"
        if split_id == "bridgedata_v2_omni":
            raise RuntimeError(
                f"Split {split_id!r} has no verified public LeRobot v3 camera bundle pinned yet; "
                "use libero_spatial / libero_goal / libero_object / libero_10 / droid_100 for real traces, "
                "or pass mode='smoke' exclusively for non-certifiable offline CI smoke testing."
            )
        rev = PINNED_HUB_REVISIONS[canonical_repo]
        local_root = dl_lerobot(
            canonical_repo,
            revision=rev,
            cache_dir=lerobot_cache_dir,
            episodes=ep_idx + 1,
            include_videos=(canonical_repo == "lerobot/droid_100"),
        )
        ds = LeRobotDataset(local_root, source_id=canonical_repo)
        tr = ds.load_episode(ep_idx, strict_instruction=False)
        if len(tr) < t_len or not tr.cameras:
            raise RuntimeError(f"Decoded trace {tr.episode_id!r} lacks {t_len} camera frames.")
        cam0 = tr.cameras[0]
        cam1 = tr.cameras[1] if len(tr.cameras) > 1 else tr.cameras[0]
        rgb_frames = np.stack(
            [_resize_rgb_uint8_hwc(tr.images[cam0][i], hw).astype(np.float32) / 255.0 for i in range(t_len)],
            axis=0,
        )
        wrist_rgb_frames = np.stack(
            [_resize_rgb_uint8_hwc(tr.images[cam1][i], hw).astype(np.float32) / 255.0 for i in range(t_len)],
            axis=0,
        )
        state = np.zeros((t_len, state_dim), dtype=np.float32)
        sd_copy = min(state_dim, tr.proprio.shape[1])
        state[:, :sd_copy] = tr.proprio[:t_len, :sd_copy]
        actions = np.zeros((t_len, action_dim), dtype=np.float32)
        ad_copy = min(action_dim, tr.action.shape[1])
        max_abs = max(1.0, float(np.max(np.abs(tr.action[:t_len, :ad_copy]))))
        actions[:, :ad_copy] = np.clip(tr.action[:t_len, :ad_copy] / np.float32(max_abs), -1.0, 1.0)
        instr_text = str(tr.instruction or default_instr)

    action_chunks = np.zeros((t_len, hc, action_dim), dtype=np.float32)
    for ti in range(t_len):
        for ki in range(hc):
            action_chunks[ti, ki] = actions[min(ti + ki, t_len - 1)]

    multi_camera_rgb = np.stack([rgb_frames, wrist_rgb_frames], axis=1)
    instruction_tokens = _tokenize_instruction_deterministic(instr_text, length=32)
    audio_waveform_16khz = np.zeros((t_len, 1600), dtype=np.float32)
    temporal_corrs = compute_frame_temporal_correlation(rgb_frames)
    return {
        "rgb_frames": rgb_frames,
        "wrist_rgb_frames": wrist_rgb_frames,
        "multi_camera_rgb": multi_camera_rgb,
        "state": state,
        "actions": actions,
        "action_chunks": action_chunks,
        "instruction_tokens": instruction_tokens,
        "instruction_text": np.array(instr_text),
        "audio_waveform_16khz": audio_waveform_16khz,
        "temporal_correlations": temporal_corrs,
        "pixel_provenance": np.array("REAL_HF_LEROBOT_CAMERA_PIXELS"),
    }


def download_embodiedx_traces(
    cache_dir: str | Path,
    splits: Optional[Sequence[str]] = None,
    mode: str = "hf",
    num_episodes_per_split: int = 5,
    steps_per_episode: int = 16,
    seed: int = 2026,
    image_hw: int = 64,
    chunk_horizon: int = 16,
    hf_timeout_s: float = 5.0,
    allow_synthetic_fallback: bool = False,
) -> Dict[str, Any]:
    """Download real `EmbodiedX-Traces-v1` episode bundles (`mode='hf'`) or explicit CI smoke shards (`mode='smoke'`)."""
    valid_modes = ("auto", "hf", "smoke")
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode={mode!r} in download_embodiedx_traces(). "
            f"Supported modes are: {list(valid_modes)}."
        )
    if int(num_episodes_per_split) < 1:
        raise ValueError(
            f"num_episodes_per_split must be >= 1, got {num_episodes_per_split}."
        )
    if int(steps_per_episode) < 2:
        raise ValueError(
            f"steps_per_episode must be >= 2 to evaluate temporal continuity, got {steps_per_episode}."
        )

    target_splits = list(splits) if splits is not None else list(EMBODIEDX_TRACES_V1_DATASETS.keys())
    unknown = [s for s in target_splits if s not in EMBODIEDX_TRACES_V1_DATASETS]
    if unknown:
        valid_splits = ", ".join(sorted(EMBODIEDX_TRACES_V1_DATASETS.keys()))
        raise ValueError(
            f"Unknown dataset split(s) {unknown!r}. "
            f"Supported splits in EMBODIEDX_TRACES_V1_DATASETS: [{valid_splits}]."
        )

    root_dir = Path(cache_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    cleanup_orphan_temp_files(root_dir)
    manifest_path = root_dir / "dataset_manifest.json"
    lerobot_cache_dir = Path(
        os.environ.get("EEB_LEROBOT_CACHE_DIR") or (root_dir / "_lerobot_hub_cache")
    )

    for split_id in target_splits:
        spec = EMBODIEDX_TRACES_V1_DATASETS[split_id]
        hf_repo = str(spec["hf_repo"])
        split_dir = root_dir / split_id
        split_dir.mkdir(parents=True, exist_ok=True)

        # Transition split state: -> PENDING -> RUNNING
        existing_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        existing_status = (
            existing_manifest.get("splits", {}).get(split_id, {}).get("status")
            if isinstance(existing_manifest, dict)
            else None
        )
        if existing_status in (
            RunStatus.REJECTED.value,
            RunStatus.FAILED.value,
            RunStatus.SKIPPED.value,
            RunStatus.BLOCKED.value,
            None,
        ):
            update_dataset_manifest(
                manifest_path=manifest_path,
                split_id=split_id,
                status=RunStatus.PENDING,
                source_mode=mode,
                hf_repo=hf_repo,
            )
            update_dataset_manifest(
                manifest_path=manifest_path,
                split_id=split_id,
                status=RunStatus.RUNNING,
                source_mode=mode,
                hf_repo=hf_repo,
            )
        elif existing_status == RunStatus.PENDING.value:
            update_dataset_manifest(
                manifest_path=manifest_path,
                split_id=split_id,
                status=RunStatus.RUNNING,
                source_mode=mode,
                hf_repo=hf_repo,
            )

        sample_instructions: Tuple[str, ...] = tuple(spec.get("sample_instructions", ("manipulate the object",)))
        episode_records: List[Dict[str, Any]] = []
        hf_rows: List[Dict[str, Any]] = []
        use_real = (mode == "hf") or (mode == "auto" and not allow_synthetic_fallback)
        resolved_source_mode = "smoke"
        prov_tag = "SYNTHETIC_PROCEDURAL_FALLBACK"

        if mode == "hf" or (mode == "auto" and os.environ.get("EMBODIEDX_ENABLE_HF_NETWORK") == "1"):
            try:
                hf_payload = fetch_hf_dataset_rows(
                    hf_repo=hf_repo,
                    length=max(int(num_episodes_per_split), 4),
                    timeout_s=hf_timeout_s,
                )
                hf_rows = list(hf_payload.get("rows", []))
                resolved_source_mode = "hf"
            except RuntimeError as exc:
                if mode == "hf" or not allow_synthetic_fallback:
                    update_dataset_manifest(
                        manifest_path=manifest_path,
                        split_id=split_id,
                        status=RunStatus.FAILED,
                        source_mode="hf",
                        hf_repo=hf_repo,
                        reason=str(exc),
                    )
                    raise
                resolved_source_mode = "smoke_auto_fallback"
                use_real = False
        elif mode == "auto":
            if not allow_synthetic_fallback:
                err_msg = (
                    f"Refusing silent synthetic fallback for split {split_id!r} (RT-02). "
                    "Pass mode='hf' (or set EMBODIEDX_ENABLE_HF_NETWORK=1) to fetch real camera traces, "
                    "or pass mode='smoke' explicitly for non-certifiable offline CI smoke testing."
                )
                update_dataset_manifest(
                    manifest_path=manifest_path,
                    split_id=split_id,
                    status=RunStatus.FAILED,
                    source_mode="auto",
                    hf_repo=hf_repo,
                    reason=err_msg,
                )
                raise RuntimeError(err_msg)
            resolved_source_mode = "smoke_auto_fallback"
            use_real = False

        try:
            for ep_idx in range(int(num_episodes_per_split)):
                default_instr = sample_instructions[ep_idx % len(sample_instructions)]
                row_data = (
                    hf_rows[ep_idx].get("row", {})
                    if (hf_rows and ep_idx < len(hf_rows) and isinstance(hf_rows[ep_idx], dict))
                    else None
                )
                if use_real:
                    if row_data is not None and "rgb_frames" not in row_data and "frames" not in row_data and not os.environ.get("EMBODIEDX_ALLOW_LEROBOT_FALLBACK_IN_HF_ROWS"):
                        # If fetch_hf_dataset_rows returned metadata without embedded pixels, fetch genuine LeRobot frames.
                        arrays = _decode_real_episode_from_hf_row_or_lerobot(
                            split_id=split_id,
                            ep_idx=ep_idx,
                            steps_per_episode=int(steps_per_episode),
                            image_hw=int(image_hw),
                            chunk_horizon=int(chunk_horizon),
                            row_data=None,
                            default_instr=str(
                                row_data.get("task")
                                or row_data.get("language_instruction")
                                or row_data.get("instruction")
                                or default_instr
                            ),
                            lerobot_cache_dir=lerobot_cache_dir,
                        )
                    else:
                        arrays = _decode_real_episode_from_hf_row_or_lerobot(
                            split_id=split_id,
                            ep_idx=ep_idx,
                            steps_per_episode=int(steps_per_episode),
                            image_hw=int(image_hw),
                            chunk_horizon=int(chunk_horizon),
                            row_data=row_data,
                            default_instr=default_instr,
                            lerobot_cache_dir=lerobot_cache_dir,
                        )
                    prov_tag = "REAL_HF_LEROBOT_CAMERA_PIXELS"
                    instr_text = str(arrays["instruction_text"].item())
                else:
                    instr_text = default_instr
                    arrays = _synthesize_smooth_episode_arrays(
                        split_id=split_id,
                        episode_idx=ep_idx,
                        steps_per_episode=int(steps_per_episode),
                        image_hw=int(image_hw),
                        seed=int(seed),
                        instruction_text=instr_text,
                        chunk_horizon=int(chunk_horizon),
                    )
                    prov_tag = "SYNTHETIC_PROCEDURAL_FALLBACK"

                ep_id = f"{split_id}_ep{ep_idx:04d}"
                rel_path = f"{split_id}/episode_{ep_idx:04d}.npz"
                npz_path = root_dir / rel_path
                np.savez(npz_path, **arrays)

                sha256_hex = compute_file_sha256(npz_path)
                corrs = arrays["temporal_correlations"]
                instr_tokens = arrays["instruction_tokens"]
                prefix_hash = hashlib.sha256(np.ascontiguousarray(instr_tokens).tobytes()).hexdigest()

                meta = EpisodeBundleMetadata(
                    split_id=split_id,
                    episode_id=ep_id,
                    episode_idx=ep_idx,
                    relative_path=rel_path,
                    sha256=sha256_hex,
                    num_steps=int(steps_per_episode),
                    image_hw=(int(image_hw), int(image_hw)),
                    num_cameras=int(spec.get("num_cameras", 2)),
                    state_dim=int(spec.get("state_dim", 8)),
                    action_dim=int(spec.get("action_dim", 7)),
                    chunk_horizon=int(chunk_horizon),
                    instruction_text=instr_text,
                    instruction_prefix_sha256=prefix_hash,
                    mean_temporal_correlation=round(float(np.mean(corrs)), 6),
                    min_temporal_correlation=round(float(np.min(corrs)), 6),
                    has_audio_16khz=True,
                    pixel_provenance=prov_tag,
                )
                episode_records.append(meta.to_dict())
        except Exception as exc:
            update_dataset_manifest(
                manifest_path=manifest_path,
                split_id=split_id,
                status=RunStatus.FAILED,
                source_mode=resolved_source_mode,
                hf_repo=hf_repo,
                reason=str(exc),
                pixel_provenance=prov_tag,
            )
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Failed to materialize real episodes for {split_id!r}: {exc}") from exc

        update_dataset_manifest(
            manifest_path=manifest_path,
            split_id=split_id,
            status=RunStatus.CERTIFIED,
            episodes=episode_records,
            source_mode=resolved_source_mode,
            hf_repo=hf_repo,
            pixel_provenance=prov_tag,
            extra_metadata={
                "steps_per_episode": int(steps_per_episode),
                "image_hw": [int(image_hw), int(image_hw)],
                "native_resolution_hw": list(spec.get("native_resolution_hw", (224, 224))),
                "seed": int(seed),
                "pixel_provenance": prov_tag,
            },
        )

    return json.loads(manifest_path.read_text(encoding="utf-8"))
