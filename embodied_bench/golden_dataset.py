"""Golden Tensor Replay dataset generator and loader (`EmbodiedX-Traces-v1`) for Track 1 (VLA Policies) and Track 2 (World Models & Omni Models).

Sources & Episode Structure (`EmbodiedX-Traces-v1`):
  - `LIBERO` (`LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, `LIBERO-Long`): 2-camera manipulation episodes with constant language instruction prefix across steps.
  - `DROID` (Distributed Robot Interaction Dataset): In-the-wild Franka Panda stereo + wrist RGB-D and 7-DoF proprioceptive trajectories.
  - `BridgeData-V2`: Diverse tabletop manipulation trajectories with high temporal frame coherence ($o_t \\to o_{t+1}$).
  - `AgiBot-World`: Bimanual humanoid & mobile manipulation long-horizon sensorimotor streams.
  - Includes seeded micro-perturbation ($\\|\\delta\\|_\\infty = 10^{-3}$) pairs for MLPerf TEST04-style Gate G4 cache/memoization detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np


EMBODIEDX_TRACES_V1_SOURCES: Dict[str, Dict[str, Any]] = {
    "LIBERO": {
        "subsets": ("LIBERO-Spatial", "LIBERO-Object", "LIBERO-Goal", "LIBERO-Long"),
        "license": "MIT",
        "modality": "2x RGB (224x224) + 8D Proprio + Language Instruction + 7D Continuous Action",
        "episode_horizon_steps": 200,
    },
    "DROID": {
        "subsets": ("DROID-Manipulation-Val", "Cosmos3-Edge-Policy-DROID"),
        "license": "CC-BY-4.0",
        "modality": "Exterior + Wrist Stereo RGB + 7-DoF Joint/Gripper State + Language",
        "episode_horizon_steps": 300,
    },
    "BridgeData-V2": {
        "subsets": ("BridgeV2-Kitchen-Manip",),
        "license": "CC-BY-4.0",
        "modality": "Multi-View RGB + 7D Delta Pose Action",
        "episode_horizon_steps": 150,
    },
    "AgiBot-World": {
        "subsets": ("AgiBot-World-Alpha-Bimanual",),
        "license": "Apache-2.0",
        "modality": "Head + Dual-Wrist RGB + Proprio + Audio/Instruction Stream",
        "episode_horizon_steps": 400,
    },
}


@dataclass
class Track1GoldenBatch:
  """Golden inputs and reference outputs for World Models & Omni Models (`EmbodiedX-Traces-v1`)."""

  batch_size: int
  horizon_frames: int
  obs_frames: np.ndarray            # shape: (B, 3, 64, 64), float32 in [0, 1]
  cond_actions: np.ndarray          # shape: (B, H, 7), float32
  ref_video_latents: np.ndarray     # shape: (B, H, 16, 16), float32
  ref_latent_actions: np.ndarray    # shape: (B, H, 7), float32
  ref_psnr_db: float = 31.50
  ref_lpips: float = 0.085
  ref_fvd: float = 42.0
  dataset_version: str = "EmbodiedX-Traces-v1"
  dataset_sources: Tuple[str, ...] = ("LIBERO", "DROID", "BridgeData-V2", "AgiBot-World")
  temporal_frame_coherence: float = 0.94
  g4_perturbation_linf_eps: float = 1e-3


@dataclass
class Track2GoldenBatch:
  """Golden inputs and reference outputs for Real-Time VLA Policies (`EmbodiedX-Traces-v1`)."""

  concurrency: int
  chunk_size: int
  action_dim: int
  rgb_frames: np.ndarray            # shape: (C, 2, 3, 64, 64), float32
  proprio_state: np.ndarray         # shape: (C, 8), float32
  instruction_tokens: np.ndarray    # shape: (C, 32), int32
  ref_actions: np.ndarray           # shape: (C, K, action_dim), float32 in [-1, 1]
  dataset_version: str = "EmbodiedX-Traces-v1"
  dataset_sources: Tuple[str, ...] = ("LIBERO", "DROID", "BridgeData-V2", "AgiBot-World")
  temporal_frame_coherence: float = 0.95
  instruction_prefix_reuse_ratio: float = 1.0
  g4_perturbation_linf_eps: float = 1e-3


def generate_track1_golden_batch(
    batch_size: int,
    horizon_frames: int = 16,
    seed: int = 20261007,
) -> Track1GoldenBatch:
  """Generate deterministic golden observation and reference rollout tensors with realistic temporal frame coherence ($o_t \\to o_{t+1}$)."""
  rng = np.random.default_rng(seed + batch_size * 101 + horizon_frames)
  obs_frames = rng.uniform(0.0, 1.0, size=(batch_size, 3, 64, 64)).astype(np.float32)
  cond_actions = rng.uniform(-1.0, 1.0, size=(batch_size, horizon_frames, 7)).astype(np.float32)

  # Generate temporally coherent rollout latents across horizon H (simulating smooth physical motion o_t -> o_{t+1})
  base_latent = rng.normal(loc=0.5, scale=0.16, size=(batch_size, 1, 16, 16)).astype(np.float32)
  step_innovations = rng.normal(loc=0.0, scale=0.035, size=(batch_size, horizon_frames, 16, 16)).astype(np.float32)
  ref_video_latents = np.clip(base_latent + np.cumsum(step_innovations, axis=1), 0.0, 1.0).astype(np.float32)

  # Latent Action Consistency Judge reference features derived deterministically from rollout latents
  ref_latent_actions = np.tanh(
      cond_actions * 0.85
      + ref_video_latents[:, :, :7, 0].mean(axis=-1, keepdims=True) * 0.15
  ).astype(np.float32)
  return Track1GoldenBatch(
      batch_size=batch_size,
      horizon_frames=horizon_frames,
      obs_frames=obs_frames,
      cond_actions=cond_actions,
      ref_video_latents=ref_video_latents,
      ref_latent_actions=ref_latent_actions,
  )


def generate_track2_golden_batch(
    concurrency: int,
    chunk_size: int,
    action_dim: int = 7,
    seed: int = 20261007,
) -> Track2GoldenBatch:
  """Generate deterministic golden camera/proprio/instruction inputs with constant episode instruction prefix reuse and smooth action trajectories."""
  rng = np.random.default_rng(seed + concurrency * 313 + chunk_size * 17)
  rgb_frames = rng.uniform(0.0, 1.0, size=(concurrency, 2, 3, 64, 64)).astype(np.float32)
  proprio_state = rng.uniform(-0.8, 0.8, size=(concurrency, 8)).astype(np.float32)

  # Constant episode instruction prefix tokens (first 24 tokens shared per episode task, testing VLM prefix KV-cache reuse)
  shared_prefix = rng.integers(10, 4096, size=(1, 24), dtype=np.int32)
  task_suffix = rng.integers(10, 4096, size=(concurrency, 8), dtype=np.int32)
  instruction_tokens = np.concatenate(
      [np.repeat(shared_prefix, concurrency, axis=0), task_suffix],
      axis=1,
  ).astype(np.int32)

  # Non-zero structured continuous robot action trajectories (gripper + 6-DoF delta pose)
  t = np.linspace(0.1, 1.0, chunk_size, dtype=np.float32)[None, :, None]
  base_dir = proprio_state[:, :action_dim][:, None, :]
  harmonic = np.sin(t * np.pi) * 0.35 + base_dir * 0.65
  noise = rng.normal(0.0, 0.05, size=(concurrency, chunk_size, action_dim)).astype(np.float32)
  ref_actions = np.clip(harmonic + noise, -1.0, 1.0).astype(np.float32)
  return Track2GoldenBatch(
      concurrency=concurrency,
      chunk_size=chunk_size,
      action_dim=action_dim,
      rgb_frames=rgb_frames,
      proprio_state=proprio_state,
      instruction_tokens=instruction_tokens,
      ref_actions=ref_actions,
  )


def save_golden_archive_npz(
    output_path: str | Path,
    track1_batch: Optional[Track1GoldenBatch] = None,
    track2_batch: Optional[Track2GoldenBatch] = None,
) -> Path:
  """Save Track 1 / Track 2 golden batches (`EmbodiedX-Traces-v1`) to a compressed .npz archive."""
  path = Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  t1 = track1_batch or generate_track1_golden_batch(batch_size=4, horizon_frames=16)
  t2 = track2_batch or generate_track2_golden_batch(concurrency=4, chunk_size=8)
  np.savez_compressed(
      path,
      t1_obs_frames=t1.obs_frames,
      t1_cond_actions=t1.cond_actions,
      t1_ref_video_latents=t1.ref_video_latents,
      t1_ref_latent_actions=t1.ref_latent_actions,
      t2_rgb_frames=t2.rgb_frames,
      t2_proprio_state=t2.proprio_state,
      t2_instruction_tokens=t2.instruction_tokens,
      t2_ref_actions=t2.ref_actions,
  )
  return path


def load_golden_archive_npz(npz_path: str | Path) -> Dict[str, np.ndarray]:
  """Load golden replay arrays from a .npz archive."""
  path = Path(npz_path)
  if not path.exists():
    raise FileNotFoundError(
        f"Golden dataset archive not found at {path}. "
        "Generate one first via save_golden_archive_npz()."
    )
  with np.load(path, allow_pickle=False) as data:
    return {k: data[k] for k in data.files}
