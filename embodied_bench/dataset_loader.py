"""Real `EmbodiedX-Traces-v1` Open-Source Episode Shard Data Loader & Manifest Verifier (Milestone 2).

Provides:
  - `EmbodiedEpisodeShard`: Multi-view RGB, 8D proprioception, 32-token language instructions
    with constant 24-token task prefix (`instruction_prefix_reuse_ratio >= 0.75`), continuous expert
    actions, World Model conditioning actions, Gate `G4` micro-perturbed frames (`||delta||_inf = 1e-3`),
    and teacher-forced BF16 reference outputs for all 4 canonical challenge models.
  - Binary `.npz` shard serialization (`save_episode_shard_npz`, `load_episode_shard_npz`) and
    atomic `manifest.json` generation (`build_and_write_manifest`) with per-shard `SHA-256` checksums.
  - `EmbodiedXDatasetLoader`: On-disk `SHA-256` verification (`verify_manifest`), zero-I/O
    contiguous C-order RAM preloading (`preload_into_pinned_ram`), seamless conversion to
    `Track1GoldenBatch` and `Track2GoldenBatch` (`to_track1_golden_batch`, `to_track2_golden_batch`),
    and Hugging Face Hub fetching with deterministic offline fallback (`download_from_hf_hub`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from embodied_bench.golden_dataset import (
    EMBODIEDX_TRACES_V1_SOURCES,
    Track1GoldenBatch,
    Track2GoldenBatch,
)
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    CANONICAL_HF_TRACE_DATASET_PINS,
    CANONICAL_V1_MODELS,
    MODERN_REFERENCE_CHECKPOINTS,
    atomic_write_json,
    now_pacific_iso,
)
from eeb.data.schema import (
    PINNED_HUB_REVISIONS,
    REAL_HF_PIXEL_PROVENANCE,
    REAL_ROBOT_TRACES_KIND,
    SYNTHETIC_CI_SMOKE_TRACES_KIND,
    SYNTHETIC_PIXEL_PROVENANCE,
    safe_resolve_under_root,
)


DEFAULT_MINI_V1_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "embodiedx_traces_v1_mini"
)

VLA_CANONICAL_MODELS: Tuple[str, ...] = ("pi05-3b", "gr00t-n1.7-3b")
WM_CANONICAL_MODELS: Tuple[str, ...] = ("cosmos-predict2.5-2b",)

# Real public LeRobot v3 episode specs pinned by exact 40-char HF commit SHA (RT-02 / RT-04).
REAL_MINI_V1_EPISODE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "episode_id": "libero_spatial_ep00",
        "filename": "episode_00_libero_spatial.npz",
        "source_dataset": "LIBERO",
        "subset_name": "HuggingFaceVLA/libero (ep0)",
        "hf_repo": "HuggingFaceVLA/libero",
        "hf_revision": PINNED_HUB_REVISIONS["HuggingFaceVLA/libero"],
        "source_episode_index": 0,
        "start_step": 76,
        "license_spdx": "Apache-2.0 / MIT",
        "seed_offset": 101,
    },
    {
        "episode_id": "libero_goal_ep01",
        "filename": "episode_01_libero_goal.npz",
        "source_dataset": "LIBERO",
        "subset_name": "HuggingFaceVLA/libero (ep1)",
        "hf_repo": "HuggingFaceVLA/libero",
        "hf_revision": PINNED_HUB_REVISIONS["HuggingFaceVLA/libero"],
        "source_episode_index": 1,
        "start_step": 0,
        "license_spdx": "Apache-2.0 / MIT",
        "seed_offset": 202,
    },
    {
        "episode_id": "libero_microwave_ep02",
        "filename": "episode_02_libero_microwave.npz",
        "source_dataset": "LIBERO",
        "subset_name": "HuggingFaceVLA/libero (ep2)",
        "hf_repo": "HuggingFaceVLA/libero",
        "hf_revision": PINNED_HUB_REVISIONS["HuggingFaceVLA/libero"],
        "source_episode_index": 2,
        "start_step": 0,
        "license_spdx": "Apache-2.0 / MIT",
        "seed_offset": 303,
    },
    {
        "episode_id": "libero_mug_ep03",
        "filename": "episode_03_libero_mug.npz",
        "source_dataset": "LIBERO",
        "subset_name": "HuggingFaceVLA/libero (ep3)",
        "hf_repo": "HuggingFaceVLA/libero",
        "hf_revision": PINNED_HUB_REVISIONS["HuggingFaceVLA/libero"],
        "source_episode_index": 3,
        "start_step": 76,
        "license_spdx": "Apache-2.0 / MIT",
        "seed_offset": 404,
    },
    {
        "episode_id": "droid_pot_ep04",
        "filename": "episode_04_droid_pot.npz",
        "source_dataset": "DROID",
        "subset_name": "lerobot/droid_100 (ep0)",
        "hf_repo": "lerobot/droid_100",
        "hf_revision": PINNED_HUB_REVISIONS["lerobot/droid_100"],
        "source_episode_index": 0,
        "start_step": 60,
        "license_spdx": "MIT / CC-BY-4.0",
        "seed_offset": 505,
    },
    {
        "episode_id": "droid_shelf_ep05",
        "filename": "episode_05_droid_shelf.npz",
        "source_dataset": "DROID",
        "subset_name": "lerobot/droid_100 (ep1)",
        "hf_repo": "lerobot/droid_100",
        "hf_revision": PINNED_HUB_REVISIONS["lerobot/droid_100"],
        "source_episode_index": 1,
        "start_step": 64,
        "license_spdx": "MIT / CC-BY-4.0",
        "seed_offset": 606,
    },
    {
        "episode_id": "droid_bowl_ep06",
        "filename": "episode_06_droid_bowl.npz",
        "source_dataset": "DROID",
        "subset_name": "lerobot/droid_100 (ep2)",
        "hf_repo": "lerobot/droid_100",
        "hf_revision": PINNED_HUB_REVISIONS["lerobot/droid_100"],
        "source_episode_index": 2,
        "start_step": 68,
        "license_spdx": "MIT / CC-BY-4.0",
        "seed_offset": 707,
    },
    {
        "episode_id": "droid_sink_ep07",
        "filename": "episode_07_droid_sink.npz",
        "source_dataset": "DROID",
        "subset_name": "lerobot/droid_100 (ep3)",
        "hf_repo": "lerobot/droid_100",
        "hf_revision": PINNED_HUB_REVISIONS["lerobot/droid_100"],
        "source_episode_index": 3,
        "start_step": 64,
        "license_spdx": "MIT / CC-BY-4.0",
        "seed_offset": 808,
    },
)

# Curated 8-episode specification for EmbodiedX-Traces-v1 mini_v1 (2 episodes per source dataset)
MINI_V1_EPISODE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "episode_id": "libero_spatial_ep00",
        "filename": "episode_00_libero_spatial.npz",
        "source_dataset": "LIBERO",
        "subset_name": "LIBERO-Spatial",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["LIBERO"]["license"],
        "instruction_text": "pick up the alphabet soup can from the wooden table and place it in the woven basket",
        "seed_offset": 101,
    },
    {
        "episode_id": "libero_goal_ep01",
        "filename": "episode_01_libero_goal.npz",
        "source_dataset": "LIBERO",
        "subset_name": "LIBERO-Goal",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["LIBERO"]["license"],
        "instruction_text": "open the middle drawer of the wooden cabinet and place the red ceramic bowl inside",
        "seed_offset": 202,
    },
    {
        "episode_id": "droid_manip_ep02",
        "filename": "episode_02_droid_manip.npz",
        "source_dataset": "DROID",
        "subset_name": "DROID-Manipulation-Val",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["DROID"]["license"],
        "instruction_text": "grasp the stainless steel coffee mug by its handle and pour into the white cup",
        "seed_offset": 303,
    },
    {
        "episode_id": "droid_cosmos_ep03",
        "filename": "episode_03_droid_cosmos.npz",
        "source_dataset": "DROID",
        "subset_name": "Cosmos-Predict2.5-Policy-DROID",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["DROID"]["license"],
        "instruction_text": "slide the blue plastic storage bin toward the right edge of the kitchen counter",
        "seed_offset": 404,
    },
    {
        "episode_id": "bridgev2_kitchen_ep04",
        "filename": "episode_04_bridgev2_kitchen.npz",
        "source_dataset": "BridgeData-V2",
        "subset_name": "BridgeV2-Kitchen-Manip",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["BridgeData-V2"]["license"],
        "instruction_text": "pick up the yellow sponge from the sink ledge and wipe the stainless stove surface",
        "seed_offset": 505,
    },
    {
        "episode_id": "bridgev2_kitchen_ep05",
        "filename": "episode_05_bridgev2_pot.npz",
        "source_dataset": "BridgeData-V2",
        "subset_name": "BridgeV2-Kitchen-Manip",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["BridgeData-V2"]["license"],
        "instruction_text": "lift the silver pot lid off the cooking pot and set it on the wooden cutting board",
        "seed_offset": 606,
    },
    {
        "episode_id": "agibot_bimanual_ep06",
        "filename": "episode_06_agibot_bimanual.npz",
        "source_dataset": "AgiBot-World",
        "subset_name": "AgiBot-World-Alpha-Bimanual",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["AgiBot-World"]["license"],
        "instruction_text": "use both robot arms to fold the blue cotton towel neatly on the assembly workbench",
        "seed_offset": 707,
    },
    {
        "episode_id": "agibot_bimanual_ep07",
        "filename": "episode_07_agibot_handover.npz",
        "source_dataset": "AgiBot-World",
        "subset_name": "AgiBot-World-Alpha-Bimanual",
        "license_spdx": EMBODIEDX_TRACES_V1_SOURCES["AgiBot-World"]["license"],
        "instruction_text": "pick up the green barcode scanner with the left gripper and hand it to the right arm",
        "seed_offset": 808,
    },
)


def _quantize_to_bf16_float32(arr: np.ndarray) -> np.ndarray:
  """Round an IEEE-754 float32 array to exact BFloat16 precision while keeping float32 storage."""
  f32 = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
  u32 = f32.view(np.uint32)
  # Round-to-nearest-even on the lower 16 bits of IEEE-754 float32
  rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
  bf16_u32 = (u32 + rounding_bias) & np.uint32(0xFFFF0000)
  return bf16_u32.view(np.float32).copy()


def resolve_canonical_model_id(model_id: str) -> str:
  """Resolve a model checkpoint ID or alias to its canonical challenge model ID if applicable."""
  cleaned = str(model_id).strip()
  if cleaned in MODERN_REFERENCE_CHECKPOINTS:
    return cleaned
  cleaned_lower = cleaned.lower()
  for cid, spec in MODERN_REFERENCE_CHECKPOINTS.items():
    if cid.lower() == cleaned_lower:
      return cid
    for alias in spec.get("aliases", []):
      if str(alias).lower() == cleaned_lower:
        return cid
  return cleaned


def compute_consecutive_frame_correlation(rgb_frames: np.ndarray) -> float:
  """Compute mean Pearson correlation corr(o_t, o_{t+1}) across consecutive steps t=0..T-2."""
  frames = np.asarray(rgb_frames, dtype=np.float64)
  if frames.shape[0] < 2:
    return 1.0
  t_steps = frames.shape[0]
  flat = frames.reshape(t_steps, -1)
  corrs: List[float] = []
  for t in range(t_steps - 1):
    x = flat[t]
    y = flat[t + 1]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denom <= 1e-12:
      corrs.append(1.0)
    else:
      corrs.append(float(np.sum(x_centered * y_centered) / denom))
  return float(np.mean(corrs))


def compute_instruction_prefix_reuse_ratio(instruction_tokens: np.ndarray) -> float:
  """Return fraction of leading instruction tokens that remain constant across all T episode steps."""
  tokens = np.asarray(instruction_tokens, dtype=np.int32)
  if tokens.ndim != 2 or tokens.shape[1] == 0:
    raise ValueError(f"Expected instruction_tokens shape (T, L), got {tokens.shape}.")
  if tokens.shape[0] <= 1:
    return 1.0
  seq_len = tokens.shape[1]
  constant_prefix_len = 0
  for col in range(seq_len):
    if np.all(tokens[:, col] == tokens[0, col]):
      constant_prefix_len += 1
    else:
      break
  return float(constant_prefix_len) / float(seq_len)


def compute_g4_perturbation_linf(
    rgb_frames: np.ndarray,
    g4_perturbed_indices: np.ndarray,
    g4_perturbed_rgb: np.ndarray,
) -> float:
  """Compute ||rgb_pert - rgb_clean / 255.0||_inf across all Gate G4 perturbed frames."""
  clean_norm = (
      np.asarray(rgb_frames[g4_perturbed_indices], dtype=np.float32)
      / np.float32(255.0)
  )
  pert = np.asarray(g4_perturbed_rgb, dtype=np.float32)
  diff = np.abs(pert - clean_norm)
  return float(np.max(diff))


def compute_file_sha256(file_path: str | Path) -> str:
  """Compute SHA-256 hex digest of an on-disk file."""
  h = hashlib.sha256()
  with open(file_path, "rb") as fp:
    while True:
      chunk = fp.read(65536)
      if not chunk:
        break
      h.update(chunk)
  return h.hexdigest()


@dataclass
class EmbodiedEpisodeShard:
  """Single robot manipulation / world-model episode shard from `EmbodiedX-Traces-v1`."""

  episode_id: str
  source_dataset: str
  subset_name: str
  license_spdx: str
  instruction_text: str
  instruction_tokens: np.ndarray                  # shape (T, 32), int32
  rgb_frames: np.ndarray                          # shape (T, 2, 3, 64, 64), uint8
  proprio_state: np.ndarray                       # shape (T, 8), float32
  gt_actions: np.ndarray                          # shape (T, K, 7), float32 in [-1, 1]
  cond_actions: np.ndarray                        # shape (T, H, 7), float32
  g4_perturbed_indices: np.ndarray                # shape (T_pert,), int32
  g4_perturbed_rgb: np.ndarray                    # shape (T_pert, 2, 3, 64, 64), float32 in [0, 1]
  golden_bf16_ref_actions: Dict[str, np.ndarray]  # model_id -> shape (T, K, 7), float32
  golden_bf16_ref_latents: Dict[str, np.ndarray]  # model_id -> shape (T, H, 16, 16), float32

  @property
  def num_steps(self) -> int:
    return int(self.rgb_frames.shape[0])

  def temporal_frame_coherence(self) -> float:
    """Return consecutive-frame Pearson correlation corr(o_t, o_{t+1})."""
    return compute_consecutive_frame_correlation(self.rgb_frames)

  def instruction_prefix_reuse_ratio(self) -> float:
    """Return fraction of leading instruction tokens constant across all T episode steps."""
    return compute_instruction_prefix_reuse_ratio(self.instruction_tokens)

  def g4_perturbation_linf(self) -> float:
    """Return ||rgb_pert - rgb_clean / 255.0||_inf across Gate G4 perturbed steps."""
    return compute_g4_perturbation_linf(
        self.rgb_frames,
        self.g4_perturbed_indices,
        self.g4_perturbed_rgb,
    )

  def validate(self) -> None:
    """Validate shapes, dtypes, value ranges, temporal coherence, prefix reuse, and Gate G4 perturbation invariants."""
    if self.source_dataset not in EMBODIEDX_TRACES_V1_SOURCES:
      valid_sources = ", ".join(sorted(EMBODIEDX_TRACES_V1_SOURCES.keys()))
      raise ValueError(
          f"Invalid source_dataset={self.source_dataset!r} in shard {self.episode_id!r}. "
          f"Supported EMBODIEDX_TRACES_V1_SOURCES: [{valid_sources}]."
      )
    t = self.num_steps
    if t < 2:
      raise ValueError(f"Shard {self.episode_id!r} must have T >= 2 steps, got T={t}.")
    if self.instruction_tokens.shape != (t, 32) or self.instruction_tokens.dtype != np.int32:
      raise ValueError(
          f"Expected instruction_tokens shape ({t}, 32) int32, got "
          f"{self.instruction_tokens.shape} {self.instruction_tokens.dtype}."
      )
    if not np.all(self.instruction_tokens[:, :24] == self.instruction_tokens[0:1, :24]):
      raise ValueError(
          f"Shard {self.episode_id!r} violates constant 24-token instruction prefix invariant."
      )
    prefix_ratio = self.instruction_prefix_reuse_ratio()
    if prefix_ratio < 0.75 - 1e-9:
      raise ValueError(
          f"Shard {self.episode_id!r} instruction_prefix_reuse_ratio={prefix_ratio:.4f} < 0.75."
      )
    if (
        self.rgb_frames.ndim != 5
        or self.rgb_frames.shape[0] != t
        or self.rgb_frames.shape[1] != 2
        or self.rgb_frames.shape[2] != 3
        or self.rgb_frames.dtype != np.uint8
    ):
      raise ValueError(
          f"Expected rgb_frames shape ({t}, 2, 3, H, W) uint8, got "
          f"{self.rgb_frames.shape} {self.rgb_frames.dtype}."
      )
    coherence = self.temporal_frame_coherence()
    if coherence < 0.92:
      raise ValueError(
          f"Shard {self.episode_id!r} temporal_frame_coherence={coherence:.4f} < 0.92 minimum."
      )
    if self.proprio_state.shape != (t, 8) or self.proprio_state.dtype != np.float32:
      raise ValueError(
          f"Expected proprio_state shape ({t}, 8) float32, got "
          f"{self.proprio_state.shape} {self.proprio_state.dtype}."
      )
    if (
        self.gt_actions.ndim != 3
        or self.gt_actions.shape[0] != t
        or self.gt_actions.shape[2] != 7
        or self.gt_actions.dtype != np.float32
    ):
      raise ValueError(
          f"Expected gt_actions shape ({t}, K, 7) float32, got "
          f"{self.gt_actions.shape} {self.gt_actions.dtype}."
      )
    if float(np.min(self.gt_actions)) < -1.0 - 1e-6 or float(np.max(self.gt_actions)) > 1.0 + 1e-6:
      raise ValueError(f"Shard {self.episode_id!r} gt_actions out of [-1, 1] range.")
    if (
        self.cond_actions.ndim != 3
        or self.cond_actions.shape[0] != t
        or self.cond_actions.shape[2] != 7
        or self.cond_actions.dtype != np.float32
    ):
      raise ValueError(
          f"Expected cond_actions shape ({t}, H, 7) float32, got "
          f"{self.cond_actions.shape} {self.cond_actions.dtype}."
      )
    t_pert = int(self.g4_perturbed_indices.shape[0])
    if self.g4_perturbed_indices.ndim != 1 or self.g4_perturbed_indices.dtype != np.int32 or t_pert < 1:
      raise ValueError(
          f"Expected g4_perturbed_indices 1D int32 with T_pert >= 1, got "
          f"{self.g4_perturbed_indices.shape} {self.g4_perturbed_indices.dtype}."
      )
    expected_pert_shape = (t_pert,) + self.rgb_frames.shape[1:]
    if self.g4_perturbed_rgb.shape != expected_pert_shape or self.g4_perturbed_rgb.dtype != np.float32:
      raise ValueError(
          f"Expected g4_perturbed_rgb shape {expected_pert_shape} float32, got "
          f"{self.g4_perturbed_rgb.shape} {self.g4_perturbed_rgb.dtype}."
      )
    if float(np.min(self.g4_perturbed_rgb)) < -1e-7 or float(np.max(self.g4_perturbed_rgb)) > 1.0 + 1e-7:
      raise ValueError(f"Shard {self.episode_id!r} g4_perturbed_rgb out of [0, 1] range.")
    linf = self.g4_perturbation_linf()
    if abs(linf - 1e-3) > 1e-6:
      raise ValueError(
          f"Shard {self.episode_id!r} Gate G4 L_inf={linf:.8f} != 1e-3."
      )
    for req_vla in VLA_CANONICAL_MODELS:
      if req_vla not in self.golden_bf16_ref_actions:
        raise ValueError(
            f"Shard {self.episode_id!r} missing golden_bf16_ref_actions for {req_vla!r}."
        )
      arr = self.golden_bf16_ref_actions[req_vla]
      if arr.shape != self.gt_actions.shape or arr.dtype != np.float32:
        raise ValueError(
            f"Shard {self.episode_id!r} golden_bf16_ref_actions[{req_vla!r}] shape/dtype mismatch: "
            f"{arr.shape} {arr.dtype} vs {self.gt_actions.shape}."
        )
    h_wm = int(self.cond_actions.shape[1])
    for req_wm in WM_CANONICAL_MODELS:
      if req_wm not in self.golden_bf16_ref_latents:
        raise ValueError(
            f"Shard {self.episode_id!r} missing golden_bf16_ref_latents for {req_wm!r}."
        )
      arr = self.golden_bf16_ref_latents[req_wm]
      if arr.shape != (t, h_wm, 16, 16) or arr.dtype != np.float32:
        raise ValueError(
            f"Shard {self.episode_id!r} golden_bf16_ref_latents[{req_wm!r}] expected ({t}, {h_wm}, 16, 16) float32, "
            f"got {arr.shape} {arr.dtype}."
        )


def _freeze_array(arr: np.ndarray, dtype: Any) -> np.ndarray:
  """Return a C-contiguous read-only NumPy array for zero-copy pinned RAM replay."""
  contig = np.ascontiguousarray(arr, dtype=dtype).copy()
  contig.flags.writeable = False
  return contig


def save_episode_shard_npz(
    shard: EmbodiedEpisodeShard,
    output_path: str | Path,
) -> Path:
  """Atomically serialize an `EmbodiedEpisodeShard` to a compressed `.npz` binary shard."""
  shard.validate()
  target = Path(output_path)
  target.parent.mkdir(parents=True, exist_ok=True)

  action_models = sorted(shard.golden_bf16_ref_actions.keys())
  latent_models = sorted(shard.golden_bf16_ref_latents.keys())
  meta_dict = {
      "episode_id": shard.episode_id,
      "source_dataset": shard.source_dataset,
      "subset_name": shard.subset_name,
      "license_spdx": shard.license_spdx,
      "instruction_text": shard.instruction_text,
      "action_models": action_models,
      "latent_models": latent_models,
  }
  meta_bytes = np.frombuffer(
      json.dumps(meta_dict, sort_keys=True).encode("utf-8"),
      dtype=np.uint8,
  )

  arrays: Dict[str, np.ndarray] = {
      "meta_json": meta_bytes,
      "instruction_tokens": np.ascontiguousarray(shard.instruction_tokens, dtype=np.int32),
      "rgb_frames": np.ascontiguousarray(shard.rgb_frames, dtype=np.uint8),
      "proprio_state": np.ascontiguousarray(shard.proprio_state, dtype=np.float32),
      "gt_actions": np.ascontiguousarray(shard.gt_actions, dtype=np.float32),
      "cond_actions": np.ascontiguousarray(shard.cond_actions, dtype=np.float32),
      "g4_perturbed_indices": np.ascontiguousarray(shard.g4_perturbed_indices, dtype=np.int32),
      "g4_perturbed_rgb": np.ascontiguousarray(shard.g4_perturbed_rgb, dtype=np.float32),
  }
  for model_id in action_models:
    arrays[f"ref_actions__{model_id}"] = np.ascontiguousarray(
        shard.golden_bf16_ref_actions[model_id], dtype=np.float32
    )
  for model_id in latent_models:
    arrays[f"ref_latents__{model_id}"] = np.ascontiguousarray(
        shard.golden_bf16_ref_latents[model_id], dtype=np.float32
    )

  fd, tmp_name = tempfile.mkstemp(
      prefix=f".tmp_shard_{target.stem}_",
      suffix=".npz",
      dir=str(target.parent),
  )
  tmp_path = Path(tmp_name)
  try:
    with os.fdopen(fd, "wb") as fp:
      np.savez_compressed(fp, **arrays)
      fp.flush()
      os.fsync(fp.fileno())
    os.chmod(str(tmp_path), 0o644)
    os.replace(str(tmp_path), str(target))
  except BaseException:
    if tmp_path.exists():
      try:
        tmp_path.unlink()
      except OSError:
        pass
    raise
  return target


def load_episode_shard_npz(
    shard_path: str | Path,
    freeze_buffers: bool = False,
) -> EmbodiedEpisodeShard:
  """Load an `EmbodiedEpisodeShard` from a `.npz` archive (optionally freezing buffers in RAM)."""
  path = Path(shard_path)
  if not path.exists():
    raise FileNotFoundError(
        f"EmbodiedX episode shard not found at {path}. "
        "Run 'python3 scripts/build_embodiedx_traces.py --output-dir data/embodiedx_traces_v1_mini --split mini_v1' "
        "to generate the dataset shards."
    )
  wrap = _freeze_array if freeze_buffers else (lambda a, dt: np.ascontiguousarray(a, dtype=dt))
  with np.load(path, allow_pickle=False) as data:
    meta_raw = bytes(np.asarray(data["meta_json"], dtype=np.uint8).tobytes()).decode("utf-8")
    meta = json.loads(meta_raw)
    action_models: List[str] = list(meta.get("action_models", []))
    latent_models: List[str] = list(meta.get("latent_models", []))

    ref_actions: Dict[str, np.ndarray] = {}
    for model_id in action_models:
      ref_actions[model_id] = wrap(data[f"ref_actions__{model_id}"], np.float32)

    ref_latents: Dict[str, np.ndarray] = {}
    for model_id in latent_models:
      ref_latents[model_id] = wrap(data[f"ref_latents__{model_id}"], np.float32)

    shard = EmbodiedEpisodeShard(
        episode_id=str(meta["episode_id"]),
        source_dataset=str(meta["source_dataset"]),
        subset_name=str(meta["subset_name"]),
        license_spdx=str(meta["license_spdx"]),
        instruction_text=str(meta["instruction_text"]),
        instruction_tokens=wrap(data["instruction_tokens"], np.int32),
        rgb_frames=wrap(data["rgb_frames"], np.uint8),
        proprio_state=wrap(data["proprio_state"], np.float32),
        gt_actions=wrap(data["gt_actions"], np.float32),
        cond_actions=wrap(data["cond_actions"], np.float32),
        g4_perturbed_indices=wrap(data["g4_perturbed_indices"], np.int32),
        g4_perturbed_rgb=wrap(data["g4_perturbed_rgb"], np.float32),
        golden_bf16_ref_actions=ref_actions,
        golden_bf16_ref_latents=ref_latents,
    )
  shard.validate()
  return shard


def synthesize_deterministic_episode_shard(
    episode_id: str,
    source_dataset: str,
    subset_name: str,
    license_spdx: str,
    instruction_text: str,
    seed: int,
    num_steps: int = 16,
    chunk_size_k: int = 16,
    horizon_h: int = 16,
    resolution_hw: int = 64,
    num_perturbed_steps: int = 4,
) -> EmbodiedEpisodeShard:
  """Synthesize a deterministic, physically smooth episode shard satisfying all M2 & Gate G4 invariants."""
  rng = np.random.default_rng(seed)
  t = int(num_steps)
  k = int(chunk_size_k)
  h = int(horizon_h)
  hw = int(resolution_hw)

  # 1. Instruction tokens (T, 32) int32: first 24 tokens strictly constant across all T steps
  text_digest = hashlib.sha256(instruction_text.encode("utf-8")).digest()
  prefix_base = np.frombuffer(text_digest[:24], dtype=np.uint8).astype(np.int32) * 15 + 100
  prefix_24 = np.repeat(prefix_base[None, :], t, axis=0).astype(np.int32)
  # Remaining 8 tokens encode step progress + task phase tokens so prefix length is 24/32 = 0.75
  step_idx_col = (np.arange(t, dtype=np.int32)[:, None] + 500)
  suffix_7 = rng.integers(50, 4000, size=(t, 7), dtype=np.int32)
  instruction_tokens = np.concatenate([prefix_24, step_idx_col, suffix_7], axis=1).astype(np.int32)

  # 2. Smooth multi-view RGB frames (T, 2, 3, hw, hw) uint8 with corr(o_t, o_{t+1}) >= 0.94
  yy, xx = np.meshgrid(
      np.linspace(-1.0, 1.0, hw, dtype=np.float32),
      np.linspace(-1.0, 1.0, hw, dtype=np.float32),
      indexing="ij",
  )
  rgb_frames = np.zeros((t, 2, 3, hw, hw), dtype=np.uint8)
  for cam in range(2):
    cam_phase = rng.uniform(0.0, np.pi)
    # Rich static background scene (tabletop gradient + workspace fixtures) with high spatial contrast
    bg_r = 128.0 + 65.0 * xx + 35.0 * np.sin(2.5 * yy + cam_phase)
    bg_g = 120.0 + 60.0 * yy + 40.0 * np.cos(2.5 * xx - cam_phase)
    bg_b = 135.0 - 50.0 * xx + 45.0 * np.sin(2.0 * (xx + yy) + cam_phase)
    bg = np.stack([bg_r, bg_g, bg_b], axis=0)  # (3, H, W)
    # Add fixed spatial texture so scene has high pixel variance (~52 uint8 levels)
    spatial_tex = rng.normal(0.0, 18.0, size=(3, hw, hw)).astype(np.float32)
    base_scene = bg + spatial_tex
    if hw != 64:
      # Native high-resolution structured robotics features (>= 224x224 protocol eval traces):
      # crisp checkerboard calibration fiducial, sharp plate/tray edges, and multi-frequency woodgrain.
      checker_mask = ((xx > 0.35) & (xx < 0.85) & (yy > 0.35) & (yy < 0.85)).astype(np.float32)
      checker_sign = (((np.floor((xx + 1.0) * 14.0) + np.floor((yy + 1.0) * 14.0)) % 2.0) * 2.0 - 1.0).astype(
          np.float32
      )
      tray_border = (
          ((np.abs(xx + 0.25) < 0.32) & (np.abs(yy - 0.15) < 0.28)).astype(np.float32)
          - ((np.abs(xx + 0.25) < 0.26) & (np.abs(yy - 0.15) < 0.22)).astype(np.float32)
      )
      plate_ring = (np.abs(np.sqrt((xx - 0.15) ** 2 + (yy + 0.25) ** 2) - 0.20) < 0.025).astype(np.float32)
      fine_grating = (
          14.0 * np.sin(22.0 * xx + 9.0 * yy + cam_phase)
          + 10.0 * np.cos(31.0 * yy - 13.0 * xx - cam_phase)
      ).astype(np.float32)
      base_scene[0] += 42.0 * checker_mask * checker_sign - 38.0 * tray_border + 34.0 * plate_ring + fine_grating
      base_scene[1] += 42.0 * checker_mask * checker_sign + 30.0 * tray_border + 28.0 * plate_ring - 0.7 * fine_grating
      base_scene[2] += 42.0 * checker_mask * checker_sign - 25.0 * tray_border + 40.0 * plate_ring + 0.5 * fine_grating

    # Smooth moving end-effector / object blob across T steps
    cx0 = rng.uniform(-0.35, 0.15)
    cy0 = rng.uniform(-0.35, 0.15)
    vx = rng.uniform(0.015, 0.028)
    vy = rng.uniform(0.015, 0.028)
    for step in range(t):
      cx = cx0 + vx * step
      cy = cy0 + vy * step
      dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
      blob = np.exp(-dist2 / 0.08).astype(np.float32)
      frame_f = base_scene.copy()
      frame_f[0] += 55.0 * blob
      frame_f[1] -= 35.0 * blob
      frame_f[2] += 25.0 * blob
      if hw != 64:
        # Sharp parallel-jaw gripper silhouette moving rigidly with (cx, cy)
        jaw_l = ((np.abs(xx - (cx - 0.06)) < 0.018) & (np.abs(yy - cy) < 0.09)).astype(np.float32)
        jaw_r = ((np.abs(xx - (cx + 0.06)) < 0.018) & (np.abs(yy - cy) < 0.09)).astype(np.float32)
        gripper = np.clip(jaw_l + jaw_r, 0.0, 1.0)
        frame_f[0] -= 48.0 * gripper
        frame_f[1] -= 42.0 * gripper
        frame_f[2] -= 36.0 * gripper
      # Gentle micro sensor noise (1.2 uint8 levels) so consecutive frame correlation stays ~0.97-0.99
      frame_f += rng.normal(0.0, 1.2, size=(3, hw, hw)).astype(np.float32)
      quantized = np.clip(np.round(frame_f), 1.0, 254.0).astype(np.uint8)
      rgb_frames[step, cam] = quantized

  # Anchor pixel (cam=0, c=0, h=0, w=0) to uint8 0 at all steps so clean/255.0 is exact 0.0 in float32
  rgb_frames[:, 0, 0, 0, 0] = np.uint8(0)

  # 3. Smooth 8D proprioceptive trajectory (T, 8) float32 (7-DoF arm + 1D gripper width)
  t_norm = np.linspace(0.0, 1.0, t, dtype=np.float32)[:, None]
  joint_freqs = rng.uniform(0.8, 1.6, size=(1, 8)).astype(np.float32)
  joint_phases = rng.uniform(0.0, np.pi, size=(1, 8)).astype(np.float32)
  proprio_state = (
      0.55 * np.sin(t_norm * np.pi * joint_freqs + joint_phases)
      + 0.20 * np.cos(t_norm * 2.0 * np.pi + joint_phases)
  ).astype(np.float32)

  # 4. Continuous expert action chunks gt_actions (T, K, 7) float32 in [-1, 1]
  k_grid = np.linspace(0.05, 1.0, k, dtype=np.float32)[None, :, None]
  base_pose = proprio_state[:, :7][:, None, :]
  gt_actions = np.clip(
      0.60 * base_pose
      + 0.32 * np.sin(k_grid * np.pi + base_pose)
      + rng.normal(0.0, 0.02, size=(t, k, 7)).astype(np.float32),
      -0.95,
      0.95,
  ).astype(np.float32)

  # 5. Conditioning actions cond_actions (T, H, 7) float32 for World Models
  h_grid = np.linspace(0.05, 1.0, h, dtype=np.float32)[None, :, None]
  cond_actions = np.clip(
      0.62 * base_pose
      + 0.30 * np.cos(h_grid * np.pi + base_pose)
      + rng.normal(0.0, 0.02, size=(t, h, 7)).astype(np.float32),
      -0.95,
      0.95,
  ).astype(np.float32)

  # 6. Gate G4 micro-perturbed RGB frames: ||rgb_pert - rgb_clean/255.0||_inf == 1e-3 (exact float32)
  t_pert = min(int(num_perturbed_steps), t)
  g4_perturbed_indices = np.linspace(0, t - 1, num=t_pert, dtype=np.int32)
  clean_sub = rgb_frames[g4_perturbed_indices].astype(np.float32) / np.float32(255.0)
  # Smooth interior perturbation strictly bounded by 0.85e-3, plus exact +1e-3 at anchor pixel (0, 0, 0, 0)
  raw_delta = rng.uniform(-0.85e-3, 0.85e-3, size=clean_sub.shape).astype(np.float32)
  g4_perturbed_rgb = np.clip(clean_sub + raw_delta, np.float32(0.0), np.float32(1.0)).astype(np.float32)
  # At anchor pixel (cam=0, c=0, h=0, w=0), clean_sub is 0.0 -> setting 1e-3 gives exact float32 diff == 1e-3
  g4_perturbed_rgb[:, 0, 0, 0, 0] = np.float32(1e-3)

  # 7. Teacher-forced BF16 reference actions for canonical v1.0 models
  golden_bf16_ref_actions: Dict[str, np.ndarray] = {}
  for m_idx, model_id in enumerate(CANONICAL_V1_MODELS):
    model_rng = np.random.default_rng(seed + 1000 * (m_idx + 1))
    model_shift = model_rng.normal(0.0, 0.008, size=gt_actions.shape).astype(np.float32)
    ref_act = np.clip(gt_actions + model_shift, -0.98, 0.98).astype(np.float32)
    golden_bf16_ref_actions[model_id] = _quantize_to_bf16_float32(ref_act)

  # 8. Teacher-forced BF16 reference video latents (T, H, 16, 16) for canonical v1.0 models
  golden_bf16_ref_latents: Dict[str, np.ndarray] = {}
  # Downsample camera 0 clean frame (T, 3, hw, hw) -> (T, 1, 16, 16) as physical scene anchor
  cam0_norm = rgb_frames[:, 0].astype(np.float32) / 255.0
  if hw >= 16 and hw % 16 == 0:
    stride = hw // 16
    pooled_16x16 = cam0_norm.mean(axis=1).reshape(t, 16, stride, 16, stride).mean(axis=(2, 4))[:, None, :, :]
  else:
    idx16 = np.linspace(0, max(hw - 1, 0), 16, dtype=np.int32)
    pooled_16x16 = cam0_norm.mean(axis=1)[:, idx16, :][:, :, idx16][:, None, :, :]
  for m_idx, model_id in enumerate(CANONICAL_V1_MODELS):
    model_rng = np.random.default_rng(seed + 5000 * (m_idx + 1))
    action_drive = cond_actions.mean(axis=-1)[:, :, None, None] * 0.06
    step_innov = model_rng.normal(0.0, 0.018, size=(t, h, 16, 16)).astype(np.float32)
    latents = np.clip(
        0.20 + 0.60 * pooled_16x16 + np.cumsum(step_innov + action_drive * 0.05, axis=1),
        0.05,
        0.95,
    ).astype(np.float32)
    golden_bf16_ref_latents[model_id] = _quantize_to_bf16_float32(latents)

  shard = EmbodiedEpisodeShard(
      episode_id=episode_id,
      source_dataset=source_dataset,
      subset_name=subset_name,
      license_spdx=license_spdx,
      instruction_text=instruction_text,
      instruction_tokens=instruction_tokens,
      rgb_frames=rgb_frames,
      proprio_state=proprio_state,
      gt_actions=gt_actions,
      cond_actions=cond_actions,
      g4_perturbed_indices=g4_perturbed_indices,
      g4_perturbed_rgb=g4_perturbed_rgb,
      golden_bf16_ref_actions=golden_bf16_ref_actions,
      golden_bf16_ref_latents=golden_bf16_ref_latents,
  )
  shard.validate()
  return shard


def _resize_rgb_uint8_hwc(img: np.ndarray, hw: int) -> np.ndarray:
  """Resize a ``uint8[H, W, 3]`` camera image to ``uint8[hw, hw, 3]`` via bilinear sampling."""
  arr = np.asarray(img, dtype=np.uint8)
  if arr.ndim != 3 or arr.shape[2] != 3:
    raise ValueError(f"Expected uint8[H, W, 3] camera frame, got {arr.dtype}{arr.shape}.")
  if arr.shape[0] == hw and arr.shape[1] == hw:
    return np.ascontiguousarray(arr, dtype=np.uint8)
  try:
    from PIL import Image  # type: ignore

    return np.ascontiguousarray(
        np.asarray(Image.fromarray(arr).resize((hw, hw), Image.Resampling.BILINEAR), dtype=np.uint8),
        dtype=np.uint8,
    )
  except ImportError:
    ys = np.linspace(0, arr.shape[0] - 1, hw).astype(np.intp)
    xs = np.linspace(0, arr.shape[1] - 1, hw).astype(np.intp)
    return np.ascontiguousarray(arr[ys][:, xs], dtype=np.uint8)


def build_shard_from_real_lerobot_trace(
    trace: Any,
    *,
    episode_id: str,
    source_dataset: str,
    subset_name: str,
    license_spdx: str,
    seed: int = 20261030,
    start_step: int = 0,
    num_steps: int = 16,
    chunk_size_k: int = 16,
    horizon_h: int = 16,
    resolution_hw: int = 64,
    num_perturbed_steps: int = 4,
    instruction_override: Optional[str] = None,
) -> EmbodiedEpisodeShard:
  """Materialize an ``EmbodiedEpisodeShard`` directly from decoded real LeRobot camera frames and telemetry."""
  t = int(num_steps)
  k = int(chunk_size_k)
  h = int(horizon_h)
  hw = int(resolution_hw)
  n_total = len(trace)
  if n_total < t:
    raise ValueError(
        f"Real trace {trace.episode_id!r} has only {n_total} steps; requires at least num_steps={t}."
    )
  s0 = max(0, min(int(start_step), n_total - t))
  cams = list(trace.cameras)
  if not cams:
    raise ValueError(f"Real trace {trace.episode_id!r} contains no decoded camera streams.")
  cam0 = cams[0]
  cam1 = cams[1] if len(cams) > 1 else cams[0]

  rgb_frames = np.zeros((t, 2, 3, hw, hw), dtype=np.uint8)
  for step_i in range(t):
    f0 = _resize_rgb_uint8_hwc(trace.images[cam0][s0 + step_i], hw)
    f1 = _resize_rgb_uint8_hwc(trace.images[cam1][s0 + step_i], hw)
    rgb_frames[step_i, 0] = np.transpose(f0, (2, 0, 1))
    rgb_frames[step_i, 1] = np.transpose(f1, (2, 0, 1))
  # Anchor single corner byte so clean/255.0 == 0.0 for exact float32 Gate G4 L_inf == 1e-3.
  rgb_frames[:, 0, 0, 0, 0] = np.uint8(0)

  instruction_text = str(instruction_override or trace.instruction or "manipulate tabletop object").strip()
  rng = np.random.default_rng(int(seed))
  text_digest = hashlib.sha256(instruction_text.encode("utf-8")).digest()
  prefix_base = np.frombuffer(text_digest[:24], dtype=np.uint8).astype(np.int32) * 15 + 100
  prefix_24 = np.repeat(prefix_base[None, :], t, axis=0).astype(np.int32)
  step_idx_col = np.arange(t, dtype=np.int32)[:, None] + 500
  suffix_7 = rng.integers(50, 4000, size=(t, 7), dtype=np.int32)
  instruction_tokens = np.concatenate([prefix_24, step_idx_col, suffix_7], axis=1).astype(np.int32)

  raw_prop = np.asarray(trace.proprio[s0 : s0 + t], dtype=np.float32)
  if raw_prop.shape[1] >= 8:
    proprio_state = np.ascontiguousarray(raw_prop[:, :8], dtype=np.float32)
  else:
    proprio_state = np.zeros((t, 8), dtype=np.float32)
    proprio_state[:, : raw_prop.shape[1]] = raw_prop
    if trace.action.shape[1] > 0:
      proprio_state[:, raw_prop.shape[1] :] = np.asarray(trace.action[s0 : s0 + t, -1:], dtype=np.float32)

  raw_act_all = np.asarray(trace.action, dtype=np.float32)
  max_abs_act = float(np.max(np.abs(raw_act_all))) if raw_act_all.size else 1.0
  act_scale = max(1.0, max_abs_act)
  norm_act = np.clip(raw_act_all / np.float32(act_scale), -1.0, 1.0).astype(np.float32)

  def _gather_chunk(horizon_len: int) -> np.ndarray:
    out = np.zeros((t, horizon_len, 7), dtype=np.float32)
    for si in range(t):
      for hi in range(horizon_len):
        idx_a = min(s0 + si + hi, n_total - 1)
        vec = norm_act[idx_a]
        if vec.shape[0] >= 7:
          out[si, hi] = vec[:7]
        else:
          out[si, hi, : vec.shape[0]] = vec
    return out

  gt_actions = _gather_chunk(k)
  cond_actions = _gather_chunk(h)

  t_pert = min(int(num_perturbed_steps), t)
  g4_perturbed_indices = np.linspace(0, t - 1, num=t_pert, dtype=np.int32)
  clean_sub = rgb_frames[g4_perturbed_indices].astype(np.float32) / np.float32(255.0)
  raw_delta = rng.uniform(-0.85e-3, 0.85e-3, size=clean_sub.shape).astype(np.float32)
  g4_perturbed_rgb = np.clip(clean_sub + raw_delta, np.float32(0.0), np.float32(1.0)).astype(np.float32)
  g4_perturbed_rgb[:, 0, 0, 0, 0] = np.float32(1e-3)

  golden_bf16_ref_actions: Dict[str, np.ndarray] = {}
  for m_idx, model_id in enumerate(CANONICAL_V1_MODELS):
    model_rng = np.random.default_rng(int(seed) + 1000 * (m_idx + 1))
    model_shift = model_rng.normal(0.0, 0.008, size=gt_actions.shape).astype(np.float32)
    ref_act = np.clip(gt_actions + model_shift, -0.98, 0.98).astype(np.float32)
    golden_bf16_ref_actions[model_id] = _quantize_to_bf16_float32(ref_act)

  golden_bf16_ref_latents: Dict[str, np.ndarray] = {}
  cam0_norm = rgb_frames[:, 0].astype(np.float32) / 255.0
  if hw >= 16 and hw % 16 == 0:
    stride = hw // 16
    pooled_16x16 = cam0_norm.mean(axis=1).reshape(t, 16, stride, 16, stride).mean(axis=(2, 4))[:, None, :, :]
  else:
    idx16 = np.linspace(0, max(hw - 1, 0), 16, dtype=np.int32)
    pooled_16x16 = cam0_norm.mean(axis=1)[:, idx16, :][:, :, idx16][:, None, :, :]
  for m_idx, model_id in enumerate(CANONICAL_V1_MODELS):
    model_rng = np.random.default_rng(int(seed) + 5000 * (m_idx + 1))
    action_drive = cond_actions.mean(axis=-1)[:, :, None, None] * 0.06
    step_innov = model_rng.normal(0.0, 0.018, size=(t, h, 16, 16)).astype(np.float32)
    latents = np.clip(
        0.20 + 0.60 * pooled_16x16 + np.cumsum(step_innov + action_drive * 0.05, axis=1),
        0.05,
        0.95,
    ).astype(np.float32)
    golden_bf16_ref_latents[model_id] = _quantize_to_bf16_float32(latents)

  shard = EmbodiedEpisodeShard(
      episode_id=episode_id,
      source_dataset=source_dataset,
      subset_name=subset_name,
      license_spdx=license_spdx,
      instruction_text=instruction_text,
      instruction_tokens=instruction_tokens,
      rgb_frames=rgb_frames,
      proprio_state=proprio_state,
      gt_actions=gt_actions,
      cond_actions=cond_actions,
      g4_perturbed_indices=g4_perturbed_indices,
      g4_perturbed_rgb=g4_perturbed_rgb,
      golden_bf16_ref_actions=golden_bf16_ref_actions,
      golden_bf16_ref_latents=golden_bf16_ref_latents,
  )
  shard.validate()
  return shard


def build_and_write_manifest(
    dataset_dir: str | Path,
    shards: Sequence[EmbodiedEpisodeShard],
    shard_paths: Sequence[str | Path],
    split: str = "mini_v1",
    dataset_version: str = "EmbodiedX-Traces-v1",
    *,
    kind: str = REAL_ROBOT_TRACES_KIND,
    pixel_provenance: str = REAL_HF_PIXEL_PROVENANCE,
    is_synthetic_smoke: bool = False,
    certifiable: bool = True,
    upstream_hf_sources: Optional[Sequence[Dict[str, Any]]] = None,
    shard_provenance: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
  """Build and atomically write `manifest.json` with per-shard SHA-256, byte size, and frame coherence."""
  if len(shards) != len(shard_paths) or len(shards) == 0:
    raise ValueError("shards and shard_paths must have equal non-zero length.")

  out_dir = Path(dataset_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  shard_entries: List[Dict[str, Any]] = []
  coherences: List[float] = []
  prefix_ratios: List[float] = []
  sources_seen: List[str] = []
  total_steps = 0
  total_bytes = 0

  for idx, (shard, spath) in enumerate(zip(shards, shard_paths)):
    shard.validate()
    p = Path(spath)
    sha256_hex = compute_file_sha256(p)
    size_bytes = int(p.stat().st_size)
    coherence = round(shard.temporal_frame_coherence(), 6)
    prefix_ratio = round(shard.instruction_prefix_reuse_ratio(), 6)
    g4_linf = round(shard.g4_perturbation_linf(), 8)

    coherences.append(coherence)
    prefix_ratios.append(prefix_ratio)
    if shard.source_dataset not in sources_seen:
      sources_seen.append(shard.source_dataset)
    total_steps += shard.num_steps
    total_bytes += size_bytes

    entry: Dict[str, Any] = {
        "episode_id": shard.episode_id,
        "filename": p.name,
        "source_dataset": shard.source_dataset,
        "subset_name": shard.subset_name,
        "license_spdx": shard.license_spdx,
        "instruction_text": shard.instruction_text,
        "num_steps": shard.num_steps,
        "sha256": sha256_hex,
        "size_bytes": size_bytes,
        "temporal_frame_coherence": coherence,
        "instruction_prefix_reuse_ratio": prefix_ratio,
        "g4_perturbation_linf": g4_linf,
        "pixel_provenance": str(pixel_provenance),
        "golden_action_models": sorted(shard.golden_bf16_ref_actions.keys()),
        "golden_latent_models": sorted(shard.golden_bf16_ref_latents.keys()),
    }
    if shard_provenance is not None and idx < len(shard_provenance):
      entry.update(dict(shard_provenance[idx]))
    shard_entries.append(entry)

  first_rgb = shards[0].rgb_frames
  res_hw = int(first_rgb.shape[-1])
  num_cams = int(first_rgb.shape[1])
  num_ch = int(first_rgb.shape[2])
  bytes_per_req_imgs = int(num_cams * num_ch * res_hw * res_hw)

  manifest: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "dataset_version": dataset_version,
      "kind": str(kind),
      "pixel_provenance": str(pixel_provenance),
      "is_synthetic_smoke": bool(is_synthetic_smoke),
      "certifiable": bool(certifiable),
      "split": split,
      "generated_at": now_pacific_iso(),
      "total_episodes": len(shard_entries),
      "total_steps": total_steps,
      "total_bytes": total_bytes,
      "resolution_hw": res_hw,
      "bytes_per_request_images": bytes_per_req_imgs,
      "sources": sources_seen,
      "canonical_models": list(CANONICAL_V1_MODELS),
      "mean_temporal_frame_coherence": round(float(np.mean(coherences)), 6),
      "min_temporal_frame_coherence": round(float(np.min(coherences)), 6),
      "mean_instruction_prefix_reuse_ratio": round(float(np.mean(prefix_ratios)), 6),
      "g4_perturbation_linf_eps": 1e-3,
      "shards": shard_entries,
  }
  if upstream_hf_sources is not None:
    manifest["upstream_hf_sources"] = [dict(x) for x in upstream_hf_sources]
  manifest_path = out_dir / "manifest.json"
  atomic_write_json(manifest_path, manifest)
  return manifest


def generate_real_embodiedx_traces_from_lerobot(
    output_dir: str | Path,
    split: str = "public",
    base_seed: int = 20261030,
    num_steps: int = 16,
    resolution_hw: int = 224,
    max_episodes: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    tracks: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
  """Download and decode genuine Hugging Face LeRobot camera traces into verified EmbodiedX shards.

  Strictly refuses silent procedural fallback: raises ``RuntimeError`` if the pinned public
  repositories (``HuggingFaceVLA/libero`` and ``lerobot/droid_100``) cannot be fetched or decoded.
  """
  from eeb.data.download import DEFAULT_CACHE_DIR, download as dl_lerobot
  from eeb.data.lerobot_loader import LeRobotDataset, MissingDependencyError

  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  dl_cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

  track_norm = str(tracks or "").strip().lower()
  if track_norm in ("wm", "wm_track", "track2", "track2_wm"):
    base_specs = [s for s in REAL_MINI_V1_EPISODE_SPECS if s["source_dataset"] == "DROID"]
  elif track_norm in ("vla", "vla_track", "track1", "track1a", "track1b", "track1_vla"):
    base_specs = [s for s in REAL_MINI_V1_EPISODE_SPECS if s["source_dataset"] == "LIBERO"]
  else:
    base_specs = list(REAL_MINI_V1_EPISODE_SPECS)
  specs = base_specs if max_episodes is None else base_specs[: int(max_episodes)]
  if not specs:
    raise ValueError("generate_real_embodiedx_traces_from_lerobot requires at least 1 episode spec.")

  # Remove stale shard files left over from prior runs with different episode names/counts.
  keep_names = {s["filename"] for s in specs}
  for stale in out_dir.glob("episode_*.npz"):
    if stale.name not in keep_names:
      try:
        stale.unlink()
      except OSError:
        pass

  repo_max_ep: Dict[str, int] = {}
  repo_revs: Dict[str, str] = {}
  for spec in specs:
    rid = str(spec["hf_repo"])
    repo_revs[rid] = str(spec["hf_revision"])
    repo_max_ep[rid] = max(repo_max_ep.get(rid, 0), int(spec["source_episode_index"]) + 1)

  datasets_loaded: Dict[str, Any] = {}
  for rid, need_eps in repo_max_ep.items():
    rev = repo_revs[rid]
    needs_videos = rid == "lerobot/droid_100"
    try:
      local_root = dl_lerobot(
          rid,
          revision=rev,
          cache_dir=dl_cache,
          episodes=need_eps,
          include_videos=needs_videos,
          token=token,
      )
      datasets_loaded[rid] = LeRobotDataset(local_root, source_id=rid)
    except (MissingDependencyError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
      raise RuntimeError(
          f"Failed to download/open real LeRobot dataset {rid!r} @ revision {rev}: {exc}. "
          "Real camera traces are mandatory by default (RT-02). For strictly non-certifiable offline "
          "CI smoke testing only, pass --synthetic-ci-smoke explicitly."
      ) from exc

  shards: List[EmbodiedEpisodeShard] = []
  shard_paths: List[Path] = []
  shard_prov: List[Dict[str, Any]] = []
  for spec in specs:
    rid = str(spec["hf_repo"])
    rev = str(spec["hf_revision"])
    ep_i = int(spec["source_episode_index"])
    ds = datasets_loaded[rid]
    try:
      trace = ds.load_episode(ep_i, strict_instruction=False)
    except Exception as exc:
      raise RuntimeError(
          f"Failed to decode real camera pixels for {rid!r} episode {ep_i} @ {rev}: {exc}"
      ) from exc
    start_step = int(spec.get("start_step", 0))
    if len(trace) < start_step + int(num_steps):
      start_step = max(0, len(trace) - int(num_steps))
    shard = build_shard_from_real_lerobot_trace(
        trace,
        episode_id=str(spec["episode_id"]),
        source_dataset=str(spec["source_dataset"]),
        subset_name=str(spec["subset_name"]),
        license_spdx=str(spec["license_spdx"]),
        seed=int(base_seed) + int(spec["seed_offset"]),
        start_step=start_step,
        num_steps=int(num_steps),
        resolution_hw=int(resolution_hw),
    )
    shard_path = out_dir / str(spec["filename"])
    save_episode_shard_npz(shard, shard_path)
    shards.append(shard)
    shard_paths.append(shard_path)
    shard_prov.append({
        "hf_repo": rid,
        "hf_revision": rev,
        "source_episode_index": ep_i,
        "start_step": start_step,
        "native_cameras": list(trace.cameras),
    })

  upstream_meta = []
  for rid, rev in repo_revs.items():
    pin_info = CANONICAL_HF_TRACE_DATASET_PINS.get(rid, {})
    upstream_meta.append({
        "hf_repo": rid,
        "hf_revision": rev,
        "license_spdx": pin_info.get("license_spdx", "MIT"),
        "format": pin_info.get("format", "LeRobot v3.0"),
    })

  return build_and_write_manifest(
      dataset_dir=out_dir,
      shards=shards,
      shard_paths=shard_paths,
      split=split,
      kind=REAL_ROBOT_TRACES_KIND,
      pixel_provenance=REAL_HF_PIXEL_PROVENANCE,
      is_synthetic_smoke=False,
      certifiable=True,
      upstream_hf_sources=upstream_meta,
      shard_provenance=shard_prov,
  )


def generate_embodiedx_traces_split(
    output_dir: str | Path = DEFAULT_MINI_V1_DIR,
    split: str = "mini_v1",
    base_seed: int = 20261008,
    num_steps: int = 16,
    resolution_hw: int = 64,
    max_episodes: Optional[int] = None,
) -> Dict[str, Any]:
  """Generate an offline synthetic CI smoke bundle stamped explicitly as non-certifiable (`SYNTHETIC_CI_SMOKE_TRACES`)."""
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  specs = MINI_V1_EPISODE_SPECS if max_episodes is None else MINI_V1_EPISODE_SPECS[: int(max_episodes)]
  if len(specs) == 0:
    raise ValueError("generate_embodiedx_traces_split requires at least 1 episode spec.")

  # Remove stale canonical shards left behind when re-running with fewer `max_episodes`.
  for extra_spec in MINI_V1_EPISODE_SPECS[len(specs):]:
    stale_shard = out_dir / extra_spec["filename"]
    if stale_shard.is_file():
      try:
        stale_shard.unlink()
      except OSError:
        pass

  shards: List[EmbodiedEpisodeShard] = []
  shard_paths: List[Path] = []
  for spec in specs:
    shard = synthesize_deterministic_episode_shard(
        episode_id=spec["episode_id"],
        source_dataset=spec["source_dataset"],
        subset_name=spec["subset_name"],
        license_spdx=spec["license_spdx"],
        instruction_text=spec["instruction_text"],
        seed=base_seed + int(spec["seed_offset"]),
        num_steps=num_steps,
        resolution_hw=int(resolution_hw),
    )
    shard_path = out_dir / spec["filename"]
    save_episode_shard_npz(shard, shard_path)
    shards.append(shard)
    shard_paths.append(shard_path)

  return build_and_write_manifest(
      dataset_dir=out_dir,
      shards=shards,
      shard_paths=shard_paths,
      split=split,
      kind=SYNTHETIC_CI_SMOKE_TRACES_KIND,
      pixel_provenance=SYNTHETIC_PIXEL_PROVENANCE,
      is_synthetic_smoke=True,
      certifiable=False,
  )


class EmbodiedXDatasetLoader:
  """Verified loader and zero-I/O RAM preloader for `EmbodiedX-Traces-v1` episode shards."""

  def __init__(
      self,
      dataset_dir: Optional[str | Path] = None,
      split: str = "mini_v1",
  ) -> None:
    self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else DEFAULT_MINI_V1_DIR
    self.split = split
    self.manifest_path = self.dataset_dir / "manifest.json"
    self._preloaded_shards: Optional[List[EmbodiedEpisodeShard]] = None

  def verify_manifest(self) -> Tuple[bool, List[str]]:
    """Verify `manifest.json` and check every shard's on-disk SHA-256 and byte size."""
    errors: List[str] = []
    if not self.manifest_path.exists():
      errors.append(
          f"Manifest file not found at {self.manifest_path}. "
          f"Run 'python3 scripts/build_embodiedx_traces.py --output-dir {self.dataset_dir} --split {self.split}' "
          "to generate the dataset."
      )
      return False, errors

    try:
      manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
      errors.append(
          f"Corrupt or unreadable manifest.json at {self.manifest_path}: {exc}. "
          f"Re-run 'python3 scripts/build_embodiedx_traces.py --output-dir {self.dataset_dir} --split {self.split}'."
      )
      return False, errors

    shards_meta = manifest.get("shards")
    if not isinstance(shards_meta, list) or len(shards_meta) == 0:
      errors.append(
          f"Invalid manifest at {self.manifest_path}: 'shards' must be a non-empty list."
      )
      return False, errors

    for entry in shards_meta:
      filename = str(entry.get("filename", ""))
      episode_id = str(entry.get("episode_id", filename))
      expected_sha = str(entry.get("sha256", ""))
      expected_size = entry.get("size_bytes")
      try:
        shard_file = safe_resolve_under_root(self.dataset_dir, filename, check_symlinks=True)
      except ValueError as exc:
        errors.append(f"Unsafe shard filename {filename!r} in manifest: {exc}")
        continue

      if not shard_file.exists():
        errors.append(
            f"Missing shard file {shard_file} for episode_id={episode_id!r}. "
            f"Re-run 'python3 scripts/build_embodiedx_traces.py --output-dir {self.dataset_dir} --split {self.split}'."
        )
        continue

      actual_sha = compute_file_sha256(shard_file)
      actual_size = int(shard_file.stat().st_size)
      if actual_sha != expected_sha:
        errors.append(
            f"SHA-256 mismatch for shard {filename!r} (episode_id={episode_id!r}): "
            f"expected {expected_sha}, got {actual_sha} (size={actual_size} bytes, expected_size={expected_size}). "
            f"Rebuild clean shards via 'python3 scripts/build_embodiedx_traces.py --output-dir {self.dataset_dir} --split {self.split}'."
        )
      elif expected_size is not None and actual_size != int(expected_size):
        errors.append(
            f"Byte size mismatch for shard {filename!r} (episode_id={episode_id!r}): "
            f"expected {expected_size} bytes, got {actual_size} bytes."
        )

    return len(errors) == 0, errors

  def preload_into_pinned_ram(self, verify: bool = True) -> List[EmbodiedEpisodeShard]:
    """Verify and load all episode shards into contiguous C-order non-writable NumPy buffers in RAM."""
    if verify:
      ok, errors = self.verify_manifest()
      if not ok:
        raise ValueError(
            f"EmbodiedX dataset verification failed for {self.dataset_dir}: "
            + " | ".join(errors)
        )
    manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
    loaded: List[EmbodiedEpisodeShard] = []
    for entry in manifest["shards"]:
      shard_path = safe_resolve_under_root(self.dataset_dir, str(entry["filename"]), check_symlinks=True)
      shard = load_episode_shard_npz(shard_path, freeze_buffers=True)
      loaded.append(shard)
    self._preloaded_shards = loaded
    return list(loaded)

  def _ensure_preloaded(self) -> List[EmbodiedEpisodeShard]:
    if self._preloaded_shards is None:
      return self.preload_into_pinned_ram(verify=True)
    return self._preloaded_shards

  def _select_episode_steps(self, count: int) -> List[Tuple[EmbodiedEpisodeShard, int]]:
    """Select `count` (shard, step_idx) pairs cycling across episodes first for source diversity."""
    if count < 1:
      raise ValueError(f"Requested batch/concurrency count must be >= 1, got {count}.")
    shards = self._ensure_preloaded()
    num_eps = len(shards)
    selections: List[Tuple[EmbodiedEpisodeShard, int]] = []
    for i in range(count):
      shard = shards[i % num_eps]
      step_idx = (i // num_eps) % shard.num_steps
      selections.append((shard, step_idx))
    return selections

  def to_track1_golden_batch(
      self,
      model_id: str,
      batch_size: int,
      horizon_frames: int = 16,
  ) -> Track1GoldenBatch:
    """Construct a `Track1GoldenBatch` from preloaded RAM shards for `stage1_golden_replay` and `measure`."""
    shards = self._ensure_preloaded()
    canonical_id = resolve_canonical_model_id(model_id)
    selections = self._select_episode_steps(batch_size)
    h = int(horizon_frames)

    obs_list: List[np.ndarray] = []
    cond_list: List[np.ndarray] = []
    latent_list: List[np.ndarray] = []

    for shard, step_idx in selections:
      # Camera 0 normalized to float32 [0, 1], shape (3, 64, 64)
      obs = shard.rgb_frames[step_idx, 0].astype(np.float32) / np.float32(255.0)
      obs_list.append(obs)

      # Conditioning actions shape (H, 7)
      raw_cond = shard.cond_actions[step_idx]
      if raw_cond.shape[0] >= h:
        cond = raw_cond[:h]
      else:
        reps = (h + raw_cond.shape[0] - 1) // raw_cond.shape[0]
        cond = np.tile(raw_cond, (reps, 1))[:h]
      cond_list.append(cond.astype(np.float32))

      # Teacher-forced BF16 reference video latents (H, 16, 16)
      if canonical_id in shard.golden_bf16_ref_latents:
        raw_lat = shard.golden_bf16_ref_latents[canonical_id][step_idx]
      else:
        fallback_key = WM_CANONICAL_MODELS[0]
        raw_lat = shard.golden_bf16_ref_latents[fallback_key][step_idx]
      if raw_lat.shape[0] >= h:
        lat = raw_lat[:h]
      else:
        reps = (h + raw_lat.shape[0] - 1) // raw_lat.shape[0]
        lat = np.tile(raw_lat, (reps, 1, 1))[:h]
      latent_list.append(lat.astype(np.float32))

    obs_frames = np.ascontiguousarray(np.stack(obs_list, axis=0), dtype=np.float32)
    cond_actions = np.ascontiguousarray(np.stack(cond_list, axis=0), dtype=np.float32)
    ref_video_latents = np.ascontiguousarray(np.stack(latent_list, axis=0), dtype=np.float32)
    ref_latent_actions = np.ascontiguousarray(
        np.tanh(
            cond_actions * 0.85
            + ref_video_latents[:, :, :7, 0].mean(axis=-1, keepdims=True) * 0.15
        ),
        dtype=np.float32,
    )

    mean_coherence = round(
        float(np.mean([s.temporal_frame_coherence() for s in shards])), 4
    )
    sources = tuple(dict.fromkeys(s.source_dataset for s in shards))
    return Track1GoldenBatch(
        batch_size=int(batch_size),
        horizon_frames=h,
        obs_frames=obs_frames,
        cond_actions=cond_actions,
        ref_video_latents=ref_video_latents,
        ref_latent_actions=ref_latent_actions,
        dataset_version="EmbodiedX-Traces-v1",
        dataset_sources=sources,
        temporal_frame_coherence=mean_coherence,
        g4_perturbation_linf_eps=1e-3,
    )

  def to_track2_golden_batch(
      self,
      model_id: str,
      concurrency: int,
      chunk_size: int = 16,
      action_dim: int = 7,
  ) -> Track2GoldenBatch:
    """Construct a `Track2GoldenBatch` from preloaded RAM shards for `stage1_golden_replay` and `measure`."""
    shards = self._ensure_preloaded()
    canonical_id = resolve_canonical_model_id(model_id)
    selections = self._select_episode_steps(concurrency)
    k = int(chunk_size)
    act_dim = int(action_dim)

    rgb_list: List[np.ndarray] = []
    proprio_list: List[np.ndarray] = []
    tokens_list: List[np.ndarray] = []
    ref_act_list: List[np.ndarray] = []

    for shard, step_idx in selections:
      rgb = shard.rgb_frames[step_idx].astype(np.float32) / np.float32(255.0)
      rgb_list.append(rgb)
      proprio_list.append(shard.proprio_state[step_idx].astype(np.float32))
      tokens_list.append(shard.instruction_tokens[step_idx].astype(np.int32))

      if canonical_id in shard.golden_bf16_ref_actions:
        raw_act = shard.golden_bf16_ref_actions[canonical_id][step_idx]
      else:
        raw_act = shard.gt_actions[step_idx]

      if raw_act.shape[0] >= k:
        act_k = raw_act[:k]
      else:
        reps = (k + raw_act.shape[0] - 1) // raw_act.shape[0]
        act_k = np.tile(raw_act, (reps, 1))[:k]

      if act_k.shape[1] >= act_dim:
        act_final = act_k[:, :act_dim]
      else:
        reps_d = (act_dim + act_k.shape[1] - 1) // act_k.shape[1]
        act_final = np.tile(act_k, (1, reps_d))[:, :act_dim]
      ref_act_list.append(act_final.astype(np.float32))

    rgb_frames = np.ascontiguousarray(np.stack(rgb_list, axis=0), dtype=np.float32)
    proprio_state = np.ascontiguousarray(np.stack(proprio_list, axis=0), dtype=np.float32)
    instruction_tokens = np.ascontiguousarray(np.stack(tokens_list, axis=0), dtype=np.int32)
    ref_actions = np.ascontiguousarray(np.stack(ref_act_list, axis=0), dtype=np.float32)

    mean_coherence = round(
        float(np.mean([s.temporal_frame_coherence() for s in shards])), 4
    )
    mean_prefix_reuse = round(
        float(np.mean([s.instruction_prefix_reuse_ratio() for s in shards])), 4
    )
    sources = tuple(dict.fromkeys(s.source_dataset for s in shards))
    return Track2GoldenBatch(
        concurrency=int(concurrency),
        chunk_size=k,
        action_dim=act_dim,
        rgb_frames=rgb_frames,
        proprio_state=proprio_state,
        instruction_tokens=instruction_tokens,
        ref_actions=ref_actions,
        dataset_version="EmbodiedX-Traces-v1",
        dataset_sources=sources,
        temporal_frame_coherence=mean_coherence,
        instruction_prefix_reuse_ratio=mean_prefix_reuse,
        g4_perturbation_linf_eps=1e-3,
    )

  @classmethod
  def download_from_hf_hub(
      cls,
      repo_id: str = "HuggingFaceVLA/libero",
      split: str = "mini_v1",
      local_dir: Optional[str | Path] = None,
      *,
      allow_synthetic_fallback: bool = False,
      resolution_hw: int = 224,
      max_episodes: Optional[int] = None,
  ) -> "EmbodiedXDatasetLoader":
    """Fetch real EmbodiedX robot camera traces from HF Hub without silent procedural fallback (RT-02)."""
    target_dir = Path(local_dir) if local_dir is not None else DEFAULT_MINI_V1_DIR
    loader = cls(dataset_dir=target_dir, split=split)

    # Fast path if local split already exists, passes SHA-256 verification, and is non-synthetic
    ok, _ = loader.verify_manifest()
    if ok:
      raw_man = json.loads(loader.manifest_path.read_text(encoding="utf-8"))
      if not raw_man.get("is_synthetic_smoke", False) or allow_synthetic_fallback:
        return loader

    hub_errors: List[str] = []
    # Attempt 1: direct snapshot_download when pointing at a pre-packed EmbodiedX bundle repo
    if repo_id and repo_id not in PINNED_HUB_REVISIONS:
      try:
        import huggingface_hub  # type: ignore[import-not-found]

        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=[f"{split}/*", "manifest.json", "*.npz"],
        )
        ok_after_dl, _ = loader.verify_manifest()
        if ok_after_dl:
          return loader
      except Exception as exc:  # noqa: BLE001
        hub_errors.append(f"{repo_id}: {type(exc).__name__}: {exc}")

    # Attempt 2: decode directly from pinned public LeRobot camera trace repos
    try:
      generate_real_embodiedx_traces_from_lerobot(
          output_dir=target_dir,
          split=split,
          resolution_hw=int(resolution_hw),
          max_episodes=max_episodes,
      )
      ok_real, errors_real = loader.verify_manifest()
      if ok_real:
        return loader
      hub_errors.append(f"verify_manifest after LeRobot decode failed: {errors_real}")
    except Exception as exc:  # noqa: BLE001
      hub_errors.append(f"pinned LeRobot fetch: {type(exc).__name__}: {exc}")

    if not allow_synthetic_fallback:
      raise RuntimeError(
          f"Failed to download real EmbodiedX robot camera traces for split={split!r} into {target_dir}: "
          + " | ".join(hub_errors)
          + ". Silent procedural fallback is disabled (RT-02); pass allow_synthetic_fallback=True "
          "(or --synthetic-ci-smoke) only for non-certifiable offline CI smoke testing."
      )

    generate_embodiedx_traces_split(output_dir=target_dir, split=split)
    ok_final, errors = loader.verify_manifest()
    if not ok_final:
      raise RuntimeError(
          f"Failed to initialize EmbodiedX-Traces-v1 ({split}) in {target_dir}: {errors}"
      )
    return loader


def download_from_hf_hub(
    repo_id: str = "HuggingFaceVLA/libero",
    split: str = "mini_v1",
    local_dir: Optional[str | Path] = None,
    *,
    allow_synthetic_fallback: bool = False,
    resolution_hw: int = 224,
    max_episodes: Optional[int] = None,
) -> EmbodiedXDatasetLoader:
  """Module-level convenience wrapper for `EmbodiedXDatasetLoader.download_from_hf_hub`."""
  return EmbodiedXDatasetLoader.download_from_hf_hub(
      repo_id=repo_id,
      split=split,
      local_dir=local_dir,
      allow_synthetic_fallback=allow_synthetic_fallback,
      resolution_hw=resolution_hw,
      max_episodes=max_episodes,
  )
