"""``eeb data pack-traces``: LeRobot root -> ``traces/<split>/episode_XXXX.npz`` + ``manifest.json``.

Output layout::

    <out_dir>/traces/manifest.json                 index of splits (+ sha256 of each split manifest)
    <out_dir>/traces/<split>/manifest.json         TraceManifest (per-file sha256 + content sha256)
    <out_dir>/traces/<split>/episode_XXXX.npz      EmbodiedXTrace container (pure NumPy)
    <out_dir>/traces/<split>/episode_XXXX.parquet  optional sidecar (``write_parquet=True`` + pyarrow)

Guarantees: temporal order and native per-episode timestamps are preserved; exactly one instruction
per episode; ``public``/``hidden`` assignment is a seeded sha256 of the episode id (stable across
machines and re-packs); every write is atomic (temp file + ``os.replace``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from eeb.data.lerobot_loader import LeRobotDataset, MissingDependencyError
from eeb.data.schema import (
    DEFAULT_HIDDEN_FRACTION,
    DEFAULT_SPLIT_SEED,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    SPLITS,
    EmbodiedXTrace,
    ManifestEpisode,
    ManifestVerification,
    TraceManifest,
    assign_split,
    atomic_write_json,
    episode_filename,
    sha256_file,
    utc_now_iso,
)

TRACES_DIRNAME = "traces"
AUTO_SPLIT = "auto"


@dataclass
class PackResult:
  out_dir: Path
  traces_dir: Path
  manifests: Dict[str, TraceManifest] = field(default_factory=dict)
  episode_files: List[Path] = field(default_factory=list)
  skipped_episodes: List[int] = field(default_factory=list)

  @property
  def episode_count(self) -> int:
    return len(self.episode_files)

  def summary(self) -> Dict[str, Any]:
    return {
        "out_dir": str(self.out_dir),
        "episodes_packed": self.episode_count,
        "splits": {s: {"episodes": m.episode_count, "steps": m.step_count} for s, m in self.manifests.items()},
        "skipped_episodes": self.skipped_episodes,
    }


def resolve_fps_stride(native_fps: float, fps: Optional[float]) -> int:
  """Integer temporal stride turning ``native_fps`` into ``fps`` (1 when ``fps`` is None)."""
  if fps is None:
    return 1
  if fps <= 0:
    raise ValueError(f"fps must be > 0, got {fps}")
  ratio = native_fps / float(fps)
  stride = int(round(ratio))
  if stride < 1 or abs(ratio - stride) > 1e-6:
    raise ValueError(
        f"requested fps={fps} is not an integer divisor of the native fps={native_fps} "
        f"(ratio {ratio:.4f}); EmbodiedX traces never resample timestamps"
    )
  return stride


def pack_traces(
    src_lerobot_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    split: str = AUTO_SPLIT,
    max_episodes: Optional[int] = None,
    cameras: Optional[Sequence[str]] = None,
    fps: Optional[float] = None,
    *,
    source_dataset: Optional[str] = None,
    source_id: Optional[str] = None,
    license: Optional[str] = None,
    seed: int = DEFAULT_SPLIT_SEED,
    hidden_fraction: float = DEFAULT_HIDDEN_FRACTION,
    episode_indices: Optional[Sequence[int]] = None,
    write_parquet: bool = False,
    decode_images: bool = True,
    strict_instruction: bool = True,
    source_revision: str = "",
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    image_dir: Optional[str | os.PathLike[str]] = None,
    log=None,
) -> PackResult:
  """Pack episodes of a local LeRobot dataset into EmbodiedX-Traces-v1 containers.

  Args:
    src_lerobot_dir: local LeRobot v2/v3 root (``meta/info.json`` present).
    out_dir: destination; traces land in ``<out_dir>/traces/<split>/``.
    split: ``"auto"`` (seeded episode hash -> public/hidden) or a forced ``"public"``/``"hidden"``.
    max_episodes: stop after this many packed episodes (episodes without local frame tables are skipped).
    cameras: short camera names to keep (default: all camera features).
    fps: target fps; must be an integer divisor of the native fps (stride subsampling, native timestamps kept).
    source_dataset: ``EMBODIEDX_TRACES_V1_SOURCES`` key (default: guessed from the dataset id).
    source_id: Hub id recorded in episode ids / manifest (default: inferred from the directory name).
    license: override the license recorded in the manifest.
    seed / hidden_fraction: split policy parameters (recorded in the manifest).
    episode_indices: explicit source episode indices to pack (default: ascending order).
    write_parquet: also write a parquet sidecar per episode (requires pyarrow).
    decode_images: ``False`` packs proprio/action/timestamps only (cameras = []), flagged in the manifest.
    strict_instruction: fail when an instruction changes within an episode.
  """
  out_dir = Path(out_dir)
  traces_dir = out_dir / TRACES_DIRNAME
  if split != AUTO_SPLIT and split not in SPLITS:
    raise ValueError(f"split must be one of {(AUTO_SPLIT,) + SPLITS}, got {split!r}")
  ds = LeRobotDataset(src_lerobot_dir, state_key=state_key, action_key=action_key, image_dir=image_dir, source_id=source_id)
  stride = resolve_fps_stride(ds.info.fps, fps)
  if write_parquet:
    try:
      import pyarrow  # type: ignore # noqa: F401
    except ImportError as e:
      raise MissingDependencyError("write_parquet=True requires pyarrow (`pip install pyarrow`)") from e

  manifests: Dict[str, TraceManifest] = {}
  result = PackResult(out_dir=out_dir, traces_dir=traces_dir)
  split_policy = {
      "method": "sha256(f'{seed}|{episode_id}')[:8] / 2**64 < hidden_fraction -> hidden",
      "seed": int(seed),
      "hidden_fraction": float(hidden_fraction),
      "forced_split": None if split == AUTO_SPLIT else split,
  }
  notes: List[str] = []
  if not decode_images:
    notes.append("images_not_packed: traces contain proprio/action/timestamps only")
  if stride > 1:
    notes.append(f"temporal_stride={stride}: every {stride}-th native frame kept, native timestamps preserved")

  indices = list(episode_indices) if episode_indices is not None else [e.episode_index for e in ds.episodes]
  for ei in indices:
    ep = ds.episode(ei)
    if not ds.has_local_data(ep):
      result.skipped_episodes.append(ei)
      continue
    trace = ds.load_episode(ei, cameras=cameras, decode_images=decode_images, source_dataset=source_dataset, strict_instruction=strict_instruction)
    trace = trace.subsample(stride)
    chosen_split = assign_split(trace.episode_id, seed=seed, hidden_fraction=hidden_fraction) if split == AUTO_SPLIT else split
    trace = trace.with_split(chosen_split)
    if license is not None:
      trace.license = license
    split_dir = traces_dir / chosen_split
    fname = episode_filename(ei)
    fpath = trace.to_npz(split_dir / fname)
    extra: Dict[str, str] = {}
    if write_parquet:
      pq_name = episode_filename(ei, ".parquet")
      write_trace_parquet(trace, split_dir / pq_name)
      extra["parquet"] = pq_name
    man = manifests.get(chosen_split)
    if man is None:
      man = TraceManifest(
          source=ds.source_id,
          source_dataset=trace.source_dataset,
          source_codebase_version=ds.info.codebase_version,
          source_revision=source_revision,
          license=trace.license,
          fps=trace.fps,
          native_fps=ds.info.fps,
          cameras=list(trace.cameras),
          split=chosen_split,
          split_policy=dict(split_policy),
          notes=list(notes),
      )
      manifests[chosen_split] = man
    man.episodes.append(
        ManifestEpisode(
            episode_id=trace.episode_id,
            file=fname,
            steps=len(trace),
            instruction=trace.instruction,
            sha256=sha256_file(fpath),
            content_sha256=trace.content_sha256(),
            source_episode_index=ei,
            cameras=list(trace.cameras),
            extra_files=extra,
        )
    )
    result.episode_files.append(fpath)
    if log:
      log(f"packed episode {ei:>6d} -> {fpath.relative_to(out_dir)}  steps={len(trace)} split={chosen_split} instruction={trace.instruction[:60]!r}")
    if max_episodes is not None and result.episode_count >= max_episodes:
      break

  if not result.episode_files:
    raise FileNotFoundError(
        f"no episodes with local frame tables found under {ds.root} (listed={len(ds.episodes)}, "
        f"skipped={len(result.skipped_episodes)}). Download some with `eeb data download {ds.source_id} --episodes 2`."
    )
  for s, man in manifests.items():
    man.save(traces_dir / s)
  write_traces_index(traces_dir, manifests)
  result.manifests = manifests
  return result


def write_traces_index(traces_dir: Path, manifests: Dict[str, TraceManifest]) -> Path:
  """``traces/manifest.json``: which splits exist and the sha256 of each split manifest."""
  index_path = traces_dir / MANIFEST_FILENAME
  existing: Dict[str, Any] = {}
  if index_path.exists():
    try:
      with open(index_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    except (OSError, ValueError):
      existing = {}
  splits = dict(existing.get("splits", {}))
  for s, man in manifests.items():
    mp = traces_dir / s / MANIFEST_FILENAME
    splits[s] = {
        "manifest": f"{s}/{MANIFEST_FILENAME}",
        "sha256": sha256_file(mp),
        "episode_count": man.episode_count,
        "step_count": man.step_count,
    }
  return atomic_write_json(index_path, {"schema_version": SCHEMA_VERSION, "updated_at": utc_now_iso(), "splits": splits})


def write_trace_parquet(trace: EmbodiedXTrace, path: Path) -> Path:
  """Parquet sidecar mirroring the LeRobot column style (images as PNG bytes structs)."""
  import pyarrow as pa  # type: ignore
  import pyarrow.parquet as pq  # type: ignore
  from PIL import Image  # type: ignore

  n = len(trace)
  cols: Dict[str, Any] = {
      "episode_id": pa.array([trace.episode_id] * n, pa.string()),
      "step": pa.array(trace.step.tolist(), pa.int64()),
      "t_ns": pa.array(trace.t_ns.tolist(), pa.int64()),
      "instruction": pa.array([trace.instruction] * n, pa.string()),
      "proprio": pa.array(trace.proprio.tolist(), pa.list_(pa.float32())),
      "action": pa.array(trace.action.tolist(), pa.list_(pa.float32())),
  }
  for cam in trace.cameras:
    blobs = []
    for i in range(n):
      buf = io.BytesIO()
      Image.fromarray(trace.images[cam][i]).save(buf, format="PNG")
      blobs.append({"bytes": buf.getvalue(), "path": f"{cam}/frame_{int(trace.step[i]):06d}.png"})
    cols[f"observation.images.{cam}"] = pa.array(blobs, pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
  table = pa.table(cols)
  tmp = path.with_suffix(path.suffix + ".tmp")
  pq.write_table(table, str(tmp), compression="zstd")
  os.replace(tmp, path)
  return path


# --------------------------------------------------------------------------------------------
# Verification / inspection of a packed directory
# --------------------------------------------------------------------------------------------
def find_traces_dir(path: str | os.PathLike[str]) -> Path:
  p = Path(path)
  if (p / TRACES_DIRNAME).is_dir():
    return p / TRACES_DIRNAME
  if p.name == TRACES_DIRNAME or (p / MANIFEST_FILENAME).exists():
    return p
  raise FileNotFoundError(f"{p} is neither an <out_dir> with traces/ nor a traces/<split> directory")


def list_split_dirs(traces_dir: Path) -> List[Path]:
  if (traces_dir / MANIFEST_FILENAME).exists() and any((traces_dir / s).is_dir() for s in SPLITS):
    return [traces_dir / s for s in SPLITS if (traces_dir / s / MANIFEST_FILENAME).exists()]
  if (traces_dir / MANIFEST_FILENAME).exists():
    return [traces_dir]
  return [traces_dir / s for s in SPLITS if (traces_dir / s / MANIFEST_FILENAME).exists()]


def verify_traces(path: str | os.PathLike[str], deep: bool = False) -> Dict[str, ManifestVerification]:
  """Verify every split manifest under ``path`` (``<out_dir>``, ``traces/`` or ``traces/<split>``)."""
  traces_dir = find_traces_dir(path)
  out: Dict[str, ManifestVerification] = {}
  for sd in list_split_dirs(traces_dir):
    man = TraceManifest.load(sd)
    out[man.split] = man.verify(sd, deep=deep)
  if not out:
    raise FileNotFoundError(f"no split manifests found under {traces_dir}")
  return out


def load_split(path: str | os.PathLike[str], split: str) -> List[EmbodiedXTrace]:
  """Load all traces of one split (verifies file sha256s first)."""
  traces_dir = find_traces_dir(path)
  sd = traces_dir / split
  man = TraceManifest.load(sd)
  ver = man.verify(sd)
  if not ver.ok:
    raise ValueError(ver.summary())
  return [EmbodiedXTrace.from_npz(sd / e.file) for e in man.episodes]
