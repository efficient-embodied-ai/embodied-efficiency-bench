"""EmbodiedX-Traces-v1 schema: per-step records, per-episode trace container, manifest, split policy.

Design (see ``docs/DATA.md`` and ``docs/ENGINEERING_PLAN.md`` §2.1 ADR-2):

* The **LeRobot v3 dataset layout is canonical**; an ``EmbodiedXTrace`` is a *derived view* of one
  episode: temporally ordered steps, native timestamps, one constant language instruction, golden
  (recorded) actions, decoded ``uint8[H, W, 3]`` camera frames and ``float32`` proprioception.
* The always-available on-disk container is a pure-NumPy ``.npz`` (no pyarrow / torch / lerobot
  needed to read it). Parquet is an optional sidecar written by ``eeb.data.pack`` when pyarrow is
  importable.
* ``EMBODIEDX_TRACES_V1_SOURCES`` is **imported** from ``embodied_bench.golden_dataset`` (the single
  source of truth for licenses / modalities) and *extended* here with the Hub dataset ids whose
  on-Hub layout was verified against ``https://huggingface.co/api/datasets/<id>/tree/main``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _load_base_sources() -> Dict[str, Dict[str, Any]]:
  """Import ``EMBODIEDX_TRACES_V1_SOURCES`` from ``embodied_bench/golden_dataset.py`` (single source of truth).

  The regular package import is tried first; if ``embodied_bench/__init__.py`` cannot initialise
  (it eagerly imports every placeholder backend), the module file is loaded directly so the data
  tooling never depends on the analytic-placeholder modules.
  """
  try:
    from embodied_bench.golden_dataset import EMBODIEDX_TRACES_V1_SOURCES as base  # type: ignore

    return base
  except Exception:  # noqa: BLE001 - fall through to the file-based import below
    pass
  import importlib.util
  import sys

  candidate = Path(__file__).resolve().parents[2] / "embodied_bench" / "golden_dataset.py"
  if not candidate.exists():
    raise ImportError(
        f"cannot locate embodied_bench/golden_dataset.py (looked at {candidate}); "
        "EMBODIEDX_TRACES_V1_SOURCES must come from that file"
    )
  mod_name = "_eeb_data_golden_dataset_src"
  spec = importlib.util.spec_from_file_location(mod_name, candidate)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  sys.modules[mod_name] = mod  # dataclasses resolve string annotations through sys.modules
  try:
    spec.loader.exec_module(mod)
  except BaseException:
    sys.modules.pop(mod_name, None)
    raise
  return getattr(mod, "EMBODIEDX_TRACES_V1_SOURCES")


_BASE_SOURCES: Dict[str, Dict[str, Any]] = _load_base_sources()

SCHEMA_VERSION = "embodiedx-traces-v1.0"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_SPLIT_SEED = 20261007
DEFAULT_HIDDEN_FRACTION = 0.2
SPLITS: Tuple[str, ...] = ("public", "hidden")
UNASSIGNED_SPLIT = "unassigned"
IMAGE_KEY_PREFIX = "image__"  # npz key prefix for per-camera frame stacks
META_KEY = "meta_json"
REAL_ROBOT_TRACES_KIND = "REAL_ROBOT_TRACES"
SYNTHETIC_CI_SMOKE_TRACES_KIND = "SYNTHETIC_CI_SMOKE_TRACES"
REAL_HF_PIXEL_PROVENANCE = "REAL_HF_LEROBOT_CAMERA_PIXELS"
SYNTHETIC_PIXEL_PROVENANCE = "SYNTHETIC_PROCEDURAL_FALLBACK"

# Canonical 40-char Hugging Face commit SHAs for public trace repos (RT-02 / RT-04).
PINNED_HUB_REVISIONS: Dict[str, str] = {
    "HuggingFaceVLA/libero": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
    "lerobot/droid_100": "87301a2d2e99340e2010c9ef0f1d8e780b08aaf9",
}


# --------------------------------------------------------------------------------------------
# Verified Hugging Face Hub sources (layout verified 2026-10-08, see docs/DATA.md for exact URLs)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HubDataset:
  """A Hugging Face Hub dataset whose on-Hub layout was verified by `eeb.data` (not invented)."""

  dataset_id: str
  codebase_version: str          # LeRobot `meta/info.json: codebase_version` (e.g. "v3.0", "v2.0")
  image_storage: str             # "parquet-embedded-png" | "video-av1"
  cameras: Tuple[str, ...]       # short camera names (feature key minus `observation.images.`)
  fps: float
  total_episodes: int
  verified_on: str               # ISO date of the layout verification
  notes: str = ""

  @property
  def hub_url(self) -> str:
    return f"https://huggingface.co/datasets/{self.dataset_id}"


VERIFIED_HUB_DATASETS: Dict[str, Tuple[HubDataset, ...]] = {
    "LIBERO": (
        HubDataset(
            dataset_id="HuggingFaceVLA/libero",
            codebase_version="v3.0",
            image_storage="parquet-embedded-png",
            cameras=("image", "image2"),
            fps=10.0,
            total_episodes=1693,
            verified_on="2026-10-08",
            notes=(
                "All 4 LIBERO suites merged (40 tasks, 273,465 frames). Frames are PNG bytes embedded "
                "in data/chunk-000/file-XXX.parquet (3 episodes/file, ~70-100 MB each); no videos/ dir. "
                "Per-episode `tasks` list lives in meta/episodes/chunk-000/file-000.parquet."
            ),
        ),
        HubDataset(
            dataset_id="lerobot/libero_10",
            codebase_version="v3.0",
            image_storage="video-av1",
            cameras=("image", "wrist_image"),
            fps=10.0,
            total_episodes=379,
            verified_on="2026-10-08",
            notes=(
                "LIBERO-10 (= LIBERO-Long). Tiny data parquets (~60 KB, 3 episodes/file); frames are AV1 "
                "yuv420p mp4 segments (videos/<key>/chunk-000/file-000.mp4 ~380 MB holds many episodes, "
                "addressed by from_timestamp/to_timestamp). Needs `av` (libdav1d) or a pre-decoded images/ dir."
            ),
        ),
        HubDataset(
            dataset_id="physical-intelligence/libero",
            codebase_version="v2.0",
            image_storage="parquet-embedded-png",
            cameras=("image", "wrist_image"),
            fps=10.0,
            total_episodes=1693,
            verified_on="2026-10-08",
            notes=(
                "LeRobot v2.0 layout: meta/episodes.jsonl + meta/tasks.jsonl, one data/chunk-XXX/"
                "episode_XXXXXX.parquet per episode (~28-37 MB) with embedded PNG frames; keys `image`, "
                "`wrist_image`, `state`, `actions`."
            ),
        ),
    ),
    "DROID": (
        HubDataset(
            dataset_id="lerobot/droid_100",
            codebase_version="v3.0",
            image_storage="video-av1",
            cameras=("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"),
            fps=15.0,
            total_episodes=100,
            verified_on="2026-10-30",
            notes=(
                "Curated 100-episode DROID validation subset (180x320 AV1 videos, 15 fps, MIT license). "
                "Pinned revision 87301a2d2e99340e2010c9ef0f1d8e780b08aaf9."
            ),
        ),
        HubDataset(
            dataset_id="lerobot/droid_1.0.1",
            codebase_version="v3.0",
            image_storage="video-av1",
            cameras=("exterior_1_left", "exterior_2_left", "wrist_left"),
            fps=15.0,
            total_episodes=95658,
            verified_on="2026-10-08",
            notes=(
                "27.6 M frames, 49,630 tasks. 180x320 AV1 videos; `observation.state` [8] and `action` [8] "
                "(joint positions + gripper) plus per-frame `language_instruction` strings. Layout verified "
                "via meta/info.json + tree listing only (no data files downloaded in this pass)."
            ),
        ),
    ),
    "BridgeData-V2": (),   # not verified in this pass -- do not invent ids
    "AgiBot-World": (),    # not verified in this pass -- do not invent ids
}

# Hub ids that were *tried* and returned HTTP 401 (HF's response for nonexistent / gated repos).
UNVERIFIED_HUB_IDS: Tuple[str, ...] = (
    "lerobot/libero_spatial",
    "lerobot/libero_object",
    "lerobot/libero_goal",
    "lerobot/droid",
)


def _extend_sources(base: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
  """Return ``base`` + a ``hub_datasets`` tuple per source (base entries are not copied by hand)."""
  out: Dict[str, Dict[str, Any]] = {}
  for name, spec in base.items():
    merged = dict(spec)
    merged["hub_datasets"] = VERIFIED_HUB_DATASETS.get(name, ())
    out[name] = merged
  return out


EMBODIEDX_TRACES_V1_SOURCES: Dict[str, Dict[str, Any]] = _extend_sources(_BASE_SOURCES)


def source_license(source_dataset: Optional[str]) -> str:
  """License string from the shared source table (empty when the source is unknown)."""
  if not source_dataset:
    return ""
  spec = EMBODIEDX_TRACES_V1_SOURCES.get(source_dataset)
  return str(spec.get("license", "")) if spec else ""


def guess_source_dataset(dataset_id_or_path: str) -> Optional[str]:
  """Map a Hub id / directory name onto an ``EMBODIEDX_TRACES_V1_SOURCES`` key (None if unknown)."""
  s = dataset_id_or_path.lower()
  if "libero" in s:
    return "LIBERO"
  if "droid" in s:
    return "DROID"
  if "bridge" in s:
    return "BridgeData-V2"
  if "agibot" in s:
    return "AgiBot-World"
  return None


# --------------------------------------------------------------------------------------------
# Hashing / JSON helpers
# --------------------------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike[str], chunk_bytes: int = 1 << 20) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(chunk_bytes), b""):
      h.update(chunk)
  return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
  """Deterministic JSON (sorted keys, no whitespace variance) used for content hashes."""
  return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (np.floating,)):
    return float(o)
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, Path):
    return str(o)
  if dataclasses.is_dataclass(o) and not isinstance(o, type):
    return dataclasses.asdict(o)
  if isinstance(o, (set, tuple)):
    return list(o)
  raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def utc_now_iso() -> str:
  return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
  """Crash-safe write: temp file in the same directory + fsync + ``os.replace``."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
  try:
    with os.fdopen(fd, "wb") as f:
      f.write(data)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
  except BaseException:
    try:
      os.unlink(tmp)
    except OSError:
      pass
    raise
  return path


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> Path:
  text = json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False, default=_json_default) + "\n"
  return atomic_write_bytes(path, text.encode("utf-8"))


def safe_resolve_under_root(
    root: str | os.PathLike[str],
    rel_path: str | os.PathLike[str],
    *,
    check_symlinks: bool = True,
) -> Path:
  """Validate that ``rel_path`` stays strictly inside ``root`` without ``..``, absolute paths, or symlink escapes."""
  raw = str(rel_path)
  if not raw or not raw.strip() or "\x00" in raw:
    raise ValueError(f"Unsafe empty or NUL path entry: {rel_path!r}")
  if "\\" in raw or raw.startswith("/"):
    raise ValueError(f"Absolute or backslash path entry is forbidden under {root!s}: {rel_path!r}")
  rel = Path(raw)
  if rel.is_absolute():
    raise ValueError(f"Absolute path entry is forbidden under {root!s}: {rel_path!r}")
  parts = rel.parts
  if not parts or any(part in ("", ".", "..") for part in parts):
    raise ValueError(f"Path traversal segment forbidden under {root!s}: {rel_path!r}")

  root_path = Path(root)
  cur = root_path
  for part in parts:
    cur = cur / part
    if check_symlinks and cur.is_symlink():
      raise ValueError(f"Symlink escape forbidden under {root_path!s}: {cur!s}")

  root_resolved = root_path.resolve()
  cand_resolved = (root_path / rel).resolve()
  if cand_resolved == root_resolved or not cand_resolved.is_relative_to(root_resolved):
    raise ValueError(f"Resolved path {cand_resolved!s} escapes root {root_resolved!s}")
  return root_path / rel


def inspect_synthetic_trace_manifest(manifest_or_path: Any) -> Tuple[bool, str]:
  """Return ``(is_synthetic, reason)`` for a trace manifest dict, ``TraceManifest``, or directory/file path."""
  data: Optional[Mapping[str, Any]] = None
  if isinstance(manifest_or_path, TraceManifest):
    data = manifest_or_path.to_dict()
  elif isinstance(manifest_or_path, Mapping):
    data = manifest_or_path
  elif isinstance(manifest_or_path, (str, os.PathLike)):
    p = Path(manifest_or_path)
    if p.is_dir():
      for cand_name in (MANIFEST_FILENAME, "dataset_manifest.json", "traces/manifest.json", "traces/public/manifest.json"):
        cand = p / cand_name
        if cand.is_file():
          p = cand
          break
    if p.is_file():
      try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
          data = loaded
      except (OSError, ValueError):
        return False, ""
  if not data:
    return False, ""

  kind = str(data.get("kind", "") or data.get("manifest_kind", ""))
  if kind == SYNTHETIC_CI_SMOKE_TRACES_KIND:
    return True, f"trace manifest kind={SYNTHETIC_CI_SMOKE_TRACES_KIND!r} is strictly for offline CI smoke tests"
  if kind == "SYNTHETIC_RANDOM_TENSORS":
    return True, "observation_source kind='SYNTHETIC_RANDOM_TENSORS' is synthetic and non-certifiable"
  if bool(data.get("is_synthetic_smoke", False)):
    return True, "trace manifest is marked is_synthetic_smoke=true"
  if "certifiable" in data and data.get("certifiable") is False:
    return True, "trace manifest is marked certifiable=false"
  prov = str(data.get("pixel_provenance", ""))
  if prov == SYNTHETIC_PIXEL_PROVENANCE:
    return True, f"trace manifest pixel_provenance={SYNTHETIC_PIXEL_PROVENANCE!r} is procedural/synthetic"
  if "is_real_camera_pixels" in data and data.get("is_real_camera_pixels") is False:
    return True, "trace manifest has is_real_camera_pixels=false"

  splits = data.get("splits")
  if isinstance(splits, Mapping):
    for split_name, split_rec in splits.items():
      if isinstance(split_rec, Mapping):
        s_prov = str(split_rec.get("pixel_provenance", ""))
        s_mode = str(split_rec.get("source_mode", ""))
        if s_prov == SYNTHETIC_PIXEL_PROVENANCE or s_mode.startswith("smoke"):
          return True, f"split {split_name!r} uses synthetic/procedural traces (source_mode={s_mode!r}, pixel_provenance={s_prov!r})"
  shards = data.get("shards")
  if isinstance(shards, list):
    for sh in shards:
      if isinstance(sh, Mapping) and str(sh.get("pixel_provenance", "")) == SYNTHETIC_PIXEL_PROVENANCE:
        return True, f"shard {sh.get('episode_id', sh.get('filename', '?'))!r} has synthetic pixel_provenance"
  return False, ""


def is_synthetic_trace_manifest(manifest_or_path: Any) -> bool:
  """Boolean predicate returning ``True`` iff ``manifest_or_path`` describes synthetic/smoke traces (RT-02)."""
  return inspect_synthetic_trace_manifest(manifest_or_path)[0]


# --------------------------------------------------------------------------------------------
# Public / hidden split policy
# --------------------------------------------------------------------------------------------
def split_hash_unit(episode_id: str, seed: int = DEFAULT_SPLIT_SEED) -> float:
  """Deterministic u in [0, 1) from ``sha256(f"{seed}|{episode_id}")`` (first 8 bytes, big-endian)."""
  digest = hashlib.sha256(f"{int(seed)}|{episode_id}".encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_split(
    episode_id: str,
    seed: int = DEFAULT_SPLIT_SEED,
    hidden_fraction: float = DEFAULT_HIDDEN_FRACTION,
) -> str:
  """Seeded episode-hash split: ``hidden`` iff u < hidden_fraction else ``public``."""
  if not (0.0 <= hidden_fraction <= 1.0):
    raise ValueError(f"hidden_fraction must be in [0, 1], got {hidden_fraction}")
  return "hidden" if split_hash_unit(episode_id, seed) < hidden_fraction else "public"


def make_episode_id(source_id: str, episode_index: int) -> str:
  """Stable, cross-dataset-unique episode id, e.g. ``HuggingFaceVLA__libero/episode_000000``."""
  slug = str(source_id).strip("/").replace("/", "__").replace(" ", "_")
  return f"{slug}/episode_{int(episode_index):06d}"


def episode_filename(episode_index: int, ext: str = ".npz") -> str:
  """``episode_XXXX`` naming (zero-padded to at least 4 digits, wider when needed)."""
  return f"episode_{int(episode_index):04d}{ext}"


# --------------------------------------------------------------------------------------------
# Per-step record
# --------------------------------------------------------------------------------------------
@dataclass
class TraceStep:
  """One observation→action step of an EmbodiedX trace (all arrays owned, no views into files)."""

  t_ns: int                        # native timestamp of this frame (ns, per-episode clock)
  step: int                        # frame index within the episode (0-based, contiguous)
  images: Dict[str, np.ndarray]    # camera -> uint8[H, W, 3]
  proprio: np.ndarray              # float32[D]
  instruction: str                 # constant across the episode
  action: np.ndarray               # float32[A]  (golden / recorded action)
  episode_id: str
  source_dataset: str              # key of EMBODIEDX_TRACES_V1_SOURCES (e.g. "LIBERO") or free text
  split: str = UNASSIGNED_SPLIT

  def validate(self) -> None:
    if not isinstance(self.t_ns, (int, np.integer)) or int(self.t_ns) < 0:
      raise ValueError(f"t_ns must be a non-negative int, got {self.t_ns!r}")
    if int(self.step) < 0:
      raise ValueError(f"step must be >= 0, got {self.step}")
    for cam, img in self.images.items():
      if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"images[{cam!r}] must be uint8[H, W, 3], got {img.dtype}{img.shape}")
    if self.proprio.dtype != np.float32 or self.proprio.ndim != 1:
      raise ValueError(f"proprio must be float32[D], got {self.proprio.dtype}{self.proprio.shape}")
    if self.action.dtype != np.float32 or self.action.ndim != 1:
      raise ValueError(f"action must be float32[A], got {self.action.dtype}{self.action.shape}")
    if not isinstance(self.instruction, str):
      raise ValueError("instruction must be a str")
    if self.split not in SPLITS + (UNASSIGNED_SPLIT,):
      raise ValueError(f"split must be one of {SPLITS + (UNASSIGNED_SPLIT,)}, got {self.split!r}")


# --------------------------------------------------------------------------------------------
# Per-episode columnar trace container
# --------------------------------------------------------------------------------------------
@dataclass
class EmbodiedXTrace:
  """One episode as columnar arrays; ``trace[i]`` materialises a ``TraceStep``.

  npz layout (``to_npz`` / ``from_npz``)::

      t_ns             int64   [T]
      step             int64   [T]
      proprio          float32 [T, D]
      action           float32 [T, A]
      image__<camera>  uint8   [T, H, W, 3]   (one key per camera, cameras sorted)
      meta_json        str     JSON: schema_version, episode_id, source_dataset, split,
                               instruction, fps, cameras, license, source (dict), step_count
  """

  episode_id: str
  source_dataset: str
  split: str
  instruction: str
  fps: float
  t_ns: np.ndarray
  step: np.ndarray
  proprio: np.ndarray
  action: np.ndarray
  images: Dict[str, np.ndarray] = field(default_factory=dict)
  license: str = ""
  source: Dict[str, Any] = field(default_factory=dict)  # dataset_id, codebase_version, episode_index, ...

  # ----- basic protocol -----
  def __len__(self) -> int:
    return int(self.t_ns.shape[0])

  @property
  def cameras(self) -> Tuple[str, ...]:
    return tuple(sorted(self.images.keys()))

  @property
  def step_count(self) -> int:
    return len(self)

  def __getitem__(self, i: int) -> TraceStep:
    n = len(self)
    if i < 0:
      i += n
    if not (0 <= i < n):
      raise IndexError(f"step {i} out of range for trace with {n} steps")
    return TraceStep(
        t_ns=int(self.t_ns[i]),
        step=int(self.step[i]),
        images={cam: np.ascontiguousarray(self.images[cam][i]) for cam in self.cameras},
        proprio=np.ascontiguousarray(self.proprio[i]),
        instruction=self.instruction,
        action=np.ascontiguousarray(self.action[i]),
        episode_id=self.episode_id,
        source_dataset=self.source_dataset,
        split=self.split,
    )

  def __iter__(self) -> Iterator[TraceStep]:
    for i in range(len(self)):
      yield self[i]

  # ----- construction -----
  @classmethod
  def from_steps(
      cls,
      steps: Sequence[TraceStep],
      fps: float,
      license: str = "",
      source: Optional[Mapping[str, Any]] = None,
  ) -> "EmbodiedXTrace":
    if not steps:
      raise ValueError("cannot build a trace from zero steps")
    first = steps[0]
    cams = tuple(sorted(first.images.keys()))
    for s in steps:
      if s.episode_id != first.episode_id:
        raise ValueError("all steps must share one episode_id")
      if s.instruction != first.instruction:
        raise ValueError("instruction must be constant within an episode")
      if tuple(sorted(s.images.keys())) != cams:
        raise ValueError("all steps must have the same camera set")
    trace = cls(
        episode_id=first.episode_id,
        source_dataset=first.source_dataset,
        split=first.split,
        instruction=first.instruction,
        fps=float(fps),
        t_ns=np.asarray([int(s.t_ns) for s in steps], dtype=np.int64),
        step=np.asarray([int(s.step) for s in steps], dtype=np.int64),
        proprio=np.stack([s.proprio for s in steps]).astype(np.float32, copy=False),
        action=np.stack([s.action for s in steps]).astype(np.float32, copy=False),
        images={cam: np.stack([s.images[cam] for s in steps]).astype(np.uint8, copy=False) for cam in cams},
        license=license,
        source=dict(source or {}),
    )
    trace.validate()
    return trace

  # ----- validation -----
  def validate(self) -> None:
    n = len(self)
    if n == 0:
      raise ValueError("trace has zero steps")
    if self.t_ns.dtype != np.int64 or self.t_ns.ndim != 1:
      raise ValueError(f"t_ns must be int64[T], got {self.t_ns.dtype}{self.t_ns.shape}")
    if self.step.dtype != np.int64 or self.step.shape != (n,):
      raise ValueError(f"step must be int64[T={n}], got {self.step.dtype}{self.step.shape}")
    if n > 1 and not np.all(np.diff(self.t_ns) > 0):
      raise ValueError("t_ns must be strictly increasing (temporal order violated)")
    if n > 1 and not np.all(np.diff(self.step) > 0):
      raise ValueError("step must be strictly increasing")
    if self.proprio.dtype != np.float32 or self.proprio.ndim != 2 or self.proprio.shape[0] != n:
      raise ValueError(f"proprio must be float32[T={n}, D], got {self.proprio.dtype}{self.proprio.shape}")
    if self.action.dtype != np.float32 or self.action.ndim != 2 or self.action.shape[0] != n:
      raise ValueError(f"action must be float32[T={n}, A], got {self.action.dtype}{self.action.shape}")
    for cam, stack in self.images.items():
      if stack.dtype != np.uint8 or stack.ndim != 4 or stack.shape[0] != n or stack.shape[-1] != 3:
        raise ValueError(f"images[{cam!r}] must be uint8[T={n}, H, W, 3], got {stack.dtype}{stack.shape}")
    if not isinstance(self.instruction, str):
      raise ValueError("instruction must be a str")
    if self.split not in SPLITS + (UNASSIGNED_SPLIT,):
      raise ValueError(f"split must be one of {SPLITS + (UNASSIGNED_SPLIT,)}, got {self.split!r}")
    if not (self.fps > 0):
      raise ValueError(f"fps must be > 0, got {self.fps}")

  # ----- metadata / hashing -----
  def meta(self) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": self.episode_id,
        "source_dataset": self.source_dataset,
        "split": self.split,
        "instruction": self.instruction,
        "fps": float(self.fps),
        "cameras": list(self.cameras),
        "license": self.license,
        "source": dict(self.source),
        "step_count": len(self),
        "proprio_dim": int(self.proprio.shape[1]),
        "action_dim": int(self.action.shape[1]),
        "image_shapes": {cam: list(self.images[cam].shape[1:]) for cam in self.cameras},
    }

  def arrays(self) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {
        "t_ns": np.ascontiguousarray(self.t_ns, dtype=np.int64),
        "step": np.ascontiguousarray(self.step, dtype=np.int64),
        "proprio": np.ascontiguousarray(self.proprio, dtype=np.float32),
        "action": np.ascontiguousarray(self.action, dtype=np.float32),
    }
    for cam in self.cameras:
      out[IMAGE_KEY_PREFIX + cam] = np.ascontiguousarray(self.images[cam], dtype=np.uint8)
    return out

  def content_sha256(self) -> str:
    """Platform-stable hash over canonical meta + raw array bytes (independent of npz compression)."""
    h = hashlib.sha256()
    h.update(canonical_json(self.meta()).encode("utf-8"))
    for key, arr in sorted(self.arrays().items()):
      h.update(key.encode("utf-8"))
      h.update(arr.dtype.str.encode("ascii"))
      h.update(canonical_json(list(arr.shape)).encode("ascii"))
      h.update(arr.tobytes(order="C"))
    return h.hexdigest()

  # ----- persistence -----
  def to_npz(self, path: str | os.PathLike[str], compress: bool = True) -> Path:
    """Write the trace atomically (temp file + ``os.replace``); returns the final path."""
    self.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(self.arrays())
    payload[META_KEY] = np.array(canonical_json(self.meta()))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
      with open(tmp, "wb") as f:
        if compress:
          np.savez_compressed(f, **payload)
        else:
          np.savez(f, **payload)
        f.flush()
        os.fsync(f.fileno())
      os.replace(tmp, path)
    except BaseException:
      try:
        os.unlink(tmp)
      except OSError:
        pass
      raise
    return path

  @classmethod
  def from_npz(cls, path: str | os.PathLike[str]) -> "EmbodiedXTrace":
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(f"trace file not found: {path}")
    with np.load(path, allow_pickle=False) as z:
      if META_KEY not in z.files:
        raise ValueError(f"{path} is not an EmbodiedX trace (missing '{META_KEY}')")
      meta = json.loads(str(z[META_KEY]))
      if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {meta.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
      images = {
          k[len(IMAGE_KEY_PREFIX):]: np.ascontiguousarray(z[k]) for k in z.files if k.startswith(IMAGE_KEY_PREFIX)
      }
      trace = cls(
          episode_id=str(meta["episode_id"]),
          source_dataset=str(meta.get("source_dataset", "")),
          split=str(meta.get("split", UNASSIGNED_SPLIT)),
          instruction=str(meta.get("instruction", "")),
          fps=float(meta["fps"]),
          t_ns=np.ascontiguousarray(z["t_ns"], dtype=np.int64),
          step=np.ascontiguousarray(z["step"], dtype=np.int64),
          proprio=np.ascontiguousarray(z["proprio"], dtype=np.float32),
          action=np.ascontiguousarray(z["action"], dtype=np.float32),
          images=images,
          license=str(meta.get("license", "")),
          source=dict(meta.get("source", {})),
      )
    trace.validate()
    return trace

  # ----- transforms -----
  def subsample(self, stride: int) -> "EmbodiedXTrace":
    """Keep every ``stride``-th step (native timestamps of kept frames are preserved)."""
    if stride < 1:
      raise ValueError("stride must be >= 1")
    if stride == 1:
      return self
    sl = slice(0, None, stride)
    return dataclasses.replace(
        self,
        fps=self.fps / stride,
        t_ns=np.ascontiguousarray(self.t_ns[sl]),
        step=np.ascontiguousarray(self.step[sl]),
        proprio=np.ascontiguousarray(self.proprio[sl]),
        action=np.ascontiguousarray(self.action[sl]),
        images={cam: np.ascontiguousarray(v[sl]) for cam, v in self.images.items()},
    )

  def with_split(self, split: str) -> "EmbodiedXTrace":
    return dataclasses.replace(self, split=split)

  def select_cameras(self, cameras: Sequence[str]) -> "EmbodiedXTrace":
    missing = [c for c in cameras if c not in self.images]
    if missing:
      raise KeyError(f"cameras {missing} not in trace cameras {list(self.cameras)}")
    return dataclasses.replace(self, images={c: self.images[c] for c in cameras})


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------
@dataclass
class ManifestEpisode:
  episode_id: str
  file: str                       # path relative to the manifest directory
  steps: int
  instruction: str
  sha256: str                     # sha256 of the file bytes
  content_sha256: str             # EmbodiedXTrace.content_sha256()
  source_episode_index: int
  cameras: List[str] = field(default_factory=list)
  extra_files: Dict[str, str] = field(default_factory=dict)  # e.g. {"parquet": "episode_0000.parquet"}


@dataclass
class TraceManifest:
  """``manifest.json`` of one ``traces/<split>/`` directory."""

  source: str                               # Hub id or local directory name the traces came from
  license: str
  fps: float
  cameras: List[str]
  split: str
  episodes: List[ManifestEpisode] = field(default_factory=list)
  schema_version: str = SCHEMA_VERSION
  source_dataset: str = ""                  # EMBODIEDX_TRACES_V1_SOURCES key when known
  source_codebase_version: str = ""         # LeRobot codebase_version of the source
  source_revision: str = ""                 # Hub revision / commit when known
  native_fps: float = 0.0
  split_policy: Dict[str, Any] = field(default_factory=dict)
  created_at: str = field(default_factory=utc_now_iso)
  notes: List[str] = field(default_factory=list)
  kind: str = REAL_ROBOT_TRACES_KIND
  pixel_provenance: str = REAL_HF_PIXEL_PROVENANCE
  is_synthetic_smoke: bool = False
  certifiable: bool = True

  @property
  def episode_count(self) -> int:
    return len(self.episodes)

  @property
  def step_count(self) -> int:
    return int(sum(e.steps for e in self.episodes))

  @property
  def sha256(self) -> Dict[str, str]:
    """``relative file -> sha256`` for every file referenced by the manifest."""
    out: Dict[str, str] = {}
    for e in self.episodes:
      out[e.file] = e.sha256
      for _, rel in sorted(e.extra_files.items()):
        out.setdefault(rel, "")  # filled by pack when the sidecar is hashed
    return out

  def to_dict(self) -> Dict[str, Any]:
    d = {
        "schema_version": self.schema_version,
        "kind": self.kind,
        "pixel_provenance": self.pixel_provenance,
        "is_synthetic_smoke": bool(self.is_synthetic_smoke),
        "certifiable": bool(self.certifiable),
        "source": self.source,
        "source_dataset": self.source_dataset,
        "source_codebase_version": self.source_codebase_version,
        "source_revision": self.source_revision,
        "license": self.license,
        "fps": float(self.fps),
        "native_fps": float(self.native_fps or self.fps),
        "cameras": list(self.cameras),
        "split": self.split,
        "split_policy": dict(self.split_policy),
        "episode_count": self.episode_count,
        "step_count": self.step_count,
        "created_at": self.created_at,
        "notes": list(self.notes),
        "episodes": [dataclasses.asdict(e) for e in self.episodes],
        "sha256": {e.file: e.sha256 for e in self.episodes},
    }
    return d

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "TraceManifest":
    if d.get("schema_version") != SCHEMA_VERSION:
      raise ValueError(f"manifest schema_version {d.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    eps = [ManifestEpisode(**{k: v for k, v in e.items() if k in ManifestEpisode.__dataclass_fields__}) for e in d.get("episodes", [])]
    dummy_root = Path(".")
    for ep in eps:
      safe_resolve_under_root(dummy_root, ep.file, check_symlinks=False)
      for rel_extra in (ep.extra_files or {}).values():
        safe_resolve_under_root(dummy_root, rel_extra, check_symlinks=False)
    kind = str(d.get("kind", REAL_ROBOT_TRACES_KIND))
    is_synth = bool(d.get("is_synthetic_smoke", kind == SYNTHETIC_CI_SMOKE_TRACES_KIND))
    prov = str(d.get("pixel_provenance", SYNTHETIC_PIXEL_PROVENANCE if is_synth else REAL_HF_PIXEL_PROVENANCE))
    cert = bool(d.get("certifiable", not is_synth and prov != SYNTHETIC_PIXEL_PROVENANCE))
    return cls(
        source=str(d["source"]),
        license=str(d.get("license", "")),
        fps=float(d["fps"]),
        cameras=list(d.get("cameras", [])),
        split=str(d["split"]),
        episodes=eps,
        schema_version=str(d["schema_version"]),
        source_dataset=str(d.get("source_dataset", "")),
        source_codebase_version=str(d.get("source_codebase_version", "")),
        source_revision=str(d.get("source_revision", "")),
        native_fps=float(d.get("native_fps", 0.0) or 0.0),
        split_policy=dict(d.get("split_policy", {})),
        created_at=str(d.get("created_at", "")),
        notes=list(d.get("notes", [])),
        kind=kind,
        pixel_provenance=prov,
        is_synthetic_smoke=is_synth,
        certifiable=cert,
    )

  def save(self, directory: str | os.PathLike[str]) -> Path:
    return atomic_write_json(Path(directory) / MANIFEST_FILENAME, self.to_dict())

  @classmethod
  def load(cls, directory_or_file: str | os.PathLike[str]) -> "TraceManifest":
    p = Path(directory_or_file)
    if p.is_dir():
      p = p / MANIFEST_FILENAME
    if not p.exists():
      raise FileNotFoundError(f"manifest not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
      return cls.from_dict(json.load(f))

  def verify(self, directory: str | os.PathLike[str], deep: bool = False) -> "ManifestVerification":
    """Recompute file sha256s (and content hashes when ``deep``) against the manifest."""
    directory = Path(directory)
    mismatches: List[str] = []
    missing: List[str] = []
    checked = 0
    for e in self.episodes:
      fp = safe_resolve_under_root(directory, e.file, check_symlinks=True)
      for rel_extra in (e.extra_files or {}).values():
        safe_resolve_under_root(directory, rel_extra, check_symlinks=True)
      if not fp.exists():
        missing.append(e.file)
        continue
      checked += 1
      actual = sha256_file(fp)
      if actual != e.sha256:
        mismatches.append(f"{e.file}: file sha256 {actual[:12]}... != manifest {e.sha256[:12]}...")
        continue
      if deep:
        trace = EmbodiedXTrace.from_npz(fp)
        c = trace.content_sha256()
        if c != e.content_sha256:
          mismatches.append(f"{e.file}: content sha256 {c[:12]}... != manifest {e.content_sha256[:12]}...")
        if len(trace) != e.steps or trace.episode_id != e.episode_id or trace.split != self.split:
          mismatches.append(f"{e.file}: metadata (steps/episode_id/split) disagrees with manifest")
    return ManifestVerification(ok=not mismatches and not missing, checked=checked, mismatches=mismatches, missing=missing)


@dataclass
class ManifestVerification:
  ok: bool
  checked: int
  mismatches: List[str] = field(default_factory=list)
  missing: List[str] = field(default_factory=list)

  def summary(self) -> str:
    status = "OK" if self.ok else "FAILED"
    parts = [f"manifest verification {status}: {self.checked} file(s) checked"]
    if self.missing:
      parts.append(f"{len(self.missing)} missing: {self.missing[:5]}")
    if self.mismatches:
      parts.append(f"{len(self.mismatches)} mismatch(es): {self.mismatches[:5]}")
    return "; ".join(parts)
