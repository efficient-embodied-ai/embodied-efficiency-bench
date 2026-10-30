"""LeRobot v2.x / v3.0 dataset reader that yields ``EmbodiedXTrace`` episodes.

Verified against the live Hub layouts on 2026-10-08 (exact URLs in ``docs/DATA.md``):

v3.0 (``HuggingFaceVLA/libero``, ``lerobot/libero_10``, ``lerobot/droid_1.0.1``)::

    meta/info.json                               codebase_version="v3.0", fps, features{...},
                                                 data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                                                 video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    meta/tasks.parquet                           columns: task_index, __index_level_0__ (= task string)
    meta/episodes/chunk-000/file-000.parquet     episode_index, length, dataset_from_index, dataset_to_index,
                                                 data/chunk_index, data/file_index, [tasks: list<str>],
                                                 videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}, stats/*
    data/chunk-000/file-000.parquet              several episodes per file; observation.state list<float>,
                                                 action list<float>, timestamp (restarts at 0.0 per episode),
                                                 frame_index, episode_index, index, task_index,
                                                 image features as struct<bytes: binary(PNG), path: string>
    videos/<key>/chunk-000/file-000.mp4          AV1 yuv420p; episode segment = [from_timestamp, to_timestamp)

v2.0 (``physical-intelligence/libero``)::

    meta/info.json                               codebase_version="v2.0", total_chunks, chunks_size,
                                                 data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                                                 video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    meta/episodes.jsonl                          {"episode_index", "tasks": [...], "length"}
    meta/tasks.jsonl                             {"task_index", "task"}
    data/chunk-000/episode_000000.parquet        one episode per file; image / wrist_image struct<bytes,path>,
                                                 state fixed_size_list<float>[8], actions fixed_size_list<float>[7]

Dependency policy
-----------------
* Parquet needs ``pyarrow`` (import-guarded; clear install hint). Frame tables may alternatively be
  ``.npz`` files with the same column names (pure NumPy fallback used by the synthetic fixture), and
  episode/task metadata may be ``.jsonl``.
* Video features are decoded with ``av`` (PyAV, bundles libdav1d for AV1) when importable; otherwise
  a pre-decoded PNG directory ``images/<feature_key>/episode_XXXXXX/frame_XXXXXX.png`` (LeRobot's
  ``DEFAULT_IMAGE_PATH``; the v3 ``episode-XXXXXX/frame-XXXXXX.png`` spelling is accepted too) is used.
* PNG bytes are decoded with Pillow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np

from eeb.data.schema import (
    PINNED_HUB_REVISIONS,
    UNASSIGNED_SPLIT,
    EmbodiedXTrace,
    guess_source_dataset,
    make_episode_id,
    safe_resolve_under_root,
    source_license,
)

PINNED_REPOS: Dict[str, str] = dict(PINNED_HUB_REVISIONS)

INFO_PATH = "meta/info.json"
V2_EPISODES_JSONL = "meta/episodes.jsonl"
V2_TASKS_JSONL = "meta/tasks.jsonl"
V3_TASKS_PARQUET = "meta/tasks.parquet"
V3_EPISODES_DIR = "meta/episodes"
CAMERA_KEY_PREFIXES = ("observation.images.", "observation.image.", "observation.")
IMAGE_DIR_TEMPLATES = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",  # LeRobot v2.x DEFAULT_IMAGE_PATH
    "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png",  # v3-style spelling
)
STATE_KEY_CANDIDATES = ("observation.state", "state", "observation.proprio", "proprio")
ACTION_KEY_CANDIDATES = ("action", "actions")
INSTRUCTION_COLUMN_CANDIDATES = ("language_instruction", "task", "instruction")


class MissingDependencyError(ImportError):
  """Raised when an optional dependency is required for the requested data path."""


def require_pyarrow():
  try:
    import pyarrow.parquet as pq  # type: ignore
  except ImportError as e:  # pragma: no cover - exercised via mock in tests
    raise MissingDependencyError(
        "Reading .parquet files requires pyarrow. Install it with `pip install pyarrow` "
        "(or `pip install 'embodied-efficiency-bench[data]'`). For an always-available pure-NumPy "
        "path, point the loader at a dataset whose frame tables are .npz files with the same column "
        "names (see docs/DATA.md §fallback)."
    ) from e
  return pq


def require_av():
  try:
    import av  # type: ignore
  except ImportError as e:  # pragma: no cover - exercised via mock in tests
    raise MissingDependencyError(
        "Decoding LeRobot video features requires PyAV. Install it with `pip install av` "
        "(wheels bundle FFmpeg + libdav1d for AV1). Alternatively provide pre-decoded frames under "
        "images/<feature_key>/episode_XXXXXX/frame_XXXXXX.png (see docs/DATA.md §video)."
    ) from e
  return av


def require_pil():
  try:
    from PIL import Image  # type: ignore
  except ImportError as e:  # pragma: no cover
    raise MissingDependencyError("Decoding PNG frames requires Pillow: `pip install pillow`.") from e
  return Image


# --------------------------------------------------------------------------------------------
# Metadata dataclasses
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LeRobotFeature:
  key: str
  dtype: str                      # "image" | "video" | "float32" | "int64" | "string" | "bool" | ...
  shape: Tuple[int, ...]
  names: Any = None
  info: Dict[str, Any] = field(default_factory=dict)  # video_info for dtype == "video"

  @property
  def is_camera(self) -> bool:
    return self.dtype in ("image", "video")


@dataclass(frozen=True)
class VideoRef:
  key: str
  chunk_index: int
  file_index: int
  from_timestamp: float
  to_timestamp: float


@dataclass
class LeRobotEpisode:
  episode_index: int
  length: int
  tasks: List[str] = field(default_factory=list)
  data_chunk_index: int = 0
  data_file_index: int = 0
  dataset_from_index: int = -1
  dataset_to_index: int = -1
  videos: Dict[str, VideoRef] = field(default_factory=dict)


@dataclass
class LeRobotInfo:
  """Parsed ``meta/info.json``."""

  root: Path
  codebase_version: str
  fps: float
  features: Dict[str, LeRobotFeature]
  data_path: str
  video_path: str
  chunks_size: int = 1000
  robot_type: str = ""
  total_episodes: int = 0
  total_frames: int = 0
  raw: Dict[str, Any] = field(default_factory=dict)

  @classmethod
  def load(cls, root: str | os.PathLike[str]) -> "LeRobotInfo":
    root = Path(root)
    p = root / INFO_PATH
    if not p.exists():
      raise FileNotFoundError(
          f"{p} not found: {root} is not a LeRobot dataset root (expected meta/info.json). "
          "Download one with `eeb data download <hub_id> --episodes 2` first."
      )
    with open(p, "r", encoding="utf-8") as f:
      raw = json.load(f)
    feats: Dict[str, LeRobotFeature] = {}
    for key, spec in (raw.get("features") or {}).items():
      shape = tuple(int(x) for x in (spec.get("shape") or ()))
      info = dict(spec.get("info") or spec.get("video_info") or {})
      feats[key] = LeRobotFeature(key=key, dtype=str(spec.get("dtype", "")), shape=shape, names=spec.get("names"), info=info)
    cv = str(raw.get("codebase_version", ""))
    major = cv.lstrip("v").split(".")[0] if cv else ""
    is_v3 = major.isdigit() and int(major) >= 3
    default_data = (
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        if is_v3 else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    default_video = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        if is_v3 else "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    fps = raw.get("fps")
    if fps is None:
      raise ValueError(f"{p}: missing required field 'fps'")
    return cls(
        root=root,
        codebase_version=cv,
        fps=float(fps),
        features=feats,
        data_path=str(raw.get("data_path") or default_data),
        video_path=str(raw.get("video_path") or default_video),
        chunks_size=int(raw.get("chunks_size") or 1000),
        robot_type=str(raw.get("robot_type") or ""),
        total_episodes=int(raw.get("total_episodes") or 0),
        total_frames=int(raw.get("total_frames") or 0),
        raw=raw,
    )

  @property
  def layout(self) -> str:
    """``"v3"`` when codebase_version >= 3 or a ``meta/episodes/`` directory exists, else ``"v2"``."""
    major = self.codebase_version.lstrip("v").split(".")[0] if self.codebase_version else ""
    if major.isdigit() and int(major) >= 3:
      return "v3"
    if (self.root / V3_EPISODES_DIR).is_dir():
      return "v3"
    return "v2"

  @property
  def image_keys(self) -> Tuple[str, ...]:
    return tuple(k for k, f in self.features.items() if f.dtype == "image")

  @property
  def video_keys(self) -> Tuple[str, ...]:
    return tuple(k for k, f in self.features.items() if f.dtype == "video")

  @property
  def camera_keys(self) -> Tuple[str, ...]:
    return tuple(k for k, f in self.features.items() if f.is_camera)

  @staticmethod
  def camera_name(key: str) -> str:
    for pre in CAMERA_KEY_PREFIXES:
      if key.startswith(pre) and len(key) > len(pre):
        return key[len(pre):]
    return key

  def camera_key(self, name: str) -> str:
    if name in self.features and self.features[name].is_camera:
      return name
    for k in self.camera_keys:
      if self.camera_name(k) == name:
        return k
    raise KeyError(f"camera {name!r} not found; available cameras: {[self.camera_name(k) for k in self.camera_keys]}")

  def resolve_state_key(self, override: Optional[str] = None) -> str:
    if override:
      if override not in self.features:
        raise KeyError(f"state key {override!r} not in features {list(self.features)}")
      return override
    for c in STATE_KEY_CANDIDATES:
      if c in self.features:
        return c
    for k in self.features:
      if k.startswith("observation.state"):
        return k
    raise KeyError(f"no proprioception feature found among {list(self.features)}; pass state_key=...")

  def resolve_action_key(self, override: Optional[str] = None) -> str:
    if override:
      if override not in self.features:
        raise KeyError(f"action key {override!r} not in features {list(self.features)}")
      return override
    for c in ACTION_KEY_CANDIDATES:
      if c in self.features:
        return c
    raise KeyError(f"no action feature found among {list(self.features)}; pass action_key=...")

  def format_path(self, template: str, **kw: Any) -> str:
    return template.format(**kw)


# --------------------------------------------------------------------------------------------
# Frame tables (parquet / npz) with a uniform column interface
# --------------------------------------------------------------------------------------------
class FrameTable:
  """Column store: ``name -> np.ndarray | list``; rows are frames."""

  def __init__(self, columns: Mapping[str, Any], num_rows: int, path: Optional[Path] = None):
    self._cols: Dict[str, Any] = dict(columns)
    self.num_rows = int(num_rows)
    self.path = path

  @property
  def names(self) -> Tuple[str, ...]:
    return tuple(self._cols.keys())

  def __contains__(self, name: str) -> bool:
    return name in self._cols

  def column(self, name: str) -> Any:
    if name not in self._cols:
      raise KeyError(f"column {name!r} not in frame table {self.path} (have {list(self._cols)})")
    return self._cols[name]

  def take(self, indices: Sequence[int]) -> "FrameTable":
    idx = np.asarray(indices, dtype=np.int64)
    out: Dict[str, Any] = {}
    for k, v in self._cols.items():
      if isinstance(v, np.ndarray):
        out[k] = v[idx]
      else:
        out[k] = [v[int(i)] for i in idx]
    return FrameTable(out, len(idx), self.path)


def _pyarrow_column_to_python(col: Any) -> Any:
  """Primitive columns -> np.ndarray; list/struct/string columns -> Python lists."""
  import pyarrow as pa  # type: ignore  # noqa: F401  (pyarrow present when this is called)

  t = col.type
  if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t):
    return col.to_numpy(zero_copy_only=False)
  return col.to_pylist()


def read_frame_table(path: str | os.PathLike[str], columns: Optional[Sequence[str]] = None) -> FrameTable:
  """Read a frame table (``.parquet`` via pyarrow, ``.npz`` via NumPy). Unknown columns are ignored."""
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"frame table not found: {path}")
  if path.suffix == ".parquet":
    pq = require_pyarrow()
    pf = pq.ParquetFile(str(path))
    available = list(pf.schema_arrow.names)
    cols = [c for c in (columns or available) if c in available]
    table = pf.read(columns=cols)
    data = {name: _pyarrow_column_to_python(table.column(name)) for name in cols}
    return FrameTable(data, table.num_rows, path)
  if path.suffix == ".npz":
    with np.load(path, allow_pickle=False) as z:
      available = list(z.files)
      cols = [c for c in (columns or available) if c in available]
      data = {name: np.ascontiguousarray(z[name]) for name in cols}
    n = len(next(iter(data.values()))) if data else 0
    return FrameTable(data, n, path)
  raise ValueError(f"unsupported frame table format {path.suffix!r} ({path}); expected .parquet or .npz")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
  rows: List[Dict[str, Any]] = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def _read_records(path: Path, drop_prefixes: Sequence[str] = ("stats/",)) -> List[Dict[str, Any]]:
  """Rows of a small metadata table (``.parquet`` or ``.jsonl``) as dicts, excluding bulky ``stats/*``."""
  if path.suffix == ".jsonl":
    return _read_jsonl(path)
  if path.suffix == ".parquet":
    pq = require_pyarrow()
    pf = pq.ParquetFile(str(path))
    cols = [c for c in pf.schema_arrow.names if not any(c.startswith(p) for p in drop_prefixes)]
    return pf.read(columns=cols).to_pylist()
  raise ValueError(f"unsupported metadata table format: {path}")


# --------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------
class LeRobotDataset:
  """Random access to the episodes of a local LeRobot v2/v3 dataset root."""

  def __init__(
      self,
      root: str | os.PathLike[str],
      *,
      state_key: Optional[str] = None,
      action_key: Optional[str] = None,
      image_dir: Optional[str | os.PathLike[str]] = None,
      video_tolerance_s: Optional[float] = None,
      source_id: Optional[str] = None,
  ):
    self.root = Path(root)
    self.info = LeRobotInfo.load(self.root)
    self.state_key = self.info.resolve_state_key(state_key)
    self.action_key = self.info.resolve_action_key(action_key)
    self.image_dir = Path(image_dir) if image_dir else self.root
    self.video_tolerance_s = float(video_tolerance_s) if video_tolerance_s else 0.5 / self.info.fps
    self.source_id = source_id or self._infer_source_id()
    self.tasks: Dict[int, str] = self._load_tasks()
    self.episodes: List[LeRobotEpisode] = self._load_episodes()
    self._by_index: Dict[int, LeRobotEpisode] = {e.episode_index: e for e in self.episodes}

  # ----- source naming -----
  def _infer_source_id(self) -> str:
    # `eeb data download` writes <cache>/<org>__<name>; HF snapshot caches look like datasets--org--name/snapshots/<rev>
    for parent in (self.root, *self.root.parents):
      n = parent.name
      if n.startswith("datasets--"):
        return n[len("datasets--"):].replace("--", "/")
    n = self.root.name
    return n.replace("__", "/") if "__" in n else n

  # ----- metadata loading -----
  def _load_tasks(self) -> Dict[int, str]:
    pq_path = self.root / V3_TASKS_PARQUET
    jsonl_path = self.root / V2_TASKS_JSONL
    rows: List[Dict[str, Any]] = []
    if pq_path.exists():
      try:
        rows = _read_records(pq_path)
      except MissingDependencyError:
        if not jsonl_path.exists():
          raise
    if not rows and jsonl_path.exists():
      rows = _read_jsonl(jsonl_path)
    tasks: Dict[int, str] = {}
    for r in rows:
      text = r.get("task", r.get("__index_level_0__", r.get("task_name")))
      if "task_index" in r and text is not None:
        tasks[int(r["task_index"])] = str(text)
    return tasks

  def _episode_meta_files(self) -> List[Path]:
    v3_dir = self.root / V3_EPISODES_DIR
    if v3_dir.is_dir():
      pqs = sorted(Path(p) for p in glob.glob(str(v3_dir / "**" / "*.parquet"), recursive=True))
      jls = sorted(Path(p) for p in glob.glob(str(v3_dir / "**" / "*.jsonl"), recursive=True))
      if pqs:
        try:
          require_pyarrow()
          return pqs
        except MissingDependencyError:
          if jls:
            return jls
          raise
      if jls:
        return jls
    for cand in (self.root / "meta" / "episodes.parquet", self.root / V2_EPISODES_JSONL):
      if cand.exists():
        return [cand]
    raise FileNotFoundError(
        f"no episode metadata under {self.root / 'meta'} (expected meta/episodes/chunk-*/file-*.parquet|.jsonl "
        "or meta/episodes.jsonl)"
    )

  def _load_episodes(self) -> List[LeRobotEpisode]:
    rows: List[Dict[str, Any]] = []
    for p in self._episode_meta_files():
      rows.extend(_read_records(p))
    eps: List[LeRobotEpisode] = []
    cs = max(1, self.info.chunks_size)
    for r in rows:
      ei = int(r["episode_index"])
      length = int(r.get("length", 0) or 0)
      tasks = r.get("tasks")
      tasks_list = [str(t) for t in tasks] if isinstance(tasks, (list, tuple)) else ([str(tasks)] if tasks else [])
      videos: Dict[str, VideoRef] = {}
      if self.info.layout == "v3":
        dci = int(r.get("data/chunk_index", r.get("data_chunk_index", ei // cs)))
        dfi = int(r.get("data/file_index", r.get("data_file_index", ei)))
        for key in self.info.video_keys:
          pre = f"videos/{key}/"
          if pre + "chunk_index" in r:
            videos[key] = VideoRef(
                key=key,
                chunk_index=int(r[pre + "chunk_index"]),
                file_index=int(r[pre + "file_index"]),
                from_timestamp=float(r.get(pre + "from_timestamp", 0.0) or 0.0),
                to_timestamp=float(r.get(pre + "to_timestamp", length / self.info.fps) or (length / self.info.fps)),
            )
      else:
        dci, dfi = ei // cs, ei
        for key in self.info.video_keys:
          videos[key] = VideoRef(key=key, chunk_index=ei // cs, file_index=ei, from_timestamp=0.0, to_timestamp=length / self.info.fps)
      eps.append(
          LeRobotEpisode(
              episode_index=ei,
              length=length,
              tasks=tasks_list,
              data_chunk_index=dci,
              data_file_index=dfi,
              dataset_from_index=int(r.get("dataset_from_index", -1)),
              dataset_to_index=int(r.get("dataset_to_index", -1)),
              videos=videos,
          )
      )
    eps.sort(key=lambda e: e.episode_index)
    return eps

  # ----- path resolution -----
  def episode(self, episode_index: int) -> LeRobotEpisode:
    try:
      return self._by_index[int(episode_index)]
    except KeyError:
      raise KeyError(f"episode {episode_index} not in dataset (have {len(self.episodes)} episodes)") from None

  def data_file_rel(self, ep: LeRobotEpisode) -> str:
    """Repo-relative frame-table path from the ``data_path`` template (what the Hub serves)."""
    return self.info.format_path(
        self.info.data_path,
        chunk_index=ep.data_chunk_index,
        file_index=ep.data_file_index,
        episode_chunk=ep.data_chunk_index,
        episode_index=ep.episode_index,
    )

  def data_file(self, ep: LeRobotEpisode, must_exist: bool = True) -> Path:
    p = self.root / self.data_file_rel(ep)
    if p.exists():
      return p
    for alt in (p.with_suffix(".npz"),):  # pure-NumPy fallback written by the synthetic fixture
      if alt.exists():
        return alt
    if must_exist:
      raise FileNotFoundError(
          f"frame table for episode {ep.episode_index} not found: {p} (nor .npz fallback). "
          f"Download it with `eeb data download {self.source_id} --episodes {ep.episode_index + 1}`."
      )
    return p

  def video_file_rel(self, ep: LeRobotEpisode, key: str) -> str:
    ref = ep.videos.get(key)
    if ref is None:
      raise KeyError(f"episode {ep.episode_index} has no video reference for {key!r}")
    return self.info.format_path(
        self.info.video_path,
        video_key=key,
        chunk_index=ref.chunk_index,
        file_index=ref.file_index,
        episode_chunk=ref.chunk_index,
        episode_index=ep.episode_index,
    )

  def video_file(self, ep: LeRobotEpisode, key: str) -> Path:
    return self.root / self.video_file_rel(ep, key)

  def has_local_data(self, ep: LeRobotEpisode) -> bool:
    return self.data_file(ep, must_exist=False).exists()

  def available_episode_indices(self) -> List[int]:
    return [e.episode_index for e in self.episodes if self.has_local_data(e)]

  # ----- frame access -----
  def read_frames(self, ep: LeRobotEpisode, columns: Optional[Sequence[str]] = None) -> FrameTable:
    """Rows of ``ep`` (only), sorted by ``frame_index``; ``columns=None`` reads everything."""
    path = self.data_file(ep)
    want = None if columns is None else list(dict.fromkeys(list(columns) + ["episode_index", "frame_index"]))
    table = read_frame_table(path, want)
    if "episode_index" in table:
      ep_col = np.asarray(table.column("episode_index")).astype(np.int64)
      rows = np.nonzero(ep_col == ep.episode_index)[0]
    else:  # one-episode-per-file layouts without the column
      rows = np.arange(table.num_rows)
    if rows.size == 0:
      data_root = self.root / "data"
      if data_root.is_dir():
        cands = sorted(Path(p) for p in glob.glob(str(data_root / "**" / "*.parquet"), recursive=True))
        cands += sorted(Path(p) for p in glob.glob(str(data_root / "**" / "*.npz"), recursive=True))
        for alt_path in cands:
          if alt_path == path:
            continue
          alt_table = read_frame_table(alt_path, want)
          if "episode_index" in alt_table:
            alt_ep = np.asarray(alt_table.column("episode_index")).astype(np.int64)
            alt_rows = np.nonzero(alt_ep == ep.episode_index)[0]
            if alt_rows.size > 0:
              table = alt_table
              path = alt_path
              rows = alt_rows
              break
      if rows.size == 0:
        raise ValueError(f"{path} contains no rows for episode {ep.episode_index}")
    if "frame_index" in table:
      fi = np.asarray(table.column("frame_index")).astype(np.int64)[rows]
      rows = rows[np.argsort(fi, kind="stable")]
    return table.take(rows)

  def _frame_instructions(self, table: FrameTable, ep: LeRobotEpisode) -> List[str]:
    if "task_index" in table and self.tasks:
      return [self.tasks.get(int(ti), "") for ti in np.asarray(table.column("task_index")).tolist()]
    for col in INSTRUCTION_COLUMN_CANDIDATES:
      if col in table:
        return [str(x) for x in list(table.column(col))]
    if ep.tasks:
      return [ep.tasks[0]] * table.num_rows
    return [""] * table.num_rows

  # ----- image decoding -----
  @staticmethod
  def decode_image_value(value: Any, root: Optional[Path] = None) -> np.ndarray:
    """One frame from an embedded ``{bytes, path}`` struct, raw bytes, a relative PNG path, or an array."""
    if isinstance(value, np.ndarray):
      if value.dtype != np.uint8:
        raise ValueError(f"raw image arrays must be uint8, got {value.dtype}")
      return value
    Image = require_pil()
    data: Optional[bytes] = None
    if isinstance(value, dict):
      data = value.get("bytes")
      if data is None and value.get("path"):
        raw_p = str(value["path"])
        if root is not None:
          p = safe_resolve_under_root(root, raw_p, check_symlinks=True)
        else:
          p = Path(raw_p)
          if ".." in p.parts or p.is_symlink():
            raise ValueError(f"Unsafe image path entry: {raw_p!r}")
        with open(p, "rb") as f:
          data = f.read()
    elif isinstance(value, (bytes, bytearray, memoryview)):
      data = bytes(value)
    elif isinstance(value, (str, os.PathLike)):
      raw_p = str(value)
      if root is not None:
        p = safe_resolve_under_root(root, raw_p, check_symlinks=True)
      else:
        p = Path(raw_p)
        if ".." in p.parts or p.is_symlink():
          raise ValueError(f"Unsafe image path entry: {raw_p!r}")
      with open(p, "rb") as f:
        data = f.read()
    if data is None:
      raise ValueError(f"cannot decode image value of type {type(value).__name__}")
    with Image.open(io.BytesIO(data)) as im:
      return np.asarray(im.convert("RGB"), dtype=np.uint8)

  def _png_dir_frames(self, ep: LeRobotEpisode, key: str, frame_indices: Sequence[int]) -> Optional[np.ndarray]:
    for tmpl in IMAGE_DIR_TEMPLATES:
      rel0 = tmpl.format(image_key=key, episode_index=ep.episode_index, frame_index=int(frame_indices[0]))
      first = safe_resolve_under_root(self.image_dir, rel0, check_symlinks=True)
      if first.exists():
        frames = [
            self.decode_image_value(
                tmpl.format(image_key=key, episode_index=ep.episode_index, frame_index=int(fi)),
                root=self.image_dir,
            )
            for fi in frame_indices
        ]
        return np.stack(frames).astype(np.uint8, copy=False)
    return None

  def episode_images(
      self,
      ep: LeRobotEpisode,
      key: str,
      table: FrameTable,
      frame_indices: Sequence[int],
      timestamps_s: Sequence[float],
  ) -> np.ndarray:
    """``uint8[T, H, W, 3]`` for one camera feature, trying embedded → PNG dir → video."""
    feat = self.info.features[key]
    if key in table:
      col = table.column(key)
      if isinstance(col, np.ndarray) and col.ndim == 4:
        return np.ascontiguousarray(col, dtype=np.uint8)
      return np.stack([self.decode_image_value(v, self.root) for v in list(col)]).astype(np.uint8, copy=False)
    png = self._png_dir_frames(ep, key, frame_indices)
    if png is not None:
      return png
    if feat.dtype == "video":
      vpath = self.video_file(ep, key)
      if not vpath.exists():
        raise FileNotFoundError(
            f"video for camera {key!r} of episode {ep.episode_index} not found at {vpath}. Either download it "
            f"(`eeb data download {self.source_id} --episodes {ep.episode_index + 1} --videos`), pre-decode frames to "
            f"images/{key}/episode_{ep.episode_index:06d}/frame_XXXXXX.png, or pass decode_images=False."
        )
      ref = ep.videos[key]
      targets = [ref.from_timestamp + float(t) for t in timestamps_s]
      return decode_video_frames(vpath, targets, tolerance_s=self.video_tolerance_s)
    raise FileNotFoundError(
        f"no frames for image feature {key!r} of episode {ep.episode_index}: not embedded in {table.path} and no "
        f"images/{key}/episode_{ep.episode_index:06d}/ directory"
    )

  # ----- episode -> trace -----
  def load_episode(
      self,
      episode_index: int,
      *,
      cameras: Optional[Sequence[str]] = None,
      decode_images: bool = True,
      split: str = UNASSIGNED_SPLIT,
      source_dataset: Optional[str] = None,
      strict_instruction: bool = True,
  ) -> EmbodiedXTrace:
    """Materialise one episode as an ``EmbodiedXTrace`` (temporal order + native timestamps preserved)."""
    ep = self.episode(episode_index)
    cam_keys: List[str] = []
    if decode_images:
      names = list(cameras) if cameras is not None else [self.info.camera_name(k) for k in self.info.camera_keys]
      cam_keys = [self.info.camera_key(n) for n in names]
    columns = [self.state_key, self.action_key, "timestamp", "frame_index", "episode_index", "index", "task_index"]
    columns += list(INSTRUCTION_COLUMN_CANDIDATES)
    columns += cam_keys  # embedded-image columns; absent for video features (ignored by read_frame_table)
    table = self.read_frames(ep, columns)
    n = table.num_rows
    if ep.length and n != ep.length:
      raise ValueError(
          f"episode {ep.episode_index}: {n} rows in {table.path} but meta says length={ep.length} "
          "(partial download or corrupted file)"
      )
    frame_idx = (
        np.asarray(table.column("frame_index")).astype(np.int64) if "frame_index" in table else np.arange(n, dtype=np.int64)
    )
    if "timestamp" in table:
      ts_raw = np.asarray(table.column("timestamp"), dtype=np.float64).reshape(-1)
    else:
      ts_raw = frame_idx.astype(np.float64) / self.info.fps
    ts, t_ns, timestamp_policy = normalize_timestamps(ts_raw, self.info.fps)
    proprio = _as_2d_float32(table.column(self.state_key), n, self.state_key)
    action = _as_2d_float32(table.column(self.action_key), n, self.action_key)
    instructions = self._frame_instructions(table, ep)
    instruction = instructions[0] if instructions else ""
    notes: List[str] = []
    if any(s != instruction for s in instructions):
      msg = (
          f"episode {ep.episode_index}: instruction changes within the episode "
          f"({len(set(instructions))} distinct values); EmbodiedX traces require one instruction per episode"
      )
      if strict_instruction:
        raise ValueError(msg + " (pass strict_instruction=False to keep the first one)")
      warnings.warn(msg + "; keeping the first", RuntimeWarning, stacklevel=2)
      notes.append("instruction_not_constant")
    images: Dict[str, np.ndarray] = {}
    for key in cam_keys:
      images[self.info.camera_name(key)] = self.episode_images(ep, key, table, frame_idx.tolist(), ts.tolist())
    src_dataset = source_dataset or guess_source_dataset(self.source_id) or ""
    trace = EmbodiedXTrace(
        episode_id=make_episode_id(self.source_id, ep.episode_index),
        source_dataset=src_dataset,
        split=split,
        instruction=instruction,
        fps=self.info.fps,
        t_ns=t_ns,
        step=frame_idx,
        proprio=proprio,
        action=action,
        images=images,
        license=source_license(src_dataset),
        source={
            "dataset_id": self.source_id,
            "codebase_version": self.info.codebase_version,
            "layout": self.info.layout,
            "robot_type": self.info.robot_type,
            "episode_index": ep.episode_index,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "camera_keys": {self.info.camera_name(k): k for k in cam_keys},
            "data_file": str(table.path.relative_to(self.root)) if table.path else "",
            "timestamp_policy": timestamp_policy,
            "notes": notes,
        },
    )
    trace.validate()
    return trace

  def iter_episodes(
      self,
      *,
      max_episodes: Optional[int] = None,
      episode_indices: Optional[Sequence[int]] = None,
      skip_missing: bool = True,
      **load_kwargs: Any,
  ) -> Iterator[EmbodiedXTrace]:
    """Yield traces in ascending episode order; ``skip_missing`` ignores episodes without local frame tables."""
    indices = list(episode_indices) if episode_indices is not None else [e.episode_index for e in self.episodes]
    produced = 0
    for ei in indices:
      ep = self.episode(ei)
      if skip_missing and not self.has_local_data(ep):
        continue
      yield self.load_episode(ei, **load_kwargs)
      produced += 1
      if max_episodes is not None and produced >= max_episodes:
        break

  def summary(self) -> Dict[str, Any]:
    return {
        "root": str(self.root),
        "source_id": self.source_id,
        "codebase_version": self.info.codebase_version,
        "layout": self.info.layout,
        "robot_type": self.info.robot_type,
        "fps": self.info.fps,
        "total_episodes_meta": self.info.total_episodes,
        "episodes_listed": len(self.episodes),
        "episodes_with_local_data": len(self.available_episode_indices()),
        "tasks": len(self.tasks),
        "cameras": {self.info.camera_name(k): {"key": k, "dtype": self.info.features[k].dtype, "shape": list(self.info.features[k].shape)} for k in self.info.camera_keys},
        "state_key": self.state_key,
        "state_dim": list(self.info.features[self.state_key].shape),
        "action_key": self.action_key,
        "action_dim": list(self.info.features[self.action_key].shape),
    }


def _as_2d_float32(col: Any, n: int, name: str) -> np.ndarray:
  arr = np.asarray(col if not isinstance(col, list) else [np.asarray(v, dtype=np.float32).reshape(-1) for v in col], dtype=np.float32)
  if arr.ndim == 1:
    arr = arr.reshape(n, -1)
  if arr.ndim != 2 or arr.shape[0] != n:
    raise ValueError(f"column {name!r} must be [T={n}, D], got {arr.shape}")
  return np.ascontiguousarray(arr, dtype=np.float32)


GRID_SNAP_TOLERANCE = 1e-3  # fraction of a frame period


def normalize_timestamps(ts_raw: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray, str]:
  """``(seconds_f64, t_ns_int64, policy)`` from LeRobot's float32 ``timestamp`` column.

  LeRobot defines ``timestamp = frame_index / fps`` but stores it as float32 (e.g. ``21.299999237``
  for frame 213 at 10 fps). When every timestamp sits on the fps grid (within 0.1 % of a frame
  period) we snap to the grid so ``t_ns`` is exact (``fps_grid_snapped``); otherwise the native
  values are kept and rounded to 1 µs (``native_rounded_us``).
  """
  ts_raw = np.asarray(ts_raw, dtype=np.float64).reshape(-1)
  k = ts_raw * float(fps)
  k_round = np.round(k)
  if ts_raw.size and np.all(np.abs(k - k_round) < GRID_SNAP_TOLERANCE):
    k_int = k_round.astype(np.int64)
    t_ns = np.round(k_int.astype(np.float64) * (1e9 / float(fps))).astype(np.int64)
    return k_int.astype(np.float64) / float(fps), t_ns, "fps_grid_snapped"
  t_ns = (np.round(ts_raw * 1e6).astype(np.int64)) * 1000
  return ts_raw, t_ns, "native_rounded_us"


# --------------------------------------------------------------------------------------------
# Video decoding (PyAV)
# --------------------------------------------------------------------------------------------
def decode_video_frames(
    path: str | os.PathLike[str],
    timestamps_s: Sequence[float],
    tolerance_s: float,
) -> np.ndarray:
  """Decode the frames nearest to ``timestamps_s`` (seconds, ascending) as ``uint8[T, H, W, 3]``.

  Seeks ~1 s before the first target (keyframe-backward), decodes sequentially until the last target
  and picks the nearest decoded frame per target; raises if any target has no frame within
  ``tolerance_s``.
  """
  av = require_av()
  targets = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
  if targets.size == 0:
    raise ValueError("timestamps_s is empty")
  if np.any(np.diff(targets) < 0):
    raise ValueError("timestamps_s must be ascending")
  first, last = float(targets[0]), float(targets[-1])
  decoded_t: List[float] = []
  decoded: List[np.ndarray] = []
  with av.open(str(path)) as container:
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if first > 1.0:
      seek_s = max(first - 1.0, 0.0)
      container.seek(int(seek_s / stream.time_base), stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
      t = frame.time
      if t is None:
        continue
      if t < first - tolerance_s:
        continue
      if t > last + tolerance_s:
        break
      decoded_t.append(float(t))
      decoded.append(frame.to_ndarray(format="rgb24"))
  if not decoded:
    raise ValueError(f"{path}: no frames decoded in [{first - tolerance_s:.4f}, {last + tolerance_s:.4f}] s")
  times = np.asarray(decoded_t)
  out: List[np.ndarray] = []
  for tgt in targets:
    j = int(np.argmin(np.abs(times - tgt)))
    if abs(times[j] - tgt) > tolerance_s:
      raise ValueError(
          f"{path}: nearest decoded frame to t={tgt:.4f}s is {times[j]:.4f}s (> tolerance {tolerance_s:.4f}s)"
      )
    out.append(decoded[j])
  return np.stack(out).astype(np.uint8, copy=False)


def detect_layout(root: str | os.PathLike[str]) -> str:
  """``"v3"`` or ``"v2"`` for a local dataset root (raises if not a LeRobot dataset)."""
  return LeRobotInfo.load(root).layout
