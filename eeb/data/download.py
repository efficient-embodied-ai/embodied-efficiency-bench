"""Hugging Face Hub download helpers for LeRobot datasets (``huggingface_hub`` is import-guarded).

* ``download(dataset_id, ...)`` wraps ``huggingface_hub.snapshot_download`` and writes a plain
  LeRobot root at ``<cache_dir>/<org>__<name>/`` (default cache ``./data_cache/``).
* ``episodes=N`` first fetches ``meta/`` only, then plans the exact ``data/`` (and optionally
  ``videos/``) files that hold the first ``N`` listed episodes, so smoke tests pull a few files
  instead of the whole dataset (``HuggingFaceVLA/libero`` is 377 x ~90 MB parquet files;
  ``lerobot/libero_10`` videos are ~380 MB per camera per chunk file).
* ``hub_tree`` / ``hub_info_json`` use only the standard library so ``eeb data inspect --hub <id>``
  can verify an on-Hub layout without installing ``huggingface_hub``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import urllib.error
import urllib.request

from eeb.data.lerobot_loader import LeRobotDataset, MissingDependencyError
from eeb.data.schema import PINNED_HUB_REVISIONS, safe_resolve_under_root

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "eeb" / "lerobot"
META_PATTERNS: Sequence[str] = ("meta/*", "meta/**", "README.md", "LICENSE", "LICENSE.*")
HF_API_TREE = "https://huggingface.co/api/datasets/{dataset_id}/tree/{revision}/{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{path}"
USER_AGENT = "embodied-efficiency-bench/eeb.data (https://github.com/efficient-embodied-ai/embodied-efficiency-bench)"


def require_hf_hub():
  try:
    import huggingface_hub  # type: ignore
  except ImportError as e:  # pragma: no cover - exercised via mock in tests
    raise MissingDependencyError(
        "Downloading from the Hugging Face Hub requires huggingface_hub. Install it with "
        "`pip install huggingface_hub` (or `pip install 'embodied-efficiency-bench[data]'`). "
        "Set HF_TOKEN for gated datasets / higher rate limits."
    ) from e
  return huggingface_hub


def local_dir_for(dataset_id: str, cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> Path:
  """``<cache_dir>/<org>__<name>`` (the loader maps ``__`` back to ``/`` for the source id)."""
  raw = str(dataset_id)
  if not raw or not raw.strip() or "\x00" in raw or "\\" in raw or raw.startswith("/"):
    raise ValueError(f"Unsafe or empty dataset_id: {dataset_id!r}")
  segs = [s for s in raw.strip().split("/") if s]
  if not segs or any(s in (".", "..") for s in segs):
    raise ValueError(f"Unsafe path segment in dataset_id: {dataset_id!r}")
  slug = "__".join(segs)
  return safe_resolve_under_root(Path(cache_dir), slug, check_symlinks=True)


# --------------------------------------------------------------------------------------------
# Offline file planner for `--episodes N`
# --------------------------------------------------------------------------------------------
@dataclass
class DownloadPlan:
  dataset_id: str
  local_dir: Path
  episode_indices: List[int]
  data_files: List[str] = field(default_factory=list)
  video_files: List[str] = field(default_factory=list)

  @property
  def allow_patterns(self) -> List[str]:
    return list(dict.fromkeys(self.data_files + self.video_files))


def plan_episode_files(
    root: str | os.PathLike[str],
    episodes: int,
    include_videos: bool = False,
    dataset_id: str = "",
) -> DownloadPlan:
  """Relative data/video files holding the first ``episodes`` listed episodes of a local ``meta/``.

  Pure function over the on-disk metadata (no network); used by ``download(..., episodes=N)`` and
  unit-tested offline against the synthetic fixture.
  """
  if episodes < 1:
    raise ValueError("episodes must be >= 1")
  ds = LeRobotDataset(root, source_id=dataset_id or None)
  chosen = ds.episodes[:episodes]
  data_files: List[str] = []
  video_files: List[str] = []
  for ep in chosen:
    # Template-derived repo-relative paths (what the Hub serves), independent of any local
    # ``.npz`` fallback file that may already sit next to the parquet path.
    rel = ds.data_file_rel(ep)
    if rel not in data_files:
      data_files.append(rel)
    if include_videos:
      for key in ds.info.video_keys:
        if key not in ep.videos:
          continue  # episode row carries no segment reference for this camera (nothing to fetch)
        vrel = ds.video_file_rel(ep, key)
        if vrel not in video_files:
          video_files.append(vrel)
  return DownloadPlan(
      dataset_id=dataset_id or ds.source_id,
      local_dir=ds.root,
      episode_indices=[e.episode_index for e in chosen],
      data_files=data_files,
      video_files=video_files,
  )


# --------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------
def download(
    dataset_id: str,
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    *,
    episodes: Optional[int] = None,
    include_videos: bool = False,
    token: Optional[str] = None,
    local_dir: Optional[str | os.PathLike[str]] = None,
    max_workers: int = 8,
) -> Path:
  """Download (part of) a LeRobot dataset repo into a plain local root and return that path.

  Args:
    dataset_id: Hub id, e.g. ``HuggingFaceVLA/libero``.
    revision: git revision / commit (``None`` = default branch).
    allow_patterns: glob patterns forwarded to ``snapshot_download`` (``None`` = everything, unless
      ``episodes`` is given).
    cache_dir: parent directory; the dataset lands in ``<cache_dir>/<org>__<name>/``.
    episodes: fetch ``meta/`` plus only the frame tables of the first N listed episodes.
    include_videos: with ``episodes``, also fetch the video files those episodes reference.
    token: HF token (defaults to ``HF_TOKEN`` env / cached login inside huggingface_hub).
    local_dir: override the destination directory.
    max_workers: parallel download workers.
  """
  hf = require_hf_hub()
  eff_cache_dir: str | os.PathLike[str] = cache_dir
  if Path(cache_dir) == DEFAULT_CACHE_DIR and os.environ.get("EEB_LEROBOT_CACHE_DIR"):
    eff_cache_dir = Path(os.environ["EEB_LEROBOT_CACHE_DIR"])
  dest = Path(local_dir) if local_dir else local_dir_for(dataset_id, eff_cache_dir)
  dest.mkdir(parents=True, exist_ok=True)
  effective_revision = revision if revision is not None else PINNED_HUB_REVISIONS.get(dataset_id)
  common = dict(repo_id=dataset_id, repo_type="dataset", revision=effective_revision, local_dir=str(dest), token=token, max_workers=max_workers)
  if episodes is None:
    hf.snapshot_download(allow_patterns=list(allow_patterns) if allow_patterns else None, **common)
    return dest
  # Two-phase: metadata first, then exactly the files the requested episodes live in.
  hf.snapshot_download(allow_patterns=list(META_PATTERNS), **common)
  plan = plan_episode_files(dest, episodes, include_videos=include_videos, dataset_id=dataset_id)
  patterns = list(plan.allow_patterns)
  if dataset_id == "HuggingFaceVLA/libero" and int(episodes) > 3:
    for fi in range((int(episodes) + 2) // 3):
      extra_rel = f"data/chunk-000/file-{fi:03d}.parquet"
      if extra_rel not in patterns:
        patterns.append(extra_rel)
  if allow_patterns:
    patterns.extend(allow_patterns)
  if patterns:
    hf.snapshot_download(allow_patterns=patterns, **common)
  return dest


# --------------------------------------------------------------------------------------------
# Stdlib Hub probes (no huggingface_hub needed)
# --------------------------------------------------------------------------------------------
def _http_get_json(url: str, timeout: float = 30.0) -> Any:
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
  tok = os.environ.get("HF_TOKEN")
  if tok:
    req.add_header("Authorization", f"Bearer {tok}")
  with urllib.request.urlopen(req, timeout=timeout) as resp:  # follows redirects (307 -> CDN)
    return json.loads(resp.read().decode("utf-8"))


def hub_tree(dataset_id: str, path: str = "", revision: str = "main", timeout: float = 30.0) -> List[Dict[str, Any]]:
  """Entries of ``https://huggingface.co/api/datasets/<id>/tree/<revision>/<path>`` (type, path, size)."""
  url = HF_API_TREE.format(dataset_id=dataset_id, revision=revision, path=path).rstrip("/")
  try:
    data = _http_get_json(url, timeout=timeout)
  except urllib.error.HTTPError as e:
    raise RuntimeError(f"GET {url} -> HTTP {e.code} ({e.reason}); nonexistent/gated repos return 401 on the Hub") from e
  if not isinstance(data, list):
    raise RuntimeError(f"GET {url} returned unexpected payload: {str(data)[:200]}")
  return [{"type": d.get("type"), "path": d.get("path"), "size": d.get("size")} for d in data]


def hub_info_json(dataset_id: str, revision: str = "main", timeout: float = 30.0) -> Dict[str, Any]:
  """``meta/info.json`` of a Hub dataset (``resolve`` URL, follows the 307 redirect to the CDN)."""
  url = HF_RESOLVE.format(dataset_id=dataset_id, revision=revision, path="meta/info.json")
  try:
    return _http_get_json(url, timeout=timeout)
  except urllib.error.HTTPError as e:
    raise RuntimeError(f"GET {url} -> HTTP {e.code} ({e.reason})") from e


def probe_hub_layout(dataset_id: str, revision: str = "main") -> Dict[str, Any]:
  """Summarise a Hub dataset's LeRobot layout from the tree listing + ``meta/info.json``."""
  top = hub_tree(dataset_id, "", revision)
  meta = hub_tree(dataset_id, "meta", revision)
  info = hub_info_json(dataset_id, revision)
  feats = info.get("features", {})
  return {
      "dataset_id": dataset_id,
      "revision": revision,
      "top_level": [e["path"] for e in top],
      "meta_files": [e["path"] for e in meta],
      "codebase_version": info.get("codebase_version"),
      "layout": "v3" if str(info.get("codebase_version", "")).lstrip("v").split(".")[0] in ("3", "4") else "v2",
      "fps": info.get("fps"),
      "total_episodes": info.get("total_episodes"),
      "total_frames": info.get("total_frames"),
      "data_path": info.get("data_path"),
      "video_path": info.get("video_path"),
      "image_features": [k for k, v in feats.items() if v.get("dtype") == "image"],
      "video_features": [k for k, v in feats.items() if v.get("dtype") == "video"],
      "has_videos_dir": any(e["path"] == "videos" for e in top),
      "urls_hit": [
          HF_API_TREE.format(dataset_id=dataset_id, revision=revision, path="").rstrip("/"),
          HF_API_TREE.format(dataset_id=dataset_id, revision=revision, path="meta"),
          HF_RESOLVE.format(dataset_id=dataset_id, revision=revision, path="meta/info.json"),
      ],
  }
