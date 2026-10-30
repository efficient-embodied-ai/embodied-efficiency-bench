"""G4 (MLPerf TEST04-style) anti-memoization perturbations for EmbodiedX traces.

A participant that caches/memoizes outputs by input identity is caught by replaying inputs that
differ from the golden trace by an imperceptible, **seeded** perturbation with
``‖δ‖∞ = eps`` (default ``1e-3`` in normalized units): the model must produce *slightly* different
outputs, a cache produces *identical* ones.

Determinism contract
--------------------
Every step's perturbation is a pure function of ``(seed, episode_id, step)``::

    rng = np.random.default_rng(int.from_bytes(sha256(f"{seed}|{episode_id}|{step}").digest()[:8], "big"))

so the LoadGen can regenerate it on the fly (recommended: no storage) and organizers can
re-materialise byte-identical artifacts for audits (``write_perturbed_split`` +
``perturbation_manifest.json`` with sha256s).

Image spaces
------------
* ``unit_float32`` (default): frames are mapped to ``float32`` in ``[0, 1]`` and ``δ`` with
  ``‖δ‖∞ == eps`` is added. Signs are flipped at the two boundaries (``x == 0 -> +|δ|``,
  ``x == 1 -> -|δ|``), so no clipping is ever needed and the bound is exact.
* ``uint8_lsb``: frames stay ``uint8``; a seeded fraction of pixels (``lsb_fraction``) moves by
  ``±1`` LSB (boundary-aware). The effective ``‖δ‖∞`` is ``1/255 ≈ 3.9e-3`` and is recorded as
  ``eps_linf_effective`` in the manifest (``1e-3`` is not representable in 8 bits).

Optional camera jitter (integer pixel shift with edge replication) and brightness jitter are
robustness probes, **not** cache probes: when enabled the manifest sets ``g4_cache_probe_valid=false``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eeb.data.schema import (
    IMAGE_KEY_PREFIX,
    MANIFEST_FILENAME,
    META_KEY,
    SCHEMA_VERSION,
    EmbodiedXTrace,
    TraceManifest,
    TraceStep,
    atomic_write_json,
    canonical_json,
    episode_filename,
    sha256_file,
    utc_now_iso,
)

G4_DEFAULT_EPS = 1e-3
DEFAULT_PERTURB_SEED = 20261007
IMAGE_SPACES: Tuple[str, ...] = ("unit_float32", "uint8_lsb")
PERTURBATION_MANIFEST_FILENAME = "perturbation_manifest.json"
PERTURBED_DIRNAME = "perturbed"


@dataclass(frozen=True)
class PerturbationConfig:
  seed: int = DEFAULT_PERTURB_SEED
  eps_linf: float = G4_DEFAULT_EPS
  image_space: str = "unit_float32"
  camera_jitter_px: int = 0
  brightness_jitter: float = 0.0
  perturb_images: bool = True
  perturb_proprio: bool = True
  lsb_fraction: float = 0.01

  def __post_init__(self) -> None:
    if self.image_space not in IMAGE_SPACES:
      raise ValueError(f"image_space must be one of {IMAGE_SPACES}, got {self.image_space!r}")
    if not (0.0 < self.eps_linf <= 1.0 / 255.0 + 1e-12):
      raise ValueError(f"eps_linf must be in (0, 1/255] so boundary sign-flips never clip, got {self.eps_linf}")
    if self.camera_jitter_px < 0:
      raise ValueError("camera_jitter_px must be >= 0")
    if not (0.0 <= self.brightness_jitter < 1.0):
      raise ValueError("brightness_jitter must be in [0, 1)")
    if not (0.0 < self.lsb_fraction <= 1.0):
      raise ValueError("lsb_fraction must be in (0, 1]")

  @property
  def eps_linf_effective(self) -> float:
    return float(self.eps_linf) if self.image_space == "unit_float32" else 1.0 / 255.0

  @property
  def g4_cache_probe_valid(self) -> bool:
    return self.camera_jitter_px == 0 and self.brightness_jitter == 0.0

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    d["eps_linf_effective"] = self.eps_linf_effective
    d["g4_cache_probe_valid"] = self.g4_cache_probe_valid
    return d


# --------------------------------------------------------------------------------------------
# Seeds and deltas
# --------------------------------------------------------------------------------------------
def derive_step_seed(seed: int, episode_id: str, step: int) -> int:
  digest = hashlib.sha256(f"{int(seed)}|{episode_id}|{int(step)}".encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big")


def step_rng(seed: int, episode_id: str, step: int) -> np.random.Generator:
  return np.random.default_rng(derive_step_seed(seed, episode_id, step))


def linf_delta(rng: np.random.Generator, shape: Sequence[int], eps: float) -> np.ndarray:
  """float32 δ with ``max|δ| == eps`` exactly (uniform directions, rescaled by the max magnitude)."""
  u = rng.uniform(-1.0, 1.0, size=tuple(int(s) for s in shape))
  m = float(np.max(np.abs(u))) if u.size else 1.0
  if m == 0.0:  # practically impossible; keep the contract anyway
    u = np.ones_like(u)
    m = 1.0
  return (u * (float(eps) / m)).astype(np.float32)


def _boundary_aware(x01: np.ndarray, delta: np.ndarray) -> np.ndarray:
  """Flip δ's sign where adding it would leave [0, 1] (only possible at exact 0 / 1 for eps <= 1/255)."""
  out = delta.copy()
  mag = np.abs(delta)
  out = np.where(x01 + delta > 1.0, -mag, out)
  out = np.where(x01 + delta < 0.0, mag, out)
  return out.astype(np.float32)


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
  """Integer pixel shift with edge replication (no wrap-around)."""
  if dx == 0 and dy == 0:
    return img
  h, w = img.shape[:2]
  pad = ((abs(dy), abs(dy)), (abs(dx), abs(dx))) + ((0, 0),) * (img.ndim - 2)
  padded = np.pad(img, pad, mode="edge")
  y0 = abs(dy) - dy
  x0 = abs(dx) - dx
  return np.ascontiguousarray(padded[y0:y0 + h, x0:x0 + w])


# --------------------------------------------------------------------------------------------
# Per-step / per-trace perturbation
# --------------------------------------------------------------------------------------------
@dataclass
class PerturbedStep:
  episode_id: str
  step: int
  images: Dict[str, np.ndarray]          # float32 [H,W,3] in [0,1]  (unit_float32)  or uint8 (uint8_lsb)
  proprio: np.ndarray                    # float32 [D]
  delta_linf: Dict[str, float] = field(default_factory=dict)   # measured ‖δ‖∞ per camera + "proprio"
  jitter: Dict[str, Any] = field(default_factory=dict)         # applied (dx, dy, brightness) per camera


def perturb_image(img_u8: np.ndarray, rng: np.random.Generator, cfg: PerturbationConfig) -> Tuple[np.ndarray, float, Dict[str, Any]]:
  """Return ``(perturbed, measured_linf_in_unit_space, jitter_info)`` for one ``uint8[H,W,3]`` frame."""
  if img_u8.dtype != np.uint8:
    raise ValueError(f"expected uint8 frame, got {img_u8.dtype}")
  info: Dict[str, Any] = {}
  base = img_u8
  if cfg.camera_jitter_px > 0:
    dx, dy = (int(v) for v in rng.integers(-cfg.camera_jitter_px, cfg.camera_jitter_px + 1, size=2))
    base = shift_image(base, dx, dy)
    info["shift_px"] = [dx, dy]
  x = base.astype(np.float32) / np.float32(255.0)
  if cfg.brightness_jitter > 0:
    b = float(rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter))
    x = np.clip(x * np.float32(1.0 + b), 0.0, 1.0).astype(np.float32)
    info["brightness_gain"] = 1.0 + b
  ref = img_u8.astype(np.float32) / np.float32(255.0)
  if cfg.image_space == "unit_float32":
    delta = _boundary_aware(x, linf_delta(rng, x.shape, cfg.eps_linf))
    y = (x + delta).astype(np.float32)
    return y, float(np.max(np.abs(y - ref))), info
  # uint8_lsb
  mask = rng.uniform(0.0, 1.0, size=x.shape) < cfg.lsb_fraction
  sign = np.where(rng.uniform(0.0, 1.0, size=x.shape) < 0.5, -1, 1).astype(np.int16)
  xi = np.round(x * 255.0).astype(np.int16)
  sign = np.where(xi <= 0, 1, sign)
  sign = np.where(xi >= 255, -1, sign)
  yi = np.where(mask, xi + sign, xi)
  y8 = np.clip(yi, 0, 255).astype(np.uint8)
  return y8, float(np.max(np.abs(y8.astype(np.float32) / 255.0 - ref))), info


def perturb_proprio(proprio: np.ndarray, rng: np.random.Generator, cfg: PerturbationConfig) -> np.ndarray:
  delta = linf_delta(rng, proprio.shape, cfg.eps_linf)
  return (proprio.astype(np.float32) + delta).astype(np.float32)


def perturb_step(step: TraceStep, cfg: PerturbationConfig = PerturbationConfig()) -> PerturbedStep:
  """Deterministic perturbation of one step (same ``(cfg.seed, episode_id, step)`` -> identical output)."""
  rng = step_rng(cfg.seed, step.episode_id, step.step)
  images: Dict[str, np.ndarray] = {}
  delta_linf: Dict[str, float] = {}
  jitter: Dict[str, Any] = {}
  for cam in sorted(step.images.keys()):   # fixed camera order -> fixed RNG consumption order
    if cfg.perturb_images:
      y, d, info = perturb_image(step.images[cam], rng, cfg)
      images[cam] = y
      delta_linf[cam] = d
      if info:
        jitter[cam] = info
    else:
      images[cam] = step.images[cam]
      delta_linf[cam] = 0.0
  if cfg.perturb_proprio:
    p = perturb_proprio(step.proprio, rng, cfg)
    delta_linf["proprio"] = float(np.max(np.abs(p - step.proprio))) if p.size else 0.0
  else:
    p = step.proprio.astype(np.float32)
    delta_linf["proprio"] = 0.0
  return PerturbedStep(episode_id=step.episode_id, step=int(step.step), images=images, proprio=p, delta_linf=delta_linf, jitter=jitter)


@dataclass
class PerturbedTrace:
  episode_id: str
  split: str
  config: PerturbationConfig
  t_ns: np.ndarray
  step: np.ndarray
  proprio: np.ndarray                    # float32 [T, D]
  images: Dict[str, np.ndarray]          # cam -> float32 [T,H,W,3] or uint8 [T,H,W,3]
  source_content_sha256: str
  max_delta_linf: Dict[str, float] = field(default_factory=dict)
  jitter: List[Dict[str, Any]] = field(default_factory=list)

  def __len__(self) -> int:
    return int(self.t_ns.shape[0])

  def meta(self) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "g4_perturbed_trace",
        "episode_id": self.episode_id,
        "split": self.split,
        "config": self.config.to_dict(),
        "source_content_sha256": self.source_content_sha256,
        "max_delta_linf": dict(self.max_delta_linf),
        "step_count": len(self),
        "cameras": sorted(self.images.keys()),
    }

  def arrays(self) -> Dict[str, np.ndarray]:
    out = {"t_ns": self.t_ns, "step": self.step, "proprio": np.ascontiguousarray(self.proprio, dtype=np.float32)}
    for cam in sorted(self.images):
      out[IMAGE_KEY_PREFIX + cam] = np.ascontiguousarray(self.images[cam])
    return out

  def content_sha256(self) -> str:
    h = hashlib.sha256(canonical_json(self.meta()).encode("utf-8"))
    for k, a in sorted(self.arrays().items()):
      h.update(k.encode()); h.update(a.dtype.str.encode()); h.update(canonical_json(list(a.shape)).encode()); h.update(a.tobytes(order="C"))
    return h.hexdigest()

  def to_npz(self, path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(self.arrays())
    payload[META_KEY] = np.array(canonical_json(self.meta()))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
      with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
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
  def from_npz(cls, path: str | os.PathLike[str]) -> "PerturbedTrace":
    with np.load(path, allow_pickle=False) as z:
      meta = json.loads(str(z[META_KEY]))
      if meta.get("kind") != "g4_perturbed_trace":
        raise ValueError(f"{path} is not a G4 perturbed trace")
      cfg_d = {k: v for k, v in meta["config"].items() if k in PerturbationConfig.__dataclass_fields__}
      return cls(
          episode_id=meta["episode_id"],
          split=meta["split"],
          config=PerturbationConfig(**cfg_d),
          t_ns=np.ascontiguousarray(z["t_ns"]),
          step=np.ascontiguousarray(z["step"]),
          proprio=np.ascontiguousarray(z["proprio"]),
          images={k[len(IMAGE_KEY_PREFIX):]: np.ascontiguousarray(z[k]) for k in z.files if k.startswith(IMAGE_KEY_PREFIX)},
          source_content_sha256=meta["source_content_sha256"],
          max_delta_linf=dict(meta.get("max_delta_linf", {})),
      )


def perturb_trace(trace: EmbodiedXTrace, cfg: PerturbationConfig = PerturbationConfig()) -> PerturbedTrace:
  """Perturb every step of a trace (columnar result; deterministic)."""
  steps = [perturb_step(s, cfg) for s in trace]
  cams = list(trace.cameras)
  images = {cam: np.stack([ps.images[cam] for ps in steps]) for cam in cams}
  proprio = np.stack([ps.proprio for ps in steps]).astype(np.float32)
  max_delta: Dict[str, float] = {}
  for key in cams + ["proprio"]:
    max_delta[key] = float(max(ps.delta_linf.get(key, 0.0) for ps in steps))
  return PerturbedTrace(
      episode_id=trace.episode_id,
      split=trace.split,
      config=cfg,
      t_ns=trace.t_ns.copy(),
      step=trace.step.copy(),
      proprio=proprio,
      images=images,
      source_content_sha256=trace.content_sha256(),
      max_delta_linf=max_delta,
      jitter=[ps.jitter for ps in steps if ps.jitter],
  )


# --------------------------------------------------------------------------------------------
# Materialisation + manifest
# --------------------------------------------------------------------------------------------
def _load_traces_for_perturbation(
    traces_dir: Path,
    split: str,
) -> Tuple[Path, Path, List[Tuple[int, str, str, EmbodiedXTrace]]]:
  """Resolve `(split_dir, manifest_path, [(source_episode_index, source_file, source_content_sha256, trace), ...])`.

  Supports:
    1. Canonical packed directory `traces/<split>/manifest.json` (`episodes` schema).
    2. Direct split directory `<split_dir>/manifest.json` (`episodes` schema).
    3. Flat EmbodiedX shard bundle `<bundle_dir>/manifest.json` (`shards` schema, e.g. `data/embodiedx_traces_v1_mini`).
  """
  split_candidate = traces_dir / split
  if (split_candidate / MANIFEST_FILENAME).exists():
    man = TraceManifest.load(split_candidate)
    ver = man.verify(split_candidate)
    if not ver.ok:
      raise ValueError(f"refusing to perturb unverified traces: {ver.summary()}")
    rows: List[Tuple[int, str, str, EmbodiedXTrace]] = []
    for e in man.episodes:
      trace = EmbodiedXTrace.from_npz(split_candidate / e.file)
      if trace.content_sha256() != e.content_sha256:
        raise ValueError(f"{e.file}: content sha256 mismatch vs manifest; refusing to perturb")
      rows.append((int(e.source_episode_index), e.file, e.content_sha256, trace))
    return split_candidate, split_candidate / MANIFEST_FILENAME, rows

  root_manifest = traces_dir / MANIFEST_FILENAME
  if root_manifest.exists():
    raw_man = json.loads(root_manifest.read_text(encoding="utf-8"))
    if isinstance(raw_man, dict) and "shards" in raw_man and "episodes" not in raw_man:
      bundle_split = str(raw_man.get("split", "public"))
      accepted_splits = {bundle_split}
      if bundle_split != "hidden":
        accepted_splits.add("public")
      if split not in accepted_splits:
        raise FileNotFoundError(
            f"Trace shard bundle {traces_dir} contains split={bundle_split!r}, not requested --split {split!r}."
        )
      from embodied_bench.dataset_loader import EmbodiedXDatasetLoader

      loader = EmbodiedXDatasetLoader(dataset_dir=traces_dir, split=bundle_split)
      shards = loader.preload_into_pinned_ram(verify=True)
      canonical_split = "hidden" if bundle_split == "hidden" else "public"
      rows = []
      for idx, sh in enumerate(shards):
        t_steps = sh.num_steps
        imgs = {
            "agentview": np.ascontiguousarray(np.transpose(sh.rgb_frames[:, 0], (0, 2, 3, 1)), dtype=np.uint8),
            "wrist": np.ascontiguousarray(np.transpose(sh.rgb_frames[:, 1], (0, 2, 3, 1)), dtype=np.uint8),
        }
        tr = EmbodiedXTrace(
            episode_id=sh.episode_id,
            source_dataset=sh.source_dataset,
            split=canonical_split,
            instruction=sh.instruction_text,
            fps=20.0,
            t_ns=np.arange(t_steps, dtype=np.int64) * 50_000_000,
            step=np.arange(t_steps, dtype=np.int64),
            proprio=np.ascontiguousarray(sh.proprio_state, dtype=np.float32),
            action=np.ascontiguousarray(sh.gt_actions[:, 0, :], dtype=np.float32),
            images=imgs,
            license=sh.license_spdx,
        )
        shard_meta = raw_man["shards"][idx] if idx < len(raw_man["shards"]) else {}
        src_file = str(shard_meta.get("filename", episode_filename(idx)))
        rows.append((idx, src_file, tr.content_sha256(), tr))
      return traces_dir, root_manifest, rows

    if isinstance(raw_man, dict) and "episodes" in raw_man:
      man = TraceManifest.load(traces_dir)
      if man.split != split:
        raise FileNotFoundError(
            f"Trace directory {traces_dir} has split={man.split!r}, not requested --split {split!r}."
        )
      ver = man.verify(traces_dir)
      if not ver.ok:
        raise ValueError(f"refusing to perturb unverified traces: {ver.summary()}")
      rows = []
      for e in man.episodes:
        trace = EmbodiedXTrace.from_npz(traces_dir / e.file)
        if trace.content_sha256() != e.content_sha256:
          raise ValueError(f"{e.file}: content sha256 mismatch vs manifest; refusing to perturb")
        rows.append((int(e.source_episode_index), e.file, e.content_sha256, trace))
      return traces_dir, root_manifest, rows

  raise FileNotFoundError(f"manifest not found: {split_candidate / MANIFEST_FILENAME}")


def write_perturbed_split(
    traces_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    cfg: PerturbationConfig = PerturbationConfig(),
    split: str = "public",
    max_episodes: Optional[int] = None,
    log=None,
) -> Path:
  """Materialise perturbed inputs for ``traces/<split>`` or flat shard bundles into ``<out_dir>/perturbed/<split>/``."""
  import fcntl
  from eeb.data.pack import find_traces_dir  # local import to avoid a cycle at module import time

  traces_dir = find_traces_dir(traces_path)
  _src_dir, src_manifest_path, source_rows = _load_traces_for_perturbation(traces_dir, split)
  out_split = Path(out_dir) / PERTURBED_DIRNAME / split
  out_split.mkdir(parents=True, exist_ok=True)

  lock_digest = hashlib.sha256(str(out_split.resolve()).encode("utf-8")).hexdigest()[:32]
  uid = getattr(os, "getuid", lambda: 0)()
  lock_dir = Path(tempfile.gettempdir()) / f"embodied_bench_locks_{uid}"
  lock_dir.mkdir(parents=True, exist_ok=True)
  lock_path = lock_dir / f"perturb.{lock_digest}.lock"

  with open(lock_path, "a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    try:
      for item in out_split.iterdir():
        if item.is_file() and item.name.startswith(".") and item.name.endswith(".tmp"):
          try:
            item.unlink()
          except OSError:
            pass
      episodes: List[Dict[str, Any]] = []
      for i, (src_ep_idx, src_file, src_csha, trace) in enumerate(source_rows):
        if max_episodes is not None and i >= max_episodes:
          break
        pt = perturb_trace(trace, cfg)
        fname = episode_filename(src_ep_idx)
        fpath = pt.to_npz(out_split / fname)
        episodes.append({
            "episode_id": trace.episode_id,
            "file": fname,
            "steps": len(pt),
            "sha256": sha256_file(fpath),
            "content_sha256": pt.content_sha256(),
            "source_file": src_file,
            "source_content_sha256": src_csha,
            "max_delta_linf": pt.max_delta_linf,
        })
        if log:
          log(
              f"perturbed {src_file} -> {fpath.relative_to(out_dir)}  "
              f"max_delta_linf={ {k: round(v, 7) for k, v in pt.max_delta_linf.items()} }"
          )
      keep_files = {e["file"] for e in episodes}
      for existing in out_split.glob("episode_*.npz"):
        if existing.name not in keep_files:
          try:
            existing.unlink()
          except OSError:
            pass
      manifest = {
          "schema_version": SCHEMA_VERSION,
          "kind": "g4_perturbation",
          "created_at": utc_now_iso(),
          "split": split,
          "config": cfg.to_dict(),
          "derivation": "np.random.default_rng(int.from_bytes(sha256(f'{seed}|{episode_id}|{step}').digest()[:8], 'big')); "
                        "cameras consumed in sorted order, then proprio",
          "source_traces_manifest": str(src_manifest_path.resolve()),
          "source_traces_manifest_sha256": sha256_file(src_manifest_path),
          "episode_count": len(episodes),
          "episodes": episodes,
          "sha256": {e["file"]: e["sha256"] for e in episodes},
      }
      return atomic_write_json(out_split / PERTURBATION_MANIFEST_FILENAME, manifest)
    finally:
      fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def verify_perturbation_manifest(manifest_path: str | os.PathLike[str]) -> Tuple[bool, List[str]]:
  """Recompute file sha256s listed in a ``perturbation_manifest.json``."""
  p = Path(manifest_path)
  if p.is_dir():
    p = p / PERTURBATION_MANIFEST_FILENAME
  with open(p, "r", encoding="utf-8") as f:
    man = json.load(f)
  problems: List[str] = []
  for fname, expected in man.get("sha256", {}).items():
    fp = p.parent / fname
    if not fp.exists():
      problems.append(f"missing {fname}")
    elif sha256_file(fp) != expected:
      problems.append(f"sha256 mismatch {fname}")
  return (not problems, problems)
