"""Golden reference outputs for Gate A (``CosSim >= 0.995`` action parity).

Organizers run each BF16 reference checkpoint **once** over the packed traces and store the
per-step action chunks here; participants' timed runs are compared against these files by the gate
code (WS-D). **This module performs no model execution** - it only defines the container and the
store/load/verify API.

Layout::

    <root>/golden/<model_id>/<split>/episode_XXXX.npz    actions float32 [T, H, A] (H = chunk horizon),
                                                         step int64 [T], t_ns int64 [T], meta_json
    <root>/golden/<model_id>/<split>/manifest.json       per-file sha256 + source trace content sha256
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from eeb.data.schema import (
    MANIFEST_FILENAME,
    META_KEY,
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_json,
    episode_filename,
    sha256_file,
    utc_now_iso,
)

GOLDEN_DIRNAME = "golden"
GOLDEN_KIND = "golden_reference_actions"


def golden_dir(root: str | os.PathLike[str], model_id: str, split: str) -> Path:
  safe_model = model_id.strip("/").replace("/", "__")
  return Path(root) / GOLDEN_DIRNAME / safe_model / split


def golden_path(root: str | os.PathLike[str], model_id: str, split: str, source_episode_index: int) -> Path:
  return golden_dir(root, model_id, split) / episode_filename(source_episode_index)


@dataclass
class GoldenEpisode:
  """Reference actions for one episode produced by one reference model."""

  model_id: str
  split: str
  episode_id: str
  source_episode_index: int
  actions: np.ndarray                       # float32 [T, H, A]  ([T, A] is promoted to H = 1)
  step: np.ndarray                          # int64 [T]
  t_ns: np.ndarray                          # int64 [T]
  trace_content_sha256: str                 # EmbodiedXTrace.content_sha256() of the input trace
  precision: str = "bf16"
  extra: Dict[str, Any] = field(default_factory=dict)  # e.g. checkpoint, framework, device, run_id

  def __post_init__(self) -> None:
    a = np.asarray(self.actions, dtype=np.float32)
    if a.ndim == 2:
      a = a[:, None, :]
    if a.ndim != 3:
      raise ValueError(f"actions must be [T, H, A] or [T, A], got shape {a.shape}")
    self.actions = np.ascontiguousarray(a)
    self.step = np.ascontiguousarray(np.asarray(self.step, dtype=np.int64).reshape(-1))
    self.t_ns = np.ascontiguousarray(np.asarray(self.t_ns, dtype=np.int64).reshape(-1))
    n = self.actions.shape[0]
    if self.step.shape != (n,) or self.t_ns.shape != (n,):
      raise ValueError(f"step/t_ns must have length T={n}, got {self.step.shape}/{self.t_ns.shape}")
    if not self.trace_content_sha256 or len(self.trace_content_sha256) != 64:
      raise ValueError("trace_content_sha256 must be the 64-hex content hash of the source trace")

  def __len__(self) -> int:
    return int(self.actions.shape[0])

  @property
  def horizon(self) -> int:
    return int(self.actions.shape[1])

  @property
  def action_dim(self) -> int:
    return int(self.actions.shape[2])

  def meta(self) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GOLDEN_KIND,
        "model_id": self.model_id,
        "split": self.split,
        "episode_id": self.episode_id,
        "source_episode_index": int(self.source_episode_index),
        "trace_content_sha256": self.trace_content_sha256,
        "precision": self.precision,
        "step_count": len(self),
        "horizon": self.horizon,
        "action_dim": self.action_dim,
        "extra": dict(self.extra),
    }

  def content_sha256(self) -> str:
    import hashlib

    h = hashlib.sha256(canonical_json(self.meta()).encode("utf-8"))
    for k, a in (("actions", self.actions), ("step", self.step), ("t_ns", self.t_ns)):
      h.update(k.encode()); h.update(a.dtype.str.encode()); h.update(canonical_json(list(a.shape)).encode()); h.update(a.tobytes(order="C"))
    return h.hexdigest()

  def to_npz(self, path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
      with open(tmp, "wb") as f:
        np.savez_compressed(f, actions=self.actions, step=self.step, t_ns=self.t_ns, **{META_KEY: np.array(canonical_json(self.meta()))})
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
  def from_npz(cls, path: str | os.PathLike[str]) -> "GoldenEpisode":
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(f"golden file not found: {path}")
    with np.load(path, allow_pickle=False) as z:
      meta = json.loads(str(z[META_KEY]))
      if meta.get("kind") != GOLDEN_KIND:
        raise ValueError(f"{path} is not a golden reference file (kind={meta.get('kind')!r})")
      return cls(
          model_id=meta["model_id"],
          split=meta["split"],
          episode_id=meta["episode_id"],
          source_episode_index=int(meta["source_episode_index"]),
          actions=z["actions"],
          step=z["step"],
          t_ns=z["t_ns"],
          trace_content_sha256=meta["trace_content_sha256"],
          precision=meta.get("precision", "bf16"),
          extra=dict(meta.get("extra", {})),
      )


# --------------------------------------------------------------------------------------------
# Store / load / verify
# --------------------------------------------------------------------------------------------
def _load_manifest(d: Path) -> Dict[str, Any]:
  p = d / MANIFEST_FILENAME
  if not p.exists():
    return {"schema_version": SCHEMA_VERSION, "kind": GOLDEN_KIND, "episodes": {}, "sha256": {}}
  with open(p, "r", encoding="utf-8") as f:
    return json.load(f)


def save_golden(root: str | os.PathLike[str], golden: GoldenEpisode) -> Path:
  """Write ``golden/<model_id>/<split>/episode_XXXX.npz`` and update that directory's manifest."""
  d = golden_dir(root, golden.model_id, golden.split)
  path = golden.to_npz(d / episode_filename(golden.source_episode_index))
  man = _load_manifest(d)
  man.update({"schema_version": SCHEMA_VERSION, "kind": GOLDEN_KIND, "model_id": golden.model_id, "split": golden.split, "updated_at": utc_now_iso()})
  man.setdefault("episodes", {})[path.name] = {
      "episode_id": golden.episode_id,
      "source_episode_index": int(golden.source_episode_index),
      "steps": len(golden),
      "horizon": golden.horizon,
      "action_dim": golden.action_dim,
      "precision": golden.precision,
      "trace_content_sha256": golden.trace_content_sha256,
      "content_sha256": golden.content_sha256(),
  }
  man.setdefault("sha256", {})[path.name] = sha256_file(path)
  atomic_write_json(d / MANIFEST_FILENAME, man)
  return path


def load_golden(root: str | os.PathLike[str], model_id: str, split: str, source_episode_index: int, verify: bool = True) -> GoldenEpisode:
  """Load one golden episode; with ``verify`` the file sha256 must match the directory manifest."""
  d = golden_dir(root, model_id, split)
  path = d / episode_filename(source_episode_index)
  if verify:
    man = _load_manifest(d)
    expected = man.get("sha256", {}).get(path.name)
    if expected is None:
      raise ValueError(f"{path.name} is not listed in {d / MANIFEST_FILENAME}")
    actual = sha256_file(path)
    if actual != expected:
      raise ValueError(f"{path}: sha256 {actual[:12]}... != manifest {expected[:12]}... (tampered or corrupted)")
  return GoldenEpisode.from_npz(path)


def verify_golden(root: str | os.PathLike[str], model_id: str, split: str) -> Dict[str, Any]:
  d = golden_dir(root, model_id, split)
  man = _load_manifest(d)
  problems: List[str] = []
  for fname, expected in man.get("sha256", {}).items():
    fp = d / fname
    if not fp.exists():
      problems.append(f"missing {fname}")
    elif sha256_file(fp) != expected:
      problems.append(f"sha256 mismatch {fname}")
  return {"ok": not problems, "checked": len(man.get("sha256", {})), "problems": problems, "dir": str(d)}


def list_golden(root: str | os.PathLike[str], model_id: str, split: str) -> List[str]:
  return sorted(_load_manifest(golden_dir(root, model_id, split)).get("sha256", {}).keys())


# --------------------------------------------------------------------------------------------
# Gate-A style comparison helpers (pure math; no model execution)
# --------------------------------------------------------------------------------------------
def action_cosine_similarity(golden: np.ndarray, candidate: np.ndarray, eps: float = 1e-12) -> float:
  """Mean per-step cosine similarity between flattened action chunks (``[T, ...]`` vs ``[T, ...]``)."""
  g = np.asarray(golden, dtype=np.float64).reshape(golden.shape[0], -1)
  c = np.asarray(candidate, dtype=np.float64).reshape(candidate.shape[0], -1)
  if g.shape != c.shape:
    raise ValueError(f"shape mismatch golden {g.shape} vs candidate {c.shape}")
  num = np.sum(g * c, axis=1)
  den = np.linalg.norm(g, axis=1) * np.linalg.norm(c, axis=1)
  cos = np.where(den > eps, num / np.maximum(den, eps), 1.0)  # both-zero rows count as identical
  return float(np.mean(cos))


def action_l2_error(golden: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
  g = np.asarray(golden, dtype=np.float64)
  c = np.asarray(candidate, dtype=np.float64)
  if g.shape != c.shape:
    raise ValueError(f"shape mismatch golden {g.shape} vs candidate {c.shape}")
  diff = (g - c).reshape(g.shape[0], -1)
  return {
      "mean_l2": float(np.mean(np.linalg.norm(diff, axis=1))),
      "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
      "rmse": float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
  }


PROTOCOL_MIN_GOLDEN_SAMPLES = 100


# --------------------------------------------------------------------------------------------
# Turnkey Golden-Reference Generator (`eeb data generate-golden`)
# --------------------------------------------------------------------------------------------
def generate_golden_reference(
    endpoint: str,
    out_path: str | os.PathLike[str],
    *,
    traces: Optional[str | os.PathLike[str]] = None,
    traces_split: str = "public",
    num_samples: int = PROTOCOL_MIN_GOLDEN_SAMPLES,
    allow_smoke_golden: bool = False,
    policy: Optional[Any] = None,
    policy_kwargs: Optional[Dict[str, Any] | str] = None,
    image_hw: Optional[tuple] = None,
    session_id: str = "golden_ref",
    seed: int = 7,
    timeout_s: float = 60.0,
    log: Optional[Any] = None,
) -> Dict[str, Any]:
  """Connect to `endpoint`, run `predict()` (`VLA_TRACK`) or `rollout()` (`WM_TRACK`) on `num_samples`
  observations from `traces` (or `SyntheticObservationSource`), and atomically write a golden `.npz`
  bundle ready for `eeb measure --golden <out.npz>`.
  """
  from eeb.loadgen.client import make_client
  from eeb.loadgen.sweep import ObservationSpec, SyntheticObservationSource, TraceBundleObservationSource

  n = int(num_samples)
  if n < 1:
    raise ValueError(f"num_samples must be >= 1, got {num_samples}")
  if n < PROTOCOL_MIN_GOLDEN_SAMPLES and not allow_smoke_golden:
    raise ValueError(
        f"generate_golden_reference requires num_samples >= {PROTOCOL_MIN_GOLDEN_SAMPLES} for "
        f"certifiable Gate A bundles (got num_samples={n}). Pass allow_smoke_golden=True "
        "(CLI: --allow-smoke-golden) only for non-certifiable CI/local smoke runs."
    )

  parsed_kwargs: Optional[Dict[str, Any]] = None
  if isinstance(policy_kwargs, str) and policy_kwargs.strip():
    parsed_kwargs = json.loads(policy_kwargs)
  elif isinstance(policy_kwargs, dict):
    parsed_kwargs = dict(policy_kwargs)

  target = Path(out_path)
  target.parent.mkdir(parents=True, exist_ok=True)
  # Clean up any crashed orphan temp files for this target before writing
  for pat in (f".{target.name}.*.tmp*", f".tmp_golden_{target.stem}*"):
    for orphan in target.parent.glob(pat):
      try:
        orphan.unlink()
      except OSError:
        pass

  with make_client(endpoint, policy=policy, policy_kwargs=parsed_kwargs, timeout_s=timeout_s) as client:
    caps = client.capabilities()
    if callable(log):
      log(f"connected to {endpoint} track={caps.track} model_id={caps.model_id} server_impl={caps.server_impl}")
    if traces is not None:
      obs_source = TraceBundleObservationSource(str(traces), split=traces_split)
    else:
      spec_overrides: Dict[str, Any] = {"seed": int(seed)}
      if image_hw is not None:
        spec_overrides["image_hw"] = (int(image_hw[0]), int(image_hw[1]))
      obs_source = SyntheticObservationSource(ObservationSpec.from_caps(caps, **spec_overrides))

    client.reset(session_id)

    if caps.track == "WM_TRACK":
      cond_frames_list: List[np.ndarray] = []
      proprios_list: List[np.ndarray] = []
      instructions_list: List[str] = []
      actions_list: List[np.ndarray] = []
      frames_list: List[np.ndarray] = []
      num_frames = int(caps.horizon or 16)
      act_dim = int(caps.action_dim or 7)
      default_h = int(caps.image_height or 64)
      default_w = int(caps.image_width or 64)

      for i in range(n):
        ctx, act_in = obs_source.make_rollout_inputs(
            session_id,
            i,
            num_frames=num_frames,
            action_dim=act_dim,
        )
        roll = client.rollout(ctx, act_in)
        if not roll.frames:
          raise ValueError(f"WM_TRACK policy returned 0 frames at sample {i} on endpoint {endpoint!r}")
        frames_i = np.stack([np.asarray(f.image, dtype=np.uint8) for f in roll.frames], axis=0)
        cond_img = (
            np.asarray(ctx.frames[0], dtype=np.uint8)
            if ctx.frames
            else np.zeros((default_h, default_w, 3), dtype=np.uint8)
        )
        cond_frames_list.append(cond_img)
        proprios_list.append(np.asarray(ctx.proprio, dtype=np.float32))
        instructions_list.append(str(ctx.instruction))
        actions_list.append(np.asarray(act_in, dtype=np.float32))
        frames_list.append(frames_i)

      context_frames = np.ascontiguousarray(np.stack(cond_frames_list, axis=0), dtype=np.uint8)
      proprios = np.ascontiguousarray(np.stack(proprios_list, axis=0), dtype=np.float32)
      instructions = np.asarray(instructions_list)
      actions = np.ascontiguousarray(np.stack(actions_list, axis=0), dtype=np.float32)
      frames = np.ascontiguousarray(np.stack(frames_list, axis=0), dtype=np.uint8)

      obs_desc = obs_source.describe()
      is_certifiable = bool(
          n >= PROTOCOL_MIN_GOLDEN_SAMPLES
          and not allow_smoke_golden
          and obs_desc.get("kind") == "EMBODIEDX_TRACES_V1_BUNDLE"
          and obs_desc.get("certifiable", True) is not False
      )
      meta_dict: Dict[str, Any] = {
          "schema_version": SCHEMA_VERSION,
          "kind": "golden_replay_bundle",
          "track": "WM_TRACK",
          "endpoint": str(endpoint),
          "model_id": caps.model_id,
          "server_impl": caps.server_impl,
          "precision": caps.precision,
          "session_id": session_id,
          "num_samples": n,
          "allow_smoke_golden": bool(allow_smoke_golden),
          "certifiable": is_certifiable,
          "num_frames": int(frames.shape[1]),
          "action_dim": int(actions.shape[2]),
          "frame_hw": [int(frames.shape[2]), int(frames.shape[3])],
          "observation_source": obs_desc,
          "generated_at": utc_now_iso(),
      }
      meta_str = canonical_json(meta_dict)
      payload: Dict[str, np.ndarray] = {
          "context_frames": context_frames,
          "image__agentview": context_frames,
          "proprios": proprios,
          "proprio": proprios,
          "instructions": instructions,
          "actions": actions,
          "ref_actions": actions,
          "frames": frames,
          "ref_frames": frames,
          "metadata_json": np.array(meta_str),
          META_KEY: np.array(meta_str),
      }
    else:
      cam_lists: Dict[str, List[np.ndarray]] = {}
      proprios_list = []
      instructions_list = []
      actions_list = []

      for i in range(n):
        obs = obs_source.make(session_id, i)
        pred = client.predict(obs)
        for cam, img in obs.images.items():
          cam_lists.setdefault(cam, []).append(np.asarray(img, dtype=np.uint8))
        proprios_list.append(np.asarray(obs.proprio, dtype=np.float32))
        instructions_list.append(str(obs.instruction))
        actions_list.append(np.asarray(pred.chunk.actions, dtype=np.float32))

      sorted_cams = sorted(cam_lists.keys())
      cam_arrays = {cam: np.ascontiguousarray(np.stack(cam_lists[cam], axis=0), dtype=np.uint8) for cam in sorted_cams}
      images = (
          np.ascontiguousarray(np.stack([cam_arrays[c] for c in sorted_cams], axis=1), dtype=np.uint8)
          if sorted_cams
          else np.zeros((n, 0), dtype=np.uint8)
      )
      proprios = np.ascontiguousarray(np.stack(proprios_list, axis=0), dtype=np.float32)
      instructions = np.asarray(instructions_list)
      actions = np.ascontiguousarray(np.stack(actions_list, axis=0), dtype=np.float32)

      obs_desc = obs_source.describe()
      is_certifiable = bool(
          n >= PROTOCOL_MIN_GOLDEN_SAMPLES
          and not allow_smoke_golden
          and obs_desc.get("kind") == "EMBODIEDX_TRACES_V1_BUNDLE"
          and obs_desc.get("certifiable", True) is not False
      )
      meta_dict = {
          "schema_version": SCHEMA_VERSION,
          "kind": "golden_replay_bundle",
          "track": "VLA_TRACK",
          "endpoint": str(endpoint),
          "model_id": caps.model_id,
          "server_impl": caps.server_impl,
          "precision": caps.precision,
          "session_id": session_id,
          "num_samples": n,
          "allow_smoke_golden": bool(allow_smoke_golden),
          "certifiable": is_certifiable,
          "horizon": int(actions.shape[1]),
          "action_dim": int(actions.shape[2]),
          "execute_horizon": caps.execute_horizon,
          "cameras": sorted_cams,
          "observation_source": obs_desc,
          "generated_at": utc_now_iso(),
      }
      meta_str = canonical_json(meta_dict)
      payload = {
          "images": images,
          "proprios": proprios,
          "proprio": proprios,
          "instructions": instructions,
          "actions": actions,
          "ref_actions": actions,
          "metadata_json": np.array(meta_str),
          META_KEY: np.array(meta_str),
      }
      for cam, arr in cam_arrays.items():
        payload[f"image__{cam}"] = arr

  fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp.npz", dir=str(target.parent))
  os.close(fd)
  try:
    with open(tmp, "wb") as f:
      np.savez_compressed(f, **payload)
      f.flush()
      os.fsync(f.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)
  except BaseException:
    try:
      os.unlink(tmp)
    except OSError:
      pass
    raise

  summary = dict(meta_dict)
  summary["out"] = str(target)
  summary["sha256"] = sha256_file(target)
  return summary
