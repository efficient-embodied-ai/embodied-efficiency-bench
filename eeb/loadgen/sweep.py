"""Concurrency sweep: warm-up, timed window, 3-run protocol, cross-run variance, percentiles.

Protocol (plan §4 M1, mirrors `embodied_bench.telemetry.run_eeb_measurement_protocol`):

  for C in concurrency list:
    for run in 1..runs:
      * C sessions (one per LoadGen slot = thread + own connection) are reset; the first
        Predict of each session gives TTFA (reset -> first action chunk, cold per session);
      * each session performs `warmup_calls` paced calls (excluded from statistics);
      * barrier; a shared budget of `timed_calls` Predict calls is drained by the C sessions
        with closed-loop pacing (think-time = execute_horizon x dt). Every call is recorded;
      * run statistics: p50/p95/p99/mean/max obs->action latency, TTFA, chunks/s, actions/s,
        deadline-miss rate, transport overhead (client latency - server compute).
    cross-run spread = (max - min) / median x 100 on p50 latency and chunks/s; must be
    < max_variance_pct (5 %) for the point to be admissible; the reported value is the
    median over runs (MLPerf style).

All timings are real wall-clock (`time.perf_counter()`); nothing here is modelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import logging
import math
import os
import platform
import socket
import statistics
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np

from eeb.loadgen.client import InProcessPolicyClient, PolicyClient, open_client
from eeb.loadgen.pacing import PacingConfig, SessionRouter, SessionSchedule, verify_sticky_trace
from eeb.protocol.policy import Caps, Context, Observation, Policy, instantiate_policy
from eeb.servers.base import parse_endpoint

_LOG = logging.getLogger("eeb.loadgen")

PROTOCOL_MIN_RUNS = 3
PROTOCOL_MIN_TIMED_CALLS = 320
PROTOCOL_MIN_TIMED_CALLS_WM = 200
PROTOCOL_MIN_WARMUP_CALLS = 30
PROTOCOL_MAX_SLOT_COUNT_SPREAD = 0.15
PROTOCOL_MAX_C1_P99_P50_RATIO = 1.5
DEFAULT_MAX_VARIANCE_PCT = 5.0
DEFAULT_MAX_P99_VARIANCE_PCT = 8.0
CANONICAL_CONCURRENCY_GRID: Tuple[int, ...] = (1, 2, 4, 8, 16)
MIN_TIMED_ANTI_LUT_PIXELS = 16


def is_canonical_concurrency_prefix(levels: Sequence[int]) -> bool:
  """Return True iff `levels` is a non-empty ordered prefix of `(1, 2, 4, 8, 16)` starting strictly at `C=1`."""
  if not levels or len(levels) > len(CANONICAL_CONCURRENCY_GRID):
    return False
  norm = tuple(int(c) for c in levels)
  return norm == CANONICAL_CONCURRENCY_GRID[: len(norm)]


def _apply_timed_anti_lut_jitter(
    img: np.ndarray,
    num_pixels: int = MIN_TIMED_ANTI_LUT_PIXELS,
) -> np.ndarray:
  """Apply per-request boundary-safe +/-1 uint8 LSB jitter on >=16 cryptographically random coordinates.

  Defeats pixel-hash, row-stride, and partial-frame lookup tables across all timed evaluation frames
  (including public `EmbodiedX-Traces-v1` episodes) without mutating `img` or shifting semantic
  vision features beyond 1/255 LSB.
  """
  arr = np.array(img, copy=True)
  if arr.size == 0 or arr.dtype != np.uint8:
    return arr
  flat = arr.reshape(-1)
  n_total = int(flat.shape[0])
  k = min(max(int(num_pixels), 1), n_total)
  raw_rand = np.frombuffer(os.urandom(k * 9), dtype=np.uint8)
  idx_words = np.frombuffer(raw_rand[: k * 8].tobytes(), dtype=np.uint64)
  chosen = (idx_words % np.uint64(n_total)).astype(np.intp)
  if len(np.unique(chosen)) < k and n_total >= k:
    seen = set()
    uniq: List[int] = []
    for cand in chosen.tolist():
      c_i = int(cand)
      while c_i in seen:
        c_i = (c_i + 1) % n_total
      seen.add(c_i)
      uniq.append(c_i)
    chosen = np.asarray(uniq, dtype=np.intp)
  dir_bits = raw_rand[k * 8 : k * 9]
  for pos, idx in enumerate(chosen.tolist()):
    v = int(flat[idx])
    if v == 0:
      flat[idx] = np.uint8(1)
    elif v == 255:
      flat[idx] = np.uint8(254)
    else:
      flat[idx] = np.uint8(v + (1 if (int(dir_bits[pos]) & 1) else -1))
  return arr


# ----------------------------------------------------------------------------- observations


@dataclass
class ObservationSpec:
  """Shape of the synthetic observations sent by the LoadGen (warm-up / harness validation only).

  Real traces (EmbodiedX-Traces-v1, WS-C) plug in through `ObservationSource`.
  """

  cameras: Tuple[str, ...] = ("agentview", "wrist")
  image_hw: Tuple[int, int] = (224, 224)
  channels: int = 3
  proprio_dim: int = 8
  instructions: Tuple[str, ...] = (
      "pick up the black bowl and place it on the plate",
      "put the wine bottle on the rack",
      "open the top drawer and put the bowl inside",
  )
  pool_size: int = 16
  seed: int = 1234

  @classmethod
  def from_caps(cls, caps: Caps, **overrides: Any) -> "ObservationSpec":
    kwargs: Dict[str, Any] = {}
    if caps.cameras:
      kwargs["cameras"] = tuple(caps.cameras)
    if caps.image_height > 0 and caps.image_width > 0:
      kwargs["image_hw"] = (int(caps.image_height), int(caps.image_width))
    if caps.proprio_dim > 0:
      kwargs["proprio_dim"] = int(caps.proprio_dim)
    kwargs.update(overrides)
    return cls(**kwargs)


class ObservationSource:
  """Returns an `Observation` for (session, step). Default: seeded synthetic pool."""

  kind = "SYNTHETIC_RANDOM_TENSORS"

  def __init__(self, spec: Optional[ObservationSpec] = None) -> None:
    self.spec = spec or ObservationSpec()
    rng = np.random.default_rng(self.spec.seed)
    h, w, c = self.spec.image_hw[0], self.spec.image_hw[1], self.spec.channels
    self._pool: Dict[str, List[np.ndarray]] = {
        cam: [rng.integers(0, 256, size=(h, w, c), dtype=np.uint8) for _ in range(self.spec.pool_size)]
        for cam in self.spec.cameras
    }
    self._proprio_pool = [rng.uniform(-1.0, 1.0, size=(self.spec.proprio_dim,)).astype(np.float32) for _ in range(self.spec.pool_size)]

  def make(self, session_id: str, step: int) -> Observation:
    import zlib

    idx = (zlib.crc32(session_id.encode("utf-8")) + int(step)) % self.spec.pool_size
    return Observation(
        session_id=session_id,
        step=int(step),
        images={cam: self._pool[cam][idx] for cam in self.spec.cameras},
        proprio=self._proprio_pool[idx],
        instruction=self.spec.instructions[idx % len(self.spec.instructions)],
    )

  def make_rollout_inputs(
      self,
      session_id: str,
      step: int,
      num_frames: int = 16,
      action_dim: int = 7,
  ) -> Tuple[Context, np.ndarray]:
    """Return `(Context, actions)` for Track 2 (`WM_TRACK`) rollouts."""
    import zlib

    obs = self.make(session_id, step)
    first_cam = self.spec.cameras[0] if self.spec.cameras else None
    cond_frames = [obs.images[first_cam]] if first_cam and first_cam in obs.images else []
    ctx = Context(
        session_id=session_id,
        frames=cond_frames,
        proprio=obs.proprio,
        instruction=obs.instruction,
        num_frames=int(num_frames),
    )
    seed_u32 = (zlib.crc32(session_id.encode("utf-8")) + int(step)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed_u32)
    actions = rng.uniform(-1.0, 1.0, size=(int(num_frames), int(action_dim))).astype(np.float32)
    return ctx, actions

  def describe(self) -> Dict[str, Any]:
    return {
        "kind": self.kind,
        "cameras": list(self.spec.cameras),
        "image_hw": list(self.spec.image_hw),
        "channels": int(self.spec.channels),
        "proprio_dim": int(self.spec.proprio_dim),
        "pool_size": int(self.spec.pool_size),
        "bytes_per_request_images": int(
            len(self.spec.cameras) * self.spec.image_hw[0] * self.spec.image_hw[1] * self.spec.channels
        ),
    }


class TraceBundleObservationSource(ObservationSource):
  """Serves real recorded observations and action sequences from a packed `EmbodiedX-Traces-v1` split."""

  kind = "EMBODIEDX_TRACES_V1_BUNDLE"

  def __init__(self, traces_path: str, split: str = "public") -> None:
    import json
    from pathlib import Path
    from eeb.data.pack import find_traces_dir, load_split
    from eeb.data.schema import (
        EmbodiedXTrace,
        ManifestEpisode,
        SYNTHETIC_CI_SMOKE_TRACES_KIND,
        TraceManifest,
        is_synthetic_trace_manifest,
    )

    self.traces_path = str(traces_path)
    self.split = str(split)
    traces_dir = find_traces_dir(traces_path)
    split_dir = traces_dir / split if (traces_dir / split).is_dir() else traces_dir
    man_file = split_dir / "manifest.json"
    raw_man = json.loads(man_file.read_text(encoding="utf-8")) if man_file.exists() else {}
    if not raw_man and (traces_dir / "manifest.json").exists():
      try:
        raw_man = json.loads((traces_dir / "manifest.json").read_text(encoding="utf-8"))
      except (OSError, ValueError):
        raw_man = {}
    self._raw_manifest: Dict[str, Any] = raw_man if isinstance(raw_man, dict) else {}
    self._is_synthetic_smoke: bool = is_synthetic_trace_manifest(self._raw_manifest)
    if self._is_synthetic_smoke:
      self.kind = SYNTHETIC_CI_SMOKE_TRACES_KIND
    if isinstance(raw_man, dict) and "shards" in raw_man and "episodes" not in raw_man:
      from embodied_bench.dataset_loader import EmbodiedXDatasetLoader

      loader = EmbodiedXDatasetLoader(dataset_dir=split_dir, split=str(raw_man.get("split", split)))
      shards = loader.preload_into_pinned_ram(verify=True)
      self.traces = []
      ep_entries: List[ManifestEpisode] = []
      for idx, sh in enumerate(shards):
        t_steps = sh.num_steps
        imgs = {
            "agentview": np.ascontiguousarray(np.transpose(sh.rgb_frames[:, 0], (0, 2, 3, 1)), dtype=np.uint8),
            "wrist": np.ascontiguousarray(np.transpose(sh.rgb_frames[:, 1], (0, 2, 3, 1)), dtype=np.uint8),
        }
        tr = EmbodiedXTrace(
            episode_id=sh.episode_id,
            source_dataset=sh.source_dataset,
            split="public",
            instruction=sh.instruction_text,
            fps=20.0,
            t_ns=np.arange(t_steps, dtype=np.int64) * 50_000_000,
            step=np.arange(t_steps, dtype=np.int64),
            proprio=np.ascontiguousarray(sh.proprio_state, dtype=np.float32),
            action=np.ascontiguousarray(sh.gt_actions[:, 0, :], dtype=np.float32),
            images=imgs,
            license=sh.license_spdx,
        )
        self.traces.append(tr)
        shard_meta = raw_man["shards"][idx] if idx < len(raw_man["shards"]) else {}
        ep_entries.append(
            ManifestEpisode(
                episode_id=sh.episode_id,
                file=str(shard_meta.get("filename", f"episode_{idx:02d}.npz")),
                steps=t_steps,
                instruction=sh.instruction_text,
                sha256=str(shard_meta.get("sha256", "")),
                content_sha256=tr.content_sha256(),
                source_episode_index=idx,
                cameras=["agentview", "wrist"],
            )
        )
      self.manifest = TraceManifest(
          source=str(split_dir.name),
          source_dataset=",".join(raw_man.get("sources", ["LIBERO"])),
          license="MIT/CC-BY-4.0/Apache-2.0",
          fps=20.0,
          cameras=["agentview", "wrist"],
          split="public",
          episodes=ep_entries,
      )
    else:
      self.manifest = TraceManifest.load(split_dir)
      if (traces_dir / split).is_dir():
        self.traces = load_split(traces_dir, split)
      else:
        ver = self.manifest.verify(split_dir)
        if not ver.ok:
          raise ValueError(ver.summary())
        self.traces = [EmbodiedXTrace.from_npz(split_dir / e.file) for e in self.manifest.episodes]
    if not self.traces:
      raise ValueError(f"No episodes found in trace bundle {traces_path!r} (split={split!r}).")
    first = self.traces[0]
    cams = first.cameras
    first_shape = first.images[cams[0]].shape[1:3] if cams else (224, 224)
    self.spec = ObservationSpec(
        cameras=cams,
        image_hw=(int(first_shape[0]), int(first_shape[1])),
        channels=3,
        proprio_dim=int(first.proprio.shape[1]),
        instructions=tuple({t.instruction for t in self.traces}),
        pool_size=sum(len(t) for t in self.traces),
    )

  def _pick_episode_and_index(self, session_id: str, step: int) -> Tuple[Any, int]:
    import zlib

    ep_idx = zlib.crc32(session_id.encode("utf-8")) % len(self.traces)
    trace = self.traces[ep_idx]
    frame_idx = int(step) % len(trace)
    return trace, frame_idx

  def make(self, session_id: str, step: int) -> Observation:
    trace, idx = self._pick_episode_and_index(session_id, step)
    ts = trace[idx]
    return Observation(
        session_id=session_id,
        step=int(step),
        images=ts.images,
        proprio=ts.proprio,
        instruction=ts.instruction,
    )

  def make_rollout_inputs(
      self,
      session_id: str,
      step: int,
      num_frames: int = 16,
      action_dim: int = 7,
  ) -> Tuple[Context, np.ndarray]:
    trace, idx = self._pick_episode_and_index(session_id, step)
    ts = trace[idx]
    cams = trace.cameras
    cond_frames = [ts.images[cams[0]]] if cams else []
    n = int(num_frames)
    actions_list: List[np.ndarray] = []
    for k in range(n):
      act = trace.action[(idx + k) % len(trace)]
      if act.shape[0] >= action_dim:
        actions_list.append(act[:action_dim])
      else:
        padded = np.zeros((action_dim,), dtype=np.float32)
        padded[: act.shape[0]] = act
        actions_list.append(padded)
    actions = np.stack(actions_list).astype(np.float32)
    ctx = Context(
        session_id=session_id,
        frames=cond_frames,
        proprio=ts.proprio,
        instruction=ts.instruction,
        num_frames=n,
    )
    return ctx, actions

  def describe(self) -> Dict[str, Any]:
    base = super().describe()
    base.update({
        "kind": self.kind,
        "manifest_kind": self._raw_manifest.get("kind", getattr(self.manifest, "kind", "")),
        "pixel_provenance": self._raw_manifest.get("pixel_provenance"),
        "is_synthetic_smoke": bool(self._is_synthetic_smoke),
        "certifiable": not bool(self._is_synthetic_smoke),
        "traces_path": self.traces_path,
        "split": self.split,
        "source": self.manifest.source,
        "source_dataset": self.manifest.source_dataset,
        "episode_count": len(self.traces),
        "step_count": int(self.spec.pool_size),
    })
    return base


# ----------------------------------------------------------------------------- config / records


@dataclass
class SweepConfig:
  concurrency: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
  warmup_calls: int = PROTOCOL_MIN_WARMUP_CALLS
  timed_calls: int = PROTOCOL_MIN_TIMED_CALLS
  runs: int = PROTOCOL_MIN_RUNS
  max_variance_pct: float = DEFAULT_MAX_VARIANCE_PCT
  max_p99_variance_pct: Optional[float] = None
  inline_canaries_enabled: bool = True
  pacing_enabled: bool = False
  execute_horizon: Optional[int] = None
  dt_ms_override: Optional[float] = None
  call_timeout_s: float = 60.0
  server_warmup_calls: int = 5
  keep_samples: bool = True

  def __post_init__(self) -> None:
    self.concurrency = [int(c) for c in self.concurrency]
    if not self.concurrency or any(c <= 0 for c in self.concurrency):
      raise ValueError(f"SweepConfig.concurrency must be a non-empty list of positive ints, got {self.concurrency}.")
    if self.timed_calls <= 0 or self.runs <= 0 or self.warmup_calls < 0:
      raise ValueError("SweepConfig requires timed_calls > 0, runs > 0, warmup_calls >= 0.")
    if self.max_variance_pct <= 0.0:
      raise ValueError("SweepConfig.max_variance_pct must be > 0.")
    if self.max_p99_variance_pct is not None and float(self.max_p99_variance_pct) <= 0.0:
      raise ValueError("SweepConfig.max_p99_variance_pct must be > 0 when specified.")

  @property
  def effective_max_p99_variance_pct(self) -> float:
    if self.max_p99_variance_pct is not None:
      return float(self.max_p99_variance_pct)
    return max(DEFAULT_MAX_P99_VARIANCE_PCT, float(self.max_variance_pct))

  def min_timed_calls_for_track(self, track: Optional[str] = None) -> int:
    if str(track or "").upper() == "WM_TRACK":
      return PROTOCOL_MIN_TIMED_CALLS_WM
    return PROTOCOL_MIN_TIMED_CALLS

  def protocol_compliant_for_track(self, track: Optional[str] = None) -> bool:
    return (
        is_canonical_concurrency_prefix(self.concurrency)
        and self.runs >= PROTOCOL_MIN_RUNS
        and self.timed_calls >= self.min_timed_calls_for_track(track)
        and self.warmup_calls >= PROTOCOL_MIN_WARMUP_CALLS
    )

  @property
  def protocol_compliant(self) -> bool:
    return self.protocol_compliant_for_track(None)

  def protocol_notes_for_track(self, track: Optional[str] = None) -> List[str]:
    min_calls = self.min_timed_calls_for_track(track)
    notes: List[str] = []
    if not is_canonical_concurrency_prefix(self.concurrency):
      notes.append(
          f"concurrency={self.concurrency} is not an ordered prefix of canonical grid {list(CANONICAL_CONCURRENCY_GRID)} starting at C=1"
      )
    if self.runs < PROTOCOL_MIN_RUNS:
      notes.append(f"runs={self.runs} < {PROTOCOL_MIN_RUNS} required by the 3-run protocol")
    if self.timed_calls < min_calls:
      notes.append(f"timed_calls={self.timed_calls} < {min_calls} required timed window")
    if self.warmup_calls < PROTOCOL_MIN_WARMUP_CALLS:
      notes.append(f"warmup_calls={self.warmup_calls} < {PROTOCOL_MIN_WARMUP_CALLS} required warm-up")
    return notes

  def protocol_notes(self) -> List[str]:
    return self.protocol_notes_for_track(None)

  def to_dict(self, track: Optional[str] = None) -> Dict[str, Any]:
    d = asdict(self)
    d["effective_max_p99_variance_pct"] = self.effective_max_p99_variance_pct
    d["protocol_compliant"] = self.protocol_compliant_for_track(track)
    d["protocol_notes"] = self.protocol_notes_for_track(track)
    return d


@dataclass
class CallRecord:
  session_id: str
  step: int
  slot: int
  concurrency: int
  run_idx: int
  t_send_rel_s: float
  latency_ms: float
  server_compute_ms: Optional[float]
  horizon: int
  dim: int
  deadline_missed: bool
  in_timed_window: bool
  request_bytes: int = 0
  response_bytes: int = 0
  shape_ok: bool = True
  finite_ok: bool = True
  error: Optional[str] = None
  ttff_ms: Optional[float] = None
  action_jerk: Optional[float] = None


def _compute_chunk_normalized_jerk(actions: np.ndarray) -> float:
  arr = np.asarray(actions, dtype=np.float64)
  if arr.ndim < 2 or arr.shape[-2] < 2 or arr.size == 0 or not np.isfinite(arr).all():
    return 0.0
  order = min(3, int(arr.shape[-2]) - 1)
  diff = np.diff(arr, n=order, axis=-2)
  jerk_rms = float(np.sqrt(np.mean(diff ** 2)))
  act_rms = float(np.sqrt(np.mean(arr ** 2)))
  return round(float(jerk_rms / max(act_rms, 1e-6)), 6)


def _percentiles(values: Sequence[float], ps: Sequence[int] = (50, 95, 99)) -> Dict[str, float]:
  if not values:
    return {f"p{p}": float("nan") for p in ps}
  arr = np.asarray(values, dtype=np.float64)
  out = {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in ps}
  out["mean"] = round(float(arr.mean()), 4)
  out["min"] = round(float(arr.min()), 4)
  out["max"] = round(float(arr.max()), 4)
  out["std"] = round(float(arr.std(ddof=0)), 4)
  out["n"] = int(arr.size)
  return out


@dataclass
class RunResult:
  concurrency: int
  run_idx: int
  window_start_wall: float
  window_end_wall: float
  window_s: float
  num_timed_calls: int
  num_warmup_calls: int
  horizon: int
  dim: int
  execute_horizon: int
  latency_ms: Dict[str, float]
  server_compute_ms: Dict[str, float]
  ttfa_ms: Dict[str, float]
  chunks_per_sec: float
  actions_per_sec: float
  executed_actions_per_sec: float
  deadline_miss_rate: float
  transport_overhead_p50_ms: Optional[float]
  pacing_oversleep_p99_ms: float
  sticky_routing_ok: bool
  sticky_violations: List[str]
  num_errors: int
  errors: List[str]
  actions_sha256: str = ""
  num_shape_mismatch: int = 0
  num_nonfinite: int = 0
  deterministic_repeat: Optional[bool] = None
  samples_latency_ms: List[float] = field(default_factory=list)
  ttff_ms: Dict[str, float] = field(default_factory=dict)
  normalized_action_jerk: Optional[float] = None
  paced_wall_chunks_per_sec: Optional[float] = None
  paced_wall_executed_actions_per_sec: Optional[float] = None
  active_service_chunks_per_sec: Optional[float] = None
  num_inline_canaries: int = 0
  num_inline_canary_failures: int = 0
  inline_canary_reasons: List[str] = field(default_factory=list)
  predictions_per_sec: Optional[float] = None
  slot_counts: List[int] = field(default_factory=list)
  slot_count_spread: float = 0.0
  c1_p99_p50_ratio: Optional[float] = None

  @property
  def passed(self) -> bool:
    return (
        self.num_errors == 0
        and self.sticky_routing_ok
        and self.num_timed_calls > 0
        and self.num_shape_mismatch == 0
        and self.num_nonfinite == 0
        and self.num_inline_canary_failures == 0
    )

  def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
    d = asdict(self)
    d["passed"] = self.passed
    if not include_samples:
      d.pop("samples_latency_ms", None)
    return d


def cross_run_spread_pct(values: Sequence[float]) -> float:
  """(max - min) / median x 100 - the same definition as embodied_bench.telemetry."""
  vals = [float(v) for v in values if v == v]  # drop NaN
  if len(vals) < 2:
    return 0.0
  med = statistics.median(vals)
  return round(((max(vals) - min(vals)) / max(abs(med), 1e-9)) * 100.0, 3)


@dataclass
class ConcurrencyResult:
  concurrency: int
  runs: List[RunResult]
  median_latency_ms: Dict[str, float]
  median_ttfa_ms: Dict[str, float]
  median_chunks_per_sec: float
  median_actions_per_sec: float
  median_executed_actions_per_sec: float
  median_deadline_miss_rate: float
  cross_run_spread_pct: Dict[str, float]
  max_variance_pct: float
  variance_passed: bool
  variance_reason: str
  all_runs_passed: bool
  median_ttff_ms: Dict[str, float] = field(default_factory=dict)
  median_normalized_action_jerk: Optional[float] = None
  median_predictions_per_sec: Optional[float] = None
  max_slot_count_spread: float = 0.0
  slot_starvation_passed: bool = True
  c1_tail_ratio_p99_p50: Optional[float] = None
  c1_tail_ratio_passed: bool = True

  @property
  def passed(self) -> bool:
    return self.variance_passed and self.all_runs_passed and self.slot_starvation_passed

  def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
    return {
        "concurrency": self.concurrency,
        "median_latency_ms": self.median_latency_ms,
        "median_ttfa_ms": self.median_ttfa_ms,
        "median_ttff_ms": self.median_ttff_ms,
        "median_normalized_action_jerk": self.median_normalized_action_jerk,
        "median_chunks_per_sec": self.median_chunks_per_sec,
        "median_predictions_per_sec": self.median_predictions_per_sec,
        "median_actions_per_sec": self.median_actions_per_sec,
        "median_executed_actions_per_sec": self.median_executed_actions_per_sec,
        "median_deadline_miss_rate": self.median_deadline_miss_rate,
        "cross_run_spread_pct": self.cross_run_spread_pct,
        "max_variance_pct": self.max_variance_pct,
        "variance_passed": self.variance_passed,
        "variance_reason": self.variance_reason,
        "max_slot_count_spread": self.max_slot_count_spread,
        "slot_starvation_passed": self.slot_starvation_passed,
        "c1_tail_ratio_p99_p50": self.c1_tail_ratio_p99_p50,
        "c1_tail_ratio_passed": self.c1_tail_ratio_passed,
        "all_runs_passed": self.all_runs_passed,
        "passed": self.passed,
        "runs": [r.to_dict(include_samples=include_samples) for r in self.runs],
    }


@dataclass
class SweepResult:
  endpoint: str
  transport: str
  caps: Caps
  config: SweepConfig
  pacing: PacingConfig
  observation_source: Dict[str, Any]
  started_at_wall: float
  finished_at_wall: float
  time_to_ready_s: float
  points: List[ConcurrencyResult]
  host: Dict[str, Any]

  @property
  def all_passed(self) -> bool:
    return all(p.passed for p in self.points)

  def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
    return {
        "endpoint": self.endpoint,
        "transport": self.transport,
        "caps": self.caps.to_dict(),
        "config": self.config.to_dict(track=self.caps.track),
        "pacing": self.pacing.to_dict(),
        "observation_source": self.observation_source,
        "started_at_wall": self.started_at_wall,
        "finished_at_wall": self.finished_at_wall,
        "wall_duration_s": round(self.finished_at_wall - self.started_at_wall, 3),
        "time_to_ready_s": self.time_to_ready_s,
        "all_passed": self.all_passed,
        "host": self.host,
        "points": [p.to_dict(include_samples=include_samples) for p in self.points],
    }


def host_info() -> Dict[str, Any]:
  import os

  kernel_ver = platform.release().split("-")[0]
  return {
      "hostname": os.environ.get("EEB_HOSTNAME", "benchmark-host"),
      "platform": f"{platform.system()}-{kernel_ver}-{platform.machine()}",
      "machine": platform.machine(),
      "python": platform.python_version(),
      "cpu_count": os.cpu_count(),
  }


# ----------------------------------------------------------------------------- the sweep


RunHook = Callable[[int, int], None]


class ConcurrencySweep:
  """Drives the sweep against one endpoint (gRPC or in-process)."""

  def __init__(
      self,
      endpoint: str,
      config: SweepConfig,
      policy: Optional[Policy] = None,
      policy_kwargs: Optional[Dict[str, Any]] = None,
      observation_source: Optional[ObservationSource] = None,
      on_run_start: Optional[RunHook] = None,
      on_run_end: Optional[Callable[[int, int, RunResult], None]] = None,
      log: Optional[Callable[[str], None]] = None,
      golden: Optional[Dict[str, Any]] = None,
  ) -> None:
    self.endpoint = endpoint
    self.config = config
    self._ep = parse_endpoint(endpoint)
    self._policy = policy
    if self._ep.is_inproc and self._policy is None:
      self._policy = instantiate_policy(self._ep.target, policy_kwargs)
    self._observation_source = observation_source
    self.on_run_start = on_run_start
    self.on_run_end = on_run_end
    self.log = log or (lambda msg: _LOG.info("%s", msg))
    self._golden = golden
    self.caps: Optional[Caps] = None
    self.pacing: Optional[PacingConfig] = None
    self.transport = "inproc" if self._ep.is_inproc else ("http" if self._ep.is_http else "grpc")
    self.call_records: List[CallRecord] = []
    self._records_lock = threading.Lock()

  # -- clients ---------------------------------------------------------------

  def _open(self) -> PolicyClient:
    if self._ep.is_inproc:
      return InProcessPolicyClient(self._policy)  # type: ignore[arg-type]
    return open_client(self.endpoint, timeout_s=self.config.call_timeout_s)

  def _record(self, rec: CallRecord) -> None:
    with self._records_lock:
      self.call_records.append(rec)

  # -- phases ----------------------------------------------------------------

  def _discover(self) -> Tuple[Caps, PacingConfig, ObservationSource]:
    with self._open() as client:
      caps = client.capabilities()
    dt_ms = float(self.config.dt_ms_override) if self.config.dt_ms_override else float(caps.dt_ms)
    exec_h = (
        int(self.config.execute_horizon)
        if self.config.execute_horizon is not None
        else (int(caps.execute_horizon) if caps.execute_horizon is not None else int(caps.horizon))
    )
    pacing = PacingConfig(
        dt_ms=dt_ms,
        horizon=int(caps.horizon),
        execute_horizon=exec_h,
        enabled=self.config.pacing_enabled,
    )
    source = self._observation_source or ObservationSource(ObservationSpec.from_caps(caps))
    self._observation_source = source
    return caps, pacing, source

  def _server_warmup(self, source: ObservationSource) -> float:
    """A few calls on a throw-away session so first-call compilation does not pollute TTFA."""
    assert self.caps is not None
    n = max(0, int(self.config.server_warmup_calls))
    t0 = time.perf_counter()
    if n == 0:
      return 0.0
    is_wm = self.caps.track == "WM_TRACK"
    warmup_sid = uuid.uuid4().hex
    with self._open() as client:
      client.reset(warmup_sid)
      for step in range(n):
        if is_wm:
          ctx, actions = source.make_rollout_inputs(
              warmup_sid,
              step,
              num_frames=int(self.caps.horizon),
              action_dim=int(self.caps.action_dim or 7),
          )
          client.rollout(ctx, actions)
        else:
          client.predict(source.make(warmup_sid, step))
    return round(time.perf_counter() - t0, 4)

  def _run_once(self, c: int, run_idx: int, source: ObservationSource) -> RunResult:
    assert self.caps is not None and self.pacing is not None
    cfg, pacing = self.config, self.pacing
    is_wm = self.caps.track == "WM_TRACK"
    router = SessionRouter(num_slots=c)
    n_slot = max(1, math.ceil(int(cfg.timed_calls) / max(c, 1)))
    done_lock = threading.Lock()
    slots_done = {"count": 0}
    all_slots_done = threading.Event()
    t_steady_end_mono = {"t": 0.0}
    t_window_start_mono = {"t": 0.0}
    t_window_start_wall = {"t": 0.0}

    def _stamp_window_start() -> None:
      t_window_start_mono["t"] = time.perf_counter()
      t_window_start_wall["t"] = time.time()

    barrier = threading.Barrier(c + 1, action=_stamp_window_start)
    per_slot_last_recv: List[float] = [0.0] * c
    slot_lats: List[List[float]] = [[] for _ in range(c)]
    ttfa_ms: List[float] = []
    ttfa_lock = threading.Lock()
    timed_records: List[CallRecord] = []
    timed_lock = threading.Lock()
    canary_lock = threading.Lock()
    canary_stats: Dict[str, Any] = {"checked": 0, "failed": 0, "reasons": []}
    errors: List[str] = []
    oversleeps: List[float] = []
    sticky_errors: List[str] = []
    slot_digests: List[str] = [""] * c
    repeat_checks: List[bool] = []
    expected_h, expected_d = int(self.caps.horizon), int(self.caps.action_dim)
    expected_img_hw = (int(self.caps.image_height or 64), int(self.caps.image_width or 64))

    golden_obj = self._golden if cfg.inline_canaries_enabled else None
    golden_n = 0
    golden_usable = False
    req_cams_golden: Tuple[str, ...] = ()
    if isinstance(golden_obj, dict) and int(golden_obj.get("n", 0) or 0) > 0:
      gn = int(golden_obj["n"])
      prop_g = golden_obj.get("proprio") if "proprio" in golden_obj else golden_obj.get("proprios")
      instr_g = golden_obj.get("instructions")
      if is_wm:
        frm_g = golden_obj.get("ref_frames") if "ref_frames" in golden_obj else golden_obj.get("frames")
        act_g = golden_obj.get("ref_actions") if "ref_actions" in golden_obj else golden_obj.get("actions")
        cams_g = golden_obj.get("cameras", {})
        cf_g = golden_obj.get("context_frames")
        cond_shape_ok = False
        if cams_g:
          ck0 = sorted(cams_g.keys())[0]
          cond_shape_ok = tuple(cams_g[ck0].shape[1:3]) == expected_img_hw
        elif cf_g is not None:
          cond_shape_ok = tuple(cf_g.shape[-3:-1]) == expected_img_hw
        if (
            frm_g is not None
            and act_g is not None
            and prop_g is not None
            and instr_g is not None
            and frm_g.ndim == 5
            and tuple(frm_g.shape[1:4]) == (expected_h, expected_img_hw[0], expected_img_hw[1])
            and tuple(act_g.shape[1:]) == (expected_h, expected_d or 7)
            and cond_shape_ok
        ):
          golden_usable = True
          golden_n = gn
      else:
        act_g = golden_obj.get("ref_actions") if "ref_actions" in golden_obj else golden_obj.get("actions")
        cams_g = golden_obj.get("cameras", {})
        req_cams_golden = tuple(self.caps.cameras) if self.caps.cameras else tuple(sorted(cams_g.keys()))
        cams_ok = bool(req_cams_golden) and all(
            cam in cams_g and tuple(cams_g[cam].shape[1:3]) == expected_img_hw
            for cam in req_cams_golden
        )
        prop_dim_ok = prop_g is not None and (
            int(self.caps.proprio_dim) <= 0
            or (prop_g.ndim == 2 and int(prop_g.shape[1]) == int(self.caps.proprio_dim))
        )
        if (
            act_g is not None
            and instr_g is not None
            and act_g.ndim == 3
            and tuple(act_g.shape[1:]) == (expected_h, expected_d)
            and cams_ok
            and prop_dim_ok
        ):
          golden_usable = True
          golden_n = gn

    def _record_canary(ok: bool, reason: str = "") -> None:
      with canary_lock:
        canary_stats["checked"] += 1
        if not ok:
          canary_stats["failed"] += 1
          if len(canary_stats["reasons"]) < 20 and reason:
            canary_stats["reasons"].append(reason)

    def worker(slot: int) -> None:
      session_id = uuid.uuid4().hex
      sched = SessionSchedule(session_id=session_id, slot=slot)
      client: Optional[PolicyClient] = None
      hasher = hashlib.sha256()
      rng_slot = np.random.default_rng(int(uuid.uuid4().hex[:8], 16))
      pert_a_set: set[int] = set()
      pert_b_set: set[int] = set()
      golden_map: Dict[int, int] = {}
      if cfg.inline_canaries_enabled:
        used_idx: set[int] = set()
        if n_slot >= 2:
          target_pairs = max(1, round(0.025 * n_slot))
          cand_starts = list(range(n_slot - 1))
          rng_slot.shuffle(cand_starts)
          for ia in cand_starts:
            if len(pert_a_set) >= target_pairs:
              break
            if ia not in used_idx and (ia + 1) not in used_idx:
              pert_a_set.add(ia)
              pert_b_set.add(ia + 1)
              used_idx.add(ia)
              used_idx.add(ia + 1)
        if golden_usable and golden_n > 0:
          rem_idx = [ix for ix in range(n_slot) if ix not in used_idx]
          if rem_idx:
            target_gold = min(len(rem_idx), max(1, round(0.05 * n_slot)))
            chosen = rng_slot.choice(rem_idx, size=target_gold, replace=False)
            for ix in chosen:
              golden_map[int(ix)] = int(rng_slot.integers(0, golden_n))
      pert_cache: Dict[str, Any] = {}

      try:
        client = self._open()
        routed = router.route(session_id, preferred_slot=slot)
        if routed != slot:
          sticky_errors.append(f"router assigned {session_id} to slot {routed}, worker slot is {slot}")

        def one_call(in_timed: bool, local_idx: Optional[int] = None) -> Optional[float]:
          is_pert_a = bool(in_timed and local_idx is not None and local_idx in pert_a_set)
          is_pert_b = bool(in_timed and local_idx is not None and local_idx in pert_b_set and "out" in pert_cache)
          gold_gi = golden_map.get(local_idx) if (in_timed and local_idx is not None) else None

          if is_wm:
            if is_pert_b:
              from eeb.loadgen.gate import BF16_SAFE_PERTURB_STEP, _perturb_bf16_safe_vector, _perturb_wire_image

              base_ctx: Context = pert_cache["ctx"]
              base_act: np.ndarray = pert_cache["act"]
              cond_b = []
              for fr in base_ctx.frames:
                fr_b, _ = _perturb_wire_image(
                    fr,
                    rng_slot,
                    eps_linf=4.0 / 255.0,
                    lsb_delta=4,
                    lsb_fraction=0.25,
                )
                cond_b.append(fr_b)
              prop_b, _ = _perturb_bf16_safe_vector(
                  base_ctx.proprio, rng_slot, step=BF16_SAFE_PERTURB_STEP
              )
              actions_in, _ = _perturb_bf16_safe_vector(
                  base_act, rng_slot, step=BF16_SAFE_PERTURB_STEP
              )
              ctx = Context(
                  session_id=session_id,
                  frames=cond_b,
                  proprio=prop_b,
                  instruction=base_ctx.instruction,
                  num_frames=base_ctx.num_frames,
              )
              call_step = int(pert_cache["step"])
            elif gold_gi is not None and golden_obj is not None:
              cams_g = golden_obj.get("cameras", {})
              cf_g = golden_obj.get("context_frames")
              prop_arr = golden_obj.get("proprio") if "proprio" in golden_obj else golden_obj.get("proprios")
              act_arr = golden_obj.get("ref_actions") if "ref_actions" in golden_obj else golden_obj.get("actions")
              if cams_g:
                ck0 = sorted(cams_g.keys())[0]
                cond_g = [cams_g[ck0][gold_gi]]
              elif cf_g is not None:
                cf_i = cf_g[gold_gi]
                cond_g = [cf_i[j] for j in range(cf_i.shape[0])] if cf_i.ndim == 4 else [cf_i]
              else:
                cond_g = []
              ctx = Context(
                  session_id=session_id,
                  frames=cond_g,
                  proprio=prop_arr[gold_gi],  # type: ignore[index]
                  instruction=golden_obj["instructions"][gold_gi],
                  num_frames=expected_h,
              )
              actions_in = act_arr[gold_gi]  # type: ignore[index]
              call_step = sched.step
            else:
              ctx, actions_in = source.make_rollout_inputs(
                  session_id,
                  sched.step,
                  num_frames=expected_h,
                  action_dim=expected_d or 7,
              )
              if in_timed and ctx.frames:
                ctx = Context(
                    session_id=ctx.session_id,
                    frames=[_apply_timed_anti_lut_jitter(f) for f in ctx.frames],
                    proprio=ctx.proprio,
                    instruction=ctx.instruction,
                    num_frames=ctx.num_frames,
                )
              call_step = sched.step

            router.record(session_id, call_step, slot)
            t_send_mono = time.perf_counter()
            try:
              res_wm = client.rollout(ctx, actions_in)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
              msg = f"{session_id} step {call_step}: {type(exc).__name__}: {str(exc)[:200]}"
              self._record(CallRecord(session_id, call_step, slot, c, run_idx, 0.0, float("nan"), None, 0, 0, False, in_timed, error=msg))
              errors.append(msg)
              sched.step += 1
              return None
            t_recv_mono = time.perf_counter()
            frames = res_wm.frames
            n_frames = len(frames)
            shape_ok = (
                n_frames == expected_h
                and all(f.image.shape[:2] == expected_img_hw and f.image.shape[2] == 3 for f in frames)
            )
            finite_ok = bool(all(np.isfinite(f.image).all() for f in frames))
            if in_timed:
              for f in frames:
                hasher.update(np.ascontiguousarray(f.image, dtype=np.uint8).tobytes())
              if shape_ok and finite_ok:
                if is_pert_a:
                  pert_cache.clear()
                  pert_cache["ctx"] = ctx
                  pert_cache["act"] = actions_in
                  pert_cache["step"] = call_step
                  pert_cache["out"] = np.stack([f.image for f in frames]).astype(np.float32)
                elif is_pert_b:
                  cand_b = np.stack([f.image for f in frames]).astype(np.float64)
                  base_a = np.asarray(pert_cache.get("out"), dtype=np.float64)
                  shift = float(np.linalg.norm(cand_b - base_a)) if cand_b.shape == base_a.shape else 0.0
                  ok_p = bool(np.isfinite(shift) and shift >= 1e-5)
                  _record_canary(
                      ok_p,
                      "" if ok_p else f"slot {slot} idx {local_idx}: inline WM perturbation L2 shift={shift:.3e} < 1e-5",
                  )
                  pert_cache.clear()
                elif gold_gi is not None and golden_obj is not None:
                  frm_arr = golden_obj.get("ref_frames") if "ref_frames" in golden_obj else golden_obj.get("frames")
                  ref_f = np.asarray(frm_arr[gold_gi], dtype=np.float32)  # type: ignore[index]
                  cand_f = np.stack([f.image for f in frames]).astype(np.float32)
                  mse_g = float(np.mean((cand_f - ref_f) ** 2))
                  psnr_g = 99.0 if mse_g <= 1e-9 else float(10.0 * np.log10((255.0 ** 2) / mse_g))
                  den_g = float(np.linalg.norm(ref_f.reshape(-1)) * np.linalg.norm(cand_f.reshape(-1)))
                  cos_g = float(np.dot(ref_f.reshape(-1), cand_f.reshape(-1)) / max(den_g, 1e-9))
                  ok_g = bool(psnr_g >= 24.0 and cos_g >= 0.985)
                  _record_canary(
                      ok_g,
                      "" if ok_g else f"slot {slot} golden[{gold_gi}]: inline WM canary psnr={psnr_g:.2f}dB cos={cos_g:.5f}",
                  )
            rec = CallRecord(
                session_id=session_id,
                step=call_step,
                slot=slot,
                concurrency=c,
                run_idx=run_idx,
                t_send_rel_s=round(t_send_mono - t_window_start_mono["t"], 6) if in_timed else -1.0,
                latency_ms=res_wm.total_ms,
                server_compute_ms=None,
                horizon=n_frames,
                dim=expected_d,
                deadline_missed=pacing.deadline_missed(res_wm.total_ms),
                in_timed_window=in_timed,
                shape_ok=shape_ok,
                finite_ok=finite_ok,
                ttff_ms=res_wm.ttff_ms,
            )
            self._record(rec)
            if in_timed:
              with timed_lock:
                timed_records.append(rec)
              slot_lats[slot].append(res_wm.total_ms)
              per_slot_last_recv[slot] = t_recv_mono
            sched.step += 1
            sched.arm(pacing, t_recv_mono)
            return res_wm.total_ms

          if is_pert_b:
            from eeb.loadgen.gate import BF16_SAFE_PERTURB_STEP, _perturb_bf16_safe_vector, _perturb_wire_image

            base_obs: Observation = pert_cache["obs"]
            imgs_b: Dict[str, np.ndarray] = {}
            for cname in sorted(base_obs.images.keys()):
              arr_b, _ = _perturb_wire_image(
                  base_obs.images[cname],
                  rng_slot,
                  eps_linf=4.0 / 255.0,
                  lsb_delta=4,
                  lsb_fraction=0.25,
              )
              imgs_b[cname] = arr_b
            prop_b, _ = _perturb_bf16_safe_vector(
                base_obs.proprio, rng_slot, step=BF16_SAFE_PERTURB_STEP
            )
            call_step = int(base_obs.step)
            obs = Observation(
                session_id=session_id,
                step=call_step,
                images=imgs_b,
                proprio=prop_b,
                instruction=base_obs.instruction,
            )
          elif gold_gi is not None and golden_obj is not None:
            prop_arr = golden_obj.get("proprio") if "proprio" in golden_obj else golden_obj.get("proprios")
            call_step = int(gold_gi)
            obs = Observation(
                session_id=session_id,
                step=call_step,
                images={cam: golden_obj["cameras"][cam][gold_gi] for cam in req_cams_golden},
                proprio=prop_arr[gold_gi],  # type: ignore[index]
                instruction=golden_obj["instructions"][gold_gi],
            )
          else:
            call_step = sched.step
            obs = source.make(session_id, call_step)
            if in_timed and obs.images:
              obs = Observation(
                  session_id=obs.session_id,
                  step=obs.step,
                  images={cam: _apply_timed_anti_lut_jitter(im) for cam, im in obs.images.items()},
                  proprio=obs.proprio,
                  instruction=obs.instruction,
              )

          router.record(session_id, call_step, slot)
          t_send_mono = time.perf_counter()
          try:
            res = client.predict(obs)  # type: ignore[union-attr]
          except Exception as exc:  # noqa: BLE001 - every transport error is a measurement error
            msg = f"{session_id} step {call_step}: {type(exc).__name__}: {str(exc)[:200]}"
            self._record(CallRecord(session_id, call_step, slot, c, run_idx, 0.0, float("nan"), None, 0, 0, False, in_timed, error=msg))
            errors.append(msg)
            sched.step += 1
            return None
          t_recv_mono = time.perf_counter()
          actions = res.chunk.actions
          shape_ok = actions.shape == (expected_h, expected_d)
          finite_ok = bool(np.isfinite(actions).all())
          call_jerk = _compute_chunk_normalized_jerk(actions) if finite_ok else 0.0
          if in_timed:
            hasher.update(np.ascontiguousarray(actions, dtype=np.float32).tobytes())
            if shape_ok and finite_ok:
              if is_pert_a:
                pert_cache.clear()
                pert_cache["obs"] = obs
                pert_cache["out"] = np.array(actions, copy=True, dtype=np.float32)
              elif is_pert_b:
                cand_b = np.asarray(actions, dtype=np.float64)
                base_a = np.asarray(pert_cache.get("out"), dtype=np.float64)
                shift = float(np.linalg.norm(cand_b - base_a)) if cand_b.shape == base_a.shape else 0.0
                ok_p = bool(np.isfinite(shift) and shift >= 1e-5)
                _record_canary(
                    ok_p,
                    "" if ok_p else f"slot {slot} idx {local_idx}: inline VLA perturbation L2 shift={shift:.3e} < 1e-5",
                )
                pert_cache.clear()
              elif gold_gi is not None and golden_obj is not None:
                act_arr = golden_obj.get("ref_actions") if "ref_actions" in golden_obj else golden_obj.get("actions")
                ref_a = np.asarray(act_arr[gold_gi], dtype=np.float32).reshape(-1)  # type: ignore[index]
                cand_a = np.asarray(actions, dtype=np.float32).reshape(-1)
                den_g = float(np.linalg.norm(ref_a) * np.linalg.norm(cand_a))
                cos_g = float(np.dot(ref_a, cand_a) / max(den_g, 1e-9))
                ok_g = bool(np.isfinite(cos_g) and cos_g >= 0.99)
                _record_canary(
                    ok_g,
                    "" if ok_g else f"slot {slot} golden[{gold_gi}]: inline VLA canary cosine_sim={cos_g:.5f} < 0.99",
                )
          rec = CallRecord(
              session_id=session_id,
              step=call_step,
              slot=slot,
              concurrency=c,
              run_idx=run_idx,
              t_send_rel_s=round(t_send_mono - t_window_start_mono["t"], 6) if in_timed else -1.0,
              latency_ms=res.latency_ms,
              server_compute_ms=res.server_compute_ms,
              horizon=res.chunk.horizon,
              dim=res.chunk.dim,
              deadline_missed=pacing.deadline_missed(res.latency_ms),
              in_timed_window=in_timed,
              request_bytes=res.request_bytes,
              response_bytes=res.response_bytes,
              shape_ok=shape_ok,
              finite_ok=finite_ok,
              action_jerk=call_jerk,
          )
          self._record(rec)
          if in_timed:
            with timed_lock:
              timed_records.append(rec)
            slot_lats[slot].append(res.latency_ms)
            per_slot_last_recv[slot] = t_recv_mono
          sched.step += 1
          sched.arm(pacing, t_recv_mono)
          return res.latency_ms

        # Cold-session TTFA (`cold_session_ttfa_*_ms`): reset() + 1st call per slot (n=C samples/run).
        # Timed warm-chunk latency (`obs_to_action_*_ms`) is collected separately after warmup below.
        t0 = time.perf_counter()
        reset_ms = client.reset(session_id)
        first = one_call(in_timed=False)
        if first is not None:
          with ttfa_lock:
            ttfa_ms.append((time.perf_counter() - t0) * 1000.0)
        _ = reset_ms
        # Per-session warm-up (paced).
        for _ in range(int(cfg.warmup_calls)):
          sched.wait()
          one_call(in_timed=False)
        barrier.wait()
        # Timed window: each slot runs its mandatory per-slot quota `n_slot = ceil(timed_calls / C)`
        # so fast/starved slot asymmetry cannot exclude slow slots from p99, and keeps driving
        # drain calls (`in_timed=False`) until all `C` slots complete so batch occupancy stays at C.
        for local_idx in range(n_slot):
          if errors:
            break
          over = sched.wait()
          if over:
            oversleeps.append(over)
          one_call(in_timed=True, local_idx=local_idx)
        with done_lock:
          slots_done["count"] += 1
          if slots_done["count"] >= c or errors:
            t_steady_end_mono["t"] = max(per_slot_last_recv) if any(per_slot_last_recv) else time.perf_counter()
            all_slots_done.set()
        while not all_slots_done.is_set() and not errors:
          sched.wait()
          one_call(in_timed=False)
        slot_digests[slot] = hasher.hexdigest()
        # Determinism probe (outside the timed window, informational): same observation/context twice.
        if slot == 0:
          try:
            if is_wm:
              ctx_p, act_p = source.make_rollout_inputs(session_id, 10_000_000, num_frames=expected_h, action_dim=expected_d or 7)
              f1 = client.rollout(ctx_p, act_p).frames
              f2 = client.rollout(ctx_p, act_p).frames
              repeat_checks.append(
                  len(f1) == len(f2) and all(np.array_equal(x.image, y.image) for x, y in zip(f1, f2))
              )
            else:
              probe = source.make(session_id, 10_000_000)
              a1 = client.predict(probe).chunk.actions
              a2 = client.predict(probe).chunk.actions
              repeat_checks.append(bool(np.array_equal(a1, a2)))
          except Exception:  # noqa: BLE001 - probe failures are not measurement errors
            pass
      except threading.BrokenBarrierError:
        errors.append(f"{session_id}: barrier broken (another worker failed before the timed window)")
        all_slots_done.set()
      except Exception as exc:  # noqa: BLE001
        errors.append(f"{session_id}: worker crashed: {type(exc).__name__}: {str(exc)[:200]}")
        all_slots_done.set()
        try:
          barrier.abort()
        except Exception:
          pass
      finally:
        with done_lock:
          if errors and not all_slots_done.is_set():
            all_slots_done.set()
        if client is not None:
          client.close()

    threads = [threading.Thread(target=worker, args=(slot,), name=f"eeb-slot-{slot}", daemon=True) for slot in range(c)]
    if self.on_run_start:
      self.on_run_start(c, run_idx)
    for t in threads:
      t.start()
    try:
      barrier.wait()  # _stamp_window_start runs atomically before any worker wakes
      window_start_wall = t_window_start_wall["t"]
    except threading.BrokenBarrierError:
      window_start_wall = time.time()
      t_window_start_mono["t"] = time.perf_counter()
      all_slots_done.set()
    for t in threads:
      t.join()
    window_end_wall = time.time()
    t_last = (
        t_steady_end_mono["t"]
        if t_steady_end_mono["t"] > 0.0
        else (max(per_slot_last_recv) if any(per_slot_last_recv) else time.perf_counter())
    )
    window_s = max(t_last - t_window_start_mono["t"], 1e-9)

    sticky_ok, sticky_viol = verify_sticky_trace(router.trace)
    sticky_viol = sticky_errors + sticky_viol
    combined = hashlib.sha256("|".join(slot_digests).encode("ascii")).hexdigest()
    lat = [r.latency_ms for r in timed_records if r.error is None]
    srv = [r.server_compute_ms for r in timed_records if r.server_compute_ms is not None]
    ttff = [r.ttff_ms for r in timed_records if r.ttff_ms is not None and r.error is None]
    jerks = [r.action_jerk for r in timed_records if r.action_jerk is not None and r.error is None]
    n_timed = len(lat)
    horizon = timed_records[0].horizon if timed_records else int(self.caps.horizon)
    dim = timed_records[0].dim if timed_records else int(self.caps.action_dim)
    wall_chunks_ps = n_timed / window_s
    lat_stats = _percentiles(lat)
    if lat and any(slot_lats):
      # Scored p99(C) is the 99th percentile over pooled steady-state completed requests at C;
      # worst-slot p50/p99 and slot_count_spread are recorded alongside as fairness diagnostics.
      lat_stats["pooled_p99"] = lat_stats["p99"]
      lat_stats["worst_slot_p50"] = round(max(float(np.percentile(sl, 50)) for sl in slot_lats if sl), 4)
      lat_stats["worst_slot_p99"] = round(max(float(np.percentile(sl, 99)) for sl in slot_lats if sl), 4)
    slot_counts = [len(sl) for sl in slot_lats]
    slot_count_spread = (
        round((max(slot_counts) - min(slot_counts)) / max(float(statistics.mean(slot_counts)), 1e-9), 6)
        if len(slot_counts) > 1 and sum(slot_counts) > 0
        else 0.0
    )
    c1_ratio = (
        round(float(lat_stats["p99"]) / max(float(lat_stats["p50"]), 1e-9), 4)
        if (c == 1 and lat)
        else None
    )
    mean_lat_ms = float(lat_stats.get("mean", 0.0) or 0.0)
    chunks_ps = wall_chunks_ps
    if pacing.enabled:
      paced_wall_chunks_ps: Optional[float] = round(wall_chunks_ps, 4)
      paced_wall_exec_aps: Optional[float] = round(
          wall_chunks_ps * pacing.effective_execute_horizon, 4
      )
      active_service_chunks_ps: Optional[float] = (
          round((1000.0 * float(c)) / mean_lat_ms, 4) if mean_lat_ms > 0.0 else None
      )
    else:
      paced_wall_chunks_ps = None
      paced_wall_exec_aps = None
      active_service_chunks_ps = None
    srv_stats = _percentiles(srv) if srv else {}
    ttff_stats = _percentiles(ttff) if ttff else {}
    mean_jerk = round(float(np.mean(jerks)), 6) if jerks else (0.0 if not is_wm else None)
    transport_p50 = round(lat_stats["p50"] - srv_stats["p50"], 4) if srv_stats and lat else None
    result = RunResult(
        concurrency=c,
        run_idx=run_idx,
        window_start_wall=window_start_wall,
        window_end_wall=window_end_wall,
        window_s=round(window_s, 6),
        num_timed_calls=n_timed,
        num_warmup_calls=c * int(cfg.warmup_calls),
        horizon=horizon,
        dim=dim,
        execute_horizon=pacing.effective_execute_horizon,
        latency_ms=lat_stats,
        server_compute_ms=srv_stats,
        ttfa_ms=_percentiles(ttfa_ms),
        chunks_per_sec=round(chunks_ps, 4),
        actions_per_sec=round(chunks_ps * horizon, 4),
        executed_actions_per_sec=round(chunks_ps * pacing.effective_execute_horizon, 4),
        deadline_miss_rate=round(sum(1 for r in timed_records if r.deadline_missed) / max(n_timed, 1), 4),
        transport_overhead_p50_ms=transport_p50,
        pacing_oversleep_p99_ms=round(float(np.percentile(oversleeps, 99)), 4) if oversleeps else 0.0,
        sticky_routing_ok=sticky_ok and not sticky_errors,
        sticky_violations=sticky_viol,
        num_errors=len(errors),
        errors=errors[:20],
        actions_sha256=combined,
        num_shape_mismatch=sum(1 for r in timed_records if not r.shape_ok),
        num_nonfinite=sum(1 for r in timed_records if not r.finite_ok),
        deterministic_repeat=(repeat_checks[0] if repeat_checks else None),
        samples_latency_ms=[round(v, 4) for v in lat] if cfg.keep_samples else [],
        ttff_ms=ttff_stats,
        normalized_action_jerk=mean_jerk,
        paced_wall_chunks_per_sec=paced_wall_chunks_ps,
        paced_wall_executed_actions_per_sec=paced_wall_exec_aps,
        active_service_chunks_per_sec=active_service_chunks_ps,
        num_inline_canaries=int(canary_stats["checked"]),
        num_inline_canary_failures=int(canary_stats["failed"]),
        inline_canary_reasons=list(canary_stats["reasons"]),
        predictions_per_sec=round(chunks_ps, 4) if is_wm else None,
        slot_counts=slot_counts,
        slot_count_spread=slot_count_spread,
        c1_p99_p50_ratio=c1_ratio,
    )
    if self.on_run_end:
      self.on_run_end(c, run_idx, result)
    return result

  def _aggregate(self, c: int, runs: List[RunResult]) -> ConcurrencyResult:
    cfg = self.config

    def med(key: Callable[[RunResult], float]) -> float:
      vals = [key(r) for r in runs if key(r) == key(r)]
      return round(float(statistics.median(vals)), 4) if vals else float("nan")

    med_lat = {k: med(lambda r, k=k: r.latency_ms.get(k, float("nan"))) for k in ("p50", "p95", "p99", "mean", "max")}
    med_ttfa = {k: med(lambda r, k=k: r.ttfa_ms.get(k, float("nan"))) for k in ("p50", "p99", "max")}
    med_ttff = (
        {k: med(lambda r, k=k: r.ttff_ms.get(k, float("nan"))) for k in ("p50", "p95", "p99", "mean", "max")}
        if any(r.ttff_ms for r in runs)
        else {}
    )
    jerk_runs = [float(r.normalized_action_jerk) for r in runs if r.normalized_action_jerk is not None]
    med_jerk = round(float(statistics.median(jerk_runs)), 6) if jerk_runs else None
    spread = {
        "latency_p50_ms": cross_run_spread_pct([r.latency_ms.get("p50", float("nan")) for r in runs]),
        "latency_p99_ms": cross_run_spread_pct([r.latency_ms.get("p99", float("nan")) for r in runs]),
        "chunks_per_sec": cross_run_spread_pct([r.chunks_per_sec for r in runs]),
    }
    p99_cap = cfg.effective_max_p99_variance_pct
    gated = {
        "latency_p50_ms": (spread["latency_p50_ms"], cfg.max_variance_pct),
        "chunks_per_sec": (spread["chunks_per_sec"], cfg.max_variance_pct),
        "latency_p99_ms": (spread["latency_p99_ms"], p99_cap),
    }
    failing = {k: (v, lim) for k, (v, lim) in gated.items() if v > lim}
    if len(runs) < 2:
      variance_passed, reason = False, f"cross-run variance needs >= 2 runs (got {len(runs)}); protocol requires {PROTOCOL_MIN_RUNS}."
    elif failing:
      variance_passed = False
      reason = (
          "cross-run spread "
          + ", ".join(f"{k}={v:.2f}%>{lim:.1f}%" for k, (v, lim) in failing.items())
          + f" (runs p50={[round(r.latency_ms.get('p50', float('nan')), 3) for r in runs]} ms, "
          + f"p99={[round(r.latency_ms.get('p99', float('nan')), 3) for r in runs]} ms). "
          "Lock clocks (`nvidia-smi -lgc` / `jetson_clocks`), stop background load, and re-run."
      )
    else:
      variance_passed = True
      reason = (
          f"cross-run spread p50/throughput <= {cfg.max_variance_pct:.1f}% and "
          f"p99 <= {p99_cap:.1f}% ({spread}) over {len(runs)} runs."
      )
    max_slot_spread = round(max((float(r.slot_count_spread) for r in runs), default=0.0), 6)
    slot_starvation_passed = max_slot_spread <= PROTOCOL_MAX_SLOT_COUNT_SPREAD
    if not slot_starvation_passed:
      reason = (
          f"{reason} Slot sample imbalance spread={max_slot_spread:.4f} > "
          f"{PROTOCOL_MAX_SLOT_COUNT_SPREAD:.2f} (RT-03 anti-starvation floor)."
      )
    pred_runs = [float(r.predictions_per_sec) for r in runs if r.predictions_per_sec is not None]
    med_preds = round(float(statistics.median(pred_runs)), 4) if pred_runs else None
    c1_tail_ratio: Optional[float] = None
    c1_tail_ok = True
    if c == 1 and med_lat.get("p50", 0.0) > 0.0 and med_lat.get("p99") == med_lat.get("p99"):
      c1_tail_ratio = round(float(med_lat["p99"]) / max(float(med_lat["p50"]), 1e-9), 4)
      c1_tail_ok = bool(c1_tail_ratio <= PROTOCOL_MAX_C1_P99_P50_RATIO)
    return ConcurrencyResult(
        concurrency=c,
        runs=runs,
        median_latency_ms=med_lat,
        median_ttfa_ms=med_ttfa,
        median_chunks_per_sec=med(lambda r: r.chunks_per_sec),
        median_actions_per_sec=med(lambda r: r.actions_per_sec),
        median_executed_actions_per_sec=med(lambda r: r.executed_actions_per_sec),
        median_deadline_miss_rate=med(lambda r: r.deadline_miss_rate),
        cross_run_spread_pct=spread,
        max_variance_pct=cfg.max_variance_pct,
        variance_passed=variance_passed,
        variance_reason=reason,
        all_runs_passed=all(r.passed for r in runs),
        median_ttff_ms=med_ttff,
        median_normalized_action_jerk=med_jerk,
        median_predictions_per_sec=med_preds,
        max_slot_count_spread=max_slot_spread,
        slot_starvation_passed=slot_starvation_passed,
        c1_tail_ratio_p99_p50=c1_tail_ratio,
        c1_tail_ratio_passed=c1_tail_ok,
    )

  def run(self) -> SweepResult:
    started = time.time()
    self.caps, self.pacing, source = self._discover()
    self.log(
        f"endpoint={self.endpoint} transport={self.transport} server_impl={self.caps.server_impl} "
        f"horizon={self.caps.horizon} dt_ms={self.pacing.dt_ms} think_time_ms={self.pacing.think_time_ms:.1f}"
    )
    ttr = self._server_warmup(source)
    points: List[ConcurrencyResult] = []
    for c in self.config.concurrency:
      runs: List[RunResult] = []
      for run_idx in range(self.config.runs):
        res = self._run_once(c, run_idx, source)
        if self.caps.track == "WM_TRACK":
          tput_desc = f"pred/s={res.chunks_per_sec:.2f} gen_fps={res.actions_per_sec:.1f}"
        else:
          tput_desc = (
              f"chunks/s={res.chunks_per_sec:.2f} exec_actions/s={res.executed_actions_per_sec:.1f} "
              f"raw_actions/s={res.actions_per_sec:.1f}"
          )
        self.log(
            f"C={c} run={run_idx + 1}/{self.config.runs}: n={res.num_timed_calls} p50={res.latency_ms.get('p50', float('nan')):.2f}ms "
            f"p99={res.latency_ms.get('p99', float('nan')):.2f}ms {tput_desc} "
            f"miss={res.deadline_miss_rate:.2%} errors={res.num_errors}"
        )
        runs.append(res)
      agg = self._aggregate(c, runs)
      self.log(f"C={c}: median p50={agg.median_latency_ms['p50']:.2f}ms p99={agg.median_latency_ms['p99']:.2f}ms spread={agg.cross_run_spread_pct} variance_passed={agg.variance_passed}")
      points.append(agg)
    return SweepResult(
        endpoint=self.endpoint,
        transport=self.transport,
        caps=self.caps,
        config=self.config,
        pacing=self.pacing,
        observation_source=source.describe(),
        started_at_wall=started,
        finished_at_wall=time.time(),
        time_to_ready_s=ttr,
        points=points,
        host=host_info(),
    )


def run_sweep(endpoint: str, config: SweepConfig, **kwargs: Any) -> SweepResult:
  """Convenience wrapper around `ConcurrencySweep(...).run()`."""
  return ConcurrencySweep(endpoint, config, **kwargs).run()


SyntheticObservationSource = ObservationSource


__all__ = [
    "CallRecord",
    "ConcurrencyResult",
    "ConcurrencySweep",
    "DEFAULT_MAX_P99_VARIANCE_PCT",
    "DEFAULT_MAX_VARIANCE_PCT",
    "ObservationSource",
    "ObservationSpec",
    "PROTOCOL_MIN_RUNS",
    "PROTOCOL_MIN_TIMED_CALLS",
    "PROTOCOL_MIN_TIMED_CALLS_WM",
    "PROTOCOL_MIN_WARMUP_CALLS",
    "RunResult",
    "SweepConfig",
    "SweepResult",
    "SyntheticObservationSource",
    "TraceBundleObservationSource",
    "cross_run_spread_pct",
    "host_info",
    "run_sweep",
]
