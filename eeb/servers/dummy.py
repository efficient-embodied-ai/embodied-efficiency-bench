"""Reference synthetic policies for harness development and CI.

`DummyVLAPolicy` and `DummyWMPolicy` do **no model inference**. They emulate a single
accelerator with a configurable synthetic compute delay (a lock + `time.sleep`) and return
deterministic, seeded outputs derived from the observation content. Their `Caps.server_impl`
is `SYNTHETIC_DUMMY`, which the report writer propagates to `execution_provenance.policy`
and which makes the resulting report permanently inadmissible for a leaderboard.

They exist to validate the *harness* (wire protocol, pacing, session stickiness, variance
check, telemetry plumbing), never to produce model numbers.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
import time
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from eeb.protocol.policy import (
    SYNTHETIC_DUMMY_SERVER_IMPL,
    ActionChunk,
    BasePolicy,
    Caps,
    Context,
    Frame,
    Observation,
)

MAX_ACTIVE_SESSIONS = 1024


def _touch_bounded_session(sessions: "OrderedDict[str, int]", session_id: str, value: int) -> None:
  if session_id in sessions:
    sessions.move_to_end(session_id)
  sessions[session_id] = value
  while len(sessions) > MAX_ACTIVE_SESSIONS:
    sessions.popitem(last=False)


def _sleep_precise(seconds: float) -> None:
  """Sleep with a short spin at the end so sub-millisecond delays are honoured on Linux."""
  if seconds <= 0.0:
    return
  deadline = time.perf_counter() + seconds
  coarse = seconds - 0.0005
  if coarse > 0.0:
    time.sleep(coarse)
  while time.perf_counter() < deadline:
    pass


class DummyVLAPolicy(BasePolicy):
  """Synthetic Track 1 policy: fixed compute delay, deterministic actions from the observation.

  Args:
    compute_delay_ms: synthetic per-call "inference" time (mutable attribute; tests change it).
    jitter_ms: extra uniform [0, jitter_ms] delay per call, drawn from a seeded RNG.
    horizon / action_dim / dt_ms: shape of the action chunk and the control period.
    exclusive_compute: if True, concurrent calls serialize on one lock, emulating a single chip
      (latency grows with concurrency; throughput saturates at 1000 / compute_delay_ms chunks/s).
    seed: base seed for the action generator.
  """

  def __init__(
      self,
      compute_delay_ms: float = 20.0,
      jitter_ms: float = 0.0,
      horizon: int = 8,
      action_dim: int = 7,
      dt_ms: float = 50.0,
      exclusive_compute: bool = True,
      seed: int = 0,
      proprio_dim: int = 8,
      cameras: Sequence[str] = ("agentview", "wrist"),
      image_hw: Tuple[int, int] = (224, 224),
      model_id: str = "dummy-vla",
      execute_horizon: Optional[int] = None,
  ) -> None:
    if compute_delay_ms < 0 or jitter_ms < 0:
      raise ValueError("compute_delay_ms and jitter_ms must be >= 0.")
    self.compute_delay_ms = float(compute_delay_ms)
    self.jitter_ms = float(jitter_ms)
    self.horizon = int(horizon)
    self.action_dim = int(action_dim)
    self.dt_ms = float(dt_ms)
    self.exclusive_compute = bool(exclusive_compute)
    self.seed = int(seed)
    self.proprio_dim = int(proprio_dim)
    self.cameras = tuple(cameras)
    self.image_hw = (int(image_hw[0]), int(image_hw[1]))
    self.model_id = model_id
    self.execute_horizon = int(execute_horizon) if execute_horizon is not None else None
    self._lock = threading.Lock()
    self._jitter_rng = np.random.default_rng(self.seed + 7919)
    self._jitter_lock = threading.Lock()
    self.num_calls = 0
    self.num_resets = 0
    self.sessions: "OrderedDict[str, int]" = OrderedDict()
    self._sessions = self.sessions

  def capabilities(self) -> Caps:
    return Caps(
        track="VLA_TRACK",
        horizon=self.horizon,
        dt_ms=self.dt_ms,
        modalities=("rgb", "proprio", "language"),
        precision="FP32",
        model_id=self.model_id,
        server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
        action_dim=self.action_dim,
        proprio_dim=self.proprio_dim,
        cameras=self.cameras,
        image_height=self.image_hw[0],
        image_width=self.image_hw[1],
        execute_horizon=self.execute_horizon,
    )

  def reset(self, session_id: str) -> None:
    with self._jitter_lock:
      self.num_resets += 1
      _touch_bounded_session(self.sessions, session_id, 0)

  def _delay_s(self) -> float:
    delay = self.compute_delay_ms
    if self.jitter_ms > 0.0:
      with self._jitter_lock:
        delay += float(self._jitter_rng.uniform(0.0, self.jitter_ms))
    return delay / 1000.0

  def actions_for(self, obs: Observation) -> np.ndarray:
    """Deterministic actions in [-1, 1] seeded by (seed, step, observation content)."""
    digest = obs.content_digest_u64()
    seed_seq = np.random.SeedSequence([self.seed & 0xFFFFFFFF, int(obs.step) & 0xFFFFFFFF, digest & 0xFFFFFFFF, (digest >> 32) & 0xFFFFFFFF])
    rng = np.random.default_rng(seed_seq)
    return rng.uniform(-1.0, 1.0, size=(self.horizon, self.action_dim)).astype(np.float32)

  def predict(self, obs: Observation) -> ActionChunk:
    delay_s = self._delay_s()
    if self.exclusive_compute:
      with self._lock:
        _sleep_precise(delay_s)
        actions = self.actions_for(obs)
    else:
      _sleep_precise(delay_s)
      actions = self.actions_for(obs)
    with self._jitter_lock:
      self.num_calls += 1
      _touch_bounded_session(self.sessions, obs.session_id, int(obs.step))
    return ActionChunk(actions=actions)


class DummyWMPolicy(BasePolicy):
  """Synthetic Track 2 world model: streams `num_frames` deterministic frames with a per-frame delay."""

  def __init__(
      self,
      frame_delay_ms: float = 10.0,
      num_frames: int = 16,
      frame_hw: Tuple[int, int] = (64, 64),
      dt_ms: float = 1000.0 / 24.0,
      action_dim: int = 7,
      exclusive_compute: bool = True,
      seed: int = 0,
      model_id: str = "dummy-wm",
      horizon: Optional[int] = None,
  ) -> None:
    if frame_delay_ms < 0:
      raise ValueError("frame_delay_ms must be >= 0.")
    resolved_frames = int(horizon) if horizon is not None else int(num_frames)
    if resolved_frames < 1:
      raise ValueError(
          f"DummyWMPolicy requires num_frames/horizon >= 1, got {resolved_frames}."
      )
    self.frame_delay_ms = float(frame_delay_ms)
    self.num_frames = resolved_frames
    self.frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
    self.dt_ms = float(dt_ms)
    self.action_dim = int(action_dim)
    self.exclusive_compute = bool(exclusive_compute)
    self.seed = int(seed)
    self.model_id = model_id
    self._lock = threading.Lock()
    self._session_lock = threading.Lock()
    self.num_rollouts = 0
    self.num_resets = 0
    self.sessions: "OrderedDict[str, int]" = OrderedDict()
    self._sessions = self.sessions

  def capabilities(self) -> Caps:
    return Caps(
        track="WM_TRACK",
        horizon=self.num_frames,
        dt_ms=self.dt_ms,
        modalities=("rgb", "proprio", "language", "actions"),
        precision="FP32",
        model_id=self.model_id,
        server_impl=SYNTHETIC_DUMMY_SERVER_IMPL,
        action_dim=self.action_dim,
        image_height=self.frame_hw[0],
        image_width=self.frame_hw[1],
    )

  def reset(self, session_id: str) -> None:
    with self._session_lock:
      self.num_resets += 1
      _touch_bounded_session(self.sessions, session_id, 0)

  def _frame(self, ctx: Context, actions: np.ndarray, index: int) -> np.ndarray:
    import zlib

    h = zlib.crc32(ctx.instruction.encode("utf-8"))
    h = zlib.crc32(np.ascontiguousarray(actions, dtype=np.float32).tobytes(), h)
    for f in ctx.frames:
      h = zlib.crc32(np.ascontiguousarray(f).tobytes()[:65536], h)
    rng = np.random.default_rng(np.random.SeedSequence([self.seed & 0xFFFFFFFF, index, h & 0xFFFFFFFF]))
    hgt, wid = self.frame_hw
    ramp = np.linspace(0, 255, wid, dtype=np.float32)[None, :, None]
    noise = rng.uniform(-16.0, 16.0, size=(hgt, wid, 3)).astype(np.float32)
    img = np.clip(ramp + noise + float(index) * 4.0, 0.0, 255.0).astype(np.uint8)
    return img

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    n = int(ctx.num_frames) if int(ctx.num_frames) > 0 else self.num_frames
    delay_s = self.frame_delay_ms / 1000.0

    def _gen() -> Iterator[Frame]:
      for i in range(n):
        _sleep_precise(delay_s)
        yield Frame(index=i, image=self._frame(ctx, actions, i), t_done=time.perf_counter())
      self.num_rollouts += 1

    if self.exclusive_compute:
      # Hold the compute lock for the whole stream (a single chip renders one rollout at a time).
      def _locked() -> Iterator[Frame]:
        with self._lock:
          yield from _gen()

      return _locked()
    return _gen()


__all__ = ["DummyVLAPolicy", "DummyWMPolicy"]
