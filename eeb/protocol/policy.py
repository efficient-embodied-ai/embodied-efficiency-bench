"""In-process `Policy` interface and plain-Python data types for the EEB participant protocol.

This module is deliberately dependency-light (stdlib + NumPy only) so that a participant
can implement a policy without importing gRPC. The wire representation lives in
`eeb/protocol/eeb.proto`; conversions between these dataclasses and the generated
protobuf messages live in `eeb.protocol.codec` (which imports grpc/protobuf lazily).

Engineering plan §5::

    class Policy(Protocol):                      # in-process adapter (dev mode)
        def capabilities(self) -> Caps: ...
        def reset(self, session_id: str) -> None: ...
        def predict(self, obs: Observation) -> ActionChunk: ...                       # T1
        def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]: ...  # T2
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import importlib
import time
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

# Server implementation tag that every synthetic / dummy policy MUST report in
# `Caps.server_impl`. Reports whose `execution_provenance.policy` equals this tag are
# never admissible for a leaderboard (see eeb.loadgen.report).
SYNTHETIC_DUMMY_SERVER_IMPL = "SYNTHETIC_DUMMY"

ACTION_DTYPE = np.dtype("<f4")  # little-endian float32 on the wire


@dataclass(frozen=True)
class Caps:
  """Static capabilities of a served policy (mirrors `eeb.v1.Caps`).

  Example (reference dummy)::

      Caps(track="VLA_TRACK", horizon=8, dt_ms=50.0, modalities=("rgb", "proprio", "language"),
           precision="FP32", model_id="dummy-vla", server_impl="SYNTHETIC_DUMMY", action_dim=7)
  """

  track: str
  horizon: int
  dt_ms: float
  modalities: Tuple[str, ...] = ("rgb", "proprio", "language")
  precision: str = "BF16"
  model_id: str = ""
  server_impl: str = ""
  action_dim: int = 7
  proprio_dim: int = 0
  cameras: Tuple[str, ...] = ()
  image_height: int = 0
  image_width: int = 0
  execute_horizon: Optional[int] = None

  def __post_init__(self) -> None:
    if int(self.horizon) <= 0:
      raise ValueError(f"Caps.horizon must be > 0, got {self.horizon}.")
    if self.execute_horizon is not None and int(self.execute_horizon) <= 0:
      raise ValueError(f"Caps.execute_horizon must be > 0 when set, got {self.execute_horizon}.")
    if float(self.dt_ms) <= 0.0:
      raise ValueError(f"Caps.dt_ms must be > 0 (milliseconds), got {self.dt_ms}.")
    if not self.server_impl:
      raise ValueError(
          "Caps.server_impl must be a non-empty implementation tag (e.g. 'openpi-pytorch-0.1.0'). "
          f"Synthetic policies must report {SYNTHETIC_DUMMY_SERVER_IMPL!r}."
      )

  @property
  def chunk_execution_s(self) -> float:
    """Wall-clock time the robot needs to execute one full chunk at the control period."""
    return float(self.horizon) * float(self.dt_ms) / 1000.0

  @property
  def is_synthetic(self) -> bool:
    return self.server_impl.upper().startswith(SYNTHETIC_DUMMY_SERVER_IMPL)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "track": self.track,
        "horizon": int(self.horizon),
        "dt_ms": float(self.dt_ms),
        "modalities": list(self.modalities),
        "precision": self.precision,
        "model_id": self.model_id,
        "server_impl": self.server_impl,
        "action_dim": int(self.action_dim),
        "proprio_dim": int(self.proprio_dim),
        "cameras": list(self.cameras),
        "image_height": int(self.image_height),
        "image_width": int(self.image_width),
        "execute_horizon": int(self.execute_horizon) if self.execute_horizon is not None else None,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Caps":
    exec_h = data.get("execute_horizon")
    return cls(
        track=str(data["track"]),
        horizon=int(data["horizon"]),
        dt_ms=float(data["dt_ms"]),
        modalities=tuple(data.get("modalities", ())),
        precision=str(data.get("precision", "")),
        model_id=str(data.get("model_id", "")),
        server_impl=str(data.get("server_impl", "")),
        action_dim=int(data.get("action_dim", 0)),
        proprio_dim=int(data.get("proprio_dim", 0)),
        cameras=tuple(data.get("cameras", ())),
        image_height=int(data.get("image_height", 0)),
        image_width=int(data.get("image_width", 0)),
        execute_horizon=int(exec_h) if exec_h is not None else None,
    )


@dataclass
class Observation:
  """One robot observation: named camera images, proprioception, and a language instruction.

  `images` maps camera name -> array of shape (H, W, C). `proprio` is a float32 vector.
  """

  session_id: str
  step: int
  images: Dict[str, np.ndarray] = field(default_factory=dict)
  proprio: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
  instruction: str = ""
  t_send: float = 0.0

  def __post_init__(self) -> None:
    self.proprio = np.ascontiguousarray(np.asarray(self.proprio, dtype=np.float32)).reshape(-1)
    for name, img in list(self.images.items()):
      arr = np.asarray(img)
      if arr.ndim == 2:
        arr = arr[:, :, None]
      if arr.ndim != 3:
        raise ValueError(
            f"Observation image {name!r} must have shape (H, W, C) or (H, W); got {arr.shape}."
        )
      self.images[name] = np.ascontiguousarray(arr)

  def content_digest_u64(self) -> int:
    """Cheap, deterministic 64-bit digest of the observation content (used by seeded dummies)."""
    import zlib

    h = zlib.crc32(self.instruction.encode("utf-8"))
    h = zlib.crc32(self.proprio.tobytes(), h)
    for name in sorted(self.images):
      h = zlib.crc32(name.encode("utf-8"), h)
      h = zlib.crc32(self.images[name].tobytes()[:65536], h)
    return int(h) & 0xFFFFFFFFFFFFFFFF


@dataclass
class ActionChunk:
  """A predicted action chunk of shape (horizon, dim) plus server-side timestamps."""

  actions: np.ndarray
  t_recv: float = 0.0
  t_done: float = 0.0

  def __post_init__(self) -> None:
    arr = np.asarray(self.actions, dtype=np.float32)
    if arr.ndim == 1:
      arr = arr[None, :]
    if arr.ndim != 2:
      raise ValueError(f"ActionChunk.actions must be 2-D (horizon, dim); got shape {arr.shape}.")
    self.actions = np.ascontiguousarray(arr)

  @property
  def horizon(self) -> int:
    return int(self.actions.shape[0])

  @property
  def dim(self) -> int:
    return int(self.actions.shape[1])

  @property
  def server_compute_ms(self) -> Optional[float]:
    if self.t_done and self.t_recv:
      return (self.t_done - self.t_recv) * 1000.0
    return None


@dataclass
class Context:
  """World-model rollout context: conditioning frames, proprio, instruction (Track 2)."""

  session_id: str
  frames: List[np.ndarray] = field(default_factory=list)
  proprio: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
  instruction: str = ""
  num_frames: int = 0
  t_send: float = 0.0

  def __post_init__(self) -> None:
    self.proprio = np.ascontiguousarray(np.asarray(self.proprio, dtype=np.float32)).reshape(-1)
    self.frames = [np.ascontiguousarray(np.asarray(f)) for f in self.frames]


@dataclass
class Frame:
  """One generated frame (Track 2)."""

  index: int
  image: np.ndarray
  t_done: float = 0.0
  camera: str = "wm"

  def __post_init__(self) -> None:
    arr = np.asarray(self.image)
    if arr.ndim == 2:
      arr = arr[:, :, None]
    if arr.ndim != 3:
      raise ValueError(f"Frame.image must have shape (H, W, C); got {arr.shape}.")
    self.image = np.ascontiguousarray(arr)


@runtime_checkable
class Policy(Protocol):
  """Structural interface every in-process policy must satisfy (plan §5)."""

  def capabilities(self) -> Caps: ...

  def reset(self, session_id: str) -> None: ...

  def predict(self, obs: Observation) -> ActionChunk: ...

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]: ...


class BasePolicy(abc.ABC):
  """Convenience ABC: subclasses implement `capabilities` and `predict` and/or `rollout`."""

  @abc.abstractmethod
  def capabilities(self) -> Caps:
    raise NotImplementedError

  def reset(self, session_id: str) -> None:  # noqa: D401 - default no-op
    """Reset per-session state. Default: no state."""
    return None

  def predict(self, obs: Observation) -> ActionChunk:
    raise NotImplementedError(
        f"{type(self).__name__} does not implement Track 1 `predict`; "
        "serve it on the WM track or implement predict()."
    )

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    raise NotImplementedError(
        f"{type(self).__name__} does not implement Track 2 `rollout`; "
        "serve it on the VLA track or implement rollout()."
    )

  def close(self) -> None:
    """Release device resources (optional)."""
    return None


def validate_policy(obj: Any) -> None:
  """Raise TypeError with an actionable message if `obj` does not satisfy `Policy`."""
  missing = [m for m in ("capabilities", "reset", "predict", "rollout") if not callable(getattr(obj, m, None))]
  if missing:
    raise TypeError(
        f"{type(obj).__name__} does not satisfy eeb.protocol.Policy: missing callable(s) {missing}. "
        "Subclass eeb.protocol.policy.BasePolicy or implement capabilities/reset/predict/rollout."
    )


def load_policy_class(spec: str) -> type:
  """Resolve a `module:Class` (or `module.Class`) spec to a class object.

  Example: ``load_policy_class("eeb.servers.dummy:DummyVLAPolicy")``.
  """
  if ":" in spec:
    mod_name, _, cls_name = spec.partition(":")
  elif "." in spec:
    mod_name, _, cls_name = spec.rpartition(".")
  else:
    raise ValueError(
        f"Invalid policy spec {spec!r}. Use '<module>:<Class>', e.g. 'eeb.servers.dummy:DummyVLAPolicy'."
    )
  try:
    module = importlib.import_module(mod_name)
  except ImportError as exc:
    raise ImportError(
        f"Cannot import policy module {mod_name!r} from spec {spec!r}: {exc}. "
        "Check PYTHONPATH / optional dependencies (e.g. `pip install -e '.[pi05]'`)."
    ) from exc
  cls = getattr(module, cls_name, None)
  if cls is None:
    raise AttributeError(f"Module {mod_name!r} has no attribute {cls_name!r} (from policy spec {spec!r}).")
  return cls


def instantiate_policy(spec: str, kwargs: Optional[Dict[str, Any]] = None) -> Any:
  """Import and construct a policy from `module:Class`, validating the Policy interface."""
  import inspect

  cls = load_policy_class(spec)
  call_kwargs = dict(kwargs or {})
  try:
    policy = cls(**call_kwargs)
  except TypeError as exc:
    try:
      sig_params = [
          p
          for p in inspect.signature(cls.__init__).parameters
          if p != "self"
      ]
    except (TypeError, ValueError):
      sig_params = []
    raise TypeError(
        f"Invalid --policy-kwargs for {cls.__name__} ({spec!r}): {exc}. "
        f"Accepted constructor parameters: {sig_params}"
    ) from exc
  validate_policy(policy)
  return policy


def now_wall() -> float:
  """Wall-clock seconds (shared between client and server on the same host)."""
  return time.time()


__all__ = [
    "ACTION_DTYPE",
    "ActionChunk",
    "BasePolicy",
    "Caps",
    "Context",
    "Frame",
    "Observation",
    "Policy",
    "SYNTHETIC_DUMMY_SERVER_IMPL",
    "instantiate_policy",
    "load_policy_class",
    "now_wall",
    "validate_policy",
]
