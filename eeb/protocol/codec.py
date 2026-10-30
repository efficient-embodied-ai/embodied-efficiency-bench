"""Conversions between `eeb.protocol.policy` dataclasses and generated `eeb_pb2` messages.

Imports the generated protobuf stubs lazily so that `eeb.protocol.policy` stays usable
without grpc/protobuf installed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from eeb.protocol.policy import ACTION_DTYPE, ActionChunk, Caps, Context, Frame, Observation


def _pb2() -> Any:
  try:
    from eeb.protocol import eeb_pb2  # noqa: WPS433 - lazy import by design
  except ImportError as exc:  # pragma: no cover - exercised only without protobuf
    raise ImportError(
        "eeb.protocol.codec requires the protobuf runtime (and grpcio for transport). "
        "Install with `pip install -e '.[grpc]'` (grpcio>=1.66.2, protobuf>=5.27.2)."
    ) from exc
  return eeb_pb2


# ----------------------------------------------------------------------------- images


def image_to_proto(camera: str, array: np.ndarray) -> Any:
  pb2 = _pb2()
  arr = np.asarray(array)
  if arr.ndim == 2:
    arr = arr[:, :, None]
  if arr.ndim != 3:
    raise ValueError(f"Image {camera!r} must be (H, W, C); got shape {arr.shape}.")
  arr = np.ascontiguousarray(arr)
  return pb2.Image(
      camera=camera,
      height=int(arr.shape[0]),
      width=int(arr.shape[1]),
      channels=int(arr.shape[2]),
      dtype=str(arr.dtype.name),
      data=arr.tobytes(order="C"),
  )


def proto_to_image(msg: Any) -> np.ndarray:
  try:
    dtype = np.dtype(msg.dtype or "uint8")
  except TypeError as exc:
    raise ValueError(f"Image {msg.camera!r} carries unknown dtype {msg.dtype!r}.") from exc
  expected = int(msg.height) * int(msg.width) * int(msg.channels) * dtype.itemsize
  if len(msg.data) != expected:
    raise ValueError(
        f"Image {msg.camera!r} payload has {len(msg.data)} bytes but shape "
        f"({msg.height}, {msg.width}, {msg.channels}) x {dtype.name} needs {expected} bytes."
    )
  return np.frombuffer(msg.data, dtype=dtype).reshape(int(msg.height), int(msg.width), int(msg.channels)).copy()


# ----------------------------------------------------------------------------- caps


def caps_to_proto(caps: Caps) -> Any:
  pb2 = _pb2()
  return pb2.Caps(
      track=caps.track,
      horizon=int(caps.horizon),
      dt_ms=float(caps.dt_ms),
      modalities=list(caps.modalities),
      precision=caps.precision,
      model_id=caps.model_id,
      server_impl=caps.server_impl,
      action_dim=int(caps.action_dim),
      proprio_dim=int(caps.proprio_dim),
      cameras=list(caps.cameras),
      image_height=int(caps.image_height),
      image_width=int(caps.image_width),
      execute_horizon=int(caps.execute_horizon) if caps.execute_horizon is not None else 0,
  )


def proto_to_caps(msg: Any) -> Caps:
  exec_h = int(getattr(msg, "execute_horizon", 0) or 0)
  return Caps(
      track=msg.track,
      horizon=int(msg.horizon),
      dt_ms=float(msg.dt_ms),
      modalities=tuple(msg.modalities),
      precision=msg.precision,
      model_id=msg.model_id,
      server_impl=msg.server_impl,
      action_dim=int(msg.action_dim),
      proprio_dim=int(msg.proprio_dim),
      cameras=tuple(msg.cameras),
      image_height=int(msg.image_height),
      image_width=int(msg.image_width),
      execute_horizon=exec_h if exec_h > 0 else None,
  )


# ----------------------------------------------------------------------------- T1


def observation_to_proto(obs: Observation) -> Any:
  pb2 = _pb2()
  return pb2.PredictRequest(
      session_id=obs.session_id,
      step=int(obs.step),
      images=[image_to_proto(name, arr) for name, arr in obs.images.items()],
      proprio_f32=np.ascontiguousarray(obs.proprio, dtype=ACTION_DTYPE).tobytes(),
      instruction=obs.instruction,
      t_send=float(obs.t_send),
  )


def proto_to_observation(msg: Any) -> Observation:
  if len(msg.proprio_f32) % 4 != 0:
    raise ValueError(
        f"PredictRequest.proprio_f32 has {len(msg.proprio_f32)} bytes; must be a multiple of 4 (float32)."
    )
  return Observation(
      session_id=msg.session_id,
      step=int(msg.step),
      images={img.camera: proto_to_image(img) for img in msg.images},
      proprio=np.frombuffer(msg.proprio_f32, dtype=ACTION_DTYPE).copy(),
      instruction=msg.instruction,
      t_send=float(msg.t_send),
  )


def action_chunk_to_proto(chunk: ActionChunk) -> Any:
  pb2 = _pb2()
  arr = np.ascontiguousarray(chunk.actions, dtype=ACTION_DTYPE)
  return pb2.PredictResponse(
      actions_f32=arr.tobytes(order="C"),
      horizon=int(arr.shape[0]),
      dim=int(arr.shape[1]),
      t_recv=float(chunk.t_recv),
      t_done=float(chunk.t_done),
  )


def proto_to_action_chunk(msg: Any) -> ActionChunk:
  horizon, dim = int(msg.horizon), int(msg.dim)
  expected = horizon * dim * ACTION_DTYPE.itemsize
  if horizon <= 0 or dim <= 0 or len(msg.actions_f32) != expected:
    raise ValueError(
        f"PredictResponse.actions_f32 has {len(msg.actions_f32)} bytes but horizon={horizon} x dim={dim} "
        f"float32 needs {expected} bytes. The server must set horizon/dim consistently with the payload."
    )
  actions = np.frombuffer(msg.actions_f32, dtype=ACTION_DTYPE).reshape(horizon, dim).copy()
  return ActionChunk(actions=actions, t_recv=float(msg.t_recv), t_done=float(msg.t_done))


# ----------------------------------------------------------------------------- T2


def context_to_proto(ctx: Context, actions: np.ndarray) -> Any:
  pb2 = _pb2()
  arr = np.asarray(actions, dtype=ACTION_DTYPE)
  if arr.ndim == 1:
    arr = arr[None, :]
  if arr.ndim != 2:
    raise ValueError(f"Rollout actions must be (horizon, dim); got shape {arr.shape}.")
  arr = np.ascontiguousarray(arr)
  return pb2.RolloutRequest(
      session_id=ctx.session_id,
      context_frames=[image_to_proto(f"ctx{i}", f) for i, f in enumerate(ctx.frames)],
      proprio_f32=np.ascontiguousarray(ctx.proprio, dtype=ACTION_DTYPE).tobytes(),
      instruction=ctx.instruction,
      actions_f32=arr.tobytes(order="C"),
      horizon=int(arr.shape[0]),
      dim=int(arr.shape[1]),
      num_frames=int(ctx.num_frames),
      t_send=float(ctx.t_send),
  )


def proto_to_context(msg: Any) -> tuple[Context, np.ndarray]:
  horizon, dim = int(msg.horizon), int(msg.dim)
  expected = horizon * dim * ACTION_DTYPE.itemsize
  if len(msg.actions_f32) != expected:
    raise ValueError(
        f"RolloutRequest.actions_f32 has {len(msg.actions_f32)} bytes but horizon={horizon} x dim={dim} "
        f"float32 needs {expected} bytes."
    )
  actions = (
      np.frombuffer(msg.actions_f32, dtype=ACTION_DTYPE).reshape(horizon, dim).copy()
      if expected > 0
      else np.zeros((0, 0), dtype=np.float32)
  )
  ctx = Context(
      session_id=msg.session_id,
      frames=[proto_to_image(img) for img in msg.context_frames],
      proprio=np.frombuffer(msg.proprio_f32, dtype=ACTION_DTYPE).copy(),
      instruction=msg.instruction,
      num_frames=int(msg.num_frames),
      t_send=float(msg.t_send),
  )
  return ctx, actions


def frame_to_proto(frame: Frame) -> Any:
  pb2 = _pb2()
  return pb2.Frame(index=int(frame.index), image=image_to_proto(frame.camera, frame.image), t_done=float(frame.t_done))


def proto_to_frame(msg: Any) -> Frame:
  return Frame(index=int(msg.index), image=proto_to_image(msg.image), t_done=float(msg.t_done), camera=msg.image.camera)


def frames_to_protos(frames: Iterable[Frame]) -> List[Any]:
  return [frame_to_proto(f) for f in frames]


__all__ = [
    "action_chunk_to_proto",
    "caps_to_proto",
    "context_to_proto",
    "frame_to_proto",
    "frames_to_protos",
    "image_to_proto",
    "observation_to_proto",
    "proto_to_action_chunk",
    "proto_to_caps",
    "proto_to_context",
    "proto_to_frame",
    "proto_to_image",
    "proto_to_observation",
]
