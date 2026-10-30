"""Reference server: Physical Intelligence pi0.5 via native JAX/Flax (`openpi`) behind the EEB `Policy` ABC.

Supports Cloud TPU (`v5e-1`, `v6e-1`, `v7`) and GPU via `jax.jit` + `flax.nnx`.
Implements static power-of-2 dynamic micro-batching (`B in {1, 2, 4, 8, 16}`), parallel per-thread CPU
input/output transforms, deterministic per-step initial flow noise, Orbax >=0.12 StepMetadata compatibility,
and explicit `jax.block_until_ready()` synchronization before latency timing completes.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from eeb.protocol.policy import ActionChunk, BasePolicy, Caps, Context, Frame, Observation
from eeb.servers.microbatch import DynamicMicroBatcher
from eeb.servers.pi05.policy import DEFAULT_IMAGE_KEY_MAP

OPENPI_JAX_INSTALL_HINT = (
    "pi0.5 JAX reference server requires `openpi` and `jax` (TPU or CUDA build):\n"
    "  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git\n"
    "  cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && uv pip install -e .\n"
    "  # On Cloud TPU VM:\n"
    "  pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html\n"
    "Then: eeb serve --policy eeb.servers.pi05.policy_jax:Pi05JaxPolicy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"checkpoint_dir\": \"gs://openpi-assets/checkpoints/pi05_libero\"}'"
)


def _install_openpi_orbax_compat() -> None:
  """Patch `openpi.models.model.restore_params` for `orbax-checkpoint>=0.12` (`StepMetadata.item_metadata`)."""
  import jax
  import numpy as _np
  import orbax.checkpoint as ocp
  from openpi.models import model as _model_mod

  if getattr(_model_mod, "_eeb_orbax_patched", False):
    return

  def _compat_restore_params(
      params_path,
      *,
      restore_type=jax.Array,
      dtype=None,
      sharding=None,
  ):
    import pathlib

    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path
    if restore_type is jax.Array and sharding is None:
      mesh = jax.sharding.Mesh(jax.devices(), ("x",))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as ckptr:
      raw_meta = ckptr.metadata(params_path)
      item_meta = getattr(raw_meta, "item_metadata", raw_meta)
      params = {"params": item_meta["params"]}
      restored = ckptr.restore(
          params_path,
          ocp.args.PyTreeRestore(
              item=params,
              restore_args=jax.tree.map(
                  lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype),
                  params,
              ),
          ),
      )["params"]
    import flax.traverse_util as traverse_util
    flat_params = traverse_util.flatten_dict(restored)
    if flat_params and all(kp[-1] == "value" for kp in flat_params):
      flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)

  _model_mod.restore_params = _compat_restore_params
  _model_mod._eeb_orbax_patched = True


class _FifoLock:
  """Strict FIFO ticket lock avoiding non-FIFO OS futex tail spikes under C=16 contention."""

  def __init__(self) -> None:
    self._cv = threading.Condition()
    self._next_ticket = 0
    self._now_serving = 0

  def __enter__(self) -> "_FifoLock":
    with self._cv:
      my_ticket = self._next_ticket
      self._next_ticket += 1
      while self._now_serving != my_ticket:
        self._cv.wait()
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    with self._cv:
      self._now_serving += 1
      self._cv.notify_all()


class Pi05JaxPolicy(BasePolicy):
  """EEB `Policy` wrapper around the canonical JAX/Flax `openpi` pi0.5 checkpoint."""

  def __init__(
      self,
      checkpoint_dir: Optional[str] = None,
      config_name: str = "pi05_libero",
      image_key_map: Optional[Dict[str, str]] = None,
      precision_tag: str = "BF16",
      dt_ms: float = 50.0,
      execute_horizon: int = 5,
      warm_start: bool = True,
      exclusive_compute: bool = True,
      enable_batching: bool = True,
      batch_buckets: Sequence[int] = (1, 2, 4, 8, 16),
      max_batch_delay_ms: float = 15.0,
      seed: int = 20261029,
  ) -> None:
    self.checkpoint_dir = os.path.expanduser(
        checkpoint_dir
        or os.environ.get("OPENPI_PI05_JAX_CHECKPOINT")
        or "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
    )
    self.config_name = config_name
    self.image_key_map = dict(image_key_map or DEFAULT_IMAGE_KEY_MAP)
    self.precision_tag = precision_tag
    self.dt_ms = float(dt_ms)
    self.execute_horizon = int(execute_horizon)
    self.exclusive_compute = bool(exclusive_compute)
    self.enable_batching = bool(enable_batching)
    self.batch_buckets: Tuple[int, ...] = tuple(int(b) for b in batch_buckets)
    self.max_batch_delay_ms = float(max_batch_delay_ms)
    self.seed = int(seed)

    self._lock = _FifoLock()
    self._horizon: Optional[int] = None
    self._action_dim: Optional[int] = None
    self._model_action_horizon: int = 10
    self._model_action_dim: int = 32
    self._openpi_version = "unknown"
    self._jax_version = "unknown"
    self._jax_backend = "unknown"
    self._jax: Any = None
    self._jnp: Any = None
    self._model_mod: Any = None
    self._fixed_rng: Any = None

    self._policy = self._load_policy()
    self._batcher: Optional[DynamicMicroBatcher[Tuple[Dict[str, Any], np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None
    if self.enable_batching and max(self.batch_buckets, default=1) > 1:
      self._batcher = DynamicMicroBatcher(
          self._execute_batch,
          buckets=self.batch_buckets,
          max_batch_delay_ms=self.max_batch_delay_ms,
          name="pi05-jax-batcher",
      )
    if warm_start:
      self._warm_start()
    import gc

    gc.collect()
    if hasattr(gc, "freeze"):
      gc.freeze()
    gc.set_threshold(700, 50000, 50000)

  def _load_policy(self) -> Any:
    try:
      import jax  # noqa: WPS433
      import jax.numpy as jnp  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("jax is not installed.\n" + OPENPI_JAX_INSTALL_HINT) from exc
    try:
      from openpi.models import model as _model_mod  # noqa: WPS433
      from openpi.policies import policy_config as _policy_config  # noqa: WPS433
      from openpi.training import config as _config  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("openpi is not installed or not importable.\n" + OPENPI_JAX_INSTALL_HINT) from exc

    _install_openpi_orbax_compat()
    self._jax = jax
    self._jnp = jnp
    self._model_mod = _model_mod
    self._fixed_rng = jax.random.key(self.seed)
    self._jax_version = str(getattr(jax, "__version__", "unknown"))
    try:
      self._jax_backend = str(jax.default_backend())
    except Exception:  # pragma: no cover
      self._jax_backend = "unknown"
    try:
      import importlib.metadata as _md

      self._openpi_version = _md.version("openpi")
    except Exception:  # pragma: no cover
      self._openpi_version = "source"

    if not self.checkpoint_dir.startswith("gs://") and not os.path.isdir(self.checkpoint_dir):
      raise FileNotFoundError(
          f"pi0.5 JAX checkpoint directory not found: {self.checkpoint_dir}. "
          "Pass --policy-kwargs '{\"checkpoint_dir\": ...}' or set $OPENPI_PI05_JAX_CHECKPOINT."
      )

    config = _config.get_config(self.config_name)
    policy = _policy_config.create_trained_policy(config, self.checkpoint_dir)
    model_obj = getattr(policy, "_model", None)
    self._model_action_horizon = int(getattr(model_obj, "action_horizon", 10))
    self._model_action_dim = int(getattr(model_obj, "action_dim", 32))
    return policy

  def _noise_for_step(self, step: int) -> np.ndarray:
    digest = hashlib.sha256(f"pi05-noise|{self.seed}|{int(step)}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.standard_normal((self._model_action_horizon, self._model_action_dim)).astype(np.float32)

  def _execute_batch(
      self,
      items: List[Tuple[Dict[str, Any], np.ndarray]],
      padded_size: int,
  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    jax = self._jax
    jnp = self._jnp
    n = len(items)
    padded = list(items) + [items[-1]] * (padded_size - n)
    t_list = [it[0] for it in padded]
    n_list = [it[1] for it in padded]

    with self._lock:
      stacked_np = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *t_list)
      batched_inputs = jax.tree.map(jnp.asarray, stacked_np)
      noise_dev = jnp.asarray(np.stack(n_list, axis=0))
      sample_kwargs = dict(self._policy._sample_kwargs)
      sample_kwargs["noise"] = noise_dev
      observation = self._model_mod.Observation.from_dict(batched_inputs)
      acts_dev = self._policy._sample_actions(self._fixed_rng, observation, **sample_kwargs)
      acts_dev = jax.block_until_ready(acts_dev)
      acts_host = np.asarray(jax.device_get(acts_dev), dtype=np.float32)

    return [(acts_host[i], np.asarray(items[i][0]["state"])) for i in range(n)]

  def _warm_start(self) -> None:
    """Pre-compile all configured static XLA bucket shapes before serving."""
    hw = (224, 224)
    obs = Observation(
        session_id="warm-start",
        step=0,
        images={name: np.zeros((hw[0], hw[1], 3), dtype=np.uint8) for name in self.image_key_map},
        proprio=np.zeros((8,), dtype=np.float32),
        instruction="warm start",
    )
    example = self._example(obs)
    transformed = self._policy._input_transform(example)
    noise = self._noise_for_step(0)
    buckets = self._batcher.buckets if self._batcher is not None else (1,)
    for b in buckets:
      raw_act, raw_state = self._execute_batch([(transformed, noise)] * int(b), int(b))[0]
    out = self._policy._output_transform({"state": raw_state, "actions": raw_act})
    actions = np.asarray(out["actions"], dtype=np.float32)
    if actions.ndim == 1:
      actions = actions[None, :]
    self._horizon, self._action_dim = int(actions.shape[0]), int(actions.shape[1])

  def capabilities(self) -> Caps:
    mode_tag = (
        f"batched{max(self.batch_buckets)}"
        if self._batcher is not None
        else "b1serial"
    )
    return Caps(
        track="VLA_TRACK",
        horizon=int(self._horizon or 10),
        dt_ms=self.dt_ms,
        modalities=("rgb", "proprio", "language"),
        precision=self.precision_tag,
        model_id=f"pi05-3b/{self.config_name}",
        server_impl=f"openpi-jax-{self._openpi_version}+jax-{self._jax_version}-{self._jax_backend}+{mode_tag}",
        action_dim=int(self._action_dim or 7),
        proprio_dim=8,
        cameras=tuple(self.image_key_map.keys()),
        image_height=224,
        image_width=224,
        execute_horizon=self.execute_horizon,
    )

  def reset(self, session_id: str) -> None:
    if self._batcher is not None:
      self._batcher.register_session(session_id)
    return None

  def _example(self, obs: Observation) -> Dict[str, Any]:
    example: Dict[str, Any] = {}
    for camera, key in self.image_key_map.items():
      if camera not in obs.images:
        raise ValueError(
            f"Observation is missing camera {camera!r} (expected cameras: {sorted(self.image_key_map)})."
        )
      example[key] = np.ascontiguousarray(obs.images[camera])
    example["observation/state"] = np.asarray(obs.proprio, dtype=np.float32)
    example["prompt"] = obs.instruction
    return example

  def predict(self, obs: Observation) -> ActionChunk:
    example = self._example(obs)
    transformed = self._policy._input_transform(example)
    noise = self._noise_for_step(obs.step)
    if self._batcher is not None:
      raw_act, raw_state = self._batcher.submit(obs.session_id, (transformed, noise))
    else:
      raw_act, raw_state = self._execute_batch([(transformed, noise)], 1)[0]
    out = self._policy._output_transform({"state": raw_state, "actions": raw_act})
    actions = np.asarray(out["actions"], dtype=np.float32)
    if actions.ndim == 1:
      actions = actions[None, :]
    self._horizon, self._action_dim = int(actions.shape[0]), int(actions.shape[1])
    return ActionChunk(actions=actions)

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    raise NotImplementedError("pi0.5 is a Track 1 VLA policy; it does not implement Track 2 rollout.")

  def close(self) -> None:
    if self._batcher is not None:
      self._batcher.close()
      self._batcher = None
    self._policy = None


__all__ = ["OPENPI_JAX_INSTALL_HINT", "Pi05JaxPolicy"]
