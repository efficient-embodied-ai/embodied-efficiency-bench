"""Reference server #1: Physical Intelligence pi0.5 (`lerobot/pi05_libero_finetuned` or `openpi` PyTorch) behind the EEB `Policy` ABC.

Supports single-chip execution across:
  - NVIDIA CUDA and AMD ROCm (`device="cuda"`, `torch.cuda.synchronize()`, optional `torch_compile=True`).
  - Google Cloud TPU via open-source **TorchTPU** (`pip install torch torch_tpu`, `device="tpu"`,
    `torch.tpu.synchronize()`, optional `torch_compile=True`), plus native JAX/Flax on Cloud TPU via
    `eeb.servers.pi05.policy_jax:Pi05JaxPolicy`.
  - Automatic probing (`device="auto"` -> CUDA/ROCm -> TorchTPU -> CPU fallback).

Backends (`backend="auto"`):
  - `"lerobot"` (default when a Hugging Face repo ID such as `"lerobot/pi05_libero_finetuned"` or a
    LeRobot checkpoint directory containing `config.json` is supplied): loads real public PyTorch
    weights strictly frozen in pure BF16 via `lerobot.policies.pi05`, runs true batched inference
    across concurrent LoadGen sessions via `DynamicMicroBatcher`, and when `torch_compile=True`
    enables SigLIP + PaliGemma + Gemma-Expert SDPA + zero-copy KV-cache cropping + `torch.compile`
    graph fusion pre-warmed across batch buckets `(1, 2, 4, 8, 16)`.
  - `"openpi"` (automatic fallback for converted `openpi` checkpoint directories and unit-test mocks).
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from eeb.protocol.policy import ActionChunk, BasePolicy, Caps, Context, Frame, Observation
from eeb.servers.base import DynamicMicroBatcher
from eeb.servers.torch_backend import (
    maybe_compile_torch_module,
    resolve_torch_device,
    synchronize_torch_device,
)

DEFAULT_LEROBOT_PI05_REPO = "lerobot/pi05_libero_finetuned_v044"
DEFAULT_LEROBOT_PI05_REVISION = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
DEFAULT_GEMMA_TOKENIZER_REPO = "unsloth/gemma-2b"
DEFAULT_GEMMA_TOKENIZER_REVISION = "7ac9d201a57c8cdc6f939069fc0f044c60197a4a"

OPENPI_INSTALL_HINT = (
    "pi0.5 reference server requires either `lerobot` (recommended public HF PyTorch path) or `openpi`, plus `torch` and `torchvision`:\n"
    "  pip install torch torchvision transformers accelerate safetensors\n"
    "  git clone https://github.com/huggingface/lerobot.git && pip install -e lerobot\n"
    "  # Google Cloud TPU via TorchTPU: pip install torch_tpu\n"
    "  # Or native JAX on Cloud TPU v6e-1: eeb.servers.pi05.policy_jax:Pi05JaxPolicy\n"
    "Then on CUDA/ROCm:\n"
    "  eeb serve --policy eeb.servers.pi05:Pi05Policy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"checkpoint_dir\": \"lerobot/pi05_libero_finetuned_v044\", \"device\": \"cuda\"}'\n"
    "Or on Cloud TPU:\n"
    "  eeb serve --policy eeb.servers.pi05:Pi05Policy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"checkpoint_dir\": \"lerobot/pi05_libero_finetuned_v044\", \"device\": \"tpu\"}'"
)

# LIBERO-style observation keys consumed by openpi's `LiberoInputs` transform.
DEFAULT_IMAGE_KEY_MAP: Dict[str, str] = {
    "agentview": "observation/image",
    "wrist": "observation/wrist_image",
}

DEFAULT_LEROBOT_IMAGE_KEY_MAP: Dict[str, str] = {
    "agentview": "observation.images.image",
    "wrist": "observation.images.image2",
}


class Pi05Policy(BasePolicy):
  """EEB `Policy` wrapper around pi0.5-3B (`lerobot/pi05_libero_finetuned` or `openpi` PyTorch).

  Locked Closed-Division parameters:
    * `execute_horizon = 5` (locked replan horizon `K_exec = 5`)
    * `horizon = 10` (10-action flow-matching chunk matching official `pi05_libero`)
    * `num_inference_steps = 10` (10 Euler flow-matching steps)
    * `obs_image_hw = (224, 224)` (2 x 224x224 RGB cameras)
    * `dt_ms = 50.0` (20 Hz control loop)
  """

  def __init__(
      self,
      checkpoint_dir: Optional[str] = None,
      config_name: str = "pi05_libero",
      device: str = "auto",
      image_key_map: Optional[Dict[str, str]] = None,
      precision_tag: str = "BF16",
      dt_ms: float = 50.0,
      horizon: int = 10,
      execute_horizon: int = 5,
      torch_compile: bool = False,
      warm_start: bool = True,
      exclusive_compute: bool = True,
      micro_batch_size: int = 16,
      batch_window_ms: float = 8.0,
      backend: str = "auto",
      tokenizer_name: str = DEFAULT_GEMMA_TOKENIZER_REPO,
      num_inference_steps: int = 10,
      obs_image_hw: Tuple[int, int] = (224, 224),
      compile_buckets: Sequence[int] = (1, 2, 4, 8, 16),
      revision: Optional[str] = None,
  ) -> None:
    self._explicit_checkpoint_dir = checkpoint_dir
    self.checkpoint_dir = os.path.expanduser(
        checkpoint_dir
        or os.environ.get("LEROBOT_PI05_CHECKPOINT")
        or os.environ.get("OPENPI_PI05_CHECKPOINT")
        or DEFAULT_LEROBOT_PI05_REPO
    )
    self.revision = (
        str(revision).strip()
        if revision
        else (DEFAULT_LEROBOT_PI05_REVISION if self.checkpoint_dir == DEFAULT_LEROBOT_PI05_REPO else None)
    )
    self.config_name = config_name
    self.device = device
    self.image_key_map = dict(image_key_map or DEFAULT_IMAGE_KEY_MAP)
    self.precision_tag = precision_tag
    self.dt_ms = float(dt_ms)
    self._requested_horizon = max(1, int(horizon))
    self.execute_horizon = int(execute_horizon)
    self.torch_compile = bool(torch_compile)
    self.exclusive_compute = bool(exclusive_compute)
    self.micro_batch_size = max(1, int(micro_batch_size))
    self.batch_window_ms = max(0.0, float(batch_window_ms))
    self.backend_requested = str(backend or "auto").strip().lower()
    self.tokenizer_name = str(tokenizer_name or DEFAULT_GEMMA_TOKENIZER_REPO)
    self.num_inference_steps = max(1, int(num_inference_steps))
    self._obs_image_hw = (int(obs_image_hw[0]), int(obs_image_hw[1]))
    self._compile_buckets = tuple(int(b) for b in compile_buckets if int(b) > 0)
    self._backend_kind = "openpi"
    self._lock = threading.Lock()
    self._batcher: Optional[DynamicMicroBatcher[Dict[str, Any], np.ndarray]] = (
        DynamicMicroBatcher(
            self._infer_batch_locked,
            max_batch_size=self.micro_batch_size,
            batch_window_ms=self.batch_window_ms,
        )
        if self.exclusive_compute and self.micro_batch_size > 1
        else None
    )
    self._horizon: Optional[int] = None
    self._action_dim: Optional[int] = None
    self._openpi_version = "unknown"
    self._lerobot_version = "unknown"
    self._torch_version = "unknown"
    self._pre: Any = None
    self._post: Any = None
    self._noise_1: Any = None
    self._timesteps: Any = None
    self._policy = self._load_policy()
    if warm_start:
      self._warm_start()

  # ------------------------------------------------------------------ backend selection & loading

  def _should_use_openpi(self) -> bool:
    if self.backend_requested == "openpi":
      return True
    if self.backend_requested == "lerobot":
      return False
    if "openpi.policies.policy_config" in sys.modules:
      return True
    if os.path.isdir(self.checkpoint_dir):
      has_lerobot_cfg = os.path.exists(os.path.join(self.checkpoint_dir, "policy_preprocessor.json")) or os.path.exists(
          os.path.join(self.checkpoint_dir, "config.json")
      )
      if not has_lerobot_cfg:
        return True
    return False

  def _load_policy(self) -> Any:
    try:
      import torch  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("torch is not installed.\n" + OPENPI_INSTALL_HINT) from exc

    self._torch_version = str(getattr(torch, "__version__", "unknown"))
    if self._should_use_openpi():
      return self._load_openpi_policy(torch)
    return self._load_lerobot_policy(torch)

  def _load_openpi_policy(self, torch: Any) -> Any:
    try:
      from openpi.policies import policy_config as _policy_config  # noqa: WPS433
      from openpi.training import config as _config  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("openpi is not installed or not importable.\n" + OPENPI_INSTALL_HINT) from exc

    self._backend_kind = "openpi"
    try:
      import importlib.metadata as _md

      self._openpi_version = _md.version("openpi")
    except Exception:  # pragma: no cover
      self._openpi_version = "source"

    if not os.path.isdir(self.checkpoint_dir):
      raise FileNotFoundError(
          f"pi0.5 checkpoint directory not found: {self.checkpoint_dir}. "
          "Pass --policy-kwargs '{\"checkpoint_dir\": \"lerobot/pi05_libero_finetuned\"}' "
          "or convert an openpi JAX checkpoint."
      )
    self.device = resolve_torch_device(self.device)
    config = _config.get_config(self.config_name)
    kwargs: Dict[str, Any] = {}
    try:
      sig = inspect.signature(_policy_config.create_trained_policy)
      if "pytorch_device" in sig.parameters:
        kwargs["pytorch_device"] = self.device
    except (TypeError, ValueError):  # pragma: no cover
      pass
    policy = _policy_config.create_trained_policy(config, self.checkpoint_dir, **kwargs)
    if self.torch_compile:
      model_attr = getattr(policy, "_model", getattr(policy, "model", None))
      if model_attr is not None:
        compiled = maybe_compile_torch_module(model_attr, device=self.device, enabled=True)
        if hasattr(policy, "_model"):
          policy._model = compiled
        else:
          policy.model = compiled
      elif hasattr(policy, "infer"):
        policy.infer = maybe_compile_torch_module(policy.infer, device=self.device, enabled=True)
    return policy

  def _load_lerobot_policy(self, torch: Any) -> Any:
    import torch.nn as nn

    try:
      import lerobot.policies.pi05.processor_pi05  # noqa: F401, WPS433
      from lerobot.policies.factory import make_pre_post_processors  # noqa: WPS433
      from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: WPS433
      from lerobot.policies.pi05.modeling_pi05 import PI05Policy as LeRobotPI05Policy  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("lerobot (`pi05`) is not installed.\n" + OPENPI_INSTALL_HINT) from exc

    self._backend_kind = "lerobot"
    try:
      import importlib.metadata as _md

      self._lerobot_version = _md.version("lerobot")
    except Exception:  # pragma: no cover
      self._lerobot_version = "source"

    self.device = resolve_torch_device(self.device)
    is_cuda = str(self.device).startswith("cuda")
    if is_cuda:
      torch.backends.cuda.matmul.allow_tf32 = False
      torch.backends.cudnn.allow_tf32 = False

    target_dtype = (
        torch.bfloat16 if self.precision_tag.upper() in ("BF16", "FP8", "NVFP4") else torch.float32
    )

    # Skip redundant random Kaiming CPU initialization on weights immediately overwritten by safetensors.
    orig_kaiming = nn.init.kaiming_uniform_
    orig_uniform = nn.init.uniform_
    orig_normal = nn.init.normal_
    nn.init.kaiming_uniform_ = lambda t, *a, **kw: t
    nn.init.uniform_ = lambda t, *a, **kw: t
    nn.init.normal_ = lambda t, *a, **kw: t
    try:
      rev_kw: Dict[str, Any] = {}
      if self.revision and not os.path.isdir(self.checkpoint_dir):
        try:
          sig_cfg = inspect.signature(PI05Config.from_pretrained)
          if "revision" in sig_cfg.parameters or any(
              p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_cfg.parameters.values()
          ):
            rev_kw["revision"] = self.revision
        except (TypeError, ValueError):
          pass
      cfg = PI05Config.from_pretrained(self.checkpoint_dir, **rev_kw)
      cfg.device = self.device
      cfg.dtype = target_dtype
      cfg.chunk_size = int(self._requested_horizon)
      cfg.n_action_steps = int(self._requested_horizon)
      cfg.num_inference_steps = self.num_inference_steps
      pol_kw: Dict[str, Any] = {"config": cfg}
      if self.revision and not os.path.isdir(self.checkpoint_dir):
        try:
          sig_pol = inspect.signature(LeRobotPI05Policy.from_pretrained)
          if "revision" in sig_pol.parameters or any(
              p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_pol.parameters.values()
          ):
            pol_kw["revision"] = self.revision
        except (TypeError, ValueError):
          pass
      policy = LeRobotPI05Policy.from_pretrained(self.checkpoint_dir, **pol_kw)
    finally:
      nn.init.kaiming_uniform_ = orig_kaiming
      nn.init.uniform_ = orig_uniform
      nn.init.normal_ = orig_normal

    policy.to(device=self.device)
    policy.eval()

    # Cast SigLIP vision tower + multi-modal projector cleanly to BF16 (LeRobot keeps them FP32
    # by default for training-time optimizer convenience; pure BF16 matches TPU v6e-1 1:1).
    pge = policy.model.paligemma_with_expert
    if target_dtype == torch.bfloat16:
      pge.paligemma.model.vision_tower.to(dtype=torch.bfloat16)
      pge.paligemma.model.multi_modal_projector.to(dtype=torch.bfloat16)

    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=self.checkpoint_dir,
        preprocessor_overrides={
            "device_processor": {"device": self.device},
            "tokenizer_processor": {"tokenizer_name": self.tokenizer_name},
        },
    )
    self._pre = pre
    self._post = post
    self._horizon = int(policy.config.chunk_size)
    self._action_dim = int(policy.config.output_features["action"].shape[0])

    gen_noise = torch.Generator(device="cpu").manual_seed(0)
    self._noise_1 = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        generator=gen_noise,
        dtype=torch.float32,
    ).to(device=self.device)
    self._timesteps = torch.linspace(
        1.0,
        1.0 / float(self.num_inference_steps),
        self.num_inference_steps,
        device=self.device,
        dtype=torch.float32,
    )

    attn_impl = "sdpa" if self.torch_compile else "eager"
    pge.paligemma.model.vision_tower.config._attn_implementation = attn_impl
    pge.paligemma.model.language_model.config._attn_implementation = attn_impl
    pge.gemma_expert.model.config._attn_implementation = attn_impl

    if self.torch_compile:
      torch._dynamo.config.cache_size_limit = max(64, int(torch._dynamo.config.cache_size_limit))
      if is_cuda:
        pge.paligemma.model.vision_tower.forward = torch.compile(
            pge.paligemma.model.vision_tower.forward, mode="default", dynamic=False
        )
        pge.paligemma.model.language_model.forward = torch.compile(
            pge.paligemma.model.language_model.forward, mode="default", dynamic=False
        )
        pge.gemma_expert.model.forward = torch.compile(
            pge.gemma_expert.model.forward, mode="default", dynamic=False
        )
      else:
        pge.paligemma.model.vision_tower.forward = maybe_compile_torch_module(
            pge.paligemma.model.vision_tower.forward,
            device=self.device,
            enabled=True,
            mode="default",
        )
        pge.paligemma.model.language_model.forward = maybe_compile_torch_module(
            pge.paligemma.model.language_model.forward,
            device=self.device,
            enabled=True,
            mode="default",
        )
        pge.gemma_expert.model.forward = maybe_compile_torch_module(
            pge.gemma_expert.model.forward,
            device=self.device,
            enabled=True,
            mode="default",
        )
    return policy

  def _sync_device(self) -> None:
    """Ensure accelerator kernels complete before measuring host wall-clock latency."""
    synchronize_torch_device(getattr(self, "device", None))

  def _warm_start(self) -> None:
    """Warm up single and batched shapes so CUDA/TorchInductor compilation never hits timed runs."""
    hw = self._obs_image_hw if self._backend_kind == "lerobot" else (224, 224)
    sample = Observation(
        session_id="warm-start",
        step=0,
        images={name: np.zeros((hw[0], hw[1], 3), dtype=np.uint8) for name in self.image_key_map},
        proprio=np.zeros((8,), dtype=np.float32),
        instruction="pick up the black bowl and place it on the plate",
    )
    if self._backend_kind == "lerobot" and self.torch_compile:
      ex = self._example(sample)
      for b in self._compile_buckets:
        if b <= self.micro_batch_size:
          self._infer_batch_locked([ex] * b)
    else:
      self.predict(sample)
    gc.collect()
    if hasattr(gc, "freeze"):
      gc.freeze()

  # ------------------------------------------------------------------ Policy API

  def capabilities(self) -> Caps:
    if self._backend_kind == "lerobot":
      return Caps(
          track="VLA_TRACK",
          horizon=int(self._horizon or self._requested_horizon),
          dt_ms=self.dt_ms,
          modalities=("rgb", "proprio", "language"),
          precision=self.precision_tag,
          model_id="pi05-3b",
          server_impl=(
              f"lerobot-pi05-{self._lerobot_version}+torch-{self._torch_version}-{getattr(self, 'device', 'unknown')}"
              + ("+sdpa+compile" if getattr(self, "torch_compile", False) else "-eager")
          ),
          action_dim=int(self._action_dim or 7),
          proprio_dim=8,
          cameras=tuple(self.image_key_map.keys()),
          image_height=self._obs_image_hw[0],
          image_width=self._obs_image_hw[1],
          execute_horizon=self.execute_horizon,
      )
    return Caps(
        track="VLA_TRACK",
        horizon=int(self._horizon or self._requested_horizon),
        dt_ms=self.dt_ms,
        modalities=("rgb", "proprio", "language"),
        precision=self.precision_tag,
        model_id=f"pi05-3b/{self.config_name}",
        server_impl=(
            f"openpi-pytorch-{self._openpi_version}+torch-{self._torch_version}-{getattr(self, 'device', 'unknown')}"
            + ("+compile" if getattr(self, "torch_compile", False) else "")
        ),
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
            f"Observation is missing camera {camera!r} (expected cameras: {sorted(self.image_key_map)}). "
            "Configure image_key_map to match the recipe's camera names."
        )
      example[key] = np.ascontiguousarray(obs.images[camera])
    example["observation/state"] = np.asarray(obs.proprio, dtype=np.float32)
    example["prompt"] = obs.instruction or "pick up the object"
    return example

  def _coerce_single_actions(self, raw_actions: Any) -> np.ndarray:
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim == 1:
      actions = actions[None, :]
    return actions

  def _infer_lerobot_batch(self, examples: Sequence[Dict[str, Any]]) -> Sequence[np.ndarray]:
    import torch
    from lerobot.policies.pi05.modeling_pi05 import clone_past_key_values, make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    orig_n = len(examples)
    run_examples: List[Dict[str, Any]] = list(examples)
    if self.torch_compile and self._compile_buckets and orig_n > 0:
      for b in sorted(self._compile_buckets):
        if b >= orig_n:
          if b > orig_n:
            run_examples = run_examples + [run_examples[0]] * (b - orig_n)
          break

    cam_names = list(self.image_key_map.keys())
    primary_key = self.image_key_map[cam_names[0]]
    wrist_key = self.image_key_map[cam_names[1]] if len(cam_names) > 1 else primary_key

    imgs_primary: List[Any] = []
    imgs_wrist: List[Any] = []
    states_list: List[Any] = []
    tasks: List[str] = []
    for ex in run_examples:
      arr_p = ex[primary_key]
      arr_w = ex[wrist_key]
      imgs_primary.append(torch.from_numpy(arr_p).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0)
      imgs_wrist.append(torch.from_numpy(arr_w).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0)
      st = np.asarray(ex["observation/state"], dtype=np.float32).reshape(-1)
      if st.shape[0] < 8:
        st_pad = np.zeros((8,), dtype=np.float32)
        st_pad[: st.shape[0]] = st
        st = st_pad
      states_list.append(torch.from_numpy(st[:8].copy()))
      tasks.append(str(ex["prompt"]))

    batched_raw = {
        "observation.images.image": torch.stack(imgs_primary, dim=0),
        "observation.images.image2": torch.stack(imgs_wrist, dim=0),
        "observation.state": torch.stack(states_list, dim=0),
        "task": tasks,
    }

    with torch.inference_mode():
      proc = self._pre(batched_raw)

      m = self._policy.model
      pge = m.paligemma_with_expert
      use_opt = self.torch_compile

      images, img_masks = self._policy._preprocess_images(proc)
      vis_dtype = pge.paligemma.model.vision_tower.dtype
      images = [img.to(dtype=vis_dtype) for img in images]
      tokens = proc[OBS_LANGUAGE_TOKENS]
      masks = proc[OBS_LANGUAGE_ATTENTION_MASK]

      prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, tokens, masks)
      prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
      prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
      att_4d = m._prepare_attention_masks_4d(prefix_att_2d, dtype=prefix_embs.dtype)

      _, past_key_values = pge.forward(
          attention_mask=att_4d,
          position_ids=prefix_pos,
          past_key_values=None,
          inputs_embeds=[prefix_embs, None],
          use_cache=True,
      )
      prefix_kv_len = int(prefix_embs.shape[1])

      bsize = int(tokens.shape[0])
      x_t = self._noise_1.expand(bsize, -1, -1).clone()
      dt = -1.0 / float(self.num_inference_steps)
      action_len = int(m.config.chunk_size)

      if use_opt:
        suffix_pad_masks = torch.ones((bsize, action_len), dtype=torch.bool, device=self.device)
        suffix_att_masks = torch.zeros((bsize, action_len), dtype=prefix_pad_masks.dtype, device=self.device)
        suffix_att_masks[:, 0] = 1
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, action_len, prefix_kv_len)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_4d = m._prepare_attention_masks_4d(full_att_2d, dtype=prefix_embs.dtype)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_pos = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        for step_idx in range(self.num_inference_steps):
          t_batch = self._timesteps[step_idx].expand(bsize)
          past_key_values.crop(prefix_kv_len)
          suffix_embs, _, _, adarms_cond = m.embed_suffix(x_t, t_batch)
          (_, suffix_out), _ = pge.forward(
              attention_mask=full_att_4d,
              position_ids=suffix_pos,
              past_key_values=past_key_values,
              inputs_embeds=[None, suffix_embs],
              use_cache=False,
              adarms_cond=[None, adarms_cond],
          )
          suffix_out = suffix_out[:, -action_len:]
          v_t = m.action_out_proj(suffix_out.to(dtype=torch.float32))
          x_t = x_t + dt * v_t
      else:
        for step_idx in range(self.num_inference_steps):
          t_batch = self._timesteps[step_idx].expand(bsize)
          suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = m.embed_suffix(x_t, t_batch)
          suffix_len = int(suffix_pad_masks.shape[1])
          prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_kv_len)
          suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
          full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
          prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
          suffix_pos = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
          full_att_4d = m._prepare_attention_masks_4d(full_att_2d, dtype=prefix_embs.dtype)
          kv_step = clone_past_key_values(past_key_values)
          (_, suffix_out), _ = pge.forward(
              attention_mask=full_att_4d,
              position_ids=suffix_pos,
              past_key_values=kv_step,
              inputs_embeds=[None, suffix_embs],
              use_cache=False,
              adarms_cond=[None, adarms_cond],
          )
          suffix_out = suffix_out[:, -action_len:]
          v_t = m.action_out_proj(suffix_out.to(dtype=torch.float32))
          x_t = x_t + dt * v_t

      raw_act = x_t[:, :, : self._action_dim]
      unnorm = self._post.process_action(raw_act)
      self._sync_device()
      arr = np.ascontiguousarray(unnorm.float().cpu().numpy(), dtype=np.float32)
      return [arr[i] for i in range(orig_n)]

  def _infer_batch_locked(self, examples: Sequence[Dict[str, Any]]) -> Sequence[np.ndarray]:
    with self._lock:
      if self._backend_kind == "lerobot":
        return self._infer_lerobot_batch(examples)

      if len(examples) == 1:
        res = self._policy.infer(examples[0])
        self._sync_device()
        return [self._coerce_single_actions(res["actions"])]

      infer_batch_fn = getattr(self._policy, "infer_batch", None)
      if callable(infer_batch_fn):
        raw_list = infer_batch_fn(list(examples))
        self._sync_device()
        return [
            self._coerce_single_actions(r["actions"] if isinstance(r, dict) else r)
            for r in raw_list
        ]

      try:
        stacked: Dict[str, Any] = {}
        keys = list(examples[0].keys())
        for k in keys:
          vals = [ex[k] for ex in examples]
          if isinstance(vals[0], np.ndarray):
            stacked[k] = np.stack(vals, axis=0)
          else:
            stacked[k] = vals
        batched_res = self._policy.infer(stacked)
        batched_act = np.asarray(batched_res["actions"], dtype=np.float32)
        if batched_act.ndim == 3 and batched_act.shape[0] == len(examples):
          self._sync_device()
          return [batched_act[i] for i in range(len(examples))]
      except (TypeError, ValueError, KeyError, IndexError, AttributeError, RuntimeError):
        pass

      outs = [self._coerce_single_actions(self._policy.infer(ex)["actions"]) for ex in examples]
      self._sync_device()
      return outs

  def predict(self, obs: Observation) -> ActionChunk:
    example = self._example(obs)
    batcher = getattr(self, "_batcher", None)
    if batcher is not None:
      actions = batcher.submit(example, session_id=obs.session_id)
    elif getattr(self, "exclusive_compute", True):
      actions = self._infer_batch_locked([example])[0]
    else:
      if self._backend_kind == "lerobot":
        actions = self._infer_lerobot_batch([example])[0]
      else:
        result = self._policy.infer(example)
        self._sync_device()
        actions = self._coerce_single_actions(result["actions"])
    self._horizon, self._action_dim = int(actions.shape[0]), int(actions.shape[1])
    return ActionChunk(actions=actions)

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    raise NotImplementedError("pi0.5 is a Track 1 VLA policy; it does not implement Track 2 rollout.")

  def close(self) -> None:
    batcher = getattr(self, "_batcher", None)
    if batcher is not None and hasattr(batcher, "close"):
      batcher.close()
    self._policy = None
    self._pre = None
    self._post = None


__all__ = [
    "DEFAULT_GEMMA_TOKENIZER_REPO",
    "DEFAULT_IMAGE_KEY_MAP",
    "DEFAULT_LEROBOT_IMAGE_KEY_MAP",
    "DEFAULT_LEROBOT_PI05_REPO",
    "OPENPI_INSTALL_HINT",
    "Pi05Policy",
]
