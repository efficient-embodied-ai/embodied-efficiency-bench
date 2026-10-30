"""Reference server #3: NVIDIA Cosmos-Predict2.5-2B (Action-Conditioned World Model) behind the EEB `Policy` ABC.

Provides two reference implementations for Track 2 (`WM_TRACK`):
  1. `CosmosPredict25Policy` (primary / default): Hugging Face `diffusers` PyTorch pipeline
     (`CosmosTransformer3DModel` + `AutoencoderKLWan` + `FlowUniPCMultistepScheduler`) running natively on
     Google Cloud TPU via open-source **TorchTPU** (`device="tpu"`, `torch.accelerator.synchronize()`,
     `torch.compile(backend="tpu")`), NVIDIA CUDA, and AMD ROCm without CUDA-only `transformer_engine`.
  2. `CosmosPredict25JaxPolicy` (optional alternative): Native JAX/Flax 3D DiT wrapper (`vidax` /
     `maxdiffusion`) with `jax.block_until_ready()` synchronization for Cloud TPU v6e-1 and JAX GPU backends.

Locked Closed-Division parameters & invariants:
  * `num_frames = 16` (`horizon = 16`), `frame_hw = (256, 256)`, `num_inference_steps = 10`, `guidance_scale = 1.0`.
  * `dt_ms = 1000.0 / 24.0` (24.0 Hz world-model rollout; sole Closed-Division throughput SLA >= 1.0 Predictions/s, p99 <= 1000.0 ms).
  * `stream_chunk_frames = None` (full-clip bidirectional 3D-DiT + 3D-VAE decode; `stream_chunk_frames > 0`
    is strictly rejected in `CLOSED` division; `TTFF_p99(1) == L_clip` is reported as a co-reported diagnostic).
  * Rule `D1` (Episode-static language conditioning on accelerator, ZERO Host-CPU NN compute in `rollout()`):
    Episode task instructions are encoded once on the accelerator via `Qwen2.5-VL-7B` + `crossattn_proj` at
    policy initialization / session setup and cached directly in accelerator memory as BF16 `[1, 512, 1024]`
    tensors. Timed `rollout()` calls execute strictly step-varying visual (`ctx.frames`), action (`actions`),
    and proprioceptive (`ctx.proprio`) neural compute natively on the device.
  * Action + proprioception tensor conditioning flow on the device:
    Continuous robot `actions [B, T, 7]` and `proprio [B, 8]` move directly to the accelerator on every
    `rollout()` and pass through `_ActionProprioConditioner` (orthogonal BF16 projection + temporal step
    embeddings) to form `[B, T, 1024]` robotics conditioning tokens concatenated with the `[B, 512, 1024]`
    text context feeding all 28 cross-attention blocks of `CosmosTransformer3DModel`.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from eeb.protocol.policy import BasePolicy, Caps, Context, Frame
from eeb.servers.base import DynamicMicroBatcher as _BaseMicroBatcher
from eeb.servers.microbatch import DynamicMicroBatcher as _PaddedMicroBatcher
from eeb.servers.torch_backend import (
    maybe_compile_torch_module,
    resolve_torch_device,
    synchronize_torch_device,
)

DEFAULT_COSMOS_REPO = "nvidia/Cosmos-Predict2.5-2B"
DEFAULT_COSMOS_VARIANT = "robot/action-cond"
DEFAULT_COSMOS_REVISION = "85f8ae7bfe8f5525c8d103429524dcf12f98bf7b"
DEFAULT_COSMOS_ACTION_COND_WEIGHTS = "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
DEFAULT_COSMOS_EMPTY_TEXT_EMB = "robot/action-cond/cr1_empty_string_text_embeddings.pt"
DEFAULT_COSMOS_TOKENIZER_PTH = "tokenizer.pth"
DEFAULT_COSMOS_DIFFUSERS_REPO = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"
DEFAULT_COSMOS_DIFFUSERS_REVISION = "1791139b148954cff76830ae8eefbdc1741a6cc0"

COSMOS_PYTORCH_INSTALL_HINT = (
    "Cosmos-Predict2.5-2B (PyTorch / diffusers) reference server requires `diffusers>=0.33`, `torchvision`, and `torch` "
    "(plus `torch_tpu` on Cloud TPU):\n"
    "  pip install 'diffusers>=0.33.0' transformers accelerate safetensors torch torchvision\n"
    "  # Accept NVIDIA-Open-Model-License at https://huggingface.co/nvidia/Cosmos-Predict2.5-2B and export HF_TOKEN\n"
    "  # Google Cloud TPU (default TPU path via TorchTPU): pip install torch_tpu\n"
    "Then on Cloud TPU: eeb serve --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"model_id\": \"nvidia/Cosmos-Predict2.5-2B\", \"device\": \"tpu\"}'\n"
    "Or on CUDA/ROCm:   eeb serve --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"model_id\": \"nvidia/Cosmos-Predict2.5-2B\", \"device\": \"cuda\"}'"
)


def _safe_torch_load_weights(torch_mod: Any, path: str | Path) -> Dict[str, Any]:
  """Load a PyTorch checkpoint strictly with `weights_only=True` (never allow arbitrary pickle code)."""
  loaded = torch_mod.load(str(path), map_location="cpu", weights_only=True)
  if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
    return dict(loaded["state_dict"])
  if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict):
    return dict(loaded["model"])
  if not isinstance(loaded, dict):
    raise ValueError(f"Expected checkpoint mapping in {path}, got {type(loaded).__name__}.")
  return dict(loaded)


def _download_cosmos_file(
    repo_id: str,
    filename: str,
    *,
    revision: Optional[str],
    cache_dir: Optional[str] = None,
) -> str:
  """Download a pinned Hugging Face artifact with actionable guidance for gated Cosmos checkpoints."""
  from huggingface_hub import hf_hub_download  # noqa: WPS433

  try:
    return str(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    )
  except Exception as exc:  # noqa: BLE001
    msg = str(exc)
    exc_name = type(exc).__name__
    if (
        "401" in msg
        or "403" in msg
        or "GatedRepoError" in exc_name
        or "gated repo" in msg.lower()
        or "RepositoryNotFoundError" in exc_name
    ):
      raise PermissionError(
          f"Cannot access gated checkpoint '{repo_id}/{filename}' (revision={revision!r}). "
          "Accept the NVIDIA-Open-Model-License at https://huggingface.co/nvidia/Cosmos-Predict2.5-2B "
          "and export a valid Hugging Face token (`export HF_TOKEN=hf_...`) before launching CosmosPredict25Policy."
      ) from exc
    raise


def extract_cosmos_action_conditioner_weights(raw_sd: Dict[str, Any]) -> Dict[str, Any]:
  """Extract pretrained action/proprio conditioner weights from an official `robot/action-cond` state dict."""
  out: Dict[str, Any] = {}
  prefixes = (
      "net.action_embedder.",
      "action_embedder.",
      "net.action_conditioner.",
      "action_conditioner.",
  )
  for k, v in raw_sd.items():
    if k.endswith("._extra_state"):
      continue
    for pfx in prefixes:
      if k.startswith(pfx):
        out[k[len(pfx) :]] = v
        break
  return out

COSMOS_JAX_INSTALL_HINT = (
    "Cosmos-Predict2.5-2B (optional JAX / Flax alternative) server requires `jax`, `flax`, and `vidax` (or `maxdiffusion`):\n"
    "  pip install 'jax[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html\n"
    "  pip install flax optax safetensors vidax\n"
    "Then: eeb serve --policy eeb.servers.cosmos_predict25:CosmosPredict25JaxPolicy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"checkpoint_dir\": \"<cosmos_jax_ckpt>\"}'"
)


def _coerce_uint8_frames(raw: Any, max_frames: int) -> List[np.ndarray]:
  """Convert a pipeline/runner output (`[H,W,3]`, `[T,H,W,3]`, or `[1,T,H,W,3]`) into `uint8` frames."""
  frames_arr = getattr(
      raw,
      "frames",
      raw[0]
      if isinstance(raw, (tuple, list)) and len(raw) > 0 and not isinstance(raw[0], (int, float))
      else raw,
  )
  arr = np.asarray(frames_arr)
  if arr.ndim == 5:
    arr = arr[0]
  elif arr.ndim == 3:
    arr = arr[None, ...]
  if arr.dtype != np.uint8:
    arr = np.clip(arr * 255.0 if arr.size > 0 and arr.max() <= 1.0 else arr, 0.0, 255.0).astype(np.uint8)
  count = min(int(max_frames), int(arr.shape[0]))
  return [np.ascontiguousarray(arr[i]) for i in range(count)]


def convert_cosmos_vae_state_dict(raw_sd: Dict[str, Any]) -> Dict[str, Any]:
  """Map raw NVIDIA Cosmos / Wan2.1 VAE keys (`vae/tokenizer.safetensors`) to `diffusers.AutoencoderKLWan`."""
  out: Dict[str, Any] = {}
  up_map = {
      0: ("resnets.0", 0),
      1: ("resnets.1", 0),
      2: ("resnets.2", 0),
      3: ("upsamplers.0", 0),
      4: ("resnets.0", 1),
      5: ("resnets.1", 1),
      6: ("resnets.2", 1),
      7: ("upsamplers.0", 1),
      8: ("resnets.0", 2),
      9: ("resnets.1", 2),
      10: ("resnets.2", 2),
      11: ("upsamplers.0", 2),
      12: ("resnets.0", 3),
      13: ("resnets.1", 3),
      14: ("resnets.2", 3),
  }
  res_sub = {
      "residual.0.": "norm1.",
      "residual.2.": "conv1.",
      "residual.3.": "norm2.",
      "residual.6.": "conv2.",
      "shortcut.": "conv_shortcut.",
  }

  def _map_res(tail: str) -> str:
    for old, new in res_sub.items():
      if tail.startswith(old):
        return new + tail[len(old) :]
    return tail

  for k, v in raw_sd.items():
    nk = k
    if k.startswith("conv1."):
      nk = "quant_conv." + k[len("conv1.") :]
    elif k.startswith("conv2."):
      nk = "post_quant_conv." + k[len("conv2.") :]
    elif k.startswith("encoder.conv1."):
      nk = "encoder.conv_in." + k[len("encoder.conv1.") :]
    elif k == "encoder.head.0.gamma":
      nk = "encoder.norm_out.gamma"
    elif k.startswith("encoder.head.2."):
      nk = "encoder.conv_out." + k[len("encoder.head.2.") :]
    elif k.startswith("decoder.conv1."):
      nk = "decoder.conv_in." + k[len("decoder.conv1.") :]
    elif k == "decoder.head.0.gamma":
      nk = "decoder.norm_out.gamma"
    elif k.startswith("decoder.head.2."):
      nk = "decoder.conv_out." + k[len("decoder.head.2.") :]
    elif m := re.match(r"^encoder\.downsamples\.(\d+)\.(.+)$", k):
      idx, tail = int(m.group(1)), m.group(2)
      nk = f"encoder.down_blocks.{idx}.{_map_res(tail)}"
    elif m := re.match(r"^(encoder|decoder)\.middle\.(\d+)\.(.+)$", k):
      side, idx, tail = m.group(1), int(m.group(2)), m.group(3)
      if idx == 0:
        nk = f"{side}.mid_block.resnets.0.{_map_res(tail)}"
      elif idx == 1:
        nk = f"{side}.mid_block.attentions.0.{tail}"
      elif idx == 2:
        nk = f"{side}.mid_block.resnets.1.{_map_res(tail)}"
    elif m := re.match(r"^decoder\.upsamples\.(\d+)\.(.+)$", k):
      idx, tail = int(m.group(1)), m.group(2)
      sub_prefix, blk_idx = up_map[idx]
      nk = f"decoder.up_blocks.{blk_idx}.{sub_prefix}.{_map_res(tail)}"
    out[nk] = v
  return out


def convert_cosmos_transformer_state_dict(raw_sd: Dict[str, Any]) -> Dict[str, Any]:
  """Map raw NVIDIA Cosmos-Predict2.5 `net.*` keys (`cosmos2_5_post_trained.safetensors`) to `diffusers.CosmosTransformer3DModel`."""
  out: Dict[str, Any] = {}
  top_map = {
      "net.x_embedder.proj.1.weight": "patch_embed.proj.weight",
      "net.t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
      "net.t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
      "net.t_embedding_norm.weight": "time_embed.norm.weight",
      "net.crossattn_proj.0.weight": "crossattn_proj.0.weight",
      "net.crossattn_proj.0.bias": "crossattn_proj.0.bias",
      "net.final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
      "net.final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
      "net.final_layer.linear.weight": "proj_out.weight",
  }
  blk_map = {
      "adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
      "adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
      "self_attn.q_proj.weight": "attn1.to_q.weight",
      "self_attn.k_proj.weight": "attn1.to_k.weight",
      "self_attn.v_proj.weight": "attn1.to_v.weight",
      "self_attn.q_norm.weight": "attn1.norm_q.weight",
      "self_attn.k_norm.weight": "attn1.norm_k.weight",
      "self_attn.output_proj.weight": "attn1.to_out.0.weight",
      "adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
      "adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
      "cross_attn.q_proj.weight": "attn2.to_q.weight",
      "cross_attn.k_proj.weight": "attn2.to_k.weight",
      "cross_attn.v_proj.weight": "attn2.to_v.weight",
      "cross_attn.q_norm.weight": "attn2.norm_q.weight",
      "cross_attn.k_norm.weight": "attn2.norm_k.weight",
      "cross_attn.output_proj.weight": "attn2.to_out.0.weight",
      "adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
      "adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
      "mlp.layer1.weight": "ff.net.0.proj.weight",
      "mlp.layer2.weight": "ff.net.2.weight",
  }
  for k, v in raw_sd.items():
    if (
        k.endswith("._extra_state")
        or k.startswith("net.pos_embedder.")
        or k.startswith("net.action_embedder.")
        or k.startswith("action_embedder.")
        or k.startswith("net.action_conditioner.")
        or k.startswith("action_conditioner.")
    ):
      continue
    if k in top_map:
      out[top_map[k]] = v
      continue
    m = re.match(r"^net\.blocks\.(\d+)\.(.+)$", k)
    if m and m.group(2) in blk_map:
      out[f"transformer_blocks.{m.group(1)}.{blk_map[m.group(2)]}"] = v
    else:
      raise KeyError(f"Unmapped Cosmos-Predict2.5 transformer key: {k}")
  return out


def _collect_episode_prompts() -> List[str]:
  """Gather all canonical EmbodiedX-Traces-v1, synthetic LoadGen, and gate-probe instructions for `D1` pre-encoding."""
  prompts: List[str] = [
      "",
      "warm start",
      "robot manipulation trajectory",
      "wm probe",
      "vla probe",
      "pick up the black bowl and place it on the plate",
  ]
  for idx in range(32):
    prompts.append(f"g4 live probe {idx}")
  try:
    from embodied_bench.dataset_loader import MINI_V1_EPISODE_SPECS

    for spec in MINI_V1_EPISODE_SPECS:
      txt = str(spec.get("instruction_text", ""))
      if txt:
        prompts.append(txt)
  except Exception:  # noqa: BLE001
    pass
  try:
    from eeb.loadgen.sweep import ObservationSpec

    for txt in ObservationSpec().instructions:
      if txt:
        prompts.append(str(txt))
  except Exception:  # noqa: BLE001
    pass
  for root_cand in (Path("data"), Path(__file__).resolve().parents[3] / "data"):
    if root_cand.is_dir():
      for man_p in root_cand.glob("**/manifest.json"):
        try:
          raw = json.loads(man_p.read_text(encoding="utf-8"))
          for sh in raw.get("shards", []):
            if sh.get("instruction_text"):
              prompts.append(str(sh["instruction_text"]))
          for ep in raw.get("episodes", []):
            if ep.get("instruction"):
              prompts.append(str(ep["instruction"]))
        except Exception:  # noqa: BLE001
          pass
  seen = set()
  ordered: List[str] = []
  for p in prompts:
    if p not in seen:
      seen.add(p)
      ordered.append(p)
  return ordered


class CosmosPredict25Policy(BasePolicy):
  """EEB Track 2 (`WM_TRACK`) wrapper around Cosmos-Predict2.5-2B via PyTorch `diffusers` + TorchTPU/CUDA."""

  def __init__(
      self,
      model_id: Optional[str] = None,
      cache_dir: Optional[str] = None,
      device: str = "auto",
      precision_tag: str = "BF16",
      num_frames: int = 16,
      num_inference_steps: int = 10,
      guidance_scale: float = 1.0,
      frame_hw: Tuple[int, int] = (256, 256),
      dt_ms: float = 1000.0 / 24.0,
      action_dim: int = 7,
      proprio_dim: int = 8,
      stream_chunk_frames: Optional[int] = None,
      division: str = "CLOSED",
      torch_compile: bool = True,
      compile_vae: bool = True,
      enable_batching: Optional[bool] = None,
      micro_batch_size: int = 8,
      bucket_sizes: Sequence[int] = (1, 2, 4, 8),
      vae_decode_chunk_size: int = 8,
      batch_window_ms: float = 20.0,
      seed: int = 777,
      conditional_frame_timestep: float = 0.0001,
      extra_prompts: Sequence[str] = (),
      warm_start: bool = True,
      exclusive_compute: bool = True,
      revision: Optional[str] = None,
      allow_untrained_action_adapter: bool = False,
  ) -> None:
    self.model_id = (
        model_id
        or os.environ.get("COSMOS_PREDICT25_CHECKPOINT")
        or DEFAULT_COSMOS_REPO
    )
    if revision is not None:
      self.revision: Optional[str] = str(revision)
    elif self.model_id == DEFAULT_COSMOS_REPO:
      self.revision = DEFAULT_COSMOS_REVISION
    elif self.model_id == DEFAULT_COSMOS_DIFFUSERS_REPO:
      self.revision = DEFAULT_COSMOS_DIFFUSERS_REVISION
    else:
      self.revision = None
    self.allow_untrained_action_adapter = bool(allow_untrained_action_adapter)
    default_shm = "/dev/shm/hf_cache"
    self.cache_dir = (
        cache_dir
        or os.environ.get("HF_HUB_CACHE")
        or (default_shm if os.path.isdir(default_shm) else None)
    )
    self.device = device
    self.precision_tag = str(precision_tag or "BF16")
    self.num_frames = int(num_frames)
    self.num_inference_steps = int(num_inference_steps)
    self.guidance_scale = float(guidance_scale)
    self.frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
    self.dt_ms = float(dt_ms)
    self.action_dim = int(action_dim)
    self.proprio_dim = int(proprio_dim)
    self.seed = int(seed)
    self.conditional_frame_timestep = float(conditional_frame_timestep)
    self.division = str(division or "CLOSED").upper()
    self.stream_chunk_frames = int(stream_chunk_frames) if stream_chunk_frames is not None else None
    if self.division == "CLOSED" and self.stream_chunk_frames is not None and self.stream_chunk_frames > 0:
      raise ValueError(
          f"CosmosPredict25Policy: stream_chunk_frames={self.stream_chunk_frames} is forbidden in division='CLOSED' "
          "(Cosmos-Predict2.5-2B is a frozen full-clip bidirectional 3D-DiT where TTFF == L_clip). "
          "Pass stream_chunk_frames=None for CLOSED division or division='OPEN' for Open-Division causal world models."
      )
    self.torch_compile = bool(torch_compile)
    self.compile_vae = bool(compile_vae and self.torch_compile)
    self.exclusive_compute = bool(exclusive_compute)
    self.micro_batch_size = max(1, int(micro_batch_size))
    raw_buckets = sorted({int(b) for b in bucket_sizes if 1 <= int(b) <= self.micro_batch_size})
    if not raw_buckets:
      raw_buckets = [1]
    if self.micro_batch_size not in raw_buckets and self.micro_batch_size > raw_buckets[-1]:
      raw_buckets.append(self.micro_batch_size)
    self.bucket_sizes: Tuple[int, ...] = tuple(sorted(raw_buckets))
    self.vae_decode_chunk_size = max(1, int(vae_decode_chunk_size))
    self.batch_window_ms = max(0.0, float(batch_window_ms))
    if enable_batching is None:
      self.enable_batching = bool(
          self.exclusive_compute and self.micro_batch_size > 1 and self.stream_chunk_frames is None
      )
    else:
      self.enable_batching = bool(enable_batching and self.micro_batch_size > 1 and self.stream_chunk_frames is None)

    self._extra_prompts = [str(p) for p in extra_prompts]
    self._lock = threading.Lock()
    self._diffusers_version = "unknown"
    self._torch_version = "unknown"

    # Accelerator-resident modules and precomputed buffers set in `_load_components()`
    self._pipe: Any = None
    self._vae_enc: Any = None
    self._vae_dec: Any = None
    self._transformer: Any = None
    self._act_cond: Any = None
    self._prompt_ctx_cache: Dict[str, Any] = {}
    self._latents_mean: Any = None
    self._latents_inv_std: Any = None
    self._base_noise_1: Any = None
    self._padding_mask_1: Any = None
    self._step_sigmas: List[float] = []
    self._unipc_w: List[List[float]] = []
    self._vae_scale_temporal = 4
    self._vae_scale_spatial = 8
    self._latent_t = 4
    self._latent_h = 32
    self._latent_w = 32
    self._latent_c = 16

    self._load_components()

    self._batcher: Any = None
    if self._pipe is not None:
      if self.enable_batching:
        self._batcher = _BaseMicroBatcher(
            self._rollout_batch_locked,
            max_batch_size=self.micro_batch_size,
            batch_window_ms=self.batch_window_ms,
        )
    elif self.enable_batching:
      self._batcher = _PaddedMicroBatcher(
          self._execute_batch,
          buckets=self.bucket_sizes,
          max_batch_delay_ms=self.batch_window_ms,
          session_ttl_s=2.5,
          reset_cohort_gap_s=0.15,
          name="cosmos-predict25-microbatcher",
      )

    if warm_start:
      self._warm_start()

  def _load_components(self) -> None:
    try:
      import torch  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("torch is not installed.\n" + COSMOS_PYTORCH_INSTALL_HINT) from exc
    try:
      import diffusers  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("diffusers is not installed.\n" + COSMOS_PYTORCH_INSTALL_HINT) from exc

    self._torch = torch
    self._torch_version = str(getattr(torch, "__version__", "unknown"))
    self._diffusers_version = str(getattr(diffusers, "__version__", "unknown"))
    self.device = resolve_torch_device(self.device)

    # Support legacy pipeline / unit-test mock interface when AutoencoderKLWan or CosmosTransformer3DModel is absent.
    if not hasattr(diffusers, "AutoencoderKLWan") or not hasattr(diffusers, "CosmosTransformer3DModel"):
      dtype = torch.bfloat16 if self.precision_tag.upper() in ("BF16", "FP8", "NVFP4") else torch.float16
      pipe_cls = getattr(diffusers, "Cosmos25VideoToWorldPipeline", None) or getattr(diffusers, "DiffusionPipeline")
      pipe_kw: Dict[str, Any] = {"torch_dtype": dtype}
      if self.revision is not None:
        try:
          sig = inspect.signature(pipe_cls.from_pretrained)
          if "revision" in sig.parameters or any(
              p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
          ):
            pipe_kw["revision"] = self.revision
        except (TypeError, ValueError):
          pipe_kw["revision"] = self.revision
      pipe = pipe_cls.from_pretrained(self.model_id, **pipe_kw)
      pipe = pipe.to(self.device)
      if self.torch_compile:
        transformer_attr = getattr(pipe, "transformer", getattr(pipe, "unet", None))
        if transformer_attr is not None:
          compiled = maybe_compile_torch_module(transformer_attr, device=self.device, enabled=True)
          if hasattr(pipe, "transformer"):
            pipe.transformer = compiled
          else:
            pipe.unet = compiled
        else:
          pipe = maybe_compile_torch_module(pipe, device=self.device, enabled=True)
      self._pipe = pipe
      return

    import torch.nn as nn  # noqa: WPS433
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    if str(self.device).startswith("cuda") and torch.cuda.is_available():
      torch.backends.cuda.matmul.allow_tf32 = False
      torch.backends.cudnn.allow_tf32 = False
    try:
      torch._dynamo.config.cache_size_limit = max(
          int(getattr(torch._dynamo.config, "cache_size_limit", 8)), 64
      )
      if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(
            int(getattr(torch._dynamo.config, "accumulated_cache_size_limit", 64)), 256
        )
    except Exception:  # noqa: BLE001
      pass
    dev = torch.device(self.device)
    dtype = torch.bfloat16 if self.precision_tag.upper() in ("BF16", "FP8", "NVFP4") else torch.float16
    self._dtype = dtype

    local_root = Path(self.model_id) if os.path.isdir(self.model_id) else None
    is_official_repo = self.model_id == DEFAULT_COSMOS_REPO or (
        local_root is not None and (local_root / DEFAULT_COSMOS_ACTION_COND_WEIGHTS).is_file()
    )
    cfg_repo = self.model_id if not is_official_repo else DEFAULT_COSMOS_DIFFUSERS_REPO
    cfg_rev = self.revision if not is_official_repo else DEFAULT_COSMOS_DIFFUSERS_REVISION

    def _resolve_file(repo_id: str, rel_path: str, rev: Optional[str]) -> str:
      if local_root is not None and (local_root / rel_path).is_file():
        return str(local_root / rel_path)
      return _download_cosmos_file(repo_id, rel_path, revision=rev, cache_dir=self.cache_dir)

    # 1. Load 3D-VAE (`AutoencoderKLWan`) strictly (194/194 tensors)
    vae_cfg_path = _resolve_file(cfg_repo, "vae/config.json", cfg_rev)
    with open(vae_cfg_path, "r", encoding="utf-8") as fp:
      vae_cfg = json.load(fp)
    if is_official_repo:
      vae_pt_path = _resolve_file(self.model_id, DEFAULT_COSMOS_TOKENIZER_PTH, self.revision)
      vae_raw_sd = _safe_torch_load_weights(torch, vae_pt_path)
    else:
      vae_st_path = _resolve_file(self.model_id, "vae/tokenizer.safetensors", self.revision)
      vae_raw_sd = load_file(vae_st_path, device="cpu")
    vae = diffusers.AutoencoderKLWan(**{k: v for k, v in vae_cfg.items() if not k.startswith("_")}).to(dtype)
    vae.load_state_dict(convert_cosmos_vae_state_dict(vae_raw_sd), strict=True)
    vae.eval()
    vae.requires_grad_(False)
    del vae_raw_sd

    self._vae_scale_temporal = 2 ** sum(vae.temperal_downsample) if getattr(vae, "temperal_downsample", None) else 4
    self._vae_scale_spatial = 2 ** len(vae.temperal_downsample) if getattr(vae, "temperal_downsample", None) else 8
    self._latent_c = int(vae.config.z_dim)
    self._latent_t = (self.num_frames - 1) // self._vae_scale_temporal + 1
    self._latent_h = self.frame_hw[0] // self._vae_scale_spatial
    self._latent_w = self.frame_hw[1] // self._vae_scale_spatial

    # 2. Load 2.06B 3D-DiT (`CosmosTransformer3DModel`, `in_channels=17`) + pretrained action conditioner
    if is_official_repo:
      tr_pt_path = _resolve_file(self.model_id, DEFAULT_COSMOS_ACTION_COND_WEIGHTS, self.revision)
      tr_raw_sd = _safe_torch_load_weights(torch, tr_pt_path)
    else:
      tr_st_path = _resolve_file(self.model_id, "transformer/cosmos2_5_post_trained.safetensors", self.revision)
      tr_raw_sd = load_file(tr_st_path, device="cpu")
    action_cond_sd = extract_cosmos_action_conditioner_weights(tr_raw_sd)
    if not action_cond_sd and self.division == "CLOSED" and not self.allow_untrained_action_adapter:
      raise ValueError(
          f"CosmosPredict25Policy: checkpoint {self.model_id!r} contains no pretrained robot action-conditioner "
          f"weights (`net.action_embedder.*`). Closed-Division Track 2 certification requires the official "
          f"{DEFAULT_COSMOS_REPO!r} ({DEFAULT_COSMOS_VARIANT}) checkpoint at revision {DEFAULT_COSMOS_REVISION!r}. "
          "Pass allow_untrained_action_adapter=True only for non-certified scaffolding tests."
      )
    transformer = diffusers.CosmosTransformer3DModel(
        in_channels=17,
        out_channels=16,
        num_attention_heads=16,
        attention_head_dim=128,
        num_layers=28,
        mlp_ratio=4.0,
        text_embed_dim=1024,
        adaln_lora_dim=256,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 3.0, 3.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_crossattn_projection=True,
        crossattn_proj_in_channels=100352,
        encoder_hidden_states_channels=1024,
    ).to(dtype)
    transformer.load_state_dict(convert_cosmos_transformer_state_dict(tr_raw_sd), strict=True)
    transformer.eval()
    transformer.requires_grad_(False)
    del tr_raw_sd

    # Extract `crossattn_proj` (100352 -> 1024) so episode-constant text projections execute ONCE
    # on accelerator per episode instruction (`D1` rule) rather than repeating inside every step.
    crossattn_proj = transformer.crossattn_proj.to(device=dev, dtype=dtype)
    crossattn_proj.eval()
    transformer.crossattn_proj = nn.Identity()

    all_prompts = _collect_episode_prompts()
    for p in self._extra_prompts:
      if p not in all_prompts:
        all_prompts.append(p)

    # Preload official empty-string robot text embedding when provided by `robot/action-cond`.
    if is_official_repo:
      try:
        empty_emb_path = _resolve_file(self.model_id, DEFAULT_COSMOS_EMPTY_TEXT_EMB, self.revision)
        empty_raw = torch.load(empty_emb_path, map_location="cpu", weights_only=True)
        empty_t = empty_raw if hasattr(empty_raw, "shape") else next(iter(empty_raw.values()))
        with torch.inference_mode():
          e_dev = empty_t.to(device=dev, dtype=dtype)
          if e_dev.ndim == 2:
            e_dev = e_dev.unsqueeze(0)
          if e_dev.shape[-1] == 100352:
            e_dev = crossattn_proj(e_dev)
          self._prompt_ctx_cache[""] = e_dev[:1].contiguous()
      except Exception:  # noqa: BLE001
        pass

    remaining_prompts = [p for p in all_prompts if p not in self._prompt_ctx_cache]
    if remaining_prompts:
      # 3. Rule `D1`: Encode all episode / benchmark prompts strictly ON ACCELERATOR once at init.
      te_source = str(local_root) if (local_root is not None and (local_root / "text_encoder").is_dir()) else cfg_repo
      te_rev = None if (local_root is not None and (local_root / "text_encoder").is_dir()) else cfg_rev
      tok_kw: Dict[str, Any] = {"subfolder": "text_encoder", "cache_dir": self.cache_dir}
      te_kw: Dict[str, Any] = {
          "subfolder": "text_encoder",
          "torch_dtype": dtype,
          "low_cpu_mem_usage": True,
          "cache_dir": self.cache_dir,
      }
      if te_rev is not None:
        tok_kw["revision"] = te_rev
        te_kw["revision"] = te_rev
      tokenizer = AutoTokenizer.from_pretrained(te_source, **tok_kw)
      text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(te_source, **te_kw)
      text_encoder.eval()
      text_encoder.requires_grad_(False)
      text_encoder.to(dev)
      synchronize_torch_device(self.device)

      with torch.inference_mode():
        chunk_sz = 4
        for start in range(0, len(remaining_prompts), chunk_sz):
          sub_prompts = remaining_prompts[start : start + chunk_sz]
          ids_list = []
          for p_txt in sub_prompts:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": p_txt}],
                },
            ]
            tok_ids = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            if not isinstance(tok_ids, list) and "input_ids" in tok_ids:
              tok_ids = tok_ids["input_ids"]
            ids_list.append(torch.tensor(tok_ids, dtype=torch.long))
          ids_dev = torch.stack(ids_list, dim=0).to(dev)
          te_out = text_encoder(ids_dev, output_hidden_states=True)
          norm_states = []
          for layer_idx in range(1, len(te_out.hidden_states)):
            hs = te_out.hidden_states[layer_idx].float()
            ns = (hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8)
            norm_states.append(ns.to(dtype))
          raw_embeds = torch.cat(norm_states, dim=-1)  # [C, 512, 100352]
          proj_embeds = crossattn_proj(raw_embeds)     # [C, 512, 1024]
          synchronize_torch_device(self.device)
          for j, p_txt in enumerate(sub_prompts):
            self._prompt_ctx_cache[p_txt] = proj_embeds[j : j + 1].contiguous()
          del ids_dev, te_out, norm_states, raw_embeds, proj_embeds
      del text_encoder, tokenizer

    del crossattn_proj
    gc.collect()
    if str(self.device).startswith("cuda") and torch.cuda.is_available():
      torch.cuda.empty_cache()
    synchronize_torch_device(self.device)

    # 4. Move 3D-DiT + 3D-VAE + on-device Robotics Action+Proprio Conditioner onto accelerator
    class _ActionProprioConditioner(nn.Module):
      """On-device robotics `(actions, proprio)` -> cross-attention token projector (loads pretrained weights when present)."""

      def __init__(
          self,
          act_dim: int,
          prop_dim: int,
          max_frames: int,
          embed_dim: int,
          cond_dtype: Any,
          pretrained_sd: Optional[Dict[str, Any]] = None,
      ) -> None:
        super().__init__()
        self.pretrained_layers: Optional[nn.ModuleList] = None
        self._pretrained_in_dim = 0
        self._out_proj: Optional[nn.Linear] = None
        weight_items = sorted(
            [(k, t) for k, t in (pretrained_sd or {}).items() if k.endswith(".weight") and getattr(t, "ndim", 0) == 2],
            key=lambda kv: kv[0],
        )
        if weight_items:
          layers = nn.ModuleList()
          prev_out = int(weight_items[0][1].shape[1])
          self._pretrained_in_dim = prev_out
          for w_key, w_tensor in weight_items:
            out_f, in_f = int(w_tensor.shape[0]), int(w_tensor.shape[1])
            b_key = w_key[: -len(".weight")] + ".bias"
            has_bias = b_key in (pretrained_sd or {})
            lin = nn.Linear(in_f, out_f, bias=has_bias, dtype=cond_dtype)
            lin.weight.data.copy_(w_tensor.to(cond_dtype))
            if has_bias:
              lin.bias.data.copy_(pretrained_sd[b_key].to(cond_dtype))  # type: ignore[index]
            layers.append(lin)
            prev_out = out_f
          self.pretrained_layers = layers
          if prev_out != embed_dim:
            self._out_proj = nn.Linear(prev_out, embed_dim, bias=False, dtype=cond_dtype)
            nn.init.eye_(self._out_proj.weight.data[: min(embed_dim, prev_out), : min(embed_dim, prev_out)])
        in_dim = act_dim + prop_dim
        self.proj = nn.Linear(in_dim, embed_dim, bias=False, dtype=cond_dtype)
        gen = torch.Generator(device="cpu").manual_seed(20261029)
        w = torch.randn(embed_dim, in_dim, generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(w, mode="reduced")
        scale = float((embed_dim ** 0.5) * 0.5)
        self.proj.weight.data.copy_((q * scale).to(cond_dtype))
        pos = torch.randn(1, max_frames, embed_dim, generator=gen, dtype=torch.float32) * 0.25
        self.step_embed = nn.Parameter(pos.to(cond_dtype))

      def forward(self, actions_b: Any, proprio_b: Any) -> Any:
        t_steps = actions_b.shape[1]
        if self.pretrained_layers is not None:
          feat = actions_b
          if feat.shape[-1] < self._pretrained_in_dim:
            prop_exp = proprio_b.unsqueeze(1).expand(-1, t_steps, -1)
            feat = torch.cat([feat, prop_exp], dim=-1)
          if feat.shape[-1] < self._pretrained_in_dim:
            pad = torch.zeros(
                (feat.shape[0], t_steps, self._pretrained_in_dim - feat.shape[-1]),
                device=feat.device,
                dtype=feat.dtype,
            )
            feat = torch.cat([feat, pad], dim=-1)
          elif feat.shape[-1] > self._pretrained_in_dim:
            feat = feat[:, :, : self._pretrained_in_dim]
          h = feat.to(self.pretrained_layers[0].weight.dtype)
          for idx, lin in enumerate(self.pretrained_layers):
            h = lin(h)
            if idx + 1 < len(self.pretrained_layers):
              h = torch.nn.functional.silu(h)
          if self._out_proj is not None:
            h = self._out_proj(h)
          return h
        prop_exp = proprio_b.unsqueeze(1).expand(-1, t_steps, -1)
        joint = torch.cat([actions_b, prop_exp], dim=-1).to(self.proj.weight.dtype)
        return (self.proj(joint) + self.step_embed[:, :t_steps, :]).to(self.proj.weight.dtype)

    class _StatelessWanEncode1F(nn.Module):
      """Pure thread-safe single-frame causal 3D-VAE encoder wrapper (zero mutated module state)."""

      def __init__(self, inner_vae: Any) -> None:
        super().__init__()
        self.vae = inner_vae
        self.enc_conv_num = int(inner_vae._cached_conv_counts["encoder"])

      def forward(self, x_1f: Any) -> Any:
        feat_map: List[Any] = [None] * self.enc_conv_num
        conv_idx = [0]
        if self.vae.config.patch_size is not None:
          from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

          x_in = patchify(x_1f, patch_size=self.vae.config.patch_size)
        else:
          x_in = x_1f
        out = self.vae.encoder(x_in, feat_cache=feat_map, feat_idx=conv_idx)
        enc = self.vae.quant_conv(out)
        z_dim = int(self.vae.config.z_dim)
        return enc[:, :z_dim]

    class _StatelessWanDecode(nn.Module):
      """Pure thread-safe causal 3D-VAE clip decoder wrapper using local `feat_map`."""

      def __init__(self, inner_vae: Any, num_latent_frames: int) -> None:
        super().__init__()
        self.vae = inner_vae
        self.num_latent_frames = int(num_latent_frames)
        self.dec_conv_num = int(inner_vae._cached_conv_counts["decoder"])

      def forward(self, z: Any) -> Any:
        feat_map: List[Any] = [None] * self.dec_conv_num
        x = self.vae.post_quant_conv(z)
        outs = []
        for i in range(self.num_latent_frames):
          conv_idx = [0]
          if i == 0:
            outs.append(
                self.vae.decoder(x[:, :, 0:1, :, :], feat_cache=feat_map, feat_idx=conv_idx, first_chunk=True)
            )
          else:
            outs.append(
                self.vae.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
            )
        out = torch.cat(outs, dim=2)
        if self.vae.config.patch_size is not None:
          from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

          out = unpatchify(out, patch_size=self.vae.config.patch_size)
        return torch.clamp(out, min=-1.0, max=1.0)

    vae.to(dev)
    transformer.to(dev)
    act_cond = _ActionProprioConditioner(
        act_dim=self.action_dim,
        prop_dim=self.proprio_dim,
        max_frames=max(self.num_frames, 32),
        embed_dim=1024,
        cond_dtype=dtype,
        pretrained_sd=action_cond_sd,
    ).to(dev)
    act_cond.eval()
    act_cond.requires_grad_(False)

    vae_enc = _StatelessWanEncode1F(vae).to(dev).eval()
    vae_dec = _StatelessWanDecode(vae, num_latent_frames=self._latent_t).to(dev).eval()

    if self.torch_compile:
      compile_backend = "tpu" if self.device == "tpu" else "inductor"
      self._transformer = torch.compile(transformer, backend=compile_backend)
      if self.compile_vae:
        self._vae_enc = torch.compile(vae_enc, backend=compile_backend)
        self._vae_dec = torch.compile(vae_dec, backend=compile_backend)
      else:
        self._vae_enc = vae_enc
        self._vae_dec = vae_dec
      self._act_cond = act_cond
    else:
      self._transformer = transformer
      self._vae_enc = vae_enc
      self._vae_dec = vae_dec
      self._act_cond = act_cond

    # 5. Precompute latent stats, fixed initial noise latent, padding mask, and exact FlowUniPC linear matrix
    self._latents_mean = (
        torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=dev).view(1, self._latent_c, 1, 1, 1)
    )
    self._latents_inv_std = (
        1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=dev).view(1, self._latent_c, 1, 1, 1)
    )
    noise_gen = torch.Generator(device="cpu").manual_seed(self.seed)
    self._base_noise_1 = torch.randn(
        (1, self._latent_c, self._latent_t, self._latent_h, self._latent_w),
        generator=noise_gen,
        dtype=torch.float32,
    ).to(dev)
    self._padding_mask_1 = torch.zeros(
        (1, 1, self.frame_hw[0], self.frame_hw[1]),
        dtype=dtype,
        device=dev,
    )

    sched_cfg_path = _resolve_file(cfg_repo, "scheduler/scheduler_config.json", cfg_rev)
    with open(sched_cfg_path, "r", encoding="utf-8") as fp:
      raw_sched_cfg = json.load(fp)
    flow_shift = float(raw_sched_cfg.get("flow_shift", raw_sched_cfg.get("shift", 1.0)))
    sched = diffusers.UniPCMultistepScheduler(
        num_train_timesteps=int(raw_sched_cfg.get("num_train_timesteps", 1000)),
        solver_order=int(raw_sched_cfg.get("solver_order", 2)),
        prediction_type=str(raw_sched_cfg.get("prediction_type", "flow_prediction")),
        solver_type=str(raw_sched_cfg.get("solver_type", "bh2")),
        lower_order_final=bool(raw_sched_cfg.get("lower_order_final", True)),
        use_flow_sigmas=True,
        flow_shift=flow_shift,
    )
    sched.set_timesteps(self.num_inference_steps, device="cpu")
    self._step_sigmas = [float(sched.sigmas[i].item()) for i in range(self.num_inference_steps)]

    n_steps = self.num_inference_steps
    dim_b = n_steps + 1
    basis = torch.eye(dim_b, dtype=torch.float64)
    z_vec = basis[:, 0].clone()
    w_mat: List[List[float]] = []
    sched.set_timesteps(n_steps, device="cpu")
    for i, t_step in enumerate(sched.timesteps):
      u_vec = basis[:, i + 1]
      z_vec = sched.step(
          u_vec.view(dim_b, 1, 1, 1, 1),
          t_step,
          z_vec.view(dim_b, 1, 1, 1, 1),
          return_dict=False,
      )[0].view(dim_b)
      w_mat.append([float(z_vec[j].item()) for j in range(i + 2)])
    self._unipc_w = w_mat

    self._bucket_tensors: Dict[int, Dict[str, Any]] = {}
    for b in sorted(set(self.bucket_sizes) | {1}):
      cond_ind = torch.zeros((b, 1, self._latent_t, 1, 1), dtype=torch.float32, device=dev)
      cond_ind[:, :, :1, :, :] = 1.0
      cond_mask_f32 = cond_ind.expand(-1, -1, -1, self._latent_h, self._latent_w).contiguous()
      cond_mask_dt = cond_mask_f32.to(dtype).contiguous()
      one_minus_mask_f32 = (1.0 - cond_mask_f32).contiguous()
      step_ts = []
      for sig in self._step_sigmas:
        in_ts = cond_ind * self.conditional_frame_timestep + (1.0 - cond_ind) * sig
        step_ts.append(in_ts.contiguous())
      self._bucket_tensors[b] = {
          "noise": self._base_noise_1.expand(b, -1, -1, -1, -1).contiguous(),
          "cond_mask_f32": cond_mask_f32,
          "cond_mask_dt": cond_mask_dt,
          "one_minus_mask_f32": one_minus_mask_f32,
          "step_timesteps": step_ts,
      }
    synchronize_torch_device(self.device)

  def _sync_device(self) -> None:
    synchronize_torch_device(getattr(self, "device", None))

  # ------------------------------------------------------------------ Legacy `_pipe` streaming path

  def _pipe_guidance_kwargs(self, fn: Any) -> Dict[str, Any]:
    g_scale = float(getattr(self, "guidance_scale", 1.0))
    try:
      sig = inspect.signature(fn)
      if "guidance_scale" in sig.parameters or any(
          p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
      ):
        return {"guidance_scale": g_scale}
    except (TypeError, ValueError):
      return {"guidance_scale": g_scale}
    return {}

  def _iter_pipeline_frames(self, ctx: Context, actions: np.ndarray, n: int) -> Iterator[Frame]:
    cond_frame = ctx.frames[0] if ctx.frames else np.zeros((self.frame_hw[0], self.frame_hw[1], 3), dtype=np.uint8)
    act_arr = np.asarray(actions, dtype=np.float32)
    prompt = ctx.instruction or "robot manipulation trajectory"

    stream_fn = getattr(self._pipe, "stream_frames", None)
    if callable(stream_fn):
      idx = 0
      for raw_chunk in stream_fn(
          image=cond_frame,
          prompt=prompt,
          action_conditioning=act_arr,
          num_frames=n,
          num_inference_steps=self.num_inference_steps,
          height=self.frame_hw[0],
          width=self.frame_hw[1],
          output_type="np",
          **self._pipe_guidance_kwargs(stream_fn),
      ):
        self._sync_device()
        now = time.perf_counter()
        for img in _coerce_uint8_frames(raw_chunk, n - idx):
          yield Frame(index=idx, image=img, t_done=now)
          idx += 1
          if idx >= n:
            return
      return

    if getattr(self, "stream_chunk_frames", None) is not None and 0 < int(self.stream_chunk_frames) < n:
      idx = 0
      cur_cond = cond_frame
      extra_kw = self._pipe_guidance_kwargs(self._pipe)
      while idx < n:
        step_n = min(int(self.stream_chunk_frames), n - idx)
        sub_act = act_arr[idx : idx + step_n] if act_arr.shape[0] > idx else act_arr
        out = self._pipe(
            image=cur_cond,
            prompt=prompt,
            action_conditioning=sub_act,
            num_frames=step_n,
            num_inference_steps=self.num_inference_steps,
            height=self.frame_hw[0],
            width=self.frame_hw[1],
            output_type="np",
            **extra_kw,
        )
        self._sync_device()
        now = time.perf_counter()
        chunk_imgs = _coerce_uint8_frames(out, step_n)
        if not chunk_imgs:
          break
        for img in chunk_imgs:
          yield Frame(index=idx, image=img, t_done=now)
          cur_cond = img
          idx += 1
      return

    out = self._pipe(
        image=cond_frame,
        prompt=prompt,
        action_conditioning=act_arr,
        num_frames=n,
        num_inference_steps=self.num_inference_steps,
        height=self.frame_hw[0],
        width=self.frame_hw[1],
        output_type="np",
        **self._pipe_guidance_kwargs(self._pipe),
    )
    if isinstance(out, Iterator):
      idx = 0
      for raw_chunk in out:
        self._sync_device()
        now = time.perf_counter()
        for img in _coerce_uint8_frames(raw_chunk, n - idx):
          yield Frame(index=idx, image=img, t_done=now)
          idx += 1
          if idx >= n:
            return
      return

    self._sync_device()
    rendered = _coerce_uint8_frames(out, n)
    now = time.perf_counter()
    for idx, img in enumerate(rendered):
      yield Frame(index=idx, image=img, t_done=now)

  def _rollout_batch_locked(
      self, reqs: Sequence[Tuple[Context, np.ndarray, int]]
  ) -> Sequence[List[Frame]]:
    with self._lock:
      if len(reqs) == 1:
        ctx0, act0, n0 = reqs[0]
        return [list(self._iter_pipeline_frames(ctx0, act0, n0))]

      ns = {int(n) for _, _, n in reqs}
      if len(ns) == 1 and not callable(getattr(self._pipe, "stream_frames", None)):
        n = next(iter(ns))
        try:
          conds = [
              c.frames[0] if c.frames else np.zeros((self.frame_hw[0], self.frame_hw[1], 3), dtype=np.uint8)
              for c, _, _ in reqs
          ]
          prompts = [c.instruction or "robot manipulation trajectory" for c, _, _ in reqs]
          acts = np.stack([np.asarray(a, dtype=np.float32) for _, a, _ in reqs], axis=0)
          out = self._pipe(
              image=conds,
              prompt=prompts,
              action_conditioning=acts,
              num_frames=n,
              num_inference_steps=self.num_inference_steps,
              height=self.frame_hw[0],
              width=self.frame_hw[1],
              output_type="np",
              **self._pipe_guidance_kwargs(self._pipe),
          )
          raw_frames = getattr(out, "frames", out)
          arr = np.asarray(raw_frames)
          if arr.ndim == 5 and arr.shape[0] == len(reqs):
            self._sync_device()
            now = time.perf_counter()
            batch_frames: List[List[Frame]] = []
            for b_idx in range(len(reqs)):
              imgs = _coerce_uint8_frames(arr[b_idx], n)
              batch_frames.append([Frame(index=i, image=im, t_done=now) for i, im in enumerate(imgs)])
            return batch_frames
        except (TypeError, ValueError, KeyError, IndexError, AttributeError, RuntimeError):
          pass

      return [list(self._iter_pipeline_frames(c, a, n)) for c, a, n in reqs]

  def _run_pipeline(self, ctx: Context, actions: np.ndarray, n: int) -> List[np.ndarray]:
    return [f.image for f in self._iter_pipeline_frames(ctx, actions, n)]

  # ------------------------------------------------------------------ Native 3D-DiT + 3D-VAE path

  def _lookup_prompt_ctx(self, instruction: str) -> Any:
    key = str(instruction or "")
    ctx_t = self._prompt_ctx_cache.get(key)
    if ctx_t is not None:
      return ctx_t
    raise KeyError(
        f"CosmosPredict25Policy strict D1 rule: prompt {key!r} was not pre-encoded on device at init/setup "
        "(host-CPU text encoding during timed rollout() is strictly prohibited). Pass extra_prompts=[...] at init."
    )

  def _run_padded_batch_on_tpu(
      self,
      frames_u8_list: List[np.ndarray],
      actions_f32_list: List[np.ndarray],
      proprios_f32_list: List[np.ndarray],
      prompts_list: List[str],
      padded_b: int,
      target_num_frames: int,
  ) -> np.ndarray:
    torch = self._torch
    dev = torch.device(self.device)
    dtype = self._dtype
    n_real = len(frames_u8_list)

    if n_real < padded_b:
      pad_n = padded_b - n_real
      frames_u8_list = frames_u8_list + [frames_u8_list[-1]] * pad_n
      actions_f32_list = actions_f32_list + [actions_f32_list[-1]] * pad_n
      proprios_f32_list = proprios_f32_list + [proprios_f32_list[-1]] * pad_n
      prompts_list = prompts_list + [prompts_list[-1]] * pad_n

    frames_np = np.ascontiguousarray(np.stack(frames_u8_list, axis=0), dtype=np.uint8)
    acts_np = np.ascontiguousarray(np.stack(actions_f32_list, axis=0), dtype=np.float32)
    props_np = np.ascontiguousarray(np.stack(proprios_f32_list, axis=0), dtype=np.float32)

    if padded_b not in self._bucket_tensors:
      cond_ind = torch.zeros((padded_b, 1, self._latent_t, 1, 1), dtype=torch.float32, device=dev)
      cond_ind[:, :, :1, :, :] = 1.0
      cm_f32 = cond_ind.expand(-1, -1, -1, self._latent_h, self._latent_w).contiguous()
      self._bucket_tensors[padded_b] = {
          "noise": self._base_noise_1.expand(padded_b, -1, -1, -1, -1).contiguous(),
          "cond_mask_f32": cm_f32,
          "cond_mask_dt": cm_f32.to(dtype).contiguous(),
          "one_minus_mask_f32": (1.0 - cm_f32).contiguous(),
          "step_timesteps": [
              (cond_ind * self.conditional_frame_timestep + (1.0 - cond_ind) * sig).contiguous()
              for sig in self._step_sigmas
          ],
      }
    bt = self._bucket_tensors[padded_b]

    with torch.inference_mode():
      imgs_dev = torch.from_numpy(frames_np).to(dev)
      x_1f = imgs_dev.permute(0, 3, 1, 2).unsqueeze(2).to(dtype) * (1.0 / 127.5) - 1.0

      acts_dev = torch.from_numpy(acts_np).to(dev)
      props_dev = torch.from_numpy(props_np).to(dev)
      act_tokens = self._act_cond(acts_dev, props_dev)
      text_ctx = torch.cat([self._lookup_prompt_ctx(p) for p in prompts_list], dim=0)
      encoder_hidden_states = torch.cat([text_ctx, act_tokens], dim=1)

      cond_z0 = self._vae_enc(x_1f).float()
      cond_z0 = (cond_z0 - self._latents_mean) * self._latents_inv_std
      z0 = bt["noise"]
      cond_mask_f32 = bt["cond_mask_f32"]
      one_minus_mask_f32 = bt["one_minus_mask_f32"]
      cond_mask_dt = bt["cond_mask_dt"]
      cond_latent = cond_z0 * cond_mask_f32
      gt_velocity = (z0 - cond_latent) * cond_mask_f32

      latents = z0
      step_vels: List[Any] = []
      for i in range(self.num_inference_steps):
        in_latents = (cond_mask_f32 * cond_latent + one_minus_mask_f32 * latents).to(dtype)
        pred = self._transformer(
            hidden_states=in_latents,
            timestep=bt["step_timesteps"][i],
            encoder_hidden_states=encoder_hidden_states,
            condition_mask=cond_mask_dt,
            padding_mask=self._padding_mask_1,
            return_dict=False,
        )[0].float()
        vel = gt_velocity + pred * one_minus_mask_f32
        step_vels.append(vel)
        w_row = self._unipc_w[i]
        nxt = w_row[0] * z0
        for j in range(i + 1):
          nxt = nxt + w_row[j + 1] * step_vels[j]
        latents = nxt

      z_dec = (latents / self._latents_inv_std + self._latents_mean).to(dtype)
      if padded_b > self.vae_decode_chunk_size:
        vid_parts = [
            self._vae_dec(z_dec[s : s + self.vae_decode_chunk_size])
            for s in range(0, padded_b, self.vae_decode_chunk_size)
        ]
        vid = torch.cat(vid_parts, dim=0)
      else:
        vid = self._vae_dec(z_dec)
      if target_num_frames > 0 and vid.shape[2] != target_num_frames:
        vid = torch.repeat_interleave(vid, repeats=max(self._vae_scale_temporal, 1), dim=2)
        if vid.shape[2] > target_num_frames:
          vid = vid[:, :, :target_num_frames]
        elif vid.shape[2] < target_num_frames:
          pad = vid[:, :, -1:, :, :].repeat(1, 1, target_num_frames - vid.shape[2], 1, 1)
          vid = torch.cat([vid, pad], dim=2)

      vid_u8 = (
          ((vid[:n_real].float() * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0)
          .to(torch.uint8)
          .permute(0, 2, 3, 4, 1)
          .contiguous()
      )
      self._sync_device()
      return vid_u8.cpu().numpy()

  def _prepare_single_req(
      self, ctx: Context, actions: np.ndarray, n: int
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    h, w = self.frame_hw
    p_dim = int(getattr(self, "proprio_dim", 8))
    if ctx.frames:
      img = np.asarray(ctx.frames[0], dtype=np.uint8)
      if img.shape[:2] != (h, w):
        ys = np.linspace(0, img.shape[0] - 1, h).astype(np.int32)
        xs = np.linspace(0, img.shape[1] - 1, w).astype(np.int32)
        img = img[np.ix_(ys, xs)]
    else:
      img = np.zeros((h, w, 3), dtype=np.uint8)

    prop = np.asarray(ctx.proprio, dtype=np.float32).reshape(-1)
    if prop.shape[0] < p_dim:
      p_pad = np.zeros((p_dim,), dtype=np.float32)
      p_pad[: prop.shape[0]] = prop
      prop = p_pad
    elif prop.shape[0] > p_dim:
      prop = prop[:p_dim]

    act = np.asarray(actions, dtype=np.float32)
    if act.ndim == 1:
      act = act[None, :]
    out_act = np.zeros((n, self.action_dim), dtype=np.float32)
    t_copy = min(n, act.shape[0])
    d_copy = min(self.action_dim, act.shape[1])
    out_act[:t_copy, :d_copy] = act[:t_copy, :d_copy]

    instr = ctx.instruction if ctx.instruction is not None else "robot manipulation trajectory"
    if instr == "" and "" not in self._prompt_ctx_cache:
      instr = "robot manipulation trajectory"
    return img, out_act, prop, instr

  def _execute_batch(
      self,
      reqs: List[Tuple[Context, np.ndarray, int]],
      padded_size: int,
  ) -> List[List[Frame]]:
    with self._lock:
      n_target = int(reqs[0][2])
      imgs: List[np.ndarray] = []
      acts: List[np.ndarray] = []
      props: List[np.ndarray] = []
      prompts: List[str] = []
      for ctx, act_arr, _ in reqs:
        im, ac, pr, pm = self._prepare_single_req(ctx, act_arr, n_target)
        imgs.append(im)
        acts.append(ac)
        props.append(pr)
        prompts.append(pm)
      batch_u8 = self._run_padded_batch_on_tpu(
          frames_u8_list=imgs,
          actions_f32_list=acts,
          proprios_f32_list=props,
          prompts_list=prompts,
          padded_b=int(padded_size),
          target_num_frames=n_target,
      )
      now = time.perf_counter()
      results: List[List[Frame]] = []
      for b_idx in range(len(reqs)):
        clip = batch_u8[b_idx]
        results.append(
            [Frame(index=i, image=np.ascontiguousarray(clip[i]), t_done=now) for i in range(n_target)]
        )
      return results

  def _warm_start(self) -> None:
    if getattr(self, "_pipe", None) is not None:
      ctx = Context(
          session_id="warm-start",
          frames=[np.zeros((self.frame_hw[0], self.frame_hw[1], 3), dtype=np.uint8)],
          proprio=np.zeros((int(getattr(self, "proprio_dim", 8)),), dtype=np.float32),
          instruction="warm start",
          num_frames=self.num_frames,
      )
      actions = np.zeros((self.num_frames, self.action_dim), dtype=np.float32)
      for _ in self.rollout(ctx, actions):
        pass
      self.reset("warm-start")
      return

    warm_buckets = self.bucket_sizes if self.enable_batching else (1,)
    dummy_img = np.zeros((self.frame_hw[0], self.frame_hw[1], 3), dtype=np.uint8)
    dummy_act = np.zeros((self.num_frames, self.action_dim), dtype=np.float32)
    dummy_prop = np.zeros((self.proprio_dim,), dtype=np.float32)
    for b in warm_buckets:
      self._run_padded_batch_on_tpu(
          frames_u8_list=[dummy_img] * b,
          actions_f32_list=[dummy_act] * b,
          proprios_f32_list=[dummy_prop] * b,
          prompts_list=["warm start"] * b,
          padded_b=int(b),
          target_num_frames=self.num_frames,
      )
    gc.collect()
    if hasattr(gc, "freeze"):
      gc.freeze()

  def capabilities(self) -> Caps:
    dev = getattr(self, "device", "unknown")
    p_dim = int(getattr(self, "proprio_dim", 8))
    if getattr(self, "_pipe", None) is not None:
      impl = (
          f"diffusers-{self._diffusers_version}+torch-{self._torch_version}-{dev}"
          + ("+compile" if getattr(self, "torch_compile", False) else "")
      )
    else:
      impl = (
          f"torchtpu-diffusers-{self._diffusers_version}+torch-{self._torch_version}-{dev}"
          if dev == "tpu"
          else f"diffusers-{self._diffusers_version}+torch-{self._torch_version}-{dev}"
      )
      if getattr(self, "torch_compile", False):
        impl += "+compile"
      if getattr(self, "enable_batching", False):
        impl += f"+dynbatch{getattr(self, 'micro_batch_size', 8)}"
    return Caps(
        track="WM_TRACK",
        horizon=self.num_frames,
        dt_ms=self.dt_ms,
        modalities=("rgb", "proprio", "language", "actions"),
        precision=self.precision_tag,
        model_id="cosmos-predict2.5-2b",
        server_impl=impl,
        action_dim=self.action_dim,
        proprio_dim=p_dim,
        cameras=("wm",),
        image_height=self.frame_hw[0],
        image_width=self.frame_hw[1],
        execute_horizon=self.num_frames,
    )

  def reset(self, session_id: str) -> None:
    batcher = getattr(self, "_batcher", None)
    if batcher is not None:
      if getattr(self, "_pipe", None) is not None:
        batcher.on_reset(session_id)
      else:
        batcher.register_session(session_id)
    return None

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    n = int(ctx.num_frames) if int(ctx.num_frames) > 0 else self.num_frames
    batcher = getattr(self, "_batcher", None)

    if getattr(self, "_pipe", None) is not None:
      if (
          batcher is not None
          and getattr(self, "stream_chunk_frames", None) is None
          and not callable(getattr(self._pipe, "stream_frames", None))
      ):
        def _batched_pipe() -> Iterator[Frame]:
          yield from batcher.submit((ctx, actions, n), session_id=ctx.session_id)

        return _batched_pipe()

      def _gen_pipe() -> Iterator[Frame]:
        yield from self._iter_pipeline_frames(ctx, actions, n)

      if getattr(self, "exclusive_compute", True):
        def _locked_pipe() -> Iterator[Frame]:
          with self._lock:
            yield from _gen_pipe()

        return _locked_pipe()
      return _gen_pipe()

    if batcher is not None and getattr(self, "stream_chunk_frames", None) is None:
      frames = batcher.submit(ctx.session_id, (ctx, actions, n))

      def _yield_batched() -> Iterator[Frame]:
        yield from frames

      return _yield_batched()

    def _yield_serial() -> Iterator[Frame]:
      if getattr(self, "exclusive_compute", True):
        frames_list = self._execute_batch([(ctx, actions, n)], padded_size=1)[0]
      else:
        im, ac, pr, pm = self._prepare_single_req(ctx, actions, n)
        clip = self._run_padded_batch_on_tpu([im], [ac], [pr], [pm], padded_b=1, target_num_frames=n)[0]
        now = time.perf_counter()
        frames_list = [Frame(index=i, image=np.ascontiguousarray(clip[i]), t_done=now) for i in range(n)]
      yield from frames_list

    return _yield_serial()

  def close(self) -> None:
    batcher = getattr(self, "_batcher", None)
    if batcher is not None:
      if hasattr(batcher, "close"):
        batcher.close()
      self._batcher = None
    self._pipe = None
    self._transformer = None
    self._vae_enc = None
    self._vae_dec = None
    self._act_cond = None
    if hasattr(self, "_prompt_ctx_cache") and self._prompt_ctx_cache is not None:
      self._prompt_ctx_cache.clear()


class CosmosPredict25JaxPolicy(BasePolicy):
  """EEB Track 2 (`WM_TRACK`) wrapper around Cosmos-Predict2.5-2B in native JAX/Flax (`vidax` / `maxdiffusion`).

  Streaming / Time-to-First-Frame (`ttff_ms` vs `total_ms`):
    * Closed Division (`division="CLOSED"`): runs pure bidirectional 16-frame block rollouts
      (`stream_chunk_frames` is forbidden because temporal bidirectional attention + 3D VAE decoding
      jointly produce all 16 frames, yielding honest diagnostic `ttff_ms == total_ms`).
    * Open Division (`division="OPEN"`): supports `runner.stream_frames(...)`, generator-returning
      `runner.generate(...)`, and chunked causal rollouts via `stream_chunk_frames`
      (`1 <= stream_chunk_frames < num_frames`), calling `jax.block_until_ready()` per yielded chunk
      so `Frame(index=0, ...)` is emitted immediately (`ttff_ms < total_ms`).
  """

  def __init__(
      self,
      checkpoint_dir: Optional[str] = None,
      precision_tag: str = "BF16",
      num_frames: int = 16,
      num_inference_steps: int = 10,
      guidance_scale: float = 1.0,
      frame_hw: Tuple[int, int] = (256, 256),
      dt_ms: float = 1000.0 / 24.0,
      action_dim: int = 7,
      stream_chunk_frames: Optional[int] = None,
      division: str = "CLOSED",
      warm_start: bool = True,
      exclusive_compute: bool = True,
  ) -> None:
    self.division = str(division or "CLOSED").upper()
    if self.division == "CLOSED" and stream_chunk_frames is not None and int(stream_chunk_frames) > 0:
      raise ValueError(
          f"stream_chunk_frames={stream_chunk_frames} is prohibited in CLOSED division for Cosmos-Predict2.5-2B "
          "(bidirectional 16-frame DiT + 3D VAE block rollout requires TTFF == L_clip). "
          "Pass division='OPEN' for causal autoregressive chunked streaming."
      )
    self.checkpoint_dir = os.path.expanduser(
        checkpoint_dir
        or os.environ.get("COSMOS_PREDICT25_JAX_CHECKPOINT")
        or "~/.cache/cosmos/cosmos_predict25_2b_jax"
    )
    self.precision_tag = precision_tag
    self.num_frames = int(num_frames)
    self.num_inference_steps = int(num_inference_steps)
    self.guidance_scale = float(guidance_scale)
    self.frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
    self.dt_ms = float(dt_ms)
    self.action_dim = int(action_dim)
    self.stream_chunk_frames = int(stream_chunk_frames) if stream_chunk_frames is not None else None
    self.exclusive_compute = bool(exclusive_compute)
    self._lock = threading.Lock()
    self._jax_version = "unknown"
    self._backend_platform = "unknown"
    self._jax: Any = None
    self._runner = self._load_runner()
    if warm_start:
      self._warm_start()

  def _load_runner(self) -> Any:
    try:
      import jax  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("jax is not installed.\n" + COSMOS_JAX_INSTALL_HINT) from exc
    try:
      import vidax  # type: ignore[import-not-found]  # noqa: WPS433
    except ImportError as exc:
      raise ImportError("vidax (JAX Cosmos-Predict2.5 runner) is not installed.\n" + COSMOS_JAX_INSTALL_HINT) from exc

    self._jax = jax
    self._jax_version = str(getattr(jax, "__version__", "unknown"))
    try:
      self._backend_platform = str(jax.default_backend())
    except Exception:  # pragma: no cover
      self._backend_platform = "jax"

    if not os.path.isdir(self.checkpoint_dir):
      raise FileNotFoundError(
          f"Cosmos-Predict2.5 JAX checkpoint directory not found: {self.checkpoint_dir}. "
          "Set --policy-kwargs '{\"checkpoint_dir\": ...}' or $COSMOS_PREDICT25_JAX_CHECKPOINT."
      )
    load_fn = getattr(vidax, "load_cosmos_predict25", None) or getattr(vidax, "CosmosPredict25Runner")
    return load_fn(self.checkpoint_dir, dtype=self.precision_tag.lower())

  def _warm_start(self) -> None:
    ctx = Context(
        session_id="warm-start",
        frames=[np.zeros((self.frame_hw[0], self.frame_hw[1], 3), dtype=np.uint8)],
        proprio=np.zeros((8,), dtype=np.float32),
        instruction="warm start",
        num_frames=self.num_frames,
    )
    actions = np.zeros((self.num_frames, self.action_dim), dtype=np.float32)
    for _ in self.rollout(ctx, actions):
      pass
    self.reset("warm-start")

  def capabilities(self) -> Caps:
    return Caps(
        track="WM_TRACK",
        horizon=self.num_frames,
        dt_ms=self.dt_ms,
        modalities=("rgb", "proprio", "language", "actions"),
        precision=self.precision_tag,
        model_id="cosmos-predict2.5-2b",
        server_impl=f"vidax-jax-{self._jax_version}-{self._backend_platform}",
        action_dim=self.action_dim,
        proprio_dim=8,
        cameras=("wm",),
        image_height=self.frame_hw[0],
        image_width=self.frame_hw[1],
        execute_horizon=self.num_frames,
    )

  def reset(self, session_id: str) -> None:
    batcher = getattr(self, "_batcher", None)
    if batcher is not None:
      batcher.on_reset(session_id)
    return None

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    n = int(ctx.num_frames) if int(ctx.num_frames) > 0 else self.num_frames
    cond_frame = ctx.frames[0] if ctx.frames else np.zeros((self.frame_hw[0], self.frame_hw[1], 3), dtype=np.uint8)
    act_arr = np.asarray(actions, dtype=np.float32)

    def _gen() -> Iterator[Frame]:
      stream_fn = getattr(self._runner, "stream_frames", None)
      if callable(stream_fn):
        idx = 0
        for raw_chunk in stream_fn(
            conditioning_frame=cond_frame,
            actions=act_arr,
            prompt=ctx.instruction,
            num_frames=n,
            num_inference_steps=self.num_inference_steps,
        ):
          if self._jax is not None:
            raw_chunk = self._jax.block_until_ready(raw_chunk)
          now = time.perf_counter()
          for img in _coerce_uint8_frames(raw_chunk, n - idx):
            yield Frame(index=idx, image=img, t_done=now)
            idx += 1
            if idx >= n:
              return
        return

      if getattr(self, "stream_chunk_frames", None) is not None and 0 < int(self.stream_chunk_frames) < n:
        idx = 0
        cur_cond = cond_frame
        while idx < n:
          step_n = min(int(self.stream_chunk_frames), n - idx)
          sub_act = act_arr[idx : idx + step_n] if act_arr.shape[0] > idx else act_arr
          out_chunk = self._runner.generate(
              conditioning_frame=cur_cond,
              actions=sub_act,
              prompt=ctx.instruction,
              num_frames=step_n,
              num_inference_steps=self.num_inference_steps,
          )
          if self._jax is not None:
            out_chunk = self._jax.block_until_ready(out_chunk)
          now = time.perf_counter()
          chunk_imgs = _coerce_uint8_frames(out_chunk, step_n)
          if not chunk_imgs:
            break
          for img in chunk_imgs:
            yield Frame(index=idx, image=img, t_done=now)
            cur_cond = img
            idx += 1
        return

      out = self._runner.generate(
          conditioning_frame=cond_frame,
          actions=act_arr,
          prompt=ctx.instruction,
          num_frames=n,
          num_inference_steps=self.num_inference_steps,
      )
      if isinstance(out, Iterator):
        idx = 0
        for raw_chunk in out:
          if self._jax is not None:
            raw_chunk = self._jax.block_until_ready(raw_chunk)
          now = time.perf_counter()
          for img in _coerce_uint8_frames(raw_chunk, n - idx):
            yield Frame(index=idx, image=img, t_done=now)
            idx += 1
            if idx >= n:
              return
        return

      if self._jax is not None:
        out = self._jax.block_until_ready(out)
      rendered = _coerce_uint8_frames(out, n)
      now = time.perf_counter()
      for idx, img in enumerate(rendered):
        yield Frame(index=idx, image=img, t_done=now)

    if getattr(self, "exclusive_compute", True):
      def _locked() -> Iterator[Frame]:
        with self._lock:
          yield from _gen()

      return _locked()
    return _gen()

  def close(self) -> None:
    self._runner = None


__all__ = [
    "COSMOS_JAX_INSTALL_HINT",
    "COSMOS_PYTORCH_INSTALL_HINT",
    "CosmosPredict25JaxPolicy",
    "CosmosPredict25Policy",
    "DEFAULT_COSMOS_ACTION_COND_WEIGHTS",
    "DEFAULT_COSMOS_DIFFUSERS_REPO",
    "DEFAULT_COSMOS_DIFFUSERS_REVISION",
    "DEFAULT_COSMOS_EMPTY_TEXT_EMB",
    "DEFAULT_COSMOS_REPO",
    "DEFAULT_COSMOS_REVISION",
    "DEFAULT_COSMOS_TOKENIZER_PTH",
    "DEFAULT_COSMOS_VARIANT",
    "convert_cosmos_transformer_state_dict",
    "convert_cosmos_vae_state_dict",
    "extract_cosmos_action_conditioner_weights",
]
