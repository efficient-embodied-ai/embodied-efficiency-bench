"""Reference server #2: NVIDIA Isaac GR00T-N1.7-3B behind the EEB `Policy` ABC.

Supports single-chip PyTorch execution across:
  - NVIDIA CUDA and AMD ROCm (`device="cuda"`, `torch.cuda.synchronize()`, optional `torch_compile=True`
    + static power-of-2 micro-batching `B in {1, 2, 4, 8, 16}`).
  - Google Cloud TPU via open-source **TorchTPU** (`pip install torch torch_tpu`, `device="tpu"`,
    `torch.accelerator.synchronize()`, optional `torch_compile=True` + static micro-batching).

Locked Closed-Division parameters:
  * `horizon = 16`
  * `execute_horizon = 8` (locked replan interval for closed-loop pacing)
  * `num_inference_timesteps = 16` (16 flow-matching DiT steps)
  * `image_height = 256`, `image_width = 256`
  * `dt_ms = 50.0` (20 Hz control loop)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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

GR00T_INSTALL_HINT = (
    "GR00T-N1.7-3B reference server requires `isaac-gr00t`, `torch`, and `torchvision` (plus `torch_tpu` on Cloud TPU):\n"
    "  pip install torch torchvision transformers accelerate safetensors\n"
    "  git clone https://github.com/NVIDIA/Isaac-GR00T.git\n"
    "  cd Isaac-GR00T && pip install --no-deps -e .\n"
    "  # Google Cloud TPU (default TPU path via TorchTPU): pip install torch torch_tpu\n"
    "  # Download checkpoint from Hugging Face (e.g., nvidia/GR00T-N1.7-LIBERO)\n"
    "Then on CUDA/ROCm:\n"
    "  eeb serve --policy eeb.servers.gr00t:Gr00tN17Policy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"model_path\": \"nvidia/GR00T-N1.7-LIBERO/libero_10\", \"device\": \"cuda\", \"torch_compile\": true}'\n"
    "Or on Cloud TPU:\n"
    "  eeb serve --policy eeb.servers.gr00t:Gr00tN17Policy --uds /tmp/eeb.sock "
    "--policy-kwargs '{\"model_path\": \"/dev/shm/gr00t_n17_libero/libero_10\", \"device\": \"tpu\", \"torch_compile\": true}'"
)

DEFAULT_GR00T_IMAGE_KEY_MAP: Dict[str, str] = {
    "agentview": "image",
    "wrist": "wrist_image",
}

CANONICAL_GR00T_N17_REPO = "nvidia/GR00T-N1.7-LIBERO"
CANONICAL_GR00T_N17_SUBDIR = "libero_10"
CANONICAL_GR00T_N17_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
DEFAULT_QWEN3_VL_CONFIG_FALLBACK = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_QWEN3_VL_CONFIG_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def _step_seed(base_seed: int, step: int) -> int:
  h = hashlib.blake2b(f"{int(base_seed)}:{int(step)}".encode("utf-8"), digest_size=8).digest()
  return int.from_bytes(h, "little") & 0x7FFF_FFFF_FFFF_FFFF


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


def _install_gr00t_v17_backbone_compat(
    qwen_fallback_repo: str = DEFAULT_QWEN3_VL_CONFIG_FALLBACK,
    qwen_fallback_revision: str = DEFAULT_QWEN3_VL_CONFIG_REVISION,
) -> None:
  """Hydrate Qwen3-VL architecture config & processor from ungated repo when Cosmos-Reason2-2B is gated.

  All 1,031 VLM + DiT pretrained parameter tensors are subsequently loaded strictly from the
  GR00T-N1.7 checkpoint safetensors (`strict=True`). Also skips CUDA-only channels_last patch_embed.
  """
  import torch
  import transformers
  import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as pg
  import gr00t.model.modules.qwen3_backbone as qb

  if getattr(qb, "_eeb_qwen3_compat_installed", False):
    return

  orig_from_pretrained = transformers.Qwen3VLForConditionalGeneration.from_pretrained
  orig_build_proc = pg.build_processor

  def _compat_qwen_from_pretrained(model_name: Any, *args: Any, **kwargs: Any) -> Any:
    if "Cosmos-Reason2" in str(model_name) and not os.path.isdir(str(model_name)):
      cfg = transformers.AutoConfig.from_pretrained(
          qwen_fallback_repo,
          revision=qwen_fallback_revision,
          trust_remote_code=True,
      )
      return transformers.Qwen3VLForConditionalGeneration._from_config(
          cfg, attn_implementation="sdpa", dtype=torch.bfloat16
      )
    return orig_from_pretrained(model_name, *args, **kwargs)

  def _compat_build_processor(model_name: Any, transformers_loading_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    kw = dict(transformers_loading_kwargs or {})
    if "Cosmos-Reason2" in str(model_name) and not os.path.isdir(str(model_name)):
      return transformers.Qwen3VLProcessor.from_pretrained(
          qwen_fallback_repo,
          revision=qwen_fallback_revision,
      )
    return orig_build_proc(model_name, kw)

  transformers.Qwen3VLForConditionalGeneration.from_pretrained = _compat_qwen_from_pretrained  # type: ignore[method-assign]
  pg.build_processor = _compat_build_processor
  if hasattr(qb.Qwen3Backbone, "_apply_vision_patch_embed_channels_last"):
    qb.Qwen3Backbone._apply_vision_patch_embed_channels_last = lambda self: None
  qb._eeb_qwen3_compat_installed = True  # type: ignore[attr-defined]


def _install_dense_vision_attention(model: Any) -> None:
  """Replace `Qwen3VLVisionAttention`'s per-image `cu_seqlens.tolist()` Python loop with dense batched SDPA."""
  import torch
  import torch.nn.functional as F
  from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

  visual = model.backbone.model.model.visual
  for blk in visual.blocks:
    attn = blk.attn

    def _dense_vis_attn(
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        _a: Any = attn,
    ) -> torch.Tensor:
      total_seq = hidden_states.shape[0]
      num_imgs = cu_seqlens.shape[0] - 1
      seq_per_img = total_seq // num_imgs
      q, k, v = (
          _a.qkv(hidden_states)
          .reshape(num_imgs, seq_per_img, 3, _a.num_heads, -1)
          .permute(2, 0, 3, 1, 4)
          .unbind(0)
      )
      assert position_embeddings is not None
      cos, sin = position_embeddings
      q, k = mq.apply_rotary_pos_emb_vision(
          q.transpose(1, 2),
          k.transpose(1, 2),
          cos.view(num_imgs, seq_per_img, -1),
          sin.view(num_imgs, seq_per_img, -1),
      )
      q, k = q.transpose(1, 2), k.transpose(1, 2)
      out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
      return _a.proj(out.transpose(1, 2).reshape(total_seq, -1))

    attn.forward = _dense_vis_attn


def _install_tpu_safe_action_attention(model: Any) -> None:
  """Work around `torch_tpu==0.2.0` broadcast boolean mask bug in `AlternateVLDiT` cross-attention."""
  import torch
  import torch.nn.functional as F
  from diffusers.models.attention_processor import Attention

  class _TpuSafeMathAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
      residual = hidden_states
      if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
      input_ndim = hidden_states.ndim
      if input_ndim == 4:
        bsz, ch, ht, wd = hidden_states.shape
        hidden_states = hidden_states.view(bsz, ch, ht * wd).transpose(1, 2)
      bsz, seq_len, _ = (
          hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
      )
      bool_mask_4d = None
      if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
          if attention_mask.ndim == 2:
            bool_mask_4d = attention_mask[:, None, None, :]
          elif attention_mask.ndim == 3:
            bool_mask_4d = attention_mask[:, None, :, :]
          else:
            bool_mask_4d = attention_mask
        else:
          attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bsz)
          attention_mask = attention_mask.view(bsz, attn.heads, -1, attention_mask.shape[-1])
      if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
      query = attn.to_q(hidden_states)
      if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
      elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
      key = attn.to_k(encoder_hidden_states)
      value = attn.to_v(encoder_hidden_states)
      inner_dim = key.shape[-1]
      head_dim = inner_dim // attn.heads
      query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
      key = key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
      value = value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
      if attn.norm_q is not None:
        query = attn.norm_q(query)
      if attn.norm_k is not None:
        key = attn.norm_k(key)
      scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
      if bool_mask_4d is not None:
        scores = scores.masked_fill(~bool_mask_4d, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        probs = probs * bool_mask_4d.any(dim=-1, keepdim=True).to(torch.float32)
        probs = probs.to(query.dtype)
      else:
        if attention_mask is not None:
          scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
      out = torch.matmul(probs, value)
      out = out.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim).to(query.dtype)
      out = attn.to_out[1](attn.to_out[0](out))
      if input_ndim == 4:
        out = out.transpose(-1, -2).reshape(bsz, ch, ht, wd)
      if attn.residual_connection:
        out = out + residual
      return out / attn.rescale_output_factor

  for mod in model.action_head.modules():
    if isinstance(mod, Attention):
      mod.set_processor(_TpuSafeMathAttnProcessor())


@dataclass
class _PreparedGr00tSample:
  vlm_text: str
  vlm_images: List[Any]
  state_tensor: Any  # [1, T_s, D_s] bf16 CPU
  embodiment_id: Any  # [1] int64 CPU
  init_noise: Any  # [1, H_act, D_act] bf16 CPU
  raw_states: Dict[str, np.ndarray]  # per-key [T_s, d] float32


class Gr00tN17Policy(BasePolicy):
  """EEB `Policy` wrapper around NVIDIA Isaac GR00T-N1.7-3B (Track 1 VLA).

  Locked Closed-Division parameters:
    * `horizon = 16`
    * `execute_horizon = 8` (locked replan interval for closed-loop pacing)
    * `dt_ms = 50.0` (20 Hz control loop)
  """

  def __init__(
      self,
      model_path: Optional[str] = None,
      embodiment_tag: str = "libero_sim",
      data_config: str = "libero",
      denoising_steps: int = 16,
      device: str = "auto",
      image_key_map: Optional[Dict[str, str]] = None,
      precision_tag: str = "BF16",
      dt_ms: float = 50.0,
      horizon: int = 16,
      execute_horizon: int = 8,
      max_token_length: int = 192,
      obs_image_hw: Tuple[int, int] = (256, 256),
      image_hw: Optional[Sequence[int]] = None,
      torch_compile: bool = False,
      enable_batching: bool = True,
      batch_buckets: Sequence[int] = (1, 2, 4, 8, 16),
      micro_batch_size: int = 16,
      batch_window_ms: float = 8.0,
      max_batch_delay_ms: Optional[float] = None,
      seed: int = 20261029,
      warm_start: bool = True,
      exclusive_compute: bool = True,
      revision: Optional[str] = None,
  ) -> None:
    default_local_home = os.path.expanduser("~/models/gr00t_n17_libero/libero_10")
    default_local_shm = "/dev/shm/gr00t_n17_libero/libero_10"
    fallback_path = (
        default_local_home
        if os.path.isdir(default_local_home)
        else (default_local_shm if os.path.isdir(default_local_shm) else CANONICAL_GR00T_N17_REPO)
    )
    self.model_path = os.path.expanduser(
        model_path or os.environ.get("GR00T_N17_CHECKPOINT") or fallback_path
    )
    self.revision = str(revision).strip() if revision else CANONICAL_GR00T_N17_REVISION
    self.embodiment_tag_str = str(embodiment_tag or "libero_sim")
    self.data_config = data_config
    self.denoising_steps = int(denoising_steps)
    self.device = device
    self.image_key_map = dict(image_key_map or DEFAULT_GR00T_IMAGE_KEY_MAP)
    self.precision_tag = precision_tag
    self.dt_ms = float(dt_ms)
    self._horizon = int(horizon)
    self.execute_horizon = int(execute_horizon)
    self.max_token_length = int(max_token_length)
    resolved_hw = image_hw if image_hw is not None else obs_image_hw
    if len(resolved_hw) != 2:
      raise ValueError(f"image_hw/obs_image_hw must have length 2 (H, W), got {resolved_hw!r}")
    self._obs_image_hw = (int(resolved_hw[0]), int(resolved_hw[1]))
    self.image_hw = self._obs_image_hw
    self.torch_compile = bool(torch_compile)
    self.enable_batching = bool(enable_batching)
    self.batch_buckets: Tuple[int, ...] = tuple(sorted({int(b) for b in batch_buckets if int(b) > 0}))
    self.micro_batch_size = max(1, int(micro_batch_size))
    resolved_window = max_batch_delay_ms if max_batch_delay_ms is not None else batch_window_ms
    self.batch_window_ms = max(0.0, float(resolved_window))
    self.max_batch_delay_ms = self.batch_window_ms
    self.seed = int(seed)
    self.exclusive_compute = bool(exclusive_compute)

    self._lock = _FifoLock()
    self._proc_lock = _FifoLock()
    self._legacy_backend = False
    self._action_dim: int = 7
    self._gr00t_version = "unknown"
    self._torch_version = "unknown"
    self._compiled_runners: Dict[int, Tuple[Any, Any, Any, Any]] = {}
    self._batcher: Optional[DynamicMicroBatcher[Any, np.ndarray]] = None

    self._policy = self._load_policy()
    if self.exclusive_compute and self.enable_batching and self.micro_batch_size > 1:
      max_b = max(self.batch_buckets) if (self.torch_compile and self.batch_buckets) else self.micro_batch_size
      self._batcher = DynamicMicroBatcher(
          self._infer_batch_dispatch,
          max_batch_size=max_b,
          batch_window_ms=self.batch_window_ms,
      )
    if warm_start:
      self._warm_start()

  def _load_policy(self) -> Any:
    try:
      import torch
    except ImportError as exc:
      raise ImportError("torch is not installed.\n" + GR00T_INSTALL_HINT) from exc

    self._torch_version = str(getattr(torch, "__version__", "unknown"))
    self.device = resolve_torch_device(self.device)
    is_cuda = str(self.device).startswith("cuda")
    if is_cuda:
      torch.backends.cuda.matmul.allow_tf32 = False
      torch.backends.cudnn.allow_tf32 = False
    elif str(self.device) == "tpu" and hasattr(torch, "set_num_threads"):
      torch.set_num_threads(1)

    # Support legacy unit-test mock interface when `gr00t.model.policy` is injected in sys.modules.
    if "gr00t.model.policy" in sys.modules:
      from gr00t.experiment.data_config import DATA_CONFIG_MAP  # type: ignore[import-not-found]
      from gr00t.model.policy import Gr00tPolicy as LegacyGr00tPolicy  # type: ignore[import-not-found]

      self._legacy_backend = True
      cfg = DATA_CONFIG_MAP.get(self.data_config)
      pol = LegacyGr00tPolicy(
          model_path=self.model_path,
          modality_config=cfg.modality_config() if cfg is not None else None,
          modality_transform=cfg.transform() if cfg is not None else None,
          embodiment_tag=self.embodiment_tag_str,
          denoising_steps=self.denoising_steps,
          device=self.device,
      )
      if self.torch_compile and hasattr(pol, "model"):
        pol.model = maybe_compile_torch_module(pol.model, device=self.device, enabled=True)
      return pol

    try:
      from gr00t.data.embodiment_tags import EmbodimentTag
      from gr00t.data.types import MessageType
      from gr00t.policy.gr00t_policy import Gr00tPolicy
      from transformers.feature_extraction_utils import BatchFeature
    except ImportError as exc:
      raise ImportError("isaac-gr00t (`gr00t`) is not installed.\n" + GR00T_INSTALL_HINT) from exc

    _ = MessageType
    _ = BatchFeature
    try:
      import importlib.metadata as _md

      self._gr00t_version = _md.version("gr00t")
    except Exception:
      self._gr00t_version = "0.1.0"

    _install_gr00t_v17_backbone_compat()
    tag_map = {t.value: t for t in EmbodimentTag}
    emb_tag = (
        tag_map.get(self.embodiment_tag_str)
        or EmbodimentTag.__members__.get(self.embodiment_tag_str.upper())
        or EmbodimentTag("libero_sim")
    )

    resolved_model_path = self.model_path
    if not os.path.isdir(resolved_model_path) and resolved_model_path in (
        CANONICAL_GR00T_N17_REPO,
        f"{CANONICAL_GR00T_N17_REPO}/{CANONICAL_GR00T_N17_SUBDIR}",
    ):
      from huggingface_hub import snapshot_download

      snap_dir = snapshot_download(
          repo_id=CANONICAL_GR00T_N17_REPO,
          revision=self.revision,
          allow_patterns=[f"{CANONICAL_GR00T_N17_SUBDIR}/*"],
      )
      resolved_model_path = os.path.join(snap_dir, CANONICAL_GR00T_N17_SUBDIR)

    policy = Gr00tPolicy(
        embodiment_tag=emb_tag,
        model_path=resolved_model_path,
        device=self.device,
        strict=True,
    )
    policy.model.action_head.num_inference_timesteps = int(self.denoising_steps)

    # Eliminate 24 `cu_seqlens.tolist()` host syncs per step on both CUDA and TPU.
    _install_dense_vision_attention(policy.model)
    if str(self.device) == "tpu":
      _install_tpu_safe_action_attention(policy.model)

    if self.torch_compile:
      self._build_compiled_buckets(policy)
    return policy

  def _build_compiled_buckets(self, policy: Any) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = self.device
    is_tpu = str(dev) == "tpu"
    if not is_tpu:
      torch._dynamo.config.cache_size_limit = max(64, int(torch._dynamo.config.cache_size_limit))

    qwen_vlm = policy.model.backbone.model
    visual = qwen_vlm.model.visual
    llm = qwen_vlm.model.language_model
    ah = policy.model.action_head
    image_token_id = int(qwen_vlm.config.image_token_id)

    class _StaticVisualBlocksWrapper(nn.Module):
      def __init__(self, vis_mod: Any, grid_thw_cpu: torch.Tensor) -> None:
        super().__init__()
        self.visual = vis_mod
        with torch.inference_mode():
          grid_dev = grid_thw_cpu.to(dev)
          pos_embeds = vis_mod.fast_pos_embed_interpolate(grid_dev).to(vis_mod.dtype)
          rotary_pos_emb = vis_mod.rot_pos_emb(grid_dev).reshape(pos_embeds.shape[0], -1)
          emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
          cos = emb.cos().to(vis_mod.dtype)
          sin = emb.sin().to(vis_mod.dtype)
          cu_seqlens = torch.repeat_interleave(
              grid_dev[:, 1] * grid_dev[:, 2], grid_dev[:, 0]
          ).cumsum(dim=0, dtype=torch.int32)
          cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self.pos_embeds = pos_embeds
        self.cos = cos
        self.sin = sin
        self.cu_seqlens = cu_seqlens

      def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        hs = self.visual.patch_embed(pixel_values) + self.pos_embeds
        pe = (self.cos, self.sin)
        ds_list: List[torch.Tensor] = []
        for layer_num, blk in enumerate(self.visual.blocks):
          hs = blk(hs, cu_seqlens=self.cu_seqlens, position_embeddings=pe)
          if layer_num in self.visual.deepstack_visual_indexes:
            idx = self.visual.deepstack_visual_indexes.index(layer_num)
            ds_list.append(self.visual.deepstack_merger_list[idx](hs))
        hs = self.visual.merger(hs)
        return (hs, *ds_list)

    class _DenseLLMActionWrapper(nn.Module):
      def __init__(self, llm_mod: Any, ah_mod: Any, img_tok_id: int) -> None:
        super().__init__()
        self.llm = llm_mod
        self.ah = ah_mod
        self.image_token_id = img_tok_id

      def forward(
          self,
          input_ids: torch.Tensor,
          pos_ids: torch.Tensor,
          attention_mask_4d: torch.Tensor,
          backbone_attn_mask_2d: torch.Tensor,
          vis_idx: torch.Tensor,
          img_embeds: torch.Tensor,
          ds0: torch.Tensor,
          ds1: torch.Tensor,
          ds2: torch.Tensor,
          state: torch.Tensor,
          emb_id: torch.Tensor,
          init_actions: torch.Tensor,
          timesteps_all: torch.Tensor,
          pos_ids_act: torch.Tensor,
      ) -> torch.Tensor:
        bsz, seq = input_ids.shape
        hdim = self.llm.config.hidden_size
        flat_vis = vis_idx.reshape(-1)
        flat_embeds = self.llm.embed_tokens(input_ids.reshape(bsz * seq)).index_copy(
            0, flat_vis, img_embeds
        )
        hs = flat_embeds.view(bsz, seq, hdim)
        pe = self.llm.rotary_emb(hs, pos_ids)
        ds_tensors = (ds0, ds1, ds2)
        for layer_idx, decoder_layer in enumerate(self.llm.layers):
          hs = decoder_layer(
              hs,
              attention_mask=attention_mask_4d,
              position_ids=pos_ids[0],
              position_embeddings=pe,
              use_cache=False,
          )
          if layer_idx < 3:
            hs = (
                hs.reshape(bsz * seq, hdim)
                .index_add(0, flat_vis, ds_tensors[layer_idx])
                .view(bsz, seq, hdim)
            )
        image_mask = input_ids == self.image_token_id
        ah_m = self.ah
        vl_embs = ah_m.vl_self_attention(ah_m.vlln(hs))
        sa_embs = ah_m.state_encoder(state.view(bsz, 1, -1), emb_id)
        actions = init_actions
        num_steps = ah_m.num_inference_timesteps
        dt = 1.0 / num_steps
        pos_emb_act = (
            ah_m.position_embedding(pos_ids_act).unsqueeze(0) if ah_m.config.add_pos_embed else None
        )
        for t_step in range(num_steps):
          tt = timesteps_all[t_step]
          af = ah_m.action_encoder(actions, tt, emb_id)
          if pos_emb_act is not None:
            af = af + pos_emb_act
          sa_feat = torch.cat((sa_embs, af), dim=1)
          if ah_m.config.use_alternate_vl_dit:
            mo = ah_m.model(
                hidden_states=sa_feat,
                encoder_hidden_states=vl_embs,
                timestep=tt,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attn_mask_2d,
                hidden_attention_mask=None,
            )
          else:
            mo = ah_m.model(
                hidden_states=sa_feat,
                encoder_hidden_states=vl_embs,
                timestep=tt,
                encoder_attention_mask=(
                    backbone_attn_mask_2d if ah_m.config.strict_vl_padding_mask else None
                ),
                hidden_attention_mask=None,
            )
          actions = actions + dt * ah_m.action_decoder(mo, emb_id)[:, -ah_m.action_horizon :]
        return actions

    dummy_obs = Observation(
        session_id="init-shape",
        step=0,
        images={
            name: np.zeros((self._obs_image_hw[0], self._obs_image_hw[1], 3), dtype=np.uint8)
            for name in self.image_key_map
        },
        proprio=np.zeros((8,), dtype=np.float32),
        instruction="warm start",
    )
    sample0 = self._prepare_single_sample(dummy_obs, policy=policy)
    vlm0 = policy.collate_fn.processor(
        text=[sample0.vlm_text],
        images=list(sample0.vlm_images),
        return_tensors="pt",
        padding="max_length",
        max_length=self.max_token_length,
        truncation=True,
    )
    thw_1 = vlm0["image_grid_thw"]  # [num_cams, 3]

    buckets = self.batch_buckets if self.enable_batching else (1,)
    pos_ids_act = torch.arange(ah.config.action_horizon, dtype=torch.long, device=dev)
    if is_tpu:
      lw_base = _DenseLLMActionWrapper(llm, ah, image_token_id).eval()
      lw_compiled: Any = torch.compile(lw_base, backend="tpu")
    else:
      # On CUDA TorchInductor, unrolling 28 LLM layers + 16 * 32 = 512 DiT blocks into a single
      # monolithic FX graph compiles ~540 blocks per static batch bucket (~15 min/bucket).
      # Compiling the 28-layer VLM prefix once and the 1-step DiT denoiser once per bucket
      # produces identical fused Triton kernels per step while shrinking compile time ~20x.
      class _CudaPrefixWrapper(nn.Module):
        def __init__(self, llm_mod: Any, ah_mod: Any, img_tok_id: int) -> None:
          super().__init__()
          self.llm = llm_mod
          self.ah = ah_mod
          self.image_token_id = img_tok_id

        def forward(
            self,
            input_ids: torch.Tensor,
            pos_ids: torch.Tensor,
            attention_mask_4d: torch.Tensor,
            vis_idx: torch.Tensor,
            img_embeds: torch.Tensor,
            ds0: torch.Tensor,
            ds1: torch.Tensor,
            ds2: torch.Tensor,
            state: torch.Tensor,
            emb_id: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
          bsz, seq = input_ids.shape
          hdim = self.llm.config.hidden_size
          flat_vis = vis_idx.reshape(-1)
          flat_embeds = self.llm.embed_tokens(input_ids.reshape(bsz * seq)).index_copy(
              0, flat_vis, img_embeds
          )
          hs = flat_embeds.view(bsz, seq, hdim)
          pe = self.llm.rotary_emb(hs, pos_ids)
          ds_tensors = (ds0, ds1, ds2)
          for layer_idx, decoder_layer in enumerate(self.llm.layers):
            hs = decoder_layer(
                hs,
                attention_mask=attention_mask_4d,
                position_ids=pos_ids[0],
                position_embeddings=pe,
                use_cache=False,
            )
            if layer_idx < 3:
              hs = (
                  hs.reshape(bsz * seq, hdim)
                  .index_add(0, flat_vis, ds_tensors[layer_idx])
                  .view(bsz, seq, hdim)
              )
          image_mask = input_ids == self.image_token_id
          vl_embs = self.ah.vl_self_attention(self.ah.vlln(hs))
          sa_embs = self.ah.state_encoder(state.view(bsz, 1, -1), emb_id)
          return vl_embs, sa_embs, image_mask

      class _CudaSingleDiTStepWrapper(nn.Module):
        def __init__(self, ah_mod: Any) -> None:
          super().__init__()
          self.ah = ah_mod

        def forward(
            self,
            actions: torch.Tensor,
            tt: torch.Tensor,
            emb_id: torch.Tensor,
            sa_embs: torch.Tensor,
            vl_embs: torch.Tensor,
            image_mask: torch.Tensor,
            backbone_attn_mask_2d: torch.Tensor,
            pos_emb_act: Optional[torch.Tensor],
            dt: float,
        ) -> torch.Tensor:
          ah_m = self.ah
          af = ah_m.action_encoder(actions, tt, emb_id)
          if pos_emb_act is not None:
            af = af + pos_emb_act
          sa_feat = torch.cat((sa_embs, af), dim=1)
          if ah_m.config.use_alternate_vl_dit:
            mo = ah_m.model(
                hidden_states=sa_feat,
                encoder_hidden_states=vl_embs,
                timestep=tt,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attn_mask_2d,
                hidden_attention_mask=None,
            )
          else:
            mo = ah_m.model(
                hidden_states=sa_feat,
                encoder_hidden_states=vl_embs,
                timestep=tt,
                encoder_attention_mask=(
                    backbone_attn_mask_2d if ah_m.config.strict_vl_padding_mask else None
                ),
                hidden_attention_mask=None,
            )
          return actions + dt * ah_m.action_decoder(mo, emb_id)[:, -ah_m.action_horizon :]

      prefix_compiled = torch.compile(
          _CudaPrefixWrapper(llm, ah, image_token_id).eval(), mode="default", dynamic=False
      )
      step_compiled = torch.compile(
          _CudaSingleDiTStepWrapper(ah).eval(), mode="default", dynamic=False
      )

      class _CudaLLMActionLoop(nn.Module):
        def __init__(self, pref_fn: Any, step_fn: Any, ah_mod: Any) -> None:
          super().__init__()
          self.pref_fn = pref_fn
          self.step_fn = step_fn
          self.ah = ah_mod

        def forward(
            self,
            input_ids: torch.Tensor,
            pos_ids: torch.Tensor,
            attention_mask_4d: torch.Tensor,
            backbone_attn_mask_2d: torch.Tensor,
            vis_idx: torch.Tensor,
            img_embeds: torch.Tensor,
            ds0: torch.Tensor,
            ds1: torch.Tensor,
            ds2: torch.Tensor,
            state: torch.Tensor,
            emb_id: torch.Tensor,
            init_actions: torch.Tensor,
            timesteps_all: torch.Tensor,
            pos_ids_act_in: torch.Tensor,
        ) -> torch.Tensor:
          vl_embs, sa_embs, image_mask = self.pref_fn(
              input_ids, pos_ids, attention_mask_4d, vis_idx, img_embeds, ds0, ds1, ds2, state, emb_id
          )
          ah_m = self.ah
          actions = init_actions
          num_steps = ah_m.num_inference_timesteps
          dt = 1.0 / num_steps
          pos_emb_act = (
              ah_m.position_embedding(pos_ids_act_in).unsqueeze(0) if ah_m.config.add_pos_embed else None
          )
          for t_step in range(num_steps):
            actions = self.step_fn(
                actions,
                timesteps_all[t_step],
                emb_id,
                sa_embs,
                vl_embs,
                image_mask,
                backbone_attn_mask_2d,
                pos_emb_act,
                dt,
            )
          return actions

      lw_compiled = _CudaLLMActionLoop(prefix_compiled, step_compiled, ah).eval()

    for b in buckets:
      thw_b = thw_1.repeat(int(b), 1)
      vw_b = _StaticVisualBlocksWrapper(visual, thw_b).eval()
      if is_tpu:
        vw_b_compiled = torch.compile(vw_b, backend="tpu")
      else:
        vw_b_compiled = torch.compile(vw_b, mode="default", dynamic=False)
      tt_b = torch.stack([
          torch.full(
              (int(b),),
              int((s / float(ah.num_inference_timesteps)) * ah.num_timestep_buckets),
              dtype=torch.long,
          )
          for s in range(ah.num_inference_timesteps)
      ]).to(dev)
      self._compiled_runners[int(b)] = (vw_b_compiled, lw_compiled, tt_b, pos_ids_act)

  def _sync_device(self) -> None:
    synchronize_torch_device(getattr(self, "device", None))

  def _warm_start(self) -> None:
    import gc

    obs = Observation(
        session_id="warm-start",
        step=0,
        images={
            name: np.zeros((self._obs_image_hw[0], self._obs_image_hw[1], 3), dtype=np.uint8)
            for name in self.image_key_map
        },
        proprio=np.zeros((8,), dtype=np.float32),
        instruction="warm start",
    )
    if not self._legacy_backend and self._compiled_runners:
      s0 = self._prepare_single_sample(obs)
      buckets = self.batch_buckets if self.enable_batching else (1,)
      for b in buckets:
        if b in self._compiled_runners:
          self._execute_batch([s0], b)
    elif not self._legacy_backend and self.enable_batching and self.batch_buckets:
      s0 = self._prepare_single_sample(obs)
      for b in self.batch_buckets:
        self._execute_batch([s0] * int(b), int(b))
    else:
      self.predict(obs)
    # Move permanent model/tokenizer/processor graphs out of Python GC Generation 2
    # so periodic Gen-2 sweeps never pause concurrent gRPC threads (~200 ms) mid-run.
    gc.collect()
    if hasattr(gc, "freeze"):
      gc.freeze()
    if str(self.device) == "tpu":
      gc.set_threshold(700, 50000, 50000)

  def capabilities(self) -> Caps:
    if self._legacy_backend:
      mode_suffix = "+compile" if getattr(self, "torch_compile", False) else ""
    else:
      mode_suffix = "+compile" if getattr(self, "torch_compile", False) else "-eager"
      if getattr(self, "enable_batching", False):
        max_b = max(self.batch_buckets) if self.batch_buckets else self.micro_batch_size
        mode_suffix += f"+microbatch{max_b}"
    return Caps(
        track="VLA_TRACK",
        horizon=int(self._horizon),
        dt_ms=self.dt_ms,
        modalities=("rgb", "proprio", "language"),
        precision=self.precision_tag,
        model_id="gr00t-n1.7-3b",
        server_impl=(
            f"isaac-gr00t-{self._gr00t_version}+torch-{self._torch_version}-{getattr(self, 'device', 'unknown')}"
            f"{mode_suffix}"
        ),
        action_dim=int(self._action_dim),
        proprio_dim=8,
        cameras=tuple(self.image_key_map.keys()),
        image_height=self._obs_image_hw[0],
        image_width=self._obs_image_hw[1],
        execute_horizon=self.execute_horizon,
    )

  def reset(self, session_id: str) -> None:
    if self._batcher is not None:
      self._batcher.register_session(session_id)
    return None

  def _resolve_camera_image(self, obs: Observation, cam_name: str, gr00t_vid_key: str, idx: int) -> np.ndarray:
    if cam_name in obs.images:
      img = obs.images[cam_name]
    elif gr00t_vid_key in obs.images:
      img = obs.images[gr00t_vid_key]
    else:
      ordered = list(obs.images.values())
      if idx < len(ordered):
        img = ordered[idx]
      elif ordered:
        img = ordered[0]
      else:
        raise ValueError(f"Observation missing camera {cam_name!r} (got {sorted(obs.images)}).")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
      arr = np.transpose(arr, (1, 2, 0))
    return np.ascontiguousarray(arr)

  def _prepare_single_sample(self, obs: Observation, policy: Any = None) -> _PreparedGr00tSample:
    import torch
    from gr00t.data.types import MessageType, VLAStepData

    pol = policy if policy is not None else self._policy
    videos: Dict[str, np.ndarray] = {}
    for idx, (cam_name, vid_key) in enumerate(self.image_key_map.items()):
      videos[vid_key] = self._resolve_camera_image(obs, cam_name, vid_key, idx)[None, ...]

    p = np.asarray(obs.proprio, dtype=np.float32).reshape(-1)
    if p.size < 8:
      p = np.pad(p, (0, 8 - p.size))
    raw_states: Dict[str, np.ndarray] = {
        "x": p[0:1][None, :],
        "y": p[1:2][None, :],
        "z": p[2:3][None, :],
        "roll": p[3:4][None, :],
        "pitch": p[4:5][None, :],
        "yaw": p[5:6][None, :],
        "gripper": p[6:8][None, :],
    }

    vsd = VLAStepData(
        images=videos,
        states=raw_states,
        actions={},
        text=obs.instruction or "",
        embodiment=pol.embodiment_tag,
    )
    proc_elem = pol.processor([{"type": MessageType.EPISODE_STEP.value, "content": vsd}])
    vlm_c = proc_elem["vlm_content"]

    ah = pol.model.action_head
    gen = torch.Generator(device="cpu").manual_seed(_step_seed(self.seed, obs.step))
    init_noise = torch.randn(
        (1, ah.config.action_horizon, ah.action_dim), generator=gen, dtype=torch.bfloat16
    )

    state_tensor = torch.from_numpy(np.asarray(proc_elem["state"])[None, ...]).to(dtype=torch.bfloat16)
    emb_id = torch.from_numpy(np.asarray(proc_elem["embodiment_id"])[None, ...]).reshape(1).to(dtype=torch.long)

    return _PreparedGr00tSample(
        vlm_text=str(vlm_c["text"]),
        vlm_images=list(vlm_c["images"]),
        state_tensor=state_tensor,
        embodiment_id=emb_id,
        init_noise=init_noise,
        raw_states=raw_states,
    )

  def _collate_vlm_batch(
      self, samples: Sequence[_PreparedGr00tSample]
  ) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

    pol = self._policy
    texts = [s.vlm_text for s in samples]
    images = [im for s in samples for im in s.vlm_images]
    with self._proc_lock:
      vlm_inputs = pol.collate_fn.processor(
          text=texts,
          images=images,
          return_tensors="pt",
          padding="max_length",
          max_length=self.max_token_length,
          truncation=True,
      )
    iids = vlm_inputs["input_ids"]
    am = vlm_inputs["attention_mask"]
    pv = vlm_inputs["pixel_values"].to(dtype=torch.bfloat16)
    thw = vlm_inputs["image_grid_thw"]

    qwen_vlm = pol.model.backbone.model
    llm_cfg = qwen_vlm.model.language_model.config
    bsz, seq = iids.shape
    pos_ids, _ = qwen_vlm.model.get_rope_index(input_ids=iids, image_grid_thw=thw, attention_mask=am)
    am4d = mq.create_causal_mask(
        config=llm_cfg,
        input_embeds=torch.empty((bsz, seq, llm_cfg.hidden_size), dtype=torch.bfloat16),
        attention_mask=am,
        cache_position=torch.arange(seq, dtype=torch.long),
        past_key_values=None,
        position_ids=pos_ids,
    )
    vis_idx = torch.nonzero(
        iids.reshape(-1) == int(qwen_vlm.config.image_token_id), as_tuple=False
    ).squeeze(-1).view(bsz, -1)
    return iids, am, pv, thw, pos_ids, am4d, vis_idx

  def _decode_physical_actions(
      self, norm_actions_np: np.ndarray, samples: Sequence[_PreparedGr00tSample]
  ) -> List[np.ndarray]:
    pol = self._policy
    n = len(samples)
    norm_slice = norm_actions_np[:n]
    batched_states = {
        k: np.stack([s.raw_states[k] for s in samples], axis=0)
        for k in pol.modality_configs["state"].modality_keys
    }
    decoded = pol.processor.decode_action(norm_slice, pol.embodiment_tag, batched_states)
    cat = np.concatenate(
        [decoded[k] for k in pol.modality_configs["action"].modality_keys], axis=-1
    ).astype(np.float32, copy=False)
    out: List[np.ndarray] = []
    for i in range(n):
      chunk = np.ascontiguousarray(cat[i, : self._horizon])
      out.append(chunk)
    return out

  def _execute_eager_batch(self, samples: Sequence[_PreparedGr00tSample]) -> List[np.ndarray]:
    import torch

    pol = self._policy
    bsz = len(samples)
    iids, am, pv, thw, _, _, _ = self._collate_vlm_batch(samples)
    inputs_dict = {
        "input_ids": iids,
        "attention_mask": am,
        "pixel_values": pv,
        "image_grid_thw": thw,
        "state": torch.cat([s.state_tensor for s in samples], dim=0),
        "embodiment_id": torch.cat([s.embodiment_id for s in samples], dim=0),
    }
    with torch.inference_mode():
      bb_in, act_in = pol.model.prepare_input(inputs_dict)
      bb_out = pol.model.backbone(bb_in)
      noise_dev = torch.cat([s.init_noise for s in samples], dim=0).to(
          device=self.device, dtype=pol.model.dtype
      )
      ah = pol.model.action_head
      vl_embs = ah.vl_self_attention(ah.vlln(bb_out["backbone_features"]))
      sa_embs = ah.state_encoder(act_in.state.view(bsz, 1, -1), act_in.embodiment_id)
      actions = noise_dev
      num_steps = ah.num_inference_timesteps
      dt = 1.0 / num_steps
      pos_emb_act = (
          ah.position_embedding(
              torch.arange(ah.config.action_horizon, dtype=torch.long, device=self.device)
          ).unsqueeze(0)
          if ah.config.add_pos_embed
          else None
      )
      for t_step in range(num_steps):
        t_disc = int((t_step / float(num_steps)) * ah.num_timestep_buckets)
        tt = torch.full((bsz,), t_disc, device=self.device, dtype=torch.long)
        af = ah.action_encoder(actions, tt, act_in.embodiment_id)
        if pos_emb_act is not None:
          af = af + pos_emb_act
        sa_feat = torch.cat((sa_embs, af), dim=1)
        if ah.config.use_alternate_vl_dit:
          mo = ah.model(
              hidden_states=sa_feat,
              encoder_hidden_states=vl_embs,
              timestep=tt,
              image_mask=bb_out["image_mask"],
              backbone_attention_mask=bb_out["backbone_attention_mask"],
              hidden_attention_mask=None,
          )
        else:
          mo = ah.model(
              hidden_states=sa_feat,
              encoder_hidden_states=vl_embs,
              timestep=tt,
              encoder_attention_mask=(
                  bb_out["backbone_attention_mask"] if ah.config.strict_vl_padding_mask else None
              ),
              hidden_attention_mask=None,
          )
        actions = actions + dt * ah.action_decoder(mo, act_in.embodiment_id)[:, -ah.action_horizon :]
      actions_f32 = actions.float()
      self._sync_device()
      norm_np = actions_f32.cpu().numpy()
    return self._decode_physical_actions(norm_np, samples)

  def _execute_batch(self, items: Sequence[_PreparedGr00tSample], bucket_b: int) -> List[np.ndarray]:
    import torch

    n_real = len(items)
    n_pad = bucket_b - n_real

    if bucket_b in self._compiled_runners:
      dev = self.device
      vw_c, lw_c, tt_b, pos_ids_act = self._compiled_runners[bucket_b]
      seq_len = self.max_token_length
      iids_r, am_r, pv_r, _, pos_r, am4d_r, vis_r = self._collate_vlm_batch(items)
      if n_pad > 0:
        iids_r = torch.cat([iids_r, iids_r[-1:].expand(n_pad, -1)], dim=0)
        am_r = torch.cat([am_r, am_r[-1:].expand(n_pad, -1)], dim=0)
        p_per_sample = pv_r.shape[0] // n_real
        pv_r = torch.cat([pv_r, pv_r[-p_per_sample:].repeat(n_pad, 1)], dim=0)
        pos_r = torch.cat([pos_r, pos_r[:, -1:, :].expand(-1, n_pad, -1)], dim=1)
        am4d_r = torch.cat([am4d_r, am4d_r[-1:].expand(n_pad, -1, -1, -1)], dim=0)
        base_local = vis_r[-1] - (n_real - 1) * seq_len
        pad_vis = torch.stack(
            [base_local + (n_real + k) * seq_len for k in range(n_pad)], dim=0
        )
        vis_r = torch.cat([vis_r, pad_vis], dim=0)

      padded = list(items) + [items[-1]] * n_pad
      iids_b = iids_r.to(dev)
      pos_b = pos_r.to(dev)
      am4d_b = am4d_r.to(dev)
      bb_mask_b = (am_r == 1).to(dev)
      vis_idx_b = vis_r.to(dev)
      pv_b = pv_r.to(dev)
      state_b = torch.cat([s.state_tensor for s in padded], dim=0).to(dev)
      emb_b = torch.cat([s.embodiment_id for s in padded], dim=0).to(dev)
      noise_b = torch.cat([s.init_noise for s in padded], dim=0).to(dev)

      with self._lock:
        with torch.inference_mode():
          ie, d0, d1, d2 = vw_c(pv_b)
          act_b = lw_c(
              iids_b,
              pos_b,
              am4d_b,
              bb_mask_b,
              vis_idx_b,
              ie,
              d0,
              d1,
              d2,
              state_b,
              emb_b,
              noise_b,
              tt_b,
              pos_ids_act,
          )
          act_f32 = act_b[:n_real].float()
          self._sync_device()
          norm_np = act_f32.cpu().numpy()
      return self._decode_physical_actions(norm_np, items)

    with self._lock:
      return self._execute_eager_batch(items)

  def _infer_batch_dispatch(self, items: Sequence[Any]) -> Sequence[np.ndarray]:
    if self._legacy_backend:
      with self._lock:
        outs = []
        for ex in items:
          raw = self._policy.get_action(ex)
          arr = np.asarray(raw["action"] if isinstance(raw, dict) and "action" in raw else raw, dtype=np.float32)
          if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
          elif arr.ndim == 1:
            arr = arr[None, :]
          outs.append(arr)
        self._sync_device()
        return outs

    n = len(items)
    if self._compiled_runners:
      bucket_b = max(self.batch_buckets)
      for b in self.batch_buckets:
        if b >= n:
          bucket_b = b
          break
      return self._execute_batch(items, bucket_b)
    return self._execute_batch(items, n)

  def _legacy_example(self, obs: Observation) -> Dict[str, Any]:
    example: Dict[str, Any] = {}
    for camera, key in self.image_key_map.items():
      if camera not in obs.images:
        raise ValueError(
            f"Observation is missing camera {camera!r} (expected cameras: {sorted(self.image_key_map)})."
        )
      example[key] = np.ascontiguousarray(obs.images[camera])[None, ...]
    example["state.proprio"] = np.asarray(obs.proprio, dtype=np.float32)[None, ...]
    example["annotation.human.action.task_description"] = [obs.instruction]
    return example

  def predict(self, obs: Observation) -> ActionChunk:
    if self._legacy_backend:
      ex = self._legacy_example(obs)
      if self._batcher is not None:
        actions = self._batcher.submit(ex, session_id=obs.session_id)
      else:
        actions = self._infer_batch_dispatch([ex])[0]
      self._horizon, self._action_dim = int(actions.shape[0]), int(actions.shape[1])
      return ActionChunk(actions=actions)

    sample = self._prepare_single_sample(obs)
    if self._batcher is not None:
      actions = self._batcher.submit(sample, session_id=obs.session_id)
    elif 1 in self._compiled_runners:
      actions = self._execute_batch([sample], 1)[0]
    elif self.exclusive_compute:
      with self._lock:
        actions = self._execute_eager_batch([sample])[0]
    else:
      actions = self._execute_eager_batch([sample])[0]
    self._action_dim = int(actions.shape[1])
    return ActionChunk(actions=actions)

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    raise NotImplementedError("GR00T-N1.7-3B is a Track 1 VLA policy; it does not implement Track 2 rollout.")

  def close(self) -> None:
    if self._batcher is not None:
      self._batcher.close()
      self._batcher = None
    self._compiled_runners.clear()
    self._policy = None


__all__ = ["DEFAULT_GR00T_IMAGE_KEY_MAP", "GR00T_INSTALL_HINT", "Gr00tN17Policy"]
