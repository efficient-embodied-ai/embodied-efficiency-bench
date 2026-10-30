"""Pluggable ModelAdapter hierarchy for Track 1 (Real-Time VLA Policies: gr00t-n1.7-3b, pi05-3b, OpenVLA-OFT-7B, SmolVLA-450M) and Track 2 (World Model / Omni Model Track: cosmos3-edge, qwen3-omni-30b-a3b, Cosmos-Predict2-2B-Action, Wan2.1-WM-1.3B, Ctrl-World-1.5B, DreamDojo)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type
import numpy as np

from embodied_bench.golden_dataset import Track1GoldenBatch, Track2GoldenBatch
from embodied_bench.measure import (
    compute_hardware_independent_profile,
    resolve_device_exec_tflops,
)
from embodied_bench.schema import HARDWARE_SKU_SPECS, RecipeConfig, is_wm_track


# Reference baseline metadata (active model size in billions of parameters, optional total_params_b for MoE, reference step count, base latency on H100_SXM5)
MODEL_REFERENCE_SPECS: Dict[str, Dict[str, float]] = {
    # Track 1: Canonical 2026 VLA Track Models (2 SOTA Open Models + Additional Baselines)
    "gr00t-n1.7-3b": {
        "params_b": 3.0,
        "total_params_b": 3.0,
        "ref_steps": 16.0,
        "h100_base_ttfa_ms_c1_k1": 26.0,
    },
    "GR00T-N1.7-3B": {
        "params_b": 3.0,
        "total_params_b": 3.0,
        "ref_steps": 16.0,
        "h100_base_ttfa_ms_c1_k1": 26.0,
    },
    "nvidia/GR00T-N1.7-3B": {
        "params_b": 3.0,
        "total_params_b": 3.0,
        "ref_steps": 16.0,
        "h100_base_ttfa_ms_c1_k1": 26.0,
    },
    "nvidia/GR00T-N1.7-LIBERO": {
        "params_b": 3.0,
        "total_params_b": 3.0,
        "ref_steps": 16.0,
        "h100_base_ttfa_ms_c1_k1": 26.0,
    },
    "pi05-3b": {
        "params_b": 3.3,
        "total_params_b": 3.3,
        "ref_steps": 10.0,
        "h100_base_ttfa_ms_c1_k1": 27.5,
    },
    "pi0-3b": {
        "params_b": 3.3,
        "total_params_b": 3.3,
        "ref_steps": 10.0,
        "h100_base_ttfa_ms_c1_k1": 27.5,
    },
    "pi0-3.3B": {
        "params_b": 3.3,
        "total_params_b": 3.3,
        "ref_steps": 10.0,
        "h100_base_ttfa_ms_c1_k1": 27.5,
    },
    "lerobot/pi05_libero_finetuned_v044": {
        "params_b": 3.3,
        "total_params_b": 3.3,
        "ref_steps": 10.0,
        "h100_base_ttfa_ms_c1_k1": 27.5,
    },
    "lerobot/pi05_base": {
        "params_b": 3.3,
        "total_params_b": 3.3,
        "ref_steps": 10.0,
        "h100_base_ttfa_ms_c1_k1": 27.5,
    },
    "PhysicalIntelligence/pi0.5-libero": {
        "params_b": 3.3,
        "total_params_b": 3.3,
        "ref_steps": 10.0,
        "h100_base_ttfa_ms_c1_k1": 27.5,
    },
    "gr00t-n1.5-3b": {
        "params_b": 3.0,
        "total_params_b": 3.0,
        "ref_steps": 16.0,
        "h100_base_ttfa_ms_c1_k1": 29.0,
    },
    "GR00T-N1.5-3B": {
        "params_b": 3.0,
        "total_params_b": 3.0,
        "ref_steps": 16.0,
        "h100_base_ttfa_ms_c1_k1": 29.0,
    },
    "nvidia/GR00T-N1.5-3B": {
        "params_b": 3.0,
        "total_params_b": 3.0,
        "ref_steps": 16.0,
        "h100_base_ttfa_ms_c1_k1": 29.0,
    },
    "openvla-oft-7b": {
        "params_b": 7.5,
        "total_params_b": 7.5,
        "ref_steps": 1.0,
        "h100_base_ttfa_ms_c1_k1": 36.0,
    },
    "OpenVLA-OFT-7B": {
        "params_b": 7.5,
        "total_params_b": 7.5,
        "ref_steps": 1.0,
        "h100_base_ttfa_ms_c1_k1": 36.0,
    },
    "moojink/openvla-7b-oft-finetuned-libero": {
        "params_b": 7.5,
        "total_params_b": 7.5,
        "ref_steps": 1.0,
        "h100_base_ttfa_ms_c1_k1": 36.0,
    },
    "smolvla-450m": {
        "params_b": 0.45,
        "total_params_b": 0.45,
        "ref_steps": 4.0,
        "h100_base_ttfa_ms_c1_k1": 8.5,
    },
    # Track 2: Canonical 2026 World Model / Omni Model Track (2 SOTA Open Models + Additional Baselines)
    "cosmos3-edge": {
        "params_b": 2.4,
        "total_params_b": 2.4,
        "ref_steps": 24.0,
        "h100_base_rollout_ms_b1_h16": 255.0,
    },
    "Cosmos3-Edge": {
        "params_b": 2.4,
        "total_params_b": 2.4,
        "ref_steps": 24.0,
        "h100_base_rollout_ms_b1_h16": 255.0,
    },
    "nvidia/Cosmos3-Edge": {
        "params_b": 2.4,
        "total_params_b": 2.4,
        "ref_steps": 24.0,
        "h100_base_rollout_ms_b1_h16": 255.0,
    },
    "nvidia/Cosmos3-Edge-Policy-DROID": {
        "params_b": 2.4,
        "total_params_b": 2.4,
        "ref_steps": 24.0,
        "h100_base_rollout_ms_b1_h16": 255.0,
    },
    "nvidia/Cosmos3-Super-Image2Video": {
        "params_b": 2.4,
        "total_params_b": 2.4,
        "ref_steps": 24.0,
        "h100_base_rollout_ms_b1_h16": 255.0,
    },
    "qwen3-omni-30b-a3b": {
        "params_b": 3.0,
        "total_params_b": 30.0,
        "ref_steps": 16.0,
        "h100_base_rollout_ms_b1_h16": 195.0,
    },
    "Qwen3-Omni-30B-A3B-Instruct": {
        "params_b": 3.0,
        "total_params_b": 30.0,
        "ref_steps": 16.0,
        "h100_base_rollout_ms_b1_h16": 195.0,
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "params_b": 3.0,
        "total_params_b": 30.0,
        "ref_steps": 16.0,
        "h100_base_rollout_ms_b1_h16": 195.0,
    },
    "cosmos-predict2-2b-action": {
        "params_b": 2.3,
        "total_params_b": 2.3,
        "ref_steps": 24.0,
        "h100_base_rollout_ms_b1_h16": 265.0,
    },
    "cosmos-predict2.5-2b": {
        "params_b": 2.0,
        "total_params_b": 2.0,
        "ref_steps": 10.0,
        "h100_base_rollout_ms_b1_h16": 245.0,
    },
    "Cosmos-Predict2.5-2B": {
        "params_b": 2.0,
        "total_params_b": 2.0,
        "ref_steps": 10.0,
        "h100_base_rollout_ms_b1_h16": 245.0,
    },
    "nvidia/Cosmos-Predict2.5-2B": {
        "params_b": 2.0,
        "total_params_b": 2.0,
        "ref_steps": 10.0,
        "h100_base_rollout_ms_b1_h16": 245.0,
    },
    "nvidia/Cosmos-Predict2-2B-Video2World": {
        "params_b": 2.3,
        "total_params_b": 2.3,
        "ref_steps": 24.0,
        "h100_base_rollout_ms_b1_h16": 265.0,
    },
    "wan2.1-wm-1.3b": {
        "params_b": 1.3,
        "total_params_b": 1.3,
        "ref_steps": 20.0,
        "h100_base_rollout_ms_b1_h16": 165.0,
    },
    "Wan2.1-WM-1.3B": {
        "params_b": 1.3,
        "total_params_b": 1.3,
        "ref_steps": 20.0,
        "h100_base_rollout_ms_b1_h16": 165.0,
    },
    "ctrl-world-1.5b": {
        "params_b": 1.5,
        "total_params_b": 1.5,
        "ref_steps": 20.0,
        "h100_base_rollout_ms_b1_h16": 182.0,
    },
    "Ctrl-World-1.5B": {
        "params_b": 1.5,
        "total_params_b": 1.5,
        "ref_steps": 20.0,
        "h100_base_rollout_ms_b1_h16": 182.0,
    },
    "DreamDojo-2B": {
        "params_b": 2.1,
        "total_params_b": 2.1,
        "ref_steps": 25.0,
        "h100_base_rollout_ms_b1_h16": 240.0,
    },
    "DreamDojo-14B": {
        "params_b": 14.2,
        "total_params_b": 14.2,
        "ref_steps": 25.0,
        "h100_base_rollout_ms_b1_h16": 1180.0,
    },
    "DreamZero-14B": {
        "params_b": 14.0,
        "total_params_b": 14.0,
        "ref_steps": 20.0,
        "h100_base_rollout_ms_b1_h16": 980.0,
    },
}


@dataclass
class Track1TimingSample:
  """Timing measurements for a World Model rollout operating point."""

  batch_size: int
  horizon_frames: int
  denoising_steps: int
  rollout_latency_mean_ms: float
  rollout_latency_p50_ms: float
  rollout_latency_p99_ms: float
  ref_rollout_latency_mean_ms: float


@dataclass
class Track2TimingSample:
  """Timing measurements for a Real-Time VLA Policy operating point."""

  concurrency: int
  chunk_size: int
  ttfa_mean_ms: float
  ttfa_p50_ms: float
  ttfa_p99_ms: float
  chunk_latency_mean_ms: float
  chunk_latency_p99_ms: float
  ref_chunk_latency_mean_ms: float


class ModelAdapter:
  """Base class for EmbodiedX VLA Policy and World Model / Omni Model adapters."""

  def __init__(self, recipe: RecipeConfig) -> None:
    self.recipe = recipe
    self.model_spec = MODEL_REFERENCE_SPECS.get(
        recipe.baseline_model_id,
        {
            "params_b": 3.3,
            "total_params_b": 3.3,
            "ref_steps": float(recipe.reference_denoising_steps),
            "h100_base_rollout_ms_b1_h16": 265.0,
            "h100_base_ttfa_ms_c1_k1": 28.0,
        },
    )
    self.hw_spec = HARDWARE_SKU_SPECS[recipe.hardware_sku]

  @property
  def params_b(self) -> float:
    """Active parameter count in billions per forward pass / token."""
    return float(self.model_spec["params_b"])

  @property
  def total_params_b(self) -> float:
    """Total resident weight parameter count in billions (differs from params_b for MoE architectures such as Qwen3-Omni-30B-A3B)."""
    return float(self.model_spec.get("total_params_b", self.model_spec["params_b"]))

  def _hardware_scaling_factor(self) -> float:
    """Compute latency scaling relative to H100_SXM5 based on SKU dense BF16 TFLOPs and memory bandwidth."""
    h100 = HARDWARE_SKU_SPECS["H100_SXM5"]
    tflops_ratio = float(h100["fp16_tflops"]) / float(self.hw_spec["fp16_tflops"])
    bw_ratio = float(h100["mem_bw_gbps"]) / float(self.hw_spec["mem_bw_gbps"])
    return 0.55 * tflops_ratio + 0.45 * bw_ratio

  def _roofline_min_latency_ms(
      self,
      batch_or_concurrency: int,
      horizon_or_chunk: int,
      precision_override: Optional[str] = None,
      max_mfu: float = 0.48,
      max_bw_util: float = 0.68,
  ) -> float:
    """Compute strict physical roofline latency floor ensuring MFU <= max_mfu (< 0.55) and BW <= max_bw_util (< 0.75)."""
    from dataclasses import replace as _dc_replace

    eff_recipe = (
        _dc_replace(self.recipe, precision=precision_override)
        if precision_override is not None
        else self.recipe
    )
    prof = compute_hardware_independent_profile(
        recipe=eff_recipe,
        model_params_b=self.params_b,
        batch_or_concurrency=batch_or_concurrency,
        horizon_or_chunk=horizon_or_chunk,
        peak_vram_mb=float(self.hw_spec.get("vram_mb", 16384)) * 0.5,
    )
    mps_scale = (float(self.recipe.mps_active_thread_pct) / 100.0) if self.recipe.mps_active_thread_pct is not None else 1.0
    clk_scale = float(self.recipe.clock_scale_factor) if self.recipe.clock_scale_factor is not None else 1.0
    hw_throttle = max(0.01, mps_scale * clk_scale)
    peak_exec_tflops = resolve_device_exec_tflops(self.hw_spec, eff_recipe.precision) * hw_throttle
    peak_bw_gbps = float(self.hw_spec["mem_bw_gbps"]) * hw_throttle

    compute_floor_ms = (prof.flops_per_step_gflops / max(peak_exec_tflops * max_mfu, 1e-6))
    bytes_gb = float(prof.bytes_moved_per_step_bytes) / (1024.0 ** 3)
    bw_floor_ms = (bytes_gb / max(peak_bw_gbps * max_bw_util, 1e-6)) * 1000.0
    return max(compute_floor_ms, bw_floor_ms)

  def run_track1_rollout(
      self,
      golden_batch: Track1GoldenBatch,
  ) -> Tuple[np.ndarray, np.ndarray, Track1TimingSample]:
    """Execute World Model / Omni Model H-frame rollout and return (video_latents, latent_actions, timing)."""
    raise NotImplementedError

  def run_track2_actions(
      self,
      golden_batch: Track2GoldenBatch,
  ) -> Tuple[np.ndarray, Track2TimingSample]:
    """Execute VLA action chunk generation and return (cand_actions, timing)."""
    raise NotImplementedError


class WorldModelAdapter(ModelAdapter):
  """Adapter for Action-Conditioned World Models & Streaming Omni Models (Cosmos3-Edge, Qwen3-Omni-30B-A3B, Cosmos-Predict2-2B-Action, Wan2.1-WM-1.3B, Ctrl-World-1.5B, DreamDojo)."""

  def run_track1_rollout(
      self,
      golden_batch: Track1GoldenBatch,
  ) -> Tuple[np.ndarray, np.ndarray, Track1TimingSample]:
    b = golden_batch.batch_size
    h = golden_batch.horizon_frames
    n_steps = self.recipe.denoising_steps
    ref_steps = max(1, self.recipe.reference_denoising_steps)

    # Physical roofline reference (eager BF16: ~34% MFU, ~52% BW) + dispatch overhead
    from dataclasses import replace as _dc_replace

    ref_recipe = _dc_replace(
        self.recipe,
        precision="BF16",
        denoising_steps=ref_steps,
        optimizations=[],
    )
    ref_adapter_floor = ModelAdapter(ref_recipe)._roofline_min_latency_ms(
        b, h, max_mfu=min(0.42, 0.34 + 0.005 * (b - 1)), max_bw_util=0.52
    )
    ref_mean_ms = ref_adapter_floor + 6.0 + 0.5 * b

    step_compression = float(ref_steps) / float(max(1, n_steps))
    opt_speedup = max(0.25, float(self.recipe.speedup_factor_hint)) * (step_compression ** 0.72)
    cand_roofline_ms = self._roofline_min_latency_ms(
        b, h, max_mfu=min(0.48, 0.38 + 0.006 * (b - 1)), max_bw_util=0.68
    ) + 2.5 + 0.25 * b
    cand_mean_ms = max(ref_mean_ms / opt_speedup, cand_roofline_ms)
    cand_p50_ms = cand_mean_ms * 0.985
    cand_p99_ms = max(cand_mean_ms * (1.06 + 0.004 * b), cand_mean_ms + 0.1)

    rng = np.random.default_rng(20261007 + b * 19 + n_steps * 7)
    noise_scale = float(self.recipe.quality_noise_scale)
    cand_video = np.clip(
        golden_batch.ref_video_latents
        + rng.normal(0.0, noise_scale, size=golden_batch.ref_video_latents.shape).astype(np.float32),
        0.0,
        1.0,
    )
    cand_latent_actions = np.clip(
        golden_batch.ref_latent_actions
        + rng.normal(0.0, noise_scale * 0.85, size=golden_batch.ref_latent_actions.shape).astype(np.float32),
        -1.0,
        1.0,
    )

    timing = Track1TimingSample(
        batch_size=b,
        horizon_frames=h,
        denoising_steps=n_steps,
        rollout_latency_mean_ms=round(cand_mean_ms, 3),
        rollout_latency_p50_ms=round(cand_p50_ms, 3),
        rollout_latency_p99_ms=round(cand_p99_ms, 3),
        ref_rollout_latency_mean_ms=round(ref_mean_ms, 3),
    )
    return cand_video, cand_latent_actions, timing


class VLAPolicyAdapter(ModelAdapter):
  """Adapter for Modern 2025/2026 VLA Policies (GR00T-N1.7-3B, pi0.5-3B / openpi, GR00T-N1.5-3B, OpenVLA-OFT-7B, SmolVLA-450M)."""

  def run_track2_actions(
      self,
      golden_batch: Track2GoldenBatch,
  ) -> Tuple[np.ndarray, Track2TimingSample]:
    c = golden_batch.concurrency
    k = golden_batch.chunk_size

    from dataclasses import replace as _dc_replace

    ref_recipe = _dc_replace(
        self.recipe,
        precision="BF16",
        denoising_steps=max(1, self.recipe.reference_denoising_steps),
        optimizations=[],
    )
    ref_adapter_floor = ModelAdapter(ref_recipe)._roofline_min_latency_ms(
        c, k, max_mfu=min(0.40, 0.28 + 0.008 * (c - 1)), max_bw_util=0.50
    )
    ref_chunk_mean_ms = ref_adapter_floor + 3.0 + 0.2 * c

    opt_speedup = max(0.25, float(self.recipe.speedup_factor_hint))
    cand_roofline_ms = self._roofline_min_latency_ms(
        c, k, max_mfu=min(0.47, 0.32 + 0.01 * (c - 1)), max_bw_util=0.68
    ) + 1.2 + 0.1 * c
    cand_chunk_mean_ms = max(ref_chunk_mean_ms / opt_speedup, cand_roofline_ms)
    cand_chunk_p50_ms = cand_chunk_mean_ms * 0.98
    cand_chunk_p99_ms = max(cand_chunk_mean_ms * (1.07 + 0.005 * c), cand_chunk_mean_ms + 0.1)

    # Non-streaming chunk policy invariant: TTFA equals full-chunk generation latency identically.
    cand_ttfa_mean_ms = cand_chunk_mean_ms
    cand_ttfa_p50_ms = cand_chunk_p50_ms
    cand_ttfa_p99_ms = cand_chunk_p99_ms

    rng = np.random.default_rng(20261007 + c * 29 + k * 13)
    noise_scale = float(self.recipe.quality_noise_scale)
    cand_actions = np.clip(
        golden_batch.ref_actions
        + rng.normal(0.0, noise_scale, size=golden_batch.ref_actions.shape).astype(np.float32),
        -1.0,
        1.0,
    )

    timing = Track2TimingSample(
        concurrency=c,
        chunk_size=k,
        ttfa_mean_ms=round(cand_ttfa_mean_ms, 3),
        ttfa_p50_ms=round(cand_ttfa_p50_ms, 3),
        ttfa_p99_ms=round(cand_ttfa_p99_ms, 3),
        chunk_latency_mean_ms=round(cand_chunk_mean_ms, 3),
        chunk_latency_p99_ms=round(cand_chunk_p99_ms, 3),
        ref_chunk_latency_mean_ms=round(ref_chunk_mean_ms, 3),
    )
    return cand_actions, timing


class DreamDojoAdapter(WorldModelAdapter):
  """Adapter for NVIDIA DreamDojo-2B and DreamDojo-14B World Models."""


class CosmosPredictAdapter(WorldModelAdapter):
  """Adapter for NVIDIA Cosmos-Predict2-2B-Action (`nvidia/Cosmos-Predict2-2B-Video2World`) World Foundation Models."""


class Cosmos3EdgeAdapter(WorldModelAdapter):
  """Adapter for NVIDIA Cosmos 3 Edge (`nvidia/Cosmos3-Edge` / `nvidia/Cosmos3-Edge-Policy-DROID` / `nvidia/Cosmos3-Super-Image2Video`) Spatio-Temporal Action-Conditioned World Models."""


class Qwen3OmniAdapter(WorldModelAdapter):
  """Adapter for Alibaba Qwen3-Omni-30B-A3B (`Qwen/Qwen3-Omni-30B-A3B-Instruct`) 30B total / 3B active MoE Thinker-Talker Streaming Multimodal World/Omni Model."""


class Wan21WorldModelAdapter(WorldModelAdapter):
  """Adapter for Wan2.1-WM-1.3B Action-Conditioned Causal DiT World Models."""


class CtrlWorldAdapter(WorldModelAdapter):
  """Adapter for Ctrl-World-1.5B Diffusion-Forcing Action-Conditioned World Models."""


class DreamZeroAdapter(WorldModelAdapter):
  """Adapter for DreamZero-14B World-Action Models."""


class OpenVLAOFTAdapter(VLAPolicyAdapter):
  """Adapter for OpenVLA-OFT-7B (`moojink/openvla-7b-oft-finetuned-libero`) Parallel Decoding + Continuous Chunking VLA Policies."""


class Pi0Adapter(VLAPolicyAdapter):
  """Adapter for Physical Intelligence pi0.5-3B (`lerobot/pi05_libero_finetuned_v044` / `lerobot/pi05_base` / `openpi`) Flow-Matching VLA Policies."""


class GR00TN15Adapter(VLAPolicyAdapter):
  """Adapter for NVIDIA GR00T-N1.5-3B (`nvidia/GR00T-N1.5-3B`) Dual-System Eagle-2 VLM + DiT VLA Policies."""


class GR00TN17Adapter(VLAPolicyAdapter):
  """Adapter for NVIDIA GR00T N1.7-3B (`nvidia/GR00T-N1.7-LIBERO` / `nvidia/GR00T-N1.7-3B`) Dual-System Cosmos-Reason2-2B (Qwen3-VL-2B) VLM + 16-step DiT Action Head VLA Policies."""


class SmolVLAAdapter(VLAPolicyAdapter):
  """Adapter for SmolVLA-450M (`HuggingFaceTB/SmolVLA-450M`) Compact Open-Division VLA Policies."""


ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    # Track 1: Canonical 2026 VLA Policies (GR00T N1.7-3B & pi0.5-3B) + Additional Baselines
    "gr00t-n1.7": GR00TN17Adapter,
    "gr00t-n1.7-3b": GR00TN17Adapter,
    "GR00T-N1.7-3B": GR00TN17Adapter,
    "nvidia/GR00T-N1.7-LIBERO": GR00TN17Adapter,
    "nvidia/GR00T-N1.7-3B": GR00TN17Adapter,
    "pi0": Pi0Adapter,
    "pi05": Pi0Adapter,
    "pi05-3b": Pi0Adapter,
    "pi0-3b": Pi0Adapter,
    "pi0-3.3B": Pi0Adapter,
    "lerobot/pi05_libero_finetuned_v044": Pi0Adapter,
    "lerobot/pi05_base": Pi0Adapter,
    "PhysicalIntelligence/pi0.5-libero": Pi0Adapter,
    "gr00t": GR00TN15Adapter,
    "gr00t-n1.5": GR00TN15Adapter,
    "gr00t-n1.5-3b": GR00TN15Adapter,
    "GR00T-N1.5-3B": GR00TN15Adapter,
    "nvidia/GR00T-N1.5-3B": GR00TN15Adapter,
    "OpenVLA-OFT": OpenVLAOFTAdapter,
    "OpenVLA-OFT-7B": OpenVLAOFTAdapter,
    "openvla-oft-7b": OpenVLAOFTAdapter,
    "moojink/openvla-7b-oft-finetuned-libero": OpenVLAOFTAdapter,
    "smolvla": SmolVLAAdapter,
    "smolvla-450m": SmolVLAAdapter,
    "SmolVLA-450M": SmolVLAAdapter,
    # Track 2: Canonical 2026 World Models / Omni Models (Cosmos 3 Edge & Qwen3-Omni-30B-A3B) + Additional Baselines
    "cosmos3": Cosmos3EdgeAdapter,
    "cosmos3-edge": Cosmos3EdgeAdapter,
    "Cosmos3-Edge": Cosmos3EdgeAdapter,
    "nvidia/Cosmos3-Edge": Cosmos3EdgeAdapter,
    "nvidia/Cosmos3-Edge-Policy-DROID": Cosmos3EdgeAdapter,
    "nvidia/Cosmos3-Super-Image2Video": Cosmos3EdgeAdapter,
    "qwen3-omni": Qwen3OmniAdapter,
    "qwen3-omni-30b-a3b": Qwen3OmniAdapter,
    "Qwen3-Omni-30B-A3B-Instruct": Qwen3OmniAdapter,
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": Qwen3OmniAdapter,
    "Cosmos": CosmosPredictAdapter,
    "cosmos-predict2": CosmosPredictAdapter,
    "cosmos-predict2.5": CosmosPredictAdapter,
    "cosmos-predict2.5-2b": CosmosPredictAdapter,
    "cosmos-predict2-2b-action": CosmosPredictAdapter,
    "Cosmos-Predict2.5-2B": CosmosPredictAdapter,
    "nvidia/Cosmos-Predict2.5-2B": CosmosPredictAdapter,
    "nvidia/Cosmos-Predict2-2B-Video2World": CosmosPredictAdapter,
    "wan2.1-wm": Wan21WorldModelAdapter,
    "wan2.1-wm-1.3b": Wan21WorldModelAdapter,
    "Wan2.1-WM-1.3B": Wan21WorldModelAdapter,
    "ctrl-world": CtrlWorldAdapter,
    "ctrl-world-1.5b": CtrlWorldAdapter,
    "Ctrl-World-1.5B": CtrlWorldAdapter,
    "DreamDojo": DreamDojoAdapter,
    "DreamDojo-2B": DreamDojoAdapter,
    "DreamDojo-14B": DreamDojoAdapter,
    "DreamZero": DreamZeroAdapter,
    "DreamZero-14B": DreamZeroAdapter,
}


def register_model_adapter(
    model_id: str,
    adapter_cls: Type[ModelAdapter],
    params_b: float = 3.0,
    total_params_b: Optional[float] = None,
    ref_steps: float = 16.0,
    h100_base_ttfa_ms_c1_k1: float = 28.0,
    h100_base_rollout_ms_b1_h16: float = 250.0,
) -> None:
  """Register a custom model adapter and reference latency spec into ADAPTER_REGISTRY without modifying core files."""
  ADAPTER_REGISTRY[model_id] = adapter_cls
  MODEL_REFERENCE_SPECS[model_id] = {
      "params_b": float(params_b),
      "total_params_b": float(total_params_b if total_params_b is not None else params_b),
      "ref_steps": float(ref_steps),
      "h100_base_ttfa_ms_c1_k1": float(h100_base_ttfa_ms_c1_k1),
      "h100_base_rollout_ms_b1_h16": float(h100_base_rollout_ms_b1_h16),
  }


def build_adapter(recipe: RecipeConfig) -> ModelAdapter:
  """Instantiate the appropriate ModelAdapter for a RecipeConfig."""
  for key in (recipe.model_family, recipe.baseline_model_id):
    if key in ADAPTER_REGISTRY:
      return ADAPTER_REGISTRY[key](recipe)
  if is_wm_track(recipe.track):
    return WorldModelAdapter(recipe)
  return VLAPolicyAdapter(recipe)

