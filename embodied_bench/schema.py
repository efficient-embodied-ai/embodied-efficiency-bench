"""Core schema definitions, Two-Clock + Unified Scenario taxonomy, Modern 2025/2026 VLA & World Model reference checkpoints, Device-Agnostic Performance Cap Envelopes (`PerformanceCapEnvelope`), hardware lookup tables, state machine validation, and atomic I/O for EmbodiedX (`embodied-efficiency-bench` / EEB v1.0)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
  import yaml
except ImportError:  # pragma: no cover
  yaml = None  # type: ignore[assignment]


LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def now_pacific_iso() -> str:
  """Return current timestamp in America/Los_Angeles (PDT/PST) with explicit offset and zone name."""
  dt = datetime.now(LOCAL_TZ)
  tz_abbr = dt.tzname() or "PT"
  return f"{dt.isoformat(timespec='seconds')} ({tz_abbr})"


class BenchmarkClock(str, Enum):
  """Two-Clock EEB execution structure: CoRL 2026 Sprint and EEB v1 Living Benchmark."""

  ROUND0_ORIN_NX_SPRINT = "round0_orin_nx_sprint"
  CORL2026_SPRINT = "corl2026_sprint"
  EEB_V1 = "eeb_v1"


class ScenarioType(str, Enum):
  """Unified EEB Scenario Taxonomy across Acting VLA Policies (Track 1) and Interactive/Evaluative World Models (Track 2)."""

  P1_ONBOARD = "P1_ONBOARD"
  P2_FLEET = "P2_FLEET"
  P3_EDGE_CLOUD = "P3_EDGE_CLOUD"
  S1_INTERACTIVE_WM = "S1_INTERACTIVE_WM"
  S2_WM_EVALUATOR = "S2_WM_EVALUATOR"


class DivisionType(str, Enum):
  """MLPerf-style benchmark divisions separating systems/compiler optimization from algorithmic co-design."""

  CLOSED = "CLOSED"
  OPEN = "OPEN"


class PowerClass(str, Enum):
  """Edge power classes for P1_ONBOARD ConstantStream (C=1) evaluation."""

  E1 = "E1"        # <= 25W (e.g. Jetson Orin NX 16GB)
  E2 = "E2"        # <= 60W (e.g. Jetson AGX Orin 64GB)
  E3 = "E3"        # <= 130W (e.g. Jetson AGX Thor)
  E_OPEN = "E_OPEN"  # Open / workstation / cloud power envelope


class PerformanceCapId(str, Enum):
  """Hardware-agnostic single-device performance cap envelope identifiers (`num_devices == 1`)."""

  CAP_SINGLE_CHIP = "CAP_SINGLE_CHIP"
  SINGLE_CHIP_UNIFIED = "SINGLE_CHIP_UNIFIED"
  CAP_SINGLE_CHIP_32GB = "CAP_SINGLE_CHIP_32GB"
  SINGLE_CHIP_32GB = "SINGLE_CHIP_32GB"
  CAP_EDGE_100T = "CAP_EDGE_100T"
  CAP_WORKSTATION_350T = "CAP_WORKSTATION_350T"
  CAP_DATACENTER_1000T = "CAP_DATACENTER_1000T"
  CAP_UNCONSTRAINED = "CAP_UNCONSTRAINED"
  SINGLE_EDGE = "SINGLE-EDGE"
  SINGLE_ACCELERATOR_STD = "SINGLE-ACCELERATOR-STD"
  SINGLE_ACCELERATOR_MAX = "SINGLE-ACCELERATOR-MAX"


PERF_CAP_ALIASES: Dict[str, str] = {
    PerformanceCapId.CAP_SINGLE_CHIP.value: PerformanceCapId.CAP_SINGLE_CHIP.value,
    PerformanceCapId.SINGLE_CHIP_UNIFIED.value: PerformanceCapId.CAP_SINGLE_CHIP.value,
    "SINGLE-CHIP": PerformanceCapId.CAP_SINGLE_CHIP.value,
    "SINGLE_CHIP": PerformanceCapId.CAP_SINGLE_CHIP.value,
    PerformanceCapId.CAP_SINGLE_CHIP_32GB.value: PerformanceCapId.CAP_SINGLE_CHIP_32GB.value,
    PerformanceCapId.SINGLE_CHIP_32GB.value: PerformanceCapId.CAP_SINGLE_CHIP_32GB.value,
    PerformanceCapId.CAP_EDGE_100T.value: PerformanceCapId.CAP_EDGE_100T.value,
    PerformanceCapId.SINGLE_EDGE.value: PerformanceCapId.CAP_EDGE_100T.value,
    PerformanceCapId.CAP_WORKSTATION_350T.value: PerformanceCapId.CAP_WORKSTATION_350T.value,
    PerformanceCapId.SINGLE_ACCELERATOR_STD.value: PerformanceCapId.CAP_WORKSTATION_350T.value,
    PerformanceCapId.CAP_DATACENTER_1000T.value: PerformanceCapId.CAP_DATACENTER_1000T.value,
    PerformanceCapId.SINGLE_ACCELERATOR_MAX.value: PerformanceCapId.CAP_DATACENTER_1000T.value,
    PerformanceCapId.CAP_UNCONSTRAINED.value: PerformanceCapId.CAP_UNCONSTRAINED.value,
}


def resolve_perf_cap_id(envelope_id: str | PerformanceCapId) -> str:
  """Resolve canonical `CAP_*` or `SINGLE_*` alias to canonical `PerformanceCapId` string."""
  raw = envelope_id.value if isinstance(envelope_id, PerformanceCapId) else str(envelope_id)
  return PERF_CAP_ALIASES.get(raw, raw)


class PerfCapSatisfactionMode(str, Enum):
  """Three ways a benchmark submission can satisfy a PerformanceCapEnvelope."""

  NATIVE_QUALIFYING_DEVICE = "native_qualifying_device"
  HARDWARE_ENFORCED_SLICE_CLOCK_CAP = "hardware_enforced_slice_clock_cap"
  ROOFLINE_NORMALIZED_LATENCY = "roofline_normalized_latency"


@dataclass(frozen=True)
class PerformanceCapEnvelope:
  """Single-device (`num_devices == 1`) compute, bandwidth, and HBM/VRAM budget envelope (`PerformanceCapEnvelope`).

  Allows participants to run on any single GPU, Google Cloud TPU, AMD Instinct, NPU, or custom ASIC
  bounded by `(max_fp8_int8_tops, max_bf16_tflops, max_hbm_gb, max_bw_gbps)`, evaluated via:
    1. Native Qualifying Device (single-device hardware specs naturally within cap),
    2. Hardware-Enforced Slice / Clock Cap (MIG / CUDA_MPS_ACTIVE_THREAD_PERCENTAGE / TPU 1-chip + locked clocks + VRAM/HBM cap), or
    3. Roofline-Normalized Effective Latency (L_norm) + Iso-TOPS / Iso-Bandwidth efficiency.
  """

  envelope_id: str
  tier_alias: str
  max_fp16_tflops: float
  max_int8_fp8_tops: float
  max_mem_bw_gbps: float
  max_vram_gb: float
  typical_power_watts: float
  example_devices: Tuple[str, ...]
  description: str
  max_num_devices: int = 1

  @property
  def max_tflops(self) -> float:
    return self.max_fp16_tflops

  @property
  def max_bf16_tflops(self) -> float:
    return self.max_fp16_tflops

  @property
  def max_fp8_int8_tops(self) -> float:
    return self.max_int8_fp8_tops

  @property
  def max_hbm_gb(self) -> float:
    return self.max_vram_gb

  @property
  def max_bw_gbps(self) -> float:
    return self.max_mem_bw_gbps

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    d["max_tflops"] = self.max_tflops
    d["max_bf16_tflops"] = self.max_bf16_tflops
    d["max_fp8_int8_tops"] = self.max_fp8_int8_tops
    d["max_hbm_gb"] = self.max_hbm_gb
    d["max_bw_gbps"] = self.max_bw_gbps
    return d


PERFORMANCE_CAP_ENVELOPES: Dict[str, PerformanceCapEnvelope] = {
    PerformanceCapId.CAP_SINGLE_CHIP.value: PerformanceCapEnvelope(
        envelope_id=PerformanceCapId.CAP_SINGLE_CHIP.value,
        tier_alias=PerformanceCapId.SINGLE_CHIP_UNIFIED.value,
        max_fp16_tflops=1000.0,
        max_int8_fp8_tops=2000.0,
        max_mem_bw_gbps=2000.0,
        max_vram_gb=64.0,
        typical_power_watts=575.0,
        example_devices=(
            "1x Google Cloud TPU v6e-1 Trillium (32GB HBM2e: 918 TFLOPS BF16 / 1,836 TOPS FP8/INT8 dense, 1,600 GB/s, 170W, $1.375/hr)",
            "1x NVIDIA GeForce RTX 5090 32GB (209.5 TFLOPS BF16 FP32-accum / 419 TOPS FP8 dense, 32GB GDDR7, 1,792 GB/s, 575W, $0.95/hr)",
            "1x NVIDIA Jetson AGX Thor 64GB (259 TFLOPS BF16 / 517.5 TOPS FP8 dense, 64GB LPDDR5X, 273 GB/s, 130W, $0.85/hr)",
            "1x NVIDIA L40S 48GB (362.05 TFLOPS BF16 / 733 TOPS FP8 dense, 864 GB/s, 350W, $1.15/hr) / RTX 4090 24GB (82.6 TFLOPS BF16 FP32-accum / 330.3 TOPS FP8 dense, 1,008 GB/s, 450W, $0.69/hr)",
            "1x NVIDIA L4 24GB (121 TFLOPS BF16 / 242 TOPS FP8 dense, 24GB GDDR6, 300 GB/s, 72W, $0.71/hr)",
            "1x NVIDIA Jetson Orin NX 16GB (23 TFLOPS FP16 / 47 TOPS INT8 dense, 16GB LPDDR5, 102.4 GB/s, 25W, $0.25/hr)",
            "Any Single-Chip GPU/TPU/NPU/ASIC natively within <=1,000 dense BF16 TFLOPS / <=2,000 FP8/INT8 TOPS, <=64GB HBM/VRAM, <=2,000 GB/s",
        ),
        description=(
            "Practical Physical-AI Single-Chip Envelope (CAP_SINGLE_CHIP / SINGLE_CHIP_UNIFIED, num_devices==1, 100% Native Uncapped): "
            "Peak dense BF16/FP16 <= 1,000 TFLOPS (dense FP8/INT8 <= 2,000 TOPS), "
            "Native HBM/VRAM <= 64 GB, Peak Memory Bandwidth <= 2,000 GB/s (2.0 TB/s)."
        ),
    ),
    PerformanceCapId.CAP_SINGLE_CHIP_32GB.value: PerformanceCapEnvelope(
        envelope_id=PerformanceCapId.CAP_SINGLE_CHIP_32GB.value,
        tier_alias=PerformanceCapId.SINGLE_CHIP_32GB.value,
        max_fp16_tflops=1000.0,
        max_int8_fp8_tops=2000.0,
        max_mem_bw_gbps=2000.0,
        max_vram_gb=32.0,
        typical_power_watts=575.0,
        example_devices=(
            "1x Google Cloud TPU v6e-1 Trillium (32GB HBM2e: 918 TFLOPS BF16 / 1,836 TOPS FP8/INT8 dense, 1,600 GB/s)",
            "1x NVIDIA GeForce RTX 5090 (32GB GDDR7: 209.5 TFLOPS BF16 FP32-accum / 419 TOPS FP8 dense, 1,792 GB/s)",
            "1x NVIDIA RTX 4090 (24GB) / NVIDIA L4 (24GB, 72W) / Jetson Orin NX (16GB)",
        ),
        description=(
            "Strict 32GB Parity Sub-Cap of Physical-AI Single-Chip Tier (CAP_SINGLE_CHIP_32GB, num_devices==1): "
            "Peak dense BF16/FP16 <= 1,000 TFLOPS (dense FP8/INT8 <= 2,000 TOPS), Native/Active HBM/VRAM <= 32 GB, Bandwidth <= 2,000 GB/s."
        ),
    ),
    PerformanceCapId.CAP_EDGE_100T.value: PerformanceCapEnvelope(
        envelope_id=PerformanceCapId.CAP_EDGE_100T.value,
        tier_alias=PerformanceCapId.SINGLE_EDGE.value,
        max_fp16_tflops=100.0,
        max_int8_fp8_tops=200.0,
        max_mem_bw_gbps=250.0,
        max_vram_gb=24.0,
        typical_power_watts=60.0,
        example_devices=(
            "NVIDIA Jetson Orin NX 16GB",
            "NVIDIA Jetson AGX Orin (24GB VRAM cap)",
            "NVIDIA RTX 4060 Ti 16GB / RTX A2000 16GB",
            "Apple M3/M4 Pro (16-24GB unified memory cap)",
            "Hailo-10H / Edge NPU / MIG / MPS + Clock-Capped GPU",
        ),
        description=(
            "Single-Device Onboard Edge Sub-Profile (CAP_EDGE_100T / SINGLE-EDGE, num_devices==1): "
            "Peak INT8/FP8 <= 200 TOPS (BF16/FP16 <= 100 TFLOPS), "
            "Peak Memory Bandwidth <= 250 GB/s, Active HBM/VRAM Budget <= 24 GB."
        ),
    ),
    PerformanceCapId.CAP_WORKSTATION_350T.value: PerformanceCapEnvelope(
        envelope_id=PerformanceCapId.CAP_WORKSTATION_350T.value,
        tier_alias=PerformanceCapId.SINGLE_ACCELERATOR_STD.value,
        max_fp16_tflops=400.0,
        max_int8_fp8_tops=800.0,
        max_mem_bw_gbps=1000.0,
        max_vram_gb=48.0,
        typical_power_watts=450.0,
        example_devices=(
            "NVIDIA L4 24GB (72W Ada Tensor Core)",
            "NVIDIA GeForce RTX 4090 24GB / L40S 48GB",
            "AMD Radeon PRO W7900 48GB / RX 7900 XTX 24GB",
        ),
        description=(
            "Single-Accelerator Standard Sub-Profile (CAP_WORKSTATION_350T / SINGLE-ACCELERATOR-STD, num_devices==1): "
            "Peak INT8/FP8 <= 800 TOPS (BF16/FP16 <= 400 TFLOPS), "
            "Peak Memory Bandwidth <= 1000 GB/s, Active HBM/VRAM Budget <= 48 GB."
        ),
    ),
    PerformanceCapId.CAP_DATACENTER_1000T.value: PerformanceCapEnvelope(
        envelope_id=PerformanceCapId.CAP_DATACENTER_1000T.value,
        tier_alias=PerformanceCapId.SINGLE_ACCELERATOR_MAX.value,
        max_fp16_tflops=1500.0,
        max_int8_fp8_tops=3000.0,
        max_mem_bw_gbps=5300.0,
        max_vram_gb=192.0,
        typical_power_watts=700.0,
        example_devices=(
            "Google Cloud TPU v6e-1 Trillium (32GB HBM2e)",
            "Google Cloud TPU v7x 1-Chip (192GB HBM3e)",
            "NVIDIA H100 SXM5 80GB / H200 141GB",
            "AMD Instinct MI300X 1-GPU (192GB HBM3)",
        ),
        description=(
            "Legacy Datacenter Alias (CAP_DATACENTER_1000T / SINGLE-ACCELERATOR-MAX, num_devices==1): "
            "Peak INT8/FP8 <= 3000 TOPS (BF16/FP16 <= 1500 TFLOPS), "
            "Peak Memory Bandwidth <= 5300 GB/s, Active HBM3/3e Budget <= 192 GB."
        ),
    ),
    PerformanceCapId.CAP_UNCONSTRAINED.value: PerformanceCapEnvelope(
        envelope_id=PerformanceCapId.CAP_UNCONSTRAINED.value,
        tier_alias="SINGLE-ACCELERATOR-UNCONSTRAINED",
        max_fp16_tflops=10000.0,
        max_int8_fp8_tops=20000.0,
        max_mem_bw_gbps=16000.0,
        max_vram_gb=256.0,
        typical_power_watts=1200.0,
        example_devices=(
            "NVIDIA Blackwell B200 SXM6 192GB (1-GPU)",
            "Google Cloud TPU v7x-1 192GB (1-Chip)",
            "AMD Instinct MI300X 192GB (1-GPU)",
            "Custom Single-Package ASIC",
        ),
        description="Unconstrained Next-Gen Single-Device Envelope (num_devices==1).",
    ),
}


class ExecutionStrategy(str, Enum):
  """Real-time action execution strategies under non-zero inference latency L in LIBERO-RT."""

  SYNCHRONOUS_HOLD = "SYNCHRONOUS_HOLD"
  ASYNC_CHUNK = "ASYNC_CHUNK"
  RTC_INPAINTING = "RTC_INPAINTING"


class SafetyProbeType(str, Enum):
  """Mid-episode reactivity and safety probes injected during LIBERO-RT closed-loop evaluation."""

  OBJECT_DISPLACEMENT = "OBJECT_DISPLACEMENT"
  GOAL_SWITCH = "GOAL_SWITCH"
  EMERGENCY_STOP = "EMERGENCY_STOP"


class LiberoRTTier(str, Enum):
  """Three-tier scaling architecture for latency-coupled LIBERO-RT evaluation."""

  TIER1_ORDERED_TRACE_REPLAY = "tier1_ordered_trace_replay"
  TIER2_CLOUD_CPU_LATENCY_REPLAY = "tier2_cloud_cpu_latency_replay"
  TIER3_WEBSOCKET_HIL_ANCHOR = "tier3_websocket_hil_anchor"


class TrackType(str, Enum):
  """EmbodiedX CoRL 2026 Dual-Track Taxonomy (plus backward-compatible v1.0 track aliases).

  - Track 1 (CoRL 2026): Real-Time VLA Policy Track (`VLA_TRACK` / `track1_vla` / legacy `track2_vla`)
  - Track 2 (CoRL 2026): Interactive Action-Conditioned World Model & Omni Track (`WM_TRACK` / `track2_wm` / legacy `track1_world_model`)
  """

  VLA_TRACK = "VLA_TRACK"
  WM_TRACK = "WM_TRACK"
  TRACK1_VLA = "track1_vla"
  TRACK2_WM = "track2_wm"
  TRACK1_WORLD_MODEL = "track1_world_model"
  TRACK2_VLA = "track2_vla"


VLA_TRACK_VALUES = frozenset({
    TrackType.VLA_TRACK.value,
    TrackType.TRACK1_VLA.value,
    TrackType.TRACK2_VLA.value,
})

WM_TRACK_VALUES = frozenset({
    TrackType.WM_TRACK.value,
    TrackType.TRACK2_WM.value,
    TrackType.TRACK1_WORLD_MODEL.value,
})


def is_vla_track(track: str | TrackType) -> bool:
  """Return True if track identifier belongs to the Real-Time VLA Policy Track."""
  val = track.value if isinstance(track, TrackType) else str(track)
  return val in VLA_TRACK_VALUES


def is_wm_track(track: str | TrackType) -> bool:
  """Return True if track identifier belongs to the Interactive World Model / Omni Model Track."""
  val = track.value if isinstance(track, TrackType) else str(track)
  return val in WM_TRACK_VALUES


# Deprecated 2024 7-token-autoregressive OpenVLA checkpoints replaced by 2025/2026 VLA baselines
DEPRECATED_BASELINE_MODEL_IDS: Tuple[str, ...] = (
    "openvla-7b-vanilla",
    "openvla/openvla-7b",
    "openvla-7b-2024-ar",
    "vanilla-openvla-7b",
)


# Locked EmbodiedX v1.0 Canonical Challenge Lineup (D2: 3 Closed-Division Reference Models)
CANONICAL_CHALLENGE_MODELS: Tuple[str, ...] = (
    "pi05-3b",
    "gr00t-n1.7-3b",
    "cosmos-predict2.5-2b",
)

# Locked EmbodiedX v1.0 3-Model Closed-Division Reference Lineup (D2)
CANONICAL_V1_MODELS: Tuple[str, ...] = CANONICAL_CHALLENGE_MODELS

# Exact pinned HuggingFace Hub repositories and immutable 40-char commit SHAs for EEB v1.0 Closed Division
CANONICAL_HF_CHECKPOINT_PINS: Dict[str, Dict[str, str]] = {
    "pi05-3b": {
        "hf_repo": "lerobot/pi05_libero_finetuned_v044",
        "hf_revision": "7152f13e34c6c115d808aea0e5d08aca74eac14b",
        "hf_subdir": "",
        "tokenizer_repo": "unsloth/gemma-2b",
        "tokenizer_revision": "7ac9d201a57c8cdc6f939069fc0f044c60197a4a",
        "osi_license": "Apache-2.0",
    },
    "gr00t-n1.7-3b": {
        "hf_repo": "nvidia/GR00T-N1.7-LIBERO",
        "hf_revision": "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
        "hf_subdir": "libero_10",
        "helper_vlm_repo": "Qwen/Qwen3-VL-2B-Instruct",
        "helper_vlm_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "osi_license": "Apache-2.0",
    },
    "cosmos-predict2.5-2b": {
        "hf_repo": "nvidia/Cosmos-Predict2.5-2B",
        "hf_revision": "85f8ae7bfe8f5525c8d103429524dcf12f98bf7b",
        "hf_subdir": "robot/action-cond",
        "hf_weights_file": "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
        "hf_text_emb_file": "robot/action-cond/cr1_empty_string_text_embeddings.pt",
        "hf_tokenizer_file": "tokenizer.pth",
        "diffusers_helper_repo": "KyleShao/Cosmos-Predict2.5-2B-Diffusers",
        "diffusers_helper_revision": "1791139b148954cff76830ae8eefbdc1741a6cc0",
        "osi_license": "NVIDIA-Open-Model-License",
    },
}

# Exact pinned HuggingFace Hub public robot trace datasets for EEB v1.0
CANONICAL_HF_TRACE_DATASET_PINS: Dict[str, Dict[str, str]] = {
    "HuggingFaceVLA/libero": {
        "hf_repo": "HuggingFaceVLA/libero",
        "hf_revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "license": "Apache-2.0",
    },
    "lerobot/droid_100": {
        "hf_repo": "lerobot/droid_100",
        "hf_revision": "87301a2d2e99340e2010c9ef0f1d8e780b08aaf9",
        "license": "MIT",
    },
}

# Planned v2.0 Open Division & Embodied Omni Roadmap Models (registered in extensible registry, not in v1.0 Closed Division)
V2_PLANNED_CHALLENGE_MODELS: Tuple[str, ...] = (
    "smolvla-450m",
    "cosmos3-edge",
    "qwen3-omni-30b-a3b",
)


# Modern 2025/2026 Reference Checkpoints for CoRL 2026 Track 1 (VLA_TRACK) and Track 2 (WM_TRACK)
MODERN_REFERENCE_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Category 1: Real-Time VLA Policy Track (VLA_TRACK) - 2 Canonical Models
    # =========================================================================
    "gr00t-n1.7-3b": {
        "checkpoint_id": "gr00t-n1.7-3b",
        "canonical_challenge_model": True,
        "aliases": [
            "gr00t",
            "gr00t-n1.7",
            "nvidia/GR00T-N1.7-LIBERO",
            "nvidia/GR00T-N1.7-LIBERO/libero_10",
            "GR00T-N1.7-3B",
            "nvidia/GR00T-N1.7-3B",
            "gr00t-n1.7-3b/libero_sim",
        ],
        "track": TrackType.VLA_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "nvidia/GR00T-N1.7-LIBERO",
        "hf_revision": "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
        "hf_subdir": "libero_10",
        "release_window": "Feb-Jul 2026",
        "framework": "Isaac-GR00T-N1.7",
        "architecture": "3B Dual-System Cosmos-Reason2-2B (Qwen3-VL-2B) VLM (System 2) + 16-Step DiT Flow Action Head (System 1, 2x256x256 RGB, H=16, K=8 @ 20 Hz)",
        "params_b": 3.0,
        "active_params_b": 3.0,
        "default_denoising_steps": 16,
        "native_action_chunk_h": 16,
        "execute_horizon_k": 8,
        "resolution_hw": 256,
        "control_freq_hz": 20.0,
        "realtime_cutoff_hz": 2.5,
        "realtime_cutoff_p99_ms": 400.0,
        "realtime_actions_per_sec_floor": 20.0,
        "action_head_type": "continuous_flow",
        "osi_license": "Apache-2.0",
        "rewarded_optimizations": [
            "dual_rate_system1_system2_async_scheduling",
            "dit_step_distillation_16_to_2_4",
            "vlm_prefix_kv_cache_reuse",
            "fp8_w4_quantization",
        ],
    },
    # Note on pi05-3b horizon: raw pi0.5-base predicts H=50 actions, whereas the
    # official canonical LIBERO benchmark checkpoint (pi05_libero /
    # lerobot/pi05_libero_finetuned_v044) has native action_horizon H=10 with locked
    # execute_horizon K=5 at 20 Hz over 10 Euler ODE steps and 2x224x224 RGB views.
    "pi05-3b": {
        "checkpoint_id": "pi05-3b",
        "canonical_challenge_model": True,
        "aliases": [
            "pi05",
            "pi0.5",
            "pi0-3b",
            "pi0-3.3B",
            "pi05-3b/pi05_libero",
            "lerobot/pi05_libero_finetuned_v044",
            "lerobot/pi05_libero_finetuned",
            "lerobot/pi05_base",
            "PhysicalIntelligence/pi0.5-libero",
            "openpi-pi05-3b",
        ],
        "track": TrackType.VLA_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "lerobot/pi05_libero_finetuned_v044",
        "hf_revision": "7152f13e34c6c115d808aea0e5d08aca74eac14b",
        "hf_subdir": "",
        "release_window": "2025-2026",
        "framework": "openpi (JAX) / lerobot & TorchTPU (PyTorch)",
        "architecture": "3.3B PaliGemma VLM + 300M Flow-Matching Action Expert (H=10, K=5 @ 20 Hz, 10 ODE steps default, 2x224x224 RGB; native JAX openpi and PyTorch lerobot/TorchTPU both certified)",
        "params_b": 3.3,
        "active_params_b": 3.3,
        "default_denoising_steps": 10,
        "native_action_chunk_h": 10,
        "execute_horizon_k": 5,
        "resolution_hw": 224,
        "certified_runtimes": ["jax-openpi", "pytorch-lerobot", "pytorch-torchtpu"],
        "control_freq_hz": 20.0,
        "realtime_cutoff_hz": 4.0,
        "realtime_cutoff_p99_ms": 250.0,
        "realtime_actions_per_sec_floor": 20.0,
        "action_head_type": "continuous_flow",
        "osi_license": "Apache-2.0",
        "rewarded_optimizations": [
            "flow_matching_step_reduction_10_to_1_2",
            "real_time_chunking_rtc",
            "vlm_prefix_kv_cache_reuse",
            "w4_fp8_quantization",
        ],
    },
    # =========================================================================
    # Planned v2.0 Roadmap / Extensible World & Omni Checkpoints (Non-v1.0 Canonical)
    # =========================================================================
    "cosmos3-edge": {
        "checkpoint_id": "cosmos3-edge",
        "canonical_challenge_model": False,
        "aliases": [
            "Cosmos3-Edge",
            "nvidia/Cosmos3-Edge",
            "nvidia/Cosmos3-Edge-Policy-DROID",
            "nvidia/Cosmos3-Super-Image2Video",
        ],
        "track": TrackType.WM_TRACK.value,
        "division": DivisionType.OPEN.value,
        "hf_repo": "nvidia/Cosmos3-Edge",
        "release_window": "May-Jul 2026",
        "framework": "cosmos3-edge",
        "architecture": "Spatio-Temporal Action-Conditioned World Model & Policy Evaluator (Cosmos3-Edge / DROID)",
        "params_b": 2.4,
        "active_params_b": 2.4,
        "default_denoising_steps": 16,
        "horizon_frames": 16,
        "osi_license": "NVIDIA-Open-Model-License",
        "rewarded_optimizations": [
            "causal_spatio_temporal_kv_caching",
            "dit_step_distillation_16_to_4",
            "fp8_nvfp4_quantization",
            "dit_feature_caching",
        ],
    },
    "qwen3-omni-30b-a3b": {
        "checkpoint_id": "qwen3-omni-30b-a3b",
        "canonical_challenge_model": False,
        "aliases": [
            "Qwen3-Omni-30B-A3B",
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "qwen3-omni-30b-a3b-instruct",
        ],
        "track": TrackType.WM_TRACK.value,
        "division": DivisionType.OPEN.value,
        "hf_repo": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "release_window": "2025-2026",
        "framework": "qwen3-omni",
        "architecture": "30B Total / 3B Active MoE Thinker-Talker Streaming Multimodal (Video+Audio+Proprio+Action) World/Omni Model",
        "params_b": 30.0,
        "active_params_b": 3.0,
        "default_denoising_steps": 4,
        "horizon_frames": 16,
        "osi_license": "Apache-2.0",
        "rewarded_optimizations": [
            "moe_expert_offload_and_fused_routing",
            "thinker_talker_async_streaming_pipeline",
            "multimodal_video_kv_cache_compression",
            "w4a8_fp8_moe_quantization",
        ],
    },
    # =========================================================================
    # Additional Supported Open Models in Extensible Registry
    # =========================================================================
    "gr00t-n1.5-3b": {
        "checkpoint_id": "gr00t-n1.5-3b",
        "canonical_challenge_model": False,
        "aliases": ["GR00T-N1.5-3B", "nvidia/GR00T-N1.5-3B"],
        "track": TrackType.VLA_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "nvidia/GR00T-N1.5-3B",
        "framework": "Isaac-GR00T",
        "architecture": "Eagle-2 VLM System 2 + Diffusion Transformer (DiT) System 1",
        "params_b": 3.0,
        "active_params_b": 3.0,
        "default_denoising_steps": 16,
        "native_action_chunk_h": 16,
        "control_freq_hz": 20.0,
        "action_head_type": "continuous_flow",
        "rewarded_optimizations": [
            "dual_rate_system1_system2_async_scheduling",
            "dit_step_distillation",
            "fp8_w4_quantization",
        ],
    },
    "openvla-oft-7b": {
        "checkpoint_id": "openvla-oft-7b",
        "canonical_challenge_model": False,
        "aliases": ["OpenVLA-OFT-7B", "moojink/openvla-7b-oft-finetuned-libero"],
        "track": TrackType.VLA_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "moojink/openvla-7b-oft-finetuned-libero",
        "framework": "openvla-oft",
        "architecture": "7.5B Prismatic VLM + Parallel Decoding + Continuous Action Chunking (replaces stale 7-token-AR 2024 OpenVLA)",
        "params_b": 7.5,
        "active_params_b": 7.5,
        "default_denoising_steps": 1,
        "native_action_chunk_h": 8,
        "control_freq_hz": 20.0,
        "action_head_type": "continuous_flow",
        "rewarded_optimizations": [
            "parallel_action_decoding",
            "continuous_action_chunking",
            "int4_awq_w4a16_quantization",
        ],
    },
    "smolvla-450m": {
        "checkpoint_id": "smolvla-450m",
        "canonical_challenge_model": False,
        "aliases": ["SmolVLA-450M", "HuggingFaceTB/SmolVLA-450M"],
        "track": TrackType.VLA_TRACK.value,
        "division": DivisionType.OPEN.value,
        "hf_repo": "HuggingFaceTB/SmolVLA-450M",
        "framework": "lerobot-smolvla",
        "architecture": "450M Compact VLM + Flow-Matching Action Chunking Policy (Open Division Reference)",
        "params_b": 0.45,
        "active_params_b": 0.45,
        "default_denoising_steps": 4,
        "native_action_chunk_h": 16,
        "control_freq_hz": 20.0,
        "action_head_type": "continuous_flow",
        "rewarded_optimizations": [
            "compact_vlm_distillation",
            "rtc_inpainting",
            "fp16_int8_edge_execution",
        ],
    },
    "cosmos-predict2-2b-action": {
        "checkpoint_id": "cosmos-predict2-2b-action",
        "canonical_challenge_model": False,
        "aliases": ["Cosmos-Predict2-2B-Action", "nvidia/Cosmos-Predict2-2B-Video2World"],
        "track": TrackType.WM_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "nvidia/Cosmos-Predict2-2B-Video2World",
        "framework": "cosmos-predict2",
        "architecture": "2.3B Action-Conditioned Video2World DiT (Robot Manipulation Split)",
        "params_b": 2.3,
        "active_params_b": 2.3,
        "default_denoising_steps": 24,
        "horizon_frames": 16,
        "rewarded_optimizations": [
            "causal_dit_step_distillation",
            "fp8_nvfp4_quantization",
            "dit_feature_caching",
        ],
    },
    "cosmos-predict2.5-2b": {
        "checkpoint_id": "cosmos-predict2.5-2b",
        "canonical_challenge_model": True,
        "aliases": [
            "cosmos",
            "cosmos-predict2.5",
            "Cosmos-Predict2.5-2B",
            "nvidia/Cosmos-Predict2.5-2B",
            "nvidia/Cosmos-Predict2.5-2B/robot/action-cond",
        ],
        "track": TrackType.WM_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "nvidia/Cosmos-Predict2.5-2B",
        "hf_revision": "85f8ae7bfe8f5525c8d103429524dcf12f98bf7b",
        "hf_subdir": "robot/action-cond",
        "framework": "cosmos-predict2.5 / torchtpu",
        "architecture": "2B Action-Conditioned Video2World 3D DiT (256x256, 16-Frame Prediction Rollout, 1.0 s Horizon, 10 Steps, CFG=1.0, ~60 TFLOPs/prediction)",
        "params_b": 2.0,
        "active_params_b": 2.0,
        "default_denoising_steps": 10,
        "horizon_frames": 16,
        "resolution_hw": 256,
        "control_freq_hz": 24.0,
        "delta_t_ms": 1000.0 / 24.0,
        "chunk_horizon_s": 1.0,
        "guidance_scale": 1.0,
        "canonical_flops_per_rollout_gflops": 60000.0,
        "canonical_flops_per_frame_gflops": 3750.0,
        "realtime_cutoff_hz": 1.0,
        "realtime_cutoff_p99_ms": 1000.0,
        "osi_license": "NVIDIA-Open-Model-License",
        "rewarded_optimizations": [
            "cross_frame_kv_cache_reuse",
            "flash_attention3_3d_dit",
            "fp8_quantization",
            "torchtpu_torch_compile_and_pallas_fusion",
        ],
    },
    "wan2.1-wm-1.3b": {
        "checkpoint_id": "wan2.1-wm-1.3b",
        "canonical_challenge_model": False,
        "aliases": ["Wan2.1-WM-1.3B", "Wan-AI/Wan2.1-T2V-1.3B-ActionWM"],
        "track": TrackType.WM_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B-ActionWM",
        "framework": "wan2.1-wm",
        "architecture": "1.3B Action-Conditioned Causal DiT World Model",
        "params_b": 1.3,
        "active_params_b": 1.3,
        "default_denoising_steps": 20,
        "horizon_frames": 16,
        "rewarded_optimizations": [
            "causal_block_kv_caching",
            "consistency_step_distillation",
            "fp8_attention_and_linear",
        ],
    },
    "ctrl-world-1.5b": {
        "checkpoint_id": "ctrl-world-1.5b",
        "canonical_challenge_model": False,
        "aliases": ["Ctrl-World-1.5B", "Ctrl-World/ctrl-world-1.5b"],
        "track": TrackType.WM_TRACK.value,
        "division": DivisionType.CLOSED.value,
        "hf_repo": "Ctrl-World/ctrl-world-1.5b",
        "framework": "ctrl-world",
        "architecture": "1.5B Diffusion-Forcing Action-Conditioned World Model",
        "params_b": 1.5,
        "active_params_b": 1.5,
        "default_denoising_steps": 20,
        "horizon_frames": 16,
        "rewarded_optimizations": [
            "diffusion_forcing_rolling_horizon",
            "fp8_quantization",
            "cuda_graph_rollout_capture",
        ],
    },
}


def register_model_checkpoint(
    checkpoint_id: Optional[str] = None,
    spec: Optional[Dict[str, Any]] = None,
    *,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
  """Register a new open-source VLA, World Model, or Omni checkpoint in `MODERN_REFERENCE_CHECKPOINTS`."""
  cid = checkpoint_id or model_id
  if not cid or not str(cid).strip():
    raise ValueError("checkpoint_id (or model_id) must be a non-empty string.")
  entry = dict(spec) if spec is not None else {}
  entry.update(kwargs)
  entry.setdefault("checkpoint_id", cid)
  entry.setdefault("canonical_challenge_model", False)
  MODERN_REFERENCE_CHECKPOINTS[cid] = entry
  return entry


class CompressionTier(str, Enum):
  """Compression tier controlling Stage 2 Gate B closed-loop success delta threshold."""

  STRICT_LOSSLESS = "strict_lossless"
  EXTREME_COMPRESSION = "extreme_compression"


class RunStatus(str, Enum):
  """Complete state machine for Stage 1 / Stage 2 benchmark runs."""

  PENDING = "PENDING"
  RUNNING = "RUNNING"
  CERTIFIED = "CERTIFIED"
  REJECTED = "REJECTED"
  BLOCKED = "BLOCKED"
  FAILED = "FAILED"
  SKIPPED = "SKIPPED"
  INVALID = "INVALID"


REASON_REQUIRED_STATUSES = frozenset({
    RunStatus.REJECTED,
    RunStatus.BLOCKED,
    RunStatus.FAILED,
    RunStatus.SKIPPED,
    RunStatus.INVALID,
})

TERMINAL_STATUSES = frozenset({
    RunStatus.CERTIFIED,
    RunStatus.REJECTED,
    RunStatus.FAILED,
    RunStatus.SKIPPED,
    RunStatus.INVALID,
})

VALID_STATE_TRANSITIONS: Dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({
        RunStatus.RUNNING,
        RunStatus.BLOCKED,
        RunStatus.SKIPPED,
        RunStatus.INVALID,
    }),
    RunStatus.RUNNING: frozenset({
        RunStatus.CERTIFIED,
        RunStatus.REJECTED,
        RunStatus.BLOCKED,
        RunStatus.FAILED,
        RunStatus.INVALID,
    }),
    RunStatus.BLOCKED: frozenset({
        RunStatus.PENDING,
        RunStatus.RUNNING,
        RunStatus.FAILED,
        RunStatus.SKIPPED,
        RunStatus.INVALID,
    }),
    RunStatus.CERTIFIED: frozenset({RunStatus.CERTIFIED}),
    RunStatus.REJECTED: frozenset({RunStatus.REJECTED, RunStatus.PENDING}),
    RunStatus.FAILED: frozenset({RunStatus.FAILED, RunStatus.PENDING}),
    RunStatus.SKIPPED: frozenset({RunStatus.SKIPPED, RunStatus.PENDING}),
    RunStatus.INVALID: frozenset({RunStatus.INVALID}),
}


# Version-indexed Edge Power Class specifications for P1_ONBOARD
POWER_CLASS_SPECS: Dict[str, Dict[str, Any]] = {
    PowerClass.E1.value: {
        "max_power_watts": 25.0,
        "default_perf_cap": PerformanceCapId.CAP_EDGE_100T.value,
        "reference_hardware_sku": "JETSON_ORIN_NX_16GB",
        "description": "Low-power onboard mobile/manipulator edge envelope (<= 25W, CAP_EDGE_100T)",
    },
    PowerClass.E2.value: {
        "max_power_watts": 60.0,
        "default_perf_cap": PerformanceCapId.CAP_EDGE_100T.value,
        "reference_hardware_sku": "JETSON_ORIN_AGX_64GB",
        "description": "Standard autonomous robot onboard module envelope (<= 60W, CAP_EDGE_100T)",
    },
    PowerClass.E3.value: {
        "max_power_watts": 130.0,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "reference_hardware_sku": "JETSON_AGX_THOR_64GB",
        "description": "High-performance humanoid onboard compute envelope (<= 130W, CAP_SINGLE_CHIP)",
    },
    PowerClass.E_OPEN.value: {
        "max_power_watts": 10000.0,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "reference_hardware_sku": "TPU_V6E_1_32GB",
        "description": "Practical Physical-AI Single-Chip power class (CAP_SINGLE_CHIP: <=2,000 TOPS / <=1,000 TFLOPS, <=64 GB native HBM/VRAM, <=2,000 GB/s)",
    },
}


# P2_FLEET Server SLO Presets: preset_name -> (control_rate_hz, max_p99_latency_ms)
P2_SLO_PRESETS: Dict[str, Tuple[float, float]] = {
    "10Hz_50ms": (10.0, 50.0),
    "5Hz_100ms": (5.0, 100.0),
    "2Hz_250ms": (2.0, 250.0),
}

P2_CONCURRENCY_GRID: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]
S1_STREAM_CONCURRENCY_GRID: List[int] = [1, 2, 4, 8, 16, 32]

# S1_INTERACTIVE_WM frame rate budgets (fps -> max frame latency in ms)
S1_FPS_TARGETS_MS: Dict[str, float] = {
    "10fps": 100.0,
    "24fps": 1000.0 / 24.0,  # 41.6667 ms
}

# LIBERO-RT physical simulation coupling constants
LIBERO_RT_CONSTANTS: Dict[str, float] = {
    "default_control_freq_hz": 20.0,
    "default_delta_t_ms": 50.0,
    "max_chunk_horizon_s": 1.0,
    "max_episode_sim_time_multiplier": 1.5,
    "default_p90_demo_duration_s": 12.0,
    "emergency_stop_max_reaction_ms": 150.0,
    "tier3_consistency_tolerance_pct": 10.0,
}


# Version-indexed Hardware SKU lookup table across NVIDIA, AMD, Google Cloud TPU, Apple Silicon, & Custom ASICs
HARDWARE_SKU_SPECS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Canonical Native Uncapped Physical-AI Single-Chip SKUs (CAP_SINGLE_CHIP)
    # =========================================================================
    # 1. Google Cloud TPU (Native Uncapped Single-Chip; Dense Silicon Ratings)
    "TPU_V6E_1_32GB": {
        "display_name": "Google Cloud TPU v6e-1 Trillium (32GB HBM2e, 1-Chip)",
        "vendor": "GOOGLE_TPU",
        "num_devices": 1,
        "vram_mb": 32768,
        "tdp_watts": 170.0,
        "idle_watts": 35.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 918.0,
        "fp8_tflops": 1836.0,
        "dense_int8_tops": 1836.0,
        "supports_native_fp8": True,
        "mem_bw_gbps": 1600.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 1.375,
        "capex_usd": 7500.0,
    },
    "TPU_V6E_1": {
        "display_name": "Google Cloud TPU v6e-1 Trillium (32GB HBM2e, 1-Chip)",
        "vendor": "GOOGLE_TPU",
        "num_devices": 1,
        "vram_mb": 32768,
        "tdp_watts": 170.0,
        "idle_watts": 35.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 918.0,
        "fp8_tflops": 1836.0,
        "dense_int8_tops": 1836.0,
        "supports_native_fp8": True,
        "mem_bw_gbps": 1600.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 1.375,
        "capex_usd": 7500.0,
    },
    "TPU_V5E_1_16GB": {
        "display_name": "Google Cloud TPU v5e-1 (16GB HBM2e, 1-Chip)",
        "vendor": "GOOGLE_TPU",
        "num_devices": 1,
        "vram_mb": 16384,
        "tdp_watts": 120.0,
        "idle_watts": 28.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 197.0,
        "fp8_tflops": 393.0,
        "dense_int8_tops": 393.0,
        "supports_native_fp8": False,
        "mem_bw_gbps": 819.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.60,
        "capex_usd": 4500.0,
    },
    # 2. NVIDIA (Native Uncapped Single-Chip; Strictly Non-Sparse Dense Silicon Ratings)
    "JETSON_AGX_THOR_64GB": {
        "display_name": "NVIDIA Jetson AGX Thor 64GB (Blackwell Robotics)",
        "vendor": "NVIDIA_TEGRA",
        "num_devices": 1,
        "vram_mb": 65536,
        "tdp_watts": 130.0,
        "idle_watts": 18.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 259.0,
        "fp8_tflops": 517.5,
        "dense_int8_tops": 517.5,
        "sparse_fp4_tops": 1035.0,
        "supports_native_fp8": True,
        "mem_bw_gbps": 273.0,
        "power_class": PowerClass.E3.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.85,
        "capex_usd": 3499.0,
    },
    "RTX_5090_32GB": {
        "display_name": "NVIDIA GeForce RTX 5090 (32GB GDDR7)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 32768,
        "tdp_watts": 575.0,
        "idle_watts": 55.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 209.5,
        "fp8_tflops": 419.0,
        "dense_int8_tops": 419.0,
        "supports_native_fp8": True,
        "mem_bw_gbps": 1792.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.95,
        "capex_usd": 1999.0,
    },
    "RTX_5090": {
        "display_name": "NVIDIA GeForce RTX 5090 (32GB GDDR7)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 32768,
        "tdp_watts": 575.0,
        "idle_watts": 55.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 209.5,
        "fp8_tflops": 419.0,
        "dense_int8_tops": 419.0,
        "supports_native_fp8": True,
        "mem_bw_gbps": 1792.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.95,
        "capex_usd": 1999.0,
    },
    "L40S_48GB": {
        "display_name": "NVIDIA L40S (48GB GDDR6)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 49152,
        "tdp_watts": 350.0,
        "idle_watts": 42.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 362.05,
        "fp8_tflops": 733.0,
        "dense_int8_tops": 733.0,
        "supports_native_fp8": True,
        "mem_bw_gbps": 864.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 1.15,
        "capex_usd": 8500.0,
    },
    "L4_24GB": {
        "display_name": "NVIDIA L4 (24GB GDDR6, 72W Ada Tensor Core)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 24576,
        "tdp_watts": 72.0,
        "idle_watts": 12.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 121.0,
        "fp8_tflops": 242.0,
        "dense_int8_tops": 242.0,
        "supports_native_fp8": True,
        "mem_bw_gbps": 300.0,
        "power_class": PowerClass.E3.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.71,
        "capex_usd": 2500.0,
    },
    "RTX_4090_24GB": {
        "display_name": "NVIDIA GeForce RTX 4090 (24GB GDDR6X)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 24576,
        "tdp_watts": 450.0,
        "idle_watts": 45.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 82.6,  # Dense BF16/FP16 Tensor Core w/ FP32 accumulate (165.2 is FP16-accum only)
        "fp8_tflops": 330.3,
        "dense_int8_tops": 330.3,
        "supports_native_fp8": True,
        "mem_bw_gbps": 1008.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.69,
        "capex_usd": 1599.0,
    },
    "RTX_4090": {
        "display_name": "NVIDIA GeForce RTX 4090 (24GB GDDR6X)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 24576,
        "tdp_watts": 450.0,
        "idle_watts": 45.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 82.6,  # Dense BF16/FP16 Tensor Core w/ FP32 accumulate (165.2 is FP16-accum only)
        "fp8_tflops": 330.3,
        "dense_int8_tops": 330.3,
        "supports_native_fp8": True,
        "mem_bw_gbps": 1008.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.69,
        "capex_usd": 1599.0,
    },
    "JETSON_ORIN_NX_16GB": {
        "display_name": "NVIDIA Jetson Orin NX 16GB (Ampere sm_87, nvpmodel 25W)",
        "vendor": "NVIDIA_TEGRA",
        "num_devices": 1,
        "vram_mb": 16384,
        "tdp_watts": 25.0,
        "idle_watts": 6.5,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 23.0,
        "fp8_tflops": 47.0,
        "dense_int8_tops": 47.0,
        "supports_native_fp8": False,
        "mem_bw_gbps": 102.4,
        "power_class": PowerClass.E1.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.25,
        "capex_usd": 999.0,
    },
    "JETSON_ORIN_AGX_64GB": {
        "display_name": "NVIDIA Jetson AGX Orin 64GB (Ampere sm_87)",
        "vendor": "NVIDIA_TEGRA",
        "num_devices": 1,
        "vram_mb": 65536,
        "tdp_watts": 60.0,
        "idle_watts": 12.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 42.5,
        "fp8_tflops": 85.0,
        "dense_int8_tops": 85.0,
        "supports_native_fp8": False,
        "mem_bw_gbps": 204.8,
        "power_class": PowerClass.E2.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.35,
        "capex_usd": 1999.0,
    },
    "RTX_4060_TI_16GB": {
        "display_name": "NVIDIA GeForce RTX 4060 Ti (16GB GDDR6)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 16384,
        "tdp_watts": 165.0,
        "idle_watts": 18.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 44.1,
        "fp8_tflops": 88.2,
        "dense_int8_tops": 88.2,
        "supports_native_fp8": True,
        "mem_bw_gbps": 240.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.32,
        "capex_usd": 499.0,
    },
    # 3. Additional Single-Chip Accelerators (AMD ROCm, Apple Silicon, Custom ASIC)
    "AMD_MI210_64GB": {
        "display_name": "AMD Instinct MI210 (64GB HBM2e, 1-GPU CDNA2)",
        "vendor": "AMD_ROCM",
        "num_devices": 1,
        "vram_mb": 65536,
        "tdp_watts": 300.0,
        "idle_watts": 42.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 181.0,
        "fp8_tflops": 181.0,
        "dense_int8_tops": 181.0,
        "supports_native_fp8": False,
        "mem_bw_gbps": 1638.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 1.10,
        "capex_usd": 12000.0,
    },
    "AMD_W7900_48GB": {
        "display_name": "AMD Radeon PRO W7900 (48GB GDDR6)",
        "vendor": "AMD_ROCM",
        "num_devices": 1,
        "vram_mb": 49152,
        "tdp_watts": 295.0,
        "idle_watts": 38.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 123.0,
        "fp8_tflops": 246.0,
        "mem_bw_gbps": 864.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.75,
        "capex_usd": 3999.0,
    },
    "AMD_RX_7900_XTX_24GB": {
        "display_name": "AMD Radeon RX 7900 XTX (24GB GDDR6)",
        "vendor": "AMD_ROCM",
        "num_devices": 1,
        "vram_mb": 24576,
        "tdp_watts": 355.0,
        "idle_watts": 42.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 123.0,
        "fp8_tflops": 246.0,
        "mem_bw_gbps": 960.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.55,
        "capex_usd": 999.0,
    },
    "AMD_MI250_1GCD_64GB": {
        "display_name": "AMD Instinct MI250 (1-GCD, 64GB HBM2e)",
        "vendor": "AMD_ROCM",
        "num_devices": 1,
        "vram_mb": 65536,
        "tdp_watts": 280.0,
        "idle_watts": 40.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 191.5,
        "fp8_tflops": 191.5,
        "dense_int8_tops": 191.5,
        "supports_native_fp8": False,
        "mem_bw_gbps": 1600.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 1.05,
        "capex_usd": 9500.0,
    },
    # 4. Apple Silicon & Custom Single-Chip ASICs (Native Uncapped Single-Chip)
    "APPLE_M4_PRO_16GB": {
        "display_name": "Apple M4 Pro (20-core GPU, 16GB Unified Memory Slice)",
        "vendor": "APPLE_SILICON",
        "num_devices": 1,
        "vram_mb": 16384,
        "tdp_watts": 45.0,
        "idle_watts": 8.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 46.0,
        "fp8_tflops": 92.0,
        "mem_bw_gbps": 240.0,
        "power_class": PowerClass.E2.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.28,
        "capex_usd": 1999.0,
    },
    "CUSTOM_ASIC_SINGLE_DEVICE": {
        "display_name": "Custom Embodied AI Single-Device ASIC (NPU / LPU / Spatial Accelerator)",
        "vendor": "CUSTOM_ASIC",
        "num_devices": 1,
        "vram_mb": 49152,
        "tdp_watts": 180.0,
        "idle_watts": 22.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 380.0,
        "fp8_tflops": 760.0,
        "mem_bw_gbps": 960.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_SINGLE_CHIP.value,
        "hourly_cost_usd": 0.90,
        "capex_usd": 5000.0,
    },
    # =========================================================================
    # Legacy Datacenter SKUs Kept Under CAP_DATACENTER_1000T / CAP_UNCONSTRAINED
    # =========================================================================
    "H100_SXM5": {
        "display_name": "NVIDIA H100 SXM5 (80GB HBM3)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 700.0,
        "idle_watts": 70.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 989.0,
        "fp8_tflops": 1979.0,
        "mem_bw_gbps": 3350.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.95,
        "capex_usd": 30000.0,
    },
    "H100_SXM5_80GB": {
        "display_name": "NVIDIA H100 SXM5 (80GB HBM3)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 700.0,
        "idle_watts": 70.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 989.0,
        "fp8_tflops": 1979.0,
        "mem_bw_gbps": 3350.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.95,
        "capex_usd": 30000.0,
    },
    "H100_SXM_80GB": {
        "display_name": "NVIDIA H100 SXM (80GB HBM3)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 700.0,
        "idle_watts": 70.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 989.0,
        "fp8_tflops": 1979.0,
        "mem_bw_gbps": 3350.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.95,
        "capex_usd": 30000.0,
    },
    "H100_80GB": {
        "display_name": "NVIDIA H100 (80GB HBM3)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 700.0,
        "idle_watts": 70.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 989.0,
        "fp8_tflops": 1979.0,
        "mem_bw_gbps": 3350.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.95,
        "capex_usd": 30000.0,
    },
    "A100_SXM_80GB": {
        "display_name": "NVIDIA A100 SXM4 (80GB HBM2e)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 400.0,
        "idle_watts": 55.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 312.0,
        "fp8_tflops": 624.0,
        "dense_int8_tops": 624.0,
        "supports_native_fp8": False,
        "mem_bw_gbps": 2039.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 1.85,
        "capex_usd": 15000.0,
    },
    "A100_80GB": {
        "display_name": "NVIDIA A100 (80GB HBM2e)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 400.0,
        "idle_watts": 55.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 312.0,
        "fp8_tflops": 624.0,
        "dense_int8_tops": 624.0,
        "supports_native_fp8": False,
        "mem_bw_gbps": 2039.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 1.85,
        "capex_usd": 15000.0,
    },
    "H200_141GB": {
        "display_name": "NVIDIA H200 SXM (141GB HBM3e)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 144384,
        "tdp_watts": 700.0,
        "idle_watts": 75.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 989.0,
        "fp8_tflops": 1979.0,
        "mem_bw_gbps": 4800.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 3.50,
        "capex_usd": 36000.0,
    },
    "B200_SXM6": {
        "display_name": "NVIDIA Blackwell B200 SXM6 (192GB HBM3e)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 196608,
        "tdp_watts": 1000.0,
        "idle_watts": 95.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 2250.0,
        "fp8_tflops": 4500.0,
        "mem_bw_gbps": 8000.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_UNCONSTRAINED.value,
        "hourly_cost_usd": 4.80,
        "capex_usd": 40000.0,
    },
    "B200_SXM6_192GB": {
        "display_name": "NVIDIA Blackwell B200 SXM6 (192GB HBM3e)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 196608,
        "tdp_watts": 1000.0,
        "idle_watts": 95.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 2250.0,
        "fp8_tflops": 4500.0,
        "mem_bw_gbps": 8000.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_UNCONSTRAINED.value,
        "hourly_cost_usd": 4.80,
        "capex_usd": 40000.0,
    },
    "B200_180GB": {
        "display_name": "NVIDIA Blackwell B200 (180GB HBM3e)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 184320,
        "tdp_watts": 1000.0,
        "idle_watts": 95.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 2250.0,
        "fp8_tflops": 4500.0,
        "mem_bw_gbps": 8000.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_UNCONSTRAINED.value,
        "hourly_cost_usd": 4.80,
        "capex_usd": 40000.0,
    },
    "B200_192GB": {
        "display_name": "NVIDIA Blackwell B200 (192GB HBM3e)",
        "vendor": "NVIDIA",
        "num_devices": 1,
        "vram_mb": 196608,
        "tdp_watts": 1000.0,
        "idle_watts": 95.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 2250.0,
        "fp8_tflops": 4500.0,
        "mem_bw_gbps": 8000.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_UNCONSTRAINED.value,
        "hourly_cost_usd": 4.80,
        "capex_usd": 40000.0,
    },
    "JETSON_THOR_128GB": {
        "display_name": "NVIDIA Jetson Thor 128GB (Blackwell Robotics)",
        "vendor": "NVIDIA_TEGRA",
        "num_devices": 1,
        "vram_mb": 131072,
        "tdp_watts": 130.0,
        "idle_watts": 18.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 259.0,
        "fp8_tflops": 517.5,
        "dense_int8_tops": 517.5,
        "mem_bw_gbps": 273.0,
        "power_class": PowerClass.E3.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 0.85,
        "capex_usd": 3499.0,
    },
    "TPU_V7X_1": {
        "display_name": "Google Cloud TPU v7x-1 (Ironwood 192GB HBM3e, 1-Chip)",
        "vendor": "GOOGLE_TPU",
        "num_devices": 1,
        "vram_mb": 196608,
        "tdp_watts": 480.0,
        "idle_watts": 55.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 2307.0,
        "fp8_tflops": 4614.0,
        "mem_bw_gbps": 7370.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_UNCONSTRAINED.value,
        "hourly_cost_usd": 3.40,
        "capex_usd": 22000.0,
    },
    "TPU_V7X_1CHIP": {
        "display_name": "Google Cloud TPU v7x (1-TensorCore / 80GB HBM3e Cap)",
        "vendor": "GOOGLE_TPU",
        "num_devices": 1,
        "vram_mb": 81920,
        "physical_vram_mb": 196608,
        "tdp_watts": 320.0,
        "idle_watts": 45.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 1150.0,
        "fp8_tflops": 2300.0,
        "mem_bw_gbps": 3685.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.40,
        "capex_usd": 14000.0,
    },
    "AMD_MI300A_128GB": {
        "display_name": "AMD Instinct MI300A / MI250 (80GB Active HBM3 Cap, 1-APU/GPU)",
        "vendor": "AMD_ROCM",
        "num_devices": 1,
        "vram_mb": 81920,
        "physical_vram_mb": 131072,
        "tdp_watts": 550.0,
        "idle_watts": 65.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 613.0,
        "fp8_tflops": 1226.0,
        "mem_bw_gbps": 3276.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.20,
        "capex_usd": 18000.0,
    },
    "AMD_MI300X_80GB_CAP": {
        "display_name": "AMD Instinct MI300X (80GB Active HBM3 Cap, 1-GPU)",
        "vendor": "AMD_ROCM",
        "num_devices": 1,
        "vram_mb": 81920,
        "physical_vram_mb": 196608,
        "tdp_watts": 750.0,
        "idle_watts": 85.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 1307.0,
        "fp8_tflops": 2614.0,
        "mem_bw_gbps": 5300.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 3.25,
        "capex_usd": 25000.0,
    },
    "AMD_MI300X_192GB": {
        "display_name": "AMD Instinct MI300X (192GB HBM3, 1-GPU)",
        "vendor": "AMD_ROCM",
        "num_devices": 1,
        "vram_mb": 81920,
        "physical_vram_mb": 196608,
        "tdp_watts": 750.0,
        "idle_watts": 85.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 1307.0,
        "fp8_tflops": 2614.0,
        "mem_bw_gbps": 5300.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 3.25,
        "capex_usd": 25000.0,
    },
    "AWS_TRAINIUM2_1CHIP": {
        "display_name": "AWS Trainium2 (1-Chip, 80GB Active HBM3 Cap)",
        "vendor": "CUSTOM_ASIC",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 500.0,
        "idle_watts": 55.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 650.0,
        "fp8_tflops": 1300.0,
        "mem_bw_gbps": 2900.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.10,
        "capex_usd": 16000.0,
    },
    "INTEL_GAUDI3_1CHIP": {
        "display_name": "Intel Gaudi 3 (1-Chip, 80GB Active HBM2e Cap)",
        "vendor": "CUSTOM_ASIC",
        "num_devices": 1,
        "vram_mb": 81920,
        "tdp_watts": 600.0,
        "idle_watts": 60.0,
        "system_overhead_watts": 0.0,
        "fp16_tflops": 917.0,
        "fp8_tflops": 1835.0,
        "mem_bw_gbps": 3700.0,
        "power_class": PowerClass.E_OPEN.value,
        "default_perf_cap": PerformanceCapId.CAP_DATACENTER_1000T.value,
        "hourly_cost_usd": 2.30,
        "capex_usd": 15650.0,
    },
}


HARDWARE_SKU_ALIASES: Dict[str, str] = {
    "NVIDIA_L40S_48GB": "L40S_48GB",
    "NVIDIA_L40S": "L40S_48GB",
    "L40S": "L40S_48GB",
    "NVIDIA_L4_24GB": "L4_24GB",
    "NVIDIA_L4": "L4_24GB",
    "L4": "L4_24GB",
    "GOOGLE_TPU_V6E_1": "TPU_V6E_1_32GB",
    "GOOGLE_TPU_V6E_1_32GB": "TPU_V6E_1_32GB",
    "GOOGLE_CLOUD_TPU_V6E_1": "TPU_V6E_1_32GB",
    "GOOGLE_TPU_V5E_1": "TPU_V5E_1_16GB",
    "GOOGLE_TPU_V5E_1_16GB": "TPU_V5E_1_16GB",
    "TPU_V5E_1": "TPU_V5E_1_16GB",
    "NVIDIA_RTX_5090_32GB": "RTX_5090_32GB",
    "NVIDIA_RTX_5090": "RTX_5090_32GB",
    "NVIDIA_RTX_4090_24GB": "RTX_4090_24GB",
    "NVIDIA_RTX_4090": "RTX_4090_24GB",
    "NVIDIA_JETSON_AGX_THOR_64GB": "JETSON_AGX_THOR_64GB",
    "NVIDIA_JETSON_THOR_64GB": "JETSON_AGX_THOR_64GB",
    "JETSON_THOR": "JETSON_AGX_THOR_64GB",
    "NVIDIA_JETSON_ORIN_NX_16GB": "JETSON_ORIN_NX_16GB",
    "ORIN_NX_16GB": "JETSON_ORIN_NX_16GB",
    "NVIDIA_JETSON_ORIN_AGX_64GB": "JETSON_ORIN_AGX_64GB",
    "AMD_INSTINCT_MI210_64GB": "AMD_MI210_64GB",
    "MI210_64GB": "AMD_MI210_64GB",
}


def resolve_hardware_sku(raw_sku: str) -> str:
  """Normalize case, dash/underscore variations, and vendor-prefixed SKU names to canonical HARDWARE_SKU_SPECS key."""
  cleaned = str(raw_sku or "").strip()
  if not cleaned:
    return ""
  if cleaned in HARDWARE_SKU_SPECS:
    return cleaned
  upper = cleaned.upper().replace("-", "_").replace(" ", "_")
  if upper in HARDWARE_SKU_SPECS:
    return upper
  return HARDWARE_SKU_ALIASES.get(upper, cleaned)


PROTOCOL_MIN_WARMUP_CALLS: int = 30
PROTOCOL_MIN_GOLDEN_SAMPLES: int = 100
PROTOCOL_MAX_C1_P99_P50_RATIO: float = 1.5
PROTOCOL_MAX_SLOT_COUNT_SPREAD: float = 0.15


CANONICAL_REALTIME_SLAS: Dict[str, Dict[str, Any]] = {
    "pi05-3b": {
        "track_key": "track1a",
        "track": TrackType.VLA_TRACK.value,
        "model_id": "pi05-3b",
        "locked_k_eval": 5,
        "native_horizon": 10,
        "control_freq_hz": 20.0,
        "max_p99_latency_ms": 250.0,
        "min_rate_hz": 4.0,
        "min_actions_per_sec": 20.0,
        "rate_unit": "Chunks/s",
        "throughput_unit": "Actions/s",
        "efficiency_unit": "Actions/J",
    },
    "gr00t-n1.7-3b": {
        "track_key": "track1b",
        "track": TrackType.VLA_TRACK.value,
        "model_id": "gr00t-n1.7-3b",
        "locked_k_eval": 8,
        "native_horizon": 16,
        "control_freq_hz": 20.0,
        "max_p99_latency_ms": 400.0,
        "min_rate_hz": 2.5,
        "min_actions_per_sec": 20.0,
        "rate_unit": "Chunks/s",
        "throughput_unit": "Actions/s",
        "efficiency_unit": "Actions/J",
    },
    "cosmos-predict2.5-2b": {
        "track_key": "track2",
        "track": TrackType.WM_TRACK.value,
        "model_id": "cosmos-predict2.5-2b",
        "locked_k_eval": 1,
        "horizon_frames": 16,
        "control_freq_hz": 1.0,
        "max_p99_latency_ms": 1000.0,
        "min_rate_hz": 1.0,
        "min_predictions_per_sec": 1.0,
        "rate_unit": "Predictions/s",
        "throughput_unit": "Predictions/s",
        "efficiency_unit": "Predictions/J",
    },
}


def resolve_canonical_model_id(raw_model: Optional[str], track: Optional[str] = None) -> Optional[str]:
  """Resolve canonical EEB v1.0 model ID from model ID, HF repo, shorthand, or track shorthand."""
  cand = str(raw_model or "").strip()
  if cand:
    if cand in MODERN_REFERENCE_CHECKPOINTS:
      return cand
    cand_lower = cand.lower()
    for mid, spec in MODERN_REFERENCE_CHECKPOINTS.items():
      if mid.lower() == cand_lower or str(spec.get("hf_repo", "")).lower() == cand_lower:
        return mid
      for alias in spec.get("aliases", []):
        if str(alias).lower() == cand_lower:
          return mid
    if "pi05" in cand_lower or "pi0.5" in cand_lower or cand_lower.startswith("pi0"):
      return "pi05-3b"
    if "gr00t" in cand_lower:
      return "gr00t-n1.7-3b"
    if "cosmos" in cand_lower:
      return "cosmos-predict2.5-2b"
  t_low = str(track or "").strip().lower()
  if t_low == "track1a":
    return "pi05-3b"
  if t_low == "track1b":
    return "gr00t-n1.7-3b"
  if t_low in ("track2", "track2_wm", "wm", "wm_track", "track1_world_model"):
    return "cosmos-predict2.5-2b"
  return None


def resolve_canonical_sla(track: str, model_id: Optional[str] = None) -> Dict[str, Any]:
  """Return canonical Gapless Real-Time SLA spec for a given track and optional model_id."""
  resolved_model = resolve_canonical_model_id(model_id, track=track)
  if resolved_model in CANONICAL_REALTIME_SLAS:
    return dict(CANONICAL_REALTIME_SLAS[resolved_model])
  if is_wm_track(track) or str(track).strip().lower() in ("track2", "wm"):
    return dict(CANONICAL_REALTIME_SLAS["cosmos-predict2.5-2b"])
  if str(track).strip().lower() == "track1b":
    return dict(CANONICAL_REALTIME_SLAS["gr00t-n1.7-3b"])
  return dict(CANONICAL_REALTIME_SLAS["pi05-3b"])


_WM_GATE_A_THRESHOLDS: Dict[str, float] = {
    "min_latent_action_cosine_sim": 0.990,
    "min_psnr_db": 28.0,
    "min_cvar1_psnr_db": 22.0,
    "min_cvar1_ssim": 0.80,
    "max_psnr_drop_db": 1.5,
    "max_lpips_increase": 0.03,
    "max_fvd_ratio": 1.05,
    "max_long_horizon_60s_drift_ratio": 1.20,
}

_VLA_GATE_A_THRESHOLDS: Dict[str, float] = {
    "min_action_cosine_sim": 0.995,
    "max_action_l1": 0.02,
    "max_action_mse": 0.001,
    "max_action_nrmse": 0.025,
    "min_ar_token_match_rate": 0.99,
}

# Version-indexed Quality & Compliance Gate thresholds lookup table
QUALITY_GATE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    TrackType.TRACK1_WORLD_MODEL.value: dict(_WM_GATE_A_THRESHOLDS),
    TrackType.WM_TRACK.value: dict(_WM_GATE_A_THRESHOLDS),
    TrackType.TRACK2_WM.value: dict(_WM_GATE_A_THRESHOLDS),
    TrackType.TRACK2_VLA.value: dict(_VLA_GATE_A_THRESHOLDS),
    TrackType.VLA_TRACK.value: dict(_VLA_GATE_A_THRESHOLDS),
    TrackType.TRACK1_VLA.value: dict(_VLA_GATE_A_THRESHOLDS),
    "wm_open_3part_gate": {
        "min_psnr_db": 28.0,
        "max_lpips": 0.12,
        "max_fvd": 120.0,
        "min_inverse_dynamics_action_cosine_sim": 0.990,
        "max_long_horizon_drift_ratio_30_60s": 1.20,
        "min_s2_policy_ranking_kendall_tau": 0.80,
    },
    "stage2_gate_b": {
        CompressionTier.STRICT_LOSSLESS.value: -2.0,
        CompressionTier.EXTREME_COMPRESSION.value: -10.0,
    },
    "g1_parity": {
        "min_ar_token_match_rate": 0.99,
        "min_continuous_cosine_sim": 0.995,
        "max_continuous_nrmse": 0.025,
        "min_wm_controllability_cosine_sim": 0.990,
        "max_wm_psnr_drop_db": 1.5,
        "max_wm_lpips_increase": 0.03,
        "max_wm_60s_drift_ratio": 1.20,
    },
    "g2_non_inferiority": {
        "round0_eligibility_min_delta_pts": -15.0,
        "round0_best_efficiency_award_min_delta_pts": -5.0,
        "round0_realtime_award_max_p99_ms": 100.0,
        "eeb_v1_closed_min_delta_pts": -2.0,
        "eeb_v1_closed_min_relative_ratio": 0.98,
        "eeb_v1_open_min_delta_pts": -5.0,
        "mcnemar_alpha": 0.05,
    },
    "g4_cache_memoization": {
        "perturbation_linf_eps": 1e-3,
        "min_l2_output_shift": 1e-5,
        "min_responsive_sample_fraction": 0.99,
        "min_jacobian_cosine_alignment": 0.70,
    },
    "s2_wm_evaluator": {
        "min_kendall_tau": 0.80,
        "required_policy_pool_size": 12.0,
    },
    "measurement_protocol": {
        "warmup_calls": float(PROTOCOL_MIN_WARMUP_CALLS),
        "timed_calls": 320.0,
        "timed_calls_wm": 200.0,
        "num_runs": 3.0,
        "min_golden_samples": float(PROTOCOL_MIN_GOLDEN_SAMPLES),
        "max_c1_p99_p50_ratio": PROTOCOL_MAX_C1_P99_P50_RATIO,
        "max_slot_count_spread": PROTOCOL_MAX_SLOT_COUNT_SPREAD,
        "max_cross_run_variance_pct": 5.0,
        "paced_freq_hz": 20.0,
    },
}


# Canonical Iso-Reactivity Slice targets per track (Hz -> max p99 latency in ms)
TRACK_CANONICAL_ISO_REPLAN_HZ_SLICES: Dict[str, Tuple[float, ...]] = {
    TrackType.VLA_TRACK.value: (2.5, 4.0, 10.0, 20.0, 30.0, 50.0),
    TrackType.WM_TRACK.value: (1.0, 2.0, 5.0, 10.0),
}

ISO_REACTIVITY_TARGETS_MS: Dict[str, float] = {
    "Iso-2.5Hz": 400.0,
    "Iso-4Hz": 250.0,
    "Iso-10Hz": 100.0,
    "Iso-20Hz": 50.0,
    "Iso-30Hz": 1000.0 / 30.0,  # 33.3333 ms
    "Iso-50Hz": 20.0,
}

WM_ISO_REACTIVITY_TARGETS_MS: Dict[str, float] = {
    "Iso-1Hz": 1000.0,  # Canonical Track 2 Gapless SLA cutoff (p99 <= 1000 ms, >= 1.0 Predictions/s)
    "Iso-2Hz": 500.0,
    "Iso-3Hz": 1000.0 / 3.0,
    "Iso-5Hz": 200.0,
    "Iso-10Hz": 100.0,
}

# Explicit Execution Provenance Tiers (CRITICAL-1 Honest Provenance Labeling)
LIVE_EMPIRICAL_PROVENANCE: Dict[str, Any] = {
    "tier": "LIVE_EMPIRICAL_BENCHMARK",
    "execution_tier": "TIER_1_LIVE_HTTP_CLIENT_SERVER",
    "execution_mode": "live_empirical_runner",
    "is_live_empirical": True,
    "is_analytic_projection": False,
    "label": "LIVE HTTP CLIENT-SERVER BENCHMARK (REAL TIMING & DUAL-EXECUTION G4)",
}

ANALYTIC_REFERENCE_PROVENANCE: Dict[str, Any] = {
    "tier": "ANALYTIC_REFERENCE_PROJECTION",
    "execution_tier": "TIER_2_REFERENCE_CALIBRATED_MODEL",
    "execution_mode": "analytic_reference_model",
    "is_live_empirical": False,
    "is_analytic_projection": True,
    "label": "ANALYTIC ROOFLINE / REFERENCE PROJECTION (SEED CURVE - NOT LIVE HARDWARE RUN)",
}


# Concrete worked examples for every schema field (Gate 6: Write For The Reader)
SCHEMA_FIELD_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "recipe_id": {
        "description": "Unique identifier for the declarative optimization recipe.",
        "example": "track1_vla_pi05_edge100t",
    },
    "clock": {
        "description": "Benchmark clock ('corl2026_sprint' / 'round0_orin_nx_sprint' for CoRL 2026 sprint or 'eeb_v1' for EEB v1 living benchmark).",
        "example": "corl2026_sprint",
    },
    "scenario": {
        "description": "Unified EEB scenario ('P1_ONBOARD', 'P2_FLEET', 'P3_EDGE_CLOUD', 'S1_INTERACTIVE_WM', or 'S2_WM_EVALUATOR').",
        "example": "P1_ONBOARD",
    },
    "division": {
        "description": "Benchmark division ('CLOSED' for frozen model/steps/chunk/resolution or 'OPEN' for algorithmic co-design/distillation).",
        "example": "OPEN",
    },
    "perf_cap": {
        "description": "Device-agnostic performance cap envelope ('CAP_EDGE_100T', 'CAP_WORKSTATION_350T', 'CAP_DATACENTER_1000T', or 'CAP_UNCONSTRAINED').",
        "example": "CAP_EDGE_100T",
    },
    "track": {
        "description": "EmbodiedX evaluation track ('VLA_TRACK' / 'track2_vla' for Real-Time VLA Policies or 'WM_TRACK' / 'track1_world_model' for Interactive World Models).",
        "example": "VLA_TRACK",
    },
    "execution_id": {
        "description": "UUID/hash binding throughput latency measurements and Co-Located Quality Gates G1-G4 to the exact same physical execution.",
        "example": "exec-20261006-225800-a91f4c2b",
    },
    "status": {
        "description": "Run lifecycle state ('PENDING', 'RUNNING', 'CERTIFIED', 'REJECTED', 'BLOCKED', 'FAILED', 'SKIPPED', 'INVALID').",
        "example": "CERTIFIED",
    },
    "reason": {
        "description": "Required non-empty human-readable explanation whenever status is REJECTED, BLOCKED, FAILED, SKIPPED, or INVALID.",
        "example": "Gate A failed: WM latent_action_cosine_sim=0.9510 < required 0.9800. Reduce quantization group size or increase denoising steps N.",
    },
    "hz_plan": {
        "description": "World Model p99 Rollout Rate (Predictions/s) computed from pooled steady-state 99th-percentile H-frame rollout latency across active slots: 1000.0 / rollout_latency_p99_ms.",
        "example": 18.42,
    },
    "hz_react": {
        "description": "VLA Policy p99 Chunk Rate (Chunks/s) [1000 / L_p99, ms] computed from pooled steady-state 99th-percentile chunk latency across active slots: 1000.0 / ttfa_p99_ms (disambiguated from K/L executed actions/s).",
        "example": 34.85,
    },
    "iso_reactivity_slices": {
        "description": "Log-linearly interpolated throughput at fixed p99 reactivity thresholds (Iso-10Hz <= 100ms, Iso-20Hz <= 50ms, Iso-30Hz <= 33.33ms, Iso-50Hz <= 20ms).",
        "example": {"Iso-10Hz": 198.4, "Iso-20Hz": 124.1, "Iso-30Hz": 72.5, "Iso-50Hz": None},
    },
}


def validate_status_and_reason(
    status: str | RunStatus,
    reason: Optional[str] = None,
    previous_status: Optional[str | RunStatus] = None,
) -> RunStatus:
  """Validate state value, transition legality, and mandatory non-empty reason field."""
  try:
    resolved = status if isinstance(status, RunStatus) else RunStatus(status)
  except ValueError as exc:
    valid_values = ", ".join(s.value for s in RunStatus)
    raise ValueError(
        f"Invalid RunStatus {status!r}. Permitted values are: [{valid_values}]. "
        "Update the 'status' field to one of the permitted RunStatus values."
    ) from exc

  if previous_status is not None:
    prev = (
        previous_status
        if isinstance(previous_status, RunStatus)
        else RunStatus(previous_status)
    )
    allowed_next = VALID_STATE_TRANSITIONS.get(prev, frozenset())
    if resolved not in allowed_next:
      allowed_str = ", ".join(s.value for s in sorted(allowed_next, key=lambda x: x.value))
      raise ValueError(
          f"Illegal state transition from {prev.value} to {resolved.value}. "
          f"Allowed transitions from {prev.value} are: [{allowed_str}]."
      )

  if resolved in REASON_REQUIRED_STATUSES:
    if reason is None or not str(reason).strip():
      raise ValueError(
          f"Status {resolved.value} requires a non-empty 'reason' explaining what failed "
          "and how to remediate it. Pass reason='<actionable explanation>'."
      )
  return resolved


# Prohibited FLOP-reducing optimization keywords in CLOSED division
CLOSED_DIVISION_FORBIDDEN_KEYWORDS: Tuple[str, ...] = (
    "distill",
    "prune",
    "pruning",
    "token_drop",
    "token_merging",
    "tome",
    "early_exit",
    "layer_skip",
    "dit_block_feature_cache",
    "ditcache",
    "step_reduction",
    "action_chunk_interpolation_skip",
    "rtc_inpainting",
    "cpu_text_encoder_offload_cache",
    "streaming_first_frame_vae_decode",
)


import re as _re

SAFE_SLUG_PATTERN = _re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_SHA1_40_PATTERN = _re.compile(r"^[0-9a-f]{40}$")


def validate_safe_identifier(value: str, field_name: str = "recipe_id") -> str:
  """Reject path traversal, hidden dotfiles, whitespace, markdown table metacharacters, or overlong slugs."""
  raw = str(value or "")
  if (
      not SAFE_SLUG_PATTERN.fullmatch(raw)
      or raw in (".", "..")
      or raw.startswith(".")
      or ".." in raw
  ):
    raise ValueError(
        f"Invalid {field_name}={value!r}. Must match ^[A-Za-z0-9_.-]{{1,80}}$, "
        f"must not start with '.', and must not contain '..'."
    )
  return raw


@dataclass
class RecipeConfig:
  """Declarative recipe configuration supporting CoRL 2026 / EEB v1 Scenarios (`P1`/`P2`/`P3`/`S1`/`S2`), `CLOSED`/`OPEN` Divisions, and Device-Agnostic `PerformanceCapEnvelope`s."""

  recipe_id: str
  track: str
  model_family: str
  model_name: str
  baseline_model_id: str
  precision: str
  hardware_sku: str
  hf_repo_id: Optional[str] = None
  hf_revision: Optional[str] = None
  hf_subdir: Optional[str] = None
  compression_tier: str = CompressionTier.STRICT_LOSSLESS.value
  # Two-Clock, Unified Scenario, Division, and Device-Agnostic Performance Cap fields
  clock: str = BenchmarkClock.EEB_V1.value
  scenario: Optional[str] = None
  division: str = DivisionType.CLOSED.value
  power_class: Optional[str] = None
  perf_cap: Optional[str] = None
  perf_cap_mode: Optional[str] = None
  num_devices: int = 1
  mps_active_thread_pct: Optional[float] = None
  clock_scale_factor: Optional[float] = None
  vram_fraction_cap: Optional[float] = None
  execution_strategy: str = ExecutionStrategy.ASYNC_CHUNK.value
  # World Model (WM_TRACK) parameters
  horizon_frames: int = 16
  stream_chunk_frames: Optional[int] = None
  denoising_steps: int = 4
  reference_denoising_steps: int = 25
  guidance_scale: float = 1.0
  batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
  session_duration_s: float = 60.0
  # VLA Policy (VLA_TRACK) parameters
  action_chunk_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
  native_action_chunk_size: int = 10
  execute_horizon: Optional[int] = None
  concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
  action_dim: int = 7
  action_head_type: str = "continuous_flow"  # 'ar_discrete' or 'continuous_flow'
  control_freq_hz: float = 20.0
  chunk_horizon_s: float = 0.40
  # Closed vs Open Division invariants
  weights_frozen: bool = True
  reference_chunk_size: int = 8
  resolution_hw: int = 224
  reference_resolution_hw: int = 224
  num_cameras: int = 2
  reference_num_cameras: int = 2
  declared_training_compute_gpu_hours: Optional[float] = 0.0
  declared_inference_flops_gflops: Optional[float] = 120.0
  osi_license: str = "Apache-2.0"
  # P3 Edge-Cloud Split network emulation parameters
  network_rtt_ms: float = 25.0
  network_jitter_ms: float = 5.0
  network_drop_rate: float = 0.01
  # Optimization & calibration metadata
  optimizations: List[str] = field(default_factory=list)
  speedup_factor_hint: float = 1.0
  quality_noise_scale: float = 0.002
  stage2_task_suite: str = "LIBERO-All"
  stage2_delta_success_pct: float = -0.5
  notes: Optional[str] = None

  def resolved_scenario(self) -> str:
    """Return explicit scenario or default scenario derived from track, perf_cap, and hardware SKU."""
    if self.scenario:
      return self.scenario
    if is_wm_track(self.track):
      return ScenarioType.S1_INTERACTIVE_WM.value
    sku_spec = HARDWARE_SKU_SPECS.get(self.hardware_sku, {})
    resolved_cap = resolve_perf_cap_id(self.perf_cap) if self.perf_cap else None
    if (
        sku_spec.get("vendor") == "NVIDIA_TEGRA"
        or self.clock in (BenchmarkClock.ROUND0_ORIN_NX_SPRINT.value, BenchmarkClock.CORL2026_SPRINT.value)
        or resolved_cap == PerformanceCapId.CAP_EDGE_100T.value
    ):
      return ScenarioType.P1_ONBOARD.value
    return ScenarioType.P2_FLEET.value

  def resolved_power_class(self) -> str:
    """Return explicit power class or SKU default power class."""
    if self.power_class:
      return self.power_class
    sku_spec = HARDWARE_SKU_SPECS.get(self.hardware_sku, {})
    return str(sku_spec.get("power_class", PowerClass.E_OPEN.value))

  def resolved_perf_cap(self) -> str:
    """Return explicit PerformanceCapEnvelope ID or infer from power class / hardware SKU (defaulting to CAP_SINGLE_CHIP)."""
    if self.perf_cap:
      return resolve_perf_cap_id(self.perf_cap)
    sku_spec = HARDWARE_SKU_SPECS.get(self.hardware_sku, {})
    if "default_perf_cap" in sku_spec:
      return resolve_perf_cap_id(str(sku_spec["default_perf_cap"]))
    pclass = self.resolved_power_class()
    pclass_spec = POWER_CLASS_SPECS.get(pclass, {})
    return resolve_perf_cap_id(str(pclass_spec.get("default_perf_cap", PerformanceCapId.CAP_SINGLE_CHIP.value)))

  def validate(self) -> None:
    """Validate recipe fields, single-device invariant (num_devices == 1), scenarios, divisions, performance caps, and LIBERO-RT horizon caps."""
    validate_safe_identifier(self.recipe_id, field_name="recipe_id")
    self.hardware_sku = resolve_hardware_sku(self.hardware_sku)
    if self.hf_revision is not None and str(self.hf_revision).strip():
      rev_norm = str(self.hf_revision).strip().lower()
      if not _SHA1_40_PATTERN.fullmatch(rev_norm):
        raise ValueError(
            f"Invalid hf_revision={self.hf_revision!r} in recipe {self.recipe_id!r}. "
            "Must be an immutable 40-character lowercase hex HuggingFace commit SHA."
        )
      self.hf_revision = rev_norm
    if int(self.num_devices) != 1:
      raise ValueError(
          f"Invalid num_devices={self.num_devices} in recipe {self.recipe_id!r}. "
          "EmbodiedX strictly enforces single-device execution (num_devices == 1: single chip / single package) "
          "bounded by a (TOPS, HBM) PerformanceCapEnvelope."
      )
    if not (is_vla_track(self.track) or is_wm_track(self.track)):
      valid_tracks = ", ".join(t.value for t in TrackType)
      raise ValueError(
          f"Invalid track={self.track!r} in recipe {self.recipe_id!r}. "
          f"Supported tracks: [{valid_tracks}]."
      )
    if self.baseline_model_id.lower() in DEPRECATED_BASELINE_MODEL_IDS:
      raise ValueError(
          f"Deprecated 2024 7-token-autoregressive baseline_model_id={self.baseline_model_id!r} in recipe {self.recipe_id!r}. "
          "Use modern 2025/2026 VLA baselines: 'gr00t-n1.7-3b' ('nvidia/GR00T-N1.7-LIBERO'), "
          "'pi05-3b' ('lerobot/pi05_libero_finetuned_v044'), "
          "or 'openvla-oft-7b' ('moojink/openvla-7b-oft-finetuned-libero')."
      )
    if self.hardware_sku not in HARDWARE_SKU_SPECS:
      valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
      raise ValueError(
          f"Unknown hardware_sku={self.hardware_sku!r} in recipe {self.recipe_id!r}. "
          f"Supported SKUs in HARDWARE_SKU_SPECS: [{valid_skus}]."
      )
    if self.compression_tier not in (
        CompressionTier.STRICT_LOSSLESS.value,
        CompressionTier.EXTREME_COMPRESSION.value,
    ):
      raise ValueError(
          f"Invalid compression_tier={self.compression_tier!r} in recipe {self.recipe_id!r}. "
          f"Use '{CompressionTier.STRICT_LOSSLESS.value}' or '{CompressionTier.EXTREME_COMPRESSION.value}'."
      )
    valid_clocks = {c.value for c in BenchmarkClock}
    if self.clock not in valid_clocks:
      raise ValueError(
          f"Invalid clock={self.clock!r} in recipe {self.recipe_id!r}. "
          f"Supported clocks: {sorted(valid_clocks)}."
      )
    scen = self.resolved_scenario()
    valid_scenarios = {s.value for s in ScenarioType}
    if scen not in valid_scenarios:
      raise ValueError(
          f"Invalid scenario={scen!r} in recipe {self.recipe_id!r}. "
          f"Supported scenarios: {sorted(valid_scenarios)}."
      )
    valid_divisions = {d.value for d in DivisionType}
    if self.division not in valid_divisions:
      raise ValueError(
          f"Invalid division={self.division!r} in recipe {self.recipe_id!r}. "
          f"Supported divisions: {sorted(valid_divisions)}."
      )
    pclass = self.resolved_power_class()
    if pclass not in POWER_CLASS_SPECS:
      raise ValueError(
          f"Invalid power_class={pclass!r} in recipe {self.recipe_id!r}. "
          f"Supported power classes: {sorted(POWER_CLASS_SPECS.keys())}."
      )
    pcap = self.resolved_perf_cap()
    if pcap not in PERFORMANCE_CAP_ENVELOPES:
      raise ValueError(
          f"Invalid perf_cap={pcap!r} in recipe {self.recipe_id!r}. "
          f"Supported performance cap envelopes: {sorted(PERFORMANCE_CAP_ENVELOPES.keys())}."
      )
    if self.perf_cap_mode is not None:
      valid_modes = {m.value for m in PerfCapSatisfactionMode}
      if self.perf_cap_mode not in valid_modes:
        raise ValueError(
            f"Invalid perf_cap_mode={self.perf_cap_mode!r} in recipe {self.recipe_id!r}. "
            f"Supported modes: {sorted(valid_modes)}."
        )
    valid_strategies = {e.value for e in ExecutionStrategy}
    if self.execution_strategy not in valid_strategies:
      raise ValueError(
          f"Invalid execution_strategy={self.execution_strategy!r} in recipe {self.recipe_id!r}. "
          f"Supported strategies: {sorted(valid_strategies)}."
      )
    max_horizon = LIBERO_RT_CONSTANTS["max_chunk_horizon_s"]
    if float(self.chunk_horizon_s) > max_horizon + 1e-9:
      raise ValueError(
          f"chunk_horizon_s={self.chunk_horizon_s} exceeds LIBERO-RT cap of {max_horizon}s in recipe {self.recipe_id!r}. "
          "Open-loop action chunks longer than 1.0s are prohibited to prevent masking stale reactivity."
      )
    validate_division_compliance(self)

  def to_dict(self) -> Dict[str, Any]:
    d = asdict(self)
    if self.execute_horizon is None:
      d.pop("execute_horizon", None)
    if self.stream_chunk_frames is None:
      d.pop("stream_chunk_frames", None)
    d["scenario"] = self.resolved_scenario()
    d["power_class"] = self.resolved_power_class()
    d["perf_cap"] = self.resolved_perf_cap()
    return d

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "RecipeConfig":
    known_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_keys}
    if "action_chunk_sizes" not in filtered and "chunk_sizes" in data and data["chunk_sizes"] is not None:
      filtered["action_chunk_sizes"] = [int(x) for x in data["chunk_sizes"]]
    if "division" not in filtered:
      track_val = str(filtered.get("track", ""))
      opts = [str(o).lower() for o in filtered.get("optimizations", [])]
      has_forbidden_opt = any(kw in o for o in opts for kw in CLOSED_DIVISION_FORBIDDEN_KEYWORDS)
      wm_step_reduced = is_wm_track(track_val) and (
          int(filtered.get("denoising_steps", 25)) != int(filtered.get("reference_denoising_steps", 25))
      )
      not_frozen = filtered.get("weights_frozen") is False
      filtered["division"] = (
          DivisionType.OPEN.value
          if (has_forbidden_opt or wm_step_reduced or not_frozen)
          else DivisionType.CLOSED.value
      )
    if (
        filtered.get("division") == DivisionType.CLOSED.value
        and filtered.get("execute_horizon") is not None
        and "action_chunk_sizes" not in data
        and "chunk_sizes" not in data
    ):
      filtered["action_chunk_sizes"] = [int(filtered["execute_horizon"])]
    cfg = cls(**filtered)
    cfg.validate()
    return cfg


def validate_division_compliance(recipe: RecipeConfig) -> List[str]:
  """Verify CLOSED vs OPEN division rules and raise ValueError on violations.

  - CLOSED division requires frozen weights, identical denoising_steps to reference_denoising_steps,
    identical resolution_hw and num_cameras, and zero FLOP-reducing model hacks (distillation, pruning,
    token dropping, DiT block caching, or step reduction).
  - OPEN division permits distillation, step reduction, chunking, and WAMs, but requires non-None
    declared training/inference compute and a non-empty OSI license.
  """
  violations: List[str] = []
  if recipe.division == DivisionType.CLOSED.value:
    if not recipe.weights_frozen:
      violations.append(
          "CLOSED division requires weights_frozen=True (no fine-tuning or distillation)."
      )
    if is_wm_track(recipe.track) and (
        recipe.denoising_steps != recipe.reference_denoising_steps
    ):
      violations.append(
          f"CLOSED division requires denoising_steps ({recipe.denoising_steps}) == "
          f"reference_denoising_steps ({recipe.reference_denoising_steps}). Move step-reduced recipes to OPEN division."
      )
    if is_wm_track(recipe.track) and (recipe.stream_chunk_frames or 0) > 0:
      violations.append(
          f"CLOSED division prohibits temporal stream chunking (stream_chunk_frames={recipe.stream_chunk_frames} > 0) "
          "because Cosmos-Predict2.5-2B is a bidirectional full-clip 3D-DiT where TTFF == L_clip. "
          "Set stream_chunk_frames=0/None in CLOSED division or declare division='OPEN' for causal chunked world models."
      )
    if recipe.resolution_hw != recipe.reference_resolution_hw:
      violations.append(
          f"CLOSED division requires resolution_hw ({recipe.resolution_hw}) == "
          f"reference_resolution_hw ({recipe.reference_resolution_hw})."
      )
    if recipe.num_cameras != recipe.reference_num_cameras:
      violations.append(
          f"CLOSED division requires num_cameras ({recipe.num_cameras}) == "
          f"reference_num_cameras ({recipe.reference_num_cameras})."
      )
    for opt in recipe.optimizations:
      opt_lower = opt.lower()
      for kw in CLOSED_DIVISION_FORBIDDEN_KEYWORDS:
        if kw in opt_lower:
          violations.append(
              f"CLOSED division forbids FLOP-reducing model modification {opt!r} (matched {kw!r}). "
              "Declare division='OPEN' for algorithmic co-design."
          )
          break
  elif recipe.division == DivisionType.OPEN.value:
    if recipe.declared_training_compute_gpu_hours is None or float(recipe.declared_training_compute_gpu_hours) < 0.0:
      violations.append(
          "OPEN division requires declared_training_compute_gpu_hours >= 0.0."
      )
    if recipe.declared_inference_flops_gflops is None or float(recipe.declared_inference_flops_gflops) <= 0.0:
      violations.append(
          "OPEN division requires declared_inference_flops_gflops > 0.0."
      )
    if not recipe.osi_license or not str(recipe.osi_license).strip():
      violations.append(
          "OPEN division requires a non-empty osi_license (e.g. 'Apache-2.0' or 'MIT')."
      )

  if violations:
    raise ValueError(
        f"Division compliance check failed for recipe {recipe.recipe_id!r} (division={recipe.division}): "
        + " ".join(violations)
    )
  return violations


def load_yaml_or_json(path: str | Path) -> Dict[str, Any]:
  """Load a declarative recipe or JSON state file with fallback parser."""
  file_path = Path(path)
  if not file_path.exists():
    raise FileNotFoundError(
        f"Configuration file not found at {file_path}. "
        "Verify the --recipe or --inputs path points to an existing YAML/JSON file."
    )
  text = file_path.read_text(encoding="utf-8")
  if file_path.suffix.lower() == ".json":
    return json.loads(text)
  if yaml is not None:
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
      raise ValueError(
          f"Expected a YAML mapping in {file_path}, got {type(loaded).__name__}."
      )
    return loaded
  return json.loads(text)


def _lock_path_for(target: Path) -> Path:
  import hashlib

  resolved = str(target.resolve())
  digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:32]
  uid = getattr(os, "getuid", lambda: 0)()
  lock_dir = Path(tempfile.gettempdir()) / f"embodied_bench_locks_{uid}"
  lock_dir.mkdir(parents=True, exist_ok=True)
  return lock_dir / f"{target.name}.{digest}.lock"


def cleanup_orphan_temp_files(directory: str | Path, prefix: str = ".tmp_embodied_") -> int:
  """Remove orphan temporary files left behind by a crashed writer process (Gate 1 Crash Path)."""
  dir_path = Path(directory)
  if not dir_path.exists():
    return 0
  removed = 0
  for item in dir_path.iterdir():
    if item.is_file() and item.name.startswith(prefix):
      try:
        item.unlink()
        removed += 1
      except OSError:
        pass
  return removed


def atomic_write_json(target_path: str | Path, payload: Dict[str, Any]) -> None:
  """Atomically write JSON with fcntl.flock, tempfile.mkstemp, os.fsync, and os.replace."""
  target = Path(target_path)
  target.parent.mkdir(parents=True, exist_ok=True)
  lock_file = _lock_path_for(target)

  with open(lock_file, "a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    try:
      cleanup_orphan_temp_files(target.parent, prefix=f".tmp_embodied_{target.name}_")
      fd, tmp_name = tempfile.mkstemp(
          prefix=f".tmp_embodied_{target.name}_",
          suffix=".json",
          dir=str(target.parent),
      )
      tmp_path = Path(tmp_name)
      try:
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_fp:
          tmp_fp.write(serialized)
          tmp_fp.flush()
          os.fsync(tmp_fp.fileno())
        os.replace(str(tmp_path), str(target))
      except BaseException:
        if tmp_path.exists():
          try:
            tmp_path.unlink()
          except OSError:
            pass
        raise
    finally:
      fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def atomic_write_text(target_path: str | Path, text: str, suffix: str = ".tmp") -> None:
  """Atomically write UTF-8 text with fcntl.flock, orphan cleanup, tempfile.mkstemp, os.fsync, and os.replace."""
  target = Path(target_path)
  target.parent.mkdir(parents=True, exist_ok=True)
  lock_file = _lock_path_for(target)

  with open(lock_file, "a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    try:
      cleanup_orphan_temp_files(target.parent, prefix=f".tmp_embodied_{target.name}_")
      fd, tmp_name = tempfile.mkstemp(
          prefix=f".tmp_embodied_{target.name}_",
          suffix=suffix,
          dir=str(target.parent),
      )
      tmp_path = Path(tmp_name)
      try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_fp:
          tmp_fp.write(text)
          tmp_fp.flush()
          os.fsync(tmp_fp.fileno())
        os.replace(str(tmp_path), str(target))
      except BaseException:
        if tmp_path.exists():
          try:
            tmp_path.unlink()
          except OSError:
            pass
        raise
    finally:
      fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


SUBMISSION_PHASE_VALUES = frozenset({
    "COMMIT_ONLY",
    "OPEN_RECIPE",
    "PROPRIETARY_AUDIT_ONLY",
})


def validate_submission_meta(
    meta: Dict[str, Any],
    *,
    require_commit_hash: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
  """Validate `submission_meta.json` used in Two-Phase Commit-Reveal and rolling leaderboard PRs."""
  import re

  violations: List[str] = []
  if not isinstance(meta, dict):
    return False, ["submission_meta.json root value must be a JSON object."], {}

  raw_sku = resolve_hardware_sku(str(meta.get("hardware_sku", "") or "").strip())
  if not raw_sku:
    violations.append("submission_meta.json missing required 'hardware_sku'.")
  elif raw_sku not in HARDWARE_SKU_SPECS:
    valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
    violations.append(
        f"submission_meta.json unknown hardware_sku={raw_sku!r}. Supported SKUs: [{valid_skus}]."
    )

  raw_div = str(meta.get("division", "") or "").strip().upper()
  valid_divs = {d.value for d in DivisionType}
  if not raw_div:
    violations.append("submission_meta.json missing required 'division' ('closed' or 'open').")
  elif raw_div not in valid_divs:
    violations.append(
        f"submission_meta.json invalid division={meta.get('division')!r}. Expected 'CLOSED' or 'OPEN'."
    )

  engine_tag_raw = meta.get("engine_tag")
  engine_tag = str(engine_tag_raw).strip() if engine_tag_raw is not None else ""
  if engine_tag_raw is not None and (not isinstance(engine_tag_raw, str) or not engine_tag):
    violations.append("submission_meta.json 'engine_tag' must be a non-empty string when provided.")
  elif engine_tag:
    try:
      validate_safe_identifier(engine_tag, field_name="engine_tag")
    except ValueError as exc:
      violations.append(f"submission_meta.json {exc}")

  team_raw = meta.get("team", meta.get("organization"))
  team = str(team_raw).strip() if team_raw is not None else ""
  if team_raw is not None and (not isinstance(team_raw, str) or not team):
    violations.append("submission_meta.json 'team' must be a non-empty string when provided.")
  elif team:
    try:
      validate_safe_identifier(team, field_name="team")
    except ValueError as exc:
      violations.append(f"submission_meta.json {exc}")

  for opt_slug_key in ("recipe_id", "model_id"):
    opt_val = meta.get(opt_slug_key)
    if opt_val is not None:
      s_val = str(opt_val).strip()
      if s_val:
        try:
          validate_safe_identifier(s_val, field_name=opt_slug_key)
        except ValueError as exc:
          violations.append(f"submission_meta.json {exc}")

  phase_raw = meta.get("submission_phase")
  if phase_raw is not None:
    phase = str(phase_raw).strip().upper()
    if phase not in SUBMISSION_PHASE_VALUES:
      violations.append(
          f"submission_meta.json invalid submission_phase={phase_raw!r}. "
          f"Supported values: {sorted(SUBMISSION_PHASE_VALUES)}."
      )
  else:
    phase = "COMMIT_ONLY" if require_commit_hash else "OPEN_RECIPE"

  raw_commit = meta.get("recipe_commit_sha256")
  commit_hex: Optional[str] = None
  needs_commit = require_commit_hash or phase == "COMMIT_ONLY"
  if raw_commit is None or (isinstance(raw_commit, str) and not raw_commit.strip()):
    if needs_commit:
      violations.append(
          "submission_meta.json requires non-empty 64-character hex 'recipe_commit_sha256' "
          "(SHA-256 digest of private reproducibility archive or container image) for Phase 1 Commit-Reveal PRs."
      )
  elif not isinstance(raw_commit, str):
    violations.append("submission_meta.json 'recipe_commit_sha256' must be a 64-character hex SHA-256 string.")
  else:
    cand = raw_commit.strip().lower()
    if cand.startswith("sha256:"):
      cand = cand[len("sha256:") :]
    if not re.fullmatch(r"[0-9a-f]{64}", cand):
      violations.append(
          f"submission_meta.json invalid recipe_commit_sha256={raw_commit!r}: expected 64 hex characters."
      )
    elif cand == "0" * 64:
      violations.append(
          "submission_meta.json invalid recipe_commit_sha256: all-zero placeholder digest is not permitted."
      )
    else:
      commit_hex = cand

  normalized: Dict[str, Any] = {
      "schema_version": str(meta.get("schema_version", "1.0.0")),
      "hardware_sku": raw_sku,
      "division": raw_div,
      "engine_tag": engine_tag or None,
      "team": team or None,
      "submission_phase": phase,
      "recipe_commit_sha256": commit_hex,
  }
  return len(violations) == 0, violations, normalized


def atomic_read_modify_write_json(
    target_path: str | Path,
    modifier_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    default_factory: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
  """Crash-safe, concurrent-safe read-modify-write under a single exclusive file lock (Gate 1).

  - Cleans up any orphan `.tmp_embodied_*` files left by crashed processes.
  - Recovers gracefully if the existing target file is corrupt or zero-byte by
    falling back to `default_factory()` and recording `recovered_from_corrupt_state=True`.
  - Holds `fcntl.LOCK_EX` across the entire read -> modify -> mkstemp -> fsync -> replace cycle.
  """
  target = Path(target_path)
  target.parent.mkdir(parents=True, exist_ok=True)
  lock_file = _lock_path_for(target)

  with open(lock_file, "a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    try:
      cleanup_orphan_temp_files(target.parent, prefix=f".tmp_embodied_{target.name}_")
      current_state: Dict[str, Any]
      if target.exists():
        try:
          raw = target.read_text(encoding="utf-8").strip()
          if not raw:
            raise json.JSONDecodeError("Empty file", "", 0)
          loaded = json.loads(raw)
          if not isinstance(loaded, dict):
            raise ValueError("Root JSON value is not an object")
          current_state = loaded
        except (json.JSONDecodeError, ValueError):
          current_state = default_factory() if default_factory else {}
          current_state["recovered_from_corrupt_state"] = True
      else:
        current_state = default_factory() if default_factory else {}

      updated_state = modifier_fn(current_state)

      fd, tmp_name = tempfile.mkstemp(
          prefix=f".tmp_embodied_{target.name}_",
          suffix=".json",
          dir=str(target.parent),
      )
      tmp_path = Path(tmp_name)
      try:
        serialized = json.dumps(updated_state, indent=2, sort_keys=True) + "\n"
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_fp:
          tmp_fp.write(serialized)
          tmp_fp.flush()
          os.fsync(tmp_fp.fileno())
        os.replace(str(tmp_path), str(target))
      except BaseException:
        if tmp_path.exists():
          try:
            tmp_path.unlink()
          except OSError:
            pass
        raise
      return updated_state
    finally:
      fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def upsert_run_record(
    registry_path: str | Path,
    run_id: str,
    status: str | RunStatus,
    metrics: Dict[str, Any],
    reason: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
  """Upsert a run record into a shared registry JSON without stale optional field carryover (Gate 1 Repeat Path)."""

  def _modify(state: Dict[str, Any]) -> Dict[str, Any]:
    runs = state.setdefault("runs", {})
    existing = runs.get(run_id)
    prev_status = existing.get("status") if isinstance(existing, dict) else None
    resolved_status = validate_status_and_reason(
        status=status,
        reason=reason,
        previous_status=prev_status,
    )
    record: Dict[str, Any] = {
        "run_id": run_id,
        "status": resolved_status.value,
        "updated_at": now_pacific_iso(),
        "metrics": dict(metrics),
        "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None,
        "notes": notes.strip() if isinstance(notes, str) and notes.strip() else None,
    }
    runs[run_id] = record
    state["updated_at"] = now_pacific_iso()
    return state

  return atomic_read_modify_write_json(
      registry_path,
      modifier_fn=_modify,
      default_factory=lambda: {"schema_version": "1.0.0", "runs": {}},
  )
