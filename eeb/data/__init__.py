"""``eeb.data`` - EmbodiedX-Traces-v1: LeRobot v2/v3 loader, trace packer, G4 perturbations, golden outputs.

Import cost is stdlib + NumPy only; ``pyarrow`` (parquet), ``av`` (video), ``huggingface_hub``
(download) and Pillow (PNG) are imported lazily behind guards with install hints.
"""

from eeb.data.download import DEFAULT_CACHE_DIR, DownloadPlan, plan_episode_files, probe_hub_layout
from eeb.data.download import download as download_dataset  # `eeb.data.download` stays the submodule
from eeb.data.golden import (
    GoldenEpisode,
    action_cosine_similarity,
    action_l2_error,
    generate_golden_reference,
    load_golden,
    save_golden,
    verify_golden,
)
from eeb.data.lerobot_loader import LeRobotDataset, LeRobotEpisode, LeRobotInfo, MissingDependencyError, decode_video_frames, detect_layout
from eeb.data.pack import PackResult, load_split, pack_traces, verify_traces
from eeb.data.perturb import PerturbationConfig, PerturbedStep, PerturbedTrace, perturb_step, perturb_trace, write_perturbed_split
from eeb.data.schema import (
    DEFAULT_HIDDEN_FRACTION,
    DEFAULT_SPLIT_SEED,
    EMBODIEDX_TRACES_V1_SOURCES,
    SCHEMA_VERSION,
    VERIFIED_HUB_DATASETS,
    EmbodiedXTrace,
    HubDataset,
    TraceManifest,
    TraceStep,
    assign_split,
    make_episode_id,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_HIDDEN_FRACTION",
    "DEFAULT_SPLIT_SEED",
    "DownloadPlan",
    "EMBODIEDX_TRACES_V1_SOURCES",
    "EmbodiedXTrace",
    "GoldenEpisode",
    "HubDataset",
    "LeRobotDataset",
    "LeRobotEpisode",
    "LeRobotInfo",
    "MissingDependencyError",
    "PackResult",
    "PerturbationConfig",
    "PerturbedStep",
    "PerturbedTrace",
    "SCHEMA_VERSION",
    "TraceManifest",
    "TraceStep",
    "VERIFIED_HUB_DATASETS",
    "action_cosine_similarity",
    "action_l2_error",
    "assign_split",
    "decode_video_frames",
    "detect_layout",
    "download_dataset",
    "generate_golden_reference",
    "load_golden",
    "load_split",
    "make_episode_id",
    "pack_traces",
    "perturb_step",
    "perturb_trace",
    "plan_episode_files",
    "probe_hub_layout",
    "save_golden",
    "verify_golden",
    "verify_traces",
    "write_perturbed_split",
]
