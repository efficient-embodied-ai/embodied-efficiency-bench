"""Reference Baseline Model Runners for EmbodiedX (`embodied-efficiency-bench` / CoRL 2026).

Exports executable reference baseline runners for EEB v1.0 canonical models and registered checkpoints:
  - `Pi05OpenPiRunner` (`pi05-3b` / `lerobot/pi05_libero_finetuned_v044`)
  - `GR00TN17Runner` (`gr00t-n1.7-3b` / `nvidia/GR00T-N1.7-3B`)
  - `CosmosPredict25WorldModelRunner` (`cosmos-predict2.5-2b` / `nvidia/Cosmos-Predict2.5-2B`)
  - `Cosmos3WorldModelRunner` (`cosmos3-edge` / `nvidia/Cosmos3-Edge`)
  - `Qwen3OmniRunner` (`qwen3-omni-30b-a3b` / `Qwen/Qwen3-Omni-30B-A3B-Instruct`)
"""

from embodied_bench.baselines.base_runner import (
    SUPPORTED_BACKENDS,
    SUPPORTED_PRECISIONS,
    SUPPORTED_WEIGHTS_MODES,
    BaseEmbodiedModelRunner,
    BaselineRunnerConfig,
    create_baseline_runner,
    quantize_dequantize_array,
    record_runner_benchmark_state,
    resolve_canonical_model_id,
)
from embodied_bench.baselines.cosmos3_wm_runner import (
    Cosmos3WorldModelRunner,
    CosmosPredict25WorldModelRunner,
)
from embodied_bench.baselines.gr00t_n17_runner import GR00TN17Runner
from embodied_bench.baselines.pi05_openpi_runner import Pi05OpenPiRunner
from embodied_bench.baselines.qwen3_omni_runner import Qwen3OmniRunner

__all__ = [
    "SUPPORTED_BACKENDS",
    "SUPPORTED_PRECISIONS",
    "SUPPORTED_WEIGHTS_MODES",
    "BaseEmbodiedModelRunner",
    "BaselineRunnerConfig",
    "Cosmos3WorldModelRunner",
    "CosmosPredict25WorldModelRunner",
    "GR00TN17Runner",
    "Pi05OpenPiRunner",
    "Qwen3OmniRunner",
    "create_baseline_runner",
    "quantize_dequantize_array",
    "record_runner_benchmark_state",
    "resolve_canonical_model_id",
]
