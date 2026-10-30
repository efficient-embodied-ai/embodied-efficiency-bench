# GR00T-N1.7-3B Reference Server (`isaac-gr00t` PyTorch CUDA / ROCm / TorchTPU)

Canonical Track 1B (`VLA_TRACK`) reference server (`eeb.servers.gr00t:Gr00tN17Policy`) for `nvidia/GR00T-N1.7-LIBERO` (`nvidia/GR00T-N1.7-3B`: `Cosmos-Reason2-2B` / `Qwen3-VL-2B` VLM backbone + 16-step DiT flow action head) under the locked evaluation contract (`2 x 256x256` RGB, `horizon = 16`, locked `execute_horizon K_eval = 8` across both `CLOSED` and `OPEN` divisions at `dt_ms = 50.0` / `20 Hz`, `denoising_steps = 16` in `CLOSED`).

* **Gapless control SLA (`H = 16, K_eval = 8`):** At `20 Hz` (`dt_ms = 50.0`), the locked `K_eval = 8` execution horizon spans `400 ms`, setting the Track 1B gapless real-time control budget to `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`).
* **Official Season 1 Competition Hardware Targets:**
  * **NVIDIA L40S (`48 GB` GDDR6, Ada SM89, `350 W` TDP, native FP8):** PyTorch `device="cuda"` with split `Prefix + Step` `torch.compile`, static batch buckets `1, 2, 4, 8, 16`, and live `NVML` board telemetry.
  * **Google Cloud TPU `v6e-1` (`32 GB` HBM2e, Trillium):** PyTorch `device="tpu"` via open-source `TorchTPU` with dense vision SDPA, TPU-safe bidirectional DiT attention, and static batch buckets `1, 2, 4, 8, 16`.

## Quick Start

```bash
# Install Isaac-GR00T + PyTorch (+ TorchTPU on Cloud TPU hosts)
git clone https://github.com/NVIDIA/Isaac-GR00T.git ../Isaac-GR00T
pip install -e ../Isaac-GR00T

# 1. Serve on Official Season 1 GPU Target (NVIDIA L40S 48 GB):
export CUDA_VISIBLE_DEVICES=0
eeb serve --policy eeb.servers.gr00t:Gr00tN17Policy --uds /tmp/eeb_gr00t.sock \
  --policy-kwargs '{"model_path": "nvidia/GR00T-N1.7-LIBERO", "device": "cuda", "torch_compile": true, "max_batch_size": 16, "execute_horizon": 8}'

eeb measure --recipe recipes/track1_vla_gr00t_n17_l40s_bf16.yaml --endpoint unix:///tmp/eeb_gr00t.sock \
  --traces ./data/embodiedx_traces_v1_256 --golden ./data/golden_v1/gr00t_n17_golden.npz \
  --concurrency 1,2,4,8,16 --warmup 20 --calls 320 --runs 3 --telemetry-source nvml --strict \
  --out results/gr00t_n17_l40s_bf16

# 2. Serve on Official Season 1 TPU Target (Google Cloud TPU v6e-1 32 GB via TorchTPU):
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0
eeb serve --policy eeb.servers.gr00t:Gr00tN17Policy --uds /tmp/eeb_gr00t_tpu.sock \
  --policy-kwargs '{"model_path": "nvidia/GR00T-N1.7-LIBERO", "device": "tpu", "torch_compile": true, "max_batch_size": 16, "execute_horizon": 8}'

eeb measure --recipe recipes/track1_vla_gr00t_n17_tpu_v6e.yaml --endpoint unix:///tmp/eeb_gr00t_tpu.sock \
  --traces ./data/embodiedx_traces_v1_256 --golden ./data/golden_v1/gr00t_n17_golden.npz \
  --concurrency 1,2,4,8,16 --warmup 20 --calls 320 --runs 3 --telemetry-source tpu_info --strict \
  --out results/gr00t_n17_tpu_v6e_bf16
```
