# pi0.5 Reference Server (`pi05-3b`: PyTorch `lerobot` / `openpi` + Native JAX/Flax `openpi`)

Canonical Track 1A (`VLA_TRACK`) reference servers for **$\pi_{0.5}$ (`pi05-3b`)** under the locked `pi05_libero` contract (`2 x 224x224` RGB, native action chunk `H = 10`, locked `execute_horizon K_eval = 5` across both `CLOSED` and `OPEN` divisions at `dt_ms = 50.0` / `20 Hz`, `10` Euler ODE denoising steps in `CLOSED`, `BF16`).

* **Action horizon and gapless control SLA (`H = 10, K_eval = 5`):** The canonical LIBERO checkpoint (`pi05_libero` in `openpi` and `lerobot/pi05_libero_finetuned`) emits `H = 10` actions with locked execution prefix `K_eval = 5` (`250 ms` execution span at `20 Hz`), setting the Track 1A gapless real-time control budget to `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`).
* **Official Season 1 Competition Hardware Targets:**
  * **NVIDIA L40S (`48 GB` GDDR6, Ada SM89, `350 W` TDP, native FP8):** `eeb.servers.pi05:Pi05Policy` (`device="cuda"`, `NVML` board telemetry).
  * **Google Cloud TPU `v6e-1` (`32 GB` HBM2e, Trillium):** `eeb.servers.pi05:Pi05JaxPolicy` (native JAX/Flax `openpi` with static micro-batch buckets `1, 2, 4, 8, 16`) or `eeb.servers.pi05:Pi05Policy` (`device="tpu"`, `TorchTPU`).

## 1. Host Prerequisites

* **NVIDIA L40S (`48 GB`, `350 W` TDP):** `CUDA_VISIBLE_DEVICES=0`, Python `>= 3.11`, `torch`, `lerobot` (or `openpi`), and `pynvml`. Under Thermal Rule `D4`, run the workload warmup (`--warmup 20..50`) at standard boost clocks first; lock SM clocks via `nvidia-smi -lgc` (calibrated on the actual model inference loop rather than synthetic GEMMs) only if inter-run thermal boost drift breaches the 3-run spread gate.
* **Google Cloud TPU `v6e-1` (`32 GB` HBM2e):** `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0`, Python `>= 3.11`, `openpi` (`jax[tpu]`, `flax`) or `torch` + `torch_tpu`, plus `tpu-info`.

## 2. Quick Start on Official Season 1 GPU Target (`NVIDIA L40S 48 GB`)

```bash
export CUDA_VISIBLE_DEVICES=0

# Terminal 1: Start PyTorch reference server on L40S
eeb serve --policy eeb.servers.pi05:Pi05Policy --uds /tmp/eeb_pi05.sock \
  --policy-kwargs '{"checkpoint_dir": "lerobot/pi05_libero_finetuned", "device": "cuda", "torch_compile": true, "precision_tag": "BF16", "execute_horizon": 5}'

# Terminal 2: Verify single-chip envelope, build native 224x224 golden bundle once, and run 3-run sweep
eeb envelope check --sku L40S_48GB
eeb data prepare-traces --out ./data/embodiedx_traces_v1_224 --image-hw 224x224
eeb data generate-golden \
  --endpoint unix:///tmp/eeb_pi05.sock \
  --traces ./data/embodiedx_traces_v1_224 \
  --num-samples 8 --seed 7 \
  --out ./data/golden_v1/pi05_golden.npz

eeb measure \
  --recipe recipes/track1_vla_pi05_l40s_bf16.yaml \
  --endpoint unix:///tmp/eeb_pi05.sock \
  --traces ./data/embodiedx_traces_v1_224 \
  --golden ./data/golden_v1/pi05_golden.npz \
  --concurrency 1,2,4,8,16 \
  --warmup 20 --calls 320 --runs 3 \
  --telemetry-source nvml \
  --strict \
  --out results/pi05_l40s_bf16
eeb verify-receipt --submission results/pi05_l40s_bf16
```

## 3. Quick Start on Official Season 1 TPU Target (`Google Cloud TPU v6e-1 32 GB`)

```bash
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_VISIBLE_DEVICES=0

# Terminal 1: Start native JAX/Flax BF16 server with static micro-batch buckets (1,2,4,8,16)
eeb serve --policy eeb.servers.pi05:Pi05JaxPolicy --uds /tmp/pi05_tpu.sock \
  --policy-kwargs '{"checkpoint_dir": "gs://openpi-assets/checkpoints/pi05_libero", "config_name": "pi05_libero", "max_batch_size": 16, "precision_tag": "BF16", "execute_horizon": 5}'

# Terminal 2: Verify single-chip TPU envelope and run 3-run concurrency sweep
eeb envelope check --sku TPU_V6E_1_32GB
eeb measure \
  --recipe recipes/track1_vla_pi05_tpu_v6e.yaml \
  --endpoint unix:///tmp/pi05_tpu.sock \
  --traces ./data/embodiedx_traces_v1_224 \
  --golden ./data/golden_v1/pi05_golden.npz \
  --concurrency 1,2,4,8,16 \
  --warmup 20 --calls 320 --runs 3 \
  --telemetry-source tpu_info \
  --strict \
  --out results/pi05_tpu_v6e_bf16
eeb verify-receipt --submission results/pi05_tpu_v6e_bf16
```
