# Cosmos-Predict2.5-2B Reference Server (`diffusers` PyTorch CUDA / ROCm / TorchTPU + Optional JAX/Flax)

Canonical Track 2 (`WM_TRACK`) reference servers for `nvidia/Cosmos-Predict2.5-2B` under the locked rollout geometry (`num_frames = 16` at `24 FPS`, `256 x 256` RGB, `10` denoising steps in `CLOSED`, `guidance_scale = 1.0`, `dt_ms = 41.67`).

* **Primary reference server:** `eeb.servers.cosmos_predict25:CosmosPredict25Policy` (`diffusers` `Cosmos2_5_PredictBaseWorldPipeline` 3D-DiT + Wan 3D-VAE with Rule `D1` on-device session prompt pre-encoding at `reset()`, linear `UniPC` step bridge, and optional `DynamicMicroBatcher`) targeting the two Official Season 1 Competition Hardware SKUs:
  * **NVIDIA L40S (`48 GB` GDDR6, Ada SM89, `350 W` TDP, native FP8):** PyTorch `device="cuda"` with `torch.cuda.synchronize()`, optional `torch_compile=True`, and `NVML` board telemetry.
  * **Google Cloud TPU `v6e-1` (`32 GB` HBM2e, Trillium):** PyTorch `device="tpu"` via open-source `TorchTPU` with `torch_compile=True`.
* **Optional alternative server:** `eeb.servers.cosmos_predict25:CosmosPredict25JaxPolicy` (native JAX/Flax `vidax` / `maxdiffusion` with `jax.block_until_ready()`).
* **Track 2 SLA and Closed Division rule (`D1` / `D7`):** Track 2 requires `p99 <= 1000.0 ms` (`>= 1.0 Pred/s` at `16 frames/pred`) for responsive MPC rollouts, with diagnostic horizon-parity slices reported up to `>= 1.5 Rollouts/s` (`24.0 Stream FPS`, `p99 <= 666.7 ms`). `stream_chunk_frames` is disabled (`None`) in `CLOSED` division and allowed only with `"division": "OPEN"` for Open-Division causal world models passing the 3-part quality gate (including `S2` Kendall `tau >= 0.80`).

## Quick Start

```bash
pip install 'diffusers>=0.33.0' transformers accelerate safetensors torch

# 1. Serve on Official Season 1 GPU Target (NVIDIA L40S 48 GB):
export CUDA_VISIBLE_DEVICES=0
eeb serve --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy --uds /tmp/eeb_wm.sock \
  --policy-kwargs '{"model_id": "nvidia/Cosmos-Predict2.5-2B", "device": "cuda", "torch_compile": true, "max_batch_size": 4}'

eeb measure --recipe recipes/track2_wm_cosmos_predict25_l40s_bf16.yaml --endpoint unix:///tmp/eeb_wm.sock \
  --traces ./data/embodiedx_traces_v1_256 --golden ./data/golden_v1/cosmos_predict25_golden.npz \
  --concurrency 1,2,4,8,16 --warmup 20 --calls 200 --runs 3 --telemetry-source nvml --strict \
  --out results/cosmos_predict25_l40s_bf16

# 2. Serve on Official Season 1 TPU Target (Google Cloud TPU v6e-1 32 GB via TorchTPU):
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0
eeb serve --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy --uds /tmp/eeb_wm_tpu.sock \
  --policy-kwargs '{"model_id": "nvidia/Cosmos-Predict2.5-2B", "device": "tpu", "torch_compile": true, "max_batch_size": 2}'

eeb measure --recipe recipes/track2_wm_cosmos_predict25_tpu_v6e.yaml --endpoint unix:///tmp/eeb_wm_tpu.sock \
  --traces ./data/embodiedx_traces_v1_256 --golden ./data/golden_v1/cosmos_predict25_golden.npz \
  --concurrency 1,2,4,8,16 --warmup 20 --calls 200 --runs 3 --telemetry-source tpu_info --strict \
  --out results/cosmos_predict25_tpu_v6e_bf16
```
