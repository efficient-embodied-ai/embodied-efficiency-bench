# Submitting to `embodied-efficiency-bench` (3rd-Party External Guide)

`embodied-efficiency-bench` (`eeb` v1.0) is a **100% open-source, standalone public benchmark** hosted under [`github.com/efficient-embodied-ai/embodied-efficiency-bench`](https://github.com/efficient-embodied-ai/embodied-efficiency-bench) for the **CoRL 2026 Workshop on Efficient Foundation Models for Real-Time Embodied AI (EEAI)**.

It has **zero proprietary or internal dependencies** and runs on **any single NVIDIA GPU/Jetson SoC, Google Cloud TPU, AMD GPU, or Custom ASIC** using Python 3.11+ (`pip install -e ".[grpc,dev]"`). See [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) for the locked v1.0 specification (D1–D9).

***

## 1. Choose Your Track & Canonical v1.0 Closed-Division Model (`D1` & `D2`)

v1.0 launches strictly with the **Closed Division** (frozen reference checkpoints and locked execution horizons $K$ / $H_{\text{frames}}$; custom kernels `Pallas`/`Triton`/`CUDA`/`ROCm`, compiler fusion via `TorchTPU` `torch.compile`/`XLA`/`jax.jit`/`TensorRT`, calibrated `FP8`/`INT8`/`INT4` quantization, cross-frame visual token caching, pre-caching frozen episode instruction text embeddings on the accelerator during episode `reset()`, and forward-only async scheduling are permitted; Host-CPU neural offload inside timed calls and `stream_chunk_frames > 0` are forbidden):

- **Track 1: Real-Time VLA Control (`VLA_TRACK`)**:
  1. **`pi05-3b`** (`Physical-Intelligence/openpi` `pi05_libero` / `lerobot/pi05_libero_finetuned`: 3.3B PaliGemma VLM + 300M Flow-Matching Action Expert, dual `224x224` RGB, 10 ODE steps, locked **`execute_horizon = 5`**, native chunk $H=10$, $\Delta t = 50.0\text{ ms}$; native JAX `openpi` and PyTorch `lerobot`/`TorchTPU` are co-equal certified runtimes).
  2. **`gr00t-n1.7-3b`** (`nvidia/GR00T-N1.7-LIBERO` / `nvidia/GR00T-N1.7-3B`: 3B `Cosmos-Reason2-2B` [`Qwen3-VL-2B`] VLM + 16-step DiT Flow Action Head, dual `256x256` RGB, locked **`execute_horizon = 8`**, native chunk $H=16$, $\Delta t = 50.0\text{ ms}$).
  - **Co-Located Track 1 Gate (`D3`)**: Native wire resolution ($\ge 224\times 224$), action $\text{CosSim} \ge 0.995$ vs. `BF16` reference + BF16-safe G4 perturbation check, single-stream $L_{p99}(1) \le 50\text{ ms}$ (`>= 20.0 Chunks/s`), and closed-loop `LIBERO-RT` task success within $\le 2.0$ percentage points of the `BF16` reference under real-time delay $d = \lceil L / \Delta t \rceil$.
- **Track 2: Interactive World Models (`WM_TRACK`)**:
  1. **`cosmos-predict2.5-2b`** (`nvidia/Cosmos-Predict2.5-2B`: 2B Spatio-Temporal Action-Conditioned Bidirectional 3D-DiT Video World Model at locked `256x256` RGB, locked **`num_frames = 16`** frames per rollout at 24 FPS, $\Delta t = 41.67\text{ ms}$).
  - **Co-Located Track 2 Gate (`D3`)**: Rollout latent/pixel $\text{CosSim} \ge 0.990$, $\text{PSNR} \ge 28.0\text{ dB}$ ($\Delta\text{FVD} \le +5\%$) vs. `BF16` reference, BF16-safe G4 perturbation reactivity, inverse-dynamics action controllability, 60s closed-loop rollout stability, and real-time horizon parity $L_{p99,\text{WM}}(1) \le 666.7\text{ ms}$ (`>= 1.5 Rollouts/s` = `24.0 Stream FPS`; $\text{TTFF}_{p99} \equiv L_{\text{clip}}$ co-reported under `D7`).

***

## 2. Run on Any Single Accelerator Under the `CAP_SINGLE_CHIP` Envelope (`D8`)

All submissions execute natively on **1 physical device (`num_devices == 1`, `TP=1, PP=1, EP=1`)** with zero artificial software memory or MPS capping and deterministically locked GPU SM clocks (`nvidia-smi -lgc f,f`):
- **Compute Cap**: $\le 1,000\text{ Dense FP32-accum BF16/FP16 TFLOPS}$ / $\le 2,000\text{ FP8/INT8 TOPS}$
- **Memory & Bandwidth Cap**: $\le 64\text{ GB}$ native device VRAM/HBM/LPDDR5X and $\le 2.0\text{ TB/s}$ memory bandwidth:
  1. **Onboard Edge Class ($\le 16\text{ GB}$)**: `JETSON_ORIN_NX_16GB` (`16GB`), `TPU_V5E_1_16GB` (`1x Cloud TPU v5e-1 16GB`).
  2. **Single-Accelerator Class ($\le 64\text{ GB}$)**:
     - **Google Cloud TPU**: `TPU_V6E_1_32GB` (`1x Cloud TPU v6e-1 Trillium 32GB`)
     - **NVIDIA**: `JETSON_AGX_THOR_64GB` (`64GB`), `RTX_5090_32GB` (`32GB`), `L40S_48GB` (`48GB`), `RTX_4090_24GB` (`24GB`, `82.6 TF` BF16), `L4_24GB` (`24GB`, `121.0 TF` BF16, `300 GB/s`, `72W`)
     - **AMD**: `AMD_MI210_64GB` (`64GB`), `AMD_W7900_48GB` (`48GB`), `AMD_RX_7900_XTX_24GB` (`24GB`)
     - **Custom ASIC**: `CUSTOM_ASIC_SINGLE_DEVICE` ($\le 64\text{ GB}$, $\le 1,000\text{ TFLOPS}$ BF16)

```bash
# 1. Verify single-chip envelope & prepare native-resolution eval traces
eeb envelope check --sku RTX_5090_32GB
eeb data prepare-traces --out ./data/eval_traces_224 --image-hw 224x224

# 2. Start your policy server (e.g. pi0.5 or GR00T-N1.7-3B; use "device": "tpu" via TorchTPU/JAX on Cloud TPU)
eeb serve --policy eeb.servers.pi05:Pi05Policy --uds /tmp/eeb.sock \
  --policy-kwargs '{"checkpoint_dir": "/path/to/pi05_ckpt", "device": "cuda"}'

# 3. Run the 3-run concurrency sweep (C = 1,2,4,8,16) and emit report.json + receipt.json + recipe.json
eeb measure \
  --recipe recipes/track1_vla_pi05_rtx5090.yaml \
  --endpoint unix:///tmp/eeb.sock \
  --traces ./data/eval_traces_224 \
  --golden ./data/libero_golden.npz \
  --execute-horizon 5 \
  --sweep 1,2,4,8,16 \
  --warmup 20 --calls 320 --runs 3 \
  --out submissions/my_team_pi05_rtx5090

# 4. Verify cryptographic receipt integrity and submission admissibility before opening a PR
eeb verify-receipt --submission submissions/my_team_pi05_rtx5090
eeb validate-submission --submission submissions/my_team_pi05_rtx5090
```

***

## 3. Open a GitHub Pull Request

Open a PR against `efficient-embodied-ai/embodied-efficiency-bench` containing:
- `submissions/<your_team>_<model>_<sku>/report.json`, `receipt.json`, and `recipe.json` (plus your recipe YAML under `recipes/<your_team>_<model>_<opt>.yaml`)
- Your reproducible container image digest (`ghcr.io/...` or `docker.io/...`) implementing the `eeb` gRPC (`eeb/protocol/eeb.proto`) or Python `Policy` interface (`eeb/protocol/policy.py`).
