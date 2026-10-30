# EmbodiedX (`embodied-efficiency-bench` v1.0): Physical Hardware Sweep & Submission Execution Guide

> **Status:** Open for Founding Season Single-Chip Hardware Submissions (`D1`–`D10`) · Workshop Leaderboard Publication  
> **Specification Authority:** [`docs/DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) (`D1`–`D10`, Locked v1.0 Closed Division)  
> **Companion Guides:** [`CLAUDE.md`](../CLAUDE.md) · [`CONTRIBUTING.md`](../CONTRIBUTING.md) · [`docs/engineering_implementation_plan.md`](engineering_implementation_plan.md) · [`docs/EmbodiedX_Benchmark_Methodology_and_Metrics.pdf`](EmbodiedX_Benchmark_Methodology_and_Metrics.pdf)

***

## Overview

All software harness components, PyTorch/`TorchTPU`/JAX accelerator adapters (`eeb/servers/`), gRPC/UDS protocol definitions (`eeb/protocol/eeb.proto`), co-located quality gates (`SANITY_ONLY` and `GOLDEN_REPLAY`), 10 Hz out-of-band power telemetry sidecars (`eeb/telemetry/sidecar.py`), single-chip hardware envelope verifiers (`eeb/envelope/check.py`), 43 canonical YAML recipes (`recipes/`), and the 10-page technical benchmark specification paper (`docs/EmbodiedX_Benchmark_Methodology_and_Metrics.pdf`) ship complete in `v1.0.0`.

At Day-1 v1.0 launch, `submissions/` starts empty so the Founding Season leaderboard tracks community and organizer single-chip submissions strictly as they are verified under ADR-005 (`PENDING` until independent organizer runner re-verification promotes them to `CERTIFIED`). This document provides the step-by-step execution guide to:
1. **Export leaderboard data for the CoRL 2026 workshop challenge page (`efficient-embodied-ai.github.io/challenge/index.html`)** (`Phase 1`).
2. **Generate official `BF16` golden reference `.npz` bundles** (`pi05_golden.npz`, `gr00t_n17_golden.npz`, `cosmos_predict25_golden.npz`) via `eeb data generate-golden` on native-resolution wire traces (`Phase 2`).
3. **Execute physical single-chip concurrency sweeps** across validated YAML recipes in `recipes/` covering NVIDIA GPUs/Jetson (`RTX_5090_32GB`, `L4_24GB`, `L40S_48GB`, `JETSON_AGX_THOR_64GB`, `JETSON_ORIN_NX_16GB`), Google Cloud TPU (`TPU_V6E_1_32GB`), and qualifying AMD Instinct GPUs (`AMD_MI210_64GB`) (`Phase 3`).
4. **Verify cryptographic receipts and compile the live leaderboard** (`Phase 4`).

***

## Phase 1: Website Leaderboard Export (`efficient-embodied-ai.github.io`)

The interactive Pareto Frontier Explorer at `efficient-embodied-ai.github.io/challenge/index.html` consumes the verified curve export produced by `scripts/export_website_leaderboard.py` (`leaderboard/embodiedx_leaderboard_data.json`).

### 1.1 Export Verified Leaderboard JSON & Markdown
```bash
.venv/bin/python scripts/export_website_leaderboard.py \
  --submissions-dir submissions \
  --output leaderboard/leaderboard_v1.json \
  --markdown-output leaderboard/LEADERBOARD.md
```
Verify that `leaderboard/embodiedx_leaderboard_data.json` cleanly encodes the 3 single-model scenarios (`pi05-3b`, `gr00t-n1.7-3b`, `cosmos-predict2.5-2b`) and hardware SKU catalogs.

### 1.2 Automated CI Workflows (`ci/workflows/`)
The repository ships validated GitHub Actions workflows in [`ci/workflows/ci_preflight.yml`](../ci/workflows/ci_preflight.yml) and [`ci/workflows/publish_leaderboard.yml`](../ci/workflows/publish_leaderboard.yml) (copy to `.github/workflows/` with `workflow` OAuth scope to activate on GitHub Actions) to validate recipes, verify submission receipts, and recompile the leaderboard on pull requests and merges.

***

## Phase 2: Official BF16 Golden Reference Generation (`eeb data generate-golden`)

Every `CERTIFIED` Closed Division run in Phase 3 requires co-located `GOLDEN_REPLAY` quality verification (`D3`) in the exact same execution (`eeb measure --golden <path.npz> --strict`):
- **Track 1 (`VLA_TRACK`):** Action cosine similarity $\text{CosSim} \ge 0.995$ vs. the frozen `BF16` reference checkpoint on `EmbodiedX-Traces-v1` + `G4` $\ell_\infty = 10^{-3}$ pixel perturbation check + `LIBERO-RT` closed-loop success within $\le 2.0$ percentage points of `BF16`.
- **Track 2 (`WM_TRACK`):** Rollout $\text{CosSim} \ge 0.990$ and $\text{PSNR} \ge 28.0\text{ dB}$ ($\Delta\text{FVD} \le +5\%$) vs. the frozen `BF16` reference checkpoint, plus horizon-parity cutoff $\ge 1.5\text{ Rollouts/s}$ ($L_{p99} \le 666.7\text{ ms}$; `stream_chunk_frames` is disabled in `CLOSED` division and $\text{TTFF}_{p99} \equiv L_{\text{clip}}$ is reported under `D7`).

Run `eeb data prepare-traces` first to pack native-resolution `EmbodiedX-Traces-v1` wire bundles (`224x224` for `pi05-3b`, `256x256` for `gr00t-n1.7-3b` and `cosmos-predict2.5-2b`), and then run `eeb data generate-golden` once per canonical reference model in `BF16` on a reference workstation GPU or Cloud TPU `v6e-1`. *(Note: `data/embodiedx_traces_v1_mini` is `64x64` strictly for synthetic / `--allow-no-device` CI smoke tests and intentionally fails the Gate `D10` `>= 224x224` native wire-resolution guard under `--strict` hardware execution.)*

```bash
mkdir -p data/golden_v1

# 1. Prepare and inspect native-resolution packed trace bundles (>= 224x224 required by Gate D10)
.venv/bin/eeb data prepare-traces --track vla --out data/embodiedx_traces_v1_224 --image-hw 224x224
.venv/bin/eeb data prepare-traces --track wm --out data/embodiedx_traces_v1_256 --image-hw 256x256
.venv/bin/eeb data inspect data/embodiedx_traces_v1_224
.venv/bin/eeb data inspect data/embodiedx_traces_v1_256
```

### 2.1 Track 1 Model A: $\pi_{0.5}$ (`pi05-3b`, Canonical `pi05_libero`: `horizon = 10`, Locked `execute_horizon = 5`, `denoising_steps = 10`, `224x224`, `dt_ms = 50.0`)
```bash
# Option A (Default) - PyTorch BF16 reference server ("device": "cuda" on GPU/ROCm, or "device": "tpu" via TorchTPU on Cloud TPU v6e-1):
.venv/bin/eeb serve \
  --policy eeb.servers.pi05:Pi05Policy \
  --uds /tmp/eeb_pi05_bf16.sock \
  --policy-kwargs '{"checkpoint_dir": "/path/to/pi05_libero_pytorch", "config_name": "pi05_libero", "device": "cuda", "precision_tag": "BF16", "execute_horizon": 5}' &

# Option B (Optional Alternative) - Native JAX/Flax BF16 server (Cloud TPU v6e-1 or GPU):
# .venv/bin/eeb serve \
#   --policy eeb.servers.pi05:Pi05JaxPolicy \
#   --uds /tmp/eeb_pi05_bf16.sock \
#   --policy-kwargs '{"checkpoint_dir": "gs://openpi-assets/checkpoints/pi05_libero", "config_name": "pi05_libero", "precision_tag": "BF16", "execute_horizon": 5}' &

# Generate deterministic BF16 golden reference bundle:
.venv/bin/eeb data generate-golden \
  --endpoint unix:///tmp/eeb_pi05_bf16.sock \
  --traces data/embodiedx_traces_v1_224 \
  --num-samples 8 \
  --seed 7 \
  --out data/golden_v1/pi05_golden.npz
```

### 2.2 Track 1 Model B: NVIDIA Isaac `GR00T-N1.7-3B` (`gr00t-n1.7-3b`, Locked `execute_horizon = 8`, `horizon = 16`, `denoising_steps = 16`)
```bash
# Use "device": "cuda" on GPU/ROCm or "device": "tpu" via open-source TorchTPU on Cloud TPU v6e-1:
.venv/bin/eeb serve \
  --policy eeb.servers.gr00t:Gr00tN17Policy \
  --uds /tmp/eeb_gr00t_bf16.sock \
  --policy-kwargs '{"model_path": "nvidia/GR00T-N1.7-LIBERO", "data_config": "libero", "denoising_steps": 16, "device": "cuda", "precision_tag": "BF16", "execute_horizon": 8}' &

.venv/bin/eeb data generate-golden \
  --endpoint unix:///tmp/eeb_gr00t_bf16.sock \
  --traces data/embodiedx_traces_v1_256 \
  --num-samples 8 \
  --seed 7 \
  --out data/golden_v1/gr00t_n17_golden.npz
```

### 2.3 Track 2 Model: NVIDIA `Cosmos-Predict2.5-2B` (`cosmos-predict2.5-2b`, Locked `num_frames = 16`, `256x256`, `dt_ms = 41.67`)
```bash
# Option A (Default) - PyTorch diffusers BF16 reference server ("device": "cuda" on GPU/ROCm, or "device": "tpu" via TorchTPU on Cloud TPU v6e-1):
.venv/bin/eeb serve \
  --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy \
  --uds /tmp/eeb_cosmos_bf16.sock \
  --policy-kwargs '{"model_id": "nvidia/Cosmos-Predict2.5-2B", "device": "cuda", "precision_tag": "BF16", "num_frames": 16, "num_inference_steps": 10}' &

# Option B (Optional Alternative) - Native JAX/Flax BF16 server (Cloud TPU v6e-1):
# .venv/bin/eeb serve \
#   --policy eeb.servers.cosmos_predict25:CosmosPredict25JaxPolicy \
#   --uds /tmp/eeb_cosmos_bf16.sock \
#   --policy-kwargs '{"checkpoint_dir": "/path/to/cosmos_predict25_2b_jax", "precision_tag": "BF16", "num_frames": 16, "num_inference_steps": 10}' &

.venv/bin/eeb data generate-golden \
  --endpoint unix:///tmp/eeb_cosmos_bf16.sock \
  --traces data/embodiedx_traces_v1_256 \
  --num-samples 8 \
  --seed 7 \
  --out data/golden_v1/cosmos_predict25_golden.npz
```

***

## Phase 3: Physical Single-Chip Baseline Sweeps (`eeb measure --recipe ... --golden ... --strict`)

All canonical v1.0 Closed Division YAML recipes in `recipes/` are schema-validated by [`scripts/validate_recipes.py`](../scripts/validate_recipes.py):
- **Primary SKU Matrix Recipes (`3 canonical models` across `RTX_5090_32GB`, `L4_24GB`, `JETSON_AGX_THOR_64GB`, `JETSON_ORIN_NX_16GB`, `TPU_V6E_1_32GB`, and `AMD_MI210_64GB`):**
  - `recipes/track1_vla_pi05_{rtx5090,l4,l4_bf16,jetson_thor,orin_nx,tpu_v6e,amd_mi210}.yaml`
  - `recipes/track1_vla_gr00t_n17_{rtx5090,l4,l4_bf16,jetson_thor,orin_nx,tpu_v6e,amd_mi210}.yaml`
  - `recipes/track2_wm_cosmos_predict25_{rtx5090,l4,l4_bf16,jetson_thor,orin_nx,tpu_v6e,amd_mi210}.yaml`
  *(Legacy compatibility recipes `track*tpu_v5e.yaml` remain schema-validated under `recipes/`.)*
- **6 Canonical Class / Tier Recipes:**
  - `recipes/track1_vla_pi05_edge.yaml` (`TPU_V6E_1_32GB`), `recipes/track1_vla_pi05_edge100t.yaml` (`RTX_4060_TI_16GB`)
  - `recipes/track1_vla_gr00t_n17_edge.yaml` (`JETSON_AGX_THOR_64GB`), `recipes/track1_vla_gr00t_edge100t.yaml` (`JETSON_ORIN_NX_16GB`)
  - `recipes/track2_wm_cosmos_predict25_edge.yaml` (`JETSON_AGX_THOR_64GB`), `recipes/track2_wm_cosmos2b_workstation350t.yaml` (`L40S_48GB`)

When `--recipe` is passed to `eeb measure`, the harness automatically inherits `concurrency_levels = [1, 2, 4, 8, 16]` and the locked `execute_horizon` (`5` for `pi05-3b`, `8` for `gr00t-n1.7-3b`, `16` frames for `cosmos-predict2.5-2b`) directly from the recipe.

### 3.1 Pre-Sweep Hardware Envelope & Power Telemetry Verification (Run on Every Host)
Before starting any sweep on a target host, verify single-chip isolation (`num_devices == 1`, $\le 64\text{ GB}$ memory) and 10 Hz out-of-band power telemetry (`nvml`, `tegrastats`, `amd_smi`, or `tpu_info`):
```bash
.venv/bin/python scripts/validate_recipes.py --recipes-dir recipes --strict
.venv/bin/eeb envelope check --sku <HARDWARE_SKU>
.venv/bin/eeb telemetry probe --seconds 2.0 --hz 10
```

### 3.2 NVIDIA GPU & Jetson Host Sweeps (`RTX_5090_32GB`, `JETSON_AGX_THOR_64GB`, `JETSON_ORIN_NX_16GB`, `RTX_4060_TI_16GB`, `L40S_48GB`)
Use `CUDA_VISIBLE_DEVICES=0` so `check_envelope()` sees `num_devices == 1`. On desktop/server GPUs (`RTX_5090_32GB`, `RTX_4060_TI_16GB`, `L40S_48GB`), telemetry uses `--telemetry-source nvml`; on Jetson (`JETSON_AGX_THOR_64GB`, `JETSON_ORIN_NX_16GB`), telemetry uses `--telemetry-source tegrastats`.

```bash
# Example SKU: RTX_5090_32GB (substitute recipes/*_jetson_thor.yaml or *_orin_nx.yaml on Jetson hosts)
export CUDA_VISIBLE_DEVICES=0

# === 1. Track 1: pi0.5 (K=5) ===
.venv/bin/eeb serve \
  --policy eeb.servers.pi05:Pi05Policy \
  --uds /tmp/eeb_pi05.sock \
  --policy-kwargs '{"checkpoint_dir": "/path/to/pi05_libero_pytorch", "device": "cuda", "precision_tag": "FP8", "execute_horizon": 5}' &

.venv/bin/eeb measure \
  --recipe recipes/track1_vla_pi05_rtx5090.yaml \
  --endpoint unix:///tmp/eeb_pi05.sock \
  --traces data/embodiedx_traces_v1_224 \
  --golden data/golden_v1/pi05_golden.npz \
  --warmup 20 --calls 320 --runs 3 \
  --telemetry-source nvml \
  --strict \
  --out submissions/measured_v1/track1_vla_pi05_rtx5090
pkill -f "/tmp/eeb_pi05.sock"

# === 2. Track 1: GR00T-N1.7-3B (K=8) ===
.venv/bin/eeb serve \
  --policy eeb.servers.gr00t:Gr00tN17Policy \
  --uds /tmp/eeb_gr00t.sock \
  --policy-kwargs '{"model_path": "nvidia/GR00T-N1.7-LIBERO", "denoising_steps": 16, "device": "cuda", "precision_tag": "FP8", "execute_horizon": 8}' &

.venv/bin/eeb measure \
  --recipe recipes/track1_vla_gr00t_n17_rtx5090.yaml \
  --endpoint unix:///tmp/eeb_gr00t.sock \
  --traces data/embodiedx_traces_v1_256 \
  --golden data/golden_v1/gr00t_n17_golden.npz \
  --warmup 20 --calls 320 --runs 3 \
  --telemetry-source nvml \
  --strict \
  --out submissions/measured_v1/track1_vla_gr00t_n17_rtx5090
pkill -f "/tmp/eeb_gr00t.sock"

# === 3. Track 2: Cosmos-Predict2.5-2B (H_frames=16) ===
.venv/bin/eeb serve \
  --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy \
  --uds /tmp/eeb_cosmos.sock \
  --policy-kwargs '{"model_id": "nvidia/Cosmos-Predict2.5-2B", "device": "cuda", "precision_tag": "FP8", "num_frames": 16, "num_inference_steps": 10}' &

.venv/bin/eeb measure \
  --recipe recipes/track2_wm_cosmos_predict25_rtx5090.yaml \
  --endpoint unix:///tmp/eeb_cosmos.sock \
  --traces data/embodiedx_traces_v1_256 \
  --golden data/golden_v1/cosmos_predict25_golden.npz \
  --warmup 20 --calls 200 --runs 3 \
  --telemetry-source nvml \
  --strict \
  --out submissions/measured_v1/track2_wm_cosmos_predict25_rtx5090
pkill -f "/tmp/eeb_cosmos.sock"
```
Repeat with the corresponding recipe YAML for:
- **Jetson AGX Thor 64GB (`--telemetry-source tegrastats`):** `recipes/track1_vla_pi05_jetson_thor.yaml`, `recipes/track1_vla_gr00t_n17_jetson_thor.yaml`, `recipes/track2_wm_cosmos_predict25_jetson_thor.yaml`, `recipes/track1_vla_gr00t_n17_edge.yaml`, `recipes/track2_wm_cosmos_predict25_edge.yaml`.
- **Jetson Orin NX 16GB (`--telemetry-source tegrastats`):** `recipes/track1_vla_pi05_orin_nx.yaml`, `recipes/track1_vla_gr00t_n17_orin_nx.yaml`, `recipes/track2_wm_cosmos_predict25_orin_nx.yaml`, `recipes/track1_vla_gr00t_edge100t.yaml`.
- **Tier Reference GPUs (`--telemetry-source nvml`):** `recipes/track1_vla_pi05_edge100t.yaml` (`RTX_4060_TI_16GB`), `recipes/track2_wm_cosmos2b_workstation350t.yaml` (`L40S_48GB`).

### 3.3 Google Cloud TPU Host Sweeps (`TPU_V6E_1_32GB`)
On a single-chip Cloud TPU `v6e-1` (`32 GB` HBM2e) VM, enforce single-chip isolation via `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0`. Verified reference configurations use native JAX/Flax (`Pi05JaxPolicy` with `jax.jit` static batch buckets) or PyTorch/`TorchTPU` (`Pi05Policy`, `Gr00tN17Policy`, and `CosmosPredict25Policy` with `device="tpu"`, `torch_compile=True`, and `torch.tpu.synchronize()`). HBM telemetry is collected via `tpu-info` (`--telemetry-source tpu_info`):

```bash
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_VISIBLE_DEVICES=0

# === 1. Track 1A: pi0.5 on TPU v6e-1 (Native JAX Pi05JaxPolicy or TorchTPU Pi05Policy) ===
.venv/bin/eeb serve \
  --policy eeb.servers.pi05:Pi05JaxPolicy \
  --uds /tmp/eeb_pi05_tpu.sock \
  --policy-kwargs '{"checkpoint_dir": "gs://openpi-assets/checkpoints/pi05_libero", "config_name": "pi05_libero", "max_batch_size": 16, "precision_tag": "BF16", "execute_horizon": 5}' &

for RECIPE in recipes/track1_vla_pi05_tpu_v6e.yaml recipes/track1_vla_pi05_edge.yaml; do
  STEM=$(basename "$RECIPE" .yaml)
  .venv/bin/eeb measure \
    --recipe "$RECIPE" \
    --endpoint unix:///tmp/eeb_pi05_tpu.sock \
    --traces data/embodiedx_traces_v1_224 \
    --golden data/golden_v1/pi05_golden.npz \
    --warmup 20 --calls 320 --runs 3 \
    --telemetry-source tpu_info \
    --strict \
    --out "submissions/measured_v1/${STEM}"
done
pkill -f "/tmp/eeb_pi05_tpu.sock"

# === 2. Track 1B: GR00T-N1.7-3B via TorchTPU on TPU v6e-1 ===
.venv/bin/eeb serve \
  --policy eeb.servers.gr00t:Gr00tN17Policy \
  --uds /tmp/eeb_gr00t_tpu.sock \
  --policy-kwargs '{"model_path": "nvidia/GR00T-N1.7-LIBERO", "denoising_steps": 16, "device": "tpu", "torch_compile": true, "max_batch_size": 16, "precision_tag": "BF16", "execute_horizon": 8}' &

.venv/bin/eeb measure \
  --recipe recipes/track1_vla_gr00t_n17_tpu_v6e.yaml \
  --endpoint unix:///tmp/eeb_gr00t_tpu.sock \
  --traces data/embodiedx_traces_v1_256 \
  --golden data/golden_v1/gr00t_n17_golden.npz \
  --warmup 20 --calls 320 --runs 3 \
  --telemetry-source tpu_info \
  --strict \
  --out submissions/measured_v1/track1_vla_gr00t_n17_tpu_v6e
pkill -f "/tmp/eeb_gr00t_tpu.sock"

# === 3. Track 2: Cosmos-Predict2.5-2B via TorchTPU on TPU v6e-1 ===
.venv/bin/eeb serve \
  --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy \
  --uds /tmp/eeb_cosmos_tpu.sock \
  --policy-kwargs '{"model_id": "nvidia/Cosmos-Predict2.5-2B", "device": "tpu", "torch_compile": true, "max_batch_size": 2, "precision_tag": "BF16", "num_frames": 16, "num_inference_steps": 10}' &

.venv/bin/eeb measure \
  --recipe recipes/track2_wm_cosmos_predict25_tpu_v6e.yaml \
  --endpoint unix:///tmp/eeb_cosmos_tpu.sock \
  --traces data/embodiedx_traces_v1_256 \
  --golden data/golden_v1/cosmos_predict25_golden.npz \
  --warmup 20 --calls 200 --runs 3 \
  --telemetry-source tpu_info \
  --strict \
  --out submissions/measured_v1/track2_wm_cosmos_predict25_tpu_v6e
pkill -f "/tmp/eeb_cosmos_tpu.sock"
```

### 3.4 AMD Instinct ROCm Host Sweeps (`AMD_MI210_64GB`)
On an AMD Instinct MI210 host (`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`), PyTorch maps ROCm devices to `"cuda"` and power telemetry is sampled at 10 Hz via `amd-smi` (`--telemetry-source amd_smi`):

```bash
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# === 1. Track 1: pi0.5 (K=5) on AMD MI210 ===
.venv/bin/eeb serve \
  --policy eeb.servers.pi05:Pi05Policy \
  --uds /tmp/eeb_pi05_rocm.sock \
  --policy-kwargs '{"checkpoint_dir": "/path/to/pi05_libero_pytorch", "device": "cuda", "precision_tag": "BF16", "execute_horizon": 5}' &

.venv/bin/eeb measure \
  --recipe recipes/track1_vla_pi05_amd_mi210.yaml \
  --endpoint unix:///tmp/eeb_pi05_rocm.sock \
  --traces data/embodiedx_traces_v1_224 \
  --golden data/golden_v1/pi05_golden.npz \
  --warmup 20 --calls 320 --runs 3 \
  --telemetry-source amd_smi \
  --strict \
  --out submissions/measured_v1/track1_vla_pi05_amd_mi210
pkill -f "/tmp/eeb_pi05_rocm.sock"

# === 2. Track 1: GR00T-N1.7-3B (K=8) on AMD MI210 ===
.venv/bin/eeb serve \
  --policy eeb.servers.gr00t:Gr00tN17Policy \
  --uds /tmp/eeb_gr00t_rocm.sock \
  --policy-kwargs '{"model_path": "nvidia/GR00T-N1.7-LIBERO", "denoising_steps": 16, "device": "cuda", "precision_tag": "BF16", "execute_horizon": 8}' &

.venv/bin/eeb measure \
  --recipe recipes/track1_vla_gr00t_n17_amd_mi210.yaml \
  --endpoint unix:///tmp/eeb_gr00t_rocm.sock \
  --traces data/embodiedx_traces_v1_256 \
  --golden data/golden_v1/gr00t_n17_golden.npz \
  --warmup 20 --calls 320 --runs 3 \
  --telemetry-source amd_smi \
  --strict \
  --out submissions/measured_v1/track1_vla_gr00t_n17_amd_mi210
pkill -f "/tmp/eeb_gr00t_rocm.sock"

# === 3. Track 2: Cosmos-Predict2.5-2B (H_frames=16) on AMD MI210 ===
.venv/bin/eeb serve \
  --policy eeb.servers.cosmos_predict25:CosmosPredict25Policy \
  --uds /tmp/eeb_cosmos_rocm.sock \
  --policy-kwargs '{"model_id": "nvidia/Cosmos-Predict2.5-2B", "device": "cuda", "precision_tag": "BF16", "num_frames": 16, "num_inference_steps": 10}' &

.venv/bin/eeb measure \
  --recipe recipes/track2_wm_cosmos_predict25_amd_mi210.yaml \
  --endpoint unix:///tmp/eeb_cosmos_rocm.sock \
  --traces data/embodiedx_traces_v1_256 \
  --golden data/golden_v1/cosmos_predict25_golden.npz \
  --warmup 20 --calls 200 --runs 3 \
  --telemetry-source amd_smi \
  --strict \
  --out submissions/measured_v1/track2_wm_cosmos_predict25_amd_mi210
pkill -f "/tmp/eeb_cosmos_rocm.sock"
```

***

## Phase 4: Leaderboard Compilation & Receipt Verification (`ADR-5 PENDING` $\to$ `CERTIFIED`)

Each completed `eeb measure --strict` run writes an atomic `report.json` + SHA-256 `receipt.json` bundle with `status = "PENDING"`, `quality_gate.quality_certified = true`, and honest hardware power provenance (`MEASURED` where board telemetry is exposed, `UNAVAILABLE` where unexposed on Cloud TPU `v6e-1`).

### 4.1 Verify Receipts & Submission Anti-Cheat Compliance
```bash
# 1. Verify cryptographic SHA-256 receipt integrity for every measured directory
for RECEIPT in submissions/measured_v1/*/receipt.json; do
  .venv/bin/eeb verify-receipt --receipt "$RECEIPT"
done

# 2. Run the full single-chip envelope, division, receipt & AST validator
for SUB_DIR in submissions/measured_v1/*; do
  .venv/bin/eeb validate-submission --submission "$SUB_DIR"
done
```

### 4.2 Recompile `leaderboard/leaderboard_v1.json` & `leaderboard/LEADERBOARD.md`
Once measured `receipt.json` bundles (or Two-Phase Commit-Reveal bundles) are placed under `submissions/`, compile the verified 4-way SemiAnalysis Pareto frontier leaderboard:
```bash
.venv/bin/python scripts/export_website_leaderboard.py \
  --submissions-dir submissions \
  --output leaderboard/leaderboard_v1.json \
  --markdown-output leaderboard/LEADERBOARD.md

# Or via the forwarded eeb CLI subcommand:
.venv/bin/eeb build-leaderboard \
  --submissions-dir submissions \
  --output leaderboard/leaderboard_v1.json \
  --markdown-output leaderboard/LEADERBOARD.md \
  --strict-verify
```

### 4.3 Final Regression Check Before Publishing Leaderboard Update
```bash
.venv/bin/pytest -q
.venv/bin/python scripts/validate_recipes.py --recipes-dir recipes --strict
```
