# EmbodiedX (`embodied-efficiency-bench` v1.0): Engineering Architecture & Implementation Plan

> [!IMPORTANT]
> **Canonical v1.0 Spec Authority (`docs/DESIGN_DECISIONS.md` D1-D10):**
> * **Owner:** EEAI Benchmark Committee
> * **Division:** **Closed Division only** (`division == "CLOSED"`) for v1.0 (Open Division deferred to v2.0 per `V2-1`; Embodied Omni track deferred per `V2-2`), enforcing the **Host-CPU Neural Offload Ban** (100% of neural forward-pass operators on the 1 target accelerator) and **Closed `stream_chunk_frames` Ban** (`stream_chunk_frames = None` / `0`).
> * **Canonical v1.0 Models, Separate Award Pools & Locked Horizons:**
>   - **Track 1 (`VLA_TRACK`: Vision-Language-Action Control):**
>     * **Workload 1A:** **$\pi_{0.5}$ (`pi05-3b` / `pi05_libero`, `openpi` PyTorch via `TorchTPU`/`CUDA`/`ROCm` & native JAX/XLA)**: locked `horizon H = 10`, `execute_horizon K = 5` (`dt_ms = 50.0`, 10 Euler flow-matching steps, dual `224x224` RGB).
>     * **Workload 1B:** **`nvidia/GR00T-N1.7-3B` / `nvidia/GR00T-N1.7-LIBERO` (PyTorch via `TorchTPU`/`CUDA`/`ROCm`)**: locked `horizon H = 16`, `execute_horizon K = 8` (`dt_ms = 50.0`, 16 DiT denoising steps, dual `256x256` RGB). Track 1A and Track 1B maintain separate championship pools (`D2`, `D9`).
>   - **Track 2 (`WM_TRACK`: Action-Conditioned World Models):**
>     * **Workload 2A:** **`nvidia/Cosmos-Predict2.5-2B` (PyTorch `diffusers` via `TorchTPU`/`CUDA`/`ROCm` & optional JAX `vidax`)**: locked `horizon_frames = 16` (`256x256` RGB, `dt_ms = 41.67` at 24 FPS, 10 steps, `CFG = 1.0`, `stream_chunk_frames = None`); under full bidirectional 3D-DiT rollout, $\text{TTFF}_{p99}(1) \equiv L_{\text{clip}}$ is co-reported strictly as diagnostic `D7` (never a `<= 100 ms` fail gate in v1.0 Closed Division).
> * **Primary Wire Protocol:** **gRPC over Unix Domain Sockets (`unix:///tmp/eeb.sock`) or loopback TCP** (`eeb/protocol/eeb.proto` with `Caps.execute_horizon = 13`).
> * **Canonical Disambiguated Pareto Axes & Anti-Gaming (`D4`, `D5`, `D9`, `D10`):**
>   - **X-Axis:** **`p99 Chunk Rate (Chunks/s) [1000 / L_p99, ms]`** with gapless real-time cutoff at **`>= 4.0 Chunks/s`** ($p99 \le 250\text{ ms}$, Track 1A `pi05-3b`) and **`>= 2.5 Chunks/s`** ($p99 \le 400\text{ ms}$, Track 1B `gr00t-n1.7-3b`), and **`p99 Prediction Rate (Predictions/s) [1000 / L_p99, ms]`** with real-time cutoff at **`>= 1.0 Predictions/s`** ($L_{p99} \le 1000\text{ ms}$, Track 2), scored from pooled steady-state $p_{99}(C)$ across active slots paired with a mandatory `<= 15%` per-slot completion fairness fence.
>   - **Y-Axis:** Steady-state wall-clock **Executed Actions/sec** (`executed_actions_per_sec`, Track 1) and **Generated Frames/sec** (`generated_frames_per_sec`, Track 2) across the 4 SemiAnalysis normalizations (`Raw/Chip`, `Per-Watt` requiring live `MEASURED` hardware telemetry via `nvmlDeviceGetTotalEnergyConsumption` in `mJ`, `tegrastats`, or `amd-smi` for official awards, `Per-$/hr Reference TCO Rate`, and `Iso-100-Dense-BF16-TFLOPS` under canonical `FP32-accum`).
>   - **LoadGen Anti-Gaming (`D10`):** Native wire resolution ($\ge 224\times 224$), opaque random `UUIDv4` `session_id`s, `~5%` inline blind `GOLDEN_REPLAY` + `G4` canaries, per-slot quota $N_{\text{slot}} = \lceil N/C \rceil$ with keep-alive drain (`timed_calls >= 320` VLA, `timed_calls >= 200` WM, `warmup >= 20`, `runs >= 3`), and cross-run variance gates $\le 5\%$ on `p50`/throughput and $\le 8\%$ on `p99` tail latency.

**Target GitHub Organization:** [`https://github.com/efficient-embodied-ai/`](https://github.com/efficient-embodied-ai/)  
**Repositories:**
1. **`efficient-embodied-ai/embodied-efficiency-bench`**: Benchmark Harness (`eeb/` + `embodied_bench/`), `EmbodiedX-Traces-v1` Data Loader, Verified Reference Model Servers, Submission Receipts & CI Validator
2. **`efficient-embodied-ai/efficient-embodied-ai.github.io`**: CoRL 2026 Workshop Website & Interactive Pareto Leaderboard (`challenge/index.html`)

***

## 0. Architectural Summary (`eeb/` Production Harness vs. `embodied_bench/` Shared Schema)

| Subsystem | Role in `embodied-efficiency-bench` v1.0 |
| :--- | :--- |
| **1. Schema, Hardware Cap & Normalizations (`embodied_bench/schema.py`, `eeb/loadgen/report.py`)** | Enforces `CANONICAL_CHALLENGE_MODELS` (`pi05-3b` $H=10, K=5$; `gr00t-n1.7-3b` $H=16, K=8$; `cosmos-predict2.5-2b` $H_{\text{frames}}=16$), Unified Single-Chip Cap (`CAP_SINGLE_CHIP`: `num_devices == 1`, $\le 64\text{ GB}$, $\le 1{,}000\text{ dense BF16 TFLOPS}$, $\le 2{,}000\text{ FP8/INT8 TOPS}$, $\le 2.0\text{ TB/s}$), and SemiAnalysis 4-Way Normalizations (`Raw`, `Perf/Watt`, `Iso-100-TOPS`, `Perf/$/hr`). |
| **2. Benchmark Harness (`eeb/protocol/eeb.proto`, `eeb/loadgen/sweep.py`, `eeb/telemetry/sidecar.py`)** | Decouples `eeb` LoadGen client from model servers over gRPC/UDS (`Predict`, `Rollout`, `Capabilities` with `execute_horizon`), executes unpaced automatic concurrency sweeps across $C, B \in \{1, 2, 4, 8, 16\}$ (`warmup >= 20`, `timed_calls >= 320` VLA / `>= 200` WM, `runs >= 3`), and collects 10 Hz out-of-band active telemetry (`NVML` / `tegrastats` / `tpu-info` / `amd-smi`, leaving unmeasured Cloud TPU power honestly `UNAVAILABLE` / `null`). |
| **3. Data Loader (`eeb/data/`, `TraceBundleObservationSource`)** | Packs real open-source episodes from Hugging Face (`lerobot/libero`, `lerobot/droid_1.0.1`) at native model wire resolution (`224x224` and `256x256`), verifies SHA-256 manifests, injects Gate `G4` micro-perturbations, and generates deterministic `BF16` golden reference `.npz` bundles (`eeb data generate-golden`). *(The bundled `64x64` `data/embodiedx_traces_v1_mini/` fixture is strictly for CPU `--allow-no-device` CI smoke tests.)* |
| **4. Reference Model Servers (`eeb/servers/{pi05,gr00t,cosmos_predict25}/`)** | Turnkey single-chip gRPC/UDS servers across `TorchTPU` (`device="tpu"`), `CUDA`, `ROCm`, and optional native `JAX/Flax`: `Pi05Policy` / `Pi05JaxPolicy` ($H=10, K=5$), `Gr00tN17Policy` ($H=16, K=8$), and `CosmosPredict25Policy` / `CosmosPredict25JaxPolicy` ($H_{\text{frames}}=16$, `stream_chunk_frames = None`, $\text{TTFF}_{p99} \equiv L_{\text{clip}}$). |
| **5. Pareto Curve Compiler & Submission Auditor (`embodied_bench/leaderboard_compiler.py`, `embodied_bench/submission_validator.py`)** | Audits SHA-256 receipts, sibling `report.json` parity, Two-Phase Commit-Reveal bundles (`bundle_summary.json` + `submission_meta.json`), `CAP_SINGLE_CHIP` hardware eligibility, and Python AST rules across `submissions/`, compiling convex Pareto frontiers and Iso-Reactivity slices (`10/20/30/50 Chunks/s` for VLA; `1.5/3.0/5.0/10.0 Rollouts/s` for WM) into `leaderboard/leaderboard_v1.json`, `leaderboard/LEADERBOARD.md`, and `leaderboard/embodiedx_leaderboard_data.json`. |

***

## 1. Repository Layout (`efficient-embodied-ai/embodied-efficiency-bench`)

```text
embodied-efficiency-bench/
├── ENGINEERING_PLAN.md                  # This architecture & v1.0 implementation plan
├── README.md                            # Quickstart, rules, hardware SKUs & CLI guide
├── CLAUDE.md / AGENTS.md / CONTRIBUTING.md  # Agent & contributor operational instructions
├── pyproject.toml                       # `eeb` CLI package (`pip install -e ".[grpc,dev]"`)
├── eeb/                                 # Primary v1.0 client-server harness package
│   ├── cli.py                           # Top-level `eeb` CLI (`serve`, `measure`, `compare`, `telemetry`, `envelope`, `data`)
│   ├── protocol/                        # `eeb.proto` gRPC contract & Python `Policy` ABC
│   ├── loadgen/                         # Closed-loop `ConcurrencySweep`, `gate.py`, and SHA-256 `report.py`
│   ├── telemetry/                       # 10 Hz out-of-band power & memory samplers (`nvml`, `tegrastats`, `amd_smi`, `tpu_info`)
│   ├── envelope/                        # Single-chip hardware envelope verifier (`CAP_SINGLE_CHIP`, `num_devices == 1`)
│   ├── data/                            # LeRobot v2.1 / v3.0 trace downloader, native-resolution packer, G4 perturb & golden generator
│   ├── sim/                             # Forward-only latency-coupled `LIBERO-RT` evaluator (`d = ceil(L / dt)`)
│   └── servers/                         # Reference single-chip servers (`pi05/`, `gr00t/`, `cosmos_predict25/`, `microbatch.py`, `torch_backend.py`)
├── embodied_bench/                      # Shared schema, Pareto math, attestation, leaderboard compiler & submission validator
├── docker/                              # Multi-vendor single-chip Dockerfiles (CUDA, Jetson Thor, ROCm MI210, Cloud TPU v6e-1)
├── recipes/                             # 43 canonical v1.0 Closed Division YAML recipes
├── scripts/                             # Recipe validator (`validate_recipes.py`) and leaderboard exporter (`export_website_leaderboard.py`)
├── data/
│   └── embodiedx_traces_v1_mini/        # 64x64 deterministic mini-split strictly for CPU CI smoke tests
├── submissions/                         # Founding Season community & organizer PR submissions (`track1_vla/`, `track2_wm/`)
├── leaderboard/                         # Compiled `leaderboard_v1.json`, `LEADERBOARD.md`, & `embodiedx_leaderboard_data.json`
├── tests/                               # Comprehensive `pytest` suite & isolated schema fixtures (`tests/fixtures/`)
└── ci/workflows/                        # GitHub Actions workflows (`ci_preflight.yml`, `publish_leaderboard.yml`)
```

***

## 2. Detailed Design of the 4 Engineering Workstreams

### Workstream 1: gRPC/UDS Client-Server Harness & Telemetry (`eeb/protocol/`, `eeb/loadgen/`, `eeb/telemetry/`)

```mermaid
sequenceDiagram
    participant CLI as eeb measure (Client LoadGen)
    participant Tel as 10Hz Telemetry Sidecar (NVML/Jetson/ROCm/TPU)
    participant Srv as Single-Chip Policy Server (unix:///tmp/eeb.sock)

    CLI->>Srv: 1. Capabilities RPC (Verify track, execute_horizon, model_id, precision)
    CLI->>Srv: 2. Reset RPC + Warmup (warmup >= 20 calls/session with opaque UUIDv4 session_id)
    CLI->>Tel: 3. Start out-of-band power (NVML mJ / board sensors) & memory sampler
    loop Concurrency Sweep (C or B in 1, 2, 4, 8, 16; 3 runs; inline blind golden+G4 canaries)
        CLI->>Srv: Predict (Track 1 VLA) or Rollout (Track 2 WM) at native >= 224x224 RGB resolution
        Srv-->>CLI: Synchronized Action Chunk (H, D_a) or 16-Frame Video Rollout
    end
    CLI->>Tel: 4. Stop telemetry & integrate Joules / Peak Memory / >= 90% coverage check
    CLI->>CLI: 5. Run Co-Located GOLDEN_REPLAY + Gate G4 Perturbation Probe (same execution_id)
    CLI->>CLI: 6. Compute 4 SemiAnalysis Normalizations & write atomic report.json + SHA-256 receipt.json
```

1. **Decoupled gRPC over Unix Domain Sockets (`eeb/protocol/eeb.proto`):**
   - Prevents dependency conflicts between `openpi` (PyTorch/JAX), `Isaac-GR00T` (PyTorch/TensorRT), and `Cosmos-Predict2.5` (`diffusers`).
   - Measures client-side wall-clock `time.perf_counter_ns()` outside the contestant process while enforcing synchronized device completion before returning responses.
2. **Anti-Gaming Closed-Loop Load Generator (`eeb/loadgen/sweep.py`):**
   - Sweeps $C \in \{1, 2, 4, 8, 16\}$ (Track 1, `timed_calls >= 320`, ensuring $\ge 20$ chunks/slot at $C=16$) and $B \in \{1, 2, 4, 8, 16\}$ (Track 2, `timed_calls >= 200`), with `warmup >= 20` and `runs >= 3`.
   - Uses opaque random `UUIDv4` `session_id`s, equal per-slot quota $N_{\text{slot}} = \lceil N/C \rceil$ with keep-alive drain until the slowest worker finishes, pooled steady-state $p_{99}(C)$ paired with a `<= 15%` per-slot fairness fence (`worst_slot_p99` retained as a diagnostic), and inline blind golden/`G4` canary probes.
3. **Honest Hardware Telemetry Sidecar (`eeb/telemetry/sidecar.py`):**
   - Prioritizes NVML cumulative hardware energy (`nvmlDeviceGetTotalEnergyConsumption` in `mJ`) alongside $\ge 10\text{ Hz}$ board polling (`tegrastats`, `amd-smi`, `tpu-info`).
   - Requires $\ge 90\%$ window coverage for `power_provenance = "MEASURED"`; leaves unexposed sensors (`TPU v6e-1`) honestly `null` (`power_provenance = "UNAVAILABLE"`).

***

### Workstream 2: Native-Resolution Trace Loader & Golden Reference Tooling (`eeb/data/`)

1. **`EmbodiedX-Traces-v1` Real Robot Episode Streams:**
   - Packs real trajectories from `HuggingFaceVLA/libero` (Apache-2.0 / MIT) and `lerobot/droid_100` (MIT / CC-BY-4.0) at native model input resolutions via `eeb data prepare-traces --track vla --out data/embodiedx_traces_v1_224 --image-hw 224x224` (`pi05-3b`) and `eeb data prepare-traces --track wm --out data/embodiedx_traces_v1_256 --image-hw 256x256` (`cosmos-predict2.5-2b`).
   - Pre-decodes RGB frames into host RAM prior to the timed measurement window and verifies per-shard SHA-256 digests.
   - Retains `data/embodiedx_traces_v1_mini/` (`64x64`) exclusively for zero-download CPU CI smoke tests (`--allow-no-device`).
2. **Co-Located Golden Reference Generation (`eeb data generate-golden`):**
   - Records deterministic `BF16` reference outputs into `.npz` bundles consumed by `eeb measure --golden <bundle.npz> --strict` for same-execution Gate `G1`/`G3`/`G4` parity verification.

***

### Workstream 3: Verified Reference Servers & PR Submission Workflow (`eeb/servers/`, `submissions/`, `ci/workflows/`)

1. **Three Canonical Reference Server Families (`eeb/servers/`):**
   - **`eeb/servers/pi05/` (`Track 1A: pi05-3b`, $H = 10, K = 5$, $10$ steps, `224x224`):** PyTorch (`Pi05Policy` for `CUDA`/`ROCm`/`TorchTPU`) and native JAX/Flax (`Pi05JaxPolicy` with static power-of-2 batch buckets for Cloud TPU `v6e-1`).
   - **`eeb/servers/gr00t/` (`Track 1B: gr00t-n1.7-3b`, $H = 16, K = 8$, $16$ steps, `256x256`):** PyTorch (`Gr00tN17Policy`) with `DynamicMicroBatcher` and `torch.compile` support across `TorchTPU` (`device="tpu"`), `CUDA`, and `ROCm`.
   - **`eeb/servers/cosmos_predict25/` (`Track 2: cosmos-predict2.5-2b`, $H_{\text{frames}} = 16$, $10$ steps, `CFG = 1.0`, `256x256`):** PyTorch `diffusers` (`CosmosPredict25Policy`) and optional JAX (`CosmosPredict25JaxPolicy`) executing full bidirectional 3D-DiT rollouts (`stream_chunk_frames = None`).
2. **Participant PR Submission Flow (`submissions/` + `ci/workflows/ci_preflight.yml`):**
   - Contestants run `eeb measure --recipe <recipe.yaml> --endpoint unix:///tmp/eeb.sock --traces <native_traces> --golden <bf16_golden.npz> --strict --out submissions/<entry_id>` on 1 qualifying chip and verify locally with `eeb validate-submission` and `eeb verify-receipt`.
   - CI validates recipes (`scripts/validate_recipes.py --recipes-dir recipes --strict`), audits every submission bundle and standalone hardware receipt, and compiles the Markdown/JSON leaderboard preview.

***

### Workstream 4: Pareto Curve Compiler & Interactive Website Leaderboard (`embodied_bench/leaderboard_compiler.py`)

1. **4-Way SemiAnalysis Pareto Frontier Solver (`eeb build-leaderboard`):**
   - Audits all submission bundles under `submissions/` (`report.json` + SHA-256 `receipt.json` and Two-Phase Commit-Reveal bundles).
   - Extracts upper-right monotone Pareto frontiers across `Raw/Chip`, `Perf/Watt` (`Actions/J`, `Frames/J`), `Iso-100-Dense-BF16-TFLOPS`, and `Perf/$/hr`.
   - Computes log-linear Iso-Reactivity slices at `10, 20, 30, 50 Chunks/s` (Track 1 VLA) and `1.5, 3.0, 5.0, 10.0 Rollouts/s` (Track 2 WM), rendering disambiguated units in `leaderboard/LEADERBOARD.md`.
2. **Website Sync (`efficient-embodied-ai.github.io/challenge/index.html`):**
   - Exports verified hardware curves and 16-char receipt SHA prefixes into `leaderboard/embodiedx_leaderboard_data.json` consumed by `efficient-embodied-ai.github.io/challenge/index.html`.

***

## 3. Completed v1.0 Delivery Milestones

| Milestone | Scope & Verified Deliverables | Status |
| :--- | :--- | :--- |
| **Milestone 1 (`M1`): gRPC/UDS Harness, Envelope & Telemetry** | `eeb/protocol/eeb.proto`, `eeb/loadgen/{sweep,gate,report}.py`, `eeb/envelope/check.py`, and `eeb/telemetry/sidecar.py` (NVML `mJ` counter + board polling). | **Completed & Verified** |
| **Milestone 2 (`M2`): Real `EmbodiedX-Traces-v1` Data & Golden Pipeline** | `eeb/data/{download,pack,perturb,golden,cli}.py`, native `224x224`/`256x256` trace packing, Gate `D10` wire-resolution guard, and CI `mini` fixture. | **Completed & Verified** |
| **Milestone 3 (`M3`): Reference Single-Chip Servers & 43 Canonical Recipes** | Single-chip `pi05-3b`, `gr00t-n1.7-3b`, and `cosmos-predict2.5-2b` reference servers (`TorchTPU`, `CUDA`, `ROCm`, `JAX`) and 43 schema-validated Closed Division YAML recipes. | **Completed & Verified** |
| **Milestone 4 (`M4`): Pareto Compiler, 10-Page Paper & Website Integration** | `leaderboard/leaderboard_v1.json`, disambiguated `leaderboard/LEADERBOARD.md`, `leaderboard/embodiedx_leaderboard_data.json`, Two-Phase Commit-Reveal validator, 10-page methodology PDF, and interactive `/challenge/` explorer. | **Completed & Verified** |
