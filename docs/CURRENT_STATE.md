# `embodied-efficiency-bench`: Current State (Honest Inventory)

> Status date: **2026-10-30 (PDT)** · Aligned with [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) (`D1`–`D10`) and [`HARDWARE_BASELINE_EXECUTION_PLAN.md`](HARDWARE_BASELINE_EXECUTION_PLAN.md).
> Update this file whenever a module changes provenance (placeholder → real).

> [!IMPORTANT]
> **Two packages live in this repository:**
> 1. **`eeb/` (Primary v1.0 Harness: Real Client-Server Protocol, LoadGen, Telemetry Sidecar, Envelope Checker, Data Tooling, and Reference Servers):** Implements the gRPC/UDS + in-process `Policy` protocol (`eeb/protocol`), closed-loop `ConcurrencySweep` (`eeb/loadgen/sweep.py`), co-located `SANITY_ONLY` / `GOLDEN_REPLAY` quality gates (`eeb/loadgen/gate.py`), atomic `report.json` + SHA-256 `receipt.json` builder (`eeb/loadgen/report.py`), 10 Hz out-of-band power telemetry sidecar (`eeb/telemetry/sidecar.py`), single-chip envelope verifier (`eeb/envelope/check.py`, probing open-source `TorchTPU` `torch.tpu` first on Cloud TPUs), LeRobot v2/v3 trace packer & G4 perturbation generator (`eeb/data/`), and turnkey single-chip reference servers for Track 1 ($\pi_{0.5}$ PyTorch/`TorchTPU` + native JAX, `GR00T-N1.7-3B` PyTorch/`TorchTPU`) and Track 2 (`Cosmos-Predict2.5-2B` PyTorch/`TorchTPU` + optional JAX). At Day-1 v1.0 launch, `submissions/` opens clean (`0` pre-populated baseline curves) for Founding Season community and organizer hardware submissions (`submissions/track1_vla/`, `submissions/track2_wm/`). Synthetic dummy policies (`DummyVLAPolicy`, `DummyWMPolicy`) hardcode `server_impl = "SYNTHETIC_DUMMY"` and always produce `status = "INVALID"`; self-measured runs produce `status = "PENDING"` until organizer verification (`CERTIFIED`), and unmeasured power fields remain `null` (`power_provenance = "UNAVAILABLE"`).
> 2. **`embodied_bench/` (Schema, Math Libraries, & Legacy Analytic Prototype):** Provides the shared `HARDWARE_SKU_SPECS`, `RecipeConfig`, `atomic_write_json`, Pareto extraction (`pareto.py`), and normalization math (`measure.py`), alongside legacy analytic/synthetic prototype modules (`adapters.py`, `golden_dataset.py`, `libero_rt.py`, `stage1_golden_replay.py`) that stamp `execution_provenance.physical_run == false` on every output.

***

## 1. Primary `eeb/` v1.0 Package Inventory

| Module | Provenance | What it does |
|---|---|---|
| `eeb/cli.py` | **Real** | Top-level `eeb` CLI (`eeb serve`, `eeb measure`, `eeb telemetry probe`, `eeb envelope check`, `eeb data {download,pack-traces,inspect,perturb,generate-golden}`), plus legacy subcommand forwarding. |
| `eeb/protocol/` | **Real** | gRPC / Protobuf wire contract (`eeb.proto`) and Python `Policy` ABC (`Capabilities`, `Reset`, `Predict` for Track 1 VLA, streaming `Rollout` for Track 2 World Models). |
| `eeb/loadgen/sweep.py` | **Real** | Closed-loop `ConcurrencySweep` across $C \in \{1, 2, 4, 8, 16\}$ (Track 1) or $B \in \{1, 2, 4, 8, 16\}$ (Track 2) with think-time pacing ($K \cdot \Delta t$), 3-run median protocol, `< 5%` cross-run variance check, and `TraceBundleObservationSource` (`--traces`). |
| `eeb/loadgen/gate.py` | **Real** | Co-located same-execution quality gate (`SANITY_ONLY` and `GOLDEN_REPLAY`: Track 1 action $\text{CosSim} \ge 0.995$, Track 2 rollout $\text{CosSim} \ge 0.990$ & $\text{PSNR} \ge 28.0\text{ dB}$). |
| `eeb/loadgen/report.py` | **Real** | Atomic `report.json` + SHA-256 `receipt.json` writer with SemiAnalysis InferenceX-style X-axis ($1000 / L_{p99}$ in `Chunks/s` for VLA or `Rollouts/s` for WM) and 4 Y-axis normalizations (`Per Chip`, `Per Watt`, `Per $/hr Reference TCO Rate`, `Iso-Compute (per 100 Dense BF16 TFLOPS) / Iso-Bandwidth`) in the `CLOSED` division. |
| `eeb/telemetry/sidecar.py` | **Real** | 10 Hz out-of-band power & memory sampler (`NVML` / `nvidia-smi`, `tegrastats`, `amd-smi`, `tpu-info`); sets `power_provenance = "UNAVAILABLE"` and `null` power/energy fields when unmeasured. |
| `eeb/envelope/check.py` | **Real** | Live single-chip hardware envelope verifier (`num_devices == 1`, $\le 64\text{ GB}$ memory, $\le 1,000\text{ BF16 TFLOPS}$, $\le 2,000\text{ FP8/INT8 TOPS}$, $\le 2.0\text{ TB/s}$ memory bandwidth; enumerates Cloud TPUs via `torch.tpu` before `jax.devices()` / `tpu-info`). |
| `eeb/data/` | **Real** | `EmbodiedX-Traces-v1` LeRobot v2/v3 trace downloader (`download.py`), packer (`pack.py`), G4 $\ell_\infty = 10^{-3}$ perturbation generator (`perturb.py`), turnkey golden reference generator (`golden.py`), and manifest schema (`schema.py`). |
| `eeb/servers/torch_backend.py` | **Real** | Shared PyTorch accelerator helper (`resolve_torch_device`, `synchronize_torch_device`, `maybe_compile_torch_module`) covering `CUDA`/`ROCm` and open-source `TorchTPU` (`pip install torch torch_tpu`, `device="tpu"`, `torch.tpu.synchronize()`, optional `torch.compile()`). |
| `eeb/servers/dummy.py` | **Synthetic self-test (`SYNTHETIC_DUMMY`)** | In-process/UDS synthetic `DummyVLAPolicy` and `DummyWMPolicy` for harness testing; stamped `server_impl = "SYNTHETIC_DUMMY"` and forced to `status = "INVALID"`. |
| `eeb/servers/pi05/` | **Real (multi-backend reference scaffold)** | Reference Track 1 server for **$\pi_{0.5}$ (`openpi`, locked $H = 10, K = 5$, dual `224x224` RGB)** in PyTorch (`policy.py`: `TorchTPU` `device="tpu"`, `CUDA`, `ROCm`) and native JAX/Flax (`policy_jax.py` with `jax.block_until_ready()`). |
| `eeb/servers/gr00t/` | **Real (multi-backend reference scaffold)** | Reference Track 1 server for **NVIDIA Isaac `GR00T-N1.7-3B` (locked $H = 16, K = 8$, 16 denoising steps, dual `256x256` RGB)** (`policy.py`: `TorchTPU` `device="tpu"`, `CUDA`, `ROCm`). |
| `eeb/servers/cosmos_predict25/` | **Real (multi-backend reference scaffold)** | Reference Track 2 server for **NVIDIA `Cosmos-Predict2.5-2B` (locked `num_frames = 16` at 24 FPS, `256x256` RGB, 10 steps, `CFG = 1.0`)** in PyTorch `diffusers` (`CosmosPredict25Policy`: `TorchTPU` `device="tpu"`, `CUDA`, `ROCm`) and optional JAX/Flax (`CosmosPredict25JaxPolicy`). |
| `submissions/` & `leaderboard/` | **Open for Founding Season PR submissions** | Day-1 launch ships with `0` pre-populated submission bundles under `submissions/` (`submissions/track1_vla/`, `submissions/track2_wm/`); verified community and organizer receipts compile cleanly into `leaderboard/leaderboard_v1.json`, `leaderboard/LEADERBOARD.md`, and `leaderboard/embodiedx_leaderboard_data.json`. |

***

## 2. Shared Math & Legacy Prototype Modules (`embodied_bench/`)

| Module | Provenance | What it does |
|---|---|---|
| `embodied_bench/schema.py` | **Real** | Recipe dataclass + `validate()`, `HARDWARE_SKU_SPECS`, `CANONICAL_CHALLENGE_MODELS`, `RunStatus` state machine, crash-safe `atomic_write_json` / `atomic_read_modify_write_json`. |
| `embodied_bench/measure.py` & `pareto.py` | **Real (math)** | Pareto-efficiency mask, frontier extraction, 4-way normalizations, log-linear iso-reactivity interpolation. |
| `embodied_bench/attestation.py` | **Real** | Canonical-JSON SHA-256 receipts (`create_submission_receipt`, `verify_submission_receipt`). |
| `embodied_bench/quality_gates.py` | **Real (math)** | Cosine-sim / NRMSE / PSNR / SHA-256 tensor digests, Gate G1–G4 decision logic, AST hot-loop device-sync auditor. |
| `embodied_bench/adapters.py`, `golden_dataset.py`, `libero_rt.py`, `stage1_golden_replay.py`, `stage2_closed_loop_verify.py` | **ANALYTIC / SYNTHETIC PLACEHOLDER** | Legacy prototype path; stamps `execution_provenance.physical_run == false` and raises `PlaceholderExecutionWarning`. |
| `tests/fixtures/seed_leaderboard.json` & `tests/fixtures/sample_submission_receipt.json` | **Placeholder schema fixtures** | Kept cleanly inside `tests/fixtures/` only for schema/CI receipt-verification unit tests (`tests/test_embodied_bench.py`); never plot as empirical hardware results (`leaderboard/` houses strictly the live verified hardware `leaderboard_v1.json` and `LEADERBOARD.md`). |
