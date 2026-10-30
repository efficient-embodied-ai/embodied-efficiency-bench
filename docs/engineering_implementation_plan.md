# EmbodiedX (`embodied-efficiency-bench` v1.0): Unified Engineering Implementation Plan & Red-Team Verification (`D1`–`D10`)

> [!IMPORTANT]
> **Status:** **Locked v1.0 Closed Division Specification (`D1`–`D10`), Adversarially Audited & Open for Founding Season Submissions (`pytest -q` passing; 43 canonical recipes validated)**  
> **Target GitHub Organization:** [`https://github.com/efficient-embodied-ai/embodied-efficiency-bench`](https://github.com/efficient-embodied-ai/embodied-efficiency-bench)  
> **Canonical Specification Authority:** [`docs/DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md) (`D1`–`D10`)  
> **Owner:** EEAI Benchmark Committee

***

## 1. Locked v1.0 Architecture & Design Decisions Summary (`D1`–`D10`)

| Decision ID | Locked v1.0 Rule | Implementation in `eeb/` & `embodied_bench/` |
| :--- | :--- | :--- |
| **`D1`** | **Closed Division Only for v1.0 (`division == "CLOSED"`):** Frozen model weights, 100% neural forward-pass execution on the single target accelerator (**Host-CPU Neural Offload Ban**), `stream_chunk_frames == 0` (**Closed `stream_chunk_frames` Ban**), and locked execution horizons ($H=10, K=5$ for $\pi_{0.5}$; $H=16, K=8$ for `GR00T-N1.7-3B`; $H_{\text{frames}}=16$ for `Cosmos-Predict2.5-2B`). Open Division (`V2-1`) and Embodied Omni Track (`V2-2`) are explicitly deferred to v2.0. | Enforced in `eeb/protocol/eeb.proto` (`Caps.execute_horizon = 13`), `eeb/loadgen/sweep.py`, `eeb/loadgen/report.py` (`CANONICAL_LOCKED_HORIZONS`), and `embodied_bench/schema.py` (`CANONICAL_CHALLENGE_MODELS`). |
| **`D2`** | **Two Canonical Tracks:**<br>• **Track 1 (`VLA_TRACK`, Separate 1A & 1B Award Pools):** $\pi_{0.5}$ (`pi05-3b` / `pi05_libero`, $H=10, K=5$, $10$ steps, dual $224\times 224$ RGB) and `GR00T-N1.7-3B` (`nvidia/GR00T-N1.7-LIBERO`, $H=16, K=8$, $16$ steps, dual $256\times 256$ RGB).<br>• **Track 2 (`WM_TRACK`):** `Cosmos-Predict2.5-2B` ($H_{\text{frames}}=16$ @ $256\times 256$ RGB, $24\text{ FPS}$, $10$ steps, $\text{CFG}=1.0$, `stream_chunk_frames = None`). | Implemented in `eeb/servers/torch_backend.py` (`TorchTPU` `device="tpu"`, `CUDA`, `ROCm`), `eeb/servers/pi05/policy.py`, `eeb/servers/pi05/policy_jax.py`, `eeb/servers/gr00t/policy.py`, and `eeb/servers/cosmos_predict25/policy.py`. |
| **`D3`** | **Co-Located Quality Gates (Measured in the Same Execution ID):**<br>• **Track 1:** Action `CosSim >= 0.995` on `EmbodiedX-Traces-v1` + BF16-safe `G4` pixel/state perturbation + closed-loop `LIBERO-RT` within $\le 2.0\text{ pp}$ of `BF16`; cutoff `p99 Chunk Rate >= 20.0 Chunks/s` ($L_{p99}(1) \le 50\text{ ms}$).<br>• **Track 2:** `CosSim >= 0.990`, `PSNR >= 28.0 dB` ($\Delta\text{FVD} \le +5\%$), sole reactivity cutoff `p99 Rollout Rate >= 1.5 Rollouts/s` (`24.0 Stream FPS`, $L_{p99,\text{WM}}(1) \le 666.7\text{ ms}$); under full bidirectional 3D-DiT rollout, $\text{TTFF}_{p99}(1) \equiv L_{\text{clip}}$ is co-reported strictly as diagnostic `D7` (never a hard fail gate in v1.0 Closed Division). | Implemented in `eeb/loadgen/gate.py` (`golden_replay_gate`, `sanity_gate`), `eeb/data/cli.py` (`eeb data generate-golden`), and `eeb/loadgen/report.py` (`eeb measure --golden ... --strict`). |
| **`D4` & `D5`** | **Automatic Concurrency Sweep & Unified InferenceX Axes:**<br>• Sweep $C \in \{1, 2, 4, 8, 16\}$ (Track 1, `timed_calls >= 320`, `warmup >= 20`, `runs >= 3`) and $B \in \{1, 2, 4, 8, 16\}$ (Track 2, `timed_calls >= 200`, `warmup >= 20`, `runs >= 3`).<br>• **X-Axis:** `p99 Chunk Rate` $1000 / L_{p99}(C)$ (`Chunks/s`, Track 1) and `p99 Rollout Rate` $1000 / L_{p99,\text{WM}}(B)$ (`Rollouts/s`, Track 2).<br>• **Y-Axis:** `Executed Actions/sec` ($1000 \cdot C \cdot K / \bar{L}(C)$) or `Generated Frames/sec` ($1000 \cdot B \cdot H_{\text{frames}} / \bar{L}_{\text{WM}}(B)$) across **4 normalizations**: `1. Per Chip`, `2. Per Watt (Actions/J, Frames/J)`, `3. Per $/hr Reference TCO Rate`, `4. Iso-Compute (per 100 Dense BF16 TFLOPS) / Iso-Bandwidth`. | Implemented in `eeb/loadgen/sweep.py` (`SweepConfig.concurrency = [1, 2, 4, 8, 16]`) and `eeb/loadgen/report.py` (`VLA_Y_METRIC = "executed_actions_per_sec"`, `WM_Y_METRIC = "generated_frames_per_sec"`). |
| **`D6`, `D7`, & `D10`** | **Default Forward-Only Async Loop, Diagnostic Telemetry, & Anti-Gaming LoadGen:** `LIBERO-RT` advances $d = \lceil L / \Delta t \rceil$ steps asynchronously without mandatory `vjp` backprop. Every point reports observation staleness ($\text{ms}$), normalized action jerk (Track 1), $\text{TTFF}_{p99} \equiv L_{\text{clip}}$ (Track 2), and peak device memory (`GB`). LoadGen enforces native wire resolution ($\ge 224\times 224$), opaque random `UUIDv4` `session_id`s, inline blind golden/G4 canaries, equal per-slot quota + keep-alive drain, pooled steady-state $p_{99}(C)$ with `slot_count_spread <= 15%`, and spread caps ($\le 5\%$ throughput/$p50$, $\le 8\%$ $p99$). | Implemented in `eeb/sim/libero_rt.py`, `eeb/loadgen/sweep.py`, `eeb/loadgen/gate.py`, and `eeb/loadgen/report.py`. |
| **`D8` & `D9`** | **Two Single-Chip Hardware Classes ($\le 64\text{ GB}$, `num_devices == 1`) & Physical SI Unit Awards:**<br>• **Onboard Edge Class ($\le 16\text{ GB}$)** & **Single-Accelerator Class ($\le 64\text{ GB}$)**.<br>• Official awards in `Actions/sec/chip`, `Frames/sec/chip`, `Actions/Joule`, and `Frames/Joule` (requiring live `MEASURED` telemetry for `Per Watt`). | Enforced in `eeb/envelope/check.py` (`ENVELOPE_MAX_MEMORY_GB = 64.0`, `num_devices == 1`) and `eeb/telemetry/sidecar.py` (`NVML` hardware `mJ` counter, `tegrastats`, `amd_smi`, `tpu_info`). |

***

## 2. Zero-Context Red-Team Audit & Remediation Summary

All findings identified by independent zero-context red-team audits are resolved and covered by regression tests in `tests/test_eeb_v1_harness.py`, `tests/test_e2e_full_pipeline.py`, and `tests/test_user_journey_seam_fixes.py`:

| Area | Verification & Anti-Theater Safeguard |
| :--- | :--- |
| **Anti-Simulation (`SYNTHETIC_DUMMY` Guard)** | `DummyVLAPolicy` and `DummyWMPolicy` hardcode `server_impl="SYNTHETIC_DUMMY"`. `build_report()` (`eeb/loadgen/report.py`) forces `status="INVALID"` on any synthetic policy, and `validate_report()` / `leaderboard_compiler.py` reject any synthetic report from entering the hardware leaderboard. |
| **Locked Horizon Wire & Report Enforcement (`D1`)** | `Caps.execute_horizon = 13` in `eeb/protocol/eeb.proto` propagates locked horizons ($H=10, K=5$ for $\pi_{0.5}$, $H=16, K=8$ for `GR00T-N1.7-3B`, $H_{\text{frames}}=16$ for `Cosmos-Predict2.5-2B`) over gRPC/UDS into `PacingConfig` and `executed_actions_per_sec`. |
| **Honest Bidirectional 3D-DiT Rollout Timing (`D1`, `D3`, `D7`)** | `CosmosPredict25Policy.rollout` runs full 16-frame bidirectional 3D-DiT denoising with `stream_chunk_frames = None` (`0`) in Closed Division and full device synchronization (`torch.cuda.synchronize()`, `torch.tpu.synchronize()`, or `jax.block_until_ready()`), evaluating real-time eligibility at $\ge 1.5\text{ Rollouts/s}$ ($24.0\text{ Stream FPS}$, $L_{p99} \le 666.7\text{ ms}$) while co-reporting $\text{TTFF}_{p99} \equiv L_{\text{clip}}$ honestly under `D7`. |
| **Honest Power Telemetry (`D5` & `D9`)** | `summarize_samples()` (`eeb/telemetry/sidecar.py`) prioritizes NVML monotonic hardware energy counters (`energy_mj`) and requires $\ge 2$ powered samples plus $\ge 90\%$ window coverage for `MEASURED` provenance; unexposed Cloud TPU power remains `null` (`UNAVAILABLE`). |
| **Golden Replay Quality & Standalone Receipt Validation (`D3`)** | `eeb data generate-golden` builds deterministic reference `.npz` bundles on native wire traces (`data/embodiedx_traces_v1_224` and `data/embodiedx_traces_v1_256`); `eeb validate-submission`, `eeb verify-bundle`, and `eeb verify-receipt` verify recipe-backed, standalone `report.json` + `receipt.json`, and Two-Phase Commit-Reveal submission bundles. |

***

## 3. Canonical `eeb` v1.0 CLI Reference

```bash
# 1. Check single-chip hardware envelope (D8: num_devices == 1, <= 64 GB)
eeb envelope check --sku TPU_V6E_1_32GB --allow-no-device

# 2. Prepare and inspect native-resolution EmbodiedX-Traces-v1 wire bundles
#    (note: data/embodiedx_traces_v1_mini is 64x64 strictly for CI smoke tests)
eeb data prepare-traces --track vla --out data/embodiedx_traces_v1_224 --image-hw 224x224
eeb data inspect data/embodiedx_traces_v1_224

# 3. Start a standalone gRPC Policy server over Unix Domain Socket (Track 1 or Track 2)
eeb serve --policy eeb.servers.pi05:Pi05Policy --uds /tmp/eeb.sock \
  --policy-kwargs '{"checkpoint_dir": "/path/to/pi05_libero_pytorch", "device": "cuda"}' &

# 4. Generate a BF16 golden reference bundle for same-execution quality verification (D3)
eeb data generate-golden --endpoint unix:///tmp/eeb.sock \
  --traces data/embodiedx_traces_v1_224 --num-samples 8 --out data/golden_v1/pi05_golden.npz

# 5. Run automatic C in {1, 2, 4, 8, 16} concurrency sweep with golden verification & strict gate (D3, D4, D5, D10)
eeb measure --recipe recipes/track1_vla_pi05_rtx5090.yaml --endpoint unix:///tmp/eeb.sock \
  --traces data/embodiedx_traces_v1_224 --golden data/golden_v1/pi05_golden.npz \
  --warmup 20 --calls 320 --runs 3 --strict --out runs/pi05_rtx5090

# 6. Validate all 43 canonical v1.0 Closed Division benchmark recipes & compile submissions
python3 scripts/validate_recipes.py --recipes-dir recipes --strict
eeb build-leaderboard --submissions-dir submissions \
  --output leaderboard/leaderboard_v1.json --markdown-output leaderboard/LEADERBOARD.md --strict-verify
```
