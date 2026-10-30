# EmbodiedX v1.0.0 Current Repository State

## Release Readiness Summary

`embodied-efficiency-bench` (`eeb` v1.0.0) is locked for the October 30, 2026 public Season 1 launch at the CoRL 2026 Workshop on Efficient Foundation Models for Real-Time Embodied AI ([EEAI @ CoRL 2026](https://efficient-embodied-ai.github.io/challenge/)).

In accordance with Option A launch policy, Season 1 opens with **`0` organizer-seeded vendor benchmark entries** across `leaderboard/` and `submissions/`. Every stage of the evaluation pipeline (deterministic real-world trace preparation from public `HuggingFaceVLA/libero` and `lerobot/droid_100` datasets, multi-runtime reference servers, concurrent closed-loop load generation, hardware telemetry capture, and eight-gate cryptographic commit-reveal bundle verification) was validated end-to-end on physical single-chip `NVIDIA L40S (48 GB)` and `Google Cloud TPU v6e-1 (32 GB)` hardware prior to release without publishing organizer baseline numbers. Community pull requests establish the founding Pareto frontier directly.

## Season 1 Milestone Calendar and Award Structure

| Event | Deadline | Scope |
| :--- | :--- | :--- |
| **Season 1 Public Launch** | **Oct 30, 2026** | v1.0.0 harness, validator gates, and live Season 1 leaderboard open for Phase-1 pull requests. |
| **CoRL 2026 Live Spotlight Freeze** | **Nov 10, 2026 (23:59 UTC)** | Valid PRs merged by Checkpoint 1 are presented live on stage on **Nov 12, 2026** at CoRL 2026 in Austin as Founding Frontier Leaders. |
| **Official Season 1 Awards Freeze** | **Dec 18, 2026 (23:59 AoE)** | Final Season 1 PR freeze plus 24-hour Phase-2 private container reveal for finalist clean-hardware re-runs. Continuous rolling leaderboard remains active in `2027+`. |

- **Three Scalar Award Categories per Track.** (1) `Interactive Reactivity (C = 1)` maximizing `1000 / p99_ms` at `C = 1` (tie-broken by `p50_ms`, requiring `p99 / p50 <= 1.5`); (2) `Real-Time Fleet Throughput` maximizing SLA-compliant `Actions/s` or `Predictions/s` over `C in {1, 2, 4, 8, 16}`; and (3) `Green Embodied Efficiency` maximizing hardware-measured `Actions/J` or `Predictions/J` (`10 Hz NVML` board power at constant cap on `NVIDIA L40S`; `--assume-power-watts` ineligible; `TPU v6e-1 = N/A`). Exact ties break via lower-step Real-Time Dominated Hypervolume wrt `(0.5 * X_SLA, 0)` over SLA-compliant discrete points.
- **Season 1 Winner Recognition.** Verified category winners receive monetary prizes, an official CoRL 2026 EEAI Workshop Award Certificate, an invitation to apply for Google Cloud credits, permanent verified leaderboard attribution, and an invitation to co-author the post-season *EmbodiedX Season 1 Systems Report*.

## End-to-End Pipeline and Target Platform Specification Matrix

| Component | Track 1A (`pi0.5-LIBERO`) | Track 1B (`GR00T N1.7-3B`) | Track 2 (`Cosmos-Predict2.5-2B`) |
| :--- | :--- | :--- | :--- |
| **Architecture Attribution** | Physical Intelligence `pi0.5` (3.3B VLM + 300M flow expert, 10 Euler ODE steps) | NVIDIA `GR00T N1.7-3B` (Cosmos-Reason2-2B VLM + 16 DiT flow steps) | NVIDIA `Cosmos-Predict2.5-2B` (2B bidirectional 3D-DiT, 10 steps, `CFG = 1.0`) |
| **Canonical HF Checkpoint and Commit SHA** | `lerobot/pi05_libero_finetuned_v044` (`7152f13e34c6c115d808aea0e5d08aca74eac14b`, ungated with pre-fit normalizers) | `nvidia/GR00T-N1.7-LIBERO` (`libero_10` checkpoint, `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`) | `nvidia/Cosmos-Predict2.5-2B` (`robot/action-cond` action-conditioned checkpoint, `85f8ae7bfe8f5525c8d103429524dcf12f98bf7b`) |
| **Public Trace Dataset and Horizon** | `HuggingFaceVLA/libero` (`86958911c0f959db2bbbdb107eb3e17c5f9c798e`), dual `224x224` RGB; `H = 10`, locked `K_eval = 5` | `HuggingFaceVLA/libero` (`86958911c0f959db2bbbdb107eb3e17c5f9c798e`), dual `256x256` RGB; `H = 16`, locked `K_eval = 8` | `lerobot/droid_100` (`87301a2d2e99340e2010c9ef0f1d8e780b08aaf9`), `256x256` RGB; locked `K_eval = 16` frames per rollout prediction |
| **Gapless Real-Time SLA (`20 Hz` = `50 ms` step)** | `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`) | `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`) | `p99 <= 1000.0 ms` (`>= 1.0 Pred/s`; strictly `Predictions/s` and `Predictions/J`) |
| **Standard GPU (`nvidia_l40s_48gb`)** | `PyTorch` (`eager` / `torch.compile`) + `10 Hz NVML` verified | `PyTorch` (`eager` / `torch.compile`) + `10 Hz NVML` verified | `PyTorch` (`eager` / `torch.compile`) + `10 Hz NVML` verified |
| **Standard TPU (`tpu_v6e_1_32gb`)** | `openpi` `JAX` (`jax.jit`) and `TorchTPU` verified | `TorchTPU` (`eager` / `torch.compile`) verified | `TorchTPU` and `JAX` reference servers verified |
| **Closed Division Gate (`--num-samples >= 100`)** | `CosSim >= 0.995` + `CVaR-1%` floor | `CosSim >= 0.995` + `CVaR-1%` floor | `SSIM >= 0.88`, `PSNR >= 26.0 dB`, `CVaR-1%(SSIM) >= 0.80`, `CVaR-1%(PSNR) >= 22.0 dB` |
| **Open Division Gate (Stage-2 Receipt)** | Latency-coupled `LIBERO-RT` within `-5.0 pp` | Latency-coupled `LIBERO-RT` within `-5.0 pp` | Closed parity floors + `LPIPS <= 0.12`, `FVD <= 120`, `InvDyn >= 0.99`, `Kendall tau >= 0.80` |
| **Pre-Seeded Leaderboard Entries** | `0` (open for community PR #1) | `0` (open for community PR #1) | `0` (open for community PR #1) |

## Protocol and Verification Invariants

1. **Standardized Single-Chip Cloud Hardware.** All Season 1 runs enforce `num_devices == 1` (`TP = PP = EP = 1`) on `NVIDIA L40S (48 GB GDDR6, Ada SM89, 350 W TDP, native FP8, 362.05 TF dense BF16)` or `Google Cloud TPU v6e-1 (32 GB HBM2e, Trillium, 918 TF dense BF16)` at stock vendor clocks and TDP.
2. **Real Traces, Pooled Steady-State `p99(C)`, and Anti-Gaming Defenses.** Zero synthetic sine-wave traces are permitted in submissions; all traces derive deterministically from public `HuggingFaceVLA/libero` and `lerobot/droid_100`. Sweeps evaluate `C in {1, 2, 4, 8, 16}` starting at `C = 1` (monotonic OOM tail truncation only; zero interpolation), `N = 3` trials per point measuring pooled steady-state `p99(C)` across active slots (`N >= 320` VLA / `N >= 200` WM) paired with the `<= 15%` per-slot completion fairness fence, within `<= 8%` `p99` spread after real-model warmup (`--warmup >= 30`), `p99 / p50 <= 1.5` at `C = 1`, `max / p50 <= 4.0` across all points, per-frame cryptographic `+-1/255` `uint8` micro-jitter, opaque `UUIDv4` session IDs, native wire payload checks, and inline Golden-IO (`--num-samples >= 100`) plus `G4` sensitivity canaries.
3. **Two-Phase Commit-Reveal Auditability.** Phase-1 PRs generated via `eeb prepare-commit-bundle` and checked with `eeb verify-bundle --strict` commit only `bundle_summary.json` and `submission_meta.json` with `recipe_commit_sha256` over `reveal_manifest.json` (pinning OCI `@sha256`, HF checkpoint commit SHA and shard SHA-256s, lockfile SHA-256, and CLI command). Final Official Season 1 Awards are decided strictly by Phase-2 organizer clean-hardware re-runs on a private held-out split (`+/- 8%` vs. Phase 1 acts strictly as an anti-fraud disqualification gate; maximum one finalist entry per lab per track and division).
