# EmbodiedX (`embodied-efficiency-bench` v1.0.0)

Open, closed-loop serving benchmark for Vision-Language-Action (VLA) policies and embodied world models at the CoRL 2026 Workshop on Efficient Foundation Models for Real-Time Embodied AI ([EEAI @ CoRL 2026](https://efficient-embodied-ai.github.io/challenge/)).

Submissions execute on standardized single-chip cloud accelerators (`NVIDIA L40S 48 GB` / `nvidia_l40s_48gb` and `Google Cloud TPU v6e-1 32 GB` / `tpu_v6e_1_32gb`) and trace the single-chip latency-throughput-energy Pareto frontier under hard real-time robot control deadlines. All evaluation traces are fetched deterministically from public real-world datasets (`HuggingFaceVLA/libero` and `lerobot/droid_100`), with zero synthetic sine-wave traces permitted in submissions. Full methodology and citations appear in [`docs/EmbodiedX_Benchmark_Methodology_and_Metrics.pdf`](docs/EmbodiedX_Benchmark_Methodology_and_Metrics.pdf), [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md), [`docs/LITERATURE_REVIEW_AXES_AND_METRICS.md`](docs/LITERATURE_REVIEW_AXES_AND_METRICS.md), and [`submissions/README.md`](submissions/README.md).

## Season 1 Schedule and Awards

Season 1 launches on October 30, 2026 with `0` pre-seeded vendor entries on the public leaderboard. The evaluation harness, public dataset trace loaders, multi-runtime reference servers (`PyTorch` eager and `torch.compile`, `openpi` `JAX`, and `TorchTPU`), and eight-gate submission verifier have been validated end-to-end on physical `NVIDIA L40S` and `Cloud TPU v6e-1` without seeding organizer benchmark scores. Community pull requests establish the founding Pareto frontier directly.

| Milestone | Date | Protocol Event |
| :--- | :--- | :--- |
| **Season 1 Launch** | **Oct 30, 2026** | Benchmark v1.0.0 harness and public Season 1 leaderboard open for Phase-1 pull requests. |
| **Checkpoint 1 (CoRL Live Spotlight Freeze)** | **Nov 10, 2026 (23:59 UTC)** | Valid pull requests merged by this deadline are featured live on stage at CoRL 2026 in Austin on **Nov 12, 2026** (12:35 workshop session) as Founding Frontier Leaders. |
| **Checkpoint 2 (Official Season 1 Awards Freeze)** | **Dec 18, 2026 (23:59 AoE)** | Final Season 1 pull request freeze followed by a 24-hour Phase-2 private container reveal for clean-hardware organizer re-runs. Rolling benchmark submissions continue in `2027+`. |

**Winner Recognition and Perks.** Verified Season 1 category winners receive **monetary prizes**, an official **CoRL 2026 EEAI Workshop Award Certificate**, **an invitation to apply for Google Cloud credits**, permanent verified attribution on the public leaderboard, and an invitation to co-author the post-season *EmbodiedX Season 1 Systems Report*.

### Three Scalar Award Categories per Track

Closed-loop serving curves can cross as concurrency rises. Season 1 evaluates each independent track pool (`Track 1A`, `Track 1B`, and `Track 2`) across three scalar award categories anchored to each track's real-time control deadline:

| Award Category | Scalar Ranking Metric | Admissibility Gate |
| :--- | :--- | :--- |
| **Interactive Reactivity (`C = 1`)** | Highest single-stream rate `1000 / p99_ms` at concurrency `C = 1` (lowest `C = 1` `p99_ms`, tie-broken by lowest `p50_ms`). | Unimodal latency tail `p99 / p50 <= 1.5` at `C = 1` plus division quality gates (`>= 100` golden samples). |
| **Real-Time Fleet Throughput** | Highest aggregate `Actions/s` (Track 1) or `Predictions/s` (Track 2) across canonical grid points `C in {1, 2, 4, 8, 16}`. | Operating point must satisfy the track's gapless pooled `p99` SLA and `<= 15%` per-slot completion fairness fence. |
| **Green Embodied Efficiency** | Highest hardware-measured `Actions/J` (Track 1) or `Predictions/J` (Track 2) across SLA-compliant points in `{1, 2, 4, 8, 16}`. | `power_provenance == "MEASURED"` from active `10 Hz NVML` board power on `NVIDIA L40S` held at a constant power cap (`--assume-power-watts` ineligible; `TPU v6e-1 = N/A` where guest VMs do not expose board wattage). |

Exact scalar ties are broken by lower-step Real-Time Dominated Hypervolume relative to reference origin `(0.5 * X_SLA, 0)` computed strictly over SLA-compliant discrete points in `{1, 2, 4, 8, 16}` without linear interpolation.

## Benchmark Tracks and Gapless Real-Time SLAs

A standard `20 Hz` robot control loop steps every `dt = 50.0 ms` (`20.0 Chunks/s` single-tick reaction ceiling). Under asynchronous action-chunk execution with locked execution horizon `K_eval`, the policy must return the next chunk within `K_eval * dt` to prevent actuator queue starvation. EmbodiedX splits the two VLA manipulation policies (`Track 1A` and `Track 1B`, separated due to a `1.82x` FLOPs-per-action difference) from world-model video prediction (`Track 2`):

| Track | Pinned Canonical Checkpoint and Public Trace Dataset | Locked Horizon | Gapless Real-Time SLA | Closed and Open Quality Gates (`--num-samples >= 100`) |
| :--- | :--- | :--- | :--- | :--- |
| **Track 1A (`pi0.5-LIBERO`)** | Physical Intelligence `pi0.5` at `lerobot/pi05_libero_finetuned_v044` (commit `7152f13e34c6c115d808aea0e5d08aca74eac14b`, ungated with pre-fit normalizers; 10 Euler ODE steps, dual `224x224` RGB traces from `HuggingFaceVLA/libero` commit `86958911c0f959db2bbbdb107eb3e17c5f9c798e`) | `H = 10`, `K_eval = 5` | `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`) | **Closed:** `CosSim >= 0.995` + `CVaR-1%` floor.<br>**Open:** Stage-2 `LIBERO-RT` within `-5.0 pp`. |
| **Track 1B (`GR00T N1.7-3B`)** | NVIDIA `GR00T N1.7-3B` at `nvidia/GR00T-N1.7-LIBERO` (`libero_10` checkpoint, commit `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`; Cosmos-Reason2-2B VLM + 16-step DiT action head, dual `256x256` RGB traces from `HuggingFaceVLA/libero`) | `H = 16`, `K_eval = 8` | `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`) | **Closed:** `CosSim >= 0.995` + `CVaR-1%` floor.<br>**Open:** Stage-2 `LIBERO-RT` within `-5.0 pp`. |
| **Track 2 (`Cosmos-Predict2.5-2B`)** | NVIDIA `Cosmos-Predict2.5-2B` at `nvidia/Cosmos-Predict2.5-2B` (`robot/action-cond` action-conditioned checkpoint, commit `85f8ae7bfe8f5525c8d103429524dcf12f98bf7b`; 2B bidirectional 3D-DiT, 10 steps, `CFG = 1.0`, `256x256` RGB traces from `lerobot/droid_100` commit `87301a2d2e99340e2010c9ef0f1d8e780b08aaf9`) | `K_eval = 16` predicted frames per rollout prediction | `p99 <= 1000.0 ms` (`>= 1.0 Pred/s`; strictly `Predictions/s` and `Predictions/J`) | **Closed:** `SSIM >= 0.88`, `PSNR >= 26.0 dB`, `CVaR-1%(SSIM) >= 0.80`, `CVaR-1%(PSNR) >= 22.0 dB`.<br>**Open:** + `LPIPS <= 0.12`, `FVD <= 120`, `InvDyn >= 0.99`, `Kendall tau >= 0.80`. |

## Standard Season 1 Hardware and Pipeline Validation Matrix

Every Season 1 submission runs on a single physical accelerator chip (`num_devices == 1`, `TP = PP = EP = 1`) at stock vendor clock and power limits. Standardizing on one cloud GPU and one cloud TPU isolates software efficiency across kernel schedules, precision choices, compiler passes, and serving runtimes so organizers can reproduce every finalist cleanly on matching cloud silicon.

| Standard Platform | Hardware Specification | Supported Runtimes and Optimization Surface | Cloud Access and Telemetry | End-to-End Harness Status |
| :--- | :--- | :--- | :--- | :--- |
| **Standard GPU (`nvidia_l40s_48gb`)** | `NVIDIA L40S` (Ada SM89, `48 GB` GDDR6 at `864 GB/s`, `350 W` TDP, native FP8 Tensor Cores, `362.05 TF` dense BF16) | `PyTorch` (`eager` / `torch.compile`), `CUDA` / `Triton` kernels, `TensorRT`, FP8 / INT4 quantization, continuous batching | Shadeform (`~$0.88-$1.15/hr`), AWS EC2 `g6e.xlarge`, or [NVIDIA Academic Grant Program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/); active `10 Hz NVML` board energy | Servers, traces, and 8-gate verifier validated end-to-end (`0` organizer scores seeded) |
| **Standard TPU (`tpu_v6e_1_32gb`)** | `Google Cloud TPU v6e-1` (Trillium single-chip slice, `32 GB` HBM2e at `1,600 GB/s`, `918 TF` dense BF16, `1,836 TOPS` INT8/FP8) | Native `openpi` `JAX` (`jax.jit`), `Pallas` kernels, `PyTorch` + open-source `TorchTPU` (`torch.compile`), `XLA` optimizations | Google Cloud (`~$1.375/hr`) or free renewable 30-day [Google TPU Research Cloud (TRC)](https://sites.research.google/trc/) grant; power reported as `N/A` | Servers, single-chip isolation, and 8-gate verifier validated end-to-end (`0` organizer scores seeded) |

Onboard edge modules remain reserved for future seasons so Season 1 stays strictly reproducible on accessible cloud hardware.

## Closed and Open Divisions with Anti-Gaming Verification

- **Locked Execution Horizon Across Both Divisions.** Both `Closed` and `Open` divisions lock `K_eval` (`5` actions for Track 1A, `8` actions for Track 1B, `16` frames per rollout prediction for Track 2) so aggregate output cannot be inflated by extending open-loop chunk lengths. All neural forward-pass operators inside timed `act()` and `rollout()` calls must execute on the single target accelerator; frozen episode instruction embeddings may be pre-cached on the accelerator during untimed episode `reset()`.
- **`Closed` Division.** Preserves checkpoint architecture and pretrained weights at the fixed denoising step count under BF16/FP16/FP8 numerics validated by inline Golden-IO parity over `--num-samples >= 100` real trace samples (`CosSim >= 0.995` + `CVaR-1%` on Track 1; `SSIM >= 0.88`, `PSNR >= 26.0 dB`, `CVaR-1%(SSIM) >= 0.80`, and `CVaR-1%(PSNR) >= 22.0 dB` on Track 2). Pseudo-causal sub-clip frame slicing (`stream_chunk_frames > 0`) on Track 2 is prohibited.
- **`Open` Division.** Permits INT4 quantization, pruning, solver step distillation, and visual token merging when accompanied by a signed Stage-2 quality receipt verifying latency-coupled `LIBERO-RT` closed-loop success (within `-5.0 pp` under actuation delay `d = ceil(L / dt)` for Track 1) or perceptual, distribution, inverse-dynamics, and policy-ranking fidelity (`SSIM >= 0.88`, `PSNR >= 26.0 dB`, `CVaR-1%(SSIM) >= 0.80`, `CVaR-1%(PSNR) >= 22.0 dB`, `LPIPS <= 0.12`, `FVD <= 120`, `InvDyn >= 0.99`, `Kendall tau >= 0.80` for Track 2).
- **Discrete Concurrency Grid, Pooled `p99(C)`, and Anti-Gaming Gates.** The closed-loop load generator sweeps `C in {1, 2, 4, 8, 16}` starting at `C = 1` (permitting only monotonic tail truncation on device OOM, with zero off-grid points and zero curve interpolation). Every point requires `N = 3` runs measuring pooled steady-state `p99(C)` across active slots (`N >= 320` chunks/run on Track 1 VLA, `N >= 200` predictions/run on Track 2 WM) paired with a mandatory `<= 15%` per-slot completion fairness fence, within `<= 8%` inter-run `p99` spread and `p99 / p50 <= 1.5` at `C = 1`. Each request applies per-frame cryptographic `+-1/255` `uint8` micro-jitter on real `HuggingFaceVLA/libero` and `lerobot/droid_100` frames, opaque `UUIDv4` session IDs, native wire-resolution checks (`>= 224x224`), `max / p50 <= 4.0` anti-cache-miss bounding, and inline blind Golden-IO and `G4` sensitivity canaries.
- **Thermal and Clock Discipline (`D4`).** Warm up on the real model workload (`--warmup >= 30`) at stock vendor boost clocks and stock board TDP first. Lock GPU SM clocks via `nvidia-smi -lgc` (calibrated directly on the live model loop instead of synthetic GEMM power burners) only when dynamic boost drift exceeds the inter-run variance gate. Above-stock TDP limits and overclocking are forbidden.

## Two-Phase Commit-Reveal Submission Protocol

1. **Phase 1 (Public Leaderboard Pull Request).** Run `eeb prepare-commit-bundle` and `eeb verify-bundle --strict` to package `bundle_summary.json` and `submission_meta.json` under `submissions/track1_vla/<entry_id>/` or `submissions/track2_wm/<entry_id>/`. `submission_meta.json` records `recipe_commit_sha256`, the SHA-256 digest of a canonical `reveal_manifest.json` pinning your immutable OCI container image (`@sha256`), Hugging Face checkpoint commit SHA and shard SHA-256 digests, lockfile SHA-256, and exact launch CLI command without leaking custom kernel code or recipe configs. Your measured Pareto curve appears immediately on the public leaderboard while custom kernels and serving code remain private until the contest freeze.
2. **Phase 2 (Post-Freeze Private Reveal and Clean-Hardware Re-run).** Within 24 hours after `Dec 18, 2026 (23:59 AoE)`, shortlisted finalists (at most one finalist slot per laboratory or organization per track and division) privately supply their matching container image and `reveal_manifest.json` to the organizers. Final Official Season 1 Awards are ranked strictly by Phase-2 organizer re-runs on clean cloud hardware using a private held-out trace split; agreement within `+/- 8%` of Phase-1 numbers acts strictly as an anti-fraud disqualification gate. Confirmed Season 1 winners publish reproducible open recipes (`Verified Open Recipe`). Rolling submissions in `2027+` support both `Verified Open Recipe` and `Proprietary / Audit-Only` attribution.

## Day-1 Quickstart Commands

```bash
git clone https://github.com/efficient-embodied-ai/embodied-efficiency-bench.git
cd embodied-efficiency-bench && pip install -e ".[vla,gpu]"

eeb data prepare-traces --track vla --split mini --out ./data/traces
eeb data generate-golden --model pi05 --traces ./data/traces --num-samples 100 --out ./data/pi05_golden.npz
eeb serve --model pi05 --checkpoint lerobot/pi05_libero_finetuned_v044 --port 8080 &

eeb measure --track track1a --division closed --hardware nvidia_l40s_48gb \
  --server http://127.0.0.1:8080 --traces ./data/traces --golden ./data/pi05_golden.npz \
  --sweep 1,2,4,8,16 --warmup 30 --calls 320 --runs 3 --strict \
  --out ./runs/my_pi05_l40s

eeb prepare-commit-bundle --run-dir ./runs/my_pi05_l40s \
  --output ./submissions/track1_vla/my_lab_pi05_l40s
eeb verify-bundle ./submissions/track1_vla/my_lab_pi05_l40s --strict
```

For single-chip isolation before launching servers, set `export CUDA_VISIBLE_DEVICES=0` on `NVIDIA L40S` or `export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0` on `Google Cloud TPU v6e-1`.

## Coding Agent and Optimization Harness Integration

`embodied-efficiency-bench` operates cleanly as an immutable single-chip evaluator for autonomous kernel and serving optimization loops:

- **Repository Agent Guides.** Point interactive coding agents or evolutionary search drivers to [`skills/embodiedx-agent-kickoff/SKILL.md`](skills/embodiedx-agent-kickoff/SKILL.md), [`skills/embodiedx-submission-auditor/SKILL.md`](skills/embodiedx-submission-auditor/SKILL.md), [`AGENTS.md`](AGENTS.md), and [`CLAUDE.md`](CLAUDE.md). Candidate improvements are scored deterministically via `eeb compare --baseline <prev_run>/report.json --candidate <new_run>/report.json --json` (`exit 0` accepts, `exit 2` rejects).
- **Cloud TPU Optimization Stacks.** Compatible with [`google-pytorch/torch_tpu`](https://github.com/google-pytorch/torch_tpu), [`google-pytorch/tpu-agent-suite`](https://github.com/google-pytorch/tpu-agent-suite), [`AI-Hypercomputer/accelerator-agents`](https://github.com/AI-Hypercomputer/accelerator-agents) (`MaxKernel` / `MaxCode` for `Pallas` and `TorchTPU`), and [`algorithmicsuperintelligence/openevolve`](https://github.com/algorithmicsuperintelligence/openevolve).
- **GPU Optimization Stacks.** Compatible with [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch), Sakana AI [`AI CUDA Engineer`](https://pub.sakana.ai/ai-cuda-engineer/), and [`ScalingIntelligence/KernelBench`](https://github.com/ScalingIntelligence/KernelBench).
