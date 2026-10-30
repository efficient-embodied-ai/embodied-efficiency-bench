# EmbodiedX Season 1 Submissions Guide

This directory stores community pull request bundles for EmbodiedX (`embodied-efficiency-bench` v1.0.0) across the two standardized Season 1 single-chip (`num_devices == 1`, `TP = PP = EP = 1`) cloud platforms:

- **Standard GPU (`nvidia_l40s_48gb`).** `NVIDIA L40S` (`48 GB` GDDR6, Ada SM89, `350 W` board TDP, native FP8 Tensor Cores, `362.05 TF` dense BF16, active `10 Hz NVML` board power telemetry). Available on demand via Shadeform (`~$0.88-$1.15/hr`), AWS EC2 `g6e.xlarge`, or the [NVIDIA Academic Grant Program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/).
- **Standard TPU (`tpu_v6e_1_32gb`).** `Google Cloud TPU v6e-1` (`32 GB` HBM2e, Trillium single-chip slice, `918 TF` dense BF16; isolate multi-chip hosts via `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0`). Available on demand via Google Cloud (`~$1.375/hr`) or free through a renewable 30-day [Google TPU Research Cloud (TRC)](https://sites.research.google/trc/) allocation.

Season 1 launches on **October 30, 2026** with **`0` pre-seeded vendor entries**. All evaluation pipelines, deterministic public trace loaders (`HuggingFaceVLA/libero` and `lerobot/droid_100`, with zero synthetic sine-wave traces permitted), reference servers (`PyTorch` eager and `torch.compile`, `openpi` `JAX`, and `TorchTPU`), and verification gates have been validated end-to-end on physical `NVIDIA L40S` and `Cloud TPU v6e-1` without seeding organizer scores so community pull requests establish the founding Pareto frontier directly.

## Season 1 Deadlines and Awards

- **Oct 30, 2026:** Season 1 leaderboard opens for Phase-1 pull requests.
- **Nov 10, 2026 (23:59 UTC):** **Checkpoint 1 (CoRL 2026 Live Spotlight Freeze).** Merged entries are featured live on stage on November 12, 2026 at the CoRL EEAI Workshop in Austin.
- **Dec 18, 2026 (23:59 AoE):** **Checkpoint 2 (Official Season 1 Awards Freeze)** followed by a 24-hour Phase-2 container reveal window for finalists. Continuous rolling submissions remain open in `2027+`.

Each track (`Track 1A`, `Track 1B`, `Track 2`) crowns winners across three scalar categories: **Interactive Reactivity (`C = 1`)** (`1000 / p99_ms` at `C = 1`, tie-broken by `p50_ms`, requiring `p99 / p50 <= 1.5`), **Real-Time Fleet Throughput** (highest SLA-compliant `Actions/s` or `Predictions/s`), and **Green Embodied Efficiency** (highest `MEASURED` `Actions/J` or `Predictions/J` via constant-cap `10 Hz NVML` board power on `L40S`; `--assume-power-watts` ineligible; `TPU v6e-1 = N/A`), with exact ties broken by lower-step Real-Time Dominated Hypervolume wrt `(0.5 * X_SLA, 0)`. Verified Season 1 category winners receive **monetary prizes**, an official **CoRL 2026 EEAI Workshop Award Certificate**, **an invitation to apply for Google Cloud credits**, and an invitation to co-author the post-season *EmbodiedX Season 1 Systems Report*.

## Track Specifications and Quality Requirements

| Track Directory and Pool | Pinned Checkpoint, Commit SHA, and Public Trace Set | Locked Horizon | Gapless Real-Time SLA | Division Verification Gates (`--num-samples >= 100`) |
| :--- | :--- | :--- | :--- | :--- |
| `submissions/track1_vla/` (**Track 1A: `pi0.5-LIBERO`**) | Physical Intelligence `pi0.5` at `lerobot/pi05_libero_finetuned_v044` (commit `7152f13e34c6c115d808aea0e5d08aca74eac14b`, ungated with pre-fit normalizers; 10 ODE steps, dual `224x224` RGB traces from `HuggingFaceVLA/libero` commit `86958911c0f959db2bbbdb107eb3e17c5f9c798e`) | `H = 10`, `K_eval = 5` | `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`) | **Closed:** `CosSim >= 0.995` + `CVaR-1%`.<br>**Open:** Stage-2 `LIBERO-RT` within `-5.0 pp`. |
| `submissions/track1_vla/` (**Track 1B: `GR00T N1.7-3B`**) | NVIDIA `GR00T N1.7-3B` at `nvidia/GR00T-N1.7-LIBERO` (`libero_10` checkpoint, commit `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`; 16 DiT steps, dual `256x256` RGB traces from `HuggingFaceVLA/libero`) | `H = 16`, `K_eval = 8` | `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`) | **Closed:** `CosSim >= 0.995` + `CVaR-1%`.<br>**Open:** Stage-2 `LIBERO-RT` within `-5.0 pp`. |
| `submissions/track2_wm/` (**Track 2: `Cosmos-Predict2.5-2B`**) | NVIDIA `Cosmos-Predict2.5-2B` at `nvidia/Cosmos-Predict2.5-2B` (`robot/action-cond` action-conditioned checkpoint, commit `85f8ae7bfe8f5525c8d103429524dcf12f98bf7b`; 10 steps, `CFG = 1.0`, `256x256` RGB traces from `lerobot/droid_100` commit `87301a2d2e99340e2010c9ef0f1d8e780b08aaf9`) | `K_eval = 16` predicted frames per rollout prediction | `p99 <= 1000.0 ms` (`>= 1.0 Pred/s`; strictly `Predictions/s` and `Predictions/J`) | **Closed:** `SSIM >= 0.88`, `PSNR >= 26.0 dB`, `CVaR-1%(SSIM) >= 0.80`, `CVaR-1%(PSNR) >= 22.0 dB`.<br>**Open:** + `LPIPS <= 0.12`, `FVD <= 120`, `InvDyn >= 0.99`, `Kendall tau >= 0.80`. |

## Two-Phase Commit-Reveal Directory Layout

Place every Phase-1 public pull request inside `submissions/track1_vla/<entry_id>/` or `submissions/track2_wm/<entry_id>/` using `eeb prepare-commit-bundle`, publishing only two public metadata files while keeping custom engine code, private `reveal_manifest.json`, and container images local until the contest freeze:

```text
submissions/track1_vla/<entry_id>/
├── bundle_summary.json      # Measured C in {1,2,4,8,16} sweep metrics, receipts, and gate outcomes
└── submission_meta.json     # Team metadata, track/division, and signed recipe_commit_sha256
```

1. **Canonical Commit Hash (`recipe_commit_sha256`).** `eeb prepare-commit-bundle` generates your private local `reveal_manifest.json` (kept inside `--run-dir` outside the git PR directory) pinning your immutable OCI container digest (`@sha256:...`), Hugging Face checkpoint commit SHA and shard SHA-256 digests, dependency lockfile SHA-256, and exact server/client CLI launch command, and writes only its SHA-256 commitment `recipe_commit_sha256` into `submission_meta.json`.
2. **Measurement Rules.** All traces must come deterministically from public `HuggingFaceVLA/libero` or `lerobot/droid_100` (zero synthetic sine-wave traces permitted). Sweep ordered canonical concurrencies `C in {1, 2, 4, 8, 16}` starting at `C = 1` (monotonic OOM tail truncation only; zero off-grid concurrencies and zero interpolation), scoring pooled steady-state `p99(C)` across active slots (`N >= 320` VLA / `N >= 200` WM) paired with the mandatory `<= 15%` per-slot completion fairness fence and `p99 / p50 <= 1.5` at `C = 1`. Warm up on the live model workload (`--warmup >= 30`) at stock vendor TDP and boost clocks, generate golden parity sets with `--num-samples >= 100`, and lock SM clocks (`nvidia-smi -lgc`, calibrated on the real model loop) only when dynamic boost trips the `<= 8%` 3-run `p99` variance gate. Per-frame `+-1/255` `uint8` micro-jitter, `max / p50 <= 4.0`, and zero host-CPU neural offload inside timed calls are strictly enforced.
3. **Phase-2 Finalist Reveal (`+24h` Post-Freeze).** Within 24 hours after `Dec 18, 2026 (23:59 AoE)`, shortlisted finalists (at most one slot per laboratory or organization per track and division) share their container image and `reveal_manifest.json` privately with the organizers. Official Season 1 Awards are ranked strictly by Phase-2 organizer clean-hardware re-runs on a private held-out trace split, with `+/- 8%` Phase-1 agreement acting strictly as an anti-fraud disqualification gate.

## Day-1 Quickstart and Local Bundle Verification Flow

Run the full six-step measurement and commit-bundle verification flow locally before opening a Phase-1 pull request:

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
