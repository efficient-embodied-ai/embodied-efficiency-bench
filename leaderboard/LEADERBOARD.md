# EmbodiedX (`embodied-efficiency-bench` v1.0): Official Season 1 Leaderboard

> **FOUNDING SEASON v1.0 OPEN FOR SUBMISSIONS (ZERO PRE-POPULATED CURVES):** EmbodiedX Season 1 launches with zero organizer-populated GPU or TPU curves. All rankings and Pareto frontiers are populated strictly from verified community and partner Phase 1 Commit-Reveal bundles (`bundle_summary.json` + `submission_meta.json`) and ranked for Official Season 1 Awards strictly by Phase 2 Organizer Clean-Hardware Re-Measurement.

- **Generated At:** `2026-10-30T12:14:30-07:00 (PDT)`
- **Official Season 1 Competition Hardware (`CAP_SINGLE_CHIP`, `num_devices == 1`):** **NVIDIA GPU Target:** `NVIDIA L40S (48 GB GDDR6, Ada SM89, 350 W TDP, native FP8)` | **Cloud TPU Target:** `Google Cloud TPU v6e-1 (32 GB HBM2e, Trillium)`
- **Season 1 Checkpoints & Awards:** **Oct 30, 2026** (v1.0 launch) -> **Nov 10, 2026 (23:59 UTC)** (Checkpoint 1: CoRL 2026 Live Spotlight Freeze) -> **Nov 12, 2026** (CoRL 2026 EEAI Workshop Live Showcase, Austin, Texas) -> **Dec 18, 2026 (23:59 AoE)** (Checkpoint 2: Official Season 1 Awards Freeze + 24h Phase 2 Reveal). Recognitions include official CoRL 2026 EEAI Workshop Award Certificates, monetary prizes, an invitation to apply for Google Cloud credits, and an invitation for verified Open-Recipe winners to contribute to the EmbodiedX Season 1 Benchmark Report.
- **Gapless Real-Time Control SLAs (Locked `K_eval` Across Both `CLOSED` and `OPEN`):** Track 1A `pi05-3b` (`K_eval = 5`, 50 ms step): `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`); Track 1B `gr00t-n1.7-3b` (`K_eval = 8`, 50 ms step): `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`); Track 2 `cosmos-predict2.5-2b` (`16 frames/pred`): `p99 <= 1000.0 ms` (`>= 1.0 Pred/s`).
- **Track Award Metrics (per `(Track, Model, Division)`):** **Award A (`Interactive Reactivity`)** highest `1000.0 / p99_ms` at `C == 1` with `p99_ms / p50_ms <= 1.5`; **Award B (`Real-Time Fleet Throughput`)** max locked-`K_eval` `Actions/s` or `Predictions/s` under SLA over `{1, 2, 4, 8, 16}`; **Award C (`Green Embodied Efficiency`)** max locked-`K_eval` `Actions/J` or `Predictions/J` under SLA (`MEASURED` telemetry only, constant `power_limit_w`); tie-broken by normalized Real-Time Dominated Hypervolume over fixed `(u_ref = 0.5, y_ref = 0.0)`.
- **Verified Submissions:** `0` | **Quarantined/Rejected:** `0`
- **Models Covered:** `Awaiting Founding Submissions (pi05-3b, gr00t-n1.7-3b, cosmos-predict2.5-2b)`
- **Vendor Families:** `Awaiting Founding Submissions` (`L40S_48GB, TPU_V6E_1_32GB`)

## 1. Season 1 Leaderboard Status (Awaiting Founding Submissions)

| Track | Canonical Model | Locked `K_eval` / Horizon | Gapless Real-Time SLA | Official GPU Target | Official TPU Target | Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `VLA_TRACK` (1A) | `pi05-3b` | `K_eval = 5` (250 ms chunk, 50 ms step) | `p99 <= 250.0 ms` (`>= 4.0 Chunks/s`) | `L40S_48GB` | `TPU_V6E_1_32GB` | Open for Phase 1 PRs |
| `VLA_TRACK` (1B) | `gr00t-n1.7-3b` | `K_eval = 8` (400 ms chunk, 50 ms step) | `p99 <= 400.0 ms` (`>= 2.5 Chunks/s`) | `L40S_48GB` | `TPU_V6E_1_32GB` | Open for Phase 1 PRs |
| `WM_TRACK` (2) | `cosmos-predict2.5-2b` | `16 frames/pred` (1.0 s horizon) | `p99 <= 1000.0 ms` (`>= 1.0 Rollouts/s`) | `L40S_48GB` | `TPU_V6E_1_32GB` | Open for Phase 1 PRs |

## 2. Submit a Phase 1 Commit-Reveal Entry

```bash
eeb measure --recipe recipes/track1_vla_pi05_l40s_bf16.yaml --endpoint grpc://127.0.0.1:50051 --golden /path/to/golden.npz --strict --out /tmp/my_run
eeb prepare-commit-bundle --source /tmp/my_run --output submissions/track1_vla/<team>_l40s_trt_20261108 --recipe-commit-sha256 <sha256_of_canonical_reveal_manifest_json> --team '<Team Name>' --engine-tag tensorrt-fp8
eeb verify-bundle --submission submissions/track1_vla/<team>_l40s_trt_20261108
```
