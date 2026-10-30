## EmbodiedX (`embodied-efficiency-bench` v1.0) Phase 1 Commit Submission PR

### 1. Target Milestone & Track
- [ ] **Checkpoint 1: CoRL 2026 Live Spotlight Freeze (`Nov 10, 2026, 23:59 UTC`; presented live `Nov 12, 2026` in Austin, TX)**
- [ ] **Checkpoint 2: Official Season 1 Awards Freeze (`Dec 18, 2026, 23:59 AoE`; Phase 2 private reveal due within 24h)**
- [ ] **Track 1A (`VLA_TRACK`)**: `pi05-3b` (locked `K_eval = 5` at 20 Hz, `p99 <= 250.0 ms`)
- [ ] **Track 1B (`VLA_TRACK`)**: `gr00t-n1.7-3b` (locked `K_eval = 8` at 20 Hz, `p99 <= 400.0 ms`)
- [ ] **Track 2 (`WM_TRACK`)**: `cosmos-predict2.5-2b` (`16 frames/pred`, `p99 <= 1000.0 ms`)

### 2. Official Season 1 Hardware Target & Division
- [ ] **`L40S_48GB`**: Single NVIDIA L40S (`48 GB` GDDR6, Ada SM89, `350 W` TDP, NVML board telemetry)
- [ ] **`TPU_V6E_1_32GB`**: Single Google Cloud TPU v6e-1 (`32 GB` HBM2e, Trillium)
- [ ] **Closed Division (`CLOSED`)**: Frozen reference weights, frozen `K_eval`, frozen denoising steps `N`, frozen native resolution and camera count.
- [ ] **Open Division (`OPEN`)**: Algorithmic modification with locked `K_eval` and passing `stage2_gate_b` receipt (`LIBERO` closed-loop SR `>= ref - 5.0 pp` for Track 1; 3-part WM gate with `S2` Kendall `tau >= 0.80` for Track 2).

### 3. Two-Phase Commit-Reveal Verification Checklist
- **Submission Folder**: `submissions/<track1_vla|track2_wm>/<team>_<sku>_<engine>_<YYYYMMDD>/`
- **Public PR Contents (Phase 1 ONLY)**: `bundle_summary.json` and `submission_meta.json` (do NOT commit proprietary kernels, model weights, Dockerfiles, or raw `report.json` during the live contest window).
- [ ] `eeb prepare-commit-bundle` generated and signed `bundle_summary.json` + `submission_meta.json`, embedding `recipe_commit_sha256` inside `attested_body`, while keeping `reveal_manifest.json` archived privately offline for Phase 2.
- [ ] `eeb verify-bundle <folder> --strict` passes locally (`status == CERTIFIED`, valid SHA-256 receipt signature, ordered concurrency prefix of `[1, 2, 4, 8, 16]` starting at `C = 1`).
- [ ] Entrant understands that Official Season 1 Track Awards (certificates, monetary prizes, an invitation to apply for Google Cloud credits, and Season 1 Systems Report invitation) are ranked strictly by Phase 2 Organizer Clean-Hardware Re-Measured scores on a private held-out trace split (`+/- 8%` vs. Phase 1 is an anti-fraud disqualification gate).
