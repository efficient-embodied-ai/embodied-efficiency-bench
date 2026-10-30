---
name: embodiedx-submission-auditor
description: Cryptographic SHA-256 receipt verifier, AST anti-cheat auditor, eeb compare verifier, CAP_SINGLE_CHIP hardware enforcer, and PR submission bundler for EmbodiedX (embodied-efficiency-bench v1.0).
---

# Skill: EmbodiedX Submission & Anti-Cheat Auditor (`embodiedx-submission-auditor`)

Use this skill before opening a Pull Request to `submissions/` or claiming a measured single-chip win on `embodied-efficiency-bench` v1.0.

## 1. Six Mandatory Pre-PR Audit Checks

1. **Single-Chip Hardware Envelope (`CAP_SINGLE_CHIP`)**:
   * `num_devices == 1` (`TP=1, PP=1, EP=1`) on a qualifying SKU (`<= 64 GB` VRAM/HBM, `<= 1,000 BF16 TFLOPS`, `<= 2,000 FP8/INT8 TOPS`, `<= 2,000 GB/s`).
   * Rejects CPU-only `--allow-no-device` self-test bundles (`status="INVALID"`, `num_devices=0`) and multi-device bypasses (`CUDA_VISIBLE_DEVICES=0,1`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`).
2. **Track & Division Compliance (`D1` & `D2`)**:
   * Track 1A (`VLA_TRACK`): `pi05-3b` (`lerobot/pi05_libero_finetuned_v044`, `execute_horizon = 5`, `H = 10`, dual `224x224` RGB, gapless SLA cutoff `>= 4.0 Chunks/s` [$L_{p99} \le 250\text{ ms}$], `N >= 320` timed chunks/run, `--warmup >= 30`).
   * Track 1B (`VLA_TRACK`): `gr00t-n1.7-3b` (`nvidia/GR00T-N1.7-LIBERO`, `execute_horizon = 8`, `H = 16`, dual `256x256` RGB, gapless SLA cutoff `>= 2.5 Chunks/s` [$L_{p99} \le 400\text{ ms}$], `N >= 320` timed chunks/run, `--warmup >= 30`).
   * Track 2 (`WM_TRACK`): `cosmos-predict2.5-2b` (`nvidia/Cosmos-Predict2.5-2B`, `num_frames = 16`, `256x256` RGB, real-time SLA cutoff `>= 1.0 Predictions/s` [$L_{p99} \le 1000\text{ ms}$], `N >= 200` timed rollouts/run, `--warmup >= 30`; `stream_chunk_frames > 0` / `streaming_first_frame_vae_decode` banned in `CLOSED` division).
   * Closed Division requires frozen checkpoint weights, full denoising step counts, and 100% neural forward-pass execution inside timed calls on the single target accelerator (pre-caching frozen episode instruction text embeddings on the accelerator during episode `reset()` is permitted under `D1`, whereas host-CPU neural offload `cpu_text_encoder_offload_cache`, `step_distillation`, and dataset fine-tuning require `OPEN` division).
3. **Co-Located Quality, Native Wire Resolution & Blind Inline Canary Gates (`G1`-`G4`, `D3`, `D10`)**:
   * Non-synthetic submissions transmitted native wire resolution ($\ge 224\times 224$ RGB, $\ge 150{,}528\text{ B/req}$).
   * Every measured operating point ran with `--golden <bf16_golden.npz>` (`quality_gate.passed == true`, `quality_gate.quality_certified == true`, and `0` inline canary failures across opaque random `UUIDv4` `session_id` calls).
   * Live BF16-safe Gate `G4` perturbation probe (`10%` RGB pixels $\pm 2\text{ LSB}$ + $10^{-2}$ state/action shift, `l2_output_shift >= 1e-5`) passed inside the same `execution_id` in seeded-shuffled order.
   * Only variance-stable (`variance_passed == true`: throughput spread `<= 5%`, pooled steady-state `latency_p99_ms` spread `<= 8%`, `slot_count_spread <= 15%`, and `C=1 p99/p50 <= 1.5`) and gate-passing (`gate_a_passed == true`) points enter `pareto_frontier`. Official `Per Watt` award eligibility further requires `power_provenance == "MEASURED"` from live hardware power sensors.
4. **AST Source Scan**:
   * Submission directory contains zero prohibited multi-GPU spawns, zero `time.sleep` tampering, and zero static output caching shortcuts.
5. **Cryptographic Bundle Verification**:
   * `receipt.json` SHA-256 signature matches `attested_body`, top-level `operating_points` / `pareto_frontier` / `summary_metrics` match the attested payload, and sibling `report.json` recomputes to the exact same attestation digest.
6. **Mechanical Improvement Check (`eeb compare`)**:
   * Candidate beats the reference baseline at the track's real-time SLA cutoff (`>= 4.0 Chunks/s` Track 1A, `>= 2.5 Chunks/s` Track 1B, or `>= 1.0 Predictions/s` Track 2) with zero quality or horizon violations.

## 2. Verbatim Audit Commands

```bash
# 1. Verify bundle cryptographic SHA-256 receipt and sibling report.json parity (exit 0)
eeb verify-receipt --submission submissions/<team_submission_dir>

# 2. Run full measured-bundle schema, envelope, division, and AST validator (exit 0)
eeb validate-submission \
  --submission submissions/<team_submission_dir> \
  --output /tmp/validation_report.json

# 3. Compare candidate bundle against the local baseline sweep (exit 0 = ACCEPT, exit 2 = REJECT)
eeb compare \
  --baseline runs/00_champion/report.json \
  --candidate submissions/<team_submission_dir>/report.json

# 4. Run latency-coupled LIBERO-RT check (Track 1) or WM policy-ranking check (Track 2)
eeb eval-libero-rt \
  --recipe submissions/<team_submission_dir>/recipe.json \
  --sweep-report submissions/<team_submission_dir>/report.json \
  --output /tmp/libero_rt_audit.json

# 5. Dry-run leaderboard compilation locally
eeb build-leaderboard \
  --submissions-dir submissions \
  --output /tmp/leaderboard_preview.json \
  --markdown-output /tmp/LEADERBOARD_PREVIEW.md
```
