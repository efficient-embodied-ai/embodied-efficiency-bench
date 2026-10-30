---
description: Start up embodied-efficiency-bench v1.0 on a single GPU/TPU/SoC, generate frozen BF16 goldens, launch eeb serve + eeb measure, score candidates with eeb compare, and verify PR receipts.
argument-hint: "<model + SKU prompt, e.g. 'Start pi0.5 on TPU v6e-1 in CLOSED division'>"
---

Execute the EmbodiedX (`embodied-efficiency-bench` v1.0) Agent & Harness Startup workflow for: `$ARGUMENTS`

1. **Bootstrap Study & Verify Single-Chip Envelope (`CAP_SINGLE_CHIP`, `num_devices == 1`)**:
   ```bash
   python3 -m embodied_bench.agent_kickoff --prompt "$ARGUMENTS"
   eeb envelope check
   eeb telemetry probe --seconds 2 --hz 10
   ```
   Read the generated `<study_dir>/STUDY_PLAN.json`, `<study_dir>/EXPANDED_AGENT_PROMPT.md`, and `skills/embodiedx-agent-kickoff/SKILL.md`.

2. **Prepare Native-Resolution Trace Bundle & Frozen `BF16` Golden Targets**:
   ```bash
   # Use --image-hw 224x224 for pi05-3b, or --image-hw 256x256 for gr00t-n1.7-3b / cosmos-predict2.5-2b:
   eeb data prepare-traces --out <study_dir>/eval_traces --image-hw 224x224
   eeb data inspect <study_dir>/eval_traces
   eeb data perturb <study_dir>/eval_traces --split public
   # Start unmodified BF16 reference server (use "device": "tpu" via TorchTPU/JAX on Cloud TPU, or "device": "cuda" on CUDA/ROCm):
   eeb serve --policy eeb.servers.pi05:Pi05Policy --uds /tmp/eeb_ref.sock --policy-kwargs '{"device": "cuda"}' &
   eeb data generate-golden <study_dir>/eval_traces --endpoint unix:///tmp/eeb_ref.sock --out <study_dir>/golden_ref.npz --split public
   ```

3. **Anchor Champion Baseline & Evaluate Candidates with `eeb measure` + `eeb compare`**:
   - Treat `eeb/loadgen/`, `eeb/telemetry/`, `eeb/envelope/`, and `embodied_bench/attestation.py` as immutable read-only evaluator files. Never strip device sync barriers or modify locked `execute_horizon` (`K=5, H=10` `pi05-3b`, `K=8, H=16` `gr00t-n1.7-3b`, `num_frames=16` `cosmos-predict2.5-2b`).
   ```bash
   eeb measure --recipe <study_dir>/baseline_recipe.yaml --endpoint unix:///tmp/eeb_ref.sock \
     --traces <study_dir>/eval_traces --golden <study_dir>/golden_ref.npz --strict --out <study_dir>/runs/00_champion
   eeb compare --baseline <study_dir>/runs/00_champion --candidate <study_dir>/runs/cand_01 --json
   ```

4. **Verify Cryptographic Receipt & Submission Package**:
   ```bash
   eeb validate-submission --submission <study_dir>/runs/cand_01
   eeb verify-receipt --submission <study_dir>/runs/cand_01
   ```
