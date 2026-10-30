# CLAUDE.md (`embodied-efficiency-bench` v1.0)

Read [`AGENTS.md`](AGENTS.md) and [`skills/embodiedx-agent-kickoff/SKILL.md`](skills/embodiedx-agent-kickoff/SKILL.md) first before modifying models, kernels, servers, recipes, or benchmark tooling.

## Quick Commands

```bash
# Run full test suite and strict recipe check
pytest -q
python3 scripts/validate_recipes.py --recipes-dir recipes --strict

# Single-chip hardware envelope & telemetry preflight
eeb envelope check --sku RTX_5090_32GB
eeb telemetry probe --seconds 2 --hz 10

# Optional study bootstrap (writes STUDY_PLAN.json + EXPANDED_AGENT_PROMPT.md + baseline_recipe.yaml)
python3 -m embodied_bench.agent_kickoff --prompt "Start pi0.5 on TPU v6e-1 in CLOSED division"

# Prepare native-resolution wire traces (>= 224x224 required by Gate D10 on real hardware;
# NOTE: data/embodiedx_traces_v1_mini is 64x64 strictly for --allow-no-device / synthetic CI smoke tests)
eeb data prepare-traces --track vla --split mini --out data/embodiedx_traces_v1_224 --image-hw 224x224

# Serve policy ("device": "tpu" via TorchTPU on Cloud TPU, "device": "cuda" on CUDA/ROCm), measure, and compare
eeb serve --model pi05 --checkpoint lerobot/pi05_libero_finetuned_v044 --uds /tmp/eeb.sock --policy-kwargs '{"device": "cuda"}' &
eeb data generate-golden data/embodiedx_traces_v1_224 --endpoint unix:///tmp/eeb.sock --out runs/golden_bf16.npz --split public
eeb measure --recipe recipes/track1_vla_pi05_l40s.yaml --endpoint unix:///tmp/eeb.sock \
  --traces data/embodiedx_traces_v1_224 --golden runs/golden_bf16.npz --warmup 30 --calls 320 --runs 3 --strict --out runs/00_champion
eeb compare --baseline runs/00_champion --candidate runs/cand_01

# Package Phase-1 commit bundle & verify cryptographic SHA-256 receipt
eeb prepare-commit-bundle --run-dir runs/cand_01 --output submissions/track1_vla/my_lab_pi05_l40s
eeb verify-bundle submissions/track1_vla/my_lab_pi05_l40s --strict
```

## Non-Negotiable Architecture Rules (v1.0)
- **Single-Chip Only (`CAP_SINGLE_CHIP`, `num_devices == 1`)**: Reject multi-GPU/multi-TPU sharding (`TP > 1`, `PP > 1`, `EP > 1`) and `> 64 GB` datacenter SKUs (`H100_SXM_80GB`, `B200_SXM6`).
- **Locked Closed-Division Scope**: Track 1 (`VLA_TRACK`: `pi05-3b` with `K=5`, `gr00t-n1.7-3b` with `K=8`) and Track 2 (`WM_TRACK`: `cosmos-predict2.5-2b` with `num_frames=16`).
- **PyTorch (`TorchTPU`) First on Cloud TPU**: Default policy servers use PyTorch across CUDA, ROCm, and Cloud TPU (`TorchTPU` / `torch_tpu`, `device="tpu"`, `torch.tpu.synchronize()`, optional `torch.compile()`); native `JAX/Flax` policy classes are supported as optional alternatives.
- **Immutable Evaluator & Co-Located Gates (`G1`-`G4`)**: Never edit `eeb/loadgen/`, `eeb/telemetry/`, `eeb/envelope/`, or `embodied_bench/attestation.py` to inflate candidate metrics, never strip device synchronization barriers inside `predict()` / `rollout()`, and always pass `--golden` on scored runs.
- **Crash-Safe Persistence**: All stateful file writes must use temp-file + `os.fsync` + `os.replace` with lockfiles stored in the OS temp directory (never leave `.lock` or orphan `.tmp` files inside repository directories).
