---
name: embodiedx-agent-kickoff
description: Agent and external harness startup skill for EmbodiedX (embodied-efficiency-bench v1.0 / CoRL 2026 EEAI Challenge). Teaches coding agents and GPU/TPU hill-climbing harnesses how to verify the single-chip envelope, build frozen BF16 goldens, launch eeb serve over UDS, run immutable eeb measure sweeps, score candidates with eeb compare (exit 0 ACCEPT / exit 2 REJECT), and package verified submission bundles.
---

# Skill: EmbodiedX v1.0 Agent & External Harness Startup (`embodiedx-agent-kickoff`)

Use this skill whenever an interactive coding agent (Claude Code, Gemini CLI, OpenAI Codex CLI, Cursor, OpenHands) or an external automated GPU/TPU hill-climbing harness needs to **start up, configure, run, and evaluate candidates** against `embodied-efficiency-bench` (`eeb` v1.0).

## 1. Benchmark Contract Summary (`D1`, `D2`, `D3`, `D8`)

1. **Divisions (`D1`)**: Season 1 tracks both **Closed Division** (locked weights, step counts, image resolution, and full precision / calibrated quantization rules) and **Open Division** (custom pruning, distillation, adaptive step schedules, and architectural changes verified via Stage-2 accuracy gates):
   * **Track 1A (`VLA_TRACK`, Flow Matching)**:
     - `pi05-3b` (`lerobot/pi05_libero_finetuned_v044` @ `7152f13e34c6c115d808aea0e5d08aca74eac14b` / `Physical-Intelligence/openpi` `pi05_libero`): dual `224x224` RGB, locked `execute_horizon = 5` ($K=5$, native LIBERO chunk $H=10$, $\Delta t = 50\text{ ms}$, 10 Euler ODE steps, gapless real-time SLA cutoff `>= 4.0 Chunks/s` [$L_{p99} \le 250\text{ ms}$]; both native JAX/Flax `openpi` and PyTorch `lerobot`/`TorchTPU` are co-equal certified Closed-Division runtimes).
   * **Track 1B (`VLA_TRACK`, DiT)**:
     - `gr00t-n1.7-3b` (`nvidia/GR00T-N1.7-LIBERO` @ `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`, `Qwen3-VL-2B` VLM + 16-step DiT flow head): dual `256x256` RGB, locked `execute_horizon = 8` ($K=8$, native chunk $H=16$, $\Delta t = 50\text{ ms}$, gapless real-time SLA cutoff `>= 2.5 Chunks/s` [$L_{p99} \le 400\text{ ms}$]).
   * **Track 2 (`WM_TRACK`)**:
     - `cosmos-predict2.5-2b` (`nvidia/Cosmos-Predict2.5-2B` @ `85f8ae7bfe8f5525c8d103429524dcf12f98bf7b`): full-clip bidirectional 3D-DiT at locked `256x256` RGB, `num_frames = 16` ($K_{\text{eval}}=16$, real-time SLA cutoff `>= 1.0 Predictions/s` [$L_{p99} \le 1000\text{ ms}$]; throughput reported strictly in `Predictions/s` and energy efficiency strictly in `Predictions/J`, with `stream_chunk_frames > 0` / `streaming_first_frame_vae_decode` forbidden in `CLOSED` division).
   * **Host-CPU Neural Offload Ban (`D1`)**: 100% of neural network forward-pass operators (`MatMul`, `Attention`, `Conv`, `MLP`, `VAE`, text encoders) inside timed `act()` / `rollout()` calls must execute on the single target accelerator (host CPU threads are restricted strictly to I/O, tokenization, orchestration, and action queuing; pre-caching frozen episode instruction text embeddings on the accelerator during episode `reset()` is permitted, while `cpu_text_encoder_offload_cache` is forbidden).
2. **Single-Chip Hardware Envelope (`CAP_SINGLE_CHIP`, `D8` & `D9`)**:
   * Strictly `num_devices == 1` (`TP=1, PP=1, EP=1`, `<= 64 GB` VRAM/HBM, `<= 1,000` FP32-accum dense BF16 TFLOPS, `<= 2,000 FP8/INT8 TOPS`, `<= 2,000 GB/s`). Lock GPU SM clocks (`nvidia-smi -lgc f,f`) at the highest fixed frequency sustainable below board TDP without throttling across $C \in \{1..16\}$.
   * Two leaderboard classes: **Onboard Edge (`<= 16 GB`)** (`JETSON_ORIN_NX_16GB`) and **Single-Accelerator (`<= 64 GB`)** (`TPU_V6E_1_32GB` [sole canonical highlighted Cloud TPU target], `JETSON_AGX_THOR_64GB`, `RTX_5090_32GB` [`209.5 TF` BF16], `L40S_48GB`, `RTX_4090_24GB` [`82.6 TF` FP32-accum BF16], `L4_24GB` [`121.0 TF` BF16, `300 GB/s`, `72W`], `AMD_MI210_64GB`, `AMD_W7900_48GB`).
   * **Official Awards (`D9`)**: Awarded independently for `Track 1A (pi0.5-3B)`, `Track 1B (GR00T-N1.7-3B)`, and `Track 2 (Cosmos-Predict2.5-2B)`. Official `Per Watt` award eligibility requires live hardware power telemetry (`power_provenance == "MEASURED"`; public GCE `Cloud TPU v6e-1` competes officially on `Y1 Per Chip`, `Y3 Per $/hr`, and `Y4 Iso-Compute`).
3. **Immutable Evaluator Rule & Anti-Gaming Protocol (`D10`)**:
   * Treat `eeb/loadgen/`, `eeb/telemetry/`, `eeb/envelope/`, and `embodied_bench/attestation.py` as **read-only evaluator code**.
   * **NEVER** remove device synchronization barriers (`synchronize_torch_device()`, `torch.cuda.synchronize()`, `torch.tpu.synchronize()`, `jax.block_until_ready()`) from `predict()` or `rollout()`.
   * **NEVER** alter the locked `execute_horizon` ($K=5$, $K=8$, or $H_{\text{frames}}=16$), set `stream_chunk_frames > 0` in Closed Division, substitute $< 224\times 224$ wire thumbnails, offload neural layers to host CPU, or inspect `session_id` (randomized `UUIDv4` with blind inline BF16-safe perturbation + golden canaries inside every timed sweep).

## 2. Step-by-Step Startup Sequence

### Step 1: Install & Single-Chip Hardware / Telemetry Preflight

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[grpc,data]"

# Lock visibility to exactly 1 physical accelerator:
#   NVIDIA GPU / Jetson: export CUDA_VISIBLE_DEVICES=0
#   Google Cloud TPU v6e-1 (or sliced multi-chip host):
#     export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0
#   AMD ROCm GPU:        export HIP_VISIBLE_DEVICES=0
eeb envelope check --sku RTX_5090_32GB
eeb telemetry probe --seconds 2 --hz 10
```

Optionally generate a persistent study scaffold (`STUDY_PLAN.json`, `EXPANDED_AGENT_PROMPT.md`, and `baseline_recipe.yaml`):

```bash
python3 -m embodied_bench.agent_kickoff \
  --prompt "Start pi0.5 on RTX 5090 in CLOSED division" \
  --study-dir studies/pi05_rtx5090_closed
```

### Step 2: Native Trace Bundle Preparation & Frozen `BF16` Golden Generation

Prepare a native-resolution evaluation trace bundle (`224x224` for `pi05-3b`, `256x256` for `gr00t-n1.7-3b` and `cosmos-predict2.5-2b`; note that committed `data/embodiedx_traces_v1_mini` is `64x64` CI-only) and generate your unmodified `BF16` reference golden file once before changing any server/kernel code:

```bash
eeb data prepare-traces --out runs/eval_traces_224 --image-hw 224x224
eeb data inspect runs/eval_traces_224
eeb data perturb runs/eval_traces_224 --split public

# Start the unmodified BF16 reference server over Unix domain socket (<= 107 bytes):
#   - CUDA / ROCm: '{"checkpoint_dir": "/path/to/ckpt", "device": "cuda", "dtype": "bfloat16"}'
#   - Cloud TPU (TorchTPU or native JAX Pi05JaxPolicy): '{"checkpoint_dir": "/path/to/ckpt", "device": "tpu", "dtype": "bfloat16"}'
eeb serve --policy eeb.servers.pi05:Pi05Policy \
  --uds /tmp/eeb_policy.sock \
  --policy-kwargs '{"checkpoint_dir": "/path/to/ckpt", "device": "cuda", "dtype": "bfloat16"}' &

# Wait until stdout prints EEB_SERVER_READY, then freeze the BF16 golden targets:
eeb data generate-golden runs/eval_traces_224 \
  --endpoint unix:///tmp/eeb_policy.sock \
  --out runs/golden_bf16.npz \
  --split public
```

### Step 3: Measure the Baseline Champion Curve (`runs/00_champion`)

```bash
eeb measure \
  --recipe recipes/track1_vla_pi05_l40s.yaml \
  --endpoint unix:///tmp/eeb_policy.sock \
  --traces runs/eval_traces_224 \
  --golden runs/golden_bf16.npz \
  --sweep 1,2,4,8,16 \
  --warmup 30 --calls 320 --runs 3 \
  --strict \
  --out runs/00_champion
```

For Track 1B (`gr00t-n1.7-3b`) and Track 2 (`cosmos-predict2.5-2b`), prepare `--image-hw 256x256` traces and use `--calls 320` (Track 1B) or `--calls 200` (Track 2).

### Step 4: Evaluate a Candidate Server & Run `eeb compare` (Fitness Function)

Point `eeb serve --policy my_pkg.fast_policy:FastPolicy --uds /tmp/eeb_cand.sock` at your candidate implementation, measure it against the same `runs/golden_bf16.npz`, and run the deterministic `eeb compare` gate:

```bash
# Fast inner-loop canary (< 2 min) for rapid screening:
eeb measure \
  --recipe recipes/track1_vla_pi05_rtx5090.yaml \
  --endpoint unix:///tmp/eeb_cand.sock \
  --traces runs/eval_traces_224 \
  --golden runs/golden_bf16.npz \
  --sweep 1,4,16 --warmup 5 --calls 50 --runs 1 \
  --out runs/cand_canary

# Deterministic fitness scorer:
#   exit 0 -> verdict="ACCEPT" (>= +2.0% gain at >= cutoff_hz, all quality & horizon gates passed)
#   exit 2 -> verdict="REJECT" (structured decision_code + reason in JSON/table output)
eeb compare \
  --baseline runs/00_champion \
  --candidate runs/cand_canary \
  --min-gain-pct 2.0 \
  --out runs/cand_canary/comparison.json
```

Before promoting a candidate to your new champion (`runs/champion`), re-measure with the full 3-run protocol (`--sweep 1,2,4,8,16 --warmup 30 --calls 320 --runs 3 --strict`) and re-verify `eeb compare` returns exit code `0`.

### Step 5: Closed-Loop Delay Check, Bundle Validation & Receipt Verification

```bash
# Track 1 latency-coupled LIBERO-RT delay audit (use eval-wm-ranking for Track 2):
eeb eval-libero-rt \
  --recipe recipes/track1_vla_pi05_rtx5090.yaml \
  --sweep-report runs/champion/report.json \
  --output runs/champion/libero_rt_eval.json

# Pre-PR submission validation and SHA-256 signature verification:
eeb validate-submission --submission runs/champion
eeb verify-receipt --submission runs/champion
```

## 3. Pairing With Open-Source GPU & TPU Agentic Hill-Climbing Frameworks

Because `eeb measure` + `eeb compare --json` provide a hermetic, tamper-evident fitness signal (`exit 0` = `ACCEPT`, `exit 2` = `REJECT`, scalar `metrics.best_output_at_cutoff.candidate`, and Reactivity descriptor `metrics.c1_reactivity_hz.candidate`), you can plug this repository directly into external optimization loops:

* **Cloud TPU Agent Frameworks**:
  1. **[`google-pytorch/tpu-agent-suite`](https://github.com/google-pytorch/tpu-agent-suite) (`TPU Agent Suite` / `tpu-agent`)**: Open-sourced alongside [`google-pytorch/torch_tpu`](https://github.com/google-pytorch/torch_tpu) (`TorchTPU`) at PyTorch Conference 2026; automates Cloud TPU model onboarding, hardware checks, profiling analysis, Pallas/TorchTPU kernel tuning, and diagnostics across Claude Code, Antigravity, Codex, and Cursor. Point it at `eeb.servers.pi05:Pi05Policy`, `Gr00tN17Policy`, or `CosmosPredict25Policy` with `"device": "tpu"` and score iterations with `eeb compare`.
  2. **[`AI-Hypercomputer/accelerator-agents`](https://github.com/AI-Hypercomputer/accelerator-agents)**: Includes `MaxKernel` and `MaxCode` for autonomous Cloud TPU `Pallas` and `TorchTPU` kernel synthesis on single-chip `TPU v6e-1`.
  3. **[`algorithmicsuperintelligence/openevolve`](https://github.com/algorithmicsuperintelligence/openevolve)**: Open-source AlphaEvolve MAP-Elites evolutionary search across TPU and GPU kernels; map `metrics.best_output_at_cutoff.candidate` to fitness and `(c1_reactivity_hz, best_power_eff_at_cutoff)` to MAP-Elites feature dimensions.
* **GPU Agentic Hill-Climbing Frameworks**:
  1. **[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)**: Single-GPU greedy hill-climbing loop; invoke `eeb measure ... && eeb compare --baseline runs/champion --candidate runs/cand` as the evaluation step (promote on `exit 0`, `git restore` on `exit 2`).
  2. **Sakana AI [`AI CUDA Engineer`](https://pub.sakana.ai/ai-cuda-engineer/)**: Agentic CUDA kernel synthesis, hardware profiling, and numerical verification pipeline for attention, MLP, flow-matching, and 3D DiT operators.
  3. **[`ScalingIntelligence/KernelBench`](https://github.com/ScalingIntelligence/KernelBench)**: PyTorch-to-CUDA/Triton kernel generation and verification harness for isolated bottleneck operators before full end-to-end `eeb measure` validation.
* **Interactive Coding Agents**:
  * Claude Code (`CLAUDE.md` and `/embodiedx-optimize`), Gemini CLI (`AGENTS.md` and `skills/`), OpenAI Codex CLI (`AGENTS.md`), Cursor, and OpenHands.
