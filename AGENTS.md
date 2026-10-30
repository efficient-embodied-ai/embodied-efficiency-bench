# Autonomous Agent & External Harness Guide (`AGENTS.md`)

Welcome to **EmbodiedX (`embodied-efficiency-bench` / `eeb` v1.0)**, the official benchmark harness for the CoRL 2026 Workshop: *"Efficient Foundation Models for Real-Time Embodied AI"* (`https://efficient-embodied-ai.github.io/challenge/`).

Every interactive coding agent (Claude Code, Gemini CLI, OpenAI Codex CLI, Cursor, OpenHands) and every external GPU/TPU optimization harness operating in this repository MUST follow the rules below. Read [`skills/embodiedx-agent-kickoff/SKILL.md`](skills/embodiedx-agent-kickoff/SKILL.md) first when starting up a new hardware node or optimization run.

## 1. Benchmark Scope & Immutable Invariants (`D1`, `D2`, `D3`, `D8`)

1. **Season 1 Benchmark Scope (`D1` & `D2`)**:
   * **Track 1A (`VLA_TRACK`, Flow Matching)**:
     - `pi05-3b` (`lerobot/pi05_libero_finetuned_v044` @ `7152f13e34c6c115d808aea0e5d08aca74eac14b` / `Physical-Intelligence/openpi` `pi05_libero`): dual `224x224` RGB, locked `execute_horizon = 5` ($K=5$, official LIBERO chunk $H=10$, $\Delta t = 50\text{ ms}$, 10 Euler ODE steps, gapless real-time SLA cutoff `>= 4.0 Chunks/s` [$L_{p99} \le 250\text{ ms}$]; both native JAX/Flax `openpi` and PyTorch `lerobot`/`TorchTPU` are certified co-equal Closed-Division runtimes).
   * **Track 1B (`VLA_TRACK`, DiT)**:
     - `gr00t-n1.7-3b` (`nvidia/GR00T-N1.7-LIBERO` @ `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`, `Qwen3-VL-2B` VLM + 16-step DiT flow head): dual `256x256` RGB, locked `execute_horizon = 8` ($K=8$, native chunk $H=16$, $\Delta t = 50\text{ ms}$, gapless real-time SLA cutoff `>= 2.5 Chunks/s` [$L_{p99} \le 400\text{ ms}$]).
   * **Track 2 (`WM_TRACK`)**:
     - `cosmos-predict2.5-2b` (`nvidia/Cosmos-Predict2.5-2B` @ `85f8ae7bfe8f5525c8d103429524dcf12f98bf7b`): full-clip bidirectional 3D-DiT at locked `256x256` RGB, rollout `num_frames = 16` ($K_{\text{eval}}=16$, 10 Euler/Flow steps, $\text{CFG}=1.0$, real-time SLA cutoff `>= 1.0 Predictions/s` [$L_{p99} \le 1000\text{ ms}$]; throughput reported strictly in `Predictions/s` and energy efficiency strictly in `Predictions/J`).
   * **Host-CPU Neural Offload Ban & Closed `stream_chunk_frames` Ban**: Weights, architecture, resolution, denoising step count, and execution horizon are strictly frozen. 100% of neural forward-pass operators (`MatMul`, `Attention`, `Conv`, `MLP`, `VAE`, text encoders) inside timed `act()` / `rollout()` calls must run on the single target accelerator (host CPU threads are limited to I/O, tokenization, orchestration, and action queuing; pre-caching frozen episode instruction text embeddings on the accelerator during episode `reset()` is permitted under `D1`, whereas `cpu_text_encoder_offload_cache` and `stream_chunk_frames > 0` / `streaming_first_frame_vae_decode` are forbidden in `CLOSED` division).

2. **Single-Chip Hardware Envelope (`CAP_SINGLE_CHIP`, `D8`)**:
   * Strictly `1` physical accelerator (`num_devices == 1`, `TP=1, PP=1, EP=1`, `<= 64 GB` memory, `<= 1,000` dense FP32-accum BF16 TFLOPS, `<= 2,000` FP8/INT8 TOPS, `<= 2,000 GB/s`), anchored on **`TPU_V6E_1_32GB`** (`Google Cloud TPU v6e-1 32GB`, sole canonical highlighted Cloud TPU target), **`RTX_5090_32GB`** (`209.5 TF` BF16), **`RTX_4090_24GB`** (`82.6 TF` FP32-accum BF16), **`L40S_48GB`**, **`L4_24GB`** (`121.0 TF` BF16, `300 GB/s`, `72W`), **`JETSON_AGX_THOR_64GB`**, and **`JETSON_ORIN_NX_16GB`** (plus compatible single-chip ROCm GPUs and ASICs).
   * Always pin a single device (`CUDA_VISIBLE_DEVICES=0` with deterministic fixed SM clock `nvidia-smi -lgc f,f` sustainable below TDP across $C=1..16$, `HIP_VISIBLE_DEVICES=0`, or on multi-chip Cloud TPU hosts `export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0`) and verify with `eeb envelope check --sku <SKU>`.

3. **Normalization Axes & Tail Accounting (`D4`, `D5`, `D9`)**:
   * **X-axis**: `p99 Chunk Rate (Chunks/s) [1000 / L_p99, ms]` (`hz_react` for `VLA_TRACK`) or `p99 Prediction Rate (Predictions/s) [1000 / L_p99, ms]` (`hz_plan` for `WM_TRACK`, scored from pooled steady-state $p_{99}(C)$ across active slots with a mandatory `<= 15%` per-slot completion fairness fence), swept across ordered canonical $C \in \{1, 2, 4, 8, 16\}$ with per-slot quota $N_{\text{slot}} = \lceil N / C \rceil$, keep-alive tail drain, throughput spread $\le 5\%$, and $p_{99}$ spread $\le 8\%$ ($N \ge 320$ Track 1, $N \ge 200$ Track 2, `--warmup >= 30`).
   * **Y-axis views computed in `report.json`**:
     1. `raw_per_chip_throughput` (steady-state wall-clock `executed_actions_per_sec` or `generated_frames_per_sec`; official `Per Chip` awards split across `Track 1A`, `Track 1B`, and `Track 2`)
     2. `power_efficiency_per_watt` (`actions/J` or `frames/J` from live hardware power telemetry, preferring `nvmlDeviceGetTotalEnergyConsumption` monotonic `mJ` counters; official `Per Watt` award eligibility requires `power_provenance == "MEASURED"`)
     3. `tco_efficiency_per_dollar` (`throughput_per_dollar_hr` in `(Output/s)/($/hr)`, plus auxiliary `units_per_dollar` and `throughput_per_1k_usd` CapEx)
     4. `iso_compute_and_bw_efficiency` (`throughput_per_100_tops` normalized against `100 Dense FP32-accum BF16 TFLOPS`, and `throughput_per_tbps_bw`)

4. **Co-Located Same-Run Quality & Anti-Cheat Gates (`D3`, `D10`, `G1`-`G4`)**:
   * **Immutable Evaluator Boundary**: Treat `eeb/loadgen/`, `eeb/telemetry/`, `eeb/envelope/`, and `embodied_bench/attestation.py` as immutable evaluator infrastructure. Never disable `synchronize_torch_device()`, `torch.cuda.synchronize()`, `torch.tpu.synchronize()`, or `jax.block_until_ready()`.
   * **`G1`, `G3`, `G4` & Blind Inline Canaries (`D10`)**: Always build native-resolution traces (`eeb data prepare-traces --image-hw 224x224` or `256x256`; never submit $64\times 64$ CI mini traces on hardware), generate `golden_bf16.npz` from the unmodified BF16 server (`eeb data generate-golden`), and pass `--golden golden_bf16.npz` to `eeb measure`. LoadGen uses opaque random `UUIDv4` `session_id`s on every call, shuffles evaluation order, and interleaves blind inline BF16-safe perturbation (`10%` RGB pixels $\pm 2\text{ LSB}$ + $10^{-2}$ state/action shift) + golden verification pairs (`~5%` of calls) directly inside the timed sweep.

## 2. Standard Startup & Evaluation Commands

```bash
# 1. Optional Turn-0 study scaffold (creates STUDY_PLAN.json + EXPANDED_AGENT_PROMPT.md + baseline_recipe.yaml)
python3 -m embodied_bench.agent_kickoff --prompt "Start pi0.5 on TPU v6e-1 in CLOSED division"

# 2. Hardware envelope, telemetry & native-resolution eval trace preflight
eeb envelope check --sku TPU_V6E_1_32GB
eeb telemetry probe --seconds 2 --hz 10
eeb data prepare-traces --out runs/eval_traces_224 --image-hw 224x224

# 3. Start policy server over UDS ("device": "tpu" via TorchTPU or Pi05JaxPolicy on Cloud TPU, "device": "cuda" on CUDA/ROCm)
eeb serve --policy eeb.servers.pi05:Pi05Policy --uds /tmp/eeb_policy.sock \
  --policy-kwargs '{"checkpoint_dir": "/path/to/ckpt", "device": "tpu"}' &

# 4. Freeze BF16 golden targets once, then run baseline & candidate sweeps
eeb data generate-golden runs/eval_traces_224 --endpoint unix:///tmp/eeb_policy.sock --out runs/golden_bf16.npz --split public
eeb measure --recipe recipes/track1_vla_pi05_tpu_v6e.yaml --endpoint unix:///tmp/eeb_policy.sock \
  --traces runs/eval_traces_224 --golden runs/golden_bf16.npz --strict --out runs/00_champion

# 5. Deterministic baseline-vs-candidate fitness check (exit 0 = ACCEPT, exit 2 = REJECT)
eeb compare --baseline runs/00_champion --candidate runs/cand_01 --json

# 6. Validate & cryptographically verify bundle before opening a PR
eeb validate-submission --submission runs/cand_01
eeb verify-receipt --submission runs/cand_01
```

## 3. Repository Skill Packages & Compatible External Harnesses

* [`skills/embodiedx-agent-kickoff/SKILL.md`](skills/embodiedx-agent-kickoff/SKILL.md): Complete agent and external harness startup protocol, plus integration guide for Cloud TPU frameworks ([`google-pytorch/tpu-agent-suite`](https://github.com/google-pytorch/tpu-agent-suite), [`AI-Hypercomputer/accelerator-agents`](https://github.com/AI-Hypercomputer/accelerator-agents), [`algorithmicsuperintelligence/openevolve`](https://github.com/algorithmicsuperintelligence/openevolve)) and GPU hill-climbing harnesses ([`karpathy/autoresearch`](https://github.com/karpathy/autoresearch), Sakana AI [`AI CUDA Engineer`](https://pub.sakana.ai/ai-cuda-engineer/), [`ScalingIntelligence/KernelBench`](https://github.com/ScalingIntelligence/KernelBench)).
* [`skills/embodiedx-submission-auditor/SKILL.md`](skills/embodiedx-submission-auditor/SKILL.md): Pre-PR cryptographic receipt, AST anti-cheat, and admissibility verification checklist.
* [`.claude/commands/embodiedx-optimize.md`](.claude/commands/embodiedx-optimize.md): Claude Code `/embodiedx-optimize` startup slash command.
