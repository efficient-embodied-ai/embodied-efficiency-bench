# EEAI @ CoRL 2026: `embodied-efficiency-bench` (EEB) Consolidated Master Proposal (v1.0)

> **Context:** Workshop = *Efficient Foundation Models for Real-Time Embodied AI (EEAI)*, CoRL 2026, **Thu Nov 12, 2026, Austin**.
> **Organizers:** Jiafan Yu, Ouais Alsharif, Changliu Liu, Dhruv Shah, Jiajun Wu, Zhuoyang Zhang.
> **Locked Specification:** See [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) (D1–D9).

## Finalized v1.0 Challenge Setup Summary

1. **Closed Division Only Across Two Canonical Tracks (`D1` & `D2`)**:
   - **Closed Division Focus (`D1`)**: Frozen reference checkpoints and locked execution horizons ($K$ / $H_{\text{frames}}$). Contestants submit optimized inference engines (custom kernels `Pallas`/`Triton`/`CUDA`/`ROCm`, graph compilation `TorchTPU` `torch.compile`/`jax.jit`/TensorRT, calibrated `FP8`/`INT8`/`INT4` quantization, cross-frame visual token caching, pre-caching frozen episode instruction text embeddings on the accelerator during episode `reset()`, and forward-only async scheduling; Host-CPU neural offload inside timed calls and `stream_chunk_frames > 0` are banned). *(Open Division with model retraining/variable $K$ and `Qwen3-Omni-30B-A3B` are deferred to v2.0 per `V2-1` and `V2-2`.)*
   - **Track 1: Real-Time VLA Control (`VLA_TRACK`)**:
     1. **`pi05-3b`** (`Physical-Intelligence/openpi` `pi05_libero` / `lerobot/pi05_libero_finetuned`: 3.3B PaliGemma VLM + 300M Flow-Matching Action Expert, dual `224x224` RGB, 10 ODE steps, locked execution horizon $K = 5$, native chunk $H = 10$, $\Delta t = 50.0\text{ ms}$; native JAX `openpi` and PyTorch `lerobot`/`TorchTPU` are co-equal certified runtimes).
     2. **`gr00t-n1.7-3b`** (`nvidia/GR00T-N1.7-LIBERO` / `nvidia/GR00T-N1.7-3B`: 3B `Cosmos-Reason2-2B` [`Qwen3-VL-2B`] VLM System 2 + 16-step DiT Flow System 1 continuous action head, dual `256x256` RGB, locked execution horizon $K = 8$, native chunk $H = 16$, $\Delta t = 50.0\text{ ms}$).
   - **Track 2: Interactive World Models (`WM_TRACK`)**:
     1. **`cosmos-predict2.5-2b`** (`nvidia/Cosmos-Predict2.5-2B`: 2B Bidirectional 3D-DiT Action-Conditioned Video World Model at locked `256x256` RGB, locked rollout length $H_{\text{frames}} = 16$ frames at 24 FPS, $\Delta t = 41.67\text{ ms}$).

2. **Two Single-Chip Hardware Classes (`num_devices == 1`, `CAP_SINGLE_CHIP`, `D8`)**:
   - **Strict Single-Device Rule (`num_devices == 1`)**: 1 chip / 1 package (`TP=1, PP=1, EP=1`) with **zero artificial memory or MPS capping** and deterministically locked GPU SM clocks (`nvidia-smi -lgc f,f`).
   - **Envelope Bounds**: $\le 1,000\text{ Dense FP32-accum BF16/FP16 TFLOPS}$, $\le 2,000\text{ FP8/INT8 TOPS}$, $\le 64\text{ GB}$ native device VRAM/HBM/LPDDR5X, $\le 2.0\text{ TB/s}$ memory bandwidth:
     1. **Onboard Edge Class ($\le 16\text{ GB}$)**: e.g., `JETSON_ORIN_NX_16GB`, `TPU_V5E_1_16GB`.
     2. **Single-Accelerator Class ($\le 64\text{ GB}$)**:
        - **Google Cloud TPU**: `TPU_V6E_1_32GB` (`1x Cloud TPU v6e-1 Trillium 32GB`)
        - **NVIDIA**: `JETSON_AGX_THOR_64GB` (`64GB`), `RTX_5090_32GB` (`32GB`), `L40S_48GB` (`48GB`), `RTX_4090_24GB` (`24GB`, `82.6 TF` BF16), `L4_24GB` (`24GB`, `121.0 TF` BF16, `300 GB/s`, `72W`)
        - **AMD**: `AMD_MI210_64GB` (`64GB`), `AMD_W7900_48GB` (`48GB`), `AMD_RX_7900_XTX_24GB` (`24GB`)
        - **Custom ASICs**: `CUSTOM_ASIC_SINGLE_DEVICE` ($\le 64\text{ GB}$, $\le 1,000\text{ TFLOPS}$ BF16).

3. **Unified SemiAnalysis InferenceX Axes (`D4`, `D5`, `D9`)**:
   - **Automatic Concurrency Sweep (`D4`)**: The harness sweeps stream concurrency $C \in \{1, 2, 4, 8, 16\}$ (Track 1, $N \ge 320$ chunks/run) or parallel candidate rollout batch $B \in \{1, 2, 4, 8, 16\}$ (Track 2, $N \ge 200$ rollouts/run).
   - **X-Axis (Right is Better $\to$)**: Per-stream Chunk / Rollout Rate: $X_{\text{VLA}}(C) = 1000 / L_{p99}(C)$ (`Chunks/s`), $X_{\text{WM}}(B) = 1000 / L_{p99,\text{WM}}(B)$ (`Rollouts/s | 16x = Stream FPS`).
   - **Y-Axis (Up is Better $\uparrow$)**: Total Chip Output ($Y_{\text{VLA}}(C) = 1000 \cdot C \cdot K / \bar{L}(C)$ [`Executed Actions/sec`], $Y_{\text{WM}}(B) = 1000 \cdot B \cdot H_{\text{frames}} / \bar{L}_{\text{WM}}(B)$ [`Generated Frames/sec`]) across **4 selectable normalizations**: (1) **Per Chip**, (2) **Per Watt (`Actions/Joule` or `Frames/Joule`)**, (3) **Per \$/hr Reference TCO Rate (`(Output/s)/(\$/hr)`)**, and (4) **Iso-Compute (`per 100 Dense BF16 TFLOPS`) & Iso-Bandwidth (`per TB/s`)**.
   - **Official Awards (`D9`)**: Awarded in immutable physical SI units (**Real-Time Performance Champion `Per Chip`** and **Real-Time Energy Efficiency Champion `Per Watt`**) at the real-time cutoff across `Track 1A`, `Track 1B`, and `Track 2`.

4. **Co-Located Same-Run Quality Gates & Latency-Coupled `LIBERO-RT` (`D3`, `D6`, `D7`)**:
   - **Track 1 Gate**: Native wire resolution ($\ge 224\times 224$), action $\text{CosSim} \ge 0.995$ vs. `BF16` reference, BF16-safe G4 anti-memoization check, single-stream $L_{p99}(1) \le 50\text{ ms}$, and closed-loop `LIBERO-RT` task success within $\le 2.0$ percentage points of the `BF16` reference under discrete delay $d = \lceil L / \Delta t \rceil$ (default forward-only async loop / `SEAM` blending, `D6`).
   - **Track 2 Gate**: Rollout latent/pixel $\text{CosSim} \ge 0.990$, $\text{PSNR} \ge 28.0\text{ dB}$ ($\Delta\text{FVD} \le +5\%$) vs. `BF16` reference, BF16-safe G4 perturbation reactivity, inverse-dynamics action controllability, 60s closed-loop rollout stability, and real-time horizon parity $L_{p99,\text{WM}}(1) \le 666.7\text{ ms}$ (`>= 1.5 Rollouts/s` = `24.0 Stream FPS`; $\text{TTFF}_{p99} \equiv L_{\text{clip}}$ co-reported under `D7`).
