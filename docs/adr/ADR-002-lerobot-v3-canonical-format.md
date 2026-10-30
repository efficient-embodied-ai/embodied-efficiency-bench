# ADR-002: LeRobot v3 dataset format is canonical; `EmbodiedX-Traces-v1` are derived views

- **Status:** Accepted: 2026-10-08 (approved with `ENGINEERING_PLAN.md` v1.0 §2.1; amend via PR to this file)
- **Deciders:** EEAI @ CoRL 2026 organizers (data lead / WS-C owner: EEAI Benchmark Committee; HF org admin: maintainer)
- **Workstream / milestone:** WS-C · Data (Oct 10–24 2026, feeds M2)
- **Supersedes:** `embodied_bench.golden_dataset` seeded-NumPy tensors (a synthetic placeholder; see `docs/CURRENT_STATE.md`)

## Context

The benchmark needs real, openly licensed robot trajectories with per-step images, proprioception, language instructions, and native timestamps for: (1) latency/throughput replay with realistic temporal coherence and instruction-prefix reuse; (2) golden reference outputs for the trace quality gate; (3) the G4 perturbation (anti-memoization) pairs; (4) LIBERO closed-loop seeds. Target sources: LIBERO (Spatial/Object/Goal/Long), DROID, BridgeData V2, AgiBot-World (with v1.0 public wire bundles built from `lerobot/libero` and `lerobot/droid_1.0.1`). Prior to M2 none of these were read: legacy `golden_dataset.py` fabricated synthetic tensors with similar shapes.

Each source historically ships in a different container (HDF5 for LIBERO, RLDS/TFDS for DROID and Bridge, custom parquet/mp4 for AgiBot). Writing and maintaining four loaders before Nov 12 is not feasible, and the reference policies we must serve (π0.5 via `openpi`/`lerobot`, GR00T N1.7, SmolVLA) already consume the LeRobot format.

## Decision

1. **LeRobot v3 (`LeRobotDataset`, parquet + mp4 + `meta/info.json`) is the canonical on-disk format** for every source. Where an official LeRobot-v3 conversion exists on the Hugging Face Hub, we consume it directly; where it does not, WS-C converts once and publishes the conversion under the `efficient-embodied-ai` HF org with the original license and attribution.
2. **`EmbodiedX-Traces-v1` is a derived view**, produced by `eeb data prepare-traces` (`eeb pack-traces`) from LeRobot v3 episodes at native model resolutions (`224x224` for Track 1 VLA and `256x256` for Track 2 WM; `64x64` `data/embodiedx_traces_v1_mini` is reserved strictly for fast unit/CI smoke tests): ordered per-step records (frames, proprio, instruction, native `timestamp`), a public split and a hidden split, seeded G4 perturbation pairs (`‖δ‖∞ = 1e-3` plus camera/lighting jitter), and organizer-generated golden outputs (`eeb data generate-golden` actions / frames from BF16 reference checkpoints). Traces preserve episode order and the constant instruction prefix so prefix-KV-cache and video-cache effects are measurable.
3. The trace schema and the golden-output procedure are versioned (`-v1`); any change that alters golden outputs bumps the version and invalidates comparisons across versions.
4. Random tensors remain acceptable **only** for harness warm-up calls (never in the timed window or the quality gate).

## Consequences

**Positive**
- One loader (`eeb.data.lerobot_loader`) instead of four; reference servers and the harness share the same decode path.
- Datasets remain where the community already finds them (HF Hub); dataset cards carry licenses (LIBERO MIT, DROID CC-BY-4.0, Bridge V2 CC-BY-4.0, AgiBot-World Apache-2.0; to be re-audited by WS-C).
- Golden outputs are reproducible by anyone with the reference checkpoint and the trace.

**Negative / costs**
- Dependency on `lerobot` (and its `torchcodec`/`av` video stack) in the harness environment; pinned in `pyproject.toml` by WS-C.
- Video-encoded frames must be decoded at replay time; `pack-traces` therefore also materializes decoded `uint8` frames for the public split (~5 GB budget) so the timed window never pays decode cost.
- Hidden-split integrity depends on organizers keeping the hidden parquet private; the public split is sufficient for self-evaluation.

**Risks**
- A source's LeRobot conversion may be incomplete (missing wrist camera, resampled timestamps). Mitigation: WS-C validates per-source field presence and documents deviations in the dataset card.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Native loaders per source (HDF5 / RLDS / custom) | 4× the code, 4× the dependency surface (TensorFlow for RLDS), no shared decode path with the reference policies. |
| Ship only pre-rendered `.npz` tensors (status quo, but real) | Opaque, not inspectable, and loses timestamps/instructions needed for pacing and prefix-reuse studies; also not how participants will test their own policies. |
| WebDataset tar shards | Good streaming story but no ecosystem alignment with the policies under test; would still require a LeRobot bridge for `openpi`/`lerobot`. |

## References

- `ENGINEERING_PLAN.md` §1 (data scope), §2.1, §4 WS-C
- LeRobot dataset format v3 documentation (Hugging Face `lerobot`)
