# ADR-001: gRPC over Unix-domain socket as the primary participant protocol (with in-process `Policy` ABC and HTTP/JSON shim)

- **Status:** Accepted: 2026-10-08 (approved with `ENGINEERING_PLAN.md` v1.0 §2.1; amend via PR to this file)
- **Deciders:** EEAI @ CoRL 2026 organizers (harness lead: maintainer; WS-A owner: EEAI Benchmark Committee)
- **Workstream / milestone:** WS-A · M1 (Harness MVP, Oct 10–17 2026)
- **Supersedes:** the in-process-only `embodied_bench.adapters.ModelAdapter` hierarchy (an analytic placeholder; see `docs/CURRENT_STATE.md`)

## Context

The benchmark must time a participant's model from *outside* the participant's process so that (a) the organizer-controlled LoadGen owns warm-up, pacing, concurrency, and the timed window; (b) non-Python stacks (C++/Rust ASIC runtimes, TensorRT, JAX on TPU, vLLM) can participate on equal footing; and (c) the quality gate and the receipt can attest that latency and outputs came from the same execution. Control loops run at 20–50 Hz, so p99 latency jitter introduced by the transport must stay well below 1 ms.

The early pre-M1 prototype had no external wire protocol: legacy `adapters.py` computed latencies from spec-sheet formulas inside the Python process.

## Decision

1. **Primary transport: gRPC over a Unix-domain socket** (`unix:///tmp/eeb.sock`), service `EmbodiedPolicy` with `Capabilities`, `Predict(PredictRequest) → PredictResponse` (Track 1 VLA), `Rollout(RolloutRequest) → stream Frame` (Track 2 `WM_TRACK`: `Cosmos-Predict2.5-2B` in v1.0, with `Qwen3-Omni` deferred to v2.0 per `D1`/`V2-2`), and `Reset(Session)`. Tensors travel as raw little-endian `float32` / `uint8` bytes plus shape metadata (no per-element protobuf encoding). Timestamps `t_send`, `t_recv`, `t_done` are carried so the harness can separate transport from compute.
2. **Dev mode: in-process Python `Policy` protocol** (`capabilities / reset / predict / rollout`) used by `eeb serve --policy module:Class` and by unit tests; the gRPC server is a thin wrapper around it.
3. **Compatibility shim: HTTP/JSON** endpoint exposing the same three calls for quick integrations; results obtained through the shim are labelled `transport=http` and are **not eligible** for the leaderboard.
4. The `.proto` file is the frozen interface (plan §5); changes after `v0.1.0` require a new ADR.

## Consequences

**Positive**
- ~50–100 µs transport overhead on UDS, no TCP stack jitter; language-neutral; the harness process can be attested independently of the participant container.
- The same client drives in-process, UDS, and HTTP backends, so tests can run on a laptop with a dummy policy.

**Negative / costs**
- Participants must ship a server (we provide reference servers for π0.5, GR00T-N1.7-3B, and Cosmos-Predict2.5-2B in v1.0, with Qwen3-Omni deferred to v2.0 per `D1`/`V2-2`, plus a Python SDK to reduce the burden).
- Image payloads (2× 224×224×3 uint8 ≈ 300 KB per step) are copied across the socket; at C=64 this is ~20 MB/s per 1 Hz (acceptable, with shared-memory tensors as a possible v2.0 optimization).
- Two code paths (gRPC + HTTP shim) to keep in sync; mitigated by generating both from the same `.proto`.

**Risks**
- Windows hosts have limited UDS support → TCP loopback fallback allowed with `transport=tcp` recorded in the report.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| HTTP/JSON only | Base64/JSON encoding of tensors and HTTP framing add milliseconds of p99 jitter at 20–50 Hz; poor fit for streamed frames. |
| In-process Python only | Cannot isolate or attest non-Python stacks; participant code shares the harness's GIL, allocator, and clocks. |
| Shared-memory ring buffer + custom framing | Fastest, but non-portable, hard to secure between containers, and a large engineering cost before Nov 12. Revisit for v1.0. |
| ZeroMQ / MQTT | Extra dependency without the schema/codegen story of gRPC; no streaming semantics as clean as gRPC server-streaming. |

## References

- `ENGINEERING_PLAN.md` §2 (architecture), §2.1 (ADR table), §5 (interfaces to freeze early)
- InferenceX / MLPerf Inference LoadGen client–server separation (prior art)
