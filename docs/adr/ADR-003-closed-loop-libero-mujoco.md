# ADR-003: Closed loop = LIBERO on MuJoCo/robosuite wrapped by `LatencyCoupledEnv`; Round 0 uses Tier-2 recorded-latency replay

- **Status:** Accepted: 2026-10-08 (approved with `ENGINEERING_PLAN.md` v1.0 §2.1; amend via PR to this file)
- **Deciders:** EEAI @ CoRL 2026 organizers (gates lead / WS-D owner: EEAI Benchmark Committee)
- **Workstream / milestone:** WS-D · M3 (Quality gates + submission pipeline, Oct 24–31 2026); Tier-1 in M6 (v1.0)
- **Supersedes:** `embodied_bench.libero_rt` and `embodied_bench.stage2_closed_loop_verify` success-rate *models* (analytic placeholders; see `docs/CURRENT_STATE.md`)

## Context

Open-loop trace replay measures speed but cannot show whether a faster, compressed policy still completes tasks when its latency is fed back into the control loop. The benchmark's distinguishing claim ("LIBERO-RT") is that inference latency `L` is coupled into the simulator as a discrete delay `d = ceil(L / Δt)` at `Δt = 50 ms` (20 Hz) under three execution strategies (`SYNC_HOLD`, `ASYNC_CHUNK`, `RTC`). Today that coupling exists only as arithmetic applied to a closed-form success-rate curve; no simulator is stepped.

Constraints: 35 days to launch; evaluations must be deterministic and cheap enough to re-run for every certified submission; GPU-in-the-loop simulation competes for the same accelerator the policy is being timed on.

## Decision

1. **Simulator:** LIBERO task suites (Spatial / Object / Goal; Long optional) on **robosuite + MuJoCo**, which is what the canonical v1.0 reference checkpoints (`openpi` π0.5-LIBERO and GR00T N1.7 LIBERO, as well as prior literature OpenVLA-OFT) were fine-tuned and reported on.
2. **Coupling wrapper:** `LatencyCoupledEnv(env, dt=0.05, strategy, chunk_size)` delays action application by `d = ceil(L / dt)` steps per policy call, implements `SYNC_HOLD` (hold last action), `ASYNC_CHUNK` (consume buffered chunk, stall if exhausted), and `RTC` (prefix-conditioned re-planning), enforces the 1.0 s chunk-horizon cap and the 1.5× p90 episode wall-time budget, and injects mid-episode reactivity probes (object displacement, goal switch, emergency stop).
3. **Round 0 / launch tier = Tier-2 "recorded-latency replay":** the policy's per-call latency schedule is *recorded* during the timed `eeb measure` run on the target chip and *replayed* into a CPU-only simulator pass where the policy runs un-timed (any hardware). `N = 50` episodes × 3 seeds per suite. This decouples the sim from the accelerator and makes re-runs reproducible.
4. **Tier-1 "GPU-in-the-loop"** (policy and simulator co-resident, live latency) is deferred to v1.0 (M6) and will be reported as a separate column, never silently mixed with Tier-2 numbers.
5. Results report `latency_free_success`, `realtime_success`, the drop in points, per-suite breakdown, reactivity-probe outcomes, and `tier`. The existing `ceil(L/Δt)` + scheduling arithmetic in `libero_rt.py` is reused inside the wrapper; its success-rate *model* is deleted.

## Consequences

**Positive**
- Deterministic, auditable, and cheap to re-run; organizers can certify submissions without owning every participant's chip for the sim pass.
- Matches the task suites the reference models were trained on, so latency-free baselines are meaningful.

**Negative / costs**
- Tier-2 replay ignores second-order effects (latency that depends on the observation distribution reached under delay). This is stated in the rules; Tier-1 closes the gap in v1.0.
- robosuite/MuJoCo + LIBERO assets add ~2 GB and a non-trivial install; shipped in the `x86-cuda` image and a CPU `sim` image.
- 50 × 3 × 3 suites ≈ 450 episodes per submission ≈ 1–3 h CPU; acceptable for certification, too slow for inner-loop development (participants may use `--episodes 10`).

**Risks**
- LIBERO flakiness / version drift (R5). Mitigation: pin LIBERO + robosuite + MuJoCo versions, cache assets, cap N, publish the exact seeds.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Tier-1 GPU-in-the-loop for every submission before Nov 12 | Simulator competes with the policy for the accelerator, non-deterministic, and the organizer re-run would need each participant's hardware. Deferred to v1.0. |
| Keep the closed-form success-rate model | Produces numbers nobody measured; cannot be part of a credible leaderboard. |
| Isaac Lab / ManiSkill as the closed-loop simulator | No matching public checkpoints for the canonical VLA models on those suites; higher GPU requirements for the organizer re-run. |
| Real-robot HIL only | Not reproducible, not scalable to external submissions; kept as an optional Tier-3 consistency anchor. |

## References

- `ENGINEERING_PLAN.md` §2.1, §4 WS-D, §7 R5
- LIBERO benchmark (Liu et al., 2023); robosuite; Real-Time Chunking (Black et al., 2025)
