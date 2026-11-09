# Checkpoint 1 notice: the Open division gate is analytic

**Status: active defect. Applies to every Open division entry merged for Checkpoint 1
(November 10, 2026, 23:59 UTC).**

## What the defect is

The Open division's admission gate is the Stage-2 `LIBERO-RT` quality receipt, specified
as closed-loop success rate within `-5.0 pp` of the frozen reference.

The shipped implementation does not run closed-loop evaluation. `eeb eval-libero-rt` and
`eeb run-stage2` compute a closed-form success-rate decay curve and never step a LIBERO
or MuJoCo environment. The code is self-tagged
`evaluation_mode = "ANALYTIC_LATENCY_COUPLED_MODEL"`, ADR-003 states that "no simulator is
stepped", and `docs/CURRENT_STATE.md` marks `embodied_bench/libero_rt.py` and
`embodied_bench/stage2_closed_loop_verify.py` as analytic placeholders.

Two consequences entrants should know about:

1. A passing Gate B receipt carries no provenance field. `GateBResult` has no
   `evaluation_mode` or `physical_run` member, and the tag that does record the analytic
   path is written only to `eeb eval-libero-rt`'s standalone output, which
   `embodied_bench/attestation.py` excludes from the attested body. A placeholder-derived
   receipt is therefore schema-indistinguishable from a simulator-backed one.
2. ADR-005 specifies a `physical_run: false` preflight rejection. It is not implemented.

This is an organizer defect. It is not a property of any submission.

## What it means for Checkpoint 1

Checkpoint 1 is a live spotlight at the CoRL 2026 EEAI workshop on November 12, 2026. It
is not an award. Official Season 1 awards are decided at Checkpoint 2, December 18, 2026,
ranked strictly by organizer clean-hardware re-runs on a private held-out trace split,
per ADR-005.

Open division entries merged for Checkpoint 1 were admitted on the analytic gate. Their
reported quality deltas come from the closed-form curve rather than from measured task
success. Those entries are correct in what they record: `gate_mode: SANITY_ONLY`,
`quality_certified: false`, and a null `golden_sha256`. Read any Open division Checkpoint
1 number as a throughput, latency and energy result whose task-quality gate was not
measured.

Closed division entries are unaffected. The Closed gate is `GOLDEN_REPLAY` action parity
at `CosSim >= 0.995` with a `CVaR-1%` floor, which is measured against the pinned
reference checkpoint and runs on real hardware.

## What we owe

1. Replace the analytic Stage-2 with a real closed-loop `LIBERO-RT` evaluation, or rename
   the gate so it does not claim closed-loop verification.
2. Publish the Stage-2 protocol: the task suites in scope, the episode and seed counts,
   and the exact seeds. These are currently inconsistent across our own documents.
3. Add a provenance field to the Gate B receipt and carry it into the attested body.
4. Implement the ADR-005 `physical_run: false` preflight rejection.

Items 1 and 2 gate the Checkpoint 2 awards. No Open division award will be issued on an
analytic receipt.
