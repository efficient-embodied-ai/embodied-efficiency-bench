# ADR-005: Submission = PR into `submissions/`; `CERTIFIED` only after an organizer re-run

- **Status:** Accepted: 2026-10-08 (approved with `ENGINEERING_PLAN.md` v1.0 §2.1; amend via PR to this file)
- **Deciders:** EEAI @ CoRL 2026 organizers (submission/infra lead / WS-D owner: EEAI Benchmark Committee)
- **Workstream / milestone:** WS-D · M3 (Oct 24–31 2026); runners WS-D/WS-E
- **Related:** `.github/ISSUE_TEMPLATE/submission.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `CONTRIBUTING.md`

## Context

A benchmark whose leaderboard accepts self-reported numbers is not credible (risk R7). InferenceX / InferenceMAX established the model: participants contribute a reproducible recipe, organizers run it on organizer-controlled hardware, and only organizer-produced results are published. We also need an audit trail (who submitted what, when, with which container digest) and a place for automated preflight checks to run before a human looks at anything.

## Decision

1. **A submission is a pull request** adding `submissions/<submission_id>/` (or `submissions/<team>/<track>/<model>/<chip>/`, including Two-Phase Commit-Reveal bundles under `submissions/track1_vla/` or `submissions/track2_wm/`) containing: `recipe.yaml` (validated by `RecipeConfig`) or Phase 1 commit metadata (`bundle_summary.json` + `submission_meta.json`), `report.json` (`eeb measure` output), and `receipt.json` (SHA-256 tamper-evident receipt binding `report.json` + host telemetry + quality gates). Standalone measured `report.json` + `receipt.json` hardware bundles are audited directly by `eeb validate-submission --submission <bundle>`. Secrets, weights, and datasets are never committed; images are pulled by digest.
2. **CI preflight on the PR** (public runner, workflow specs under `ci/workflows/ci_preflight.yml` and `ci/workflows/publish_leaderboard.yml`): recipe schema validation (`scripts/validate_recipes.py --recipes-dir recipes --strict`), single-chip envelope check, receipt SHA-256 verification (`eeb verify-receipt`), standalone and recipe-backed submission validation (`eeb validate-submission`), reward-hacking AST scan, and quality-gate parity checks. Failures block merge with a human-readable reason.
3. **Organizer re-run** on self-hosted runners (canonical NVIDIA + TPU hosts; AMD when sponsored), triggered by an organizer label (`rerun:<chip>`). The re-run executes the participant image against the **hidden** trace split and the Tier-2 LIBERO-RT pass, produces a *new* receipt signed with the organizer key, and attaches it to the PR.
4. **State machine:** `SUBMITTED → PREFLIGHT_OK → REPRODUCED → CERTIFIED | REJECTED(reason)`. Every non-success state requires a non-empty `reason` (reusing `RunStatus` / `validate_status_and_reason`). Only `CERTIFIED` entries are written to `leaderboard/*.json`; `results.json` from the participant is never plotted.
5. **Receipts must carry provenance:** the attested body will include `execution_provenance`, the image digest, the trace version, and the runner identity; any receipt with `physical_run: false` is rejected at preflight.
6. Organizers may re-run any certified entry at any time; a reproduction outside ±5 % demotes the entry to `REJECTED(reason="reproduction drift")` with notice to the team.

## Consequences

**Positive**
- No self-reported numbers on the board; full audit trail in git history; reviewers see diffs, not uploads.
- Preflight catches the cheap mistakes (bad YAML, multi-device recipes, broken receipts) before a human or a GPU is involved.

**Negative / costs**
- Requires self-hosted runners and an organizer signing key (M3), plus the queueing/notification plumbing that goes with them.
- Turnaround depends on organizer hardware availability; the plan sets expectations (launch = organizer baselines + open call; first external results cutoff Jan 2027).
- PR-based flow is unfamiliar to some teams; mitigated by the submission issue template and a `eeb submit --dry-run` packer.

**Risks**
- Participant images that cannot run unattended (interactive license prompts, missing weights) → preflight requires a `--self-test` entrypoint that must pass on the public split.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Web form / file upload | No audit trail, no diff review, no automated checks, invites self-reported numbers. |
| Trust participant receipts (no re-run) | Receipts prove integrity of *what the participant ran*, not that the hardware or clocks were as declared. |
| Organizer runs everything from source (no participant image) | Would make organizers responsible for every stack's build; images by digest are reproducible and keep the participant accountable. |

## References

- `ENGINEERING_PLAN.md` §2 (publication path), §2.1, §4 WS-D, §7 R7
- InferenceX / MLPerf Inference submission and review process (prior art)
