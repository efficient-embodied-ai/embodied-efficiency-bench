# ADR-006: Leaderboard = static GitHub Pages generated from `leaderboard/*.json`

- **Status:** Accepted: 2026-10-08; updated 2026-10-30 for v1.0 Season 1 public launch
- **Deciders:** EEAI @ CoRL 2026 organizers (site/infra lead / WS-E owner: EEAI Benchmark Committee)
- **Workstream / milestone:** WS-E · M4 (Nov 1–8 2026)
- **Related:** workshop challenge website explorer (`efficient-embodied-ai.github.io/challenge/index.html`, linked from `#challenge`, populated from `leaderboard/embodiedx_leaderboard_data.json`)

## Context

The leaderboard must show Pareto frontiers (p99 latency vs. throughput) with unified Primary throughput (`Executed Action Steps/s` / `Generated Env-State Frames/s`) plus three Secondary efficiency normalizations (`per-Watt`, `per-100-BF16-TFLOPS`, `per-$/hr`) for `C ∈ {1, 2, 4, 8, 16}`, per track and per chip, and must be auditable: every plotted point links directly to a cryptographically attested receipt. There is no budget or operator capacity for a hosted backend, and the data volume (tens to hundreds of verified points) is compact.

## Decision

1. **Source of truth:** `leaderboard/leaderboard_v1.json`, `leaderboard/LEADERBOARD.md`, and `leaderboard/embodiedx_leaderboard_data.json` in this repository, regenerated strictly via `eeb build-leaderboard --submissions-dir submissions --strict-verify` and `eeb export-website` from verified receipts (ADR-005). Each entry carries `submission_id`, `receipt_hash`, `execution_provenance`, `power_provenance`, `power_source`, quality-gate outcomes, and the full concurrency curve.
2. **Rendering:** static GitHub Pages (`efficient-embodied-ai.github.io/challenge/index.html`) powered strictly by verified curve payloads exported to `leaderboard/embodiedx_leaderboard_data.json`; no server-side database is required.
3. **Regeneration:** `ci/workflows/publish_leaderboard.yml` validates recipes and rebuilds `leaderboard/leaderboard_v1.json`, `leaderboard/LEADERBOARD.md`, and `leaderboard/embodiedx_leaderboard_data.json` on changes touching `submissions/` or `leaderboard/`.
4. **Honesty rules on the page:** entries with `power_provenance == "UNAVAILABLE"` (Cloud TPU VMs without physical guest rail sensors, ADR-004) omit fabricated watts/Joules cleanly; non-qualifying runs (`status == "INVALID"`) stay excluded from `leaderboard/leaderboard_v1.json`; no point is plotted without a verified receipt SHA-256. At Day-1 v1.0 launch, `submissions/` opens empty (`num_verified_submissions == 0`) with zero synthetic seed rows on the live board, inviting Founding Season community and organizer submissions; schema unit-test fixtures (`seed_leaderboard.json`, `sample_submission_receipt.json`) reside solely under `tests/fixtures/`.

## Consequences

**Positive**
- Zero backend to operate or secure; the leaderboard is a git log; anyone can fork and re-render.
- Auditable: JSON diff per certification; receipts linkable by hash.

**Negative / costs**
- No per-user interactivity beyond client-side filtering; no live queueing status (acceptable).
- Large JSON (if traces or per-point telemetry were embedded) would slow page load → the site JSON contains summaries only; telemetry CSVs stay in `submissions/`.
- GitHub Pages caching can lag a few minutes after regeneration.

**Risks**
- Website and repo diverge if the explorer is edited by hand → the explorer's data path is the generated JSON only; hand-edits are rejected in review (CODEOWNERS on `leaderboard/`).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Hosted dashboard (Streamlit / Grafana / custom API) | Operational burden, auth/secrets, and a single point of failure for a volunteer-run workshop. |
| Hugging Face Space leaderboard | Viable, but splits the audit trail from the repo and duplicates the existing explorer; may be added later as a mirror. |
| Manually curated table in README | Not scalable, no normalization toggles, error-prone. |

## References

- `ENGINEERING_PLAN.md` §2 (publication), §2.1, §4 WS-E
- InferenceMAX / InferenceX static dashboards (prior art)
