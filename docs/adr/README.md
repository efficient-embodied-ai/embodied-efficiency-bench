# Architecture Decision Records (ADRs)

Decisions for `embodied-efficiency-bench` (EmbodiedX), mirroring §2.1 of [`../ENGINEERING_PLAN.md`](../ENGINEERING_PLAN.md). One file per decision; never edit history (supersede with a new ADR and link both ways).

| # | Title | Status | Workstream · Milestone |
|---|---|---|---|
| [ADR-001](ADR-001-protocol-grpc-uds.md) | gRPC over Unix-domain socket as the primary participant protocol; in-process `Policy` ABC for dev; HTTP/JSON shim | Accepted (2026-10-08) | WS-A · M1 |
| [ADR-002](ADR-002-lerobot-v3-canonical-format.md) | LeRobot v3 dataset format is canonical; `EmbodiedX-Traces-v1` are derived views | Accepted (2026-10-08) | WS-C · M2 |
| [ADR-003](ADR-003-closed-loop-libero-mujoco.md) | Closed loop = LIBERO on MuJoCo/robosuite via `LatencyCoupledEnv`; Round 0 uses Tier-2 recorded-latency replay | Accepted (2026-10-08) | WS-D · M3 (Tier-1 in M6) |
| [ADR-004](ADR-004-power-telemetry-live-sampling.md) | Power via live 10 Hz sampling (`pynvml`, `tegrastats`, `amd-smi`); unmeasured Cloud TPU power flagged `UNAVAILABLE` (`null`) | Accepted (2026-10-08) | WS-A · M1 |
| [ADR-005](ADR-005-submission-pr-rerun-certified.md) | Submission = PR into `submissions/`; `CERTIFIED` only after organizer re-run | Accepted (2026-10-08) | WS-D · M3 |
| [ADR-006](ADR-006-static-leaderboard.md) | Leaderboard = static GitHub Pages generated from `leaderboard/*.json` | Accepted (2026-10-08) | WS-E · M4 |

## Conventions

- **Statuses:** `Proposed` → `Accepted` → (`Deprecated` | `Superseded by ADR-NNN`). All six above were accepted together with Engineering Plan v1.0 on 2026-10-08; organizers may amend any of them by PR (update the file, bump the date, note the change under *Status*).
- **Template:** Title · Status · Deciders · Workstream/milestone · Context · Decision (numbered) · Consequences (positive / negative / risks) · Alternatives considered (table) · References.
- **Numbering:** next free number is **ADR-007**. Candidates already identified: receipt body must include `execution_provenance` + image digest (from `docs/CURRENT_STATE.md` §1), `.proto` versioning policy after `v0.1.0` (ADR-001 §4).
- **Placeholder honesty:** an ADR that replaces a placeholder module names it under *Supersedes* and links to [`../CURRENT_STATE.md`](../CURRENT_STATE.md); when the replacement lands, update that inventory in the same PR.
