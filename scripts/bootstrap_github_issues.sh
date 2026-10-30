#!/usr/bin/env bash
# bootstrap_github_issues.sh -- one-shot project bootstrap for embodied-efficiency-bench (EmbodiedX).
#
# Creates, on a GitHub repository you own:
#   1. Labels   : track:vla, track:wm, ws:A..ws:E, M0..M6, hardware:{nvidia,amd,tpu,asic}, blocked
#                 (+ bug, submission, model-request, hardware-request used by .github/ISSUE_TEMPLATE/*)
#   2. Milestones: M0..M6 with the due dates from docs/ENGINEERING_PLAN.md §4 (2026, Pacific time)
#   3. Issues   : one per checklist item in docs/ENGINEERING_PLAN.md §4 (title, body with Definition of Done,
#                 labels, milestone). 32 issues total.
#
# SAFETY MODEL
#   * Default mode is --dry-run: nothing is executed; the exact `gh` commands are printed (shell-quoted,
#     copy-pasteable). `gh` does not even need to be installed for a dry run.
#   * Writes happen only with --apply AND an explicit --repo owner/name. --apply refuses to run without it.
#   * --apply is idempotent: labels are created with --force (update in place), milestones and issues are
#     skipped when an item with the same title already exists (any state), so re-running never duplicates.
#
# USAGE
#   scripts/bootstrap_github_issues.sh --dry-run                       # print plan against the default repo name
#   scripts/bootstrap_github_issues.sh --dry-run --repo myorg/myrepo   # print plan against a specific repo
#   scripts/bootstrap_github_issues.sh --apply   --repo myorg/myrepo   # create labels + milestones + issues
#   scripts/bootstrap_github_issues.sh --apply   --repo myorg/myrepo --skip-issues   # labels + milestones only
#
# REQUIREMENTS (apply mode only): gh >= 2.40 authenticated with `repo` scope (`gh auth login`), write access.
#
# Exit codes: 0 ok · 2 usage error · 3 gh missing / unauthenticated / repo not accessible.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DEFAULT_DRY_RUN_REPO="efficient-embodied-ai/embodied-efficiency-bench"   # proposed name, plan §8 item 3
PLAN_URL_FRAGMENT="docs/ENGINEERING_PLAN.md"

MODE="dry-run"
REPO=""
DO_LABELS=1
DO_MILESTONES=1
DO_ISSUES=1

usage() {
  sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "ERROR: $*" >&2
  exit "${2:-2}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run" ;;
    --apply) MODE="apply" ;;
    --repo)
      [[ $# -ge 2 ]] || die "--repo requires a value (owner/name)"
      REPO="$2"; shift ;;
    --repo=*) REPO="${1#--repo=}" ;;
    --skip-labels) DO_LABELS=0 ;;
    --skip-milestones) DO_MILESTONES=0 ;;
    --skip-issues) DO_ISSUES=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$MODE" == "apply" ]]; then
  [[ -n "$REPO" ]] || die "--apply requires an explicit --repo owner/name (refusing to guess a target repository)."
  [[ "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "--repo must look like owner/name, got: '$REPO'"
  command -v gh >/dev/null 2>&1 || die "gh CLI not found; install from https://cli.github.com/ and run 'gh auth login'." 3
  gh auth status >/dev/null 2>&1 || die "gh is not authenticated; run: gh auth login" 3
  gh repo view "$REPO" --json nameWithOwner >/dev/null 2>&1 || die "cannot access repository '$REPO' with the current gh account." 3
else
  if [[ -n "$REPO" && ! "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    die "--repo must look like owner/name, got: '$REPO'"
  fi
  REPO="${REPO:-$DEFAULT_DRY_RUN_REPO}"
fi

echo "# $SCRIPT_NAME -- mode=$MODE repo=$REPO (labels=$DO_LABELS milestones=$DO_MILESTONES issues=$DO_ISSUES)"
if [[ "$MODE" == "dry-run" ]]; then
  echo "# DRY RUN: nothing will be created. Re-run with '--apply --repo owner/name' to execute the commands below."
fi

# Print a shell-quoted command line (copy-pasteable) and, in apply mode, execute it.
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$MODE" == "apply" ]]; then
    "$@"
  fi
}

# ---------------------------------------------------------------------------------------------------------------
# 1. Labels
# ---------------------------------------------------------------------------------------------------------------
N_LABELS=0
label() { # name color description
  N_LABELS=$((N_LABELS + 1))
  run gh label create "$1" --repo "$REPO" --color "$2" --description "$3" --force
}

if [[ "$DO_LABELS" -eq 1 ]]; then
  echo
  echo "## Labels"
  label "track:vla"         "1d76db" "Track 1 -- real-time VLA policies (GR00T-N1.7-3B, pi0.5-3.3B, ...)"
  label "track:wm"          "5319e7" "Track 2 -- world / video models (Cosmos-Predict2.5-2B, ...)"
  label "ws:A"              "0e8a16" "Workstream A -- harness: protocol, LoadGen, telemetry, envelope (M1)"
  label "ws:B"              "2cbe4e" "Workstream B -- reference servers + TPU path + Docker images (M2)"
  label "ws:C"              "5bd36b" "Workstream C -- data: LeRobot v3 loader, EmbodiedX-Traces-v1, golden outputs"
  label "ws:D"              "8ee89a" "Workstream D -- quality gates, LIBERO-RT, submission pipeline, runners (M3)"
  label "ws:E"              "b9f0c1" "Workstream E -- organizer baselines, leaderboard site, docs (M4)"
  label "M0"                "fbca04" "Milestone M0 -- Bootstrap (Oct 8–10, 2026)"
  label "M1"                "f9d71c" "Milestone M1 -- Harness MVP (Oct 10–17, 2026)"
  label "M2"                "f7e27a" "Milestone M2 -- 4 reference servers + TPU path (Oct 17–24, 2026)"
  label "M3"                "f5ec9e" "Milestone M3 -- Gates + submission pipeline (Oct 24–31, 2026)"
  label "M4"                "f3f2b8" "Milestone M4 -- Organizer baselines + leaderboard site (Nov 1–8, 2026)"
  label "M5"                "e4e669" "Milestone M5 -- Launch at CoRL 2026 (Nov 9–12, 2026)"
  label "M6"                "d4c5f9" "Milestone M6 -- v1.0 living benchmark (Nov 13, 2026 → Jan 2027)"
  label "hardware:nvidia"   "76b900" "Needs / concerns NVIDIA GPUs or Jetson"
  label "hardware:amd"      "ed1c24" "Needs / concerns AMD Instinct / Radeon Pro (ROCm)"
  label "hardware:tpu"      "4285f4" "Needs / concerns Google Cloud TPU"
  label "hardware:asic"     "6a737d" "Needs / concerns other accelerators / custom ASICs"
  label "blocked"           "b60205" "Blocked on an external dependency, decision, or hardware"
  label "bug"               "d73a4a" "Something is broken (see .github/ISSUE_TEMPLATE/bug_report.md)"
  label "submission"        "0052cc" "Benchmark submission intake (see .github/ISSUE_TEMPLATE/submission.md)"
  label "model-request"     "bfd4f2" "Request to add a model to the registry"
  label "hardware-request"  "c5def5" "Request to add a hardware SKU / vendor path"
fi

# ---------------------------------------------------------------------------------------------------------------
# 2. Milestones (due dates from ENGINEERING_PLAN.md §4; midnight America/Los_Angeles expressed in UTC:
#    PDT = UTC-7 until Nov 1, 2026; PST = UTC-8 afterwards)
# ---------------------------------------------------------------------------------------------------------------
N_MILESTONES=0
EXISTING_MILESTONES=""
if [[ "$MODE" == "apply" && "$DO_MILESTONES" -eq 1 ]]; then
  EXISTING_MILESTONES="$(gh api "repos/$REPO/milestones?state=all&per_page=100" --paginate --jq '.[].title' 2>/dev/null || true)"
fi

milestone() { # title due_on description
  N_MILESTONES=$((N_MILESTONES + 1))
  if [[ "$MODE" == "apply" ]] && grep -Fxq -- "$1" <<<"$EXISTING_MILESTONES"; then
    echo "# skip (exists): milestone '$1'"
    return 0
  fi
  run gh api --method POST "repos/$REPO/milestones" -f "title=$1" -f "due_on=$2" -f "description=$3"
}

if [[ "$DO_MILESTONES" -eq 1 ]]; then
  echo
  echo "## Milestones"
  milestone "M0" "2026-10-10T07:00:00Z" "Bootstrap (Oct 8–10). DoD: repo public-ready, CI green, issues filed, owners assigned. Plan §4."
  milestone "M1" "2026-10-17T07:00:00Z" "WS-A Harness MVP (Oct 10–17). DoD: real p99 latency + Joules for pi0.5 at C=1..16 on one NVIDIA chip, reproducible ±5%."
  milestone "M2" "2026-10-24T07:00:00Z" "WS-B reference servers + TPU path; WS-C data lands (Oct 17–24). DoD: all 3 canonical v1.0 Closed Division models (pi05-3b, gr00t-n1.7-3b, cosmos-predict2.5-2b) produce real reports on ≥1 chip; pi0.5 on NVIDIA and TPU; eeb data download + measure runs offline."
  milestone "M3" "2026-10-31T07:00:00Z" "WS-D quality gates + submission pipeline (Oct 24–31). DoD: a dummy external submission goes PR → preflight → re-run → leaderboard automatically."
  milestone "M4" "2026-11-08T08:00:00Z" "WS-E organizer baselines + leaderboard (Nov 1–8). DoD: public leaderboard shows only measured, receipt-backed points."
  milestone "M5" "2026-11-12T08:00:00Z" "Launch at CoRL 2026, Austin (Nov 9–12). DoD: rules frozen, harness tagged v0.1.0, ≥6 organizer baseline curves live."
  milestone "M6" "2027-01-31T08:00:00Z" "v1.0 living benchmark (Nov 13, 2026 → Jan 2027). AMD runner, Tier-1 LIBERO-RT, FVD + S2 evaluator, P3 edge-cloud, registry round 2, first results cutoff + awards."
fi

# ---------------------------------------------------------------------------------------------------------------
# 3. Issues -- one per checklist item in ENGINEERING_PLAN.md §4
# ---------------------------------------------------------------------------------------------------------------
N_ISSUES=0
EXISTING_ISSUES=""
if [[ "$MODE" == "apply" && "$DO_ISSUES" -eq 1 ]]; then
  EXISTING_ISSUES="$(gh issue list --repo "$REPO" --state all --limit 500 --json title --jq '.[].title' 2>/dev/null || true)"
fi

issue() { # milestone labels title body
  N_ISSUES=$((N_ISSUES + 1))
  if [[ "$MODE" == "apply" ]] && grep -Fxq -- "$3" <<<"$EXISTING_ISSUES"; then
    echo "# skip (exists): issue '$3'"
    return 0
  fi
  run gh issue create --repo "$REPO" --title "$3" --label "$2" --milestone "$1" --body "$4"
}

if [[ "$DO_ISSUES" -eq 1 ]]; then
  echo
  echo "## Issues"

  # ---- M0 -- Bootstrap (Oct 8–10) · Owner: maintainer -------------------------------------------------------------
  issue "M0" "M0" "M0: Create GitHub repo under the workshop org (Apache-2.0, branch protection, CODEOWNERS)" "$(cat <<'EOF'
## Context
Plan §4 M0, item 1. Repository name proposed in plan §8: `github.com/efficient-embodied-ai/embodied-efficiency-bench` (Apache-2.0).

## Definition of Done
- [ ] Repository exists under the workshop org, public (or private until M5 with a dated flip plan).
- [ ] `LICENSE` (Apache-2.0 full text), `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.github/CODEOWNERS` present and `@efficient-embodied-ai/organizers` team exists.
- [ ] Branch protection on `main`: PR required, CI status checks required, no force-push.
- [ ] Issue templates and PR template render correctly.

## Links
- docs/ENGINEERING_PLAN.md §4 M0, §8
EOF
)"

  issue "M0" "M0" "M0: Commit current code with explicit ANALYTIC PLACEHOLDER banners (adapters.py, golden_dataset.py, libero_rt.py)" "$(cat <<'EOF'
## Context
Plan §4 M0, item 2 and §0. Nothing in `embodied_bench/` runs a model, loads a real episode, or steps a simulator; every report must say so.

## Definition of Done
- [ ] Module banners + `EXECUTION_PROVENANCE` constants on `adapters.py`, `golden_dataset.py`, `libero_rt.py`, `stage2_closed_loop_verify.py`, `telemetry.py`.
- [ ] Every Stage-1 / `eeb measure` report, CLI summary, and leaderboard entry carries `execution_provenance{adapters,dataset,closed_loop,telemetry,physical_run=false}`.
- [ ] `docs/CURRENT_STATE.md` inventory matches the code (file/line refs).
- [ ] A test fails if the provenance block is removed.

## Links
- docs/CURRENT_STATE.md · docs/ENGINEERING_PLAN.md §0
EOF
)"

  issue "M0" "M0" "M0: CI -- ruff + pytest + recipe schema validation on every PR" "$(cat <<'EOF'
## Context
Plan §4 M0, item 3.

## Definition of Done
- [ ] `.github/workflows/ci_preflight.yml` runs `ruff check .`, `pytest -q`, and `python3 scripts/validate_recipes.py` on `push` to `main` and on every `pull_request`.
- [ ] Matrix covers the minimum supported Python (3.10/3.11) and current (3.13).
- [ ] CI is green on `main`; the checks are marked *required* in branch protection (see repo-creation issue).

## Links
- .github/workflows/ci_preflight.yml · scripts/validate_recipes.py
EOF
)"

  issue "M0" "M0" "M0: Land ENGINEERING_PLAN + ADR-001..006; open milestones/issues via scripts/bootstrap_github_issues.sh" "$(cat <<'EOF'
## Context
Plan §4 M0, item 4.

## Definition of Done
- [ ] `docs/ENGINEERING_PLAN.md`, `docs/CURRENT_STATE.md`, `docs/adr/ADR-001..006` + `docs/adr/README.md` merged.
- [ ] `scripts/bootstrap_github_issues.sh --apply --repo <org>/<repo>` executed once; labels, milestones M0–M6, and all §4 issues exist.
- [ ] Owners assigned on every WS-A..WS-E issue (plan §8 item 4).

## Links
- scripts/README.md · docs/adr/README.md
EOF
)"

  issue "M0" "M0,hardware:nvidia,hardware:amd,hardware:tpu" "M0: Kick off hardware procurement (plan §6)" "$(cat <<'EOF'
## Context
Plan §4 M0, item 5 and §6. Needed: 1× NVIDIA dev/CI box (RTX 5090 32 GB or L40S 48 GB, cloud ≈ $1/hr), 1× Cloud TPU v6e-1 (GCP project + quota), 1× AMD chip (MI210 64 GB / W7900 48 GB via AMD / TensorWave sponsorship), 1× Jetson AGX Thor dev kit (NVIDIA sponsorship), HF org `efficient-embodied-ai`, GitHub org + self-hosted runner tokens.

## Definition of Done
- [ ] Each row of plan §6 has an owner and a status (☐ → requested → secured) tracked here.
- [ ] NVIDIA box and TPU v6e-1 reachable by WS-A/WS-B by **Oct 14** (M1 needs the NVIDIA box).
- [ ] AMD status decided by **Oct 24** (R6: launch with NVIDIA + TPU measured, AMD "coming soon" if not secured).

## Links
- docs/ENGINEERING_PLAN.md §6, §7 (R6)
EOF
)"

  # ---- WS-A · M1 -- Harness MVP (Oct 10–17) --------------------------------------------------------------------
  issue "M1" "M1,ws:A,track:vla,track:wm" "WS-A: eeb.proto -- Capabilities, Predict(obs)→actions[H×D] (T1), Rollout(ctx, actions)→frames (T2), Reset" "$(cat <<'EOF'
## Context
Plan §4 WS-A item 1, §5 interfaces to freeze early, ADR-001.

## Definition of Done
- [ ] `eeb/protocol/eeb.proto` with `EmbodiedPolicy` service (`Capabilities`, `Predict`, `Rollout` server-streaming, `Reset`), raw little-endian float32 tensor payloads + shape metadata, `t_send/t_recv/t_done` timestamps.
- [ ] Generated Python stubs checked in; codegen command documented; stubs excluded from `ruff`.
- [ ] Round-trip unit test: encode → decode equals input for images, proprio, actions, frames.
- [ ] `.proto` marked frozen for `v0.1.0` (changes need ADR-007+).

## Links
- docs/adr/ADR-001-protocol-grpc-uds.md
EOF
)"

  issue "M1" "M1,ws:A" "WS-A: In-process Policy ABC + gRPC/UDS server skeleton + client" "$(cat <<'EOF'
## Context
Plan §4 WS-A item 2, ADR-001.

## Definition of Done
- [ ] `eeb.protocol.Policy` protocol (`capabilities / reset / predict / rollout`) with a dummy policy for tests.
- [ ] `eeb serve --policy module:Class --uds /tmp/eeb.sock` wraps any `Policy` in a gRPC/UDS server; `eeb.loadgen.Client` talks to `unix://` and in-process backends through one interface.
- [ ] HTTP/JSON shim present and labelled `transport=http` (not leaderboard-eligible).
- [ ] Test: client ↔ UDS server round trip under 1 ms p99 transport overhead for a 2×224×224×3 observation on the CI runner (informational threshold, logged).

## Links
- docs/adr/ADR-001-protocol-grpc-uds.md
EOF
)"

  issue "M1" "M1,ws:A,track:vla" "WS-A: LoadGen -- concurrency sweep, session-sticky routing, pacing, 20-call warm-up, 1,000-call window, 3-run <5% variance, p50/p99, TTFA" "$(cat <<'EOF'
## Context
Plan §4 WS-A item 3 and §1 metrics. Replaces `embodied_bench.telemetry.run_eeb_measurement_protocol`, which fabricates its 3 runs.

## Definition of Done
- [ ] `C ∈ {1,2,4,8,16,32,64}` sweep; each session replays one episode in order (session-sticky); think-time = chunk execution time at 20 Hz.
- [ ] Harness-owned 20-call warm-up on non-test observations, then a 1,000-call timed window; `time_to_ready_s` reported separately.
- [ ] 3 runs; report median; fail the point if cross-run spread ≥ 5 %.
- [ ] Metrics per point: p50/p99 obs→action latency, TTFA, actions/s (T1) or frames/s (T2), exact tensors from the timed window kept for the gate (G3).
- [ ] Report carries `execution_provenance.adapters = "PHYSICAL_RUN"` only when a real server answered every timed call.

## Links
- docs/ENGINEERING_PLAN.md §1, §4 WS-A · docs/CURRENT_STATE.md
EOF
)"

  issue "M1" "M1,ws:A,hardware:nvidia,hardware:amd,hardware:tpu" "WS-A: Telemetry sidecar -- 10 Hz pynvml / tegrastats / amd-smi; Joules/action; peak memory; tpu-info best-effort" "$(cat <<'EOF'
## Context
Plan §4 WS-A item 4, ADR-004. Replaces the TDP-table power model in `embodied_bench.telemetry`.

## Definition of Done
- [ ] Separate sidecar process samples at 10 Hz for the whole timed window (+5 s idle baseline) → `telemetry.csv` (`t, power_w, mem_used_mb, temp_c, sm_clock_mhz, source`).
- [ ] Backends: `pynvml`, `tegrastats`, `amd-smi`; `tpu-info` best-effort with `power_source=estimated` when no per-chip power.
- [ ] Energy per action/frame integrated over the window (not P×t of one latency); `mean_power_w`, `idle_power_w`, `peak_mem_mb` reported.
- [ ] Parsers from `embodied_bench.telemetry` reused with their tests; new tests for the integrator and the `estimated` fallback.

## Links
- docs/adr/ADR-004-power-telemetry-live-sampling.md
EOF
)"

  issue "M1" "M1,ws:A" "WS-A: Envelope check wired to live device query (num_devices==1, memory, spec TOPS)" "$(cat <<'EOF'
## Context
Plan §4 WS-A item 5 and §1 hardware envelope (`≤ 64 GB`, `≤ 1,000 BF16 TFLOPS` / `≤ 2,000 FP8/INT8 TOPS`, `≤ 2.0 TB/s`, `N=1`).

## Definition of Done
- [ ] `eeb/envelope/hardware_specs.yaml` (moved from `HARDWARE_SKU_SPECS`) + a live query that confirms exactly one visible device, its total memory, and matches it to a SKU row (or `CUSTOM_ASIC` with declared specs).
- [ ] Mismatch between recipe `hardware_sku` and the live device fails the run with a reason.
- [ ] Peak memory from the telemetry sidecar is checked against the 64 GB cap at the end of the run.
- [ ] Tests with a fake device enumerator for pass / multi-device / over-cap cases.

## Links
- docs/ENGINEERING_PLAN.md §1
EOF
)"

  issue "M1" "M1,ws:A,track:vla,hardware:nvidia" "WS-A: Reference server #1 -- π0.5 via openpi (PyTorch) on 1× RTX/L40S → first real eeb measure report" "$(cat <<'EOF'
## Context
Plan §4 WS-A item 6 and the M1 DoD. This is the first physical number in the project.

## Definition of Done
- [ ] `eeb/servers/pi05/` serves `lerobot/pi05_libero_finetuned_v044` (openpi, PyTorch) over gRPC/UDS.
- [ ] `eeb measure --recipe recipes/track1_vla_pi05_edge.yaml --endpoint unix:///tmp/eeb.sock --sweep 1,2,4,8,16` completes on one NVIDIA chip with real p99 latency and Joules per action; three runs within ±5 %.
- [ ] Report has `execution_provenance.physical_run == true` and the receipt verifies.
- [ ] Numbers posted in this issue with chip, driver, clocks, and container digest -- **not** on the leaderboard until M4.

## Links
- docs/ENGINEERING_PLAN.md §4 WS-A (M1 DoD)
EOF
)"

  # ---- WS-C -- Data: EmbodiedX-Traces-v1 (Oct 10–24, parallel; lands with M2) --------------------------------
  issue "M2" "M2,ws:C" "WS-C: eeb.data.lerobot_loader -- LIBERO (Spatial/Object/Goal/Long), DROID, Bridge V2, AgiBot-World from HF LeRobot v3" "$(cat <<'EOF'
## Context
Plan §4 WS-C item 1, ADR-002. Replaces the seeded-NumPy `embodied_bench.golden_dataset`.

## Definition of Done
- [ ] One loader over `LeRobotDataset` yielding ordered per-step records (frames per camera, proprio, instruction, native timestamp, action) for all four sources.
- [ ] Per-source field-presence validation with documented deviations (e.g. missing wrist camera).
- [ ] Offline cache under `data_cache/` (gitignored); `eeb data download --trace vla_libero_public` works on a laptop.
- [ ] Tests on a tiny fixture dataset (no network in CI).

## Links
- docs/adr/ADR-002-lerobot-v3-canonical-format.md
EOF
)"

  issue "M2" "M2,ws:C" "WS-C: eeb pack-traces -- per-step frames, proprio, instruction, native timestamps → parquet (+ mp4); preserves temporal coherence and instruction prefix" "$(cat <<'EOF'
## Context
Plan §4 WS-C item 2, ADR-002.

## Definition of Done
- [ ] `eeb pack-traces --source libero --split public --out traces/` writes parquet (+ mp4 and decoded uint8 frames for the public split) with episode order and constant instruction prefix preserved.
- [ ] Trace schema versioned `EmbodiedX-Traces-v1`; a manifest with SHA-256 per shard.
- [ ] Size budget for the public split documented (~5 GB total across sources).

## Links
- docs/adr/ADR-002-lerobot-v3-canonical-format.md
EOF
)"

  issue "M2" "M2,ws:C" "WS-C: Public/hidden split + G4 perturbation generator (seeded ‖δ‖∞ = 1e-3 + camera/lighting jitter)" "$(cat <<'EOF'
## Context
Plan §4 WS-C item 3. G4 = MLPerf TEST04-style anti-memoization check.

## Definition of Done
- [ ] Deterministic public/hidden episode split per source (seeded, documented); hidden shards stored privately by organizers.
- [ ] Perturbation generator produces paired observations (`‖δ‖∞ = 1e-3` on pixels + mild camera/lighting jitter) with the seed recorded in the trace manifest.
- [ ] Gate G4 consumes the pairs and flags identical outputs on perturbed inputs (positive + negative-control tests).
EOF
)"

  issue "M2" "M2,ws:C,track:vla,track:wm" "WS-C: Golden outputs -- BF16 reference checkpoints run once by organizers → stored actions/frames for Gate A (CosSim ≥ 0.995, PSNR/LPIPS)" "$(cat <<'EOF'
## Context
Plan §4 WS-C item 4. Replaces the closed-form "reference outputs" in `embodied_bench.golden_dataset`.

## Definition of Done
- [ ] For each canonical model, organizers run the BF16 reference on the public and hidden traces once and store outputs (T1 actions; T2 frames) with checkpoint hash, container digest, and chip.
- [ ] Gate A thresholds set from reference-vs-reference re-run noise (CosSim ≥ 0.995 for T1; PSNR/LPIPS bands for T2) and recorded in the manifest.
- [ ] Golden outputs are versioned with the trace (`-v1`).
EOF
)"

  issue "M2" "M2,ws:C" "WS-C: Publish efficient-embodied-ai/EmbodiedX-Traces-v1 (~5 GB) on HF Hub with dataset card + license audit" "$(cat <<'EOF'
## Context
Plan §4 WS-C item 5 and the WS-C DoD.

## Definition of Done
- [ ] Dataset repo exists under the `efficient-embodied-ai` HF org with a card listing sources, licenses (LIBERO MIT, DROID CC-BY-4.0, Bridge V2 CC-BY-4.0, AgiBot-World Apache-2.0 -- re-audited), attribution, split definitions, and the G4 seed.
- [ ] `eeb data download --trace vla_libero_public` then `eeb measure --trace vla_libero_public` runs offline on a laptop with the dummy policy.
EOF
)"

  # ---- WS-B · M2 -- Reference servers + TPU path (Oct 17–24) ---------------------------------------------------
  issue "M2" "M2,ws:B,track:vla,hardware:nvidia" "WS-B: Server #2 -- GR00T-N1.7-3B (Isaac-GR00T, LIBERO checkpoints), dual-rate System-2/System-1 exposed" "$(cat <<'EOF'
## Context
Plan §4 WS-B item 1.

## Definition of Done
- [ ] `eeb/servers/gr00t/` serves `nvidia/GR00T-N1.7-3B` LIBERO checkpoints over gRPC/UDS; `Capabilities` reports both System-2 (VLM) and System-1 (DiT) rates and the native chunk size.
- [ ] Real `eeb measure` report on ≥ 1 NVIDIA chip with receipt; Gate A against WS-C golden outputs.
EOF
)"

  issue "M2" "M2,ws:B,track:wm,hardware:nvidia" "WS-B: Server #3 -- Cosmos-Predict2.5-2B action-conditioned Rollout (16 frames, 10 steps)" "$(cat <<'EOF'
## Context
Plan §4 WS-B item 2: canonical Track 2 World Model (`cosmos-predict2.5-2b`, `nvidia/Cosmos-Predict2.5-2B`, Apache-2.0 compatible weights).

## Definition of Done
- [ ] `eeb/servers/cosmos_predict25/` implements `Rollout(ctx, actions) → stream Frame` for `cosmos-predict2.5-2b` (`horizon_frames=16`, `denoising_steps=10`).
- [ ] Real `eeb measure` report (frames/s, per-frame p99, TTFF) on ≥ 1 chip; PSNR/LPIPS/IDM gate against golden frames.
EOF
)"

  issue "M2" "M2,ws:B,track:vla,hardware:tpu" "WS-B: TPU v6e-1 path -- π0.5 via openpi JAX; tpu-info telemetry; document which models have TPU-native stacks (R3)" "$(cat <<'EOF'
## Context
Plan §4 WS-B item 4; risk R3 (no TPU-native stack for GR00T / Cosmos-Predict2.5-2B); ADR-004 (TPU power `estimated`).

## Definition of Done
- [ ] π0.5 (openpi JAX) served on 1× TPU v6e-1 over gRPC/UDS; real `eeb measure` report with `power_source=estimated` clearly set.
- [ ] `tpu-info` parsing validated against the live VM image; fallback path tested.
- [ ] A table in `docs/` stating per model: TPU-native stack available (yes/no/planned).
EOF
)"

  issue "M2" "M2,ws:B,hardware:nvidia,hardware:amd,hardware:tpu" "WS-B: Docker images -- x86-cuda, jetson, rocm, tpu" "$(cat <<'EOF'
## Context
Plan §4 WS-B item 5. Existing `docker/Dockerfile.{x86-cuda,jetson-orin}` only package the placeholder CLI.

## Definition of Done
- [ ] Four images build in CI (or on the matching self-hosted runner) and run `eeb --help` + the dummy-policy smoke test.
- [ ] Images are published by digest; the digest format matches what `submissions/<...>/image.digest` expects (ADR-005).
- [ ] Default `CMD`s no longer run the analytic placeholder path.
EOF
)"

  # ---- WS-D · M3 -- Quality gates + submission pipeline (Oct 24–31) --------------------------------------------
  issue "M3" "M3,ws:D,track:vla,track:wm" "WS-D: Trace gate on the exact tensors from the timed run (G3 same-execution SHA-256): T1 action CosSim/L2; T2 PSNR/LPIPS (FVD optional)" "$(cat <<'EOF'
## Context
Plan §4 WS-D item 1. The gate math in `embodied_bench.quality_gates` is reusable; what changes is that inputs come from the timed LoadGen window.

## Definition of Done
- [ ] LoadGen stores timed-window outputs (or their SHA-256 + a sampled subset) and the gate recomputes metrics from exactly those tensors; execution IDs must match (G3).
- [ ] T1: CosSim / L2 vs golden; T2: PSNR / LPIPS (FVD optional, flagged).
- [ ] Negative-control tests: swapped execution id, tampered tensor, memoized output (G4).
EOF
)"

  issue "M3" "M3,ws:D,track:vla" "WS-D: LIBERO-RT -- LatencyCoupledEnv over LIBERO/robosuite with SYNC_HOLD / ASYNC_CHUNK / RTC, reactivity probes; Tier-2 replay; N=50 episodes × 3 seeds on Spatial/Object/Goal" "$(cat <<'EOF'
## Context
Plan §4 WS-D item 2, ADR-003. Replaces the closed-form success-rate model in `embodied_bench.libero_rt` / `stage2_closed_loop_verify` (the `ceil(L/Δt)` + scheduling arithmetic is reused).

## Definition of Done
- [ ] `LatencyCoupledEnv` wraps robosuite/MuJoCo LIBERO; applies `d = ceil(L/Δt)` delay per strategy; enforces the 1.0 s chunk cap and 1.5× p90 budget; injects the three reactivity probes.
- [ ] Tier-2: latency schedule recorded by `eeb measure` replayed into a CPU-only sim pass; `eeb eval-libero-rt --tier 2 --episodes 50 --seeds 3` on Spatial/Object/Goal.
- [ ] Output: latency-free vs real-time success, per-suite breakdown, probe outcomes, `tier`.
- [ ] Pinned LIBERO/robosuite/MuJoCo versions; assets cached; seeds published (R5).
EOF
)"

  issue "M3" "M3,ws:D,track:wm" "WS-D: WM controllability -- frozen inverse-dynamics probe (small IDM on LIBERO/DROID) → CosSim ≥ 0.95; 30–60 s drift ratio" "$(cat <<'EOF'
## Context
Plan §4 WS-D item 3.

## Definition of Done
- [ ] A small frozen IDM trained on LIBERO/DROID (weights + hash published) recovers actions from generated frame pairs; gate passes at CosSim ≥ 0.95 vs conditioning actions.
- [ ] 30–60 s rollout drift ratio computed and thresholded; both metrics in the T2 report.
- [ ] Positive/negative-control tests with a scrambled-action rollout.
EOF
)"

  issue "M3" "M3,ws:D" "WS-D: Submission flow -- submissions/<team>/<track>/<model>/<chip>/{recipe.yaml, image.digest, results.json, telemetry.csv, receipt.json}; CI preflight; SUBMITTED → PREFLIGHT_OK → REPRODUCED → CERTIFIED | REJECTED(reason)" "$(cat <<'EOF'
## Context
Plan §4 WS-D item 4, ADR-005.

## Definition of Done
- [ ] Directory contract documented in `docs/SUBMISSION_GUIDE.md`; `eeb submit --dry-run` packs it.
- [ ] CI preflight on PRs touching `submissions/`: schema, envelope, receipt signature, gate recompute, license/attribution; `physical_run: false` receipts rejected.
- [ ] State machine persisted per submission with mandatory non-empty reasons; only `CERTIFIED` reaches `leaderboard/*.json`.
- [ ] Receipt attested body extended with `execution_provenance`, image digest, trace version, runner identity (ADR-007 candidate).
EOF
)"

  issue "M3" "M3,ws:D,hardware:nvidia,hardware:tpu,hardware:amd" "WS-D: Self-hosted GitHub runners (InferenceX-style runners.yaml) on canonical NVIDIA + TPU hosts; AMD if sponsored" "$(cat <<'EOF'
## Context
Plan §4 WS-D item 5; depends on hardware procurement (M0) and the M3 DoD: a dummy external submission goes PR → preflight → re-run → leaderboard automatically.

## Definition of Done
- [ ] `runners.yaml` lists runner labels per chip; `rerun.yml` workflow triggered by an organizer `rerun:<chip>` label runs the participant image by digest on the hidden split and posts the organizer-signed receipt to the PR.
- [ ] Runner hosts hardened (ephemeral runners, no secrets in participant containers).
- [ ] End-to-end dry run with a dummy submission documented in this issue.
EOF
)"

  # ---- WS-E · M4 -- Organizer baselines + leaderboard (Nov 1–8) -------------------------------------------------
  issue "M4" "M4,ws:E,track:vla,track:wm,hardware:nvidia,hardware:tpu,hardware:amd" "WS-E: Baseline sweeps -- 3 models × {RTX 5090 or L40S, TPU v6e-1, (AMD if available)} × C∈{1..16}; Closed Division BF16 reference + FP8/INT8 optimized point" "$(cat <<'EOF'
## Context
Plan §4 WS-E item 1 and the M5 DoD (≥ 6 organizer baseline curves live).

## Definition of Done
- [ ] For each canonical model and available chip: BF16 reference sweep + one optimized point, 3 runs each, receipts verified, gates passed.
- [ ] Raw reports + telemetry archived under `submissions/organizers/...`.
- [ ] Any chip not secured (R6) is listed as "coming soon" on the site, never as a projected number.
EOF
)"

  issue "M4" "M4,ws:E" "WS-E: eeb pareto --normalize {raw,per_watt,per_100tops,per_dollar} → leaderboard/*.json; replace illustrative curves on the website with measured ones" "$(cat <<'EOF'
## Context
Plan §4 WS-E item 2, ADR-006. The two placeholder files in `leaderboard/` (generated by the analytic pipeline on 2026-10-07) must be regenerated or moved to test fixtures.

## Definition of Done
- [ ] `eeb pareto --inputs results/*.json --normalize per_watt --slo-ms 50 --out leaderboard/` writes per-track JSON from `CERTIFIED` receipts only, each point carrying `attestation_sha256`, `execution_provenance`, `power_source`, `tier`.
- [ ] Website explorer reads only generated JSON; hand-authored curves removed; "illustrative" banner removed only when measured data is live.
EOF
)"

  issue "M4" "M4,ws:E" "WS-E: GitHub Pages site; nightly regeneration workflow" "$(cat <<'EOF'
## Context
Plan §4 WS-E item 3, ADR-006.

## Definition of Done
- [ ] `leaderboard/site/` static explorer deployed via `pages.yml`; page shows generation commit SHA + timestamp.
- [ ] Nightly workflow (and on merges touching `leaderboard/` or `submissions/`) re-runs `eeb pareto` and republishes.
- [ ] `estimated` power points and Tier-2 vs Tier-1 results visually distinguished; every point links to its receipt.
EOF
)"

  issue "M4" "M4,ws:E" "WS-E: Docs -- RULES.md v1.0, SUBMISSION_GUIDE.md, Quickstart, FAQ" "$(cat <<'EOF'
## Context
Plan §4 WS-E item 4 and §3 repo layout (`docs/`).

## Definition of Done
- [ ] `docs/RULES.md` v1.0 (tracks, envelope, metrics, gates, tiers, certification, re-run rights) frozen before M5.
- [ ] `docs/SUBMISSION_GUIDE.md` matching the `submissions/` contract (ADR-005); Quickstart for `eeb serve` + `eeb measure` with the dummy policy; FAQ covering R1–R7.
- [ ] README status section updated to point at measured results once they exist.
EOF
)"

  # ---- M5 -- Launch at CoRL (Nov 9–12) --------------------------------------------------------------------------
  issue "M5" "M5" "M5: Talk/demo -- live eeb measure on a Jetson/laptop + leaderboard walkthrough; open call for submissions" "$(cat <<'EOF'
## Context
Plan §4 M5. DoD for the milestone: rules frozen, harness tagged `v0.1.0`, ≥ 6 organizer baseline curves live.

## Definition of Done
- [ ] Demo script rehearsed: `eeb serve` (dummy or π0.5) + `eeb measure` live on a Jetson or laptop, then the leaderboard walkthrough.
- [ ] `v0.1.0` tag cut after RULES.md freeze; release notes list what is measured vs. "coming soon".
- [ ] Open call for submissions published with the Jan 2027 first-results cutoff (plan §8 item 1 / R1).
EOF
)"

  # ---- M6 -- v1.0 living benchmark (Nov 13 → Jan 2027) ----------------------------------------------------------
  issue "M6" "M6,hardware:amd,track:vla,track:wm" "M6: v1.0 living benchmark -- AMD runner, Tier-1 GPU sim-in-the-loop LIBERO-RT, FVD + S2 policy-ranking evaluator, P3 edge-cloud emulation, registry round 2, first results cutoff + awards" "$(cat <<'EOF'
## Context
Plan §4 M6 (tracking issue; split into child issues when scheduled).

## Definition of Done
- [ ] AMD self-hosted runner online and at least one certified AMD baseline.
- [ ] Tier-1 GPU-in-the-loop LIBERO-RT reported as a separate column (ADR-003).
- [ ] FVD + S2 policy-ranking evaluator for T2; `P3` edge-cloud emulation (RTT/jitter/drop) scenario.
- [ ] Registry round 2 (`MolmoAct2`, `LingBot-World`, `Qwen3.5`) with golden outputs.
- [ ] First external results cutoff + awards process at a 2027 venue announced.
EOF
)"
fi

echo
echo "# Summary: labels=$N_LABELS milestones=$N_MILESTONES issues=$N_ISSUES mode=$MODE repo=$REPO"
if [[ "$MODE" == "dry-run" ]]; then
  echo "# Nothing was created. To execute: $SCRIPT_NAME --apply --repo <owner>/<name>"
fi
