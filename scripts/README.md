# `scripts/`

Operational helpers for organizers and CI verification. None of these scripts fabricate benchmark measurements.

| Script | Purpose | Safe by default? |
|---|---|---|
| [`bootstrap_github_issues.sh`](bootstrap_github_issues.sh) | Create the repository's GitHub labels, milestones `M0` through `M6`, and tracking issues. | **Yes**: `--dry-run` is the default and prints shell-quoted `gh` commands. Writes require `--apply --repo owner/name`. Idempotent on re-run. |
| [`validate_recipes.py`](validate_recipes.py) | CI schema gate: loads every `recipes/*.yaml|yml|json` via `load_yaml_or_json()`, parses it with `RecipeConfig.from_dict()` (`validate()`), and enforces Season 1 `CAP_SINGLE_CHIP` and division rules across all 43 recipes (`pi05-3b`, `gr00t-n1.7-3b`, `cosmos-predict2.5-2b`). | Yes (read-only). Exit `0` all valid, `1` any failure (`FAIL <path>: <error>`), `2` no files found. |
| [`export_website_leaderboard.py`](export_website_leaderboard.py) | Recompiles `leaderboard/leaderboard_v1.json`, `leaderboard/LEADERBOARD.md`, and `leaderboard/embodiedx_leaderboard_data.json` strictly from verified community submissions (`bundle_summary.json` + `submission_meta.json` or full verified bundles) under `submissions/` (yielding `0` entries on Day 1 prior to community PRs). | Reads verified cryptographic receipts and overwrites the canonical leaderboard JSON/Markdown outputs when invoked explicitly. |

## `bootstrap_github_issues.sh`

```bash
# Preview everything (no gh needed, nothing created):
scripts/bootstrap_github_issues.sh --dry-run
scripts/bootstrap_github_issues.sh --dry-run --repo efficient-embodied-ai/embodied-efficiency-bench

# Execute against a repository you administer (gh must be authenticated with repo scope):
scripts/bootstrap_github_issues.sh --apply --repo efficient-embodied-ai/embodied-efficiency-bench
```

Safety behaviour:

| Invocation | Result |
|---|---|
| `--apply` without `--repo` | exits `2`: `--apply requires an explicit --repo owner/name` |
| `--apply --repo badformat` | exits `2`: `--repo must look like owner/name` |
| `--apply` with `gh` missing / unauthenticated / repo not accessible | exits `3` with the remediation command |
| unknown flag | exits `2` and prints usage |

## `validate_recipes.py`

```bash
python3 scripts/validate_recipes.py --recipes-dir recipes --strict   # all 43 recipes (including Official Season 1 L40S_48GB + TPU_V6E_1 targets)
python3 scripts/validate_recipes.py recipes/track1_vla_pi05_l40s_bf16.yaml
```

Negative control (expected to fail with exit `1`): copy a recipe, set `num_devices: 2`, and pass it in (`RecipeConfig.validate()` rejects multi-device recipes). Exercised by `tests/test_embodied_bench.py::test_recipe_schema_validator_script_fires_on_violation_and_stays_silent_on_valid`.

## Conventions for New Scripts

- Default to a non-mutating mode; require an explicit flag plus an explicit target for anything that writes outside the repo.
- Print the exact commands you would run (`printf '%q '`) so a reviewer can inspect them verbatim.
- Never fabricate benchmark numbers (`export_website_leaderboard.py` reads verified cryptographic receipts only and emits an empty `0`-entry leaderboard when `submissions/` has zero community submissions).
