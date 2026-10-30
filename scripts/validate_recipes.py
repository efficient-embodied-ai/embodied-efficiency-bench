#!/usr/bin/env python3
"""Validate every recipe file against `embodied_bench.schema.RecipeConfig` and EEB v1.0 Closed Division rules.

Each `recipes/*.yaml` / `recipes/*.json` file is loaded with `load_yaml_or_json()` and
parsed with `RecipeConfig.from_dict()`, which runs `RecipeConfig.validate()` (track /
division / single-device / perf-cap rules). For canonical repository recipes under
`recipes/`, this validator also enforces the EEB v1.0 Closed Division invariants:
  - `baseline_model_id` in `CANONICAL_V1_MODELS` (`pi05-3b`, `gr00t-n1.7-3b`, `cosmos-predict2.5-2b`)
  - `division == "CLOSED"` and `weights_frozen is True`
  - Locked `execute_horizon == 5` (`pi05-3b`), `execute_horizon == 8` (`gr00t-n1.7-3b`),
    or `horizon_frames == 16` (`cosmos-predict2.5-2b`)
  - `concurrency_levels == [1, 2, 4, 8, 16]`

Usage (from the repository root):
  python3 scripts/validate_recipes.py                 # validates recipes/*.yaml and recipes/*.json
  python3 scripts/validate_recipes.py path/a.yaml ... # validates only the given files

Exit codes:
  0  every recipe parsed and validated
  1  at least one recipe failed (each failure is printed as `FAIL <path>: <error>`)
  2  no recipe files were found (misconfigured path or empty directory)

This script only checks *schema and v1.0 Closed Division rule* validity; it does not
execute a hardware benchmark and says nothing about whether a recipe's numbers were
measured on physical hardware.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from embodied_bench.schema import (  # noqa: E402
    CANONICAL_V1_MODELS,
    RecipeConfig,
    is_vla_track,
    is_wm_track,
    load_yaml_or_json,
)

RECIPE_SUFFIXES = (".yaml", ".yml", ".json")
V1_LOCKED_EXECUTE_HORIZONS = {
    "pi05-3b": 5,
    "gr00t-n1.7-3b": 8,
}
V1_LOCKED_NATIVE_CHUNKS = {
    "pi05-3b": 10,
    "gr00t-n1.7-3b": 16,
}
V1_LOCKED_RESOLUTIONS_HW = {
    "pi05-3b": 224,
    "gr00t-n1.7-3b": 256,
    "cosmos-predict2.5-2b": 256,
}
V1_LOCKED_WM_HORIZON_FRAMES = 16
V1_CONCURRENCY_SWEEP = [1, 2, 4, 8, 16]


def discover_recipe_paths(recipes_dir: Path) -> list[Path]:
  """Return every recipe file under `recipes_dir`, sorted for deterministic output."""
  if not recipes_dir.is_dir():
    return []
  return sorted(p for p in recipes_dir.iterdir() if p.is_file() and p.suffix.lower() in RECIPE_SUFFIXES)


def _is_canonical_repo_recipe(path: Path) -> bool:
  try:
    resolved = path.resolve()
    return (REPO_ROOT / "recipes").resolve() in resolved.parents
  except OSError:
    return False


def validate_recipe(path: Path, enforce_v1_closed: Optional[bool] = None) -> Optional[str]:
  """Validate one recipe; return None on success or a one-line error description on failure."""
  try:
    cfg = RecipeConfig.from_dict(load_yaml_or_json(path))
  except Exception as exc:  # noqa: BLE001 - any parse/validation error is a CI failure
    return f"{type(exc).__name__}: {exc}"
  if not cfg.recipe_id:
    return "ValueError: recipe_id is empty"

  check_v1 = _is_canonical_repo_recipe(path) if enforce_v1_closed is None else enforce_v1_closed
  if check_v1:
    if cfg.division != "CLOSED":
      return f"ValueError: EEB v1.0 canonical recipes must declare division='CLOSED', got {cfg.division!r}"
    if not cfg.weights_frozen:
      return "ValueError: EEB v1.0 Closed Division requires weights_frozen=True"
    if cfg.baseline_model_id not in CANONICAL_V1_MODELS:
      return (
          f"ValueError: baseline_model_id={cfg.baseline_model_id!r} is not in "
          f"CANONICAL_V1_MODELS={CANONICAL_V1_MODELS}"
      )
    if list(cfg.concurrency_levels) != V1_CONCURRENCY_SWEEP:
      return (
          f"ValueError: concurrency_levels={cfg.concurrency_levels!r} must equal "
          f"{V1_CONCURRENCY_SWEEP} for EEB v1.0 Closed Division"
      )
    expected_res = V1_LOCKED_RESOLUTIONS_HW.get(cfg.baseline_model_id)
    if expected_res is not None:
      if cfg.resolution_hw != expected_res or cfg.reference_resolution_hw != expected_res:
        return (
            f"ValueError: {cfg.baseline_model_id} Closed Division requires canonical "
            f"resolution_hw=reference_resolution_hw={expected_res}, "
            f"got resolution_hw={cfg.resolution_hw!r}, reference_resolution_hw={cfg.reference_resolution_hw!r}"
        )
    if is_vla_track(cfg.track):
      expected_h = V1_LOCKED_EXECUTE_HORIZONS.get(cfg.baseline_model_id)
      if expected_h is not None and cfg.execute_horizon != expected_h:
        return (
            f"ValueError: {cfg.baseline_model_id} requires locked execute_horizon={expected_h}, "
            f"got {cfg.execute_horizon!r}"
        )
      if expected_h is not None and list(cfg.action_chunk_sizes) != [expected_h]:
        return (
            f"ValueError: {cfg.baseline_model_id} Closed Division requires action_chunk_sizes=[{expected_h}], "
            f"got {cfg.action_chunk_sizes!r}"
        )
      expected_native_h = V1_LOCKED_NATIVE_CHUNKS.get(cfg.baseline_model_id)
      if expected_native_h is not None and cfg.native_action_chunk_size != expected_native_h:
        return (
            f"ValueError: {cfg.baseline_model_id} Closed Division requires "
            f"native_action_chunk_size={expected_native_h}, got {cfg.native_action_chunk_size!r}"
        )
    elif is_wm_track(cfg.track):
      if cfg.horizon_frames != V1_LOCKED_WM_HORIZON_FRAMES:
        return (
            f"ValueError: {cfg.baseline_model_id} requires locked horizon_frames={V1_LOCKED_WM_HORIZON_FRAMES}, "
            f"got {cfg.horizon_frames!r}"
        )
      if cfg.resolution_hw != 256:
        return (
            f"ValueError: {cfg.baseline_model_id} Closed Division requires canonical resolution_hw=256, "
            f"got {cfg.resolution_hw!r}"
        )
      if float(cfg.guidance_scale) != 1.0:
        return (
            f"ValueError: {cfg.baseline_model_id} Closed Division requires canonical guidance_scale=1.0, "
            f"got {cfg.guidance_scale!r}"
        )
  return None


def main(argv: Optional[Sequence[str]] = None) -> int:
  args = list(sys.argv[1:] if argv is None else argv)
  if args and any(a in ("-h", "--help") for a in args):
    print(__doc__)
    return 0

  enforce_v1_closed: Optional[bool] = None
  recipes_dir: Optional[Path] = None
  positional_paths: list[Path] = []
  idx = 0
  while idx < len(args):
    tok = args[idx]
    if tok == "--strict":
      enforce_v1_closed = True
      idx += 1
    elif tok == "--no-strict":
      enforce_v1_closed = False
      idx += 1
    elif tok in ("--recipes-dir", "--dir") and idx + 1 < len(args):
      recipes_dir = Path(args[idx + 1])
      idx += 2
    elif tok.startswith("--recipes-dir="):
      recipes_dir = Path(tok.split("=", 1)[1])
      idx += 1
    else:
      p = Path(tok)
      if p.is_dir():
        positional_paths.extend(discover_recipe_paths(p))
      else:
        positional_paths.append(p)
      idx += 1

  if positional_paths:
    paths = positional_paths
  elif recipes_dir is not None:
    paths = discover_recipe_paths(recipes_dir)
  else:
    paths = discover_recipe_paths(REPO_ROOT / "recipes")

  if not paths:
    print("ERROR: no recipe files found (expected recipes/*.yaml|*.yml|*.json or explicit paths).", file=sys.stderr)
    return 2

  failures = 0
  for path in paths:
    err = validate_recipe(path, enforce_v1_closed=enforce_v1_closed)
    try:
      rel = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
      rel = path
    if err is None:
      print(f"OK   {rel}")
    else:
      failures += 1
      print(f"FAIL {rel}: {err}")

  total = len(paths)
  if failures:
    print(f"\n{failures}/{total} recipe(s) FAILED schema validation.", file=sys.stderr)
    return 1
  print(f"\nAll {total} recipe(s) passed EEB v1.0 Closed Division RecipeConfig schema validation.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
