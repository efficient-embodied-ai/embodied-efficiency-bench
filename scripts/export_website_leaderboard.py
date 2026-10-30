#!/usr/bin/env python3
"""CLI entrypoint to compile verified single-chip submissions into `leaderboard/leaderboard_v1.json` and `leaderboard/LEADERBOARD.md`.

Supports any flag order for `--submissions-dir`, `--output`, `--markdown-output`,
`--generate-reference-seeds`, and `--no-strict-verify`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from embodied_bench.leaderboard_compiler import (  # noqa: E402
    compile_submissions_to_leaderboard,
    generate_reference_submissions,
    render_pr_comment_markdown,
)
from embodied_bench.schema import CANONICAL_V1_MODELS, atomic_write_json, atomic_write_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="export_website_leaderboard.py",
      description=(
          "Discover and cryptographically verify single-chip receipt.json files "
          "under submissions/, enforce CAP_SINGLE_CHIP (num_devices == 1, <=64GB VRAM, "
          "<=1000 BF16 TFLOPS) and G1-G4 compliance, and export EEB v1.0 Closed Division "
          "4-way SemiAnalysis Pareto frontiers to leaderboard/leaderboard_v1.json and "
          "leaderboard/LEADERBOARD.md."
      ),
  )
  parser.add_argument(
      "--submissions-dir",
      "-s",
      type=Path,
      default=REPO_ROOT / "submissions",
      help="Directory to recursively scan for verified receipt.json or bundle_summary.json files (default: submissions).",
  )
  parser.add_argument(
      "--output",
      "--output-json",
      "--leaderboard-out",
      "-o",
      dest="output",
      type=Path,
      default=REPO_ROOT / "leaderboard" / "leaderboard_v1.json",
      help="Output path for compiled leaderboard JSON (default: leaderboard/leaderboard_v1.json).",
  )
  parser.add_argument(
      "--markdown-output",
      "--output-md",
      "--markdown-out",
      "-m",
      dest="markdown_output",
      type=Path,
      default=None,
      help="Optional path to write GitHub-Flavored Markdown summary table (e.g. leaderboard/LEADERBOARD.md).",
  )
  parser.add_argument(
      "--website-output",
      "--website-out",
      dest="website_output",
      type=Path,
      default=None,
      help="Optional path to also export website leaderboard JSON (e.g. leaderboard/embodiedx_leaderboard_data.json).",
  )
  parser.add_argument(
      "--generate-reference-seeds",
      action="store_true",
      help="Generate/refresh reference baseline submissions inside --submissions-dir before compiling.",
  )
  parser.add_argument(
      "--no-strict-verify",
      action="store_true",
      help="Quarantine invalid receipts into rejected_submissions instead of raising ValueError.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.generate_reference_seeds:
    generate_reference_submissions(args.submissions_dir)

  leaderboard_data = compile_submissions_to_leaderboard(
      submissions_dir=args.submissions_dir,
      output_json_path=args.output,
      strict_verify=not args.no_strict_verify,
  )
  leaderboard_data["canonical_models"] = list(CANONICAL_V1_MODELS)

  def _slim_points_in_obj(obj: object) -> object:
    if isinstance(obj, list):
      return [_slim_points_in_obj(x) for x in obj]
    if isinstance(obj, dict):
      if "point_id" in obj and "semianalysis_normalizations" in obj:
        return {k: v for k, v in obj.items() if k != "semianalysis_normalizations"}
      return {k: _slim_points_in_obj(v) for k, v in obj.items()}
    return obj

  leaderboard_data = _slim_points_in_obj(leaderboard_data)  # type: ignore[assignment]
  atomic_write_json(args.output, leaderboard_data)

  md_text = render_pr_comment_markdown(leaderboard_data)
  prov_summary = leaderboard_data.get("execution_provenance_summary", {})
  has_analytic = int(prov_summary.get("num_analytic_reference_submissions", 0)) > 0
  if has_analytic and "ILLUSTRATIVE_SEED" not in md_text:
    md_lines = md_text.splitlines()
    banner = (
        "- **Execution Provenance:** `ILLUSTRATIVE_SEED` / `PENDING` physical hardware measurement "
        "(`physical_run: false`; reference baseline seeds only)"
    )
    if len(md_lines) >= 4:
      md_lines.insert(4, banner)
      md_text = "\n".join(md_lines) + "\n"

  if args.markdown_output is not None:
    atomic_write_text(args.markdown_output, md_text)

  if args.website_output is not None:
    from leaderboard.export_to_website import export_verified_runs_to_website_json

    export_verified_runs_to_website_json(
        leaderboard_json_path=args.output,
        output_website_json_path=args.website_output,
    )

  print(
      f"[export_website_leaderboard] Compiled {leaderboard_data['num_verified_submissions']} verified "
      f"single-chip submissions across models={leaderboard_data['models_represented']} and "
      f"vendors={leaderboard_data['vendors_represented']} -> {args.output}"
  )
  if args.markdown_output is not None:
    print(f"[export_website_leaderboard] Wrote Markdown table -> {args.markdown_output}")
  if args.website_output is not None:
    print(f"[export_website_leaderboard] Wrote website JSON -> {args.website_output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
