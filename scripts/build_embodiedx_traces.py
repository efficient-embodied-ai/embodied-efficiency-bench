#!/usr/bin/env python3
"""CLI builder for the `EmbodiedX-Traces-v1` Open-Source Golden Dataset (`mini_v1` split).

Usage:
  python3 scripts/build_embodiedx_traces.py --output-dir data/embodiedx_traces_v1_mini --split mini_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

# Ensure repository root is importable when invoked as python3 scripts/build_embodiedx_traces.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from embodied_bench.dataset_loader import (  # noqa: E402
    DEFAULT_MINI_V1_DIR,
    EmbodiedXDatasetLoader,
    generate_embodiedx_traces_split,
)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Generate EmbodiedX-Traces-v1 episode shards (.npz) and manifest.json."
  )
  parser.add_argument(
      "--output-dir",
      type=str,
      default=str(DEFAULT_MINI_V1_DIR),
      help="Target directory for episode shards and manifest.json.",
  )
  parser.add_argument(
      "--split",
      type=str,
      default="mini_v1",
      help="Dataset split identifier (default: mini_v1).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=20261008,
      help="Base deterministic RNG seed (default: 20261008).",
  )
  parser.add_argument(
      "--num-steps",
      type=int,
      default=16,
      help="Number of steps T per episode shard (default: 16).",
  )
  parser.add_argument(
      "--resolution-hw",
      type=int,
      default=64,
      help="Spatial RGB resolution H=W per camera view (64 for fast CI mini_v1, 224 for pi0.5 eval, 256 for GR00T/Cosmos eval).",
  )
  parser.add_argument(
      "--max-episodes",
      type=int,
      default=None,
      help="Optional cap on number of episodes generated (default: all 8 canonical episodes).",
  )
  return parser


def main(argv: Optional[List[str]] = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)

  out_dir = Path(args.output_dir)
  manifest = generate_embodiedx_traces_split(
      output_dir=out_dir,
      split=args.split,
      base_seed=args.seed,
      num_steps=args.num_steps,
      resolution_hw=args.resolution_hw,
      max_episodes=args.max_episodes,
  )

  loader = EmbodiedXDatasetLoader(dataset_dir=out_dir, split=args.split)
  verified, errors = loader.verify_manifest()
  if not verified:
    sys.stderr.write(
        "ERROR: Manifest SHA-256 verification failed after generation:\n"
        + "\n".join(errors)
        + "\n"
    )
    return 1

  summary = {
      "status": "OK",
      "dataset_version": manifest["dataset_version"],
      "split": manifest["split"],
      "output_dir": str(out_dir.resolve()),
      "total_episodes": manifest["total_episodes"],
      "total_steps": manifest["total_steps"],
      "total_bytes": manifest["total_bytes"],
      "resolution_hw": manifest.get("resolution_hw", int(args.resolution_hw)),
      "bytes_per_request_images": manifest.get("bytes_per_request_images"),
      "sources": manifest["sources"],
      "canonical_models": manifest["canonical_models"],
      "mean_temporal_frame_coherence": manifest["mean_temporal_frame_coherence"],
      "min_temporal_frame_coherence": manifest["min_temporal_frame_coherence"],
      "mean_instruction_prefix_reuse_ratio": manifest["mean_instruction_prefix_reuse_ratio"],
      "g4_perturbation_linf_eps": manifest["g4_perturbation_linf_eps"],
      "sha256_verified": verified,
  }
  print(json.dumps(summary, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
