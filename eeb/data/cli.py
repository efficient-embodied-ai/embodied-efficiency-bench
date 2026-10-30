"""``eeb data ...`` subcommands.

WS-A's ``eeb/cli.py`` imports this lazily::

    from eeb.data.cli import register_cli
    register_cli(subparsers)          # adds `eeb data {download,pack-traces,inspect,perturb}`

Each leaf parser sets ``func`` (``args.func(args) -> int``), matching the existing CLI convention.
The module is also runnable on its own (``python -m eeb.data ...`` / ``python -m eeb.data.cli ...``)
so the data tooling works before/without the umbrella CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from eeb.data.download import DEFAULT_CACHE_DIR, download, plan_episode_files, probe_hub_layout
from eeb.data.golden import generate_golden_reference
from eeb.data.lerobot_loader import LeRobotDataset, MissingDependencyError
from eeb.data.pack import AUTO_SPLIT, find_traces_dir, pack_traces, verify_traces
from eeb.data.perturb import IMAGE_SPACES, PerturbationConfig, verify_perturbation_manifest, write_perturbed_split
from eeb.data.schema import DEFAULT_HIDDEN_FRACTION, DEFAULT_SPLIT_SEED, SPLITS, TraceManifest


def _print(msg: str) -> None:
  print(msg, flush=True)


def _csv(value: Optional[str]) -> Optional[list]:
  if value is None or value == "":
    return None
  return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------------
def _cmd_download(args: argparse.Namespace) -> int:
  try:
    if args.plan_only:
      plan = plan_episode_files(args.plan_only, args.episodes or 1, include_videos=args.videos, dataset_id=args.dataset_id)
      _print(json.dumps({"dataset_id": plan.dataset_id, "episode_indices": plan.episode_indices, "data_files": plan.data_files, "video_files": plan.video_files}, indent=2))
      return 0
    dest = download(
        args.dataset_id,
        revision=args.revision,
        allow_patterns=_csv(args.allow_patterns),
        cache_dir=args.cache_dir,
        episodes=args.episodes,
        include_videos=args.videos,
        token=args.token,
    )
  except MissingDependencyError as e:
    _print(f"ERROR: {e}")
    return 2
  ds = LeRobotDataset(dest, source_id=args.dataset_id)
  summary = ds.summary()
  _print(f"downloaded {args.dataset_id} -> {dest}")
  _print(json.dumps(summary, indent=2))
  return 0


# --------------------------------------------------------------------------------------------
# pack-traces
# --------------------------------------------------------------------------------------------
def _cmd_pack_traces(args: argparse.Namespace) -> int:
  try:
    result = pack_traces(
        args.src,
        args.out,
        split=args.split,
        max_episodes=args.max_episodes,
        cameras=_csv(args.cameras),
        fps=args.fps,
        source_dataset=args.source_dataset,
        source_id=args.source_id,
        license=args.license,
        seed=args.seed,
        hidden_fraction=args.hidden_fraction,
        episode_indices=[int(x) for x in _csv(args.episodes)] if args.episodes else None,
        write_parquet=args.parquet,
        decode_images=not args.no_images,
        strict_instruction=not args.allow_instruction_change,
        source_revision=args.source_revision or "",
        state_key=args.state_key,
        action_key=args.action_key,
        image_dir=args.image_dir,
        log=_print,
    )
  except (MissingDependencyError, FileNotFoundError, ValueError, KeyError) as e:
    _print(f"ERROR: {e}")
    return 2
  _print(json.dumps(result.summary(), indent=2))
  for split, man in result.manifests.items():
    _print(f"manifest: {result.traces_dir / split / 'manifest.json'}  episodes={man.episode_count} steps={man.step_count} cameras={man.cameras}")
  return 0


# --------------------------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------------------------
def _cmd_inspect(args: argparse.Namespace) -> int:
  if args.hub:
    try:
      _print(json.dumps(probe_hub_layout(args.hub, revision=args.revision or "main"), indent=2))
      return 0
    except RuntimeError as e:
      _print(f"ERROR: {e}")
      return 2
  p = Path(args.path)
  if (p / "meta" / "info.json").exists():
    try:
      ds = LeRobotDataset(p)
    except MissingDependencyError as e:
      _print(f"ERROR: {e}")
      return 2
    out = ds.summary()
    out["first_episodes"] = [
        {"episode_index": e.episode_index, "length": e.length, "tasks": e.tasks[:1], "local_data": ds.has_local_data(e)}
        for e in ds.episodes[: args.limit]
    ]
    _print(json.dumps(out, indent=2))
    return 0
  if (p / "perturbation_manifest.json").exists():
    ok, problems = verify_perturbation_manifest(p)
    _print(f"perturbation manifest {'OK' if ok else 'FAILED'}: {problems}")
    return 0 if ok else 1
  pert_dirs = []
  for base_cand in (p, p / "perturbed"):
    if base_cand.is_dir():
      for sp in SPLITS:
        if (base_cand / sp / "perturbation_manifest.json").exists():
          pert_dirs.append(base_cand / sp)
  if pert_dirs and not (p / "traces").is_dir() and not (p / "manifest.json").exists():
    rc = 0
    for pd in pert_dirs:
      ok, problems = verify_perturbation_manifest(pd)
      _print(f"[{pd.name}] perturbation manifest {'OK' if ok else 'FAILED'}: {problems}")
      rc |= 0 if ok else 1
    return rc
  try:
    traces_dir = find_traces_dir(p)
  except FileNotFoundError as e:
    _print(f"ERROR: {e}")
    return 2
  man_file = traces_dir / "manifest.json"
  if man_file.exists():
    try:
      raw_man = json.loads(man_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
      raw_man = {}
    if isinstance(raw_man, dict) and "shards" in raw_man and "episodes" not in raw_man:
      from eeb.loadgen.sweep import TraceBundleObservationSource

      try:
        src = TraceBundleObservationSource(str(traces_dir))
      except Exception as e:  # noqa: BLE001
        _print(f"ERROR: {e}")
        return 1
      man = src.manifest
      split_tag = str(raw_man.get("split", man.split))
      _print(f"[{split_tag}] source={man.source} fps={man.fps} cameras={man.cameras} episodes={man.episode_count} steps={man.step_count}")
      for e in man.episodes[: args.limit]:
        _print(f"  {e.file}  steps={e.steps:>5d}  sha256={e.sha256[:16]}...  instruction={e.instruction[:70]!r}")
      _print(f"  manifest verification OK: {man.episode_count} file(s) checked")
      return 0
  results = verify_traces(traces_dir, deep=args.deep)
  rc = 0
  for split, ver in results.items():
    man = TraceManifest.load(traces_dir / split if (traces_dir / split).is_dir() else traces_dir)
    _print(f"[{split}] source={man.source} fps={man.fps} cameras={man.cameras} episodes={man.episode_count} steps={man.step_count}")
    for e in man.episodes[: args.limit]:
      _print(f"  {e.file}  steps={e.steps:>5d}  sha256={e.sha256[:16]}...  instruction={e.instruction[:70]!r}")
    _print("  " + ver.summary())
    rc |= 0 if ver.ok else 1
  return rc


# --------------------------------------------------------------------------------------------
# perturb
# --------------------------------------------------------------------------------------------
def _cmd_perturb(args: argparse.Namespace) -> int:
  try:
    cfg = PerturbationConfig(
        seed=args.seed,
        eps_linf=args.eps,
        image_space=args.image_space,
        camera_jitter_px=args.camera_jitter_px,
        brightness_jitter=args.brightness_jitter,
        perturb_images=not args.no_images,
        perturb_proprio=not args.no_proprio,
    )
    out_root = args.out if args.out else args.traces
    log_fn = (lambda _msg: None) if getattr(args, "quiet", False) else _print
    manifest_path = write_perturbed_split(args.traces, out_root, cfg, split=args.split, max_episodes=args.max_episodes, log=log_fn)
  except (FileNotFoundError, ValueError) as e:
    _print(f"ERROR: {e}")
    return 2
  _print(f"perturbation manifest: {manifest_path}")
  _print(json.dumps(cfg.to_dict(), indent=2))
  return 0


# --------------------------------------------------------------------------------------------
# generate-golden
# --------------------------------------------------------------------------------------------
def _parse_hw(raw: Optional[str]):
  if not raw:
    return None
  txt = str(raw).strip().lower()
  if "x" not in txt:
    try:
      side = int(txt)
      if side <= 0:
        raise ValueError
      return (side, side)
    except ValueError as exc:
      raise ValueError(f"--image-hw expects HxW like 224x224 or 256x256, got {raw!r}.") from exc
  try:
    h, w = txt.split("x")
    hi, wi = int(h), int(w)
    if hi <= 0 or wi <= 0:
      raise ValueError
    return (hi, wi)
  except ValueError as exc:
    raise ValueError(f"--image-hw expects HxW like 224x224 or 256x256, got {raw!r}.") from exc


def _cmd_prepare_traces(args: argparse.Namespace) -> int:
  import fcntl
  import os

  from embodied_bench.dataset_loader import (
      EmbodiedXDatasetLoader,
      generate_embodiedx_traces_split,
      generate_real_embodiedx_traces_from_lerobot,
  )
  from embodied_bench.schema import _lock_path_for, cleanup_orphan_temp_files
  from eeb.loadgen.sweep import TraceBundleObservationSource

  log = (lambda msg: None) if getattr(args, "quiet", False) else (lambda msg: _print(f"[eeb data prepare-traces] {msg}"))
  try:
    track_arg = str(getattr(args, "track", None) or "all").strip()
    is_wm_track_hint = track_arg.lower() in ("wm", "wm_track", "track2", "track2_wm")
    default_hw = 256 if (is_wm_track_hint or track_arg.lower() == "track1b") else 224
    synthetic_smoke = bool(getattr(args, "synthetic_ci_smoke", False))

    res_hw_arg = getattr(args, "resolution_hw", None)
    hw_tuple = _parse_hw(getattr(args, "image_hw", None))
    if res_hw_arg is not None and hw_tuple is not None and (hw_tuple[0] != int(res_hw_arg) or hw_tuple[1] != int(res_hw_arg)):
      raise ValueError(
          f"Conflicting resolution arguments: --resolution-hw {res_hw_arg} vs --image-hw {args.image_hw!r}."
      )
    if res_hw_arg is not None:
      hw = int(res_hw_arg)
    elif hw_tuple is not None:
      if hw_tuple[0] != hw_tuple[1]:
        raise ValueError(
            f"prepare-traces expects square HxW resolution (e.g. 224x224 or 256x256), got {args.image_hw!r}."
        )
      hw = int(hw_tuple[0])
    else:
      hw = default_hw
    if hw < 16:
      raise ValueError(f"resolution_hw must be >= 16 (protocol minimum: 224 for VLA/pi05, 256 for GR00T/Cosmos), got {hw}.")
    if args.episodes is not None and int(args.episodes) < 1:
      raise ValueError(f"--episodes must be >= 1, got {args.episodes}.")
    if int(args.steps) < 2:
      raise ValueError(f"--steps must be >= 2, got {args.steps}.")

    eff_cache_dir = (
        getattr(args, "cache_dir", None)
        or os.environ.get("EEB_LEROBOT_CACHE_DIR")
        or (Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "eeb" / "lerobot")
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_path_for(out_dir / ".prepare_traces")
    mode_label = "SYNTHETIC_CI_SMOKE" if synthetic_smoke else f"REAL_ROBOT_TRACES(track={track_arg})"
    log(
        f"materializing EmbodiedX-Traces-v1 [{mode_label}] split={args.split!r} at {hw}x{hw} RGB "
        f"(episodes={args.episodes or 'default'}, steps={args.steps}) -> {out_dir}"
    )
    with open(lock_file, "a+", encoding="utf-8") as lock_fp:
      fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
      try:
        cleanup_orphan_temp_files(out_dir, prefix=".tmp_embodied_")
        cleanup_orphan_temp_files(out_dir, prefix=".tmp_shard_")
        for item in out_dir.iterdir():
          if item.is_file() and item.name.startswith(".") and item.name.endswith(".tmp"):
            try:
              item.unlink()
            except OSError:
              pass
        if synthetic_smoke:
          manifest = generate_embodiedx_traces_split(
              output_dir=out_dir,
              split=args.split,
              base_seed=int(args.seed),
              num_steps=int(args.steps),
              resolution_hw=hw,
              max_episodes=args.episodes,
          )
        else:
          manifest = generate_real_embodiedx_traces_from_lerobot(
              output_dir=out_dir,
              split=args.split,
              base_seed=int(args.seed),
              num_steps=int(args.steps),
              resolution_hw=hw,
              max_episodes=args.episodes,
              cache_dir=eff_cache_dir,
              tracks=track_arg,
              token=getattr(args, "token", None),
          )
        loader = EmbodiedXDatasetLoader(dataset_dir=out_dir, split=args.split)
        verified, errors = loader.verify_manifest()
        if not verified:
          raise RuntimeError("Manifest SHA-256 verification failed: " + "; ".join(errors))
        src = TraceBundleObservationSource(str(out_dir), split="public")
        obs_desc = src.describe()
      finally:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
  except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
    _print(f"ERROR: {e}")
    return 2

  summary = {
      "status": "OK",
      "output_dir": str(out_dir.resolve()),
      "dataset_version": manifest["dataset_version"],
      "kind": manifest.get("kind"),
      "pixel_provenance": manifest.get("pixel_provenance"),
      "is_synthetic_smoke": bool(manifest.get("is_synthetic_smoke", synthetic_smoke)),
      "certifiable": bool(manifest.get("certifiable", not synthetic_smoke)),
      "split": manifest["split"],
      "total_episodes": manifest["total_episodes"],
      "total_steps": manifest["total_steps"],
      "total_bytes": manifest["total_bytes"],
      "image_hw": [hw, hw],
      "cameras": obs_desc["cameras"],
      "bytes_per_request_images_vla": int(obs_desc["bytes_per_request_images"]),
      "bytes_per_request_images_wm": int(hw * hw * int(obs_desc["channels"])),
      "admissible_wire_resolution": bool(hw >= 224 and int(hw * hw * int(obs_desc["channels"])) >= 150_528),
      "mean_temporal_frame_coherence": manifest["mean_temporal_frame_coherence"],
      "min_temporal_frame_coherence": manifest["min_temporal_frame_coherence"],
      "sha256_verified": True,
  }
  _print(json.dumps(summary, indent=2))
  return 0


def _cmd_generate_golden(args: argparse.Namespace) -> int:
  from eeb.cli import inject_checkpoint_kwarg, resolve_policy_spec

  log = (lambda msg: None) if getattr(args, "quiet", False) else (lambda msg: _print(f"[eeb data generate-golden] {msg}"))
  try:
    traces_opt = getattr(args, "traces", None)
    traces_pos = getattr(args, "traces_pos", None)
    if traces_opt is not None and traces_pos is not None and Path(traces_opt).resolve() != Path(traces_pos).resolve():
      raise ValueError(
          f"Conflicting traces arguments: positional {traces_pos!r} vs --traces {traces_opt!r}; supply only one."
      )
    effective_traces = traces_opt if traces_opt is not None else traces_pos

    raw_endpoint = getattr(args, "endpoint", None)
    raw_model = getattr(args, "model", None)
    if raw_endpoint and raw_model:
      raise ValueError("Pass either --endpoint/--server OR --model (with optional --checkpoint), not both.")
    if not raw_endpoint and not raw_model:
      raise ValueError("generate-golden requires either --endpoint/--server <url> or --model <pi05|gr00t|cosmos|module:Class>.")

    policy_kwargs_str = getattr(args, "policy_kwargs", None)
    if raw_model:
      policy_spec = resolve_policy_spec(str(raw_model))
      resolved_endpoint = f"inproc:{policy_spec}"
      if getattr(args, "checkpoint", None):
        parsed_kw = json.loads(policy_kwargs_str) if policy_kwargs_str else {}
        if not isinstance(parsed_kw, dict):
          raise ValueError("--policy-kwargs must decode to a JSON object.")
        parsed_kw = inject_checkpoint_kwarg(policy_spec, str(args.checkpoint), parsed_kw)
        policy_kwargs_str = json.dumps(parsed_kw)
    else:
      ep_s = str(raw_endpoint)
      if ep_s.startswith("inproc:"):
        resolved_endpoint = f"inproc:{resolve_policy_spec(ep_s[len('inproc:'):])}"
      else:
        resolved_endpoint = ep_s

    split_arg = str(getattr(args, "traces_split", "public") or "public")
    eff_split = "public" if split_arg == "auto" else split_arg
    hw = _parse_hw(getattr(args, "image_hw", None))
    summary = generate_golden_reference(
        endpoint=resolved_endpoint,
        out_path=args.out,
        traces=effective_traces,
        traces_split=eff_split,
        num_samples=args.num_samples,
        session_id=args.session_id,
        seed=args.seed,
        policy_kwargs=policy_kwargs_str,
        image_hw=hw,
        allow_smoke_golden=bool(getattr(args, "allow_smoke_golden", False)),
        log=log,
    )
  except (FileNotFoundError, ValueError, KeyError, ConnectionError, RuntimeError, json.JSONDecodeError) as e:
    _print(f"ERROR: {e}")
    return 2
  _print(json.dumps(summary, indent=2))
  return 0


# --------------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------------
def register_cli(subparsers) -> None:
  """Attach the ``data`` command group (download / pack-traces / inspect / perturb / generate-golden) to ``subparsers``."""
  p_data = subparsers.add_parser("data", help="EmbodiedX-Traces-v1 data tooling (LeRobot v2/v3 -> traces, G4 perturbations, golden references)")
  data_sub = p_data.add_subparsers(dest="data_command", required=True)
  add_data_commands(data_sub)


def add_data_commands(data_sub) -> None:
  """Add the four leaf commands to an ``argparse`` subparsers object."""
  p_dl = data_sub.add_parser("download", help="Download (part of) a LeRobot dataset from the Hugging Face Hub")
  p_dl.add_argument("dataset_id", help="Hub id, e.g. HuggingFaceVLA/libero")
  p_dl.add_argument("--revision", default=None)
  p_dl.add_argument("--cache-dir", "--out", dest="cache_dir", default=str(DEFAULT_CACHE_DIR), help="parent dir; dataset lands in <cache>/<org>__<name>/ (default ./data_cache)")
  p_dl.add_argument("--episodes", type=int, default=None, help="fetch meta/ + only the frame tables of the first N episodes")
  p_dl.add_argument("--videos", action="store_true", help="with --episodes, also fetch referenced video files (large!)")
  p_dl.add_argument("--allow-patterns", default=None, help="comma-separated glob patterns forwarded to snapshot_download")
  p_dl.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env / cached login)")
  p_dl.add_argument("--plan-only", metavar="LOCAL_ROOT", default=None, help="offline: print the files --episodes N would fetch for an already-downloaded meta/")
  p_dl.set_defaults(func=_cmd_download)

  p_pk = data_sub.add_parser("pack-traces", help="Pack a local LeRobot dataset into traces/<split>/episode_XXXX.npz + manifest.json")
  p_pk.add_argument("src", help="local LeRobot v2/v3 root (meta/info.json)")
  p_pk.add_argument("--out", required=True, help="output directory (traces/ is created inside)")
  p_pk.add_argument("--split", default=AUTO_SPLIT, choices=(AUTO_SPLIT,) + SPLITS, help="auto = seeded episode-hash split")
  p_pk.add_argument("--max-episodes", type=int, default=None)
  p_pk.add_argument("--episodes", default=None, help="comma-separated explicit source episode indices")
  p_pk.add_argument("--cameras", default=None, help="comma-separated short camera names (default: all)")
  p_pk.add_argument("--fps", type=float, default=None, help="target fps (integer divisor of native fps; stride subsampling)")
  p_pk.add_argument("--source-dataset", default=None, help="EMBODIEDX_TRACES_V1_SOURCES key, e.g. LIBERO (default: guessed)")
  p_pk.add_argument("--source-id", default=None, help="Hub id to record (default: inferred from directory name)")
  p_pk.add_argument("--source-revision", default=None)
  p_pk.add_argument("--license", default=None, help="override license recorded in the manifest")
  p_pk.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
  p_pk.add_argument("--hidden-fraction", type=float, default=DEFAULT_HIDDEN_FRACTION)
  p_pk.add_argument("--parquet", action="store_true", help="also write parquet sidecars (needs pyarrow)")
  p_pk.add_argument("--no-images", action="store_true", help="pack proprio/action/timestamps only (metadata smoke tests)")
  p_pk.add_argument("--allow-instruction-change", action="store_true", help="keep the first instruction if it changes mid-episode")
  p_pk.add_argument("--state-key", default=None)
  p_pk.add_argument("--action-key", default=None)
  p_pk.add_argument("--image-dir", default=None, help="root holding pre-decoded images/<key>/episode_XXXXXX/frame_XXXXXX.png")
  p_pk.set_defaults(func=_cmd_pack_traces)

  p_in = data_sub.add_parser("inspect", help="Summarise a LeRobot root, verify a packed traces dir, or probe a Hub dataset layout")
  p_in.add_argument("path", nargs="?", default=".", help="LeRobot root, <out_dir>, traces/, traces/<split>, or perturbed/<split>")
  p_in.add_argument("--hub", default=None, help="probe https://huggingface.co/api/datasets/<id>/tree/main instead of a local path")
  p_in.add_argument("--revision", default=None)
  p_in.add_argument("--deep", action="store_true", help="also recompute content hashes (loads every trace)")
  p_in.add_argument("--limit", type=int, default=5, help="episodes to list")
  p_in.set_defaults(func=_cmd_inspect)

  p_pt = data_sub.add_parser("perturb", help="Materialise seeded G4 perturbations (‖δ‖∞ = eps) for traces/<split>")
  p_pt.add_argument("traces", help="<out_dir> of pack-traces, traces/ dir, or flat EmbodiedX shard bundle dir")
  p_pt.add_argument("--out", default=None, help="output directory (perturbed/<split>/ is created inside; default: <traces>)")
  p_pt.add_argument("--split", default="public", choices=SPLITS)
  p_pt.add_argument("--max-episodes", type=int, default=None)
  p_pt.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
  p_pt.add_argument("--eps", type=float, default=1e-3, help="‖δ‖∞ in normalized units (<= 1/255)")
  p_pt.add_argument("--image-space", default="unit_float32", choices=IMAGE_SPACES)
  p_pt.add_argument("--camera-jitter-px", type=int, default=0)
  p_pt.add_argument("--brightness-jitter", type=float, default=0.0)
  p_pt.add_argument("--no-images", action="store_true", help="do not perturb images")
  p_pt.add_argument("--no-proprio", action="store_true", help="do not perturb proprio")
  p_pt.add_argument("--quiet", action="store_true", help="suppress per-episode progress output")
  p_pt.set_defaults(func=_cmd_perturb)

  p_gg = data_sub.add_parser(
      "generate-golden",
      help="Generate a golden-reference .npz from a baseline endpoint for golden_replay_gate() (VLA_TRACK or WM_TRACK)",
  )
  p_gg.add_argument(
      "traces_pos",
      nargs="?",
      default=None,
      help="Optional positional EmbodiedX-Traces-v1 directory (alias for --traces)",
  )
  p_gg.add_argument(
      "--endpoint",
      "--server",
      dest="endpoint",
      default=None,
      help="Baseline server endpoint: http://host:port | inproc:<module:Class> | unix:///path.sock | tcp://host:port",
  )
  p_gg.add_argument(
      "--model",
      "--policy",
      dest="model",
      default=None,
      help="Baseline policy shorthand (pi05, gr00t, cosmos, dummy-vla, dummy-wm) or '<module>:<Class>' when --endpoint is omitted",
  )
  p_gg.add_argument(
      "--checkpoint",
      default=None,
      help="Optional local checkpoint directory or pinned Hugging Face repo ID when using --model",
  )
  p_gg.add_argument(
      "--traces",
      default=None,
      help="Optional EmbodiedX-Traces-v1 directory (or shard directory); defaults to SyntheticObservationSource",
  )
  p_gg.add_argument(
      "--traces-split",
      "--split",
      dest="traces_split",
      default="public",
      choices=("auto",) + SPLITS,
      help="Split to load when traces points to a multi-split bundle root (default: public)",
  )
  p_gg.add_argument(
      "--num-samples",
      type=int,
      default=100,
      help="Number of observations to record into the golden .npz (protocol minimum: 100; default: 100)",
  )
  p_gg.add_argument(
      "--allow-smoke-golden",
      action="store_true",
      help="Allow fewer than 100 golden samples (stamps allow_smoke_golden=True; non-admissible for leaderboard)",
  )
  p_gg.add_argument(
      "--out",
      "--output",
      dest="out",
      required=True,
      help="Output path for the generated golden .npz file",
  )
  p_gg.add_argument(
      "--session-id",
      default="golden_ref",
      help="Session ID passed to client.reset() and predict/rollout (default: golden_ref)",
  )
  p_gg.add_argument(
      "--seed",
      type=int,
      default=7,
      help="Deterministic seed passed to client.reset(session_id, seed) (default: 7)",
  )
  p_gg.add_argument(
      "--policy-kwargs",
      default=None,
      help="Optional JSON object of constructor kwargs when --endpoint is inproc:<module:Class>",
  )
  p_gg.add_argument(
      "--image-hw",
      default=None,
      help="Optional synthetic observation image size HxW (e.g. 224x224) when --traces is omitted",
  )
  p_gg.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress progress messages on stdout (still prints JSON summary)",
  )
  p_gg.set_defaults(func=_cmd_generate_golden)

  p_pt_eval = data_sub.add_parser(
      "prepare-traces",
      aliases=["generate-eval-traces"],
      help=(
          "Download and materialise real pinned LeRobot EmbodiedX-Traces-v1 shards "
          "(224x224 for pi0.5; 256x256 for GR00T-N1.7 / Cosmos-Predict2.5) for hardware sweeps"
      ),
  )
  p_pt_eval.add_argument(
      "--out",
      "--output-dir",
      dest="out",
      required=True,
      help="Output directory for generated native-resolution shards and manifest.json",
  )
  p_pt_eval.add_argument(
      "--track",
      default="all",
      choices=[
          "all",
          "vla",
          "wm",
          "track1a",
          "track1b",
          "track2",
          "track1_vla",
          "track2_wm",
          "VLA_TRACK",
          "WM_TRACK",
          "TRACK1A",
          "TRACK1B",
          "TRACK2",
      ],
      help="Target track for trace bundle preparation (default: all; 'vla'/'track1a' -> 224x224 LIBERO; 'track1b' -> 256x256 LIBERO; 'wm'/'track2' -> 256x256 DROID)",
  )
  p_pt_eval.add_argument(
      "--cache-dir",
      default=None,
      help="Directory for cached Hugging Face LeRobot parquet/mp4 downloads (default: ./data_cache or $EEB_LEROBOT_CACHE_DIR)",
  )
  p_pt_eval.add_argument(
      "--token",
      default=None,
      help="Optional Hugging Face auth token",
  )
  p_pt_eval.add_argument(
      "--synthetic-ci-smoke",
      action="store_true",
      help="Generate explicit offline non-certifiable SYNTHETIC_CI_SMOKE_TRACES instead of real LeRobot camera frames",
  )
  p_pt_eval.add_argument(
      "--image-hw",
      default=None,
      help="Native camera resolution HxW (default: 224x224; use 256x256 for GR00T-N1.7-3B and Cosmos-Predict2.5-2B)",
  )
  p_pt_eval.add_argument(
      "--resolution-hw",
      type=int,
      default=None,
      help="Square RGB side length (e.g. 224 or 256; alias for --image-hw NxN)",
  )
  p_pt_eval.add_argument(
      "--split",
      default="public",
      help="Dataset split tag written into manifest.json (default: public)",
  )
  p_pt_eval.add_argument(
      "--episodes",
      "--max-episodes",
      dest="episodes",
      type=int,
      default=None,
      help="Number of canonical episodes to generate (default: all 8 episodes across LIBERO/DROID/BridgeData/OXE)",
  )
  p_pt_eval.add_argument(
      "--steps",
      "--num-steps",
      dest="steps",
      type=int,
      default=16,
      help="Steps T per episode shard (default: 16)",
  )
  p_pt_eval.add_argument(
      "--seed",
      type=int,
      default=20261008,
      help="Deterministic RNG base seed (default: 20261008)",
  )
  p_pt_eval.add_argument(
      "--quiet",
      action="store_true",
      help="Suppress progress messages on stdout (still prints JSON summary)",
  )
  p_pt_eval.set_defaults(func=_cmd_prepare_traces)


def build_arg_parser() -> argparse.ArgumentParser:
  """Standalone parser: ``python -m eeb.data <command> ...`` mirrors ``eeb data <command> ...``."""
  parser = argparse.ArgumentParser(prog="eeb data", description="EmbodiedX-Traces-v1 data tooling (standalone entry point)")
  subparsers = parser.add_subparsers(dest="data_command", required=True)
  add_data_commands(subparsers)
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  args = build_arg_parser().parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
  sys.exit(main())
