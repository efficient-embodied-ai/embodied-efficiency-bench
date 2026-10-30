"""`eeb` command-line interface (WS-A / M1).

    eeb serve      --policy <module:Class> [--uds PATH | --tcp HOST:PORT] [--policy-kwargs JSON]
    eeb measure    --recipe R --endpoint unix:///...|inproc:<module:Class> --sweep 1,2,4,8,16
                   --warmup 20 --calls 320 --runs 3 --out results/
    eeb compare    --baseline runs/champion --candidate runs/cand_01 [--cutoff-hz F] [--min-gain-pct 2.0]
    eeb telemetry  probe [--seconds 1] [--hz 10] [--source nvml|tegrastats|amd_smi|tpu_info]
    eeb envelope   check [--sku SKU] [--allow-no-device]
    eeb data ...   (registered lazily from eeb.data.cli.register_cli when WS-C is present)
    eeb <legacy>   eval-libero-rt | eval-wm-ranking | run-stage1 | run-stage2 | compute-pareto |
                   verify-receipt  -> forwarded to embodied_bench.cli

Exit codes: 0 success (a report with status INVALID is still a successful measurement unless
`--strict` is given), 1 usage / runtime error, 2 strict-mode admissibility failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

LEGACY_SUBCOMMANDS = (
    "eval-libero-rt",
    "eval-wm-ranking",
    "run-stage1",
    "run-stage2",
    "compute-pareto",
    "pareto",
    "verify-receipt",
    "verify-bundle",
    "prepare-commit-bundle",
    "validate-submission",
    "download-data",
    "benchmark",
    "export-website",
    "build-leaderboard",
    "agent-kickoff",
)


def _parse_kwargs(raw: Optional[str]) -> Dict[str, Any]:
  if not raw:
    return {}
  try:
    data = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise SystemExit(f"--policy-kwargs must be a JSON object, got {raw!r}: {exc}") from exc
  if not isinstance(data, dict):
    raise SystemExit("--policy-kwargs must decode to a JSON object (e.g. '{\"compute_delay_ms\": 20}').")
  return data


def _parse_int_list(raw: str, flag: str) -> List[int]:
  try:
    vals = [int(tok) for tok in raw.replace(" ", "").split(",") if tok]
  except ValueError as exc:
    raise SystemExit(f"{flag} expects a comma-separated list of ints, got {raw!r}.") from exc
  if not vals or any(v <= 0 for v in vals):
    raise SystemExit(f"{flag} needs positive ints, got {raw!r}.")
  return vals


def _parse_hw(raw: Optional[str]):
  if not raw:
    return None
  try:
    h, w = raw.lower().split("x")
    return (int(h), int(w))
  except ValueError as exc:
    raise SystemExit(f"--image-hw expects HxW like 224x224, got {raw!r}.") from exc


# ----------------------------------------------------------------------------- serve

_POLICY_MODEL_SHORTHANDS: Dict[str, str] = {
    "pi05": "eeb.servers.pi05:Pi05Policy",
    "pi0.5": "eeb.servers.pi05:Pi05Policy",
    "pi05-3b": "eeb.servers.pi05:Pi05Policy",
    "lerobot/pi05_libero_finetuned_v044": "eeb.servers.pi05:Pi05Policy",
    "physicalintelligence/pi0.5-libero": "eeb.servers.pi05:Pi05Policy",
    "gr00t": "eeb.servers.gr00t:Gr00tN17Policy",
    "gr00t-n1.7": "eeb.servers.gr00t:Gr00tN17Policy",
    "gr00t-n1.7-3b": "eeb.servers.gr00t:Gr00tN17Policy",
    "nvidia/gr00t-n1.7-libero": "eeb.servers.gr00t:Gr00tN17Policy",
    "nvidia/gr00t-n1.7-3b": "eeb.servers.gr00t:Gr00tN17Policy",
    "cosmos": "eeb.servers.cosmos_predict25:CosmosPredict25Policy",
    "cosmos-predict2.5": "eeb.servers.cosmos_predict25:CosmosPredict25Policy",
    "cosmos-predict2.5-2b": "eeb.servers.cosmos_predict25:CosmosPredict25Policy",
    "nvidia/cosmos-predict2.5-2b": "eeb.servers.cosmos_predict25:CosmosPredict25Policy",
    "dummy": "eeb.servers.dummy:DummyVLAPolicy",
    "dummy-vla": "eeb.servers.dummy:DummyVLAPolicy",
    "dummy-wm": "eeb.servers.dummy:DummyWMPolicy",
}


def resolve_policy_spec(raw_spec: str) -> str:
  key = str(raw_spec).strip()
  if key.lower().startswith("inproc:"):
    key = key[len("inproc:") :].strip()
  return _POLICY_MODEL_SHORTHANDS.get(key.lower(), key)


def inject_checkpoint_kwarg(
    policy_spec: str,
    checkpoint: Optional[str],
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Map CLI ``--checkpoint`` to the policy class constructor's checkpoint parameter name."""
  import inspect

  from eeb.protocol.policy import load_policy_class

  out: Dict[str, Any] = dict(kwargs or {})
  if not checkpoint:
    return out
  explicit_keys = ("checkpoint", "checkpoint_dir", "model_path")
  if any(k in out for k in explicit_keys):
    return out
  cls = load_policy_class(policy_spec)
  params = inspect.signature(cls.__init__).parameters
  for candidate in explicit_keys:
    if candidate in params:
      out[candidate] = str(checkpoint)
      return out
  if "model_id" in params and cls.__module__ != "eeb.servers.dummy":
    out.setdefault("model_id", str(checkpoint))
    return out
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
    out["checkpoint"] = str(checkpoint)
  return out


def cmd_serve(args: argparse.Namespace) -> int:
  from eeb.protocol.policy import instantiate_policy
  from eeb.servers.base import normalize_and_validate_uds_path, serve_policy

  raw_policy = getattr(args, "policy", None) or getattr(args, "model", None)
  if not raw_policy:
    raise SystemExit("eeb serve: pass --model <pi05|gr00t|cosmos|module:Class> (or --policy <module:Class>).")
  if getattr(args, "policy", None) and getattr(args, "model", None):
    if resolve_policy_spec(args.policy) != resolve_policy_spec(args.model):
      raise SystemExit(f"eeb serve: conflicting --policy {args.policy!r} and --model {args.model!r}.")
  policy_spec = resolve_policy_spec(raw_policy)

  transport_count = (
      int(bool(getattr(args, "uds", None)))
      + int(bool(getattr(args, "tcp", None)))
      + int(bool(getattr(args, "http", None)))
      + int(getattr(args, "port", None) is not None)
  )
  if transport_count != 1:
    raise SystemExit(
        "eeb serve: pass exactly one transport target: --port PORT [--host HOST] [--grpc], "
        "--uds PATH, --tcp HOST:PORT, or --http HOST:PORT."
    )
  if getattr(args, "uds", None):
    uds_path = normalize_and_validate_uds_path(args.uds)
    endpoint = f"unix://{uds_path}"
  elif getattr(args, "tcp", None):
    endpoint = f"tcp://{args.tcp}"
  elif getattr(args, "http", None):
    h_raw = str(args.http).strip()
    endpoint = h_raw if h_raw.startswith(("http://", "https://")) else f"http://{h_raw}"
  else:
    host = str(getattr(args, "host", None) or "127.0.0.1").strip()
    port = int(args.port)
    if port < 0 or port > 65535:
      raise SystemExit(f"eeb serve: --port must be in 0..65535, got {port}.")
    scheme = "tcp" if getattr(args, "grpc", False) else "http"
    endpoint = f"{scheme}://{host}:{port}"

  kwargs = inject_checkpoint_kwarg(policy_spec, getattr(args, "checkpoint", None), _parse_kwargs(args.policy_kwargs))

  policy = instantiate_policy(policy_spec, kwargs)
  caps = policy.capabilities()
  server = serve_policy(policy, endpoint, max_workers=args.max_workers, serialize_calls=args.serialize)
  if hasattr(server, "http_url"):
    bound = str(server.http_url)
  elif endpoint.startswith("tcp://") and endpoint.endswith(":0"):
    bound = f"tcp://{endpoint[6:].rsplit(':', 1)[0]}:{server.eeb_bound_port}"
  else:
    bound = endpoint
  print(
      f"eeb serve: {policy_spec} server_impl={caps.server_impl} track={caps.track} horizon={caps.horizon} "
      f"dt_ms={caps.dt_ms} listening on {bound}",
      flush=True,
  )
  print("EEB_SERVER_READY", flush=True)
  stop = {"flag": False}

  def _handler(signum: int, _frame: Any) -> None:
    stop["flag"] = True

  signal.signal(signal.SIGINT, _handler)
  signal.signal(signal.SIGTERM, _handler)
  deadline = time.time() + args.serve_seconds if args.serve_seconds and args.serve_seconds > 0 else None
  try:
    while not stop["flag"]:
      if deadline is not None and time.time() >= deadline:
        break
      time.sleep(0.2)
  finally:
    server.stop(grace=2.0)
    close = getattr(policy, "close", None)
    if callable(close):
      close()
    print("eeb serve: stopped", flush=True)
  return 0


# ----------------------------------------------------------------------------- measure


def cmd_measure(args: argparse.Namespace) -> int:
  from embodied_bench.schema import RecipeConfig, load_yaml_or_json, resolve_hardware_sku

  from eeb.envelope.check import check_envelope
  from eeb.loadgen.client import open_client
  from eeb.loadgen.gate import load_golden_npz, run_gate
  from eeb.loadgen.report import ReportInputs, build_report, report_to_markdown_summary, validate_report, write_report_bundle
  from eeb.loadgen.sweep import ConcurrencySweep, ObservationSource, ObservationSpec, SweepConfig, TraceBundleObservationSource
  from eeb.protocol.policy import instantiate_policy
  from eeb.servers.base import parse_endpoint
  from eeb.telemetry.sidecar import TelemetrySidecar

  log = (lambda msg: None) if args.quiet else (lambda msg: print(f"[eeb measure] {msg}", flush=True))

  track_hint_raw = getattr(args, "track", None)
  track_hint = None
  track_subpool = None
  if track_hint_raw:
    th_low = str(track_hint_raw).strip().lower()
    if th_low in ("wm", "wm_track", "track2", "track2_wm"):
      track_hint = "WM_TRACK"
      track_subpool = "track2"
    else:
      track_hint = "VLA_TRACK"
      if th_low in ("track1a", "track1b"):
        track_subpool = th_low
  div_hint_raw = getattr(args, "division", None)
  div_hint = str(div_hint_raw).strip().upper() if div_hint_raw else None

  recipe_cfg: Optional[RecipeConfig] = None
  recipe_dict: Optional[Dict[str, Any]] = None
  if args.recipe:
    recipe_cfg = RecipeConfig.from_dict(load_yaml_or_json(args.recipe))
    recipe_dict = recipe_cfg.to_dict()
    sku = resolve_hardware_sku(args.sku) if args.sku else recipe_cfg.hardware_sku
    if track_hint and recipe_cfg.track != track_hint:
      raise SystemExit(f"eeb measure: --track {track_hint!r} conflicts with recipe track {recipe_cfg.track!r}.")
    if div_hint and str(recipe_cfg.division).upper() != div_hint:
      raise SystemExit(f"eeb measure: --division {div_hint!r} conflicts with recipe division {recipe_cfg.division!r}.")
  else:
    sku = resolve_hardware_sku(args.sku) if args.sku else None

  if args.sweep is not None:
    sweep_levels = _parse_int_list(args.sweep, "--sweep")
  elif recipe_cfg is not None and (recipe_cfg.concurrency_levels or recipe_cfg.batch_sizes):
    raw_levels = (
        recipe_cfg.concurrency_levels
        if recipe_cfg.track == "VLA_TRACK" and recipe_cfg.concurrency_levels
        else (recipe_cfg.batch_sizes or recipe_cfg.concurrency_levels)
    )
    sweep_levels = [int(c) for c in raw_levels]
  else:
    sweep_levels = [1, 2, 4, 8] if track_hint == "WM_TRACK" else [1, 2, 4, 8, 16]

  effective_execute_horizon = args.execute_horizon
  if effective_execute_horizon is None and recipe_cfg is not None:
    if recipe_cfg.execute_horizon is not None:
      effective_execute_horizon = int(recipe_cfg.execute_horizon)
    elif recipe_cfg.track == "WM_TRACK" and recipe_cfg.horizon_frames:
      effective_execute_horizon = int(recipe_cfg.horizon_frames)

  ep = parse_endpoint(args.endpoint)
  inproc_target = resolve_policy_spec(ep.target) if ep.is_inproc else ep.target
  policy = instantiate_policy(inproc_target, _parse_kwargs(args.policy_kwargs)) if ep.is_inproc else None
  resolved_endpoint = f"inproc:{inproc_target}" if ep.is_inproc else args.endpoint

  is_wm_for_calls = (
      (recipe_cfg is not None and recipe_cfg.track == "WM_TRACK")
      or track_hint == "WM_TRACK"
      or (policy is not None and policy.capabilities().track == "WM_TRACK")
  )
  effective_calls = int(args.calls) if args.calls is not None else (200 if is_wm_for_calls else 320)

  config = SweepConfig(
      concurrency=sweep_levels,
      warmup_calls=args.warmup,
      timed_calls=effective_calls,
      runs=args.runs,
      max_variance_pct=args.max_variance_pct,
      max_p99_variance_pct=getattr(args, "max_p99_variance_pct", None),
      pacing_enabled=bool(getattr(args, "paced", False) and not getattr(args, "no_pacing", False)),
      execute_horizon=effective_execute_horizon,
      dt_ms_override=args.dt_ms,
      call_timeout_s=args.call_timeout_s,
      keep_samples=not args.no_samples,
  )

  log("envelope check ...")
  allow_no_dev = bool(args.allow_no_device or (policy is not None and policy.capabilities().is_synthetic))
  envelope = check_envelope(hardware_sku=sku, allow_no_device=allow_no_dev)
  log(f"envelope: passed={envelope.passed} num_devices={envelope.num_devices} reason={envelope.reason}")

  sidecar = TelemetrySidecar.auto(hz=args.telemetry_hz, device_index=args.device_index, prefer=None if args.telemetry_source in (None, "auto") else args.telemetry_source)
  log(f"telemetry source: {sidecar.source.describe()}")
  telemetry_by_run: Dict[Any, Any] = {}

  def on_run_end(c: int, run_idx: int, res: Any) -> None:
    # Give the sampler a chance to record the last sample inside the window.
    time.sleep(min(0.15, 1.5 / args.telemetry_hz))
    telemetry_by_run[(c, run_idx)] = sidecar.summarize(res.window_start_wall, res.window_end_wall)

  obs_source = None
  hw = _parse_hw(args.image_hw)
  if args.traces:
    obs_source = TraceBundleObservationSource(args.traces, split=args.traces_split)
    log(f"observation source: {obs_source.describe()}")

  golden = load_golden_npz(args.golden) if args.golden else None
  sweep = ConcurrencySweep(
      resolved_endpoint,
      config,
      policy=policy,
      observation_source=obs_source,
      golden=golden,
      on_run_end=on_run_end,
      log=log,
  )
  if hw is not None and obs_source is None:
    with sweep._open() as probe:  # noqa: SLF001 - need caps to size the synthetic observations
      caps_probe = probe.capabilities()
    sweep._observation_source = ObservationSource(ObservationSpec.from_caps(caps_probe, image_hw=hw))  # noqa: SLF001

  sidecar.start()
  try:
    result = sweep.run()
  finally:
    sidecar.stop()

  if track_hint and result.caps.track != track_hint:
    raise SystemExit(
        f"eeb measure: --track {track_hint!r} does not match server capability track {result.caps.track!r}."
    )
  if track_subpool in ("track1a", "track1b") and not result.caps.is_synthetic:
    mid_low = str(result.caps.model_id or "").lower()
    if track_subpool == "track1a" and "gr00t" in mid_low:
      raise SystemExit(
          f"eeb measure: --track track1a expects pi0.5 (VLA Track 1A), but server reported model_id={result.caps.model_id!r}."
      )
    if track_subpool == "track1b" and "pi05" in mid_low:
      raise SystemExit(
          f"eeb measure: --track track1b expects GR00T N1.7-3B (VLA Track 1B), but server reported model_id={result.caps.model_id!r}."
      )
  if recipe_dict is None and (track_hint or div_hint):
    from pathlib import Path as _Path

    inferred_id = f"adhoc_{_Path(args.out).name or result.caps.model_id}".replace(" ", "_")
    recipe_dict = {
        "_adhoc_cli": True,
        "recipe_id": inferred_id,
        "track": track_hint or result.caps.track,
        "division": div_hint or "CLOSED",
        "hardware_sku": sku or "",
        "baseline_model_id": result.caps.model_id,
        "model_name": result.caps.model_id,
        "precision": result.caps.precision or "BF16",
    }

  if not envelope.passed and envelope.num_devices == 0 and result.caps.is_synthetic:
    envelope = check_envelope(hardware_sku=sku, allow_no_device=True)

  gates = {}
  gate_client = open_client(resolved_endpoint, policy=policy)
  try:
    for point in result.points:
      exec_id = f"{(recipe_dict or {}).get('recipe_id', 'adhoc')}-C{point.concurrency}-{int(result.started_at_wall)}"
      gates[point.concurrency] = run_gate(point, result.caps, exec_id, client=gate_client, golden=golden)
  finally:
    gate_client.close()

  report = build_report(
      ReportInputs(
          sweep=result,
          recipe=recipe_dict,
          recipe_path=args.recipe,
          envelope=envelope,
          telemetry_by_run=telemetry_by_run,
          telemetry_source=sidecar.source.describe(),
          gates=gates,
          allow_sanity_only=bool(args.allow_sanity_only),
          assumed_power_w=args.assume_power_watts,
          notes=list(args.note or []),
      )
  )
  problems = validate_report(report)
  if problems:
    raise SystemExit("internal error: generated report failed validation: " + "; ".join(problems))
  paths = write_report_bundle(report, args.out)
  if args.json:
    print(json.dumps({"status": report["status"], "reason": report["reason"], **paths}, indent=2))
  else:
    print(report_to_markdown_summary(report))
    print(f"report:  {paths['report']}\nreceipt: {paths['receipt']} (sha256 {paths['attestation_sha256'][:16]}...)")
  if args.strict:
    if not report.get("quality_gate", {}).get("passed", False):
      return 2
    if not report.get("quality_gate", {}).get("quality_certified", False) and not args.allow_sanity_only:
      return 2
    if report["status"] != "PENDING":
      if result.caps.is_synthetic and report.get("quality_gate", {}).get("quality_certified", False):
        hard_failures = [
            f
            for f in report.get("admissibility_findings", [])
            if not f.startswith("policy server_impl=")
            and not f.startswith("envelope check ")
            and not f.startswith("sweep below protocol minimums:")
            and not f.startswith("observation wire resolution below protocol minimum:")
        ]
        if hard_failures:
          return 2
      else:
        return 2
  return 0


# ----------------------------------------------------------------------------- telemetry / envelope


def cmd_telemetry_probe(args: argparse.Namespace) -> int:
  from eeb.telemetry.sidecar import TelemetrySidecar, detect_sources

  sources = detect_sources(args.device_index)
  sidecar = TelemetrySidecar.auto(hz=args.hz, device_index=args.device_index, prefer=None if args.source in (None, "auto") else args.source)
  sidecar.start()
  t0 = time.time()
  time.sleep(max(0.0, args.seconds))
  t1 = time.time()
  sidecar.stop()
  summary = sidecar.summarize(t0, t1)
  out = {
      "sources": [s.describe() for s in sources],
      "selected": sidecar.source.describe(),
      "latest_sample": sidecar.latest(),
      "window_summary": summary.to_dict(),
  }
  print(json.dumps(out, indent=2, default=str))
  return 0


def cmd_envelope_check(args: argparse.Namespace) -> int:
  from embodied_bench.schema import resolve_hardware_sku
  from eeb.envelope.check import check_envelope

  resolved_sku = resolve_hardware_sku(args.sku) if args.sku else None
  res = check_envelope(hardware_sku=resolved_sku, allow_no_device=args.allow_no_device)
  if args.json:
    print(json.dumps(res.to_dict(), indent=2))
  else:
    print(f"envelope: passed={res.passed} num_devices={res.num_devices}")
    print(f"reason:   {res.reason}")
    if res.device:
      print(f"device:   {res.device['name']} ({res.device['vendor']}, {res.measured_total_memory_mb} MB, via {res.device['source']})")
    if res.spec:
      print(f"spec:     {res.hardware_sku}: {res.spec}")
  return 0 if res.passed else 1


# ----------------------------------------------------------------------------- compare


def cmd_compare(args: argparse.Namespace) -> int:
  from eeb.loadgen.compare import compare_reports, format_comparison_markdown, write_comparison_json

  result = compare_reports(
      baseline_path=args.baseline,
      candidate_path=args.candidate,
      cutoff_hz=args.cutoff_hz,
      min_gain_pct=args.min_gain_pct,
      allow_sanity_only=bool(args.allow_sanity_only),
  )
  if args.out:
    out_file = write_comparison_json(result, args.out)
    result["output"] = str(out_file)
  if args.json:
    print(json.dumps(result, indent=2, sort_keys=True))
  else:
    print(format_comparison_markdown(result))
  return int(result["exit_code"])


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
  epilog = (
      "Submission, evaluation & leaderboard subcommands (run `eeb <cmd> --help` for details):\n"
      "  prepare-commit-bundle  package a measured run into a Phase-1 two-file PR directory\n"
      "  verify-bundle          verify a Phase-1 commit bundle or full run directory (--strict)\n"
      "  verify-receipt         verify SHA-256 signature & bundle integrity of receipt.json / report.json\n"
      "  validate-submission    validate submission recipe + measured bundle admissibility pre-PR\n"
      "  compute-pareto         aggregate report.json files into Pareto frontiers & Iso-Reactivity slices\n"
      "  build-leaderboard      verify submission receipts and compile leaderboard JSON + Markdown\n"
      "  export-website         export compiled leaderboard JSON for website visualization\n"
      "  eval-libero-rt         run latency-coupled LIBERO-RT closed-loop delay evaluation\n"
      "  eval-wm-ranking        run Track 2 world-model policy-ranking correlation evaluation\n"
      "  download-data          download or synthesize EmbodiedX-Traces-v1 episode bundles\n"
      "  agent-kickoff          generate an optimization study blueprint for a recipe"
  )
  parser = argparse.ArgumentParser(
      prog="eeb",
      description="EmbodiedX / embodied-efficiency-bench harness CLI",
      epilog=epilog,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  sub = parser.add_subparsers(dest="command")

  p_serve = sub.add_parser("serve", help="serve a Policy over HTTP (--port / --http) or gRPC (--uds / --tcp)")
  p_serve.add_argument("--policy", default=None, help="policy spec '<module>:<Class>' or shorthand (pi05, gr00t, cosmos, dummy)")
  p_serve.add_argument("--model", default=None, help="model shorthand (pi05, gr00t, cosmos, dummy) or '<module>:<Class>'")
  p_serve.add_argument("--checkpoint", default=None, help="optional local checkpoint directory or pinned Hugging Face repo ID")
  p_serve.add_argument("--uds", "--unix", dest="uds", default=None, help="Unix-domain gRPC socket path (<= 107 bytes, e.g. /tmp/eeb.sock)")
  p_serve.add_argument("--tcp", default=None, help="gRPC host:port, e.g. 127.0.0.1:50051")
  p_serve.add_argument("--http", default=None, help="HTTP host:port, e.g. 127.0.0.1:8000")
  p_serve.add_argument("--host", default="127.0.0.1", help="bind host when using --port (default: 127.0.0.1)")
  p_serve.add_argument("--port", type=int, default=None, help="listen TCP port (serves HTTP by default; pass --grpc for gRPC)")
  p_serve.add_argument("--grpc", action="store_true", help="serve gRPC instead of HTTP when --port is specified")
  p_serve.add_argument("--policy-kwargs", help="JSON object of constructor kwargs")
  p_serve.add_argument("--max-workers", type=int, default=16)
  p_serve.add_argument("--serialize", action="store_true", help="serialize predict/rollout calls on one lock (non-thread-safe policies)")
  p_serve.add_argument("--serve-seconds", type=float, default=0.0, help="auto-stop after N seconds (tests / CI); 0 = run until SIGINT")
  p_serve.set_defaults(func=cmd_serve)

  p_meas = sub.add_parser("measure", help="run the LoadGen concurrency sweep and write report + receipt")
  p_meas.add_argument("--recipe", help="recipe YAML/JSON (embodied_bench.schema.RecipeConfig)")
  p_meas.add_argument(
      "--track",
      default=None,
      choices=[
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
      help="track selector when running without --recipe (e.g. track1a, track1b, track2, vla, wm)",
  )
  p_meas.add_argument("--division", default=None, choices=["closed", "open", "CLOSED", "OPEN"], help="division selector when running without --recipe")
  p_meas.add_argument(
      "--endpoint",
      "--server",
      "--server-url",
      dest="endpoint",
      required=True,
      help="http://host:port | unix:///path.sock | tcp://host:port | inproc:<module:Class>",
  )
  p_meas.add_argument("--sweep", default=None, help="comma-separated concurrency levels (default: recipe concurrency_levels/batch_sizes or 1,2,4,8,16)")
  p_meas.add_argument("--warmup", type=int, default=30, help="warm-up calls per run (protocol minimum: 30)")
  p_meas.add_argument(
      "--calls",
      "--timed",
      dest="calls",
      type=int,
      default=None,
      help="timed calls per run, shared by all sessions (protocol: 320 VLA_TRACK = >=20/slot at C=16; 200 WM_TRACK)",
  )
  p_meas.add_argument("--runs", type=int, default=3, help="runs per concurrency level (protocol: 3)")
  p_meas.add_argument("--out", "--output", "--output-dir", dest="out", required=True, help="output directory for report.json + receipt.json")
  p_meas.add_argument("--max-variance-pct", type=float, default=5.0, help="maximum allowed cross-run spread %% on p50 latency and chunks/s (default: 5.0)")
  p_meas.add_argument("--max-p99-variance-pct", type=float, default=None, help="maximum allowed cross-run spread %% on p99 tail latency (default: max(8.0, max_variance_pct))")
  p_meas.add_argument("--policy-kwargs", help="JSON kwargs for inproc:<module:Class> endpoints")
  p_meas.add_argument("--paced", action="store_true", help="opt into closed-loop wall-clock sleep pacing up to T_exec = K_exec * dt_ms (default: unpaced continuous queuing)")
  p_meas.add_argument("--no-pacing", action="store_true", help="force unpaced continuous queuing (default; overrides --paced)")
  p_meas.add_argument("--execute-horizon", type=int, default=None, help="actions executed per chunk before the next request (default: recipe or server caps)")
  p_meas.add_argument("--dt-ms", type=float, default=None, help="override control period reported by the server")
  p_meas.add_argument("--image-hw", default=None, help="synthetic observation image size HxW (default: server caps or 224x224)")
  p_meas.add_argument("--traces", default=None, help="EmbodiedX-Traces-v1 bundle directory (or traces/<split>) for real recorded observations")
  p_meas.add_argument("--traces-split", default="public", choices=["public", "hidden"], help="split to load when --traces points to a bundle root")
  p_meas.add_argument("--golden", default=None, help="golden NPZ for a GOLDEN_REPLAY gate in the same execution")
  p_meas.add_argument("--allow-sanity-only", action="store_true", help="allow SANITY_ONLY gate (without --golden) in --strict mode")
  p_meas.add_argument("--sku", "--hardware", "--hardware-sku", dest="sku", default=None, help="hardware SKU (e.g. L40S_48GB, TPU_V6E_1_32GB)")
  p_meas.add_argument("--allow-no-device", action="store_true", help="accept a CPU-only host (harness self-test; report stays INVALID)")
  p_meas.add_argument("--assume-power-watts", type=float, default=None, help="operator-supplied power when telemetry is unavailable (labelled ESTIMATED)")
  p_meas.add_argument("--telemetry-source", default="auto", choices=["auto", "nvml", "tegrastats", "amd_smi", "tpu_info"])
  p_meas.add_argument("--telemetry-hz", type=float, default=10.0)
  p_meas.add_argument("--device-index", type=int, default=0)
  p_meas.add_argument("--call-timeout-s", type=float, default=60.0)
  p_meas.add_argument("--no-samples", action="store_true", help="omit per-call latency samples from the report")
  p_meas.add_argument("--note", action="append", help="free-text note appended to the report (repeatable)")
  p_meas.add_argument("--strict", action="store_true", help="exit 2 unless the report status is PENDING (admissible)")
  p_meas.add_argument("--json", action="store_true", help="print a JSON summary instead of the table")
  p_meas.add_argument("--quiet", action="store_true")
  p_meas.set_defaults(func=cmd_measure)

  p_tel = sub.add_parser("telemetry", help="telemetry utilities")
  tel_sub = p_tel.add_subparsers(dest="telemetry_command")
  p_probe = tel_sub.add_parser("probe", help="detect sources, sample for a few seconds, print provenance")
  p_probe.add_argument("--seconds", type=float, default=1.0)
  p_probe.add_argument("--hz", type=float, default=10.0)
  p_probe.add_argument("--source", default="auto", choices=["auto", "nvml", "tegrastats", "amd_smi", "tpu_info"])
  p_probe.add_argument("--device-index", type=int, default=0)
  p_probe.set_defaults(func=cmd_telemetry_probe)

  p_env = sub.add_parser("envelope", help="single-chip envelope utilities")
  env_sub = p_env.add_subparsers(dest="envelope_command")
  p_check = env_sub.add_parser("check", help="enumerate devices and enforce num_devices == 1")
  p_check.add_argument("--sku", "--hardware", "--hardware-sku", dest="sku", default=None, help="hardware SKU from embodied_bench.schema.HARDWARE_SKU_SPECS")
  p_check.add_argument("--allow-no-device", action="store_true")
  p_check.add_argument("--json", action="store_true")
  p_check.set_defaults(func=cmd_envelope_check)

  p_cmp = sub.add_parser(
      "compare",
      help="compare candidate report.json vs baseline champion report (exit 0 ACCEPT, exit 2 REJECT)",
  )
  p_cmp.add_argument("--baseline", required=True, help="baseline report.json file or bundle directory")
  p_cmp.add_argument("--candidate", required=True, help="candidate report.json file or bundle directory")
  p_cmp.add_argument(
      "--cutoff-hz",
      type=float,
      default=None,
      help="minimum gapless real-time reactivity cutoff in Hz (default: 4.0 pi05-3b, 2.5 gr00t-n1.7-3b, 1.0 WM_TRACK)",
  )
  p_cmp.add_argument(
      "--min-gain-pct",
      type=float,
      default=2.0,
      help="minimum required gain %% at >= cutoff_hz to ACCEPT over baseline (default: 2.0)",
  )
  p_cmp.add_argument(
      "--allow-sanity-only",
      action="store_true",
      help="accept SANITY_ONLY quality_gate candidates (for harness self-tests only)",
  )
  p_cmp.add_argument("--out", default=None, help="optional path to write full comparison JSON atomically")
  p_cmp.add_argument("--json", action="store_true", help="print machine-readable JSON output instead of table")
  p_cmp.set_defaults(func=cmd_compare)

  # WS-C data CLI (optional, lazily registered).
  try:
    from eeb.data import cli as _data_cli  # type: ignore[import-not-found]  # noqa: WPS433

    register = getattr(_data_cli, "register_cli", None)
    if callable(register):
      register(sub)
  except Exception:  # noqa: BLE001 - WS-C absent or broken must never break the harness CLI
    pass

  # Submission verification, closed-loop evaluation, leaderboard compilation, website export, and agent-kickoff.
  try:
    from embodied_bench.cli import register_embodied_bench_subcommands

    register_embodied_bench_subcommands(
        sub,
        skip_names=("measure", "serve", "data"),
        include_pareto_alias=True,
    )
  except Exception:  # noqa: BLE001
    pass
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
  parser = build_parser()
  args = parser.parse_args(argv)
  func = getattr(args, "func", None)
  if func is None:
    parser.print_help()
    return 1
  try:
    return int(func(args) or 0)
  except (
      ValueError,
      FileNotFoundError,
      ConnectionError,
      ImportError,
      TypeError,
      AttributeError,
      RuntimeError,
  ) as exc:
    print(f"eeb {args.command}: error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
  sys.exit(main())
