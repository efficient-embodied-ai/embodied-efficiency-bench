"""Unit tests for the locked EmbodiedX v1.0 harness additions:
  * Reference policy scaffolds: Pi05Policy, Pi05JaxPolicy, Gr00tN17Policy, CosmosPredict25Policy, CosmosPredict25JaxPolicy
  * Track 2 (`WM_TRACK`) `rollout(ctx, actions)` sweep, sanity gate, golden replay gate, and report normalizations
  * `TraceBundleObservationSource` (`eeb measure --traces <bundle_dir>`)
  * Default `division == "CLOSED"` and `executed_actions_per_sec` / `frames_per_sec` normalization
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from eeb.cli import main as eeb_main
from eeb.data.schema import EmbodiedXTrace, ManifestEpisode, TraceManifest, sha256_file
from eeb.loadgen.client import InProcessPolicyClient
from eeb.loadgen.gate import golden_replay_gate, load_golden_npz, run_gate, sanity_gate
from eeb.loadgen.report import ReportInputs, build_report, validate_report
from eeb.loadgen.sweep import ConcurrencySweep, SweepConfig, TraceBundleObservationSource
from eeb.servers.cosmos_predict25 import CosmosPredict25JaxPolicy, CosmosPredict25Policy
from eeb.servers.dummy import DummyVLAPolicy, DummyWMPolicy
from eeb.servers.gr00t import Gr00tN17Policy
from eeb.servers.pi05 import Pi05JaxPolicy, Pi05Policy


def test_reference_policy_scaffolds_import_and_raise_actionable_hints() -> None:
  """All 5 reference policy scaffolds import cleanly without heavy ML frameworks installed."""
  for cls in (Pi05Policy, Pi05JaxPolicy, Gr00tN17Policy, CosmosPredict25Policy, CosmosPredict25JaxPolicy):
    with pytest.raises((ImportError, FileNotFoundError)):
      cls(warm_start=False)


def test_wm_track_concurrency_sweep_gate_and_report(tmp_path: Path) -> None:
  """ConcurrencySweep drives `rollout(ctx, actions)` when `caps.track == 'WM_TRACK'`."""
  wm_policy = DummyWMPolicy(frame_delay_ms=3.0, num_frames=4, frame_hw=(32, 32))
  cfg = SweepConfig(
      concurrency=[1, 2],
      warmup_calls=2,
      timed_calls=6,
      runs=2,
      max_variance_pct=80.0,
      max_p99_variance_pct=150.0,
      pacing_enabled=False,
      server_warmup_calls=1,
  )
  sweep = ConcurrencySweep(
      "inproc:eeb.servers.dummy:DummyWMPolicy",
      cfg,
      policy=wm_policy,
  )
  res = sweep.run()
  assert res.caps.track == "WM_TRACK"
  assert res.all_passed is True
  assert len(res.points) == 2

  # Build a golden frame replay NPZ from the deterministic DummyWMPolicy
  client = InProcessPolicyClient(wm_policy)
  ctx0, act0 = sweep._observation_source.make_rollout_inputs("golden-test", 0, num_frames=4, action_dim=7)  # type: ignore[union-attr]
  roll0 = client.rollout(ctx0, act0)
  ref_frames = np.stack([f.image for f in roll0.frames])[None, ...]  # [1, 4, 32, 32, 3]
  golden_path = tmp_path / "wm_golden.npz"
  np.savez(
      golden_path,
      ref_frames=ref_frames,
      ref_actions=act0[None, ...],
      proprio=ctx0.proprio[None, ...],
      instructions=np.array([ctx0.instruction]),
      image__agentview=ctx0.frames[0][None, ...],
  )
  golden_data = load_golden_npz(golden_path)
  gates = {}
  for pt in res.points:
    g = run_gate(pt, res.caps, f"wm-C{pt.concurrency}", client=client, golden=golden_data)
    assert g.passed is True
    assert g.quality_certified is True
    gates[pt.concurrency] = g

  report = build_report(ReportInputs(sweep=res, gates=gates))
  assert validate_report(report) == []
  assert report["division"] == "CLOSED"
  assert report["track"] == "WM_TRACK"
  for op in report["operating_points"]:
    assert op["frames_per_sec"] is not None
    assert op["frames_per_sec"] > 0.0


def test_trace_bundle_observation_source_and_cli_measure(tmp_path: Path) -> None:
  """`TraceBundleObservationSource` and `eeb measure --traces <bundle_dir>` replay real packed traces."""
  split_dir = tmp_path / "traces" / "public"
  split_dir.mkdir(parents=True, exist_ok=True)
  t = 12
  trace = EmbodiedXTrace(
      episode_id="test_libero/episode_000000",
      source_dataset="LIBERO",
      split="public",
      instruction="pick up the red mug",
      fps=20.0,
      t_ns=np.arange(t, dtype=np.int64) * 50_000_000,
      step=np.arange(t, dtype=np.int64),
      proprio=np.zeros((t, 8), dtype=np.float32),
      action=np.ones((t, 7), dtype=np.float32) * 0.25,
      images={
          "agentview": np.zeros((t, 64, 64, 3), dtype=np.uint8),
          "wrist": np.zeros((t, 64, 64, 3), dtype=np.uint8),
      },
      license="MIT",
  )
  ep_file = trace.to_npz(split_dir / "episode_0000.npz")
  man = TraceManifest(
      source="test_libero",
      source_dataset="LIBERO",
      license="MIT",
      fps=20.0,
      cameras=["agentview", "wrist"],
      split="public",
      episodes=[
          ManifestEpisode(
              episode_id=trace.episode_id,
              file="episode_0000.npz",
              steps=t,
              instruction=trace.instruction,
              sha256=sha256_file(ep_file),
              content_sha256=trace.content_sha256(),
              source_episode_index=0,
              cameras=["agentview", "wrist"],
          )
      ],
  )
  man.save(split_dir)

  src = TraceBundleObservationSource(str(tmp_path), split="public")
  obs = src.make("sess-0", 3)
  assert obs.instruction == "pick up the red mug"
  assert set(obs.images.keys()) == {"agentview", "wrist"}
  ctx, act = src.make_rollout_inputs("sess-0", 3, num_frames=6, action_dim=7)
  assert act.shape == (6, 7)
  assert np.allclose(act, 0.25)

  out_dir = tmp_path / "out_vla"
  rc = eeb_main([
      "measure",
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--policy-kwargs",
      '{"compute_delay_ms": 0.5, "horizon": 10, "action_dim": 7}',
      "--execute-horizon",
      "5",
      "--traces",
      str(tmp_path),
      "--sweep",
      "1,2",
      "--warmup",
      "1",
      "--calls",
      "4",
      "--runs",
      "2",
      "--no-pacing",
      "--allow-no-device",
      "--sku",
      "RTX_4090_24GB",
      "--out",
      str(out_dir),
      "--quiet",
  ])
  assert rc == 0
  import json

  rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
  assert rep["division"] == "CLOSED"
  assert rep["execution_provenance"]["observations"] == "EMBODIEDX_TRACES_V1_BUNDLE"
  op0 = rep["operating_points"][0]
  assert op0["execute_horizon"] == 5
  assert op0["horizon"] == 10
  # Effective per-chip throughput normalization uses executed_actions_per_sec (50% of actions_per_sec)
  assert np.isclose(op0["raw_per_chip_throughput"], op0["executed_actions_per_sec"], rtol=1e-3)

  # Gate 3: Test reversed CLI flag order as well
  out_dir_rev = tmp_path / "out_vla_rev"
  rc_rev = eeb_main([
      "measure",
      "--out",
      str(out_dir_rev),
      "--sku",
      "RTX_4090_24GB",
      "--allow-no-device",
      "--no-pacing",
      "--runs",
      "2",
      "--calls",
      "4",
      "--warmup",
      "1",
      "--sweep",
      "1",
      "--traces",
      str(tmp_path),
      "--traces-split",
      "public",
      "--execute-horizon",
      "5",
      "--policy-kwargs",
      '{"compute_delay_ms": 0.5, "horizon": 10, "action_dim": 7}',
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--quiet",
  ])
  assert rc_rev == 0


def test_three_paths_and_negative_controls_wm_golden_replay(tmp_path: Path) -> None:
  """Gate 1 (repeat + concurrent write_report_bundle) and Gate 4 (WM_TRACK golden replay negative control)."""
  from concurrent.futures import ThreadPoolExecutor
  from eeb.loadgen.report import write_report_bundle

  wm_policy = DummyWMPolicy(frame_delay_ms=0.1, num_frames=4, frame_hw=(32, 32))
  cfg = SweepConfig(concurrency=[1], warmup_calls=1, timed_calls=4, runs=2, max_variance_pct=50.0, pacing_enabled=False, server_warmup_calls=0)
  res = ConcurrencySweep("inproc:eeb.servers.dummy:DummyWMPolicy", cfg, policy=wm_policy).run()
  client = InProcessPolicyClient(wm_policy)
  ctx0, act0 = res.points[0].runs[0], None
  obs_src = TraceBundleObservationSource if False else None  # unused
  from eeb.loadgen.sweep import ObservationSource, ObservationSpec

  src = ObservationSource(ObservationSpec.from_caps(res.caps))
  ctx0, act0 = src.make_rollout_inputs("neg-ctrl", 0, num_frames=4, action_dim=7)
  roll0 = client.rollout(ctx0, act0)
  clean_frames = np.stack([f.image for f in roll0.frames])[None, ...]

  # Negative control: invert pixels so PSNR drops below 26 dB -> golden_replay_gate MUST fail
  corrupted_frames = (255 - clean_frames).astype(np.uint8)
  bad_golden = {
      "ref_frames": corrupted_frames,
      "ref_actions": act0[None, ...],
      "cameras": {"agentview": ctx0.frames[0][None, ...]},
      "proprio": ctx0.proprio[None, ...],
      "instructions": [ctx0.instruction],
      "n": 1,
  }
  bad_gate = run_gate(res.points[0], res.caps, "wm-neg-C1", client=client, golden=bad_golden)
  assert bad_gate.passed is False
  assert bad_gate.quality_certified is False
  assert "GOLDEN_REPLAY gate failed" in bad_gate.reason

  # Repeat & Concurrent writer path on write_report_bundle
  rep = build_report(ReportInputs(sweep=res, gates={1: bad_gate}))
  assert rep["status"] == "INVALID"
  assert rep["reason"]

  def _writer(idx: int) -> str:
    b = write_report_bundle(rep, tmp_path / "concurrent_bundle", basename=f"report_{idx}")
    return b["attestation_sha256"]

  with ThreadPoolExecutor(max_workers=4) as ex:
    digests = list(ex.map(_writer, range(4)))
  assert len(set(digests)) == 1


def test_remediated_v1_findings_caps_execute_horizon_ttff_and_d3_gates(tmp_path: Path) -> None:
  """Verify remediations for Findings #37.1, #37.2, #37.3, #37.4, and #37.5."""
  import inspect
  from embodied_bench.schema import CANONICAL_V1_MODELS, QUALITY_GATE_THRESHOLDS, RecipeConfig
  from eeb.protocol.policy import Caps
  from eeb.telemetry.sidecar import TelemetrySummary

  # 1. Finding #37.1: Caps.execute_horizon round-trip & automatic propagation in ConcurrencySweep._discover()
  caps = Caps(
      track="VLA_TRACK",
      horizon=50,
      dt_ms=50.0,
      server_impl="SYNTHETIC_DUMMY",
      execute_horizon=5,
  )
  assert Caps.from_dict(caps.to_dict()).execute_horizon == 5

  sig_gr00t = inspect.signature(Gr00tN17Policy.__init__)
  assert sig_gr00t.parameters["denoising_steps"].default == 16
  assert sig_gr00t.parameters["execute_horizon"].default == 8

  class _LockedHorizonDummyVLA(DummyVLAPolicy):
    def capabilities(self) -> Caps:
      base = super().capabilities()
      return Caps(
          track=base.track,
          horizon=50,
          dt_ms=base.dt_ms,
          modalities=base.modalities,
          precision=base.precision,
          model_id="pi05-3b",
          server_impl=base.server_impl,
          action_dim=base.action_dim,
          proprio_dim=base.proprio_dim,
          cameras=base.cameras,
          image_height=base.image_height,
          image_width=base.image_width,
          execute_horizon=5,
      )

  vla_pol = _LockedHorizonDummyVLA(compute_delay_ms=0.1, horizon=50, action_dim=7)
  cfg_auto = SweepConfig(concurrency=[1], warmup_calls=1, timed_calls=3, runs=2, pacing_enabled=False, server_warmup_calls=0)
  res_auto = ConcurrencySweep("inproc:eeb.servers.dummy:DummyVLAPolicy", cfg_auto, policy=vla_pol).run()
  assert res_auto.pacing.effective_execute_horizon == 5
  assert res_auto.points[0].runs[0].execute_horizon == 5

  # 2. Finding #37.2 & #37.5 + D7/D3 Addendum: per-call ttff_ms diagnostic, observation_staleness_ms, peak_mem_used_gb, and CLOSED stream_chunk_frames ban
  wm_fast = DummyWMPolicy(frame_delay_ms=0.2, num_frames=4, frame_hw=(32, 32))
  res_wm_fast = ConcurrencySweep("inproc:eeb.servers.dummy:DummyWMPolicy", cfg_auto, policy=wm_fast).run()
  tel_summary = TelemetrySummary(
      source="nvml",
      provenance="MEASURED",
      num_samples=5,
      window_s=1.0,
      mean_power_w=120.0,
      peak_power_w=135.0,
      energy_j=120.0,
      peak_mem_used_mb=8192.0,
      mean_util_pct=85.0,
      sample_hz_actual=10.0,
      power_coverage_pct=100.0,
  )
  tel_by_run = {(1, 0): tel_summary, (1, 1): tel_summary}
  rep_fast = build_report(ReportInputs(sweep=res_wm_fast, telemetry_by_run=tel_by_run))
  op_fast = rep_fast["operating_points"][0]
  assert op_fast["ttff_p99_ms"] is not None
  assert op_fast["observation_staleness_ms"] is not None and op_fast["observation_staleness_ms"] > op_fast["obs_to_action_p99_ms"]
  assert op_fast["peak_mem_used_gb"] == 8.0
  assert not any("single-stream ttff_p99_ms" in f for f in rep_fast["admissibility_findings"])

  # Lookalike: full-clip 3D-DiT TTFF == L_clip = 620.0 ms is recorded as D7 diagnostic without disqualifying CLOSED division
  res_wm_fast.points[0].median_ttff_ms["p99"] = 620.0
  rep_fullclip = build_report(ReportInputs(sweep=res_wm_fast, telemetry_by_run=tel_by_run))
  assert rep_fullclip["operating_points"][0]["ttff_p99_ms"] == 620.0
  assert not any("ttff_p99_ms" in f for f in rep_fullclip["admissibility_findings"])

  # Negative control: CLOSED division with stream_chunk_frames=4 is blocked, while OPEN division allows it
  rep_bad_chunk = build_report(
      ReportInputs(
          sweep=res_wm_fast,
          telemetry_by_run=tel_by_run,
          recipe={"division": "CLOSED", "stream_chunk_frames": 4},
      )
  )
  assert any("CLOSED division stream_chunk_frames=4 violation" in f for f in rep_bad_chunk["admissibility_findings"])
  rep_open_chunk = build_report(
      ReportInputs(
          sweep=res_wm_fast,
          telemetry_by_run=tel_by_run,
          recipe={"division": "OPEN", "stream_chunk_frames": 4},
      )
  )
  assert not any("stream_chunk_frames" in f for f in rep_open_chunk["admissibility_findings"])

  # 3. Finding #37.3 & #37.4: D3 thresholds, default sweep [1,2,4,8,16], CANONICAL_V1_MODELS, and RecipeConfig.division == "CLOSED"
  assert SweepConfig().concurrency == [1, 2, 4, 8, 16]
  assert QUALITY_GATE_THRESHOLDS["VLA_TRACK"]["min_action_cosine_sim"] == 0.995
  assert CANONICAL_V1_MODELS == ("pi05-3b", "gr00t-n1.7-3b", "cosmos-predict2.5-2b")
  rcfg = RecipeConfig(
      recipe_id="test_default_closed",
      track="VLA_TRACK",
      model_family="pi05",
      model_name="pi05-3b",
      baseline_model_id="pi05-3b",
      precision="BF16",
      hardware_sku="RTX_4090_24GB",
  )
  assert rcfg.division == "CLOSED"


def test_grpc_wire_execute_horizon_roundtrip_and_uds_sweep(tmp_path: Path) -> None:
  """Finding #37.1 across gRPC: `Caps.execute_horizon` (field 13) round-trips over UDS via `eeb.protocol.server` and `GrpcPolicyClient`."""
  from eeb.loadgen.client import GrpcPolicyClient
  from eeb.protocol import eeb_pb2
  from eeb.protocol.server import caps_to_proto, proto_to_caps, serve_policy

  assert eeb_pb2.Caps.DESCRIPTOR.fields_by_name["execute_horizon"].number == 13
  pol = DummyVLAPolicy(compute_delay_ms=0.2, horizon=50, action_dim=7, execute_horizon=5)
  proto_msg = caps_to_proto(pol.capabilities())
  assert proto_msg.execute_horizon == 5
  assert proto_to_caps(proto_msg).execute_horizon == 5

  sock_path = tmp_path / "eeb_test_exec_h.sock"
  endpoint = f"unix://{sock_path}"
  srv = serve_policy(pol, endpoint, max_workers=4)
  try:
    with GrpcPolicyClient(endpoint) as client:
      remote_caps = client.capabilities()
      assert remote_caps.horizon == 50
      assert remote_caps.execute_horizon == 5

    cfg = SweepConfig(concurrency=[1], warmup_calls=1, timed_calls=3, runs=2, pacing_enabled=False, server_warmup_calls=0)
    res_uds = ConcurrencySweep(endpoint, cfg).run()
    assert res_uds.caps.execute_horizon == 5
    assert res_uds.pacing.effective_execute_horizon == 5
    assert res_uds.points[0].runs[0].execute_horizon == 5
  finally:
    srv.stop(grace=1.0)


def test_locked_horizon_enforcement_y_metric_jerk_and_sanity_only_strict(tmp_path: Path) -> None:
  """Findings #37.1b-c, #37.3c, #37.4a, #37.5: locked horizon checks, `executed_actions_per_sec` Y_METRIC, `normalized_action_jerk`, `SANITY_ONLY` strict rejection, and `--recipe` inheritance."""
  import json
  from eeb.cli import LEGACY_SUBCOMMANDS
  from eeb.loadgen.report import Y_METRIC, extract_iso_reactivity_slices
  from eeb.protocol.policy import Caps

  for legacy_cmd in ("validate-submission", "download-data", "benchmark"):
    assert legacy_cmd in LEGACY_SUBCOMMANDS

  assert Y_METRIC == "executed_actions_per_sec"

  pol = DummyVLAPolicy(compute_delay_ms=0.2, horizon=50, action_dim=7, execute_horizon=5)
  cfg = SweepConfig(concurrency=[1, 2], warmup_calls=1, timed_calls=4, runs=2, pacing_enabled=False, server_warmup_calls=0)
  res = ConcurrencySweep("inproc:eeb.servers.dummy:DummyVLAPolicy", cfg, policy=pol).run()

  gates_sanity = {pt.concurrency: sanity_gate(pt, res.caps, f"sanity-C{pt.concurrency}") for pt in res.points}
  for g in gates_sanity.values():
    assert g.passed is True
    assert g.quality_certified is False
    assert g.normalized_action_jerk is not None and g.normalized_action_jerk >= 0.0

  # 1. Positive control: CLOSED division with wrong execute_horizon for pi05 (10 != 5), gr00t (5 != 8), cosmos-predict2.5 (8 != 16)
  bad_pi05_pol = DummyVLAPolicy(compute_delay_ms=0.1, horizon=50, action_dim=7, execute_horizon=10)
  res_bad_pi05 = ConcurrencySweep("inproc:eeb.servers.dummy:DummyVLAPolicy", cfg, policy=bad_pi05_pol).run()
  rep_bad_pi05 = build_report(
      ReportInputs(
          sweep=res_bad_pi05,
          recipe={"recipe_id": "bad_pi05", "model_family": "pi05", "model_name": "pi05-3b", "division": "CLOSED"},
          gates=gates_sanity,
          allow_sanity_only=True,
      )
  )
  assert any("CLOSED division locked execute_horizon violation" in f and "execute_horizon=5" in f for f in rep_bad_pi05["admissibility_findings"])

  rep_bad_gr00t = build_report(
      ReportInputs(
          sweep=res,
          recipe={"recipe_id": "bad_gr00t", "model_family": "gr00t", "model_name": "gr00t-n1.7-3b", "division": "CLOSED"},
          gates=gates_sanity,
          allow_sanity_only=True,
      )
  )
  assert any("CLOSED division locked execute_horizon violation" in f and "execute_horizon=8" in f for f in rep_bad_gr00t["admissibility_findings"])

  rep_bad_cosmos = build_report(
      ReportInputs(
          sweep=res,
          recipe={"recipe_id": "bad_cosmos", "model_family": "cosmos-predict2.5", "model_name": "cosmos-predict2.5-2b", "division": "CLOSED"},
          gates=gates_sanity,
          allow_sanity_only=True,
      )
  )
  assert any("CLOSED division locked execute_horizon violation" in f and "execute_horizon=16" in f for f in rep_bad_cosmos["admissibility_findings"])

  # 2. Negative control: CLOSED division with matching execute_horizon=5 for pi05-3b
  rep_ok_pi05 = build_report(
      ReportInputs(
          sweep=res,
          recipe={"recipe_id": "ok_pi05", "model_family": "pi05", "model_name": "pi05-3b", "division": "CLOSED"},
          gates=gates_sanity,
          allow_sanity_only=False,
      )
  )
  assert not any("CLOSED division locked execute_horizon violation" in f for f in rep_ok_pi05["admissibility_findings"])
  assert rep_ok_pi05["y_metric_name"] == "executed_actions_per_sec"
  assert rep_ok_pi05["quality_gate"]["quality_certified"] is False
  assert rep_ok_pi05["quality_gate"]["normalized_action_jerk"] is not None
  assert any("not quality_certified" in f for f in rep_ok_pi05["admissibility_findings"])
  assert rep_ok_pi05["summary"]["peak_executed_actions_per_sec"] is not None
  assert rep_ok_pi05["summary"]["peak_actions_per_sec"] is not None
  # Since horizon=50 and execute_horizon=5, peak_executed_actions_per_sec == 0.1 * peak_actions_per_sec
  assert np.isclose(
      rep_ok_pi05["summary"]["peak_executed_actions_per_sec"],
      0.1 * rep_ok_pi05["summary"]["peak_actions_per_sec"],
      rtol=1e-3,
  )
  for op in rep_ok_pi05["operating_points"]:
    assert op["normalized_action_jerk"] is not None
  slices = extract_iso_reactivity_slices(rep_ok_pi05["pareto_frontier"])
  assert isinstance(slices, dict)

  # 3. CLI `--recipe` inheritance (concurrency_levels + execute_horizon) and `--strict` vs `--allow-sanity-only`
  recipe_file = tmp_path / "pi05_test_recipe.json"
  recipe_file.write_text(
      json.dumps({
          "recipe_id": "pi05_cli_recipe_test",
          "track": "VLA_TRACK",
          "model_family": "pi05",
          "model_name": "pi05-3b",
          "baseline_model_id": "pi05-3b",
          "precision": "BF16",
          "hardware_sku": "RTX_4090_24GB",
          "division": "CLOSED",
          "concurrency_levels": [1, 2],
          "execute_horizon": 5,
      }),
      encoding="utf-8",
  )
  out_recipe_dir = tmp_path / "out_recipe_cli"
  rc_strict_fail = eeb_main([
      "measure",
      "--recipe",
      str(recipe_file),
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--policy-kwargs",
      '{"compute_delay_ms": 0.2, "horizon": 50, "action_dim": 7}',
      "--warmup",
      "1",
      "--calls",
      "3",
      "--runs",
      "2",
      "--no-pacing",
      "--allow-no-device",
      "--strict",
      "--out",
      str(out_recipe_dir),
      "--quiet",
  ])
  # Without --golden (and on synthetic/no-device), --strict MUST exit with code 2
  assert rc_strict_fail == 2
  rep_cli = json.loads((out_recipe_dir / "report.json").read_text(encoding="utf-8"))
  assert [op["concurrency"] for op in rep_cli["operating_points"]] == [1, 2]
  assert rep_cli["operating_points"][0]["execute_horizon"] == 5
  assert rep_cli["quality_gate"]["quality_certified"] is False


def test_cosmos_predict25_stepwise_streaming_rollout_ttff_vs_total() -> None:
  """Finding #37.2b: `CosmosPredict25Policy.rollout` and `CosmosPredict25JaxPolicy.rollout` yield frame 0 immediately so `ttff_ms < total_ms`."""
  import time
  from eeb.protocol.policy import Context

  class _MockStreamingDiffusersPipe:
    def stream_frames(self, **kwargs):
      n = int(kwargs.get("num_frames", 4))
      h = int(kwargs.get("height", 32))
      w = int(kwargs.get("width", 32))
      for i in range(n):
        time.sleep(0.004)
        yield np.full((1, h, w, 3), i * 10, dtype=np.uint8)

    def __call__(self, **kwargs):
      n = int(kwargs.get("num_frames", 4))
      h = int(kwargs.get("height", 32))
      w = int(kwargs.get("width", 32))
      time.sleep(0.004 * n)
      return np.zeros((1, n, h, w, 3), dtype=np.uint8)

  # Test CosmosPredict25Policy with stream_frames generator
  py_pol = object.__new__(CosmosPredict25Policy)
  py_pol.model_id = "nvidia/Cosmos-Predict2.5-2B"
  py_pol.device = "cpu"
  py_pol.precision_tag = "BF16"
  py_pol.num_frames = 4
  py_pol.num_inference_steps = 4
  py_pol.frame_hw = (32, 32)
  py_pol.dt_ms = 1000.0 / 24.0
  py_pol.action_dim = 7
  py_pol.stream_chunk_frames = None
  py_pol.exclusive_compute = True
  import threading
  py_pol._lock = threading.Lock()
  py_pol._diffusers_version = "0.33.0"
  py_pol._torch_version = "2.6.0"
  py_pol._pipe = _MockStreamingDiffusersPipe()

  client_py = InProcessPolicyClient(py_pol)
  ctx = Context(
      session_id="stream-test",
      frames=[np.zeros((32, 32, 3), dtype=np.uint8)],
      proprio=np.zeros((8,), dtype=np.float32),
      instruction="stream rollout",
      num_frames=4,
  )
  actions = np.zeros((4, 7), dtype=np.float32)
  roll_py = client_py.rollout(ctx, actions)
  assert len(roll_py.frames) == 4
  assert roll_py.ttff_ms > 0.0
  assert roll_py.total_ms > roll_py.ttff_ms * 1.8

  # Also test CosmosPredict25JaxPolicy with stream_chunk_frames=1
  class _MockJaxRunner:
    def generate(self, **kwargs):
      n = int(kwargs.get("num_frames", 1))
      time.sleep(0.004 * n)
      return np.zeros((n, 32, 32, 3), dtype=np.uint8)

  jax_pol = object.__new__(CosmosPredict25JaxPolicy)
  jax_pol.checkpoint_dir = "/tmp/mock_cosmos_jax"
  jax_pol.precision_tag = "BF16"
  jax_pol.num_frames = 4
  jax_pol.num_inference_steps = 4
  jax_pol.frame_hw = (32, 32)
  jax_pol.dt_ms = 1000.0 / 24.0
  jax_pol.action_dim = 7
  jax_pol.stream_chunk_frames = 1
  jax_pol.exclusive_compute = True
  jax_pol._lock = threading.Lock()
  jax_pol._jax_version = "0.5.0"
  jax_pol._backend_platform = "tpu"
  jax_pol._jax = None
  jax_pol._runner = _MockJaxRunner()

  client_jax = InProcessPolicyClient(jax_pol)
  roll_jax = client_jax.rollout(ctx, actions)
  assert len(roll_jax.frames) == 4
  assert roll_jax.ttff_ms > 0.0
  assert roll_jax.total_ms > roll_jax.ttff_ms * 1.8


def test_engineering_plan_and_website_export_alignment() -> None:
  """Findings #39.2 and #38.3c: ENGINEERING_PLAN.md, docs/ENGINEERING_PLAN.md, and website leaderboard export align with DESIGN_DECISIONS.md."""
  repo_root = Path(__file__).resolve().parents[1]
  for rel in ("ENGINEERING_PLAN.md", "docs/ENGINEERING_PLAN.md"):
    txt = (repo_root / rel).read_text(encoding="utf-8")
    assert "Closed Division" in txt
    assert "pi05-3b" in txt
    assert "GR00T-N1.7-3B" in txt
    assert "Cosmos-Predict2.5-2B" in txt
    assert "executed_actions_per_sec" in txt
    assert "eeb.proto" in txt

  import json

  web_payload = json.loads((repo_root / "leaderboard/embodiedx_leaderboard_data.json").read_text(encoding="utf-8"))
  assert web_payload["num_verified_submissions"] == 0
  assert "TPU_V6E_1_32GB" in web_payload["cross_vendor_single_chip_skus"]
  assert "L40S_48GB" in web_payload["cross_vendor_single_chip_skus"]


def test_eeb_data_generate_golden_and_strict_measure_vla_and_wm(tmp_path: Path) -> None:
  """Release Blocker 4: `eeb data generate-golden` + `eeb measure --golden <path.npz> --strict` for Track 1 (VLA_TRACK) and Track 2 (WM_TRACK)."""
  from concurrent.futures import ThreadPoolExecutor
  import json
  from eeb.data.golden import generate_golden_reference

  repo_root = Path(__file__).resolve().parents[1]
  mini_traces_dir = repo_root / "data" / "embodiedx_traces_v1_mini"

  # 1. Track 1 (VLA_TRACK): generate-golden with --traces (shard directory) and synthetic fallback + both flag orders
  vla_golden_npz = tmp_path / "vla_golden.npz"
  orphan_tmp = tmp_path / ".tmp_golden_vla_golden_stale.npz"
  orphan_tmp.write_bytes(b"corrupt_orphan")

  rc_gen_vla = eeb_main([
      "data",
      "generate-golden",
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--traces",
      str(mini_traces_dir),
      "--num-samples",
      "100",
      "--out",
      str(vla_golden_npz),
      "--quiet",
  ])
  assert rc_gen_vla == 0
  assert vla_golden_npz.exists()
  assert not orphan_tmp.exists(), "Orphan .tmp_golden_* file must be cleaned up on write"

  # Verify VLA .npz contents match golden_replay_gate contract
  with np.load(vla_golden_npz, allow_pickle=False) as d_vla:
    assert "actions" in d_vla.files
    assert "proprios" in d_vla.files
    assert "instructions" in d_vla.files
    assert "images" in d_vla.files
    assert "metadata_json" in d_vla.files
    assert d_vla["actions"].shape == (100, 8, 7)

  # Run `eeb measure --golden <vla_golden.npz> --strict`
  vla_out_dir = tmp_path / "measure_vla_strict"
  rc_meas_vla = eeb_main([
      "measure",
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--traces",
      str(mini_traces_dir),
      "--golden",
      str(vla_golden_npz),
      "--sweep",
      "1,2",
      "--warmup",
      "1",
      "--calls",
      "4",
      "--runs",
      "2",
      "--no-pacing",
      "--strict",
      "--out",
      str(vla_out_dir),
      "--quiet",
  ])
  assert rc_meas_vla == 0
  rep_vla = json.loads((vla_out_dir / "report.json").read_text(encoding="utf-8"))
  assert rep_vla["quality_certified"] is True
  assert rep_vla["gate_mode"] == "GOLDEN_REPLAY"
  assert rep_vla["quality_gate"]["passed"] is True
  assert rep_vla["quality_gate"]["quality_certified"] is True
  assert rep_vla["quality_gate"]["mode"] == "GOLDEN_REPLAY"
  assert rep_vla["quality_gate"]["gate_mode"] == "GOLDEN_REPLAY"
  for op in rep_vla["operating_points"]:
    assert op["quality_certified"] is True
    assert op["gate_a_mode"] == "GOLDEN_REPLAY"
    assert op["gate_mode"] == "GOLDEN_REPLAY"
    assert op["co_located_gate_a"]["action_cosine_sim"] == pytest.approx(1.0, abs=1e-6)
    assert op["co_located_gate_a"]["action_cosine_sim"] >= 0.995

  # 2. Track 2 (WM_TRACK): generate-golden in reversed flag order + measure --golden --strict in reversed flag order
  wm_golden_npz = tmp_path / "wm_golden.npz"
  rc_gen_wm = eeb_main([
      "data",
      "generate-golden",
      "--quiet",
      "--out",
      str(wm_golden_npz),
      "--num-samples",
      "100",
      "--traces",
      str(mini_traces_dir),
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyWMPolicy",
  ])
  assert rc_gen_wm == 0
  assert wm_golden_npz.exists()

  with np.load(wm_golden_npz, allow_pickle=False) as d_wm:
    assert "context_frames" in d_wm.files
    assert "proprios" in d_wm.files
    assert "instructions" in d_wm.files
    assert "actions" in d_wm.files
    assert "frames" in d_wm.files
    assert "metadata_json" in d_wm.files
    assert d_wm["frames"].shape[0] == 100
    assert d_wm["frames"].ndim == 5

  wm_out_dir = tmp_path / "measure_wm_strict"
  rc_meas_wm = eeb_main([
      "measure",
      "--quiet",
      "--out",
      str(wm_out_dir),
      "--strict",
      "--no-pacing",
      "--runs",
      "2",
      "--calls",
      "4",
      "--warmup",
      "1",
      "--sweep",
      "1,2",
      "--traces",
      str(mini_traces_dir),
      "--golden",
      str(wm_golden_npz),
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyWMPolicy",
  ])
  assert rc_meas_wm == 0
  rep_wm = json.loads((wm_out_dir / "report.json").read_text(encoding="utf-8"))
  assert rep_wm["quality_certified"] is True
  assert rep_wm["gate_mode"] == "GOLDEN_REPLAY"
  assert rep_wm["quality_gate"]["passed"] is True
  assert rep_wm["quality_gate"]["quality_certified"] is True
  assert rep_wm["quality_gate"]["mode"] == "GOLDEN_REPLAY"
  assert rep_wm["quality_gate"]["gate_mode"] == "GOLDEN_REPLAY"
  for op in rep_wm["operating_points"]:
    assert op["quality_certified"] is True
    assert op["gate_a_mode"] == "GOLDEN_REPLAY"
    assert op["gate_mode"] == "GOLDEN_REPLAY"
    assert op["co_located_gate_a"]["frame_psnr_db"] == pytest.approx(99.0, abs=1e-4)
    assert op["co_located_gate_a"]["frame_psnr_db"] >= 28.0
    assert op["co_located_gate_a"]["action_cosine_sim"] == pytest.approx(1.0, abs=1e-6)
    assert op["co_located_gate_a"]["action_cosine_sim"] >= 0.990

  # 3. Gate 1 Concurrent-writer path on generate_golden_reference
  def _gen_worker(idx: int) -> int:
    s = generate_golden_reference(
        endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy",
        out_path=tmp_path / f"concurrent_golden_{idx}.npz",
        num_samples=2,
        allow_smoke_golden=True,
    )
    return int(s["num_samples"])

  with ThreadPoolExecutor(max_workers=4) as pool:
    counts = list(pool.map(_gen_worker, range(4)))
  assert counts == [2, 2, 2, 2]

  # 4. Gate 4 Negative Control: corrupted golden actions must cause `eeb measure --golden --strict` to return 2
  bad_vla_npz = tmp_path / "bad_vla_golden.npz"
  with np.load(vla_golden_npz, allow_pickle=False) as d_orig:
    bad_arrays = {k: d_orig[k] for k in d_orig.files}
  bad_arrays["actions"] = -bad_arrays["actions"]
  bad_arrays["ref_actions"] = -bad_arrays["ref_actions"]
  np.savez_compressed(bad_vla_npz, **bad_arrays)

  bad_out_dir = tmp_path / "measure_vla_bad_strict"
  rc_bad = eeb_main([
      "measure",
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--golden",
      str(bad_vla_npz),
      "--sweep",
      "1",
      "--warmup",
      "1",
      "--calls",
      "3",
      "--runs",
      "2",
      "--no-pacing",
      "--strict",
      "--out",
      str(bad_out_dir),
      "--quiet",
  ])
  assert rc_bad == 2
  rep_bad = json.loads((bad_out_dir / "report.json").read_text(encoding="utf-8"))
  assert rep_bad["quality_certified"] is False
  assert rep_bad["quality_gate"]["passed"] is False
  assert rep_bad["quality_gate"]["quality_certified"] is False

  # 5. `eeb data inspect` on mini_v1 shard manifest (both flag orders + corrupted-manifest negative control)
  assert eeb_main(["data", "inspect", str(mini_traces_dir), "--limit", "2"]) == 0
  assert eeb_main(["data", "inspect", "--limit", "2", str(mini_traces_dir)]) == 0
  bad_shard_dir = tmp_path / "bad_shard_bundle"
  bad_shard_dir.mkdir(parents=True, exist_ok=True)
  (bad_shard_dir / "manifest.json").write_text(
      json.dumps({"schema_version": "1.0.0", "split": "mini_v1", "shards": [{"episode_id": "missing_ep", "filename": "missing.npz"}]}),
      encoding="utf-8",
  )
  assert eeb_main(["data", "inspect", str(bad_shard_dir)]) == 1


def test_v1_release_version_and_submission_dir_cli_alias(tmp_path: Path) -> None:
  """Verify `eeb.__version__ == '1.0.0'`, `--submission` / `--submission-dir` CLI aliases in both flag orders, and doc alignment."""
  import json
  import eeb
  import embodied_bench
  from embodied_bench.cli import main as eb_main

  assert eeb.__version__ == "1.0.0"
  assert embodied_bench.__version__ == "1.0.0"

  repo_root = Path(__file__).resolve().parents[1]
  sub_dir = tmp_path / "nvidia_l40s_gr00t_n17_fp8"
  sub_dir.mkdir(parents=True, exist_ok=True)
  (sub_dir / "recipe.yaml").write_text(
      (repo_root / "recipes" / "track1_vla_gr00t_n17_l40s.yaml").read_text(encoding="utf-8"),
      encoding="utf-8",
  )
  out1 = tmp_path / "val_sub_1.json"
  out2 = tmp_path / "val_sub_2.json"
  out3 = tmp_path / "val_sub_3.json"

  # 1. `--submission` standard flag
  assert eb_main(["validate-submission", "--submission", str(sub_dir), "--output", str(out1)]) == 0
  r1 = json.loads(out1.read_text(encoding="utf-8"))
  assert r1["passed"] is True
  assert r1["violations"] == []

  # 2. `--submission-dir` alias (order 1: --submission-dir first)
  assert eb_main(["validate-submission", "--submission-dir", str(sub_dir), "--output", str(out2)]) == 0
  r2 = json.loads(out2.read_text(encoding="utf-8"))
  assert r2["passed"] is True
  assert r2["violations"] == []

  # 3. `--submission-dir` alias via forwarded `eeb validate-submission` in reversed flag order (--output first)
  assert eeb_main(["validate-submission", "--output", str(out3), "--submission-dir", str(sub_dir)]) == 0
  r3 = json.loads(out3.read_text(encoding="utf-8"))
  assert r3["passed"] is True
  assert r3["violations"] == []

  # 3b. Negative control: missing recipe in submission dir raises FileNotFoundError
  empty_sub_dir = tmp_path / "empty_sub"
  empty_sub_dir.mkdir(parents=True, exist_ok=True)
  with pytest.raises(FileNotFoundError, match="No 'recipe.yaml'"):
    eb_main(["validate-submission", "--submission-dir", str(empty_sub_dir)])

  # 4. Verify zero em-dashes in docs and LITERATURE_REVIEW_AXES_AND_METRICS.md alignment
  lit_txt = (repo_root / "docs" / "LITERATURE_REVIEW_AXES_AND_METRICS.md").read_text(encoding="utf-8")
  assert "\u2014" not in lit_txt
  assert "p99" in lit_txt and "1000" in lit_txt
  assert "Actions/s" in lit_txt and "Predictions/s" in lit_txt
  assert "0.995" in lit_txt and "0.99" in lit_txt and "26.0\\text{ dB}" in lit_txt

  tex_txt = (repo_root / "docs" / "EmbodiedX_Benchmark_Methodology_and_Metrics.tex").read_text(encoding="utf-8")
  assert "---" not in tex_txt and "\u2014" not in tex_txt


def _make_mock_torch_and_torchtpu(
    *,
    cuda_available: bool = False,
    tpu_available: bool = True,
    tpu_device_count: int = 1,
    tpu_device_name: str = "Google Cloud TPU v6e-1",
    tpu_total_memory_bytes: int = 32 * 1024**3,
    attach_tpu_on_import: bool = True,
    sync_error: Exception | None = None,
):
  """Create isolated mock `torch` and `torch_tpu` modules for unit testing."""
  from concurrent.futures import ThreadPoolExecutor  # noqa: F401
  import threading
  import types

  state = {"cuda_sync_calls": 0, "tpu_sync_calls": 0, "compile_calls": 0}
  lock = threading.Lock()

  torch_mod = types.ModuleType("torch")
  torch_mod.__version__ = "2.7.0"
  torch_mod.bfloat16 = "bfloat16"
  torch_mod.float16 = "float16"

  cuda_ns = types.SimpleNamespace()
  cuda_ns.is_available = lambda: cuda_available

  def _cuda_sync(*_args, **_kwargs):
    with lock:
      state["cuda_sync_calls"] += 1

  cuda_ns.synchronize = _cuda_sync
  torch_mod.cuda = cuda_ns

  def _compile(fn, **kwargs):
    with lock:
      state["compile_calls"] += 1
    return types.SimpleNamespace(_wrapped=fn, _compile_kwargs=kwargs, __call__=fn) if not callable(fn) else _wrap_callable(fn, kwargs)

  def _wrap_callable(fn, kwargs):
    def _wrapped(*a, **kw):
      return fn(*a, **kw)

    _wrapped._compiled_mode = kwargs.get("mode")
    return _wrapped

  torch_mod.compile = _compile

  tpu_ns = types.SimpleNamespace()
  tpu_ns.is_available = lambda: tpu_available
  tpu_ns.device_count = lambda: tpu_device_count
  tpu_ns.get_device_name = lambda idx=0: f"{tpu_device_name}:{idx}"
  tpu_ns.get_device_properties = lambda idx=0: types.SimpleNamespace(
      name=f"{tpu_device_name}:{idx}",
      total_memory=tpu_total_memory_bytes,
  )

  def _tpu_sync():
    with lock:
      state["tpu_sync_calls"] += 1
    if sync_error is not None:
      raise sync_error

  tpu_ns.synchronize = _tpu_sync
  if attach_tpu_on_import:
    torch_mod.tpu = tpu_ns

  torchtpu_mod = types.ModuleType("torch_tpu")
  torchtpu_mod._tpu_ns = tpu_ns
  return torch_mod, torchtpu_mod, tpu_ns, state


def test_torchtpu_device_resolution_positive_and_negative_controls(monkeypatch: pytest.MonkeyPatch) -> None:
  """Gate 3 & Gate 4: Verify TorchTPU resolution priority, install hints, bounds checks, and silent lookalikes."""
  import importlib
  import importlib.util
  import sys

  from eeb.servers import base as servers_base
  from eeb.servers.torch_backend import (
      TORCHTPU_INSTALL_HINT,
      maybe_compile_torch_module,
      resolve_torch_device,
      synchronize_torch_device,
  )

  assert servers_base.TORCHTPU_INSTALL_HINT == TORCHTPU_INSTALL_HINT
  assert servers_base.resolve_torch_device is resolve_torch_device
  assert servers_base.synchronize_torch_device is synchronize_torch_device
  assert servers_base.maybe_compile_torch_module is maybe_compile_torch_module
  assert "pip install torch torch_tpu" in TORCHTPU_INSTALL_HINT

  # Positive 1: explicit `tpu` and `tpu:0` resolve cleanly with TorchTPU present
  torch_mod, torchtpu_mod, _tpu_ns, _state = _make_mock_torch_and_torchtpu(
      cuda_available=False, tpu_available=True, tpu_device_count=1
  )
  monkeypatch.setitem(sys.modules, "torch", torch_mod)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_mod)
  assert resolve_torch_device("tpu") == "tpu"
  assert resolve_torch_device("tpu:0") == "tpu:0"
  assert resolve_torch_device("auto") == "tpu"

  # Positive 2: `auto` prefers CUDA/ROCm when available
  torch_cuda, torchtpu_cuda, _, _ = _make_mock_torch_and_torchtpu(
      cuda_available=True, tpu_available=True, tpu_device_count=1
  )
  monkeypatch.setitem(sys.modules, "torch", torch_cuda)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_cuda)
  assert resolve_torch_device("auto") == "cuda"
  assert resolve_torch_device("cuda:0") == "cuda:0"

  # Positive 3 & Negative: `auto` falls back to CPU unless `allow_cpu_fallback=False`
  torch_none, torchtpu_none, _, _ = _make_mock_torch_and_torchtpu(
      cuda_available=False, tpu_available=False, tpu_device_count=0
  )
  monkeypatch.setitem(sys.modules, "torch", torch_none)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_none)
  assert resolve_torch_device("auto", allow_cpu_fallback=True) == "cpu"
  with pytest.raises(RuntimeError, match=r"torch\.tpu\.is_available\(\)"):
    resolve_torch_device("auto", allow_cpu_fallback=False)

  # Negative Control 1: Missing `torch_tpu` raises actionable ImportError mentioning pip install
  monkeypatch.setitem(sys.modules, "torch_tpu", None)
  real_import_module = importlib.import_module

  def _blocked_torchtpu_import(name: str, package: str | None = None):
    if name == "torch_tpu":
      raise ImportError("No module named 'torch_tpu'")
    return real_import_module(name, package)

  monkeypatch.setattr(importlib, "import_module", _blocked_torchtpu_import)
  with pytest.raises(ImportError, match="pip install torch torch_tpu"):
    resolve_torch_device("tpu")

  # Negative Control 2: `torch_tpu` installed cleanly, but `is_available() == False`
  monkeypatch.setattr(importlib, "import_module", real_import_module)
  monkeypatch.setitem(sys.modules, "torch", torch_none)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_none)
  with pytest.raises(RuntimeError, match="v6e-1 or v5e-1"):
    resolve_torch_device("tpu")

  # Negative Control 3: Out-of-range TPU index & malformed TPU index string
  monkeypatch.setitem(sys.modules, "torch", torch_mod)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_mod)
  with pytest.raises(RuntimeError, match=r"device_count\(\)` is 1"):
    resolve_torch_device("tpu:1")
  with pytest.raises(ValueError, match="Invalid TorchTPU device index"):
    resolve_torch_device("tpu:abc")

  # Negative Control 4: Legacy `xla:0` emits DeprecationWarning referencing TorchTPU and maps to `tpu:0`
  with pytest.warns(DeprecationWarning, match="TorchTPU"):
    assert resolve_torch_device("xla:0") == "tpu:0"

  # Negative Control 5: Unsupported backend string raises ValueError
  with pytest.raises(ValueError, match="Unsupported PyTorch device"):
    resolve_torch_device("npu:0")

  # Gate 4 Lookalike Negative Control: `cpu` and `None` synchronization stay completely silent even without torch
  monkeypatch.setitem(sys.modules, "torch", None)
  monkeypatch.setitem(sys.modules, "torch_tpu", None)
  assert synchronize_torch_device("cpu") is None
  assert synchronize_torch_device("") is None
  assert synchronize_torch_device(None) is None


def test_torchtpu_sync_compile_and_pytorch_servers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """Verify TorchTPU synchronization barrier propagation, `torch.compile`, and all 3 PyTorch reference scaffolds."""
  import sys
  import types

  from eeb.protocol.policy import Context, Observation
  from eeb.servers.cosmos_predict25.policy import CosmosPredict25Policy
  from eeb.servers.gr00t.policy import Gr00tN17Policy
  from eeb.servers.pi05.policy import Pi05Policy
  from eeb.servers.torch_backend import maybe_compile_torch_module, synchronize_torch_device

  # 1. Synchronization barrier & runtime error propagation (not silently swallowed)
  torch_err, torchtpu_err, _, err_state = _make_mock_torch_and_torchtpu(
      tpu_available=True, sync_error=RuntimeError("TPU HBM async trap")
  )
  monkeypatch.setitem(sys.modules, "torch", torch_err)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_err)
  with pytest.raises(RuntimeError, match="TPU HBM async trap"):
    synchronize_torch_device("tpu")
  assert err_state["tpu_sync_calls"] == 1

  # 2. Clean TorchTPU environment for server scaffold verification
  torch_mod, torchtpu_mod, _, state = _make_mock_torch_and_torchtpu(
      cuda_available=False, tpu_available=True, tpu_device_count=1
  )
  monkeypatch.setitem(sys.modules, "torch", torch_mod)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_mod)

  synchronize_torch_device("tpu")
  synchronize_torch_device("tpu:0")
  assert state["tpu_sync_calls"] == 2

  def _sample_fn(x: int) -> int:
    return x + 1

  assert maybe_compile_torch_module(_sample_fn, device="tpu", enabled=False) is _sample_fn
  compiled_fn = maybe_compile_torch_module(_sample_fn, device="tpu", enabled=True, mode="reduce-overhead")
  assert compiled_fn(4) == 5
  assert getattr(compiled_fn, "_compiled_mode", None) == "reduce-overhead"
  assert state["compile_calls"] == 1

  # 3. Pi05Policy on TorchTPU with torch_compile=True
  ckpt_dir = tmp_path / "pi05_pytorch_ckpt"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  openpi_mod = types.ModuleType("openpi")
  openpi_policies = types.ModuleType("openpi.policies")
  openpi_training = types.ModuleType("openpi.training")
  openpi_policy_cfg = types.ModuleType("openpi.policies.policy_config")
  openpi_train_cfg = types.ModuleType("openpi.training.config")
  openpi_train_cfg.get_config = lambda name: {"name": name}

  class _FakeOpenPiLoaded:
    def __init__(self, pytorch_device: str = "cpu") -> None:
      self.pytorch_device = pytorch_device
      self._model = lambda x: x

    def infer(self, example):
      assert "prompt" in example
      return {"actions": np.ones((50, 7), dtype=np.float32) * 0.25}

  def _create_trained_policy(config, checkpoint_dir, pytorch_device: str = "cpu"):
    assert checkpoint_dir == str(ckpt_dir)
    return _FakeOpenPiLoaded(pytorch_device=pytorch_device)

  openpi_policy_cfg.create_trained_policy = _create_trained_policy
  monkeypatch.setitem(sys.modules, "openpi", openpi_mod)
  monkeypatch.setitem(sys.modules, "openpi.policies", openpi_policies)
  monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", openpi_policy_cfg)
  monkeypatch.setitem(sys.modules, "openpi.training", openpi_training)
  monkeypatch.setitem(sys.modules, "openpi.training.config", openpi_train_cfg)

  sync_before = state["tpu_sync_calls"]
  compile_before = state["compile_calls"]
  pi_pol = Pi05Policy(
      checkpoint_dir=str(ckpt_dir),
      device="tpu",
      precision_tag="FP8",
      torch_compile=True,
      warm_start=False,
  )
  assert pi_pol.device == "tpu"
  assert pi_pol._policy.pytorch_device == "tpu"
  assert state["compile_calls"] == compile_before + 1
  assert pi_pol.capabilities().server_impl.endswith("-tpu+compile")

  obs = Observation(
      session_id="s-tpu",
      step=0,
      images={
          "agentview": np.zeros((224, 224, 3), dtype=np.uint8),
          "wrist": np.zeros((224, 224, 3), dtype=np.uint8),
      },
      proprio=np.zeros((8,), dtype=np.float32),
      instruction="pick block",
  )
  chunk = pi_pol.predict(obs)
  assert chunk.actions.shape == (50, 7)
  assert state["tpu_sync_calls"] == sync_before + 1

  # 4. Gr00tN17Policy on TorchTPU with torch_compile=True
  gr00t_mod = types.ModuleType("gr00t")
  gr00t_exp = types.ModuleType("gr00t.experiment")
  gr00t_dc = types.ModuleType("gr00t.experiment.data_config")
  gr00t_model = types.ModuleType("gr00t.model")
  gr00t_pol_mod = types.ModuleType("gr00t.model.policy")
  gr00t_dc.DATA_CONFIG_MAP = {}

  class _FakeGr00tPolicy:
    def __init__(self, **kwargs) -> None:
      self.kwargs = kwargs
      self.model = lambda x: x

    def get_action(self, example):
      assert "state.proprio" in example
      return {"action": np.ones((1, 16, 7), dtype=np.float32) * 0.5}

  gr00t_pol_mod.Gr00tPolicy = _FakeGr00tPolicy
  monkeypatch.setitem(sys.modules, "gr00t", gr00t_mod)
  monkeypatch.setitem(sys.modules, "gr00t.experiment", gr00t_exp)
  monkeypatch.setitem(sys.modules, "gr00t.experiment.data_config", gr00t_dc)
  monkeypatch.setitem(sys.modules, "gr00t.model", gr00t_model)
  monkeypatch.setitem(sys.modules, "gr00t.model.policy", gr00t_pol_mod)

  sync_before = state["tpu_sync_calls"]
  compile_before = state["compile_calls"]
  gr_pol = Gr00tN17Policy(
      model_path="nvidia/GR00T-N1.7-3B",
      device="tpu",
      precision_tag="FP8",
      torch_compile=True,
      warm_start=False,
  )
  assert gr_pol.device == "tpu"
  assert gr_pol._policy.kwargs["device"] == "tpu"
  assert state["compile_calls"] == compile_before + 1
  assert gr_pol.capabilities().server_impl.endswith("-tpu+compile")
  gr_chunk = gr_pol.predict(obs)
  assert gr_chunk.actions.shape == (16, 7)
  assert state["tpu_sync_calls"] == sync_before + 1

  # 5. CosmosPredict25Policy on TorchTPU with torch_compile=True and streaming chunks
  diffusers_mod = types.ModuleType("diffusers")
  diffusers_mod.__version__ = "0.33.1"

  class _FakeCosmosPipe:
    def __init__(self) -> None:
      self.moved_to = None
      self.transformer = lambda x: x

    @classmethod
    def from_pretrained(cls, _model_id, torch_dtype=None):
      inst = cls()
      inst.torch_dtype = torch_dtype
      return inst

    def to(self, dev: str):
      self.moved_to = dev
      return self

    def __call__(self, *, num_frames: int, height: int, width: int, **_kw):
      return types.SimpleNamespace(frames=np.zeros((1, num_frames, height, width, 3), dtype=np.uint8))

  diffusers_mod.Cosmos25VideoToWorldPipeline = _FakeCosmosPipe
  monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)

  # CLOSED division strictly bans pseudo-causal stream_chunk_frames > 0
  with pytest.raises(ValueError, match="stream_chunk_frames=4 is forbidden in division='CLOSED'"):
    CosmosPredict25Policy(
        device="tpu",
        precision_tag="FP8",
        num_frames=16,
        frame_hw=(32, 32),
        stream_chunk_frames=4,
        torch_compile=True,
        warm_start=False,
    )

  sync_before = state["tpu_sync_calls"]
  compile_before = state["compile_calls"]
  cos_pol = CosmosPredict25Policy(
      device="tpu",
      precision_tag="FP8",
      num_frames=16,
      frame_hw=(32, 32),
      stream_chunk_frames=4,
      division="OPEN",
      torch_compile=True,
      warm_start=False,
  )
  assert cos_pol.device == "tpu"
  assert cos_pol._pipe.moved_to == "tpu"
  assert state["compile_calls"] == compile_before + 1
  assert cos_pol.capabilities().server_impl.endswith("-tpu+compile")

  ctx = Context(
      session_id="wm-tpu",
      frames=[np.zeros((32, 32, 3), dtype=np.uint8)],
      proprio=np.zeros((8,), dtype=np.float32),
      instruction="push block",
      num_frames=16,
  )
  frames = list(cos_pol.rollout(ctx, np.zeros((16, 7), dtype=np.float32)))
  assert len(frames) == 16
  # 16 frames / stream_chunk_frames=4 -> 4 chunk barrier syncs on TorchTPU
  assert state["tpu_sync_calls"] == sync_before + 4


def test_envelope_and_host_detectors_prioritize_torchtpu(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `eeb/envelope/check.py`, `SingleChipHardwareVerifier`, and `detect_host_accelerator` prefer TorchTPU before JAX/tpu-info."""
  import importlib.util
  import os
  import sys
  import types

  import eeb.envelope.check as env_check
  from embodied_bench.agent_kickoff import detect_host_accelerator
  from embodied_bench.harness.hardware_verifier import SingleChipHardwareVerifier

  # Isolate host GPU / CLI probes so only TPU paths are exercised
  monkeypatch.setattr(env_check, "which", lambda _cmd: None)
  monkeypatch.setattr(env_check.os.path, "exists", lambda _p: False)
  monkeypatch.delenv("EEB_HARDWARE_SKU", raising=False)

  real_find_spec = importlib.util.find_spec

  def _no_jax_find_spec(name: str, package: str | None = None):
    if name in ("jax", "torch_tpu", "torch"):
      return None
    return real_find_spec(name, package)

  monkeypatch.setattr(importlib.util, "find_spec", _no_jax_find_spec)
  monkeypatch.setitem(sys.modules, "jax", None)

  # 1. Cloud TPU v6e-1 (32 GiB) passes with TorchTPU alone (no jax, no tpu-info)
  torch_v6e, torchtpu_v6e, _, _ = _make_mock_torch_and_torchtpu(
      tpu_available=True,
      tpu_device_count=1,
      tpu_device_name="Cloud TPU v6e-1",
      tpu_total_memory_bytes=32 * 1024**3,
      attach_tpu_on_import=True,
  )
  monkeypatch.setitem(sys.modules, "torch", torch_v6e)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_v6e)

  res_v6e = env_check.check_envelope(hardware_sku="TPU_V6E_1_32GB")
  assert res_v6e.passed is True
  assert res_v6e.num_devices == 1
  assert res_v6e.device is not None
  assert res_v6e.device["vendor"] == "GOOGLE_TPU"
  assert res_v6e.device["source"] == "torch.tpu"
  assert res_v6e.measured_total_memory_mb == pytest.approx(32768.0)
  assert res_v6e.violations == []

  verifier = SingleChipHardwareVerifier(hardware_sku="TPU_V6E_1_32GB", allow_live_probe=True)
  assert verifier.detect_physical_device_count() == 1

  host_v6e = detect_host_accelerator(probe_live=True)
  assert host_v6e.hardware_sku == "TPU_V6E_1_32GB"
  assert host_v6e.detection_source == "torchtpu_live"
  assert host_v6e.detected_device_count == 1
  assert host_v6e.qualifies_cap_single_chip is True

  # 2. Cloud TPU v5e-1 (16 GiB) passes and maps to TPU_V5E_1_16GB
  torch_v5e, torchtpu_v5e, _, _ = _make_mock_torch_and_torchtpu(
      tpu_available=True,
      tpu_device_count=1,
      tpu_device_name="Cloud TPU v5e-1",
      tpu_total_memory_bytes=16 * 1024**3,
  )
  monkeypatch.setitem(sys.modules, "torch", torch_v5e)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_v5e)

  res_v5e = env_check.check_envelope(hardware_sku="TPU_V5E_1_16GB")
  assert res_v5e.passed is True
  assert res_v5e.measured_total_memory_mb == pytest.approx(16384.0)
  host_v5e = detect_host_accelerator(probe_live=True)
  assert host_v5e.hardware_sku == "TPU_V5E_1_16GB"
  assert host_v5e.detection_source == "torchtpu_live"

  # 3. Negative Control: 4 visible TPU chips fails single-chip envelope with actionable reason
  torch_4x, torchtpu_4x, _, _ = _make_mock_torch_and_torchtpu(
      tpu_available=True,
      tpu_device_count=4,
      tpu_device_name="Cloud TPU v6e",
      tpu_total_memory_bytes=32 * 1024**3,
  )
  monkeypatch.setitem(sys.modules, "torch", torch_4x)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_4x)

  res_4x = env_check.check_envelope(hardware_sku="TPU_V6E_1_32GB")
  assert res_4x.passed is False
  assert res_4x.num_devices == 4
  assert res_4x.reason
  assert any("num_devices == 1" in v and "TPU_VISIBLE_DEVICES=0" in v for v in res_4x.violations)
  assert verifier.detect_physical_device_count() == 4
  host_4x = detect_host_accelerator(probe_live=True)
  assert host_4x.detected_device_count == 4
  assert host_4x.qualifies_cap_single_chip is False

  # 4. Fallback to `jax.devices()` when `torch_tpu` is absent
  monkeypatch.setitem(sys.modules, "torch", None)
  monkeypatch.setitem(sys.modules, "torch_tpu", None)
  jax_mod = types.ModuleType("jax")
  fake_jax_dev = types.SimpleNamespace(
      platform="tpu",
      id=0,
      device_kind="TPU v6 lite",
      memory_stats=lambda: {"bytes_limit": 32 * 1024**3},
  )
  jax_mod.devices = lambda: [fake_jax_dev]
  monkeypatch.setitem(sys.modules, "jax", jax_mod)

  res_jax = env_check.check_envelope(hardware_sku="TPU_V6E_1_32GB")
  assert res_jax.passed is True
  assert res_jax.device is not None
  assert res_jax.device["source"] == "jax.devices()"


def test_torchtpu_crash_repeat_and_concurrent_paths(monkeypatch: pytest.MonkeyPatch) -> None:
  """Gate 1 Three Paths: Crash-mid-enumeration, repeated cross-mode calls, and 8-worker concurrent invocations."""
  from concurrent.futures import ThreadPoolExecutor
  import importlib.util
  import sys
  import types

  import eeb.envelope.check as env_check
  from eeb.servers.torch_backend import resolve_torch_device, synchronize_torch_device

  monkeypatch.setattr(env_check, "which", lambda _cmd: None)
  monkeypatch.setattr(env_check.os.path, "exists", lambda _p: False)
  monkeypatch.setattr(importlib.util, "find_spec", lambda _name, _pkg=None: None)
  monkeypatch.setitem(sys.modules, "jax", None)

  # 1. Crash path: `torch.tpu.device_count()` raises mid-enumeration -> clean fallback, 0 partial devices
  broken_torch = types.ModuleType("torch")
  broken_tpu = types.SimpleNamespace(
      device_count=lambda: (_ for _ in ()).throw(RuntimeError("libtpu initialization failed"))
  )
  broken_torch.tpu = broken_tpu
  monkeypatch.setitem(sys.modules, "torch", broken_torch)
  monkeypatch.setitem(sys.modules, "torch_tpu", types.ModuleType("torch_tpu"))

  crash_inv = env_check.enumerate_devices()
  assert crash_inv.num_devices == 0
  assert any("torch.tpu failed: RuntimeError" in n for n in crash_inv.notes)

  # 2. Repeat path: alternate TPU -> CPU -> allow_no_device without stale cached devices
  torch_mod, torchtpu_mod, _, state = _make_mock_torch_and_torchtpu(
      cuda_available=False,
      tpu_available=True,
      tpu_device_count=1,
      tpu_total_memory_bytes=32 * 1024**3,
  )
  monkeypatch.setitem(sys.modules, "torch", torch_mod)
  monkeypatch.setitem(sys.modules, "torch_tpu", torchtpu_mod)

  assert resolve_torch_device("tpu") == "tpu"
  assert resolve_torch_device("cpu") == "cpu"
  r_live = env_check.check_envelope(hardware_sku="TPU_V6E_1_32GB")
  r_empty = env_check.check_envelope(
      hardware_sku="TPU_V6E_1_32GB",
      inventory=env_check.DeviceInventory(),
      allow_no_device=True,
  )
  r_live_2 = env_check.check_envelope(hardware_sku="TPU_V6E_1_32GB")
  assert r_live.passed is True and r_live.num_devices == 1
  assert r_empty.passed is True and r_empty.num_devices == 0
  assert r_live_2.passed is True and r_live_2.num_devices == 1

  # 3. Concurrent path: 8 threads x 32 tasks invoking resolve + sync + check_envelope concurrently
  def _worker(task_id: int) -> tuple[str, bool, int]:
    dev = resolve_torch_device("auto")
    synchronize_torch_device(dev)
    chk = env_check.check_envelope(hardware_sku="TPU_V6E_1_32GB")
    return (dev, chk.passed, chk.num_devices + (task_id * 0))

  with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(_worker, range(32)))

  assert results == [("tpu", True, 1)] * 32
  assert state["tpu_sync_calls"] == 32







