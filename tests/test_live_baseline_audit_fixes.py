"""Regression and adversarial self-review tests for live baseline legitimacy audit fixes.

Covers:
  - [P0-3] Active-cohort DynamicMicroBatcher (zero dead-sleep wait at C=1 and C=4, cohort shrink
    on reset/TTL expiry, exception propagation without deadlock, concurrent thread safety).
  - [P0-4] Native 224x224 / 256x256 `eeb data prepare-traces` CLI (crash-leftover cleanup, repeat
    overwrite without stale shards, concurrent thread + subprocess flock safety, reversed flag
    order) and non-synthetic wire-resolution admissibility gates in report/CLI/leaderboard.
  - [P0-5] BF16-safe Gate G4 and inline perturbation canaries (multi-ULP vector steps + 10% wire
    image pixel perturbation surviving exact IEEE BF16 rounding while rejecting static and
    vision-blind memoizers).
  - [P0-6] Protocol timed-call floor (`320` VLA = >=20 samples/slot at C=16; `200` WM).
  - [P1-1] & [P1-2] Hardware SKU specs (`RTX_4090_24GB = 82.6` FP16-FP32-accum TFLOPS vs
    `L4_24GB = 121.0`), canonical model contracts (`pi05-3b` H=10/K=5/224, `gr00t-n1.7-3b`
    Cosmos-Reason2-2B H=16/K=8/256), `L4_24GB` baseline recipes, and Closed-Division guards.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Dict, Iterator

import numpy as np
import pytest

from eeb.cli import main as eeb_main
from eeb.envelope.check import ENVELOPE_MAX_BF16_TFLOPS, ENVELOPE_MAX_FP8_INT8_TOPS, ENVELOPE_MAX_MEM_BW_TBPS, ENVELOPE_MAX_MEMORY_GB, EnvelopeCheckResult, sku_spec_summary
from eeb.loadgen.client import InProcessPolicyClient
from eeb.loadgen.gate import (
    BF16_SAFE_PERTURB_STEP,
    MIN_ADMISSIBLE_WIRE_IMAGE_BYTES,
    MIN_ADMISSIBLE_WIRE_RESOLUTION_HW,
    evaluate_wire_observation_resolution,
    run_gate,
    run_live_g4_perturbation_probe,
)
from eeb.loadgen.report import ReportInputs, build_report
from eeb.loadgen.sweep import (
    PROTOCOL_MIN_TIMED_CALLS,
    PROTOCOL_MIN_TIMED_CALLS_WM,
    ConcurrencySweep,
    ObservationSource,
    ObservationSpec,
    SweepConfig,
    TraceBundleObservationSource,
)
from eeb.protocol.policy import ActionChunk, BasePolicy, Caps, Context, Frame, Observation
from eeb.servers.base import DynamicMicroBatcher
from embodied_bench.agent_kickoff import run_agent_kickoff
from embodied_bench.attestation import create_submission_receipt
from embodied_bench.dataset_loader import EmbodiedXDatasetLoader
from embodied_bench.leaderboard_compiler import _audit_single_receipt
from embodied_bench.schema import (
    CLOSED_DIVISION_FORBIDDEN_KEYWORDS,
    HARDWARE_SKU_SPECS,
    MODERN_REFERENCE_CHECKPOINTS,
    RecipeConfig,
    atomic_write_json,
    load_yaml_or_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _round_to_bf16_f32(arr: np.ndarray) -> np.ndarray:
  """Exact IEEE-754 float32 -> bfloat16 round-to-nearest-even projected back to float32."""
  f32 = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
  u32 = f32.view(np.uint32)
  nan_mask = np.isnan(f32)
  lsb = (u32 >> np.uint32(16)) & np.uint32(1)
  rounded = (u32 + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
  out = rounded.view(np.float32).copy()
  if np.any(nan_mask):
    out[nan_mask] = np.nan
  return out


class BF16QuantizedVLAPolicy(BasePolicy):
  """Realistic non-synthetic VLA policy stub that rounds all normalized inputs to exact BF16."""

  def __init__(self, image_hw: int = 224) -> None:
    self._hw = int(image_hw)

  def capabilities(self) -> Caps:
    return Caps(
        track="VLA_TRACK",
        horizon=10,
        execute_horizon=5,
        dt_ms=50.0,
        precision="BF16",
        model_id="pi05-3b",
        server_impl="openpi-bf16-test-0.1.0",
        action_dim=7,
        proprio_dim=8,
        cameras=("agentview", "wrist"),
        image_height=self._hw,
        image_width=self._hw,
    )

  def predict(self, obs: Observation) -> ActionChunk:
    t0 = time.perf_counter()
    prop_bf16 = _round_to_bf16_f32(obs.proprio + np.float32(1.0)) - np.float32(1.0)
    vis_feats = np.zeros((7,), dtype=np.float32)
    for idx, cam in enumerate(sorted(obs.images.keys())):
      # Normalize uint8 RGB into [1.0, 2.0] before BF16 quantization (worst-case unit-scale ULP = 2^-7)
      norm = np.float32(1.0) + obs.images[cam].astype(np.float32) / np.float32(255.0)
      img_bf16 = _round_to_bf16_f32(norm) - np.float32(1.5)
      flat = img_bf16.reshape(-1)
      usable = (flat.size // 7) * 7
      pooled = flat[:usable].reshape(7, -1).mean(axis=1)
      vis_feats += pooled * np.float32(1.6 + 0.4 * idx)

    prop_pad = np.zeros((7,), dtype=np.float32)
    n_p = min(7, prop_bf16.size)
    prop_pad[:n_p] = prop_bf16[:n_p]
    base_vec = np.tanh(prop_pad * np.float32(0.8) + vis_feats)
    t_axis = np.linspace(0.8, 1.2, 10, dtype=np.float32)[:, None]
    actions = _round_to_bf16_f32(t_axis * base_vec[None, :])
    t1 = time.perf_counter()
    return ActionChunk(actions=actions, t_recv=t0, t_done=t1)


class VisionBlindBF16VLAPolicy(BF16QuantizedVLAPolicy):
  """Negative-control policy that responds to proprioception but completely ignores camera images."""

  def predict(self, obs: Observation) -> ActionChunk:
    t0 = time.perf_counter()
    prop_bf16 = _round_to_bf16_f32(obs.proprio + np.float32(1.0)) - np.float32(1.0)
    prop_pad = np.zeros((7,), dtype=np.float32)
    n_p = min(7, prop_bf16.size)
    prop_pad[:n_p] = prop_bf16[:n_p]
    t_axis = np.linspace(0.8, 1.2, 10, dtype=np.float32)[:, None]
    actions = _round_to_bf16_f32(t_axis * np.tanh(prop_pad * np.float32(1.5))[None, :])
    return ActionChunk(actions=actions, t_recv=t0, t_done=time.perf_counter())


class StaticMemoizingVLAPolicy(BF16QuantizedVLAPolicy):
  """Negative-control policy that returns a fixed cached trajectory regardless of input perturbations."""

  def predict(self, obs: Observation) -> ActionChunk:
    del obs
    t0 = time.perf_counter()
    t_axis = np.linspace(-0.2, 0.2, 10, dtype=np.float32)[:, None]
    actions = np.repeat(t_axis, 7, axis=1).astype(np.float32)
    return ActionChunk(actions=actions, t_recv=t0, t_done=time.perf_counter())


class BF16QuantizedWMPolicy(BasePolicy):
  """Realistic non-synthetic Track 2 world-model stub that quantizes frames and actions to BF16."""

  def __init__(self, image_hw: int = 256) -> None:
    self._hw = int(image_hw)

  def capabilities(self) -> Caps:
    return Caps(
        track="WM_TRACK",
        horizon=16,
        execute_horizon=16,
        dt_ms=1000.0 / 24.0,
        precision="BF16",
        model_id="cosmos-predict2.5-2b",
        server_impl="cosmos-bf16-test-0.1.0",
        action_dim=7,
        proprio_dim=8,
        cameras=("agentview",),
        image_height=self._hw,
        image_width=self._hw,
    )

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    n_frames = int(ctx.num_frames or 16)
    base_img = (
        ctx.frames[0].astype(np.float32)
        if ctx.frames
        else np.full((self._hw, self._hw, 3), 127.0, dtype=np.float32)
    )
    cond_bf16 = _round_to_bf16_f32(np.float32(1.0) + base_img / np.float32(255.0)) - np.float32(1.0)
    act_bf16 = _round_to_bf16_f32(actions + np.float32(1.0)) - np.float32(1.0)
    prop_bf16 = _round_to_bf16_f32(ctx.proprio + np.float32(1.0)) - np.float32(1.0)
    prop_scalar = float(np.mean(prop_bf16)) if prop_bf16.size else 0.0
    vis_scalar = float(np.mean(cond_bf16))

    for i in range(n_frames):
      act_row = act_bf16[i % len(act_bf16)] if len(act_bf16) else np.zeros((7,), dtype=np.float32)
      shift = np.float32(float(np.mean(act_row)) * 0.25 + prop_scalar * 0.15 + vis_scalar * 0.20 + i * 0.005)
      frame_f = np.clip((cond_bf16 + shift) * np.float32(255.0), 0.0, 255.0)
      yield Frame(
          index=i,
          image=np.rint(frame_f).astype(np.uint8),
          t_done=time.perf_counter(),
          camera="wm",
      )


def _passing_single_chip_envelope(sku: str = "L4_24GB") -> EnvelopeCheckResult:
  spec_full = HARDWARE_SKU_SPECS[sku]
  dev = {
      "index": 0,
      "name": spec_full["display_name"],
      "vendor": spec_full["vendor"],
      "total_memory_mb": float(spec_full["vram_mb"]),
      "source": "unit-test",
      "uuid": "GPU-TEST-0",
  }
  return EnvelopeCheckResult(
      passed=True,
      num_devices=1,
      reason="single-chip envelope verified (unit test)",
      hardware_sku=sku,
      device=dev,
      measured_total_memory_mb=float(spec_full["vram_mb"]),
      memory_provenance="MEASURED",
      spec=sku_spec_summary(sku),
      spec_provenance="SPEC_TABLE",
      envelope_limits={
          "max_memory_gb": ENVELOPE_MAX_MEMORY_GB,
          "max_bf16_tflops": ENVELOPE_MAX_BF16_TFLOPS,
          "max_fp8_int8_tops": ENVELOPE_MAX_FP8_INT8_TOPS,
          "max_mem_bw_tbps": ENVELOPE_MAX_MEM_BW_TBPS,
      },
      violations=[],
      inventory={"num_devices": 1, "devices": [dev]},
  )


# =============================================================================
# [P0-5] BF16 Sub-ULP Perturbation Canary Fix (Gate 4 lookalikes + violations)
# =============================================================================


def test_p05_bf16_rounding_swallows_1e3_but_preserves_safe_step() -> None:
  """Prove mathematically why delta=1e-3 failed on BF16 and BF16_SAFE_PERTURB_STEP=1e-2 survives."""
  base = np.array([1.0, -1.0, 0.85, -0.75], dtype=np.float32)
  swallowed = _round_to_bf16_f32(base + np.float32(1e-3))
  assert np.array_equal(swallowed, _round_to_bf16_f32(base))

  assert BF16_SAFE_PERTURB_STEP >= 1e-2
  preserved = _round_to_bf16_f32(base + np.float32(BF16_SAFE_PERTURB_STEP))
  assert np.all(preserved != _round_to_bf16_f32(base))


def test_p05_bf16_quantized_vla_passes_live_g4_and_inline_canaries() -> None:
  """Lookalike PASS: genuine BF16-quantizing VLA policy passes live G4 AND inline sweep canaries."""
  policy = BF16QuantizedVLAPolicy(image_hw=224)
  client = InProcessPolicyClient(policy)
  g4 = run_live_g4_perturbation_probe(client=client, caps=policy.capabilities(), execution_id="bf16-vla-g4")
  assert g4["passed"] is True, g4
  assert g4["vision_only_passed"] is True, g4
  assert g4["min_output_l2_diff"] > 0.0

  cfg = SweepConfig(
      concurrency=[1, 2],
      warmup_calls=2,
      timed_calls=16,
      runs=1,
      max_variance_pct=100.0,
      inline_canaries_enabled=True,
      server_warmup_calls=1,
  )
  src = ObservationSource(ObservationSpec.from_caps(policy.capabilities(), pool_size=4))
  res = ConcurrencySweep(endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy", config=cfg, policy=policy, observation_source=src).run()
  for pt in res.points:
    assert pt.runs[0].num_inline_canaries >= 1
    assert pt.runs[0].num_inline_canary_failures == 0, pt.runs[0].inline_canary_reasons
    assert pt.all_runs_passed is True


def test_p05_bf16_quantized_wm_passes_live_g4_and_inline_canaries() -> None:
  """Lookalike PASS: genuine BF16-quantizing Track 2 WM policy passes live G4 AND inline canaries."""
  policy = BF16QuantizedWMPolicy(image_hw=64)
  client = InProcessPolicyClient(policy)
  g4 = run_live_g4_perturbation_probe(client=client, caps=policy.capabilities(), execution_id="bf16-wm-g4", num_probes=2)
  assert g4["passed"] is True, g4
  assert g4["vision_only_passed"] is True, g4

  cfg = SweepConfig(
      concurrency=[1],
      warmup_calls=1,
      timed_calls=8,
      runs=1,
      max_variance_pct=100.0,
      inline_canaries_enabled=True,
      server_warmup_calls=1,
  )
  src = ObservationSource(ObservationSpec.from_caps(policy.capabilities(), pool_size=4))
  res = ConcurrencySweep(endpoint="inproc:eeb.servers.dummy:DummyWMPolicy", config=cfg, policy=policy, observation_source=src).run()
  assert res.points[0].runs[0].num_inline_canaries >= 1
  assert res.points[0].runs[0].num_inline_canary_failures == 0, res.points[0].runs[0].inline_canary_reasons


def test_p05_negative_controls_reject_static_and_vision_blind_memoizers() -> None:
  """Negative controls: static cache hit policy AND vision-blind policy both fail with reasons."""
  static_pol = StaticMemoizingVLAPolicy(image_hw=224)
  g4_static = run_live_g4_perturbation_probe(
      client=InProcessPolicyClient(static_pol),
      caps=static_pol.capabilities(),
      execution_id="neg-static",
  )
  assert g4_static["passed"] is False
  assert g4_static["cache_hit_suspected"] is True
  assert g4_static["message"].strip()

  cfg = SweepConfig(
      concurrency=[1],
      warmup_calls=1,
      timed_calls=10,
      runs=1,
      max_variance_pct=100.0,
      inline_canaries_enabled=True,
      server_warmup_calls=0,
  )
  src = ObservationSource(ObservationSpec.from_caps(static_pol.capabilities(), pool_size=4))
  res = ConcurrencySweep(endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy", config=cfg, policy=static_pol, observation_source=src).run()
  assert res.points[0].runs[0].num_inline_canary_failures >= 1
  assert res.points[0].runs[0].inline_canary_reasons
  assert res.points[0].all_runs_passed is False

  blind_pol = VisionBlindBF16VLAPolicy(image_hw=224)
  g4_blind = run_live_g4_perturbation_probe(
      client=InProcessPolicyClient(blind_pol),
      caps=blind_pol.capabilities(),
      execution_id="neg-blind",
  )
  assert g4_blind["vision_only_passed"] is False
  assert g4_blind["passed"] is False
  assert "vision" in g4_blind["message"].lower()


# =============================================================================
# [P0-3] Active-Cohort DynamicMicroBatcher (Gate 1 crash / repeat / concurrent)
# =============================================================================


def test_p03_microbatcher_zero_dead_sleep_at_c1_c4_and_cohort_shrink() -> None:
  """At C=1 and C=4 with a huge 60 ms batch_window_ms, active-cohort tracking dispatches in < 8 ms."""
  batch_sizes = []
  lock = threading.Lock()

  def _batch_fn(items):
    with lock:
      batch_sizes.append(len(items))
    return [x * 2 for x in items]

  batcher = DynamicMicroBatcher(_batch_fn, max_batch_size=16, batch_window_ms=60.0, session_ttl_s=0.08)

  # 1. C=1 warmup + steady state: after 1.5 ms initial cohort discovery, wait is ~0 ms (<< 60 ms)
  assert batcher.submit(10, session_id="sess-0") == 20
  c1_latencies_ms = []
  for i in range(12):
    t0 = time.perf_counter()
    assert batcher.submit(i, session_id="sess-0") == i * 2
    c1_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
  assert statistics.median(c1_latencies_ms) < 8.0, c1_latencies_ms
  assert batcher.active_session_count() == 1

  # 2. C=4 synchronized cohort: discovers all 4 sessions and immediately fires batch size 4
  batcher.on_reset()
  barrier = threading.Barrier(4)
  c4_latencies_ms: Dict[int, list[float]] = {k: [] for k in range(4)}

  def _worker(slot: int) -> None:
    sid = f"c4-sess-{slot}"
    for step in range(8):
      barrier.wait(timeout=2.0)
      t0 = time.perf_counter()
      out = batcher.submit(slot * 100 + step, session_id=sid)
      dt_ms = (time.perf_counter() - t0) * 1000.0
      assert out == (slot * 100 + step) * 2
      if step >= 1:
        c4_latencies_ms[slot].append(dt_ms)

  threads = [threading.Thread(target=_worker, args=(k,)) for k in range(4)]
  for t in threads:
    t.start()
  for t in threads:
    t.join(timeout=5.0)
    assert not t.is_alive()

  flat_c4 = [x for vals in c4_latencies_ms.values() for x in vals]
  assert statistics.median(flat_c4) < 15.0, flat_c4
  assert batcher.max_observed_batch_size == 4
  assert batcher.active_session_count() == 4

  # 3. Repeat path: explicit on_reset shrinks cohort 4 -> 1 so C=1 immediately avoids 60 ms waits
  for k in (1, 2, 3):
    batcher.on_reset(f"c4-sess-{k}")
  assert batcher.active_session_count() == 1
  post_reset_ms = []
  for i in range(6):
    t0 = time.perf_counter()
    assert batcher.submit(i, session_id="c4-sess-0") == i * 2
    post_reset_ms.append((time.perf_counter() - t0) * 1000.0)
  assert statistics.median(post_reset_ms) < 8.0, post_reset_ms

  # 4. TTL expiry path: concurrent 4-session cohort drops out after session_ttl_s
  batcher.on_reset()
  ttl_barrier = threading.Barrier(4)

  def _ttl_worker(k: int) -> None:
    ttl_barrier.wait(timeout=2.0)
    assert batcher.submit(k, session_id=f"ttl-{k}") == k * 2

  ttl_threads = [threading.Thread(target=_ttl_worker, args=(k,)) for k in range(4)]
  for t in ttl_threads:
    t.start()
  for t in ttl_threads:
    t.join(timeout=3.0)
  assert batcher.active_session_count() == 4
  time.sleep(0.12)
  assert batcher.active_session_count() == 0


def test_p03_microbatcher_crash_recovery_and_high_concurrency_routing() -> None:
  """Gate 1 crash + concurrent paths: exceptions never deadlock waiters and outputs never cross."""
  mode = {"state": "boom"}

  def _flaky_batch_fn(items):
    if mode["state"] == "boom":
      raise RuntimeError("simulated device OOM")
    if mode["state"] == "short":
      return items[:-1]
    time.sleep(0.001)
    return [(sid, step, step * 3 + 1) for (sid, step) in items]

  batcher = DynamicMicroBatcher(_flaky_batch_fn, max_batch_size=8, batch_window_ms=2.0)

  with pytest.raises(RuntimeError, match="simulated device OOM"):
    batcher.submit(("s0", 1), session_id="s0")

  mode["state"] = "short"
  with pytest.raises(RuntimeError, match="returned 0 outputs for batch of size 1"):
    batcher.submit(("s0", 2), session_id="s0")

  # Recovery + 12 concurrent worker threads x 20 calls
  mode["state"] = "ok"
  batcher.on_reset()

  def _caller(tid: int) -> None:
    sid = f"worker-{tid}"
    for step in range(20):
      got_sid, got_step, got_val = batcher.submit((sid, step), session_id=sid)
      assert got_sid == sid
      assert got_step == step
      assert got_val == step * 3 + 1

  with ThreadPoolExecutor(max_workers=12) as pool:
    futs = [pool.submit(_caller, tid) for tid in range(12)]
    for fut in futs:
      fut.result(timeout=10.0)

  assert batcher.num_items >= 12 * 20 + 2


# =============================================================================
# [P0-4] Native Trace Preparation CLI & Wire-Resolution Admissibility Gate
# =============================================================================


def test_p04_prepare_traces_crash_repeat_concurrent_and_flag_orders(tmp_path: Path) -> None:
  """Gate 1 + Gate 3: orphan recovery, repeat overwrite, both flag orders, and concurrent flock."""
  out_224 = tmp_path / "eval_224"
  out_224.mkdir(parents=True)
  orphan_tmp = out_224 / ".episode_00.npz.99999.tmp"
  orphan_tmp.write_bytes(b"partial-crash-bytes")
  orphan_shard_tmp = out_224 / ".tmp_shard_episode_00_ab12cd.npz"
  orphan_shard_tmp.write_bytes(b"partial-shard-bytes")

  rc1 = eeb_main([
      "data",
      "prepare-traces",
      "--out",
      str(out_224),
      "--image-hw",
      "224x224",
      "--episodes",
      "2",
      "--steps",
      "4",
      "--quiet",
  ])
  assert rc1 == 0
  assert not orphan_tmp.exists()
  assert not orphan_shard_tmp.exists()
  assert list(out_224.glob("*.lock")) == []

  src_224 = TraceBundleObservationSource(str(out_224), split="public")
  desc_224 = src_224.describe()
  assert desc_224["image_hw"] == [224, 224]
  assert desc_224["episode_count"] == 2
  assert desc_224["bytes_per_request_images"] == 2 * 224 * 224 * 3

  # Verify native spatial texture sharpness (crisp checkerboard/tray edges, not flat wash)
  sample_img = src_224.make("tex-check", 0).images["agentview"].astype(np.float32)
  grad_mag = float(np.mean(np.abs(np.diff(sample_img, axis=0))) + np.mean(np.abs(np.diff(sample_img, axis=1))))
  assert grad_mag > 1.5, f"Expected crisp spatial edges in native 224x224 frame, got grad_mag={grad_mag}"

  # Repeat overwrite onto same --out with fewer episodes and reversed flag order via alias
  rc2 = eeb_main([
      "data",
      "generate-eval-traces",
      "--quiet",
      "--steps",
      "3",
      "--episodes",
      "1",
      "--resolution-hw",
      "256",
      "--output-dir",
      str(out_224),
  ])
  assert rc2 == 0
  assert len(list(out_224.glob("episode_*.npz"))) == 1, "Stale shard survived repeat prepare-traces run"
  ok_man, msg_man = EmbodiedXDatasetLoader(dataset_dir=out_224).verify_manifest()
  assert ok_man is True, msg_man
  src_256 = TraceBundleObservationSource(str(out_224), split="public")
  desc_256 = src_256.describe()
  assert desc_256["image_hw"] == [256, 256]
  assert desc_256["episode_count"] == 1
  assert desc_256["bytes_per_request_images"] == 2 * 256 * 256 * 3

  # Concurrent thread + subprocess writers against shared target directory
  shared_out = tmp_path / "concurrent_224"

  def _thread_writer(seed_val: int) -> int:
    return eeb_main([
        "data",
        "prepare-traces",
        "--out",
        str(shared_out),
        "--image-hw",
        "224x224",
        "--episodes",
        "1",
        "--steps",
        "3",
        "--seed",
        str(seed_val),
        "--quiet",
    ])

  proc = subprocess.Popen(
      [
          sys.executable,
          "-m",
          "eeb.cli",
          "data",
          "prepare-traces",
          "--out",
          str(shared_out),
          "--image-hw",
          "224x224",
          "--episodes",
          "1",
          "--steps",
          "3",
          "--seed",
          "77",
          "--quiet",
      ],
      cwd=str(REPO_ROOT),
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )
  with ThreadPoolExecutor(max_workers=3) as pool:
    rcs = list(pool.map(_thread_writer, [11, 22, 33]))
  sub_rc = proc.wait(timeout=30.0)
  assert sub_rc == 0, proc.stderr.read() if proc.stderr else ""
  assert rcs == [0, 0, 0]
  assert list(shared_out.glob("*.lock")) == []
  assert list(shared_out.glob("*.tmp")) == []
  ok_shared, msg_shared = EmbodiedXDatasetLoader(dataset_dir=shared_out).verify_manifest()
  assert ok_shared is True, msg_shared

  # Regression guard: 64x64 CI mini bundle remains SHA-256 intact
  ok_mini, msg_mini = EmbodiedXDatasetLoader(dataset_dir=REPO_ROOT / "data" / "embodiedx_traces_v1_mini").verify_manifest()
  assert ok_mini is True, msg_mini


def test_p04_wire_resolution_gate_rejects_64x64_and_admits_native_224_and_256(tmp_path: Path) -> None:
  """Gate 2 + Gate 4: non-synthetic 64x64 run is INVALID / exit 2; native 224x224 run is PENDING."""
  assert MIN_ADMISSIBLE_WIRE_IMAGE_BYTES == 150_528
  assert MIN_ADMISSIBLE_WIRE_RESOLUTION_HW == 224

  mini_dir = REPO_ROOT / "data" / "embodiedx_traces_v1_mini"
  src_64 = TraceBundleObservationSource(str(mini_dir), split="public")
  ok_64, b_64, hw_64, reason_64 = evaluate_wire_observation_resolution(src_64.describe(), is_wm=False)
  assert ok_64 is False
  assert b_64 == 24_576
  assert hw_64 == (64, 64)
  assert "prepare-traces" in reason_64

  ok_wm224, b_wm224, hw_wm224, _ = evaluate_wire_observation_resolution(
      {"image_hw": [224, 224], "channels": 3, "cameras": ["agentview"]},
      is_wm=True,
  )
  assert ok_wm224 is True
  assert b_wm224 == 150_528
  assert hw_wm224 == (224, 224)

  policy = BF16QuantizedVLAPolicy(image_hw=224)
  cfg = SweepConfig(
      concurrency=[1],
      warmup_calls=30,
      timed_calls=PROTOCOL_MIN_TIMED_CALLS,
      runs=3,
      max_variance_pct=100.0,
      inline_canaries_enabled=False,
      server_warmup_calls=1,
      keep_samples=False,
  )
  recipe_dict = load_yaml_or_json(REPO_ROOT / "recipes" / "track1_vla_pi05_l4.yaml")
  env_ok = _passing_single_chip_envelope("L4_24GB")

  # Negative control: real non-synthetic policy fed 64x64 CI thumbnails -> INVALID
  res_bad = ConcurrencySweep(
      endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy",
      config=cfg,
      policy=policy,
      observation_source=src_64,
  ).run()
  gate_bad = run_gate(res_bad.points[0], policy.capabilities(), "exec-bad-64")
  rep_bad = build_report(
      ReportInputs(
          sweep=res_bad,
          recipe=recipe_dict,
          envelope=env_ok,
          gates={1: gate_bad},
          allow_sanity_only=True,
      )
  )
  assert rep_bad["status"] == "INVALID"
  assert "observation wire resolution below protocol minimum" in rep_bad["reason"]

  # Negative control via CLI: `eeb measure --strict` with non-synthetic policy on 64x64 returns exit 2
  cli_bad_out = tmp_path / "cli_bad_64"
  rc_bad = eeb_main([
      "measure",
      "--endpoint",
      f"inproc:{__name__}:BF16QuantizedVLAPolicy",
      "--traces",
      str(mini_dir),
      "--sweep",
      "1",
      "--warmup",
      "1",
      "--calls",
      "4",
      "--runs",
      "2",
      "--max-variance-pct",
      "100",
      "--allow-sanity-only",
      "--allow-no-device",
      "--strict",
      "--quiet",
      "--out",
      str(cli_bad_out),
  ])
  assert rc_bad == 2
  cli_rep = json.loads((cli_bad_out / "report.json").read_text(encoding="utf-8"))
  assert cli_rep["status"] == "INVALID"
  assert any("observation wire resolution below protocol minimum" in f for f in cli_rep["admissibility_findings"])

  # Lookalike PASS: exact same non-synthetic policy fed prepared 224x224 traces -> PENDING
  eval_224_dir = tmp_path / "eval_224_bundle"
  assert eeb_main([
      "data",
      "prepare-traces",
      "--out",
      str(eval_224_dir),
      "--image-hw",
      "224x224",
      "--episodes",
      "1",
      "--steps",
      "4",
      "--quiet",
  ]) == 0
  src_224 = TraceBundleObservationSource(str(eval_224_dir), split="public")
  res_good = ConcurrencySweep(
      endpoint="inproc:eeb.servers.dummy:DummyVLAPolicy",
      config=cfg,
      policy=policy,
      observation_source=src_224,
  ).run()
  gate_good = run_gate(res_good.points[0], policy.capabilities(), "exec-good-224")
  rep_good = build_report(
      ReportInputs(
          sweep=res_good,
          recipe=recipe_dict,
          envelope=env_ok,
          gates={1: gate_good},
          allow_sanity_only=True,
      )
  )
  assert rep_good["status"] == "PENDING", rep_good["admissibility_findings"]

  # Synthetic DummyVLAPolicy exemption: CI 64x64 dummy smoke runs with --golden still return exit 0 with --strict
  dummy_golden = tmp_path / "dummy_64_golden.npz"
  assert eeb_main([
      "data",
      "generate-golden",
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--traces",
      str(mini_dir),
      "--num-samples",
      "100",
      "--out",
      str(dummy_golden),
      "--quiet",
  ]) == 0
  cli_dummy_out = tmp_path / "cli_dummy_64"
  rc_dummy = eeb_main([
      "measure",
      "--endpoint",
      "inproc:eeb.servers.dummy:DummyVLAPolicy",
      "--traces",
      str(mini_dir),
      "--golden",
      str(dummy_golden),
      "--sweep",
      "1",
      "--warmup",
      "1",
      "--calls",
      "4",
      "--runs",
      "2",
      "--max-variance-pct",
      "100",
      "--allow-no-device",
      "--strict",
      "--quiet",
      "--out",
      str(cli_dummy_out),
  ])
  assert rc_dummy == 0


def test_p04_leaderboard_compiler_audits_wire_resolution_in_receipts(tmp_path: Path) -> None:
  """Leaderboard receipt auditor rejects non-synthetic 64x64 receipts and accepts 224x224 receipts."""
  base_report = {
      "schema_version": "1.0.0",
      "recipe_id": "test_pi05_l4_closed",
      "track": "VLA_TRACK",
      "model_name": "pi0.5-3B-OpenPi-Native-BF16-L4",
      "baseline_model_id": "pi05-3b",
      "precision": "BF16",
      "compression_tier": "strict_lossless",
      "division": "CLOSED",
      "hardware_sku": "L4_24GB",
      "num_devices": 1,
      "status": "CERTIFIED",
      "reason": "certified organizer hardware run",
      "pareto_frontier": [
          {
              "point_id": "C1",
              "execution_id": "exec-lb-1",
              "hz_react": 20.0,
              "executed_actions_per_sec": 80.0,
              "obs_to_action_p99_ms": 50.0,
              "gate_a_passed": True,
              "co_located_gate_a": {"execution_id": "exec-lb-1", "passed": True},
          }
      ],
      "x_metric_name": "hz_react",
      "y_metric_name": "executed_actions_per_sec",
      "latency_metric_name": "obs_to_action_p99_ms",
  }

  sub_bad = tmp_path / "sub_bad"
  sub_bad.mkdir()
  rec_bad = create_submission_receipt(report=base_report, output_path=None)
  rec_bad["num_devices"] = 1
  atomic_write_json(sub_bad / "receipt.json", rec_bad)
  atomic_write_json(
      sub_bad / "sweep_results.json",
      {
          **base_report,
          "server_provenance": {"is_synthetic": False},
          "observation_source": {"image_hw": [64, 64], "bytes_per_request_images": 24576},
      },
  )
  passed_bad, violations_bad, _ = _audit_single_receipt(sub_bad / "receipt.json")
  assert passed_bad is False
  assert any("Observation wire resolution violation" in v for v in violations_bad), violations_bad

  sub_good = tmp_path / "sub_good"
  sub_good.mkdir()
  rec_good = create_submission_receipt(report=base_report, output_path=None)
  rec_good["num_devices"] = 1
  atomic_write_json(sub_good / "receipt.json", rec_good)
  atomic_write_json(
      sub_good / "sweep_results.json",
      {
          **base_report,
          "server_provenance": {"is_synthetic": False},
          "observation_source": {"image_hw": [224, 224], "bytes_per_request_images": 301056},
      },
  )
  passed_good, violations_good, _ = _audit_single_receipt(sub_good / "receipt.json")
  assert passed_good is True, violations_good


# =============================================================================
# [P0-6], [P1-1], [P1-2] Schema, Hardware SKU, Recipe, & Kickoff Invariants
# =============================================================================


def test_p06_and_p1_schema_recipe_and_kickoff_invariants(tmp_path: Path) -> None:
  """Gate 4 + Gate 5: protocol floors, SKU TFLOPS, model specs, L4 recipes, and Closed guards."""
  assert PROTOCOL_MIN_TIMED_CALLS == 320
  assert PROTOCOL_MIN_TIMED_CALLS_WM == 200

  cfg_200 = SweepConfig(concurrency=[1, 2, 4, 8, 16], warmup_calls=30, timed_calls=200, runs=3)
  assert cfg_200.protocol_compliant_for_track("VLA_TRACK") is False
  assert cfg_200.protocol_compliant_for_track("WM_TRACK") is True
  cfg_320 = SweepConfig(concurrency=[1, 2, 4, 8, 16], warmup_calls=30, timed_calls=320, runs=3)
  assert cfg_320.protocol_compliant_for_track("VLA_TRACK") is True
  assert cfg_320.protocol_compliant_for_track("WM_TRACK") is True
  # Negative control (RT3): non-canonical concurrency slice skipping C=1 fails protocol compliance
  cfg_skip_c1 = SweepConfig(concurrency=[16], warmup_calls=30, timed_calls=320, runs=3)
  assert cfg_skip_c1.protocol_compliant_for_track("VLA_TRACK") is False
  assert cfg_skip_c1.protocol_compliant_for_track("WM_TRACK") is False

  # [P1-2] RTX 4090 FP16-with-FP32-accumulate = 82.6 TFLOPS vs L4 = 121.0 TFLOPS
  assert HARDWARE_SKU_SPECS["RTX_4090_24GB"]["fp16_tflops"] == pytest.approx(82.6)
  assert HARDWARE_SKU_SPECS["RTX_4090"]["fp16_tflops"] == pytest.approx(82.6)
  assert HARDWARE_SKU_SPECS["RTX_4090_24GB"]["fp8_tflops"] == pytest.approx(330.3)
  assert HARDWARE_SKU_SPECS["L4_24GB"]["fp16_tflops"] == pytest.approx(121.0)

  # [P1-1] Canonical model architectures, resolutions, and horizons
  pi05 = MODERN_REFERENCE_CHECKPOINTS["pi05-3b"]
  assert pi05["native_action_chunk_h"] == 10
  assert pi05["execute_horizon_k"] == 5
  assert pi05["default_denoising_steps"] == 10
  assert pi05["resolution_hw"] == 224
  assert "pytorch-torchtpu" in pi05["certified_runtimes"]

  groot = MODERN_REFERENCE_CHECKPOINTS["gr00t-n1.7-3b"]
  assert groot["hf_repo"] == "nvidia/GR00T-N1.7-LIBERO"
  assert "Cosmos-Reason2-2B" in groot["architecture"]
  assert groot["native_action_chunk_h"] == 16
  assert groot["execute_horizon_k"] == 8
  assert groot["default_denoising_steps"] == 16
  assert groot["resolution_hw"] == 256

  cosmos = MODERN_REFERENCE_CHECKPOINTS["cosmos-predict2.5-2b"]
  assert cosmos["default_denoising_steps"] == 10
  assert cosmos["horizon_frames"] == 16
  assert cosmos["resolution_hw"] == 256

  # Closed Division negative controls vs Open/Closed lookalikes
  assert "cpu_text_encoder_offload_cache" in CLOSED_DIVISION_FORBIDDEN_KEYWORDS
  assert "streaming_first_frame_vae_decode" in CLOSED_DIVISION_FORBIDDEN_KEYWORDS

  for forbidden_kw in ("cpu_text_encoder_offload_cache", "streaming_first_frame_vae_decode"):
    with pytest.raises(ValueError, match=forbidden_kw):
      RecipeConfig.from_dict({
          **load_yaml_or_json(REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_l4.yaml"),
          "division": "CLOSED",
          "optimizations": [forbidden_kw],
      })

    ok_open = RecipeConfig.from_dict({
        **load_yaml_or_json(REPO_ROOT / "recipes" / "track2_wm_cosmos_predict25_l4.yaml"),
        "division": "OPEN",
        "compression_tier": "extreme_compression",
        "declared_training_compute_gpu_hours": 0.0,
        "declared_inference_flops_gflops": 120.0,
        "osi_license": "Apache-2.0",
        "optimizations": [forbidden_kw],
    })
    assert forbidden_kw in ok_open.optimizations

  ok_closed_lookalike = RecipeConfig.from_dict({
      **load_yaml_or_json(REPO_ROOT / "recipes" / "track1_vla_gr00t_n17_l4.yaml"),
      "division": "CLOSED",
      "optimizations": ["vlm_kv_cache_prefix_reuse", "torch_compile_inductor_bf16"],
  })
  assert "vlm_kv_cache_prefix_reuse" in ok_closed_lookalike.optimizations

  # Kickoff blueprints wire exact model-specific native resolution and track timed-call floors
  for rel_path, model_id, expected_calls, expected_hw in [
      ("recipes/track1_vla_pi05_l4.yaml", "pi05-3b", 320, 224),
      ("recipes/track1_vla_gr00t_n17_l4.yaml", "gr00t-n1.7-3b", 320, 256),
      ("recipes/track2_wm_cosmos_predict25_l4.yaml", "cosmos-predict2.5-2b", 200, 256),
  ]:
    recipe_file = REPO_ROOT / rel_path
    assert recipe_file.exists()
    loaded_recipe = RecipeConfig.from_dict(load_yaml_or_json(recipe_file))
    assert loaded_recipe.hardware_sku == "L4_24GB"
    assert loaded_recipe.resolution_hw == expected_hw

    study_dir = tmp_path / recipe_file.stem
    plan = run_agent_kickoff(
        model_id=model_id,
        hardware_sku="L4_24GB",
        division="CLOSED",
        study_dir=study_dir,
    )
    assert plan["passed"] is True
    gen_recipe = RecipeConfig.from_dict(load_yaml_or_json(Path(plan["recipe_path"])))
    assert gen_recipe.resolution_hw == expected_hw
    assert gen_recipe.reference_resolution_hw == expected_hw

    m1_cmds = "\n".join(plan["milestones"]["M1_HARDWARE_AND_DATA_PREFLIGHT"]["cli_commands"])
    assert f"eeb data prepare-traces --out {study_dir / 'eval_traces'} --image-hw {expected_hw}x{expected_hw}" in m1_cmds

    expanded_md = Path(plan["expanded_prompt_path"]).read_text(encoding="utf-8")
    assert "prepare-traces" in expanded_md
    assert f">= {expected_calls}" in expanded_md
    assert "lsb_delta=4" in expanded_md
