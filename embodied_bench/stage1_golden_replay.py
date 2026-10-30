"""Stage 1 (<3 min) Golden Tensor Replay, EEB Measurement Protocol (`eeb measure`), Device-Agnostic Performance Cap Envelopes, Co-Located Gate A & Compliance Gates G1-G4 evaluation runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from embodied_bench.adapters import build_adapter
from embodied_bench.golden_dataset import (
    Track1GoldenBatch,
    Track2GoldenBatch,
    generate_track1_golden_batch,
    generate_track2_golden_batch,
)
from embodied_bench.measure import (
    compute_semianalysis_normalizations,
    evaluate_performance_cap_envelope,
)
from embodied_bench.pareto import (
    compute_iso_reactivity_slices,
    extract_pareto_frontier,
    extract_semianalysis_4way_pareto_frontiers,
)
from embodied_bench.quality_gates import (
    evaluate_compliance_suite_g1_g4,
    evaluate_vla_gate_a,
    evaluate_wm_gate_a,
    evaluate_wm_open_3part_quality_gate,
)
from embodied_bench.scenarios import evaluate_recipe_scenario, evaluate_s2_wm_evaluator
from embodied_bench.schema import (
    ANALYTIC_REFERENCE_PROVENANCE,
    LIVE_EMPIRICAL_PROVENANCE,
    RecipeConfig,
    RunStatus,
    atomic_write_json,
    is_wm_track,
    load_yaml_or_json,
    now_pacific_iso,
    validate_status_and_reason,
)
from embodied_bench.stage2_closed_loop_verify import (
    run_stage2_libero_rt_and_g2,
    run_stage2_verification,
)
from embodied_bench.telemetry import (
    TelemetryMonitor,
    run_eeb_measurement_protocol,
)


def _make_execution_id(recipe_id: str, point_tag: str) -> str:
  digest = hashlib.sha256(f"{recipe_id}:{point_tag}".encode("utf-8")).hexdigest()[:12]
  return f"exec-{recipe_id}-{point_tag}-{digest}"


def run_stage1_golden_replay(
    recipe: RecipeConfig,
    include_stage2: bool = True,
    output_path: Optional[str | Path] = None,
    warmup_calls: int = 30,
    timed_calls: int = 320,
    num_runs: int = 3,
    perf_cap: Optional[str] = None,
    perf_cap_mode: Optional[str] = None,
    mps_active_thread_pct: Optional[float] = None,
    clock_scale_factor: Optional[float] = None,
    vram_fraction_cap: Optional[float] = None,
    live_runner: bool = False,
    server_url: Optional[str] = None,
) -> Dict[str, Any]:
  """Run Stage 1 Golden Tensor Replay + Co-Located Gate A + G1-G4 Compliance + EEB Measurement Protocol + PerformanceCapEnvelope evaluation."""
  if perf_cap is not None:
    recipe.perf_cap = perf_cap
  if perf_cap_mode is not None:
    recipe.perf_cap_mode = perf_cap_mode
  if mps_active_thread_pct is not None:
    recipe.mps_active_thread_pct = mps_active_thread_pct
  if clock_scale_factor is not None:
    recipe.clock_scale_factor = clock_scale_factor
  if vram_fraction_cap is not None:
    recipe.vram_fraction_cap = vram_fraction_cap

  recipe.validate()
  adapter = build_adapter(recipe)
  telemetry_monitor = TelemetryMonitor(recipe.hardware_sku, allow_live_probe=False)
  wm_mode = is_wm_track(recipe.track)
  is_live_mode = bool(live_runner or server_url)

  operating_points: List[Dict[str, Any]] = []
  all_gate_a_passed = True
  first_failure_reason: Optional[str] = None

  # Track reference and candidate arrays on primary operating point for G1-G4 compliance audit
  primary_exec_id = ""
  primary_head_type = "world_model" if wm_mode else recipe.action_head_type
  primary_ref_out: Optional[np.ndarray] = None
  primary_cand_out: Optional[np.ndarray] = None
  primary_pert_cand_out: Optional[np.ndarray] = None
  primary_pert_ref_out: Optional[np.ndarray] = None
  primary_ref_la: Optional[np.ndarray] = None
  primary_cand_la: Optional[np.ndarray] = None
  primary_gate_a_metrics: Dict[str, float] = {}

  if wm_mode:
    x_key = "hz_plan"
    y_key = "predictions_per_sec"
    latency_key = "rollout_latency_p99_ms"
    mean_latency_key = "rollout_latency_mean_ms"
    p50_latency_key = "rollout_latency_p50_ms"

    for idx, b in enumerate(recipe.batch_sizes):
      point_tag = f"B{b}_H{recipe.horizon_frames}_N{recipe.denoising_steps}"
      exec_id = _make_execution_id(recipe.recipe_id, point_tag)
      golden = generate_track1_golden_batch(
          batch_size=b, horizon_frames=recipe.horizon_frames
      )
      cand_video, cand_latent_actions, timing = adapter.run_track1_rollout(golden)
      hw_snap = telemetry_monitor.measure_operating_point(
          model_params_b=adapter.params_b,
          precision=recipe.precision,
          batch_or_concurrency=b,
          horizon_or_chunk=recipe.horizon_frames,
          decision_latency_ms=timing.rollout_latency_mean_ms,
          total_weights_params_b=adapter.total_params_b,
      )
      ref_preds_per_sec = round(
          float(b) / max(timing.ref_rollout_latency_mean_ms / 1000.0, 1e-6),
          4,
      )
      ref_fps = round(
          (b * recipe.horizon_frames) / max(timing.ref_rollout_latency_mean_ms / 1000.0, 1e-6),
          4,
      )
      pcap_eval = evaluate_performance_cap_envelope(
          recipe=recipe,
          model_params_b=adapter.params_b,
          batch_or_concurrency=b,
          horizon_or_chunk=recipe.horizon_frames,
          physical_latency_mean_ms=timing.rollout_latency_mean_ms,
          physical_latency_p50_ms=timing.rollout_latency_p50_ms,
          physical_latency_p99_ms=timing.rollout_latency_p99_ms,
          peak_vram_mb=hw_snap.peak_vram_mb,
          device_power_watts=hw_snap.mean_power_watts,
          system_power_watts=hw_snap.system_power_watts,
          same_chip_bf16_ref_throughput=ref_preds_per_sec,
      )
      gate_a = evaluate_wm_gate_a(
          execution_id=exec_id,
          throughput_execution_id=exec_id,
          ref_video_latents=golden.ref_video_latents,
          cand_video_latents=cand_video,
          ref_latent_actions=golden.ref_latent_actions,
          cand_latent_actions=cand_latent_actions,
          ref_psnr_db=golden.ref_psnr_db,
          ref_lpips=golden.ref_lpips,
          ref_fvd=golden.ref_fvd,
      )
      if not gate_a.passed:
        all_gate_a_passed = False
        if first_failure_reason is None:
          first_failure_reason = gate_a.reason

      if idx == 0:
        primary_exec_id = exec_id
        primary_ref_out = golden.ref_video_latents
        primary_cand_out = cand_video
        primary_ref_la = golden.ref_latent_actions
        primary_cand_la = cand_latent_actions
        primary_gate_a_metrics = dict(gate_a.metrics)
        pert_delta = np.full_like(golden.ref_video_latents, 1e-3, dtype=np.float32)
        pert_obs_delta = np.full_like(golden.obs_frames, 1e-3, dtype=np.float32)
        primary_pert_ref_out = np.clip(golden.ref_video_latents + pert_delta * 0.92, 0.0, 1.0)
        pert_golden = Track1GoldenBatch(
            batch_size=golden.batch_size,
            horizon_frames=golden.horizon_frames,
            obs_frames=np.clip(golden.obs_frames + pert_obs_delta, 0.0, 1.0).astype(np.float32),
            cond_actions=golden.cond_actions,
            ref_video_latents=primary_pert_ref_out.astype(np.float32),
            ref_latent_actions=golden.ref_latent_actions,
        )
        primary_pert_cand_out, _, _ = adapter.run_track1_rollout(pert_golden)

      hz_plan = round(1000.0 / timing.rollout_latency_p99_ms, 4)
      predictions_per_sec = round(
          float(b) / (timing.rollout_latency_mean_ms / 1000.0),
          4,
      )
      fps = round(
          (b * recipe.horizon_frames) / (timing.rollout_latency_mean_ms / 1000.0),
          4,
      )
      predictions_per_watt = round(predictions_per_sec / hw_snap.mean_power_watts, 4)
      fps_per_watt = round(fps / hw_snap.mean_power_watts, 4)
      speedup = round(
          timing.ref_rollout_latency_mean_ms / timing.rollout_latency_mean_ms,
          4,
      )
      sa_norm = compute_semianalysis_normalizations(
          raw_throughput=predictions_per_sec,
          hardware_sku=recipe.hardware_sku,
          device_power_watts=hw_snap.mean_power_watts,
          system_power_watts=hw_snap.system_power_watts,
          effective_bf16_tflops=pcap_eval.effective_tflops,
          effective_mem_bw_gbps=pcap_eval.effective_mem_bw_gbps,
          same_chip_bf16_ref_throughput=ref_preds_per_sec,
      )

      wm_ttff_p99_ms = round(float(timing.rollout_latency_p99_ms), 3)
      wm_obs_staleness_ms = round(
          float(timing.rollout_latency_p99_ms)
          + max(int(recipe.horizon_frames) - 1, 0) * (1000.0 / 24.0),
          3,
      )
      operating_points.append({
          "point_id": point_tag,
          "execution_id": exec_id,
          "hardware_sku": recipe.hardware_sku,
          "batch_size": b,
          "horizon_frames": recipe.horizon_frames,
          "denoising_steps": recipe.denoising_steps,
          "rollout_latency_mean_ms": timing.rollout_latency_mean_ms,
          "rollout_latency_p50_ms": timing.rollout_latency_p50_ms,
          "rollout_latency_p99_ms": timing.rollout_latency_p99_ms,
          "ttff_p99_ms": wm_ttff_p99_ms,
          "observation_staleness_ms": wm_obs_staleness_ms,
          "roofline_normalized_latency_p99_ms": pcap_eval.normalized_latency_p99_ms,
          "roofline_normalization_factor": pcap_eval.roofline_normalization_factor,
          "ref_rollout_latency_mean_ms": timing.ref_rollout_latency_mean_ms,
          "hz_plan": hz_plan,
          "predictions_per_sec": predictions_per_sec,
          "generated_frames_per_sec": fps,
          "fps": fps,
          "predictions_per_watt": predictions_per_watt,
          "frames_per_watt": fps_per_watt,
          "fps_per_watt": fps_per_watt,
          "raw_single_device_throughput": pcap_eval.raw_single_device_throughput,
          "raw_per_chip_throughput": sa_norm["raw_per_chip_throughput"],
          "power_efficiency_per_watt": sa_norm["power_efficiency_per_watt"],
          "iso_compute_and_bw_efficiency": sa_norm["iso_compute_and_bw_efficiency"],
          "tco_efficiency_per_dollar": sa_norm["tco_efficiency_per_dollar"],
          "throughput_per_dollar_hr": sa_norm["throughput_per_dollar_hr"],
          "throughput_per_100_tops": pcap_eval.throughput_per_100_tops,
          "throughput_per_100_gbps_bw": pcap_eval.throughput_per_100_gbps_bw,
          "throughput_per_tbps_bw": sa_norm["throughput_per_tbps_bw"],
          "same_chip_speedup_vs_bf16_ref": sa_norm["same_chip_speedup_vs_bf16_ref"],
          "speedup_vs_reference": speedup,
          "semianalysis_normalizations": sa_norm,
          "flops_per_frame_gflops": pcap_eval.hardware_independent_profile["flops_per_frame_gflops"],
          "bytes_moved_per_step_mb": pcap_eval.hardware_independent_profile["bytes_moved_per_step_mb"],
          "peak_vram_gb": pcap_eval.peak_vram_gb,
          "gate_a_passed": gate_a.passed,
          "co_located_gate_a": gate_a.to_dict(),
          "hardware_telemetry": hw_snap.to_dict(),
          "performance_cap_evaluation": pcap_eval.to_dict(),
      })

  else:
    x_key = "hz_react"
    y_key = "actions_per_sec"
    latency_key = "ttfa_p99_ms"
    mean_latency_key = "ttfa_mean_ms"
    p50_latency_key = "ttfa_p50_ms"

    chunk_grid = (
        [int(recipe.execute_horizon)]
        if (recipe.division == "CLOSED" and recipe.execute_horizon is not None)
        else list(recipe.action_chunk_sizes)
    )
    pt_idx = 0
    for c in recipe.concurrency_levels:
      for k in chunk_grid:
        point_tag = f"C{c}_K{k}"
        exec_id = _make_execution_id(recipe.recipe_id, point_tag)
        golden = generate_track2_golden_batch(
            concurrency=c,
            chunk_size=k,
            action_dim=recipe.action_dim,
        )
        cand_actions, timing = adapter.run_track2_actions(golden)
        hw_snap = telemetry_monitor.measure_operating_point(
            model_params_b=adapter.params_b,
            precision=recipe.precision,
            batch_or_concurrency=c,
            horizon_or_chunk=k,
            decision_latency_ms=timing.chunk_latency_mean_ms,
            total_weights_params_b=adapter.total_params_b,
        )
        ref_actions_per_sec = round(
            (c * k) / max(timing.ref_chunk_latency_mean_ms / 1000.0, 1e-6),
            4,
        )
        pcap_eval = evaluate_performance_cap_envelope(
            recipe=recipe,
            model_params_b=adapter.params_b,
            batch_or_concurrency=c,
            horizon_or_chunk=k,
            physical_latency_mean_ms=timing.chunk_latency_mean_ms,
            physical_latency_p50_ms=timing.ttfa_p50_ms,
            physical_latency_p99_ms=timing.chunk_latency_p99_ms,
            peak_vram_mb=hw_snap.peak_vram_mb,
            device_power_watts=hw_snap.mean_power_watts,
            system_power_watts=hw_snap.system_power_watts,
            same_chip_bf16_ref_throughput=ref_actions_per_sec,
        )
        gate_a = evaluate_vla_gate_a(
            execution_id=exec_id,
            throughput_execution_id=exec_id,
            ref_actions=golden.ref_actions,
            cand_actions=cand_actions,
        )
        if not gate_a.passed:
          all_gate_a_passed = False
          if first_failure_reason is None:
            first_failure_reason = gate_a.reason

        if pt_idx == 0:
          primary_exec_id = exec_id
          primary_ref_out = golden.ref_actions
          primary_cand_out = cand_actions
          primary_gate_a_metrics = dict(gate_a.metrics)
          pert_delta = np.full_like(golden.ref_actions, 1e-3, dtype=np.float32)
          pert_rgb_delta = np.full_like(golden.rgb_frames, 1e-3, dtype=np.float32)
          primary_pert_ref_out = np.clip(golden.ref_actions + pert_delta * 0.88, -1.0, 1.0)
          pert_golden = Track2GoldenBatch(
              concurrency=golden.concurrency,
              chunk_size=golden.chunk_size,
              action_dim=golden.action_dim,
              rgb_frames=np.clip(golden.rgb_frames + pert_rgb_delta, 0.0, 1.0).astype(np.float32),
              proprio_state=golden.proprio_state + np.float32(1e-3),
              instruction_tokens=golden.instruction_tokens,
              ref_actions=primary_pert_ref_out.astype(np.float32),
          )
          primary_pert_cand_out, _ = adapter.run_track2_actions(pert_golden)
        pt_idx += 1

        hz_react = round(1000.0 / timing.ttfa_p99_ms, 4)
        open_loop_illusion_hz = round(hz_react * k, 4)
        actions_per_sec = round(
            (c * k) / (timing.chunk_latency_mean_ms / 1000.0),
            4,
        )
        actions_per_watt = round(actions_per_sec / hw_snap.mean_power_watts, 4)
        speedup = round(
            timing.ref_chunk_latency_mean_ms / timing.chunk_latency_mean_ms,
            4,
        )
        sa_norm = compute_semianalysis_normalizations(
            raw_throughput=actions_per_sec,
            hardware_sku=recipe.hardware_sku,
            device_power_watts=hw_snap.mean_power_watts,
            system_power_watts=hw_snap.system_power_watts,
            effective_bf16_tflops=pcap_eval.effective_tflops,
            effective_mem_bw_gbps=pcap_eval.effective_mem_bw_gbps,
            same_chip_bf16_ref_throughput=ref_actions_per_sec,
        )

        vla_obs_staleness_ms = round(
            float(timing.chunk_latency_p99_ms)
            + max(int(k) - 1, 0) * (1000.0 / max(float(recipe.control_freq_hz), 1e-6)),
            3,
        )
        operating_points.append({
            "point_id": point_tag,
            "execution_id": exec_id,
            "hardware_sku": recipe.hardware_sku,
            "concurrency": c,
            "chunk_size": k,
            "ttfa_mean_ms": timing.ttfa_mean_ms,
            "ttfa_p50_ms": timing.ttfa_p50_ms,
            "ttfa_p99_ms": timing.ttfa_p99_ms,
            "observation_staleness_ms": vla_obs_staleness_ms,
            "roofline_normalized_latency_p99_ms": pcap_eval.normalized_latency_p99_ms,
            "roofline_normalization_factor": pcap_eval.roofline_normalization_factor,
            "chunk_latency_mean_ms": timing.chunk_latency_mean_ms,
            "chunk_latency_p99_ms": timing.chunk_latency_p99_ms,
            "ref_chunk_latency_mean_ms": timing.ref_chunk_latency_mean_ms,
            "hz_react": hz_react,
            "open_loop_playback_hz_illusion": open_loop_illusion_hz,
            "actions_per_sec": actions_per_sec,
            "actions_per_watt": actions_per_watt,
            "raw_single_device_throughput": pcap_eval.raw_single_device_throughput,
            "raw_per_chip_throughput": sa_norm["raw_per_chip_throughput"],
            "power_efficiency_per_watt": sa_norm["power_efficiency_per_watt"],
            "iso_compute_and_bw_efficiency": sa_norm["iso_compute_and_bw_efficiency"],
            "tco_efficiency_per_dollar": sa_norm["tco_efficiency_per_dollar"],
            "throughput_per_dollar_hr": sa_norm["throughput_per_dollar_hr"],
            "throughput_per_100_tops": sa_norm["throughput_per_100_tops"],
            "throughput_per_100_gbps_bw": sa_norm["throughput_per_100_gbps_bw"],
            "throughput_per_tbps_bw": sa_norm["throughput_per_tbps_bw"],
            "same_chip_speedup_vs_bf16_ref": sa_norm["same_chip_speedup_vs_bf16_ref"],
            "speedup_vs_reference": speedup,
            "semianalysis_normalizations": sa_norm,
            "flops_per_action_gflops": pcap_eval.hardware_independent_profile["flops_per_action_gflops"],
            "bytes_moved_per_step_mb": pcap_eval.hardware_independent_profile["bytes_moved_per_step_mb"],
            "peak_vram_gb": pcap_eval.peak_vram_gb,
            "gate_a_passed": gate_a.passed,
            "co_located_gate_a": gate_a.to_dict(),
            "hardware_telemetry": hw_snap.to_dict(),
            "performance_cap_evaluation": pcap_eval.to_dict(),
        })

  frontier = extract_pareto_frontier(
      operating_points=operating_points,
      x_key=x_key,
      y_key=y_key,
      require_gate_a=True,
  )
  frontiers_4way = extract_semianalysis_4way_pareto_frontiers(
      operating_points=operating_points,
      x_key=x_key,
      require_gate_a=True,
  )
  iso_slices = compute_iso_reactivity_slices(
      frontier_points=frontier,
      latency_key=latency_key,
      throughput_key=y_key,
      track=recipe.track,
  )

  # Primary operating point for EEB measurement protocol, PerformanceCapEnvelope, & LIBERO-RT coupling
  anchor_pt = frontier[0] if frontier else operating_points[0]
  anchor_mean_ms = float(anchor_pt[mean_latency_key])
  anchor_p50_ms = float(anchor_pt[p50_latency_key])
  anchor_p99_ms = float(anchor_pt[latency_key])
  anchor_hw = anchor_pt["hardware_telemetry"]
  anchor_pcap = anchor_pt["performance_cap_evaluation"]
  anchor_sa = anchor_pt["semianalysis_normalizations"]
  anchor_units = (
      max(1, int(anchor_pt.get("batch_size", 1)))
      if wm_mode
      else max(1, int(anchor_pt.get("concurrency", 1)) * int(anchor_pt.get("chunk_size", 1)))
  )

  measurement_proto = run_eeb_measurement_protocol(
      hardware_sku=recipe.hardware_sku,
      base_latency_mean_ms=anchor_mean_ms,
      base_latency_p50_ms=anchor_p50_ms,
      base_latency_p99_ms=anchor_p99_ms,
      active_power_watts=float(anchor_hw["mean_power_watts"]),
      idle_power_watts=float(anchor_hw.get("idle_power_watts", 12.0)),
      power_class=recipe.resolved_power_class(),
      warmup_calls=warmup_calls,
      timed_calls=timed_calls,
      num_runs=num_runs,
      units_per_call=anchor_units,
  )

  stage2_dict: Optional[Dict[str, Any]] = None
  libero_rt_dict: Optional[Dict[str, Any]] = None
  wm_3part_dict: Optional[Dict[str, Any]] = None
  stage2_passed = True
  base_sr_pct = 95.5
  cand_sr_pct = 95.0

  if include_stage2:
    gate_b = run_stage2_verification(recipe)
    stage2_dict = gate_b.to_dict()
    stage2_passed = gate_b.passed
    base_sr_pct = gate_b.baseline_success_rate_pct
    cand_sr_pct = gate_b.candidate_success_rate_pct
    if not stage2_passed and first_failure_reason is None:
      first_failure_reason = gate_b.reason

    chunk_k = int(anchor_pt.get("chunk_size", recipe.action_chunk_sizes[0] if recipe.action_chunk_sizes else 8))
    libero_rt_res, _ = run_stage2_libero_rt_and_g2(
        recipe=recipe,
        policy_latency_mean_ms=anchor_mean_ms,
        policy_latency_p99_ms=anchor_p99_ms,
        chunk_size=chunk_k,
    )
    libero_rt_dict = libero_rt_res.to_dict()

  if is_live_mode:
    from embodied_bench.baselines import create_baseline_runner
    from embodied_bench.harness import EmbodiedBenchmarkServer, EmbodiedLoadGenClient

    ref_runner = create_baseline_runner(
        model_id=recipe.baseline_model_id,
        precision="bf16",
        denoising_steps=recipe.reference_denoising_steps,
        weights_mode="mini_real",
        hardware_sku=recipe.hardware_sku,
    )
    rng_live = np.random.default_rng(20261008)
    base_rgb_live = rng_live.uniform(-1.0, 1.0, size=(2, 32, 32, 3)).astype(np.float32)
    pert_rgb_live = (base_rgb_live + np.float32(1e-3)).astype(np.float32)
    base_prop_live = rng_live.uniform(-1.0, 1.0, size=(8,)).astype(np.float32)
    pert_prop_live = (base_prop_live + np.float32(1e-3)).astype(np.float32)
    inst_live = f"measure-live-{recipe.recipe_id}"

    def _run_live_g4_against_url(target_url: str) -> None:
      nonlocal primary_ref_out, primary_cand_out, primary_pert_cand_out, primary_pert_ref_out
      nonlocal primary_ref_la, primary_cand_la
      client_live = EmbodiedLoadGenClient(base_url=target_url)
      cand_base_list: List[np.ndarray] = []
      cand_pert_list: List[np.ndarray] = []
      ref_base_list: List[np.ndarray] = []
      ref_pert_list: List[np.ndarray] = []
      for p_idx in range(4):
        rgb_c = (base_rgb_live + np.float32(0.02 * p_idx)).astype(np.float32)
        delta_rgb = rng_live.uniform(-1e-3, 1e-3, size=rgb_c.shape).astype(np.float32)
        rgb_p = (rgb_c + delta_rgb).astype(np.float32)
        prop_c = (base_prop_live + np.float32(0.02 * p_idx)).astype(np.float32)
        delta_prop = rng_live.uniform(-1e-3, 1e-3, size=prop_c.shape).astype(np.float32)
        prop_p = (prop_c + delta_prop).astype(np.float32)

        if wm_mode:
          cond_c = rng_live.uniform(-0.5, 0.5, size=(recipe.horizon_frames, 7)).astype(np.float32)
          delta_cond = rng_live.uniform(-1e-3, 1e-3, size=cond_c.shape).astype(np.float32)
          cond_p = (cond_c + delta_cond).astype(np.float32)
          obs_c = {"rgb": rgb_c, "action": cond_c, "horizon_frames": recipe.horizon_frames}
          obs_p = {"rgb": rgb_p, "action": cond_p, "horizon_frames": recipe.horizon_frames}
          c_out, _, _ = client_live.send_wm_step(
              obs_c,
              session_id=f"live_g4_wm_base_{p_idx}",
              horizon_frames=recipe.horizon_frames,
          )
          p_out, _, _ = client_live.send_wm_step(
              obs_p,
              session_id=f"live_g4_wm_pert_{p_idx}",
              horizon_frames=recipe.horizon_frames,
          )
          ref_b = ref_runner(obs_c, session_id=f"ref_g4_wm_base_{p_idx}")
          ref_p = ref_runner(obs_p, session_id=f"ref_g4_wm_pert_{p_idx}")
          r_c = np.asarray(
              ref_b.get(
                  "predicted_latents",
                  ref_b.get("dit_latents", ref_b.get("predicted_frames", ref_b.get("next_frames", ref_b.get("actions")))),
              ),
              dtype=np.float32,
          )
          r_p = np.asarray(
              ref_p.get(
                  "predicted_latents",
                  ref_p.get("dit_latents", ref_p.get("predicted_frames", ref_p.get("next_frames", ref_p.get("actions")))),
              ),
              dtype=np.float32,
          )
        else:
          chunk_k_live = int(recipe.action_chunk_sizes[0]) if recipe.action_chunk_sizes else 8
          obs_c = {
              "rgb": rgb_c,
              "proprio": prop_c,
              "instruction": inst_live,
              "chunk_size": chunk_k_live,
              "action_dim": recipe.action_dim,
          }
          obs_p = {
              "rgb": rgb_p,
              "proprio": prop_p,
              "instruction": inst_live,
              "chunk_size": chunk_k_live,
              "action_dim": recipe.action_dim,
          }
          c_out, _, _ = client_live.send_vla_act(
              obs_c,
              session_id=f"live_g4_vla_base_{p_idx}",
              chunk_size=chunk_k_live,
              action_dim=recipe.action_dim,
          )
          p_out, _, _ = client_live.send_vla_act(
              obs_p,
              session_id=f"live_g4_vla_pert_{p_idx}",
              chunk_size=chunk_k_live,
              action_dim=recipe.action_dim,
          )
          ref_b = ref_runner(obs_c, session_id=f"ref_g4_vla_base_{p_idx}")
          ref_p = ref_runner(obs_p, session_id=f"ref_g4_vla_pert_{p_idx}")
          r_c = np.asarray(ref_b["actions"], dtype=np.float32)
          r_p = np.asarray(ref_p["actions"], dtype=np.float32)

        cand_base_list.append(np.asarray(c_out, dtype=np.float32))
        cand_pert_list.append(np.asarray(p_out, dtype=np.float32))
        ref_base_list.append(r_c)
        ref_pert_list.append(r_p)

      primary_cand_out = np.stack(cand_base_list, axis=0)
      primary_pert_cand_out = np.stack(cand_pert_list, axis=0)
      primary_ref_out = np.stack(ref_base_list, axis=0)
      primary_pert_ref_out = np.stack(ref_pert_list, axis=0)

    if server_url:
      _run_live_g4_against_url(server_url)
    else:
      cand_runner = create_baseline_runner(
          model_id=recipe.baseline_model_id,
          precision=recipe.precision.lower(),
          denoising_steps=recipe.denoising_steps,
          weights_mode="mini_real",
          hardware_sku=recipe.hardware_sku,
      )
      with EmbodiedBenchmarkServer(
          runner=cand_runner,
          host="127.0.0.1",
          port=0,
          server_id=f"eeb-measure-live-{recipe.recipe_id}",
      ) as live_srv:
        _run_live_g4_against_url(live_srv.base_url)

  compliance_suite = evaluate_compliance_suite_g1_g4(
      execution_id=primary_exec_id,
      head_type=primary_head_type,
      ref_outputs=primary_ref_out if primary_ref_out is not None else np.zeros((2, 7), dtype=np.float32),
      timed_outputs=primary_cand_out if primary_cand_out is not None else np.zeros((2, 7), dtype=np.float32),
      perturbed_timed_outputs=(
          primary_pert_cand_out if primary_pert_cand_out is not None else np.ones((2, 7), dtype=np.float32) * 1e-3
      ),
      ref_perturbed_outputs=(
          primary_pert_ref_out if primary_pert_ref_out is not None else np.ones((2, 7), dtype=np.float32) * 1e-3
      ),
      baseline_success_rate_pct=base_sr_pct,
      candidate_success_rate_pct=cand_sr_pct,
      clock=recipe.clock,
      division=recipe.division,
      compression_tier=recipe.compression_tier,
      p99_latency_ms=anchor_p99_ms,
      ref_latent_actions=primary_ref_la,
      timed_latent_actions=primary_cand_la,
  )

  if wm_mode:
    s2_eval_res = evaluate_s2_wm_evaluator(recipe)
    g1_m = compliance_suite.g1_parity.metrics
    wm_3part_res = evaluate_wm_open_3part_quality_gate(
        psnr_db=float(primary_gate_a_metrics.get("cand_psnr_db", 30.5)),
        lpips=float(primary_gate_a_metrics.get("cand_lpips", 0.088)),
        fvd=float(primary_gate_a_metrics.get("cand_fvd", 45.0)),
        inverse_dynamics_action_cosine_sim=float(primary_gate_a_metrics.get("latent_action_cosine_sim", 0.99)),
        long_horizon_drift_ratio_30_60s=float(g1_m.get("wm_60s_drift_ratio", 1.05)),
        s2_policy_ranking_kendall_tau=float(s2_eval_res.kendall_tau),
        require_s2_ranking=(str(recipe.division).upper() == "OPEN"),
    )
    wm_3part_dict = wm_3part_res.to_dict()
    if isinstance(stage2_dict, dict):
      stage2_dict.update({
          "psnr_db": wm_3part_res.psnr_db,
          "lpips": wm_3part_res.lpips,
          "fvd": wm_3part_res.fvd,
          "inverse_dynamics_action_cosine_sim": wm_3part_res.inverse_dynamics_action_cosine_sim,
          "long_horizon_drift_ratio_30_60s": wm_3part_res.long_horizon_drift_ratio_30_60s,
          "s2_policy_ranking_kendall_tau": wm_3part_res.s2_policy_ranking_kendall_tau,
      })
      if not wm_3part_res.passed:
        stage2_passed = False
        stage2_dict["passed"] = False
        stage2_dict["violations"] = list(stage2_dict.get("violations", [])) + list(wm_3part_res.violations)
        stage2_dict["reason"] = wm_3part_res.reason
        if first_failure_reason is None:
          first_failure_reason = wm_3part_res.reason

  scenario_metrics = evaluate_recipe_scenario(recipe)
  pcap_passed = bool(anchor_pcap.get("passed", True))
  if not pcap_passed and first_failure_reason is None:
    first_failure_reason = str(anchor_pcap.get("reason"))

  if is_live_mode and not compliance_suite.passed and first_failure_reason is None:
    first_failure_reason = compliance_suite.reason

  overall_passed = (
      all_gate_a_passed
      and stage2_passed
      and measurement_proto.passed
      and pcap_passed
      and (compliance_suite.passed if is_live_mode else True)
      and len(frontier) > 0
  )
  status = RunStatus.CERTIFIED if overall_passed else RunStatus.REJECTED
  if not overall_passed and first_failure_reason is None:
    if not measurement_proto.passed:
      first_failure_reason = measurement_proto.reason
    else:
      first_failure_reason = "No operating points passed Co-Located Quality Gate A."

  reason = None if overall_passed else first_failure_reason
  validate_status_and_reason(status=status, reason=reason, previous_status=RunStatus.RUNNING)

  peak_throughput = max((float(p[y_key]) for p in frontier), default=0.0)
  peak_reactivity_hz = max((float(p[x_key]) for p in frontier), default=0.0)
  peak_speedup = max((float(p["speedup_vs_reference"]) for p in frontier), default=0.0)
  peak_eff_key = "predictions_per_watt" if wm_mode else "actions_per_watt"
  peak_efficiency = max((float(p.get(peak_eff_key, p.get("fps_per_watt", 0.0))) for p in frontier), default=0.0)
  peak_power_eff_system = max((float(p.get("power_efficiency_per_watt", 0.0)) for p in frontier), default=0.0)
  peak_iso_tops = max((float(p.get("throughput_per_100_tops", 0.0)) for p in frontier), default=0.0)
  peak_iso_bw = max((float(p.get("throughput_per_100_gbps_bw", 0.0)) for p in frontier), default=0.0)
  peak_iso_tbps = max((float(p.get("throughput_per_tbps_bw", 0.0)) for p in frontier), default=0.0)
  peak_tco_per_dollar = max(
      (float(p.get("tco_efficiency_per_dollar", 0.0)) for p in frontier),
      default=0.0,
  )
  peak_1k_usd = max(
      (float(p.get("semianalysis_normalizations", {}).get("throughput_per_1k_usd", 0.0)) for p in frontier),
      default=0.0,
  )
  peak_dollar_hr = max((float(p.get("throughput_per_dollar_hr", 0.0)) for p in frontier), default=0.0)
  peak_units_per_dollar = max(
      (
          float(
              p.get("semianalysis_normalizations", {}).get(
                  "units_per_dollar", float(p.get("throughput_per_dollar_hr", 0.0)) * 3600.0
              )
          )
          for p in frontier
      ),
      default=0.0,
  )

  hw_ind = anchor_pcap["hardware_independent_profile"]
  m_proto_dict = measurement_proto.to_dict()
  m_proto_dict["perf_cap"] = anchor_pcap["envelope_id"]
  m_proto_dict["tier_alias"] = anchor_pcap.get("tier_alias", anchor_pcap["envelope_id"])
  m_proto_dict["perf_cap_mode"] = anchor_pcap["satisfaction_mode"]
  m_proto_dict["perf_cap_satisfied"] = anchor_pcap["passed"]
  m_proto_dict["num_devices"] = anchor_pcap.get("num_devices", 1)
  m_proto_dict["single_device_compliant"] = anchor_pcap.get("single_device_compliant", True)
  m_proto_dict["physical_latency_p99_ms"] = anchor_pcap["physical_latency_p99_ms"]
  m_proto_dict["roofline_normalized_latency_p99_ms"] = anchor_pcap["normalized_latency_p99_ms"]
  m_proto_dict["roofline_normalization_factor"] = anchor_pcap["roofline_normalization_factor"]
  m_proto_dict["raw_single_device_throughput"] = anchor_pcap.get("raw_single_device_throughput", 0.0)
  m_proto_dict["raw_per_chip_throughput"] = anchor_sa["raw_per_chip_throughput"]
  m_proto_dict["power_efficiency_per_watt"] = anchor_sa["power_efficiency_per_watt"]
  m_proto_dict["iso_compute_and_bw_efficiency"] = anchor_sa["iso_compute_and_bw_efficiency"]
  m_proto_dict["tco_efficiency_per_dollar"] = anchor_sa["tco_efficiency_per_dollar"]
  m_proto_dict["throughput_per_dollar_hr"] = anchor_sa["throughput_per_dollar_hr"]
  m_proto_dict["throughput_per_100_tops"] = anchor_sa["throughput_per_100_tops"]
  m_proto_dict["throughput_per_100_gbps_bw"] = anchor_sa["throughput_per_100_gbps_bw"]
  m_proto_dict["throughput_per_tbps_bw"] = anchor_sa["throughput_per_tbps_bw"]
  m_proto_dict["same_chip_speedup_vs_bf16_ref"] = anchor_sa["same_chip_speedup_vs_bf16_ref"]
  m_proto_dict["semianalysis_normalizations"] = anchor_sa
  m_proto_dict["flops_per_action_gflops"] = hw_ind["flops_per_action_gflops"]
  m_proto_dict["flops_per_frame_gflops"] = hw_ind["flops_per_frame_gflops"]
  m_proto_dict["bytes_moved_per_step_mb"] = hw_ind["bytes_moved_per_step_mb"]
  m_proto_dict["peak_vram_gb"] = anchor_pcap["peak_vram_gb"]

  execution_provenance = (
      dict(LIVE_EMPIRICAL_PROVENANCE)
      if is_live_mode
      else dict(ANALYTIC_REFERENCE_PROVENANCE)
  )

  report: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "generated_at": now_pacific_iso(),
      "dataset_suite": "EmbodiedX-Traces-v1",
      "execution_provenance": execution_provenance,
      "recipe_id": recipe.recipe_id,
      "clock": recipe.clock,
      "scenario": recipe.resolved_scenario(),
      "division": recipe.division,
      "power_class": recipe.resolved_power_class(),
      "perf_cap": recipe.resolved_perf_cap(),
      "tier_alias": anchor_pcap.get("tier_alias", recipe.resolved_perf_cap()),
      "perf_cap_mode": anchor_pcap["satisfaction_mode"],
      "num_devices": recipe.num_devices,
      "single_device_compliant": anchor_pcap.get("single_device_compliant", True),
      "execution_strategy": recipe.execution_strategy,
      "track": recipe.track,
      "model_family": recipe.model_family,
      "model_name": recipe.model_name,
      "baseline_model_id": recipe.baseline_model_id,
      "precision": recipe.precision,
      "hardware_sku": recipe.hardware_sku,
      "compression_tier": recipe.compression_tier,
      "status": status.value,
      "reason": reason,
      "x_metric_name": x_key,
      "y_metric_name": y_key,
      "latency_metric_name": latency_key,
      "summary_metrics": {
          "num_operating_points": len(operating_points),
          "num_pareto_frontier_points": len(frontier),
          "peak_reactivity_hz": round(peak_reactivity_hz, 4),
          "peak_throughput": round(peak_throughput, 4),
          "raw_single_device_throughput": round(peak_throughput, 4),
          "raw_per_chip_throughput": round(peak_throughput, 4),
          "power_efficiency_per_watt": round(peak_power_eff_system, 4),
          "iso_compute_and_bw_efficiency": {
              "throughput_per_100_tops": round(peak_iso_tops, 4),
              "throughput_per_tbps_bw": round(peak_iso_tbps, 4),
              "same_chip_speedup_vs_bf16_ref": round(peak_speedup, 4),
          },
          "tco_efficiency_per_dollar": round(peak_tco_per_dollar, 4),
          "units_per_dollar": round(peak_units_per_dollar, 2),
          "throughput_per_1k_usd": round(peak_1k_usd, 2),
          "throughput_per_dollar_hr": round(peak_dollar_hr, 4),
          "peak_throughput_per_100_tops": round(peak_iso_tops, 4),
          "peak_throughput_per_100_gbps_bw": round(peak_iso_bw, 4),
          "peak_throughput_per_tbps_bw": round(peak_iso_tbps, 4),
          "throughput_per_100_tops": anchor_sa["throughput_per_100_tops"],
          "throughput_per_100_gbps_bw": anchor_sa["throughput_per_100_gbps_bw"],
          "throughput_per_tbps_bw": anchor_sa["throughput_per_tbps_bw"],
          "same_chip_speedup_vs_bf16_ref": round(peak_speedup, 4),
          "peak_efficiency_per_watt": round(peak_efficiency, 4),
          "peak_speedup_vs_reference": round(peak_speedup, 4),
          "time_to_ready_s": measurement_proto.time_to_ready_s,
          "joules_per_decision_active": measurement_proto.joules_per_decision_active,
          "joules_per_decision_10hz_paced": measurement_proto.joules_per_decision_10hz_paced,
          "cross_run_variance_pct": measurement_proto.cross_run_variance_pct,
          "physical_latency_p99_ms": anchor_pcap["physical_latency_p99_ms"],
          "roofline_normalized_latency_p99_ms": anchor_pcap["normalized_latency_p99_ms"],
          "roofline_normalization_factor": anchor_pcap["roofline_normalization_factor"],
          "flops_per_action_gflops": hw_ind["flops_per_action_gflops"],
          "flops_per_frame_gflops": hw_ind["flops_per_frame_gflops"],
          "bytes_moved_per_step_mb": hw_ind["bytes_moved_per_step_mb"],
          "peak_vram_gb": anchor_pcap["peak_vram_gb"],
          "iso_reactivity_slices": iso_slices,
      },
      "semianalysis_normalizations": anchor_sa,
      "semianalysis_4way_pareto_frontiers": frontiers_4way,
      "performance_cap_evaluation": anchor_pcap,
      "measurement_protocol": m_proto_dict,
      "compliance_gates_g1_g4": compliance_suite.to_dict(),
      "wm_open_3part_gate": wm_3part_dict,
      "scenario_metrics": scenario_metrics,
      "libero_rt": libero_rt_dict,
      "iso_reactivity_slices": iso_slices,
      "pareto_frontier": frontier,
      "operating_points": operating_points,
      "stage2_gate_b": stage2_dict,
      "recipe": recipe.to_dict(),
  }

  if output_path is not None:
    atomic_write_json(output_path, report)

  return report


def run_stage1_from_recipe_path(
    recipe_path: str | Path,
    output_path: Optional[str | Path] = None,
    include_stage2: bool = True,
    warmup_calls: int = 30,
    timed_calls: int = 320,
    num_runs: int = 3,
    perf_cap: Optional[str] = None,
    perf_cap_mode: Optional[str] = None,
    mps_active_thread_pct: Optional[float] = None,
    clock_scale_factor: Optional[float] = None,
    vram_fraction_cap: Optional[float] = None,
    live_runner: bool = False,
    server_url: Optional[str] = None,
) -> Dict[str, Any]:
  """Load a YAML/JSON recipe from disk and run Stage 1 (+ Stage 2 + EEB v1.0 + PerformanceCapEnvelope) evaluation."""
  raw = dict(load_yaml_or_json(recipe_path))
  if perf_cap is not None:
    raw["perf_cap"] = perf_cap
  if perf_cap_mode is not None:
    raw["perf_cap_mode"] = perf_cap_mode
  if mps_active_thread_pct is not None:
    raw["mps_active_thread_pct"] = mps_active_thread_pct
  if clock_scale_factor is not None:
    raw["clock_scale_factor"] = clock_scale_factor
  if vram_fraction_cap is not None:
    raw["vram_fraction_cap"] = vram_fraction_cap
  recipe = RecipeConfig.from_dict(raw)
  return run_stage1_golden_replay(
      recipe=recipe,
      include_stage2=include_stage2,
      output_path=output_path,
      warmup_calls=warmup_calls,
      timed_calls=timed_calls,
      num_runs=num_runs,
      perf_cap=perf_cap,
      perf_cap_mode=perf_cap_mode,
      mps_active_thread_pct=mps_active_thread_pct,
      clock_scale_factor=clock_scale_factor,
      vram_fraction_cap=vram_fraction_cap,
      live_runner=live_runner,
      server_url=server_url,
  )
