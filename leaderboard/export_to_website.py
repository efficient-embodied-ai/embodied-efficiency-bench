"""Export verified EmbodiedX (`embodied-efficiency-bench`) Stage 1/2 runs or leaderboard JSON into a compact, frontend-ready JSON payload for `efficient-embodied-ai.github.io`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from embodied_bench.measure import compute_semianalysis_normalizations
from embodied_bench.pareto import (
    compute_iso_reactivity_slices,
    extract_pareto_frontier,
    extract_semianalysis_4way_pareto_frontiers,
)
from embodied_bench.schema import (
    ANALYTIC_REFERENCE_PROVENANCE,
    CANONICAL_V1_MODELS,
    HARDWARE_SKU_SPECS,
    LIVE_EMPIRICAL_PROVENANCE,
    MODERN_REFERENCE_CHECKPOINTS,
    PERFORMANCE_CAP_ENVELOPES,
    PerformanceCapId,
    RunStatus,
    TrackType,
    atomic_read_modify_write_json,
    atomic_write_json,
    cleanup_orphan_temp_files,
    is_wm_track,
    load_yaml_or_json,
    now_pacific_iso,
)


# Canonical cross-vendor single-chip SKUs featured on the EmbodiedX website selector
CANONICAL_WEBSITE_SINGLE_CHIP_SKUS: Dict[str, str] = {
    "TPU_V6E_1_32GB": "Google Cloud TPU v6e-1 32GB",
    "TPU_V5E_1_16GB": "TPU v5e-1 16GB",
    "JETSON_AGX_THOR_64GB": "NVIDIA Jetson AGX Thor 64GB",
    "RTX_5090_32GB": "RTX 5090 32GB",
    "L40S_48GB": "L40S 48GB",
    "RTX_4090_24GB": "RTX 4090 24GB",
    "L4_24GB": "NVIDIA L4 24GB",
    "AMD_MI210_64GB": "AMD Instinct MI210 64GB",
    "AMD_W7900_48GB": "Radeon PRO W7900 48GB",
}

# Canonical 4 SemiAnalysis Y-axis normalization modes
SEMIANALYSIS_Y_AXIS_NORMALIZATIONS: Dict[str, Dict[str, str]] = {
    "raw_per_chip_throughput": {
        "key": "raw_per_chip_throughput",
        "label": "Raw Per-Chip Throughput (Actions/s/chip or Pred/s/chip)",
        "vla_unit": "Actions/sec/chip",
        "wm_unit": "Predictions/sec/chip (16-Frame Rollouts/s)",
        "description": "Raw single-chip throughput at 100% native uncapped single-device execution (num_devices == 1).",
    },
    "power_efficiency_per_watt": {
        "key": "power_efficiency_per_watt",
        "label": "Power Efficiency / Perf-per-Watt (Actions/J or Pred/J)",
        "vla_unit": "Actions/Joule (Actions/s/W)",
        "wm_unit": "Predictions/Joule (Pred/s/W)",
        "description": "Throughput divided by measured active device power watts (zero synthetic host/system overhead).",
    },
    "throughput_per_100_tops": {
        "key": "throughput_per_100_tops",
        "label": "Iso-Compute Efficiency (Throughput per 100 Dense BF16 TFLOPS)",
        "vla_unit": "Actions/s per 100 Dense BF16 TFLOPS",
        "wm_unit": "Predictions/s per 100 Dense BF16 TFLOPS",
        "description": "Hardware-normalized architectural utilization: Throughput / (Dense_BF16_TFLOPS / 100), plus TB/s BW auxiliary efficiency.",
    },
    "throughput_per_dollar_hr": {
        "key": "throughput_per_dollar_hr",
        "label": "TCO Efficiency (Perf per $/hr Cloud/Edge Rate)",
        "vla_unit": "(Actions/s)/($/hr)",
        "wm_unit": "(Predictions/s)/($/hr)",
        "description": "Hourly-rate cost-normalized throughput: Throughput / Reference_Hourly_Cost_USD in (Output/s)/($/hr).",
    },
}


def _enrich_operating_point(
    pt: Dict[str, Any],
    hardware_sku: str,
    x_key: str,
    y_key: str,
    lat_key: str,
) -> Dict[str, Any]:
  """Ensure an operating point has all 4 SemiAnalysis Y-axis normalizations and compact frontend fields."""
  enriched = dict(pt)
  raw_tput = float(
      enriched.get(
          "predictions_per_sec" if y_key == "predictions_per_sec" else "raw_per_chip_throughput",
          enriched.get(
              "raw_per_chip_throughput",
              enriched.get(
                  "raw_single_device_throughput",
                  enriched.get(
                      y_key,
                      enriched.get(
                          "predictions_per_sec",
                          enriched.get(
                              "generated_frames_per_sec",
                              enriched.get(
                                  "executed_actions_per_sec",
                                  enriched.get("actions_per_sec", enriched.get("frames_per_sec", enriched.get("fps", 0.0))),
                              ),
                          ),
                      ),
                  ),
              ),
          ),
      )
  )
  hw_tel = enriched.get("hardware_telemetry") if isinstance(enriched.get("hardware_telemetry"), dict) else {}
  existing_sa = (
      enriched.get("semianalysis_normalizations")
      if isinstance(enriched.get("semianalysis_normalizations"), dict)
      else {}
  )
  power_unavail = (
      hw_tel.get("power_provenance") in ("UNAVAILABLE", "ESTIMATED")
      or existing_sa.get("power_provenance") in ("UNAVAILABLE", "ESTIMATED")
      or enriched.get("power_provenance") in ("UNAVAILABLE", "ESTIMATED")
      or ("power_efficiency_per_watt" in enriched and enriched["power_efficiency_per_watt"] is None)
  )
  dev_w = hw_tel.get("mean_power_watts")
  sys_w = hw_tel.get("system_power_watts")
  speedup = float(enriched.get("speedup_vs_reference", enriched.get("same_chip_speedup_vs_bf16_ref", 1.0)))
  ref_tput = (raw_tput / speedup) if speedup > 0.0 else raw_tput

  sa = compute_semianalysis_normalizations(
      raw_throughput=raw_tput,
      hardware_sku=hardware_sku,
      device_power_watts=float(dev_w) if dev_w is not None else None,
      system_power_watts=float(sys_w) if sys_w is not None else None,
      same_chip_bf16_ref_throughput=ref_tput,
  )
  if power_unavail:
    sa["power_efficiency_per_watt"] = None
    sa["system_power_watts"] = None
  enriched[x_key] = round(float(enriched.get(x_key, 0.0)), 4)
  enriched[y_key] = round(raw_tput, 4)
  if lat_key in enriched and enriched[lat_key] is not None:
    enriched[lat_key] = round(float(enriched[lat_key]), 3)
  enriched["hardware_sku"] = hardware_sku
  enriched["raw_per_chip_throughput"] = sa["raw_per_chip_throughput"]
  enriched["power_efficiency_per_watt"] = sa["power_efficiency_per_watt"]
  enriched["throughput_per_100_tops"] = sa["throughput_per_100_tops"]
  enriched["throughput_per_tbps_bw"] = sa["throughput_per_tbps_bw"]
  enriched["throughput_per_100_gbps_bw"] = sa["throughput_per_100_gbps_bw"]
  enriched["same_chip_speedup_vs_bf16_ref"] = sa["same_chip_speedup_vs_bf16_ref"]
  enriched["throughput_per_dollar_hr"] = sa["throughput_per_dollar_hr"]
  enriched["throughput_per_1k_usd"] = sa["throughput_per_1k_usd"]
  enriched["tco_efficiency_per_dollar"] = sa["tco_efficiency_per_dollar"]
  enriched["iso_compute_and_bw_efficiency"] = sa["iso_compute_and_bw_efficiency"]
  enriched["semianalysis_normalizations"] = sa
  return enriched


def _normalize_submission_record(raw_sub: Dict[str, Any]) -> Dict[str, Any]:
  """Convert a Stage 1/2 report or leaderboard submission entry into a compact website submission record."""
  track = str(raw_sub.get("track", TrackType.VLA_TRACK.value))
  wm_flag = is_wm_track(track)
  x_key = str(raw_sub.get("x_metric_name", "hz_plan" if wm_flag else "hz_react"))
  y_key = str(raw_sub.get("y_metric_name", "predictions_per_sec" if wm_flag else "actions_per_sec"))
  lat_key = str(
      raw_sub.get(
          "latency_metric_name",
          "rollout_latency_p99_ms" if wm_flag else "ttfa_p99_ms",
      )
  )
  sku = str(raw_sub.get("hardware_sku", "TPU_V6E_1_32GB"))
  sku_spec = HARDWARE_SKU_SPECS.get(sku, HARDWARE_SKU_SPECS["TPU_V6E_1_32GB"])

  raw_points = raw_sub.get("operating_points") or raw_sub.get("pareto_frontier") or []
  enriched_points = [
      _enrich_operating_point(pt, hardware_sku=sku, x_key=x_key, y_key=y_key, lat_key=lat_key)
      for pt in raw_points
  ]
  compact_points: List[Dict[str, Any]] = []
  for pt in enriched_points:
    c_item: Dict[str, Any] = {
        "point_id": pt.get("point_id"),
        "execution_id": pt.get("execution_id"),
        "hardware_sku": sku,
        x_key: pt[x_key],
        y_key: pt[y_key],
        lat_key: pt.get(lat_key),
        "raw_per_chip_throughput": pt["raw_per_chip_throughput"],
        "power_efficiency_per_watt": pt["power_efficiency_per_watt"],
        "throughput_per_100_tops": pt["throughput_per_100_tops"],
        "throughput_per_dollar_hr": pt["throughput_per_dollar_hr"],
        "tco_efficiency_per_dollar": pt["tco_efficiency_per_dollar"],
        "throughput_per_tbps_bw": pt["throughput_per_tbps_bw"],
        "same_chip_speedup_vs_bf16_ref": pt["same_chip_speedup_vs_bf16_ref"],
        "gate_a_passed": bool(pt.get("gate_a_passed", True)),
    }
    if pt.get("ttff_p99_ms") is not None:
      c_item["ttff_p99_ms"] = round(float(pt["ttff_p99_ms"]), 3)
    if pt.get("observation_staleness_ms") is not None:
      c_item["observation_staleness_ms"] = round(float(pt["observation_staleness_ms"]), 3)
    compact_points.append(c_item)

  frontier = extract_pareto_frontier(
      operating_points=compact_points,
      x_key=x_key,
      y_key=y_key,
      require_gate_a=True,
  )
  raw_frontiers_4way = extract_semianalysis_4way_pareto_frontiers(
      operating_points=compact_points,
      x_key=x_key,
      require_gate_a=True,
  )
  frontiers_4way = {
      norm_k: [
          {k: v for k, v in pt.items() if k != "semianalysis_normalizations"}
          for pt in pts
      ]
      for norm_k, pts in raw_frontiers_4way.items()
  }
  y_axis_frontiers = {
      "raw_per_chip_throughput": frontiers_4way["raw_per_chip_throughput"],
      "power_efficiency_per_watt": frontiers_4way["power_efficiency_per_watt"],
      "throughput_per_100_tops": frontiers_4way["iso_compute_and_bw_efficiency"],
      "throughput_per_dollar_hr": frontiers_4way["tco_efficiency_per_dollar"],
  }
  iso_slices = raw_sub.get("iso_reactivity_slices") or compute_iso_reactivity_slices(
      frontier_points=frontier,
      latency_key=lat_key,
      throughput_key=y_key,
      track=track,
  )

  summary = raw_sub.get("summary_metrics") if isinstance(raw_sub.get("summary_metrics"), dict) else {}
  peak_x = max(
      (float(p.get(x_key, 0.0)) for p in frontier if p.get(x_key) is not None),
      default=float(raw_sub.get("peak_reactivity_hz") or summary.get("peak_reactivity_hz") or 0.0),
  )
  peak_raw = max(
      (float(p.get("raw_per_chip_throughput", 0.0)) for p in frontier if p.get("raw_per_chip_throughput") is not None),
      default=float(raw_sub.get("raw_per_chip_throughput") or summary.get("raw_per_chip_throughput") or 0.0),
  )
  watt_candidates = [
      float(p["power_efficiency_per_watt"])
      for p in frontier
      if p.get("power_efficiency_per_watt") is not None
  ]
  if watt_candidates:
    peak_watt: Optional[float] = round(max(watt_candidates), 4)
  elif raw_sub.get("power_efficiency_per_watt") is not None:
    peak_watt = round(float(raw_sub["power_efficiency_per_watt"]), 4)
  elif summary.get("power_efficiency_per_watt") is not None:
    peak_watt = round(float(summary["power_efficiency_per_watt"]), 4)
  else:
    peak_watt = None
  peak_100_tops = max(
      (float(p.get("throughput_per_100_tops", 0.0)) for p in frontier if p.get("throughput_per_100_tops") is not None),
      default=float(raw_sub.get("peak_throughput_per_100_tops") or summary.get("peak_throughput_per_100_tops") or 0.0),
  )
  peak_tco_rate = max(
      (float(p.get("tco_efficiency_per_dollar", 0.0)) for p in frontier if p.get("tco_efficiency_per_dollar") is not None),
      default=float(
          raw_sub.get("tco_efficiency_per_dollar")
          or summary.get("tco_efficiency_per_dollar")
          or raw_sub.get("throughput_per_dollar_hr")
          or summary.get("throughput_per_dollar_hr")
          or 0.0
      ),
  )
  peak_dollar_hr = peak_tco_rate
  peak_tco_1k = max(
      (
          float(p.get("throughput_per_1k_usd", 0.0))
          for p in frontier
          if p.get("throughput_per_1k_usd") is not None
      ),
      default=float(
          raw_sub.get("throughput_per_1k_usd")
          or summary.get("throughput_per_1k_usd")
          or 0.0
      ),
  )
  peak_tbps = max(
      (float(p.get("throughput_per_tbps_bw", 0.0)) for p in frontier if p.get("throughput_per_tbps_bw") is not None),
      default=float(raw_sub.get("peak_throughput_per_tbps_bw") or summary.get("peak_throughput_per_tbps_bw") or 0.0),
  )
  peak_speedup = max(
      (float(p.get("same_chip_speedup_vs_bf16_ref", 1.0)) for p in frontier if p.get("same_chip_speedup_vs_bf16_ref") is not None),
      default=float(raw_sub.get("same_chip_speedup_vs_bf16_ref") or summary.get("same_chip_speedup_vs_bf16_ref") or 1.0),
  )

  # Compact frontier points for frontend plotting
  compact_frontier: List[Dict[str, Any]] = []
  for pt in frontier:
    cf_item: Dict[str, Any] = {
        "point_id": pt.get("point_id"),
        "execution_id": pt.get("execution_id"),
        "hardware_sku": sku,
        x_key: pt[x_key],
        y_key: pt[y_key],
        lat_key: pt.get(lat_key),
        "raw_per_chip_throughput": pt["raw_per_chip_throughput"],
        "power_efficiency_per_watt": pt["power_efficiency_per_watt"],
        "throughput_per_100_tops": pt["throughput_per_100_tops"],
        "throughput_per_dollar_hr": pt["throughput_per_dollar_hr"],
        "throughput_per_1k_usd": pt.get("throughput_per_1k_usd", pt["tco_efficiency_per_dollar"]),
        "tco_efficiency_per_dollar": pt["tco_efficiency_per_dollar"],
        "throughput_per_tbps_bw": pt["throughput_per_tbps_bw"],
        "same_chip_speedup_vs_bf16_ref": pt["same_chip_speedup_vs_bf16_ref"],
        "gate_a_passed": bool(pt.get("gate_a_passed", True)),
    }
    if pt.get("ttff_p99_ms") is not None:
      cf_item["ttff_p99_ms"] = round(float(pt["ttff_p99_ms"]), 3)
    if pt.get("observation_staleness_ms") is not None:
      cf_item["observation_staleness_ms"] = round(float(pt["observation_staleness_ms"]), 3)
    compact_frontier.append(cf_item)

  baseline_id = str(raw_sub.get("baseline_model_id", ""))
  return {
      "recipe_id": str(raw_sub.get("recipe_id", "")),
      "track": TrackType.WM_TRACK.value if wm_flag else TrackType.VLA_TRACK.value,
      "division": str(raw_sub.get("division", "CLOSED")),
      "clock": str(raw_sub.get("clock", "corl2026_sprint")),
      "scenario": str(raw_sub.get("scenario", "S1_INTERACTIVE_WM" if wm_flag else "P1_ONBOARD")),
      "perf_cap": str(raw_sub.get("perf_cap", PerformanceCapId.CAP_SINGLE_CHIP.value)),
      "num_devices": int(raw_sub.get("num_devices", 1)),
      "model_family": str(raw_sub.get("model_family", baseline_id)),
      "model_name": str(raw_sub.get("model_name", baseline_id)),
      "model_id": str(raw_sub.get("model_id") or baseline_id),
      "baseline_model_id": baseline_id,
      "canonical_challenge_model": baseline_id in CANONICAL_V1_MODELS,
      "precision": str(raw_sub.get("precision", "BF16")),
      "hardware_sku": sku,
      "hardware_display_name": CANONICAL_WEBSITE_SINGLE_CHIP_SKUS.get(
          sku, str(sku_spec.get("display_name", sku))
      ),
      "vendor": str(sku_spec.get("vendor", "UNKNOWN")),
      "status": str(raw_sub.get("status", RunStatus.CERTIFIED.value)),
      "reason": raw_sub.get("reason"),
      "submission_phase": str(raw_sub.get("submission_phase", "OPEN_RECIPE")),
      "recipe_badge": str(raw_sub.get("recipe_badge", "OPEN_RECIPE_INCLUDED")),
      "recipe_commit_sha256": raw_sub.get("recipe_commit_sha256"),
      "engine_tag": raw_sub.get("engine_tag"),
      "submitter_team": raw_sub.get("submitter_team"),
      "award_metrics": raw_sub.get("award_metrics"),
      "execution_provenance": raw_sub.get("execution_provenance") or dict(ANALYTIC_REFERENCE_PROVENANCE),
      "x_metric_name": x_key,
      "y_metric_name": y_key,
      "latency_metric_name": lat_key,
      "peak_reactivity_hz": round(peak_x, 4),
      "raw_per_chip_throughput": round(peak_raw, 4),
      "power_efficiency_per_watt": peak_watt,
      "throughput_per_100_tops": round(peak_100_tops, 4),
      "throughput_per_dollar_hr": round(peak_dollar_hr, 4),
      "throughput_per_1k_usd": round(peak_tco_1k, 4),
      "throughput_per_tbps_bw": round(peak_tbps, 4),
      "same_chip_speedup_vs_bf16_ref": round(peak_speedup, 4),
      "iso_compute_and_bw_efficiency": {
          "throughput_per_100_tops": round(peak_100_tops, 4),
          "throughput_per_tbps_bw": round(peak_tbps, 4),
          "same_chip_speedup_vs_bf16_ref": round(peak_speedup, 4),
      },
      "tco_efficiency_per_dollar": round(peak_tco_rate, 4),
      "iso_reactivity_slices": iso_slices,
      "pareto_frontier": compact_frontier,
      "semianalysis_4way_pareto_frontiers": frontiers_4way,
      "y_axis_frontiers": y_axis_frontiers,
      "attestation_sha256": raw_sub.get("attestation_sha256"),
  }


def _load_submissions_from_source(
    source: Union[str, Path, Dict[str, Any], Sequence[Union[str, Path, Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
  """Load raw submission or Stage 1/2 report dictionaries from path(s) or in-memory object(s)."""
  if isinstance(source, dict):
    if "submissions" in source and isinstance(source["submissions"], list):
      return [dict(item) for item in source["submissions"]]
    return [dict(source)]

  if isinstance(source, (str, Path)):
    loaded = load_yaml_or_json(source)
    if "submissions" in loaded and isinstance(loaded["submissions"], list):
      return [dict(item) for item in loaded["submissions"]]
    return [dict(loaded)]

  items: List[Dict[str, Any]] = []
  for elem in source:
    items.extend(_load_submissions_from_source(elem))
  return items


def export_verified_runs_to_website_json(
    leaderboard_json_path: Union[str, Path, Dict[str, Any], Sequence[Union[str, Path, Dict[str, Any]]]],
    output_website_json_path: str | Path,
) -> Dict[str, Any]:
  """Compile verified Stage 1/2 runs or leaderboard JSON into the frontend JSON payload for `efficient-embodied-ai.github.io`.

  Includes:
    - All 3 canonical EEB v1.0 Closed Division challenge models (`pi05-3b`, `gr00t-n1.7-3b`, `cosmos-predict2.5-2b`).
    - All 8 canonical cross-vendor single-chip SKUs (`Google Cloud TPU v6e-1 32GB`, `TPU v5e-1 16GB`,
      `NVIDIA Jetson AGX Thor 64GB`, `RTX 5090 32GB`, `L40S 48GB`, `RTX 4090 24GB`,
      `AMD Instinct MI210 64GB`, `Radeon PRO W7900 48GB`).
    - All 4 SemiAnalysis Y-axis normalizations (`raw_per_chip_throughput`, `power_efficiency_per_watt`,
      `throughput_per_100_tops`, `throughput_per_dollar_hr`).
    - Atomic persistence via `atomic_write_json` and `atomic_read_modify_write_json`.
  """
  raw_submissions = _load_submissions_from_source(leaderboard_json_path)
  normalized_all = [_normalize_submission_record(s) for s in raw_submissions]
  verified_runs = [
      s for s in normalized_all if s["status"] == RunStatus.CERTIFIED.value
  ]

  # Separate pools for global Pareto extraction
  vla_pool: List[Dict[str, Any]] = []
  wm_pool: List[Dict[str, Any]] = []
  for sub in verified_runs:
    wm_flag = is_wm_track(sub["track"])
    for pt in sub["pareto_frontier"]:
      tagged = dict(pt)
      tagged["recipe_id"] = sub["recipe_id"]
      tagged["model_name"] = sub["model_name"]
      tagged["baseline_model_id"] = sub["baseline_model_id"]
      tagged["precision"] = sub["precision"]
      tagged["hardware_sku"] = sub["hardware_sku"]
      tagged["hardware_display_name"] = sub["hardware_display_name"]
      tagged["vendor"] = sub["vendor"]
      if wm_flag:
        wm_pool.append(tagged)
      else:
        vla_pool.append(tagged)

  raw_vla_4way = extract_semianalysis_4way_pareto_frontiers(
      operating_points=vla_pool,
      x_key="hz_react",
      require_gate_a=True,
  )
  raw_wm_4way = extract_semianalysis_4way_pareto_frontiers(
      operating_points=wm_pool,
      x_key="hz_plan",
      require_gate_a=True,
  )
  vla_4way = {
      norm_k: [
          {k: v for k, v in pt.items() if k != "semianalysis_normalizations"}
          for pt in pts
      ]
      for norm_k, pts in raw_vla_4way.items()
  }
  wm_4way = {
      norm_k: [
          {k: v for k, v in pt.items() if k != "semianalysis_normalizations"}
          for pt in pts
      ]
      for norm_k, pts in raw_wm_4way.items()
  }

  vla_y_axis_frontiers = {
      "raw_per_chip_throughput": vla_4way["raw_per_chip_throughput"],
      "power_efficiency_per_watt": vla_4way["power_efficiency_per_watt"],
      "throughput_per_100_tops": vla_4way["iso_compute_and_bw_efficiency"],
      "throughput_per_dollar_hr": vla_4way["tco_efficiency_per_dollar"],
  }
  wm_y_axis_frontiers = {
      "raw_per_chip_throughput": wm_4way["raw_per_chip_throughput"],
      "power_efficiency_per_watt": wm_4way["power_efficiency_per_watt"],
      "throughput_per_100_tops": wm_4way["iso_compute_and_bw_efficiency"],
      "throughput_per_dollar_hr": wm_4way["tco_efficiency_per_dollar"],
  }

  # Canonical 3-model EEB v1.0 Closed Division registry + per-model verified runs
  canonical_models_payload: Dict[str, Dict[str, Any]] = {}
  for model_id in CANONICAL_V1_MODELS:
    ckpt_meta = dict(MODERN_REFERENCE_CHECKPOINTS.get(model_id, {}))
    model_runs = [s for s in verified_runs if s["baseline_model_id"] == model_id]
    canonical_models_payload[model_id] = {
        "model_id": model_id,
        "track": ckpt_meta.get("track", TrackType.VLA_TRACK.value),
        "division": "CLOSED",
        "hf_repo": ckpt_meta.get("hf_repo") or ckpt_meta.get("hf_repo_id", ""),
        "architecture": ckpt_meta.get("architecture", ""),
        "params_b": ckpt_meta.get("params_b", 3.0),
        "active_params_b": ckpt_meta.get("active_params_b", ckpt_meta.get("params_b", 3.0)),
        "osi_license": ckpt_meta.get("osi_license") or ckpt_meta.get("license", "Apache-2.0"),
        "rewarded_optimizations": ckpt_meta.get("rewarded_optimizations", []),
        "num_verified_runs": len(model_runs),
        "verified_runs": model_runs,
    }

  # Cross-vendor single-chip SKUs catalog
  cross_vendor_skus_payload: Dict[str, Dict[str, Any]] = {}
  for sku_id, short_label in CANONICAL_WEBSITE_SINGLE_CHIP_SKUS.items():
    spec = HARDWARE_SKU_SPECS[sku_id]
    sku_runs = [s for s in verified_runs if s["hardware_sku"] == sku_id]
    cross_vendor_skus_payload[sku_id] = {
        "sku_id": sku_id,
        "short_label": short_label,
        "display_name": spec["display_name"],
        "vendor": spec["vendor"],
        "num_devices": int(spec.get("num_devices", 1)),
        "vram_gb": round(float(spec["vram_mb"]) / 1024.0, 1),
        "fp16_tflops": float(spec["fp16_tflops"]),
        "fp8_tflops": float(spec["fp8_tflops"]),
        "mem_bw_gbps": float(spec["mem_bw_gbps"]),
        "tdp_watts": float(spec["tdp_watts"]),
        "hourly_cost_usd": float(spec["hourly_cost_usd"]),
        "perf_cap": str(spec.get("default_perf_cap", PerformanceCapId.CAP_SINGLE_CHIP.value)),
        "num_verified_runs": len(sku_runs),
    }

  # Build top-level website payload
  vla_verified = [s for s in verified_runs if not is_wm_track(s["track"])]
  wm_verified = [s for s in verified_runs if is_wm_track(s["track"])]

  num_live = sum(
      1
      for s in verified_runs
      if isinstance(s.get("execution_provenance"), dict)
      and bool(s["execution_provenance"].get("is_live_empirical", False))
  )
  num_analytic = len(verified_runs) - num_live
  prov_summary = {
      "num_live_empirical_submissions": num_live,
      "num_analytic_reference_submissions": num_analytic,
      "provenance_banner": (
          "FOUNDING_SEASON_OPEN_AWAITING_SUBMISSIONS"
          if len(verified_runs) == 0
          else (
              "MIXED_LIVE_AND_ANALYTIC_REFERENCE"
              if (num_live > 0 and num_analytic > 0)
              else (
                  "LIVE_EMPIRICAL_RUNNER_EXECUTION"
                  if num_live > 0
                  else "ANALYTIC_ROOFLINE_REFERENCE_PROJECTION (SEEDED REFERENCE BASELINES ARE ANALYTIC PROJECTIONS, NOT LIVE HARDWARE RUNS)"
              )
          )
      ),
  }

  from embodied_bench.leaderboard_compiler import (
      GAPLESS_REALTIME_SLA_SPEC,
      SEASON1_OFFICIAL_HARDWARE_TARGETS,
      SUBMISSION_PROTOCOL_SPEC,
      compute_track_award_rankings,
  )

  award_rankings = compute_track_award_rankings(
      [s for s in verified_runs if isinstance(s.get("award_metrics"), dict)]
  )

  website_payload: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "site": "efficient-embodied-ai.github.io",
      "benchmark_suite": "EmbodiedX / embodied-efficiency-bench v1.0",
      "generated_at": now_pacific_iso(),
      "execution_provenance_summary": prov_summary,
      "season1_official_hardware_targets": dict(SEASON1_OFFICIAL_HARDWARE_TARGETS),
      "submission_protocol": dict(SUBMISSION_PROTOCOL_SPEC),
      "gapless_realtime_sla": dict(GAPLESS_REALTIME_SLA_SPEC),
      "award_rankings": award_rankings,
      "performance_cap_envelope": PERFORMANCE_CAP_ENVELOPES[
          PerformanceCapId.CAP_SINGLE_CHIP.value
      ].to_dict(),
      "canonical_models": canonical_models_payload,
      "canonical_model_ids": list(CANONICAL_V1_MODELS),
      "cross_vendor_single_chip_skus": cross_vendor_skus_payload,
      "cross_vendor_sku_labels": list(CANONICAL_WEBSITE_SINGLE_CHIP_SKUS.values()),
      "y_axis_normalizations": {
          norm_key: {
              **norm_meta,
              "vla_global_frontier": vla_y_axis_frontiers[norm_key],
              "wm_global_frontier": wm_y_axis_frontiers[norm_key],
          }
          for norm_key, norm_meta in SEMIANALYSIS_Y_AXIS_NORMALIZATIONS.items()
      },
      "y_axis_keys": list(SEMIANALYSIS_Y_AXIS_NORMALIZATIONS.keys()),
      "num_total_submissions": len(normalized_all),
      "num_verified_submissions": len(verified_runs),
      "tracks": {
          TrackType.VLA_TRACK.value: {
              "track_id": TrackType.VLA_TRACK.value,
              "title": "Track 1: Real-Time VLA Policy Track (pi0.5-3B [1A] & GR00T-N1.7-3B [1B], Closed Division)",
              "x_key": "hz_react",
              "x_axis_label": "p99 Chunk Rate (Chunks/s) [1000 / L_p99, ms]",
              "y_key": "actions_per_sec",
              "num_verified_submissions": len(vla_verified),
              "global_pareto_frontier": vla_y_axis_frontiers["raw_per_chip_throughput"],
              "semianalysis_4way_global_pareto_frontiers": vla_4way,
              "y_axis_frontiers": vla_y_axis_frontiers,
              "submissions": vla_verified,
          },
          TrackType.WM_TRACK.value: {
              "track_id": TrackType.WM_TRACK.value,
              "title": "Track 2: Action-Conditioned World Model Track (Cosmos-Predict2.5-2B, Closed Division)",
              "x_key": "hz_plan",
              "x_axis_label": "p99 Rollout Rate (Rollouts/s | 16x = Stream FPS)",
              "y_key": "predictions_per_sec",
              "num_verified_submissions": len(wm_verified),
              "global_pareto_frontier": wm_y_axis_frontiers["raw_per_chip_throughput"],
              "semianalysis_4way_global_pareto_frontiers": wm_4way,
              "y_axis_frontiers": wm_y_axis_frontiers,
              "submissions": wm_verified,
          },
      },
      "entries": verified_runs,
      "submissions": verified_runs,
  }

  out_path = Path(output_website_json_path)

  # Write atomically via atomic_write_json and update export metadata via atomic_read_modify_write_json
  atomic_write_json(out_path, website_payload)
  result = atomic_read_modify_write_json(
      target_path=out_path,
      modifier_fn=lambda state: {**state, "exported_at": now_pacific_iso()},
      default_factory=lambda: website_payload,
  )
  return result


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="export_to_website",
      description="Export verified EmbodiedX runs or leaderboard JSON to website payload JSON.",
  )
  parser.add_argument(
      "--input",
      "-i",
      required=True,
      nargs="+",
      help="Path(s) to leaderboard.json or Stage 1/2 evaluation report JSON file(s).",
  )
  parser.add_argument(
      "--output",
      "-o",
      required=True,
      help="Destination path for website leaderboard JSON.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  input_arg: Union[str, Sequence[str]] = (
      args.input[0] if len(args.input) == 1 else args.input
  )
  payload = export_verified_runs_to_website_json(
      leaderboard_json_path=input_arg,
      output_website_json_path=args.output,
  )
  print(
      f"Exported {payload['num_verified_submissions']} verified run(s) across "
      f"{len(payload['canonical_model_ids'])} canonical models and "
      f"{len(payload['cross_vendor_single_chip_skus'])} cross-vendor single-chip SKUs to {args.output}."
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
