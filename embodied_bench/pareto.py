"""Pareto frontier domination stripping and log-linear Iso-Reactivity Slicing for EmbodiedX."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from embodied_bench.measure import compute_semianalysis_normalizations
from embodied_bench.schema import (
    ISO_REACTIVITY_TARGETS_MS,
    WM_ISO_REACTIVITY_TARGETS_MS,
    is_wm_track,
)


def is_pareto_efficient(xy_points: np.ndarray, tie_atol: float = 1e-9) -> np.ndarray:
  """Return boolean mask of non-dominated points where BOTH X (Reactivity Hz) and Y (Throughput) are maximized.

  A point i is dominated if there exists j != i such that:
    X[j] >= X[i] - tie_atol and Y[j] >= Y[i] - tie_atol
    and (X[j] > X[i] + tie_atol or Y[j] > Y[i] + tie_atol).
  Exact `(X, Y)` ties (within `tie_atol`) both survive as Co-SOTA frontier entries so no submission
  from one vendor arbitrarily shadows an identical-performing submission from another vendor.
  """
  pts = np.asarray(xy_points, dtype=np.float64)
  if pts.ndim != 2 or pts.shape[1] != 2:
    raise ValueError(
        f"Expected xy_points of shape (N, 2), got {pts.shape}. "
        "Pass an array of [reactivity_hz, throughput] pairs."
    )
  n = pts.shape[0]
  keep = np.ones(n, dtype=bool)
  tol = max(0.0, float(tie_atol))
  for i in range(n):
    for j in range(n):
      if i == j:
        continue
      ge_both = (pts[j, 0] >= pts[i, 0] - tol) and (pts[j, 1] >= pts[i, 1] - tol)
      gt_either = (pts[j, 0] > pts[i, 0] + tol) or (pts[j, 1] > pts[i, 1] + tol)
      if ge_both and gt_either:
        keep[i] = False
        break
  return keep


def extract_pareto_frontier(
    operating_points: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
    require_gate_a: bool = True,
) -> List[Dict[str, Any]]:
  """Filter admissible operating points, strip dominated points, preserve Co-SOTA ties, and sort deterministically."""
  admissible: List[Dict[str, Any]] = []
  optional_y_keys = {
      "power_efficiency_per_watt",
      "actions_per_watt",
      "predictions_per_watt",
      "frames_per_watt",
  }
  for pt in operating_points:
    if require_gate_a and not pt.get("gate_a_passed", True):
      continue
    if x_key not in pt or (y_key not in pt and y_key not in optional_y_keys):
      raise KeyError(
          f"Operating point missing required keys ({x_key!r}, {y_key!r}). "
          f"Available keys: {sorted(pt.keys())}"
      )
    if pt.get(x_key) is None or pt.get(y_key) is None:
      continue
    admissible.append(dict(pt))

  if not admissible:
    return []

  xy = np.array([[float(p[x_key]), float(p[y_key])] for p in admissible], dtype=np.float64)
  mask = is_pareto_efficient(xy)
  frontier = [admissible[i] for i in range(len(admissible)) if bool(mask[i])]
  # Sort from highest reactivity Hz (lowest latency) to lowest reactivity Hz (highest throughput),
  # with deterministic lexicographic tie-breaking for Co-SOTA points.
  frontier.sort(
      key=lambda item: (
          -float(item[x_key]),
          float(item[y_key]),
          str(item.get("recipe_id", item.get("submission_id", ""))),
          str(item.get("hardware_sku", "")),
          str(item.get("point_id", "")),
      )
  )
  return frontier


def extract_semianalysis_4way_pareto_frontiers(
    operating_points: Sequence[Dict[str, Any]],
    x_key: str,
    require_gate_a: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
  """Extract the 4 SemiAnalysis-style normalized Pareto frontiers against `x_key` (`hz_react` or `hz_plan`).

  Returns a dictionary with keys:
    - `raw_per_chip_throughput`: Frontier maximizing `raw_per_chip_throughput` vs `x_key`.
    - `power_efficiency_per_watt`: Frontier maximizing `power_efficiency_per_watt` vs `x_key`.
    - `iso_compute_and_bw_efficiency`: Frontier maximizing `throughput_per_100_tops` vs `x_key`.
    - `tco_efficiency_per_dollar`: Frontier maximizing `tco_efficiency_per_dollar` (`throughput_per_1k_usd`) vs `x_key`.
  """
  enriched: List[Dict[str, Any]] = []
  for pt in operating_points:
    item = dict(pt)
    raw_tput = float(
        item.get(
            "raw_per_chip_throughput",
            item.get(
                "raw_single_device_throughput",
                item.get(
                    "predictions_per_sec",
                    item.get(
                        "executed_actions_per_sec",
                        item.get(
                            "actions_per_sec",
                            item.get("generated_frames_per_sec", item.get("frames_per_sec", item.get("fps", 0.0))),
                        ),
                    ),
                ),
            ),
        )
    )
    sku = str(item.get("hardware_sku", "RTX_4090_24GB"))
    hw_tel = item.get("hardware_telemetry") if isinstance(item.get("hardware_telemetry"), dict) else {}
    existing_sa = (
        item.get("semianalysis_normalizations")
        if isinstance(item.get("semianalysis_normalizations"), dict)
        else {}
    )
    power_unavail = (
        hw_tel.get("power_provenance") == "UNAVAILABLE"
        or existing_sa.get("power_provenance") == "UNAVAILABLE"
        or ("power_efficiency_per_watt" in item and item["power_efficiency_per_watt"] is None)
    )
    dev_w = hw_tel.get("mean_power_watts")
    sys_w = hw_tel.get("system_power_watts")
    ref_speedup = float(item.get("speedup_vs_reference", 1.0))
    ref_tput = (raw_tput / ref_speedup) if ref_speedup > 0 else raw_tput
    sa = compute_semianalysis_normalizations(
        raw_throughput=raw_tput,
        hardware_sku=sku,
        device_power_watts=float(dev_w) if dev_w is not None else None,
        system_power_watts=float(sys_w) if sys_w is not None else None,
        same_chip_bf16_ref_throughput=ref_tput,
    )
    if power_unavail:
      sa["power_efficiency_per_watt"] = None
      sa["system_power_watts"] = None
      item["power_efficiency_per_watt"] = None
    else:
      item.setdefault("power_efficiency_per_watt", sa["power_efficiency_per_watt"])
    item.setdefault("raw_per_chip_throughput", sa["raw_per_chip_throughput"])
    item.setdefault("throughput_per_100_tops", sa["throughput_per_100_tops"])
    item.setdefault("throughput_per_tbps_bw", sa["throughput_per_tbps_bw"])
    item.setdefault("same_chip_speedup_vs_bf16_ref", sa["same_chip_speedup_vs_bf16_ref"])
    item.setdefault("throughput_per_1k_usd", sa["throughput_per_1k_usd"])
    item.setdefault("tco_efficiency_per_dollar", sa["tco_efficiency_per_dollar"])
    item.setdefault("throughput_per_dollar_hr", sa["throughput_per_dollar_hr"])
    item.setdefault("semianalysis_normalizations", sa)
    enriched.append(item)

  return {
      "raw_per_chip_throughput": extract_pareto_frontier(
          enriched, x_key=x_key, y_key="raw_per_chip_throughput", require_gate_a=require_gate_a
      ),
      "power_efficiency_per_watt": extract_pareto_frontier(
          enriched, x_key=x_key, y_key="power_efficiency_per_watt", require_gate_a=require_gate_a
      ),
      "iso_compute_and_bw_efficiency": extract_pareto_frontier(
          enriched, x_key=x_key, y_key="throughput_per_100_tops", require_gate_a=require_gate_a
      ),
      "tco_efficiency_per_dollar": extract_pareto_frontier(
          enriched, x_key=x_key, y_key="tco_efficiency_per_dollar", require_gate_a=require_gate_a
      ),
  }


def interpolate_log_linear_at_latency(
    latencies_ms: Sequence[float],
    throughputs: Sequence[float],
    target_latency_ms: float,
) -> Optional[float]:
  """Compute log-linearly interpolated throughput at `target_latency_ms`.

  Returns None if the fastest operating point exceeds `target_latency_ms` (cannot meet real-time budget).
  """
  if not latencies_ms or len(latencies_ms) != len(throughputs):
    return None

  pairs = sorted(
      zip([float(l) for l in latencies_ms], [float(t) for t in throughputs]),
      key=lambda item: item[0],
  )
  min_lat, _ = pairs[0]
  if min_lat > target_latency_ms + 1e-9:
    return None

  # Check exact match first
  for lat, tput in pairs:
    if abs(lat - target_latency_ms) <= 1e-9:
      return round(tput, 4)

  # If all measured points are faster than or equal to target_latency_ms, return max achieved throughput
  max_lat, _ = pairs[-1]
  if max_lat <= target_latency_ms:
    return round(max(t for _, t in pairs), 4)

  # Find tightest bracketing pair: low_lat <= target_latency_ms < high_lat
  low_lat, low_tput = pairs[0]
  high_lat, high_tput = pairs[-1]
  for idx in range(len(pairs) - 1):
    l0, y0 = pairs[idx]
    l1, y1 = pairs[idx + 1]
    if l0 <= target_latency_ms <= l1:
      low_lat, low_tput = l0, y0
      high_lat, high_tput = l1, y1
      break

  if low_lat <= 0.0 or high_lat <= 0.0 or low_tput <= 0.0 or high_tput <= 0.0:
    alpha = (target_latency_ms - low_lat) / max(high_lat - low_lat, 1e-12)
    return round(low_tput + alpha * (high_tput - low_tput), 4)

  log_l0 = math.log(low_lat)
  log_l1 = math.log(high_lat)
  if abs(log_l1 - log_l0) <= 1e-12:
    return round(max(low_tput, high_tput), 4)

  alpha = (math.log(target_latency_ms) - log_l0) / (log_l1 - log_l0)
  log_y = (1.0 - alpha) * math.log(low_tput) + alpha * math.log(high_tput)
  return round(math.exp(log_y), 4)


def compute_iso_reactivity_slices(
    frontier_points: Sequence[Dict[str, Any]],
    latency_key: str,
    throughput_key: str,
    track: Optional[str] = None,
) -> Dict[str, Optional[float]]:
  """Compute track-calibrated Iso-Reactivity throughput slices on a Pareto frontier."""
  valid_pairs = [
      (float(p[latency_key]), float(p[throughput_key]))
      for p in frontier_points
      if p.get(latency_key) is not None and p.get(throughput_key) is not None
  ]
  latencies = [lat for lat, _ in valid_pairs]
  throughputs = [tp for _, tp in valid_pairs]

  target_map: Dict[str, float] = {}
  if track is not None and is_wm_track(track):
    target_map.update(WM_ISO_REACTIVITY_TARGETS_MS)
  target_map.update(ISO_REACTIVITY_TARGETS_MS)

  slices: Dict[str, Optional[float]] = {}
  for slice_name, budget_ms in target_map.items():
    slices[slice_name] = interpolate_log_linear_at_latency(
        latencies_ms=latencies,
        throughputs=throughputs,
        target_latency_ms=budget_ms,
    )
  return slices
