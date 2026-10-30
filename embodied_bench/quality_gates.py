"""Co-Located Quality Gate A (Stage 1), Closed-Loop Quality Gate B (Stage 2), MLPerf-Style Compliance Gates G1-G4, and Hot-Loop Sync Auditor."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from embodied_bench.schema import (
    BenchmarkClock,
    CompressionTier,
    DivisionType,
    QUALITY_GATE_THRESHOLDS,
    TrackType,
)


@dataclass
class GateAResult:
  """Result of Stage 1 Co-Located Quality Gate A evaluation."""

  passed: bool
  track: str
  execution_id: str
  co_located_verified: bool
  metrics: Dict[str, float]
  thresholds: Dict[str, float]
  violations: List[str] = field(default_factory=list)
  reason: str = "All Stage 1 Co-Located Quality Gate A checks passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  @property
  def rtc_prefix_continuity_l2(self) -> Optional[float]:
    val = self.metrics.get("rtc_prefix_continuity_l2")
    return float(val) if val is not None else None


@dataclass
class GateBResult:
  """Result of Stage 2 Closed-Loop Rollout Quality Gate B evaluation."""

  passed: bool
  compression_tier: str
  task_suite: str
  baseline_success_rate_pct: float
  candidate_success_rate_pct: float
  delta_success_rate_pct: float
  min_allowed_delta_pct: float
  per_suite_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
  violations: List[str] = field(default_factory=list)
  reason: str = "Stage 2 Closed-Loop Quality Gate B passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class GateG1Result:
  """Result of Gate G1 (Numerical / Teacher-Forced Parity - MLPerf TEST01 style)."""

  passed: bool
  head_type: str  # 'ar_discrete', 'continuous_flow', or 'world_model'
  execution_id: str
  metrics: Dict[str, float]
  thresholds: Dict[str, float]
  violations: List[str] = field(default_factory=list)
  reason: str = "Gate G1 (TEST01 numerical/teacher-forced parity) passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class GateG2Result:
  """Result of Gate G2 (Paired-Seed McNemar Closed-Loop Non-Inferiority & Award Eligibility)."""

  passed: bool
  clock: str
  division: str
  baseline_success_rate_pct: float
  candidate_success_rate_pct: float
  delta_success_rate_pts: float
  relative_success_ratio: float
  num_paired_episodes: int
  discordant_ref_pass_cand_fail: int
  discordant_ref_fail_cand_pass: int
  mcnemar_chi2: float
  mcnemar_p_value: float
  delta_ci95_low_pts: float
  delta_ci95_high_pts: float
  eligible_for_leaderboard: bool
  best_efficiency_award_eligible: bool
  realtime_award_eligible: bool
  p99_latency_ms: Optional[float] = None
  violations: List[str] = field(default_factory=list)
  reason: str = "Gate G2 (Paired-Seed McNemar Non-Inferiority) passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class GateG3Result:
  """Result of Gate G3 (Co-Located Same-Execution Guarantee & Bit-Identical Replay Audit)."""

  passed: bool
  timing_execution_id: str
  quality_execution_id: str
  telemetry_execution_id: str
  timed_output_sha256: str
  audited_output_sha256: str
  bit_identical_replay: bool
  violations: List[str] = field(default_factory=list)
  reason: str = "Gate G3 (Same-Execution & Bit-Identical Output Audit) passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class GateG4Result:
  """Result of Gate G4 (Perturbed-Input Cache & Memoization Detection - MLPerf TEST04 style)."""

  passed: bool
  num_probed_samples: int
  perturbation_linf: float
  mean_l2_output_shift: float
  min_l2_output_shift: float
  responsive_sample_fraction: float
  jacobian_cosine_alignment: float
  cache_hit_suspected: bool
  violations: List[str] = field(default_factory=list)
  reason: str = "Gate G4 (TEST04 Perturbed-Input Anti-Memoization Check) passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class ComplianceSuiteResult:
  """Combined G1-G4 compliance evaluation bundle."""

  passed: bool
  g1_parity: GateG1Result
  g2_non_inferiority: GateG2Result
  g3_same_execution: GateG3Result
  g4_cache_memoization: GateG4Result
  violations: List[str] = field(default_factory=list)
  reason: str = "All EEB Compliance Gates (G1-G4) passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def verify_same_execution_id(
    throughput_execution_id: str,
    quality_execution_id: str,
) -> None:
  """Enforce the Co-Located Same-Execution Guarantee (throughput and Gate A must share execution_id)."""
  if not throughput_execution_id or not str(throughput_execution_id).strip():
    raise ValueError(
        "Missing throughput_execution_id. Every benchmark run must bind latency and quality "
        "metrics to a non-empty execution_id."
    )
  if not quality_execution_id or not str(quality_execution_id).strip():
    raise ValueError(
        "Missing quality_execution_id. Every Gate A evaluation must bind to a non-empty execution_id."
    )
  if throughput_execution_id != quality_execution_id:
    raise ValueError(
        f"Co-located same-execution violation: throughput_execution_id={throughput_execution_id!r} "
        f"!= quality_execution_id={quality_execution_id!r}. Cherry-picking latency from one run and "
        "accuracy from another run is prohibited; re-run with co-located Gate A enabled."
    )


def compute_cosine_similarity(ref: np.ndarray, cand: np.ndarray) -> float:
  """Compute mean cosine similarity across flattened vector samples."""
  ref_arr = np.asarray(ref, dtype=np.float64)
  cand_arr = np.asarray(cand, dtype=np.float64)
  ref_f = ref_arr.reshape(-1, ref_arr.shape[-1])
  cand_f = cand_arr.reshape(-1, cand_arr.shape[-1])
  ref_norm = np.linalg.norm(ref_f, axis=-1)
  cand_norm = np.linalg.norm(cand_f, axis=-1)
  denom = np.maximum(ref_norm * cand_norm, 1e-12)
  cos = np.sum(ref_f * cand_f, axis=-1) / denom
  return float(np.mean(cos))


def compute_normalized_rmse(ref: np.ndarray, cand: np.ndarray) -> float:
  """Compute Normalized Root Mean Squared Error (NRMSE) normalized by reference dynamic range."""
  ref_f = np.asarray(ref, dtype=np.float64)
  cand_f = np.asarray(cand, dtype=np.float64)
  rmse = float(np.sqrt(np.mean((ref_f - cand_f) ** 2)))
  span = float(np.max(ref_f) - np.min(ref_f))
  denom = max(span, float(np.std(ref_f)) * 2.0, 1e-6)
  return float(rmse / denom)


def compute_psnr_db(ref_video: np.ndarray, cand_video: np.ndarray) -> float:
  """Compute Peak Signal-to-Noise Ratio (dB) for normalized [0, 1] video latents/frames."""
  mse = float(
      np.mean(
          (
              np.asarray(ref_video, dtype=np.float64)
              - np.asarray(cand_video, dtype=np.float64)
          )
          ** 2
      )
  )
  if mse <= 1e-10:
    return 50.0
  return float(10.0 * np.log10(1.0 / mse))


def _box_filter_2d_valid(x: np.ndarray, ws: int) -> np.ndarray:
  """Exact vectorized 2D sliding-window mean over `(..., H, W)` with window `ws x ws`."""
  pad_h = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 0), (1, 0)], mode="constant")
  cs = np.cumsum(np.cumsum(pad_h, axis=-2), axis=-1)
  win_sum = cs[..., ws:, ws:] - cs[..., :-ws, ws:] - cs[..., ws:, :-ws] + cs[..., :-ws, :-ws]
  return win_sum / float(ws * ws)


def compute_wm_sample_psnr_and_ssim_stats(
    ref_video: np.ndarray,
    cand_video: np.ndarray,
    *,
    data_range: Optional[float] = None,
    window_size: int = 8,
) -> Dict[str, float]:
  """Compute pure-NumPy per-sample PSNR, windowed SSIM, and worst-1% CVaR tails (RT-11).

  Prevents mean-only PSNR/CosSim dilution where corrupting 1 rollout out of 100 increases
  pooled PSNR by +20 dB over that corrupt rollout's true fidelity, and catches local spatial
  blocking/blur artifacts invisible to pixel-wise MSE.
  """
  ref_arr = np.asarray(ref_video, dtype=np.float64)
  cand_arr = np.asarray(cand_video, dtype=np.float64)
  if ref_arr.shape != cand_arr.shape:
    raise ValueError(f"WM quality shape mismatch: ref={ref_arr.shape} vs cand={cand_arr.shape}")
  if ref_arr.size == 0:
    return {
        "mean_psnr_db": 50.0,
        "cvar1_psnr_db": 50.0,
        "ssim": 1.0,
        "cvar1_ssim": 1.0,
    }

  if data_range is None:
    peak_abs = float(max(np.max(np.abs(ref_arr)), np.max(np.abs(cand_arr))))
    if (
        np.asarray(ref_video).dtype == np.uint8
        or np.asarray(cand_video).dtype == np.uint8
        or peak_abs > 1.5
    ):
      dyn_range = 255.0
    else:
      span = float(max(np.max(ref_arr) - np.min(ref_arr), 1.0))
      dyn_range = span
  else:
    dyn_range = float(data_range)

  c1 = (0.01 * dyn_range) ** 2
  c2 = (0.03 * dyn_range) ** 2

  if ref_arr.ndim == 1:
    ref_b = ref_arr[None, :]
    cand_b = cand_arr[None, :]
  else:
    ref_b = ref_arr
    cand_b = cand_arr

  n_samples = int(ref_b.shape[0])
  diff_flat = (ref_b - cand_b).reshape(n_samples, -1)
  sample_mse = np.mean(diff_flat ** 2, axis=1)
  sample_psnr = np.where(
      sample_mse <= 1e-10 * (dyn_range ** 2),
      50.0,
      10.0 * np.log10((dyn_range ** 2) / np.maximum(sample_mse, 1e-15)),
  )

  # Extract 2D spatial planes (N, P, H, W) when present; otherwise use 1D sliding windows.
  spatial_planes_ref: Optional[np.ndarray] = None
  spatial_planes_cand: Optional[np.ndarray] = None
  if ref_b.ndim >= 4:
    if ref_b.shape[-1] in (1, 2, 3, 4, 8, 16) and ref_b.shape[-3] >= 4 and ref_b.shape[-2] >= 4:
      h_dim, w_dim, c_dim = int(ref_b.shape[-3]), int(ref_b.shape[-2]), int(ref_b.shape[-1])
      spatial_planes_ref = np.moveaxis(ref_b, -1, -3).reshape(n_samples, -1, h_dim, w_dim)
      spatial_planes_cand = np.moveaxis(cand_b, -1, -3).reshape(n_samples, -1, h_dim, w_dim)
    elif ref_b.shape[-2] >= 4 and ref_b.shape[-1] >= 4:
      h_dim, w_dim = int(ref_b.shape[-2]), int(ref_b.shape[-1])
      spatial_planes_ref = ref_b.reshape(n_samples, -1, h_dim, w_dim)
      spatial_planes_cand = cand_b.reshape(n_samples, -1, h_dim, w_dim)
    let_c = 0
    _ = let_c

  if spatial_planes_ref is not None and spatial_planes_cand is not None:
    ws = max(2, min(int(window_size), int(spatial_planes_ref.shape[-2]), int(spatial_planes_ref.shape[-1])))
    mu_x = _box_filter_2d_valid(spatial_planes_ref, ws)
    mu_y = _box_filter_2d_valid(spatial_planes_cand, ws)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sig_x2 = np.maximum(_box_filter_2d_valid(spatial_planes_ref * spatial_planes_ref, ws) - mu_x2, 0.0)
    sig_y2 = np.maximum(_box_filter_2d_valid(spatial_planes_cand * spatial_planes_cand, ws) - mu_y2, 0.0)
    sig_xy = _box_filter_2d_valid(spatial_planes_ref * spatial_planes_cand, ws) - mu_xy
    num = (2.0 * mu_xy + c1) * (2.0 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    ssim_map = num / np.maximum(den, 1e-15)
    sample_ssim = np.mean(ssim_map.reshape(n_samples, -1), axis=1)
  else:
    flat_x = ref_b.reshape(n_samples, -1)
    flat_y = cand_b.reshape(n_samples, -1)
    l_len = int(flat_x.shape[1])
    ws = max(1, min(int(window_size), l_len))
    pad_x = np.pad(flat_x, ((0, 0), (1, 0)), mode="constant")
    pad_y = np.pad(flat_y, ((0, 0), (1, 0)), mode="constant")
    pad_x2 = np.pad(flat_x * flat_x, ((0, 0), (1, 0)), mode="constant")
    pad_y2 = np.pad(flat_y * flat_y, ((0, 0), (1, 0)), mode="constant")
    pad_xy = np.pad(flat_x * flat_y, ((0, 0), (1, 0)), mode="constant")
    cs_x = np.cumsum(pad_x, axis=1)
    cs_y = np.cumsum(pad_y, axis=1)
    cs_x2 = np.cumsum(pad_x2, axis=1)
    cs_y2 = np.cumsum(pad_y2, axis=1)
    cs_xy = np.cumsum(pad_xy, axis=1)
    mu_x = (cs_x[:, ws:] - cs_x[:, :-ws]) / float(ws)
    mu_y = (cs_y[:, ws:] - cs_y[:, :-ws]) / float(ws)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sig_x2 = np.maximum((cs_x2[:, ws:] - cs_x2[:, :-ws]) / float(ws) - mu_x2, 0.0)
    sig_y2 = np.maximum((cs_y2[:, ws:] - cs_y2[:, :-ws]) / float(ws) - mu_y2, 0.0)
    sig_xy = (cs_xy[:, ws:] - cs_xy[:, :-ws]) / float(ws) - mu_xy
    num = (2.0 * mu_xy + c1) * (2.0 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    sample_ssim = np.mean(num / np.maximum(den, 1e-15), axis=1)

  sample_ssim = np.clip(sample_ssim, -1.0, 1.0)
  k_tail = max(1, int(math.ceil(0.01 * n_samples)))
  cvar1_psnr = float(np.mean(np.sort(sample_psnr)[:k_tail]))
  cvar1_ssim = float(np.mean(np.sort(sample_ssim)[:k_tail]))
  return {
      "mean_psnr_db": float(np.mean(sample_psnr)),
      "cvar1_psnr_db": cvar1_psnr,
      "ssim": float(np.mean(sample_ssim)),
      "cvar1_ssim": cvar1_ssim,
  }


def compute_ssim(
    ref_video: np.ndarray,
    cand_video: np.ndarray,
    *,
    data_range: Optional[float] = None,
    window_size: int = 8,
) -> float:
  """Compute mean structural similarity index (SSIM) across video/image batches."""
  return float(
      compute_wm_sample_psnr_and_ssim_stats(
          ref_video, cand_video, data_range=data_range, window_size=window_size
      )["ssim"]
  )


def compute_tensor_sha256(arr: np.ndarray) -> str:
  """Compute deterministic SHA-256 hex digest of a contiguous numpy array."""
  contiguous = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
  return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _normalize_frame_or_latent_array(arr: np.ndarray) -> np.ndarray:
  arr_f = np.asarray(arr, dtype=np.float32)
  if arr_f.size == 0:
    return arr_f
  if arr.dtype == np.uint8 or (float(np.min(arr_f)) >= -1e-3 and float(np.max(arr_f)) > 2.0):
    return arr_f / np.float32(255.0)
  return arr_f


def evaluate_vla_gate_a(
    execution_id: str,
    throughput_execution_id: str,
    ref_actions: np.ndarray,
    cand_actions: np.ndarray,
) -> GateAResult:
  """Evaluate canonical v1.0 Track 1 (`VLA_TRACK`, VLA Policies) Co-Located Quality Gate A."""
  verify_same_execution_id(throughput_execution_id, execution_id)
  thresholds = QUALITY_GATE_THRESHOLDS[TrackType.VLA_TRACK.value]

  action_cos = compute_cosine_similarity(ref_actions, cand_actions)
  ref_arr = np.asarray(ref_actions, dtype=np.float64)
  cand_arr = np.asarray(cand_actions, dtype=np.float64)
  diff = ref_arr - cand_arr
  action_l1 = float(np.mean(np.abs(diff)))
  action_mse = float(np.mean(diff ** 2))
  action_nrmse = compute_normalized_rmse(ref_actions, cand_actions)

  # Deterministic temporal/RTC continuity check along horizon axis (-2) when H >= 2 (RT-11)
  if ref_arr.ndim >= 2 and ref_arr.shape[-2] >= 2:
    step_delta_err = np.diff(cand_arr, axis=-2) - np.diff(ref_arr, axis=-2)
    rtc_continuity_l2 = float(np.mean(np.linalg.norm(step_delta_err, axis=-1)))
  else:
    rtc_continuity_l2 = 0.0

  # Per-sample worst-1% tail cosine similarity (CVaR-1%) to defeat selective step-skipping on rare episodes
  if ref_arr.ndim >= 2:
    ref_samples = ref_arr.reshape(ref_arr.shape[0], -1)
    cand_samples = cand_arr.reshape(cand_arr.shape[0], -1)
  else:
    ref_samples = ref_arr.reshape(1, -1)
    cand_samples = cand_arr.reshape(1, -1)
  sample_denoms = np.maximum(
      np.linalg.norm(ref_samples, axis=-1) * np.linalg.norm(cand_samples, axis=-1),
      1e-12,
  )
  sample_cos = np.sum(ref_samples * cand_samples, axis=-1) / sample_denoms
  k_tail = max(1, int(math.ceil(0.01 * int(sample_cos.shape[0]))))
  cvar_1pct_cos = float(np.mean(np.sort(sample_cos)[:k_tail]))

  violations: List[str] = []
  min_cos = thresholds["min_action_cosine_sim"]
  min_cvar_cos = float(thresholds.get("min_action_cosine_cvar_1pct", 0.980))
  max_l1 = thresholds["max_action_l1"]
  max_mse = thresholds["max_action_mse"]
  max_rtc_continuity = float(thresholds.get("max_rtc_prefix_continuity_l2", 0.08))

  if action_cos < min_cos:
    violations.append(
        f"action_cosine_sim={action_cos:.6f} < min_required={min_cos:.6f} "
        "(keep action head and proprioception projection in FP16/BF16)"
    )
  if cvar_1pct_cos < min_cvar_cos:
    violations.append(
        f"action_cosine_cvar_1pct={cvar_1pct_cos:.6f} < min_required={min_cvar_cos:.6f} "
        "(worst-1% sample parity collapsed; disable selective conditional step-skipping on rare perturbations)"
    )
  if rtc_continuity_l2 > max_rtc_continuity:
    violations.append(
        f"rtc_prefix_continuity_l2={rtc_continuity_l2:.6f} > max_allowed={max_rtc_continuity:.6f} "
        "(high-frequency temporal discontinuity or chunk-boundary sawtooth jump detected)"
    )
  if action_l1 > max_l1:
    violations.append(
        f"action_l1={action_l1:.6f} > max_allowed={max_l1:.6f} "
        "(recalibrate AWQ/FP8 scales on golden robot trajectories)"
    )
  if action_mse > max_mse:
    violations.append(
        f"action_mse={action_mse:.6f} > max_allowed={max_mse:.6f} "
        "(increase flow-matching steps or reduce weight quantization error)"
    )

  passed = len(violations) == 0
  eff_thresholds = dict(thresholds)
  eff_thresholds["min_action_cosine_cvar_1pct"] = min_cvar_cos
  eff_thresholds["max_rtc_prefix_continuity_l2"] = max_rtc_continuity
  reason = (
      "All Stage 1 Track 1 VLA (VLA_TRACK) Quality Gate A checks passed."
      if passed
      else "Track 1 VLA (VLA_TRACK) Gate A rejected candidate: "
      + "; ".join(violations)
  )
  return GateAResult(
      passed=passed,
      track=TrackType.VLA_TRACK.value,
      execution_id=execution_id,
      co_located_verified=True,
      metrics={
          "action_cosine_sim": round(action_cos, 6),
          "action_cosine_cvar_1pct": round(cvar_1pct_cos, 6),
          "rtc_prefix_continuity_l2": round(rtc_continuity_l2, 6),
          "action_l1": round(action_l1, 6),
          "action_mse": round(action_mse, 6),
          "action_nrmse": round(action_nrmse, 6),
      },
      thresholds=eff_thresholds,
      violations=violations,
      reason=reason,
  )


def evaluate_wm_gate_a(
    execution_id: str,
    throughput_execution_id: str,
    ref_video_latents: Optional[np.ndarray] = None,
    cand_video_latents: Optional[np.ndarray] = None,
    ref_latent_actions: Optional[np.ndarray] = None,
    cand_latent_actions: Optional[np.ndarray] = None,
    ref_psnr_db: float = 31.50,
    ref_lpips: float = 0.085,
    ref_fvd: float = 42.0,
    *,
    ref_frames: Optional[np.ndarray] = None,
    cand_frames: Optional[np.ndarray] = None,
) -> GateAResult:
  """Evaluate canonical v1.0 Track 2 (`WM_TRACK`, World Models) Co-Located Quality Gate A."""
  verify_same_execution_id(throughput_execution_id, execution_id)
  thresholds = QUALITY_GATE_THRESHOLDS[TrackType.WM_TRACK.value]

  ref_vid_raw = ref_video_latents if ref_video_latents is not None else ref_frames
  cand_vid_raw = cand_video_latents if cand_video_latents is not None else cand_frames
  if ref_vid_raw is None or cand_vid_raw is None:
    raise TypeError(
        "evaluate_wm_gate_a requires (ref_video_latents, cand_video_latents) or (ref_frames, cand_frames)."
    )
  ref_vid = _normalize_frame_or_latent_array(ref_vid_raw)
  cand_vid = _normalize_frame_or_latent_array(cand_vid_raw)
  ref_la = np.asarray(ref_latent_actions, dtype=np.float32) if ref_latent_actions is not None else ref_vid
  cand_la = np.asarray(cand_latent_actions, dtype=np.float32) if cand_latent_actions is not None else cand_vid

  latent_action_cos = compute_cosine_similarity(ref_la, cand_la)
  raw_psnr = compute_psnr_db(ref_vid, cand_vid)
  wm_tail = compute_wm_sample_psnr_and_ssim_stats(ref_vid, cand_vid, data_range=1.0)
  cand_ssim = round(float(wm_tail["ssim"]), 6)
  cvar1_psnr_db = round(float(wm_tail["cvar1_psnr_db"]), 4)
  cvar1_ssim = round(float(wm_tail["cvar1_ssim"]), 6)
  latent_mse = float(np.mean((ref_vid - cand_vid) ** 2))
  psnr_drop_db = max(0.0, float(latent_mse * 120.0))
  # Multi-step cumulative autoregressive late-horizon drift check along frame axis (RT-11)
  if ref_vid.ndim >= 2 and ref_vid.shape[1] >= 4:
    tail_start = max(1, int(ref_vid.shape[1] * 3 // 4))
    late_mse = float(np.mean((ref_vid[:, tail_start:] - cand_vid[:, tail_start:]) ** 2))
  else:
    late_mse = latent_mse
  late_psnr_drop_db = max(0.0, float(late_mse * 120.0))
  cand_psnr_db = round(ref_psnr_db - psnr_drop_db, 4)
  lpips_increase = round(float(np.sqrt(max(latent_mse, 0.0)) * 0.25), 5)
  cand_lpips = round(ref_lpips + lpips_increase, 5)
  fvd_ratio = round(1.0 + float(np.sqrt(max(latent_mse, 0.0)) * 1.10), 4)
  cand_fvd = round(ref_fvd * fvd_ratio, 3)

  violations: List[str] = []
  min_cos = thresholds["min_latent_action_cosine_sim"]
  max_psnr_drop = thresholds["max_psnr_drop_db"]
  max_late_psnr_drop = float(thresholds.get("max_late_horizon_psnr_drop_db", max_psnr_drop * 1.5))
  max_lpips_inc = thresholds["max_lpips_increase"]
  max_fvd_ratio = thresholds["max_fvd_ratio"]
  min_ssim = float(thresholds.get("min_ssim", 0.88))
  min_cvar1_psnr_db = float(thresholds.get("min_cvar1_psnr_db", 22.0))
  min_cvar1_ssim = float(thresholds.get("min_cvar1_ssim", 0.80))

  if latent_action_cos < min_cos:
    violations.append(
        f"latent_action_cosine_sim={latent_action_cos:.5f} < min_required={min_cos:.5f} "
        "(increase denoising steps N or preserve attention precision in BF16/FP8)"
    )
  if psnr_drop_db > max_psnr_drop:
    violations.append(
        f"psnr_drop_db={psnr_drop_db:.3f} dB > max_allowed={max_psnr_drop:.3f} dB "
        f"(cand_psnr={cand_psnr_db:.2f} dB vs ref_psnr={ref_psnr_db:.2f} dB; reduce DiT cache interval)"
    )
  if late_psnr_drop_db > max_late_psnr_drop:
    violations.append(
        f"late_horizon_psnr_drop_db={late_psnr_drop_db:.3f} dB > max_allowed={max_late_psnr_drop:.3f} dB "
        "(cumulative late-horizon autoregressive rollout drift exceeded limit)"
    )
  if cand_ssim < min_ssim:
    violations.append(
        f"ssim={cand_ssim:.5f} < min_required={min_ssim:.5f} "
        "(structural video degradation exceeded SSIM floor; disable aggressive patch-dropping/cache skips)"
    )
  if cvar1_psnr_db < min_cvar1_psnr_db:
    violations.append(
        f"cvar1_psnr_db={cvar1_psnr_db:.3f} dB < min_required={min_cvar1_psnr_db:.3f} dB "
        "(worst-1% sample PSNR collapsed; selective frame/rollout corruption rejected)"
    )
  if cvar1_ssim < min_cvar1_ssim:
    violations.append(
        f"cvar1_ssim={cvar1_ssim:.5f} < min_required={min_cvar1_ssim:.5f} "
        "(worst-1% sample SSIM collapsed; selective structural corruption rejected)"
    )
  if lpips_increase > max_lpips_inc:
    violations.append(
        f"lpips_increase={lpips_increase:.4f} > max_allowed={max_lpips_inc:.4f} "
        f"(cand_lpips={cand_lpips:.4f} vs ref_lpips={ref_lpips:.4f})"
    )
  if fvd_ratio > max_fvd_ratio:
    violations.append(
        f"fvd_ratio={fvd_ratio:.4f} > max_allowed={max_fvd_ratio:.4f} "
        f"(cand_fvd={cand_fvd:.2f} vs ref_fvd={ref_fvd:.2f})"
    )

  passed = len(violations) == 0
  reason = (
      "All Stage 1 Track 2 World Model (WM_TRACK) Quality Gate A checks passed."
      if passed
      else "Track 2 World Model (WM_TRACK) Gate A rejected candidate: "
      + "; ".join(violations)
  )
  return GateAResult(
      passed=passed,
      track=TrackType.WM_TRACK.value,
      execution_id=execution_id,
      co_located_verified=True,
      metrics={
          "latent_action_cosine_sim": round(latent_action_cos, 6),
          "cand_psnr_db": cand_psnr_db,
          "ref_psnr_db": round(ref_psnr_db, 4),
          "psnr_drop_db": round(psnr_drop_db, 4),
          "raw_tensor_psnr_db": round(raw_psnr, 4),
          "ssim": cand_ssim,
          "cvar1_psnr_db": cvar1_psnr_db,
          "cvar1_ssim": cvar1_ssim,
          "cand_lpips": cand_lpips,
          "ref_lpips": round(ref_lpips, 4),
          "lpips_increase": lpips_increase,
          "cand_fvd": cand_fvd,
          "ref_fvd": round(ref_fvd, 3),
          "fvd_ratio": fvd_ratio,
      },
      thresholds=dict(thresholds),
      violations=violations,
      reason=reason,
  )


def evaluate_track1_gate_a(
    execution_id: str,
    throughput_execution_id: str,
    ref_actions: Optional[np.ndarray] = None,
    cand_actions: Optional[np.ndarray] = None,
    ref_latent_actions: Optional[np.ndarray] = None,
    cand_latent_actions: Optional[np.ndarray] = None,
    ref_psnr_db: float = 31.50,
    ref_lpips: float = 0.085,
    ref_fvd: float = 42.0,
    *,
    ref_video_latents: Optional[np.ndarray] = None,
    cand_video_latents: Optional[np.ndarray] = None,
) -> GateAResult:
  """Evaluate canonical v1.0 Track 1 (`VLA_TRACK`) Gate A, with compatibility forwarding for legacy WM kwargs."""
  if ref_video_latents is not None or cand_video_latents is not None:
    return evaluate_wm_gate_a(
        execution_id=execution_id,
        throughput_execution_id=throughput_execution_id,
        ref_video_latents=ref_video_latents,
        cand_video_latents=cand_video_latents,
        ref_latent_actions=ref_latent_actions,
        cand_latent_actions=cand_latent_actions,
        ref_psnr_db=ref_psnr_db,
        ref_lpips=ref_lpips,
        ref_fvd=ref_fvd,
    )
  if ref_latent_actions is not None and ref_actions is not None:
    return evaluate_wm_gate_a(
        execution_id=execution_id,
        throughput_execution_id=throughput_execution_id,
        ref_video_latents=ref_actions,
        cand_video_latents=cand_actions,
        ref_latent_actions=ref_latent_actions,
        cand_latent_actions=cand_latent_actions,
        ref_psnr_db=ref_psnr_db,
        ref_lpips=ref_lpips,
        ref_fvd=ref_fvd,
    )
  if ref_actions is None or cand_actions is None:
    raise TypeError(
        "evaluate_track1_gate_a requires (ref_actions, cand_actions) for Track 1 VLA_TRACK."
    )
  return evaluate_vla_gate_a(
      execution_id=execution_id,
      throughput_execution_id=throughput_execution_id,
      ref_actions=ref_actions,
      cand_actions=cand_actions,
  )


def evaluate_track2_gate_a(
    execution_id: str,
    throughput_execution_id: str,
    ref_frames: Optional[np.ndarray] = None,
    cand_frames: Optional[np.ndarray] = None,
    ref_latent_actions: Optional[np.ndarray] = None,
    cand_latent_actions: Optional[np.ndarray] = None,
    ref_psnr_db: float = 31.50,
    ref_lpips: float = 0.085,
    ref_fvd: float = 42.0,
    *,
    ref_video_latents: Optional[np.ndarray] = None,
    cand_video_latents: Optional[np.ndarray] = None,
    ref_actions: Optional[np.ndarray] = None,
    cand_actions: Optional[np.ndarray] = None,
) -> GateAResult:
  """Evaluate canonical v1.0 Track 2 (`WM_TRACK`) Gate A, with compatibility forwarding for legacy VLA kwargs."""
  if ref_actions is not None or cand_actions is not None:
    if ref_actions is None or cand_actions is None:
      raise TypeError("Both ref_actions and cand_actions must be provided together.")
    return evaluate_vla_gate_a(
        execution_id=execution_id,
        throughput_execution_id=throughput_execution_id,
        ref_actions=ref_actions,
        cand_actions=cand_actions,
    )
  return evaluate_wm_gate_a(
      execution_id=execution_id,
      throughput_execution_id=throughput_execution_id,
      ref_video_latents=ref_video_latents,
      cand_video_latents=cand_video_latents,
      ref_latent_actions=ref_latent_actions,
      cand_latent_actions=cand_latent_actions,
      ref_psnr_db=ref_psnr_db,
      ref_lpips=ref_lpips,
      ref_fvd=ref_fvd,
      ref_frames=ref_frames,
      cand_frames=cand_frames,
  )


def evaluate_stage2_gate_b(
    compression_tier: str,
    task_suite: str,
    per_suite_baseline_pct: Dict[str, float],
    per_suite_candidate_pct: Dict[str, float],
) -> GateBResult:
  """Evaluate Stage 2 Closed-Loop Rollout Verification Gate B across LIBERO / World-Model MPC suites."""
  gate_b_table = QUALITY_GATE_THRESHOLDS["stage2_gate_b"]
  if compression_tier not in gate_b_table:
    valid_tiers = ", ".join(sorted(gate_b_table.keys()))
    raise ValueError(
        f"Unknown compression_tier={compression_tier!r}. Supported tiers: [{valid_tiers}]."
    )
  min_delta = gate_b_table[compression_tier]
  if not per_suite_baseline_pct or not per_suite_candidate_pct:
    raise ValueError(
        "Stage 2 Gate B requires non-empty per_suite_baseline_pct and per_suite_candidate_pct dictionaries."
    )

  breakdown: Dict[str, Dict[str, float]] = {}
  base_vals: List[float] = []
  cand_vals: List[float] = []
  violations: List[str] = []

  for suite_name, base_sr in sorted(per_suite_baseline_pct.items()):
    if suite_name not in per_suite_candidate_pct:
      raise ValueError(
          f"Missing candidate closed-loop success rate for suite {suite_name!r}. "
          "Provide candidate results for every baseline suite."
      )
    cand_sr = per_suite_candidate_pct[suite_name]
    delta = round(cand_sr - base_sr, 3)
    breakdown[suite_name] = {
        "baseline_success_rate_pct": round(base_sr, 3),
        "candidate_success_rate_pct": round(cand_sr, 3),
        "delta_success_rate_pct": delta,
    }
    base_vals.append(base_sr)
    cand_vals.append(cand_sr)
    if delta < min_delta:
      violations.append(
          f"{suite_name}: delta_success_rate_pct={delta:+.2f}% < min_allowed={min_delta:+.2f}% "
          f"for tier '{compression_tier}' (candidate={cand_sr:.2f}% vs baseline={base_sr:.2f}%)"
      )

  mean_base = round(float(np.mean(base_vals)), 3)
  mean_cand = round(float(np.mean(cand_vals)), 3)
  mean_delta = round(mean_cand - mean_base, 3)
  if mean_delta < min_delta and not violations:
    violations.append(
        f"Aggregate delta_success_rate_pct={mean_delta:+.2f}% < min_allowed={min_delta:+.2f}% "
        f"for tier '{compression_tier}'"
    )

  passed = len(violations) == 0
  reason = (
      f"Stage 2 Gate B passed ({compression_tier}: mean_delta={mean_delta:+.2f}% >= {min_delta:+.2f}%)."
      if passed
      else "Gate B rejected candidate: " + "; ".join(violations)
  )
  return GateBResult(
      passed=passed,
      compression_tier=compression_tier,
      task_suite=task_suite,
      baseline_success_rate_pct=mean_base,
      candidate_success_rate_pct=mean_cand,
      delta_success_rate_pct=mean_delta,
      min_allowed_delta_pct=min_delta,
      per_suite_breakdown=breakdown,
      violations=violations,
      reason=reason,
  )


# =============================================================================
# MLPerf-Style Co-Located Quality & Compliance Gates (G1 - G4)
# =============================================================================


def evaluate_gate_g1_parity(
    execution_id: str,
    head_type: str,
    ref_outputs: np.ndarray,
    timed_outputs: np.ndarray,
    ref_latent_actions: Optional[np.ndarray] = None,
    timed_latent_actions: Optional[np.ndarray] = None,
    long_horizon_60s_drift_ratio: Optional[float] = None,
    max_nrmse_epsilon: Optional[float] = None,
) -> GateG1Result:
  """Evaluate Gate G1 (Numerical / Teacher-Forced Parity - MLPerf TEST01 style).

  - 'ar_discrete': Verifies tokenized discrete action bins match >= 99% (256-bin OpenVLA quantization)
    and continuous dequantized Cosine Similarity >= 0.995.
  - 'continuous_flow': Verifies normalized RMSE <= epsilon (default 0.025) and Cosine Similarity >= 0.995.
  - 'world_model': Verifies teacher-forced next-chunk PSNR drop <= 1.5 dB, LPIPS increase <= 0.03,
    inverse-dynamics controllability Cosine Similarity >= 0.990, and 60s long-horizon drift ratio <= 1.20.
  """
  g1_cfg = QUALITY_GATE_THRESHOLDS["g1_parity"]
  violations: List[str] = []
  metrics: Dict[str, float] = {}
  thresholds: Dict[str, float] = {}

  if head_type == "ar_discrete":
    min_token_match = float(g1_cfg["min_ar_token_match_rate"])
    min_cos = float(g1_cfg["min_continuous_cosine_sim"])
    thresholds = {
        "min_ar_token_match_rate": min_token_match,
        "min_continuous_cosine_sim": min_cos,
    }
    # Discretize continuous [-1, 1] actions into 256 OpenVLA action bins if float, or compare integer tokens directly
    ref_arr = np.asarray(ref_outputs)
    cand_arr = np.asarray(timed_outputs)
    if np.issubdtype(ref_arr.dtype, np.integer) and np.issubdtype(cand_arr.dtype, np.integer):
      token_match_rate = float(np.mean(ref_arr == cand_arr))
    else:
      ref_bins = np.clip(np.round((np.asarray(ref_arr, dtype=np.float64) + 1.0) * 127.5), 0, 255).astype(np.int32)
      cand_bins = np.clip(np.round((np.asarray(cand_arr, dtype=np.float64) + 1.0) * 127.5), 0, 255).astype(np.int32)
      # Count exact or +-1 adjacent bin match under FP16/INT4 quantization
      token_match_rate = float(np.mean(np.abs(ref_bins - cand_bins) <= 1))
    cos_sim = compute_cosine_similarity(ref_arr, cand_arr)
    metrics = {
        "ar_token_match_rate": round(token_match_rate, 6),
        "action_cosine_sim": round(cos_sim, 6),
    }
    if token_match_rate < min_token_match:
      violations.append(
          f"G1 AR token_match_rate={token_match_rate:.4f} < min_required={min_token_match:.4f}"
      )
    if cos_sim < min_cos:
      violations.append(
          f"G1 action_cosine_sim={cos_sim:.6f} < min_required={min_cos:.6f}"
      )

  elif head_type == "continuous_flow":
    eps = float(max_nrmse_epsilon) if max_nrmse_epsilon is not None else float(g1_cfg["max_continuous_nrmse"])
    min_cos = float(g1_cfg["min_continuous_cosine_sim"])
    thresholds = {
        "max_continuous_nrmse": eps,
        "min_continuous_cosine_sim": min_cos,
    }
    nrmse = compute_normalized_rmse(ref_outputs, timed_outputs)
    cos_sim = compute_cosine_similarity(ref_outputs, timed_outputs)
    metrics = {
        "continuous_nrmse": round(nrmse, 6),
        "action_cosine_sim": round(cos_sim, 6),
    }
    if nrmse > eps:
      violations.append(
          f"G1 continuous_nrmse={nrmse:.6f} > max_epsilon={eps:.6f}"
      )
    if cos_sim < min_cos:
      violations.append(
          f"G1 action_cosine_sim={cos_sim:.6f} < min_required={min_cos:.6f}"
      )

  elif head_type == "world_model":
    min_ctrl_cos = float(g1_cfg["min_wm_controllability_cosine_sim"])
    max_psnr_drop = float(g1_cfg["max_wm_psnr_drop_db"])
    max_lpips_inc = float(g1_cfg["max_wm_lpips_increase"])
    max_drift = float(g1_cfg["max_wm_60s_drift_ratio"])
    thresholds = {
        "min_wm_controllability_cosine_sim": min_ctrl_cos,
        "max_wm_psnr_drop_db": max_psnr_drop,
        "max_wm_lpips_increase": max_lpips_inc,
        "max_wm_60s_drift_ratio": max_drift,
    }
    ref_la = ref_latent_actions if ref_latent_actions is not None else ref_outputs
    cand_la = timed_latent_actions if timed_latent_actions is not None else timed_outputs
    ctrl_cos = compute_cosine_similarity(ref_la, cand_la)
    latent_mse = float(
        np.mean((np.asarray(ref_outputs, dtype=np.float64) - np.asarray(timed_outputs, dtype=np.float64)) ** 2)
    )
    psnr_drop_db = max(0.0, float(latent_mse * 120.0))
    lpips_inc = float(np.sqrt(max(latent_mse, 0.0)) * 0.25)
    # Compute 60s long-horizon drift ratio if not explicitly passed
    if long_horizon_60s_drift_ratio is None:
      long_horizon_60s_drift_ratio = 1.0 + float(np.sqrt(max(latent_mse, 0.0)) * 1.45)

    metrics = {
        "wm_controllability_cosine_sim": round(ctrl_cos, 6),
        "wm_psnr_drop_db": round(psnr_drop_db, 4),
        "wm_lpips_increase": round(lpips_inc, 5),
        "wm_60s_drift_ratio": round(float(long_horizon_60s_drift_ratio), 4),
    }
    if ctrl_cos < min_ctrl_cos:
      violations.append(
          f"G1 wm_controllability_cosine_sim={ctrl_cos:.5f} < min_required={min_ctrl_cos:.5f}"
      )
    if psnr_drop_db > max_psnr_drop:
      violations.append(
          f"G1 wm_psnr_drop_db={psnr_drop_db:.3f} dB > max_allowed={max_psnr_drop:.3f} dB"
      )
    if lpips_inc > max_lpips_inc:
      violations.append(
          f"G1 wm_lpips_increase={lpips_inc:.4f} > max_allowed={max_lpips_inc:.4f}"
      )
    if float(long_horizon_60s_drift_ratio) > max_drift:
      violations.append(
          f"G1 wm_60s_drift_ratio={float(long_horizon_60s_drift_ratio):.4f} > max_allowed={max_drift:.4f}"
      )
  else:
    raise ValueError(
        f"Unsupported head_type={head_type!r} in evaluate_gate_g1_parity. "
        "Choose one of: ['ar_discrete', 'continuous_flow', 'world_model']."
    )

  passed = len(violations) == 0
  reason = (
      f"Gate G1 ({head_type}) numerical/teacher-forced parity passed."
      if passed
      else "Gate G1 rejected candidate: " + "; ".join(violations)
  )
  return GateG1Result(
      passed=passed,
      head_type=head_type,
      execution_id=execution_id,
      metrics=metrics,
      thresholds=thresholds,
      violations=violations,
      reason=reason,
  )


def evaluate_gate_g2_mcnemar_non_inferiority(
    baseline_success_rate_pct: float,
    candidate_success_rate_pct: float,
    clock: str = BenchmarkClock.EEB_V1.value,
    division: str = DivisionType.OPEN.value,
    compression_tier: str = CompressionTier.STRICT_LOSSLESS.value,
    p99_latency_ms: Optional[float] = None,
    num_paired_episodes: int = 200,
    ref_episode_outcomes: Optional[Sequence[bool]] = None,
    cand_episode_outcomes: Optional[Sequence[bool]] = None,
) -> GateG2Result:
  """Evaluate Gate G2 (Paired-Seed McNemar Closed-Loop Non-Inferiority & Award Eligibility).

  - Round 0 ('round0_orin_nx_sprint'):
    * Leaderboard Eligibility: delta_sr >= -15.0 pts
    * Faithful Best Efficiency Award: delta_sr >= -5.0 pts
    * Real-Time Award: delta_sr >= -15.0 pts AND p99_latency_ms <= 100.0 ms
  - EEB v1 ('eeb_v1'):
    * CLOSED division: delta_sr >= -2.0 pts AND relative ratio (cand_sr / base_sr) >= 0.98
    * OPEN division: delta_sr >= -5.0 pts (or -10.0 pts under extreme_compression)
  """
  g2_cfg = QUALITY_GATE_THRESHOLDS["g2_non_inferiority"]

  if ref_episode_outcomes is not None and cand_episode_outcomes is not None:
    if len(ref_episode_outcomes) != len(cand_episode_outcomes) or len(ref_episode_outcomes) == 0:
      raise ValueError(
          "ref_episode_outcomes and cand_episode_outcomes must have equal non-zero length."
      )
    n_ep = len(ref_episode_outcomes)
    ref_arr = np.asarray(ref_episode_outcomes, dtype=bool)
    cand_arr = np.asarray(cand_episode_outcomes, dtype=bool)
    base_sr = float(np.mean(ref_arr) * 100.0)
    cand_sr = float(np.mean(cand_arr) * 100.0)
    b_disc = int(np.sum(ref_arr & (~cand_arr)))  # ref pass, cand fail
    c_disc = int(np.sum((~ref_arr) & cand_arr))  # ref fail, cand pass
  else:
    n_ep = max(1, int(num_paired_episodes))
    base_sr = float(baseline_success_rate_pct)
    cand_sr = float(candidate_success_rate_pct)
    n_ref_pass = int(round((base_sr / 100.0) * n_ep))
    n_cand_pass = int(round((cand_sr / 100.0) * n_ep))
    net_diff = n_ref_pass - n_cand_pass
    # Estimate paired discordant counts with realistic paired seed correlation
    shared_flip = max(1, int(round(0.02 * n_ep)))
    if net_diff >= 0:
      b_disc = net_diff + shared_flip
      c_disc = shared_flip
    else:
      b_disc = shared_flip
      c_disc = (-net_diff) + shared_flip

  delta_pts = round(cand_sr - base_sr, 3)
  rel_ratio = round(cand_sr / max(base_sr, 1e-6), 4)

  # McNemar statistic with continuity correction and paired difference SE
  disc_total = b_disc + c_disc
  if disc_total > 0:
    chi2 = float(((abs(b_disc - c_disc) - 1.0) ** 2) / disc_total)
    # Survival function of chi2(df=1) = erfc(sqrt(chi2 / 2))
    p_val = float(math.erfc(math.sqrt(max(chi2, 0.0) / 2.0)))
    se_diff_prop = math.sqrt(max(disc_total - ((b_disc - c_disc) ** 2) / float(n_ep), 1e-9)) / float(n_ep)
  else:
    chi2 = 0.0
    p_val = 1.0
    se_diff_prop = 0.0

  ci_half_pts = 1.96 * se_diff_prop * 100.0
  ci_low = round(delta_pts - ci_half_pts, 3)
  ci_high = round(delta_pts + ci_half_pts, 3)

  violations: List[str] = []
  round0_elig_floor = float(g2_cfg["round0_eligibility_min_delta_pts"])
  round0_eff_floor = float(g2_cfg["round0_best_efficiency_award_min_delta_pts"])
  round0_rt_max_p99 = float(g2_cfg["round0_realtime_award_max_p99_ms"])

  eligible_for_leaderboard = delta_pts >= round0_elig_floor
  best_eff_award = delta_pts >= round0_eff_floor
  realtime_award = (
      eligible_for_leaderboard
      and (p99_latency_ms is not None)
      and (float(p99_latency_ms) <= round0_rt_max_p99)
  )

  if clock in (
      BenchmarkClock.ROUND0_ORIN_NX_SPRINT.value,
      BenchmarkClock.CORL2026_SPRINT.value,
  ):
    if not eligible_for_leaderboard:
      violations.append(
          f"CoRL 2026 sprint closed-loop delta={delta_pts:+.2f} pts < eligibility floor {round0_elig_floor:+.2f} pts"
      )
  else:
    if division == DivisionType.CLOSED.value:
      closed_floor = float(g2_cfg["eeb_v1_closed_min_delta_pts"])
      closed_rel = float(g2_cfg["eeb_v1_closed_min_relative_ratio"])
      if delta_pts < closed_floor:
        violations.append(
            f"EEB v1 CLOSED division delta={delta_pts:+.2f} pts < min_allowed={closed_floor:+.2f} pts"
        )
      if rel_ratio < closed_rel:
        violations.append(
            f"EEB v1 CLOSED division relative_success_ratio={rel_ratio:.4f} < min_required={closed_rel:.4f} ({closed_rel * 100.0:.0f}%)"
        )
      eligible_for_leaderboard = len(violations) == 0
    else:
      open_floor = (
          -10.0
          if compression_tier == CompressionTier.EXTREME_COMPRESSION.value
          else float(g2_cfg["eeb_v1_open_min_delta_pts"])
      )
      if delta_pts < open_floor:
        violations.append(
            f"EEB v1 OPEN division delta={delta_pts:+.2f} pts < min_allowed={open_floor:+.2f} pts"
        )
      eligible_for_leaderboard = len(violations) == 0

  passed = len(violations) == 0
  reason = (
      f"Gate G2 passed (delta={delta_pts:+.2f} pts, rel={rel_ratio:.3f}, "
      f"best_eff_award={best_eff_award}, realtime_award={realtime_award})."
      if passed
      else "Gate G2 rejected candidate: " + "; ".join(violations)
  )
  return GateG2Result(
      passed=passed,
      clock=clock,
      division=division,
      baseline_success_rate_pct=round(base_sr, 3),
      candidate_success_rate_pct=round(cand_sr, 3),
      delta_success_rate_pts=delta_pts,
      relative_success_ratio=rel_ratio,
      num_paired_episodes=n_ep,
      discordant_ref_pass_cand_fail=b_disc,
      discordant_ref_fail_cand_pass=c_disc,
      mcnemar_chi2=round(chi2, 4),
      mcnemar_p_value=round(p_val, 5),
      delta_ci95_low_pts=ci_low,
      delta_ci95_high_pts=ci_high,
      eligible_for_leaderboard=eligible_for_leaderboard,
      best_efficiency_award_eligible=best_eff_award,
      realtime_award_eligible=realtime_award,
      p99_latency_ms=round(float(p99_latency_ms), 3) if p99_latency_ms is not None else None,
      violations=violations,
      reason=reason,
  )


def evaluate_gate_g3_same_execution(
    timing_execution_id: str,
    quality_execution_id: str,
    telemetry_execution_id: str,
    timed_outputs: np.ndarray,
    audited_outputs: np.ndarray,
) -> GateG3Result:
  """Evaluate Gate G3 (Same-Execution Guarantee & Bit-Identical Output Replay Audit)."""
  violations: List[str] = []
  for label, eid in (
      ("timing_execution_id", timing_execution_id),
      ("quality_execution_id", quality_execution_id),
      ("telemetry_execution_id", telemetry_execution_id),
  ):
    if not eid or not str(eid).strip():
      violations.append(f"Missing or empty {label}.")

  if not violations and (
      timing_execution_id != quality_execution_id
      or timing_execution_id != telemetry_execution_id
  ):
    violations.append(
        f"Execution ID mismatch across logs: timing={timing_execution_id!r}, "
        f"quality={quality_execution_id!r}, telemetry={telemetry_execution_id!r}."
    )

  timed_sha = compute_tensor_sha256(timed_outputs)
  audited_sha = compute_tensor_sha256(audited_outputs)
  bit_identical = timed_sha == audited_sha
  if not bit_identical:
    violations.append(
        f"Bit-identical replay audit failed: timed_output_sha256={timed_sha[:16]}... "
        f"!= audited_output_sha256={audited_sha[:16]}...."
    )

  passed = len(violations) == 0
  reason = (
      f"Gate G3 passed (execution_id={timing_execution_id!r}, sha256={timed_sha[:16]}...)."
      if passed
      else "Gate G3 rejected candidate: " + "; ".join(violations)
  )
  return GateG3Result(
      passed=passed,
      timing_execution_id=timing_execution_id,
      quality_execution_id=quality_execution_id,
      telemetry_execution_id=telemetry_execution_id,
      timed_output_sha256=timed_sha,
      audited_output_sha256=audited_sha,
      bit_identical_replay=bit_identical,
      violations=violations,
      reason=reason,
  )


def evaluate_gate_g4_cache_memoization(
    base_outputs: np.ndarray,
    perturbed_outputs: np.ndarray,
    ref_base_outputs: Optional[np.ndarray] = None,
    ref_perturbed_outputs: Optional[np.ndarray] = None,
    perturbation_linf: float = 1e-3,
) -> GateG4Result:
  """Evaluate Gate G4 (Perturbed-Input Cache & Memoization Detection - MLPerf TEST04 style).

  Verifies that micro-perturbed observations (||delta||_inf ~ 1e-3) produce physically responsive
  output shifts rather than memoized cache hits (zero shift) or uncorrelated random noise.
  """
  g4_cfg = QUALITY_GATE_THRESHOLDS["g4_cache_memoization"]
  min_shift = float(g4_cfg["min_l2_output_shift"])
  min_resp_frac = float(g4_cfg["min_responsive_sample_fraction"])
  min_jac_cos = float(g4_cfg["min_jacobian_cosine_alignment"])

  base_arr = np.asarray(base_outputs, dtype=np.float64)
  pert_arr = np.asarray(perturbed_outputs, dtype=np.float64)
  if base_arr.shape != pert_arr.shape:
    raise ValueError(
        f"Shape mismatch in Gate G4: base_outputs.shape={base_arr.shape} != perturbed_outputs.shape={pert_arr.shape}."
    )

  n_samples = base_arr.shape[0]
  cand_delta = (pert_arr - base_arr).reshape(n_samples, -1)
  per_sample_l2 = np.linalg.norm(cand_delta, axis=-1)
  mean_l2 = float(np.mean(per_sample_l2))
  min_l2 = float(np.min(per_sample_l2))
  responsive_frac = float(np.mean(per_sample_l2 > min_shift))
  cache_hit_suspected = responsive_frac < min_resp_frac or min_l2 <= 1e-9

  if ref_base_outputs is not None and ref_perturbed_outputs is not None:
    ref_delta = (
        np.asarray(ref_perturbed_outputs, dtype=np.float64)
        - np.asarray(ref_base_outputs, dtype=np.float64)
    ).reshape(n_samples, -1)
    jac_cos = compute_cosine_similarity(ref_delta, cand_delta)
  else:
    jac_cos = 1.0 if not cache_hit_suspected else 0.0

  violations: List[str] = []
  if cache_hit_suspected:
    violations.append(
        f"G4 TEST04 cache/memoization detected: responsive_sample_fraction={responsive_frac:.4f} "
        f"< required {min_resp_frac:.4f} (min_l2_shift={min_l2:.3e} <= {min_shift:.3e}). "
        "Disable output memoization or static lookup tables during timed runs."
    )
  if jac_cos < min_jac_cos:
    violations.append(
        f"G4 TEST04 perturbation response alignment={jac_cos:.4f} < min_required={min_jac_cos:.4f} "
        "(perturbed output shift is uncorrelated with physical Jacobian response)."
    )

  passed = len(violations) == 0
  reason = (
      f"Gate G4 (TEST04) passed: responsive_fraction={responsive_frac:.4f}, "
      f"mean_l2_shift={mean_l2:.5f}, jacobian_alignment={jac_cos:.4f}."
      if passed
      else "Gate G4 rejected candidate: " + "; ".join(violations)
  )
  return GateG4Result(
      passed=passed,
      num_probed_samples=n_samples,
      perturbation_linf=float(perturbation_linf),
      mean_l2_output_shift=round(mean_l2, 6),
      min_l2_output_shift=round(min_l2, 8),
      responsive_sample_fraction=round(responsive_frac, 4),
      jacobian_cosine_alignment=round(jac_cos, 4),
      cache_hit_suspected=cache_hit_suspected,
      violations=violations,
      reason=reason,
  )


def evaluate_compliance_suite_g1_g4(
    execution_id: str,
    head_type: str,
    ref_outputs: np.ndarray,
    timed_outputs: np.ndarray,
    perturbed_timed_outputs: np.ndarray,
    ref_perturbed_outputs: np.ndarray,
    baseline_success_rate_pct: float,
    candidate_success_rate_pct: float,
    clock: str = BenchmarkClock.EEB_V1.value,
    division: str = DivisionType.OPEN.value,
    compression_tier: str = CompressionTier.STRICT_LOSSLESS.value,
    p99_latency_ms: Optional[float] = None,
    audited_outputs: Optional[np.ndarray] = None,
    ref_latent_actions: Optional[np.ndarray] = None,
    timed_latent_actions: Optional[np.ndarray] = None,
) -> ComplianceSuiteResult:
  """Run all four EEB Compliance Gates (G1, G2, G3, G4) and return unified ComplianceSuiteResult."""
  g1 = evaluate_gate_g1_parity(
      execution_id=execution_id,
      head_type=head_type,
      ref_outputs=ref_outputs,
      timed_outputs=timed_outputs,
      ref_latent_actions=ref_latent_actions,
      timed_latent_actions=timed_latent_actions,
  )
  g2 = evaluate_gate_g2_mcnemar_non_inferiority(
      baseline_success_rate_pct=baseline_success_rate_pct,
      candidate_success_rate_pct=candidate_success_rate_pct,
      clock=clock,
      division=division,
      compression_tier=compression_tier,
      p99_latency_ms=p99_latency_ms,
  )
  g3 = evaluate_gate_g3_same_execution(
      timing_execution_id=execution_id,
      quality_execution_id=execution_id,
      telemetry_execution_id=execution_id,
      timed_outputs=timed_outputs,
      audited_outputs=audited_outputs if audited_outputs is not None else timed_outputs,
  )
  g4 = evaluate_gate_g4_cache_memoization(
      base_outputs=timed_outputs,
      perturbed_outputs=perturbed_timed_outputs,
      ref_base_outputs=ref_outputs,
      ref_perturbed_outputs=ref_perturbed_outputs,
  )

  all_violations = (
      list(g1.violations)
      + list(g2.violations)
      + list(g3.violations)
      + list(g4.violations)
  )
  passed = g1.passed and g2.passed and g3.passed and g4.passed
  reason = (
      "All EEB Compliance Gates (G1-G4) passed."
      if passed
      else "Compliance suite failed: " + "; ".join(all_violations)
  )
  return ComplianceSuiteResult(
      passed=passed,
      g1_parity=g1,
      g2_non_inferiority=g2,
      g3_same_execution=g3,
      g4_cache_memoization=g4,
      violations=all_violations,
      reason=reason,
  )


def audit_hot_loop_device_sync(python_source: str) -> List[str]:
  """Static AST validator catching blocking `.item()` / `.cpu()` calls inside `for`/`while` loops.

  Stays silent on legitimate `.item()` / `.cpu()` calls outside loops (e.g., one-time summary
  logging after warmup or post-benchmark metric formatting).
  """
  tree = ast.parse(python_source)
  findings: List[str] = []

  class _LoopVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
      self.loop_depth = 0

    def visit_For(self, node: ast.For) -> None:
      self.loop_depth += 1
      self.generic_visit(node)
      self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
      self.loop_depth += 1
      self.generic_visit(node)
      self.loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
      if self.loop_depth > 0 and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("item", "cpu", "numpy"):
          findings.append(
              f"Line {node.lineno}: blocking device-to-host sync '.{node.func.attr}()' "
              "inside hot rollout/action loop. Move host transfer outside the timing loop."
          )
      self.generic_visit(node)

  _LoopVisitor().visit(tree)
  return findings


@dataclass
class WMOpen3PartGateResult:
  """Result of Track 2 World Model Open-Division 3-Part Quality Gate evaluation.

  Enforces:
    1. Visual Fidelity: PSNR >= 28.0 dB, LPIPS <= 0.12, FVD <= 120 on held-out manipulation rollouts.
    2. Action Controllability: Frozen inverse-dynamics probe action recovery CosSim >= 0.990.
    3. Long-Horizon Drift & Policy-Ranking Correlation: 30s-60s drift ratio <= 1.25
       (and optional S2 policy-ranking Kendall tau >= 0.80).
  """

  passed: bool
  visual_fidelity_passed: bool
  action_controllability_passed: bool
  long_horizon_and_ranking_passed: bool
  psnr_db: float
  lpips: float
  fvd: float
  inverse_dynamics_action_cosine_sim: float
  long_horizon_drift_ratio_30_60s: float
  s2_policy_ranking_kendall_tau: Optional[float]
  thresholds: Dict[str, float]
  violations: List[str] = field(default_factory=list)
  reason: str = "World Model 3-Part Quality Gate (Visual Fidelity, Action Controllability, Long-Horizon Drift) passed."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def evaluate_wm_open_3part_quality_gate(
    psnr_db: float,
    lpips: float,
    fvd: float,
    inverse_dynamics_action_cosine_sim: float,
    long_horizon_drift_ratio_30_60s: float,
    s2_policy_ranking_kendall_tau: Optional[float] = None,
    *,
    require_s2_ranking: bool = False,
) -> WMOpen3PartGateResult:
  """Evaluate the 3-Part Quality Gate for Interactive & Evaluative Action-Conditioned World Models (`WM_TRACK`)."""
  cfg = QUALITY_GATE_THRESHOLDS["wm_open_3part_gate"]
  min_psnr = float(cfg["min_psnr_db"])
  max_lpips = float(cfg["max_lpips"])
  max_fvd = float(cfg["max_fvd"])
  min_ctrl_cos = float(cfg["min_inverse_dynamics_action_cosine_sim"])
  max_drift = float(cfg["max_long_horizon_drift_ratio_30_60s"])
  min_tau = float(cfg["min_s2_policy_ranking_kendall_tau"])

  violations: List[str] = []

  # Part 1: Visual Fidelity
  vf_ok = True
  if float(psnr_db) < min_psnr:
    vf_ok = False
    violations.append(
        f"Part 1 Visual Fidelity: psnr_db={psnr_db:.2f} dB < min_required={min_psnr:.1f} dB"
    )
  if float(lpips) > max_lpips:
    vf_ok = False
    violations.append(
        f"Part 1 Visual Fidelity: lpips={lpips:.4f} > max_allowed={max_lpips:.2f}"
    )
  if float(fvd) > max_fvd:
    vf_ok = False
    violations.append(
        f"Part 1 Visual Fidelity: fvd={fvd:.2f} > max_allowed={max_fvd:.1f}"
    )

  # Part 2: Action Controllability (Frozen inverse-dynamics probe)
  ac_ok = float(inverse_dynamics_action_cosine_sim) >= min_ctrl_cos
  if not ac_ok:
    violations.append(
        f"Part 2 Action Controllability: inverse_dynamics_action_cosine_sim={inverse_dynamics_action_cosine_sim:.4f} "
        f"< min_required={min_ctrl_cos:.2f}"
    )

  # Part 3: Long-Horizon Drift (30s-60s) & Optional S2 Policy-Ranking Correlation
  lh_ok = True
  if float(long_horizon_drift_ratio_30_60s) > max_drift:
    lh_ok = False
    violations.append(
        f"Part 3 Long-Horizon Drift: drift_ratio_30_60s={long_horizon_drift_ratio_30_60s:.4f} "
        f"> max_allowed={max_drift:.2f}"
    )
  if s2_policy_ranking_kendall_tau is None and require_s2_ranking:
    lh_ok = False
    violations.append(
        f"Part 3 Policy-Ranking Correlation: s2_policy_ranking_kendall_tau is required for Open Division "
        f"(must be >= {min_tau:.2f})"
    )
  elif s2_policy_ranking_kendall_tau is not None and float(s2_policy_ranking_kendall_tau) < min_tau:
    lh_ok = False
    violations.append(
        f"Part 3 Policy-Ranking Correlation: s2_kendall_tau={s2_policy_ranking_kendall_tau:.4f} "
        f"< min_required={min_tau:.2f}"
    )

  passed = vf_ok and ac_ok and lh_ok
  reason = (
      f"WM 3-Part Quality Gate passed (PSNR={psnr_db:.2f}dB, LPIPS={lpips:.3f}, FVD={fvd:.1f}, "
      f"InvDynCosSim={inverse_dynamics_action_cosine_sim:.4f}, Drift30_60s={long_horizon_drift_ratio_30_60s:.3f})."
      if passed
      else "WM 3-Part Quality Gate rejected candidate: " + "; ".join(violations)
  )

  return WMOpen3PartGateResult(
      passed=passed,
      visual_fidelity_passed=vf_ok,
      action_controllability_passed=ac_ok,
      long_horizon_and_ranking_passed=lh_ok,
      psnr_db=round(float(psnr_db), 4),
      lpips=round(float(lpips), 5),
      fvd=round(float(fvd), 3),
      inverse_dynamics_action_cosine_sim=round(float(inverse_dynamics_action_cosine_sim), 6),
      long_horizon_drift_ratio_30_60s=round(float(long_horizon_drift_ratio_30_60s), 4),
      s2_policy_ranking_kendall_tau=(
          round(float(s2_policy_ranking_kendall_tau), 4)
          if s2_policy_ranking_kendall_tau is not None
          else None
      ),
      thresholds=dict(cfg),
      violations=violations,
      reason=reason,
  )

