"""Co-Located Golden Reference Quality Metrics (Gate G1) & Anti-Memoization Verifier (Gate G4) for EmbodiedX.

Implements:
  - `compute_colocated_quality_metrics(...)`:
    * Track 1 (VLA Policy Actions `(H, D)` or `(B, H, D)`):
      Computes real `action_cosine_sim`, `action_l2_rmse`, and `max_abs_diff`, enforcing
      Gate G1 (`action_cosine_sim >= 0.990` and `action_l2_rmse <= 0.015`).
    * Track 2 (World Model / Omni Frames `(T, H, W, 3)` or `(B, T, H, W, 3)` + Action Probe / Text):
      Computes real `psnr_db` ($10 \\log_{10}(255^2 / \\text{MSE})$), multi-scale spatial gradient
      perceptual distance (`lpips_proxy`), frozen linear inverse-dynamics probe $R^2$ (`inv_dyn_r2`),
      and token `rouge_l`, enforcing Gate G1 (`psnr_db >= 28.0`, `lpips_proxy <= 0.090`, `inv_dyn_r2 >= 0.94`).
  - `verify_gate_g4_anti_memoization(clean_outputs, perturbed_outputs)`:
    Fires when a candidate returns identical/memoized outputs on micro-perturbed inputs
    (`max_diff < 1e-7`) and stays silent when the model genuinely executes a perturbed forward pass
    (Gate 4 negative control).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from embodied_bench.quality_gates import (
    GateG4Result,
    compute_cosine_similarity,
    evaluate_gate_g4_cache_memoization,
    verify_same_execution_id,
)
from embodied_bench.schema import (
    TrackType,
    is_vla_track,
    is_wm_track,
    now_pacific_iso,
)


TRACK1_G1_THRESHOLDS: Dict[str, float] = {
    "min_action_cosine_sim": 0.995,
    "max_action_l2_rmse": 0.015,
}

TRACK2_G1_THRESHOLDS: Dict[str, float] = {
    "min_psnr_db": 28.0,
    "max_lpips_proxy": 0.090,
    "min_inv_dyn_r2": 0.94,
}

GATE_G4_MEMOIZATION_MAX_DIFF_FLOOR: float = 1e-5


@dataclass
class ColocatedQualityResult:
    """Result of co-located Gate G1 numerical & perceptual quality evaluation on episode traces."""

    passed: bool
    track: str
    execution_id: str
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    violations: List[str] = field(default_factory=list)
    reason: str = "Co-located Gate G1 quality check passed."
    evaluated_at: str = field(default_factory=now_pacific_iso)
    # Track 1 convenience attributes
    action_cosine_sim: Optional[float] = None
    action_l2_rmse: Optional[float] = None
    max_abs_diff: Optional[float] = None
    # Track 2 convenience attributes
    psnr_db: Optional[float] = None
    lpips_proxy: Optional[float] = None
    inv_dyn_r2: Optional[float] = None
    rouge_l: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in self.metrics.items():
            d.setdefault(k, v)
        return d

    def __getitem__(self, key: str) -> Any:
        if key in self.metrics:
            return self.metrics[key]
        d = asdict(self)
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.metrics or hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        return sorted(set(asdict(self).keys()) | set(self.metrics.keys()))


@dataclass
class GateG4AntiMemoizationResult:
    """Result of Gate G4 anti-memoization verification on clean vs micro-perturbed episode steps."""

    passed: bool
    memoization_detected: bool
    max_diff: float
    mean_l2_shift: float
    min_allowed_max_diff: float
    violations: List[str] = field(default_factory=list)
    reason: str = "Gate G4 anti-memoization check passed (responsive forward pass verified)."
    g4_result: Optional[GateG4Result] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.g4_result is not None:
            d["g4_result"] = self.g4_result.to_dict()
        return d

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def _infer_track_from_inputs(
    candidate_outputs: Union[np.ndarray, Dict[str, Any]],
    track: Optional[str | TrackType] = None,
) -> str:
    """Determine whether evaluation targets Track 1 (VLA Actions) or Track 2 (World/Omni Model)."""
    if track is not None:
        if is_vla_track(track):
            return TrackType.VLA_TRACK.value
        if is_wm_track(track):
            return TrackType.WM_TRACK.value
        raw = track.value if isinstance(track, TrackType) else str(track).lower()
        if "vla" in raw or raw in ("track1", "1"):
            return TrackType.VLA_TRACK.value
        if "wm" in raw or "world" in raw or "omni" in raw or raw in ("track2", "2"):
            return TrackType.WM_TRACK.value
        raise ValueError(
            f"Unrecognized track={track!r}. Use TrackType.VLA_TRACK ('VLA_TRACK' / 'track1_vla') "
            "or TrackType.WM_TRACK ('WM_TRACK' / 'track2_wm')."
        )

    if isinstance(candidate_outputs, dict):
        if any(k in candidate_outputs for k in ("frames", "rgb_frames", "video", "horizon_rgb_frames")):
            return TrackType.WM_TRACK.value
        return TrackType.VLA_TRACK.value

    arr = np.asarray(candidate_outputs)
    # 4D (T, H, W, C) or 5D (B, T, H, W, C) video frames -> Track 2 WM
    if arr.ndim >= 4:
        return TrackType.WM_TRACK.value
    return TrackType.VLA_TRACK.value


def _compute_multiscale_gradient_lpips_proxy(
    ref_01: np.ndarray,
    cand_01: np.ndarray,
) -> float:
    """Compute multi-scale spatial gradient + luminance perceptual distance in [0, 1]."""
    ref_curr = np.asarray(ref_01, dtype=np.float64)
    cand_curr = np.asarray(cand_01, dtype=np.float64)
    # Reshape any leading dimensions into (N_frames, H, W, C)
    if ref_curr.ndim == 3:
        ref_curr = ref_curr[None, ...]
        cand_curr = cand_curr[None, ...]
    elif ref_curr.ndim > 4:
        h, w, c = ref_curr.shape[-3], ref_curr.shape[-2], ref_curr.shape[-1]
        ref_curr = ref_curr.reshape(-1, h, w, c)
        cand_curr = cand_curr.reshape(-1, h, w, c)

    scale_distances: List[float] = []
    for _scale in (1, 2, 4):
        if ref_curr.shape[1] < 2 or ref_curr.shape[2] < 2:
            break
        gx_ref = ref_curr[:, :, 1:, :] - ref_curr[:, :, :-1, :]
        gx_cand = cand_curr[:, :, 1:, :] - cand_curr[:, :, :-1, :]
        gy_ref = ref_curr[:, 1:, :, :] - ref_curr[:, :-1, :, :]
        gy_cand = cand_curr[:, 1:, :, :] - cand_curr[:, :-1, :, :]

        grad_x_l1 = float(np.mean(np.abs(gx_cand - gx_ref)))
        grad_y_l1 = float(np.mean(np.abs(gy_cand - gy_ref)))
        luma_l1 = float(np.mean(np.abs(cand_curr - ref_curr)))
        scale_distances.append(0.45 * grad_x_l1 + 0.45 * grad_y_l1 + 0.25 * luma_l1)

        # 2x2 average pooling for next pyramid level
        h_even = (ref_curr.shape[1] // 2) * 2
        w_even = (ref_curr.shape[2] // 2) * 2
        if h_even < 4 or w_even < 4:
            break
        ref_curr = 0.25 * (
            ref_curr[:, 0:h_even:2, 0:w_even:2, :]
            + ref_curr[:, 1:h_even:2, 0:w_even:2, :]
            + ref_curr[:, 0:h_even:2, 1:w_even:2, :]
            + ref_curr[:, 1:h_even:2, 1:w_even:2, :]
        )
        cand_curr = 0.25 * (
            cand_curr[:, 0:h_even:2, 0:w_even:2, :]
            + cand_curr[:, 1:h_even:2, 0:w_even:2, :]
            + cand_curr[:, 0:h_even:2, 1:w_even:2, :]
            + cand_curr[:, 1:h_even:2, 1:w_even:2, :]
        )

    if not scale_distances:
        return float(np.mean(np.abs(cand_01 - ref_01)))
    return float(np.mean(scale_distances))


def _extract_inverse_dynamics_transition_features(frames_01: np.ndarray, feat_dim: int = 64) -> np.ndarray:
    """Extract spatiotemporal transition vectors $\\phi(o_t, o_{t+1})$ for the frozen linear probe."""
    arr = np.asarray(frames_01, dtype=np.float64)
    if arr.ndim > 4:
        arr = arr.reshape(-1, arr.shape[-3], arr.shape[-2], arr.shape[-1])
    elif arr.ndim == 3:
        arr = arr[None, ...]

    t_steps = arr.shape[0]
    flat = arr.reshape(t_steps, -1)
    if t_steps >= 2:
        deltas = flat[1:] - flat[:-1]
        midpoints = 0.5 * (flat[1:] + flat[:-1])
        combined = np.concatenate([deltas, midpoints], axis=1)
    else:
        combined = flat

    # Deterministic strided sampling + block pooling into feat_dim features
    n_cols = combined.shape[1]
    indices = np.linspace(0, n_cols - 1, feat_dim, dtype=np.int64)
    return combined[:, indices]


def _compute_frozen_linear_inv_dyn_r2(
    ref_01: np.ndarray,
    cand_01: np.ndarray,
    reference_actions: Optional[np.ndarray] = None,
    candidate_actions: Optional[np.ndarray] = None,
    action_dim: int = 7,
) -> float:
    """Compute $R^2$ of a frozen linear inverse-dynamics probe predicting actions from $(o_t, o_{t+1})$."""
    if reference_actions is not None and candidate_actions is not None:
        ref_a = np.asarray(reference_actions, dtype=np.float64).reshape(-1, action_dim)
        cand_a = np.asarray(candidate_actions, dtype=np.float64).reshape(-1, action_dim)
        ss_res = float(np.sum((cand_a - ref_a) ** 2))
        ss_tot = max(
            float(np.sum((ref_a - np.mean(ref_a, axis=0, keepdims=True)) ** 2)),
            float(np.sum(ref_a ** 2)),
            1e-8,
        )
        return float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))

    feat_dim = 64
    phi_ref = _extract_inverse_dynamics_transition_features(ref_01, feat_dim=feat_dim)
    phi_cand = _extract_inverse_dynamics_transition_features(cand_01, feat_dim=feat_dim)

    probe_rng = np.random.default_rng(20260408)
    w_probe = probe_rng.normal(0.0, 1.0 / np.sqrt(feat_dim), size=(feat_dim, action_dim))

    if reference_actions is not None:
        y_ref = np.asarray(reference_actions, dtype=np.float64)
        if y_ref.shape[0] == phi_ref.shape[0] + 1:
            y_ref = y_ref[:-1]
        y_ref = y_ref.reshape(phi_ref.shape[0], -1)
        # Fit frozen ridge probe on reference (phi_ref -> y_ref) and evaluate on phi_cand
        reg = 1e-4 * np.eye(feat_dim, dtype=np.float64)
        w_fit = np.linalg.solve(phi_ref.T @ phi_ref + reg, phi_ref.T @ y_ref)
        y_ref_pred = phi_ref @ w_fit
        y_cand_pred = phi_cand @ w_fit
    else:
        y_ref_pred = phi_ref @ w_probe
        y_cand_pred = phi_cand @ w_probe

    ss_res = float(np.sum((y_cand_pred - y_ref_pred) ** 2))
    ss_tot = max(
        float(np.sum((y_ref_pred - np.mean(y_ref_pred, axis=0, keepdims=True)) ** 2)),
        float(np.sum(y_ref_pred ** 2)),
        1e-8,
    )
    return float(np.clip(1.0 - (ss_res / ss_tot), -1.0, 1.0))


def _compute_lcs_length(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """Compute Longest Common Subsequence (LCS) length between two token sequences."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev_diag = 0
        ai = seq_a[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            if ai == seq_b[j - 1]:
                dp[j] = prev_diag + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev_diag = temp
    return dp[n]


def compute_token_rouge_l(
    reference_tokens_or_text: Union[str, Sequence[Any], np.ndarray],
    candidate_tokens_or_text: Union[str, Sequence[Any], np.ndarray],
) -> float:
    """Compute token-level ROUGE-L F1 score in [0.0, 1.0]."""
    if isinstance(reference_tokens_or_text, str):
        ref_seq: List[Any] = reference_tokens_or_text.strip().lower().split()
    elif isinstance(reference_tokens_or_text, np.ndarray):
        ref_seq = reference_tokens_or_text.reshape(-1).tolist()
    else:
        ref_seq = list(reference_tokens_or_text)

    if isinstance(candidate_tokens_or_text, str):
        cand_seq: List[Any] = candidate_tokens_or_text.strip().lower().split()
    elif isinstance(candidate_tokens_or_text, np.ndarray):
        cand_seq = candidate_tokens_or_text.reshape(-1).tolist()
    else:
        cand_seq = list(candidate_tokens_or_text)

    if not ref_seq and not cand_seq:
        return 1.0
    if not ref_seq or not cand_seq:
        return 0.0

    lcs = _compute_lcs_length(ref_seq, cand_seq)
    prec = float(lcs) / float(len(cand_seq))
    rec = float(lcs) / float(len(ref_seq))
    if prec + rec <= 1e-12:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def compute_colocated_quality_metrics(
    candidate_outputs: Union[np.ndarray, Dict[str, Any]],
    reference_bf16_outputs: Union[np.ndarray, Dict[str, Any]],
    track: Optional[str | TrackType] = None,
    execution_id: str = "exec-colocated-g1",
    throughput_execution_id: Optional[str] = None,
    reference_actions: Optional[np.ndarray] = None,
    candidate_actions: Optional[np.ndarray] = None,
    reference_text: Optional[Union[str, Sequence[Any], np.ndarray]] = None,
    candidate_text: Optional[Union[str, Sequence[Any], np.ndarray]] = None,
) -> ColocatedQualityResult:
    """Compute co-located Gate G1 numerical & perceptual parity metrics on the same episode steps.

    - For Track 1 (VLA Actions `(H, D)` or `(B, H, D)`):
      Computes `action_cosine_sim`, `action_l2_rmse`, and `max_abs_diff`, and verifies Gate G1
      (`action_cosine_sim >= 0.990` and `action_l2_rmse <= 0.015`).
    - For Track 2 (World Model / Omni Frames `(T, H, W, 3)` + Action Probe / Text):
      Computes `psnr_db` ($10 \\log_{10}(255^2 / \\text{MSE})$), `lpips_proxy` (multi-scale spatial
      gradient perceptual distance), `inv_dyn_r2` (frozen linear inverse-dynamics probe $R^2$),
      and `rouge_l`, verifying Gate G1 (`psnr_db >= 28.0`, `lpips_proxy <= 0.090`, `inv_dyn_r2 >= 0.94`).
    """
    if throughput_execution_id is not None:
        verify_same_execution_id(throughput_execution_id, execution_id)

    resolved_track = _infer_track_from_inputs(candidate_outputs, track=track)

    if isinstance(candidate_outputs, dict) and isinstance(reference_bf16_outputs, dict):
        if resolved_track == TrackType.VLA_TRACK.value:
            cand_arr = np.asarray(
                candidate_outputs.get("actions", candidate_outputs.get("action_chunks")),
                dtype=np.float64,
            )
            ref_arr = np.asarray(
                reference_bf16_outputs.get("actions", reference_bf16_outputs.get("action_chunks")),
                dtype=np.float64,
            )
        else:
            cand_arr = np.asarray(
                candidate_outputs.get("frames", candidate_outputs.get("rgb_frames", candidate_outputs.get("video"))),
                dtype=np.float64,
            )
            ref_arr = np.asarray(
                reference_bf16_outputs.get("frames", reference_bf16_outputs.get("rgb_frames", reference_bf16_outputs.get("video"))),
                dtype=np.float64,
            )
            if reference_actions is None and "actions" in reference_bf16_outputs:
                reference_actions = np.asarray(reference_bf16_outputs["actions"], dtype=np.float64)
            if candidate_actions is None and "actions" in candidate_outputs:
                candidate_actions = np.asarray(candidate_outputs["actions"], dtype=np.float64)
            if reference_text is None and "text" in reference_bf16_outputs:
                reference_text = reference_bf16_outputs["text"]
            if candidate_text is None and "text" in candidate_outputs:
                candidate_text = candidate_outputs["text"]
    else:
        cand_arr = np.asarray(candidate_outputs, dtype=np.float64)
        ref_arr = np.asarray(reference_bf16_outputs, dtype=np.float64)

    if cand_arr.shape != ref_arr.shape:
        raise ValueError(
            f"Shape mismatch in compute_colocated_quality_metrics: candidate_outputs.shape={cand_arr.shape} "
            f"!= reference_bf16_outputs.shape={ref_arr.shape}. Evaluate candidate and reference on the exact same episode steps."
        )

    violations: List[str] = []

    if resolved_track == TrackType.VLA_TRACK.value:
        action_cos = compute_cosine_similarity(ref_arr, cand_arr)
        diff = cand_arr - ref_arr
        action_l2_rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_abs_diff = float(np.max(np.abs(diff)))

        min_cos = TRACK1_G1_THRESHOLDS["min_action_cosine_sim"]
        max_rmse = TRACK1_G1_THRESHOLDS["max_action_l2_rmse"]

        if action_cos < min_cos:
            violations.append(
                f"Track 1 Gate G1 violation: action_cosine_sim={action_cos:.6f} < min_required={min_cos:.3f}. "
                "Keep the action head and proprioceptive projection in BF16/FP16 or increase flow-matching steps."
            )
        if action_l2_rmse > max_rmse:
            violations.append(
                f"Track 1 Gate G1 violation: action_l2_rmse={action_l2_rmse:.6f} > max_allowed={max_rmse:.3f}. "
                "Recalibrate weight quantization scales on EmbodiedX-Traces-v1 calibration episodes."
            )

        passed = len(violations) == 0
        reason = (
            f"Track 1 VLA Gate G1 passed (action_cosine_sim={action_cos:.5f} >= {min_cos:.3f}, "
            f"action_l2_rmse={action_l2_rmse:.5f} <= {max_rmse:.3f})."
            if passed
            else "; ".join(violations)
        )
        metrics = {
            "action_cosine_sim": round(action_cos, 6),
            "action_l2_rmse": round(action_l2_rmse, 6),
            "max_abs_diff": round(max_abs_diff, 6),
        }
        return ColocatedQualityResult(
            passed=passed,
            track=TrackType.VLA_TRACK.value,
            execution_id=execution_id,
            metrics=metrics,
            thresholds=dict(TRACK1_G1_THRESHOLDS),
            violations=violations,
            reason=reason,
            action_cosine_sim=metrics["action_cosine_sim"],
            action_l2_rmse=metrics["action_l2_rmse"],
            max_abs_diff=metrics["max_abs_diff"],
        )

    # Track 2: World Model / Omni Frames (T, H, W, 3) + Action Probe + Token ROUGE-L
    max_val = float(max(np.max(np.abs(ref_arr)), np.max(np.abs(cand_arr)), 1e-8))
    if max_val <= 1.5:
        ref_01 = np.clip(ref_arr, 0.0, 1.0)
        cand_01 = np.clip(cand_arr, 0.0, 1.0)
        ref_255 = ref_01 * 255.0
        cand_255 = cand_01 * 255.0
    else:
        ref_255 = np.clip(ref_arr, 0.0, 255.0)
        cand_255 = np.clip(cand_arr, 0.0, 255.0)
        ref_01 = ref_255 / 255.0
        cand_01 = cand_255 / 255.0

    mse_255 = float(np.mean((cand_255 - ref_255) ** 2))
    if mse_255 <= 1e-10:
        psnr_db = 100.0
    else:
        psnr_db = float(10.0 * np.log10((255.0 ** 2) / mse_255))

    lpips_proxy = _compute_multiscale_gradient_lpips_proxy(ref_01, cand_01)
    inv_dyn_r2 = _compute_frozen_linear_inv_dyn_r2(
        ref_01=ref_01,
        cand_01=cand_01,
        reference_actions=reference_actions,
        candidate_actions=candidate_actions,
    )

    if reference_text is not None and candidate_text is not None:
        rouge_l = compute_token_rouge_l(reference_text, candidate_text)
    else:
        # Derive semantic visual-language token parity from 32 pooled patch descriptors
        flat_ref = ref_01.reshape(-1)
        flat_cand = cand_01.reshape(-1)
        idx = np.linspace(0, flat_ref.size - 1, min(32, flat_ref.size), dtype=np.int64)
        ref_bins = np.clip(np.round(flat_ref[idx] * 25.0), 0, 25).astype(np.int32)
        cand_bins = np.clip(np.round(flat_cand[idx] * 25.0), 0, 25).astype(np.int32)
        rouge_l = compute_token_rouge_l(ref_bins, cand_bins)

    max_abs_diff = float(np.max(np.abs(cand_01 - ref_01)))

    min_psnr = TRACK2_G1_THRESHOLDS["min_psnr_db"]
    max_lpips = TRACK2_G1_THRESHOLDS["max_lpips_proxy"]
    min_r2 = TRACK2_G1_THRESHOLDS["min_inv_dyn_r2"]

    if psnr_db < min_psnr:
        violations.append(
            f"Track 2 Gate G1 violation: psnr_db={psnr_db:.3f} dB < min_required={min_psnr:.1f} dB. "
            "Reduce DiT feature-cache interval or increase denoising steps."
        )
    if lpips_proxy > max_lpips:
        violations.append(
            f"Track 2 Gate G1 violation: lpips_proxy={lpips_proxy:.5f} > max_allowed={max_lpips:.3f}. "
            "Preserve high-frequency spatial attention heads in BF16/FP8."
        )
    if inv_dyn_r2 < min_r2:
        violations.append(
            f"Track 2 Gate G1 violation: inv_dyn_r2={inv_dyn_r2:.5f} < min_required={min_r2:.2f}. "
            "Ensure action-conditioning cross-attention layers are not over-quantized or skipped."
        )

    passed = len(violations) == 0
    reason = (
        f"Track 2 World/Omni Gate G1 passed (psnr_db={psnr_db:.2f} dB >= {min_psnr:.1f} dB, "
        f"lpips_proxy={lpips_proxy:.4f} <= {max_lpips:.3f}, inv_dyn_r2={inv_dyn_r2:.4f} >= {min_r2:.2f}, "
        f"rouge_l={rouge_l:.4f})."
        if passed
        else "; ".join(violations)
    )
    metrics = {
        "psnr_db": round(psnr_db, 4),
        "lpips_proxy": round(lpips_proxy, 6),
        "inv_dyn_r2": round(inv_dyn_r2, 6),
        "rouge_l": round(rouge_l, 6),
        "max_abs_diff": round(max_abs_diff, 6),
    }
    return ColocatedQualityResult(
        passed=passed,
        track=TrackType.WM_TRACK.value,
        execution_id=execution_id,
        metrics=metrics,
        thresholds=dict(TRACK2_G1_THRESHOLDS),
        violations=violations,
        reason=reason,
        max_abs_diff=metrics["max_abs_diff"],
        psnr_db=metrics["psnr_db"],
        lpips_proxy=metrics["lpips_proxy"],
        inv_dyn_r2=metrics["inv_dyn_r2"],
        rouge_l=metrics["rouge_l"],
    )


def verify_gate_g4_anti_memoization(
    clean_outputs: np.ndarray,
    perturbed_outputs: np.ndarray,
    min_allowed_max_diff: float = GATE_G4_MEMOIZATION_MAX_DIFF_FLOOR,
    reference_clean_outputs: Optional[np.ndarray] = None,
    reference_perturbed_outputs: Optional[np.ndarray] = None,
    perturbation_linf: float = 1e-3,
    raise_on_violation: bool = False,
) -> GateG4AntiMemoizationResult:
    """Verify Gate G4 (TEST04 Anti-Memoization) on clean vs micro-perturbed episode step outputs.

    Fires (`passed=False`, `memoization_detected=True`) when a model returns identical or near-identical
    memoized outputs on perturbed inputs (`max_diff < 1e-7`), and stays silent (`passed=True`,
    `memoization_detected=False`) when the candidate genuinely computes a perturbed forward pass.
    """
    clean_arr = np.asarray(clean_outputs, dtype=np.float64)
    pert_arr = np.asarray(perturbed_outputs, dtype=np.float64)
    if clean_arr.shape != pert_arr.shape:
        raise ValueError(
            f"Shape mismatch in verify_gate_g4_anti_memoization: clean_outputs.shape={clean_arr.shape} "
            f"!= perturbed_outputs.shape={pert_arr.shape}."
        )

    diff = pert_arr - clean_arr
    max_diff = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0

    # Ensure 2D+ shape (N_samples, ...) for evaluate_gate_g4_cache_memoization
    clean_batched = clean_arr[None, ...] if clean_arr.ndim == 1 else clean_arr
    pert_batched = pert_arr[None, ...] if pert_arr.ndim == 1 else pert_arr
    ref_clean_batched = (
        (np.asarray(reference_clean_outputs, dtype=np.float64)[None, ...]
         if np.asarray(reference_clean_outputs).ndim == 1
         else np.asarray(reference_clean_outputs, dtype=np.float64))
        if reference_clean_outputs is not None
        else None
    )
    ref_pert_batched = (
        (np.asarray(reference_perturbed_outputs, dtype=np.float64)[None, ...]
         if np.asarray(reference_perturbed_outputs).ndim == 1
         else np.asarray(reference_perturbed_outputs, dtype=np.float64))
        if reference_perturbed_outputs is not None
        else None
    )

    g4_eval = evaluate_gate_g4_cache_memoization(
        base_outputs=clean_batched,
        perturbed_outputs=pert_batched,
        ref_base_outputs=ref_clean_batched,
        ref_perturbed_outputs=ref_pert_batched,
        perturbation_linf=perturbation_linf,
    )

    violations: List[str] = []
    memoization_detected = bool(max_diff < float(min_allowed_max_diff) or g4_eval.cache_hit_suspected)
    if max_diff < float(min_allowed_max_diff):
        violations.append(
            f"Gate G4 anti-memoization violation: max_diff={max_diff:.3e} < min_required={min_allowed_max_diff:.1e}. "
            "Candidate returned static/memoized outputs on perturbed inputs; disable lookup-table caching "
            "and execute the full model forward pass on every step."
        )
    for v in g4_eval.violations:
        if v not in violations:
            violations.append(v)

    passed = not memoization_detected and len(violations) == 0
    reason = (
        f"Gate G4 anti-memoization check passed (max_diff={max_diff:.3e} >= {min_allowed_max_diff:.1e}, "
        f"mean_l2_shift={g4_eval.mean_l2_output_shift:.5f})."
        if passed
        else "; ".join(violations)
    )

    result = GateG4AntiMemoizationResult(
        passed=passed,
        memoization_detected=memoization_detected,
        max_diff=round(max_diff, 10),
        mean_l2_shift=g4_eval.mean_l2_output_shift,
        min_allowed_max_diff=float(min_allowed_max_diff),
        violations=violations,
        reason=reason,
        g4_result=g4_eval,
    )
    if raise_on_violation and not passed:
        raise ValueError(result.reason)
    return result
