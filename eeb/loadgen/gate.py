"""Co-located gate for `eeb measure` reports.

Two modes, always executed in the same process / against the same server as the timed sweep
and tagged with the operating point's `execution_id` (G3 same-execution rule):

* `SANITY_ONLY` (default): structural checks on the exact tensors returned during the timed
  window - every chunk had shape (caps.horizon, caps.action_dim), all values finite, no transport
  errors, session-sticky routing held - plus a SHA-256 over the returned actions. This is NOT a
  quality certification (`quality_certified=False`); it only proves the run produced well-formed
  outputs and gives WS-D a digest to compare against golden outputs later.
* `GOLDEN_REPLAY`: additionally replays a golden NPZ (`ref_actions`, `proprio`, `instructions`,
  `image__<camera>`) through the same endpoint and evaluates
  `embodied_bench.quality_gates.evaluate_vla_gate_a` (CosSim / L1 / MSE thresholds).

Golden NPZ layout (all arrays share the leading N)::

    ref_actions : float32 [N, H, D]
    proprio     : float32 [N, P]
    instructions: str     [N]
    image__<cam>: uint8   [N, h, w, c]   (one array per camera)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional
import uuid

import numpy as np

from eeb.loadgen.client import PolicyClient
from eeb.loadgen.sweep import ConcurrencyResult
from eeb.protocol.policy import Caps, Context, Observation

GATE_MODE_SANITY = "SANITY_ONLY"
GATE_MODE_GOLDEN = "GOLDEN_REPLAY"
MIN_ADMISSIBLE_WIRE_IMAGE_BYTES = 224 * 224 * 3  # 150,528 B (1x 224x224x3 uint8 RGB frame)
MIN_ADMISSIBLE_WIRE_RESOLUTION_HW = 224
BF16_SAFE_PERTURB_STEP = 1e-2  # > 1 BF16 ULP at unit scale (2^-7 = 0.0078125)


def evaluate_wire_observation_resolution(
    observation_source: Optional[Dict[str, Any]],
    *,
    is_wm: bool = False,
) -> tuple[bool, int, tuple[int, int], str]:
  """Verify that the LoadGen request wire payload sends native-resolution RGB frames (>= 224x224).

  Catches accidental hardware benchmark runs fed the lightweight 64x64 CI thumbnail bundle
  (`data/embodiedx_traces_v1_mini`, 24,576 B/req VLA or 12,288 B/req WM), which understates
  gRPC/protobuf wire serialization by >12x and forces host CPU image upsampling inside `predict()`.
  """
  obs = dict(observation_source or {})
  raw_hw = obs.get("image_hw")
  if isinstance(raw_hw, (list, tuple)) and len(raw_hw) >= 2:
    h, w = int(raw_hw[0] or 0), int(raw_hw[1] or 0)
  else:
    h, w = 0, 0
  channels = int(obs.get("channels", 3) or 3)
  cams = obs.get("cameras") or []
  num_frames = 1 if is_wm else max(len(cams), 1)
  computed_bytes = int(num_frames * h * w * channels)
  recorded_bytes = obs.get("bytes_per_request_images")
  if not is_wm and isinstance(recorded_bytes, (int, float)) and not isinstance(recorded_bytes, bool):
    wire_bytes = int(recorded_bytes)
  else:
    wire_bytes = computed_bytes

  if wire_bytes < MIN_ADMISSIBLE_WIRE_IMAGE_BYTES or min(h, w) < MIN_ADMISSIBLE_WIRE_RESOLUTION_HW:
    reason = (
        f"observation wire resolution below protocol minimum: image_hw={h}x{w}, "
        f"bytes_per_request_images={wire_bytes} B (< {MIN_ADMISSIBLE_WIRE_IMAGE_BYTES} B = 224x224x3 RGB; "
        "64x64 CI thumbnail traces are strictly for unit tests -- materialize native 224x224 or 256x256 "
        "eval traces via 'eeb data prepare-traces --out data/embodiedx_eval_224 --image-hw 224x224')"
    )
    return False, wire_bytes, (h, w), reason
  return True, wire_bytes, (h, w), f"admissible wire resolution ({h}x{w}, {wire_bytes} B/req)"


def compute_normalized_action_jerk(actions: np.ndarray) -> float:
  """Compute dimensionless RMS discrete action jerk along the horizon axis (-2).

  For action arrays of shape (H, D) or (N, H, D), computes the discrete 3rd difference
  (or highest difference order <= 3 supported by H >= 2) along axis -2, normalized by
  the RMS action magnitude.
  """
  arr = np.asarray(actions, dtype=np.float64)
  if arr.ndim < 2 or arr.shape[-2] < 2 or arr.size == 0 or not np.isfinite(arr).all():
    return 0.0
  order = min(3, int(arr.shape[-2]) - 1)
  diff = np.diff(arr, n=order, axis=-2)
  jerk_rms = float(np.sqrt(np.mean(diff ** 2)))
  act_rms = float(np.sqrt(np.mean(arr ** 2)))
  return round(float(jerk_rms / max(act_rms, 1e-6)), 6)


@dataclass
class CoLocatedGateResult:
  execution_id: str
  mode: str
  passed: bool
  quality_certified: bool
  checks: Dict[str, Any] = field(default_factory=dict)
  violations: List[str] = field(default_factory=list)
  reason: str = ""
  golden: Optional[Dict[str, Any]] = None
  normalized_action_jerk: Optional[float] = None

  def to_dict(self) -> Dict[str, Any]:
    out = asdict(self)
    out["gate_mode"] = self.mode
    if isinstance(self.golden, dict):
      g_metrics = self.golden.get("metrics") or {}
      for k in (
          "action_cosine_sim",
          "action_cosine_cvar_1pct",
          "psnr_db",
          "frame_psnr_db",
          "frame_cosine_sim",
          "ssim",
          "cvar1_psnr_db",
          "cvar1_ssim",
      ):
        if k in g_metrics:
          out[k] = g_metrics[k]
        elif k in self.golden:
          out[k] = self.golden[k]
      if self.golden.get("golden_sha256"):
        out["golden_sha256"] = str(self.golden["golden_sha256"])
      if "num_samples" in self.golden:
        out["num_samples"] = int(self.golden["num_samples"])
      if "psnr_db" in g_metrics and "frame_psnr_db" not in out:
        out["frame_psnr_db"] = g_metrics["psnr_db"]
    return out


def sanity_gate(point: ConcurrencyResult, caps: Caps, execution_id: str) -> CoLocatedGateResult:
  """Structural checks over the timed-window records of one concurrency point."""
  violations: List[str] = []
  n_calls = sum(r.num_timed_calls for r in point.runs)
  n_shape = sum(r.num_shape_mismatch for r in point.runs)
  n_nonfinite = sum(r.num_nonfinite for r in point.runs)
  n_errors = sum(r.num_errors for r in point.runs)
  sticky = all(r.sticky_routing_ok for r in point.runs)
  if n_calls == 0:
    violations.append("no timed calls completed")
  if n_shape:
    if caps.track == "WM_TRACK":
      violations.append(
          f"{n_shape}/{n_calls} rollouts had frame shape != (num_frames={caps.horizon}, H={caps.image_height}, W={caps.image_width}, 3)"
      )
    else:
      violations.append(f"{n_shape}/{n_calls} chunks had shape != (horizon={caps.horizon}, dim={caps.action_dim})")
  if n_nonfinite:
    violations.append(f"{n_nonfinite}/{n_calls} chunks contained NaN/Inf")
  if n_errors:
    violations.append(f"{n_errors} transport/policy errors during the runs")
  if not sticky:
    violations.append("session-sticky routing violated")
  n_canary_checked = sum(int(getattr(r, "num_inline_canaries", 0) or 0) for r in point.runs)
  n_canary_failed = sum(int(getattr(r, "num_inline_canary_failures", 0) or 0) for r in point.runs)
  if n_canary_failed > 0:
    canary_msgs: List[str] = []
    for r in point.runs:
      for m in getattr(r, "inline_canary_reasons", []) or []:
        if m not in canary_msgs:
          canary_msgs.append(m)
    detail = f" ({'; '.join(canary_msgs[:3])})" if canary_msgs else ""
    violations.append(
        f"{n_canary_failed}/{max(n_canary_checked, n_canary_failed)} inline blind sweep canaries failed{detail}"
    )
  b1_ttff_p99: Optional[float] = None
  if caps.track == "WM_TRACK" and int(point.concurrency) == 1:
    ttff_stats = getattr(point, "median_ttff_ms", None)
    raw_ttff_p99 = ttff_stats.get("p99") if isinstance(ttff_stats, dict) else None
    if isinstance(raw_ttff_p99, (int, float)) and not isinstance(raw_ttff_p99, bool) and np.isfinite(float(raw_ttff_p99)):
      b1_ttff_p99 = float(raw_ttff_p99)
  passed = not violations
  norm_jerk: Optional[float] = None
  if caps.track != "WM_TRACK":
    if getattr(point, "median_normalized_action_jerk", None) is not None:
      norm_jerk = float(point.median_normalized_action_jerk)  # type: ignore[arg-type]
    else:
      run_jerks = [
          float(r.normalized_action_jerk)
          for r in point.runs
          if getattr(r, "normalized_action_jerk", None) is not None
      ]
      norm_jerk = round(float(statistics.median(run_jerks)), 6) if run_jerks else 0.0
  checks: Dict[str, Any] = {
      "num_timed_calls": n_calls,
      "num_shape_mismatch": n_shape,
      "num_nonfinite": n_nonfinite,
      "num_errors": n_errors,
      "sticky_routing_ok": sticky,
      "num_inline_canaries": n_canary_checked,
      "num_inline_canary_failures": n_canary_failed,
      "deterministic_repeat": [r.deterministic_repeat for r in point.runs],
      "actions_sha256_per_run": [r.actions_sha256 for r in point.runs],
  }
  if b1_ttff_p99 is not None:
    checks["b1_ttff_p99_ms"] = round(b1_ttff_p99, 4)
    checks["b1_ttff_p99_ms_diagnostic"] = round(b1_ttff_p99, 4)
  if norm_jerk is not None:
    checks["normalized_action_jerk"] = norm_jerk
  return CoLocatedGateResult(
      execution_id=execution_id,
      mode=GATE_MODE_SANITY,
      passed=passed,
      quality_certified=False,
      checks=checks,
      violations=violations,
      reason=(
          "SANITY_ONLY: well-formed outputs in the same execution; NOT a quality gate (no golden reference supplied)."
          if passed
          else "SANITY_ONLY gate failed: " + "; ".join(violations)
      ),
      normalized_action_jerk=norm_jerk,
  )


def load_golden_npz(path: str | Path) -> Dict[str, Any]:
  import json

  p_obj = Path(path)
  golden_sha256 = hashlib.sha256(p_obj.read_bytes()).hexdigest()
  with np.load(p_obj, allow_pickle=False) as data:
    keys = set(data.keys())
    has_actions = "ref_actions" in keys or "actions" in keys
    has_frames = "ref_frames" in keys or "frames" in keys
    if not has_actions and not has_frames:
      raise ValueError(
          f"Golden file {path} has neither 'ref_actions'/'actions' nor 'ref_frames'/'frames' array (keys: {sorted(keys)})."
      )
    act_key = "ref_actions" if "ref_actions" in keys else ("actions" if "actions" in keys else None)
    frm_key = "ref_frames" if "ref_frames" in keys else ("frames" if "frames" in keys else None)
    ref_frames = np.asarray(data[frm_key], dtype=np.uint8) if frm_key is not None else None
    ref = (
        np.asarray(data[act_key], dtype=np.float32)
        if act_key is not None
        else np.zeros((int(ref_frames.shape[0]), 1, 7), dtype=np.float32)  # type: ignore[union-attr]
    )
    n = int(ref.shape[0] if act_key is not None else ref_frames.shape[0])  # type: ignore[union-attr]
    meta: Dict[str, Any] = {}
    for mkey in ("metadata_json", "meta_json"):
      if mkey in keys:
        try:
          raw_m = data[mkey]
          if raw_m.dtype == np.uint8:
            meta = json.loads(bytes(raw_m.tobytes()).decode("utf-8"))
          else:
            meta = json.loads(str(raw_m.item() if raw_m.ndim == 0 else raw_m[0]))
          break
        except Exception:
          pass
    cams = {k[len("image__"):]: np.asarray(data[k]) for k in keys if k.startswith("image__")}
    if not cams:
      if "context_frames" in keys:
        cf = np.asarray(data["context_frames"], dtype=np.uint8)
        cams["agentview"] = cf[:, 0] if cf.ndim == 5 else cf
      elif "images" in keys:
        imgs = np.asarray(data["images"], dtype=np.uint8)
        if imgs.ndim == 5:
          cam_names = meta.get("cameras") or [f"cam_{i}" for i in range(imgs.shape[1])]
          for idx, cname in enumerate(cam_names):
            if idx < imgs.shape[1]:
              cams[str(cname)] = imgs[:, idx]
        elif imgs.ndim == 4:
          cams["agentview"] = imgs
    prop_key = "proprio" if "proprio" in keys else ("proprios" if "proprios" in keys else None)
    proprio = np.asarray(data[prop_key], dtype=np.float32) if prop_key is not None else np.zeros((n, 0), np.float32)
    instr = [str(x) for x in data["instructions"]] if "instructions" in keys else [""] * n
    for name, arr in cams.items():
      if arr.shape[0] != n:
        raise ValueError(f"Golden image array {name!r} has N={arr.shape[0]} but golden set has N={n}.")
    out: Dict[str, Any] = {
        "ref_actions": ref,
        "actions": ref,
        "cameras": cams,
        "proprio": proprio,
        "proprios": proprio,
        "instructions": instr,
        "n": n,
        "metadata": meta,
        "golden_sha256": golden_sha256,
    }
    if "context_frames" in keys:
      out["context_frames"] = np.asarray(data["context_frames"], dtype=np.uint8)
    if "images" in keys:
      out["images"] = np.asarray(data["images"], dtype=np.uint8)
    if ref_frames is not None:
      out["ref_frames"] = ref_frames
      out["frames"] = ref_frames
    return out


def golden_replay_gate(client: PolicyClient, golden: Dict[str, Any], execution_id: str, session_prefix: str = "golden") -> Dict[str, Any]:
  """Replay golden observations/contexts through `client` and evaluate Gate A on returned actions or frames."""
  from embodied_bench.quality_gates import (
      compute_wm_sample_psnr_and_ssim_stats,
      evaluate_vla_gate_a,
  )

  caps = client.capabilities()
  golden_meta = dict(golden.get("metadata") or {})
  raw_frames_arr = golden.get("ref_frames") if "ref_frames" in golden else golden.get("frames")
  raw_actions_arr = golden.get("ref_actions") if "ref_actions" in golden else golden.get("actions")
  proprios_arr = golden.get("proprio") if "proprio" in golden else golden.get("proprios")
  seed_int = int(hashlib.sha256(f"golden-order:{execution_id}:{caps.track}".encode("utf-8")).hexdigest()[:8], 16)
  rng_order = np.random.default_rng(seed_int)
  order = [int(i) for i in rng_order.permutation(int(golden["n"]))]
  if caps.track == "WM_TRACK" and raw_frames_arr is not None:
    ref_frames = np.asarray(raw_frames_arr, dtype=np.float32)
    cand_frames_by_idx: List[Optional[np.ndarray]] = [None] * int(golden["n"])
    sess_id = uuid.uuid4().hex
    client.reset(sess_id)
    cam_keys = sorted(golden.get("cameras", {}).keys())
    ctx_frames_arr = golden.get("context_frames")
    for i in order:
      if cam_keys:
        cond = [golden["cameras"][cam_keys[0]][i]]
      elif ctx_frames_arr is not None:
        cf_i = ctx_frames_arr[i]
        cond = [cf_i[j] for j in range(cf_i.shape[0])] if cf_i.ndim == 4 else [cf_i]
      else:
        cond = []
      ctx = Context(
          session_id=sess_id,
          frames=cond,
          proprio=proprios_arr[i],  # type: ignore[index]
          instruction=golden["instructions"][i],
          num_frames=int(ref_frames.shape[1]),
      )
      act_in = raw_actions_arr[i]  # type: ignore[index]
      roll = client.rollout(ctx, act_in)
      cand_frames_by_idx[i] = np.stack([f.image for f in roll.frames]).astype(np.float32)
    cand_arr = np.stack([arr for arr in cand_frames_by_idx if arr is not None])
    if cand_arr.shape != ref_frames.shape:
      return {
          "passed": False,
          "execution_id": execution_id,
          "reason": f"candidate frames shape {cand_arr.shape} != golden {ref_frames.shape}",
          "metrics": {},
      }
    mse = float(np.mean((cand_arr - ref_frames) ** 2))
    psnr = 99.0 if mse <= 1e-9 else float(10.0 * np.log10((255.0 ** 2) / mse))
    wm_tail = compute_wm_sample_psnr_and_ssim_stats(ref_frames, cand_arr, data_range=255.0)
    ssim_val = float(wm_tail["ssim"])
    cvar1_psnr = float(wm_tail["cvar1_psnr_db"])
    cvar1_ssim = float(wm_tail["cvar1_ssim"])
    flat_ref = ref_frames.reshape(-1)
    flat_cand = cand_arr.reshape(-1)
    denom = float(np.linalg.norm(flat_ref) * np.linalg.norm(flat_cand))
    cos_sim = float(np.dot(flat_ref, flat_cand) / max(denom, 1e-9))
    wm_viol: List[str] = []
    if psnr < 28.0:
      wm_viol.append(f"psnr_db={psnr:.2f} < 28.0")
    if ssim_val < 0.88:
      wm_viol.append(f"ssim={ssim_val:.4f} < 0.88")
    if cvar1_psnr < 22.0:
      wm_viol.append(f"cvar1_psnr_db={cvar1_psnr:.2f} < 22.0")
    if cvar1_ssim < 0.80:
      wm_viol.append(f"cvar1_ssim={cvar1_ssim:.4f} < 0.80")
    if cos_sim < 0.990:
      wm_viol.append(f"cosine_sim={cos_sim:.5f} < 0.990")
    passed = len(wm_viol) == 0
    reason_msg = (
        f"WM_TRACK golden replay psnr_db={psnr:.2f} ssim={ssim_val:.4f} "
        f"cvar1_psnr_db={cvar1_psnr:.2f} cvar1_ssim={cvar1_ssim:.4f} cosine_sim={cos_sim:.5f}"
    )
    if wm_viol:
      reason_msg = f"{reason_msg} ({'; '.join(wm_viol)})"
    return {
        "passed": passed,
        "execution_id": execution_id,
        "reason": reason_msg,
        "violations": wm_viol,
        "num_samples": int(golden["n"]),
        "allow_smoke_golden": bool(golden_meta.get("allow_smoke_golden", False)),
        "golden_certifiable": bool(
            golden_meta.get("certifiable", int(golden["n"]) >= 100 and not golden_meta.get("allow_smoke_golden", False))
        ),
        "golden_sha256": golden.get("golden_sha256"),
        "psnr_db": round(psnr, 3),
        "frame_psnr_db": round(psnr, 3),
        "ssim": round(ssim_val, 6),
        "cvar1_psnr_db": round(cvar1_psnr, 3),
        "cvar1_ssim": round(cvar1_ssim, 6),
        "action_cosine_sim": round(cos_sim, 6),
        "metrics": {
            "psnr_db": round(psnr, 3),
            "frame_psnr_db": round(psnr, 3),
            "ssim": round(ssim_val, 6),
            "cvar1_psnr_db": round(cvar1_psnr, 3),
            "cvar1_ssim": round(cvar1_ssim, 6),
            "frame_cosine_sim": round(cos_sim, 6),
            "action_cosine_sim": round(cos_sim, 6),
        },
    }

  cand_by_idx: List[Optional[np.ndarray]] = [None] * int(golden["n"])
  sess_id = uuid.uuid4().hex
  client.reset(sess_id)
  for i in order:
    obs = Observation(
        session_id=sess_id,
        step=i,
        images={cam: arr[i] for cam, arr in golden.get("cameras", {}).items()},
        proprio=proprios_arr[i],  # type: ignore[index]
        instruction=golden["instructions"][i],
    )
    cand_by_idx[i] = client.predict(obs).chunk.actions
  cand_arr = np.stack([arr for arr in cand_by_idx if arr is not None]).astype(np.float32)
  ref_arr = np.asarray(raw_actions_arr, dtype=np.float32)
  if cand_arr.shape != ref_arr.shape:
    return {
        "passed": False,
        "execution_id": execution_id,
        "reason": f"candidate actions shape {cand_arr.shape} != golden {ref_arr.shape}",
        "metrics": {},
    }
  res = evaluate_vla_gate_a(execution_id=execution_id, throughput_execution_id=execution_id, ref_actions=ref_arr, cand_actions=cand_arr)
  out = res.to_dict() if hasattr(res, "to_dict") else asdict(res)
  cand_jerk = compute_normalized_action_jerk(cand_arr)
  ref_jerk = compute_normalized_action_jerk(ref_arr)
  out.setdefault("metrics", {})["normalized_action_jerk"] = cand_jerk
  out["metrics"]["ref_normalized_action_jerk"] = ref_jerk
  out["normalized_action_jerk"] = cand_jerk
  out["num_samples"] = int(golden["n"])
  out["allow_smoke_golden"] = bool(golden_meta.get("allow_smoke_golden", False))
  out["golden_certifiable"] = bool(
      golden_meta.get("certifiable", int(golden["n"]) >= 100 and not golden_meta.get("allow_smoke_golden", False))
  )
  if golden.get("golden_sha256"):
    out["golden_sha256"] = str(golden["golden_sha256"])
  return out


def _perturb_bf16_safe_vector(
    arr: np.ndarray,
    rng: np.random.Generator,
    step: float = BF16_SAFE_PERTURB_STEP,
) -> tuple[np.ndarray, float]:
  """Perturb every element of a float vector/matrix by >= 1 BF16 ULP without mutating `arr`.

  At unit magnitude `|x| in [1, 2)`, 1 BF16 ULP is `2^-7 = 0.0078125`, so a `1e-3` delta
  collapses to exact zero when cast to `torch.bfloat16` / `jnp.bfloat16`. This helper shifts
  ALL elements by random `+/- max(step, 1.5 * 2^-7 * |x|)` (`step >= 1e-2`), guaranteeing every
  coordinate changes its BF16 representation across arbitrary dynamic ranges.
  """
  x = np.array(arr, copy=True, dtype=np.float32)
  if x.size == 0:
    return x, 0.0
  min_step = np.float32(max(float(step), BF16_SAFE_PERTURB_STEP))
  bf16_scale_step = (np.abs(x) * np.float32(1.5 * (2.0 ** -7))).astype(np.float32)
  elem_mag = np.maximum(min_step, bf16_scale_step)
  signs = np.where(rng.uniform(0.0, 1.0, size=x.shape) < 0.5, np.float32(-1.0), np.float32(1.0))
  y = (x + signs * elem_mag).astype(np.float32)
  return y, float(np.max(np.abs(y - x)))


def _perturb_wire_image(
    arr: np.ndarray,
    rng: np.random.Generator,
    eps_linf: float,
    *,
    lsb_delta: int = 1,
    lsb_fraction: float = 0.10,
) -> tuple[np.ndarray, float]:
  """Perturb one `uint8` or floating camera frame without clipping or mutating `arr`, guaranteeing >=1 changed element."""
  from eeb.data.perturb import PerturbationConfig, linf_delta, perturb_image

  img = np.asarray(arr)
  if img.size == 0:
    return img.copy(), 0.0
  step_i = max(1, int(lsb_delta))
  frac = min(max(float(lsb_fraction), 1e-6), 1.0)
  if img.dtype == np.uint8:
    if step_i == 1:
      cfg = PerturbationConfig(
          eps_linf=min(float(eps_linf), 1.0 / 255.0),
          image_space="uint8_lsb",
          lsb_fraction=frac,
      )
      y8, img_linf, _ = perturb_image(img, rng, cfg)
      if np.array_equal(y8, img):
        y8 = img.copy()
        flat = y8.reshape(-1)
        flat[0] = np.uint8(int(flat[0]) - 1 if int(flat[0]) > 0 else 1)
        img_linf = float(1.0 / 255.0)
      return y8, float(img_linf)

    xi = img.astype(np.int16, copy=True)
    # Use spatially coherent coarse macro-blocks + per-channel coherent directional signs
    # so perturbations neither cancel inside 14x14/16x16 ViT patch projections (unlike i.i.d.
    # Rademacher noise) nor vanish under zero-DC spatial LayerNorm/edge filters (unlike full-frame DC shifts).
    if xi.ndim == 3:
      h_i, w_i, c_i = xi.shape
      grid = 4
      block_probs = rng.uniform(0.0, 1.0, size=(grid, grid))
      block_mask = block_probs < max(frac, 0.25)
      if not bool(block_mask.any()):
        block_mask.reshape(-1)[int(rng.integers(0, block_mask.size))] = True
      if bool(block_mask.all()):
        block_mask.reshape(-1)[int(rng.integers(0, block_mask.size))] = False
      row_bins = np.minimum((np.arange(h_i) * grid) // max(h_i, 1), grid - 1)
      col_bins = np.minimum((np.arange(w_i) * grid) // max(w_i, 1), grid - 1)
      mask = np.broadcast_to(block_mask[row_bins[:, None], col_bins[None, :], None], xi.shape).copy()
      ch_signs = np.where(rng.uniform(0.0, 1.0, size=(1, 1, c_i)) < 0.5, -step_i, step_i).astype(np.int16)
      signs = np.broadcast_to(ch_signs, xi.shape).copy()
    else:
      mask = rng.uniform(0.0, 1.0, size=xi.shape) < frac
      if not bool(mask.any()):
        mask.reshape(-1)[int(rng.integers(0, mask.size))] = True
      s_scalar = np.int16(-step_i if float(rng.uniform(0.0, 1.0)) < 0.5 else step_i)
      signs = np.full(xi.shape, s_scalar, dtype=np.int16)
    # Boundary-safe sign flip in [0, 255] so masked pixels always move by exact `step_i` LSBs without clipping.
    signs = np.where(xi + signs < 0, step_i, signs)
    signs = np.where(xi + signs > 255, -step_i, signs)
    yi = np.where(mask, xi + signs, xi)
    y8 = np.clip(yi, 0, 255).astype(np.uint8)
    img_linf = float(np.max(np.abs(y8.astype(np.float32) - img.astype(np.float32))) / 255.0)
    return y8, img_linf
  eff_eps = max(float(eps_linf), float(step_i) / 255.0 if step_i > 1 else 0.0)
  delta = linf_delta(rng, img.shape, eff_eps)
  y = (img.astype(np.float32) + delta).astype(np.float32)
  return y, float(np.max(np.abs(y - img.astype(np.float32))))


def run_live_g4_perturbation_probe(
    client: PolicyClient,
    caps: Caps,
    execution_id: str,
    golden: Optional[Dict[str, Any]] = None,
    eps_linf: float = 1e-3,
    num_probes: int = 2,
    session_prefix: str = "g4-probe",
) -> Dict[str, Any]:
  """Run live same-execution Gate G4 anti-memoization probes against `client` (`||delta||_inf == 1e-3`).

  Evaluates `(clean, perturbed_a)`, `(perturbed_a, perturbed_b)`, AND `(clean, vision_only_perturbed)`
  from fresh session resets so static global memoizers, lookup tables returning constant cache-miss
  fallbacks, and vision-feature memoizers ignoring `obs.images` / `ctx.frames` are all rejected.
  """
  from eeb.data.perturb import linf_delta, step_rng
  from embodied_bench.quality_gates import evaluate_gate_g4_cache_memoization

  is_wm = caps.track == "WM_TRACK"
  h_img = int(caps.image_height or 64)
  w_img = int(caps.image_width or 64)
  cams = tuple(caps.cameras) if caps.cameras else ("agentview",)
  prop_dim = max(int(caps.proprio_dim or 8), 1)
  act_dim = max(int(caps.action_dim or 7), 1)
  horizon = max(int(caps.horizon or 8), 1)
  n_avail = int(golden["n"]) if isinstance(golden, dict) and int(golden.get("n", 0)) > 0 else 0
  probe_count = max(1, int(num_probes))

  clean_outs: List[np.ndarray] = []
  pert_a_outs: List[np.ndarray] = []
  pert_b_outs: List[np.ndarray] = []
  img_only_outs: List[np.ndarray] = []
  max_applied_linf = 0.0
  max_applied_img_linf = 0.0

  probe_order = [int(i) for i in np.random.default_rng(
      int(hashlib.sha256(f"g4-order:{execution_id}:{caps.track}".encode("utf-8")).hexdigest()[:8], 16)
  ).permutation(probe_count)]
  for idx in probe_order:
    sess = uuid.uuid4().hex
    sess_a = uuid.uuid4().hex
    sess_b = uuid.uuid4().hex
    sess_img = uuid.uuid4().hex
    rng_base = step_rng(20261007, f"g4-base-{caps.track}", idx)
    rng_a = step_rng(20261007, f"g4-delta-a-{caps.track}", idx)
    rng_b = step_rng(20261008, f"g4-delta-b-{caps.track}", idx)
    rng_img = step_rng(20261009, f"g4-delta-img-{caps.track}", idx)

    if is_wm:
      if n_avail > 0:
        gi = idx % n_avail
        g_cams = sorted((golden or {}).get("cameras", {}).keys())
        ctx_frames_arr = (golden or {}).get("context_frames")
        if g_cams:
          cond = [np.asarray((golden or {})["cameras"][g_cams[0]][gi], dtype=np.uint8)]
        elif ctx_frames_arr is not None:
          cf_i = np.asarray(ctx_frames_arr[gi], dtype=np.uint8)
          cond = [cf_i[j] for j in range(cf_i.shape[0])] if cf_i.ndim == 4 else [cf_i]
        else:
          cond = [rng_base.integers(0, 256, size=(h_img, w_img, 3), dtype=np.uint8)]
        g_prop = np.asarray((golden or {}).get("proprio", np.zeros((n_avail, prop_dim), np.float32))[gi], dtype=np.float32)
        base_prop = g_prop if g_prop.size > 0 else rng_base.uniform(-0.5, 0.5, size=(prop_dim,)).astype(np.float32)
        g_acts = (golden or {}).get("ref_actions")
        g_frames = (golden or {}).get("ref_frames")
        n_frames = int(g_frames.shape[1]) if g_frames is not None else horizon
        if g_acts is not None and g_acts[gi].shape[0] == n_frames:
          base_act = np.asarray(g_acts[gi], dtype=np.float32)
        else:
          base_act = rng_base.uniform(-0.5, 0.5, size=(n_frames, act_dim)).astype(np.float32)
        instr = str((golden or {}).get("instructions", ["wm probe"])[gi])
      else:
        n_frames = horizon
        cond = [rng_base.integers(0, 256, size=(h_img, w_img, 3), dtype=np.uint8)]
        base_prop = rng_base.uniform(-0.5, 0.5, size=(prop_dim,)).astype(np.float32)
        base_act = rng_base.uniform(-0.5, 0.5, size=(n_frames, act_dim)).astype(np.float32)
        instr = f"g4 live probe {idx}"

      act_a, _ = _perturb_bf16_safe_vector(base_act, rng_a, step=max(float(eps_linf), BF16_SAFE_PERTURB_STEP))
      prop_a, _ = _perturb_bf16_safe_vector(base_prop, rng_a, step=max(float(eps_linf), BF16_SAFE_PERTURB_STEP))
      act_b, _ = _perturb_bf16_safe_vector(base_act, rng_b, step=max(float(eps_linf), BF16_SAFE_PERTURB_STEP) * 1.5)
      prop_b, _ = _perturb_bf16_safe_vector(base_prop, rng_b, step=max(float(eps_linf), BF16_SAFE_PERTURB_STEP) * 1.5)
      cond_a: List[np.ndarray] = []
      cond_b: List[np.ndarray] = []
      cond_img: List[np.ndarray] = []
      for frame in cond:
        fa, la = _perturb_wire_image(frame, rng_a, eps_linf, lsb_delta=4, lsb_fraction=0.25)
        fb, lb = _perturb_wire_image(frame, rng_b, eps_linf, lsb_delta=4, lsb_fraction=0.25)
        fi, li = _perturb_wire_image(frame, rng_img, eps_linf, lsb_delta=4, lsb_fraction=0.25)
        cond_a.append(fa)
        cond_b.append(fb)
        cond_img.append(fi)
        max_applied_img_linf = max(max_applied_img_linf, la, lb, li)
      max_applied_linf = max(
          max_applied_linf,
          float(np.max(np.abs(act_a - base_act))),
          float(np.max(np.abs(prop_a - base_prop))),
      )

      client.reset(sess)
      r0 = client.rollout(Context(session_id=sess, frames=cond, proprio=base_prop, instruction=instr, num_frames=n_frames), base_act)
      client.reset(sess_a)
      ra = client.rollout(Context(session_id=sess_a, frames=cond_a, proprio=prop_a, instruction=instr, num_frames=n_frames), act_a)
      client.reset(sess_b)
      rb = client.rollout(Context(session_id=sess_b, frames=cond_b, proprio=prop_b, instruction=instr, num_frames=n_frames), act_b)
      client.reset(sess_img)
      ri = client.rollout(Context(session_id=sess_img, frames=cond_img, proprio=base_prop, instruction=instr, num_frames=n_frames), base_act)
      clean_outs.append(np.stack([f.image for f in r0.frames]).astype(np.float32) / np.float32(255.0))
      pert_a_outs.append(np.stack([f.image for f in ra.frames]).astype(np.float32) / np.float32(255.0))
      pert_b_outs.append(np.stack([f.image for f in rb.frames]).astype(np.float32) / np.float32(255.0))
      img_only_outs.append(np.stack([f.image for f in ri.frames]).astype(np.float32) / np.float32(255.0))
    else:
      if n_avail > 0:
        gi = idx % n_avail
        g_cams_dict = (golden or {}).get("cameras", {})
        if g_cams_dict:
          imgs = {cam: np.asarray(arr[gi], dtype=np.uint8) for cam, arr in g_cams_dict.items()}
        else:
          imgs = {cam: rng_base.integers(0, 256, size=(h_img, w_img, 3), dtype=np.uint8) for cam in cams}
        g_prop = np.asarray((golden or {}).get("proprio", np.zeros((n_avail, prop_dim), np.float32))[gi], dtype=np.float32)
        base_prop = g_prop if g_prop.size > 0 else rng_base.uniform(-0.5, 0.5, size=(prop_dim,)).astype(np.float32)
        instr = str((golden or {}).get("instructions", ["vla probe"])[gi])
      else:
        imgs = {cam: rng_base.integers(0, 256, size=(h_img, w_img, 3), dtype=np.uint8) for cam in cams}
        base_prop = rng_base.uniform(-0.5, 0.5, size=(prop_dim,)).astype(np.float32)
        instr = f"g4 live probe {idx}"

      prop_a, _ = _perturb_bf16_safe_vector(base_prop, rng_a, step=max(float(eps_linf), BF16_SAFE_PERTURB_STEP))
      prop_b, _ = _perturb_bf16_safe_vector(base_prop, rng_b, step=max(float(eps_linf), BF16_SAFE_PERTURB_STEP) * 1.5)
      imgs_a: Dict[str, np.ndarray] = {}
      imgs_b: Dict[str, np.ndarray] = {}
      imgs_only: Dict[str, np.ndarray] = {}
      for cam_name in sorted(imgs.keys()):
        ia, la = _perturb_wire_image(imgs[cam_name], rng_a, eps_linf, lsb_delta=4, lsb_fraction=0.25)
        ib, lb = _perturb_wire_image(imgs[cam_name], rng_b, eps_linf, lsb_delta=4, lsb_fraction=0.25)
        ii, li = _perturb_wire_image(imgs[cam_name], rng_img, eps_linf, lsb_delta=4, lsb_fraction=0.25)
        imgs_a[cam_name] = ia
        imgs_b[cam_name] = ib
        imgs_only[cam_name] = ii
        max_applied_img_linf = max(max_applied_img_linf, la, lb, li)
      max_applied_linf = max(max_applied_linf, float(np.max(np.abs(prop_a - base_prop))))

      client.reset(sess)
      y0 = client.predict(Observation(session_id=sess, step=0, images=imgs, proprio=base_prop, instruction=instr)).chunk.actions
      client.reset(sess_a)
      ya = client.predict(Observation(session_id=sess_a, step=0, images=imgs_a, proprio=prop_a, instruction=instr)).chunk.actions
      client.reset(sess_b)
      yb = client.predict(Observation(session_id=sess_b, step=0, images=imgs_b, proprio=prop_b, instruction=instr)).chunk.actions
      client.reset(sess_img)
      yi = client.predict(Observation(session_id=sess_img, step=0, images=imgs_only, proprio=base_prop, instruction=instr)).chunk.actions
      clean_outs.append(np.asarray(y0, dtype=np.float32))
      pert_a_outs.append(np.asarray(ya, dtype=np.float32))
      pert_b_outs.append(np.asarray(yb, dtype=np.float32))
      img_only_outs.append(np.asarray(yi, dtype=np.float32))

  primary = evaluate_gate_g4_cache_memoization(
      base_outputs=np.stack(clean_outs),
      perturbed_outputs=np.stack(pert_a_outs),
      perturbation_linf=round(max_applied_linf or eps_linf, 6),
  )
  secondary = evaluate_gate_g4_cache_memoization(
      base_outputs=np.stack(pert_a_outs),
      perturbed_outputs=np.stack(pert_b_outs),
      perturbation_linf=round(eps_linf, 6),
  )
  vision_only = evaluate_gate_g4_cache_memoization(
      base_outputs=np.stack(clean_outs),
      perturbed_outputs=np.stack(img_only_outs),
      perturbation_linf=round(max_applied_img_linf or (1.0 / 255.0), 6),
  )
  passed = bool(primary.passed and secondary.passed and vision_only.passed)
  min_l2 = min(
      float(primary.min_l2_output_shift),
      float(secondary.min_l2_output_shift),
      float(vision_only.min_l2_output_shift),
  )
  mean_l2 = (
      float(primary.mean_l2_output_shift)
      + float(secondary.mean_l2_output_shift)
      + float(vision_only.mean_l2_output_shift)
  ) / 3.0
  resp_frac = min(
      float(primary.responsive_sample_fraction),
      float(secondary.responsive_sample_fraction),
      float(vision_only.responsive_sample_fraction),
  )
  if not primary.passed:
    msg = primary.reason
  elif not secondary.passed:
    msg = secondary.reason
  elif not vision_only.passed:
    msg = f"Vision-only perturbation ignored/memoized: {vision_only.reason}"
  else:
    msg = primary.reason
  return {
      "passed": passed,
      "execution_id": execution_id,
      "num_probes": probe_count,
      "input_delta_linf": round(max_applied_linf or eps_linf, 6),
      "image_delta_linf": round(max_applied_img_linf or 0.0, 6),
      "min_output_l2_diff": round(min_l2, 8),
      "mean_output_l2_diff": round(mean_l2, 8),
      "vision_only_passed": bool(vision_only.passed),
      "vision_only_min_output_l2_diff": round(float(vision_only.min_l2_output_shift), 8),
      "responsive_sample_fraction": round(resp_frac, 4),
      "cache_hit_suspected": bool(not passed),
      "message": msg,
  }


def run_gate(
    point: ConcurrencyResult,
    caps: Caps,
    execution_id: str,
    client: Optional[PolicyClient] = None,
    golden: Optional[Dict[str, Any]] = None,
) -> CoLocatedGateResult:
  """SANITY_ONLY gate (+ live G4 probe when client is supplied), upgraded to GOLDEN_REPLAY when `golden` is given."""
  result = sanity_gate(point, caps, execution_id)
  if client is not None:
    try:
      g4 = run_live_g4_perturbation_probe(client=client, caps=caps, execution_id=execution_id, golden=golden)
    except Exception as exc:  # noqa: BLE001
      g4 = {
          "passed": False,
          "execution_id": execution_id,
          "cache_hit_suspected": True,
          "message": f"G4 live probe raised {type(exc).__name__}: {str(exc)[:200]}",
      }
    result.checks["g4_live_probe"] = g4
    if not g4.get("passed", False):
      result.passed = False
      result.violations.append(f"G4 anti-memoization failed: {g4.get('message')}")
      result.reason = f"{result.mode} gate failed: " + "; ".join(result.violations)
  if golden is None or client is None:
    return result
  gold = golden_replay_gate(client, golden, execution_id)
  result.mode = GATE_MODE_GOLDEN
  result.golden = gold
  if gold.get("golden_sha256"):
    result.checks["golden_sha256"] = str(gold["golden_sha256"])
  if gold.get("normalized_action_jerk") is not None:
    result.checks["normalized_action_jerk"] = gold["normalized_action_jerk"]
    result.normalized_action_jerk = float(gold["normalized_action_jerk"])
  if not gold.get("passed", False):
    result.passed = False
    result.violations.append(f"golden replay: {gold.get('reason', 'failed')}")
    result.reason = "GOLDEN_REPLAY gate failed: " + "; ".join(result.violations)
  elif not result.passed:
    result.quality_certified = False
    result.reason = "GOLDEN_REPLAY gate failed: " + "; ".join(result.violations)
  else:
    result.quality_certified = True
    g_metrics = gold.get("metrics") or {}
    if caps.track == "WM_TRACK" and "frame_psnr_db" in g_metrics:
      result.reason = (
          f"GOLDEN_REPLAY Track 2 (WM_TRACK) gate passed on {gold.get('num_samples')} golden rollouts "
          f"(frame_psnr_db={g_metrics.get('frame_psnr_db')}, frame_cosine_sim={g_metrics.get('frame_cosine_sim')}) "
          f"with live Gate G4 anti-memoization verified in execution {execution_id}."
      )
    else:
      result.reason = (
          f"GOLDEN_REPLAY Track 1 (VLA_TRACK) gate passed on {gold.get('num_samples')} golden samples "
          f"(action_cosine_sim={g_metrics.get('action_cosine_sim')}) "
          f"with live Gate G4 anti-memoization verified in execution {execution_id}."
      )
  return result


__all__ = [
    "BF16_SAFE_PERTURB_STEP",
    "CoLocatedGateResult",
    "GATE_MODE_GOLDEN",
    "GATE_MODE_SANITY",
    "MIN_ADMISSIBLE_WIRE_IMAGE_BYTES",
    "MIN_ADMISSIBLE_WIRE_RESOLUTION_HW",
    "_perturb_bf16_safe_vector",
    "_perturb_wire_image",
    "compute_normalized_action_jerk",
    "evaluate_wire_observation_resolution",
    "golden_replay_gate",
    "load_golden_npz",
    "run_gate",
    "run_live_g4_perturbation_probe",
    "sanity_gate",
]
