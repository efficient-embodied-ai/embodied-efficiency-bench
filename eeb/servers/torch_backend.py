"""Shared PyTorch device resolution, TorchTPU (`torch_tpu`) setup, synchronization, and `torch.compile` helpers.

Default Cloud TPU execution path for EmbodiedX v1.0 PyTorch servers:
  - Install open-source TorchTPU (`pip install torch torch_tpu`).
  - Pass `device="tpu"` (or `device="auto"`, which probes CUDA/ROCm, then TorchTPU, then CPU fallback).
  - End every `predict()` / streamed `rollout()` chunk with `synchronize_torch_device(device)` so LoadGen
    latency timers record complete accelerator execution (`torch.cuda.synchronize()` on GPU and
    `torch.accelerator.synchronize()` / `torch.tpu.synchronize()` on Cloud TPU).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any, Optional
import warnings


TORCHTPU_INSTALL_HINT = (
    "Google Cloud TPU execution via TorchTPU requires the open-source `torch_tpu` backend:\n"
    "  pip install torch torch_tpu\n"
    "Verify the TPU chip is visible (`python -c \"import torch, torch_tpu; print(torch.accelerator.device_count())\"`)\n"
    "and pass `device=\"tpu\"` (or `device=\"auto\"`) to the policy server."
)


def _ensure_torch() -> Any:
  try:
    import torch  # noqa: WPS433

    return torch
  except ImportError as exc:
    raise ImportError(
        "PyTorch (`torch`) is not installed. Install `torch` (and `torch_tpu` for Google Cloud TPU hosts):\n"
        + TORCHTPU_INSTALL_HINT
    ) from exc


def _accelerator_is_tpu(torch_mod: Any) -> bool:
  acc = getattr(torch_mod, "accelerator", None)
  if acc is None:
    return False
  cur_fn = getattr(acc, "current_accelerator", None)
  avail_fn = getattr(acc, "is_available", None)
  if not callable(cur_fn) or not callable(avail_fn):
    return False
  try:
    if not bool(avail_fn()):
      return False
    cur = cur_fn()
    return str(getattr(cur, "type", cur)) == "tpu"
  except Exception:
    return False


def _torchtpu_is_available(torch_mod: Any, tpu_ns: Any) -> bool:
  """Check TorchTPU availability across both legacy `torch.tpu.is_available()` and `torch_tpu>=0.2` (`torch.accelerator`)."""
  is_avail = getattr(tpu_ns, "is_available", None)
  if callable(is_avail):
    try:
      return bool(is_avail())
    except RuntimeError as exc:
      if "internal to TorchTPU" not in str(exc):
        raise
  return _accelerator_is_tpu(torch_mod)


def _torchtpu_device_count(torch_mod: Any, tpu_ns: Any) -> int:
  """Return visible TPU device count across legacy `torch.tpu.device_count()` and `torch_tpu>=0.2` (`torch.accelerator`)."""
  dev_count_fn = getattr(tpu_ns, "device_count", None)
  if callable(dev_count_fn):
    try:
      return int(dev_count_fn())
    except RuntimeError as exc:
      if "internal to TorchTPU" not in str(exc):
        raise
  acc = getattr(torch_mod, "accelerator", None)
  if acc is not None and _accelerator_is_tpu(torch_mod):
    acc_count = getattr(acc, "device_count", None)
    if callable(acc_count):
      return int(acc_count())
  return 0


def _try_import_torchtpu(torch_mod: Any) -> bool:
  """Best-effort `torch_tpu` import for `device='auto'` probing without failing non-TPU hosts."""
  tpu_ns = getattr(torch_mod, "tpu", None)
  if tpu_ns is not None and (callable(getattr(tpu_ns, "is_available", None)) or _accelerator_is_tpu(torch_mod)):
    return True
  in_sys = sys.modules.get("torch_tpu", False)
  if in_sys is None:
    return False
  if in_sys is False:
    try:
      spec = importlib.util.find_spec("torch_tpu")
    except (ImportError, ValueError):
      spec = None
    if spec is None:
      return False
  try:
    importlib.import_module("torch_tpu")
  except Exception:
    return False
  tpu_ns = getattr(torch_mod, "tpu", None)
  return tpu_ns is not None


def _require_torchtpu(torch_mod: Any, requested_device: str) -> Any:
  """Import `torch_tpu` for an explicit `tpu` device and return `torch.tpu`."""
  try:
    importlib.import_module("torch_tpu")
  except ImportError as exc:
    raise ImportError(
        f"Requested TorchTPU device {requested_device!r}, but `torch_tpu` is not installed.\n"
        + TORCHTPU_INSTALL_HINT
    ) from exc

  tpu_ns = getattr(torch_mod, "tpu", None)
  if tpu_ns is None:
    raise RuntimeError(
        f"Imported `torch_tpu` for device {requested_device!r}, but `torch.tpu` namespace was not registered."
    )
  if not _torchtpu_is_available(torch_mod, tpu_ns):
    raise RuntimeError(
        f"Requested TorchTPU device {requested_device!r}, but TorchTPU reports no available TPU device. "
        "Ensure a single Cloud TPU chip (v6e-1 or v5e-1) is attached and visible (`TPU_VISIBLE_DEVICES=0`)."
    )
  return tpu_ns


def _normalize_legacy_xla_device(raw_device: str) -> str:
  lowered = raw_device.strip().lower()
  if lowered == "xla" or lowered.startswith("xla:"):
    suffix = lowered[3:]
    mapped = f"tpu{suffix}"
    warnings.warn(
        f"Legacy device {raw_device!r} is deprecated in EmbodiedX v1.0; mapping to native TorchTPU "
        f"device {mapped!r}. Install open-source `torch_tpu` and pass `device='tpu'` directly.",
        DeprecationWarning,
        stacklevel=3,
    )
    return mapped
  return lowered


def resolve_torch_device(device: str = "auto", *, allow_cpu_fallback: bool = True) -> str:
  """Resolve a PyTorch accelerator device string supporting CUDA/ROCm, TorchTPU (`tpu`), and `auto`."""
  raw = (device or "auto").strip()
  normalized = _normalize_legacy_xla_device(raw or "auto")
  torch_mod = _ensure_torch()

  if normalized in ("", "auto"):
    cuda_ns = getattr(torch_mod, "cuda", None)
    if cuda_ns is not None and callable(getattr(cuda_ns, "is_available", None)) and bool(cuda_ns.is_available()):
      return "cuda"
    if _try_import_torchtpu(torch_mod):
      tpu_ns = getattr(torch_mod, "tpu", None)
      if tpu_ns is not None and _torchtpu_is_available(torch_mod, tpu_ns):
        return "tpu"
    if allow_cpu_fallback:
      return "cpu"
    raise RuntimeError(
        "No PyTorch accelerator detected for device='auto' (`torch.cuda.is_available()` and "
        "`torch.tpu.is_available()` are both False). Attach a single GPU or Cloud TPU, or pass device='cpu'."
    )

  if normalized == "tpu" or normalized.startswith("tpu:"):
    tpu_ns = _require_torchtpu(torch_mod, raw)
    if ":" in normalized:
      idx_str = normalized.split(":", 1)[1]
      if not idx_str.isdigit():
        raise ValueError(f"Invalid TorchTPU device index in {raw!r}; expected 'tpu' or 'tpu:<index>'.")
      idx = int(idx_str)
      count = _torchtpu_device_count(torch_mod, tpu_ns)
      if count > 0 and idx >= count:
        raise RuntimeError(
            f"Requested device {raw!r}, but `torch.tpu.device_count()` is {count} "
            "(EmbodiedX v1.0 enforces `num_devices == 1`, index 0)."
        )
    return normalized

  if normalized == "cuda" or normalized.startswith("cuda:"):
    cuda_ns = getattr(torch_mod, "cuda", None)
    if cuda_ns is None or not callable(getattr(cuda_ns, "is_available", None)) or not bool(cuda_ns.is_available()):
      raise RuntimeError(
          f"Requested CUDA/ROCm device {raw!r}, but `torch.cuda.is_available()` returned False."
      )
    return normalized

  if normalized == "cpu" or normalized == "mps" or normalized.startswith("mps:"):
    return normalized

  raise ValueError(
      f"Unsupported PyTorch device {raw!r}; expected 'auto', 'cuda', 'tpu', or 'cpu'."
  )


def synchronize_torch_device(device: Optional[str]) -> None:
  """Block until pending async kernels complete on `device`."""
  if not device:
    return
  normalized = _normalize_legacy_xla_device(str(device))
  if normalized in ("", "cpu"):
    return

  torch_mod = _ensure_torch()
  if normalized == "cuda" or normalized.startswith("cuda:"):
    sync_fn = getattr(getattr(torch_mod, "cuda", None), "synchronize", None)
    if callable(sync_fn):
      if ":" in normalized:
        sync_fn(normalized)
      else:
        sync_fn()
    return

  if normalized == "tpu" or normalized.startswith("tpu:"):
    tpu_ns = getattr(torch_mod, "tpu", None)
    if tpu_ns is None:
      tpu_ns = _require_torchtpu(torch_mod, str(device))
    sync_fn = getattr(tpu_ns, "synchronize", None)
    if callable(sync_fn):
      try:
        sync_fn()
        return
      except RuntimeError as exc:
        if "internal to TorchTPU" not in str(exc):
          raise
    acc = getattr(torch_mod, "accelerator", None)
    acc_sync = getattr(acc, "synchronize", None) if acc is not None else None
    if callable(acc_sync):
      acc_sync()
      return
    raise RuntimeError(
        f"TorchTPU synchronize() is unavailable for device {device!r}; check your `torch_tpu` installation."
    )

  if normalized == "mps" or normalized.startswith("mps:"):
    mps_ns = getattr(torch_mod, "mps", None)
    sync_fn = getattr(mps_ns, "synchronize", None)
    if callable(sync_fn):
      sync_fn()


def maybe_compile_torch_module(
    module: Any,
    *,
    device: str,
    enabled: bool = False,
    mode: Optional[str] = None,
) -> Any:
  """Optionally wrap `module` (or callable) in `torch.compile(...)` for TorchTPU / CUDA / ROCm graph fusion."""
  if not enabled or module is None:
    return module
  torch_mod = _ensure_torch()
  compile_fn = getattr(torch_mod, "compile", None)
  if not callable(compile_fn):
    raise RuntimeError(
        f"`torch_compile=True` requested on device {device!r}, but `torch.compile` is unavailable in "
        f"PyTorch {getattr(torch_mod, '__version__', 'unknown')}."
    )
  kwargs: dict[str, Any] = {}
  if mode:
    kwargs["mode"] = mode
  return compile_fn(module, **kwargs)


__all__ = [
    "TORCHTPU_INSTALL_HINT",
    "maybe_compile_torch_module",
    "resolve_torch_device",
    "synchronize_torch_device",
]
