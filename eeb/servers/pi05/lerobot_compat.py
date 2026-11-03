"""Resolve the lerobot APIs the pi0.5 PyTorch server uses, across lerobot releases.

The server was written against a lerobot whose pi0.5 module had, at once:
  * `clone_past_key_values` and `make_att_2d_masks` at module level in `lerobot.policies.pi05.modeling_pi05`, and
  * a model method `_prepare_attention_masks_4d(att_2d_masks, dtype=...)`.
No release has both: v0.4.4 and v0.5.x have `make_att_2d_masks` and the method without `dtype`, and no
`clone_past_key_values`; v0.6.0 adds `clone_past_key_values`, still without `dtype`; v0.6.1 moved the helpers into
`lerobot.policies.common.vla_utils` and replaced the method with `prepare_attention_masks_4d(att_2d_masks, dtype=...)`.
`PI05Config.dtype` is a string ("bfloat16" or "float32") in every release, and concrete config classes reject the
checkpoint's `type` key. These helpers detect which shape is installed and adapt, so one server runs on a pinned lerobot.
Pure Python: nothing here imports torch or lerobot until it is called.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

_HELPER_MODULES = ("lerobot.policies.pi05.modeling_pi05", "lerobot.policies.common.vla_utils")


def resolve_helper(name: str) -> Any | None:
  """A module-level lerobot helper, looked up where releases keep it. None when no installed module has it."""
  for module_name in _HELPER_MODULES:
    try:
      module = importlib.import_module(module_name)
    except ImportError:
      continue
    if hasattr(module, name):
      return getattr(module, name)
  return None


def clone_past_key_values(past_key_values: Any) -> Any:
  """lerobot's own helper when this release has one; otherwise a deep copy of the cache (what the helper does)."""
  helper = resolve_helper("clone_past_key_values")
  return helper(past_key_values) if helper is not None else copy.deepcopy(past_key_values)


def make_att_2d_masks(pad_masks: Any, att_masks: Any) -> Any:
  helper = resolve_helper("make_att_2d_masks")
  if helper is None:
    raise ImportError("this lerobot release has no `make_att_2d_masks` in pi05.modeling_pi05 or common.vla_utils")
  return helper(pad_masks, att_masks)


def attention_mask_4d(model: Any, att_2d_masks: Any, dtype: Any) -> Any:
  """The additive 4D attention mask in `dtype`, from whichever API this lerobot release has."""
  method = getattr(model, "_prepare_attention_masks_4d", None)
  if method is not None:
    try:
      return method(att_2d_masks, dtype=dtype)
    except TypeError:  # this release's method takes no dtype
      return method(att_2d_masks).to(dtype=dtype)
  helper = resolve_helper("prepare_attention_masks_4d")
  if helper is None:
    raise ImportError("this lerobot release has neither `_prepare_attention_masks_4d` nor `prepare_attention_masks_4d`")
  return helper(att_2d_masks, dtype=dtype)


def config_dtype(config: Any, torch_dtype: Any) -> Any:
  """The value to store in `config.dtype`: lerobot declares it as a string and checks `precision == "bfloat16"` /
  `== "float32"` when it builds the model, so a torch dtype matches neither branch (and is rejected as an invalid precision)."""
  if isinstance(getattr(config, "dtype", None), str):
    return str(torch_dtype).removeprefix("torch.")
  return torch_dtype


def load_config(config_cls: Any, checkpoint: str, **kwargs: Any) -> Any:
  """`config_cls.from_pretrained`, falling back to the base `PreTrainedConfig` loader, which consumes the checkpoint's
  `type` key and returns the right subclass. Concrete subclasses reject that key in some releases."""
  try:
    return config_cls.from_pretrained(checkpoint, **kwargs)
  except Exception as first:  # noqa: BLE001 - the parser's error type varies by release
    base = None
    for module_name in ("lerobot.configs.policies", "lerobot.configs"):
      try:
        base = getattr(importlib.import_module(module_name), "PreTrainedConfig")
        break
      except (ImportError, AttributeError):
        continue
    if base is None:
      raise
    config = base.from_pretrained(checkpoint, **kwargs)
    if not isinstance(config, config_cls):
      raise first
    return config
