"""AMD Instinct / Radeon telemetry via `amd-smi` (JSON) with `rocm-smi` (CSV) fallback.

Field names differ across ROCm releases, so both parsers match keys by substring and return
None for anything missing (never a fabricated value). UNTESTED on AMD hardware in this repo -
the parsers are covered by unit tests on captured/representative outputs only.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from eeb.telemetry.source import TelemetrySource, make_sample, run_cli, unavailable_sample, which


def _find_number(node: Any, *needles: str) -> Optional[float]:
  """Depth-first search for the first numeric leaf whose key contains all `needles` (lowercase)."""
  if isinstance(node, dict):
    for key, val in node.items():
      k = str(key).lower()
      if all(n in k for n in needles):
        if isinstance(val, dict) and "value" in val:
          val = val["value"]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
          return float(val)
        if isinstance(val, str):
          try:
            return float(val.split()[0])
          except (ValueError, IndexError):
            pass
    for val in node.values():
      found = _find_number(val, *needles)
      if found is not None:
        return found
  elif isinstance(node, list):
    for item in node:
      found = _find_number(item, *needles)
      if found is not None:
        return found
  return None


def parse_amd_smi_json(text: str, gpu_index: int = 0) -> Dict[str, Any]:
  """Parse `amd-smi metric --power --mem-usage --usage --json` output for one GPU."""
  data = json.loads(text)
  entries = data if isinstance(data, list) else [data]
  entry = None
  for item in entries:
    if isinstance(item, dict) and int(item.get("gpu", -1)) == gpu_index:
      entry = item
      break
  if entry is None:
    entry = entries[0] if entries else {}
  power_w = _find_number(entry.get("power", entry), "socket_power") or _find_number(entry.get("power", entry), "power")
  used_mb = _find_number(entry.get("mem_usage", entry), "used", "vram")
  util = _find_number(entry.get("usage", entry), "gfx_activity")
  if util is None:
    util = _find_number(entry.get("usage", entry), "gfx")
  return make_sample("amd-smi", power_w=power_w, mem_used_mb=used_mb, util_pct=util)


def parse_rocm_smi_csv(text: str) -> Dict[str, Any]:
  """Parse `rocm-smi --showpower --showmeminfo vram --showuse --csv` (header + first data row)."""
  lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.startswith("=")]
  if len(lines) < 2:
    raise ValueError(f"rocm-smi CSV needs a header and a data row, got: {text!r}")
  headers = [h.strip().lower() for h in lines[0].split(",")]
  values = [v.strip() for v in lines[1].split(",")]
  row = dict(zip(headers, values))

  def _pick(*needles: str) -> Optional[float]:
    for key, val in row.items():
      if all(n in key for n in needles):
        try:
          return float(val)
        except ValueError:
          return None
    return None

  power_w = _pick("power", "(w)")
  used_b = _pick("used memory", "(b)")
  util = _pick("gpu use")
  return make_sample(
      "rocm-smi",
      power_w=power_w,
      mem_used_mb=None if used_b is None else used_b / (1024.0 * 1024.0),
      util_pct=util,
  )


class AmdSmiSource(TelemetrySource):
  name = "amd_smi"

  def __init__(self, gpu_index: int = 0) -> None:
    self.gpu_index = int(gpu_index)
    self._amd_smi = which("amd-smi")
    self._rocm_smi = which("rocm-smi")

  def available(self) -> bool:
    return bool(self._amd_smi or self._rocm_smi)

  def unavailable_reason(self) -> str:
    return "neither amd-smi nor rocm-smi on PATH (not a ROCm host)"

  def sample(self) -> Dict[str, Any]:
    if self._amd_smi:
      out = run_cli([self._amd_smi, "metric", "--power", "--mem-usage", "--usage", "--json", "-g", str(self.gpu_index)])
      if out:
        try:
          return parse_amd_smi_json(out, self.gpu_index)
        except (ValueError, json.JSONDecodeError) as exc:
          last = f"amd-smi JSON parse failed: {exc}"
      else:
        last = "amd-smi returned no output"
    else:
      last = "amd-smi not on PATH"
    if self._rocm_smi:
      out = run_cli([self._rocm_smi, "--showpower", "--showmeminfo", "vram", "--showuse", "--csv", "-d", str(self.gpu_index)])
      if out:
        try:
          return parse_rocm_smi_csv(out)
        except ValueError as exc:
          last = f"rocm-smi CSV parse failed: {exc}"
      else:
        last = f"{last}; rocm-smi returned no output"
    return unavailable_sample(self.name, last)


__all__ = ["AmdSmiSource", "parse_amd_smi_json", "parse_rocm_smi_csv"]
