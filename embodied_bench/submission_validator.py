"""Third-party PR submission directory & recipe validator, single-chip hardware envelope auditor (`CAP_SINGLE_CHIP`), static AST anti-cheat source scanner (`Gate 4`), and crash-safe stateful submission registry (`Gates 1 & 2`) for EmbodiedX (`embodied-efficiency-bench`)."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import io
import os
from pathlib import Path
import re
import tokenize
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SAFE_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _audit_directory_symlinks_and_containment(root_dir: Path) -> List[str]:
  """Verify `root_dir` contains zero symbolic links and all files stay inside `root_dir.resolve()` (RT-07)."""
  errors: List[str] = []
  if root_dir.is_symlink():
    errors.append(
        f"Submission root path {root_dir} is a symbolic link (RT-07: symlinks are prohibited in submission bundles)."
    )
    return errors
  if not root_dir.is_dir():
    return errors
  resolved_root = root_dir.resolve()
  for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
    cur = Path(dirpath)
    for name in list(dirnames) + list(filenames):
      entry = cur / name
      if entry.is_symlink():
        errors.append(
            f"Submission bundle contains prohibited symbolic link {entry.relative_to(root_dir).as_posix()!r} (RT-07)."
        )
        continue
      try:
        entry.resolve().relative_to(resolved_root)
      except ValueError:
        errors.append(
            f"Submission entry {entry.relative_to(root_dir).as_posix()!r} escapes bundle root {resolved_root} (RT-07)."
        )
    dirnames[:] = [d for d in dirnames if not (cur / d).is_symlink()]
  return errors


def _audit_untrusted_slug(field_name: str, value: Optional[str]) -> Optional[str]:
  """Validate untrusted string IDs against `SAFE_SLUG_PATTERN` (`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`) (RT-07)."""
  if value is None or str(value) == "":
    return None
  s = str(value)
  if ".." in s or not SAFE_SLUG_PATTERN.fullmatch(s):
    return (
        f"Invalid {field_name}={s!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ with no '..' or shell/markdown metacharacters (RT-07)."
    )
  return None

from embodied_bench.attestation import verify_submission_receipt
from embodied_bench.schema import (
    CANONICAL_CHALLENGE_MODELS,
    DivisionType,
    HARDWARE_SKU_SPECS,
    MODERN_REFERENCE_CHECKPOINTS,
    PERFORMANCE_CAP_ENVELOPES,
    PerformanceCapId,
    RecipeConfig,
    RunStatus,
    atomic_read_modify_write_json,
    atomic_write_json,
    cleanup_orphan_temp_files,
    load_yaml_or_json,
    now_pacific_iso,
    resolve_perf_cap_id,
    validate_division_compliance,
    validate_status_and_reason,
    validate_submission_meta,
)


# Environment variables that control visible accelerator devices
_DEVICE_VISIBILITY_ENV_VARS: frozenset[str] = frozenset({
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "TPU_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "HABANA_VISIBLE_MODULES",
})

# Multi-device parallelism keyword arguments / config variables prohibited under CAP_SINGLE_CHIP
_PARALLELISM_SIZE_IDENTIFIERS: frozenset[str] = frozenset({
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "expert_parallel_size",
    "context_parallel_size",
    "decode_context_parallel_size",
    "num_devices",
    "world_size",
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
})

# Timer attributes protected against monkeypatching / mocking
_PROTECTED_TIMER_ATTRS: frozenset[str] = frozenset({
    "perf_counter",
    "perf_counter_ns",
    "monotonic",
    "monotonic_ns",
})

# Quality gate and attestation identifiers protected against hardcoding / mocking
_PROTECTED_GATE_FLAGS: frozenset[str] = frozenset({
    "gate_a_passed",
    "gate_b_passed",
    "compliance_passed",
    "single_device_compliant",
})

_MULTI_DEVICE_ENV_ASSIGN_RE = re.compile(
    r"\b(?P<var>CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|TPU_VISIBLE_DEVICES)"
    r"\s*=\s*['\"]?(?P<val>[0-9]+\s*,\s*[0-9, a-zA-Z_-]+)['\"]?"
)

_TPU_BOUNDS_ASSIGN_RE = re.compile(
    r"\bTPU_CHIPS_PER_HOST_BOUNDS\s*=\s*['\"]?(?P<val>[0-9]+\s*,\s*[0-9]+\s*,\s*[0-9]+)['\"]?"
)


@dataclass(frozen=True)
class AntiCheatViolation:
  """Single static AST / source anti-cheat violation with actionable remediation."""

  rule_id: str
  file_path: str
  line: int
  col: int
  code_snippet: str
  message: str
  remediation: str

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class SourceAuditResult:
  """Result of auditing Python source files in a submission package for prohibited benchmark cheats."""

  passed: bool
  files_scanned: List[str]
  violations: List[AntiCheatViolation] = field(default_factory=list)
  reason: str = "All Python source files passed static AST anti-cheat audit."
  audited_at: str = field(default_factory=now_pacific_iso)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "passed": self.passed,
        "files_scanned": list(self.files_scanned),
        "violations": [v.to_dict() for v in self.violations],
        "reason": self.reason,
        "audited_at": self.audited_at,
    }


@dataclass
class HardwareEnvelopeCheckResult:
  """Result of checking a hardware SKU and recipe against the single-chip envelope (`CAP_SINGLE_CHIP`)."""

  passed: bool
  envelope_id: str
  hardware_sku: str
  vendor: str
  num_devices: int
  vram_gb: float
  fp16_tflops: float
  fp8_tflops: float
  mem_bw_gbps: float
  limits: Dict[str, float]
  violations: List[str] = field(default_factory=list)
  reason: str = "Single-chip hardware envelope satisfied."

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class SubmissionValidationReport:
  """Comprehensive validation report for a 3rd-party EmbodiedX submission folder or recipe."""

  submission_id: str
  submission_path: str
  recipe_path: str
  recipe_id: str
  track: str
  division: str
  clock: str
  scenario: str
  model_name: str
  baseline_model_id: str
  canonical_challenge_model: bool
  precision: str
  hardware_sku: str
  perf_cap: str
  num_devices: int
  passed: bool
  status: str
  reason: Optional[str]
  schema_valid: bool
  division_compliant: bool
  hardware_envelope_check: Dict[str, Any]
  source_audit: Dict[str, Any]
  has_measurements: bool = False
  validation_scope: str = "RECIPE_ONLY_NO_MEASUREMENTS"
  receipt_verified: Optional[bool] = None
  receipt_message: Optional[str] = None
  submission_phase: str = "OPEN_RECIPE"
  recipe_commit_sha256: Optional[str] = None
  engine_tag: Optional[str] = None
  submitter_team: Optional[str] = None
  violations: List[str] = field(default_factory=list)
  validated_at: str = field(default_factory=now_pacific_iso)

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def _is_multi_device_visible_string(val: str) -> bool:
  """Return True if a VISIBLE_DEVICES value selects more than 1 device (e.g. '0,1')."""
  cleaned = val.strip().strip("'\"")
  if not cleaned or cleaned in ("-1", "none", "None", "void"):
    return False
  parts = [p.strip() for p in cleaned.split(",") if p.strip()]
  return len(parts) > 1


def _is_multi_chip_tpu_bounds_string(val: str) -> bool:
  """Return True if TPU_CHIPS_PER_HOST_BOUNDS product exceeds 1 chip (e.g. '2,2,1')."""
  cleaned = val.strip().strip("'\"")
  parts = [p.strip() for p in cleaned.split(",") if p.strip()]
  if not parts:
    return False
  try:
    dims = [int(p) for p in parts]
  except ValueError:
    return False
  product = 1
  for d in dims:
    product *= d
  return product > 1


def _collect_docstring_expr_ids(tree: ast.AST) -> Set[int]:
  """Collect `id(node)` for all module, class, and function docstring `ast.Expr` nodes."""
  doc_ids: Set[int] = set()
  for node in ast.walk(tree):
    if isinstance(
        node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
      if node.body and isinstance(node.body[0], ast.Expr):
        first_val = node.body[0].value
        if isinstance(first_val, ast.Constant) and isinstance(first_val.value, str):
          doc_ids.add(id(node.body[0]))
  return doc_ids


def _expr_contains_raw_bytes_key(node: ast.AST) -> bool:
  """Return True if an index/key AST expression hashes or serializes raw input tensor bytes."""
  for sub in ast.walk(node):
    if isinstance(sub, ast.Call):
      # Match `<tensor>.tobytes()`, `<tensor>.data.tobytes()`, `<tensor>.numpy().tobytes()`
      if isinstance(sub.func, ast.Attribute) and sub.func.attr in (
          "tobytes",
          "tostring",
          "data_ptr",
      ):
        return True
      # Match `hash(obs.tobytes())`, `hashlib.sha256(obs)`, `pickle.dumps(obs)`
      if isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
        mod_name = sub.func.value.id
        fn_name = sub.func.attr
        if mod_name == "hashlib" and fn_name in ("md5", "sha1", "sha256", "sha512", "blake2b"):
          return True
        if mod_name in ("pickle", "marshal") and fn_name == "dumps":
          return True
    if isinstance(sub, ast.Attribute) and sub.attr == "tobytes":
      return True
  return False


def _extract_dotted_name(node: ast.AST) -> str:
  """Return dotted name for `ast.Name` / `ast.Attribute` chain, or empty string."""
  parts: List[str] = []
  curr: Optional[ast.AST] = node
  while isinstance(curr, ast.Attribute):
    parts.append(curr.attr)
    curr = curr.value
  if isinstance(curr, ast.Name):
    parts.append(curr.id)
    return ".".join(reversed(parts))
  return ""


def _extract_int_literal(node: ast.AST) -> Optional[int]:
  """Return integer value if `node` is an integer literal constant, else None."""
  if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
    return int(node.value)
  if (
      isinstance(node, ast.UnaryOp)
      and isinstance(node.op, ast.USub)
      and isinstance(node.operand, ast.Constant)
      and isinstance(node.operand.value, int)
      and not isinstance(node.operand.value, bool)
  ):
    return -int(node.operand.value)
  return None


def _audit_single_python_source(
    source_text: str,
    file_label: str,
) -> List[AntiCheatViolation]:
  """Run AST + non-comment/non-docstring inspection on a single Python file."""
  violations: List[AntiCheatViolation] = []
  lines = source_text.splitlines()

  def _snippet(lineno: int) -> str:
    if 1 <= lineno <= len(lines):
      return lines[lineno - 1].strip()
    return ""

  try:
    tree = ast.parse(source_text, filename=file_label)
  except SyntaxError as exc:
    lineno = int(exc.lineno or 1)
    col = int(exc.offset or 0)
    violations.append(
        AntiCheatViolation(
            rule_id="SYNTAX_ERROR",
            file_path=file_label,
            line=lineno,
            col=col,
            code_snippet=_snippet(lineno),
            message=f"Python syntax error while parsing submission source: {exc.msg}.",
            remediation="Fix Python syntax errors in the submission file before running validation.",
        )
    )
    return violations

  docstring_expr_ids = _collect_docstring_expr_ids(tree)

  class _AntiCheatVisitor(ast.NodeVisitor):
    """AST visitor ignoring docstring nodes while auditing executable statements and expressions."""

    def __init__(self) -> None:
      self.loop_depth = 0

    def visit_For(self, node: ast.For) -> None:
      self.loop_depth += 1
      self.generic_visit(node)
      self.loop_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
      self.loop_depth += 1
      self.generic_visit(node)
      self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
      self.loop_depth += 1
      self.generic_visit(node)
      self.loop_depth -= 1

    def visit_Expr(self, node: ast.Expr) -> None:
      if id(node) in docstring_expr_ids:
        return
      self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
      # Rule 1: Hot-loop output memoization / lookup table keyed by raw input bytes (`_memo_cache[obs.tobytes()]`)
      if _expr_contains_raw_bytes_key(node.slice):
        container_name = _extract_dotted_name(node.value) or "<cache>"
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        violations.append(
            AntiCheatViolation(
                rule_id="CHEAT_RAW_INPUT_MEMOIZATION",
                file_path=file_label,
                line=lineno,
                col=col,
                code_snippet=_snippet(lineno),
                message=(
                    f"Prohibited raw-input byte memoization detected on {container_name!r} "
                    "indexed by '.tobytes()' / raw input hash."
                ),
                remediation=(
                    "Remove raw observation byte memoization tables. Only legitimate per-session_id "
                    "VLM language-prefix KV caching (`session_kv_cache[session_id]`) is permitted."
                ),
            )
        )

      # Rule 4: `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"` handled in visit_Assign
      self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
      if node.name == "verify_submission_receipt":
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        violations.append(
            AntiCheatViolation(
                rule_id="CHEAT_GATE_OR_RECEIPT_BYPASS",
                file_path=file_label,
                line=lineno,
                col=col,
                code_snippet=_snippet(lineno),
                message="Redefining 'verify_submission_receipt' inside submission code is prohibited.",
                remediation="Import and use canonical 'verify_submission_receipt' from 'embodied_bench.attestation'.",
            )
        )
      self.generic_visit(node)

    def _check_assignment_target_and_value(
        self,
        target: ast.AST,
        value: Optional[ast.AST],
        lineno: int,
        col: int,
    ) -> None:
      dotted_target = _extract_dotted_name(target)
      target_leaf = dotted_target.split(".")[-1] if dotted_target else ""

      # Rule 2: Overriding `time.perf_counter` / `time.perf_counter_ns`
      if (
          dotted_target in (
              "time.perf_counter",
              "time.perf_counter_ns",
              "time.monotonic",
              "time.monotonic_ns",
          )
          or target_leaf in _PROTECTED_TIMER_ATTRS
      ):
        violations.append(
            AntiCheatViolation(
                rule_id="CHEAT_TIMER_OVERRIDE_OR_MOCK",
                file_path=file_label,
                line=lineno,
                col=col,
                code_snippet=_snippet(lineno),
                message=f"Overriding or reassigning high-resolution timer {dotted_target!r} is prohibited.",
                remediation="Do not reassign 'time.perf_counter' or 'time.perf_counter_ns'; call them read-only for timing.",
            )
        )

      # Rule 3: Hardcoding `gate_a_passed = True` or overriding `verify_submission_receipt`
      if target_leaf == "verify_submission_receipt":
        violations.append(
            AntiCheatViolation(
                rule_id="CHEAT_GATE_OR_RECEIPT_BYPASS",
                file_path=file_label,
                line=lineno,
                col=col,
                code_snippet=_snippet(lineno),
                message="Overriding 'verify_submission_receipt' is prohibited.",
                remediation="Use 'embodied_bench.attestation.verify_submission_receipt' without modification.",
            )
        )
      if target_leaf in _PROTECTED_GATE_FLAGS and isinstance(value, ast.Constant) and value.value is True:
        violations.append(
            AntiCheatViolation(
                rule_id="CHEAT_GATE_OR_RECEIPT_BYPASS",
                file_path=file_label,
                line=lineno,
                col=col,
                code_snippet=_snippet(lineno),
                message=f"Hardcoding {target_leaf}=True in submission code is prohibited.",
                remediation=f"Allow the EmbodiedX harness quality gates to compute {target_leaf} from actual outputs.",
            )
        )

      # Rule 3 & Rule 4 on Subscript assignment: e.g. `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"` or `report["gate_a_passed"] = True`
      if isinstance(target, ast.Subscript):
        if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
          key_str = target.slice.value
          if key_str in _PROTECTED_GATE_FLAGS and isinstance(value, ast.Constant) and value.value is True:
            violations.append(
                AntiCheatViolation(
                    rule_id="CHEAT_GATE_OR_RECEIPT_BYPASS",
                    file_path=file_label,
                    line=lineno,
                    col=col,
                    code_snippet=_snippet(lineno),
                    message=f"Hardcoding [{key_str!r}] = True in submission code is prohibited.",
                    remediation="Do not hardcode quality gate results in dictionaries.",
                )
            )
          if key_str in _DEVICE_VISIBILITY_ENV_VARS and isinstance(value, ast.Constant) and isinstance(value.value, str):
            if _is_multi_device_visible_string(value.value):
              violations.append(
                  AntiCheatViolation(
                      rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                      file_path=file_label,
                      line=lineno,
                      col=col,
                      code_snippet=_snippet(lineno),
                      message=(
                          f"Multi-device environment assignment {key_str}={value.value!r} violates the "
                          "single-chip rule (num_devices == 1)."
                      ),
                      remediation=f"Restrict {key_str} to a single device index such as '0'.",
                  )
              )
          if key_str == "TPU_CHIPS_PER_HOST_BOUNDS" and isinstance(value, ast.Constant) and isinstance(value.value, str):
            if _is_multi_chip_tpu_bounds_string(value.value):
              violations.append(
                  AntiCheatViolation(
                      rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                      file_path=file_label,
                      line=lineno,
                      col=col,
                      code_snippet=_snippet(lineno),
                      message=(
                          f"Multi-chip TPU bounds TPU_CHIPS_PER_HOST_BOUNDS={value.value!r} violates "
                          "single-chip execution (1,1,1)."
                      ),
                      remediation="Set TPU_CHIPS_PER_HOST_BOUNDS='1,1,1' and TPU_VISIBLE_DEVICES='0'.",
                  )
              )
          if key_str in _PARALLELISM_SIZE_IDENTIFIERS and value is not None:
            int_val = _extract_int_literal(value)
            if int_val is not None and int_val > 1:
              violations.append(
                  AntiCheatViolation(
                      rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                      file_path=file_label,
                      line=lineno,
                      col=col,
                      code_snippet=_snippet(lineno),
                      message=(
                          f"Multi-device config [{key_str!r}] = {int_val} violates CAP_SINGLE_CHIP "
                          "(num_devices == 1)."
                      ),
                      remediation=f"Set {key_str}=1 for single-chip execution.",
                  )
              )

      # Rule 4: Direct variable assignment `tensor_parallel_size = 2` or `CUDA_VISIBLE_DEVICES = "0,1"`
      if target_leaf in _PARALLELISM_SIZE_IDENTIFIERS and value is not None:
        int_val = _extract_int_literal(value)
        if int_val is not None and int_val > 1:
          violations.append(
              AntiCheatViolation(
                  rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                  file_path=file_label,
                  line=lineno,
                  col=col,
                  code_snippet=_snippet(lineno),
                  message=f"Setting {target_leaf}={int_val} (> 1) violates single-chip execution.",
                  remediation=f"Set {target_leaf}=1 to comply with CAP_SINGLE_CHIP.",
              )
          )
      if target_leaf in _DEVICE_VISIBILITY_ENV_VARS and isinstance(value, ast.Constant) and isinstance(value.value, str):
        if _is_multi_device_visible_string(value.value):
          violations.append(
              AntiCheatViolation(
                  rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                  file_path=file_label,
                  line=lineno,
                  col=col,
                  code_snippet=_snippet(lineno),
                  message=f"Multi-device setting {target_leaf}={value.value!r} violates single-chip execution.",
                  remediation=f"Set {target_leaf}='0' for single-device execution.",
              )
          )

    def visit_Assign(self, node: ast.Assign) -> None:
      lineno = getattr(node, "lineno", 1)
      col = getattr(node, "col_offset", 0)
      for target in node.targets:
        self._check_assignment_target_and_value(target, node.value, lineno, col)
      self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
      lineno = getattr(node, "lineno", 1)
      col = getattr(node, "col_offset", 0)
      self._check_assignment_target_and_value(node.target, node.value, lineno, col)
      self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
      for k_node, v_node in zip(node.keys, node.values):
        if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
          key_str = k_node.value
          lineno = getattr(k_node, "lineno", getattr(node, "lineno", 1))
          col = getattr(k_node, "col_offset", getattr(node, "col_offset", 0))
          if key_str in _PROTECTED_GATE_FLAGS and isinstance(v_node, ast.Constant) and v_node.value is True:
            violations.append(
                AntiCheatViolation(
                    rule_id="CHEAT_GATE_OR_RECEIPT_BYPASS",
                    file_path=file_label,
                    line=lineno,
                    col=col,
                    code_snippet=_snippet(lineno),
                    message=f"Hardcoding dict entry {key_str!r}: True in submission source is prohibited.",
                    remediation="Remove hardcoded quality gate pass flags.",
                )
            )
          if key_str in _DEVICE_VISIBILITY_ENV_VARS and isinstance(v_node, ast.Constant) and isinstance(v_node.value, str):
            if _is_multi_device_visible_string(v_node.value):
              violations.append(
                  AntiCheatViolation(
                      rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                      file_path=file_label,
                      line=lineno,
                      col=col,
                      code_snippet=_snippet(lineno),
                      message=f"Multi-device dict entry {key_str!r}: {v_node.value!r} violates single-chip execution.",
                      remediation=f"Set {key_str!r} to '0'.",
                  )
              )
          if key_str in _PARALLELISM_SIZE_IDENTIFIERS:
            int_val = _extract_int_literal(v_node)
            if int_val is not None and int_val > 1:
              violations.append(
                  AntiCheatViolation(
                      rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                      file_path=file_label,
                      line=lineno,
                      col=col,
                      code_snippet=_snippet(lineno),
                      message=f"Multi-device dict entry {key_str!r}: {int_val} violates CAP_SINGLE_CHIP.",
                      remediation=f"Set {key_str!r}: 1.",
                  )
              )
      self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
      func_name = _extract_dotted_name(node.func)
      func_leaf = func_name.split(".")[-1] if func_name else ""
      lineno = getattr(node, "lineno", 1)
      col = getattr(node, "col_offset", 0)

      # Rule 1: `.get(obs.tobytes())` / `.setdefault(obs.tobytes(), ...)` on a cache dict
      if func_leaf in ("get", "setdefault", "__getitem__", "__setitem__") and node.args:
        if _expr_contains_raw_bytes_key(node.args[0]):
          violations.append(
              AntiCheatViolation(
                  rule_id="CHEAT_RAW_INPUT_MEMOIZATION",
                  file_path=file_label,
                  line=lineno,
                  col=col,
                  code_snippet=_snippet(lineno),
                  message=f"Cache lookup via {func_name}() keyed by raw input '.tobytes()' is prohibited.",
                  remediation="Remove raw observation byte memoization; only per-session_id language KV caching is permitted.",
              )
          )

      # Rule 2 & Rule 3: `unittest.mock.patch("time.perf_counter_ns")`, `setattr(time, "perf_counter_ns", ...)`, `monkeypatch.setattr(...)`
      if func_leaf in ("patch", "object", "setattr"):
        str_args: List[str] = []
        target_names: List[str] = []
        for arg in node.args:
          if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            str_args.append(arg.value)
          else:
            dname = _extract_dotted_name(arg)
            if dname:
              target_names.append(dname)
        for kw in node.keywords:
          if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            str_args.append(kw.value.value)

        combined_tokens = str_args + target_names
        if any(
            tok in _PROTECTED_TIMER_ATTRS
            or any(tok.endswith(f".{attr}") for attr in _PROTECTED_TIMER_ATTRS)
            for tok in combined_tokens
        ):
          violations.append(
              AntiCheatViolation(
                  rule_id="CHEAT_TIMER_OVERRIDE_OR_MOCK",
                  file_path=file_label,
                  line=lineno,
                  col=col,
                  code_snippet=_snippet(lineno),
                  message=f"Mocking or patching timing function via {func_name}() is prohibited.",
                  remediation="Remove mock/patch/setattr calls targeting time.perf_counter or time.perf_counter_ns.",
              )
          )

        if any("verify_submission_receipt" in tok for tok in combined_tokens):
          violations.append(
              AntiCheatViolation(
                  rule_id="CHEAT_GATE_OR_RECEIPT_BYPASS",
                  file_path=file_label,
                  line=lineno,
                  col=col,
                  code_snippet=_snippet(lineno),
                  message=f"Mocking or patching 'verify_submission_receipt' via {func_name}() is prohibited.",
                  remediation="Remove any mock or monkeypatch targeting verify_submission_receipt.",
              )
          )

      # Rule 4: `os.putenv("CUDA_VISIBLE_DEVICES", "0,1")` or `os.environ.update({"CUDA_VISIBLE_DEVICES": "0,1"})`
      if func_leaf in ("putenv", "setdefault") and len(node.args) >= 2:
        if (
            isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
          env_key = node.args[0].value
          env_val = node.args[1].value
          if env_key in _DEVICE_VISIBILITY_ENV_VARS and _is_multi_device_visible_string(env_val):
            violations.append(
                AntiCheatViolation(
                    rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                    file_path=file_label,
                    line=lineno,
                    col=col,
                    code_snippet=_snippet(lineno),
                    message=f"Multi-device {func_name}({env_key!r}, {env_val!r}) violates single-chip rule.",
                    remediation=f"Set {env_key}='0'.",
                )
            )

      # Rule 4: Keyword arguments in function/class calls such as `LLM(..., tensor_parallel_size=2)`
      for kw in node.keywords:
        if kw.arg in _PARALLELISM_SIZE_IDENTIFIERS:
          int_val = _extract_int_literal(kw.value)
          if int_val is not None and int_val > 1:
            violations.append(
                AntiCheatViolation(
                    rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                    file_path=file_label,
                    line=lineno,
                    col=col,
                    code_snippet=_snippet(lineno),
                    message=(
                        f"Call {func_name}(..., {kw.arg}={int_val}) requests {int_val} devices, "
                        "violating CAP_SINGLE_CHIP (num_devices == 1)."
                    ),
                    remediation=f"Pass {kw.arg}=1 for single-chip execution.",
                )
            )

      # Rule 5: Blocking device-to-host sync (`.item()`, `.cpu()`, `.numpy()`) inside a `for`/`while` loop
      if (
          self.loop_depth > 0
          and isinstance(node.func, ast.Attribute)
          and node.func.attr in ("item", "cpu", "numpy")
          and not node.args
          and not node.keywords
      ):
        violations.append(
            AntiCheatViolation(
                rule_id="CHEAT_HOT_LOOP_DEVICE_SYNC",
                file_path=file_label,
                line=lineno,
                col=col,
                code_snippet=_snippet(lineno),
                message=(
                    f"Blocking device-to-host synchronization '.{node.func.attr}()' detected inside "
                    "a hot 'for'/'while' loop."
                ),
                remediation=(
                    "Move '.item()', '.cpu()', and '.numpy()' host transfers outside the hot inference/rollout loop."
                ),
            )
        )

      self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
      # Check non-docstring string constants in executable code (e.g. subprocess commands `"CUDA_VISIBLE_DEVICES=0,1 python ..."`)
      if isinstance(node.value, str):
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        for match in _MULTI_DEVICE_ENV_ASSIGN_RE.finditer(node.value):
          var_name = match.group("var")
          val_str = match.group("val")
          if _is_multi_device_visible_string(val_str):
            violations.append(
                AntiCheatViolation(
                    rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                    file_path=file_label,
                    line=lineno,
                    col=col,
                    code_snippet=_snippet(lineno),
                    message=(
                        f"Executable string constant sets {var_name}={val_str!r} across multiple devices, "
                        "violating CAP_SINGLE_CHIP."
                    ),
                    remediation=f"Set {var_name}=0 (single device).",
                )
            )
        for match in _TPU_BOUNDS_ASSIGN_RE.finditer(node.value):
          val_str = match.group("val")
          if _is_multi_chip_tpu_bounds_string(val_str):
            violations.append(
                AntiCheatViolation(
                    rule_id="CHEAT_MULTI_DEVICE_BYPASS",
                    file_path=file_label,
                    line=lineno,
                    col=col,
                    code_snippet=_snippet(lineno),
                    message=(
                        f"Executable string constant sets TPU_CHIPS_PER_HOST_BOUNDS={val_str!r}, "
                        "violating single-chip execution."
                    ),
                    remediation="Set TPU_CHIPS_PER_HOST_BOUNDS=1,1,1.",
                )
            )

  visitor = _AntiCheatVisitor()
  visitor.visit(tree)

  # Deduplicate violations by (rule_id, file_path, line)
  deduped: List[AntiCheatViolation] = []
  seen: Set[Tuple[str, str, int]] = set()
  for v in violations:
    key = (v.rule_id, v.file_path, v.line)
    if key not in seen:
      seen.add(key)
      deduped.append(v)

  return deduped


def audit_submission_python_sources(
    target: Optional[str | Path | Dict[str, str]] = None,
    *,
    source_code: Optional[str] = None,
) -> SourceAuditResult:
  """Audit Python sources in a submission folder, single `.py` file, or inline source dictionary/string (`Gate 4`).

  Detects prohibited benchmark cheats:
    1. `CHEAT_RAW_INPUT_MEMOIZATION`: Hot-loop output memoization / lookup table keyed by raw input bytes (`_memo_cache[obs.tobytes()]`).
    2. `CHEAT_TIMER_OVERRIDE_OR_MOCK`: Overriding or mocking `time.perf_counter` / `time.perf_counter_ns`.
    3. `CHEAT_GATE_OR_RECEIPT_BYPASS`: Hardcoding `gate_a_passed = True` or bypassing `verify_submission_receipt`.
    4. `CHEAT_MULTI_DEVICE_BYPASS`: Multi-device environment bypass (`CUDA_VISIBLE_DEVICES="0,1"`, `tensor_parallel_size=2`).

  Negative-Control Immunity (`Gate 4`):
    - Stays 100% silent on reading `time.perf_counter_ns()` (`t0 = time.perf_counter_ns()`),
    - Stays 100% silent on per-`session_id` VLM language-prefix KV caching (`session_kv_cache[session_id] = encode_prompt(instruction)`),
    - Stays 100% silent on comments (`# ...`) and docstrings (`\"\"\"...\"\"\"`) mentioning prohibited patterns,
    - Stays 100% silent on single-device `CUDA_VISIBLE_DEVICES="0"` or `tensor_parallel_size=1`.
  """
  sources_to_scan: List[Tuple[str, str]] = []

  if source_code is not None:
    label = str(target) if target is not None else "<inline_source.py>"
    sources_to_scan.append((label, source_code))
  elif isinstance(target, dict):
    for fname, code in sorted(target.items()):
      sources_to_scan.append((str(fname), str(code)))
  elif isinstance(target, (str, Path)):
    path_obj = Path(target)
    if path_obj.is_symlink():
      all_violations_pre = [
          AntiCheatViolation(
              rule_id="SYMLINK_PROHIBITED",
              file_path=str(path_obj),
              line=1,
              col=0,
              code_snippet=str(path_obj),
              message=f"Symbolic link {path_obj} is prohibited in submission packages (RT-07).",
              remediation="Replace symbolic links with self-contained regular files inside the submission bundle.",
          )
      ]
      return SourceAuditResult(
          passed=False,
          files_scanned=[],
          violations=all_violations_pre,
          reason=all_violations_pre[0].message,
      )
    if path_obj.is_file():
      if path_obj.suffix.lower() == ".py":
        sources_to_scan.append((str(path_obj), path_obj.read_text(encoding="utf-8")))
    elif path_obj.is_dir():
      resolved_root = path_obj.resolve()
      py_files: List[Path] = []
      for dirpath, dirnames, filenames in os.walk(path_obj, followlinks=False):
        cur = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not (cur / d).is_symlink())
        for fn in sorted(filenames):
          cand = cur / fn
          if cand.is_symlink() or not cand.is_file() or cand.suffix.lower() != ".py":
            continue
          cand.resolve().relative_to(resolved_root)
          py_files.append(cand)
      for py_file in py_files:
        sources_to_scan.append((str(py_file), py_file.read_text(encoding="utf-8")))
    else:
      # If someone passed inline Python source string directly as the first positional argument
      if "\n" in str(target) or "=" in str(target) or "import " in str(target):
        sources_to_scan.append(("<inline_source.py>", str(target)))
      else:
        raise FileNotFoundError(
            f"Submission source target not found at {path_obj}. "
            "Pass an existing submission directory, .py file path, or source_code='...'."
        )
  else:
    raise ValueError(
        "audit_submission_python_sources requires either target=<path_or_dict> or source_code='<python_code>'."
    )

  all_violations: List[AntiCheatViolation] = []
  scanned_labels: List[str] = []
  for file_label, text in sources_to_scan:
    scanned_labels.append(file_label)
    all_violations.extend(_audit_single_python_source(text, file_label))

  passed = len(all_violations) == 0
  if passed:
    reason = (
        f"Static AST anti-cheat audit passed across {len(scanned_labels)} Python file(s) "
        "with 0 violations."
    )
  else:
    formatted = "; ".join(
        f"[{v.rule_id}] {Path(v.file_path).name}:{v.line} ({v.message} Remediation: {v.remediation})"
        for v in all_violations
    )
    reason = f"Static AST anti-cheat audit failed with {len(all_violations)} violation(s): {formatted}"

  return SourceAuditResult(
      passed=passed,
      files_scanned=scanned_labels,
      violations=all_violations,
      reason=reason,
  )


def check_single_chip_hardware_eligibility(
    recipe_or_sku: RecipeConfig | str,
    envelope_id: str | PerformanceCapId = PerformanceCapId.CAP_SINGLE_CHIP,
) -> HardwareEnvelopeCheckResult:
  """Check hardware SKU and recipe compliance against the single-chip envelope (`CAP_SINGLE_CHIP`).

  Verifies:
    - `num_devices == 1`
    - `vram_gb <= 64.0`
    - `fp16_tflops <= 1000.0`
    - `fp8_tflops <= 2000.0`
    - `mem_bw_gbps <= 2000.0`
  """
  resolved_env_id = resolve_perf_cap_id(envelope_id)
  if resolved_env_id not in PERFORMANCE_CAP_ENVELOPES:
    valid_envs = ", ".join(sorted(PERFORMANCE_CAP_ENVELOPES.keys()))
    raise ValueError(
        f"Unknown PerformanceCapEnvelope {envelope_id!r}. Supported envelopes: [{valid_envs}]."
    )
  cap = PERFORMANCE_CAP_ENVELOPES[resolved_env_id]

  if isinstance(recipe_or_sku, RecipeConfig):
    sku = recipe_or_sku.hardware_sku
    recipe_num_devices = int(recipe_or_sku.num_devices)
  else:
    sku = str(recipe_or_sku)
    recipe_num_devices = 1

  if sku not in HARDWARE_SKU_SPECS:
    valid_skus = ", ".join(sorted(HARDWARE_SKU_SPECS.keys()))
    raise ValueError(
        f"Unknown hardware_sku={sku!r}. Supported SKUs in HARDWARE_SKU_SPECS: [{valid_skus}]."
    )

  sku_spec = HARDWARE_SKU_SPECS[sku]
  sku_num_devices = int(sku_spec.get("num_devices", 1))
  effective_num_devices = max(recipe_num_devices, sku_num_devices)
  vram_gb = round(float(sku_spec["vram_mb"]) / 1024.0, 4)
  fp16_tflops = float(sku_spec["fp16_tflops"])
  fp8_tflops = float(sku_spec.get("fp8_tflops", fp16_tflops * 2.0))
  mem_bw_gbps = float(sku_spec["mem_bw_gbps"])
  vendor = str(sku_spec.get("vendor", "UNKNOWN"))

  violations: List[str] = []
  if effective_num_devices != 1 or effective_num_devices > cap.max_num_devices:
    violations.append(
        f"num_devices={effective_num_devices} exceeds single-chip limit (max_num_devices={cap.max_num_devices})."
    )
  if vram_gb > cap.max_vram_gb + 1e-6:
    violations.append(
        f"hardware_sku={sku!r} vram_gb={vram_gb:.1f} GB exceeds {cap.envelope_id} max_vram_gb={cap.max_vram_gb:.1f} GB."
    )
  if fp16_tflops > cap.max_fp16_tflops + 1e-6:
    violations.append(
        f"hardware_sku={sku!r} fp16_tflops={fp16_tflops:.1f} TFLOPS exceeds {cap.envelope_id} max_fp16_tflops={cap.max_fp16_tflops:.1f} TFLOPS."
    )
  if fp8_tflops > cap.max_int8_fp8_tops + 1e-6:
    violations.append(
        f"hardware_sku={sku!r} fp8_tflops={fp8_tflops:.1f} TOPS exceeds {cap.envelope_id} max_int8_fp8_tops={cap.max_int8_fp8_tops:.1f} TOPS."
    )
  if mem_bw_gbps > cap.max_mem_bw_gbps + 1e-6:
    violations.append(
        f"hardware_sku={sku!r} mem_bw_gbps={mem_bw_gbps:.1f} GB/s exceeds {cap.envelope_id} max_mem_bw_gbps={cap.max_mem_bw_gbps:.1f} GB/s."
    )

  passed = len(violations) == 0
  limits = {
      "max_num_devices": float(cap.max_num_devices),
      "max_vram_gb": float(cap.max_vram_gb),
      "max_fp16_tflops": float(cap.max_fp16_tflops),
      "max_fp8_tflops": float(cap.max_int8_fp8_tops),
      "max_mem_bw_gbps": float(cap.max_mem_bw_gbps),
  }
  if passed:
    reason = (
        f"Hardware SKU {sku!r} ({vendor}) natively satisfies {cap.envelope_id} "
        f"(num_devices={effective_num_devices}, vram={vram_gb:.1f}/{cap.max_vram_gb:.1f}GB, "
        f"fp16={fp16_tflops:.1f}/{cap.max_fp16_tflops:.1f}TFLOPS, "
        f"fp8={fp8_tflops:.1f}/{cap.max_int8_fp8_tops:.1f}TOPS, "
        f"bw={mem_bw_gbps:.1f}/{cap.max_mem_bw_gbps:.1f}GB/s)."
    )
  else:
    reason = (
        f"Hardware SKU {sku!r} failed {cap.envelope_id} eligibility check: "
        + " ".join(violations)
    )

  return HardwareEnvelopeCheckResult(
      passed=passed,
      envelope_id=cap.envelope_id,
      hardware_sku=sku,
      vendor=vendor,
      num_devices=effective_num_devices,
      vram_gb=vram_gb,
      fp16_tflops=fp16_tflops,
      fp8_tflops=fp8_tflops,
      mem_bw_gbps=mem_bw_gbps,
      limits=limits,
      violations=violations,
      reason=reason,
  )


_NON_RECIPE_SUBMISSION_ARTIFACTS: frozenset[str] = frozenset({
    "report.json",
    "receipt.json",
    "submission_receipt.json",
    "bundle_summary.json",
    "submission_meta.json",
    "sweep_results.json",
    "manifest.json",
    "dataset_manifest.json",
    "perturbation_manifest.json",
    "probe_result.json",
})


def _resolve_submission_and_recipe_path(
    submission_path: str | Path,
    *,
    allow_standalone_measured_bundle: bool = False,
) -> Tuple[Path, Optional[Path], str]:
  """Resolve `(submission_dir, recipe_file_path, submission_id)` from a directory or recipe file path."""
  p = Path(submission_path)
  if not p.exists():
    raise FileNotFoundError(
        f"Submission path {p} does not exist. "
        "Pass a valid submission directory ('submissions/<submission_id>') or recipe.yaml path."
    )
  if p.is_file():
    if allow_standalone_measured_bundle and p.name.lower() in {
        "receipt.json",
        "submission_receipt.json",
        "bundle_summary.json",
        "submission_meta.json",
        "report.json",
        "sweep_results.json",
    }:
      return p.parent, None, p.parent.name or p.stem
    return p.parent, p, p.parent.name or p.stem

  for candidate_name in ("recipe.yaml", "recipe.yml", "recipe.json"):
    candidate = p / candidate_name
    if candidate.is_file():
      return p, candidate, p.name

  yaml_files = sorted(
      f
      for f in p.iterdir()
      if f.is_file()
      and f.suffix.lower() in (".yaml", ".yml", ".json")
      and not f.name.startswith(".")
      and f.name.lower() not in _NON_RECIPE_SUBMISSION_ARTIFACTS
      and not f.name.lower().endswith("_receipt.json")
      and not f.name.lower().endswith("_report.json")
  )
  if yaml_files:
    return p, yaml_files[0], p.name

  measured_bundle_files = sorted(
      f.name
      for f in p.iterdir()
      if f.is_file()
      and f.name.lower()
      in {
          "receipt.json",
          "submission_receipt.json",
          "bundle_summary.json",
          "submission_meta.json",
          "report.json",
          "sweep_results.json",
      }
  )
  if allow_standalone_measured_bundle and measured_bundle_files:
    return p, None, p.name

  present_artifacts = sorted(
      f.name for f in p.iterdir() if f.is_file() and f.name.lower() in _NON_RECIPE_SUBMISSION_ARTIFACTS
  )
  hint = (
      f" (found non-recipe bundle files {present_artifacts}; re-run 'eeb measure --recipe <recipe.yaml> --out {p}', "
      "supply 'bundle_summary.json' + 'submission_meta.json' for Phase 1 Commit-Reveal, or copy 'recipe.yaml' into the directory)"
      if present_artifacts
      else ""
  )
  raise FileNotFoundError(
      f"No 'recipe.yaml' (or 'recipe.json') found inside submission directory {p}{hint}. "
      "Every submission folder under 'submissions/<track>/<submission_id>/' must include either a declarative 'recipe.yaml'/'recipe.json', "
      "a Phase 1 Commit-Reveal 'bundle_summary.json' + 'submission_meta.json', or a standalone measured 'report.json' + 'receipt.json' bundle."
  )


def _validate_standalone_measured_bundle(
    sub_dir: Path,
    sub_id: str,
    *,
    registry_path: Optional[str | Path] = None,
    require_single_chip_cap: bool = True,
) -> SubmissionValidationReport:
  """Validate a standalone real-hardware submission folder (`report.json` + `receipt.json` or Phase-1 `bundle_summary.json` + `submission_meta.json`) without `recipe.yaml`."""
  from embodied_bench.leaderboard_compiler import (
      _audit_single_receipt,
      _canonicalize_model_id,
      _canonicalize_track,
  )

  violations: List[str] = []
  violations.extend(_audit_directory_symlinks_and_containment(sub_dir))
  sub_slug_err = _audit_untrusted_slug("submission_id", sub_id)
  if sub_slug_err:
    violations.append(sub_slug_err)
  bundle_summary_path = sub_dir / "bundle_summary.json"
  has_bundle_summary = bundle_summary_path.is_file() and not bundle_summary_path.is_symlink()
  meta_path = sub_dir / "submission_meta.json"
  has_meta = meta_path.is_file() and not meta_path.is_symlink()

  receipt_candidates = [
      sub_dir / "receipt.json",
      sub_dir / "submission_receipt.json",
      bundle_summary_path,
  ]
  receipt_path = next((c for c in receipt_candidates if c.is_file() and not c.is_symlink()), None)
  measured_report_files = [
      c for c in (sub_dir / "report.json", sub_dir / "sweep_results.json") if c.is_file() and not c.is_symlink()
  ]
  report_path = measured_report_files[0] if measured_report_files else None
  is_phase1_commit_reveal = bool(has_bundle_summary and has_meta and report_path is None)

  normalized_meta: Dict[str, Any] = {}
  if has_meta:
    try:
      loaded_meta = load_yaml_or_json(meta_path)
      meta_ok, meta_errs, normalized_meta = validate_submission_meta(
          loaded_meta,
          require_commit_hash=(report_path is None),
      )
      if not meta_ok:
        violations.extend(meta_errs)
    except Exception as exc:
      violations.append(f"Failed to parse {meta_path}: {exc}")

  sweep_data: Dict[str, Any] = {}
  if report_path is not None:
    try:
      loaded_report = load_yaml_or_json(report_path)
      if isinstance(loaded_report, dict):
        sweep_data = loaded_report
      else:
        violations.append(f"Report payload at {report_path} must be a JSON object.")
    except Exception as exc:
      violations.append(f"Failed to parse {report_path}: {exc}")

  receipt_verified: Optional[bool] = None
  receipt_message: Optional[str] = None
  ctx: Dict[str, Any] = {}
  receipt: Dict[str, Any] = {}
  body: Dict[str, Any] = {}

  if receipt_path is None:
    receipt_verified = False
    names = ", ".join(f.name for f in measured_report_files) or ("submission_meta.json" if has_meta else "report.json")
    receipt_message = (
        f"Submission folder contains [{names}] without a cryptographic "
        "'receipt.json' or Phase 1 'bundle_summary.json'. Regenerate the bundle with 'eeb measure'."
    )
    violations.append(receipt_message)
  elif report_path is None and not is_phase1_commit_reveal:
    receipt_verified, receipt_message = verify_submission_receipt(receipt_path)
    if not receipt_verified:
      violations.append(receipt_message or f"Receipt verification failed for {receipt_path}.")
    if has_bundle_summary and not has_meta:
      violations.append(
          f"Phase 1 Commit-Reveal submission folder {sub_dir} has 'bundle_summary.json' without 'report.json'/'recipe.yaml' "
          "and is missing required 'submission_meta.json' (with 'hardware_sku', 'division', and 64-hex 'recipe_commit_sha256')."
      )
    else:
      violations.append(
          f"Standalone measured submission folder {sub_dir} has no 'recipe.yaml'/'recipe.json' "
          "and is missing 'report.json' (or 'sweep_results.json'); both 'report.json' and 'receipt.json' are required "
          "(or provide Phase 1 'bundle_summary.json' + 'submission_meta.json' with 'recipe_commit_sha256')."
      )
    try:
      loaded_receipt = load_yaml_or_json(receipt_path)
      if isinstance(loaded_receipt, dict):
        receipt = loaded_receipt
        if isinstance(receipt.get("attested_body"), dict):
          body = receipt["attested_body"]
    except Exception:
      pass
  else:
    receipt_verified, receipt_message = verify_submission_receipt(receipt_path)
    audit_ok, audit_violations, ctx = _audit_single_receipt(receipt_path)
    if isinstance(ctx.get("receipt"), dict):
      receipt = ctx["receipt"]
    if isinstance(ctx.get("body"), dict):
      body = ctx["body"]
    if isinstance(ctx.get("sweep_data"), dict) and ctx["sweep_data"]:
      sweep_data = ctx["sweep_data"]
    if not audit_ok:
      for v in audit_violations:
        if v not in violations:
          violations.append(v)

  schema_valid = bool((report_path is not None or is_phase1_commit_reveal) and (receipt or sweep_data))
  division_compliant = True

  caps = sweep_data.get("caps") if isinstance(sweep_data.get("caps"), dict) else {}
  env_check_raw = sweep_data.get("envelope_check") if isinstance(sweep_data.get("envelope_check"), dict) else {}

  recipe_id = str(
      ctx.get("recipe_id")
      or receipt.get("recipe_id")
      or body.get("recipe_id")
      or sweep_data.get("recipe_id")
      or sub_id
  )
  raw_model_id = str(
      body.get("baseline_model_id")
      or receipt.get("baseline_model_id")
      or sweep_data.get("baseline_model_id")
      or body.get("model_id")
      or caps.get("model_id")
      or ""
  )
  baseline_model_id = _canonicalize_model_id(raw_model_id) if raw_model_id else ""
  model_name = str(
      body.get("model_name")
      or receipt.get("model_name")
      or sweep_data.get("model_name")
      or baseline_model_id
  )
  raw_track = str(
      body.get("track")
      or receipt.get("track")
      or sweep_data.get("track")
      or caps.get("track")
      or ""
  )
  track = _canonicalize_track(raw_track, baseline_model_id)
  division = str(
      receipt.get("division")
      or body.get("division")
      or sweep_data.get("division")
      or normalized_meta.get("division")
      or DivisionType.CLOSED.value
  ).upper()
  clock = str(
      receipt.get("clock")
      or body.get("clock")
      or sweep_data.get("clock")
      or "corl2026_sprint"
  )
  scenario = str(receipt.get("scenario") or sweep_data.get("scenario") or "P1_ONBOARD")
  precision = str(
      body.get("precision")
      or receipt.get("precision")
      or caps.get("precision")
      or "BF16"
  ).upper()
  hardware_sku = str(
      ctx.get("hardware_sku")
      or env_check_raw.get("hardware_sku")
      or body.get("hardware_sku")
      or receipt.get("hardware_sku")
      or sweep_data.get("hardware_sku")
      or normalized_meta.get("hardware_sku")
      or ""
  )
  if normalized_meta:
    meta_sku = normalized_meta.get("hardware_sku")
    if meta_sku and hardware_sku and meta_sku != hardware_sku:
      msg = (
          f"submission_meta.json hardware_sku={meta_sku!r} does not match signed bundle "
          f"hardware_sku={hardware_sku!r}."
      )
      if msg not in violations:
        violations.append(msg)
    meta_div = normalized_meta.get("division")
    if meta_div and division and meta_div != division:
      msg = (
          f"submission_meta.json division={meta_div!r} does not match signed bundle "
          f"division={division!r}."
      )
      if msg not in violations:
        violations.append(msg)
  perf_cap = resolve_perf_cap_id(
      str(receipt.get("perf_cap") or sweep_data.get("perf_cap") or "CAP_SINGLE_CHIP")
  )
  raw_num_devices = ctx.get(
      "num_devices",
      receipt.get("num_devices", sweep_data.get("num_devices", env_check_raw.get("num_devices", 1))),
  )
  num_devices = int(raw_num_devices) if raw_num_devices is not None else 1

  for src_obj in (receipt, body, sweep_data):
    for par_key in _PARALLELISM_SIZE_IDENTIFIERS:
      if par_key in src_obj and isinstance(src_obj[par_key], (int, float)) and int(src_obj[par_key]) > 1:
        msg = f"Bundle sets {par_key}={src_obj[par_key]} (> 1), violating single-chip execution."
        if msg not in violations:
          violations.append(msg)

  is_canonical = (
      baseline_model_id in CANONICAL_CHALLENGE_MODELS
      or bool(MODERN_REFERENCE_CHECKPOINTS.get(baseline_model_id, {}).get("canonical_challenge_model", False))
  )
  if division == DivisionType.CLOSED.value and not is_canonical:
    division_compliant = False
    violations.append(
        f"Division compliance check failed: standalone CLOSED bundle has non-canonical "
        f"baseline_model_id={raw_model_id!r} (canonicalized={baseline_model_id!r})."
    )

  target_envelope = (
      PerformanceCapId.CAP_SINGLE_CHIP.value if require_single_chip_cap else perf_cap
  )
  try:
    hw_check = check_single_chip_hardware_eligibility(hardware_sku, envelope_id=target_envelope)
    if num_devices != 1 and hw_check.passed:
      hw_check.passed = False
      hw_check.num_devices = num_devices
      hw_check.violations.append(
          f"num_devices={num_devices} violates single-chip execution (num_devices == 1)."
      )
      hw_check.reason = " ".join(hw_check.violations)
  except ValueError as exc:
    hw_check = HardwareEnvelopeCheckResult(
        passed=False,
        envelope_id=str(target_envelope),
        hardware_sku=hardware_sku,
        vendor="UNKNOWN",
        num_devices=num_devices,
        vram_gb=0.0,
        fp16_tflops=0.0,
        fp8_tflops=0.0,
        mem_bw_gbps=0.0,
        limits={},
        violations=[str(exc)],
        reason=str(exc),
    )

  if not hw_check.passed:
    for hv in hw_check.violations:
      if hv not in violations:
        violations.append(hv)

  for slug_field, slug_val in (
      ("recipe_id", recipe_id),
      ("hardware_sku", hardware_sku),
      ("baseline_model_id", raw_model_id or baseline_model_id),
      ("team", ctx.get("submitter_team") or normalized_meta.get("team") or body.get("team") or receipt.get("team")),
      ("engine_tag", ctx.get("engine_tag") or normalized_meta.get("engine_tag") or body.get("engine_tag") or receipt.get("engine_tag")),
  ):
    slug_err = _audit_untrusted_slug(slug_field, slug_val)
    if slug_err and slug_err not in violations:
      violations.append(slug_err)

  if is_phase1_commit_reveal:
    allowed_p1_files = {"bundle_summary.json", "submission_meta.json", "README.md"}
    extra_p1_files = sorted(
        f.relative_to(sub_dir).as_posix()
        for f in sub_dir.rglob("*")
        if f.is_file() and f.name not in allowed_p1_files and not f.name.startswith(".")
    )
    if extra_p1_files:
      violations.append(
          f"Phase 1 Commit-Reveal bundle contains unauthorized non-commit file(s) {extra_p1_files} "
          f"(only {sorted(allowed_p1_files)} permitted in Phase 1 PRs to prevent leaking private server/kernel source before Phase 2 reveal)."
      )

  source_audit = audit_submission_python_sources(sub_dir)
  if not source_audit.passed:
    violations.append(source_audit.reason)

  passed = (
      schema_valid
      and division_compliant
      and hw_check.passed
      and source_audit.passed
      and (receipt_verified is True)
      and len(violations) == 0
  )
  status = RunStatus.CERTIFIED.value if passed else RunStatus.REJECTED.value
  reason = None if passed else "; ".join(violations)
  validate_status_and_reason(status=status, reason=reason)

  effective_recipe_path = str(report_path or receipt_path or sub_dir)
  sub_phase = str(
      ctx.get("submission_phase")
      or normalized_meta.get("submission_phase")
      or ("COMMIT_ONLY" if is_phase1_commit_reveal else "STANDALONE_BUNDLE")
  )
  val_scope = "PHASE1_COMMIT_REVEAL_BUNDLE" if is_phase1_commit_reveal else "FULL_MEASURED_BUNDLE"
  report = SubmissionValidationReport(
      submission_id=sub_id,
      submission_path=str(sub_dir),
      recipe_path=effective_recipe_path,
      recipe_id=recipe_id,
      track=track,
      division=division,
      clock=clock,
      scenario=scenario,
      model_name=model_name,
      baseline_model_id=baseline_model_id,
      canonical_challenge_model=is_canonical,
      precision=precision,
      hardware_sku=hardware_sku,
      perf_cap=perf_cap,
      num_devices=num_devices,
      passed=passed,
      status=status,
      reason=reason,
      schema_valid=schema_valid,
      division_compliant=division_compliant,
      hardware_envelope_check=hw_check.to_dict(),
      source_audit=source_audit.to_dict(),
      has_measurements=True,
      validation_scope=val_scope,
      receipt_verified=receipt_verified,
      receipt_message=receipt_message,
      submission_phase=sub_phase,
      recipe_commit_sha256=(
          ctx.get("recipe_commit_sha256") or normalized_meta.get("recipe_commit_sha256")
      ),
      engine_tag=ctx.get("engine_tag") or normalized_meta.get("engine_tag"),
      submitter_team=ctx.get("submitter_team") or normalized_meta.get("team"),
      violations=violations,
  )

  if registry_path is not None:
    record_submission_validation(
        registry_path=registry_path,
        submission_id=sub_id,
        status=status,
        reason=reason,
        recipe_id=recipe_id,
        hardware_sku=hardware_sku,
        division=division,
        track=track,
        violations=violations if violations else None,
    )

  return report


def validate_submission_directory(
    submission_path: str | Path,
    *,
    registry_path: Optional[str | Path] = None,
    require_single_chip_cap: bool = True,
) -> SubmissionValidationReport:
  """Validate a 3rd-party PR submission directory (`submissions/<submission_id>/`) or `recipe.yaml` file.

  Performs:
    1. `RecipeConfig.from_dict(raw)` schema validation & `CLOSED` vs `OPEN` division compliance
       (or standalone `report.json` + cryptographic `receipt.json` validation for ad-hoc real-hardware bundles).
    2. Single-chip hardware envelope eligibility (`CAP_SINGLE_CHIP`: `num_devices == 1`, `vram_gb <= 64.0`,
       `fp16_tflops <= 1000.0`, `fp8_tflops <= 2000.0`, `mem_bw_gbps <= 2000.0`) against `HARDWARE_SKU_SPECS`.
    3. Static AST / source anti-cheat audit (`audit_submission_python_sources`) across all `.py` files in the submission folder.
    4. Optional cryptographic `submission_receipt.json` verification if present in the submission folder.
    5. Optional persistence into `registry_path` via `record_submission_validation`.
  """
  sub_dir, recipe_file, sub_id = _resolve_submission_and_recipe_path(
      submission_path, allow_standalone_measured_bundle=True
  )
  if recipe_file is None:
    return _validate_standalone_measured_bundle(
        sub_dir,
        sub_id,
        registry_path=registry_path,
        require_single_chip_cap=require_single_chip_cap,
    )
  symlink_errs = _audit_directory_symlinks_and_containment(sub_dir)
  if recipe_file.is_symlink():
    symlink_errs.append(f"Recipe file {recipe_file} is a symbolic link (RT-07).")
  else:
    try:
      recipe_file.resolve().relative_to(sub_dir.resolve())
    except ValueError:
      symlink_errs.append(f"Recipe file {recipe_file} escapes submission directory {sub_dir} (RT-07).")
  raw = load_yaml_or_json(recipe_file) if not recipe_file.is_symlink() else {}

  violations: List[str] = list(symlink_errs)
  sub_slug_err = _audit_untrusted_slug("submission_id", sub_id)
  if sub_slug_err:
    violations.append(sub_slug_err)
  schema_valid = True
  division_compliant = True
  recipe_cfg: Optional[RecipeConfig] = None

  # Check raw multi-device keys before dataclass filtering
  for par_key in _PARALLELISM_SIZE_IDENTIFIERS:
    if par_key in raw and isinstance(raw[par_key], (int, float)) and int(raw[par_key]) > 1:
      violations.append(
          f"Recipe sets {par_key}={raw[par_key]} (> 1), violating single-chip execution."
      )

  try:
    recipe_cfg = RecipeConfig.from_dict(raw)
  except ValueError as exc:
    msg = str(exc)
    if "Division compliance check failed" in msg:
      division_compliant = False
    else:
      schema_valid = False
    violations.append(msg)

  recipe_id = str(raw.get("recipe_id", sub_id))
  track = str(raw.get("track", "VLA_TRACK"))
  division = str(raw.get("division", "OPEN"))
  clock = str(raw.get("clock", "corl2026_sprint"))
  scenario = recipe_cfg.resolved_scenario() if recipe_cfg else str(raw.get("scenario", "P1_ONBOARD"))
  model_name = str(raw.get("model_name", ""))
  baseline_model_id = str(raw.get("baseline_model_id", ""))
  precision = str(raw.get("precision", "FP8"))
  hardware_sku = str(raw.get("hardware_sku", ""))
  perf_cap = recipe_cfg.resolved_perf_cap() if recipe_cfg else resolve_perf_cap_id(str(raw.get("perf_cap", "CAP_SINGLE_CHIP")))
  num_devices = int(raw.get("num_devices", 1))

  is_canonical = (
      baseline_model_id in CANONICAL_CHALLENGE_MODELS
      or bool(MODERN_REFERENCE_CHECKPOINTS.get(baseline_model_id, {}).get("canonical_challenge_model", False))
  )

  # Hardware envelope check
  target_envelope = (
      PerformanceCapId.CAP_SINGLE_CHIP.value
      if require_single_chip_cap
      else perf_cap
  )
  try:
    hw_check = check_single_chip_hardware_eligibility(
        recipe_cfg if recipe_cfg is not None else hardware_sku,
        envelope_id=target_envelope,
    )
    if int(raw.get("num_devices", 1)) != 1 and hw_check.passed:
      hw_check.passed = False
      hw_check.num_devices = int(raw.get("num_devices", 1))
      hw_check.violations.append(
          f"num_devices={raw.get('num_devices')} violates single-chip execution (num_devices == 1)."
      )
      hw_check.reason = " ".join(hw_check.violations)
  except ValueError as exc:
    hw_check = HardwareEnvelopeCheckResult(
        passed=False,
        envelope_id=str(target_envelope),
        hardware_sku=hardware_sku,
        vendor="UNKNOWN",
        num_devices=num_devices,
        vram_gb=0.0,
        fp16_tflops=0.0,
        fp8_tflops=0.0,
        mem_bw_gbps=0.0,
        limits={},
        violations=[str(exc)],
        reason=str(exc),
    )

  if not hw_check.passed:
    violations.extend(hw_check.violations)

  # Static AST / Source Anti-Cheat Audit across Python files in submission directory
  source_audit = audit_submission_python_sources(sub_dir)
  if not source_audit.passed:
    violations.append(source_audit.reason)

  # Optional submission_meta.json check when paired with open recipe.yaml
  meta_path = sub_dir / "submission_meta.json"
  normalized_meta: Dict[str, Any] = {}
  if meta_path.is_file():
    try:
      loaded_meta = load_yaml_or_json(meta_path)
      meta_ok, meta_errs, normalized_meta = validate_submission_meta(
          loaded_meta, require_commit_hash=False
      )
      if not meta_ok:
        violations.extend(meta_errs)
      else:
        if normalized_meta.get("hardware_sku") and hardware_sku and normalized_meta["hardware_sku"] != hardware_sku:
          violations.append(
              f"submission_meta.json hardware_sku={normalized_meta['hardware_sku']!r} != recipe hardware_sku={hardware_sku!r}."
          )
        if normalized_meta.get("division") and division and normalized_meta["division"] != division.upper():
          violations.append(
              f"submission_meta.json division={normalized_meta['division']!r} != recipe division={division!r}."
          )
    except Exception as exc:
      violations.append(f"Failed to parse {meta_path}: {exc}")

  for slug_field, slug_val in (
      ("recipe_id", recipe_id),
      ("hardware_sku", hardware_sku),
      ("baseline_model_id", baseline_model_id),
      ("team", normalized_meta.get("team")),
      ("engine_tag", normalized_meta.get("engine_tag")),
  ):
    slug_err = _audit_untrusted_slug(slug_field, slug_val)
    if slug_err and slug_err not in violations:
      violations.append(slug_err)

  # Receipt & measured-bundle integrity audit (supports receipt.json, submission_receipt.json, and bundle_summary.json)
  receipt_candidates = [
      sub_dir / "receipt.json",
      sub_dir / "submission_receipt.json",
      sub_dir / "bundle_summary.json",
  ]
  receipt_path = next((c for c in receipt_candidates if c.is_file() and not c.is_symlink()), None)
  measured_report_files = [
      c for c in (sub_dir / "report.json", sub_dir / "sweep_results.json") if c.is_file() and not c.is_symlink()
  ]
  has_measurements = bool(receipt_path is not None or measured_report_files)
  validation_scope = (
      "FULL_MEASURED_BUNDLE" if has_measurements else "RECIPE_ONLY_NO_MEASUREMENTS"
  )
  receipt_verified: Optional[bool] = None
  receipt_message: Optional[str] = None

  if receipt_path is not None:
    receipt_verified, receipt_message = verify_submission_receipt(receipt_path)
    if not receipt_verified:
      violations.append(receipt_message or f"Receipt verification failed for {receipt_path}.")
    else:
      try:
        rcpt_data = load_yaml_or_json(receipt_path)
      except Exception as exc:
        receipt_verified = False
        receipt_message = f"Failed to parse {receipt_path}: {exc}"
        violations.append(receipt_message)
        rcpt_data = {}
      rcpt_body = rcpt_data.get("attested_body") if isinstance(rcpt_data.get("attested_body"), dict) else {}
      rcpt_status = str(rcpt_body.get("status") or rcpt_data.get("status") or "")
      rcpt_reason = rcpt_body.get("reason") or rcpt_data.get("reason")
      rcpt_num_dev = rcpt_data.get("num_devices")
      if rcpt_status not in (RunStatus.CERTIFIED.value, RunStatus.PENDING.value):
        violations.append(
            f"Measured bundle receipt has status={rcpt_status!r} (not admissible for leaderboard submission; "
            f"requires CERTIFIED or PENDING single-chip hardware run): {rcpt_reason}"
        )
      if rcpt_num_dev is not None and int(rcpt_num_dev) != 1:
        violations.append(
            f"Measured bundle receipt recorded num_devices={rcpt_num_dev} != 1 "
            "(CPU-only --allow-no-device runs have num_devices=0 and multi-GPU runs are prohibited under CAP_SINGLE_CHIP)."
        )
      from embodied_bench.leaderboard_compiler import _audit_single_receipt

      audit_ok, audit_violations, _ = _audit_single_receipt(receipt_path)
      if not audit_ok:
        for av in audit_violations:
          if av not in violations:
            violations.append(av)
  elif measured_report_files:
    receipt_verified = False
    names = ", ".join(f.name for f in measured_report_files)
    receipt_message = (
        f"Submission folder contains measured artifact(s) [{names}] without a cryptographic "
        "'receipt.json' (or 'submission_receipt.json' / 'bundle_summary.json'). Do not delete receipt.json; regenerate the bundle with 'eeb measure'."
    )
    violations.append(receipt_message)
  else:
    receipt_message = (
        "No receipt.json or measured report.json present; validated declarative recipe, "
        "single-chip SKU envelope, and Python AST only (RECIPE_ONLY_NO_MEASUREMENTS)."
    )

  passed = (
      schema_valid
      and division_compliant
      and hw_check.passed
      and source_audit.passed
      and (receipt_verified is not False)
      and len(violations) == 0
  )
  status = RunStatus.CERTIFIED.value if passed else RunStatus.REJECTED.value
  reason = None if passed else "; ".join(violations)
  validate_status_and_reason(status=status, reason=reason)

  report = SubmissionValidationReport(
      submission_id=sub_id,
      submission_path=str(sub_dir),
      recipe_path=str(recipe_file),
      recipe_id=recipe_id,
      track=track,
      division=division,
      clock=clock,
      scenario=scenario,
      model_name=model_name,
      baseline_model_id=baseline_model_id,
      canonical_challenge_model=is_canonical,
      precision=precision,
      hardware_sku=hardware_sku,
      perf_cap=perf_cap,
      num_devices=num_devices,
      passed=passed,
      status=status,
      reason=reason,
      schema_valid=schema_valid,
      division_compliant=division_compliant,
      hardware_envelope_check=hw_check.to_dict(),
      source_audit=source_audit.to_dict(),
      has_measurements=has_measurements,
      validation_scope=validation_scope,
      receipt_verified=receipt_verified,
      receipt_message=receipt_message,
      submission_phase=str(normalized_meta.get("submission_phase") or "OPEN_RECIPE"),
      recipe_commit_sha256=normalized_meta.get("recipe_commit_sha256"),
      engine_tag=normalized_meta.get("engine_tag"),
      submitter_team=normalized_meta.get("team"),
      violations=violations,
  )

  if registry_path is not None:
    record_submission_validation(
        registry_path=registry_path,
        submission_id=sub_id,
        status=status,
        reason=reason,
        recipe_id=recipe_id,
        hardware_sku=hardware_sku,
        division=division,
        track=track,
        violations=violations if violations else None,
    )

  return report


# Alias for CLI / external callers
validate_submission = validate_submission_directory


def verify_submission_bundle(
    submission_path: str | Path,
    *,
    strict: bool = True,
    registry_path: Optional[str | Path] = None,
    require_single_chip_cap: bool = True,
) -> SubmissionValidationReport:
  """Verify a complete measured submission bundle (Phase 1 Commit-Reveal or Phase 2 Open-Recipe).

  Unlike `validate_submission_directory` (which also accepts recipe-only structural templates),
  `verify_submission_bundle` strictly requires cryptographic hardware measurements (`receipt.json`,
  `submission_receipt.json`, or Phase 1 `bundle_summary.json` + `submission_meta.json`), and when
  `strict=True` applies full leaderboard admissibility gates (`_audit_single_receipt`, non-synthetic
  hardware run, `num_devices == 1`, `CAP_SINGLE_CHIP`, `G1-G4`, and wire resolution >= 224x224).
  """
  from embodied_bench.leaderboard_compiler import _audit_single_receipt

  report = validate_submission_directory(
      submission_path=submission_path,
      registry_path=None,
      require_single_chip_cap=require_single_chip_cap,
  )
  extra_violations: List[str] = []
  sub_dir = Path(report.submission_path)

  if not report.has_measurements or report.receipt_verified is not True:
    if not report.has_measurements:
      extra_violations.append(
          f"Submission bundle {sub_dir} has no cryptographic hardware measurements "
          "(expected 'receipt.json' + 'report.json', or Phase 1 'bundle_summary.json' + 'submission_meta.json')."
      )
    elif report.receipt_message and report.receipt_message not in report.violations:
      extra_violations.append(report.receipt_message)

  if strict and report.has_measurements:
    receipt_candidates = [
        sub_dir / "receipt.json",
        sub_dir / "submission_receipt.json",
        sub_dir / "bundle_summary.json",
    ]
    manifest = next((c for c in receipt_candidates if c.is_file()), None)
    if manifest is not None:
      audit_ok, audit_violations, ctx = _audit_single_receipt(manifest)
      if not audit_ok:
        for v in audit_violations:
          if v not in report.violations and v not in extra_violations:
            extra_violations.append(v)
      else:
        op_pts = ctx.get("receipt", {}).get("operating_points") or ctx.get("sweep_data", {}).get("operating_points") or ctx.get("body", {}).get("frontier_points") or []
        admissible_pts = [
            pt for pt in op_pts
            if isinstance(pt, dict)
            and bool(pt.get("gate_a_passed", True))
            and pt.get("variance_passed", True) is not False
        ]
        if not admissible_pts:
          extra_violations.append(
              f"Submission bundle {sub_dir} contains 0 gate-passing variance-stable operating points."
          )

  if extra_violations:
    merged_violations = list(report.violations) + extra_violations
    reason = "; ".join(merged_violations)
    validate_status_and_reason(status=RunStatus.REJECTED.value, reason=reason)
    report.passed = False
    report.status = RunStatus.REJECTED.value
    report.reason = reason
    report.violations = merged_violations

  if registry_path is not None:
    record_submission_validation(
        registry_path=registry_path,
        submission_id=report.submission_id,
        status=report.status,
        reason=report.reason,
        recipe_id=report.recipe_id,
        hardware_sku=report.hardware_sku,
        division=report.division,
        track=report.track,
        violations=report.violations if report.violations else None,
    )
  return report


def prepare_phase1_commit_bundle(
    source_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    recipe_commit_sha256: str,
    team: Optional[str] = None,
    engine_tag: Optional[str] = None,
    division: Optional[str] = None,
    hardware_sku: Optional[str] = None,
) -> Dict[str, str]:
  """Create a clean Phase 1 Commit-Reveal PR folder (`bundle_summary.json` + `submission_meta.json`).

  Redacts raw per-sample run logs, local hostnames/file paths, and private server recipes/flags
  while preserving the unmodified cryptographic `attested_body`, `attestation_sha256`, hardware
  envelope check, quality gates, and summary operating points needed for live leaderboard display.
  """
  from embodied_bench.attestation import create_submission_receipt

  src = Path(source_bundle_dir)
  out = Path(output_dir)
  if not src.is_dir():
    raise FileNotFoundError(f"Source measurement bundle directory not found: {src}")

  report_path = next((p for p in (src / "report.json", src / "sweep_results.json") if p.is_file()), None)
  receipt_path = next((p for p in (src / "receipt.json", src / "submission_receipt.json", src / "bundle_summary.json") if p.is_file()), None)
  if receipt_path is None and report_path is None:
    raise FileNotFoundError(
        f"No 'receipt.json' or 'report.json' found in {src}. Run 'eeb measure --strict' first."
    )

  if receipt_path is not None:
    ok_sig, sig_msg = verify_submission_receipt(receipt_path)
    if not ok_sig:
      raise ValueError(f"Cannot package unverified source bundle: {sig_msg}")
    base_summary = dict(load_yaml_or_json(receipt_path))
    if report_path is not None:
      rep_data = load_yaml_or_json(report_path)
      for pub_key in ("envelope_check", "observation_source", "quality_gate", "quality_certified", "gate_mode"):
        if pub_key not in base_summary and pub_key in rep_data and rep_data[pub_key] is not None:
          base_summary[pub_key] = rep_data[pub_key]
  else:
    assert report_path is not None
    rep_data = load_yaml_or_json(report_path)
    base_summary = create_submission_receipt(rep_data)

  # Strip local filesystem hints from observation_source and raw samples from operating_points
  obs_src = base_summary.get("observation_source")
  if isinstance(obs_src, dict):
    cleaned_obs = {
        k: v
        for k, v in obs_src.items()
        if k not in ("path", "bundle_dir", "traces_path", "local_path")
    }
    base_summary["observation_source"] = cleaned_obs

  ops = base_summary.get("operating_points")
  if isinstance(ops, list):
    cleaned_ops: List[Dict[str, Any]] = []
    for op in ops:
      if isinstance(op, dict):
        item = {k: v for k, v in op.items() if k != "runs"}
        cleaned_ops.append(item)
    base_summary["operating_points"] = cleaned_ops

  body = base_summary.get("attested_body") if isinstance(base_summary.get("attested_body"), dict) else {}
  resolved_sku = str(
      hardware_sku
      or base_summary.get("hardware_sku")
      or body.get("hardware_sku")
      or ""
  ).strip()
  resolved_div = str(
      division
      or base_summary.get("division")
      or body.get("division")
      or DivisionType.CLOSED.value
  ).strip().upper()
  base_summary["bundle_format"] = "embodiedx_phase1_bundle_summary_v1"

  meta_payload: Dict[str, Any] = {
      "schema_version": "1.0.0",
      "submission_phase": "COMMIT_ONLY",
      "hardware_sku": resolved_sku,
      "division": resolved_div,
      "recipe_commit_sha256": recipe_commit_sha256,
  }
  if team:
    meta_payload["team"] = team
  if engine_tag:
    meta_payload["engine_tag"] = engine_tag

  meta_ok, meta_errs, norm_meta = validate_submission_meta(meta_payload, require_commit_hash=True)
  if not meta_ok:
    raise ValueError("Invalid Phase 1 submission metadata: " + "; ".join(meta_errs))

  from embodied_bench.attestation import compute_attestation_sha256

  orig_sha = str(base_summary.get("attestation_sha256", "") or "")
  signed_body = dict(body)
  signed_body["division"] = resolved_div
  signed_body["recipe_commit_sha256"] = str(norm_meta["recipe_commit_sha256"])
  if orig_sha and "phase1_source_attestation_sha256" not in signed_body:
    signed_body["phase1_source_attestation_sha256"] = orig_sha
  base_summary["division"] = resolved_div
  base_summary["recipe_commit_sha256"] = str(norm_meta["recipe_commit_sha256"])
  if orig_sha and "phase1_source_attestation_sha256" not in base_summary:
    base_summary["phase1_source_attestation_sha256"] = orig_sha
    norm_meta["phase1_source_attestation_sha256"] = orig_sha
  base_summary["attested_body"] = signed_body
  base_summary["attestation_sha256"] = compute_attestation_sha256(signed_body)

  out.mkdir(parents=True, exist_ok=True)
  summary_out = out / "bundle_summary.json"
  meta_out = out / "submission_meta.json"
  atomic_write_json(summary_out, base_summary)
  atomic_write_json(meta_out, norm_meta)

  ok_out, out_msg = verify_submission_receipt(summary_out)
  if not ok_out:
    raise ValueError(f"Generated bundle_summary.json failed verification: {out_msg}")

  return {
      "bundle_summary": str(summary_out),
      "submission_meta": str(meta_out),
      "attestation_sha256": str(base_summary["attestation_sha256"]),
      "recipe_commit_sha256": str(norm_meta["recipe_commit_sha256"]),
  }


def record_submission_validation(
    registry_path: str | Path,
    submission_id: str,
    status: str | RunStatus,
    *,
    reason: Optional[str] = None,
    recipe_id: Optional[str] = None,
    hardware_sku: Optional[str] = None,
    division: Optional[str] = None,
    track: Optional[str] = None,
    violations: Optional[Sequence[str]] = None,
    notes: Optional[str] = None,
    receipt_sha256: Optional[str] = None,
    previous_status: Optional[str | RunStatus] = None,
    enforce_transitions: bool = False,
) -> Dict[str, Any]:
  """Persist a submission validation outcome into a shared JSON registry (`Gates 1 & 2`).

  Guarantees:
    - Gate 1 (Crash Path): Cleans up orphan `.tmp_embodied_*` files via `cleanup_orphan_temp_files` and
      recovers from corrupt or zero-byte JSON files with `recovered_from_corrupt_state=True`.
    - Gate 1 (Repeat-Call Path): Constructs a clean record dictionary on every call so optional fields
      (`reason`, `notes`, `violations`, `receipt_sha256`, `recipe_id`, `hardware_sku`, `division`, `track`)
      never carry over stale values from a prior call when omitted.
    - Gate 1 (Concurrent-Writer Path): Uses `atomic_read_modify_write_json` (`fcntl.flock` + `tempfile.mkstemp`
      + `os.fsync` + `os.replace`) for safe concurrent updates across threads and subprocesses.
    - Gate 2 (State Machine Completeness): Supports all 8 `RunStatus` states (`PENDING`, `RUNNING`,
      `CERTIFIED`, `REJECTED`, `BLOCKED`, `FAILED`, `SKIPPED`, `INVALID`) and enforces a non-empty
      `reason` whenever status is in `{REJECTED, BLOCKED, FAILED, SKIPPED, INVALID}`.
  """
  if not submission_id or not str(submission_id).strip():
    raise ValueError("submission_id must be a non-empty string.")

  def _modify(state: Dict[str, Any]) -> Dict[str, Any]:
    submissions = state.setdefault("submissions", {})
    existing = submissions.get(submission_id)
    prev = previous_status
    if prev is None and enforce_transitions and isinstance(existing, dict):
      prev = existing.get("status")

    resolved_status = validate_status_and_reason(
        status=status,
        reason=reason,
        previous_status=prev,
    )
    cleaned_reason = (
        reason.strip() if isinstance(reason, str) and reason.strip() else None
    )
    cleaned_notes = (
        notes.strip() if isinstance(notes, str) and notes.strip() else None
    )
    cleaned_violations = [str(v) for v in violations] if violations else []

    record: Dict[str, Any] = {
        "submission_id": str(submission_id),
        "status": resolved_status.value,
        "reason": cleaned_reason,
        "recipe_id": str(recipe_id) if recipe_id is not None else None,
        "hardware_sku": str(hardware_sku) if hardware_sku is not None else None,
        "division": str(division) if division is not None else None,
        "track": str(track) if track is not None else None,
        "violations": cleaned_violations,
        "notes": cleaned_notes,
        "receipt_sha256": str(receipt_sha256) if receipt_sha256 is not None else None,
        "updated_at": now_pacific_iso(),
    }
    submissions[str(submission_id)] = record
    state["schema_version"] = str(state.get("schema_version", "1.0.0"))
    state["num_submissions"] = len(submissions)
    state["num_certified"] = sum(
        1
        for rec in submissions.values()
        if isinstance(rec, dict) and rec.get("status") == RunStatus.CERTIFIED.value
    )
    state["updated_at"] = now_pacific_iso()
    return state

  return atomic_read_modify_write_json(
      target_path=registry_path,
      modifier_fn=_modify,
      default_factory=lambda: {
          "schema_version": "1.0.0",
          "submissions": {},
          "num_submissions": 0,
          "num_certified": 0,
      },
  )
