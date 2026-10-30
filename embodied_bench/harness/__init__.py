"""Phase 1 Client-Server Benchmark Harness, LoadGen Client, Latency-Coupled Closed-Loop LIBERO-RT Environment, and Single-Chip Hardware Verifier for EmbodiedX (`embodied-efficiency-bench`)."""

from embodied_bench.harness.hardware_verifier import (
    MultiChipAuditResult,
    SingleChipHardwareVerifier,
    SingleChipTelemetrySummary,
    SingleChipVerificationResult,
    TelemetrySamplePoint,
    audit_command_line_for_multi_chip_bypass,
    parse_rocm_smi_json,
    verify_single_chip_envelope,
)
from embodied_bench.harness.libero_rt_env import (
    EpisodeRolloutTrace,
    HAS_LIBERO_MUJOCO,
    LiberoRTClosedLoopEnv,
    LiberoRTClosedLoopResult,
)
from embodied_bench.harness.loadgen_client import (
    DEFAULT_LOADGEN_CONCURRENCY_GRID,
    EmbodiedLoadGenClient,
    LoadGenStreamResult,
    record_loadgen_session_state,
)
from embodied_bench.harness.server_protocol import (
    EmbodiedBenchmarkServer,
    ReferenceEmbodiedRunner,
    SUPPORTED_SERVER_ENDPOINTS,
    SessionStateRecord,
    decode_ndarray_b64,
    decode_payload_arrays,
    encode_ndarray_b64,
    encode_payload_arrays,
    is_encoded_ndarray,
)

__all__ = [
    "DEFAULT_LOADGEN_CONCURRENCY_GRID",
    "EmbodiedBenchmarkServer",
    "EmbodiedLoadGenClient",
    "EpisodeRolloutTrace",
    "HAS_LIBERO_MUJOCO",
    "LiberoRTClosedLoopEnv",
    "LiberoRTClosedLoopResult",
    "LoadGenStreamResult",
    "MultiChipAuditResult",
    "ReferenceEmbodiedRunner",
    "SUPPORTED_SERVER_ENDPOINTS",
    "SessionStateRecord",
    "SingleChipHardwareVerifier",
    "SingleChipTelemetrySummary",
    "SingleChipVerificationResult",
    "TelemetrySamplePoint",
    "audit_command_line_for_multi_chip_bypass",
    "decode_ndarray_b64",
    "decode_payload_arrays",
    "encode_ndarray_b64",
    "encode_payload_arrays",
    "is_encoded_ndarray",
    "parse_rocm_smi_json",
    "record_loadgen_session_state",
    "verify_single_chip_envelope",
]
