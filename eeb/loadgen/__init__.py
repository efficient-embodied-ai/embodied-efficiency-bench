"""EEB LoadGen: transport-agnostic client, pacing + session-sticky routing, concurrency sweep, report."""

from eeb.loadgen.client import GrpcPolicyClient, InProcessPolicyClient, PolicyClient, PredictResult, RolloutResult, open_client
from eeb.loadgen.pacing import PacingConfig, SessionRouter, StickinessViolation, verify_sticky_trace
from eeb.loadgen.sweep import (
    ConcurrencyResult,
    ConcurrencySweep,
    ObservationSource,
    ObservationSpec,
    RunResult,
    SweepConfig,
    SweepResult,
    cross_run_spread_pct,
    run_sweep,
)

__all__ = [
    "ConcurrencyResult",
    "ConcurrencySweep",
    "GrpcPolicyClient",
    "InProcessPolicyClient",
    "ObservationSource",
    "ObservationSpec",
    "PacingConfig",
    "PolicyClient",
    "PredictResult",
    "RolloutResult",
    "RunResult",
    "SessionRouter",
    "StickinessViolation",
    "SweepConfig",
    "SweepResult",
    "cross_run_spread_pct",
    "open_client",
    "run_sweep",
    "verify_sticky_trace",
]
