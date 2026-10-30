"""`eeb` - EmbodiedX / embodied-efficiency-bench measuring harness (WS-A / M1 Harness MVP).

Subpackages (engineering plan §3):

* `eeb.protocol`  - `eeb.proto` wire schema, generated gRPC stubs, in-process `Policy` ABC.
* `eeb.servers`   - gRPC servicer over Unix-domain socket / TCP, reference dummies, π0.5 scaffold.
* `eeb.loadgen`   - client, pacing + session-sticky routing, concurrency sweep, JSON report + receipt.
* `eeb.telemetry` - 10 Hz power/memory sidecar with nvml / tegrastats / amd-smi / tpu-info sources.
* `eeb.envelope`  - live single-chip envelope check (`num_devices == 1`, memory, spec TOPS).
* `eeb.cli`       - `eeb serve | measure | telemetry probe | envelope check`.

The core harness depends only on the standard library, NumPy, and grpcio; everything else
(pynvml, torch, openpi, jax) is imported lazily behind guards. Scoring / schema / attestation
logic is reused from the existing `embodied_bench` package.
"""

from __future__ import annotations

__version__ = "1.0.0"

# Execution-provenance vocabulary shared by the harness, telemetry, and reports.
PROVENANCE_MEASURED = "MEASURED"
PROVENANCE_ESTIMATED = "ESTIMATED"
PROVENANCE_UNAVAILABLE = "UNAVAILABLE"
LOADGEN_PROVENANCE_REAL_WALLCLOCK = "REAL_WALLCLOCK"

__all__ = [
    "LOADGEN_PROVENANCE_REAL_WALLCLOCK",
    "PROVENANCE_ESTIMATED",
    "PROVENANCE_MEASURED",
    "PROVENANCE_UNAVAILABLE",
    "__version__",
]
