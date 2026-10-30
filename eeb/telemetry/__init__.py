"""Cross-vendor telemetry sidecar (10 Hz) with honest `MEASURED | ESTIMATED | UNAVAILABLE` provenance."""

from eeb.telemetry.sidecar import (
    TelemetrySidecar,
    TelemetrySummary,
    detect_sources,
    joules_per_action,
    select_source,
    summarize_samples,
)
from eeb.telemetry.source import NullSource, TelemetrySource, make_sample, unavailable_sample

__all__ = [
    "NullSource",
    "TelemetrySidecar",
    "TelemetrySource",
    "TelemetrySummary",
    "detect_sources",
    "joules_per_action",
    "make_sample",
    "select_source",
    "summarize_samples",
    "unavailable_sample",
]
