"""gRPC server adapter re-exports for `eeb.protocol` (`PolicyServicer`, `serve_policy`).

The primary implementation lives in `eeb.servers.base` and `eeb.protocol.codec` (including
`execute_horizon` wire serialization on `Caps`).
"""

from __future__ import annotations

from eeb.protocol.codec import caps_to_proto, proto_to_caps
from eeb.servers.base import Endpoint, PolicyServicer, build_servicer, parse_endpoint, serve_policy

__all__ = [
    "Endpoint",
    "PolicyServicer",
    "build_servicer",
    "caps_to_proto",
    "parse_endpoint",
    "proto_to_caps",
    "serve_policy",
]
