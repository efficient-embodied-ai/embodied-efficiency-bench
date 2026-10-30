"""EEB participant protocol: `eeb.proto` wire schema, generated stubs, and the in-process `Policy` ABC.

Importing this package never requires grpc/protobuf; the generated stubs are imported
lazily by `eeb.protocol.codec`, `eeb.servers.base`, and `eeb.loadgen.client`.
"""

from eeb.protocol.policy import (
    ACTION_DTYPE,
    SYNTHETIC_DUMMY_SERVER_IMPL,
    ActionChunk,
    BasePolicy,
    Caps,
    Context,
    Frame,
    Observation,
    Policy,
    instantiate_policy,
    load_policy_class,
    validate_policy,
)

__all__ = [
    "ACTION_DTYPE",
    "SYNTHETIC_DUMMY_SERVER_IMPL",
    "ActionChunk",
    "BasePolicy",
    "Caps",
    "Context",
    "Frame",
    "Observation",
    "Policy",
    "instantiate_policy",
    "load_policy_class",
    "validate_policy",
]
