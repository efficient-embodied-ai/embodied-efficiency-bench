"""Reference policy servers.

* `eeb.servers.base`  - gRPC servicer / `serve_policy()` over UDS or TCP for any `Policy`.
* `eeb.servers.dummy` - `DummyVLAPolicy`, `DummyWMPolicy` (`server_impl=SYNTHETIC_DUMMY`, no model).
* `eeb.servers.pi05`  - pi0.5 (`pi05-3b`) via `lerobot`/`openpi` PyTorch (CUDA/ROCm/TorchTPU) and native JAX/Flax (`openpi`) targeting NVIDIA L40S 48 GB and Google Cloud TPU v6e-1 32 GB.
* `eeb.servers.gr00t` - GR00T-N1.7-3B via `isaac-gr00t` PyTorch (CUDA/ROCm/TorchTPU) targeting NVIDIA L40S 48 GB and Google Cloud TPU v6e-1 32 GB.
* `eeb.servers.cosmos_predict25` - Cosmos-Predict2.5-2B via `diffusers` PyTorch (CUDA/ROCm/TorchTPU) and optional JAX/Flax targeting NVIDIA L40S 48 GB and Google Cloud TPU v6e-1 32 GB.
"""

from eeb.servers.base import Endpoint, PolicyServicer, parse_endpoint, serve_policy
from eeb.servers.dummy import DummyVLAPolicy, DummyWMPolicy

__all__ = [
    "DummyVLAPolicy",
    "DummyWMPolicy",
    "Endpoint",
    "PolicyServicer",
    "parse_endpoint",
    "serve_policy",
]
