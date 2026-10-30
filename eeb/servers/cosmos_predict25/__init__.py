"""NVIDIA Cosmos-Predict2.5-2B (PyTorch `diffusers` and native JAX/Flax) reference-server scaffolds."""

from eeb.servers.cosmos_predict25.policy import (
    COSMOS_JAX_INSTALL_HINT,
    COSMOS_PYTORCH_INSTALL_HINT,
    CosmosPredict25JaxPolicy,
    CosmosPredict25Policy,
)

__all__ = [
    "COSMOS_JAX_INSTALL_HINT",
    "COSMOS_PYTORCH_INSTALL_HINT",
    "CosmosPredict25JaxPolicy",
    "CosmosPredict25Policy",
]
