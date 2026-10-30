"""NVIDIA Isaac GR00T-N1.7-3B reference-server scaffold."""

from eeb.servers.gr00t.policy import DEFAULT_GR00T_IMAGE_KEY_MAP, GR00T_INSTALL_HINT, Gr00tN17Policy

GR00TPolicy = Gr00tN17Policy

__all__ = ["DEFAULT_GR00T_IMAGE_KEY_MAP", "GR00T_INSTALL_HINT", "GR00TPolicy", "Gr00tN17Policy"]
