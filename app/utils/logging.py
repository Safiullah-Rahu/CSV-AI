"""Lightweight logger factory.

Avoids ``logging.basicConfig`` side effects (Streamlit configures its own
handlers). Each module gets a child logger of the ``csvai`` root.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False


def _configure_root(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("csvai")
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger under ``csvai.<name>``."""
    _configure_root(level or "INFO")
    return logging.getLogger(f"csvai.{name}")
