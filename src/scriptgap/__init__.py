"""Script Is the Missing Interface — experiment code.

Importing this package installs repo-local cache isolation (see ``env.py``)
before any heavy third-party import can read the default ``$HOME`` paths.
"""

from __future__ import annotations

from . import env as _env

ROOT = _env.bootstrap()

__all__ = ["ROOT", "__version__"]
__version__ = "0.1.0"
