"""Uniform tqdm helpers.

All long-running loops in this codebase — dataset download, transform
generation, inference, judging, tokenisation, figure rendering — go through
``bar()`` or ``track()`` so the terminal output is consistent and every stage
reports progress.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Sized, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

#: Honour ``SCRIPTGAP_NO_PROGRESS=1`` for cron/CI runs where bars are noise.
_DISABLED = os.environ.get("SCRIPTGAP_NO_PROGRESS", "").strip() in {"1", "true", "yes"}

_DEFAULTS = dict(
    dynamic_ncols=True,
    leave=True,
    smoothing=0.05,
    mininterval=0.5,
)


def bar(desc: str, total: int | None = None, unit: str = "it", **kwargs) -> tqdm:
    """A configured, manually-advanced progress bar."""
    opts = {**_DEFAULTS, "disable": _DISABLED, **kwargs}
    return tqdm(total=total, desc=desc, unit=unit, **opts)


def track(
    iterable: Iterable[T],
    desc: str,
    total: int | None = None,
    unit: str = "it",
    **kwargs,
) -> Iterator[T]:
    """Wrap an iterable in a progress bar, inferring ``total`` when possible."""
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)
    opts = {**_DEFAULTS, "disable": _DISABLED, **kwargs}
    return tqdm(iterable, desc=desc, total=total, unit=unit, **opts)
