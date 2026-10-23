"""Run each model in its own process.

In-process teardown (``models/gpu.py``) is best-effort: vLLM allocates through
its own executor, keeps module-level distributed state, and builds a CUDA
context that the caching allocator does not always hand back. On a single 48 GB
card, a few hundred MB left stranded by model *n* is exactly what makes model
*n+1* fail to allocate its KV cache.

Process isolation removes the question. Each model is evaluated in a child
process; when it exits, the OS reclaims **all** of its device memory —
allocator caches, CUDA context and everything else — regardless of how well the
library cleaned up after itself.

This is safe to do because the run is already checkpointed per shard: the child
writes into the same ``runs/<run_id>/`` tree the parent would have used, and a
child that dies mid-shard leaves a resumable partial shard exactly as an
in-process crash would.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Sequence

from ..models.registry import ModelSpec

log = logging.getLogger(__name__)

#: Exit code the inference stage uses for "time budget exhausted".
BUDGET_EXHAUSTED = 2


def should_isolate(mode: str, specs: Sequence[ModelSpec], backend_default: str) -> bool:
    """Decide whether to spawn a process per model.

    ``auto`` isolates only when it actually buys something: more than one model,
    and a backend whose memory is hard to reclaim in-process. A single model has
    nothing to protect against, and the ``hf``/``anthropic`` backends release
    cleanly.
    """
    mode = (mode or "auto").lower()
    if mode == "always":
        return len(specs) > 0
    if mode == "never":
        return False
    if mode != "auto":
        raise SystemExit(
            f"inference.isolate_models must be auto|always|never, got {mode!r}"
        )
    if len(specs) < 2:
        return False
    return any((spec.backend or backend_default).lower() == "vllm" for spec in specs)


def strip_flag(argv: list[str], flag: str, takes_value: bool = True) -> list[str]:
    """Remove every occurrence of ``--flag value`` / ``--flag=value``."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == flag:
            skip = takes_value
            continue
        if token.startswith(f"{flag}="):
            continue
        out.append(token)
    return out


def child_argv(base_argv: list[str], model: str, run_id: str, remaining: float | None) -> list[str]:
    """Build the child command line for one model."""
    argv = strip_flag(base_argv, "--model")
    argv = strip_flag(argv, "--run-id")
    argv = strip_flag(argv, "--run-prefix")
    argv = strip_flag(argv, "--max-seconds")

    argv += ["--model", model, "--run-id", run_id]
    if remaining is not None:
        argv += ["--max-seconds", f"{max(1.0, remaining):.0f}"]
    # Stop the child from isolating again.
    argv += ["--set", "inference.isolate_models=never"]
    return [sys.executable, "-m", "scriptgap.cli", *argv]


def run_isolated(
    specs: Sequence[ModelSpec],
    base_argv: list[str],
    run_id: str,
    max_seconds: float | None = None,
    continue_on_error: bool = True,
) -> int:
    """Evaluate each model in a child process, in order. Returns an exit code."""
    started = time.time()
    failed: dict[str, int] = {}

    for position, spec in enumerate(specs, start=1):
        remaining = None
        if max_seconds is not None:
            remaining = max_seconds - (time.time() - started)
            if remaining <= 1.0:
                log.warning(
                    "time budget exhausted before %s — resume with --run-id %s",
                    spec.name,
                    run_id,
                )
                return BUDGET_EXHAUSTED

        command = child_argv(list(base_argv), spec.name, run_id, remaining)
        log.info(
            "=== model %d/%d: %s (isolated process) ===", position, len(specs), spec.name
        )
        log.debug("  %s", " ".join(command))

        completed = subprocess.run(command, check=False)

        if completed.returncode == BUDGET_EXHAUSTED:
            log.warning(
                "%s hit the time budget; stopping. Resume with --run-id %s",
                spec.name,
                run_id,
            )
            return BUDGET_EXHAUSTED
        if completed.returncode != 0:
            failed[spec.name] = completed.returncode
            log.error(
                "%s exited with code %d. Its shards are checkpointed — rerun with "
                "--run-id %s to continue from where it stopped.",
                spec.name,
                completed.returncode,
                run_id,
            )
            if not continue_on_error:
                return completed.returncode
            log.warning("continuing with the remaining model(s)")
            continue

        log.info("%s finished; its process exited, so the GPU is fully reclaimed", spec.name)

    if failed:
        log.error("%d of %d model(s) failed: %s", len(failed), len(specs), failed)
        return 1
    return 0
