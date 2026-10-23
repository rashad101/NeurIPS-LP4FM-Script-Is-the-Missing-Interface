"""File + console logging that co-operates with tqdm progress bars."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Any

from tqdm.auto import tqdm

from . import env
from .paths import LOGS_DIR, RunPaths

_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

#: Handler-ownership tags. ``setup_logging`` is called once per stage, and the
#: ``all`` pipeline runs six stages in one process — without these tags each
#: call appended another console + file handler to the root logger and every
#: line was emitted once per stage already started (six copies by the end).
_TAG_CONSOLE = "_scriptgap_console"
_TAG_GLOBAL_FILE = "_scriptgap_global_file"
_TAG_STAGE_FILE = "_scriptgap_stage_file"


class TqdmHandler(logging.Handler):
    """Emit through ``tqdm.write`` so log lines never shred a progress bar."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - I/O
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(
    stage: str,
    run: RunPaths | None = None,
    level: str = "INFO",
    quiet_console: bool = False,
) -> logging.Logger:
    """Attach handlers for ``stage``, replacing any previous stage's.

    Every stage writes two files: the per-run log inside
    ``runs/<run_id>/logs/<stage>.log`` and an append-only global log under
    ``logs/scriptgap.log``, so a crashed run still leaves a trace even if the
    run directory is later cleaned up.

    Safe to call repeatedly in one process: the console and global-file handlers
    are installed once, and the per-stage file handler is swapped rather than
    stacked.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)
    existing = {tag for tag in (_TAG_CONSOLE, _TAG_GLOBAL_FILE)
                for h in root.handlers if getattr(h, tag, False)}

    # Global log file — install once.
    if _TAG_GLOBAL_FILE not in existing:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        gh = logging.FileHandler(LOGS_DIR / "scriptgap.log", mode="a", encoding="utf-8")
        gh.setFormatter(formatter)
        setattr(gh, _TAG_GLOBAL_FILE, True)
        root.addHandler(gh)

    # Console — install once.
    if not quiet_console and _TAG_CONSOLE not in existing:
        ch = TqdmHandler()
        ch.setFormatter(formatter)
        setattr(ch, _TAG_CONSOLE, True)
        root.addHandler(ch)

    # Per-stage run file — swap out the previous stage's.
    if run is not None:
        for handler in [h for h in root.handlers if getattr(h, _TAG_STAGE_FILE, False)]:
            root.removeHandler(handler)
            handler.close()
        sh = logging.FileHandler(run.log_file(stage), mode="a", encoding="utf-8")
        sh.setFormatter(formatter)
        setattr(sh, _TAG_STAGE_FILE, True)
        root.addHandler(sh)

    # Third-party libraries are extremely chatty at INFO — httpx in particular
    # logs one line per HTTP request, which buries our own progress output
    # during dataset resolution.
    for noisy in (
        "urllib3",
        "filelock",
        "fsspec",
        "datasets",
        "huggingface_hub",
        "matplotlib",
        "httpx",
        "httpcore",
        "requests",
        "vllm",
        "anthropic",
        "openai",
        "PIL",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(f"scriptgap.{stage}")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": git_sha(),
        "cwd": os.getcwd(),
        "caches": env.describe(),
    }
    try:  # torch is an optional extra on CPU-only checkouts
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1024**3, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        snap["torch"] = None
    return snap


def write_manifest(run: RunPaths, stage: str, config: dict[str, Any]) -> None:
    """Record (or extend) the run manifest with this stage's configuration."""
    manifest: dict[str, Any] = {}
    if run.manifest.exists():
        try:
            manifest = json.loads(run.manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("run_id", run.run_id)
    manifest.setdefault("environment", environment_snapshot())
    stages = manifest.setdefault("stages", {})
    stages[stage] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config,
    }
    run.manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
