"""Run-directory layout.

Runs are separable two ways, as required by the experiment plan:

* **by time** — every invocation gets a ``run_id``; the default is a UTC
  timestamp (``20260730-141205``), so ``runs/`` is chronologically sorted.
* **by algorithm** — inside a run, predictions are sharded by
  ``stage / model / task__lang__condition``, so a single model or a single
  stage can be re-run, inspected, or deleted without touching the others.

::

    runs/<run_id>/
      manifest.json                     run config + git sha + environment
      logs/<stage>.log                  one log file per stage (also mirrored to console)
      predictions/<stage>/<model>/<shard>.jsonl
      predictions/<stage>/<model>/<shard>.done      written on shard completion
      raw/safety/<model>/<shard>.jsonl  full safety generations (gitignored, access-controlled)
      metrics/script_gap_results.parquet
      metrics/*.csv
      figures/*.pdf|png
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

from . import ROOT

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
LOGS_DIR = ROOT / "logs"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str) -> str:
    """Filesystem-safe token (model ids contain ``/``, conditions contain spaces)."""
    return _SAFE.sub("-", text.strip()).strip("-") or "unnamed"


def new_run_id(prefix: str | None = None) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slug(prefix)}-{stamp}" if prefix else stamp


@dataclass(frozen=True)
class RunPaths:
    """Resolved directory layout for one run."""

    run_id: str
    root: Path

    @classmethod
    def create(cls, run_id: str | None = None, prefix: str | None = None) -> "RunPaths":
        rid = run_id or new_run_id(prefix)
        rp = cls(run_id=rid, root=RUNS_DIR / rid)
        for d in (rp.root, rp.logs, rp.predictions, rp.raw, rp.metrics, rp.figures):
            d.mkdir(parents=True, exist_ok=True)
        return rp

    # -- top level ---------------------------------------------------------
    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    # -- sharding ----------------------------------------------------------
    def stage_dir(self, stage: str, model: str | None = None) -> Path:
        d = self.predictions / slug(stage)
        if model is not None:
            d = d / slug(model)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def shard(self, stage: str, model: str, shard: str) -> Path:
        return self.stage_dir(stage, model) / f"{slug(shard)}.jsonl"

    def raw_shard(self, stage: str, model: str, shard: str) -> Path:
        d = self.raw / slug(stage) / slug(model)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{slug(shard)}.jsonl"

    def log_file(self, stage: str) -> Path:
        return self.logs / f"{slug(stage)}.log"


def latest_run() -> RunPaths | None:
    """Most recent run directory, for ``--run-id latest``."""
    if not RUNS_DIR.exists():
        return None
    candidates = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not candidates:
        return None
    return RunPaths(run_id=candidates[-1].name, root=candidates[-1])


def resolve_run(run_id: str | None, prefix: str | None = None) -> RunPaths:
    """``None`` -> new run, ``"latest"`` -> most recent, otherwise that id."""
    if run_id == "latest":
        found = latest_run()
        if found is None:
            raise SystemExit("no runs found under runs/ — start one without --run-id first")
        # make sure sub-dirs exist even for a run created by an older version
        return RunPaths.create(run_id=found.run_id)
    return RunPaths.create(run_id=run_id, prefix=prefix)
