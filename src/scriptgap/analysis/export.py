"""Bundle a run into a self-contained, checksummed archive.

``runs/`` is gitignored — results live on one machine's disk. This stage
produces a single artefact that survives losing that machine and that someone
else can re-analyse without this repository:

* the released ``script_gap_results.parquet`` (prompts and completions included)
* every prediction shard, verbatim
* the rendered suite and its manifest, so items can be traced to their transform
* metrics CSVs, figures, the run manifest, per-stage logs
* the calibration measurement, if one was run
* ``EXPORT.json`` — SHA-256 of every file, row counts, and a completeness report

**Safety completions are excluded by default.** ``runs/<id>/raw/safety/`` holds
raw model output for harmful-intent prompts; it is kept out of the bundle unless
``--include-raw`` is passed explicitly, so an archive cannot leak it by accident.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

from .. import ROOT, __version__
from ..checkpoint import read_jsonl, write_json_atomic
from ..config import Config
from ..data.build import suite_dir
from ..paths import RunPaths
from ..progress import track

log = logging.getLogger(__name__)

EXPORTS_DIR = ROOT / "exports"

#: Copied wholesale when present.
_RUN_SUBDIRS = ("metrics", "figures", "logs", "predictions")

#: Suite files that make the predictions interpretable on their own.
_SUITE_FILES = ("base.jsonl", "rendered.jsonl", "suite_manifest.json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _copy_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def _completeness(run: RunPaths) -> dict[str, Any]:
    """Shard-level completeness, so a partial run cannot be mistaken for a full one."""
    root = run.predictions / "inference"
    if not root.exists():
        return {"inference_shards": 0, "complete_shards": 0, "incomplete": []}

    shards = sorted(root.glob("*/*.jsonl"))
    incomplete: list[str] = []
    total_records = 0
    for shard in shards:
        records = sum(1 for _ in read_jsonl(shard))
        total_records += records
        if not shard.with_suffix(shard.suffix + ".done").exists():
            incomplete.append(f"{shard.parent.name}/{shard.stem} ({records} records, no .done)")

    return {
        "inference_shards": len(shards),
        "complete_shards": len(shards) - len(incomplete),
        "prediction_records": total_records,
        "incomplete": incomplete,
    }


def run(
    cfg: Config,
    run_paths: RunPaths,
    out: Path | None = None,
    include_raw: bool = False,
    compress: bool = True,
) -> Path:
    """Assemble the bundle. Returns the archive path (or directory if uncompressed)."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stage_dir = EXPORTS_DIR / f"{run_paths.run_id}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    counts: dict[str, int] = {}

    for name in _RUN_SUBDIRS:
        counts[name] = _copy_tree(run_paths.root / name, stage_dir / name)

    if run_paths.manifest.exists():
        shutil.copy2(run_paths.manifest, stage_dir / "manifest.json")

    # The suite: without it, a prediction's condition is a bare label.
    suite_src = suite_dir(cfg)
    suite_dst = stage_dir / "suite"
    suite_dst.mkdir(exist_ok=True)
    suite_copied = 0
    for name in _SUITE_FILES:
        source = suite_src / name
        if source.exists():
            shutil.copy2(source, suite_dst / name)
            suite_copied += 1
    counts["suite"] = suite_copied

    calibration = ROOT / "data" / "calibration"
    if calibration.exists():
        counts["calibration"] = _copy_tree(calibration, stage_dir / "calibration")

    human_check = ROOT / "data" / "human_check"
    if human_check.exists():
        counts["human_check"] = _copy_tree(human_check, stage_dir / "human_check")

    if include_raw:
        counts["raw"] = _copy_tree(run_paths.raw, stage_dir / "raw")
        log.warning(
            "--include-raw: this bundle contains raw model completions for "
            "harmful-intent prompts. Do not distribute it."
        )
    else:
        if (run_paths.raw / "safety").exists():
            log.info(
                "excluding runs/%s/raw/safety (pass --include-raw to override)",
                run_paths.run_id,
            )

    # Checksums over everything that landed.
    files = sorted(p for p in stage_dir.rglob("*") if p.is_file())
    checksums = {
        str(p.relative_to(stage_dir)): {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in track(files, desc="checksumming", unit="file")
    }

    completeness = _completeness(run_paths)
    export_manifest = {
        "run_id": run_paths.run_id,
        "scriptgap_version": __version__,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "includes_raw_safety": include_raw,
        "file_counts": counts,
        "completeness": completeness,
        "total_bytes": sum(v["bytes"] for v in checksums.values()),
        "files": checksums,
    }
    write_json_atomic(stage_dir / "EXPORT.json", export_manifest)

    if completeness["incomplete"]:
        log.warning(
            "%d shard(s) have no .done marker — this run is partial:",
            len(completeness["incomplete"]),
        )
        for entry in completeness["incomplete"][:10]:
            log.warning("    %s", entry)

    log.info(
        "bundled %d files (%.1f MB) covering %d prediction records",
        len(files) + 1,
        export_manifest["total_bytes"] / 1024**2,
        completeness.get("prediction_records", 0),
    )

    if not compress:
        log.info("export -> %s", stage_dir)
        return stage_dir

    archive = Path(out) if out else EXPORTS_DIR / f"scriptgap_{run_paths.run_id}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage_dir, arcname=run_paths.run_id)
    shutil.rmtree(stage_dir)

    log.info("export -> %s (%.1f MB)", archive, archive.stat().st_size / 1024**2)
    return archive


def verify(archive: Path) -> dict[str, Any]:
    """Re-check an exported bundle against its own checksums."""
    import json

    if archive.is_dir():
        root = archive
    else:
        raise SystemExit("verify expects an extracted export directory")

    manifest_path = root / "EXPORT.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found — not a scriptgap export")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mismatched: list[str] = []
    missing: list[str] = []
    for name, meta in track(manifest["files"].items(), desc="verifying", unit="file"):
        path = root / name
        if not path.exists():
            missing.append(name)
            continue
        if sha256_file(path) != meta["sha256"]:
            mismatched.append(name)

    result = {
        "run_id": manifest.get("run_id"),
        "files": len(manifest["files"]),
        "missing": missing,
        "mismatched": mismatched,
        "ok": not missing and not mismatched,
    }
    if result["ok"]:
        log.info("export %s verified: %d files intact", result["run_id"], result["files"])
    else:
        log.error("export %s FAILED: %d missing, %d mismatched",
                  result["run_id"], len(missing), len(mismatched))
    return result
