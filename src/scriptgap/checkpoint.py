"""Crash-safe, resumable JSONL prediction store.

Design constraints from the experiment plan: ~13k generations over ~10 GPU
hours, on a single machine, and a failed run must be resumable rather than
restarted. So:

* Predictions are appended to a **JSONL shard** and flushed (plus ``fsync`` on
  a configurable interval) so at most the last few records are ever lost.
* Every record carries a stable ``uid``. On resume we read the shard, keep the
  uids we already have, and skip those items. A half-written trailing line
  (the classic kill -9 artefact) is detected and truncated rather than
  poisoning the run.
* A sibling ``<shard>.done`` marker records completion, so an orchestrator can
  skip a finished shard without parsing it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, tolerating a truncated final line."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Only the last line may legitimately be torn.
                log.warning("%s:%d is not valid JSON — ignoring (torn write?)", path, lineno)


def repair_jsonl(path: Path) -> int:
    """Drop trailing garbage from a shard. Returns the number of good records."""
    if not path.exists():
        return 0
    good: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                log.warning("truncating %s at a malformed record", path)
                break
            good.append(stripped)
    tmp = path.with_suffix(path.suffix + ".repair")
    tmp.write_text("\n".join(good) + ("\n" if good else ""), encoding="utf-8")
    tmp.replace(path)
    return len(good)


def completed_uids(path: Path) -> set[str]:
    """uids already present in a shard — the resume set."""
    return {r["uid"] for r in read_jsonl(path) if isinstance(r, dict) and "uid" in r}


class ShardWriter:
    """Append-only JSONL writer with periodic durability and resume support.

    Usage::

        with ShardWriter(path) as w:
            todo = [it for it in items if it["uid"] not in w.done]
            for item in todo:
                w.write(predict(item))
            w.mark_done()
    """

    def __init__(self, path: Path, flush_every: int = 10, fsync: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.fsync = fsync

        # Repair before reading so a torn tail does not shrink the resume set.
        if self.path.exists():
            repair_jsonl(self.path)
        self.done: set[str] = completed_uids(self.path)
        self.resumed_count = len(self.done)

        self._fh = self.path.open("a", encoding="utf-8")
        self._since_flush = 0
        self.written = 0

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- api ---------------------------------------------------------------
    def write(self, record: dict[str, Any]) -> None:
        if "uid" not in record:
            raise ValueError("prediction records must carry a 'uid' field")
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.done.add(record["uid"])
        self.written += 1
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._flush()

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.write(record)

    def _flush(self) -> None:
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())
        self._since_flush = 0

    @property
    def marker(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".done")

    def is_done(self) -> bool:
        return self.marker.exists()

    def mark_done(self, meta: dict[str, Any] | None = None) -> None:
        self._flush()
        payload = {"records": len(self.done), **(meta or {})}
        self.marker.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def close(self) -> None:
        if not self._fh.closed:
            self._flush()
            self._fh.close()


def prune_shard(path: Path, keep: "Callable[[dict[str, Any]], bool]") -> tuple[int, int]:
    """Drop records that ``keep`` rejects. Returns ``(kept, dropped)``.

    Used to evict predictions whose prompt fingerprint no longer matches the
    rendered suite, so a resume regenerates them instead of silently keeping
    output produced under different transform settings.
    """
    if not path.exists():
        return (0, 0)
    records = list(read_jsonl(path))
    kept = [r for r in records if keep(r)]
    dropped = len(records) - len(kept)
    if dropped:
        write_jsonl_atomic(path, kept)
        # the shard is no longer complete
        path.with_suffix(path.suffix + ".done").unlink(missing_ok=True)
    return (len(kept), dropped)


def shard_is_done(path: Path) -> bool:
    """True when the shard's ``.done`` marker exists (cheap skip check)."""
    return path.with_suffix(path.suffix + ".done").exists()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically replace ``path`` with ``payload`` as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Atomically write a whole JSONL file. Returns the record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                n += 1
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
        return n
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
