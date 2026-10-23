"""Resumability tests.

The requirement is that a killed run resumes rather than restarts, so these
tests simulate the failure modes that actually happen: a hard kill mid-write
(torn final line), a re-open, and a duplicate-work check.
"""

from __future__ import annotations

import json

from scriptgap.checkpoint import (
    ShardWriter,
    completed_uids,
    read_jsonl,
    repair_jsonl,
    shard_is_done,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_writer_appends_and_resumes(tmp_path):
    path = tmp_path / "shard.jsonl"

    with ShardWriter(path, flush_every=1) as writer:
        writer.write({"uid": "a", "value": 1})
        writer.write({"uid": "b", "value": 2})
        assert writer.resumed_count == 0

    with ShardWriter(path) as writer:
        assert writer.done == {"a", "b"}
        assert writer.resumed_count == 2
        writer.write({"uid": "c", "value": 3})

    assert completed_uids(path) == {"a", "b", "c"}
    assert len(list(read_jsonl(path))) == 3


def test_torn_final_line_is_repaired_not_lost(tmp_path):
    """kill -9 mid-write leaves a partial line; the earlier records must survive."""
    path = tmp_path / "shard.jsonl"
    path.write_text(
        json.dumps({"uid": "a"}) + "\n" + json.dumps({"uid": "b"}) + "\n" + '{"uid": "c", "val',
        encoding="utf-8",
    )

    with ShardWriter(path) as writer:
        assert writer.done == {"a", "b"}, "repair dropped good records"
        writer.write({"uid": "c", "recovered": True})

    records = list(read_jsonl(path))
    assert [r["uid"] for r in records] == ["a", "b", "c"]
    assert records[-1]["recovered"] is True


def test_repair_returns_the_good_record_count(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_text(json.dumps({"uid": "a"}) + "\n" + "{broken", encoding="utf-8")
    assert repair_jsonl(path) == 1


def test_done_marker(tmp_path):
    path = tmp_path / "shard.jsonl"
    assert not shard_is_done(path)
    with ShardWriter(path) as writer:
        writer.write({"uid": "a"})
        writer.mark_done({"note": "finished"})
    assert shard_is_done(path)
    marker = json.loads(path.with_suffix(".jsonl.done").read_text(encoding="utf-8"))
    assert marker["records"] == 1
    assert marker["note"] == "finished"


def test_write_requires_a_uid(tmp_path):
    with ShardWriter(tmp_path / "s.jsonl") as writer:
        try:
            writer.write({"no_uid": True})
        except ValueError:
            return
    raise AssertionError("a record without a uid should be rejected")


def test_missing_file_reads_as_empty(tmp_path):
    assert list(read_jsonl(tmp_path / "absent.jsonl")) == []
    assert completed_uids(tmp_path / "absent.jsonl") == set()


def test_atomic_writers_leave_no_temp_files(tmp_path):
    write_json_atomic(tmp_path / "a.json", {"x": 1})
    write_jsonl_atomic(tmp_path / "b.jsonl", [{"uid": "1"}, {"uid": "2"}])
    assert json.loads((tmp_path / "a.json").read_text())["x"] == 1
    assert len(list(read_jsonl(tmp_path / "b.jsonl"))) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_does_not_clobber_on_failure(tmp_path):
    """A failed atomic write must leave the previous content intact."""
    path = tmp_path / "a.json"
    write_json_atomic(path, {"good": True})

    class Unserialisable:
        pass

    def bad_records():
        yield {"uid": "1"}
        raise RuntimeError("boom")

    try:
        write_jsonl_atomic(path, bad_records())
    except RuntimeError:
        pass
    assert json.loads(path.read_text())["good"] is True
    assert not list(tmp_path.glob("*.tmp"))
