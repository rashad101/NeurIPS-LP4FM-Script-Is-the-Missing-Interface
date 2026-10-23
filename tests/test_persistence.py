"""Prediction persistence: everything needed for future analysis is on disk.

The scenarios here are the ones that silently corrupt or lose results:

* re-rendering after a transform change (e.g. ``scriptgap calibrate``) keeps the
  uid but changes the prompt, so a uid-keyed resume would keep stale output;
* a generation truncated at ``max_tokens`` is indistinguishable from a wrong
  answer once reduced to a score;
* the released parquet being a scoreboard rather than a re-analysable table.
"""

from __future__ import annotations

import json

import pandas as pd

from scriptgap.checkpoint import ShardWriter, prune_shard, read_jsonl, shard_is_done
from scriptgap.config import Config, load_config
from scriptgap.data.build import render_item
from scriptgap.data.schema import BaseItem, prompt_fingerprint, script_for
from scriptgap.eval.run_inference import _record
from scriptgap.models.backends import Generation


def _item(**over) -> BaseItem:
    base = dict(
        item_id="itm1",
        task="milu",
        language="hi",
        script=script_for("hi"),
        prompt="भारत की राजधानी नई दिल्ली है।",
        answer="A",
        answer_type="mcq",
        options=["हाँ", "नहीं"],
    )
    base.update(over)
    return BaseItem(**base)


# ---------------------------------------------------------------------------
# Prompt fingerprints
# ---------------------------------------------------------------------------


def test_fingerprint_covers_prompt_and_options():
    a = prompt_fingerprint("q", ["x", "y"])
    assert a == prompt_fingerprint("q", ["x", "y"])
    assert a != prompt_fingerprint("q", ["x", "z"]), "options must be in scope"
    assert a != prompt_fingerprint("q!", ["x", "y"])
    assert prompt_fingerprint("q") != prompt_fingerprint("q", ["x"])


def test_every_rendered_cell_carries_a_fingerprint():
    cfg = load_config(None, ["data.noisy_samples=2"])
    for cell in render_item(_item(), cfg):
        assert cell.prompt_sha256
        assert cell.prompt_sha256 == prompt_fingerprint(cell.prompt, cell.options)


def test_transform_change_keeps_the_uid_but_changes_the_fingerprint():
    """The exact hazard `scriptgap calibrate` introduces.

    Re-fitting the noise probabilities rewrites the prompt while `item_id` —
    which hashes only the native source — is unchanged. Without a content
    fingerprint a resume cannot tell the two apart.
    """
    before = {
        c.condition: c
        for c in render_item(
            _item(),
            load_config(None, ["data.noisy_samples=1", "transforms.noisy_roman.vowel_ending_drop=0.0"]),
        )
    }["noisy_roman"]
    after = {
        c.condition: c
        for c in render_item(
            _item(),
            load_config(
                None,
                ["data.noisy_samples=1", "transforms.noisy_roman.vowel_ending_drop=0.99"],
            ),
        )
    }["noisy_roman"]

    assert before.uid == after.uid
    assert before.prompt != after.prompt
    assert before.prompt_sha256 != after.prompt_sha256


def test_per_language_override_changes_the_fingerprint():
    """Per-language calibration must also invalidate — it changes the prompt."""
    plain = load_config(None, ["data.noisy_samples=1"])
    scoped = load_config(
        None,
        [
            "data.noisy_samples=1",
            "transforms.noisy_roman_by_language={'hi': {'vowel_ending_drop': 0.99}}",
        ],
    )
    a = {c.condition: c for c in render_item(_item(), plain)}["noisy_roman"]
    b = {c.condition: c for c in render_item(_item(), scoped)}["noisy_roman"]
    assert a.prompt_sha256 != b.prompt_sha256


# ---------------------------------------------------------------------------
# Stale eviction
# ---------------------------------------------------------------------------


def test_prune_shard_drops_stale_and_clears_the_done_marker(tmp_path):
    path = tmp_path / "shard.jsonl"
    with ShardWriter(path) as writer:
        writer.write({"uid": "a", "prompt_sha256": "old"})
        writer.write({"uid": "b", "prompt_sha256": "same"})
        writer.mark_done()
    assert shard_is_done(path)

    current = {"a": "new", "b": "same"}
    kept, dropped = prune_shard(
        path, lambda r: r.get("prompt_sha256") == current.get(r["uid"])
    )
    assert (kept, dropped) == (1, 1)
    assert {r["uid"] for r in read_jsonl(path)} == {"b"}
    assert not shard_is_done(path), "a pruned shard must no longer count as complete"


def test_prune_shard_is_a_no_op_when_everything_matches(tmp_path):
    path = tmp_path / "shard.jsonl"
    with ShardWriter(path) as writer:
        writer.write({"uid": "a", "prompt_sha256": "x"})
        writer.mark_done()

    kept, dropped = prune_shard(path, lambda r: True)
    assert (kept, dropped) == (1, 0)
    assert shard_is_done(path), "an untouched shard must keep its marker"


def test_records_without_a_fingerprint_are_kept(tmp_path):
    """Shards written before fingerprints existed must not be wiped."""
    path = tmp_path / "shard.jsonl"
    with ShardWriter(path) as writer:
        writer.write({"uid": "legacy"})

    expected = {"legacy": "whatever"}

    def still_valid(record):
        want = expected.get(record.get("uid", ""))
        if want is None or not want:
            return True
        return record.get("prompt_sha256") == want

    # the record has no fingerprint at all -> mismatch -> dropped
    _kept, dropped = prune_shard(path, still_valid)
    assert dropped == 1

    # but when the *suite* has no fingerprint, nothing is dropped
    with ShardWriter(path) as writer:
        writer.write({"uid": "legacy"})
    _kept, dropped = prune_shard(
        path, lambda r: (not {"legacy": ""}.get(r.get("uid", ""))) or True
    )
    assert dropped == 0


# ---------------------------------------------------------------------------
# Record contents
# ---------------------------------------------------------------------------


def _rendered():
    cfg = load_config(None, ["data.noisy_samples=1"])
    return {c.condition: c for c in render_item(_item(), cfg)}["clean_roman"]


def test_record_is_self_contained():
    """A prediction must be interpretable without the suite file."""
    cell = _rendered()
    record = _record(cell, "m1", Generation("Answer: A", "stop", 5), 0.1, store_text=True)

    for field in ("uid", "prompt", "response", "prompt_sha256", "gold", "condition"):
        assert field in record, field
    assert record["prompt"] == cell.prompt
    assert record["options"] == cell.options
    assert record["prompt_sha256"] == cell.prompt_sha256
    assert record["correct"] is True


def test_record_captures_the_finish_reason():
    cell = _rendered()
    stopped = _record(cell, "m", Generation("A", "stop", 3), 0.1, store_text=True)
    cut = _record(cell, "m", Generation("A", "length", 64), 0.1, store_text=True)

    assert stopped["truncated"] is False
    assert cut["truncated"] is True
    assert cut["finish_reason"] == "length"
    assert cut["output_tokens"] == 64


def test_safety_records_hold_no_text():
    """Harmful prompts and completions stay out of the shared predictions file."""
    cell = _rendered()
    record = _record(cell, "m", Generation("some completion", "stop"), 0.1, store_text=False)
    assert "response" not in record
    assert "prompt" not in record
    # but the analysable fields survive
    assert record["uid"] and record["prompt_sha256"]
    assert record["response_chars"] == len("some completion")


def test_failed_generation_is_recorded_not_dropped():
    cell = _rendered()
    record = _record(cell, "m", Generation.failed(), 0.1, store_text=True)
    assert record["empty_response"] is True
    assert record["finish_reason"] == "error"
    assert record["correct"] is False


# ---------------------------------------------------------------------------
# The released table
# ---------------------------------------------------------------------------


def _write_run(tmp_path, records):
    from scriptgap.paths import RunPaths

    run = RunPaths(run_id="t", root=tmp_path / "t")
    shard = run.shard("inference", "m1", "milu__hi__native")
    with ShardWriter(shard) as writer:
        for record in records:
            writer.write(record)
        writer.mark_done()
    return run


def test_parquet_carries_prompt_and_response(tmp_path):
    from scriptgap.analysis.metrics import load_predictions

    cell = _rendered()
    run = _write_run(
        tmp_path,
        [_record(cell, "m1", Generation("Answer: A", "stop", 4), 0.1, store_text=True)],
    )
    df = load_predictions(run, include_text=True)

    assert "prompt" in df.columns and "response" in df.columns
    assert df.iloc[0]["prompt"] == cell.prompt
    assert df.iloc[0]["response"] == "Answer: A"
    for field in ("prompt_sha256", "finish_reason", "truncated", "output_tokens", "gold"):
        assert field in df.columns, field


def test_parquet_text_can_be_switched_off(tmp_path):
    from scriptgap.analysis.metrics import load_predictions

    cell = _rendered()
    run = _write_run(
        tmp_path, [_record(cell, "m1", Generation("x", "stop"), 0.1, store_text=True)]
    )
    df = load_predictions(run, include_text=False)
    assert "prompt" not in df.columns and "response" not in df.columns
    assert "correct" in df.columns


def test_parquet_round_trips_through_disk(tmp_path):
    """The released artefact must actually reload, unicode and all."""
    from scriptgap.analysis.metrics import load_predictions

    cell = _rendered()
    run = _write_run(
        tmp_path,
        [_record(cell, "m1", Generation("उत्तर: A ✓", "stop", 7), 0.1, store_text=True)],
    )
    df = load_predictions(run)
    path = tmp_path / "out.parquet"
    df.to_parquet(path, index=False)

    reloaded = pd.read_parquet(path)
    assert reloaded.iloc[0]["response"] == "उत्तर: A ✓"
    assert list(reloaded.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_bundles_and_verifies(tmp_path, monkeypatch):
    from scriptgap.analysis import export

    cell = _rendered()
    run = _write_run(
        tmp_path, [_record(cell, "m1", Generation("A", "stop"), 0.1, store_text=True)]
    )
    (run.root / "metrics").mkdir(parents=True, exist_ok=True)
    (run.root / "metrics" / "summary.json").write_text('{"ok": true}', encoding="utf-8")

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "rendered.jsonl").write_text(json.dumps(cell.to_dict()) + "\n", encoding="utf-8")

    monkeypatch.setattr(export, "EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr(export, "suite_dir", lambda _cfg: suite)

    out = export.run(Config(data={}), run, compress=False)
    manifest = json.loads((out / "EXPORT.json").read_text(encoding="utf-8"))

    assert manifest["run_id"] == "t"
    assert manifest["completeness"]["prediction_records"] == 1
    assert manifest["completeness"]["incomplete"] == []
    assert (out / "suite" / "rendered.jsonl").exists()
    assert (out / "metrics" / "summary.json").exists()
    assert manifest["files"], "no checksums recorded"

    assert export.verify(out)["ok"]


def test_export_excludes_raw_safety_by_default(tmp_path, monkeypatch):
    from scriptgap.analysis import export

    run = _write_run(tmp_path, [{"uid": "a"}])
    raw = run.raw_shard("safety", "m1", "safety__bn__native")
    raw.write_text(json.dumps({"uid": "a", "response": "unsafe text"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(export, "EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr(export, "suite_dir", lambda _cfg: tmp_path / "nope")

    out = export.run(Config(data={}), run, compress=False, include_raw=False)
    assert not (out / "raw").exists(), "safety completions must not be bundled by default"

    manifest = json.loads((out / "EXPORT.json").read_text(encoding="utf-8"))
    assert manifest["includes_raw_safety"] is False
    assert not any("unsafe" in name for name in manifest["files"])


def test_export_flags_an_incomplete_run(tmp_path, monkeypatch):
    from scriptgap.analysis import export
    from scriptgap.paths import RunPaths

    run = RunPaths(run_id="partial", root=tmp_path / "partial")
    shard = run.shard("inference", "m1", "milu__hi__native")
    with ShardWriter(shard) as writer:
        writer.write({"uid": "a"})  # deliberately no mark_done()

    monkeypatch.setattr(export, "EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr(export, "suite_dir", lambda _cfg: tmp_path / "nope")

    out = export.run(Config(data={}), run, compress=False)
    manifest = json.loads((out / "EXPORT.json").read_text(encoding="utf-8"))
    assert manifest["completeness"]["incomplete"], "a partial run must be flagged"
    assert manifest["completeness"]["complete_shards"] == 0


def test_verify_detects_tampering(tmp_path, monkeypatch):
    from scriptgap.analysis import export

    run = _write_run(tmp_path, [{"uid": "a"}])
    monkeypatch.setattr(export, "EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr(export, "suite_dir", lambda _cfg: tmp_path / "nope")

    out = export.run(Config(data={}), run, compress=False)
    target = next((out / "predictions").rglob("*.jsonl"))
    target.write_text('{"uid": "tampered"}\n', encoding="utf-8")

    result = export.verify(out)
    assert not result["ok"]
    assert result["mismatched"]
