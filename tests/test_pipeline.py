"""Config, rendering, sharding and metrics — the plumbing between stages.

Uses synthetic base items so it runs offline with no dataset download.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scriptgap.config import CONDITIONS, MULTI_SAMPLE_CONDITIONS, Config, apply_overrides, load_config
from scriptgap.data.build import render_item
from scriptgap.data.schema import BaseItem, RenderedItem, script_for
from scriptgap.eval.run_inference import build_shards


def _cfg(**overrides) -> Config:
    base = {
        "seed": 11,
        "data": {"noisy_samples": 2, "roundtrip_threshold": 0.9},
        "transforms": {
            "clean_roman": {"scheme": "ISO", "strip_diacritics": True},
            "noisy_roman": {"schwa_deletion": 0.5, "long_vowel_collapse": 0.5},
            "native_noisy": {"matra_drop": 0.3, "nukta_drop": 0.5},
            "codemix": {"insertion_rate": 0.9, "max_insertions": 6},
        },
    }
    base.update(overrides)
    return Config(data=base)


def _item(language: str = "hi", task: str = "milu") -> BaseItem:
    return BaseItem(
        item_id="item0001",
        task=task,
        language=language,
        script=script_for(language),
        prompt="भारत की राजधानी क्या है?",
        answer="A",
        answer_type="mcq",
        options=["नई दिल्ली", "मुंबई", "कोलकाता", "चेन्नई"],
        meta={"subject": "geography"},
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_loads_and_has_five_conditions():
    cfg = load_config()
    assert len(CONDITIONS) == 5
    assert cfg.get("seed") is not None
    assert cfg.get("data.milu.languages")


def test_dotted_overrides_coerce_types():
    out = apply_overrides({"a": {"b": 1}}, ["a.b=7", "a.c=true", "a.d=[1,2]"])
    assert out["a"] == {"b": 7, "c": True, "d": [1, 2]}


def test_config_require_raises_on_missing_key():
    with pytest.raises(SystemExit):
        _cfg().require("nope.missing")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_produces_every_condition_with_the_right_sample_counts():
    cfg = _cfg()
    rendered = list(render_item(_item(), cfg))
    n_samples = cfg.get("data.noisy_samples")

    by_condition: dict[str, int] = {}
    for cell in rendered:
        by_condition[cell.condition] = by_condition.get(cell.condition, 0) + 1

    assert set(by_condition) == set(CONDITIONS)
    for condition in CONDITIONS:
        expected = n_samples if condition in MULTI_SAMPLE_CONDITIONS else 1
        assert by_condition[condition] == expected, condition


def test_render_keeps_uids_unique_and_stable():
    cfg = _cfg()
    first = list(render_item(_item(), cfg))
    second = list(render_item(_item(), cfg))
    uids = [c.uid for c in first]
    assert len(uids) == len(set(uids)), "uid collision would silently drop cells on resume"
    assert uids == [c.uid for c in second]
    assert [c.prompt for c in first] == [c.prompt for c in second]


def test_render_preserves_the_gold_answer_across_conditions():
    """Transforms must never touch the answer key."""
    for cell in render_item(_item(), cfg=_cfg()):
        assert cell.answer == "A"
        assert cell.answer_type == "mcq"
        assert cell.options and len(cell.options) == 4


def test_native_condition_is_untouched():
    native = [c for c in render_item(_item(), _cfg()) if c.condition == "native"]
    assert len(native) == 1
    assert native[0].prompt == _item().prompt
    assert native[0].options == _item().options


def test_roman_conditions_actually_change_the_prompt():
    cells = {c.condition: c for c in render_item(_item(), _cfg()) if c.sample_idx == 0}
    assert cells["clean_roman"].prompt != cells["native"].prompt
    assert cells["clean_roman"].options != cells["native"].options


def test_gate_flag_propagates_to_every_condition_of_an_item():
    cells = list(render_item(_item(), _cfg()))
    flags = {c.roundtrip_pass for c in cells}
    assert len(flags) == 1, "the gate must be an item-level decision, not per-condition"


def test_rendered_item_round_trips_through_dict():
    cell = next(iter(render_item(_item(), _cfg())))
    assert RenderedItem.from_dict(cell.to_dict()) == cell


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def _cells() -> list[RenderedItem]:
    cfg = _cfg()
    out: list[RenderedItem] = []
    for language in ("hi", "bn"):
        for task in ("milu", "math"):
            item = _item(language=language, task=task)
            item.item_id = f"{task}-{language}"
            out.extend(render_item(item, cfg))
    return out


def test_shards_are_keyed_by_task_language_condition():
    shards = build_shards(_cells())
    keys = {s.key for s in shards}
    assert "milu__hi__native" in keys
    assert "math__bn__clean_roman" in keys
    # 2 languages x 2 tasks x 5 conditions
    assert len(shards) == 2 * 2 * 5


def test_shard_filters_compose():
    shards = build_shards(_cells(), tasks=["milu"], conditions=["native"], languages=["hi"])
    assert len(shards) == 1
    assert shards[0].key == "milu__hi__native"


def test_shard_limit_caps_items():
    shards = build_shards(_cells(), conditions=["noisy_roman"], limit_per_shard=1)
    assert all(len(s.items) == 1 for s in shards)


def test_shard_items_are_deterministically_ordered():
    a = build_shards(_cells())
    b = build_shards(list(reversed(_cells())))
    assert [s.key for s in a] == [s.key for s in b]
    assert [[i.uid for i in s.items] for s in a] == [[i.uid for i in s.items] for s in b]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_gap_frame_computes_native_minus_condition():
    from scriptgap.analysis.metrics import _accuracy_frame, _gap_frame

    df = pd.DataFrame(
        [
            {"model": "m", "task": "milu", "language": "hi", "condition": "native",
             "correct": True, "parse_failed": False, "empty_response": False},
            {"model": "m", "task": "milu", "language": "hi", "condition": "native",
             "correct": True, "parse_failed": False, "empty_response": False},
            {"model": "m", "task": "milu", "language": "hi", "condition": "clean_roman",
             "correct": True, "parse_failed": False, "empty_response": False},
            {"model": "m", "task": "milu", "language": "hi", "condition": "clean_roman",
             "correct": False, "parse_failed": False, "empty_response": False},
        ]
    )
    accuracy = _accuracy_frame(df, n_boot=50, seed=0)
    gaps = _gap_frame(accuracy)

    assert set(gaps["condition"]) == {"clean_roman"}, "the baseline must be excluded"
    row = gaps.iloc[0]
    assert row["gap"] == pytest.approx(0.5)
    assert row["relative_gap"] == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_mean():
    from scriptgap.analysis.metrics import bootstrap_ci

    values = [1.0] * 60 + [0.0] * 40
    lo, hi = bootstrap_ci(values, n_samples=500, seed=1)
    assert lo is not None and hi is not None
    assert lo < 0.6 < hi


def test_bootstrap_ci_returns_none_for_tiny_samples():
    from scriptgap.analysis.metrics import bootstrap_ci

    assert bootstrap_ci([1.0], n_samples=100) == (None, None)


def test_h2_variance_needs_specialised_models(caplog):
    """H2 is a variance claim; with fewer than two models it must warn."""
    from scriptgap.analysis.metrics import _h2_variance

    gaps = pd.DataFrame(
        [{"model": "only-one", "task": "milu", "language": "hi",
          "condition": "clean_roman", "gap": 0.2}]
    )
    out = _h2_variance(gaps, ["only-one"])
    assert len(out) == 1
    assert "variance" in caplog.text.lower()


def test_repeated_setup_logging_does_not_duplicate_output(tmp_path, monkeypatch):
    """Regression: `scriptgap all` runs six stages in one process.

    Each stage calls setup_logging. Without handler tagging, every call appended
    another console+file handler and each line was emitted once per stage
    already started — six copies of every log line by the end of the pipeline.
    """
    import logging as _logging

    from scriptgap import logging_utils, paths

    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    run = paths.RunPaths.create(run_id="logtest")

    root = _logging.getLogger()
    saved = list(root.handlers)
    for handler in saved:
        root.removeHandler(handler)
    try:
        counts = []
        for stage in ("build-data", "inference", "fertility", "metrics", "figures"):
            logging_utils.setup_logging(stage, run)
            counts.append(len(root.handlers))
        # console + global file + exactly one stage file, no matter how many
        # stages have run.
        assert counts[-1] == counts[0] == 3, counts
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)


def test_h2_variance_reports_spread_across_specialised_models():
    from scriptgap.analysis.metrics import _h2_variance

    gaps = pd.DataFrame(
        [
            {"model": "spec-a", "task": "milu", "language": "hi", "condition": "clean_roman", "gap": 0.05},
            {"model": "spec-b", "task": "milu", "language": "hi", "condition": "clean_roman", "gap": 0.35},
            {"model": "generalist", "task": "milu", "language": "hi", "condition": "clean_roman", "gap": 0.9},
        ]
    )
    out = _h2_variance(gaps, ["spec-a", "spec-b"])
    row = out.iloc[0]
    assert row["n_models"] == 2, "the generalist must not enter the H2 statistic"
    assert row["gap_range"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Config completeness
# ---------------------------------------------------------------------------


def test_every_declared_noise_operator_has_a_probability():
    """An absent key silently disables an operator.

    `scriptgap calibrate` only emits the six operators it can measure from
    Dakshina's single-word lexicons; `char_swap` and `sms_isms` cannot be
    measured there. Pasting a calibration over the config without them would
    turn them off with no error.
    """
    from scriptgap.data.noise import ROMAN_OPS, probs_for_language

    cfg = load_config()
    declared = {name for name, _fn in ROMAN_OPS}
    for language in ("hi", "bn", "ta", "te", "mr", "unknown-language"):
        probs = probs_for_language(cfg, language, "noisy_roman")
        missing = declared - set(probs)
        assert not missing, f"{language} is missing operator(s): {sorted(missing)}"
        assert all(0.0 <= float(v) <= 1.0 for v in probs.values())


def test_per_language_blocks_inherit_unmeasurable_operators():
    """Per-language calibration must not drop the hand-set operators."""
    from scriptgap.data.noise import probs_for_language

    cfg = load_config()
    globals_ = cfg.get("transforms.noisy_roman")
    for language in cfg.get("transforms.noisy_roman_by_language", {}):
        probs = probs_for_language(cfg, language, "noisy_roman")
        assert probs["char_swap"] == globals_["char_swap"]
        assert probs["sms_isms"] == globals_["sms_isms"]


def test_native_duplicate_conditions_match_the_renderer():
    """The metrics drop-list must cover every condition that can echo native.

    If a renderer starts returning the native prompt on a no-op and the metrics
    list is not updated, those cells score a guaranteed zero gap and drag the
    headline number toward zero.
    """
    from scriptgap.analysis.metrics import NATIVE_DUPLICATE_CONDITIONS
    from scriptgap.config import ALL_CONDITIONS

    for condition in NATIVE_DUPLICATE_CONDITIONS:
        assert condition in ALL_CONDITIONS, condition
    # noisy_roman is deliberately excluded: a no-op there equals the *clean*
    # romanisation, which is still a romanised observation.
    assert "noisy_roman" not in NATIVE_DUPLICATE_CONDITIONS
    assert "native" not in NATIVE_DUPLICATE_CONDITIONS


def test_uneven_truncation_is_flagged_as_a_confound(tmp_path, caplog):
    """Truncation that differs by condition manufactures a gap.

    A condition cut off at `max_tokens` more often than the baseline scores
    worse for a reason unrelated to orthography, and the difference is
    indistinguishable from a real register effect.
    """
    import logging

    from scriptgap.analysis import metrics
    from scriptgap.checkpoint import ShardWriter
    from scriptgap.paths import RunPaths

    run = RunPaths(run_id="trunc", root=tmp_path / "trunc")
    shard = run.shard("inference", "m1", "milu__hi__mixed")
    with ShardWriter(shard) as writer:
        # enough rows per condition that the spread cannot be sampling noise
        for i in range(2 * metrics.TRUNCATION_MIN_N + 20):
            # native truncates constantly, clean_roman never
            condition = "native" if i % 2 == 0 else "clean_roman"
            writer.write({
                "uid": f"u{i}", "item_id": f"i{i}", "model": "m1", "task": "milu",
                "language": "hi", "script": "Devanagari", "condition": condition,
                "sample_idx": 0, "answer_type": "mcq", "gold": "A", "parsed": "A",
                "correct": True, "parse_failed": False, "empty_response": False,
                "roundtrip_pass": True, "transform_applied": True,
                "prompt_sha256": "x", "finish_reason": "length" if condition == "native" else "stop",
                "truncated": condition == "native", "output_tokens": 8,
                "latency_s": 0.1, "response_chars": 5, "prompt": "p", "response": "A",
            })
        writer.mark_done()

    cfg = load_config(None, ["metrics.bootstrap_samples=20"])
    with caplog.at_level(logging.ERROR):
        summary = metrics.compute(cfg, run, specialised_models=[], gate=False)

    assert summary["truncation_confound_warning"] is True
    assert summary["truncation_spread_across_conditions"] == pytest.approx(1.0)
    assert "UNEVEN across conditions" in caplog.text


def test_even_truncation_is_not_flagged(tmp_path):
    from scriptgap.analysis import metrics
    from scriptgap.checkpoint import ShardWriter
    from scriptgap.paths import RunPaths

    run = RunPaths(run_id="trunc2", root=tmp_path / "trunc2")
    shard = run.shard("inference", "m1", "milu__hi__mixed")
    with ShardWriter(shard) as writer:
        for i in range(2 * metrics.TRUNCATION_MIN_N + 20):
            condition = "native" if i % 2 == 0 else "clean_roman"
            writer.write({
                "uid": f"u{i}", "item_id": f"i{i}", "model": "m1", "task": "milu",
                "language": "hi", "script": "Devanagari", "condition": condition,
                "sample_idx": 0, "answer_type": "mcq", "gold": "A", "parsed": "A",
                "correct": True, "parse_failed": False, "empty_response": False,
                "roundtrip_pass": True, "transform_applied": True,
                "prompt_sha256": "x", "finish_reason": "length" if i < 4 else "stop",
                "truncated": i < 4, "output_tokens": 8,
                "latency_s": 0.1, "response_chars": 5, "prompt": "p", "response": "A",
            })
        writer.mark_done()

    summary = metrics.compute(
        load_config(None, ["metrics.bootstrap_samples=20"]), run, specialised_models=[], gate=False
    )
    assert "truncation_confound_warning" not in summary


def test_truncation_confound_is_not_flagged_on_a_tiny_sample(tmp_path, caplog):
    """With 4 rows per condition, one truncation is a 25-point spread.

    Flagging that would fire on every smoke run and train the reader to ignore
    the alarm, which costs more than the alarm is worth.
    """
    import logging

    from scriptgap.analysis import metrics
    from scriptgap.checkpoint import ShardWriter
    from scriptgap.paths import RunPaths

    run = RunPaths(run_id="tiny", root=tmp_path / "tiny")
    shard = run.shard("inference", "m1", "milu__hi__mixed")
    with ShardWriter(shard) as writer:
        for i in range(8):  # 4 per condition
            condition = "native" if i % 2 == 0 else "clean_roman"
            writer.write({
                "uid": f"u{i}", "item_id": f"i{i}", "model": "m1", "task": "milu",
                "language": "hi", "script": "Devanagari", "condition": condition,
                "sample_idx": 0, "answer_type": "mcq", "gold": "A", "parsed": "A",
                "correct": True, "parse_failed": False, "empty_response": False,
                "roundtrip_pass": True, "transform_applied": True,
                "prompt_sha256": "x",
                "finish_reason": "length" if i == 0 else "stop",
                "truncated": i == 0, "output_tokens": 8,
                "latency_s": 0.1, "response_chars": 5, "prompt": "p", "response": "A",
            })
        writer.mark_done()

    with caplog.at_level(logging.INFO):
        summary = metrics.compute(
            load_config(None, ["metrics.bootstrap_samples=20"]),
            run, specialised_models=[], gate=False,
        )

    assert "truncation_confound_warning" not in summary
    assert summary["truncation_min_rows_per_condition"] == 4
    assert "too few to distinguish" in caplog.text
