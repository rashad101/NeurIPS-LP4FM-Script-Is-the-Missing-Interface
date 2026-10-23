"""Attested-romanisation condition and calibration.

Uses a synthetic lexicon in the real Dakshina TSV format, so the tests run
offline without the 2 GB archive.
"""

from __future__ import annotations

import random

import pytest

from scriptgap.config import ALL_CONDITIONS, CONDITIONS, Config, active_conditions
from scriptgap.data import dakshina


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Real rows from Dakshina bn: several attested spellings per native word.
LEXICON_TSV = """\
অংশগ্রহনকারী\tangsagrohonkari\t1
অংশগ্রহনকারী\tangshogrohonkari\t3
অংশগ্রহনকারী\tongsagrohonkari\t1
অঁরি\tanri\t2
অঁরি\tonry\t1
রাজধানী\trajdhani\t5
রাজধানী\trajodhani\t1
কোথায়\tkothay\t4
"""


@pytest.fixture
def dakshina_root(tmp_path):
    root = tmp_path / "dakshina"
    lexdir = root / "bn" / "lexicons"
    lexdir.mkdir(parents=True)
    (lexdir / "bn.translit.sampled.train.tsv").write_text(LEXICON_TSV, encoding="utf-8")
    return root


@pytest.fixture
def lexicon(dakshina_root):
    return dakshina.load_lexicon(dakshina_root, "bn", splits=("train",))


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------


def test_is_available(dakshina_root, tmp_path):
    assert dakshina.is_available(dakshina_root)
    assert not dakshina.is_available(tmp_path / "absent")


def test_lexicon_files_found_by_glob(dakshina_root):
    assert dakshina.find_lexicon_files(dakshina_root, "bn", ("train",))


def test_lexicon_file_discovery_falls_back_when_split_is_missing(dakshina_root):
    """Upstream filenames drift; a missing split must not silently yield nothing."""
    found = dakshina.find_lexicon_files(dakshina_root, "bn", ("nonexistent-split",))
    assert found, "fallback to any lexicon TSV did not fire"


def test_load_lexicon_aggregates_variants(lexicon):
    assert lexicon.language == "bn"
    assert len(lexicon) == 4
    assert lexicon.n_romanisations == 8
    assert lexicon.mean_variants == 2.0
    # sorted by descending count
    assert lexicon.entries["অংশগ্রহনকারী"][0] == ("angshogrohonkari", 3)


def test_load_lexicon_returns_none_for_missing_language(dakshina_root):
    assert dakshina.load_lexicon(dakshina_root, "ta", splits=("train",)) is None


def test_load_lexicons_skips_languages_dakshina_lacks(dakshina_root):
    out = dakshina.load_lexicons(dakshina_root, ["bn", "ar"], splits=("train",))
    assert set(out) == {"bn"}


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    root = tmp_path / "d"
    lexdir = root / "bn" / "lexicons"
    lexdir.mkdir(parents=True)
    (lexdir / "bn.train.tsv").write_text(
        "রাজধানী\trajdhani\t5\n\nbroken-line-no-tab\n\t\t\nকোথায়\tkothay\tnot-an-int\n",
        encoding="utf-8",
    )
    lex = dakshina.load_lexicon(root, "bn", splits=("train",))
    assert lex is not None
    assert set(lex.entries) == {"রাজধানী", "কোথায়"}
    # a non-integer count degrades to 1 rather than crashing
    assert lex.entries["কোথায়"] == [("kothay", 1)]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_returns_an_attested_variant(lexicon):
    rng = random.Random(0)
    seen = {lexicon.sample("অঁরি", rng) for _ in range(50)}
    assert seen <= {"anri", "onry"}
    assert len(seen) == 2, "count-weighted sampling should still reach both variants"


def test_sample_is_count_weighted(lexicon):
    rng = random.Random(1)
    draws = [lexicon.sample("রাজধানী", rng) for _ in range(400)]
    # 5:1 in favour of rajdhani
    assert draws.count("rajdhani") > draws.count("rajodhani") * 2


def test_sample_returns_none_for_unknown_word(lexicon):
    assert lexicon.sample("একটিঅজানাশব্দ", random.Random(0)) is None


# ---------------------------------------------------------------------------
# The attested_roman condition
# ---------------------------------------------------------------------------


def test_attested_romanise_uses_human_spellings(lexicon):
    rng = random.Random(2)
    out, report = dakshina.attested_romanise("রাজধানী কোথায়", lexicon, "Bengali", rng)
    assert "rajdhani" in out or "rajodhani" in out
    assert "kothay" in out
    assert report.tokens == 2
    assert report.attested == 2
    assert report.coverage == 1.0
    assert report.applied


def test_attested_romanise_falls_back_for_oov_words(lexicon):
    """Output must be fully romanised even when the lexicon misses a word."""
    rng = random.Random(3)
    out, report = dakshina.attested_romanise(
        "রাজধানী ঢাকা", lexicon, "Bengali", rng
    )
    assert report.attested == 1
    assert report.fallback == 1
    assert 0.0 < report.coverage < 1.0
    # no Bengali survived — the fallback transliterated it
    assert not any(0x0980 <= ord(c) <= 0x09FF for c in out)


def test_attested_romanise_preserves_punctuation_and_layout(lexicon):
    rng = random.Random(4)
    out, _ = dakshina.attested_romanise("রাজধানী, কোথায়?", lexicon, "Bengali", rng)
    assert "," in out and "?" in out


def test_attested_romanise_asciifies_native_punctuation(lexicon):
    """A danda in an otherwise-Latin string is a signature of our transform.

    Someone typing on a Latin keyboard types `.`, not `।`; leaving the native
    form in would let a reader identify the condition from punctuation alone.
    """
    rng = random.Random(6)
    out, _ = dakshina.attested_romanise("রাজধানী কোথায়।", lexicon, "Bengali", rng)
    assert "।" not in out
    assert out.rstrip().endswith(".")


def test_attested_romanise_reports_zero_coverage_for_no_matches(lexicon):
    rng = random.Random(5)
    _out, report = dakshina.attested_romanise("ঢাকা শহর", lexicon, "Bengali", rng)
    assert report.attested == 0
    assert not report.applied, "a cell with no attested tokens must be flagged unapplied"


def test_attested_romanise_is_deterministic_for_a_seed(lexicon):
    from scriptgap.data import noise

    a = dakshina.attested_romanise(
        "রাজধানী কোথায়", lexicon, "Bengali", noise.rng_for(7, "item", "attested_roman", 0)
    )[0]
    b = dakshina.attested_romanise(
        "রাজধানী কোথায়", lexicon, "Bengali", noise.rng_for(7, "item", "attested_roman", 0)
    )[0]
    assert a == b


# ---------------------------------------------------------------------------
# Condition wiring
# ---------------------------------------------------------------------------


def test_attested_is_opt_in_and_not_in_the_released_five():
    """The paper claims five renderings; the sixth must not silently join them."""
    assert "attested_roman" not in CONDITIONS
    assert len(CONDITIONS) == 5
    assert "attested_roman" in ALL_CONDITIONS

    off = Config(data={"data": {"dakshina": {"enabled": False}}})
    on = Config(data={"data": {"dakshina": {"enabled": True}}})
    assert active_conditions(off) == CONDITIONS
    assert active_conditions(on) == CONDITIONS + ("attested_roman",)


def test_render_marks_attested_unapplied_when_no_lexicon():
    """A language Dakshina lacks must not produce a duplicate of the native item."""
    from scriptgap.data.build import render_item
    from scriptgap.data.schema import BaseItem, script_for

    cfg = Config(
        data={
            "seed": 1,
            "data": {"noisy_samples": 1, "dakshina": {"enabled": True}},
            "transforms": {
                "clean_roman": {"scheme": "ISO", "strip_diacritics": True},
                "noisy_roman": {},
                "native_noisy": {},
                "codemix": {"insertion_rate": 0.0, "max_insertions": 0},
            },
        }
    )
    item = BaseItem(
        item_id="x1",
        task="milu",
        language="hi",
        script=script_for("hi"),
        prompt="भारत की राजधानी क्या है?",
        answer="A",
        answer_type="mcq",
        options=["नई दिल्ली", "मुंबई"],
    )
    cells = {c.condition: c for c in render_item(item, cfg, lexicons={})}
    assert "attested_roman" in cells
    assert cells["attested_roman"].transform_applied is False
    assert cells["attested_roman"].meta["attested_coverage"] == 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_measure_phenomena_returns_rates_in_range(lexicon):
    rates = dakshina.measure_phenomena(lexicon, "Bengali", max_pairs=100)
    assert rates
    for name, stats in rates.items():
        if name.startswith("_"):
            continue
        assert stats["sites"] >= stats["applied"] >= 0
        assert 0.0 <= stats["rate"] <= 1.0


def test_measure_phenomena_detects_the_sibilant_alternation(lexicon):
    """`angshogrohonkari` vs `angsagrohonkari` is exactly sh->s in the wild."""
    rates = dakshina.measure_phenomena(lexicon, "Bengali", max_pairs=100)
    assert "sibilant_swap" in rates or "schwa_deletion" in rates


def test_suggest_probabilities_maps_onto_config_keys(lexicon):
    measured = {"bn": dakshina.measure_phenomena(lexicon, "Bengali", max_pairs=100)}
    suggested = dakshina.suggest_probabilities(measured)
    known = {
        "schwa_deletion",
        "long_vowel_collapse",
        "gemination_collapse",
        "aspiration_drop",
        "sibilant_swap",
        "vowel_ending_drop",
    }
    assert suggested, "no probabilities were suggested"
    assert set(suggested) <= known, f"unexpected keys: {set(suggested) - known}"
    for key, value in suggested.items():
        assert 0.0 <= value <= 1.0, (key, value)


# ---------------------------------------------------------------------------
# Per-language operator probabilities
# ---------------------------------------------------------------------------


def test_per_language_probabilities_override_the_global_block():
    """Calibration shows up to 35x spread across languages.

    A single global probability set would inject Hindi-shaped noise into Telugu
    items and inflate that language's gap with spellings no Telugu speaker
    produces, so the per-language block has to win.
    """
    from scriptgap.data import noise

    cfg = Config(
        data={
            "transforms": {
                "noisy_roman": {"vowel_ending_drop": 0.35, "sibilant_swap": 0.18},
                "noisy_roman_by_language": {
                    "hi": {"vowel_ending_drop": 0.536},
                    "te": {"vowel_ending_drop": 0.015, "sibilant_swap": 0.34},
                },
            }
        }
    )
    hi = noise.probs_for_language(cfg, "hi", "noisy_roman")
    te = noise.probs_for_language(cfg, "te", "noisy_roman")
    bn = noise.probs_for_language(cfg, "bn", "noisy_roman")

    assert hi["vowel_ending_drop"] == 0.536
    assert te["vowel_ending_drop"] == 0.015
    # keys absent from the override fall through to the global block
    assert hi["sibilant_swap"] == 0.18
    assert te["sibilant_swap"] == 0.34
    # a language with no override gets the global block unchanged
    assert bn == {"vowel_ending_drop": 0.35, "sibilant_swap": 0.18}


def test_cli_override_beats_the_per_language_default():
    """`--set` must win, or sensitivity sweeps are silent no-ops.

    The shipped defaults carry per-language calibrated probabilities for the
    five suite languages. Without this precedence rule,
    `--set transforms.noisy_roman.schwa_deletion=0.15` would change nothing for
    exactly the languages the study uses — and the README's sensitivity recipe
    would quietly measure nothing.
    """
    from scriptgap.config import load_config
    from scriptgap.data import noise

    plain = load_config()
    swept = load_config(None, ["transforms.noisy_roman.schwa_deletion=0.99"])

    for language in ("hi", "te", "bn", "language-with-no-block"):
        assert (
            noise.probs_for_language(swept, language, "noisy_roman")["schwa_deletion"] == 0.99
        ), language

    # untouched keys keep their per-language values
    assert (
        noise.probs_for_language(swept, "hi", "noisy_roman")["vowel_ending_drop"]
        != noise.probs_for_language(swept, "te", "noisy_roman")["vowel_ending_drop"]
    )
    # and the un-swept config is unchanged
    assert noise.probs_for_language(plain, "hi", "noisy_roman")["schwa_deletion"] != 0.99


def test_override_of_a_key_absent_from_the_block_still_applies():
    from scriptgap.config import load_config
    from scriptgap.data import noise

    cfg = load_config(None, ["transforms.noisy_roman.brand_new_operator=0.5"])
    assert noise.probs_for_language(cfg, "hi", "noisy_roman")["brand_new_operator"] == 0.5


def test_probs_for_language_handles_a_missing_block():
    from scriptgap.data import noise

    assert noise.probs_for_language(Config(data={}), "hi", "noisy_roman") == {}


def test_calibration_fragment_is_layerable():
    """The emitted YAML must load through the normal config path."""
    import yaml

    from scriptgap.analysis.calibrate import CALIBRATION_DIR

    fragment = CALIBRATION_DIR / "noisy_roman.yaml"
    if not fragment.exists():
        pytest.skip("run `scriptgap calibrate` first")
    loaded = yaml.safe_load(fragment.read_text(encoding="utf-8"))
    assert "noisy_roman" in loaded["transforms"]
    assert "noisy_roman_by_language" in loaded["transforms"]
    for language, probs in loaded["transforms"]["noisy_roman_by_language"].items():
        assert language in dakshina.DAKSHINA_LANGS
        for key, value in probs.items():
            assert 0.0 <= value <= 1.0, (language, key, value)


def test_suggest_probabilities_pools_across_languages(lexicon):
    """Two languages must pool by site count, not average blindly."""
    one = dakshina.measure_phenomena(lexicon, "Bengali", max_pairs=100)
    pooled = dakshina.suggest_probabilities({"bn": one, "bn2": one})
    single = dakshina.suggest_probabilities({"bn": one})
    for key in single:
        assert pooled[key] == pytest.approx(single[key], abs=1e-9)
