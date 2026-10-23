"""Transform-layer invariants.

These matter because a bug here silently becomes a "finding": if the clean
romanisation is wrong, the H1 gap is an artefact; if the noisy transform never
fires, H3 measures nothing; if a transform leaks the answer or changes the
script it claims to preserve, the whole causal story collapses.
"""

from __future__ import annotations

import unicodedata

import pytest

from scriptgap.data import arabizi, codemix, noise, translit

SAMPLES = [
    ("Devanagari", "hi", "भारत की राजधानी नई दिल्ली है।"),
    ("Bengali", "bn", "বাংলাদেশের রাজধানী ঢাকা।"),
    ("Tamil", "ta", "இந்தியாவின் தலைநகரம் புது டெல்லி."),
    ("Telugu", "te", "భారతదేశ రాజధాని న్యూఢిల్లీ."),
]


def test_a_transliteration_backend_is_installed():
    assert translit.available_backends(), "no transliteration backend available"


@pytest.mark.parametrize("script,lang,text", SAMPLES)
def test_reversible_form_round_trips(script, lang, text):
    """The gate statistic must be high on well-formed input.

    This is the confound check from spec §4: if the tooling cannot recover the
    original, a measured drop may be our transliteration failing rather than
    the model.
    """
    result = translit.romanise(text, script, scheme="ISO", strip=True)
    assert result.roundtrip_score >= 0.90, (
        f"{lang}: round-trip {result.roundtrip_score:.3f} — transliteration is lossy"
    )


@pytest.mark.parametrize(
    "script,text",
    [
        # Digits: `2` round-trips through Aksharamukha as `২`/`२`. Without digit
        # normalisation this failed the gate on every numeric item, which is the
        # entire maths task.
        ("Bengali", "ভিনসেন্ট $2.50 দরে 3টি প্যাকেজ কিনলেন।"),
        ("Devanagari", "200 में से 5 भाग निकालें।"),
        # Embedded Latin: the reverse pass reads `lowercase` as romanised Indic
        # and returns `लॊwऎर्चसॆ` unless the span is masked.
        ("Devanagari", "अंग्रेज़ी अक्षर (lowercase) में लिखें।"),
        ("Devanagari", "JSON ऑब्जेक्ट के रूप में उत्तर दें।"),
        ("Bengali", "উত্তর JSON format এ দিন।"),
    ],
)
def test_gate_is_not_fooled_by_digits_or_embedded_latin(script, text):
    """Regression: the gate must measure transliteration, not its own artefacts.

    Both failure modes here silently discarded 25-50% of sound items, which
    would have shrunk the study without anyone noticing.
    """
    result = translit.romanise(text, script, scheme="ISO", strip=True)
    assert result.roundtrip_score >= 0.95, (
        f"gate score {result.roundtrip_score:.3f} on {text!r}"
    )


def test_embedded_latin_survives_into_the_emitted_string():
    """The masking is internal: the model must see the real English word."""
    result = translit.romanise("अक्षर (lowercase) में लिखें।", "Devanagari", strip=True)
    assert "lowercase" in result.emitted
    assert "lowercase" in result.reversible
    # no sentinel leaked into the string the model sees
    assert not any(0xE0000 <= ord(c) <= 0xE007F for c in result.emitted)
    assert not any(0xE000 <= ord(c) <= 0xF8FF for c in result.emitted)


@pytest.mark.parametrize("script", ["Bengali", "Devanagari", "Tamil", "Telugu"])
def test_sentinels_survive_both_directions(script):
    """The masking depends on this and nothing else.

    Private-use codepoints failed here: Aksharamukha's reverse pass reads an
    unmapped codepoint as an implicit vowel and turned U+E000 into a real
    letter, corrupting the text the mask was protecting.
    """
    for index in range(6):
        sentinel = chr(translit._SENTINEL_BASE + index)
        forward, _backend = translit.to_roman(sentinel, script, "ISO")
        assert sentinel in forward, f"{script}: sentinel lost going to roman"
        assert sentinel in translit.to_native(forward, script, "ISO"), (
            f"{script}: sentinel lost coming back"
        )


def test_many_latin_runs_do_not_corrupt_the_output():
    text = "यह " + " ".join(f"word{i}" for i in range(40)) + " है।"
    result = translit.romanise(text, "Devanagari", strip=True)
    for i in range(40):
        assert f"word{i}" in result.emitted


def test_canonical_normalisation_unifies_digits_and_danda():
    from scriptgap import textnorm

    assert textnorm.canonical("२०० है।") == textnorm.canonical("200 है.")
    assert textnorm.normalise_digits("৪২") == "42"


@pytest.mark.parametrize("script,lang,text", SAMPLES)
def test_clean_roman_is_ascii_and_differs(script, lang, text):
    result = translit.romanise(text, script, scheme="ISO", strip=True)
    assert result.emitted != text
    # De-diacritisation should leave a keyboard-typable string.
    letters = [c for c in result.emitted if c.isalpha()]
    assert letters, "clean romanisation produced no letters"
    assert all(ord(c) < 128 for c in letters), f"non-ASCII survived: {result.emitted!r}"


@pytest.mark.parametrize("script,lang,text", SAMPLES)
def test_native_noise_preserves_script(script, lang, text):
    """Condition 4 must stay in the native script — that is its whole point."""
    rng = noise.rng_for(1, text, "native_noisy", 0)
    out = noise.native_noise(text, script, rng, matra_drop=0.9, matra_swap=0.9, nukta_drop=0.9)
    base = noise._BLOCK_BASE[script]
    native_chars = [c for c in out if base <= ord(c) < base + 0x80]
    assert native_chars, f"{lang}: native noise destroyed the script"
    # No Latin letters should have appeared.
    assert not any("A" <= c <= "z" and c.isalpha() for c in out)


def test_noisy_roman_fires_at_high_probability():
    """With every operator at p=1 the string must change, or H3 measures nothing."""
    clean = "bharata ki rajadhani nai dilli hai."
    probs = {name: 1.0 for name, _ in noise.ROMAN_OPS}
    out, report = noise.noisy_roman(clean, noise.rng_for(2, "x", "noisy", 0), probs)
    assert out != clean
    assert report.changed
    assert report.operators_fired


def test_noisy_roman_is_a_no_op_at_zero_probability():
    clean = "bharata ki rajadhani nai dilli hai."
    out, report = noise.noisy_roman(clean, noise.rng_for(2, "x", "noisy", 0), {})
    assert out == clean
    assert not report.changed


def test_transforms_are_deterministic_given_the_seed():
    """Reproducibility: the same cell must regenerate byte-identically."""
    probs = {name: 0.5 for name, _ in noise.ROMAN_OPS}
    first = noise.noisy_roman("bharata ki rajadhani", noise.rng_for(9, "item", "noisy_roman", 1), probs)[0]
    second = noise.noisy_roman("bharata ki rajadhani", noise.rng_for(9, "item", "noisy_roman", 1), probs)[0]
    assert first == second
    different_sample = noise.noisy_roman(
        "bharata ki rajadhani", noise.rng_for(9, "item", "noisy_roman", 2), probs
    )[0]
    assert isinstance(different_sample, str)


def test_schwa_deletion_keeps_a_vowel_in_the_word():
    """`kamal` -> `kaml`, never `kml` — the word must stay pronounceable."""
    out = noise.op_schwa_deletion("kamala", noise.rng_for(3, "x", "y", 0), 1.0)
    assert any(c in "aeiou" for c in out), out


def test_sms_isms_only_touch_known_words():
    out = noise.op_sms_isms("you are the rajadhani", noise.rng_for(4, "x", "y", 0), 1.0)
    assert "rajadhani" in out  # not in the lexicon, must survive
    assert "u" in out.split() and "r" in out.split()


@pytest.mark.parametrize("lang", ["hi", "bn", "ta", "te"])
def test_codemix_lexicons_are_non_trivial(lang):
    assert len(codemix.lexicon_for(lang)) >= 30


def test_codemix_respects_the_free_morpheme_constraint():
    """A suffixed token must not be split: only bare matches are replaced.

    `समयसीमा` (deadline) contains the lexicon key `समय` (time) but is a single
    word, so replacing inside it would violate the free-morpheme constraint and
    produce ungrammatical output.
    """
    rng = noise.rng_for(5, "x", "codemix", 0)
    out, report = codemix.code_mix("समयसीमा बहुत जल्दी है", "hi", rng, 1.0, 6, discourse_rate=0.0)
    assert "समयसीमा" in out
    assert "time" not in out
    assert report.lexical == 0
    assert report.candidates == 0


def test_codemix_replaces_the_bare_form_of_the_same_word():
    """Control for the test above: the bare token *is* replaced."""
    rng = noise.rng_for(5, "x", "codemix", 0)
    out, report = codemix.code_mix("समय बहुत जल्दी है", "hi", rng, 1.0, 6, discourse_rate=0.0)
    assert "time" in out
    assert report.lexical == 1


def test_codemix_reports_zero_when_nothing_matches():
    rng = noise.rng_for(5, "x", "codemix", 0)
    out, report = codemix.code_mix("xyzzy plugh", "hi", rng, 1.0, 6, discourse_rate=0.0)
    assert out == "xyzzy plugh"
    assert not report.applied


def test_codemix_separates_lexical_from_discourse():
    rng = noise.rng_for(6, "x", "codemix", 0)
    _out, report = codemix.code_mix("समय कम है", "hi", rng, 1.0, 6, discourse_rate=1.0)
    assert report.lexical >= 1
    assert report.discourse == 1
    assert report.insertions == report.lexical + report.discourse


def test_arabizi_is_latin_and_uses_digit_conventions():
    out = arabizi.to_arabizi("ما هي عاصمة مصر؟")
    assert "؟" not in out and "?" in out
    assert "3" in out  # ain -> 3, the signature Arabizi convention
    assert not arabizi.has_arabic(out)


def test_arabizi_noisy_differs_from_clean_sometimes():
    text = "ما هي عاصمة جمهورية مصر العربية؟"
    clean = arabizi.to_arabizi(text)
    variants = {
        arabizi.to_arabizi(text, rng=noise.rng_for(i, text, "noisy", 0)) for i in range(10)
    }
    assert any(v != clean for v in variants)


def test_similarity_is_bounded_and_reflexive():
    assert translit.similarity("abc", "abc") == 1.0
    assert translit.similarity("", "abc") == 0.0
    assert 0.0 <= translit.similarity("abcd", "abce") <= 1.0


def test_strip_diacritics_normalises_to_nfc():
    out = translit.strip_diacritics("śrī bhārata")
    assert out == unicodedata.normalize("NFC", out)
    assert "sh" in out
