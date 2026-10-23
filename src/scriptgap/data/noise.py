"""Conditions 3 and 4 — ad-hoc romanisation noise, and script-internal noise.

**Condition 3** (``noisy_roman``) models how people actually romanise: schwa
deletion, long/short vowel collapse, gemination collapse, aspiration drop,
sibilant confusion, chat spellings and the odd transposition. The operators run
over the *reversible* ISO form, where ``ā`` and ``a`` or ``ś`` and ``s`` are
still distinguishable, and the result is de-diacritised afterwards — so a cell
where nothing fired is byte-identical to the clean romanisation rather than to
something arbitrary.

Their probabilities are **fitted, not hand-set**: ``scriptgap calibrate``
measures each phenomenon against Dakshina's native-speaker romanisations, and
the rates turn out to differ by up to 35x across languages, so they are applied
per language. See :func:`probs_for_language` for the precedence rule.

**Condition 4** (``native_noise``) is the control that makes the causal claim
work: script-internal perturbation — dropped or swapped matras, lost nuktas,
anusvara confusion, dropped viramas — which isolates "misspelled" from
"different script". Without it, a drop on condition 3 could be dismissed as
generic typo-robustness. It is *not* fitted: Dakshina provides native->roman
pairs, so it can calibrate condition 3 but there is no comparable corpus of
attested native-script misspellings.

Every operator draws from a per-cell seeded RNG (:func:`rng_for`), so any single
cell regenerates byte-identically.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import textnorm
from ..config import Config

#: Word runs. Uses the shared tokeniser so combining marks — Indic matras, and
#: ISO-15919's decomposed ``r̥`` — stay inside their word.
_WORD = textnorm.WORD_TOKEN

#: Vowels in both the ISO and the de-diacritised registers.
_VOWELS = set("aeiouāīūēōĕŏạẹịọụ")

#: Consonants that carry an aspirated counterpart written ``Ch`` in ISO.
_ASPIRABLE = "bcdgjkptṭḍ"


def rng_for(seed: Any, item_id: str, condition: str, sample_idx: int) -> random.Random:
    """A per-cell RNG. The same four inputs always give the same stream."""
    return random.Random(f"{seed}␟{item_id}␟{condition}␟{sample_idx}")


def _has_vowel(word: str) -> bool:
    return any(c.lower() in _VOWELS for c in word)


def _per_word(text: str, fn: Callable[[str], str]) -> str:
    return _WORD.sub(lambda m: fn(m.group(0)), text)


# ---------------------------------------------------------------------------
# Condition 3 operators
# ---------------------------------------------------------------------------


def op_schwa_deletion(text: str, rng: random.Random, p: float) -> str:
    """Drop word-internal schwa: ``kamala`` -> ``kamla``, never ``kml``.

    The word has to stay pronounceable, so a deletion that would leave no vowel
    at all is skipped — an unpronounceable string is not a spelling anyone
    produces, and it would make the condition measure gibberish tolerance.
    """

    def apply(word: str) -> str:
        chars = list(word)
        # Right to left, so earlier indices stay valid as we delete.
        for index in range(len(chars) - 2, 0, -1):
            if chars[index].lower() != "a":
                continue
            before, after = chars[index - 1].lower(), chars[index + 1].lower()
            if before in _VOWELS or after in _VOWELS:
                continue
            if rng.random() >= p:
                continue
            candidate = chars[:index] + chars[index + 1 :]
            if _has_vowel("".join(candidate)):
                chars = candidate
        return "".join(chars)

    return _per_word(text, apply)


def op_vowel_ending_drop(text: str, rng: random.Random, p: float) -> str:
    """Drop a word-final vowel: ``bharata`` -> ``bharat``.

    Calibration found this to be the dominant word-final phenomenon, and the one
    the hand-set defaults had set *too low* (0.54 in Hindi against a configured
    0.20).
    """

    def apply(word: str) -> str:
        if len(word) < 3 or word[-1].lower() not in _VOWELS:
            return word
        if rng.random() >= p:
            return word
        shortened = word[:-1]
        return shortened if _has_vowel(shortened) else word

    return _per_word(text, apply)


_LONG_VOWELS = {"ā": "a", "ī": "i", "ū": "u", "ē": "e", "ō": "o"}
_LONG_DIGRAPHS = (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"))


def op_long_vowel_collapse(text: str, rng: random.Random, p: float) -> str:
    """Collapse a long vowel onto its short partner (``ā``/``aa`` -> ``a``)."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        matched = False
        for digraph, replacement in _LONG_DIGRAPHS:
            if text[index : index + 2].lower() == digraph:
                out.append(replacement if rng.random() < p else text[index : index + 2])
                index += 2
                matched = True
                break
        if matched:
            continue
        short = _LONG_VOWELS.get(char.lower())
        if short is not None and rng.random() < p:
            out.append(short.upper() if char.isupper() else short)
        else:
            out.append(char)
        index += 1
    return "".join(out)


def op_gemination_collapse(text: str, rng: random.Random, p: float) -> str:
    """``dilli`` -> ``dili`` — a doubled consonant written once."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        doubled = (
            char.isalpha()
            and char.lower() == following.lower()
            and char.lower() not in _VOWELS
        )
        if doubled and rng.random() < p:
            out.append(char)
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def op_aspiration_drop(text: str, rng: random.Random, p: float) -> str:
    """``bharata`` -> ``barata``: the aspirate written without its ``h``."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if char.lower() in _ASPIRABLE and following.lower() == "h" and rng.random() < p:
            out.append(char)
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


_SIBILANTS = {"ś": "s", "ṣ": "s", "Ś": "S", "Ṣ": "S"}


def op_sibilant_swap(text: str, rng: random.Random, p: float) -> str:
    """``ś``/``ṣ``/``sh`` <-> ``s`` — the commonest attested confusion."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if text[index : index + 2].lower() == "sh" and rng.random() < p:
            out.append(char)  # keep the case of the leading s
            index += 2
            continue
        plain = _SIBILANTS.get(char)
        if plain is not None and rng.random() < p:
            out.append(plain)
        else:
            out.append(char)
        index += 1
    return "".join(out)


def op_char_swap(text: str, rng: random.Random, p: float) -> str:
    """Adjacent transposition — ordinary typing noise, not a phonology."""

    def apply(word: str) -> str:
        if len(word) < 3 or rng.random() >= p:
            return word
        cut = rng.randrange(len(word) - 1)
        chars = list(word)
        chars[cut], chars[cut + 1] = chars[cut + 1], chars[cut]
        return "".join(chars)

    return _per_word(text, apply)


#: Chat spellings. Not fitted — Dakshina's lexicons are single-word native->roman
#: pairs, which cannot exhibit English chat forms, so this stays hand-set.
SMS_LEXICON: dict[str, str] = {
    "you": "u",
    "your": "ur",
    "you're": "ur",
    "are": "r",
    "please": "plz",
    "thanks": "thx",
    "thank": "thx",
    "and": "n",
    "to": "2",
    "too": "2",
    "for": "4",
    "be": "b",
    "see": "c",
    "why": "y",
    "what": "wat",
    "that": "dat",
    "the": "d",
    "this": "dis",
    "with": "wid",
    "because": "bcoz",
    "before": "b4",
    "people": "ppl",
    "message": "msg",
    "tomorrow": "tmrw",
    "okay": "k",
    "good": "gud",
    "night": "nite",
    "right": "rite",
    "great": "gr8",
    "later": "l8r",
    "nahi": "nhi",
    "nahin": "nhi",
    "hai": "h",
    "kya": "kia",
    "kaise": "kese",
    "mujhe": "muje",
    "bahut": "bht",
    "kuch": "kch",
}


def op_sms_isms(text: str, rng: random.Random, p: float) -> str:
    """Rewrite known chat spellings only.

    Restricted to a lexicon on purpose: a general character-level rule would
    mangle the item's content words, and the resulting drop would measure our
    own corruption rather than a register users produce.
    """

    def apply(word: str) -> str:
        replacement = SMS_LEXICON.get(word.lower())
        if replacement is None or rng.random() >= p:
            return word
        return replacement.upper() if word.isupper() else replacement

    return _per_word(text, apply)


#: Declared operators, in application order. ``probs_for_language`` must be able
#: to supply a probability for every name here; an absent key silently disables
#: the operator, which is why ``tests/test_pipeline.py`` asserts completeness.
ROMAN_OPS: tuple[tuple[str, Callable[[str, random.Random, float], str]], ...] = (
    ("aspiration_drop", op_aspiration_drop),
    ("sibilant_swap", op_sibilant_swap),
    ("gemination_collapse", op_gemination_collapse),
    ("long_vowel_collapse", op_long_vowel_collapse),
    ("schwa_deletion", op_schwa_deletion),
    ("vowel_ending_drop", op_vowel_ending_drop),
    ("sms_isms", op_sms_isms),
    ("char_swap", op_char_swap),
)


@dataclass
class NoiseReport:
    """Which operators actually fired on one cell."""

    changed: bool = False
    operators_fired: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return self.changed


def noisy_roman(
    text: str,
    rng: random.Random,
    probs: dict[str, float],
) -> tuple[str, NoiseReport]:
    """Apply the condition-3 operators. Returns ``(text, report)``.

    An operator with no probability in ``probs`` is **off**, not defaulted — the
    config is the single source of truth for what is enabled.
    """
    report = NoiseReport()
    out = text
    for name, operator in ROMAN_OPS:
        probability = float(probs.get(name, 0.0) or 0.0)
        if probability <= 0.0:
            continue
        before = out
        out = operator(out, rng, probability)
        if out != before:
            report.operators_fired.append(name)
    report.changed = out != text
    return out, report


# ---------------------------------------------------------------------------
# Condition 4 — script-internal noise
# ---------------------------------------------------------------------------

#: Base codepoint of each Brahmic block. The blocks are laid out in parallel, so
#: matras, virama, nukta and anusvara sit at the same offsets in each.
_BLOCK_BASE: dict[str, int] = {
    "Devanagari": 0x0900,
    "Bengali": 0x0980,
    "Gurmukhi": 0x0A00,
    "Gujarati": 0x0A80,
    "Oriya": 0x0B00,
    "Tamil": 0x0B80,
    "Telugu": 0x0C00,
    "Kannada": 0x0C80,
    "Malayalam": 0x0D00,
    "Sinhala": 0x0D80,
}

_CANDRABINDU = 0x01
_ANUSVARA = 0x02
_NUKTA = 0x3C
_VIRAMA = 0x4D
#: Dependent vowel signs that are assigned in every block above.
_MATRA_OFFSETS = (0x3E, 0x3F, 0x40, 0x41, 0x42, 0x47, 0x48, 0x4B, 0x4C)


def _offset(char: str, base: int) -> int | None:
    point = ord(char) - base
    return point if 0 <= point < 0x80 else None


def native_noise(
    text: str,
    script: str,
    rng: random.Random,
    matra_drop: float = 0.0,
    matra_swap: float = 0.0,
    nukta_drop: float = 0.0,
    anusvara_swap: float = 0.0,
    char_swap: float = 0.0,
    virama_drop: float = 0.0,
) -> str:
    """Perturb text **within its own script** (condition 4).

    Never emits a character outside the source block, which is the whole point:
    condition 4 has to hold script constant so that condition 3's drop cannot be
    attributed to generic typo sensitivity.
    """
    base = _BLOCK_BASE.get(script)
    if base is None:
        return text

    matras = [chr(base + off) for off in _MATRA_OFFSETS]
    out: list[str] = []

    for char in text:
        offset = _offset(char, base)
        if offset is None:
            out.append(char)
            continue

        if offset in _MATRA_OFFSETS or 0x3E <= offset <= 0x4C:
            if rng.random() < matra_drop:
                continue
            if rng.random() < matra_swap:
                out.append(rng.choice(matras))
                continue
        elif offset == _NUKTA:
            if rng.random() < nukta_drop:
                continue
        elif offset == _VIRAMA:
            if rng.random() < virama_drop:
                continue
        elif offset in (_ANUSVARA, _CANDRABINDU):
            if rng.random() < anusvara_swap:
                flipped = _CANDRABINDU if offset == _ANUSVARA else _ANUSVARA
                out.append(chr(base + flipped))
                continue
        out.append(char)

    noised = "".join(out)
    if char_swap > 0.0:
        noised = _swap_native_chars(noised, base, rng, char_swap)
    return noised


def _swap_native_chars(text: str, base: int, rng: random.Random, p: float) -> str:
    """Transpose adjacent in-block characters — a native-keyboard typo."""
    chars = list(text)
    for index in range(len(chars) - 1):
        if _offset(chars[index], base) is None or _offset(chars[index + 1], base) is None:
            continue
        if rng.random() < p:
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


# ---------------------------------------------------------------------------
# Probability resolution
# ---------------------------------------------------------------------------


def probs_for_language(cfg: Config, language: str, block: str = "noisy_roman") -> dict[str, float]:
    """Operator probabilities for one language.

    Precedence is ``--set`` > per-language > global:

    * the global ``transforms.<block>`` block is the base;
    * ``transforms.<block>_by_language.<lang>`` overrides it, because the
      attested rates are nowhere near uniform (final-vowel dropping runs at
      0.536 in Hindi and 0.015 in Telugu — a single global set would inject
      Hindi-shaped noise into Telugu items);
    * an explicit ``--set transforms.<block>.<op>=x`` beats both. Without that
      last rule a sensitivity sweep would be a silent no-op for exactly the five
      languages the suite ships calibrated values for.
    """
    global_block = dict(cfg.get(f"transforms.{block}", {}) or {})
    by_language = dict(
        (cfg.get(f"transforms.{block}_by_language", {}) or {}).get(language, {}) or {}
    )

    merged: dict[str, Any] = {**global_block, **by_language}
    overridden = getattr(cfg, "overridden", set()) or set()
    for key, value in global_block.items():
        if f"transforms.{block}.{key}" in overridden:
            merged[key] = value

    return {key: float(value) for key, value in merged.items() if value is not None}
