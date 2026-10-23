"""Condition 2 — scheme-based romanisation, and the back-transliteration gate.

Two strings are produced per item and they do different jobs:

``reversible``
    The ISO-15919 form, diacritics intact. This is the **gate statistic**: it is
    transliterated back to the native script and compared with the source, which
    measures whether the *tooling* preserved the content.

``emitted``
    The same string de-diacritised into a keyboard-typable form — what the model
    is actually shown. De-diacritisation is a deliberate model of user behaviour
    rather than a tool defect, so it must not be charged to the gate; its own
    round-trip score is kept as a secondary diagnostic.

Two instrument bugs shaped the implementation, both of which silently discarded
25-50% of sound items before they were found:

1. **Digits.** ASCII ``2`` comes back from Aksharamukha as ``२``/``২``. No
   information is lost, but a naive string comparison scored it as a mismatch on
   every numeric item — that is the entire maths task. Comparison therefore
   happens in :func:`scriptgap.textnorm.canonical` form.
2. **Embedded Latin.** Latin spans already present in the source (``(lowercase)``,
   ``JSON``) are read as romanised Indic on the reverse pass and come back as
   ``लॊwऎर्चसॆ``. They are masked for the duration of the round trip. The mask
   uses Unicode **Tag** characters: private-use codepoints do not work, because
   Aksharamukha's reverse pass treats an unmapped codepoint as a consonant with
   an implicit vowel and turns ``U+E000`` into a real letter.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .. import textnorm

log = logging.getLogger(__name__)

#: Base of the masking sentinel range (Unicode Tag characters, U+E0000..U+E007F).
#: Tag characters survive Aksharamukha in both directions untouched; private-use
#: codepoints do not.
_SENTINEL_BASE = 0xE0000
_SENTINEL_LIMIT = 0x80

#: Latin runs already present in a native-script source: English loanwords,
#: format names, parenthesised glosses. Masked across the round trip so the
#: reverse pass cannot read them as romanised Indic.
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[._'-][A-Za-z0-9]+)*")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

#: Script name -> ``indic_transliteration.sanscript`` scheme, for the fallback.
_SANSCRIPT_SCHEMES = {
    "Devanagari": "devanagari",
    "Bengali": "bengali",
    "Gujarati": "gujarati",
    "Gurmukhi": "gurmukhi",
    "Kannada": "kannada",
    "Malayalam": "malayalam",
    "Oriya": "oriya",
    "Tamil": "tamil",
    "Telugu": "telugu",
}


@lru_cache(maxsize=1)
def available_backends() -> tuple[str, ...]:
    """Transliteration backends importable in this environment, best first."""
    found: list[str] = []
    try:  # pragma: no cover - environment dependent
        import aksharamukha.transliterate  # noqa: F401

        found.append("aksharamukha")
    except Exception:  # pragma: no cover
        pass
    try:  # pragma: no cover - environment dependent
        import indic_transliteration.sanscript  # noqa: F401

        found.append("indic_transliteration")
    except Exception:  # pragma: no cover
        pass
    return tuple(found)


def _require_backend() -> str:
    backends = available_backends()
    if not backends:
        raise SystemExit(
            "no transliteration backend installed — "
            "`uv pip install aksharamukha indic-transliteration`"
        )
    return backends[0]


def to_roman(text: str, script: str, scheme: str = "ISO") -> tuple[str, str]:
    """Native script -> ``scheme``. Returns ``(text, backend_used)``."""
    backend = _require_backend()
    if backend == "aksharamukha":
        from aksharamukha import transliterate

        return transliterate.process(script, scheme, text), backend

    from indic_transliteration import sanscript  # pragma: no cover - fallback

    source = _SANSCRIPT_SCHEMES.get(script)
    if source is None:
        return text, backend
    return sanscript.transliterate(text, source, sanscript.IAST), backend


def to_native(text: str, script: str, scheme: str = "ISO") -> str:
    """``scheme`` -> native script (the reverse leg of the gate)."""
    backend = _require_backend()
    if backend == "aksharamukha":
        from aksharamukha import transliterate

        return transliterate.process(scheme, script, text)

    from indic_transliteration import sanscript  # pragma: no cover - fallback

    target = _SANSCRIPT_SCHEMES.get(script)
    if target is None:
        return text
    return sanscript.transliterate(text, sanscript.IAST, target)


# ---------------------------------------------------------------------------
# Masking embedded Latin
# ---------------------------------------------------------------------------


def _mask_latin(text: str) -> tuple[str, list[str]]:
    """Replace Latin runs with sentinels. Returns ``(masked, originals)``."""
    originals: list[str] = []

    def swap(match: re.Match[str]) -> str:
        if len(originals) >= _SENTINEL_LIMIT:
            # More Latin runs than sentinels: leave the rest in place rather
            # than reusing a sentinel, which would restore the wrong word.
            return match.group(0)
        originals.append(match.group(0))
        return chr(_SENTINEL_BASE + len(originals) - 1)

    return _LATIN_RUN.sub(swap, text), originals


def _unmask(text: str, originals: list[str]) -> str:
    for index, original in enumerate(originals):
        text = text.replace(chr(_SENTINEL_BASE + index), original)
    return text


def has_sentinel(text: str) -> bool:
    return any(_SENTINEL_BASE <= ord(c) < _SENTINEL_BASE + _SENTINEL_LIMIT for c in text)


# ---------------------------------------------------------------------------
# De-diacritisation
# ---------------------------------------------------------------------------

#: Applied before the generic combining-mark strip, for the cases where simply
#: dropping the mark would give a letter no one types. ``ś``/``ṣ`` are written
#: ``sh``; vocalic ``r̥`` is written ``ri``; ISO ``c`` is written ``ch``.
_PRE_STRIP: tuple[tuple[str, str], ...] = (
    ("r̥̄", "ri"),
    ("r̥", "ri"),
    ("l̥̄", "li"),
    ("l̥", "li"),
    ("ś", "sh"),
    ("Ś", "Sh"),
    ("ṣ", "sh"),
    ("Ṣ", "Sh"),
    ("c", "ch"),
    ("C", "Ch"),
    ("ṁ", "n"),
    ("ṃ", "n"),
)


def strip_diacritics(text: str) -> str:
    """ISO-15919 -> the de-diacritised form a Latin keyboard produces."""
    out = unicodedata.normalize("NFC", text)
    for source, target in _PRE_STRIP:
        out = out.replace(source, target)
    out = unicodedata.normalize("NFD", out)
    out = "".join(c for c in out if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", out)


# ---------------------------------------------------------------------------
# Similarity and the gate
# ---------------------------------------------------------------------------


def similarity(left: str, right: str) -> float:
    """Character-level similarity in [0, 1] (1.0 when identical)."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


@dataclass
class Romanisation:
    """Both romanised forms of one item, plus the gate statistics."""

    #: What the model is shown (de-diacritised unless ``strip=False``).
    emitted: str
    #: The ISO form with diacritics — the gate statistic's subject.
    reversible: str
    #: Similarity of ``reversible`` transliterated back vs. the source.
    roundtrip_score: float
    #: Same for ``emitted``. Reported as a diagnostic only: de-diacritisation is
    #: a deliberate model of user behaviour, not a tooling failure.
    emitted_roundtrip_score: float
    #: The recovered native-script string, kept for error inspection.
    recovered: str
    backend: str

    @property
    def changed(self) -> bool:
        return bool(self.emitted)


def romanise(
    text: str,
    script: str,
    scheme: str = "ISO",
    strip: bool = True,
) -> Romanisation:
    """Romanise ``text`` and measure how well it transliterates back."""
    source = unicodedata.normalize("NFC", text or "")
    masked, originals = _mask_latin(source)

    forward, backend = to_roman(masked, script, scheme)
    reversible = _unmask(forward, originals)

    stripped = strip_diacritics(forward)
    emitted = _unmask(stripped, originals) if strip else reversible

    # The gate: compare in canonical form so a digit that came back in native
    # script, or a danda that came back as a full stop, is not scored as loss.
    recovered_masked = to_native(forward, script, scheme)
    recovered = _unmask(recovered_masked, originals)
    score = similarity(textnorm.canonical(recovered_masked), textnorm.canonical(masked))

    try:
        emitted_recovered = to_native(stripped, script, scheme)
        emitted_score = similarity(
            textnorm.canonical(emitted_recovered), textnorm.canonical(masked)
        )
    except Exception:  # pragma: no cover - diagnostic only, never fatal
        emitted_score = 0.0

    return Romanisation(
        emitted=emitted,
        reversible=reversible,
        roundtrip_score=round(score, 6),
        emitted_roundtrip_score=round(emitted_score, 6),
        recovered=recovered,
        backend=backend,
    )
