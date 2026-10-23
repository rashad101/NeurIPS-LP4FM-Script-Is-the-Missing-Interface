"""Shared text normalisation used by both the transforms and the scorers.

Kept in one place so the round-trip gate and the answer parsers agree on what
"the same string" means. If these diverged, the gate could discard items the
scorer would have handled fine (or worse, the reverse).
"""

from __future__ import annotations

import re
import unicodedata

#: Native-script digit -> ASCII, for every Brahmic block plus Arabic-Indic.
#: The Brahmic blocks put digits at offset 0x66-0x6F within the block, so the
#: table is generated rather than typed out.
_DIGIT_BASES = (
    0x0966,  # Devanagari
    0x09E6,  # Bengali
    0x0A66,  # Gurmukhi
    0x0AE6,  # Gujarati
    0x0B66,  # Oriya
    0x0BE6,  # Tamil
    0x0C66,  # Telugu
    0x0CE6,  # Kannada
    0x0D66,  # Malayalam
    0x0660,  # Arabic-Indic
    0x06F0,  # Extended Arabic-Indic
)

DIGIT_MAP: dict[int, str] = {
    base + d: str(d) for base in _DIGIT_BASES for d in range(10)
}

#: Native-script sentence terminators that mean "full stop".
_DANDA_MAP = {ord("।"): ".", ord("॥"): ".", ord("۔"): "."}

#: Punctuation a Latin-keyboard writer types instead of the native form. Used by
#: the romanised conditions: leaving a danda in an otherwise-Latin string is a
#: tell-tale of a scripted transform rather than of the register itself.
PUNCTUATION_MAP: dict[int, str] = {
    **_DANDA_MAP,
    ord("،"): ",",
    ord("؛"): ";",
    ord("؟"): "?",
    ord("“"): '"',
    ord("”"): '"',
    ord("‘"): "'",
    ord("’"): "'",
}

_WS = re.compile(r"\s+")

#: Characters that end a word. Everything else is word-internal.
#:
#: Defined as a *negative* class on purpose. The obvious `[^\W\d_]+` is wrong
#: for every script this study uses: Indic dependent vowel signs (matras),
#: viramas and nuktas are combining marks, which `\w` does not match, so
#: `রাজধানী` tokenised as five fragments instead of one word — and every
#: whole-token lookup (the code-mix free-morpheme check, the Dakshina lexicon)
#: silently missed. ISO-15919's decomposed `r̥` breaks the same way.
_SEPARATORS = (
    r"\s\d"
    r".,;:!?()\[\]{}<>\"'`/\\|@#$%^&*+=~_"
    r"—–…।॥،؛؟“”‘’"
)

#: One word: a maximal run of non-separator characters, marks included.
WORD_TOKEN = re.compile(rf"[^{_SEPARATORS}]+")


def normalise_digits(text: str) -> str:
    """Map every native-script digit to its ASCII equivalent."""
    return text.translate(DIGIT_MAP)


def asciify_punctuation(text: str) -> str:
    """Native-script punctuation -> the ASCII a Latin keyboard produces."""
    return text.translate(PUNCTUATION_MAP)


def canonical(text: str) -> str:
    """NFC, ASCII digits, unified full stops, collapsed whitespace.

    This is the comparison form for the round-trip gate: two strings that differ
    only in digit script or spacing carry the same information, and treating
    them as different would make the gate reject sound items.
    """
    out = unicodedata.normalize("NFC", text)
    out = out.translate(DIGIT_MAP)
    out = out.translate(_DANDA_MAP)
    return _WS.sub(" ", out).strip()
