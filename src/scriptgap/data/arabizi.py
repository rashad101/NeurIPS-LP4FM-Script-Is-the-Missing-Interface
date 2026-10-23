"""Arabizi — the universality sidebar (spec §6, figure 6).

Arabic speakers romanise with a convention Brahmic scripts have no equivalent
of: **digits stand in for consonants with no Latin counterpart** (``ع`` -> ``3``,
``ح`` -> ``7``, ``ء`` -> ``2``). Running the same pipeline over an Arabic item
set shows the script gap is not an Indic artefact.

Unlike conditions 2-5 there is **no round-trip gate here**: Arabizi is
intentionally irreversible — short vowels are unwritten in Arabic and are
supplied ad libitum in Arabizi — so those items are reported separately from the
gated Indic results.
"""

from __future__ import annotations

import random
import re
import unicodedata

from .. import textnorm

_ARABIC_RANGE = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

#: Consonant map. The first entry is the canonical spelling; the rest are
#: attested alternatives a sampler can draw from.
_CONSONANTS: dict[str, tuple[str, ...]] = {
    "ا": ("a",), "أ": ("a", "2a"), "إ": ("i", "2i"), "آ": ("aa", "a"),
    "ب": ("b",), "ت": ("t",), "ث": ("th", "s"), "ج": ("j", "g"),
    "ح": ("7", "h"), "خ": ("5", "kh"), "د": ("d",), "ذ": ("z", "dh"),
    "ر": ("r",), "ز": ("z",), "س": ("s",), "ش": ("sh", "ch"),
    "ص": ("9", "s"), "ض": ("d", "9'"), "ط": ("6", "t"), "ظ": ("z", "8"),
    "ع": ("3",), "غ": ("3'", "gh"), "ف": ("f",), "ق": ("8", "q", "2"),
    "ك": ("k",), "ل": ("l",), "م": ("m",), "ن": ("n",),
    "ه": ("h",), "و": ("w", "o", "u"), "ي": ("y", "i", "ee"),
    "ى": ("a",), "ة": ("a", "eh"), "ء": ("2",), "ئ": ("2",), "ؤ": ("2",),
    "ﻻ": ("la",),
}

#: Harakat, superscript alef and tatweel: dropped, as Arabizi writers drop them.
_DIACRITICS = re.compile("[ً-ْٰـ]")


def has_arabic(text: str) -> bool:
    """True when any Arabic-script character remains."""
    return any(
        any(low <= ord(c) <= high for low, high in _ARABIC_RANGE) for c in text
    )


def to_arabizi(text: str, rng: random.Random | None = None) -> str:
    """Arabic script -> Arabizi.

    With ``rng`` the per-character alternatives are sampled, which is how the
    stochastic samples of this condition differ from one another; without it the
    canonical spelling is used and the output is deterministic.
    """
    source = unicodedata.normalize("NFC", text or "")
    source = _DIACRITICS.sub("", source)
    # A Latin keyboard produces Latin punctuation: leaving `؟` in an otherwise
    # romanised string would identify the condition from punctuation alone.
    source = textnorm.asciify_punctuation(textnorm.normalise_digits(source))

    out: list[str] = []
    for char in source:
        options = _CONSONANTS.get(char)
        if options is None:
            out.append(char)
            continue
        if rng is not None and len(options) > 1:
            out.append(rng.choice(options))
        else:
            out.append(options[0])
    return "".join(out)
