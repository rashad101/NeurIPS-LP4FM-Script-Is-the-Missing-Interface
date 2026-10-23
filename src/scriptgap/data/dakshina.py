"""Attested human romanisation (Dakshina) — the answer to "you measured your
own noise model".

Conditions 2-5 are rule-generated, which invites exactly that objection. This
module supplies both replies, from the same source:

**Validation** — :func:`attested_romanise` builds the opt-in ``attested_roman``
condition by replacing each native word with a romanisation a native speaker
actually produced, sampled count-weighted from Dakshina's lexicons. Scheme
transliteration is used only as a fallback for out-of-lexicon words, and the
fraction of tokens that came from real data is recorded per cell as
``attested_coverage`` so results can be conditioned on it.

**Calibration** — :func:`measure_phenomena` measures how often each condition-3
operator's phenomenon actually occurs in attested spellings, relative to the
scheme transliteration of the same word, and :func:`suggest_probabilities` pools
those measurements into operator probabilities. That is what makes condition 3
fitted rather than hand-set.

The lexicons also quantify the repertoire claim directly: speakers produce 1.8
(Hindi) to 3.6 (Bengali) distinct attested spellings per word. There is no
canonical written form even at the level of a single word.

Only the lexicon TSVs are kept from the ~2 GB archive (~29 MB, 37 files).
"""

from __future__ import annotations

import logging
import random
import re
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import ROOT, textnorm
from ..config import Config
from ..progress import track
from . import noise as noise_mod
from . import translit

log = logging.getLogger(__name__)

DAKSHINA_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"

#: The twelve languages Dakshina covers — all five suite languages are here.
DAKSHINA_LANGS: tuple[str, ...] = (
    "bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur",
)

#: Operators whose phenomenon is measurable from single-word native->roman
#: pairs. ``char_swap`` and ``sms_isms`` are not: a lexicon of isolated words
#: cannot exhibit adjacent-key transposition or English chat spellings, so they
#: stay hand-set and this module never emits them.
MEASURABLE_OPERATORS: tuple[str, ...] = (
    "schwa_deletion",
    "long_vowel_collapse",
    "gemination_collapse",
    "aspiration_drop",
    "sibilant_swap",
    "vowel_ending_drop",
)

#: Shared tokeniser: without it a Bengali word tokenises into one fragment per
#: matra and never matches its own lexicon key.
_TOKEN = textnorm.WORD_TOKEN
_LONG_VOWELS = "āīūēō"


# ---------------------------------------------------------------------------
# Locating the corpus
# ---------------------------------------------------------------------------


def dakshina_root(cfg: Config) -> Path:
    configured = str(cfg.get("data.dakshina.root", "data/dakshina"))
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def is_available(root: Path | str) -> bool:
    """True when at least one lexicon TSV is present under ``root``."""
    root = Path(root)
    if not root.is_dir():
        return False
    return any(root.glob("*/lexicons/*.tsv"))


def download(root: Path | str, url: str = DAKSHINA_URL) -> Path:
    """Stream the archive and keep only the lexicon TSVs.

    The full tar is ~2 GB; the lexicons are ~29 MB. It is read as a stream and
    never written to disk in full.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    log.info("streaming Dakshina from %s (~2 GB read, lexicons only are kept)", url)

    kept = 0
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed, documented URL
        with tarfile.open(fileobj=response, mode="r|*") as tar:
            for member in tar:
                if not member.isfile() or "/lexicons/" not in member.name:
                    continue
                if not member.name.endswith(".tsv"):
                    continue
                parts = Path(member.name).parts
                # <release>/<lang>/lexicons/<file>.tsv -> <root>/<lang>/lexicons/<file>.tsv
                try:
                    index = parts.index("lexicons")
                except ValueError:  # pragma: no cover - defensive
                    continue
                target = root.joinpath(*parts[index - 1 :])
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                if extracted is None:  # pragma: no cover - defensive
                    continue
                target.write_bytes(extracted.read())
                kept += 1

    log.info("kept %d lexicon file(s) under %s", kept, root)
    return root


def ensure(cfg: Config) -> Path | None:
    """Return the corpus root, downloading it if the config allows."""
    root = dakshina_root(cfg)
    if is_available(root):
        return root
    if cfg.get("data.dakshina.auto_download", False):
        return download(root, str(cfg.get("data.dakshina.url", DAKSHINA_URL)))
    log.warning(
        "Dakshina not found at %s and auto_download is false — "
        "run `scriptgap download --dakshina`",
        root,
    )
    return None


def find_lexicon_files(
    root: Path | str, language: str, splits: Sequence[str] = ("train", "dev")
) -> list[Path]:
    """Lexicon TSVs for one language.

    Upstream filenames drift between releases, so a split that matches nothing
    falls back to every lexicon TSV present rather than silently yielding an
    empty lexicon — which would look like "this language has no data".
    """
    lexdir = Path(root) / language / "lexicons"
    if not lexdir.is_dir():
        return []
    every = sorted(lexdir.glob("*.tsv"))
    matched = [
        path
        for path in every
        if any(split in path.name.replace("_", ".").split(".") for split in splits)
    ]
    if not matched and every:
        log.debug(
            "%s: no lexicon matched splits %s — falling back to all %d file(s)",
            language,
            list(splits),
            len(every),
        )
    return matched or every


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------


@dataclass
class Lexicon:
    """Native word -> attested romanisations, count-weighted."""

    language: str
    entries: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def n_romanisations(self) -> int:
        return sum(len(variants) for variants in self.entries.values())

    @property
    def mean_variants(self) -> float:
        """Distinct attested spellings per word — the repertoire statistic."""
        if not self.entries:
            return 0.0
        return round(self.n_romanisations / len(self.entries), 4)

    def sample(self, word: str, rng: random.Random) -> str | None:
        """Draw an attested spelling, weighted by how often it was observed."""
        variants = self.entries.get(word)
        if not variants:
            return None
        spellings = [spelling for spelling, _count in variants]
        weights = [max(1, count) for _spelling, count in variants]
        return rng.choices(spellings, weights=weights, k=1)[0]


def load_lexicon(
    root: Path | str, language: str, splits: Sequence[str] = ("train", "dev")
) -> Lexicon | None:
    """Load and aggregate one language's lexicons, or ``None`` if absent."""
    files = find_lexicon_files(root, language, splits)
    if not files:
        return None

    counts: dict[str, dict[str, int]] = {}
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            native, roman = parts[0].strip(), parts[1].strip()
            if not native or not roman:
                continue
            try:
                count = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 1
            except ValueError:
                # A malformed count is not a reason to lose the pair.
                count = 1
            bucket = counts.setdefault(native, {})
            bucket[roman] = bucket.get(roman, 0) + max(1, count)

    if not counts:
        return None

    entries = {
        native: sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))
        for native, variants in counts.items()
    }
    return Lexicon(
        language=language,
        entries=entries,
        files=[str(path) for path in files],
    )


def load_lexicons(
    root: Path | str,
    languages: Iterable[str],
    splits: Sequence[str] = ("train", "dev"),
) -> dict[str, Lexicon]:
    """Load several languages, skipping the ones Dakshina does not cover."""
    out: dict[str, Lexicon] = {}
    for language in languages:
        lexicon = load_lexicon(root, language, splits)
        if lexicon is None:
            log.info("Dakshina has no lexicon for %s — skipping", language)
            continue
        out[language] = lexicon
        log.info(
            "%s: %d native words, %d attested romanisations (%.2f per word)",
            language,
            len(lexicon),
            lexicon.n_romanisations,
            lexicon.mean_variants,
        )
    return out


# ---------------------------------------------------------------------------
# The attested_roman condition
# ---------------------------------------------------------------------------


@dataclass
class AttestedReport:
    """Where each word's spelling came from."""

    tokens: int = 0
    #: Tokens replaced with a real human spelling.
    attested: int = 0
    #: Tokens the lexicon did not have, filled by scheme transliteration.
    fallback: int = 0

    @property
    def coverage(self) -> float:
        return round(self.attested / self.tokens, 6) if self.tokens else 0.0

    @property
    def applied(self) -> bool:
        """False when no token came from attested data.

        Such a cell is a scheme transliteration wearing the attested label, so
        it must not be counted as an observation of this condition.
        """
        return self.attested > 0


def attested_romanise(
    text: str,
    lexicon: Lexicon | None,
    script: str,
    rng: random.Random,
) -> tuple[str, AttestedReport]:
    """Rewrite native words as native speakers spelled them."""
    report = AttestedReport()
    if lexicon is None:
        return text, report

    base = noise_mod._BLOCK_BASE.get(script)

    def in_script(word: str) -> bool:
        if base is None:
            return True
        return any(base <= ord(c) < base + 0x80 for c in word)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if not in_script(word):
            return word
        report.tokens += 1
        sampled = lexicon.sample(word, rng)
        if sampled is not None:
            report.attested += 1
            return sampled
        report.fallback += 1
        # Fall back to the scheme transliteration, so the output is fully
        # romanised rather than a mixture of scripts.
        return translit.romanise(word, script, strip=True).emitted

    out = _TOKEN.sub(replace, text)
    # Someone typing on a Latin keyboard types `.`, not `।`; leaving the native
    # form would let a reader identify the condition from punctuation alone.
    return textnorm.asciify_punctuation(textnorm.normalise_digits(out)), report


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _reference_forms(word: str, script: str, scheme: str) -> tuple[str, str]:
    """``(iso_with_diacritics, de-diacritised)`` for one native word."""
    iso, _backend = translit.to_roman(word, script, scheme)
    return iso, translit.strip_diacritics(iso).lower()


def _closer(attested: str, transformed: str, reference: str) -> bool:
    """Does the transformed form explain the attested spelling better?"""
    return translit.similarity(attested, transformed) > translit.similarity(attested, reference)


def measure_phenomena(
    lexicon: Lexicon,
    script: str,
    scheme: str = "ISO",
    max_pairs: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure how often each operator's phenomenon occurs in attested data.

    For every (native word, attested spelling) pair, each operator is applied to
    the scheme transliteration of the word at p=1. If the operator has nothing
    to act on, the pair is not a *site* for it. If it does, the pair counts as
    *applied* when the transformed form is closer to what the speaker actually
    wrote than the untransformed reference is.

    Returns ``{operator: {"sites": n, "applied": n, "rate": r}}`` plus
    underscore-prefixed provenance keys.
    """
    operators = {name: fn for name, fn in noise_mod.ROMAN_OPS if name in MEASURABLE_OPERATORS}
    tally: dict[str, dict[str, int]] = {
        name: {"sites": 0, "applied": 0} for name in MEASURABLE_OPERATORS
    }

    rng = random.Random(f"{seed}␟measure␟{lexicon.language}")
    pairs: list[tuple[str, str]] = [
        (native, spelling)
        for native, variants in lexicon.entries.items()
        for spelling, _count in variants
    ]
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)

    examined = 0
    for native, spelling in track(
        pairs, desc=f"measuring {lexicon.language}", unit="pair", leave=False
    ):
        try:
            iso, plain = _reference_forms(native, script, scheme)
        except Exception:  # pragma: no cover - a single bad word is not fatal
            continue
        if not plain:
            continue
        examined += 1
        attested = spelling.strip().lower()

        for name, operator in operators.items():
            if name == "long_vowel_collapse":
                # Invisible in a de-diacritised reference (`ā` and `a` are both
                # `a`), so it is measured the other way round: the site is a
                # long vowel in the ISO form, and it counts as applied when the
                # attested spelling is closer to the collapsed form than to the
                # doubled one a length-marking speller would write.
                if not any(vowel in iso.lower() for vowel in _LONG_VOWELS):
                    continue
                doubled = _double_long_vowels(iso).lower()
                tally[name]["sites"] += 1
                if not _closer(attested, doubled, plain):
                    tally[name]["applied"] += 1
                continue

            transformed = operator(plain, random.Random(0), 1.0)
            if transformed == plain:
                continue  # the phenomenon has nowhere to apply in this word
            tally[name]["sites"] += 1
            if _closer(attested, transformed, plain):
                tally[name]["applied"] += 1

    rates: dict[str, Any] = {}
    for name, counts in tally.items():
        if not counts["sites"]:
            continue
        rates[name] = {
            "sites": counts["sites"],
            "applied": counts["applied"],
            "rate": round(counts["applied"] / counts["sites"], 6),
        }

    rates["_language"] = lexicon.language
    rates["_pairs_examined"] = examined
    rates["_native_words"] = len(lexicon)
    return rates


def _double_long_vowels(text: str) -> str:
    """``bhārata`` -> ``bhaarata`` — how a length-marking speller would write it."""
    out: list[str] = []
    for char in text:
        plain = translit.strip_diacritics(char) or char
        out.append(plain * 2 if char.lower() in _LONG_VOWELS else plain)
    return "".join(out)


def suggest_probabilities(measured: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Pool per-language measurements into operator probabilities.

    Pooling is **by site count**, not a mean of rates: a language with 200 sites
    should not weigh the same as one with 20,000.
    """
    totals: dict[str, dict[str, int]] = {}
    for rates in measured.values():
        for name, stats in (rates or {}).items():
            if name.startswith("_") or name not in MEASURABLE_OPERATORS:
                continue
            if not isinstance(stats, dict):
                continue
            bucket = totals.setdefault(name, {"sites": 0, "applied": 0})
            bucket["sites"] += int(stats.get("sites", 0))
            bucket["applied"] += int(stats.get("applied", 0))

    return {
        name: round(counts["applied"] / counts["sites"], 3)
        for name, counts in sorted(totals.items())
        if counts["sites"] > 0
    }
