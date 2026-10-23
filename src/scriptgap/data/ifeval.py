"""Verifiable Hindi instruction items, and the verifiers that grade them.

Constraints are **structural only** — word and sentence counts, list shapes,
punctuation, casing, JSON validity. A constraint quoting a literal phrase
("include the word ``सुरक्षा``") would be rewritten by our own transforms, so the
phrase the model was asked to emit would differ per condition and part of the
measured gap would be an artefact of the instruction rather than of the model.

Casing rules apply to **embedded Latin only**: Devanagari has no case, so a
response written entirely in native script satisfies them vacuously rather than
failing for a property its script cannot express.

Items are generated deterministically from a seed. Every generated item is
checked against :data:`_INCOMPATIBLE`, because a self-contradictory item scores
0 for every model in every condition and adds noise that looks like a gap.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

_WORDS = re.compile(r"\S+")
#: Sentence terminators including the danda — an item written in Hindi ends its
#: sentences with `।`, and a verifier that only knew `.` would fail it unfairly.
_SENTENCE_END = re.compile(r"[.!?।॥]+")
_LATIN = re.compile(r"[A-Za-z]")
_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def _word_count(text: str) -> int:
    return len(_WORDS.findall(text.strip()))


def _sentence_count(text: str) -> int:
    return len([part for part in _SENTENCE_END.split(text) if part.strip()])


def _strip_fence(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = _FENCE.sub("", body).strip()
    return body


def v_word_count_max(text: str, n: int = 50, **_: Any) -> bool:
    return _word_count(text) <= int(n)


def v_word_count_min(text: str, n: int = 5, **_: Any) -> bool:
    return _word_count(text) >= int(n)


def v_no_commas(text: str, **_: Any) -> bool:
    return "," not in text and "،" not in text


def v_no_digits(text: str, **_: Any) -> bool:
    return not any(c.isdigit() for c in text)


def v_no_question_marks(text: str, **_: Any) -> bool:
    return "?" not in text and "؟" not in text


def v_json_object(text: str, **_: Any) -> bool:
    body = _strip_fence(text)
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict)


def v_bullet_list_exact(text: str, n: int = 3, **_: Any) -> bool:
    return len(_BULLET.findall(text)) == int(n)


def v_numbered_list_exact(text: str, n: int = 3, **_: Any) -> bool:
    return len(_NUMBERED.findall(text)) == int(n)


def v_wrap_in_quotes(text: str, **_: Any) -> bool:
    body = text.strip()
    if len(body) < 2:
        return False
    return (body[0], body[-1]) in {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")}


def v_single_paragraph(text: str, **_: Any) -> bool:
    return "\n\n" not in text.strip()


def v_paragraph_count_exact(text: str, n: int = 2, **_: Any) -> bool:
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paragraphs) == int(n)


def v_sentence_count_exact(text: str, n: int = 3, **_: Any) -> bool:
    return _sentence_count(text) == int(n)


def v_uppercase_latin(text: str, **_: Any) -> bool:
    """Every Latin letter uppercase. Vacuously true for pure native script."""
    return all(c.isupper() for c in text if _LATIN.match(c))


def v_lowercase_latin(text: str, **_: Any) -> bool:
    return all(c.islower() for c in text if _LATIN.match(c))


#: Constraint name -> verifier. The name is what appears in an item's
#: ``constraints`` list and in the per-constraint scoring detail.
BY_NAME: dict[str, Callable[..., bool]] = {
    "word_count_max": v_word_count_max,
    "word_count_min": v_word_count_min,
    "no_commas": v_no_commas,
    "no_digits": v_no_digits,
    "no_question_marks": v_no_question_marks,
    "json_object": v_json_object,
    "bullet_list_exact": v_bullet_list_exact,
    "numbered_list_exact": v_numbered_list_exact,
    "wrap_in_quotes": v_wrap_in_quotes,
    "single_paragraph": v_single_paragraph,
    "paragraph_count_exact": v_paragraph_count_exact,
    "sentence_count_exact": v_sentence_count_exact,
    "uppercase_latin": v_uppercase_latin,
    "lowercase_latin": v_lowercase_latin,
}


def verify(response: str, constraints: list[dict[str, Any]]) -> tuple[bool, dict[str, bool]]:
    """Check every constraint. Returns ``(all_passed, per_constraint)``.

    An unknown constraint **fails closed**: silently passing it would credit the
    model for satisfying something that was never checked.
    """
    per: dict[str, bool] = {}
    for constraint in constraints or []:
        name = str(constraint.get("type", ""))
        verifier = BY_NAME.get(name)
        if verifier is None:
            per[name] = False
            continue
        params = constraint.get("params") or {}
        try:
            per[name] = bool(verifier(response or "", **params))
        except Exception:
            per[name] = False
    return (all(per.values()) if per else False), per


# ---------------------------------------------------------------------------
# Item generation
# ---------------------------------------------------------------------------

#: Constraint pairs that cannot both hold. Checked at generation time — an item
#: that is impossible to satisfy scores 0 in every condition for every model,
#: which is noise that reads like a gap.
_INCOMPATIBLE: tuple[frozenset[str], ...] = (
    frozenset({"uppercase_latin", "lowercase_latin"}),
    frozenset({"word_count_max", "word_count_min"}),
    frozenset({"bullet_list_exact", "numbered_list_exact"}),
    frozenset({"json_object", "bullet_list_exact"}),
    frozenset({"json_object", "numbered_list_exact"}),
    frozenset({"json_object", "wrap_in_quotes"}),
    frozenset({"json_object", "sentence_count_exact"}),
    frozenset({"json_object", "paragraph_count_exact"}),
    frozenset({"json_object", "no_commas"}),
    frozenset({"json_object", "no_digits"}),
    frozenset({"bullet_list_exact", "wrap_in_quotes"}),
    frozenset({"numbered_list_exact", "wrap_in_quotes"}),
    frozenset({"bullet_list_exact", "sentence_count_exact"}),
    frozenset({"numbered_list_exact", "sentence_count_exact"}),
    frozenset({"bullet_list_exact", "paragraph_count_exact"}),
    frozenset({"numbered_list_exact", "paragraph_count_exact"}),
    frozenset({"single_paragraph", "paragraph_count_exact"}),
    frozenset({"wrap_in_quotes", "paragraph_count_exact"}),
    frozenset({"wrap_in_quotes", "sentence_count_exact"}),
)

#: Ordinary Hindi prose stems. Content is deliberately mundane and
#: non-controversial: the task under test is instruction *following*, so the
#: stem should contribute as little difficulty as possible.
_STEMS: tuple[str, ...] = (
    "बारिश के मौसम के बारे में लिखिए।",
    "एक अच्छी किताब पढ़ने के फायदे बताइए।",
    "सुबह जल्दी उठने की आदत के बारे में लिखिए।",
    "अपने पसंदीदा त्योहार का वर्णन कीजिए।",
    "साइकिल चलाने के फायदे बताइए।",
    "एक अच्छे शिक्षक के गुण बताइए।",
    "पानी बचाने के तरीके बताइए।",
    "किसी पहाड़ी जगह की यात्रा का वर्णन कीजिए।",
    "स्वस्थ भोजन के बारे में लिखिए।",
    "पेड़ लगाने का महत्व समझाइए।",
    "समय के सही उपयोग के बारे में लिखिए।",
    "अपने शहर की सबसे अच्छी बात बताइए।",
    "खेल खेलने के फायदे बताइए।",
    "एक अच्छे दोस्त की पहचान क्या है, बताइए।",
    "सर्दियों की सुबह का वर्णन कीजिए।",
    "पुस्तकालय के उपयोग के बारे में लिखिए।",
    "साफ-सफाई के महत्व पर लिखिए।",
    "किसी नई भाषा को सीखने के फायदे बताइए।",
    "बगीचे में काम करने के अनुभव के बारे में लिखिए।",
    "रेल यात्रा का वर्णन कीजिए।",
    "संगीत सुनने के फायदे बताइए।",
    "अपने गाँव या मोहल्ले का वर्णन कीजिए।",
    "नियमित व्यायाम के महत्व पर लिखिए।",
    "बरसात में सड़क पर चलने की सावधानियाँ बताइए।",
    "किसी पुराने मित्र से मिलने के अनुभव के बारे में लिखिए।",
)

#: How each constraint is phrased to the model, and how its parameter is drawn.
#: The instruction wrapper stays English (see the methodological note in the
#: README): only the item content varies across conditions, so a drop cannot be
#: the model failing to parse *our* instruction.
_PHRASING: dict[str, str] = {
    "word_count_max": "Answer in at most {n} words.",
    "word_count_min": "Answer in at least {n} words.",
    "no_commas": "Do not use any commas in your answer.",
    "no_digits": "Do not use any digits in your answer.",
    "no_question_marks": "Do not use any question marks in your answer.",
    "json_object": "Reply with a single valid JSON object and nothing else.",
    "bullet_list_exact": "Reply as a bulleted list of exactly {n} items, each starting with '- '.",
    "numbered_list_exact": "Reply as a numbered list of exactly {n} items.",
    "wrap_in_quotes": "Wrap your entire answer in double quotation marks.",
    "single_paragraph": "Reply in a single paragraph with no blank lines.",
    "paragraph_count_exact": "Reply in exactly {n} paragraphs separated by a blank line.",
    "sentence_count_exact": "Reply in exactly {n} sentences.",
    "uppercase_latin": "Write any Latin-script letters in UPPERCASE.",
    "lowercase_latin": "Write any Latin-script letters in lowercase.",
}

#: Parameter ranges, so a generated item is neither trivial nor impossible.
_PARAM_RANGES: dict[str, tuple[int, int]] = {
    "word_count_max": (25, 80),
    "word_count_min": (15, 40),
    "bullet_list_exact": (2, 5),
    "numbered_list_exact": (2, 5),
    "paragraph_count_exact": (2, 3),
    "sentence_count_exact": (2, 4),
}


def _compatible(name: str, chosen: set[str]) -> bool:
    candidate = chosen | {name}
    return not any(pair <= candidate for pair in _INCOMPATIBLE)


def generate_items(n: int, seed: int = 0, language: str = "hi") -> list[dict[str, Any]]:
    """Generate ``n`` verifiable instruction items, deterministically."""
    rng = random.Random(f"{seed}␟ifeval␟{language}")
    names = sorted(BY_NAME)
    items: list[dict[str, Any]] = []

    for index in range(n):
        stem = _STEMS[index % len(_STEMS)]

        chosen: list[str] = []
        selected: set[str] = set()
        wanted = rng.choice((1, 2, 2, 3))
        for name in rng.sample(names, len(names)):
            if len(chosen) >= wanted:
                break
            if _compatible(name, selected):
                chosen.append(name)
                selected.add(name)

        constraints: list[dict[str, Any]] = []
        instructions: list[str] = []
        for name in chosen:
            params: dict[str, Any] = {}
            if name in _PARAM_RANGES:
                low, high = _PARAM_RANGES[name]
                params["n"] = rng.randint(low, high)
            constraints.append({"type": name, "params": params})
            instructions.append(_PHRASING[name].format(**params))

        items.append(
            {
                "id": f"ifeval-{language}-{index:04d}",
                "language": language,
                "prompt": stem,
                "instructions": instructions,
                "constraints": constraints,
            }
        )
    return items
