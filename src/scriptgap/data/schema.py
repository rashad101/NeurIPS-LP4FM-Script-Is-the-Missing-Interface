"""Item types, the language/script map, and prompt fingerprints.

Two record types travel through the pipeline:

``BaseItem``
    One native-script gold item, exactly as it came out of its source dataset.
    Its ``item_id`` hashes the **native source only**, so it is stable across
    every change to the transform settings.

``RenderedItem``
    One (item x condition x sample) cell — what a model actually sees. Its
    ``uid`` is ``item_id::condition::sample_idx``, which is likewise stable
    across transform changes; that is deliberate (a cell keeps its identity when
    it is re-rendered) and it is why every cell also carries a
    ``prompt_sha256`` content fingerprint. The uid says *which* cell this is;
    the fingerprint says *what it currently contains*. A resume needs both:
    re-running ``scriptgap calibrate`` changes the prompt without changing the
    uid, and predictions generated under the old settings have to be evicted
    rather than silently kept.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: The evaluation tasks, in the order they are reported.
TASKS: tuple[str, ...] = ("milu", "math", "ifeval", "safety", "arabizi")

#: How each task's gold answer is scored (see ``eval/scoring.py``).
ANSWER_TYPES: dict[str, str] = {
    "milu": "mcq",
    "math": "number",
    "ifeval": "constraints",
    "safety": "refusal",
    "arabizi": "mcq",
}

#: ISO 639-1 code -> Unicode script name, as Aksharamukha names them.
LANGUAGE_SCRIPT: dict[str, str] = {
    "hi": "Devanagari",
    "mr": "Devanagari",
    "ne": "Devanagari",
    "sa": "Devanagari",
    "bn": "Bengali",
    "as": "Bengali",
    "gu": "Gujarati",
    "pa": "Gurmukhi",
    "or": "Oriya",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "si": "Sinhala",
    "ur": "Urdu",
    "sd": "Sindhi",
    "ar": "Arabic",
    "en": "Latin",
}

#: MILU ships its configs under language *names*; everything else uses codes.
MILU_LANG_CODE: dict[str, str] = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Odia": "or",
    "Punjabi": "pa",
    "English": "en",
}

#: Inverse of the above, for logging a code back as a MILU config name.
MILU_CODE_LANG: dict[str, str] = {code: name for name, code in MILU_LANG_CODE.items()}


def script_for(language: str) -> str:
    """Script name for a language code (``"hi"`` -> ``"Devanagari"``)."""
    return LANGUAGE_SCRIPT.get((language or "").strip().lower(), "Latin")


def answer_type_for(task: str) -> str:
    return ANSWER_TYPES.get(task, "mcq")


def item_id_for(task: str, language: str, prompt: str, options: list[str] | None = None) -> str:
    """Stable id hashing the **native source only**.

    Deliberately independent of the transform settings: re-rendering an item
    under new noise probabilities must produce the same id, so a run can tell
    "same cell, new contents" from "a different cell".
    """
    payload = json.dumps([task, language, prompt, options], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{task}-{language}-{digest}"


def prompt_fingerprint(prompt: str, options: list[str] | None = None) -> str:
    """Content hash of what the model will be shown.

    Covers the options as well as the prompt: the transforms rewrite both, and a
    fingerprint that ignored the options would fail to notice a re-render that
    changed only the answer choices.
    """
    payload = json.dumps([prompt, options], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BaseItem:
    """A native-script gold item, before any transform."""

    item_id: str
    task: str
    language: str
    script: str
    prompt: str
    answer: str
    answer_type: str
    options: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaseItem":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class RenderedItem:
    """One (item x condition x sample) cell — what a model is actually sent."""

    uid: str
    item_id: str
    task: str
    language: str
    script: str
    condition: str
    sample_idx: int
    prompt: str
    answer: str
    answer_type: str
    options: list[str] | None = None
    #: Item-level: did the clean romanisation of this item survive the
    #: back-transliteration gate? Copied onto every condition of the item so a
    #: prediction can be filtered without joining back to the item.
    roundtrip_pass: bool = True
    #: Did this condition's transform actually change anything? A cell where it
    #: did not is a byte-identical copy of the native prompt and must not be
    #: scored as an observation of its nominal condition.
    transform_applied: bool = True
    prompt_sha256: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_sha256:
            self.prompt_sha256 = prompt_fingerprint(self.prompt, self.options)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RenderedItem":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def make_uid(item_id: str, condition: str, sample_idx: int) -> str:
    """``item_id::condition::sample_idx`` — stable across re-renders."""
    return f"{item_id}::{condition}::{sample_idx}"
