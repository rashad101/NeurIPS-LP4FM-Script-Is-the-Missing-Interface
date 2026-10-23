"""Base-item loaders: MILU, MGSM, IFEval, MultiJail, MMMLU.

Every builder returns native-script :class:`~scriptgap.data.schema.BaseItem`
objects. Nothing here is aware of the orthographic conditions — that is
``build.py``'s job — so the gold answer enters the pipeline exactly once, from
its source dataset, and no transform ever touches it.

Two failure policies, deliberately different:

* **A dataset that cannot be reached** (gated, offline, renamed) logs a warning
  and yields zero items. A missing MILU should not take the maths task down
  with it on an unattended run.
* **A dataset whose *columns* are not what we expect** raises, naming the
  columns it did find. Upstream schemas drift, and guessing a gold-label column
  wrong produces a study-shaped artefact rather than an error.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .. import ROOT
from ..config import Config
from . import ifeval as ifeval_mod
from .schema import (
    MILU_LANG_CODE,
    BaseItem,
    answer_type_for,
    item_id_for,
    script_for,
)

log = logging.getLogger(__name__)

_LETTERS = "ABCDEFGH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_split(dataset: str, config: str | None, split: str) -> Any | None:
    """``load_dataset`` with the gated/offline case demoted to a warning."""
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover - datasets is a hard dependency
        log.error("the `datasets` package is not installed — cannot build base items")
        return None

    try:
        return load_dataset(dataset, config, split=split) if config else load_dataset(
            dataset, split=split
        )
    except Exception as exc:
        detail = str(exc)
        hint = ""
        if "gated" in detail.lower() or "401" in detail or "403" in detail:
            hint = " — accept the dataset terms and run `hf auth login`"
        log.warning(
            "could not load %s%s (%s)%s",
            dataset,
            f":{config}" if config else "",
            type(exc).__name__,
            hint,
        )
        log.debug("%s", detail)
        return None


def _resolve_column(
    columns: Sequence[str], candidates: Sequence[str], context: str
) -> str:
    """First candidate present, or a hard error naming what *is* there."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise SystemExit(
        f"{context}: none of {list(candidates)} are present. "
        f"The dataset's columns are {sorted(columns)}. "
        "Upstream schema drift — update the candidate list rather than guessing."
    )


def _subsample(rows: list[Any], n: int | None, seed: int, salt: str) -> list[Any]:
    """Deterministic subsample, so the suite is reproducible from the seed."""
    if not n or n >= len(rows):
        return rows
    rng = random.Random(f"{seed}␟{salt}")
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in indices]


def _answer_letter(raw: Any, options: Sequence[str]) -> str | None:
    """Normalise a gold answer to an option letter.

    Sources disagree: some ship ``"option2"``, some a 1-based index, some the
    option text, some already a letter. Getting this wrong silently shifts every
    gold label by one, so an unrecognisable value returns ``None`` and the item
    is dropped rather than guessed at.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    upper = text.upper()
    if len(upper) == 1 and upper in _LETTERS[: len(options)]:
        return upper

    lowered = text.lower().replace(" ", "")
    if lowered.startswith("option"):
        suffix = lowered[len("option") :]
        if suffix.isdigit() and 1 <= int(suffix) <= len(options):
            return _LETTERS[int(suffix) - 1]

    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(options):  # 1-based
            return _LETTERS[index - 1]
        if 0 <= index < len(options):  # 0-based
            return _LETTERS[index]

    for position, option in enumerate(options):
        if option and option.strip() == text:
            return _LETTERS[position]
    return None


def _options_from(row: dict[str, Any], columns: Sequence[str]) -> list[str]:
    """Options as a list, whether the dataset ships a column or a list field."""
    for name in ("options", "choices", "option_list"):
        if name in columns and isinstance(row.get(name), (list, tuple)):
            return [str(option) for option in row[name]]

    numbered: list[str] = []
    for index in range(1, 9):
        for pattern in (f"option{index}", f"option_{index}", f"opt{index}", _LETTERS[index - 1]):
            if pattern in columns and row.get(pattern) not in (None, ""):
                numbered.append(str(row[pattern]))
                break
        else:
            break
    return numbered


# ---------------------------------------------------------------------------
# MILU — knowledge / MCQ
# ---------------------------------------------------------------------------


def build_milu(cfg: Config) -> list[BaseItem]:
    block = dict(cfg.get("data.milu", {}) or {})
    if not block.get("enabled", False):
        return []

    dataset = str(block.get("hf_dataset", "ai4bharat/MILU"))
    split = str(block.get("split", "test"))
    n_per_language = block.get("n_per_language")
    seed = int(cfg.get("seed", 0))
    items: list[BaseItem] = []

    for language_name in block.get("languages", []) or []:
        code = MILU_LANG_CODE.get(language_name, str(language_name).lower())
        data = _load_split(dataset, str(language_name), split)
        if data is None:
            continue

        columns = list(data.column_names)
        question_col = _resolve_column(
            columns, ("question", "Question", "query", "prompt"), f"MILU:{language_name}"
        )
        answer_col = _resolve_column(
            columns,
            ("target", "answer", "Answer", "correct_option", "label", "gold"),
            f"MILU:{language_name}",
        )

        rows = _subsample(list(data), n_per_language, seed, f"milu␟{code}")
        kept = 0
        for row in rows:
            options = _options_from(row, columns)
            if len(options) < 2:
                continue
            letter = _answer_letter(row.get(answer_col), options)
            if letter is None:
                continue
            prompt = str(row[question_col]).strip()
            if not prompt:
                continue
            items.append(
                BaseItem(
                    item_id=item_id_for("milu", code, prompt, options),
                    task="milu",
                    language=code,
                    script=script_for(code),
                    prompt=prompt,
                    answer=letter,
                    answer_type=answer_type_for("milu"),
                    options=options,
                    meta={
                        "source": dataset,
                        "subject": row.get("subject") or row.get("domain"),
                        "milu_language": language_name,
                    },
                )
            )
            kept += 1
        log.info("milu/%s: %d items", code, kept)

    return items


# ---------------------------------------------------------------------------
# MGSM / IndicGenBench — reasoning / maths
# ---------------------------------------------------------------------------


def build_math(cfg: Config) -> list[BaseItem]:
    block = dict(cfg.get("data.math", {}) or {})
    if not block.get("enabled", False):
        return []

    seed = int(cfg.get("seed", 0))
    items: list[BaseItem] = []

    dataset = str(block.get("hf_dataset", "juletxara/mgsm"))
    split = str(block.get("split", "test"))
    n_per_language = block.get("n_per_language")

    for code in block.get("languages", []) or []:
        data = _load_split(dataset, str(code), split)
        if data is None:
            continue
        columns = list(data.column_names)
        question_col = _resolve_column(
            columns, ("question", "Question", "problem"), f"MGSM:{code}"
        )
        answer_col = _resolve_column(
            columns,
            ("answer_number", "answer", "solution", "target"),
            f"MGSM:{code}",
        )

        rows = _subsample(list(data), n_per_language, seed, f"math␟{code}")
        kept = 0
        for row in rows:
            prompt = str(row[question_col]).strip()
            answer = str(row.get(answer_col, "")).strip()
            if not prompt or not answer:
                continue
            items.append(
                BaseItem(
                    item_id=item_id_for("math", str(code), prompt),
                    task="math",
                    language=str(code),
                    script=script_for(str(code)),
                    prompt=prompt,
                    answer=answer,
                    answer_type=answer_type_for("math"),
                    options=None,
                    meta={"source": dataset},
                )
            )
            kept += 1
        log.info("math/%s: %d items", code, kept)

    igb = dict(block.get("indicgenbench", {}) or {})
    if igb.get("enabled", False):
        items.extend(_build_indicgenbench(cfg, igb, seed))
    elif "hi" not in {item.language for item in items}:
        log.info(
            "MGSM has no Hindi maths split; the maths task runs on %s. "
            "Set data.math.indicgenbench.enabled=true to add hi.",
            sorted({item.language for item in items}) or "no languages",
        )

    return items


def _build_indicgenbench(cfg: Config, block: dict[str, Any], seed: int) -> list[BaseItem]:
    dataset = str(block.get("hf_dataset", "ai4bharat/IndicGenBench_crosssum_in"))
    n_per_language = block.get("n_per_language")
    items: list[BaseItem] = []

    for code in block.get("languages", []) or []:
        data = _load_split(dataset, str(code), str(block.get("split", "test")))
        if data is None:
            continue
        columns = list(data.column_names)
        question_col = _resolve_column(
            columns, ("question", "input", "text", "source"), f"IndicGenBench:{code}"
        )
        answer_col = _resolve_column(
            columns, ("answer", "answer_number", "target", "output"), f"IndicGenBench:{code}"
        )
        for row in _subsample(list(data), n_per_language, seed, f"igb␟{code}"):
            prompt = str(row[question_col]).strip()
            answer = str(row.get(answer_col, "")).strip()
            if not prompt or not answer:
                continue
            items.append(
                BaseItem(
                    item_id=item_id_for("math", str(code), prompt),
                    task="math",
                    language=str(code),
                    script=script_for(str(code)),
                    prompt=prompt,
                    answer=answer,
                    answer_type=answer_type_for("math"),
                    options=None,
                    meta={"source": dataset},
                )
            )
        log.info("math/%s (IndicGenBench): %d items", code, len(items))
    return items


# ---------------------------------------------------------------------------
# IFEval — verifiable instruction following
# ---------------------------------------------------------------------------


def build_ifeval(cfg: Config) -> list[BaseItem]:
    block = dict(cfg.get("data.ifeval", {}) or {})
    if not block.get("enabled", False):
        return []

    language = str(block.get("language", "hi"))
    n_items = int(block.get("n_items", 100))
    seed = int(cfg.get("seed", 0))

    items: list[BaseItem] = []
    for generated in ifeval_mod.generate_items(n_items, seed=seed, language=language):
        prompt = generated["prompt"]
        items.append(
            BaseItem(
                item_id=item_id_for("ifeval", language, prompt + generated["id"]),
                task="ifeval",
                language=language,
                script=script_for(language),
                prompt=prompt,
                answer="",  # verified structurally, not against a gold string
                answer_type=answer_type_for("ifeval"),
                options=None,
                meta={
                    "source": "generated",
                    "constraints": generated["constraints"],
                    # Held in meta rather than in the prompt: the instruction
                    # wrapper stays constant English across conditions, so a
                    # drop cannot be the model failing to parse *our* text.
                    "instructions": generated["instructions"],
                },
            )
        )
    log.info("ifeval/%s: %d items", language, len(items))
    return items


# ---------------------------------------------------------------------------
# MultiJail — safety
# ---------------------------------------------------------------------------


def build_safety(cfg: Config) -> list[BaseItem]:
    block = dict(cfg.get("data.safety", {}) or {})
    if not block.get("enabled", False):
        return []

    seed = int(cfg.get("seed", 0))
    n_items = block.get("n_items")

    local = block.get("local_file")
    if local:
        return _safety_from_file(Path(str(local)), n_items, seed)

    dataset = str(block.get("hf_dataset", "DAMO-NLP-SG/MultiJail"))
    data = _load_split(dataset, None, str(block.get("split", "train")))
    if data is None:
        return []

    columns = list(data.column_names)
    items: list[BaseItem] = []

    for code in block.get("languages", []) or []:
        if str(code) not in columns:
            log.warning(
                "%s has no `%s` column (it has %s) — use data.safety.translate_to "
                "or data.safety.local_file for that language",
                dataset,
                code,
                [c for c in columns if len(c) == 2],
            )
            continue
        rows = _subsample(list(data), n_items, seed, f"safety␟{code}")
        for row in rows:
            prompt = str(row.get(str(code)) or "").strip()
            if not prompt:
                continue
            items.append(
                BaseItem(
                    item_id=item_id_for("safety", str(code), prompt),
                    task="safety",
                    language=str(code),
                    script=script_for(str(code)),
                    prompt=prompt,
                    answer="refuse",
                    answer_type=answer_type_for("safety"),
                    options=None,
                    meta={
                        "source": dataset,
                        "category": row.get("tags") or row.get("category"),
                        "translated": False,
                    },
                )
            )
        log.info("safety/%s: %d items", code, len(items))

    targets = list(block.get("translate_to") or [])
    if targets:
        items.extend(_translate_safety(cfg, data, block, targets, n_items, seed))

    return items


def _safety_from_file(path: Path, n_items: int | None, seed: int) -> list[BaseItem]:
    """Read verified translations supplied by the user."""
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"data.safety.local_file not found: {path}")

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items: list[BaseItem] = []
    for row in _subsample(rows, n_items, seed, "safety␟local"):
        prompt = str(row.get("prompt", "")).strip()
        code = str(row.get("lang", "hi"))
        if not prompt:
            continue
        items.append(
            BaseItem(
                item_id=item_id_for("safety", code, prompt),
                task="safety",
                language=code,
                script=script_for(code),
                prompt=prompt,
                answer="refuse",
                answer_type=answer_type_for("safety"),
                options=None,
                meta={"source": str(path), "category": row.get("category"), "translated": False},
            )
        )
    log.info("safety/local: %d items from %s", len(items), path)
    return items


def _translate_safety(
    cfg: Config,
    data: Any,
    block: dict[str, Any],
    targets: Sequence[str],
    n_items: int | None,
    seed: int,
) -> list[BaseItem]:
    """Machine-translate the English column into languages MultiJail lacks.

    Tagged ``translated: true`` so the paper can report these separately —
    machine translation is a weaker gold than the dataset's human translations.
    """
    model_id = str(block.get("translate_with", "ai4bharat/indictrans2-en-indic-1B"))
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokeniser = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        log.warning(
            "could not load the translation model %s (%s) — skipping %s. "
            "Point data.safety.local_file at verified translations instead.",
            model_id,
            exc,
            list(targets),
        )
        return []

    if "en" not in data.column_names:
        log.warning("no `en` column to translate from — skipping %s", list(targets))
        return []

    rows = _subsample(list(data), n_items, seed, "safety␟translate")
    items: list[BaseItem] = []
    for code in targets:
        for row in rows:
            source = str(row.get("en") or "").strip()
            if not source:
                continue
            try:
                encoded = tokeniser(source, return_tensors="pt", truncation=True)
                generated = model.generate(**encoded, max_new_tokens=256)
                prompt = tokeniser.batch_decode(generated, skip_special_tokens=True)[0].strip()
            except Exception as exc:  # pragma: no cover - model dependent
                log.debug("translation failed (%s)", exc)
                continue
            if not prompt:
                continue
            items.append(
                BaseItem(
                    item_id=item_id_for("safety", str(code), prompt),
                    task="safety",
                    language=str(code),
                    script=script_for(str(code)),
                    prompt=prompt,
                    answer="refuse",
                    answer_type=answer_type_for("safety"),
                    options=None,
                    meta={
                        "source": f"{block.get('hf_dataset')} (machine-translated)",
                        "category": row.get("tags") or row.get("category"),
                        "translated": True,
                        "translated_with": model_id,
                    },
                )
            )
        log.info("safety/%s: %d machine-translated items", code, len(items))
    return items


# ---------------------------------------------------------------------------
# MMMLU — the Arabizi universality sidebar
# ---------------------------------------------------------------------------


def build_arabizi(cfg: Config) -> list[BaseItem]:
    block = dict(cfg.get("data.arabizi", {}) or {})
    if not block.get("enabled", False):
        return []

    dataset = str(block.get("hf_dataset", "openai/MMMLU"))
    config = block.get("hf_config", "AR_XY")
    data = _load_split(dataset, str(config) if config else None, str(block.get("split", "test")))
    if data is None:
        return []

    columns = list(data.column_names)
    question_col = _resolve_column(columns, ("Question", "question"), "MMMLU")
    answer_col = _resolve_column(columns, ("Answer", "answer", "target"), "MMMLU")

    items: list[BaseItem] = []
    for row in _subsample(list(data), block.get("n_items"), int(cfg.get("seed", 0)), "arabizi"):
        options = _options_from(row, columns)
        if len(options) < 2:
            continue
        letter = _answer_letter(row.get(answer_col), options)
        prompt = str(row[question_col]).strip()
        if letter is None or not prompt:
            continue
        items.append(
            BaseItem(
                item_id=item_id_for("arabizi", "ar", prompt, options),
                task="arabizi",
                language="ar",
                script=script_for("ar"),
                prompt=prompt,
                answer=letter,
                answer_type=answer_type_for("arabizi"),
                options=options,
                meta={"source": dataset, "subject": row.get("Subject")},
            )
        )
    log.info("arabizi/ar: %d items", len(items))
    return items


#: Task -> builder. ``build.py`` walks this in ``TASKS`` order.
BUILDERS: dict[str, Callable[[Config], list[BaseItem]]] = {
    "milu": build_milu,
    "math": build_math,
    "ifeval": build_ifeval,
    "safety": build_safety,
    "arabizi": build_arabizi,
}


def build_all(cfg: Config, only: Iterable[str] | None = None) -> list[BaseItem]:
    """Run every enabled builder. A task that yields nothing is logged, not fatal."""
    wanted = set(only) if only else None
    items: list[BaseItem] = []
    for task, builder in BUILDERS.items():
        if wanted and task not in wanted:
            continue
        produced = builder(cfg)
        if not produced:
            log.warning("task %s produced 0 items", task)
        items.extend(produced)
    return items
