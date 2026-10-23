"""H6 — within-language fertility: is the register users prefer cheaper to serve?

Prior token-tax work compares *across* languages (Hindi costs more than
English). The contribution here is the *within-language* contrast: for the same
content, does the romanised rendering cost more or fewer tokens than the native
one, under the tokenisers actually deployed?

For every item we tokenise the native prompt and each romanised prompt with
every tokeniser in scope and record:

* ``tokens`` — raw count
* ``fertility`` — tokens per character (the standard normalisation, so a longer
  romanised string is not credited as "more efficient" merely for being longer)
* ``token_ratio`` — romanised tokens / native tokens, the headline number

CPU-only, so this can run while the GPU is busy or afterwards on a laptop.
Results are checkpointed per tokeniser, so adding a tokeniser later does not
re-tokenise the whole suite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..checkpoint import ShardWriter, read_jsonl
from ..config import Config
from ..data.build import load_rendered
from ..models.registry import ModelSpec
from ..paths import RunPaths
from ..progress import bar

log = logging.getLogger(__name__)

STAGE = "fertility"

#: Conditions compared against ``native``.
COMPARED_CONDITIONS = (
    "clean_roman",
    "noisy_roman",
    "attested_roman",
    "codemix",
    "native_noisy",
)


@dataclass
class Tokeniser:
    """A named callable that returns a token count for a string."""

    name: str
    kind: str  # "model" | "external" | "tiktoken"
    encode: Callable[[str], int]


def _hf_tokeniser(name: str, source: str, kind: str) -> Tokeniser | None:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    except Exception as exc:
        log.warning("could not load tokeniser %s from %s (%s) — skipping", name, source, exc)
        return None

    def encode(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return Tokeniser(name=name, kind=kind, encode=encode)


def _tiktoken_tokeniser(encoding: str) -> Tokeniser | None:
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding)
    except Exception as exc:
        log.warning("could not load tiktoken encoding %s (%s) — skipping", encoding, exc)
        return None

    return Tokeniser(
        name=f"tiktoken:{encoding}",
        kind="tiktoken",
        encode=lambda text: len(enc.encode(text, disallowed_special=())),
    )


def collect_tokenisers(cfg: Config, specs: Sequence[ModelSpec]) -> list[Tokeniser]:
    """Model tokenisers + the external ones named in the config."""
    fcfg = dict(cfg.get("fertility", {}) or {})
    out: list[Tokeniser] = []
    seen: set[str] = set()

    for spec in specs:
        if not spec.hf_id:
            continue  # API-served models expose no local tokeniser
        source = spec.resolve_path()
        if source in seen:
            continue
        seen.add(source)
        tok = _hf_tokeniser(spec.name, source, kind="model")
        if tok:
            out.append(tok)

    for source in fcfg.get("external_tokenizers", []) or []:
        if source in seen:
            continue
        seen.add(source)
        tok = _hf_tokeniser(f"external:{source}", source, kind="external")
        if tok:
            out.append(tok)

    for encoding in fcfg.get("tiktoken_encodings", []) or []:
        tok = _tiktoken_tokeniser(encoding)
        if tok:
            out.append(tok)

    log.info("fertility tokenisers: %s", [t.name for t in out])
    return out


def run(
    cfg: Config,
    run_paths: RunPaths,
    specs: Sequence[ModelSpec],
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    items = load_rendered(cfg)
    want_tasks = set(tasks) if tasks else None

    # Index by item so native/roman pairs line up. For the stochastic
    # conditions we use sample 0, so the pairing is one-to-one.
    native: dict[str, Any] = {}
    variants: dict[tuple[str, str], Any] = {}
    for item in items:
        if want_tasks and item.task not in want_tasks:
            continue
        if item.condition == "native":
            native[item.item_id] = item
        elif item.condition in COMPARED_CONDITIONS and item.sample_idx == 0:
            variants[(item.item_id, item.condition)] = item

    if not native:
        raise SystemExit("no native-condition items found — run `scriptgap build-data` first")

    tokenisers = collect_tokenisers(cfg, specs)
    if not tokenisers:
        raise SystemExit("no tokenisers could be loaded — check fertility.* in the config")

    summary: dict[str, Any] = {"tokenisers": {}}

    for tokeniser in tokenisers:
        shard_path = run_paths.shard(STAGE, tokeniser.name, "pairs")
        rows: list[dict[str, Any]] = []

        with ShardWriter(shard_path, flush_every=500) as writer:
            rows.extend(read_jsonl(shard_path))
            todo = [
                (item_id, base)
                for item_id, base in native.items()
                if f"{item_id}::{tokeniser.name}" not in writer.done
            ]
            if writer.resumed_count:
                log.info(
                    "%s: %d pairs already tokenised, %d to go",
                    tokeniser.name,
                    writer.resumed_count,
                    len(todo),
                )

            progress = bar(f"fertility {tokeniser.name}", total=len(todo), unit="item")
            for item_id, base in todo:
                native_chars = max(1, len(base.prompt))
                try:
                    native_tokens = tokeniser.encode(base.prompt)
                except Exception as exc:
                    log.warning("tokenisation failed for %s (%s)", item_id, exc)
                    progress.update(1)
                    continue

                record: dict[str, Any] = {
                    "uid": f"{item_id}::{tokeniser.name}",
                    "item_id": item_id,
                    "tokeniser": tokeniser.name,
                    "tokeniser_kind": tokeniser.kind,
                    "task": base.task,
                    "language": base.language,
                    "script": base.script,
                    "native_tokens": native_tokens,
                    "native_chars": native_chars,
                    "native_fertility": round(native_tokens / native_chars, 6),
                }

                for condition in COMPARED_CONDITIONS:
                    variant = variants.get((item_id, condition))
                    if variant is None:
                        continue
                    chars = max(1, len(variant.prompt))
                    try:
                        tokens = tokeniser.encode(variant.prompt)
                    except Exception:
                        continue
                    record[f"{condition}_tokens"] = tokens
                    record[f"{condition}_chars"] = chars
                    record[f"{condition}_fertility"] = round(tokens / chars, 6)
                    record[f"{condition}_token_ratio"] = (
                        round(tokens / native_tokens, 6) if native_tokens else None
                    )

                writer.write(record)
                rows.append(record)
                progress.update(1)
            progress.close()
            writer.mark_done({"tokeniser": tokeniser.name})

        summary["tokenisers"][tokeniser.name] = _summarise(rows)
        head = summary["tokenisers"][tokeniser.name].get("clean_roman_token_ratio_mean")
        log.info(
            "%-40s clean_roman/native token ratio = %s",
            tokeniser.name,
            f"{head:.3f}" if head is not None else "n/a",
        )

    return summary


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    out: dict[str, Any] = {"n": len(rows), "native_fertility_mean": mean("native_fertility")}
    for condition in COMPARED_CONDITIONS:
        out[f"{condition}_fertility_mean"] = mean(f"{condition}_fertility")
        out[f"{condition}_token_ratio_mean"] = mean(f"{condition}_token_ratio")
    return out
