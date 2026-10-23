"""Render base items into the five (or six) orthographic conditions.

This is the stage that makes the study a *controlled* comparison: one native
item in, N cells out, differing in nothing but orthography. The gold answer, the
option set and the task wrapper are held constant; only the item's surface form
changes.

Two resumable artefacts under ``data/suite/``:

``base.jsonl``
    Native-script gold items. Slow and network-bound, so it is reused unless
    ``--force``.

``rendered.jsonl``
    Every cell. Resumes per cell, keyed on uid.

``suite_manifest.json``
    Item counts, per-language gate pass rates, code-mix insertion counts and
    lexicon coverage, and the transform probabilities actually used. That last
    part matters: the operator probabilities are per-language and can be
    overridden three ways, so the manifest is what makes a rendered suite
    interpretable after the fact.

The gate is applied **per item, not per condition**: an item whose clean
romanisation does not transliterate back is flagged on every one of its cells,
so a prediction can be filtered without joining back to the suite. Failing items
are flagged, never deleted — the exclusion stays auditable and reversible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .. import ROOT
from ..checkpoint import ShardWriter, read_jsonl, write_json_atomic, write_jsonl_atomic
from ..config import Config, active_conditions
from ..paths import DATA_DIR
from ..progress import track
from . import arabizi, codemix, dakshina, noise, translit
from .builders import build_all
from .schema import BaseItem, RenderedItem, make_uid

log = logging.getLogger(__name__)

BASE_FILE = "base.jsonl"
RENDERED_FILE = "rendered.jsonl"
MANIFEST_FILE = "suite_manifest.json"

#: Keyword arguments :func:`scriptgap.data.noise.native_noise` accepts. The
#: config block is filtered against this so an unknown key is reported rather
#: than raising a TypeError deep inside a render loop.
_NATIVE_NOISE_OPS: tuple[str, ...] = (
    "matra_drop",
    "matra_swap",
    "nukta_drop",
    "anusvara_swap",
    "char_swap",
    "virama_drop",
)


def suite_dir(cfg: Config) -> Path:
    configured = str(cfg.get("data.out_dir", "data/suite"))
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# Base items
# ---------------------------------------------------------------------------


def build_base_items(
    cfg: Config,
    force: bool = False,
    only: Iterable[str] | None = None,
) -> list[BaseItem]:
    """Load the native-script gold items, building them if needed."""
    out = suite_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    path = out / BASE_FILE

    if path.exists() and not force:
        items = [BaseItem.from_dict(record) for record in read_jsonl(path)]
        if only:
            wanted = set(only)
            items = [item for item in items if item.task in wanted]
        if items:
            log.info("reusing %d base items from %s (pass --force to rebuild)", len(items), path)
            return items

    log.info("building base items%s", f" for {sorted(set(only))}" if only else "")
    items = build_all(cfg, only=only)
    if not items:
        raise SystemExit(
            "no base items were built — every configured task produced zero. "
            "Check dataset access (`hf auth login`) and data.* in the config."
        )

    write_jsonl_atomic(path, [item.to_dict() for item in items])
    log.info("wrote %d base items -> %s", len(items), path)
    return items


# ---------------------------------------------------------------------------
# Rendering one item
# ---------------------------------------------------------------------------


def _split_parts(parts: Sequence[str], item: BaseItem) -> tuple[str, list[str] | None]:
    """``[prompt, *options]`` back into ``(prompt, options)``."""
    prompt = parts[0]
    if not item.options:
        return prompt, None
    return prompt, list(parts[1:])


def render_item(
    item: BaseItem,
    cfg: Config,
    lexicons: dict[str, Any] | None = None,
) -> Iterator[RenderedItem]:
    """Yield every condition-and-sample cell for one base item.

    Deterministic: the same item and config always produce byte-identical cells,
    because every stochastic transform draws from a per-cell seeded RNG.
    """
    seed = cfg.get("seed", 0)
    n_samples = max(1, int(cfg.get("data.noisy_samples", 3) or 1))
    scheme = str(cfg.get("transforms.clean_roman.scheme", "ISO"))
    strip = bool(cfg.get("transforms.clean_roman.strip_diacritics", True))
    threshold = float(cfg.get("data.roundtrip_threshold", 0.95))
    conditions = active_conditions(cfg)

    parts = [item.prompt] + list(item.options or [])
    # Arabizi is intentionally irreversible — short vowels are unwritten in
    # Arabic and supplied ad libitum — so the sidebar has no round-trip gate and
    # is reported separately from the gated Indic results.
    is_arabizi = item.task == "arabizi" or item.script == "Arabic"

    gate_score = 1.0
    emitted_gate_score = 1.0
    if is_arabizi:
        clean_parts = [arabizi.to_arabizi(part) for part in parts]
        reversible_parts = list(clean_parts)
        roundtrip_pass = True
    else:
        romanised = [translit.romanise(part, item.script, scheme, strip) for part in parts]
        clean_parts = [r.emitted for r in romanised]
        reversible_parts = [r.reversible for r in romanised]
        gate_score = min((r.roundtrip_score for r in romanised), default=1.0)
        emitted_gate_score = min((r.emitted_roundtrip_score for r in romanised), default=1.0)
        roundtrip_pass = gate_score >= threshold

    def cell(
        condition: str,
        sample_idx: int,
        rendered_parts: Sequence[str],
        applied: bool,
        meta: dict[str, Any] | None = None,
    ) -> RenderedItem:
        prompt, options = _split_parts(rendered_parts, item)
        payload = dict(item.meta or {})
        payload.update(
            {
                "roundtrip_score": round(gate_score, 6),
                "emitted_roundtrip_score": round(emitted_gate_score, 6),
            }
        )
        payload.update(meta or {})
        return RenderedItem(
            uid=make_uid(item.item_id, condition, sample_idx),
            item_id=item.item_id,
            task=item.task,
            language=item.language,
            script=item.script,
            condition=condition,
            sample_idx=sample_idx,
            prompt=prompt,
            answer=item.answer,
            answer_type=item.answer_type,
            options=options,
            roundtrip_pass=roundtrip_pass,
            transform_applied=applied,
            meta=payload,
        )

    for condition in conditions:
        if condition == "native":
            yield cell("native", 0, parts, applied=True)

        elif condition == "clean_roman":
            yield cell("clean_roman", 0, clean_parts, applied=clean_parts != parts)

        elif condition == "noisy_roman":
            for sample_idx in range(n_samples):
                yield _render_noisy_roman(
                    item, cfg, cell, sample_idx, parts, reversible_parts,
                    clean_parts, seed, strip, is_arabizi,
                )

        elif condition == "native_noisy":
            for sample_idx in range(n_samples):
                yield _render_native_noisy(item, cfg, cell, sample_idx, parts, seed)

        elif condition == "codemix":
            yield _render_codemix(item, cfg, cell, parts, seed)

        elif condition == "attested_roman":
            for sample_idx in range(n_samples):
                yield _render_attested(item, cell, sample_idx, parts, seed, lexicons)


def _render_noisy_roman(
    item, cfg, cell, sample_idx, parts, reversible_parts, clean_parts, seed, strip, is_arabizi
) -> RenderedItem:
    rng = noise.rng_for(seed, item.item_id, "noisy_roman", sample_idx)

    if is_arabizi:
        rendered = [arabizi.to_arabizi(part, rng=rng) for part in parts]
        return cell(
            "noisy_roman", sample_idx, rendered, applied=rendered != clean_parts
        )

    probs = noise.probs_for_language(cfg, item.language, "noisy_roman")
    rendered: list[str] = []
    fired: set[str] = set()
    applied = False
    for reversible in reversible_parts:
        # Operators run on the ISO form, where `ā` vs `a` and `ś` vs `s` are
        # still distinguishable, then the result is de-diacritised. So a cell
        # where nothing fired is byte-identical to the clean romanisation
        # rather than to something arbitrary.
        noised, report = noise.noisy_roman(reversible, rng, probs)
        rendered.append(translit.strip_diacritics(noised) if strip else noised)
        fired.update(report.operators_fired)
        applied = applied or report.changed

    return cell(
        "noisy_roman",
        sample_idx,
        rendered,
        applied=applied,
        meta={"noise_operators": sorted(fired)},
    )


def _render_native_noisy(item, cfg, cell, sample_idx, parts, seed) -> RenderedItem:
    rng = noise.rng_for(seed, item.item_id, "native_noisy", sample_idx)
    probs = noise.probs_for_language(cfg, item.language, "native_noisy")
    kwargs = {name: probs[name] for name in _NATIVE_NOISE_OPS if name in probs}
    unknown = set(probs) - set(_NATIVE_NOISE_OPS)
    if unknown:
        log.warning(
            "transforms.native_noisy has unknown operator(s) %s — ignored",
            sorted(unknown),
        )

    rendered = [noise.native_noise(part, item.script, rng, **kwargs) for part in parts]
    return cell("native_noisy", sample_idx, rendered, applied=rendered != list(parts))


def _render_codemix(item, cfg, cell, parts, seed) -> RenderedItem:
    rng = noise.rng_for(seed, item.item_id, "codemix", 0)
    rate = float(cfg.get("transforms.codemix.insertion_rate", 0.3) or 0.0)
    max_insertions = int(cfg.get("transforms.codemix.max_insertions", 6) or 0)
    discourse_rate = float(cfg.get("transforms.codemix.discourse_rate", 0.25) or 0.0)

    rendered: list[str] = []
    lexical = discourse = candidates = 0
    for index, part in enumerate(parts):
        mixed, report = codemix.code_mix(
            part,
            item.language,
            rng,
            rate,
            max_insertions,
            # Discourse markers belong in the question, not spliced into an
            # answer option, where they would change what the option means.
            discourse_rate=discourse_rate if index == 0 else 0.0,
        )
        rendered.append(mixed)
        lexical += report.lexical
        discourse += report.discourse
        candidates += report.candidates

    return cell(
        "codemix",
        0,
        rendered,
        applied=(lexical + discourse) > 0,
        meta={
            "codemix_lexical": lexical,
            "codemix_discourse": discourse,
            "codemix_candidates": candidates,
        },
    )


def _render_attested(item, cell, sample_idx, parts, seed, lexicons) -> RenderedItem:
    rng = noise.rng_for(seed, item.item_id, "attested_roman", sample_idx)
    lexicon = (lexicons or {}).get(item.language)

    rendered: list[str] = []
    tokens = attested = fallback = 0
    for part in parts:
        out, report = dakshina.attested_romanise(part, lexicon, item.script, rng)
        rendered.append(out)
        tokens += report.tokens
        attested += report.attested
        fallback += report.fallback

    coverage = round(attested / tokens, 6) if tokens else 0.0
    return cell(
        "attested_roman",
        sample_idx,
        rendered,
        # A cell with no attested token is a scheme transliteration wearing the
        # attested label — or, with no lexicon at all, a copy of the native
        # item. Either way it is not an observation of this condition.
        applied=attested > 0,
        meta={
            "attested_coverage": coverage,
            "attested_tokens": attested,
            "attested_fallback": fallback,
        },
    )


# ---------------------------------------------------------------------------
# Building the whole suite
# ---------------------------------------------------------------------------


def _load_lexicons(cfg: Config) -> dict[str, Any]:
    """Dakshina lexicons for the opt-in ``attested_roman`` condition."""
    if not cfg.get("data.dakshina.enabled", False):
        return {}
    root = dakshina.ensure(cfg)
    if root is None:
        log.warning(
            "data.dakshina.enabled is true but no corpus was found — "
            "attested_roman cells will be marked unapplied"
        )
        return {}
    splits = list(cfg.get("data.dakshina.splits", ["train", "dev"]))
    return dakshina.load_lexicons(root, dakshina.DAKSHINA_LANGS, splits)


def build(
    cfg: Config,
    force: bool = False,
    only: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build base items and render every condition. Returns the suite manifest."""
    out = suite_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    rendered_path = out / RENDERED_FILE

    items = build_base_items(cfg, force=force, only=only)
    lexicons = _load_lexicons(cfg)
    conditions = active_conditions(cfg)
    log.info("rendering %d items x %s", len(items), list(conditions))

    if force and rendered_path.exists():
        rendered_path.unlink()
        rendered_path.with_suffix(rendered_path.suffix + ".done").unlink(missing_ok=True)

    stats = _Stats()
    written = 0
    with ShardWriter(rendered_path, flush_every=500) as writer:
        if writer.resumed_count:
            log.info("resuming: %d cells already rendered", writer.resumed_count)
        for item in track(items, desc="rendering items", unit="item"):
            for cell in render_item(item, cfg, lexicons):
                stats.observe(cell)
                if cell.uid in writer.done:
                    continue
                writer.write(cell.to_dict())
                written += 1
        writer.mark_done({"cells": stats.cells})

    manifest = stats.to_manifest(cfg, items, lexicons, conditions)
    write_json_atomic(out / MANIFEST_FILE, manifest)

    log.info(
        "rendered %d cells (%d newly written) from %d items -> %s",
        stats.cells,
        written,
        len(items),
        rendered_path,
    )
    for language, gate in sorted(manifest["gate"].items()):
        log.info(
            "  gate %-4s pass %d/%d (%.1f%%), mean score %.3f",
            language,
            gate["passed"],
            gate["items"],
            100 * gate["pass_rate"],
            gate["mean_score"],
        )
    if stats.unapplied:
        log.info("cells whose transform did not fire: %s", dict(sorted(stats.unapplied.items())))

    _write_codemix_spotcheck(cfg, out, rendered_path)
    return manifest


class _Stats:
    """Counters for the suite manifest."""

    def __init__(self) -> None:
        self.cells = 0
        self.by_condition: dict[str, int] = {}
        self.by_task: dict[str, int] = {}
        self.by_language: dict[str, int] = {}
        self.unapplied: dict[str, int] = {}
        self.gate_items: dict[str, set[str]] = {}
        self.gate_passed: dict[str, set[str]] = {}
        self.gate_scores: dict[str, list[float]] = {}
        self.codemix_lexical = 0
        self.codemix_discourse = 0
        self.codemix_candidates = 0
        self.attested_coverage: list[float] = []

    def observe(self, cell: RenderedItem) -> None:
        self.cells += 1
        self.by_condition[cell.condition] = self.by_condition.get(cell.condition, 0) + 1
        self.by_task[cell.task] = self.by_task.get(cell.task, 0) + 1
        self.by_language[cell.language] = self.by_language.get(cell.language, 0) + 1
        if not cell.transform_applied:
            self.unapplied[cell.condition] = self.unapplied.get(cell.condition, 0) + 1

        if cell.condition == "native":
            self.gate_items.setdefault(cell.language, set()).add(cell.item_id)
            if cell.roundtrip_pass:
                self.gate_passed.setdefault(cell.language, set()).add(cell.item_id)
            score = cell.meta.get("roundtrip_score")
            if score is not None:
                self.gate_scores.setdefault(cell.language, []).append(float(score))

        if cell.condition == "codemix":
            self.codemix_lexical += int(cell.meta.get("codemix_lexical", 0))
            self.codemix_discourse += int(cell.meta.get("codemix_discourse", 0))
            self.codemix_candidates += int(cell.meta.get("codemix_candidates", 0))

        if cell.condition == "attested_roman":
            self.attested_coverage.append(float(cell.meta.get("attested_coverage", 0.0)))

    def to_manifest(
        self,
        cfg: Config,
        items: Sequence[BaseItem],
        lexicons: dict[str, Any],
        conditions: Sequence[str],
    ) -> dict[str, Any]:
        gate: dict[str, dict[str, Any]] = {}
        for language, item_ids in self.gate_items.items():
            passed = len(self.gate_passed.get(language, set()))
            scores = self.gate_scores.get(language, [])
            gate[language] = {
                "items": len(item_ids),
                "passed": passed,
                "pass_rate": round(passed / len(item_ids), 4) if item_ids else 0.0,
                "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            }

        languages = sorted(self.by_language)
        return {
            "conditions": list(conditions),
            "noisy_samples": int(cfg.get("data.noisy_samples", 3)),
            "roundtrip_threshold": float(cfg.get("data.roundtrip_threshold", 0.95)),
            "stats": {
                "items": len(items),
                "cells": self.cells,
                "by_condition": dict(sorted(self.by_condition.items())),
                "by_task": dict(sorted(self.by_task.items())),
                "by_language": dict(sorted(self.by_language.items())),
                "transform_not_applied": dict(sorted(self.unapplied.items())),
            },
            "gate": gate,
            "codemix": {
                "lexical_insertions": self.codemix_lexical,
                "discourse_insertions": self.codemix_discourse,
                "candidate_sites": self.codemix_candidates,
                "lexicon_sizes": {
                    language: len(codemix.lexicon_for(language)) for language in languages
                },
            },
            "attested": {
                "lexicons": sorted(lexicons),
                "mean_coverage": (
                    round(sum(self.attested_coverage) / len(self.attested_coverage), 4)
                    if self.attested_coverage
                    else None
                ),
            },
            # The probabilities actually used, per language. Three things can
            # set these (global block, per-language block, --set), so without
            # this a rendered suite cannot be interpreted after the fact.
            "transform_probabilities": {
                "noisy_roman": {
                    language: noise.probs_for_language(cfg, language, "noisy_roman")
                    for language in languages
                },
                "native_noisy": {
                    language: noise.probs_for_language(cfg, language, "native_noisy")
                    for language in languages
                },
                "codemix": dict(cfg.get("transforms.codemix", {}) or {}),
                "clean_roman": dict(cfg.get("transforms.clean_roman", {}) or {}),
            },
        }


def _write_codemix_spotcheck(cfg: Config, out: Path, rendered_path: Path, n: int = 50) -> None:
    """Native/code-mixed pairs for the spec's 50-item native-speaker review.

    The repo can generate the sheet but not fill it in; the reviewer columns are
    left blank on purpose.
    """
    natives: dict[str, str] = {}
    mixed: list[dict[str, Any]] = []
    for record in read_jsonl(rendered_path):
        if record.get("condition") == "native":
            natives[record["item_id"]] = record.get("prompt", "")
        elif record.get("condition") == "codemix" and record.get("transform_applied"):
            mixed.append(record)

    if not mixed:
        return

    import random as _random

    rng = _random.Random(f"{cfg.get('seed', 0)}␟codemix-spotcheck")
    rng.shuffle(mixed)
    rows = [
        {
            "uid": record["uid"],
            "language": record["language"],
            "native": natives.get(record["item_id"], ""),
            "codemixed": record.get("prompt", ""),
            "lexical_insertions": record.get("meta", {}).get("codemix_lexical"),
            "discourse_insertions": record.get("meta", {}).get("codemix_discourse"),
            "is_natural": "",
            "violates_free_morpheme": "",
            "reviewer_notes": "",
        }
        for record in mixed[:n]
    ]
    target = DATA_DIR / "human_check" / f"codemix_spotcheck_{len(rows)}.jsonl"
    write_jsonl_atomic(target, rows)
    log.info("wrote %d code-mix pairs for native-speaker review -> %s", len(rows), target)


# ---------------------------------------------------------------------------
# Reading the suite back
# ---------------------------------------------------------------------------


def load_rendered(cfg: Config) -> list[RenderedItem]:
    """Every rendered cell, as written by :func:`build`."""
    path = suite_dir(cfg) / RENDERED_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found — run `scriptgap build-data` first")
    items = [RenderedItem.from_dict(record) for record in read_jsonl(path)]
    if not items:
        raise SystemExit(f"{path} is empty — re-run `scriptgap build-data --force`")
    return items
