"""Inference driver: the GPU-day workhorse.

Work is partitioned into **shards** keyed ``task__language__condition``, one
directory per model. That is what makes runs separable by algorithm as well as
by time: a shard can be deleted and regenerated, a single model re-run, or a
single condition added later, without touching anything else.

Within a shard, generations are issued in write-chunks and appended to the
shard's JSONL after every chunk, so a crash loses at most one chunk. On restart
the shard's existing uids are skipped.

``--max-seconds`` lets a run stop cleanly inside a compute budget; the next
invocation with the same ``--run-id`` picks up exactly where it left off.

**Only one model is resident at a time.** The roster is walked sequentially: a
backend is built, every shard for that model is generated, the backend is torn
down and the card verified free, and only then does the next model load. On a
single 48 GB card that is what allows a 24 GB fp8 Sarvam-M and a 24 GB Gemma-3
in the same pass. See ``models/gpu.py`` for the teardown, and
``inference.isolate_models`` for per-model process isolation when in-process
teardown is not enough.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..checkpoint import ShardWriter, prune_shard, shard_is_done
from ..config import Config
from ..data.build import load_rendered
from ..data.schema import RenderedItem
from ..models import prompts as prompt_mod
from ..models import gpu
from ..models.backends import Backend, Generation, make_backend
from ..models.registry import ModelSpec
from ..paths import RunPaths
from ..progress import bar, track
from . import scoring

log = logging.getLogger(__name__)

STAGE = "inference"


class BudgetExhausted(Exception):
    """Raised internally when ``--max-seconds`` is hit; the run exits cleanly."""


@dataclass
class Shard:
    task: str
    language: str
    condition: str
    items: list[RenderedItem]

    @property
    def key(self) -> str:
        return f"{self.task}__{self.language}__{self.condition}"


def build_shards(
    items: Sequence[RenderedItem],
    tasks: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    limit_per_shard: int | None = None,
) -> list[Shard]:
    """Group rendered cells into shards, applying the CLI filters."""
    want_tasks = set(tasks) if tasks else None
    want_conditions = set(conditions) if conditions else None
    want_languages = set(languages) if languages else None

    grouped: dict[tuple[str, str, str], list[RenderedItem]] = defaultdict(list)
    for item in items:
        if want_tasks and item.task not in want_tasks:
            continue
        if want_conditions and item.condition not in want_conditions:
            continue
        if want_languages and item.language not in want_languages:
            continue
        grouped[(item.task, item.language, item.condition)].append(item)

    shards: list[Shard] = []
    for (task, language, condition), group in sorted(grouped.items()):
        group.sort(key=lambda i: (i.item_id, i.sample_idx))
        if limit_per_shard:
            group = group[:limit_per_shard]
        shards.append(Shard(task=task, language=language, condition=condition, items=group))
    return shards


def _record(
    item: RenderedItem,
    model_name: str,
    generation: Generation,
    elapsed: float,
    store_text: bool,
) -> dict[str, Any]:
    """Build one prediction record.

    Records are written to be **self-contained**: the prompt that was sent, its
    fingerprint, the raw completion, and why generation stopped. Re-analysis
    should never have to re-derive the prompt by joining against a suite file
    that may since have been rebuilt.
    """
    response = generation.text
    result = scoring.score(
        item.answer_type,
        response,
        item.answer,
        options=item.options,
        meta=item.meta,
    )
    record: dict[str, Any] = {
        "uid": item.uid,
        "item_id": item.item_id,
        "model": model_name,
        "task": item.task,
        "language": item.language,
        "script": item.script,
        "condition": item.condition,
        "sample_idx": item.sample_idx,
        "answer_type": item.answer_type,
        "gold": item.answer,
        "parsed": result.parsed,
        "correct": result.correct,
        "parse_failed": result.parse_failed,
        "detail": result.detail,
        "roundtrip_pass": item.roundtrip_pass,
        "transform_applied": item.transform_applied,
        "prompt_sha256": item.prompt_sha256,
        "response_chars": len(response or ""),
        "empty_response": not (response or "").strip(),
        "finish_reason": generation.finish_reason,
        "truncated": generation.truncated,
        "output_tokens": generation.output_tokens,
        "latency_s": round(elapsed, 4),
    }
    if store_text:
        # Safety completions and prompts are held out of the shared predictions
        # file and written to runs/<id>/raw/safety/ instead (see the ethics note).
        record["prompt"] = item.prompt
        record["options"] = item.options
        record["response"] = response
    return record


def run_shard(
    backend: Backend,
    spec: ModelSpec,
    shard: Shard,
    run: RunPaths,
    cfg: Config,
    deadline: float | None,
    overall: Any | None = None,
) -> dict[str, int]:
    """Generate, score and checkpoint one shard. Returns counters.

    ``overall`` is the model-level generation bar. It is advanced for resumed
    and skipped items too, so it always shows true position within the model
    rather than only the work done since this process started.
    """
    inference_cfg = dict(cfg.get("inference", {}) or {})
    max_tokens_block = dict(inference_cfg.get("max_new_tokens", {}) or {})
    write_chunk = int(inference_cfg.get("vllm", {}).get("chunk_size", 64) or 64)
    flush_every = int(inference_cfg.get("checkpoint_every", 10))

    # The safety task keeps completions out of the shared predictions file; only
    # the aggregate label lives there (ethics statement, spec §8).
    is_safety = shard.task == "safety"
    shard_path = run.shard(STAGE, spec.name, shard.key)

    # Evict predictions whose prompt no longer matches the rendered suite. The
    # uid alone cannot detect this: it identifies a cell, not its contents, so a
    # re-render after `scriptgap calibrate` keeps the uid and changes the text.
    fingerprints = {item.uid: item.prompt_sha256 for item in shard.items}

    def still_valid(record: dict[str, Any]) -> bool:
        expected = fingerprints.get(record.get("uid", ""))
        if expected is None or not expected:
            return True  # not in this shard's scope, or suite predates fingerprints
        return record.get("prompt_sha256") == expected

    _kept, stale = prune_shard(shard_path, still_valid)
    if stale:
        log.warning(
            "shard %s/%s: dropped %d prediction(s) whose prompt changed since they "
            "were generated — regenerating them",
            spec.name,
            shard.key,
            stale,
        )

    if shard_is_done(shard_path):
        log.info("shard %s/%s already complete — skipping", spec.name, shard.key)
        if overall is not None:
            overall.update(len(shard.items))
        return {"skipped": len(shard.items)}

    max_new = int(max_tokens_block.get(shard.task, 256))
    counters = {
        "generated": 0,
        "resumed": 0,
        "correct": 0,
        "parse_failed": 0,
        "truncated": 0,
        "stale_regenerated": stale,
    }

    with ShardWriter(shard_path, flush_every=flush_every) as writer:
        todo = [item for item in shard.items if item.uid not in writer.done]
        counters["resumed"] = writer.resumed_count
        # Items already on disk still count toward the model's position.
        if overall is not None and len(shard.items) > len(todo):
            overall.update(len(shard.items) - len(todo))
        if not todo:
            writer.mark_done({"model": spec.name, "shard": shard.key})
            log.info("shard %s/%s: nothing to do (%d cached)", spec.name, shard.key, writer.resumed_count)
            return counters

        log.info(
            "shard %s/%s: %d to generate (%d cached), max_new_tokens=%d",
            spec.name,
            shard.key,
            len(todo),
            writer.resumed_count,
            max_new,
        )

        raw_writer = None
        if is_safety:
            raw_writer = ShardWriter(run.raw_shard("safety", spec.name, shard.key), flush_every=flush_every)

        # One bar per shard. The backend reports into it via on_progress rather
        # than drawing its own, so the terminal shows models > shards > items
        # and not a third bar per write-chunk.
        progress = bar(f"{spec.name} · {shard.key}", total=len(todo), unit="gen", leave=False)
        try:
            for start in range(0, len(todo), write_chunk):
                if deadline is not None and time.time() > deadline:
                    raise BudgetExhausted(shard.key)

                chunk = todo[start : start + write_chunk]
                messages = [prompt_mod.build_messages(item) for item in chunk]

                t0 = time.time()
                responses = backend.generate(
                    messages,
                    max_new_tokens=max_new,
                    on_progress=progress.update,
                )
                per_item = (time.time() - t0) / max(1, len(chunk))

                for item, generation in zip(chunk, responses):
                    record = _record(
                        item, spec.name, generation, per_item, store_text=not is_safety
                    )
                    writer.write(record)
                    counters["generated"] += 1
                    if overall is not None:
                        overall.update(1)
                    if record["correct"]:
                        counters["correct"] += 1
                    if record["parse_failed"]:
                        counters["parse_failed"] += 1
                    if record["truncated"]:
                        counters["truncated"] += 1

                    if raw_writer is not None:
                        # The full record for safety, kept out of the shared file.
                        raw_writer.write(
                            {
                                "uid": item.uid,
                                "model": spec.name,
                                "prompt": item.prompt,
                                "response": generation.text,
                                "finish_reason": generation.finish_reason,
                            }
                        )
        finally:
            progress.close()
            if raw_writer is not None:
                raw_writer.close()

        writer.mark_done({"model": spec.name, "shard": shard.key, **counters})

    if counters["generated"]:
        log.info(
            "shard %s/%s done: %d generated, %d parse failures, %d truncated, %s",
            spec.name,
            shard.key,
            counters["generated"],
            counters["parse_failed"],
            counters["truncated"],
            f"acc={counters['correct'] / counters['generated']:.3f}"
            if not is_safety
            else "safety (aggregate only)",
        )
        if counters["truncated"]:
            log.warning(
                "  %d/%d generations hit max_new_tokens=%d — raise "
                "inference.max_new_tokens.%s or those look like errors",
                counters["truncated"],
                counters["generated"],
                max_new,
                shard.task,
            )
    return counters


def run(
    cfg: Config,
    run_paths: RunPaths,
    specs: Sequence[ModelSpec],
    tasks: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    limit_per_shard: int | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Run every (model x shard) cell, resuming whatever already exists."""
    items = load_rendered(cfg)
    shards = build_shards(
        items,
        tasks=tasks,
        conditions=conditions,
        languages=languages,
        limit_per_shard=limit_per_shard,
    )
    if not shards:
        raise SystemExit("no shards matched the given --task/--condition/--language filters")

    total_cells = sum(len(s.items) for s in shards)
    log.info(
        "plan: %d models x %d shards = %d generations (before resume)",
        len(specs),
        len(shards),
        total_cells * len(specs),
    )
    for shard in shards:
        log.debug("  shard %-40s %d items", shard.key, len(shard.items))

    deadline = time.time() + max_seconds if max_seconds else None
    summary: dict[str, Any] = {"models": {}, "budget_exhausted": False, "failed_models": {}}
    # On an unattended multi-hour run, one model failing to load (OOM, missing
    # weights, a bad quantisation combination) should not cost the remaining
    # models. The failure is recorded and reported at the end instead.
    continue_on_error = bool(cfg.get("inference.continue_on_model_error", True))

    gpu.log_memory("before the roster")
    min_free = float(cfg.get("inference.min_free_fraction", gpu.DEFAULT_FREE_FRACTION))

    model_bar = bar("models", total=len(specs), unit="model")
    for position, spec in enumerate(specs, start=1):
        inference_cfg = dict(cfg.get("inference", {}) or {})
        inference_cfg.setdefault("seed", cfg.get("seed"))
        totals = {
            "generated": 0,
            "resumed": 0,
            "correct": 0,
            "parse_failed": 0,
            "skipped": 0,
            "truncated": 0,
            "stale_regenerated": 0,
        }

        log.info(
            "=== model %d/%d: %s (%s / %s) ===",
            position,
            len(specs),
            spec.name,
            spec.role,
            spec.backend,
        )
        started = time.time()
        # Item-level progress across the whole model. The per-shard bar below
        # resets every shard and the shard bar only moves once per shard, so on
        # a multi-hour model neither answers "how far in are we, and until
        # when?". This one does, and it counts resumed items so the answer stays
        # right after a restart.
        generation_bar = bar(
            f"generations [{spec.name}]", total=total_cells, unit="gen", leave=False
        )
        try:
            with make_backend(spec, inference_cfg) as backend:
                for shard in track(shards, desc=f"shards [{spec.name}]", unit="shard", leave=False):
                    counters = run_shard(
                        backend, spec, shard, run_paths, cfg, deadline, overall=generation_bar
                    )
                    for key, value in counters.items():
                        totals[key] = totals.get(key, 0) + value
        except BudgetExhausted as exc:
            summary["budget_exhausted"] = True
            log.warning(
                "time budget exhausted during %s/%s — re-run with the same --run-id to resume",
                spec.name,
                exc,
            )
        except KeyboardInterrupt:
            log.warning("interrupted — predictions up to the last chunk are on disk")
            raise
        except Exception as exc:
            summary["failed_models"][spec.name] = f"{type(exc).__name__}: {exc}"
            log.exception("model %s failed", spec.name)
            if not continue_on_error:
                raise
            log.warning(
                "continuing with the remaining model(s); %s's shards are checkpointed "
                "and will resume on a later run",
                spec.name,
            )
        finally:
            generation_bar.close()
            # The backend's own close() already ran via the context manager;
            # this verifies the card actually came back before the next model
            # tries to allocate, and says so plainly if it did not.
            released, devices = gpu.wait_until_free(min_free_fraction=min_free, timeout=5.0)
            totals["wall_seconds"] = round(time.time() - started, 1)
            totals["gpu_released"] = released
            if devices:
                totals["gpu_free_gb_after"] = [d.free_gb for d in devices]
            summary["models"][spec.name] = totals
            log.info("model %s totals: %s", spec.name, totals)

            if devices and not released and position < len(specs):
                log.warning(
                    "only %.0f%% of the card is free after unloading %s, and %d model(s) "
                    "remain. If the next load OOMs, set inference.isolate_models=always.",
                    100 * min(d.free_fraction for d in devices),
                    spec.name,
                    len(specs) - position,
                )

        model_bar.update(1)
        if summary["budget_exhausted"]:
            break
    model_bar.close()

    if summary["failed_models"]:
        log.error(
            "%d of %d model(s) failed: %s",
            len(summary["failed_models"]),
            len(specs),
            ", ".join(summary["failed_models"]),
        )
    return summary
