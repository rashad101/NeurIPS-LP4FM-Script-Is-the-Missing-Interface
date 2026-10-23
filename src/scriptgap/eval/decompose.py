"""H5 — failure-mode decomposition: comprehension vs. action.

For each romanised failure the judge sees three things: the *native-script*
version of the item (the ground truth of what was asked), the romanised prompt
the model actually received, and the model's response. It then labels the
failure:

``misunderstood_topic``
    The response is about something else — the model did not recover the item's
    content from the romanised string. A comprehension failure.

``understood_but_wrong``
    The response is on-topic and addresses the right question, but the answer or
    the action is wrong. A post-comprehension failure.

``refused_or_empty`` / ``unclear``
    Escape hatches, so the judge is never forced into a bad label. Reported
    separately rather than folded into either arm.

Why this matters for the position: an *action-dominant* profile means a
normalisation front-end — which only attacks the comprehension stage — is
addressing the wrong part of the pipeline (paper §7a, §10a). The two escape
labels exist so that claim rests on the two substantive labels alone.

Judging is checkpointed per model, so an interrupted or rate-limited judging
pass resumes instead of re-paying for the whole set.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..checkpoint import ShardWriter, read_jsonl, write_jsonl_atomic
from ..config import Config
from ..data.build import load_rendered
from ..paths import RunPaths
from ..progress import bar
from .. import ROOT

log = logging.getLogger(__name__)

STAGE = "decompose"

LABELS = ("misunderstood_topic", "understood_but_wrong", "refused_or_empty", "unclear")

#: The two labels the H5 claim is actually about.
SUBSTANTIVE_LABELS = ("misunderstood_topic", "understood_but_wrong")

JUDGE_SYSTEM = """You are an expert annotator for a study of how language models handle romanised Indic text.

You will be shown:
1. NATIVE ITEM — the original question in its native script (the ground truth of what was asked).
2. ROMANISED PROMPT — the same question rewritten in Latin script, which is what the model received.
3. MODEL RESPONSE — what the model produced. It is known to be incorrect.

Decide *where* the failure happened and return one label:

- "misunderstood_topic": the response addresses a different question, entity, or
  domain than the native item. The model failed to recover the content of the
  item from the romanised string.
- "understood_but_wrong": the response clearly addresses the question that was
  actually asked — right topic, right entities, right task — but the answer,
  computation, or action is wrong.
- "refused_or_empty": the response is empty, a refusal, boilerplate, or pure
  restatement with no attempt at an answer.
- "unclear": you genuinely cannot tell which of the first two applies.

Judge only comprehension versus action. Do not re-grade correctness — the
response is already known to be wrong. Be strict about
"understood_but_wrong": it requires positive evidence that the model grasped
the specific question, not merely that it produced fluent text."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {
            "type": "string",
            "description": "One sentence quoting or paraphrasing the decisive part of the response.",
        },
    },
    "required": ["label", "confidence", "evidence"],
    "additionalProperties": False,
}

#: Conditions whose failures are in scope for H5. ``attested_roman`` is included
#: when it was rendered: a failure on text a native speaker actually wrote is the
#: most defensible instance of the phenomenon H5 decomposes.
ROMANISED_CONDITIONS = ("clean_roman", "noisy_roman", "attested_roman")

#: Tasks with a well-defined "topic" for the judge to check against.
DECOMPOSABLE_TASKS = ("milu", "math", "arabizi")


@dataclass
class Failure:
    uid: str
    model: str
    task: str
    language: str
    condition: str
    native_prompt: str
    romanised_prompt: str
    response: str
    gold: str
    options: list[str] | None


# ---------------------------------------------------------------------------
# Selecting failures
# ---------------------------------------------------------------------------


def collect_failures(
    run: RunPaths,
    cfg: Config,
    models: Sequence[str] | None = None,
    n_per_model: int = 100,
    only_gated: bool = True,
) -> dict[str, list[Failure]]:
    """Gather romanised failures per model, subsampled to ``n_per_model``."""
    rendered = load_rendered(cfg)
    native: dict[str, Any] = {i.item_id: i for i in rendered if i.condition == "native"}
    by_uid: dict[str, Any] = {i.uid: i for i in rendered}

    pred_root = run.predictions / "inference"
    if not pred_root.exists():
        raise SystemExit(f"{pred_root} not found — run `scriptgap run-inference` first")

    seed = int(cfg.get("seed", 0))
    out: dict[str, list[Failure]] = {}

    model_dirs = sorted(p for p in pred_root.iterdir() if p.is_dir())
    for model_dir in model_dirs:
        model_name = model_dir.name
        if models and model_name not in {m for m in models}:
            continue

        pool: list[Failure] = []
        shard_files = sorted(model_dir.glob("*.jsonl"))
        progress = bar(f"scanning {model_name}", total=len(shard_files), unit="shard")
        for shard_file in shard_files:
            for record in read_jsonl(shard_file):
                if record.get("condition") not in ROMANISED_CONDITIONS:
                    continue
                if record.get("task") not in DECOMPOSABLE_TASKS:
                    continue
                if record.get("correct") is not False:
                    continue
                if only_gated and not record.get("roundtrip_pass", True):
                    continue
                item = by_uid.get(record["uid"])
                base = native.get(record.get("item_id", ""))
                if item is None or base is None:
                    continue
                pool.append(
                    Failure(
                        uid=record["uid"],
                        model=model_name,
                        task=record["task"],
                        language=record["language"],
                        condition=record["condition"],
                        native_prompt=base.prompt,
                        romanised_prompt=item.prompt,
                        response=record.get("response", ""),
                        gold=record.get("gold", ""),
                        options=item.options,
                    )
                )
            progress.update(1)
        progress.close()

        rng = random.Random(f"{seed}␟decompose␟{model_name}")
        rng.shuffle(pool)
        out[model_name] = pool[:n_per_model]
        log.info(
            "%s: %d romanised failures in scope, judging %d",
            model_name,
            len(pool),
            len(out[model_name]),
        )
    return out


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class Judge:
    """Anthropic-backed LLM judge with structured output."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "high", max_retries: int = 5) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise SystemExit(
                "the H5 judge needs the API extra: `uv pip install -e '.[api]'`"
            ) from exc
        self.anthropic = anthropic
        self.model = model
        self.effort = effort
        self.max_retries = max_retries
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.info("ANTHROPIC_API_KEY unset; the SDK will use an `ant auth login` profile")
        self.client = anthropic.Anthropic()

    def _user_block(self, failure: Failure) -> str:
        parts = [
            "NATIVE ITEM (native script, ground truth of what was asked):",
            failure.native_prompt,
            "",
            "ROMANISED PROMPT (what the model actually received):",
            failure.romanised_prompt,
        ]
        if failure.options:
            parts += ["", "OPTIONS SHOWN TO THE MODEL:", *(f"- {o}" for o in failure.options)]
        parts += [
            "",
            f"GOLD ANSWER: {failure.gold}",
            "",
            "MODEL RESPONSE (known to be incorrect):",
            failure.response or "<empty>",
        ]
        return "\n".join(parts)

    def judge(self, failure: Failure) -> dict[str, Any]:
        delay = 2.0
        for _attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": self._user_block(failure)}],
                    output_config={
                        "effort": self.effort,
                        "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                    },
                )
                if response.stop_reason == "refusal":
                    return {"label": "unclear", "confidence": "low", "evidence": "judge refused"}
                text = next((b.text for b in response.content if b.type == "text"), "")
                parsed = json.loads(text)
                if parsed.get("label") not in LABELS:
                    parsed["label"] = "unclear"
                return parsed
            except json.JSONDecodeError as exc:
                log.warning("judge returned unparseable JSON (%s); retrying", exc)
            except self.anthropic.RateLimitError:
                log.warning("judge rate limited; backing off %.0fs", delay)
            except self.anthropic.APIStatusError as exc:
                if exc.status_code < 500:
                    log.warning("judge API error %s: %s", exc.status_code, exc.message)
                    return {"label": "unclear", "confidence": "low", "evidence": f"api error {exc.status_code}"}
                log.warning("judge server error %s; backing off %.0fs", exc.status_code, delay)
            except self.anthropic.APIConnectionError as exc:
                log.warning("judge connection error (%s); backing off %.0fs", exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        return {"label": "unclear", "confidence": "low", "evidence": "judge exhausted retries"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _sample_for_human_check(
    judged: dict[str, list[dict[str, Any]]], n: int, seed: int
) -> list[dict[str, Any]]:
    """Blind sample for the 30-item human agreement check (spec §5)."""
    flat = [record for records in judged.values() for record in records]
    rng = random.Random(f"{seed}␟human-check")
    rng.shuffle(flat)
    out = []
    for record in flat[:n]:
        out.append(
            {
                "uid": record["uid"],
                "model": record["model"],
                "task": record["task"],
                "condition": record["condition"],
                "native_prompt": record["native_prompt"],
                "romanised_prompt": record["romanised_prompt"],
                "response": record["response"],
                "gold": record["gold"],
                # judge label withheld so the human annotation is blind
                "human_label": "",
                "human_notes": "",
            }
        )
    return out


def run(
    cfg: Config,
    run_paths: RunPaths,
    models: Iterable[str] | None = None,
) -> dict[str, Any]:
    dcfg = dict(cfg.get("decompose", {}) or {})
    if not dcfg.get("enabled", True):
        log.info("decompose.enabled is false — skipping H5")
        return {"skipped": True}

    n_per_model = int(dcfg.get("n_failures_per_model", 100))
    failures = collect_failures(
        run_paths, cfg, models=list(models) if models else None, n_per_model=n_per_model
    )
    if not any(failures.values()):
        log.warning("no romanised failures found — nothing to decompose")
        return {"judged": 0}

    judge = Judge(
        model=str(dcfg.get("judge_model", "claude-opus-5")),
        effort=str(dcfg.get("judge_effort", "high")),
    )

    judged: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {"models": {}}

    for model_name, pool in failures.items():
        if not pool:
            continue
        shard_path = run_paths.shard(STAGE, model_name, "failures")
        records: list[dict[str, Any]] = []

        with ShardWriter(shard_path, flush_every=5) as writer:
            records.extend(read_jsonl(shard_path))
            todo = [f for f in pool if f.uid not in writer.done]
            log.info(
                "judging %s: %d items (%d already judged)",
                model_name,
                len(todo),
                writer.resumed_count,
            )

            progress = bar(f"judge {model_name}", total=len(todo), unit="item")
            for failure in todo:
                verdict = judge.judge(failure)
                record = {
                    "uid": failure.uid,
                    "model": model_name,
                    "task": failure.task,
                    "language": failure.language,
                    "condition": failure.condition,
                    "native_prompt": failure.native_prompt,
                    "romanised_prompt": failure.romanised_prompt,
                    "response": failure.response,
                    "gold": failure.gold,
                    "judge_model": judge.model,
                    "label": verdict.get("label", "unclear"),
                    "confidence": verdict.get("confidence", "low"),
                    "evidence": verdict.get("evidence", ""),
                }
                writer.write(record)
                records.append(record)
                progress.update(1)
            progress.close()
            writer.mark_done({"model": model_name})

        judged[model_name] = records
        counts: dict[str, int] = {label: 0 for label in LABELS}
        for record in records:
            counts[record.get("label", "unclear")] = counts.get(record.get("label", "unclear"), 0) + 1
        substantive = sum(counts[label] for label in SUBSTANTIVE_LABELS)
        summary["models"][model_name] = {
            "counts": counts,
            "n": len(records),
            "action_share_of_substantive": (
                round(counts["understood_but_wrong"] / substantive, 4) if substantive else None
            ),
        }
        log.info("%s decomposition: %s", model_name, summary["models"][model_name])

    human_n = int(dcfg.get("human_check_n", 30))
    if human_n and judged:
        target = ROOT / "data" / "human_check" / f"decompose_{run_paths.run_id}.jsonl"
        written = write_jsonl_atomic(
            target, _sample_for_human_check(judged, human_n, int(cfg.get("seed", 0)))
        )
        log.info("wrote %d blind items for human agreement check -> %s", written, target)
        summary["human_check_file"] = str(target)

    return summary
