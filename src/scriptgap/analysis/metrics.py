"""Aggregate predictions into the paper's statistics (H1-H6).

Produces, under ``runs/<run_id>/metrics/``:

``script_gap_results.parquet``
    The released long-format table: one row per (model, item, condition, sample),
    carrying the prompt, the raw completion, the prompt fingerprint and the
    finish reason. It is deliberately **self-contained** — a future re-analysis
    should not need the run's JSONL shards or a suite file that may since have
    been rebuilt. Safety rows carry labels only (see the ethics note).

``accuracy_by_cell.csv``
    Accuracy per (model, task, language, condition) with bootstrap CIs.

``gaps.csv``
    Native-minus-condition gaps — H1's headline numbers.

``h2_variance.csv``
    Cross-model variance of the gap *among Indic-specialised models*. This is
    the H2 statistic: the loop diagnosis predicts high variance, not uniform
    failure, so a specialised model that does well on one register while
    gapping on another supports the hypothesis rather than refuting it.

``h3_clean_vs_noisy.csv``
    Per-task clean-minus-noisy delta — "curated transliteration understates the
    deployed gap".

``h4_refusal.csv``
    Aggregate refusal rates by condition. No completions are included.

``h5_decomposition.csv`` / ``h6_fertility.csv``
    The two new measurements, if those stages have been run.

``summary.json``
    Everything above, condensed, for quick inspection and for the README table.

Gating: rows whose item failed the back-transliteration round-trip gate are
excluded from the reported gaps by default (spec §4) but kept in the parquet,
so the exclusion is auditable and reversible.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..checkpoint import read_jsonl, write_json_atomic
from ..config import ALL_CONDITIONS, Config
from ..paths import RunPaths
from ..progress import track

log = logging.getLogger(__name__)

BASELINE_CONDITION = "native"

#: Conditions whose renderer emits the *native prompt verbatim* when its
#: transform does not fire (no code-mix insertion matched, the noise draw was a
#: no-op, no attested lexicon for the language). Such a cell is an unlabelled
#: duplicate of the baseline and must not be scored as its nominal condition.
NATIVE_DUPLICATE_CONDITIONS = ("codemix", "native_noisy", "attested_roman", "clean_roman")

#: Escalate from warning to error once truncation rates differ across conditions
#: by this much. Uneven truncation lands directly in the gap and is
#: indistinguishable from a register effect.
TRUNCATION_SPREAD_ALERT = 0.10

#: ...but only once each condition has enough observations for the spread to
#: mean anything. With 4 rows per condition a single truncated generation is a
#: 25-point spread, so a smoke run would trip the alarm every time and train the
#: reader to ignore it.
TRUNCATION_MIN_N = 30


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


#: Analysis columns, always present.
_CORE_FIELDS = (
    "uid",
    "item_id",
    "model",
    "task",
    "language",
    "script",
    "condition",
    "sample_idx",
    "answer_type",
    "gold",
    "parsed",
    "correct",
    "parse_failed",
    "empty_response",
    "roundtrip_pass",
    "transform_applied",
    "prompt_sha256",
    "finish_reason",
    "truncated",
    "output_tokens",
    "latency_s",
    "response_chars",
)

#: Text columns. Dropped only if `metrics.include_text` is false — they are what
#: make the released table re-analysable (error analysis, re-scoring under a
#: different parser, qualitative inspection).
_TEXT_FIELDS = ("prompt", "response")


def load_predictions(
    run: RunPaths, stage: str = "inference", include_text: bool = True
) -> pd.DataFrame:
    root = run.predictions / stage
    if not root.exists():
        raise SystemExit(f"{root} not found — run `scriptgap run-inference` first")

    shard_files = sorted(root.glob("*/*.jsonl"))
    if not shard_files:
        raise SystemExit(f"no prediction shards under {root}")

    fields = _CORE_FIELDS + (_TEXT_FIELDS if include_text else ())
    rows: list[dict[str, Any]] = []
    for shard in track(shard_files, desc="loading predictions", unit="shard"):
        for record in read_jsonl(shard):
            detail = record.get("detail") or {}
            row = {name: record.get(name) for name in fields}
            # flatten the scoring detail that analysis actually groups on
            row["refused"] = detail.get("refused")
            row["n_constraints"] = detail.get("n_constraints")
            row["n_satisfied"] = detail.get("n_satisfied")
            row["ifeval_per_constraint"] = (
                json.dumps(detail["per_constraint"], ensure_ascii=False)
                if detail.get("per_constraint")
                else None
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            f"every shard under {root} was empty — nothing to aggregate"
        )
    for flag in ("parse_failed", "empty_response", "roundtrip_pass", "transform_applied"):
        df[flag] = df[flag].fillna(flag in {"roundtrip_pass", "transform_applied"}).astype(bool)
    df["truncated"] = df["truncated"].fillna(False).astype(bool)

    log.info(
        "loaded %d prediction rows over %d models (%d columns, text=%s)",
        len(df),
        df["model"].nunique(),
        len(df.columns),
        include_text,
    )
    missing_text = include_text and df["prompt"].isna().all()
    if missing_text:
        log.warning(
            "no prompt text in any prediction — these shards predate the "
            "self-contained record format; re-run inference to capture it"
        )
    return df


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Iterable[float], n_samples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for a mean. Returns (lo, hi)."""
    array = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if array.size < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, array.size, size=(n_samples, array.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 4), round(float(hi), 4))


def _accuracy_frame(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Accuracy per (model, task, language, condition), with CIs."""
    scored = df[df["correct"].notna()]
    records: list[dict[str, Any]] = []
    groups = list(scored.groupby(["model", "task", "language", "condition"], dropna=False))

    for (model, task, language, condition), group in track(
        groups, desc="accuracy cells", unit="cell"
    ):
        values = group["correct"].astype(float)
        lo, hi = bootstrap_ci(values, n_samples=n_boot, seed=seed)
        records.append(
            {
                "model": model,
                "task": task,
                "language": language,
                "condition": condition,
                "n": int(len(group)),
                "accuracy": round(float(values.mean()), 4),
                "ci_lo": lo,
                "ci_hi": hi,
                "parse_failure_rate": round(float(group["parse_failed"].mean()), 4),
                "empty_rate": round(float(group["empty_response"].mean()), 4),
            }
        )
    return pd.DataFrame(records)


def _gap_frame(accuracy: pd.DataFrame) -> pd.DataFrame:
    """Native-minus-condition gap per (model, task, language, condition)."""
    if accuracy.empty:
        return pd.DataFrame()
    baseline = (
        accuracy[accuracy["condition"] == BASELINE_CONDITION]
        .set_index(["model", "task", "language"])["accuracy"]
        .rename("native_accuracy")
    )
    merged = accuracy.merge(baseline, left_on=["model", "task", "language"], right_index=True, how="left")
    merged["gap"] = (merged["native_accuracy"] - merged["accuracy"]).round(4)
    merged["relative_gap"] = np.where(
        merged["native_accuracy"] > 0,
        (merged["gap"] / merged["native_accuracy"]).round(4),
        np.nan,
    )
    return merged[merged["condition"] != BASELINE_CONDITION].reset_index(drop=True)


def _h2_variance(gaps: pd.DataFrame, specialised: list[str]) -> pd.DataFrame:
    """Cross-model variance of the gap among Indic-specialised models."""
    if gaps.empty:
        return pd.DataFrame()
    subset = gaps[gaps["model"].isin(specialised)]
    if subset.empty:
        log.warning(
            "no Indic-specialised models found in the predictions; H2's variance "
            "statistic needs at least two (check the `role` field in configs/models.yaml)"
        )
        return pd.DataFrame()
    if subset["model"].nunique() < 2:
        log.warning(
            "only %d Indic-specialised model in the predictions; H2 is a variance "
            "claim and needs >= 2 to be testable",
            subset["model"].nunique(),
        )

    grouped = subset.groupby(["task", "language", "condition"], dropna=False)["gap"]
    out = grouped.agg(
        n_models="count",
        gap_mean="mean",
        gap_std="std",
        gap_min="min",
        gap_max="max",
    ).reset_index()
    out["gap_range"] = (out["gap_max"] - out["gap_min"]).round(4)
    for column in ("gap_mean", "gap_std", "gap_min", "gap_max"):
        out[column] = out[column].round(4)
    return out


def _h3_clean_vs_noisy(accuracy: pd.DataFrame) -> pd.DataFrame:
    """Clean-minus-noisy accuracy delta, per (model, task, language)."""
    wide = accuracy.pivot_table(
        index=["model", "task", "language"], columns="condition", values="accuracy"
    ).reset_index()
    if "clean_roman" not in wide or "noisy_roman" not in wide:
        return pd.DataFrame()
    wide["clean_minus_noisy"] = (wide["clean_roman"] - wide["noisy_roman"]).round(4)
    keep = ["model", "task", "language", "clean_roman", "noisy_roman", "clean_minus_noisy"]
    if BASELINE_CONDITION in wide:
        wide["native_minus_clean"] = (wide[BASELINE_CONDITION] - wide["clean_roman"]).round(4)
        wide["native_minus_noisy"] = (wide[BASELINE_CONDITION] - wide["noisy_roman"]).round(4)
        keep = [BASELINE_CONDITION] + keep[3:] + ["native_minus_clean", "native_minus_noisy"]
        keep = ["model", "task", "language", BASELINE_CONDITION, "clean_roman", "noisy_roman",
                "clean_minus_noisy", "native_minus_clean", "native_minus_noisy"]
    return wide[keep].reset_index(drop=True)


def _h4_refusal(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Aggregate refusal rates on the safety set — no completions included."""
    safety = df[(df["task"] == "safety") & df["refused"].notna()]
    if safety.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (model, language, condition), group in safety.groupby(
        ["model", "language", "condition"], dropna=False
    ):
        values = group["refused"].astype(float)
        lo, hi = bootstrap_ci(values, n_samples=n_boot, seed=seed)
        records.append(
            {
                "model": model,
                "language": language,
                "condition": condition,
                "n": int(len(group)),
                "refusal_rate": round(float(values.mean()), 4),
                "ci_lo": lo,
                "ci_hi": hi,
                "empty_rate": round(float(group["empty_response"].mean()), 4),
            }
        )
    frame = pd.DataFrame(records)
    baseline = (
        frame[frame["condition"] == BASELINE_CONDITION]
        .set_index(["model", "language"])["refusal_rate"]
        .rename("native_refusal_rate")
    )
    frame = frame.merge(baseline, left_on=["model", "language"], right_index=True, how="left")
    frame["refusal_drop"] = (frame["native_refusal_rate"] - frame["refusal_rate"]).round(4)
    return frame


def _h5_decomposition(run: RunPaths) -> pd.DataFrame:
    root = run.predictions / "decompose"
    if not root.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for shard in sorted(root.glob("*/*.jsonl")):
        for record in read_jsonl(shard):
            rows.append(
                {
                    "model": record.get("model"),
                    "task": record.get("task"),
                    "language": record.get("language"),
                    "condition": record.get("condition"),
                    "label": record.get("label"),
                    "confidence": record.get("confidence"),
                }
            )
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    counts = (
        raw.groupby(["model", "label"]).size().unstack(fill_value=0).reset_index()
    )
    for label in ("misunderstood_topic", "understood_but_wrong", "refused_or_empty", "unclear"):
        if label not in counts:
            counts[label] = 0
    counts["n"] = counts[
        ["misunderstood_topic", "understood_but_wrong", "refused_or_empty", "unclear"]
    ].sum(axis=1)
    substantive = counts["misunderstood_topic"] + counts["understood_but_wrong"]
    counts["substantive_n"] = substantive
    counts["action_share"] = np.where(
        substantive > 0, (counts["understood_but_wrong"] / substantive).round(4), np.nan
    )
    counts["comprehension_share"] = np.where(
        substantive > 0, (counts["misunderstood_topic"] / substantive).round(4), np.nan
    )
    return counts


def _h6_fertility(run: RunPaths) -> pd.DataFrame:
    root = run.predictions / "fertility"
    if not root.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for shard in sorted(root.glob("*/*.jsonl")):
        rows.extend(read_jsonl(shard))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def compute(
    cfg: Config,
    run: RunPaths,
    specialised_models: list[str] | None = None,
    gate: bool = True,
) -> dict[str, Any]:
    n_boot = int(cfg.get("metrics.bootstrap_samples", 2000))
    seed = int(cfg.get("seed", 0))

    # A RunPaths built directly (rather than via .create()) has no directories
    # yet; make the outputs land regardless of how the caller constructed it.
    run.metrics.mkdir(parents=True, exist_ok=True)
    include_text = bool(cfg.get("metrics.include_text", True))
    df = load_predictions(run, include_text=include_text)

    # Released long-format table keeps *everything*, gate flag included.
    parquet_name = str(cfg.get("metrics.out_name", "script_gap_results.parquet"))
    parquet_path = run.metrics / parquet_name
    df.to_parquet(parquet_path, index=False)
    log.info("wrote %s (%d rows)", parquet_path, len(df))

    analysed = df
    excluded = 0
    if gate:
        excluded = int((~df["roundtrip_pass"]).sum())
        analysed = df[df["roundtrip_pass"]]
        if excluded:
            log.info(
                "round-trip gate excluded %d of %d rows (%.1f%%) from the reported gaps",
                excluded,
                len(df),
                100 * excluded / max(1, len(df)),
            )

    # A codemix cell where no insertion fired is an unlabelled duplicate of the
    # native item, so it must not be counted as a code-mixed observation.
    # A cell whose transform did not fire can be a byte-identical copy of the
    # native prompt. Scoring it as its nominal condition contributes a
    # guaranteed zero gap and drags the mean toward zero — it *understates* the
    # effect being measured. Calibration made this more likely, not less: the
    # fitted probabilities are lower than the hand-set ones, so no-op draws are
    # more common.
    duplicate_counts: dict[str, int] = {}
    for condition in NATIVE_DUPLICATE_CONDITIONS:
        mask = (analysed["condition"] == condition) & (~analysed["transform_applied"])
        count = int(mask.sum())
        if count:
            duplicate_counts[condition] = count
            analysed = analysed[~mask]
    unapplied = sum(duplicate_counts.values())
    if duplicate_counts:
        log.info(
            "dropped %d row(s) that were byte-identical to the native prompt: %s",
            unapplied,
            duplicate_counts,
        )

    # `noisy_roman` is different: when no operator fires the cell equals the
    # *clean* romanisation, which is still a legitimate romanised observation.
    # It is kept, but it biases the H3 clean-vs-noisy delta toward zero, so the
    # rate is reported rather than silently absorbed.
    noop_noise = int(
        ((analysed["condition"] == "noisy_roman") & (~analysed["transform_applied"])).sum()
    )
    n_noisy = int((analysed["condition"] == "noisy_roman").sum())
    if noop_noise:
        log.warning(
            "%d/%d noisy_roman cells (%.1f%%) had no operator fire and equal the clean "
            "romanisation — this biases the H3 delta toward zero",
            noop_noise,
            n_noisy,
            100 * noop_noise / max(1, n_noisy),
        )

    accuracy = _accuracy_frame(analysed, n_boot=n_boot, seed=seed)
    gaps = _gap_frame(accuracy)
    h2 = _h2_variance(gaps, specialised_models or [])
    h3 = _h3_clean_vs_noisy(accuracy)
    h4 = _h4_refusal(analysed, n_boot=n_boot, seed=seed)
    h5 = _h5_decomposition(run)
    h6 = _h6_fertility(run)

    outputs: dict[str, pd.DataFrame] = {
        "accuracy_by_cell": accuracy,
        "gaps": gaps,
        "h2_variance": h2,
        "h3_clean_vs_noisy": h3,
        "h4_refusal": h4,
        "h5_decomposition": h5,
        "h6_fertility": h6,
    }
    for name, frame in outputs.items():
        if frame is None or frame.empty:
            log.info("%s: empty (stage not run or no matching rows)", name)
            continue
        path = run.metrics / f"{name}.csv"
        frame.to_csv(path, index=False)
        log.info("wrote %s (%d rows)", path, len(frame))

    summary = _summarise(outputs, excluded_rows=excluded, total_rows=len(df))
    summary["rows_dropped_as_native_duplicates"] = duplicate_counts
    summary["noisy_roman_noop_rate"] = (
        round(noop_noise / n_noisy, 4) if n_noisy else None
    )

    # Truncation masquerades as error once a generation is reduced to a score,
    # so it is surfaced rather than buried.
    if "truncated" in df.columns and df["truncated"].any():
        by_condition = df.groupby("condition")["truncated"].mean().round(4)
        summary["truncation_rate_by_condition"] = by_condition.to_dict()
        summary["truncation_rate_overall"] = round(float(df["truncated"].mean()), 4)
        spread = float(by_condition.max() - by_condition.min())
        summary["truncation_spread_across_conditions"] = round(spread, 4)
        worst = by_condition.idxmax()
        log.warning(
            "%.1f%% of generations were truncated at max_new_tokens (worst: %s at %.1f%%) "
            "— raise inference.max_new_tokens or these will score as errors",
            100 * df["truncated"].mean(),
            worst,
            100 * by_condition.max(),
        )
        # Uneven truncation does not cancel out: if one condition is cut off more
        # often than another, the difference lands in the gap and is
        # indistinguishable from a register effect. This is the single most
        # likely way to manufacture a script gap by accident.
        smallest_cell = int(df.groupby("condition").size().min())
        summary["truncation_min_rows_per_condition"] = smallest_cell

        if spread >= TRUNCATION_SPREAD_ALERT and smallest_cell < TRUNCATION_MIN_N:
            log.info(
                "truncation spread is %.0f%% but the smallest condition has only %d "
                "rows — too few to distinguish an uneven cap from sampling noise, "
                "so this is not being flagged as a confound",
                100 * spread,
                smallest_cell,
            )
        elif spread >= TRUNCATION_SPREAD_ALERT:
            summary["truncation_confound_warning"] = True
            log.error(
                "truncation is UNEVEN across conditions (%.0f%% spread: %s at %.0f%% vs "
                "%s at %.0f%%). This biases the gap directly — a condition that is cut "
                "off more often scores worse for a reason unrelated to orthography. "
                "Raise inference.max_new_tokens.<task> until the spread is small before "
                "reporting any gap.",
                100 * spread,
                worst,
                100 * by_condition.max(),
                by_condition.idxmin(),
                100 * by_condition.min(),
            )
    write_json_atomic(run.metrics / "summary.json", summary)
    _log_headlines(summary)
    return summary


def _summarise(
    outputs: dict[str, pd.DataFrame], excluded_rows: int, total_rows: int
) -> dict[str, Any]:
    accuracy = outputs["accuracy_by_cell"]
    gaps = outputs["gaps"]
    h2 = outputs["h2_variance"]
    h3 = outputs["h3_clean_vs_noisy"]
    h4 = outputs["h4_refusal"]
    h5 = outputs["h5_decomposition"]
    h6 = outputs["h6_fertility"]

    summary: dict[str, Any] = {
        "rows_total": total_rows,
        "rows_excluded_by_gate": excluded_rows,
        "conditions": list(ALL_CONDITIONS),
    }

    if not accuracy.empty:
        summary["accuracy_by_model_condition"] = (
            accuracy.pivot_table(index="model", columns="condition", values="accuracy")
            .round(4)
            .to_dict()
        )
    if not gaps.empty:
        summary["h1_mean_gap_by_condition"] = (
            gaps.groupby("condition")["gap"].mean().round(4).to_dict()
        )
        summary["h1_mean_gap_by_model_condition"] = (
            gaps.pivot_table(index="model", columns="condition", values="gap").round(4).to_dict()
        )
    if not h2.empty:
        summary["h2_specialised_gap_spread"] = {
            "mean_std_across_models": round(float(h2["gap_std"].mean(skipna=True)), 4),
            "mean_range_across_models": round(float(h2["gap_range"].mean(skipna=True)), 4),
            "max_range": round(float(h2["gap_range"].max()), 4),
            "n_cells": int(len(h2)),
        }
    if not h3.empty:
        summary["h3_clean_minus_noisy_by_task"] = (
            h3.groupby("task")["clean_minus_noisy"].mean().round(4).to_dict()
        )
    # Validation of the synthetic noise model: if the attested condition was
    # rendered, how far is our rule-generated register from the real one?
    if not gaps.empty and {"noisy_roman", "attested_roman"} <= set(gaps["condition"]):
        wide = gaps.pivot_table(
            index=["model", "task", "language"], columns="condition", values="gap"
        )
        if {"noisy_roman", "attested_roman"} <= set(wide.columns):
            delta = (wide["noisy_roman"] - wide["attested_roman"]).dropna()
            summary["synthetic_vs_attested"] = {
                "mean_gap_synthetic": round(float(wide["noisy_roman"].mean()), 4),
                "mean_gap_attested": round(float(wide["attested_roman"].mean()), 4),
                "mean_abs_difference": round(float(delta.abs().mean()), 4),
                "correlation": (
                    round(float(wide["noisy_roman"].corr(wide["attested_roman"])), 4)
                    if len(wide) > 2
                    else None
                ),
                "n_cells": int(len(delta)),
            }
    if not h4.empty:
        summary["h4_refusal_by_condition"] = (
            h4.groupby("condition")["refusal_rate"].mean().round(4).to_dict()
        )
    if not h5.empty:
        summary["h5_action_share_by_model"] = dict(
            zip(h5["model"], h5["action_share"].round(4))
        )
    if not h6.empty and "clean_roman_token_ratio" in h6:
        summary["h6_clean_roman_token_ratio_by_tokeniser"] = (
            h6.groupby("tokeniser")["clean_roman_token_ratio"].mean().round(4).to_dict()
        )
    return summary


def _log_headlines(summary: dict[str, Any]) -> None:
    log.info("---- headline numbers ----")
    for key in (
        "h1_mean_gap_by_condition",
        "h2_specialised_gap_spread",
        "h3_clean_minus_noisy_by_task",
        "h4_refusal_by_condition",
        "h5_action_share_by_model",
        "h6_clean_roman_token_ratio_by_tokeniser",
        "synthetic_vs_attested",
    ):
        if key in summary:
            log.info("%-42s %s", key, summary[key])
        else:
            log.info("%-42s (not available — stage not run)", key)
