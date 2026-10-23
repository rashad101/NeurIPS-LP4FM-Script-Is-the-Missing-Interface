"""The six paper figures (spec §6).

Colour decisions, in the order the method requires — form first, colour last:

* **Fig 1 (gap heatmap)** encodes *polarity*: a gap can be negative when a
  condition beats native script. So it uses the **diverging** blue↔red pair with
  a neutral gray midpoint pinned at zero — not a sequential ramp, which would
  hide the sign, and not a rainbow.
* **Figs 2, 4, 6** encode *identity* across ≤ 4 series, so they use the
  categorical slots in fixed order (never cycled).
* **Fig 5** is a scatter, which puts every pair of series on screen at once, so
  it is capped at the **first three** categorical slots — the set that validates
  on the all-pairs list.

Palettes were validated with the six checks (lightness band, chroma floor, CVD
separation, normal-vision floor, contrast) in both light and dark. Three light
steps land below 3:1 against the surface, so the relief rule applies: every
figure ships **visible direct value labels**, and ``analysis/metrics.py`` writes
the same numbers to CSV as the table view. Identity is therefore never carried
by colour alone.

Paper figures deliberately commit to the light surface for print. ``--theme
dark`` renders the separately-validated dark steps (not an automatic flip).
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import ALL_CONDITIONS
from ..paths import RunPaths
from ..progress import track

log = logging.getLogger(__name__)

CONDITION_ORDER = list(ALL_CONDITIONS)
CONDITION_LABEL = {
    "native": "native",
    "clean_roman": "clean\nroman",
    "noisy_roman": "noisy\nroman",
    "native_noisy": "native\nnoisy",
    "codemix": "code-\nmixed",
    "attested_roman": "attested\nroman",
}
DECOMP_ORDER = ("misunderstood_topic", "understood_but_wrong", "refused_or_empty", "unclear")
DECOMP_LABEL = {
    "misunderstood_topic": "misunderstood topic\n(comprehension)",
    "understood_but_wrong": "understood, wrong answer\n(action)",
    "refused_or_empty": "refused / empty",
    "unclear": "unclear",
}


@dataclass(frozen=True)
class Theme:
    """One validated colour instance. Light is the print default."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    #: categorical slots, fixed order — never cycled
    series: tuple[str, ...]
    #: diverging poles + neutral midpoint
    diverge_low: str
    diverge_mid: str
    diverge_high: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e3e2de",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    diverge_low="#2a78d6",
    diverge_mid="#f0efec",
    diverge_high="#d03b3b",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    grid="#3a3a37",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    diverge_low="#3987e5",
    diverge_mid="#383835",
    diverge_high="#d03b3b",
)

THEMES = {"light": LIGHT, "dark": DARK}


# ---------------------------------------------------------------------------
# Plot scaffolding
# ---------------------------------------------------------------------------


def _mpl():
    import matplotlib

    matplotlib.use("Agg")  # headless: no display on the GPU box
    import matplotlib.pyplot as plt

    return plt


def px(n: float, dpi: int) -> float:
    """Convert pixels to points, so mark specs hold at any dpi."""
    return n * 72.0 / dpi


def _style(ax, theme: Theme, xgrid: bool = False, ygrid: bool = True) -> None:
    """Recessive grid and axes; no chartjunk."""
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme.text_secondary, labelsize=8, length=3, width=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=theme.grid, linewidth=0.7, alpha=0.9)
    if xgrid:
        ax.xaxis.grid(True, color=theme.grid, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


#: Subtitles are wrapped to this many characters. An unwrapped one-liner sets
#: the figure's tight bounding box and squashes the plot into a corner.
_SUBTITLE_WRAP = 92


def _title(ax, theme: Theme, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional wrapped subtitle above the axes.

    Offsets are in **points**, not axes fractions, so the spacing is identical
    whatever the figure's size or aspect — axes-fraction offsets drifted into
    the title as figures grew taller.
    """
    lines = textwrap.wrap(subtitle, _SUBTITLE_WRAP) if subtitle else []
    line_pts = 11.0
    base = 7.0

    for i, line in enumerate(lines):
        ax.annotate(
            line,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, base + line_pts * (len(lines) - 1 - i)),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=theme.text_secondary,
            annotation_clip=False,
        )

    ax.annotate(
        title,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, base + line_pts * len(lines) + 4.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=11.5,
        color=theme.text_primary,
        annotation_clip=False,
    )


def _diverging_cmap(theme: Theme):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "gap", [theme.diverge_low, theme.diverge_mid, theme.diverge_high], N=256
    )


def _save(fig, run: RunPaths, name: str, formats: Sequence[str], dpi: int, theme: Theme) -> list[Path]:
    written: list[Path] = []
    suffix = "" if theme.name == "light" else f"_{theme.name}"
    for fmt in formats:
        path = run.figures / f"{name}{suffix}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.surface)
        written.append(path)
    _mpl().close(fig)
    return written


def _read_csv(run: RunPaths, name: str) -> pd.DataFrame:
    path = run.metrics / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("could not read %s (%s)", path, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figure 1 — script-gap heatmap
# ---------------------------------------------------------------------------


def fig_gap_heatmap(run: RunPaths, theme: Theme, formats, dpi) -> list[Path]:
    gaps = _read_csv(run, "gaps")
    if gaps.empty:
        log.info("fig1 skipped: gaps.csv is empty")
        return []

    plt = _mpl()
    pivot = gaps.pivot_table(index=["task", "condition"], columns="model", values="gap")
    order = [
        (task, condition)
        for task in sorted({t for t, _ in pivot.index})
        for condition in CONDITION_ORDER
        if (task, condition) in pivot.index
    ]
    pivot = pivot.reindex(order)

    n_rows, n_cols = pivot.shape
    width = max(6.0, 1.5 * n_cols + 3.6)
    fig, ax = plt.subplots(
        figsize=(width, 0.44 * n_rows + 2.2), facecolor=theme.surface
    )

    limit = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))) or 0.01
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        cmap=_diverging_cmap(theme),
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{task} · {condition}" for task, condition in pivot.index], fontsize=8
    )
    ax.tick_params(colors=theme.text_secondary, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Direct value labels: the relief for sub-3:1 cells, and the numbers the
    # paper quotes.
    for row in range(n_rows):
        for col in range(n_cols):
            value = pivot.iat[row, col]
            if pd.isna(value):
                ax.text(col, row, "–", ha="center", va="center", color=theme.text_secondary, fontsize=7)
                continue
            # ink colour, not the cell colour; flip to the surface tone on dark cells
            intense = abs(value) > 0.55 * limit
            ax.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=theme.surface if intense else theme.text_primary,
            )

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, aspect=28)
    bar.set_label("accuracy gap vs native  (+ = worse than native)", color=theme.text_secondary, fontsize=8)
    bar.ax.tick_params(colors=theme.text_secondary, labelsize=7)
    bar.outline.set_visible(False)

    _title(
        ax,
        theme,
        "H1/H2 · Script gap by task and model",
        "Native-script accuracy minus condition accuracy. Spread across the Indic-specialised "
        "columns is the H2 statistic — the loop predicts variance, not uniform failure.",
    )
    fig.patch.set_facecolor(theme.surface)
    return _save(fig, run, "fig1_gap_heatmap", formats, dpi, theme)


# ---------------------------------------------------------------------------
# Figure 2 — clean vs noisy
# ---------------------------------------------------------------------------


def fig_clean_vs_noisy(run: RunPaths, theme: Theme, formats, dpi) -> list[Path]:
    accuracy = _read_csv(run, "accuracy_by_cell")
    if accuracy.empty:
        log.info("fig2 skipped: accuracy_by_cell.csv is empty")
        return []

    # attested_roman sits beside the synthetic registers when it was rendered,
    # so the figure shows directly how close the rule model gets to real usage.
    series_conditions = [
        c
        for c in ("native", "clean_roman", "noisy_roman", "attested_roman")
        if c in set(accuracy["condition"])
    ]
    if len(series_conditions) < 2:
        log.info("fig2 skipped: needs at least the clean and noisy conditions")
        return []

    frame = (
        accuracy[accuracy["condition"].isin(series_conditions)]
        .groupby(["task", "condition"])["accuracy"]
        .mean()
        .unstack("condition")
        .reindex(columns=series_conditions)
    )
    if frame.empty:
        return []

    plt = _mpl()
    tasks = list(frame.index)
    n_series = len(series_conditions)
    fig, ax = plt.subplots(figsize=(1.9 * len(tasks) + 2.6, 4.0), facecolor=theme.surface)

    # The 2px surface gap between adjacent bars is drawn as a surface-coloured
    # edge stroke (below), which is resolution-independent. `slot_gap` is a small
    # extra breathing space in data units so the group reads as a group.
    group_width = 0.76
    slot_gap = 0.012
    width = (group_width - slot_gap * (n_series - 1)) / n_series
    positions = np.arange(len(tasks), dtype=float)

    for i, condition in enumerate(series_conditions):
        offset = -group_width / 2 + i * (width + slot_gap) + width / 2
        values = frame[condition].to_numpy(dtype=float)
        bars = ax.bar(
            positions + offset,
            values,
            width=width,
            label=condition.replace("_", " "),
            color=theme.series[i],
            edgecolor=theme.surface,
            linewidth=px(2, dpi),
        )
        for rect, value in zip(bars, values):
            if pd.isna(value):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=theme.text_secondary,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("accuracy (mean over models and languages)", color=theme.text_secondary, fontsize=9)
    ax.set_ylim(0, min(1.0, float(np.nanmax(frame.to_numpy(dtype=float))) * 1.15 + 0.04))
    _style(ax, theme)

    # Below the axes, so it can never sit on top of a tall bar.
    legend = ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=n_series,
    )
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)

    _title(
        ax,
        theme,
        "H3 · Curated transliteration understates the deployed gap",
        "Clean scheme-based romanisation vs the ad-hoc register users actually type"
        + (
            ", with attested native-speaker romanisation for comparison."
            if "attested_roman" in series_conditions
            else "."
        ),
    )
    return _save(fig, run, "fig2_clean_vs_noisy", formats, dpi, theme)


# ---------------------------------------------------------------------------
# Figure 3 — safety refusal table
# ---------------------------------------------------------------------------


def fig_safety_table(run: RunPaths, theme: Theme, formats, dpi) -> list[Path]:
    refusal = _read_csv(run, "h4_refusal")
    if refusal.empty:
        log.info("fig3 skipped: h4_refusal.csv is empty (safety task not run?)")
        return []

    pivot = (
        refusal.pivot_table(index="model", columns="condition", values="refusal_rate")
        .reindex(columns=[c for c in CONDITION_ORDER if c in set(refusal["condition"])])
    )

    plt = _mpl()
    fig, ax = plt.subplots(
        figsize=(1.5 * len(pivot.columns) + 3.2, 0.42 * len(pivot.index) + 2.2),
        facecolor=theme.surface,
    )
    ax.axis("off")

    cell_text = [
        [("–" if pd.isna(v) else f"{v:.2f}") for v in row] for row in pivot.to_numpy(dtype=float)
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=list(pivot.index),
        colLabels=[c.replace("_", " ") for c in pivot.columns],
        cellLoc="center",
        rowLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(theme.grid)
        cell.set_linewidth(0.6)
        cell.set_facecolor(theme.surface)
        text = cell.get_text()
        text.set_color(theme.text_primary if row > 0 else theme.text_secondary)
        if row == 0:
            text.set_fontweight("medium")

    _title(
        ax,
        theme,
        "H4 · Refusal rate on identical harmful-intent prompts, by condition",
        "Aggregate rates only; no completions are reproduced. Safety inherits the same closed "
        "loop as capability, so a native-script number overstates real protection.",
    )
    return _save(fig, run, "fig3_safety_refusal", formats, dpi, theme)


# ---------------------------------------------------------------------------
# Figure 4 — failure decomposition
# ---------------------------------------------------------------------------


def fig_decomposition(run: RunPaths, theme: Theme, formats, dpi) -> list[Path]:
    decomp = _read_csv(run, "h5_decomposition")
    if decomp.empty:
        log.info("fig4 skipped: h5_decomposition.csv is empty (run `scriptgap decompose`)")
        return []

    labels = [label for label in DECOMP_ORDER if label in decomp.columns]
    if not labels:
        return []

    counts = decomp.set_index("model")[labels].astype(float)
    totals = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.div(totals, axis=0).fillna(0.0)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(shares) + 2.5), facecolor=theme.surface)

    positions = np.arange(len(shares), dtype=float)
    left = np.zeros(len(shares))
    for i, label in enumerate(labels):
        values = shares[label].to_numpy()
        ax.barh(
            positions,
            values,
            left=left,
            height=0.6,
            label=DECOMP_LABEL.get(label, label),
            color=theme.series[i],
            # 2px surface gap between stacked segments
            edgecolor=theme.surface,
            linewidth=px(2, dpi),
        )
        for y, (value, offset) in enumerate(zip(values, left)):
            if value < 0.06:  # too narrow to label legibly
                continue
            ax.text(
                offset + value / 2,
                y,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=7.4,
                color=theme.surface if i in (0, 5, 6) else theme.text_primary,
            )
        left = left + values

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{model}  (n={int(totals.iloc[i]) if not pd.isna(totals.iloc[i]) else 0})"
         for i, model in enumerate(shares.index)],
        fontsize=8.5,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of judged romanised failures", color=theme.text_secondary, fontsize=9)
    ax.invert_yaxis()
    _style(ax, theme, xgrid=True, ygrid=False)

    legend = ax.legend(
        frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2
    )
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)

    _title(
        ax,
        theme,
        "H5 · Where romanised failures happen",
        "An action-dominant profile means a normalisation front-end attacks the wrong stage.",
    )
    return _save(fig, run, "fig4_decomposition", formats, dpi, theme)


# ---------------------------------------------------------------------------
# Figure 5 — fertility scatter
# ---------------------------------------------------------------------------


def fig_fertility(run: RunPaths, theme: Theme, formats, dpi) -> list[Path]:
    fertility = _read_csv(run, "h6_fertility")
    if fertility.empty or "clean_roman_tokens" not in fertility.columns:
        log.info("fig5 skipped: h6_fertility.csv is empty (run `scriptgap fertility`)")
        return []

    frame = fertility.dropna(subset=["native_tokens", "clean_roman_tokens"])
    if frame.empty:
        return []

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=theme.surface)

    # A scatter shows every pair of series at once, so the palette is capped at
    # the three slots that validate on the all-pairs list.
    kinds = [k for k in ("model", "external", "tiktoken") if k in set(frame["tokeniser_kind"])]
    for i, kind in enumerate(kinds[:3]):
        subset = frame[frame["tokeniser_kind"] == kind]
        ax.scatter(
            subset["native_tokens"],
            subset["clean_roman_tokens"],
            s=14,  # >= 8px marker
            color=theme.series[i],
            edgecolor=theme.surface,
            linewidth=px(2, dpi),
            alpha=0.75,
            label=f"{kind} tokenisers",
        )

    upper = float(
        max(frame["native_tokens"].max(), frame["clean_roman_tokens"].max())
    ) * 1.05
    ax.plot([0, upper], [0, upper], color=theme.text_secondary, linewidth=1.0, linestyle="--", alpha=0.6)
    # Horizontal, offset off the line: rotating to 45 degrees only matches the
    # line's *visual* slope when the axes are exactly square, which the figure
    # size does not guarantee.
    ax.annotate(
        "equal cost",
        xy=(upper * 0.72, upper * 0.72),
        xytext=(8, -4),
        textcoords="offset points",
        color=theme.text_secondary,
        fontsize=8,
        ha="left",
        va="top",
    )

    ratio = float((frame["clean_roman_tokens"] / frame["native_tokens"].replace(0, np.nan)).mean())
    ax.text(
        0.03,
        0.96,
        f"mean romanised/native token ratio = {ratio:.2f}\n"
        + ("romanised is cheaper to serve" if ratio < 1 else "romanised costs more"),
        transform=ax.transAxes,
        color=theme.text_primary,
        fontsize=9,
        va="top",
        ha="left",
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("native-script tokens per item", color=theme.text_secondary, fontsize=9)
    ax.set_ylabel("clean-romanised tokens per item", color=theme.text_secondary, fontsize=9)
    _style(ax, theme, xgrid=True, ygrid=True)

    if len(kinds) > 1:
        legend = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for text in legend.get_texts():
            text.set_color(theme.text_secondary)

    _title(
        ax,
        theme,
        "H6 · Within a language, the register users prefer is cheaper to serve",
        "Points below the dashed line cost fewer tokens than the native-script original.",
    )
    return _save(fig, run, "fig5_fertility_scatter", formats, dpi, theme)


# ---------------------------------------------------------------------------
# Figure 6 — Arabizi sidebar
# ---------------------------------------------------------------------------


def fig_arabizi(run: RunPaths, theme: Theme, formats, dpi) -> list[Path]:
    accuracy = _read_csv(run, "accuracy_by_cell")
    if accuracy.empty:
        return []
    subset = accuracy[accuracy["task"] == "arabizi"]
    if subset.empty:
        log.info("fig6 skipped: no Arabizi rows (enable data.arabizi and re-run)")
        return []

    frame = (
        subset.groupby("condition")["accuracy"]
        .mean()
        .reindex([c for c in CONDITION_ORDER if c in set(subset["condition"])])
    )

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(1.15 * len(frame) + 2.4, 4.0), facecolor=theme.surface)

    # A single series: the title names it, so no legend box.
    values = frame.to_numpy(dtype=float)
    bars = ax.bar(
        range(len(frame)),
        values,
        width=0.62,
        color=theme.series[0],
        edgecolor=theme.surface,
        linewidth=px(2, dpi),
    )
    for rect, value in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            value + 0.012,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=theme.text_secondary,
        )

    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels([CONDITION_LABEL.get(c, c) for c in frame.index], fontsize=8.5)
    ax.set_ylabel("accuracy (mean over models)", color=theme.text_secondary, fontsize=9)
    ax.set_ylim(0, min(1.0, float(np.nanmax(values)) * 1.22 + 0.02))
    _style(ax, theme)

    _title(
        ax,
        theme,
        "Universality sidebar · Arabic script vs Arabizi",
        "The same five-condition pipeline on a 100-item Arabic slice: the claim is about "
        "writing systems, not one region.",
    )
    return _save(fig, run, "fig6_arabizi_sidebar", formats, dpi, theme)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FIGURES = (
    ("fig1_gap_heatmap", fig_gap_heatmap),
    ("fig2_clean_vs_noisy", fig_clean_vs_noisy),
    ("fig3_safety_refusal", fig_safety_table),
    ("fig4_decomposition", fig_decomposition),
    ("fig5_fertility_scatter", fig_fertility),
    ("fig6_arabizi_sidebar", fig_arabizi),
)


def render_all(
    run: RunPaths,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 200,
    theme_name: str = "light",
) -> dict[str, Any]:
    theme = THEMES.get(theme_name)
    if theme is None:
        raise SystemExit(f"unknown theme {theme_name!r}; choose light or dark")

    written: dict[str, list[str]] = {}
    for name, fn in track(FIGURES, desc="rendering figures", unit="fig"):
        try:
            paths = fn(run, theme, formats, dpi)
        except Exception as exc:
            log.exception("figure %s failed: %s", name, exc)
            continue
        if paths:
            written[name] = [str(p) for p in paths]
            log.info("%s -> %s", name, ", ".join(p.name for p in paths))

    if not written:
        log.warning("no figures were produced — run `scriptgap metrics` first")
    return {"figures": written, "theme": theme.name}
