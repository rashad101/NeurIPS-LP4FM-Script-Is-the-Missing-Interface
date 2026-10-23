"""Fit the condition-3 noise operators to attested human romanisation.

The spec asks for the ad-hoc romanisation operators to be "calibrated on
Dakshina ad-hoc portions". This stage does that measurement and reports it two
ways:

1. **Suggested probabilities** — a ready-to-layer YAML fragment, so
   ``transforms.noisy_roman`` stops being hand-tuned.
2. **A comparison against the current config** — which is itself a reportable
   result: *how closely does a rule-based noise model approximate the register
   native speakers actually produce?* An operator whose configured probability
   is far from its attested rate is a finding, not just a knob to turn.

Output goes to ``data/calibration/`` (shared across runs, since it is a property
of the corpus rather than of any one run) and is copied into the run's metrics
directory for provenance.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

from .. import ROOT
from ..checkpoint import write_json_atomic
from ..config import Config
from ..data import dakshina as dakshina_mod
from ..data.schema import script_for
from ..paths import RunPaths
from ..progress import track

log = logging.getLogger(__name__)

CALIBRATION_DIR = ROOT / "data" / "calibration"


def _suite_languages(cfg: Config) -> list[str]:
    """Languages the configured suite will actually use."""
    from ..data.schema import MILU_LANG_CODE

    languages: set[str] = set()
    data = cfg.get("data", {}) or {}

    milu = data.get("milu") or {}
    if milu.get("enabled"):
        for name in milu.get("languages", []):
            code = MILU_LANG_CODE.get(name)
            if code:
                languages.add(code)

    math = data.get("math") or {}
    if math.get("enabled"):
        languages.update(math.get("languages", []))
        igb = math.get("indicgenbench") or {}
        if igb.get("enabled"):
            languages.update(igb.get("languages", []))

    ifeval = data.get("ifeval") or {}
    if ifeval.get("enabled"):
        languages.add(ifeval.get("language", "hi"))

    safety = data.get("safety") or {}
    if safety.get("enabled"):
        languages.update(safety.get("languages", []))
        languages.update(safety.get("translate_to") or [])

    return sorted(languages & set(dakshina_mod.DAKSHINA_LANGS))


def run(cfg: Config, run_paths: RunPaths | None = None) -> dict[str, Any]:
    """Measure attested phenomenon rates and compare them to the config."""
    root = dakshina_mod.ensure(cfg)
    if root is None:
        raise SystemExit(
            "Dakshina is required for calibration. Run `scriptgap download --dakshina`, "
            "or set data.dakshina.root to an extracted copy."
        )

    languages = _suite_languages(cfg)
    if not languages:
        raise SystemExit(
            "none of the configured suite languages are covered by Dakshina "
            f"(it covers {', '.join(dakshina_mod.DAKSHINA_LANGS)})"
        )
    log.info("calibrating on: %s", ", ".join(languages))

    splits = list(cfg.get("data.dakshina.splits", ["train", "dev"]))
    max_pairs = int(cfg.get("data.dakshina.calibrate_max_pairs", 20000))
    scheme = str(cfg.get("transforms.clean_roman.scheme", "ISO"))
    seed = int(cfg.get("seed", 0))

    measured: dict[str, Any] = {}
    lexicon_stats: dict[str, Any] = {}

    for language in track(languages, desc="calibrating languages", unit="lang"):
        lexicon = dakshina_mod.load_lexicon(root, language, splits)
        if lexicon is None:
            continue
        lexicon_stats[language] = {
            "native_words": len(lexicon),
            "attested_romanisations": lexicon.n_romanisations,
            "mean_variants_per_word": lexicon.mean_variants,
            "files": lexicon.files,
        }
        measured[language] = dakshina_mod.measure_phenomena(
            lexicon,
            script_for(language),
            scheme=scheme,
            max_pairs=max_pairs,
            seed=seed,
        )

    if not measured:
        raise SystemExit("no lexicons could be loaded — check data.dakshina.root")

    suggested = dakshina_mod.suggest_probabilities(measured)
    # Per-language, since the pooled value is a poor fit for any single language
    # once the spread is this wide.
    per_language = {
        language: dakshina_mod.suggest_probabilities({language: rates})
        for language, rates in measured.items()
    }
    spread = {
        operator: {
            "min": min(v[operator] for v in per_language.values() if operator in v),
            "max": max(v[operator] for v in per_language.values() if operator in v),
        }
        for operator in suggested
        if any(operator in v for v in per_language.values())
    }
    for operator, bounds in spread.items():
        bounds["ratio"] = (
            round(bounds["max"] / bounds["min"], 1) if bounds["min"] > 0 else None
        )
    configured = {
        key: float(value)
        for key, value in (cfg.get("transforms.noisy_roman", {}) or {}).items()
    }

    comparison = []
    for key, attested_rate in sorted(suggested.items()):
        current = configured.get(key)
        comparison.append(
            {
                "operator": key,
                "configured": current,
                "attested": attested_rate,
                "delta": round(attested_rate - current, 4) if current is not None else None,
            }
        )

    payload = {
        "source": "dakshina",
        "dakshina_root": str(root),
        "splits": splits,
        "scheme": scheme,
        "max_pairs_per_language": max_pairs,
        "lexicons": lexicon_stats,
        "measured_rates": measured,
        "suggested_probabilities": suggested,
        "suggested_probabilities_by_language": per_language,
        "cross_language_spread": spread,
        "configured_probabilities": configured,
        "comparison": comparison,
    }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    json_path = CALIBRATION_DIR / "dakshina_rates.json"
    write_json_atomic(json_path, payload)

    # A layerable config fragment: `-c data/calibration/noisy_roman.yaml`
    unchanged = {
        key: value for key, value in configured.items() if key not in suggested
    }
    fragment = {
        "transforms": {
            # Pooled values as the fallback for languages with no Dakshina data.
            "noisy_roman": {**unchanged, **suggested},
            # Per-language overrides — this is the part that matters.
            "noisy_roman_by_language": per_language,
        },
    }
    yaml_path = CALIBRATION_DIR / "noisy_roman.yaml"
    spread_note = "\n".join(
        f"#   {operator:<22} {bounds['min']:.3f} - {bounds['max']:.3f}"
        + (f"  ({bounds['ratio']}x across languages)" if bounds.get("ratio") else "")
        for operator, bounds in sorted(spread.items())
    )
    header = (
        "# Condition-3 operator probabilities fitted to attested human\n"
        "# romanisations (Dakshina lexicons). Generated by `scriptgap calibrate`.\n"
        f"# Languages: {', '.join(languages)}\n"
        "#\n"
        "# `noisy_roman` is the pooled fallback; `noisy_roman_by_language` is what\n"
        "# actually applies. Per-language values matter because the attested rates\n"
        "# are far from uniform:\n"
        f"{spread_note}\n"
        "#\n"
        "# Operators with no attested measurement keep their configured value.\n"
        "# Layer this over the defaults:\n"
        "#   scriptgap build-data -c data/calibration/noisy_roman.yaml --force\n\n"
    )
    yaml_path.write_text(
        header + yaml.safe_dump(fragment, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )

    if run_paths is not None:
        write_json_atomic(run_paths.metrics / "calibration.json", payload)

    log.info("calibration -> %s", json_path)
    log.info("layerable config -> %s", yaml_path)
    log.info("---- cross-language spread in the attested rates ----")
    for operator, bounds in sorted(spread.items()):
        log.info(
            "%-24s %.3f - %.3f%s",
            operator,
            bounds["min"],
            bounds["max"],
            f"  ({bounds['ratio']}x)" if bounds.get("ratio") else "",
        )
    log.info("---- pooled operator probabilities: configured vs attested ----")
    log.info("%-24s %10s %10s %8s", "operator", "config", "attested", "delta")
    for row in comparison:
        log.info(
            "%-24s %10s %10.3f %8s",
            row["operator"],
            f"{row['configured']:.3f}" if row["configured"] is not None else "-",
            row["attested"],
            f"{row['delta']:+.3f}" if row["delta"] is not None else "-",
        )

    return payload
