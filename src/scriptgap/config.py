"""YAML configuration loading, merging and dotted-key overrides."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import ROOT

CONFIG_DIR = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

#: The five orthographic renderings of every item (paper §2 taxonomy + native).
#: This is the released suite and the paper's headline framing.
CONDITIONS: tuple[str, ...] = (
    "native",  # 1. gold native script
    "clean_roman",  # 2. scheme-based transliteration (Aksharamukha / ISO-15919)
    "noisy_roman",  # 3. ad-hoc romanisation with stochastic spelling noise
    "native_noisy",  # 4. script-internal noise (matra/typo perturbation)
    "codemix",  # 5. code-mixed (matrix Indic + English insertions)
)

#: Opt-in conditions, rendered only when their data source is configured. Kept
#: out of ``CONDITIONS`` so the paper's "five orthographic renderings" claim
#: still describes the released suite exactly.
#:
#: ``attested_roman`` substitutes real native-speaker romanisations from
#: Dakshina. It is the validation condition for condition 3: it answers
#: "did you measure your own noise model, or the register users produce?"
OPTIONAL_CONDITIONS: tuple[str, ...] = ("attested_roman",)

#: Every condition the codebase knows about, in display order.
ALL_CONDITIONS: tuple[str, ...] = CONDITIONS + OPTIONAL_CONDITIONS

#: Conditions that are sampled more than once (stochastic transforms).
MULTI_SAMPLE_CONDITIONS: tuple[str, ...] = ("noisy_roman", "native_noisy", "attested_roman")


def active_conditions(cfg: "Config") -> tuple[str, ...]:
    """The five core conditions, plus any opt-in condition that is configured."""
    active = list(CONDITIONS)
    if cfg.get("data.dakshina.enabled", False):
        active.append("attested_roman")
    return tuple(active)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value as YAML (so ints/bools/lists work)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(
    cfg: dict[str, Any],
    overrides: list[str] | None,
    record: set[str] | None = None,
) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (returns a new dict).

    ``record`` collects the dotted keys that were set, so downstream code can
    tell an explicit command-line override from a config-file default.
    """
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        node: Any = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _coerce(raw.strip())
        if record is not None:
            record.add(dotted.strip())
    return out


@dataclass
class Config:
    """Loaded configuration plus its provenance, for the run manifest."""

    data: dict[str, Any]
    sources: list[str] = field(default_factory=list)
    #: dotted keys set explicitly with ``--set``. An explicit override is meant
    #: to win over a config-file default, including a per-language default.
    overridden: set[str] = field(default_factory=set)

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise SystemExit(f"missing required config key: {dotted}")
        return value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def dumps(self) -> str:
        return json.dumps(self.data, indent=2, ensure_ascii=False, default=str)


def load_config(
    paths: list[str | Path] | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load ``default.yaml``, then each extra file, then dotted overrides."""
    files: list[Path] = [DEFAULT_CONFIG]
    for p in paths or []:
        path = Path(p)
        if not path.exists():
            # allow bare names like `models` to mean `configs/models.yaml`
            alt = CONFIG_DIR / f"{path.stem}.yaml"
            if alt.exists():
                path = alt
            else:
                raise SystemExit(f"config file not found: {p}")
        files.append(path)

    merged: dict[str, Any] = {}
    used: list[str] = []
    for path in files:
        if not path.exists():
            raise SystemExit(f"config file not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise SystemExit(f"{path} must contain a YAML mapping at the top level")
        merged = deep_merge(merged, loaded)
        used.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))

    recorded: set[str] = set()
    merged = apply_overrides(merged, overrides, record=recorded)
    return Config(data=merged, sources=used, overridden=recorded)
