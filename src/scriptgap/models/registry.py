"""The model roster: parsing ``configs/models.yaml``, and fetching weights.

Weights land in ``models/<name>/`` inside the repo rather than in a user-level
cache, so a checkout is self-contained and a run can be reproduced from it (see
``env.py``). Backends load from that directory when it exists and fall back to
the hub id otherwise.

H2 is a **variance** claim, so :func:`specialised_models` is what the metrics
stage uses to decide which columns enter that statistic. With fewer than two
Indic-specialised models the claim has nothing to measure, and the metrics stage
says so rather than reporting a spread of one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..paths import MODELS_DIR, slug

log = logging.getLogger(__name__)

#: Roles that count as Indic-specialised for the H2 statistic.
_SPECIALISED_MARKERS = ("indic-special", "indic_special", "specialised", "specialized")

#: What a real set of weights looks like on disk.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth")


@dataclass
class ModelSpec:
    """One entry from the roster."""

    name: str
    hf_id: str | None = None
    #: Provider-side model id for API backends (``backend: anthropic``).
    api_model: str | None = None
    role: str = "generalist"
    backend: str = "vllm"
    enabled: bool = True
    #: Needs an accepted licence plus `hf auth login`.
    gated: bool = False
    quantization: str | None = None
    max_model_len: int | None = None
    dtype: str | None = None
    #: Qwen3 emits <think> blocks unless the chat template is told otherwise.
    disable_thinking: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def local_dir(self) -> Path:
        return MODELS_DIR / slug(self.name)

    def is_downloaded(self) -> bool:
        """True only when actual **weights** are on disk.

        Not merely "the directory is non-empty": a download that dies partway —
        the usual case being a gated repo where the small files fetch fine and
        the 401 lands on the first shard — leaves ``config.json``, the tokenizer
        and the chat template behind with no weights at all. Counting that as
        downloaded makes the retry a silent no-op, and the failure resurfaces
        much later as an unintelligible load error on the GPU day.
        """
        if not self.local_dir.is_dir():
            return False
        return any(
            path.suffix in _WEIGHT_SUFFIXES
            for path in self.local_dir.rglob("*")
            if path.is_file()
        )

    def resolve_path(self) -> str:
        """Local weights when present, otherwise the hub id."""
        if self.is_downloaded():
            return str(self.local_dir)
        return self.hf_id or self.api_model or self.name

    @property
    def is_specialised(self) -> bool:
        role = (self.role or "").lower().replace(" ", "-")
        return any(marker in role for marker in _SPECIALISED_MARKERS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "api_model": self.api_model,
            "role": self.role,
            "backend": self.backend,
            "enabled": self.enabled,
            "gated": self.gated,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "disable_thinking": self.disable_thinking,
            "resolved_path": self.resolve_path(),
            "downloaded": self.is_downloaded(),
        }


_KNOWN_FIELDS = {
    "name", "hf_id", "api_model", "role", "backend", "enabled", "gated",
    "quantization", "max_model_len", "dtype", "disable_thinking",
}


def load_models(entries: Sequence[dict[str, Any]] | None) -> list[ModelSpec]:
    """Parse the ``models:`` block of the config."""
    if not entries:
        raise SystemExit(
            "no models configured — pass `-c configs/models.yaml` "
            "(or add a `models:` block to your config)"
        )

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise SystemExit(f"each model entry needs a `name`; got {entry!r}")
        name = str(entry["name"])
        if name in seen:
            raise SystemExit(f"duplicate model name in the roster: {name!r}")
        seen.add(name)
        known = {key: value for key, value in entry.items() if key in _KNOWN_FIELDS}
        extra = {key: value for key, value in entry.items() if key not in _KNOWN_FIELDS}
        known["name"] = name
        specs.append(ModelSpec(**known, extra=extra))
    return specs


def select_models(
    specs: Sequence[ModelSpec],
    only: Iterable[str] | None = None,
    include_disabled: bool = False,
) -> list[ModelSpec]:
    """Apply ``--model`` and the ``enabled`` flag.

    An explicit ``--model`` overrides ``enabled: false`` — asking for a model by
    name is a clearer statement of intent than the roster default.
    """
    if only:
        wanted = list(only)
        by_name = {spec.name: spec for spec in specs}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise SystemExit(
                f"unknown model(s) {missing}; the roster has {sorted(by_name)}"
            )
        return [by_name[name] for name in wanted]

    chosen = [spec for spec in specs if spec.enabled or include_disabled]
    if not chosen:
        raise SystemExit(
            "every model in the roster is disabled — enable one, or pass --model"
        )
    return chosen


def specialised_models(specs: Sequence[ModelSpec]) -> list[ModelSpec]:
    """The Indic-specialised subset — the models H2's variance is measured over."""
    return [spec for spec in specs if spec.is_specialised]


def describe(specs: Sequence[ModelSpec]) -> list[dict[str, Any]]:
    """Roster summary for the run manifest."""
    return [spec.to_dict() for spec in specs]


def download_model(spec: ModelSpec, force: bool = False) -> Path | None:
    """Fetch weights into ``models/<name>/``. Returns the directory."""
    if spec.backend == "anthropic" or not spec.hf_id:
        log.info("%s is API-served — nothing to download", spec.name)
        return None

    target = spec.local_dir
    if spec.is_downloaded() and not force:
        log.info("%s already present at %s", spec.name, target)
        return target
    if target.is_dir() and any(target.iterdir()):
        log.info(
            "%s has metadata but no weights at %s — resuming an interrupted download",
            spec.name,
            target,
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise SystemExit("huggingface_hub is not installed") from exc

    log.info("downloading %s (%s) -> %s", spec.name, spec.hf_id, target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=spec.hf_id,
            local_dir=str(target),
            ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
        )
    except Exception as exc:
        detail = str(exc)
        if spec.gated or "gated" in detail.lower() or "401" in detail or "403" in detail:
            log.error(
                "%s is gated: accept the licence at https://huggingface.co/%s "
                "and run `hf auth login`",
                spec.name,
                spec.hf_id,
            )
        log.error("could not download %s (%s)", spec.name, exc)
        return None

    log.info("%s ready at %s", spec.name, target)
    return target
