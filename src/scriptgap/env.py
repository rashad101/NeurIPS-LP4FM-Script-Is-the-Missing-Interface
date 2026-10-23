"""Repo-local cache isolation.

Every third-party library we touch (HuggingFace, torch, vLLM, triton, tiktoken,
matplotlib, aksharamukha) defaults to writing caches under ``$HOME``. This
module redirects all of them into ``<repo>/.cache`` and ``<repo>/models`` so a
checkout is fully self-contained and nothing lands in the home directory.

``bootstrap()`` is invoked from ``scriptgap/__init__.py`` before any heavy
import, so importing anything from ``scriptgap`` is enough to get isolation.
Setting the variables in ``run.sh`` as well is belt-and-braces for subprocesses
that never import us (e.g. a bare ``python -c`` or the vLLM worker spawner).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set once ``bootstrap()`` has run, so repeated imports are cheap and idempotent.
_BOOTSTRAPPED = False


def repo_root() -> Path:
    """Return the repository root (three levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def bootstrap(root: Path | None = None) -> Path:
    """Point every known cache environment variable inside the repo.

    Pre-existing values are respected — if the caller already exported
    ``HF_HOME`` (for example to share a large model cache across checkouts on a
    GPU box) we do not override it.
    """
    global _BOOTSTRAPPED
    root = Path(root) if root is not None else repo_root()
    if _BOOTSTRAPPED:
        return root

    # Captured before the defaults below can redirect it — see _ensure_hf_token.
    caller_hf_home = os.environ.get("HF_HOME")

    cache = root / ".cache"
    models = root / "models"

    # dir-creating side effects are cheap and make the first run less noisy
    for sub in ("huggingface", "torch", "triton", "vllm", "tiktoken", "matplotlib", "xdg", "tmp"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    defaults = {
        # HuggingFace: HF_HOME covers hub+datasets+metrics on modern versions,
        # but the older explicit vars are still read by some code paths.
        "HF_HOME": str(cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface" / "transformers"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        # torch / vllm / triton
        "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch" / "extensions"),
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "VLLM_CONFIG_ROOT": str(cache / "vllm"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        # misc
        "TIKTOKEN_CACHE_DIR": str(cache / "tiktoken"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "TMPDIR": str(cache / "tmp"),
        # aksharamukha writes a small state dir next to its package by default;
        # this keeps any stray writes local too.
        "AKSHARAMUKHA_DATA": str(cache / "aksharamukha"),
        # Keep tokenizers quiet inside the dataloader forks.
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    _ensure_hf_token(caller_hf_home)

    _BOOTSTRAPPED = True
    return root


def _ensure_hf_token(caller_hf_home: str | None) -> None:
    """Keep ``hf auth login`` working across the HF_HOME redirection above.

    ``huggingface_hub`` reads the saved token from ``$HF_HOME/token``. Moving
    HF_HOME into the repo therefore hides a token that was written under a
    different HF_HOME, and the symptom is misleading: ``hf auth login`` still
    reports "already logged in" while every gated repo returns 401. Gated
    *datasets* are worse still — ``datasets`` silently falls back to whatever is
    cached, so a task quietly builds fewer items instead of failing.

    So: if the effective HF_HOME has no token, look where one plausibly is and
    promote it to ``HF_TOKEN``, which the hub reads regardless of HF_HOME.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return

    effective = os.environ.get("HF_HOME")
    if effective and (Path(effective) / "token").is_file():
        return  # the normal case: the token is where the hub will look

    home = Path.home()
    candidates = [
        Path(caller_hf_home) / "token" if caller_hf_home else None,
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    ]
    for path in candidates:
        if path is None:
            continue
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            os.environ["HF_TOKEN"] = token
            return


def describe() -> dict[str, str]:
    """Cache-related environment, for logging at run start."""
    keys = (
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "VLLM_CACHE_ROOT",
        "TRITON_CACHE_DIR",
        "TIKTOKEN_CACHE_DIR",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "HOME",
    )
    return {k: os.environ.get(k, "<unset>") for k in keys}
