"""Generation backends: vLLM, transformers, Anthropic.

All three present the same surface — build one, call :meth:`Backend.generate`
with a batch of chat messages, close it — so the inference driver never branches
on which engine is in use.

Two properties matter to the study rather than to convenience:

* **Teardown is real.** ``close()`` drops the engine's references and verifies
  the card came back (see ``gpu.py``). Only one model is ever resident, which is
  what lets a 48 GB card host the whole roster in one pass.
* **A failed generation is a record, not a gap.** Anything that goes wrong for
  one item yields ``Generation.failed()`` — an empty completion with
  ``finish_reason="error"`` — so it is scored as a failure and counted, rather
  than silently dropping the row and shrinking that cell's denominator.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import gpu
from .registry import ModelSpec

log = logging.getLogger(__name__)

Message = dict[str, str]
ProgressFn = Callable[[int], Any]


class _NullStream:
    """A file object that swallows everything written to it."""

    def write(self, *_args: Any) -> int:
        return 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _forwarding_tqdm(sink: ProgressFn) -> type:
    """A tqdm subclass that reports into ``sink`` instead of drawing itself.

    vLLM's ``use_tqdm`` accepts a factory, so handing it this makes it advance
    the caller's shard bar once per *completed request* rather than once per
    64-item write chunk. Without it a big model shows no movement for minutes at
    a time. Nothing is rendered, so the terminal still shows
    models > shards > items and not a third bar per chunk.

    It is silenced by writing to a null stream rather than by ``disable=True``:
    a disabled tqdm reports ``elapsed == 0``, and vLLM divides by that to report
    throughput — which raises ``ZeroDivisionError`` and kills the shard. The bar
    therefore stays fully live, it just renders nowhere.
    """
    from tqdm.auto import tqdm as _tqdm

    class _Forwarding(_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["file"] = _NullStream()
            kwargs["disable"] = False
            kwargs.pop("dynamic_ncols", None)
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> Any:
            if n:
                sink(int(n))
            return super().update(n)

        @property
        def format_dict(self) -> dict[str, Any]:
            # Belt and braces for the division above: a chunk that completes
            # inside one clock tick must not read back as zero elapsed time.
            stats = super().format_dict
            if not stats.get("elapsed"):
                stats["elapsed"] = 1e-6
            return stats

    return _Forwarding


@dataclass
class Generation:
    """One completion, plus why it stopped.

    ``finish_reason`` is carried all the way into the released table: a
    generation cut off at ``max_tokens`` is indistinguishable from a wrong
    answer once reduced to a score, and truncation that differs by condition
    manufactures a script gap.
    """

    text: str
    finish_reason: str = "stop"
    output_tokens: int = 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @classmethod
    def failed(cls, reason: str = "error") -> "Generation":
        return cls(text="", finish_reason=reason, output_tokens=0)


class Backend:
    """Common lifecycle. Subclasses implement :meth:`generate`."""

    def __init__(self, spec: ModelSpec, cfg: dict[str, Any]) -> None:
        self.spec = spec
        self.cfg = dict(cfg or {})

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
        max_new_tokens: int,
        desc: str = "",
        on_progress: ProgressFn | None = None,
    ) -> list[Generation]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- shared helpers ----------------------------------------------------
    @property
    def temperature(self) -> float:
        return float(self.cfg.get("temperature", 0.0) or 0.0)

    @property
    def top_p(self) -> float:
        return float(self.cfg.get("top_p", 1.0) or 1.0)


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


def _disable_flashinfer_sampler_if_unusable() -> None:
    """Fall back to PyTorch sampling when FlashInfer cannot target this GPU.

    FlashInfer JIT-compiles its sampling kernels with the system ``nvcc``. When
    that toolkit is older than the GPU needs — CUDA 12.8 against an SM 12.0
    Blackwell card, which FlashInfer requires >= 12.9 to build for — it collects
    an *empty* target-architecture set and then reports the emptiness as
    ``RuntimeError: FlashInfer requires GPUs with sm75 or higher``. That message
    sends you looking at the GPU, which is far newer than sm75; the real problem
    is the toolkit.

    The kernels are a throughput optimisation, not a correctness requirement
    (this study samples greedily at temperature 0), so the run continues on
    vLLM's PyTorch sampler. Set ``VLLM_USE_FLASHINFER_SAMPLER`` yourself to
    override either way.
    """
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
        return
    try:
        from flashinfer.compilation_context import CompilationContext

        usable = bool(CompilationContext().TARGET_CUDA_ARCHS)
    except Exception:
        return  # no FlashInfer at all: vLLM already handles that path
    if usable:
        return

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    log.warning(
        "FlashInfer cannot build sampling kernels for this GPU/toolkit "
        "combination — falling back to vLLM's PyTorch sampler. This costs a "
        "little throughput and changes no results."
    )


def _supported_engine_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop engine arguments this vLLM build does not accept.

    vLLM's engine arguments move between releases — 0.26 removed ``swap_space``,
    which `configs/default.yaml` still sets — and an unknown keyword is a
    ``TypeError`` at construction. On an unattended roster that surfaces as
    "model failed" after the engine has already spent minutes loading, with the
    real cause buried in a traceback.

    Unknown keys are therefore dropped with a warning naming them, so the run
    continues on a config written for a different vLLM and the mismatch is still
    visible in the log.
    """
    try:
        import dataclasses

        from vllm.engine.arg_utils import EngineArgs

        accepted = {field.name for field in dataclasses.fields(EngineArgs)}
    except Exception:  # pragma: no cover - unknown vLLM layout: pass through
        return kwargs

    kept = {key: value for key, value in kwargs.items() if key in accepted}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        try:
            import vllm

            version = vllm.__version__
        except Exception:  # pragma: no cover
            version = "?"
        log.warning(
            "vLLM %s does not accept %s — ignoring. Remove them from "
            "inference.vllm.* in the config to silence this.",
            version,
            dropped,
        )
    return kept


class VLLMBackend(Backend):
    """Local generation through vLLM — the GPU-day default."""

    def __init__(self, spec: ModelSpec, cfg: dict[str, Any]) -> None:
        super().__init__(spec, cfg)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise SystemExit(
                "backend 'vllm' is unavailable — run `./setup_env.sh --gpu`, "
                "or set inference.backend=hf"
            ) from exc

        from transformers import AutoTokenizer

        vllm_cfg = dict(self.cfg.get("vllm", {}) or {})
        source = spec.resolve_path()
        log.info("loading %s with vLLM from %s", spec.name, source)

        # Must happen before LLM() spawns its EngineCore child, which inherits
        # this environment and is where the sampler is actually constructed.
        _disable_flashinfer_sampler_if_unusable()

        kwargs: dict[str, Any] = {
            "model": source,
            "dtype": spec.dtype or vllm_cfg.get("dtype", "bfloat16"),
            "gpu_memory_utilization": float(vllm_cfg.get("gpu_memory_utilization", 0.90)),
            "max_model_len": int(spec.max_model_len or vllm_cfg.get("max_model_len", 4096)),
            "enforce_eager": bool(vllm_cfg.get("enforce_eager", False)),
            "swap_space": int(vllm_cfg.get("swap_space", 4)),
            "trust_remote_code": True,
        }
        if spec.quantization:
            kwargs["quantization"] = spec.quantization
        if self.cfg.get("seed") is not None:
            kwargs["seed"] = int(self.cfg["seed"])

        self._SamplingParams = SamplingParams
        self.llm = LLM(**_supported_engine_args(kwargs))
        self.tokeniser = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        gpu.log_memory(f"after loading {spec.name}")

    def _render(self, messages: Sequence[Message]) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.spec.disable_thinking:
            # Qwen3 and friends: without this the model spends the token budget
            # on a <think> block and the answer is truncated away.
            kwargs["enable_thinking"] = False
        try:
            return self.tokeniser.apply_chat_template(list(messages), **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokeniser.apply_chat_template(list(messages), **kwargs)
        except Exception:
            return "\n\n".join(message["content"] for message in messages)

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
        max_new_tokens: int,
        desc: str = "",
        on_progress: ProgressFn | None = None,
    ) -> list[Generation]:
        prompts = [self._render(messages) for messages in batch]
        params = self._SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(max_new_tokens),
        )

        reported = {"n": 0}

        def sink(n: int) -> None:
            reported["n"] += n
            if on_progress:
                on_progress(n)

        outputs = self.llm.generate(
            prompts, params, use_tqdm=_forwarding_tqdm(sink) if on_progress else False
        )

        results: list[Generation] = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                Generation(
                    text=completion.text or "",
                    finish_reason=completion.finish_reason or "stop",
                    output_tokens=len(completion.token_ids or []),
                )
            )

        # Reconcile: if a future vLLM stops driving the factory, the caller's bar
        # must still end up in the right place rather than silently stalling.
        missing = len(results) - reported["n"]
        if on_progress and missing > 0:
            on_progress(missing)
        return results

    def close(self) -> None:
        for attribute in ("llm", "tokeniser"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        gpu.teardown(self.spec.name)


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------


class HFBackend(Backend):
    """transformers generation — CPU-friendly, used for smoke runs."""

    def __init__(self, spec: ModelSpec, cfg: dict[str, Any]) -> None:
        super().__init__(spec, cfg)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit("backend 'hf' is unavailable — transformers is not installed") from exc

        source = spec.resolve_path()
        log.info("loading %s with transformers from %s", spec.name, source)
        self.torch = torch
        self.tokeniser = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        if self.tokeniser.pad_token_id is None:
            self.tokeniser.pad_token = self.tokeniser.eos_token
        self.tokeniser.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        gpu.log_memory(f"after loading {spec.name}")

    def _render(self, messages: Sequence[Message]) -> str:
        try:
            return self.tokeniser.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return "\n\n".join(message["content"] for message in messages)

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
        max_new_tokens: int,
        desc: str = "",
        on_progress: ProgressFn | None = None,
    ) -> list[Generation]:
        torch = self.torch
        batch_size = max(1, int(self.cfg.get("batch_size", 8)))
        prompts = [self._render(messages) for messages in batch]
        results: list[Generation] = []

        for start in range(0, len(prompts), batch_size):
            window = prompts[start : start + batch_size]
            encoded = self.tokeniser(
                window, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else None,
                        top_p=self.top_p if self.temperature > 0 else None,
                        pad_token_id=self.tokeniser.pad_token_id,
                    )
            except Exception as exc:
                log.warning("generation failed for a batch of %d (%s)", len(window), exc)
                results.extend(Generation.failed() for _ in window)
                if on_progress:
                    on_progress(len(window))
                continue

            prompt_length = encoded["input_ids"].shape[1]
            for row in generated:
                new_tokens = row[prompt_length:]
                text = self.tokeniser.decode(new_tokens, skip_special_tokens=True)
                produced = int((new_tokens != self.tokeniser.pad_token_id).sum())
                results.append(
                    Generation(
                        text=text,
                        finish_reason="length" if produced >= int(max_new_tokens) else "stop",
                        output_tokens=produced,
                    )
                )
            if on_progress:
                on_progress(len(window))

        return results

    def close(self) -> None:
        for attribute in ("model", "tokeniser"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        gpu.teardown(self.spec.name)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicBackend(Backend):
    """API generation, for the optional frontier reference model."""

    def __init__(self, spec: ModelSpec, cfg: dict[str, Any]) -> None:
        super().__init__(spec, cfg)
        try:
            import anthropic
        except ImportError as exc:
            raise SystemExit(
                "backend 'anthropic' is unavailable — run `./setup_env.sh --with-api`"
            ) from exc

        self.anthropic = anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.info("ANTHROPIC_API_KEY unset; the SDK will use an `ant auth login` profile")
        self.client = anthropic.Anthropic()
        self.model = spec.api_model or spec.hf_id or spec.name
        self.max_retries = int(self.cfg.get("api_max_retries", 5))

    def generate(
        self,
        batch: Sequence[Sequence[Message]],
        max_new_tokens: int,
        desc: str = "",
        on_progress: ProgressFn | None = None,
    ) -> list[Generation]:
        results = [self._one(messages, max_new_tokens) for messages in batch]
        if on_progress:
            on_progress(len(results))
        return results

    def _one(self, messages: Sequence[Message], max_new_tokens: int) -> Generation:
        from ..eval.scoring import API_REFUSAL_SENTINEL

        delay = 2.0
        for _attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=int(max_new_tokens),
                    temperature=self.temperature,
                    messages=[dict(message) for message in messages],
                )
                if response.stop_reason == "refusal":
                    # A classifier-side decline. Marked rather than dropped, so
                    # H4 can count it as the refusal it is.
                    return Generation(API_REFUSAL_SENTINEL, "refusal", 0)
                text = "".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                )
                return Generation(
                    text=text,
                    finish_reason=response.stop_reason or "stop",
                    output_tokens=getattr(response.usage, "output_tokens", 0),
                )
            except self.anthropic.RateLimitError:
                log.warning("rate limited; backing off %.0fs", delay)
            except self.anthropic.APIStatusError as exc:
                if exc.status_code < 500:
                    log.warning("API error %s: %s", exc.status_code, exc.message)
                    return Generation.failed()
                log.warning("server error %s; backing off %.0fs", exc.status_code, delay)
            except self.anthropic.APIConnectionError as exc:
                log.warning("connection error (%s); backing off %.0fs", exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

        log.error("giving up on one generation after %d attempts", self.max_retries)
        return Generation.failed()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BACKENDS: dict[str, type[Backend]] = {
    "vllm": VLLMBackend,
    "hf": HFBackend,
    "transformers": HFBackend,
    "anthropic": AnthropicBackend,
}


def make_backend(spec: ModelSpec, cfg: dict[str, Any] | None = None) -> Backend:
    """Build the backend for one model spec."""
    cfg = dict(cfg or {})
    name = str(spec.backend or cfg.get("backend") or "vllm").lower()
    backend = BACKENDS.get(name)
    if backend is None:
        raise SystemExit(
            f"unknown backend {name!r} for model {spec.name}; "
            f"choose one of {sorted(BACKENDS)}"
        )
    return backend(spec, cfg)
