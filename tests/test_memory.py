"""One model resident at a time.

The roster is evaluated sequentially so a 48 GB card can host a 24 GB fp8
Sarvam-M and a 24 GB Gemma-3 in the same pass. Two things have to hold:

* a backend is torn down before the next one is constructed — never two live at
  once;
* teardown actually drops the references that keep an engine alive, rather than
  appearing to.

These use a fake backend so the invariant is tested without a GPU.
"""

from __future__ import annotations

import pytest

from scriptgap.config import Config
from scriptgap.eval import isolate, run_inference
from scriptgap.models.backends import Generation
from scriptgap.models.registry import ModelSpec


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Records construction and teardown order across all instances."""

    events: list[str] = []
    live: set[str] = set()
    max_live: int = 0

    def __init__(self, spec, cfg):
        self.spec = spec
        type(self).events.append(f"load:{spec.name}")
        type(self).live.add(spec.name)
        type(self).max_live = max(type(self).max_live, len(type(self).live))

    def generate(self, batch, max_new_tokens, desc="", on_progress=None):
        if on_progress:
            on_progress(len(batch))
        return [Generation("Answer: A", "stop", 3) for _ in batch]

    def close(self):
        type(self).events.append(f"unload:{self.spec.name}")
        type(self).live.discard(self.spec.name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @classmethod
    def reset(cls):
        cls.events = []
        cls.live = set()
        cls.max_live = 0


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """A tiny rendered suite plus a run directory, with no real model anywhere."""
    from scriptgap.data.build import render_item
    from scriptgap.data.schema import BaseItem, script_for
    from scriptgap.paths import RunPaths

    cfg = Config(
        data={
            "seed": 1,
            "data": {"noisy_samples": 1},
            "transforms": {
                "clean_roman": {"scheme": "ISO", "strip_diacritics": True},
                "noisy_roman": {}, "native_noisy": {},
                "codemix": {"insertion_rate": 0.0, "max_insertions": 0},
            },
            "inference": {"max_new_tokens": {"milu": 8}, "checkpoint_every": 1},
        }
    )
    item = BaseItem(
        item_id="i1", task="milu", language="hi", script=script_for("hi"),
        prompt="भारत की राजधानी क्या है?", answer="A", answer_type="mcq",
        options=["नई दिल्ली", "मुंबई"],
    )
    cells = list(render_item(item, cfg))
    monkeypatch.setattr(run_inference, "load_rendered", lambda _cfg: cells)
    monkeypatch.setattr(run_inference, "make_backend", _FakeBackend)

    _FakeBackend.reset()
    return cfg, RunPaths.create(run_id="mem-test") if False else RunPaths(
        run_id="mem", root=tmp_path / "mem"
    )


def _specs(n: int) -> list[ModelSpec]:
    return [ModelSpec(name=f"m{i}", backend="fake") for i in range(n)]


def test_only_one_backend_is_live_at_a_time(suite):
    cfg, run = suite
    run_inference.run(cfg, run, _specs(3))

    assert _FakeBackend.max_live == 1, (
        f"{_FakeBackend.max_live} backends were live simultaneously — "
        "models must be loaded one at a time"
    )
    assert not _FakeBackend.live, "a backend was left loaded at the end of the run"


def test_each_model_is_unloaded_before_the_next_loads(suite):
    cfg, run = suite
    run_inference.run(cfg, run, _specs(3))

    events = _FakeBackend.events
    assert events == [
        "load:m0", "unload:m0",
        "load:m1", "unload:m1",
        "load:m2", "unload:m2",
    ], events


def test_backend_is_unloaded_even_when_a_shard_raises(suite, monkeypatch):
    """A crash mid-model must not leak the model into the next one's budget."""
    cfg, run = suite

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated generation failure")

    monkeypatch.setattr(_FakeBackend, "generate", explode)
    run_inference.run(cfg, run, _specs(2))

    assert not _FakeBackend.live, "the backend stayed loaded after an exception"
    assert _FakeBackend.max_live == 1
    # each model is still unloaded before the next loads
    assert _FakeBackend.events == [
        "load:m0", "unload:m0", "load:m1", "unload:m1",
    ], _FakeBackend.events


def test_one_failing_model_does_not_cost_the_rest(suite, monkeypatch):
    """The default on a multi-hour unattended run is to carry on.

    A model that cannot load (OOM, missing weights, an unsupported quantisation)
    should not discard the models after it — their shards are independent.
    """
    cfg, run = suite
    calls = {"n": 0}
    original = _FakeBackend.generate

    def fail_first_model(self, *args, **kwargs):
        if self.spec.name == "m0":
            calls["n"] += 1
            raise RuntimeError("simulated OOM")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_FakeBackend, "generate", fail_first_model)
    summary = run_inference.run(cfg, run, _specs(3))

    assert "m0" in summary["failed_models"]
    assert "simulated OOM" in summary["failed_models"]["m0"]
    # the later models still ran
    assert summary["models"]["m1"]["generated"] > 0
    assert summary["models"]["m2"]["generated"] > 0
    assert not _FakeBackend.live


def test_failure_can_be_made_fatal(suite, monkeypatch):
    cfg, run = suite
    cfg.data["inference"]["continue_on_model_error"] = False

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated generation failure")

    monkeypatch.setattr(_FakeBackend, "generate", explode)
    with pytest.raises(RuntimeError):
        run_inference.run(cfg, run, _specs(2))

    assert not _FakeBackend.live, "the backend stayed loaded after a fatal error"


def test_budget_exhaustion_still_unloads(suite):
    cfg, run = suite
    summary = run_inference.run(cfg, run, _specs(3), max_seconds=-1.0)

    assert summary["budget_exhausted"] is True
    assert not _FakeBackend.live, "hitting the time budget left a model resident"


# ---------------------------------------------------------------------------
# Isolation policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,backends,expected",
    [
        ("auto", ["vllm", "vllm"], True),      # the case that matters
        ("auto", ["vllm"], False),             # nothing to protect against
        ("auto", ["hf", "hf"], False),         # hf releases cleanly
        ("auto", ["hf", "vllm"], True),        # any vLLM in the roster
        ("auto", ["anthropic", "anthropic"], False),
        ("always", ["hf"], True),
        ("never", ["vllm", "vllm"], False),
    ],
)
def test_isolation_policy(mode, backends, expected):
    specs = [ModelSpec(name=f"m{i}", backend=b) for i, b in enumerate(backends)]
    assert isolate.should_isolate(mode, specs, "vllm") is expected


def test_unknown_isolation_mode_is_rejected():
    with pytest.raises(SystemExit):
        isolate.should_isolate("sometimes", _specs(2), "vllm")


# ---------------------------------------------------------------------------
# Child command construction
# ---------------------------------------------------------------------------


def test_child_argv_targets_one_model_and_shares_the_run():
    argv = ["run-inference", "-c", "configs/models.yaml", "--task", "milu"]
    command = isolate.child_argv(argv, "sarvam-m", "run42", None)

    assert command[1:3] == ["-m", "scriptgap.cli"]
    assert command.count("--model") == 1
    assert command[command.index("--model") + 1] == "sarvam-m"
    assert command[command.index("--run-id") + 1] == "run42"
    # the child must not isolate again
    assert "inference.isolate_models=never" in command
    # unrelated flags survive
    assert "--task" in command and "configs/models.yaml" in command


def test_child_argv_replaces_a_pre_existing_model_flag():
    """A parent invoked with --model must not pass that filter down verbatim."""
    for argv in (
        ["run-inference", "--model", "other", "--run-id", "old"],
        ["run-inference", "--model=other", "--run-id=old"],
    ):
        command = isolate.child_argv(argv, "target", "new", None)
        assert command.count("--model") == 1
        assert command[command.index("--model") + 1] == "target"
        assert "other" not in command
        assert "old" not in command
        assert command[command.index("--run-id") + 1] == "new"


def test_child_argv_passes_the_remaining_budget():
    command = isolate.child_argv(["run-inference"], "m", "r", 1234.6)
    assert command[command.index("--max-seconds") + 1] == "1235"

    # and never a non-positive one
    command = isolate.child_argv(["run-inference"], "m", "r", 0.2)
    assert float(command[command.index("--max-seconds") + 1]) >= 1.0


def test_strip_flag_handles_both_spellings():
    argv = ["a", "--x", "1", "--x=2", "b"]
    assert isolate.strip_flag(argv, "--x") == ["a", "b"]
    assert isolate.strip_flag(["a", "--flag", "b"], "--flag", takes_value=False) == ["a", "b"]


# ---------------------------------------------------------------------------
# GPU helpers (must be safe on a CPU-only box)
# ---------------------------------------------------------------------------


def test_gpu_helpers_are_no_ops_without_cuda():
    from scriptgap.models import gpu

    assert gpu.snapshot() == [] or gpu.available()
    gpu.release(deep=True)  # must not raise
    ok, devices = gpu.wait_until_free(timeout=0.1)
    if not gpu.available():
        assert ok and devices == []
        assert gpu.teardown("nothing") is True
        assert gpu.assert_headroom(999.0) is True


def test_device_memory_arithmetic():
    from scriptgap.models.gpu import DeviceMemory

    device = DeviceMemory(index=0, name="X", free_gb=12.0, total_gb=48.0)
    assert device.used_gb == 36.0
    assert device.free_fraction == pytest.approx(0.25)
    assert "12.0/48.0" in str(device)
