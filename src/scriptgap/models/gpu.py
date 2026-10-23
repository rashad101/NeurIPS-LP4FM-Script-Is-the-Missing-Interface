"""Device memory accounting and verified teardown between models.

The roster is walked one model at a time so a single 48 GB card can host a
24 GB fp8 Sarvam-M and a 24 GB Gemma-3 in the same pass. That only works if
model *n*'s memory is actually back before model *n+1* allocates its KV cache,
and "actually" is the operative word: calling ``del`` on an engine is not the
same as the allocator handing the memory to the driver.

So teardown here is **verified, not assumed** — the card is polled until it is
free again, and if it does not come back that is logged plainly with the fix
(``inference.isolate_models=always``) rather than being discovered as an OOM on
the next model.

Every function is a no-op on a CPU-only box, so the same code path runs in
tests and on a laptop.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Treat the card as reclaimed at this free fraction. Not 1.0: CUDA context and
#: cuBLAS workspaces are a legitimate residual that never comes back in-process.
DEFAULT_FREE_FRACTION = 0.80

_BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class DeviceMemory:
    """A snapshot of one device's memory, in GiB."""

    index: int
    name: str
    free_gb: float
    total_gb: float

    @property
    def used_gb(self) -> float:
        return round(self.total_gb - self.free_gb, 4)

    @property
    def free_fraction(self) -> float:
        return self.free_gb / self.total_gb if self.total_gb else 1.0

    def __str__(self) -> str:
        return (
            f"cuda:{self.index} {self.name} "
            f"{self.free_gb:.1f}/{self.total_gb:.1f} GiB free "
            f"({100 * self.free_fraction:.0f}%)"
        )


def available() -> bool:
    """True when CUDA is usable. Everything else degrades to a no-op."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def snapshot() -> list[DeviceMemory]:
    """Current free/total memory per device (empty list without CUDA)."""
    if not available():
        return []
    try:
        import torch

        out: list[DeviceMemory] = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            out.append(
                DeviceMemory(
                    index=index,
                    name=torch.cuda.get_device_name(index),
                    free_gb=round(free / _BYTES_PER_GB, 3),
                    total_gb=round(total / _BYTES_PER_GB, 3),
                )
            )
        return out
    except Exception as exc:  # pragma: no cover - driver dependent
        log.debug("could not read device memory (%s)", exc)
        return []


def log_memory(label: str = "") -> list[DeviceMemory]:
    devices = snapshot()
    for device in devices:
        log.info("GPU %s%s", f"[{label}] " if label else "", device)
    return devices


def release(deep: bool = False) -> None:
    """Drop cached allocations. ``deep`` also tears down distributed state.

    vLLM keeps module-level distributed state and its own executor; without the
    deep pass those keep the CUDA context alive and the next model starts a few
    hundred MB short — which on a single card is exactly the difference between
    fitting and OOM.
    """
    gc.collect()
    if not available():
        return
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if deep:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as exc:  # pragma: no cover - vLLM version dependent
                log.debug("could not destroy the process group (%s)", exc)
            try:
                from vllm.distributed.parallel_state import (  # type: ignore
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception as exc:  # pragma: no cover - vLLM version dependent
                log.debug("no vLLM parallel state to destroy (%s)", exc)
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - driver dependent
        log.debug("release failed (%s)", exc)


def wait_until_free(
    min_free_fraction: float = DEFAULT_FREE_FRACTION,
    timeout: float = 30.0,
    poll: float = 0.5,
) -> tuple[bool, list[DeviceMemory]]:
    """Poll until every device is free again. Returns ``(released, devices)``."""
    if not available():
        return True, []

    deadline = time.time() + max(0.0, timeout)
    devices = snapshot()
    while True:
        devices = snapshot()
        if not devices or all(d.free_fraction >= min_free_fraction for d in devices):
            return True, devices
        if time.time() >= deadline:
            return False, devices
        release()
        time.sleep(poll)


def teardown(
    label: str,
    min_free_fraction: float = DEFAULT_FREE_FRACTION,
    timeout: float = 30.0,
) -> bool:
    """Release everything and verify the card came back. Logs what happened."""
    before = snapshot()
    release(deep=True)
    released, after = wait_until_free(min_free_fraction, timeout=timeout)

    if not after:
        return True

    reclaimed = 0.0
    for old, new in zip(before, after):
        reclaimed += max(0.0, new.free_gb - old.free_gb)

    if released:
        log.info("unloaded %s (reclaimed +%.1f GiB); %s", label, reclaimed, after[0])
    else:
        log.warning(
            "GPU memory was not fully released after unloading %s "
            "(reclaimed +%.1f GiB, %s). vLLM does not always hand back its CUDA "
            "context in-process — set inference.isolate_models=always to run each "
            "model in its own process instead.",
            label,
            reclaimed,
            after[0],
        )
    return released


def assert_headroom(required_gb: float) -> bool:
    """True when some device has ``required_gb`` free (or there is no CUDA)."""
    devices = snapshot()
    if not devices:
        return True
    best = max(device.free_gb for device in devices)
    if best >= required_gb:
        return True
    log.warning("only %.1f GiB free; %.1f GiB wanted", best, required_gb)
    return False
