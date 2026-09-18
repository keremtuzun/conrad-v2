"""HIL measurement primitives: fixed-rate scheduler, latency statistics, resource probes.

No new dependencies: CPU from ``time.process_time``, Python heap from ``tracemalloc``, RSS from
``/proc/self/statm`` when it exists (Linux targets), GPU from torch only when CUDA is present.
Anything that cannot be measured is reported as ``NOT_AVAILABLE`` - never as zero.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import Field

from conrad.schemas.base import ConradModel

NOT_AVAILABLE = "NOT_AVAILABLE"
Clock = Callable[[], int]
Sleep = Callable[[float], None]


class FixedRateScheduler:
    """Absolute-deadline scheduler: tick k is due at ``start + k * period``.

    A late cycle is counted as an overrun; the scheduler then skips to the next future tick instead of
    bursting to catch up (bursts would hide the deadline miss from the control loop).
    """

    def __init__(
        self, period_ns: int, clock: Clock = time.perf_counter_ns, sleep: Sleep = time.sleep
    ) -> None:
        if period_ns <= 0:
            raise ValueError("period_ns must be positive")
        self.period_ns = period_ns
        self._clock = clock
        self._sleep = sleep
        self._start: int | None = None
        self._tick = 0
        self.skipped_ticks = 0

    def start(self) -> int:
        self._start = self._clock()
        self._tick = 0
        return self._start

    def deadline_ns(self) -> int:
        if self._start is None:
            raise RuntimeError("scheduler not started")
        return self._start + (self._tick + 1) * self.period_ns

    def wait_next(self) -> bool:
        """Sleep until the next tick. Returns False when the tick was already missed."""
        deadline = self.deadline_ns()
        now = self._clock()
        on_time = now <= deadline
        if on_time:
            self._sleep((deadline - now) / 1e9)
            self._tick += 1
        else:
            assert self._start is not None
            late_ticks = (now - self._start) // self.period_ns
            self.skipped_ticks += int(late_ticks - self._tick)
            self._tick = int(late_ticks)
        return on_time


class LatencyStats(ConradModel):
    count: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None

    @classmethod
    def of(cls, samples_ns: list[int]) -> LatencyStats:
        if not samples_ns:
            return cls(count=0, mean_ms=None, p50_ms=None, p95_ms=None, p99_ms=None, max_ms=None)
        a = np.asarray(samples_ns, dtype=np.float64) / 1e6
        return cls(
            count=len(samples_ns),
            mean_ms=float(a.mean()),
            p50_ms=float(np.percentile(a, 50)),
            p95_ms=float(np.percentile(a, 95)),
            p99_ms=float(np.percentile(a, 99)),
            max_ms=float(a.max()),
        )


class ResourceSnapshot(ConradModel):
    wall_ns: int
    cpu_process_s: float
    python_heap_bytes: int | None
    rss_bytes: int | None


class ResourceReport(ConradModel):
    cpu_utilisation_fraction: float | None = Field(description="process CPU time / wall time, all cores")
    python_heap_start_bytes: int | None
    python_heap_end_bytes: int | None
    python_heap_peak_bytes: int | None
    python_heap_growth_bytes: int | None
    rss_start_bytes: int | None
    rss_end_bytes: int | None
    rss_source: str
    gpu_status: str
    gpu_memory_bytes: int | None = None
    notes: tuple[str, ...] = ()


def _rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    pages = int(statm.read_text().split()[1])
    return pages * int(sysconf("SC_PAGE_SIZE"))


def default_gpu_probe() -> tuple[str, int | None]:
    """(status, bytes allocated). Imports torch only if it is already importable; never raises."""
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch as torch_mod
        except ImportError:
            return NOT_AVAILABLE + ":torch_not_installed", None
        torch = torch_mod
    if not torch.cuda.is_available():
        return NOT_AVAILABLE + ":no_cuda_device", None
    return "CUDA:" + torch.cuda.get_device_name(0), int(torch.cuda.memory_allocated(0))


class ResourceSampler:
    def __init__(self, gpu_probe: Callable[[], tuple[str, int | None]] = default_gpu_probe) -> None:
        self._gpu_probe = gpu_probe
        self._started_tracemalloc = False
        self.start_snapshot: ResourceSnapshot | None = None

    def snapshot(self) -> ResourceSnapshot:
        heap = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else None
        return ResourceSnapshot(
            wall_ns=time.perf_counter_ns(),
            cpu_process_s=time.process_time(),
            python_heap_bytes=heap,
            rss_bytes=_rss_bytes(),
        )

    def start(self) -> ResourceSnapshot:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True
        tracemalloc.reset_peak()
        self.start_snapshot = self.snapshot()
        return self.start_snapshot

    def mark_baseline(self) -> None:
        """Re-baseline after warm-up so one-off allocations are not reported as growth."""
        self.start_snapshot = self.snapshot()
        tracemalloc.reset_peak()

    def finish(self) -> ResourceReport:
        if self.start_snapshot is None:
            raise RuntimeError("ResourceSampler.start() was not called")
        end = self.snapshot()
        peak = tracemalloc.get_traced_memory()[1] if tracemalloc.is_tracing() else None
        if self._started_tracemalloc:
            tracemalloc.stop()
        s = self.start_snapshot
        wall = (end.wall_ns - s.wall_ns) / 1e9
        status, gpu_mem = self._gpu_probe()
        heap_growth = (
            None
            if s.python_heap_bytes is None or end.python_heap_bytes is None
            else end.python_heap_bytes - s.python_heap_bytes
        )
        return ResourceReport(
            cpu_utilisation_fraction=None if wall <= 0 else (end.cpu_process_s - s.cpu_process_s) / wall,
            python_heap_start_bytes=s.python_heap_bytes,
            python_heap_end_bytes=end.python_heap_bytes,
            python_heap_peak_bytes=peak,
            python_heap_growth_bytes=heap_growth,
            rss_start_bytes=s.rss_bytes,
            rss_end_bytes=end.rss_bytes,
            rss_source="/proc/self/statm" if end.rss_bytes is not None else NOT_AVAILABLE,
            gpu_status=status,
            gpu_memory_bytes=gpu_mem,
            notes=("python heap via tracemalloc excludes native (C/CUDA) allocations",),
        )
