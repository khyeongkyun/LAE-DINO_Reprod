# UPDATE: latency-per-block
"""Per-block inference latency recording for LAEDINO.

Only active when a detector is built with ``per_block_latency_measure=True``.
Timings are accumulated across the whole test run, excluding a leading
warmup window, and are only reduced across distributed ranks when
:meth:`BlockLatencyRecorder.sync_and_report` is called (see
``mmdet.engine.hooks.LatencyLoggerHook``).
"""
import time
from collections import defaultdict
from typing import Dict, List

import torch
from mmengine.dist import all_gather_object, get_rank, is_distributed


class BlockLatencyRecorder:
    """Accumulates named block latencies (ms) across a test run.

    Args:
        enabled (bool): Whether recording is active at all.
        warmup_iters (int): Number of leading calls to :meth:`start_iter`
            (i.e. leading inference calls) to discard before accumulating
            stats, so CUDA/cudnn/tokenizer/model lazy-init warmup doesn't
            skew the reported numbers.
    """

    def __init__(self, enabled: bool = False, warmup_iters: int = 10) -> None:
        self.enabled = enabled
        self.warmup_iters = warmup_iters
        self._iter_idx = -1
        self._records: Dict[str, List[float]] = defaultdict(list)

    def start_iter(self) -> None:
        """Mark the start of a new top-level inference call."""
        if self.enabled:
            self._iter_idx += 1

    @property
    def past_warmup(self) -> bool:
        return self._iter_idx >= self.warmup_iters

    def reset(self) -> None:
        """Clear accumulated records and restart the warmup window."""
        self._iter_idx = -1
        self._records = defaultdict(list)

    def cuda_block(self, name: str) -> '_CudaBlockTimer':
        """Context manager timing a GPU-bound block via CUDA events."""
        return _CudaBlockTimer(self, name)

    def cpu_block(self, name: str) -> '_CpuBlockTimer':
        """Context manager timing a CPU-bound block via perf_counter."""
        return _CpuBlockTimer(self, name)

    def _record(self, name: str, elapsed_ms: float) -> None:
        self._records[name].append(elapsed_ms)

    def sync_and_report(self, logger) -> None:
        """Gather per-rank records across all ranks and log a summary.

        Must be called on every rank (it performs a collective
        all-gather); only rank 0 logs the resulting table.
        """
        if not self.enabled:
            return

        if is_distributed():
            gathered = all_gather_object(dict(self._records))
        else:
            gathered = [dict(self._records)]

        merged: Dict[str, List[float]] = defaultdict(list)
        for rank_records in gathered:
            for name, values in rank_records.items():
                merged[name].extend(values)

        if get_rank() != 0:
            return

        if not merged:
            logger.info('Per-block latency: no samples recorded '
                        '(warmup window not exceeded).')
            return

        header = f'{"block":<20}{"n":>8}{"mean(ms)":>12}{"std(ms)":>12}'
        lines = [
            f'Per-block latency (warmup={self.warmup_iters} iters excluded, '
            'gathered across all ranks):', header
        ]
        for name in sorted(merged):
            values = torch.tensor(merged[name])
            lines.append(
                f'{name:<20}{values.numel():>8}'
                f'{values.mean().item():>12.3f}{values.std().item():>12.3f}')
        logger.info('\n'.join(lines))


class _CudaBlockTimer:
    """CUDA-event-based timer for a single named block."""

    def __init__(self, recorder: BlockLatencyRecorder, name: str) -> None:
        self.recorder = recorder
        self.name = name
        self._active = False

    def __enter__(self) -> '_CudaBlockTimer':
        self._active = self.recorder.enabled and self.recorder.past_warmup
        if self._active:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._active:
            self._end.record()
            # Only waits for this event's stream, not the whole device.
            self._end.synchronize()
            self.recorder._record(self.name,
                                  self._start.elapsed_time(self._end))
        return False


class _CpuBlockTimer:
    """perf_counter-based timer for a single named CPU-bound block."""

    def __init__(self, recorder: BlockLatencyRecorder, name: str) -> None:
        self.recorder = recorder
        self.name = name
        self._active = False

    def __enter__(self) -> '_CpuBlockTimer':
        self._active = self.recorder.enabled and self.recorder.past_warmup
        if self._active:
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._active:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self.recorder._record(self.name, elapsed_ms)
        return False
