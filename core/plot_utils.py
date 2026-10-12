"""Helpers that keep long-recording plots fast without hiding data.

A one-hour recording at 30 fps is ~108,000 samples; drawing every sample as
its own path vertex or polygon takes seconds per figure, so traces are
decimated to their per-bucket extremes and boolean states are drawn as runs.
"""

from typing import Tuple

import numpy as np

DEFAULT_MAX_TRACE_POINTS = 5000


def decimate_min_max(
    x,
    y,
    max_points: int = DEFAULT_MAX_TRACE_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a trace to at most ``max_points`` points, keeping every peak.

    The trace is split into equal buckets and each bucket contributes its
    minimum and maximum sample in time order, so spikes and dips survive and
    the plotted envelope is identical to the full trace. Traces that already
    fit are returned unchanged.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    count = min(len(x), len(y))
    x, y = x[:count], y[:count]
    if max_points < 4 or count <= max_points:
        return x, y

    bucket = -(-count // (max_points // 2 - 1))  # ceil division
    full = (count // bucket) * bucket
    blocks = y[:full].reshape(-1, bucket)
    offsets = np.arange(0, full, bucket)
    low = blocks.argmin(axis=1) + offsets
    high = blocks.argmax(axis=1) + offsets
    indices = [np.minimum(low, high), np.maximum(low, high)]
    if full < count:
        tail = y[full:]
        tail_low = int(tail.argmin()) + full
        tail_high = int(tail.argmax()) + full
        indices[0] = np.append(indices[0], min(tail_low, tail_high))
        indices[1] = np.append(indices[1], max(tail_low, tail_high))
    # Interleave (first, second) per bucket; drop the repeat when a bucket's
    # minimum and maximum are the same sample.
    order = np.empty(len(indices[0]) * 2, dtype=np.int64)
    order[0::2] = indices[0]
    order[1::2] = indices[1]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = order[1:] != order[:-1]
    order = order[keep]
    return x[order], y[order]


def true_runs(flags) -> Tuple[np.ndarray, np.ndarray]:
    """Return (start, stop) index arrays of each run of True in ``flags``.

    ``stop`` is exclusive, so run ``k`` covers ``flags[start[k]:stop[k]]``.
    """
    flags = np.asarray(flags, dtype=bool)
    padded = np.concatenate(([False], flags, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return changes[0::2], changes[1::2]


def run_spans(times, flags) -> list:
    """Return ``[(start_time, width), ...]`` for each run of True flags.

    Each sample is taken to last until the next one, so a run covers
    ``times[start]`` to ``times[stop]`` (or one sample step past the final
    sample). Single-sample runs therefore stay visible.
    """
    times = np.asarray(times, dtype=float)
    starts, stops = true_runs(flags[:len(times)])
    if len(starts) == 0:
        return []
    if len(times) > 1:
        last_step = times[-1] - times[-2]
    else:
        last_step = 0.0
    extended = np.append(times, times[-1] + last_step)
    begin = extended[starts]
    return list(zip(begin.tolist(), (extended[stops] - begin).tolist()))
