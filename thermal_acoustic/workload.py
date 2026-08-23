"""A fixed, deterministic synthetic heat-generation trace (Watts) representing a bursty
compute workload -- idle baseline with a few burst windows layered in, similar in shape to
what a small embedded/telecom module sees under intermittent traffic or duty-cycled work.
"""
from __future__ import annotations

import numpy as np

_BASE_BURSTS = [(15, 30), (45, 70), (90, 110)]  # (start, end) step indices, inclusive-exclusive


def heat_trace(n_steps: int = 120, base_w: float = 15.0, burst_w: float = 35.0) -> np.ndarray:
    trace = np.full(n_steps, base_w, dtype=float)
    for start, end in _BASE_BURSTS:
        end = min(end, n_steps)
        if start < n_steps:
            trace[start:end] += burst_w
    return trace


def sample_heat_trace(
    rng: np.random.Generator,
    n_steps: int = 120,
    base_w: float = 15.0,
    burst_w: float = 35.0,
    start_jitter_steps: int = 10,
    duration_scale_range: tuple[float, float] = (0.6, 1.4),
    magnitude_scale_range: tuple[float, float] = (0.7, 1.3),
) -> np.ndarray:
    """A randomized draw from the same idle-plus-three-bursts *family* as `heat_trace()`,
    instead of one fixed realization of it. Each burst independently gets a jittered start
    time (uniform integer offset in +/-`start_jitter_steps`), a rescaled duration, and a
    rescaled magnitude -- representing the kind of workload uncertainty a single fixed
    trace can't capture (traffic bursts don't arrive on a fixed schedule with a fixed
    size in a real system). `heat_trace()` is approximately this distribution's mean
    realization, not a separate model -- same base load, same three nominal burst
    windows, same burst magnitude, just without the randomization.
    """
    trace = np.full(n_steps, base_w, dtype=float)
    for start, end in _BASE_BURSTS:
        nominal_duration = end - start
        duration = max(1, int(round(nominal_duration * rng.uniform(*duration_scale_range))))
        jitter = int(rng.integers(-start_jitter_steps, start_jitter_steps + 1))
        jittered_start = max(0, start + jitter)
        jittered_end = min(n_steps, jittered_start + duration)
        magnitude = burst_w * rng.uniform(*magnitude_scale_range)
        if jittered_start < n_steps:
            trace[jittered_start:jittered_end] += magnitude
    return trace
