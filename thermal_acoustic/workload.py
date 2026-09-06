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
    jitter_scale: float = 1.0,
) -> np.ndarray:
    """A randomized draw from the same idle-plus-three-bursts *family* as `heat_trace()`,
    instead of one fixed realization of it. Each burst independently gets a jittered start
    time (uniform integer offset in +/-`start_jitter_steps`), a rescaled duration, and a
    rescaled magnitude -- representing the kind of workload uncertainty a single fixed
    trace can't capture (traffic bursts don't arrive on a fixed schedule with a fixed
    size in a real system). `heat_trace()` is approximately this distribution's mean
    realization, not a separate model -- same base load, same three nominal burst
    windows, same burst magnitude, just without the randomization.

    `jitter_scale` uniformly scales how much workload uncertainty is present, without
    changing its shape: it multiplies `start_jitter_steps` directly, and scales
    `duration_scale_range`/`magnitude_scale_range`'s spread *around their shared center of
    1.0* (so `jitter_scale=1.0` reproduces the defaults above, and `jitter_scale=0.0`
    collapses the distribution onto `heat_trace()` exactly -- no start jitter, and every
    duration/magnitude multiplier pinned to 1.0). This is what lets the same sampler be
    used to ask "how does robustness change as workload uncertainty itself grows or
    shrinks," not just "is this policy robust to the one default amount of it."
    """
    start_jitter_steps = int(round(start_jitter_steps * jitter_scale))
    duration_scale_range = _scale_range_around_one(duration_scale_range, jitter_scale)
    magnitude_scale_range = _scale_range_around_one(magnitude_scale_range, jitter_scale)

    trace = np.full(n_steps, base_w, dtype=float)
    for start, end in _BASE_BURSTS:
        nominal_duration = end - start
        duration = max(1, int(round(nominal_duration * rng.uniform(*duration_scale_range))))
        jitter = int(rng.integers(-start_jitter_steps, start_jitter_steps + 1)) if start_jitter_steps > 0 else 0
        jittered_start = max(0, start + jitter)
        jittered_end = min(n_steps, jittered_start + duration)
        magnitude = burst_w * rng.uniform(*magnitude_scale_range)
        if jittered_start < n_steps:
            trace[jittered_start:jittered_end] += magnitude
    return trace


def _scale_range_around_one(value_range: tuple[float, float], scale: float) -> tuple[float, float]:
    """Scale a (low, high) range's half-width around its center, keeping the center fixed.
    Only meaningful for ranges centered on 1.0 (both scale ranges in this module are), which
    is what makes `scale=0.0` collapse the range to exactly `(1.0, 1.0)`."""
    low, high = value_range
    center = (low + high) / 2.0
    half_width = (high - low) / 2.0
    return (center - half_width * scale, center + half_width * scale)
