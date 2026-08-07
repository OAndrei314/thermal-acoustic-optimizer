"""A fixed, deterministic synthetic heat-generation trace (Watts) representing a bursty
compute workload -- idle baseline with a few burst windows layered in, similar in shape to
what a small embedded/telecom module sees under intermittent traffic or duty-cycled work.
"""
from __future__ import annotations

import numpy as np


def heat_trace(n_steps: int = 120, base_w: float = 15.0, burst_w: float = 35.0) -> np.ndarray:
    t = np.arange(n_steps)
    trace = np.full(n_steps, base_w, dtype=float)
    bursts = [(15, 30), (45, 70), (90, 110)]  # (start, end) step indices, inclusive-exclusive
    for start, end in bursts:
        end = min(end, n_steps)
        if start < n_steps:
            trace[start:end] += burst_w
    return trace
