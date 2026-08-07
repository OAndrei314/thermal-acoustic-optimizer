"""Two baseline control curves to compare the optimized one against -- both are realistic
starting points a real engineer would reach for before doing any optimization."""
from __future__ import annotations

import numpy as np


def always_on_policy(n_points: int) -> np.ndarray:
    """The overly-conservative baseline: fan at full speed regardless of temperature.
    Maximally safe, presumably wastes power and is loud."""
    return np.ones(n_points)


def linear_ramp_policy(n_points: int) -> np.ndarray:
    """The naive first-guess baseline: fan speed ramps linearly from low to high across
    the temperature range."""
    return np.linspace(0.15, 1.0, n_points)
