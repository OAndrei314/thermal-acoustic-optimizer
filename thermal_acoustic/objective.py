"""Turns a simulated policy run into a single scalar score to minimize: mean power +
mean acoustic noise, with a large penalty if the safety temperature limit is ever exceeded
(scaled by how badly it was exceeded, so "barely over" and "way over" aren't equally bad)."""
from __future__ import annotations

import numpy as np

from .simulate import T_SAFETY_MAX_C, simulate_policy

SAFETY_PENALTY_BASE = 1000.0
SAFETY_PENALTY_PER_DEGREE = 20.0


def evaluate_policy(
    control_points: np.ndarray,
    temp_breakpoints: np.ndarray,
    heat_w: np.ndarray,
    power_weight: float = 1.0,
    noise_weight: float = 1.0,
    sensor_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    result = simulate_policy(
        control_points, temp_breakpoints, heat_w,
        sensor_noise_std=sensor_noise_std, rng=rng,
    )
    mean_power = float(np.mean(result.powers))
    mean_noise = float(np.mean(result.noises_db))

    score = power_weight * mean_power + noise_weight * mean_noise
    if result.safety_violated:
        overshoot = result.max_temp - T_SAFETY_MAX_C
        score += SAFETY_PENALTY_BASE + SAFETY_PENALTY_PER_DEGREE * overshoot

    return {
        "mean_power_w": mean_power,
        "mean_noise_db": mean_noise,
        "max_temp_c": result.max_temp,
        "safety_violated": result.safety_violated,
        "score": score,
    }
