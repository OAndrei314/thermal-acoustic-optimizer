"""Monte-Carlo evaluation of a *fixed* control policy under sensor measurement noise.

This answers a different question than the noiseless objective in objective.py. The
objective asks "what's the best achievable power/noise score for this workload." This
module asks "how often does this policy actually stay under the safety limit once the
temperature sensor the controller reads is imperfect" -- which is the question that
matters before shipping a control curve that was tuned assuming perfect feedback.
"""
from __future__ import annotations

import numpy as np

from .simulate import simulate_policy


def evaluate_robustness(
    control_points: np.ndarray,
    temp_breakpoints: np.ndarray,
    heat_w: np.ndarray,
    sensor_noise_std: float,
    n_trials: int = 200,
    seed: int = 0,
) -> dict:
    if sensor_noise_std <= 0:
        raise ValueError("evaluate_robustness needs sensor_noise_std > 0; use evaluate_policy for the noiseless case")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(seed)
    max_temps = np.empty(n_trials)
    powers = np.empty(n_trials)
    noises = np.empty(n_trials)
    violations = 0

    for i in range(n_trials):
        result = simulate_policy(
            control_points, temp_breakpoints, heat_w,
            sensor_noise_std=sensor_noise_std, rng=rng,
        )
        max_temps[i] = result.max_temp
        powers[i] = float(np.mean(result.powers))
        noises[i] = float(np.mean(result.noises_db))
        if result.safety_violated:
            violations += 1

    return {
        "n_trials": n_trials,
        "sensor_noise_std": sensor_noise_std,
        "safety_violation_rate": violations / n_trials,
        "mean_max_temp_c": float(np.mean(max_temps)),
        "worst_max_temp_c": float(np.max(max_temps)),
        "mean_power_w": float(np.mean(powers)),
        "mean_noise_db": float(np.mean(noises)),
    }
