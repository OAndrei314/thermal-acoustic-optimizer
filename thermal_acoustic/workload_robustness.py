"""Monte-Carlo evaluation of a *fixed* control policy against a distribution of
workload traces, instead of the single fixed trace in `workload.heat_trace()`.

`robustness.py` asks "how often does this policy stay safe when the *sensor* the
controller reads is noisy." This module asks a different question that matters just
as much before shipping a control curve: "how often does this policy stay safe when
the *workload itself* -- burst timing, duration, and magnitude -- varies around the
one nominal trace it may have been tuned against."
"""
from __future__ import annotations

import numpy as np

from .objective import evaluate_policy
from .workload import sample_heat_trace


def evaluate_workload_robustness(
    control_points: np.ndarray,
    temp_breakpoints: np.ndarray,
    n_trials: int = 200,
    seed: int = 0,
    jitter_scale: float = 1.0,
) -> dict:
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(seed)
    max_temps = np.empty(n_trials)
    powers = np.empty(n_trials)
    noises = np.empty(n_trials)
    violations = 0

    for i in range(n_trials):
        heat_w = sample_heat_trace(rng, jitter_scale=jitter_scale)
        ev = evaluate_policy(control_points, temp_breakpoints, heat_w)
        max_temps[i] = ev["max_temp_c"]
        powers[i] = ev["mean_power_w"]
        noises[i] = ev["mean_noise_db"]
        if ev["safety_violated"]:
            violations += 1

    return {
        "n_trials": n_trials,
        "safety_violation_rate": violations / n_trials,
        "mean_max_temp_c": float(np.mean(max_temps)),
        "worst_max_temp_c": float(np.max(max_temps)),
        "mean_power_w": float(np.mean(powers)),
        "mean_noise_db": float(np.mean(noises)),
    }
