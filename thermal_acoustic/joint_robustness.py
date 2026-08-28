"""Monte-Carlo evaluation of a *fixed* control policy under sensor noise and workload
uncertainty *together*, in the same trial.

`robustness.py` and `workload_robustness.py` each Monte-Carlo one source of
uncertainty at a time, holding the other one fixed (the true workload in the sensor
case, a perfect sensor in the workload case). Real deployments don't get to pick
one axis of uncertainty to worry about -- the sensor is noisy *and* the workload
varies, at the same time, on the same run. This module answers the question the
other two can't: does a policy that was made robust to one axis alone stay robust
once both are active simultaneously, or does compounding the two failure modes
erode the safety margin faster than either one predicts on its own.
"""
from __future__ import annotations

import numpy as np

from .simulate import simulate_policy
from .workload import sample_heat_trace


def evaluate_joint_robustness(
    control_points: np.ndarray,
    temp_breakpoints: np.ndarray,
    sensor_noise_std: float,
    n_trials: int = 200,
    seed: int = 0,
) -> dict:
    """Each trial draws a fresh random workload trace (`workload.sample_heat_trace`)
    *and* runs it with additive Gaussian sensor read noise, so both sources of
    uncertainty are active on every single rollout rather than being evaluated
    independently."""
    if sensor_noise_std <= 0:
        raise ValueError("evaluate_joint_robustness needs sensor_noise_std > 0; use "
                          "evaluate_workload_robustness for the perfect-sensor case")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(seed)
    max_temps = np.empty(n_trials)
    powers = np.empty(n_trials)
    noises = np.empty(n_trials)
    violations = 0

    for i in range(n_trials):
        heat_w = sample_heat_trace(rng)
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
