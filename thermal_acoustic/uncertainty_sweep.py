"""Does the single-axis-robustness-transfer gap found in `joint_robustness.py` narrow or
widen as the uncertainty sources themselves scale up or down?

The joint-robustness result (see README) was only ever measured at one sensor-noise
magnitude (1.5 degC std) and one workload-jitter magnitude (`sample_heat_trace`'s
defaults) -- so it answers "at this specific amount of uncertainty, single-axis robustness
doesn't transfer" but not whether that gap is a generic property of compounding two
failure modes, or an artifact of those two particular magnitudes happening to interact
badly. This module re-runs the same three-way comparison (sensor-robust vs
workload-robust vs jointly-robust, evaluated under joint noise) across a range of
uncertainty scales, using `jitter_scale` (see `workload.sample_heat_trace`) to scale
workload uncertainty and a plain multiplier on `sensor_noise_std` to scale sensor
uncertainty, moved together so "uncertainty scale" means the same relative thing on both
axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from .joint_robustness import evaluate_joint_robustness
from .optimize import optimize_policy
from .workload import sample_heat_trace


@dataclass(frozen=True)
class UncertaintyScalePoint:
    scale: float
    sensor_noise_std: float
    sensor_robust_violation_rate: float
    workload_robust_violation_rate: float
    jointly_robust_violation_rate: float

    @property
    def best_single_axis_violation_rate(self) -> float:
        return min(self.sensor_robust_violation_rate, self.workload_robust_violation_rate)

    @property
    def transfer_gap(self) -> float:
        """How much worse the best single-axis-robust policy is than the jointly-robust
        one, under joint noise at this scale. Zero would mean single-axis robustness
        transfers perfectly; the joint-robustness README result found this to be large
        (46-70 percentage points) at scale=1.0."""
        return self.best_single_axis_violation_rate - self.jointly_robust_violation_rate


def sweep_uncertainty_scale(
    temp_breakpoints: np.ndarray,
    heat_w: np.ndarray,
    init: np.ndarray,
    scales: list[float],
    base_sensor_noise_std: float = 1.5,
    iterations: int = 500,
    trials_per_eval: int = 20,
    eval_trials: int = 300,
    seed: int = 0,
) -> list[UncertaintyScalePoint]:
    """For each scale in `scales`, sets sensor_noise_std = base_sensor_noise_std * scale
    and workload jitter_scale = scale, trains a sensor-only-robust, a workload-only-robust,
    and a jointly-robust policy at that magnitude, evaluates all three under joint
    sensor+workload noise at that same magnitude, and reports each policy's violation rate
    plus the resulting single-axis-transfer gap. `scale=0.0` is invalid (there would be no
    uncertainty to be robust to); use a small positive value instead.
    """
    if any(scale <= 0 for scale in scales):
        raise ValueError("uncertainty scales must be > 0 (no uncertainty to train against at scale=0)")

    points = []
    for scale in scales:
        sensor_noise_std = base_sensor_noise_std * scale
        sampler = partial(sample_heat_trace, jitter_scale=scale)

        sensor_result = optimize_policy(
            temp_breakpoints, heat_w, init=init,
            iterations=iterations, seed=seed,
            sensor_noise_std=sensor_noise_std,
            noise_trials_per_eval=trials_per_eval,
            reevaluate_incumbent=True,
        )
        workload_result = optimize_policy(
            temp_breakpoints, heat_w, init=init,
            iterations=iterations, seed=seed,
            workload_sampler=sampler,
            noise_trials_per_eval=trials_per_eval,
            reevaluate_incumbent=True,
        )
        joint_result = optimize_policy(
            temp_breakpoints, heat_w, init=init,
            iterations=iterations, seed=seed,
            sensor_noise_std=sensor_noise_std,
            workload_sampler=sampler,
            noise_trials_per_eval=trials_per_eval,
            reevaluate_incumbent=True,
        )

        def joint_violation_rate(control_points: np.ndarray) -> float:
            evaluation = evaluate_joint_robustness(
                control_points, temp_breakpoints,
                sensor_noise_std=sensor_noise_std, n_trials=eval_trials, seed=seed,
                jitter_scale=scale,
            )
            return evaluation["safety_violation_rate"]

        points.append(
            UncertaintyScalePoint(
                scale=scale,
                sensor_noise_std=sensor_noise_std,
                sensor_robust_violation_rate=joint_violation_rate(sensor_result.control_points),
                workload_robust_violation_rate=joint_violation_rate(workload_result.control_points),
                jointly_robust_violation_rate=joint_violation_rate(joint_result.control_points),
            )
        )
    return points
