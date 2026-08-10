"""A from-scratch (1+1)-evolution-strategy local search: perturb the current control curve
with decaying-magnitude Gaussian noise, keep the perturbation only if it improves the
score. Simple, transparent, and enough for a ~5-8 dimensional bounded problem like this one
-- a full optimization library would be overkill for the actual dimensionality here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import evaluate_policy


@dataclass(frozen=True)
class OptimizeResult:
    control_points: np.ndarray
    score: float
    history: list[float]  # best score after each iteration, for a convergence plot


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    power_weight: float
    noise_weight: float
    control_points: np.ndarray
    evaluation: dict


def optimize_policy(
    temp_breakpoints: np.ndarray,
    heat_w: np.ndarray,
    init: np.ndarray,
    iterations: int = 500,
    seed: int = 0,
    initial_step: float = 0.3,
    step_decay: float = 0.995,
    power_weight: float = 1.0,
    noise_weight: float = 1.0,
    sensor_noise_std: float = 0.0,
    noise_trials_per_eval: int = 5,
) -> OptimizeResult:
    """sensor_noise_std > 0 makes this a *robust* optimization: each candidate is scored
    as the mean over `noise_trials_per_eval` independent noisy-sensor rollouts instead of
    one noiseless rollout. A candidate that only looks good because it hugs the safety
    limit under perfect feedback will, on average, cross the limit on some of those
    rollouts and pick up the safety penalty -- so the search is naturally pushed away from
    the wall, without any explicit margin term in the objective.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1_000_000) if sensor_noise_std > 0 else None
    n_points = len(init)

    def score_of(control_points: np.ndarray) -> float:
        if sensor_noise_std <= 0:
            return evaluate_policy(control_points, temp_breakpoints, heat_w, power_weight, noise_weight)["score"]
        trial_scores = [
            evaluate_policy(
                control_points, temp_breakpoints, heat_w, power_weight, noise_weight,
                sensor_noise_std=sensor_noise_std, rng=noise_rng,
            )["score"]
            for _ in range(noise_trials_per_eval)
        ]
        return float(np.mean(trial_scores))

    current = np.array(init, dtype=float)
    current_score = score_of(current)
    best = current.copy()
    best_score = current_score
    history = [best_score]

    step = initial_step
    for _ in range(iterations):
        candidate = np.clip(current + rng.normal(0, step, size=n_points), 0.0, 1.0)
        cand_score = score_of(candidate)
        if cand_score < current_score:
            current, current_score = candidate, cand_score
            if current_score < best_score:
                best, best_score = current.copy(), current_score
        step *= step_decay
        history.append(best_score)

    return OptimizeResult(control_points=best, score=best_score, history=history)


def optimize_tradeoff_sweep(
    temp_breakpoints: np.ndarray,
    heat_w: np.ndarray,
    init: np.ndarray,
    weights: list[tuple[float, float]],
    iterations: int = 500,
    seed: int = 0,
) -> list[ParetoPoint]:
    """Optimize the same curve under several power/noise weightings."""
    points = []
    for idx, (power_weight, noise_weight) in enumerate(weights):
        result = optimize_policy(
            temp_breakpoints,
            heat_w,
            init=init,
            iterations=iterations,
            seed=seed + idx,
            power_weight=power_weight,
            noise_weight=noise_weight,
        )
        evaluation = evaluate_policy(
            result.control_points,
            temp_breakpoints,
            heat_w,
            power_weight=power_weight,
            noise_weight=noise_weight,
        )
        label = f"power={power_weight:g},noise={noise_weight:g}"
        points.append(
            ParetoPoint(
                label=label,
                power_weight=power_weight,
                noise_weight=noise_weight,
                control_points=result.control_points,
                evaluation=evaluation,
            )
        )
    return points


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Return safe points not dominated on power, noise, and max temperature."""
    safe_points = [point for point in points if not point.evaluation["safety_violated"]]
    frontier = []
    for point in safe_points:
        power = point.evaluation["mean_power_w"]
        noise = point.evaluation["mean_noise_db"]
        temp = point.evaluation["max_temp_c"]
        dominated = False
        for other in safe_points:
            if other is point:
                continue
            other_power = other.evaluation["mean_power_w"]
            other_noise = other.evaluation["mean_noise_db"]
            other_temp = other.evaluation["max_temp_c"]
            no_worse = other_power <= power and other_noise <= noise and other_temp <= temp
            strictly_better = other_power < power or other_noise < noise or other_temp < temp
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda item: item.evaluation["mean_power_w"])
