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
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    n_points = len(init)

    current = np.array(init, dtype=float)
    current_score = evaluate_policy(current, temp_breakpoints, heat_w, power_weight, noise_weight)["score"]
    best = current.copy()
    best_score = current_score
    history = [best_score]

    step = initial_step
    for _ in range(iterations):
        candidate = np.clip(current + rng.normal(0, step, size=n_points), 0.0, 1.0)
        cand_score = evaluate_policy(candidate, temp_breakpoints, heat_w, power_weight, noise_weight)["score"]
        if cand_score < current_score:
            current, current_score = candidate, cand_score
            if current_score < best_score:
                best, best_score = current.copy(), current_score
        step *= step_decay
        history.append(best_score)

    return OptimizeResult(control_points=best, score=best_score, history=history)
