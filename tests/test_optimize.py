import numpy as np

from thermal_acoustic.objective import evaluate_policy
from thermal_acoustic.optimize import optimize_policy, optimize_tradeoff_sweep, pareto_frontier
from thermal_acoustic.policies import linear_ramp_policy
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.workload import heat_trace


def test_optimizer_beats_its_own_starting_point_and_stays_safe():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)
    baseline_score = evaluate_policy(baseline, temp_breakpoints, heat_w)["score"]

    result = optimize_policy(temp_breakpoints, heat_w, init=baseline, iterations=500, seed=0)
    final = evaluate_policy(result.control_points, temp_breakpoints, heat_w)

    assert final["score"] < baseline_score
    assert not final["safety_violated"]
    assert result.score == final["score"]


def test_optimizer_history_is_monotonically_non_increasing():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    result = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=200, seed=1
    )
    assert all(a >= b for a, b in zip(result.history, result.history[1:]))


def test_tradeoff_sweep_returns_safe_frontier_points():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    points = optimize_tradeoff_sweep(
        temp_breakpoints,
        heat_w,
        init=linear_ramp_policy(6),
        weights=[(1.0, 0.5), (1.0, 1.0), (1.0, 2.0)],
        iterations=150,
        seed=2,
    )
    frontier = pareto_frontier(points)

    assert len(points) == 3
    assert frontier
    assert all(not point.evaluation["safety_violated"] for point in frontier)
