import numpy as np

from thermal_acoustic.objective import evaluate_policy
from thermal_acoustic.optimize import optimize_policy, optimize_tradeoff_sweep, pareto_frontier
from thermal_acoustic.policies import always_on_policy, linear_ramp_policy
from thermal_acoustic.report import render_markdown_report
from thermal_acoustic.robustness import evaluate_robustness
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.workload import heat_trace


def test_report_includes_pareto_sweep_when_provided():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baselines = {
        "always_on": always_on_policy(6),
        "linear_ramp": linear_ramp_policy(6),
    }
    evaluations = {
        name: evaluate_policy(cp, temp_breakpoints, heat_w) for name, cp in baselines.items()
    }
    optimized = optimize_policy(
        temp_breakpoints,
        heat_w,
        init=baselines["linear_ramp"],
        iterations=100,
        seed=0,
    )
    evaluations["optimized"] = evaluate_policy(optimized.control_points, temp_breakpoints, heat_w)
    points = optimize_tradeoff_sweep(
        temp_breakpoints,
        heat_w,
        init=baselines["linear_ramp"],
        weights=[(1.0, 0.5), (1.0, 1.0)],
        iterations=100,
        seed=0,
    )

    report = render_markdown_report(evaluations, points, pareto_frontier(points))

    assert "Efficiency / Thermal-Margin Tradeoff Sweep" in report
    assert "power=1,noise=0.5" in report
    assert "frontier" in report
    assert "Money question" not in report


def test_report_includes_robustness_section_when_provided():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    optimized = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=100, seed=0,
    )
    evaluations = {"optimized": evaluate_policy(optimized.control_points, temp_breakpoints, heat_w)}
    robustness = {
        "optimized": evaluate_robustness(
            optimized.control_points, temp_breakpoints, heat_w,
            sensor_noise_std=1.5, n_trials=30, seed=0,
        )
    }

    report = render_markdown_report(evaluations, robustness=robustness)

    assert "Sensor-Noise Robustness" in report
    assert "violation rate" in report
