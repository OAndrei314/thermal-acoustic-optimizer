import numpy as np

from thermal_acoustic.objective import evaluate_policy
from thermal_acoustic.optimize import optimize_policy, optimize_tradeoff_sweep, pareto_frontier
from thermal_acoustic.policies import always_on_policy, linear_ramp_policy
from thermal_acoustic.report import render_markdown_report
from thermal_acoustic.robustness import evaluate_robustness
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.workload import heat_trace
from thermal_acoustic.joint_robustness import evaluate_joint_robustness
from thermal_acoustic.uncertainty_sweep import UncertaintyScalePoint
from thermal_acoustic.workload_robustness import evaluate_workload_robustness


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


def test_report_includes_workload_distribution_section_when_provided():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    optimized = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=100, seed=0,
    )
    evaluations = {"optimized": evaluate_policy(optimized.control_points, temp_breakpoints, heat_w)}
    workload_robustness = {
        "optimized": evaluate_workload_robustness(
            optimized.control_points, temp_breakpoints, n_trials=30, seed=0,
        )
    }

    report = render_markdown_report(evaluations, workload_robustness=workload_robustness)

    assert "Workload-Distribution Robustness" in report
    assert "sample_heat_trace" in report


def test_report_includes_joint_robustness_section_when_provided():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    optimized = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=100, seed=0,
    )
    evaluations = {"optimized": evaluate_policy(optimized.control_points, temp_breakpoints, heat_w)}
    joint_robustness = {
        "optimized": evaluate_joint_robustness(
            optimized.control_points, temp_breakpoints,
            sensor_noise_std=1.5, n_trials=30, seed=0,
        )
    }

    report = render_markdown_report(evaluations, joint_robustness=joint_robustness)

    assert "Joint Sensor+Workload Robustness" in report
    assert "violation rate" in report


def test_report_includes_uncertainty_scale_sweep_section_when_provided():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    optimized = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=100, seed=0,
    )
    evaluations = {"optimized": evaluate_policy(optimized.control_points, temp_breakpoints, heat_w)}
    uncertainty_scale_points = [
        UncertaintyScalePoint(
            scale=0.5,
            sensor_noise_std=0.75,
            sensor_robust_violation_rate=0.30,
            workload_robust_violation_rate=0.20,
            jointly_robust_violation_rate=0.02,
        ),
        UncertaintyScalePoint(
            scale=1.0,
            sensor_noise_std=1.5,
            sensor_robust_violation_rate=0.70,
            workload_robust_violation_rate=0.46,
            jointly_robust_violation_rate=0.06,
        ),
    ]

    report = render_markdown_report(evaluations, uncertainty_scale_points=uncertainty_scale_points)

    assert "Uncertainty-Scale Sweep" in report
    assert "transfer gap" in report
    assert "40.0%" in report  # scale=1.0's transfer gap: 0.46 - 0.06
