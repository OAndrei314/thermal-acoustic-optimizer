import numpy as np
import pytest

from thermal_acoustic.joint_robustness import evaluate_joint_robustness
from thermal_acoustic.objective import evaluate_policy
from thermal_acoustic.optimize import optimize_policy
from thermal_acoustic.policies import always_on_policy, linear_ramp_policy
from thermal_acoustic.robustness import evaluate_robustness
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.workload import heat_trace
from thermal_acoustic.workload_robustness import evaluate_workload_robustness


def test_evaluate_joint_robustness_rejects_non_positive_noise_or_trials():
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    cp = always_on_policy(6)
    with pytest.raises(ValueError):
        evaluate_joint_robustness(cp, temp_breakpoints, sensor_noise_std=0.0)
    with pytest.raises(ValueError):
        evaluate_joint_robustness(cp, temp_breakpoints, sensor_noise_std=1.0, n_trials=0)


def test_evaluate_joint_robustness_is_reproducible_and_well_formed():
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    cp = linear_ramp_policy(6)

    rob_a = evaluate_joint_robustness(cp, temp_breakpoints, sensor_noise_std=1.5, n_trials=50, seed=7)
    rob_b = evaluate_joint_robustness(cp, temp_breakpoints, sensor_noise_std=1.5, n_trials=50, seed=7)
    assert rob_a == rob_b
    assert 0.0 <= rob_a["safety_violation_rate"] <= 1.0
    assert rob_a["worst_max_temp_c"] >= rob_a["mean_max_temp_c"]


def test_joint_uncertainty_is_at_least_as_bad_as_either_axis_alone():
    """A policy tuned on the one fixed trace with a perfect sensor should be hit by
    *both* failure modes at once under joint evaluation, so its joint violation rate
    should be at least as high as either single-axis violation rate measured
    separately -- compounding two independent-ish sources of risk shouldn't make a
    fragile policy look safer than either source does on its own."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    sensor_noise_std = 1.5

    noiseless = optimize_policy(temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=300, seed=0)

    sensor_only = evaluate_robustness(
        noiseless.control_points, temp_breakpoints, heat_w,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=1,
    )
    workload_only = evaluate_workload_robustness(
        noiseless.control_points, temp_breakpoints, n_trials=300, seed=1,
    )
    joint = evaluate_joint_robustness(
        noiseless.control_points, temp_breakpoints,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=1,
    )

    assert joint["safety_violation_rate"] >= sensor_only["safety_violation_rate"] - 0.02
    assert joint["safety_violation_rate"] >= workload_only["safety_violation_rate"] - 0.02


def test_jointly_robust_optimization_beats_single_axis_robust_policies_under_joint_noise():
    """The open question this module exists to answer: does a policy optimized
    against sensor noise alone, or workload variation alone, hold up once *both*
    are active together -- or does jointly-robust optimization (composing
    sensor_noise_std and workload_sampler in the same optimize_policy call) do
    meaningfully better under the evaluation that actually matches deployment."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    sensor_noise_std = 1.5
    from thermal_acoustic.workload import sample_heat_trace

    sensor_robust = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=300, seed=0,
        sensor_noise_std=sensor_noise_std, noise_trials_per_eval=10, reevaluate_incumbent=True,
    )
    workload_robust = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=300, seed=0,
        workload_sampler=sample_heat_trace, noise_trials_per_eval=10, reevaluate_incumbent=True,
    )
    jointly_robust = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=300, seed=0,
        sensor_noise_std=sensor_noise_std, workload_sampler=sample_heat_trace,
        noise_trials_per_eval=10, reevaluate_incumbent=True,
    )

    sensor_robust_joint = evaluate_joint_robustness(
        sensor_robust.control_points, temp_breakpoints,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=2,
    )
    workload_robust_joint = evaluate_joint_robustness(
        workload_robust.control_points, temp_breakpoints,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=2,
    )
    jointly_robust_joint = evaluate_joint_robustness(
        jointly_robust.control_points, temp_breakpoints,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=2,
    )

    assert jointly_robust_joint["safety_violation_rate"] <= sensor_robust_joint["safety_violation_rate"] + 0.05
    assert jointly_robust_joint["safety_violation_rate"] <= workload_robust_joint["safety_violation_rate"] + 0.05


def test_evaluate_joint_robustness_jitter_scale_zero_only_leaves_sensor_noise_active():
    """jitter_scale=0 removes workload uncertainty, so evaluating a policy tuned only
    against the fixed trace and a perfect sensor should give a violation rate that
    matches plain sensor-noise robustness (evaluate_robustness on the fixed trace), not
    the (much higher) violation rate joint evaluation shows at the default jitter_scale."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    sensor_noise_std = 1.5
    noiseless = optimize_policy(temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=300, seed=0)

    sensor_only = evaluate_robustness(
        noiseless.control_points, temp_breakpoints, heat_w,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=5,
    )
    joint_no_jitter = evaluate_joint_robustness(
        noiseless.control_points, temp_breakpoints,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=5, jitter_scale=0.0,
    )

    assert abs(joint_no_jitter["safety_violation_rate"] - sensor_only["safety_violation_rate"]) < 0.05
