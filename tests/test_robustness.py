import numpy as np
import pytest

from thermal_acoustic.objective import evaluate_policy
from thermal_acoustic.optimize import optimize_policy
from thermal_acoustic.policies import always_on_policy, linear_ramp_policy
from thermal_acoustic.robustness import evaluate_robustness
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C, simulate_policy
from thermal_acoustic.workload import heat_trace


def test_simulate_policy_requires_rng_for_sensor_noise():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    with pytest.raises(ValueError):
        simulate_policy(always_on_policy(6), temp_breakpoints, heat_w, sensor_noise_std=1.0)


def test_zero_sensor_noise_matches_noiseless_simulation_exactly():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    cp = linear_ramp_policy(6)

    plain = simulate_policy(cp, temp_breakpoints, heat_w)
    noisy_but_zero = simulate_policy(
        cp, temp_breakpoints, heat_w, sensor_noise_std=0.0, rng=np.random.default_rng(0)
    )
    assert np.array_equal(plain.temps, noisy_but_zero.temps)


def test_sensor_noise_is_reproducible_given_the_same_rng_state():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    cp = linear_ramp_policy(6)

    result_a = simulate_policy(
        cp, temp_breakpoints, heat_w, sensor_noise_std=2.0, rng=np.random.default_rng(42)
    )
    result_b = simulate_policy(
        cp, temp_breakpoints, heat_w, sensor_noise_std=2.0, rng=np.random.default_rng(42)
    )
    assert np.array_equal(result_a.temps, result_b.temps)


def test_evaluate_robustness_rejects_non_positive_noise_or_trials():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    cp = always_on_policy(6)
    with pytest.raises(ValueError):
        evaluate_robustness(cp, temp_breakpoints, heat_w, sensor_noise_std=0.0)
    with pytest.raises(ValueError):
        evaluate_robustness(cp, temp_breakpoints, heat_w, sensor_noise_std=1.0, n_trials=0)


def test_evaluate_robustness_is_reproducible_and_well_formed():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    cp = linear_ramp_policy(6)

    rob_a = evaluate_robustness(cp, temp_breakpoints, heat_w, sensor_noise_std=1.5, n_trials=50, seed=7)
    rob_b = evaluate_robustness(cp, temp_breakpoints, heat_w, sensor_noise_std=1.5, n_trials=50, seed=7)
    assert rob_a == rob_b
    assert 0.0 <= rob_a["safety_violation_rate"] <= 1.0
    assert rob_a["worst_max_temp_c"] >= rob_a["mean_max_temp_c"]


def test_policy_optimized_right_at_the_safety_wall_is_fragile_under_sensor_noise():
    """The noiseless optimizer is documented to converge right up against the safety
    limit. This test checks that this is a real fragility, not just a documentation claim:
    under realistic sensor noise, that same policy should violate safety a meaningful
    fraction of the time."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    noiseless = optimize_policy(temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=500, seed=0)
    noiseless_eval = evaluate_policy(noiseless.control_points, temp_breakpoints, heat_w)
    assert not noiseless_eval["safety_violated"]
    assert noiseless_eval["max_temp_c"] > T_SAFETY_MAX_C - 1.0  # hugging the wall

    rob = evaluate_robustness(
        noiseless.control_points, temp_breakpoints, heat_w,
        sensor_noise_std=1.5, n_trials=300, seed=0,
    )
    assert rob["safety_violation_rate"] > 0.1


def test_robust_optimization_reduces_safety_violation_rate_under_the_same_noise():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    sensor_noise_std = 1.5

    noiseless = optimize_policy(temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=500, seed=0)
    robust = optimize_policy(
        temp_breakpoints, heat_w, init=linear_ramp_policy(6), iterations=500, seed=0,
        sensor_noise_std=sensor_noise_std,
    )

    noiseless_rob = evaluate_robustness(
        noiseless.control_points, temp_breakpoints, heat_w,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=1,
    )
    robust_rob = evaluate_robustness(
        robust.control_points, temp_breakpoints, heat_w,
        sensor_noise_std=sensor_noise_std, n_trials=300, seed=1,
    )

    assert robust_rob["safety_violation_rate"] < noiseless_rob["safety_violation_rate"]
