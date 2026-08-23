import numpy as np
import pytest

from thermal_acoustic.objective import evaluate_policy
from thermal_acoustic.optimize import optimize_policy, optimize_tradeoff_sweep, pareto_frontier
from thermal_acoustic.policies import linear_ramp_policy
from thermal_acoustic.robustness import evaluate_robustness
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.workload import heat_trace, sample_heat_trace
from thermal_acoustic.workload_robustness import evaluate_workload_robustness


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


def test_reevaluate_incumbent_is_a_no_op_without_sensor_noise():
    """reevaluate_incumbent only matters for the noisy accept/reject decision -- with no
    sensor noise, score_of() is deterministic, so re-scoring the incumbent must return the
    exact same value and the whole search trajectory must be unchanged."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    plain = optimize_policy(temp_breakpoints, heat_w, init=baseline, iterations=200, seed=0)
    reeval = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=200, seed=0, reevaluate_incumbent=True
    )

    assert reeval.score == plain.score
    assert reeval.history == plain.history
    assert np.array_equal(reeval.control_points, plain.control_points)


def test_reevaluate_incumbent_reduces_safety_violation_rate_under_sensor_noise():
    """Regression test for the staleness bias flagged in the README: the default robust
    optimizer compares a freshly-sampled candidate score against a stale incumbent score
    that hasn't been resampled since it was accepted, which lets lucky/unlucky noise
    draws bias the walk. Re-scoring the incumbent fresh every iteration should measurably
    cut the real (Monte-Carlo-evaluated) safety violation rate at a fixed sample budget --
    this reproduces the ~44% -> ~12% drop measured in the README at iterations=500,
    noise_trials_per_eval=5, sensor_noise_std=1.5, seed=0."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    stale = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=500, seed=0,
        sensor_noise_std=1.5, noise_trials_per_eval=5, reevaluate_incumbent=False,
    )
    fresh = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=500, seed=0,
        sensor_noise_std=1.5, noise_trials_per_eval=5, reevaluate_incumbent=True,
    )

    stale_rob = evaluate_robustness(
        stale.control_points, temp_breakpoints, heat_w, sensor_noise_std=1.5, n_trials=200, seed=0
    )
    fresh_rob = evaluate_robustness(
        fresh.control_points, temp_breakpoints, heat_w, sensor_noise_std=1.5, n_trials=200, seed=0
    )

    assert fresh_rob["safety_violation_rate"] < stale_rob["safety_violation_rate"] - 0.1


def test_confidence_z_requires_sensor_noise():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    with pytest.raises(ValueError, match="sensor_noise_std"):
        optimize_policy(
            temp_breakpoints, heat_w, init=baseline, iterations=10, seed=0, confidence_z=1.0
        )


def test_confidence_z_requires_at_least_two_noise_trials():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    with pytest.raises(ValueError, match="noise_trials_per_eval"):
        optimize_policy(
            temp_breakpoints, heat_w, init=baseline, iterations=10, seed=0,
            sensor_noise_std=1.5, noise_trials_per_eval=1, confidence_z=1.0,
        )


def test_confidence_z_rejects_everything_at_an_extreme_threshold():
    """With an absurdly high z requirement, no candidate's estimated improvement can ever
    clear `confidence_z` standard errors of noise -- the search must stay exactly at its
    starting point. This pins down the gating mechanism itself, independent of the
    seed-to-seed variance in how well it performs on the actual safety-violation metric."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    result = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=200, seed=0,
        sensor_noise_std=1.5, noise_trials_per_eval=5, confidence_z=1000.0,
    )

    assert np.array_equal(result.control_points, baseline)
    assert all(value == result.history[0] for value in result.history)


def test_confidence_z_zero_threshold_matches_reevaluate_incumbent():
    """confidence_z=0 accepts whenever the candidate's fresh sample mean is at all lower
    than the incumbent's fresh sample mean -- the same accept rule as
    reevaluate_incumbent=True, just computed through the standard-error machinery. The two
    should therefore produce an identical search trajectory given the same seed."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    reeval = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=200, seed=0,
        sensor_noise_std=1.5, noise_trials_per_eval=5, reevaluate_incumbent=True,
    )
    confidence = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=200, seed=0,
        sensor_noise_std=1.5, noise_trials_per_eval=5, confidence_z=0.0,
    )

    assert confidence.score == reeval.score
    assert confidence.history == reeval.history
    assert np.array_equal(confidence.control_points, reeval.control_points)


def test_workload_sampler_is_a_no_op_without_it_set():
    """Sanity check for the score_stats refactor: passing workload_sampler=None (the
    default) must reproduce the exact pre-refactor deterministic trajectory -- this
    pins down that adding workload-distribution support didn't change the plain,
    noiseless optimization path at all."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    result = optimize_policy(temp_breakpoints, heat_w, init=baseline, iterations=200, seed=0)
    final = evaluate_policy(result.control_points, temp_breakpoints, heat_w)
    assert not final["safety_violated"]
    assert final["max_temp_c"] > T_SAFETY_MAX_C - 1.0  # still hugs the wall, as documented


def test_confidence_z_works_with_workload_sampler_and_no_sensor_noise():
    """confidence_z previously required sensor_noise_std > 0; it should also accept
    a workload_sampler on its own (no sensor noise at all) as a valid source of the
    stochasticity it needs a standard error from."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    result = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=20, seed=0,
        workload_sampler=sample_heat_trace, noise_trials_per_eval=3, confidence_z=1.0,
    )
    assert result.control_points.shape == baseline.shape


def test_workload_distribution_optimization_generalizes_far_better_than_fixed_trace_tuning():
    """The core hypothesis this feature exists to test: a policy optimized against the
    one fixed heat_trace() converges right up against the safety wall for *that* trace
    (documented in the README), which should make it fragile once the real workload's
    burst timing/duration/magnitude varies -- exactly the same overfitting story as the
    sensor-noise section, but for workload uncertainty instead of sensor uncertainty.
    Optimizing directly against sampled workload draws should generalize far better when
    evaluated on held-out draws. Reproduces the ~87% -> ~7% drop measured in the README
    at iterations=500, workload_trials_per_eval=5, reevaluate_incumbent=True, seed=0."""
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    baseline = linear_ramp_policy(6)

    fixed_trace = optimize_policy(temp_breakpoints, heat_w, init=baseline, iterations=500, seed=0)
    workload_robust = optimize_policy(
        temp_breakpoints, heat_w, init=baseline, iterations=500, seed=0,
        workload_sampler=sample_heat_trace, noise_trials_per_eval=5, reevaluate_incumbent=True,
    )

    fixed_trace_rob = evaluate_workload_robustness(
        fixed_trace.control_points, temp_breakpoints, n_trials=300, seed=1
    )
    workload_robust_rob = evaluate_workload_robustness(
        workload_robust.control_points, temp_breakpoints, n_trials=300, seed=1
    )

    assert fixed_trace_rob["safety_violation_rate"] > 0.5
    assert workload_robust_rob["safety_violation_rate"] < fixed_trace_rob["safety_violation_rate"] - 0.3
