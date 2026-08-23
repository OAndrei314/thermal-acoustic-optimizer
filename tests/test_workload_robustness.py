import numpy as np
import pytest

from thermal_acoustic.policies import always_on_policy, linear_ramp_policy
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.workload_robustness import evaluate_workload_robustness


def test_evaluate_workload_robustness_rejects_non_positive_trials():
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    with pytest.raises(ValueError):
        evaluate_workload_robustness(always_on_policy(6), temp_breakpoints, n_trials=0)


def test_evaluate_workload_robustness_is_reproducible_and_well_formed():
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    cp = linear_ramp_policy(6)

    rob_a = evaluate_workload_robustness(cp, temp_breakpoints, n_trials=50, seed=7)
    rob_b = evaluate_workload_robustness(cp, temp_breakpoints, n_trials=50, seed=7)
    assert rob_a == rob_b
    assert 0.0 <= rob_a["safety_violation_rate"] <= 1.0
    assert rob_a["worst_max_temp_c"] >= rob_a["mean_max_temp_c"]


def test_always_on_policy_is_far_more_robust_than_a_wall_hugging_policy():
    """always_on runs the fan at full speed no matter what -- it's the maximally
    conservative baseline. It isn't perfectly safe under the sampled workload
    distribution (an extreme burst-overlap draw can still exceed even full-fan
    cooling -- see K_COOL's docstring in simulate.py), but its violation rate
    should be a small fraction of a policy tuned to hug the safety wall on one
    fixed trace (see test_optimize.py's fixed-trace-vs-workload-distribution test
    for that comparison, which shows ~87%)."""
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    rob = evaluate_workload_robustness(always_on_policy(6), temp_breakpoints, n_trials=300, seed=0)
    assert rob["safety_violation_rate"] < 0.05
