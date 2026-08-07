import numpy as np

from thermal_acoustic.objective import evaluate_policy
from thermal_acoustic.policies import always_on_policy
from thermal_acoustic.simulate import BASE_DB, K_NOISE_DB, K_POWER_W, T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.workload import heat_trace


def test_always_on_policy_has_exact_max_power_and_noise():
    # At fan_speed=1.0 constantly, power and noise are deterministic constants regardless
    # of the workload -- a useful exact check on the power/noise formulas themselves.
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    ev = evaluate_policy(always_on_policy(6), temp_breakpoints, heat_w)
    assert ev["mean_power_w"] == K_POWER_W
    assert ev["mean_noise_db"] == BASE_DB + K_NOISE_DB


def test_unsafe_policy_score_is_much_higher_than_safe_policy():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    safe = evaluate_policy(always_on_policy(6), temp_breakpoints, heat_w)
    unsafe = evaluate_policy(np.zeros(6), temp_breakpoints, heat_w)
    assert unsafe["safety_violated"]
    assert not safe["safety_violated"]
    assert unsafe["score"] > safe["score"] + 1000  # penalty should dominate
