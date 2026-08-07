import numpy as np

from thermal_acoustic.policies import always_on_policy
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C, simulate_policy
from thermal_acoustic.workload import heat_trace


def test_simulate_produces_correct_length_arrays():
    heat_w = heat_trace(n_steps=50)
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    result = simulate_policy(always_on_policy(6), temp_breakpoints, heat_w)
    assert len(result.temps) == 50
    assert len(result.fan_speeds) == 50
    assert len(result.powers) == 50
    assert len(result.noises_db) == 50


def test_zero_fan_speed_violates_safety_on_default_workload():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    result = simulate_policy(np.zeros(6), temp_breakpoints, heat_w)
    assert result.safety_violated
    assert result.max_temp > T_SAFETY_MAX_C


def test_full_fan_speed_keeps_default_workload_safe():
    heat_w = heat_trace()
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    result = simulate_policy(always_on_policy(6), temp_breakpoints, heat_w)
    assert not result.safety_violated
    assert result.max_temp < T_SAFETY_MAX_C
