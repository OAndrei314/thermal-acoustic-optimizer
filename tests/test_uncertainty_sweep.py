import numpy as np
import pytest

from thermal_acoustic.policies import linear_ramp_policy
from thermal_acoustic.simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from thermal_acoustic.uncertainty_sweep import UncertaintyScalePoint, sweep_uncertainty_scale
from thermal_acoustic.workload import heat_trace


def test_sweep_uncertainty_scale_rejects_non_positive_scales():
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    with pytest.raises(ValueError):
        sweep_uncertainty_scale(
            temp_breakpoints, heat_trace(), init=linear_ramp_policy(6), scales=[1.0, 0.0],
        )


def test_sweep_uncertainty_scale_returns_one_point_per_scale_with_scaled_sensor_std():
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    points = sweep_uncertainty_scale(
        temp_breakpoints, heat_trace(), init=linear_ramp_policy(6),
        scales=[0.5, 1.0],
        base_sensor_noise_std=1.5,
        iterations=30,
        trials_per_eval=3,
        eval_trials=20,
        seed=0,
    )

    assert len(points) == 2
    assert all(isinstance(point, UncertaintyScalePoint) for point in points)
    assert points[0].sensor_noise_std == pytest.approx(0.75)
    assert points[1].sensor_noise_std == pytest.approx(1.5)
    for point in points:
        assert 0.0 <= point.sensor_robust_violation_rate <= 1.0
        assert 0.0 <= point.workload_robust_violation_rate <= 1.0
        assert 0.0 <= point.jointly_robust_violation_rate <= 1.0


def test_sweep_uncertainty_scale_is_reproducible():
    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, 6)
    kwargs = dict(
        temp_breakpoints=temp_breakpoints,
        heat_w=heat_trace(),
        init=linear_ramp_policy(6),
        scales=[1.0],
        base_sensor_noise_std=1.5,
        iterations=30,
        trials_per_eval=3,
        eval_trials=20,
        seed=3,
    )
    points_a = sweep_uncertainty_scale(**kwargs)
    points_b = sweep_uncertainty_scale(**kwargs)
    assert points_a == points_b


def test_transfer_gap_and_best_single_axis_properties():
    point = UncertaintyScalePoint(
        scale=1.0,
        sensor_noise_std=1.5,
        sensor_robust_violation_rate=0.70,
        workload_robust_violation_rate=0.46,
        jointly_robust_violation_rate=0.06,
    )
    assert point.best_single_axis_violation_rate == pytest.approx(0.46)
    assert point.transfer_gap == pytest.approx(0.40)
