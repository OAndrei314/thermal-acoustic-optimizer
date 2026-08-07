"""A deliberately simplified thermal/acoustic/power model -- not a reproduction of any real
chassis's thermal design, just physically-plausible enough (heat balance ODE, fan affinity
laws for power, superlinear acoustic scaling with speed) to make curve optimization mean
something.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_AMBIENT_C = 25.0
THERMAL_CAPACITANCE = 8.0  # W*min per degC: how much net imbalance raises T by 1degC/min
# K_COOL is picked so that full fan speed can hold peak workload heat (50W) at ~65degC --
# comfortably under the 85degC safety limit, so the safety constraint is actually
# satisfiable and the optimization problem is meaningful (a K_COOL too low makes every
# policy unsafe regardless of fan curve, which is a modeling bug, not a control problem).
K_COOL = 1.25  # heat-removal coefficient at fan_speed=1.0
COOL_EXPONENT = 1.0  # how heat removal capacity scales with fan speed
K_POWER_W = 8.0  # fan power at full speed (affinity law: power ~ speed^3)
BASE_DB = 25.0  # ambient/idle acoustic floor
K_NOISE_DB = 35.0  # acoustic contribution at full speed
NOISE_EXPONENT = 1.5  # superlinear -- fans get disproportionately loud at high speed
T_SAFETY_MAX_C = 85.0


@dataclass(frozen=True)
class SimResult:
    temps: np.ndarray
    fan_speeds: np.ndarray
    powers: np.ndarray
    noises_db: np.ndarray
    max_temp: float
    safety_violated: bool


def fan_speed_for_temp(control_points: np.ndarray, temp_breakpoints: np.ndarray, t: float) -> float:
    """control_points[i] is the fan speed (0..1) to use at temp_breakpoints[i]; linear
    interpolation between breakpoints, clamped at the ends."""
    speed = np.interp(t, temp_breakpoints, control_points)
    return float(np.clip(speed, 0.0, 1.0))


def simulate_policy(
    control_points: np.ndarray,
    temp_breakpoints: np.ndarray,
    heat_w: np.ndarray,
    dt_min: float = 1.0,
) -> SimResult:
    n = len(heat_w)
    temps = np.zeros(n)
    fan_speeds = np.zeros(n)
    powers = np.zeros(n)
    noises = np.zeros(n)

    T = T_AMBIENT_C
    for i in range(n):
        speed = fan_speed_for_temp(control_points, temp_breakpoints, T)
        heat_removed = K_COOL * (speed**COOL_EXPONENT) * max(T - T_AMBIENT_C, 0.0)
        dT = (heat_w[i] - heat_removed) / THERMAL_CAPACITANCE * dt_min
        T = T + dT

        temps[i] = T
        fan_speeds[i] = speed
        powers[i] = K_POWER_W * speed**3
        noises[i] = BASE_DB + K_NOISE_DB * speed**NOISE_EXPONENT

    max_temp = float(np.max(temps))
    return SimResult(
        temps=temps,
        fan_speeds=fan_speeds,
        powers=powers,
        noises_db=noises,
        max_temp=max_temp,
        safety_violated=max_temp > T_SAFETY_MAX_C,
    )
