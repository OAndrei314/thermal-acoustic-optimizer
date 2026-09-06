"""Renders thermal-control optimization results into markdown."""
from __future__ import annotations

from .optimize import ParetoPoint
from .uncertainty_sweep import UncertaintyScalePoint


def _policy_row(name: str, ev: dict) -> str:
    return (
        f"| {name} | {ev['mean_power_w']:.2f} | {ev['mean_noise_db']:.2f} | "
        f"{ev['max_temp_c']:.1f} | {'YES' if ev['safety_violated'] else 'no'} |"
    )


def render_markdown_report(
    evaluations: dict[str, dict],
    pareto_points: list[ParetoPoint] | None = None,
    frontier: list[ParetoPoint] | None = None,
    robustness: dict[str, dict] | None = None,
    workload_robustness: dict[str, dict] | None = None,
    joint_robustness: dict[str, dict] | None = None,
    uncertainty_scale_points: list[UncertaintyScalePoint] | None = None,
) -> str:
    lines = [
        "# Thermal / Acoustic Fan Control Report",
        "",
        "## Research question",
        "",
        "Starting from an always-on baseline and a naive linear ramp, how much power and",
        "acoustic noise can a from-scratch local-search optimizer save on a fixed synthetic",
        "workload, without ever crossing the safety temperature limit?",
        "",
        "## Why this matters",
        "",
        "Fan power and acoustic behavior are real product constraints in telecom/embedded",
        "hardware -- acoustic limits are often a hard customer/regulatory requirement, and",
        "fan power is a direct line item in a system's power budget.",
        "",
        "## Results",
        "",
        "| policy | mean power (W) | mean noise (dB) | max temp (°C) | safety violated |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for name, ev in evaluations.items():
        lines.append(_policy_row(name, ev))

    if pareto_points:
        frontier_labels = {point.label for point in frontier or []}
        lines.extend(
            [
                "",
                "## Efficiency / Thermal-Margin Tradeoff Sweep",
                "",
                "| weights | mean power (W) | mean noise (dB) | max temp (C) | safety violated | frontier |",
                "| --- | ---: | ---: | ---: | :---: | :---: |",
            ]
        )
        for point in pareto_points:
            ev = point.evaluation
            lines.append(
                f"| {point.label} | {ev['mean_power_w']:.2f} | {ev['mean_noise_db']:.2f} | "
                f"{ev['max_temp_c']:.1f} | {'YES' if ev['safety_violated'] else 'no'} | "
                f"{'yes' if point.label in frontier_labels else 'no'} |"
            )

        lines.extend(
            [
                "",
                "The frontier marks safe policies that are not dominated on power, noise,",
                "and maximum temperature by another safe policy in the sweep.",
            ]
        )

    if robustness:
        lines.extend(
            [
                "",
                "## Sensor-Noise Robustness (Monte Carlo)",
                "",
                "The noiseless optimizer assumes the controller reads the true temperature",
                "exactly. This section re-evaluates policies with additive Gaussian noise on",
                "the *measured* temperature the controller reacts to, while the real thermal",
                "state (and the safety check) still uses the true temperature.",
                "",
                "| policy | violation rate | mean max temp (°C) | worst max temp (°C) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, rob in robustness.items():
            lines.append(
                f"| {name} | {rob['safety_violation_rate']:.1%} | "
                f"{rob['mean_max_temp_c']:.1f} | {rob['worst_max_temp_c']:.1f} |"
            )
        lines.extend(
            [
                "",
                f"({next(iter(robustness.values()))['n_trials']} trials, "
                f"sensor noise std = {next(iter(robustness.values()))['sensor_noise_std']:.2f} °C)",
            ]
        )

    if workload_robustness:
        lines.extend(
            [
                "",
                "## Workload-Distribution Robustness (Monte Carlo)",
                "",
                "The noiseless optimizer and the sensor-noise sections above both tune and",
                "evaluate against `workload.heat_trace()`, one fixed burst-timing/duration/",
                "magnitude realization. This section re-evaluates policies against a",
                "*distribution* of workload traces (`workload.sample_heat_trace`, jittered",
                "burst start, duration, and magnitude) instead -- the question here is",
                "whether a policy tuned on the one fixed trace generalizes, or overfits to",
                "its exact shape.",
                "",
                "| policy | violation rate | mean max temp (°C) | worst max temp (°C) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, rob in workload_robustness.items():
            lines.append(
                f"| {name} | {rob['safety_violation_rate']:.1%} | "
                f"{rob['mean_max_temp_c']:.1f} | {rob['worst_max_temp_c']:.1f} |"
            )
        lines.extend(
            [
                "",
                f"({next(iter(workload_robustness.values()))['n_trials']} trials)",
            ]
        )

    if joint_robustness:
        lines.extend(
            [
                "",
                "## Joint Sensor+Workload Robustness (Monte Carlo)",
                "",
                "The two sections above each Monte-Carlo one source of uncertainty at a",
                "time, holding the other fixed. This section draws both a random workload",
                "trace *and* noisy sensor reads on every trial, so both failure modes are",
                "active simultaneously -- the question is whether a policy made robust to",
                "one axis alone stays robust once both are live at once, or trades one",
                "failure mode for the other.",
                "",
                "| policy | violation rate | mean max temp (°C) | worst max temp (°C) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, rob in joint_robustness.items():
            lines.append(
                f"| {name} | {rob['safety_violation_rate']:.1%} | "
                f"{rob['mean_max_temp_c']:.1f} | {rob['worst_max_temp_c']:.1f} |"
            )
        lines.extend(
            [
                "",
                f"({next(iter(joint_robustness.values()))['n_trials']} trials, "
                f"sensor noise std = {next(iter(joint_robustness.values()))['sensor_noise_std']:.2f} °C)",
            ]
        )

    if uncertainty_scale_points:
        lines.extend(
            [
                "",
                "## Uncertainty-Scale Sweep",
                "",
                "The joint-robustness section above was only ever measured at one sensor-noise",
                "magnitude and one workload-jitter magnitude. This sweep scales both together",
                "(scale=1.0 reproduces that section's magnitude exactly) and re-runs the",
                "sensor-robust / workload-robust / jointly-robust three-way comparison at each",
                "scale, to see whether the single-axis-transfer gap is a generic property of",
                "compounding two failure modes or an artifact of the one magnitude tested before.",
                "",
                "| scale | sensor std (°C) | sensor-robust | workload-robust | jointly-robust | transfer gap |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for point in uncertainty_scale_points:
            lines.append(
                f"| {point.scale:g} | {point.sensor_noise_std:.2f} | "
                f"{point.sensor_robust_violation_rate:.1%} | "
                f"{point.workload_robust_violation_rate:.1%} | "
                f"{point.jointly_robust_violation_rate:.1%} | "
                f"{point.transfer_gap:.1%} |"
            )
        lines.extend(
            [
                "",
                "\"transfer gap\" is the best single-axis-robust policy's violation rate minus",
                "the jointly-robust policy's, at that scale -- how much single-axis robustness",
                "leaves on the table relative to defending against the compound failure mode",
                "directly.",
            ]
        )

    lines.append("")
    return "\n".join(lines)
