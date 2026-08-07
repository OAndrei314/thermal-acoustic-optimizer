"""Renders thermal-control optimization results into markdown."""
from __future__ import annotations

from .optimize import ParetoPoint


def _policy_row(name: str, ev: dict) -> str:
    return (
        f"| {name} | {ev['mean_power_w']:.2f} | {ev['mean_noise_db']:.2f} | "
        f"{ev['max_temp_c']:.1f} | {'YES' if ev['safety_violated'] else 'no'} |"
    )


def render_markdown_report(
    evaluations: dict[str, dict],
    pareto_points: list[ParetoPoint] | None = None,
    frontier: list[ParetoPoint] | None = None,
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
        "## Money question",
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

    lines.append("")
    return "\n".join(lines)
