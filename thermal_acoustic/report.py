"""Renders a comparison of baseline vs. optimized policies into markdown."""
from __future__ import annotations


def render_markdown_report(evaluations: dict[str, dict]) -> str:
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
        lines.append(
            f"| {name} | {ev['mean_power_w']:.2f} | {ev['mean_noise_db']:.2f} | "
            f"{ev['max_temp_c']:.1f} | {'YES' if ev['safety_violated'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)
