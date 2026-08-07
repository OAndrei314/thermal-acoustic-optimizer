"""`python -m thermal_acoustic.cli optimize ...`"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .objective import evaluate_policy
from .optimize import optimize_policy, optimize_tradeoff_sweep, pareto_frontier
from .policies import always_on_policy, linear_ramp_policy
from .report import render_markdown_report
from .simulate import T_AMBIENT_C, T_SAFETY_MAX_C
from .workload import heat_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thermal-acoustic")
    parser.add_argument("--n-points", type=int, default=6, help="control-curve breakpoints")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="run a power/noise objective-weight sweep and report the safe frontier",
    )
    parser.add_argument(
        "--noise-weights",
        default="0.5,1,2,4",
        help="comma-separated noise weights for the Pareto sweep; power weight stays 1",
    )
    parser.add_argument("--report", help="optional markdown report path")
    args = parser.parse_args(argv)

    temp_breakpoints = np.linspace(T_AMBIENT_C, T_SAFETY_MAX_C, args.n_points)
    heat_w = heat_trace()

    baselines = {
        "always_on": always_on_policy(args.n_points),
        "linear_ramp": linear_ramp_policy(args.n_points),
    }

    evaluations = {
        name: evaluate_policy(cp, temp_breakpoints, heat_w) for name, cp in baselines.items()
    }

    result = optimize_policy(
        temp_breakpoints, heat_w, init=baselines["linear_ramp"],
        iterations=args.iterations, seed=args.seed,
    )
    evaluations["optimized"] = evaluate_policy(result.control_points, temp_breakpoints, heat_w)

    print(f"{'policy':<15} {'power(W)':<10} {'noise(dB)':<11} {'max_temp(C)':<13} {'safe':<6}")
    for name, ev in evaluations.items():
        print(
            f"{name:<15} {ev['mean_power_w']:<10.2f} {ev['mean_noise_db']:<11.2f} "
            f"{ev['max_temp_c']:<13.1f} {'NO' if ev['safety_violated'] else 'yes':<6}"
        )

    pareto_points = None
    frontier = None
    if args.pareto:
        noise_weights = [float(part.strip()) for part in args.noise_weights.split(",") if part.strip()]
        weights = [(1.0, noise_weight) for noise_weight in noise_weights]
        pareto_points = optimize_tradeoff_sweep(
            temp_breakpoints,
            heat_w,
            init=baselines["linear_ramp"],
            weights=weights,
            iterations=args.iterations,
            seed=args.seed,
        )
        frontier = pareto_frontier(pareto_points)
        frontier_labels = {point.label for point in frontier}

        print("")
        print(f"{'weights':<18} {'power(W)':<10} {'noise(dB)':<11} {'max_temp(C)':<13} {'frontier':<8}")
        for point in pareto_points:
            ev = point.evaluation
            print(
                f"{point.label:<18} {ev['mean_power_w']:<10.2f} {ev['mean_noise_db']:<11.2f} "
                f"{ev['max_temp_c']:<13.1f} {'yes' if point.label in frontier_labels else 'no':<8}"
            )

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(render_markdown_report(evaluations, pareto_points, frontier))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
