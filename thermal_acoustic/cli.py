"""`python -m thermal_acoustic.cli optimize ...`"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .objective import evaluate_policy
from .optimize import optimize_policy, optimize_tradeoff_sweep, pareto_frontier
from .policies import always_on_policy, linear_ramp_policy
from .report import render_markdown_report
from .robustness import evaluate_robustness
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
    parser.add_argument(
        "--sensor-noise-std",
        type=float,
        default=0.0,
        help="if > 0, also Monte-Carlo evaluate the optimized policy's safety under this "
        "much Gaussian sensor read noise (degC), and run a second noise-aware 'robust' "
        "optimization pass for comparison",
    )
    parser.add_argument("--noise-trials", type=int, default=200, help="Monte-Carlo trials for the robustness check")
    parser.add_argument(
        "--noise-trials-per-eval",
        type=int,
        default=5,
        help="Monte-Carlo samples averaged per candidate score during robust optimization "
        "(higher = more accurate noisy-score estimate, slower to run)",
    )
    parser.add_argument(
        "--compare-reevaluate-incumbent",
        action="store_true",
        help="also run the robust optimization pass with a fresh-noise incumbent "
        "re-score each iteration (instead of the default stale incumbent score), and "
        "report both side by side",
    )
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

    robustness = None
    if args.sensor_noise_std > 0:
        robustness = {}
        robustness["optimized"] = evaluate_robustness(
            result.control_points, temp_breakpoints, heat_w,
            sensor_noise_std=args.sensor_noise_std, n_trials=args.noise_trials, seed=args.seed,
        )

        robust_result = optimize_policy(
            temp_breakpoints, heat_w, init=baselines["linear_ramp"],
            iterations=args.iterations, seed=args.seed,
            sensor_noise_std=args.sensor_noise_std,
            noise_trials_per_eval=args.noise_trials_per_eval,
        )
        evaluations["robust_optimized"] = evaluate_policy(robust_result.control_points, temp_breakpoints, heat_w)
        robustness["robust_optimized"] = evaluate_robustness(
            robust_result.control_points, temp_breakpoints, heat_w,
            sensor_noise_std=args.sensor_noise_std, n_trials=args.noise_trials, seed=args.seed,
        )

        if args.compare_reevaluate_incumbent:
            reeval_result = optimize_policy(
                temp_breakpoints, heat_w, init=baselines["linear_ramp"],
                iterations=args.iterations, seed=args.seed,
                sensor_noise_std=args.sensor_noise_std,
                noise_trials_per_eval=args.noise_trials_per_eval,
                reevaluate_incumbent=True,
            )
            evaluations["robust_optimized_reeval"] = evaluate_policy(
                reeval_result.control_points, temp_breakpoints, heat_w
            )
            robustness["robust_optimized_reeval"] = evaluate_robustness(
                reeval_result.control_points, temp_breakpoints, heat_w,
                sensor_noise_std=args.sensor_noise_std, n_trials=args.noise_trials, seed=args.seed,
            )

        print("")
        print(f"sensor noise std: {args.sensor_noise_std:.2f} degC, {args.noise_trials} Monte-Carlo trials")
        print(f"{'policy':<18} {'violation rate':<15} {'mean max_temp(C)':<18} {'worst max_temp(C)':<18}")
        for name, rob in robustness.items():
            print(
                f"{name:<18} {rob['safety_violation_rate']:<15.1%} "
                f"{rob['mean_max_temp_c']:<18.1f} {rob['worst_max_temp_c']:<18.1f}"
            )

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(render_markdown_report(evaluations, pareto_points, frontier, robustness))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
