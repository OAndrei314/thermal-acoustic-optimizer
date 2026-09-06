import pytest

from thermal_acoustic.cli import main


def test_cli_compare_reevaluate_incumbent_runs_and_reports_both_variants(tmp_path, capsys):
    report_path = tmp_path / "report.md"
    exit_code = main(
        [
            "--n-points",
            "6",
            "--iterations",
            "50",
            "--seed",
            "0",
            "--sensor-noise-std",
            "1.5",
            "--noise-trials",
            "20",
            "--noise-trials-per-eval",
            "3",
            "--compare-reevaluate-incumbent",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "robust_optimized_reeval" in printed

    report = report_path.read_text()
    assert "robust_optimized" in report
    assert "robust_optimized_reeval" in report


def test_cli_confidence_z_runs_and_reports_the_confidence_variant(tmp_path, capsys):
    report_path = tmp_path / "report.md"
    exit_code = main(
        [
            "--n-points",
            "6",
            "--iterations",
            "50",
            "--seed",
            "0",
            "--sensor-noise-std",
            "1.5",
            "--noise-trials",
            "20",
            "--noise-trials-per-eval",
            "3",
            "--confidence-z",
            "1.0",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "robust_optimized_confidence" in printed

    report = report_path.read_text()
    assert "robust_optimized_confidence" in report


def test_cli_workload_distribution_runs_and_reports_the_workload_robust_variant(tmp_path, capsys):
    report_path = tmp_path / "report.md"
    exit_code = main(
        [
            "--n-points",
            "6",
            "--iterations",
            "50",
            "--seed",
            "0",
            "--workload-distribution",
            "--workload-trials",
            "20",
            "--workload-trials-per-eval",
            "3",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "workload_robust_optimized" in printed

    report = report_path.read_text()
    assert "workload_robust_optimized" in report
    assert "Workload-Distribution Robustness" in report


def test_cli_joint_robustness_requires_sensor_noise_and_workload_distribution():
    with pytest.raises(SystemExit) as exc_info:
        main(["--joint-robustness"])
    assert exc_info.value.code != 0


def test_cli_joint_robustness_runs_and_reports_the_jointly_robust_variant(tmp_path, capsys):
    report_path = tmp_path / "report.md"
    exit_code = main(
        [
            "--n-points",
            "6",
            "--iterations",
            "50",
            "--seed",
            "0",
            "--sensor-noise-std",
            "1.5",
            "--noise-trials",
            "20",
            "--noise-trials-per-eval",
            "3",
            "--workload-distribution",
            "--workload-trials",
            "20",
            "--workload-trials-per-eval",
            "3",
            "--joint-robustness",
            "--joint-trials-per-eval",
            "3",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "jointly_robust_optimized" in printed

    report = report_path.read_text()
    assert "jointly_robust_optimized" in report
    assert "Joint Sensor+Workload Robustness" in report


def test_cli_uncertainty_scale_sweep_requires_sensor_noise_std():
    with pytest.raises(SystemExit) as exc_info:
        main(["--uncertainty-scale-sweep"])
    assert exc_info.value.code != 0


def test_cli_uncertainty_scale_sweep_runs_and_reports_each_scale(tmp_path, capsys):
    report_path = tmp_path / "report.md"
    exit_code = main(
        [
            "--n-points",
            "6",
            "--iterations",
            "30",
            "--seed",
            "0",
            "--sensor-noise-std",
            "1.5",
            "--noise-trials",
            "20",
            "--joint-trials-per-eval",
            "3",
            "--uncertainty-scale-sweep",
            "--uncertainty-scales",
            "0.5,1.5",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "uncertainty-scale sweep" in printed
    assert "0.5" in printed and "1.5" in printed

    report = report_path.read_text()
    assert "Uncertainty-Scale Sweep" in report
    assert "transfer gap" in report


def test_cli_workload_reevaluate_incumbent_reports_both_variants(tmp_path, capsys):
    report_path = tmp_path / "report.md"
    exit_code = main(
        [
            "--n-points",
            "6",
            "--iterations",
            "50",
            "--seed",
            "0",
            "--workload-distribution",
            "--workload-trials",
            "20",
            "--workload-trials-per-eval",
            "3",
            "--workload-reevaluate-incumbent",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "workload_robust_optimized_reeval" in printed

    report = report_path.read_text()
    assert "workload_robust_optimized" in report
    assert "workload_robust_optimized_reeval" in report
