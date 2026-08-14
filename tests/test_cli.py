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
