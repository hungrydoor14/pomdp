"""Find the history-dependent T2-PD example used by manuscript Case 3."""

from pathlib import Path

from t2_policy_dependent_case_search import main


if __name__ == "__main__":
    main(
        default_target=(1, (0, 1)),
        default_output_json=Path("graphing/case_study-c3.json"),
        report_title="T2-PD certifies / T2-DSE fails after restricted attack",
        certificate_label="T2-PD",
    )
