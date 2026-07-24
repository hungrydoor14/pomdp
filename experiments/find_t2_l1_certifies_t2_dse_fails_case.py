"""Reproduce the seed-140 Lemma 1 / T2-DSE separation example."""

from pathlib import Path

from t2_policy_dependent_case_search import main


if __name__ == "__main__":
    main(
        default_target=(1, (1, 1)),
        default_output_json=Path("graphing/case_study-l1.json"),
        report_title="Lemma 1 certifies / T2-DSE fails after restricted attack",
        certificate_label="Lemma 1 reduction",
    )
