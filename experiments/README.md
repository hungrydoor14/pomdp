# Experiment scripts

## Result naming

- `T1-DSE`: Lemma 1
- `T1-IFF`: Theorem 1
- `T2-DSE`: Theorems 2 and 3
- `T2-PD`: planned Theorem 4
- `T2-IFF`: planned main theorem (Theorem 5)

Python modules spell these labels with underscores (for example,
`find_t2_dse_failure_unteachable_case.py`); human-readable labels and generated
artifact names use hyphens (for example, `T2-DSE` and `t2-dse_case_study-c1.png`).

`two_period_joint_policy_experiments.py`

Shared two-period factored-POMDP helpers and joint-policy calculations used by
the other experiment scripts.

`find_fixed_sequence_cases.py`

Searches for cases where one fixed open-loop sequence is preferred.

`find_neighbor_pattern_cases.py`

Searches for local-neighbor and root-deviation patterns in two-period policy
trees.

`find_t2_l1_certifies_t2_dse_fails_case.py`

Reproduces the seed-140 example used to illustrate Lemma 1 while the
pointwise T2-DSE condition fails. Its default target is the stationary tree
`(a1, a1, a1)`, and it writes `graphing/case_study-l1.json`.

`find_t2_pd_certifies_t2_dse_fails_case.py`

Searches for the history-dependent target used by Case 3, certified by T2-PD
while the pointwise T2-DSE condition fails. Its default target is
`(a1, a0, a1)`, and it writes `graphing/case_study-c3.json`.

`t2_policy_dependent_case_search.py`

Shared exhaustive-search and reporting implementation used by both entry
points. It checks the 81 attacker policies whose rows are `(1,0)`, `(0,1)`,
or `(0.5,0.5)`.

`find_t2_dse_inducible_observed_model_case.py`

Finds the Case 1-style inducible observed-model example used for T2-DSE.

`find_t2_dse_transition_mislearning_case.py`

Finds the Case 2-style example where the learned observed transition model
changes under attacker-induced hidden-state mixtures.

`find_fake_observed_coverage_case.py`

Constructs a coverage diagnostic where the observed state \(S_1\) appears fully
covered, but some hidden states \((S_1,S_2)\) have zero samples.

`find_t2_dse_failure_unteachable_case.py`

Searches for a Case 4 example where T2-DSE's range condition fails and the
stationary two-period target is not strictly teachable—the simple converse of
Case 1. It prints the same observed-model case-study sections as Case 1. The
attacker search is exhaustive over policy rows `(1,0)`, `(0,1)`, and
`(0.5,0.5)`.

`find_t2_dse_pd_fail_teachable_case.py`

Searches for the nominally missing quadrant where T2-DSE and T2-PD both fail
but direct policy-tree enumeration says that the target is strictly teachable.
Under the current two-period T2-PD definition this is a consistency diagnostic:
the PD margin equals the direct teachability margin, so no witness is expected.

`plot_inducible_observed_model_diagnostics.py`

Produces the older diagnostic plots for the inducible observed-model case.

`plot_joint_tree_case_study.py`

Plots diagnostics for the joint-tree case study.

`compare_memoryless_to_joint_tree_cases.py`

Compares memoryless open-loop plans against joint policy trees.

`print_joint_tree_dominant_history_cases.py`

Prints candidate joint-tree cases with dominant histories.
