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

Verifies that the reduced T2-PD margin—computed from separate same-root and
optimal-continuation comparisons—equals the margin obtained by direct
enumeration of every competing two-period policy tree.

`compare_grid_continuous_t2_pd.py`

Compares the best T2-PD teaching margin from the restricted 81-policy grid
with a continuous four-parameter attacker search, using multiple numerical
restarts and reporting the binding observed state and competing tree. The
default continuous domain includes boundary policies and enforces the theorem's
observed state-action coverage condition; `--coverage-floor` optionally imposes
the stronger full-state action-support condition. The continuous result is a
numerical lower bound on the unrestricted supremum. A reported sign separation
therefore proves that the restricted enumeration missed a positive-margin
attacker, without claiming that the continuous optimizer found the global
maximum. The current script searches the restricted target family that uses the
same rooted tree at both initial observed states.

`verify_case4_unrestricted_t2_pd.py`

Verifies the unrestricted negative T2-PD result for Case 4 by partitioning
the binary Bayes-plausible mixture space into four linear regions and solving
the max-margin LP in each region. It checks the reported overall margin and
maximizing induced mixture collection against the Appendix C.3 values.

`t2_c3_epsilon_robustness.py`

Evaluates the full-support relaxation of the Case 3 attacker over
`epsilon` in `[0,1]`. It reports the largest connected interval from zero
with a positive T2-PD margin, records changes in the binding observed state
and competitor tree, and writes the margin curve and plot under `outputs/t2/`.

`decompose_t2_pd_mechanisms.py`

Evaluates the T2-PD margin under the original, counterfactual reward-only,
counterfactual transition-only, and full attacked models. It reconstructs
Case 2.1 from its exact generator values and performs a seeded randomized
search over behaviorally inducible mixture collections for a joint-only
attacker witness. Each sampled collection is converted into a concrete
attacker policy, reconstructed from that policy, and only then evaluated.
The report includes the binding observed state and
competing rooted tree for every model. The search uses the restricted target
family in which the same rooted tree is evaluated at both initial states. By
default it retains the best witness over the configured search; use
`--stop-after-paper-witness` to stop once the requested substantive margin
threshold is reached. Each reported mixture witness is converted back into an
attacker policy and checked by reconstructing its induced mixtures.

`plot_inducible_observed_model_diagnostics.py`

Produces the older diagnostic plots for the inducible observed-model case.

`plot_joint_tree_case_study.py`

Plots diagnostics for the joint-tree case study.

`compare_memoryless_to_joint_tree_cases.py`

Compares memoryless open-loop plans against joint policy trees.

`print_joint_tree_dominant_history_cases.py`

Prints candidate joint-tree cases with dominant histories.
