# Experiment scripts

`two_period_joint_policy_experiments.py`

Shared two-period factored-POMDP helpers and joint-policy calculations used by
the other experiment scripts.

`find_fixed_sequence_cases.py`

Searches for cases where one fixed open-loop sequence is preferred.

`find_neighbor_pattern_cases.py`

Searches for local-neighbor and root-deviation patterns in two-period policy
trees.

`find_inducible_observed_model_case.py`

Finds the Case 1-style inducible observed-model example used for Theorem 3.

`find_transition_mislearning_case.py`

Finds the Case 2-style example where the learned observed transition model
changes under attacker-induced hidden-state mixtures.

`find_fake_observed_coverage_case.py`

Constructs a coverage diagnostic where the observed state \(S_1\) appears fully
covered, but some hidden states \((S_1,S_2)\) have zero samples.

`plot_inducible_observed_model_diagnostics.py`

Produces the older diagnostic plots for the inducible observed-model case.

`plot_joint_tree_case_study.py`

Plots diagnostics for the joint-tree case study.

`compare_memoryless_to_joint_tree_cases.py`

Compares memoryless open-loop plans against joint policy trees.

`print_joint_tree_dominant_history_cases.py`

Prints candidate joint-tree cases with dominant histories.
