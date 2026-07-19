# Graphing files

`plot_observed_model_case_study.py`

Renders the two-panel observed-model figure comparing the original observed model
against the attacker-induced observed model.

Common commands:

```bash
python3 graphing/plot_observed_model_case_study.py
```

`common.py`

Shared parser and helpers for case-study inputs. It reads the newer JSON case
files and still supports the older txt format.

`case_study.json`

Active input file used by `plot_observed_model_case_study.py`.

`case_study-c1.json`

Case 1 data. This is the Theorem 3-style example where the observed transition
model is fixed and the attack changes the learned observed rewards/values.

`case_study-c2.json`

Case 2 data. This is the transition-mislearning example where the attack changes
the induced observed transition model.

`case_study-c3.json`

Case 3 data. This is the fake observed-coverage example where observed
state-action coverage is complete but hidden-state coverage is incomplete.
