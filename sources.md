# Sources

This is a preliminary rough list of sources and where they were used, in order to make the paper writing easier.

## Solver setup

* Prof. Wu discussion 1
* https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf
  * POMDP setup / partial observation background

## Random Reward Teachability work

### Confounding / hidden information

* Prof. Wu discussion June 5, 2026
* https://proceedings.mlr.press/v235/hong24d.html
  * Helped with confounded POMDP terminology and hidden information framing.
  * Not directly useful for the math because their setup is different.

### Attacker as information designer

* Prof. Wu discussion June 5, 2026
* https://web.stanford.edu/~gentzkow/research/BayesianPersuasion.pdf
  * I did not implement Bayesian Persuasion.
  * Mostly useful for the “attacker controls information / victim learns from what they see” idea.
  * In my setup, the attacker chooses pi_dagger, generates training data, hides S2, and the victim learns an observed MDP from (S1, action, reward, next_S1) using MLE. Then the victim solves that learned MDP and may learn pi_star.

### Exact POMDP solution / witness algorithm background

* Prof. Wu discussion June 5, 2026
* https://pomdp.org/tutorial/witness.html
  * Basic background on witness algorithm.
  * Better source needed.
  * Prefer citing the technical report below for formal claims.
* https://www.researchgate.net/publication/2626779_The_Witness_Algorithm_Solving_Partially_Observable_Markov_Decision_Processes
  * Went deeper than the POMDP tutorial.
  * Use the original technical report citation rather than ResearchGate.

## Notes on Professor Discussions

Professor discussions are not included in `references.bib`. If a specific unpublished
idea must be attributed, describe it in the text as a personal communication, subject
to the professor's approval. For example:

