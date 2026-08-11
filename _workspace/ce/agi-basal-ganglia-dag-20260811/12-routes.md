# Candidate routes and falsification

## Model comparison

| Candidate | Use | Status |
|---|---|---|
| Inverse tree | Reverse inhibitory reduction only | Useful metaphor, not a standalone biological model |
| Soft decision tree | Sparse routing baseline | Implementable but overly rigid |
| XGBoost | Strong static prediction baseline | Not a basal-ganglia mechanism |
| Hierarchical mixture-of-experts | Parallel hierarchical routing skeleton | Best statistical base |
| Recurrent inhibitory DAG | HMoE + reverse competition + signed TD state | Primary hypothesis |

XGBoost adds frozen trees to fit successive loss residuals. The proposed model
reuses one directed graph online, carries temporal state, applies structured
competition, and updates eligible edges with signed TD error. Removing state,
competition, and online credit reduces it toward an additive tree ensemble.

## Required falsification tests

1. A high-level inhibitory lesion must affect a subtree; a leaf lesion must be
   localized.
2. Proposal activity must precede structured competitive inhibition.
3. State reset must remove history dependence under identical current input.
4. Feedback shuffle and sign flip must remove or reverse learning benefit.
5. Inhibition removal must selectively damage high-conflict trials.
6. Recurrent DAG must beat a feedforward DAG and matched recurrent flat policy,
   not merely XGBoost.
7. Stationary and linear null tasks must not manufacture a DAG advantage.
8. Every feedback edge must cross a microtime boundary; same-pass cycles fail
   the topology audit.

## Next benchmark

Use eight actions with shared subproblems, context recombination, block
switches, and delayed signed feedback. Compare flat softmax, race/DDM, XGBoost,
hard tree, feedforward sparse DAG, recurrent sparse DAG, feedback shuffle,
feedback sign flip, and direction/topology shuffle at matched compute and
description cost.
