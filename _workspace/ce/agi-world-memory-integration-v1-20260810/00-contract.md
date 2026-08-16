# Research contract — G9-CBM V1

Status: FROZEN_FOR_LANES

## 1. Question

In a preregistered synthetic, action-conditioned compositional world, does adding
an observed-only episodic long-term memory (LTM) and a provenance-preserving,
world-constrained offline recombination path to one shared causal world model
improve both autonomous 20-step prediction and counterfactual action selection,
without increasing false recall or invalid transitions?

This is a component-integration experiment. It is not a test of human dreaming,
biological hippocampal dynamics, consciousness, or AGI.

## 2. Inherited evidence and non-negotiable history

- G9-CB V4 passed its registered one-step synthetic gate.
- G9-CB V5, V6, and V7 produced finite 20-step rollouts but failed their
  registered robustness gates. Those failures remain active counterevidence.
- G7-M V2 passed its fixed-96-item known-slot LTM and constrained-recombination
  gate. Its hard exemplar completion is not called a recurrent attractor.
- No result from either family may be relabelled as integrated planning evidence.

## 3. Definitions

- **World model**: a candidate that predicts future observed states from an
  observed prefix and a proposed action sequence. It may not read held-out
  futures, evaluator latents, or counterfactual targets.
- **Observed-only episodic LTM**: a persistent, cue-addressable store containing
  only real wake episodes. Synthetic records are forbidden.
- **Dream-like constrained recombination**: a bounded offline operation that
  recombines only observed fragments permitted by an inferred validity graph.
  Every output is tagged `synthetic/hypothetical`, is unavailable as an episode
  identity, and may update only a slow schema/world component.
- **Planning**: choose one action sequence from a fixed candidate set using only
  candidate rollouts, then score the selected action against evaluator outcomes.
- **Invalid transition**: a transition violating a generator-level registered
  state, port, context, continuity, or action constraint. Generator truth is used
  only by the evaluator after prediction/selection.

## 4. Domain and comparison

- Domain: a small CPU-only synthetic compositional causal family with stable
  mechanisms, changing context, partial episode cues, valid unseen combinations,
  action-conditioned futures, and deliberately invalid action/fragment lures.
- Horizon: one origin, H5 diagnostic and H20 primary autonomous rollout.
- Factorial cells use identical worlds, prefixes, candidate actions, budgets, and
  seeds: `M00` no LTM/no dream, `M10` LTM only, `M01` dream only, `M11` both.
- Required controls: persistence, frozen G9-CB-style causal rollout, schema-only
  fallback, oracle evaluator (diagnostic only), shuffled episode binding, and
  unconstrained recombination lesion.
- The implementation must reuse the verified G7-M V2 provenance boundary or
  prove byte/semantic equivalence before registered seeds are opened.

## 5. Claims to test

- **C1 [prediction]**: the marginal LTM effect reduces H20 rollout error.
- **C2 [prediction]**: the marginal dream-like effect reduces unseen-valid H20
  rollout error.
- **C3 [prediction]**: `M11` reduces action-selection regret and increases task
  success relative to `M00` while satisfying both absolute component gates.
- **C4 [safety prediction]**: gains do not come from invalid transitions,
  synthetic-as-real provenance, held-out reads, or episodic-store contamination.
- **C5 [reported, not required positive]**: all factorial interactions and their
  paired confidence intervals are reported. The word `synergy` is permitted only
  when the relevant interaction CI lower bound is strictly positive.

## 6. Frozen primary gates

All gates are all-of and are computed per seed before averaging. Paired 95%
Student-t intervals use the seed as the independent unit.

- H20 state NRMSE: `M11 <= 0.90 * M00`, paired benefit CI lower `> 0`, strict
  seed win fraction `>= 0.65`.
- H20 unseen-valid NRMSE: each dream cell improves its matched no-dream cell by
  at least 10%, paired benefit CI lower `> 0`, strict win fraction `>= 0.65`.
- Planning regret: `M11 <= 0.80 * M00`, paired benefit CI lower `> 0`; selected
  action success improves by at least 0.10 with paired CI lower `> 0`.
- Absolute `M11` requirements: finite outputs, H20/H5 error ratio `<= 2.0`, task
  success `>= 0.75`, invalid predicted-transition rate `<= 0.01`.
- Memory safety: accepted-wrong episode rate `<= 0.05`, unstored-lure false recall
  mean `<= 0.05`, synthetic tagged `recalled` `<= 0.01`, synthetic-to-LTM inserts
  `= 0`.
- Constraint/audit safety: held-out future reads, evaluator latent reads, observed
  record overwrite, cross-context invalid splice, nonfinite outputs, and test
  access before unlock must all equal `0`.
- No antagonism: `M11` may degrade neither the LTM-only recall error nor the
  dream-only unseen-valid error by more than 2% under paired upper-CI checks.
- Resource: CPU-only, no downloads, no external trajectory files, and the same
  registered compute budget in all factorial cells.

## 7. Split and lock contract

- Train/calibration: seeds `83100..83139` (40).
- Validation: seeds `84100..84139` (40), exactly one registered run.
- Locked test: seeds `85100..85159` (60), inaccessible until validation passes
  every performance, safety, integrity, and resource gate.
- Unit and development seeds must be outside all registered ranges. Any pilot
  must be declared development evidence and may not use validation/test seeds.
- Before train/calibration: freeze raw LF preregistration SHA, implementation SHA,
  runner SHA, tests SHA, inherited-module SHA, and equivalence proof.
- Calibration is train-only, written once with raw SHA, and reused byte-identically
  by validation and test.
- Any post-registration change to generator, model, thresholds, metrics, gates,
  seeds, or provenance rules creates V2 with fresh seeds. A failed artifact is
  preserved and its locked test remains unopened.

## 8. Allowed conclusion

PASS would support only this statement: in the registered small synthetic family,
the specified observed-only memory and constrained synthetic schema path improve
the specified world-model rollout/planning metrics without violating the listed
safety gates. It would not establish general world modelling, biological memory,
dreaming, sleep, consciousness, or AGI.

## 9. Stop conditions

- Stop before implementation if the lanes cannot define a leak-free generator,
  common-budget factorial comparison, and evaluator-independent candidate API.
- Stop before validation on any unresolved P0/P1 contract or integrity defect.
- Stop after validation FAIL; do not open test.
