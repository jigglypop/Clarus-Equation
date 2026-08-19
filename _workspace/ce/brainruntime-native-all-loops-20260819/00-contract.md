# BrainRuntime-native memory loops 6--10 research contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/_archive/agi-v9-runtime-integration-20260812`

## Question

Can the already implemented temporal-memory and replay ideas be executed through the real
`BrainRuntime` state transition and recurrent weight matrix, rather than only through the
stand-alone NumPy consolidation matrices, and what happens when the same executable path is
extended through temporal selection (Loop 6), agent routing (Loop 7), replay consolidation
(Loop 8), intervention transfer (Loop 9), and bounded self-prediction (Loop 10)?

## Fixed status boundary

This is a computational experiment. Passing a loop establishes only the behavior measured by
that loop in the repository runtime and task generator. It does not establish biological
memory, cortical consolidation, consciousness, AGI, or equivalence between SCCs and cognitive
modules. A failed gate remains a reported STOP and does not authorize threshold changes after
the result is seen.

## Native-runtime criterion

A result counts as native only if all of the following hold:

1. Encoding, replay, and recall trajectories are produced by `BrainRuntime.step` or
   `RuntimeAgent.step`.
2. Loop 8 consolidation changes `BrainRuntime.weight`; a separate association matrix cannot
   satisfy this criterion.
3. Before independent recall, the temporal episodic rows and `HippocampusMemory` rows are both
   removed.
4. Recall is decoded from the runtime activation reached by a cue followed by free recurrent
   rollout. The decoder may compare against a fixed codebook but may not contain trained
   cue-to-value associations.
5. The disabled/default runtime path remains byte-behavior compatible under focused tests.

## Loop definitions and preregistered gates

All reported aggregate advantages use matched seeds and identical initial weights.

### Loop 6 -- valid-time temporal selection

The latest non-deleted event for each `(subject, relation)` is the only event eligible for
replay. Gate: selected-version accuracy `>= 0.99`, deleted/unknown abstention `>= 0.99`, and a
temporal-order shuffle ablation must reduce latest-value accuracy by at least `0.20`.

### Loop 7 -- selective RuntimeAgent route

Only explicit memory queries may trigger the temporal route; supplied context takes precedence,
and disabled mode must equal the base agent action. Gate: route precision and recall accuracy
both `>= 0.99`, with zero temporal reads in disabled mode.

### Loop 8 -- native replay and independent recall

Latest-valid episodes are replayed through the runtime in NREM, after which both external stores
are detached. Gate: every evaluated seed has nonzero finite recurrent-weight drift; clean recall
accuracy `>= 0.80`; 15% cue-corruption accuracy `>= 0.65`; unknown abstention `>= 0.95`; native
replay exceeds no-replay and target-shuffled controls by at least `0.20` clean accuracy; and
target-attractor cosine improves over the cue-only state by at least `0.05` on average.

### Loop 9 -- intervention transfer

After consolidation, one cue coordinate block is intervened on without changing the target
codebook. Gate: the native model predicts the held-out intervention target at accuracy `>= 0.70`
and exceeds a replay-target-shuffled control by at least `0.20`. This is a synthetic causal
transfer result, not real-world causal discovery.

### Loop 10 -- bounded self-prediction and metacognitive response

The self-model receives only current runtime observables and the committed action and predicts
the next native activation summary. Gate: next-state prediction MSE improves on a persistence
baseline by at least 10%, error scores are finite, and high-error interventions produce a larger
metacognitive correction depth than matched non-intervened transitions. This does not test
phenomenal consciousness.

## Fixed development protocol

- Development seeds: `97101..97108`.
- Confirmation seeds: `98101..98132`; evaluated once after implementation and focused tests.
- Runtime backend: PyTorch CPU, because the experiment requires observable native weight updates.
- Primary dimension: 48; replay epochs: 12; free-rollout horizon: 6.
- The implementation may fix runtime bugs or add an opt-in native controller, but it may not
  mutate default-off behavior.
- Per-loop raw results, configuration, source commit, and SHA-256 of the result JSON are retained
  under `artifacts/`.

## Required controls

- no replay;
- target-shuffled replay;
- temporal-order shuffle;
- episodic/hippocampal cutoff audit;
- default-off compatibility;
- snapshot/restore parity after native consolidation;
- exact seed and configuration replay.

## Completion condition

The run is complete when all five loops have executable results, every control above is reported,
focused tests pass, the confirmation command is reproducible, and the final report preserves GO
and STOP separately for each loop. The run may complete with scientific STOP results.
