# Frozen route and counterexample map

Status: COMPLETE

| Route | Mechanism | Required adverse controls | Independent verdicts |
|---|---|---|---|
| T1 | unchanged M1 on held-out factor combination | zero/sign/time/reset/no-replay/target-shuffle | factor transfer only |
| M2 | positive-minus-negative lag correlation | no-write, target-shuffle, identical phase, positive-only, negative-only, sign reverse | binding; factor transfer |
| M3 | frozen teacher-forced replay residual plus cached cue credit | predictor-only, transition shuffle, delayed error, sign flip, no replay, target shuffle | prediction; binding; factor transfer |

## Implementation order

1. Implement T1 without touching the frozen M1 learner or threshold. A failure is final for T1.
2. Implement M2 in a new isolated module and first prove identical-phase zero update, equal phase
   schedules, and row-post/column-pre direction. Then run binding and factor transfer.
3. Implement M3 predictor fitting and held-out prediction audit before enabling any recurrent write.
   Only a frozen predictor may feed the error learner. Run binding and factor transfer separately.
4. Use development seeds only for the at most two logged implementation revisions. Freeze source,
   thresholds, and results before opening confirmation once.

## Complete counterexamples

- T1 is rejected if held-out decoding survives target shuffle or store cutoff fails.
- M2 is rejected as contrastive acquisition if identical phases write weight, negative/positive phase
  counts differ, or a target-shuffled control matches it.
- M3 prediction is rejected if the model sees post-state features, is refit on held-out transitions,
  or fails persistence. M3 binding/transfer is rejected independently if predictor-only or shuffled
  transition controls match it.
- Success in any route cannot be called biological consolidation, causal abstraction in a real brain,
  self-model, or consciousness.
