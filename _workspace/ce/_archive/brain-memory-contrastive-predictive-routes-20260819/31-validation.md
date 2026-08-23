# T1/M2/M3 development validation

Status: COMPLETE

Independent development circuits are seeds `97301..97316`. Confirmation seeds `99301..99332`
remain sealed because no registered route cleared its development gate.

## Verdict table

| Route | Absolute task result | Decisive control/result | Circuit GO | Verdict |
|---|---:|---|---:|---|
| T1 frozen-M1 factor transfer | held-out 11/16 | five failures all decoded trained `(1,0)` | 11/16 | STOP |
| M2 binding | clean 0/16 | `C- = 0`; positive-only delta identical 16/16 | 0/16 | STOP |
| M2 factor transfer | held-out 0/16 | `C- = 0`; positive-only delta identical 16/16 | 0/16 | STOP |
| M3 predictor | mean MSE ratio `1.065739` | minimum ratio `0.970721` > required `0.90` | 0/16 | STOP |
| M3 binding | clean/task gate 16/16 | transition-order shuffle also 16/16 | 0/16 | STOP |
| M3 factor transfer | held-out 1/16 | transition-order shuffle 8/16 | 0/16 | STOP |

## T1: memory does not imply stable composition

The frozen M1 mechanism transferred the missing `(1,1)` factor combination in 11/16 circuits
(`68.75%`), below the registered 80% circuit gate. All five failures selected the observed `(1,0)`
combination, consistent with the declared factor-frequency imbalance. Decoder-only accuracy was
zero. Held-out exclusion, schedule contract, zero stores, snapshot restore, codebook parity, and the
unchanged frozen configuration passed 16/16. No M1 parameter was revised.

## M2: the negative phase is structurally null

Across all circuits, the reset-plus-zero-replay negative phase produced correlation norm exactly
`0.0`. Consequently the learned update and positive-only update had the same hash 16/16. The
positive collector was nonzero and recurrent weight changed by about `0.20--0.22`, with positive
association contrast, but clean and corrupt recall remained zero for every circuit. Held-out factor
accuracy was also zero. The identical-phase control wrote exactly zero; projection, schedule,
snapshot, cutoff, finite-state, and dense/CSR audits all passed.

This rejects the registered contrastive mechanism twice: subtraction contributed nothing, and the
remaining positive lag write did not create a native attractor at the frozen scale.

## M3 predictor: current-state ridge does not beat persistence

The final fixed-point-source predictor used exactly 64 adjacent fit rows and 16 independent held-out
forks. Mean model MSE was `0.00835502`; mean persistence MSE was `0.00784298`; mean ratio was
`1.065739` (range `0.970721--1.504718`). Predictor parameters remained hash-identical during score,
fit/score rows were disjoint, replay-vector reconstruction was exact, and recurrent weight never
changed. The prediction claim is therefore an empirical STOP, not an implementation failure.

## M3 memory: binding capability without predictive specificity

The normal teacher-forced residual route passed every absolute binding endpoint in all 16 circuits:
clean accuracy 1.0, corrupt at least 2/3, deleted and unknown abstention 1.0, and positive attractor
gain. However the fixed cyclic residual-credit derangement also achieved clean accuracy 1.0 in all
16 circuits. Normal control advantage was therefore exactly zero. Predictor-only, delayed-error,
sign-flip, zero-replay, and target-shuffle controls were all zero.

For the unseen `(1,1)` factor combination, the normal route succeeded in only 1/16 circuits while
the transition-order shuffle succeeded in 8/16. Mean normal advantage over controls was `-0.4375`.
The held-out combination was absent from every task row, replay value, update construction,
threshold, and calibration route.

The honest result is a strong supervised replay-writing capability whose claimed fine temporal
error-credit alignment is not identified. Because the predictor gate also failed, this cannot be
called validated predictive-error learning.

## Mechanical validation

Focused source tests passed `7/7`. The final adjacent regression set covering this module, frozen
M0/M1, native loops, and STDP direction passed `35/35` with cache disabled and an external temporary
base. Only the existing PyTorch sparse-CSR beta warnings were emitted. No confirmation seed was
accessed.

Primary artifacts:

- `artifacts/t1-development-results-v2-audited.json`
- `artifacts/m2-binding-development-results-v2-frozen.json`
- `artifacts/m2-factor-development-results-v2-frozen.json`
- `artifacts/m3-predictor-development-results-v2-frozen.json`
- `artifacts/m3-binding-development-results-v2-frozen.json`
- `artifacts/m3-factor-development-results-v2-frozen.json`
