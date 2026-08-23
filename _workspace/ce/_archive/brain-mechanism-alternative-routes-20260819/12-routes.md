# Frozen alternative-route map

Status: COMPLETE

## Route table

| Route | Changed method | Primary question | Strongest falsifier | Maximum allowed claim |
|---|---|---|---|---|
| M0 | bounded supervised SVD rank sweep | What recurrent rank/structure is sufficient? | equal-norm random low-rank write | supervised capacity threshold |
| M1 | block-delayed three-factor eligibility | Can local eligibility plus scalar modulation acquire binding? | sign/time shuffle and eligibility reset | synthetic three-factor acquisition |
| M2 | positive-minus-negative contrastive phases | Can phase contrast acquire binding/transfer? | identical phase and target shuffle | supervised contrastive acquisition |
| M3 | native next-state prediction error | Can transition prediction drive a useful recurrent write? | transition shuffle and predictor-only arm | predictive plasticity; binding only if separately passed |
| G1 | randomized signed edge intervention | Does a known weight change induce a specific frozen metric and trajectory change? | matched scrambled edge, gain, and noise arms | simulator-level interventional chain |
| G2 | fixed-weight drift/noise factorial | Does the metric add prediction beyond direct dynamics? | equal-budget direct state-space model | operational metric utility |
| G3 | randomized successful local learner | Does metric change mediate independently learned recall? | shuffled contingency and norm-matched random weight | simulator-level learned mediation |
| C1 | frozen predicted-risk action gate | Is self-prediction used before action to reduce matched-coverage loss? | same-coverage random and shuffled-risk gates | task-bounded prediction-guided control |
| S1 | matched SCC feedback lesion | Does declared feedback structure causally support recovery? | degree/mass/spectrum-matched non-feedback lesion | functional role of declared recurrence |

## Rejected shortcuts

- Retuning the failed per-tick scalar STDP route is not a new method and cannot enter this universe.
- Injecting a NumPy association output during recall violates native independent readout.
- Using M0 as evidence for local learning or G3 mediation is prohibited.
- Choosing a metric, SCC threshold, lesion, risk threshold, or endpoint after confirmation outcomes is
  inspected invalidates that route.
- Weight drift alone, metric distance alone, prediction MSE alone, or SCC membership alone is never a
  task-level success.
- A PFC pseudopopulation covariance result cannot fill a missing runtime causal arrow.

## Implementation order

1. **M0 diagnostic first.** Determine the minimum supervised rank and recurrence term needed for the
   already confirmed task. This bounds the structural problem before testing learning rules.
2. **M1 next.** It is the closest biologically motivated replacement for the failed Route A while
   changing the credit timing and providing decisive scalar-gate lesions. Its gate is the frozen
   target-blind `+1.0` block-end clock pulse; it is not reward learning. A native transient-state
   reset between phases makes eligibility the only permitted temporal bridge.
3. **M2 and M3 separately.** M2 tests phase-based supervised acquisition. M3 first tests prediction,
   and reaches memory or factor claims only through additional gates.
4. **G1 before G2/G3.** Direct randomized weight intervention first validates the estimator and null
   behavior. G2 then asks whether geometry adds anything beyond direct dynamics. G3 waits for a local
   learner to pass.
5. **C1 uses but does not modify the frozen Loop-10 result.** It is implemented as an isolated
   benchmark so prediction evidence cannot be retroactively re-labelled as control.
6. **S1 is isolated last.** Its effective-edge rule and matched lesion generator freeze independently
   of memory and control endpoints.

One implementation owner changes code sequentially. Read-only audits occur only after each stable
route snapshot. Development failures remain first-class results; no confirmation seed is opened to
repair a method.

## Immediate selected slice

The first implementation slice is M0 plus M1. It is bounded, directly addresses the currently known
failure, and supplies the structural rank ceiling and the strongest local-credit alternative before
more elaborate geometry or control code is introduced. Completing this slice does not complete the
whole run; the remaining frozen routes stay pending.
