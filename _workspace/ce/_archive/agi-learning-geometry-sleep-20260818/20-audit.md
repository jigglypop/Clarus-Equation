# CE status audit: learned geometry and sleep rearrangement

Status: COMPLETE
CE_RUN: agi-learning-geometry-sleep-20260818
Snapshot: contract and lanes are read-only for this audit; active documents were checked at their current line locations.

## Gate

Gate: PASS

The mathematical lane closes `LGS-T1` and `LGS-T2`, and supplies complete counterexamples to `LGS-N1`, `LGS-N2`, and `LGS-N3`. The source ledger is conservative: it records 22 deduplicated sources, does not establish the full `Delta W -> Delta g -> Delta x(t)` chain, and explicitly leaves several sample/data-access details to later verification. The scoped re-audit of the active documents found the required parent wording corrected: the sleep schedule is explicitly a software model choice, the scalar contraction is conditional, NREM/REM mechanisms are hypotheses, catastrophic-forgetting language is predictive, and `W -> g` now has an explicit typed-object and `Phi_q` boundary. No P0 remains in the audited scope. This is a documentation closure gate, not a runtime or scientific-release gate.

## Claim audit

| Claim | Status | Priority | Evidence and required disposition |
|---|---|---:|---|
| `LGS-T1` | [정리] | -- | `11-math.md` (1) proves the one-edge APSP identity for finite directed positive-cost multigraphs, including unreachable pairs and parallel arcs. `verify_lgs_math.py` reports exhaustive finite checking. Preserve as a typed shortest-path theorem; do not call it a metric theorem. |
| `LGS-T2` | [정리] | -- | `11-math.md` (2)--(4) gives the necessary-and-sufficient strict distance-change set and separates tie-induced optimal-path changes. Preserve the distinction between distance values and route sets. |
| `LGS-N1` | 삭제/경계 대체 | P0 | Topology, direction, multiplicity, and transition-dynamics counterexamples in `11-math.md` §P0 refute structural equivalence of shortcut addition and a metric deformation. Delete any such parent wording; retain only cost-observable operational emulation under a declared protocol. |
| `LGS-T3` | [정의] | -- | `11-math.md` defines operational equivalence only for fixed state/action set, cost observable, tie rule, policy class, and intervention protocol. It does not imply trajectory-law or topology equivalence. Keep with this scope. |
| `LGS-N2` | 삭제/경계 대체 | P0 | `11-math.md` §P0 shows activity/context dependence and a `GL(r)` gauge non-identifiability; without a declared `Phi_c`, `W -> g` is not even a defined map. In `15_Equations.md:148-152`, state that `d_g` is a chosen/modelled input and not uniquely inferred from `W`; distinguish Riemannian SPD distance from directed shortest-path quasi-distance. |
| `LGS-H1` | [미완성] | P1 | `11-math.md` claim table and `12-routes.md` R1 require a frozen, training-only `Phi_c`, matched raw-W/activity/latency baselines, held-out contexts and intervention. No source directly supplies the full longitudinal chain (`10-sources.md` Verification notes). Do not promote to result. |
| `LGS-H2` | [미완성] | P1 | `11-math.md` (5) supports only the declared critical-path/fixed-delay model; `12-routes.md` R2 requires independent calibration and join/parallel intervention. Keep as a testable hypothesis, not a universal cognition-time law. |
| `LGS-H3` | [미완성] | P1 | `12-routes.md` R3 requires usage dose, matched untrained pairs, and global-scaling/exposure controls. No source ledger entry measures this effective-cost change directly. |
| `LGS-H4` | [미완성] | P1 | `12-routes.md` R1 requires context-conditioned cost/gain intervention on fixed anatomy. Existing evidence measures representation or communication changes, not a unique neural geometry (`10-sources.md` E01--E03, Verification notes). |
| `LGS-H5` | [미완성] | P1 | `12-routes.md` R4 requires sleep/wake and time-awake controls, local/global estimator declared in advance, and replay coupling ablation. E13--E20 support narrower sleep, replay, synchrony, or representation results, not selective `W -> g` rearrangement. |
| `LGS-N3` | 삭제/경계 대체 | P0 | Delay, integration overhead, and parallel critical-path counterexamples in `11-math.md` §P0 refute universal `L_effective/v_neural`. Replace by equation (5) and state the serial zero-overhead assumptions needed for reduction to `L/v`. |
| `LGS-N4` | 삭제/경계 대체 | P1 | The fixed serial claim about NREM noise-curvature removal and REM recombination is not established. `12-routes.md` R4 explicitly keeps local asynchronous maintenance plus periodic synchronization as a competing model. In `1_AGI.md:112-114` and `3_Sleep.md:143-220`, retain the software schedule as a design hypothesis and remove biological/causal wording. |
| `LGS-H6` | [미완성] | P1 | Contract §6.3 and `12-routes.md` require independent splits for learning, context, and sleep, matched baselines, primary endpoints, and kill tests. The source ledger explicitly says no record measures all three links. Keep as an integrated preregistered prediction only. |
| `LGS-X1` | 활성 제외 | -- | `00-contract.md` excludes the AGI-sufficiency bridge, and `1_AGI.md:15,39,61` correctly says the current system is not AGI and that narrow synthetic results are not AGI evidence. Preserve the exclusion. |

## Active-document correction scope (verified complete)

The following required corrections were verified in the current stable diff; no runtime code change was required or authorized by this audit.

1. `docs/7_AGI/3_Sleep.md:48,58-64`: separate the mathematical toy contraction from the biological sleep-deprivation result. The cited sleep evidence does not establish the CE contraction, its 2--3-cycle interpretation, or the causal explanation.
2. `docs/7_AGI/3_Sleep.md:143-178,190-220`: label NREM curvature flattening and REM non-selected-path recombination as implementation hypotheses. Remove language implying that these are established neural mechanisms, fixed serial algorithms, or literal curvature/path quantities. Preserve the existing holdout/ablation caveats.
3. `docs/7_AGI/3_Sleep.md:310-320`: replace “naturally solves catastrophic forgetting” and “identical to the brain” with the preregistered prediction plus baseline/ablation requirement. The `4.87/26.2/68.9` numerical recurrence is a stated scalar map, not evidence of model or brain convergence.
4. `docs/7_AGI/3_Sleep.md:332-361`: mark the contraction inequality as conditional on the explicitly defined state map and assumptions; remove the inference from human sleep deprivation to proof of the AI schedule. Retain the bounded-residual versus wake-only statement only as a kill-tested model prediction.
5. `docs/7_AGI/1_AGI.md:110-116`: keep the schedule as a CE-AGI design proposal and explicitly point to `LGS-N4/H5/H6` as unclosed. Do not describe the NREM/REM sequence as a biological algorithm.
6. `docs/7_AGI/15_Equations.md:148-168`: add the typed-object boundary from `11-math.md`: `d_g` is a chosen SPD/Riemannian distance or other declared cost object; it is not uniquely determined by raw `W`, and directed graph costs are not automatically Riemannian metrics. Any `W -> g` map must name `Phi_c(W,A,c)` and its measurement protocol.

These edits are textual scope corrections and are now present. The valid T1/T2 mathematics and operational R1--R4 experimental routes remain intact.

## Source-access caveats

`10-sources.md` is a primary-source ledger, not a proof of the CE hypotheses. It records 22 unique citations after deduplication. E02, E05, E06, E07, E10, E12, E15--E20 and parts of E11 have explicitly noted sample-size, full-text, source-data, or code-access checks still required before quantitative reuse. E01--E22 collectively support narrower representation, connectivity, sleep, replay, and synchrony observations, but none measures all three links in `Delta W -> Delta g -> Delta x(t)`. The audit therefore assigns no empirical claim a stronger status than [미완성] without the route-specific controls.

## Implementation allowance

Allowed after this Gate: the small proof/verification fixture already named by the contract, focused tests for T1/T2, and the six verified documentation corrections above. Not allowed under this audit: changing runtime behavior, promoting any H-claim to [정리] or [산출], asserting a universal neural metric, or claiming an AGI bridge. The Gate is PASS for the audited documentation scope; empirical H-claims remain [미완성] until the route-specific controls and data are executed.

## Re-audit evidence

The stable `git diff` for `docs/7_AGI/1_AGI.md`, `docs/7_AGI/3_Sleep.md`, and `docs/7_AGI/15_Equations.md` was inspected after the corrections. The diff removes the universal/bootstrap and biological-algorithm wording, labels the affine contraction and ratio table as conditional toy mathematics, converts NREM/REM claims to model choices and hypotheses, adds matched-control/kill-test language for continual learning, and adds the directed-cost versus SPD/Riemannian type boundary plus `Phi_q(W,A,tau,q)`. A repository search found no remaining audited parent wording for structural shortcut/metric equivalence, universal `L_effective/v_neural`, or biological NREM/REM identity in these three active documents.
