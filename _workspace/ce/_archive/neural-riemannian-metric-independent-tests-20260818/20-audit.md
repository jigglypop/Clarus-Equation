# Independent-test run audit

Status: COMPLETE
Gate: PASS
Audited snapshot: `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`, predecessor `40-final-report.md` and `artifacts/candidate-equation-registry.md`, plus `artifacts/check_counterexamples.py`.

## Re-audit verdict

All eight prior P1 findings are resolved in the revised stable snapshot.

1. `check_counterexamples.py` now constructs `C_{Q,1}=I Q I^T` and checks equality with `Q`, including a common nonzero ridge. It also executes the flat polar-coordinate cancellation check. Focused fixture result: `counterexample spot checks: PASS`.
2. The contract now scopes Tier A completion to each route's declared hypothesis, and explicitly excludes `H_G`-only A3/A4 from full-chain status.
3. A1 is now explicitly the smallest *proposed* full-chain design under stated assumptions. It includes target-specific intervention, negative-target/off-target and matched-rate controls, exclusion assumptions, sensitivity bounds, and retained direct structural-to-path mediation.
4. MICrONS is now labeled Tier B for a static bridge only and explicitly not `H_W`.
5. `10-sources.md` now contains a capability matrix with scale, identity, structural signal, activity signal, perturbation, longitudinal support, and claim tier, plus a separate technology inventory.
6. The common frame now freezes `Phi`, the `g -> (v,Q)` bridge, parameter budget, unconstrained `v,Q` alternative, and direct `W^s -> x` path before outcome access.
7. Physical folds `h`, baseline producers `P-h/P-W/P-D/P-C`, and induced deformation are operationally separated. Producer selection is pre-intervention only and includes placebo deformation and complexity-matched flat-pullback nulls.
8. The completion criterion now distinguishes documented counterexamples from executable spot checks, so the fixture's limited scope is no longer overstated.

## Remaining status notes

- `00-contract.md` is complete and its claim boundary is consistent with the route portfolio.
- `10-sources.md` separates measurement capability from theory evidence and assigns endpoint-only sources to static or Tier B scopes rather than to `H_W`.
- `11-math.md` contains the required chart laws, `h/g_0/g_t` decomposition, flat-pullback counterexample, rank-one sign/SPD condition, distance-versus-first-passage counterexample, non-unique `W -> g` counterexample, and same-source circularity warning.
- `12-routes.md` covers the required independent families, collapses dependent variants, states route-specific claim scopes, and qualifies A1's full-chain status by its identification assumptions.
- The predecessor E17 report and candidate registry remain consistent: E17 is retrospective activity-derived feasibility and does not establish structural `W^s`, physical folds, a nonconstant `g_0`, or the causal chain.

## Gate decision

`Gate: PASS`. No P0/P1 issue remains in the audited snapshot. The focused counterexample fixture passes. The result is documentation and route-design closure; it is not biological confirmation of the neural Riemannian hypothesis.

## Final-report audit: PASS

The completed `40-final-report.md` is consistent with the PASS lanes. It answers the pre-existing-fold question by separating physical surface metric `h`, baseline state metric `g_0`, and induced field `g_t`; it retains the flat-pullback and placebo controls; and it states that E17 did not measure folds, direct `W^s`, or a nonconstant `g_0`. The route ranking remains correctly scoped: A3 is an `H_G` precursor, A1/A2 are full-chain designs only under their stated exclusion and mediation conditions, and A5 is conditional on a declared field-evolution law. The current hypothesis status is appropriately unconfirmed: the result establishes a testable research program and retrospective activity-derived feasibility, not biological validation. No new P0/P1 issue was found.
