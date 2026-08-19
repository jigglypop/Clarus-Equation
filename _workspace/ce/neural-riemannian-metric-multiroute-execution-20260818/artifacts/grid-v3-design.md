# Grid-Torus v3 design

Status: FROZEN_PRE_OUTCOME

For each rat-module, a seeded random 20% of 10 s wake blocks forms C. A scaler
and six-component PCA fit C only. Wake's remainder is split A/B; REM and SWS
are independently split A/B. C enters no endpoint. This disclosed pre-outcome
amendment replaces the earlier random-half rule after chart-dependence review.

The same C-fitted chart is used for all states and both roles. The primary
contrast uses topology A and mobility B; the swap uses topology B and mobility
A. Topology requires two finite cosine-Ripser H1 lifetimes. Mobility is the
SPD precision `(C_delta + lambda R_C)^-1` with increments only within blocks.

Per-state wake contrasts and same-state A/B references are reported separately
for symmetric normalized padded-lifetime topology distance, defined as
`||a-b|| / max((||a||+||b||)/2, eps)`, and affine-invariant SPD distance.
`dissociation_compatible` is only a one-split heuristic: both roles
need topology ratio <= 1 and metric ratio > 1. It has no p-value or population
meaning, proves neither curvature nor Riemannian geometry, and observes neither
structural W nor physical fold geometry h.
