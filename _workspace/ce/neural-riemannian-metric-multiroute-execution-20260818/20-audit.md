# Preimplementation audit, revision 1

Status: COMPLETE  
Audit snapshot: `00-contract.md`, `10-sources.md`, `11-math.md`, and
`12-routes.md`, read without opening route outcomes.  
Audit date: 2026-08-18

## Gate decision

`Gate: PASS`

The revision resolves the previous P1 findings. Every frozen route now has a
single declared disposition, the physical/anatomical metric `h` is separated
from pre-period `g0` and post-state `gt`, and the flat-pullback and anatomy
controls are in the execution contract. The synthetic acceptance rule is now
machine-definite: G1 requires at least 18/20, G2--G6 permit at most 5/100
Holm-surviving metric-specific false positives, G3 curvature permits at most
1/20, and missing/nonfinite seeds fail rather than shrinking denominators.

## P0 findings

None.

## P1 findings

None remaining in the stable preimplementation snapshot.

The source identity correction for R-DANDI-37 is internally consistent: the
official 000037 record is an Openscope visual-cortex dataset and is therefore
`UNTESTABLE_MISSING_INPUT` for the frozen longitudinal M1 route. It is not a
substitute route and is not an access block. R-GRID-TORUS has an exact archive
hash, frozen code commit, and explicit REM/SWS intervals, and remains eligible
only for topology-metric dissociation. R-BCI is now one legal status,
`PARTIAL_DESCRIPTIVE`, with exact Dryad file identities/digests and a
documented HTTP WAF failure; the official derived-table path is not treated as
a raw-input metric test. R-CELEGANS is likewise a local structural fixture,
`PARTIAL_DESCRIPTIVE`, with a frozen local hash and unresolved source-object
license, not biological evidence.

## P2 findings and disclosures

### P2-PROTOCOL-001: BCI schema inspection after the original freeze

After the original contract/math freeze and before this implementation gate,
the parent opened only the first rows/headers of several official BCI derived
CSV files to determine their schema. Numeric values were visible. No
candidate, threshold, split, horizon, or endpoint was changed, and the BCI
route is now descriptive-only and excluded from confirmatory inference. This
is a transparent protocol deviation that must be copied into the execution
and final ledgers. No BCI numeric result may be used to select a model, tune a
threshold, define a split, or enter the Holm family.

### P2-FAMILY-001: keep BCI out of the confirmatory family in the executable

`11-math.md` describes the confirmatory family as eligible primary contrasts;
the route matrix correctly marks BCI `PARTIAL_DESCRIPTIVE`. The implementation
must materialize the family from final `ELIGIBLE` statuses and assert that BCI
is absent, rather than relying on prose interpretation. The same assertion
should exclude R-CELEGANS, R-E17-F2, and R-SYNTH as already specified.

### P2-SCOPE-001: preserve the narrow claim boundaries

Existing folds belong to physical `h` or explicit anatomical nuisance fields;
`g0` is fit only on the training/pre-period block; `gt` is compared to `g0`
through the declared relative-deformation statistics. E17's sessionwise
constant SPD candidates do not measure folded `h` or an independently observed
nonconstant `g0`. GRID-TORUS supplies topology-metric dissociation only; the
C. elegans route supplies a structural/synthetic fixture only; BCI supplies a
behavioral accessibility description only. None supplies
`Delta W^s -> Delta g -> Delta p`.

## Independence and circularity audit

The frozen estimands and route matrix correctly require animal/subject or
independently randomized circuit units, hold out the target endpoint, and keep
cells, spines, trials, windows, paths, and sessions nested. They explicitly
kill shared fitted `J,Q`, shared response trials, outcome-driven preprocessing,
identity-permutation survival, direct-dynamics/gain/noise ties, and
distance-to-first-passage substitutions. E17 F2 remains
`INELIGIBLE_DEPENDENT`; E17 F3/F4/F5 and E19/E15 remain descriptive where
chronology or identity is absent; MICrONS remains static partial descriptive.

No route outcome was used by this audit. The BCI schema inspection exception is
the sole post-freeze disclosure and is quarantined as described above.

## Route disposition audit

All 13 frozen routes have a pre-outcome disposition: R-SYNTH and R-GRID-TORUS
are eligible for their restricted computational claims; R-DANDI-37 and
R-ALLOPTICAL are untestable missing-input routes; R-BCI and R-CELEGANS are
partial descriptive routes; R-MICRONS is partial descriptive; E17 F3/F4/F5
and sleep E19/E15 are partial descriptive; E17 F2 is ineligible dependent.
The matrix provides a fallback or exact missing-input rule for every route.

## Referee-level verdict

The implementation gate is open for the frozen eligible estimator and
topology-metric routes, subject to the P2 disclosure and the machine assertion
that BCI cannot enter confirmatory inference. Passing the estimator does not
promote the biological theory. The portfolio still lacks a same-unit public
measurement of structural change, independently fitted `g0 -> gt`, and a
held-out trajectory endpoint, so the full chain remains untested.

Gate: PASS
