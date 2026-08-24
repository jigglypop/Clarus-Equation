# Status audit: equality, dimensionless mathematics, and alternative dimensions

Status: COMPLETE
Snapshot audited: revised `11-math.md`, `10-sources.md`, `12-routes.md`,
`artifacts/verify_equality_dimension_math.py`, and stabilized production
dimensionless implementation/tests, with the unchanged contract as scope
authority. Reported certificate: `OK`; focused pytest: `23 passed`; numeric
boundary spot-checks (nextafter, subnormal, and extreme ratios) also pass.

## Gate summary

Gate: PASS

The previous P0 domain error is closed: E4 now states that the form-degree
domain requires `n>=2` and that `dim Lambda^1=dim Lambda^2` has the sole
admissible solution `n=3`; the formal `n=0` continuation is explicitly
out-of-domain.  The revised artifact now exercises typed-zero targets,
SPD-versus-indefinite C3, positive-torus/constant-rank guards, the two-level
affine exception, and the three-level nonlinear failure.  The production
`log_equality_defect` now returns exact zero only for exact equality, uses a
finite `log1p` relative form for ordinary unequal positive inputs, and falls
back to a log-difference for extreme ratios; its zero-set and finite-output
P1 are closed by the boundary spot-checks.  No P0 remains.

## Claim status and findings

| ID | Status | Formal classification | P0/P1/P2 finding |
|---|---|---|---|
| E1 | supported | **Theorem**, with typed-zero and physical-kind ceiling | Character agreement proves only dimensional compatibility. `typed_equal` additionally requires the same physical kind/target type; equal characters alone do not identify energy with torque. Preserve typed-zero semantics and reject bare numerical equality across unlike dimensions. |
| E2 | supported with conditions | **Theorem** on stated domains; C3 is a conditional theorem | C2 requires positive finite nonzero `S`; `log_equality_defect` has exact-zero equality semantics and finite relative/fallback evaluation for unequal positive finite inputs. C3 requires symmetric positive-definite `Sigma` (or an explicitly invertible transformation law plus SPD); invertibility alone is false. Buckingham--Pi exhaustion is only positive/nonzero, rational-exponent, constant-rank/local. |
| E3 | supported with conditions | **Theorem** on fixed support/base measure; normalization rule is an **axiom/declared structure** | Affine defect changes with beta compensation preserve the finite family. The nonzero-beta, at-least-two-level premise is required for uniqueness; beta=0 and constant-one-level supports are degenerate controls. Nonlinear transforms preserve zero set/order but not generally the finite-beta family, with accidental one-/two-level exceptions. |
| E4 | supported | **Theorem** for Hodge and binomial classifications; cross-product result is a cited **theorem** | Domain correction is complete: `n>=2` for `(p,q)=(1,2)`, with `n=3` only. Keep signature/reality qualification: Lorentzian 3+1 real two-form self-duality is not available under the stated convention. Algebraic special dimensions do not select observed spacetime dimension. |
| E5 | supported with boundary cases | **Derived theorem** by power counting; conformality/marginality are **conditional classifications** | D=4 Yang--Mills and scalar phi^4, D=6 phi^3, and D=2 Newton marginality are engineering statements only. D=2/D=3 gravity exceptions must not be promoted to UV completion or physical existence. |
| E6 | supported as source map | **Empirical/source-grounded status table**, not a theorem | Kaluza--Klein, ADD, RS, string critical dimensions and M-theory are model/consistency routes. PDG/LHC values are model-, channel-, confidence-, luminosity-, and version-dependent constraints. No reviewed source establishes an extra macroscopic dimension. |
| E7 | supported with explicit ceiling | **Axiom/taxonomy plus falsifier registry** | The active CE branch remains `d=3,D=4`; `D_eff`, internal, compact, configuration/path, and spectral dimensions need distinct type/operational maps. Extra-dimensional existence remains **unproven/incomplete**, not a derived result. |
| E8 | supported | **Executable certificate plus production implementation/tests**; no physical existence claim | `reality_stone.clarus.dimensionless` now has typed equality, finite-output/overflow guards, SPD Mahalanobis validation, and beta compensation checks; `tests/test_dimensionless.py` covers the new boundaries. The certificate and 23 focused tests do not constitute evidence for extra-dimensional existence. |

## Priority findings

### P0

No open P0 remains.  The prior E4 domain finding is closed in the revised
math lane and executable certificate. Preserve the parent-claim
deletions/narrowings already identified by the
   lanes: no untyped-zero equality, no indefinite-covariance zero-set claim,
   no inference of physical spacetime dimension from Hodge/binomial/cross
   product coincidences, and no inference of extra-dimensional existence from
   model consistency or null searches.

### P1

The numeric zero-set/finite-output item for `log_equality_defect` is closed by
the production implementation, nextafter/subnormal/extreme-ratio spot-checks,
and 23 focused tests. Remaining P1 items are:

1. Keep the distinction between a field-theory power-counting result and
   physical existence/UV completion explicit in both ledger and paper.
2. Before any extra-dimensional existence claim, freeze geometry/topology,
   compactification or warp parameters, field localization, observable,
   likelihood/data set, confidence level, and 4D control.  The current source
   lane does not supply this bridge, which is appropriate for the present
   claim ceiling.
3. Keep the central-force stability derivation and CKM/toy reproduction
   derivations tagged as conditional theorem/calculation or empirical/toy
   ansatz, respectively; neither selects a physical dimension nor turns a
   reproduced value into a prediction without a fixed action and data bridge.

### P2

1. Improve source presentation/encoding artifacts (several source-lane symbols
   are rendered imperfectly) without changing the mathematical conclusions.
2. Extend binomial enumeration beyond the current finite certificate only if a
   later release needs a larger regression domain; the strict-unimodality proof
   already supplies the general theorem.

## Exact permitted central statements

The following are safe to carry forward: equality of nonzero quantities is
unit-covariant exactly when their dimension characters agree, followed by a
separate physical-kind/target-bundle check; typed zero is covariant only as the
zero section of its target; C2 and C3 are unit-invariant on their stated
domains with C3's SPD requirement; Pi groups locally exhaust rational-exponent
monomial invariants on the positive, constant-rank stratum; finite-beta Gibbs
families are invariant under affine defect reparameterization only with the
corresponding beta compensation under the nonzero-beta/nondegenerate-support
premise, with beta=0, constant-one-level, and accidental two-level exceptions;
Hodge/binomial/cross-product results are mathematical classifications; power
counting classifies engineering dimensions only; central-force and CKM/toy
reproductions retain their conditional or empirical tags; and extra-dimensional
routes remain model-dependent with constraints but no confirmed observation in
the reviewed sources.

## Recommended order and file scope

1. **Ledger first:** the smallest relevant dimensionless/formal-mathematics
   ledger, carrying forward the corrected E4 domain statement and preserving the E1--E8
   statuses above.
2. **Paper second:** the equality/pre-equality reader path and the
   dimension-uniqueness derivation, using the frozen ledger read-only.
3. **Code is now frozen for this run:** the focused extension is present in
   `reality_stone.clarus.dimensionless` and `tests/test_dimensionless.py`, with
   finite-output overflow guards. Do not encode extra-dimensional physical
   existence.

No cosmological-constant or dark-abundance file is in scope.  Existing dirty
cosmology/quantum work is unrelated and must be preserved.

## Referee disposition

Internal-only / arXiv-ready only after the ordinary ledger and paper freezes
and source metadata are reviewed. The formal research gate itself is PASS.
