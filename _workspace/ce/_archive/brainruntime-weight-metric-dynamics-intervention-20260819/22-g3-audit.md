# G3-D diagnostic audit

Status: PASS

Gate: the frozen non-mediation diagnostic contract passed independent mathematical and adversarial
audits. Implementation is authorized.

Original learned mediation is permanently blocked: treatment changes W, and both response summary
and recall are post-treatment functions of W. G3-D is a separate falsification-oriented diagnostic
with independent probes, continuous recall, matched M1 controls, and a calibration-null W lesion.

Closed audit points:

- response and recall contrasts are paired within the same adverse condition before correlation;
- the simultaneous bootstrap takes separate family minima for mean response contrast, mean recall
  contrast, and same-condition correlation;
- AIRM uses the declared float64 symmetric-eigendecomposition convention with no silent clamp;
- every calibration-null candidate starts from a fresh matched post-W snapshot and is ranked by the
  ordered six-horizon response stack;
- a null-lesion falsifier additionally requires the declared `C` summary itself to remain within the
  frozen AIRM tolerance;
- the coordinate-permuted structural arm is a fresh zero-gate branch with explicit `P W P^T`
  reconstruction, no clipping, no tensor alias, and hash audits.

Implementation must use an explicit 48-by-48 permutation matrix and keep the three bootstrap minima
separate because their units differ.

The first implementation audit required four additional compliance repairs before development:
full-default M1 parity at excluded seed 97699, explicit temporal-store zero evidence, direct strict
SPD validation of both AIRM inputs, and an exact unique-stage seed validator. These repairs are now
implemented. A later adversarial audit found a confirmation-library bypass; confirmation now
requires library-level manifest verification and parses/recomputes the bound development artifact
before either execution or summary. The repaired stable snapshot awaits final re-audit.

A final low-level audit found that public single-seed/range helpers could still compute official
confirmation seeds. Those helpers now reject confirmation overlap before any circuit runs; only the
verified stage path reaches the private unchecked executor.

The first development execution is not admissible evidence. Every lesion install missed the exact
delta-reconstruction tolerance solely at the float32 addition boundary (256/256 residuals between
`1.64868e-7` and `1.88207e-7`). Seeds `97701..97716` are retired after outcome inspection. The
proposed repair freezes each float32-representable target before installation, retains the original
Gaussian direction/norm and all scientific thresholds, and moves development to untouched
`97801..97816`. Independent amendment audit is required before code changes or rerun.

The fixed native install headroom is `.250001`, while the actual candidate remains gated at
`.25 +/- 1e-6`; this prevents a permitted float32 norm overshoot from being clipped. Replacement
bootstrap seed `97898` is frozen before any replacement outcome.

The amended implementation passed 8/8 focused tests and stable independent audits. It constructs
and audits the representable target on each fresh branch, rejects nonrepresentable candidates before
calibration, and blocks both retired and confirmation seed ranges at every public raw entrypoint.
Fresh replacement development `97801..97816` is authorized; confirmation remains sealed.

Final stable-snapshot audits found no remaining P0/P1. Focused validation passed 7/7 tests,
including full-default M1 parity, zero-store probes, strict AIRM, exact stage units, and all public
confirmation-seal bypasses. Development execution is authorized; confirmation remains sealed.
