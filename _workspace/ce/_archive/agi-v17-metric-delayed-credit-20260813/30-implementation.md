# AGI V17 metric-only delayed cue: implementation

Status: COMPLETE

## 1. Implemented scope

[Definition] The production candidate is
`reality_stone/python/reality_stone/clarus/homogeneous_signed_cue.py`. It
implements the registered homogeneous escape with one persistent dataclass
field, `HomogeneousSignedCueState.factor`, encoding one
$G\in\operatorname{SPD}(d+1)$ by a canonical lower-triangular factor with
positive diagonal. The public package exports:

- `HomogeneousSignedCueState`;
- `SignedCueReadout`;
- `HomogeneousSignedCueCertificate`;
- `HomogeneousSignedCue`.

The implementation also exposes the strict original-space V16 state as a
no-go control. It does not implement a general recurrent controller, a learned
credit-assignment rule, an infinite SCC runtime, biological dynamics,
cosmology, or AGI.

## 2. Frozen homogeneous write and readout

[Derived] For a dimensionless public unit reference $u$ and cue sign
$s\in\{-1,+1\}$, `lift_cue` forms

$$
z_s=(su,1).
$$

Starting from $G_0=I_{d+1}$, `write_cue` delegates the fixed analytic write to
the V16 positive-factor metric flow with $\eta=1$, prewrite prediction $p=2$
and observed cost $c=4$. In exact arithmetic the result is

$$
G_1=I_{d+1}+\frac12z_sz_s^T. \tag{V17.I1}
$$

The method rejects a state/reference pair whose prewrite prediction is not the
registered value within a declared binary64 tolerance. This prevents the
one-cue fixture from being silently presented as a general repeated-memory
update.

`lift_action` forms $y_a=(au,-1)$ in fixed action order $(-1,+1)$.
`terminal_costs` evaluates the two quadratic costs and `readout` selects the
unique lower-cost action. The independent exact formula is

$$
y_a^TG_1y_a=2+\frac12(sa-1)^2, \tag{V17.I2}
$$

so the correct and wrong costs are respectively 2 and 4 and the exact margin
is 2. This construction writes the cue immediately and later recalls its sign;
it does not infer causal credit from the delayed reward.

## 3. State and chart boundary

[Definition] At $d=3$, the augmented factor represents ten independent real
coordinates, compared with six for the original-space metric. The four added
coordinates are the three-component spatial--homogeneous cross block and the
last scalar. In block form,

$$
G=\begin{pmatrix}Q&b\\b^T&\gamma\end{pmatrix},
$$

and the cue-odd memory lives in the covector block $b$. Packaging $b$ and
$\gamma$ inside one factor field does not make this a strict D1 metric-only
state.

[Axiom: model choice] The last coordinate is a declared homogeneous splitting.
For a spatial chart $J\in GL(d)$, the only claimed chart action is

$$
A=\operatorname{diag}(J,1),\qquad
G\mapsto A^{-T}GA^{-1}. \tag{V17.I3}
$$

The implementation accepts a transported initial metric through
`make_state_from_metric`; it does not reset that metric to identity, reproject
the result, or renormalize $Ju$. The certificate explicitly leaves general
$GL(d+1)$ semantics false. Public $u$ is a transient input supplied again at
readout, not a hidden persistent field.

`snapshot` and `from_snapshot` preserve the factor exactly. The certificate
reports one persistent field, ten ambient real coordinates, four added
coordinates and zero optimizer-state fields, and records the narrow scope as
negative flags for general delayed credit, infinite-SCC intelligence growth,
biology, cosmology and AGI evidence.

## 4. Strict no-go controls

[Derived] `strict_write` uses the inherited original-space V16 factor update.
For binary64 reproducibility it first chooses a canonical representative of
the projective pair $\{x,-x\}$, normalizes zero spellings, and then performs
the sign-even update. This transient canonicalization stores no orientation;
it ensures that the two exact-theory branches also have byte-identical factor
serializations rather than QR-branch or signed-zero spelling differences.

`serialize_strict_state` serializes every factor entry by exact hexadecimal
spelling. `strict_terminal_distribution` returns the same frozen action law
$(1/2,1/2)$ for both signs. The evaluator forms sign-independent,
permutation-equivariant aggregate controls at the preregistered finite sizes
$N\in\{1,2,4,8,16,64\}$. These controls test the finite no-go; they are not an
implementation of a countably infinite or infinite-depth system.

## 5. Evaluator and contamination controls

[Observed procedure] `artifacts/run_v17_benchmark.py` fixes $\eta=1$, $c=4$,
the two embeddings, all thresholds and both seed ranges. It computes reference
costs and scored decisions independently of the production `readout` helper.
Development used seeds 1,719,000--1,719,063 only and performed no
hyperparameter search.

Before seed 1,720,000 was accessed, the contract, development result,
production module, public export and evaluator were SHA-256 sealed. The
confirmation path binds the canonical repository root and imported production
module, rejects path traversal and nonexact manifest coverage, verifies the
fixed protocol, exclusively creates an opening receipt, verifies the manifest
again after evaluation and exclusively writes the result. The existing receipt
or result makes a second opening fail closed. The first result stores all 256
per-seed paired summaries.

## 6. Numeric safeguards and tests

[Implementation result] Focused tests cover exact and near-axis references,
both signs, chart singular-value endpoints $0.25$ and $4$, exact snapshot
round trips, one-field introspection, public exports, nonfinite and zero input
rejection, wrong state types, certificate boundaries and the killing ablation
that deletes the homogeneous coordinate. The ablation restores sign-paired
spatial metrics and a tied quadratic decision.

Evaluator tests cover deterministic draws, independent scoring, lossless
signed-zero serialization, exact manifest coverage, traversal rejection,
canonical import binding, protocol tampering, receipt-before-seed ordering,
second-open rejection, forged-capability rejection, midrun mutation and
failure-consumed capabilities. V17 conditional information and the normalized
lift margin are registered as dimensionless quantities in
`dimensionless_checker.py` and checked in `tests/test_dimensionless.py`.

## 7. Implementation artifacts

- production: `reality_stone/python/reality_stone/clarus/homogeneous_signed_cue.py`;
- public export: `reality_stone/python/reality_stone/clarus/__init__.py`;
- inherited factor flow:
  `reality_stone/python/reality_stone/clarus/covariant_metric_flow.py`;
- dimension audit:
  `reality_stone/python/reality_stone/clarus/dimensionless_checker.py`;
- production tests: `tests/test_homogeneous_signed_cue.py`;
- evaluator tests: `tests/test_v17_benchmark.py`;
- evaluator and records: `artifacts/run_v17_benchmark.py`,
  `artifacts/development-results.json`, `artifacts/confirmation-manifest.json`,
  `artifacts/confirmation-opened.json` and
  `artifacts/confirmation-results.json`.

