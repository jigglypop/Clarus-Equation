# AGI V16 covariant metric flow: implementation

Status: COMPLETE

## 1. Implemented scope

[Definition] The production learner is
`reality_stone/python/reality_stone/clarus/covariant_metric_flow.py`. Its only
persistent semantic state is `CovariantMetricState.factor`, a canonical
lower-triangular factor with positive diagonal encoding $g=LL^T$. It does not
persist an independent metric copy, optimizer moment, replay buffer, role head,
or eligibility trace. For $d$ dimensions the semantic state therefore has
$d(d+1)/2$ degrees of freedom and one persisted dataclass field.

The public package exports these five production types:

- `CovariantMetricConfig`;
- `CovariantMetricState`;
- `RouteChoice`;
- `MetricFlowCertificate`;
- `CovariantMetricFlow`.

The implementation also repairs the inherited V15 numeric obligations in
`unified_metric.py`, registers the V16 dimensionless residual and normalized
regret in `dimensionless_checker.py`, and adds focused regression tests. It does
not add raw sensory encoding, delayed credit, learned semantic OOD, continuum
geometry, biology, cosmology, or an unrestricted AGI controller.

## 2. Metric-flow update

[Derived] The public `update` method implements equation (V16.1) through a
positive factor congruence. It computes the stable log prediction and normalized
direction from binary mantissa/exponent dot products, constructs an orthogonal
basis with that direction as its first column, scales only that congruence
direction by $e^{-\eta r/2}$, and restores the unique lower-triangular
positive-diagonal representation with QR.

This route avoids evaluating the dangerous subtractive form of

$$
g^+=g+\frac{e^{-\eta r}-1}{p}(gx)(gx)^T
$$

when $e^{-\eta r}-1<0$. At $\eta=1$, the first factor column uses a scaled
square-root ratio rather than first forming $c/p$. Accepted states have finite
entries and strictly positive diagonal; nonrepresentable predictions, factors,
metric entries, route costs, or updates raise an explicit exception.

[Implementation boundary] The mathematical rank-one congruence has $O(d^2)$
structure. This reference implementation deliberately performs a full QR to
canonicalize the stored factor, so its numerical retriangularization is
$O(d^3)$. No $O(d^2)$ production-runtime claim is made.

## 3. Readouts and state boundary

[Definition] `predict` evaluates $x^Tgx$, `residual` evaluates the dimensionless
$\log(p/c)$, and `route_costs` sums quadratic costs over the displacements of a
candidate route. `choose_route` applies the preregistered
$64\epsilon\max(1,\max_k|p_k|)$ tie tolerance and returns the lowest-index
representative while retaining every declared minimizer.

`snapshot` and `from_snapshot` copy only the canonical factor. The certificate
reports the proved exact-arithmetic properties while explicitly leaving raw
perception, delayed credit, continuum geometry, fixed-rate noisy point
convergence, and AGI evidence false.

## 4. Inherited numeric repairs

[Implementation result] The V15 `UnifiedMetricCore` repairs implement the
contract's R1--R4 obligations:

- strict Dijkstra relaxation owns the representative predecessor;
- tie counting runs afterward on a distance-oriented DAG and cannot rewrite
  the representative path;
- reconstruction has a visited-set and $N-1$-hop guard, while source-to-self is
  uniquely represented;
- local quadratic forms and edge lengths use scaled product decompositions;
- endpoint averaging and metric symmetrization avoid adding two huge finite
  operands before scaling;
- nonrepresentable distances and affine transports raise explicit exceptions;
- the surprise Boolean is decided in log space independently of a displayed
  ratio that may saturate to zero or infinity;
- target comparison uses a relative tolerance without an absolute unit floor.

R5 is implemented by the V16 factor/congruence update described in Section 2.
The registered scalar extremes and near-equality residual receive dedicated
tests.

## 5. Evaluator and contamination controls

`artifacts/run_v16_benchmark.py` implements development-rate selection and the
sealed confirmation protocol independently of the production learner's route
helper. Development selected $\eta=0.4$ for V16, $0.2$ for the additive
full-matrix comparator, and $0.05$ for the conformal control.

[Observed procedure] Before confirmation, SHA-256 values were frozen for the
contract, production module, public export file, evaluator, development result,
and selected-rate file. `artifacts/confirmation-opened.json` was created with
exclusive creation before seed access. The evaluator then wrote
`artifacts/confirmation-results.json` once; a second opening is rejected.
Neither mathematical proofs nor numeric unit gates are encoded as booleans in
that confirmation JSON, so its `learning_chart_closed_loop_pass` field is only
the conjunction of G-LEARN, G-CHART, and G-CLOSED-LOOP.

## 6. Implementation artifacts

- production: `reality_stone/python/reality_stone/clarus/covariant_metric_flow.py`;
- public export: `reality_stone/python/reality_stone/clarus/__init__.py`;
- inherited repairs: `reality_stone/python/reality_stone/clarus/unified_metric.py`;
- dimension audit registration:
  `reality_stone/python/reality_stone/clarus/dimensionless_checker.py`;
- unit tests: `tests/test_covariant_metric_flow.py`,
  `tests/test_unified_metric.py`, and `tests/test_dimensionless.py`;
- seal/evaluator tests: `tests/test_v16_benchmark.py`;
- evaluator and results: `artifacts/run_v16_benchmark.py`,
  `artifacts/development-results.json`, `artifacts/confirmation-manifest.json`,
  `artifacts/confirmation-opened.json`, and
  `artifacts/confirmation-results.json`.

