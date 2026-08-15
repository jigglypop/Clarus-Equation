# AGI V17 metric-only delayed cue: scored validation

Status: COMPLETE

## 1. Decision structure and verdict

The two V17 decisions have different meanings and are evaluated separately:

$$
\begin{aligned}
\text{METRIC-ONLY NO-GO CLOSED}
&=\text{G-MATH}\land\text{G-DIMENSIONLESS}
\land\text{G-STRICT-NO-GO}\land\text{G-NUMERIC},\\
\text{HOMOGENEOUS LIFT NARROW GO}
&=\text{H1 implemented}\land\text{G-LIFT}\land\text{G-NUMERIC}.
\end{aligned} \tag{V17.V1}
$$

Every registered conjunct passes. The exact run verdict is therefore:

$$
\boxed{\text{V17 METRIC-ONLY NO-GO CLOSED}}
$$

and

$$
\boxed{\text{V17 HOMOGENEOUS LIFT NARROW GO}}.
$$

These are compatible: the first applies to a strict original-space
$g\in\operatorname{SPD}(3)$, while the second adds a homogeneous splitting and
four ambient real coordinates. The AGI decision remains `AGI STOP`; `AGI GO`
is not authorized.

## 2. G-MATH

[Theorem] Full $GL(d)$ covariance includes the chart $J=-I$. Because this
chart fixes every covariant metric but sends $x$ to $-x$, every admissible
fixed-seed update obeys

$$
U(g,-x,c)=U(g,x,c). \tag{V17.V2}
$$

The registered pointwise fixed-seed condition extends this equality to allowed
randomized updates and then to their averaged laws.

[No-go theorem] Under D1--D4, the two sign branches use the same public
reference, initial state and whole seed realization. Equation V17.V2 gives the
same post-cue state, and induction through later sign-independent inputs gives
the same terminal decision state. A common terminal policy therefore has
balanced accuracy exactly $1/2$ and expected 0--1 regret $1/2$.

[No-go theorem] The same equality induction closes every finite causal event
depth and finite component count satisfying D5. The countable extension is a
theorem only for a declared measurable countable product/trajectory system or
projectively compatible finite laws with a measurable terminal kernel. An
undefined infinite event-depth output is not covered.

[Theorem] An exact solver needs
$H(S\mid G_T,U)=0$ and $I(S;G_T\mid U)=1$ bit for the balanced task. This is a
conditional two-class information separation, not a marginal-information,
coordinate-minimality or exact-real capacity theorem.

[Theorem for registered fixture] The independent homogeneous calculation gives
costs 2 and 4 and exact margin 2. Under
$A=\operatorname{diag}(J,1)$, transporting the initial state, write and actions
preserves both costs without reprojection. The math lane has no open P0 or P1,
so `G-MATH PASS`.

## 3. G-DIMENSIONLESS and G-NUMERIC

[Derived] All synthetic cue/action coordinates, the homogeneous coordinate,
entropy counts, $p/c$, $\log(p/c)$, loss and regret are declared
dimensionless. The checker registers
$I_{\mathrm{V17}}=H(S\mid U)-H(S\mid G_T,U)$ and the normalized action margin

$$
\delta_{\mathrm{V17}}
=\frac{c_{\rm wrong}-c_{\rm correct}}{c_{\rm correct}}.
$$

Both checks pass. This is dimensional consistency only, not physical evidence.
Therefore `G-DIMENSIONLESS PASS`.

[Numerical result] The focused V17, V16/V15 and dimensionless suite passed
`100 passed in 15.21s`. It includes
`tests/test_homogeneous_signed_cue.py`, `tests/test_v17_benchmark.py`,
`tests/test_dimensionless.py`, `tests/test_covariant_metric_flow.py` and
`tests/test_unified_metric.py`. The 16-file SCC/metric-related expanded slice
passed `337 passed in 23.66s`.

Ruff passed the V17 production module, evaluator, tests and dimensionless
changes. The package export file passed with the repository's pre-existing
F401 findings ignored; those eight inherited import warnings were not created
by V17. Compileall and the post-confirmation diff/integrity check passed. The
CE contract, lanes and formal gate hooks also passed. A repository-wide run was
attempted with a 600-second limit. It reached 73% and displayed inherited
failures/errors before timing out, so it produced no valid final total. No
repository-wide green claim is made; the scored V17 and related slices above
are the completed regression evidence.

The focused tests include every G-NUMERIC killing case: axes and near axes,
both signs, chart endpoints, exact snapshots, finite/zero rejection,
state-field introspection and deletion of the homogeneous coordinate. Public
results are finite in the registered binary64 domain. Therefore
`G-NUMERIC PASS`.

## 4. Sealed confirmation integrity

[Observed procedure] The fixed confirmation block 1,720,000--1,720,255 was
opened once after five bound artifacts were sealed. The three record hashes
are:

| Record | SHA-256 |
|---|---|
| confirmation manifest | `898ab27369cd5580fc0b7f67f44fe048bfbf6361d55d574ce652b9ef571a63d1` |
| opening receipt | `2022f4778ae47e49627913442de61064234586f707762dea12678291f0a81ed1` |
| confirmation result | `35324a7fe1f4570a5d66c3cca6ed65298191c1c651eec73cdec24dbca677e01f` |

The result reports `manifest_verified: true`; every current bound-file hash
matches the manifest. The receipt records the frozen protocol before seed
access, and evaluator tests verify exclusive creation and fail-closed second
opening.

| Bound artifact | Sealed SHA-256 |
|---|---|
| `00-contract.md` | `51e662cf504991d5241adfa1a7e625fdf96105d5d29b12717ef989f1385df487` |
| `artifacts/development-results.json` | `5fbcbab1a5c0570e795686fab50a42bce905527cc10edadaa266de5b1bfba4dd` |
| `artifacts/run_v17_benchmark.py` | `4e4575493d7af252c0f8791a2949e4ee773a8006b2a9c2bb17cfee3659480c5e` |
| `reality_stone/python/reality_stone/clarus/__init__.py` | `d54692f9fe85cfb8cf772c6f69ff4d014b6b7502d2dabd687a5de0f02bee915f` |
| `reality_stone/python/reality_stone/clarus/homogeneous_signed_cue.py` | `5e8de35dda08d238a6f9fbdcf68a5377f18a7db5b7008f2f50a6c8d56fcb72d3` |

[Numerical result] An independent read-only rescore used only the stored
per-seed JSON; it did not import or call the evaluator, confirmation function
or seed helpers. It found 256 unique consecutive seeds, exactly 1,720,000
through 1,720,255; all 256 strict pairs; both lift signs for 512 total
branches; and no missing registered ensemble size. Strict state and ensemble
serializations and their SHA-256 values were internally consistent, the
recomputed summaries matched the reported summaries, and all 17 registered
gate booleans were true. Open P0/P1 findings were 0/0. This recomputation did
not reopen or regenerate the block.

[Incomplete: integrity boundary] The result JSON does not contain an internal execution
timestamp, and the receipt/result have no external signature or public hash
anchor. Filesystem modification times put manifest before receipt before
result, but that ordering is procedural evidence rather than external proof.
Per-seed entries store scored costs and defects, not raw lift factors or
certificate objects; therefore the independent audit rederived scored gates
from per-seed data but cross-checked metric-state finiteness and the
ten-coordinate certificate against the sealed production source and aggregate
record rather than reconstructing them from raw per-seed matrices.

## 5. G-STRICT-NO-GO

[Numerical result] The strict original-space control produced:

| Metric | Result | Registered gate | Status |
|---|---:|---:|---:|
| finite seed rate | $1.0$ | $1.0$ | PASS |
| exact serialized state equality rate | $1.0$ | $1.0$ | PASS |
| action-distribution equality rate | $1.0$ | $1.0$ | PASS |
| balanced accuracy | $0.5$ | $0.5$ | PASS |
| balanced regret | $0.5$ | $0.5$ | PASS |

For every $N\in\{1,2,4,8,16,64\}$, serialized aggregate equality and action
distribution equality were $1.0$, while balanced accuracy and regret remained
$0.5$. The stored hashes of the plus and minus aggregate were equal for every
seed and registered $N$. Thus `G-STRICT-NO-GO PASS`.

This numerical gate confirms the registered implementation control; equation
V17.V2 and the coupling proof, not the finite score, establish the mathematical
no-go. The finite ensembles do not empirically establish an infinite SCC.

## 6. G-LIFT

[Numerical result] The homogeneous one-factor candidate produced:

| Metric | Result | Registered gate | Status |
|---|---:|---:|---:|
| finite seed rate | $1.0$ | $1.0$ | PASS |
| action accuracy, 512 branches | $1.0$ | $1.0$ | PASS |
| mean regret | $0$ | $0$ | PASS |
| minimum wrong-minus-correct margin | $1.999999999999996$ | $\ge1.999999999$ | PASS |
| transported action agreement | $1.0$ | $1.0$ | PASS |
| maximum relative quadratic-cost defect | $4.4408920985006072\times10^{-15}$ | $\le10^{-10}$ | PASS |
| maximum relative metric-transport defect | $9.9841522918352446\times10^{-16}$ | descriptive | PASS |
| maximum relative reference-metric defect | $8.5023397051973222\times10^{-16}$ | descriptive | PASS |
| persistent state fields / optimizer fields | $1/0$ | $1/0$ | PASS |
| ambient coordinates / delta over strict metric | $10/+4$ | $10/+4$ | PASS |

All lift gate booleans in the sealed result are true. Therefore `G-LIFT PASS`.

## 7. Validation boundary

[Incomplete] No result in this run establishes general delayed credit
assignment, learning an eligibility rule from delayed reward, variable or
unknown terminal query directions, arbitrary-length memory, noisy finite-
precision robustness, infinite event-depth dynamics, or an unconditional
countable-agent limit. The lift is a target-aware analytic one-cue memory
primitive.

[Incomplete] No biological-brain model, spacetime identification,
consciousness, semantic OOD, autonomous tool use or unrestricted intelligence
was tested. Consequently the two V17 decisions pass while the AGI decision
remains `STOP / GO not authorized`.
