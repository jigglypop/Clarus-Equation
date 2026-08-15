# AGI V16 covariant metric flow: scored validation

Status: COMPLETE

## 1. Decision structure

The result has three separately established parts:

$$
\text{V16 NARROW GO}
=\text{G-MATH}\land\text{G-NUMERIC}
\land
(\text{G-LEARN}\land\text{G-CHART}\land\text{G-CLOSED-LOOP}).
$$

`artifacts/confirmation-results.json` records only the final parenthesized
confirmation conjunction as `learning_chart_closed_loop_pass`. G-MATH comes
from the proof lane and G-NUMERIC from the implementation regression lane.
Consequently the JSON boolean alone is not the V16 verdict.

All three parts pass in this run. Their conjunction yields `V16 NARROW GO` in
the scope fixed by `00-contract.md`. `AGI GO` remains forbidden.

## 2. G-MATH

[Theorem] M1--M5 close as follows: V16.1 preserves SPD and has the stated
determinant law; it is covariant under every $J\in GL(d)$ without reprojection;
the same-observation residual contracts by $1-\eta$; the update is one AIRM
natural-gradient exponential-map step; and noiseless quadratic measurements
identify $g$ iff their rank-one matrices span $\operatorname{Sym}(d)$.

[Theorem] The finite noiseless spanning, uniformly bounded-gap schedule
converges to $g_*$ by the strict Burg/log-det decrement proved in `11-math.md`.

[No-go theorem] Fixed-rate point convergence under persistent multiplicative
noise is false, including on an allowed bounded-gap spanning schedule. This
retained limitation does not fail M1--M5; stochastic stationary-risk and
diminishing-rate theorems remain incomplete.

[Numerical verification] The deterministic math fixture used seed 160013 and
768 trials in dimensions 2--4. It reported:

| Check | Maximum defect or terminal result |
|---|---:|
| affine covariance | $7.1304\times10^{-13}$ |
| AIRM update identity | $9.5844\times10^{-15}$ |
| same-observation contraction | $3.8858\times10^{-14}$ |
| determinant law | $3.9524\times10^{-14}$ |
| Burg decrement identity | $1.9007\times10^{-13}$ |
| minimum updated eigenvalue | $2.4540\times10^{-2}$ |
| bounded-gap final Frobenius error | $8.9099\times10^{-16}$ |
| nonspanning measurement defect | $0$ |

The AIRM-error killing fixture also preserved the required counterexample: one
valid update increases invariant RMS error from $5.99146$ to $8.23318$. These
numbers test the algebra; the proofs, not the random agreement, establish
G-MATH. Therefore `G-MATH PASS`.

## 3. G-NUMERIC

[Numerical result] Focused tests cover R1--R5 and the inherited V15 killing
fixtures:

- R1/R2: the $2\to1\to0$ chain at $10^{-16}$ scale terminates with its unique
  simple path, source-to-self is unique, goal ties do not acquire an absolute
  unit floor, and an injected predecessor cycle raises explicitly;
- R3: local and edge lengths pass at coordinate scales
  $10^k$, $k\in\{-150,-16,0,16,150\}$; huge finite metric symmetrization and
  projection remain finite; nonrepresentable distances and affine transports
  are explicitly rejected;
- R4: surprise gates remain correct for reference scales $10^{-200}$ and
  $10^{200}$ even when the displayed normalized ratio saturates;
- R5: the scalar cases $(1,1,10^{-300},1)$,
  $(1,10^{-150},1,1)$, and $(10^{308},1,10^{-308},1/2)$ produce the registered
  positive representable results, near-equality residuals are retained, and a
  nonrepresentable factor update raises explicitly.

The combined focused suite completed with 63 passing tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe' -m pytest `
  tests/test_unified_metric.py tests/test_covariant_metric_flow.py `
  tests/test_dimensionless.py tests/test_v16_benchmark.py `
  -q -p no:cacheprovider `
  --basetemp 'C:\Users\dongh\AppData\Local\Temp\codex-v16-run-reports-combined'
```

Result: `63 passed in 3.85s`. The dimensionless checks include
$r_{\mathrm{V16}}=\log(p/c)$ and normalized regret. Ruff reported `All checks
passed!` for the production, repaired numeric, evaluator, math-verifier, and
focused test files. Therefore `G-NUMERIC PASS` within the contract's declared
representable binary64 domain.

## 4. Sealed confirmation integrity

[Observed procedure] The confirmation block 918000--918255 was opened exactly
once after the rates and required artifacts were frozen. The opening receipt
records manifest SHA-256
`fa06dc24bc7be0cd00395c8ab57288cfad01ccf98da1560df809901393787df8`.
The evaluator reports `manifest_verified: true` and reproduces the selected
rates V16 $0.4$, additive $0.2$, and conformal $0.05$.

The evaluator tests independently verify required-manifest coverage, traversal
rejection, rejection of an alternate rate path before opening, exclusive
receipt/result creation, second-open rejection, and the development affine
chart fixture.

## 5. G-LEARN confirmation score

[Numerical result] Each learner used 256 confirmation seeds. V16 produced:

| Metric | V16 | Gate | Result |
|---|---:|---:|---:|
| finite episode rate | $1.0$ | $1.0$ | PASS |
| held-out route accuracy | $0.9642334$ | $\ge0.90$ | PASS |
| mean normalized held-out regret | $0.000439384$ | $\le0.05$ | PASS |
| median invariant metric error | $0.0339121$ | $\le0.25$ | PASS |
| regret improvement over identity | $0.3879170$ | $\ge0.10$ | PASS |
| regret improvement over conformal | $0.3879170$ | $\ge0.05$ | PASS |
| V16 regret minus additive regret | $-0.0120556$ | $\le0.02$ | PASS |

For context, additive route accuracy was $0.8910522$ with regret $0.0124950$;
identity and conformal each had route accuracy $0.2483521$ and regret
$0.3883564$. The equal identity/conformal action performance is structural on
the unit-vector protocol, not evidence that conformal learning recovered the
hidden anisotropy. Therefore `G-LEARN PASS`.

## 6. G-CHART confirmation score

[Numerical result] All transformed episodes remained finite, and original and
transported V16 runs had action agreement $1.0$. The maximum relative
prediction defect was $2.6735\times10^{-13}$ against the preregistered
$10^{-10}$ bound. The additional maximum relative transported-metric defect was
$3.0244\times10^{-14}$. Therefore `G-CHART PASS`.

This is a chart metamorphism of the same vector problem. It is not semantic OOD
or a raw-perception test.

## 7. G-CLOSED-LOOP confirmation score

[Numerical result] V16 mean normalized online regret after step 32 was
$0.1660816$, versus identity $0.6533468$. The improvement was $0.4872651$,
above the required $0.05$. The evaluator chooses an action before observing its
cost and updates only from the executed displacement and returned scalar cost.
Therefore `G-CLOSED-LOOP PASS`.

## 8. Verdict and boundary

G-MATH, G-NUMERIC, G-LEARN, G-CHART, and G-CLOSED-LOOP all pass. The exact
registered conjunction therefore gives `V16 NARROW GO`.

This is evidence for a one-state affine-covariant SPD metric-learning primitive
and a finite synthetic vector-observation action--environment--update loop. It
does not establish raw sensory representation, delayed credit, learned
compute-matched semantic OOD, tool use, continuum SCC convergence, biological
fidelity, cosmology, unrestricted intelligence, or AGI. `AGI GO` is not claimed.

