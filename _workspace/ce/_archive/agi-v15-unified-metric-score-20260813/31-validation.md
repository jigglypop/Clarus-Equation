# AGI V15 unified metric: actual scored validation

Status: COMPLETE

## 1. Frozen artifact

The scored module SHA-256 was
`0599FC3B212F924424DE0675266881F8F1A6611D880382533708CD55F2529BE4`
and matched the preregistered hash. No SUT change occurred before or after the
score in this run.

## 2. Mathematical proof score

| Claim | Symbolic status | Killing result | Score |
|---|---|---:|---:|
| F1 affine readout covariance | theorem | maximum relative error $6.868\times10^{-16}$ | 1/1 |
| F2 spectral clipping covariance | complete counterexample | defect $5/9$ | 1/1 |
| F3 static direction symmetry | no-go theorem | forward/reverse error $0$ | 1/1 |
| F4 symmetric singleton goal | no-go theorem | minimizers $(1,2)$ retained | 1/1 |
| F5 finite-to-continuum identification | complete counterexample | length difference $0.5$ | 1/1 |
| **Total** |  |  | **5/5, MATH PASS** |

F2--F5 are scored as successful proofs of limitations. They are not positive
AGI capabilities.

## 3. Held-out ordinary-scale finite correctness

Seeds 915000--915255 were not used by the predecessor. Graph dimension was
2--4, node count 5--9, and all tensors were SPD with sampled eigenvalues in the
declared interval.

| Metric | Result | Gate |
|---|---:|---:|
| finite trial rate | 256/256 = 100% | PASS |
| reference shortest-cost agreement | 100% | PASS |
| returned simple-path validity | 100% | PASS |
| declared-tolerance goal agreement | 256/256 = 100% | PASS |
| maximum relative scalar error | $3.5955\times10^{-16}$ | PASS |

This slice is `PASS`. It is qualified by the adversarial scale failures below.

## 4. Coordinate and permutation metamorphism

| Metric | Result |
|---|---:|
| affine edge maximum relative error | $1.8728\times10^{-14}$ |
| affine path-cost agreement | 256/256 |
| affine goal agreement | 256/256 |
| permutation path-cost agreement | 256/256 |
| permutation goal agreement | 256/256 |

The declared $10^{-10}$ threshold passed. This is a coordinate metamorphic
readout test, not semantic OOD. The transport generator itself uses the SUT
helper; independent support for its formula is the F1 proof, not this score.

## 5. Positive-scale robustness failure

The preregistered complete three-node fixture failed by nontermination. A second
adversarial chain with the unique path $(2,1,0)$ and exact cost $2\times10^{-16}$
also failed to terminate within one second.

For an edge relaxation, the implementation uses

$$
q=D_{\mathrm{pop}}(u)+w_{uv},\qquad
\tau=128\epsilon\max(1,|q|,|D(v)|).
$$

When $|q-D(v)|\le\tau$, it can update path count and predecessor without a
strict distance-direction or source guard. On the unique-chain fixture the
internal arrays become

$$
D=(2\times10^{-16},10^{-16},0),\qquad P=(1,0,1),
$$

so $P(0)=1$ and $P(1)=0$ form a predecessor cycle. Public path reconstruction
then loops. This is a complete implementation counterexample, hence
`FINITE CORE GO = false` even though the bounded ordinary-scale slice passed.

## 6. Further finite-input adversarial score

The additional script reproduced eight failures out of eight probes:

1. costs $10^{-16}$ and $2\times10^{-16}$ are reported as tied goals;
2. source-to-self path is reported non-unique;
3. `reference_scale=1e-200` raises `ZeroDivisionError` after squaring underflows;
4. `reference_scale=1e200`, threshold zero, maps a strictly positive exact ratio
   to zero and returns a closed gate;
5. valid finite points $\pm10^{308}$ produce a NaN edge;
6. a valid finite large symmetric metric projects to nonfinite output;
7. a large finite affine Jacobian underflows an SPD eigenvalue to zero;
8. a small finite affine Jacobian overflows the transported metric.

These are robustness failures. They do not contradict the bounded random score;
they falsify a stronger all-finite-input interpretation.

## 7. Oracle navigation utility

On seeds 916000--916255, the environment's cost and V15's supplied metric used
the same endpoint-average formula. The identity arm used the same graph-search
algorithm and call count but did not receive that privileged metric.

| Metric | V15 oracle metric | Identity metric |
|---|---:|---:|
| exact route choice | 256/256 = 100% | 168/256 = 65.625% |
| mean normalized regret | 0 | 0.1909883315 |
| paired mean-regret improvement | 0.1909883315 | — |

Thus the preregistered `ORACLE UTILITY GO` passes. This proves that, when given
the objective metric, the implementation optimizes the matching synthetic
objective. It does not test inference of $g$, learning, or semantic world
understanding.

Post-score descriptive statistics, not preregistered confirmatory statistics:

- V15 Wilson 95% interval: $[98.52\%,100.00\%]$;
- identity Wilson 95% interval: $[59.61\%,71.17\%]$;
- discordant wins $(88,0)$, exact one-sided sign/McNemar
  $p=2^{-88}=3.23117\times10^{-27}$.

The small $p$ is evidence only for this oracle-information comparison.

## 8. Autonomous AGI gates

| Gate | Executable scored evidence | Score |
|---|---|---:|
| A1 raw observations $\to g_t$ learning | none; update accepts an external tensor | 0 |
| A2 perception--action--environment loop | none | 0 |
| A3 delayed temporal credit assignment | none | 0 |
| A4 learned compute-matched compositional OOD | none | 0 |
| **Total** |  | **0/4** |

Under the frozen decision rule, internal AGI qualification is 0% and the AGI
verdict is `STOP`.

## 9. Regression and reproducibility checks

| Command slice | Result |
|---|---:|
| F1--F5 verification script | exit 0; 5/5 |
| scored benchmark | deterministic replay matched captured JSON |
| adversarial numeric script | exit 0; 8 reported failures |
| focused V15 unit suite | 17 passed, one pytest cache warning |
| CE core plus V15 slice | 72 passed, three warnings |
| SCC compatibility plus V15 | 114 passed, one pytest cache warning |
| scorer/adversarial Ruff | all checks passed |
| scorer/adversarial compileall | exit 0 |

Warnings were two existing PyTorch sparse notices and a local pytest-cache
creation warning. The repository-wide suite was not rerun because the dirty
worktree has known unrelated missing-fixture and policy failures; relevant
slices and the scored scripts were run directly.

The separate CE constants harness reproduced bootstrap residual
$2.08\times10^{-17}$, scorecard 11/12 PASS with one CAUTION at $-1.80\sigma$,
dimension checks 7/7, and overall `CAUTION`. Those physics/constant results are
not evidence for or against V15 task capability.

