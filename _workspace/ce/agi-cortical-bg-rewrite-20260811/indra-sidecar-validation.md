# Orbit sidecar task, budget, and scaling validation

## Registered outcome

The fixed delayed interception task used 54 balanced episodes at each cover
size `N = 64, 128, 256`. The action label was generated from future
reward/hazard utility by the environment's dense rollout. The sidecar received
only the initial quotient state, localized perturbations, tied kernel, and
decision rule; it did not receive the future dense state or label.

Across every size:

- dense reference accuracy and normalized utility: `1.0`, `1.0`;
- exact quotient-plus-local-cone accuracy and normalized utility: `1.0`, `1.0`;
- quotient-only accuracy: `0.3333333333`;
- quotient-only normalized utility: `0.5015`, `0.5031`, and `0.5372`;
- cyclic-shift action failures: `0`;
- fixed-radius task kernel versus generic exact sidecar action failures: `0`.

This establishes decision sufficiency for this analytic task. It does not test
learning: the utility decoder is fixed, and the labels are consequences of the
same registered environment dynamics rather than an external natural dataset.

## Limited-budget curve

| budget | action accuracy | maximum actual state error | certified bound | violations | certified coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.878948 | 0.878948 | 0 | 0.0 |
| 1 | 0.5833 | 0.845702 | 0.845702 | 0 | 0.0 |
| 2 | 0.5000 | 0.785909 | 0.785909 | 0 | 0.0 |
| 4 | 0.8333 | 0.457153 | 0.481090 | 0 | 0.0 |
| 8 | 0.9167 | 0.151061 | 0.158983 | 0 | 0.0 |
| 16 | 1.0000 | 0.075589 | 0.075589 | 0 | 0.0 |
| 64 | 1.0000 | 2.78e-17 | 0.0 | 0 | 1.0 |

The non-monotonic `B=1` versus `B=2` action accuracy is retained honestly:
magnitude-selected recurrent supports need not be nested. State errors stayed
inside the preregistered delay-resolved Lipschitz certificate at every budget.
No hard action was declared certified unless its top-two utility margin exceeded
twice the propagated score error. There were zero false certificates. At
`B=16`, empirical actions happened to be correct but the bound still required
abstention; only the exact `B=64` lane certified all audited actions.

## Resource result

At the locked `N=256` task lane:

- dense stored state scalars: `3840`;
- quotient plus sparse sidecar state scalars: at most `165`;
- state-scalar ratio: `0.04297`;
- median dense repeated-batch time: about `0.00585 s`;
- median fixed-radius sidecar time: about `0.00101 s`;
- time ratio: `0.1723` (`5.8x` faster).

The first generic Python dictionary sidecar was approximately equal to or
slower than vectorized dense NumPy at this small cover. The passing timing is
from a fixed-radius patch executor whose support and weights are compiled into
the registered task path. It matched the generic exact sidecar on every
episode. Therefore the resource GO applies to fixed finite-range kernels that
admit this patch execution, not yet to arbitrary delayed graphs.

## Verification and status

- combined prior and new regression: `40 passed`;
- Ruff: all new and changed files passed;
- CE dimensionless checker: passed;
- all nine registered behavioral, error, symmetry, memory, time, and matched
  kernel gates: passed.

Readiness score: `100/100` on the frozen synthetic sidecar rubric.

Verdict: `GO` for a feature-off, read-only BrainRuntime sidecar restricted to
the validated finite-range kernel family. `HOLD` remains for general runtime
replacement, learned/plastic kernels, arbitrary graphs, and claims of general
AGI or biological equivalence.
