# Validation

## Mathematical and regression checks

The new core and the existing dimensionless audit were executed in the project
virtual environment:

```text
25 passed
Ruff: All checks passed
```

Covered invariants include explicit HOLD normalization, increased HOLD under
increased common STN drive, unchanged conditional action identity, exact
action-softmax common-offset no-effect to floating precision, reconvergent-flow
mass conservation, posterior splitting of multi-path credit, finite topological
execution, invalid reverse-edge rejection, and the existing CE dimensionless
tests.

## Experiment A: held-out recombination

Forty untouched deterministic seeds were evaluated. Training used 12 of the 16
goal–subaction pairs; the four diagonal recombinations were OOD.

| Arm | OOD accuracy | NLL | Brier |
|---|---:|---:|---:|
| atomic flat | 0.000000 | 5.791132 | 1.203510 |
| strict tree | 0.000000 | 3.695888 | 1.343921 |
| shared DAG | 0.999544 | 0.108132 | 0.019798 |
| factorized flat | 0.999544 | 0.108132 | 0.019798 |
| destroyed shared identity | 0.000000 | 3.627629 | 1.233167 |

Paired 95% lower confidence bounds for shared-DAG accuracy minus atomic, tree,
and destroyed controls were all `0.999247`. The sharing-identity gate passed.
Shared-DAG minus factorized-flat was exactly `0.0`, so DAG-specificity failed.

Verdict: `FACTORIZATION_GO_DAG_UNRESOLVED`.

This is a strong but deliberately narrow result. It proves that reusable factor
identity, rather than an atomic output or duplicated tree branch, solves this
static recombination task. It does not prove that reconvergent DAG topology is
needed. A temporal start/maintain/terminate option task is required for that.
