# Validation record: quantum-neighbor bootstrap

Status: COMPLETE

## Focused mechanical checks

| Check | Result | Scope |
|---|---|---|
| `.codex/hooks/python.cmd doctor` | PASS | system Python 3.11.9; bytecode disabled; repository `PYTHONPATH` active |
| `artifacts/verify_quantum_neighbor_bootstrap.py` | PASS | four-state absorption, state-11 closure counterexample, scalar root, exponential-lifetime overdispersion |
| focused quantum/branching/dimensionless pytest set | **51 passed** | `test_quantum_jump_bridge.py`, `test_multispace_bootstrap.py`, `test_bootstrap_solver.py`, `test_dimensionless.py` |
| focused reader-policy nodes | **3 passed** | fixed-point domain/measure term, relative links, renderable math delimiters |
| `git diff --check` | PASS | no whitespace errors; only line-ending warnings |
| research harness contract/lanes/gate | PASS | contract, independent lanes, initial and revised formal gate |

The run-local exact certificate reported

```text
absorbing_pi=[1.0, 0.0, 0.0, 0.0]
exact_dn1_11=-1 naive_linear=1
poisson_qext=0.04864671964
exp_lifetime_mean=0.4 variance=0.56
```

These are certificates for the declared finite model and fixed-point input,
not empirical observations or a derivation of dark abundance.

## Dimensionless gate

`kappa_ij` and `gamma_i` have dimension `T^-1`, while the jump operators have
dimension `T^-1/2`. Only `kappa_ij tau`, `gamma_i tau`, the offspring matrix
`A`, and its row sum or spectral radius `D` are dimensionless. Therefore the
Poisson exponential is dimensionally valid only after a time scale and the
branching-limit interpretation have been supplied. The detailed audit is in
`13-dimensionless.md`.

## Known unrelated policy failures

The full `tests/test_canonical_document_policy.py` result is **2 failed,
9 passed**. Both remaining failures are outside the approved files and existed
independently of this QNB change:

- `docs/9_등호이전/README.md` contains a machine-verdict word flagged by the
  global policy test;
- `.codex` and `.claude` policy mirrors have pre-existing drift and two policy
  files contain machine-verdict vocabulary.

The QNB-relevant node that initially failed (`x_0 in [0,1/D]` and residual
measure preservation) was repaired and passes, as do the global relative-link
and math-delimiter nodes. The unrelated failures were not edited because they
are outside this research contract.

## Interpretation ceiling

Passing code, dimensions, and document gates establishes internal consistency
of the declared model. It does not establish a universal quantum ontology,
the network-to-branching scaling theorem, the residual gravitational map, or
observed DM/DE abundance.

