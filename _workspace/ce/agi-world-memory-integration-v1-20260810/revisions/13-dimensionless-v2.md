# G9-CBM V2 dimensionless audit

Status: COMPLETE

Gate: PASS

This audit checks dimensional consistency only. It is not evidence that the
generator, memory mechanism, dream-like completion, or planning claim is true.
No registered seed or experiment runner was executed.

The contract declares the synthetic state and action coordinates dimensionless,
so every generator coefficient, residual, fingerprint, standardizer, threshold,
and learned regression coefficient below also has dimension vector
`(M,L,T,Theta)=(0,0,0,0)`.

| core argument or quantity | dimension vector | dimensionless | normalization / reason |
|---|---:|---:|---|
| `x`, `a`, `q_episode`, `b`, `eta`, `x_hat`, public goal `g` | `(0,0,0,0)` | yes | synthetic benchmark coordinates are declared dimensionless |
| `tanh(x[source])` argument | `(0,0,0,0)` | yes | `x` is dimensionless |
| `f_hat(x,a)` and codec residual `x_next-f_hat` | `(0,0,0,0)` | yes | sums/subtractions use like dimensionless coordinates |
| raw codec `T_raw` and residual-only schema `S_raw` | `(0,0,0,0)` | yes | concatenated dimensionless residual/signature coordinates |
| standardized codec `(T_raw-mu_codec)/sigma_codec` | `(0,0,0,0)` | yes | ratio of like coordinates; positive scale floor is dimensionless |
| standardized state `(v-mu_x)/sigma_x` | `(0,0,0,0)` | yes | ratio of like state coordinates |
| masked cosine confidence and `tau` | `(0,0,0,0)` | yes | normalized inner product and like threshold |
| residual join RMS and its train quantile | `(0,0,0,0)` | yes | root mean square of dimensionless residual differences |
| squared rollout error, NRMSE, cost `J`, regret `R` | `(0,0,0,0)` | yes | standardized state error and dimensionless action penalty |
| coverage, validity/error rates, relative reductions | `(0,0,0,0)` | yes | counts or ratios of like positive quantities |
| factorial effects, Student-t CI, strict-win fractions | `(0,0,0,0)` | yes | differences/averages of like dimensionless metrics |

The raw/standardized boundary is an explicit numerical-coordinate boundary, not
a physical-unit repair: recall returns raw coordinates, schema/dream storage may
use standardized coordinates internally, and inverse-standardization occurs
before state rollout. No raw-minus-standardized subtraction is permitted.

Byte counts and wall-clock seconds are resource-ledger quantities, not inputs to
`tanh`, probability, fixed-point, similarity, rollout, or scientific score
formulae. They therefore remain outside the dimensionless mathematical core.

Existing mechanical checks were run with the repository's established virtual
environment and the isolated worktree on `PYTHONPATH`:

```text
python -m pytest -q tests/test_dimensionless.py
7 passed in 4.74s

python reality_stone/python/reality_stone/clarus/dimensionless.py
exit code 0
```

The first `uv` attempt failed before test collection because OneDrive does not
support the requested cache hardlink operation. It produced no scientific
result; the successful commands above avoided dependency mutation and network
access.
