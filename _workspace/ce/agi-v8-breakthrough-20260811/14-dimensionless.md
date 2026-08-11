# Dimensionless audit

Status: COMPLETE

The candidate is

\[
\widehat x_h^{(g)}=p_h+g\bigl(s_h-p_h\bigr),
\qquad
g^*=\Pi_{[0,1]}
\frac{\sum \langle (s-p)/\sigma,(y-p)/\sigma\rangle}
     {\sum \lVert(s-p)/\sigma\rVert^2},
\]

where `s` is the frozen sparse-parent path, `p` is the persistence path,
`y` is an already observed training target, and each coordinate scale
`sigma_j` is estimated from the observational training split only.

| Core argument | Dimension vector | Dimensionless? | Normalization |
|---|---:|---|---|
| `(s_j-p_j)/sigma_j` | `(0,0,0,0)` | yes | same-coordinate training scale |
| `(y_j-p_j)/sigma_j` | `(0,0,0,0)` | yes | same-coordinate training scale |
| numerator of `g` | `(0,0,0,0)` | yes | inner product of normalized residuals |
| denominator of `g` | `(0,0,0,0)` | yes | squared norm of normalized direction |
| clipped gain `g` | `(0,0,0,0)` | yes | ratio of dimensionless scalars |
| normalized H20 RMSE | `(0,0,0,0)` | yes | residual divided by `sigma_j` |
| `log(R_candidate/R_control)` | `(0,0,0,0)` | yes | ratio of positive normalized errors |

The affine prediction is dimensionally coherent because `p`, `s`, and `y`
share the same synthetic coordinate units and `g` is dimensionless.  The
projection `Pi_[0,1]` acts on a dimensionless scalar.

Machine checks:

- `tests/test_dimensionless.py`: 10 passed.
- `reality_stone/python/reality_stone/clarus/dimensionless.py`: exit 0.

This establishes dimensional consistency only.  It does not prove OOD
performance, sparse causal necessity, or an AGI bridge.

