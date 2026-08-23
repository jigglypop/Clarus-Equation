# BA-TR26 dimensionless audit

| Core quantity | Dimension vector | Status | Normalization |
|---|---:|---|---|
| $r_i,y_i,X,Y,K$ | $(0,0,0,0)$ | PASS | normalized simulator coordinates |
| singular-value ratio and $\kappa_2(X)$ | $(0,0,0,0)$ | PASS | ratio of like quantities |
| affine/span/rectangle residual | $(0,0,0,0)$ | PASS | norm divided by like norm |
| normalized packet column $W_j/\|W_j\|_2$ | $(0,0,0,0)$ | PASS | same weight scale cancels |
| binding residual and relative margin | $(0,0,0,0)$ | PASS | normalized content space |
| delay and packet counts | discrete count | PASS | simulator ticks/events |

No exponential, logarithm, probability, or fixed-point kernel receives a
dimensional input.  Repository dimensionless regression: `17 passed`.
Dimensionlessness establishes type consistency only; it does not establish the
global affine law, biology, or physical meaning.
