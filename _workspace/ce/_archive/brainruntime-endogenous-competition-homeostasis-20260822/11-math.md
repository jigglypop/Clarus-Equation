# Mathematics

Status: COMPLETE

For one fresh source pulse, all four hidden rows receive the same positive
presynaptic scalar multiplied by one column of $B$. At $r=0$, monotonicity
therefore preserves the four distinct seeded levels. The operator

$$
c_i=[v_i-\max_{j\ne i}v_j]_+
$$

has exactly one positive coordinate when the maximum is unique and is the zero
vector when all coordinates tie. It supplies no coordinate-order tie break.

The frozen weights obey

$$
B_{\min}=1-0.2\frac{3}{\sqrt{20}},
\qquad
B_{\max}=1+0.2\frac{3}{\sqrt{20}}.
$$

Once a coordinate has received $r\simeq1$, $\lambda=1$ gives

$$
e^{-\lambda}B_{\max}<B_{\min},
$$

so any unused coordinate outranks it before max-relative competition. This is
conditional on the measured novelty commit and washout gates; it is not a
general theorem for arbitrary pulse trains or runtime parameters. Uniform
$B$, equal thresholds, and $r_0=0$ remain permutation symmetric and must
abstain.

The packet envelope is required because an external pulse leaves a decaying
activation tail. Without it, the same source burst could write several usage
updates and fill multiple hidden coordinates before the next source. The
positive-increment envelope makes the first rising packet dominant while
remaining a floating state rather than a Boolean winner or occupancy mask.

## Dimensionless audit

| quantity | dimension vector | result |
|---|---:|---|
| normalized $a,d,B,\xi,v,c,p,m,\nu,q,r$ | $(0,0,0,0)$ | dimensionless |
| $\gamma,\lambda,\alpha,\delta,\rho,\varepsilon$ | $(0,0,0,0)$ | dimensionless |
| $L,D_c$ and seed | unitless tick/count/index | not physical time |

The argument $-\lambda r$ of `exp` is dimensionless, the two terms inside the
competition subtraction have equal dimensions, and every normalization divides
like quantities. This establishes dimensional consistency only, not biological
or physical interpretation.
