# AGI V16 covariant metric flow: independent mathematical verification

Status: COMPLETE

This lane audits only M1--M5, H1, and the mathematical boundaries of R1--R5
from `00-contract.md`.  It does not use finite learning-benchmark success as a
substitute for proof.  The deterministic killing fixture is
`artifacts/verify_v16_math.py`; its captured output is
`artifacts/verify_v16_math.log`.

## 1. Verdict

| Item | Status | Exact boundary |
|---|---|---|
| M1 SPD preservation | **THEOREM** | Every finite allowed input, $0<\eta\leq1$ |
| M2 affine covariance | **THEOREM** | Every $J\in GL(d)$; no reprojection |
| M3 same-observation contraction | **THEOREM** | Exact arithmetic |
| M4 AIRM natural-gradient identity | **THEOREM** | Exponential-map step at the current $g$; not a geodesic line search |
| M5 identifiability iff | **THEOREM** | Finite noiseless quadratic measurements; uniqueness over SPD |
| H1a noiseless bounded-gap convergence | **THEOREM** | Finite spanning direction family, uniformly bounded visit gaps, fixed $0<\eta\leq1$ |
| H1b fixed-rate noisy point convergence | **FALSE** | Already false in allowed dimension $d=2$ under a bounded-gap spanning schedule |
| H1c stochastic-risk/diminishing-rate convergence | **INCOMPLETE** | Requires a separately stated stochastic theorem |
| R1--R4 | **IMPLEMENTATION OBLIGATIONS** | Their numerical claims are not established by V16.1's geometry |

Thus the mathematical core M1--M5 closes.  H1 is now sharply divided: the
finite noiseless bounded-gap case closes, while the contract's fixed-rate noisy
point-convergence interpretation is refuted and weaker stochastic targets
remain incomplete.  No finite benchmark substitutes for those distinctions.

## 2. Common rank-one representation

Fix $g\in\mathbb S_{++}^d$, $x\ne0$, $c>0$, and put

$$
p=x^Tgx,\qquad r=\log(p/c),\qquad
u=\frac{g^{1/2}x}{\sqrt p},\qquad P=uu^T.
$$

Then $u^Tu=1$, $P^2=P$, and (V16.1) is exactly

$$
g^+=g^{1/2}\left[I+(e^{-\eta r}-1)P\right]g^{1/2}
=g^{1/2}e^{-\eta rP}g^{1/2}. \tag{1}
$$

The middle matrix has eigenvalue $e^{-\eta r}>0$ along $u$ and eigenvalue $1$
on $u^\perp$.  This identity drives M1, M3, M4, and the determinant formula.

## 3. M1: SPD preservation and exact determinant structure

For every nonzero $v$, (1) gives

$$
v^Tg^+v
=w^Te^{-\eta rP}w>0,
\qquad w=g^{1/2}v\ne0.
$$

Therefore $g^+\in\mathbb S_{++}^d$ without clipping.  The matrix determinant
lemma, or (1), also gives the exact formula

$$
\det g^+=\det g\,e^{-\eta r}
=\det g\left(\frac cp\right)^\eta. \tag{2}
$$

The relative congruence $g^{-1/2}g^+g^{-1/2}$ has spectrum
$\{e^{-\eta r},1,\ldots,1\}$.  This is the correct eigen-structure claim.
It does **not** say that $g$ and $g^+$ share ordinary Euclidean eigenvectors or
that only one ordinary eigenvalue changes; those stronger statements are false
unless the observed direction is aligned with an eigendirection of $g$.

## 4. M2: general affine covariance

Let $y=Jx$, $h=J^{-T}gJ^{-1}$, and retain the same scalar $c$.  Then

$$
y^Thy=x^Tgx=p,
\qquad hy=J^{-T}gx.
$$

Substitution into (V16.1) yields

$$
h^+
=J^{-T}gJ^{-1}
+\frac{e^{-\eta r}-1}{p}(J^{-T}gx)(J^{-T}gx)^T
=J^{-T}g^+J^{-1}.
$$

Thus the update is covariant for every invertible affine Jacobian.  Translation
does not enter a displacement.  Any fixed-chart spectral projection applied
before or after the update would break this theorem and is not covered.

## 5. M3: exact residual contraction

Writing $a=e^{-\eta r}-1$ and using $x^Tgx=p$,

$$
p^+=x^Tg^+x=p+\frac a p p^2=p(1+a)=pe^{-\eta r}.
$$

Since $e^{-\eta r}=(c/p)^\eta$,

$$
p^+=p^{1-\eta}c^\eta,
\qquad
\log(p^+/c)=(1-\eta)\log(p/c).
$$

For $\eta=1$ the selected observation is fit exactly.  This is a statement
about querying the same $x$ immediately after one update, not about other
directions or global metric error.

## 6. M4: AIRM exponential-map natural gradient

Use the affine-invariant Riemannian metric on SPD matrices,

$$
\langle U,V\rangle_g
=\operatorname{tr}(g^{-1}Ug^{-1}V),
$$

and the loss $L(g)=\tfrac12r^2$.  For a symmetric tangent perturbation $H$,

$$
dL_g[H]
=r\,\frac{x^THx}{p}
=\operatorname{tr}\left(\frac r pxx^TH\right).
$$

If $\operatorname{grad}^{\rm AIRM}L$ denotes the Riemannian gradient, its
defining equation for every $H$ is

$$
\operatorname{tr}
\left(g^{-1}\operatorname{grad}^{\rm AIRM}L\,g^{-1}H\right)
=dL_g[H].
$$

Hence

$$
\operatorname{grad}^{\rm AIRM}L
=\frac r p(gx)(gx)^T. \tag{3}
$$

The AIRM exponential map is

$$
\operatorname{Exp}_g(H)
=g^{1/2}\exp(g^{-1/2}Hg^{-1/2})g^{1/2}.
$$

For $H=-\eta\operatorname{grad}^{\rm AIRM}L$, equations (3) and the definition
of $u$ give $g^{-1/2}Hg^{-1/2}=-\eta r uu^T$.  Since

$$
e^{suu^T}=I+(e^s-1)uu^T,
$$

the exponential step is exactly (1), hence exactly (V16.1).  M4 is therefore
proved, not merely numerically matched.

Boundary: this is one explicit Riemannian-gradient step whose tangent is
computed at the current state.  It is not the exact Riemannian gradient-flow
solution for a finite time, because the gradient changes along that flow.

## 7. M5: identifiability if and only if

Let $A_t=x_tx_t^T$ and define the linear measurement map

$$
\mathcal A:\operatorname{Sym}(d)\to\mathbb R^T,
\qquad
(\mathcal A G)_t=\langle A_t,G\rangle_F=x_t^TGx_t.
$$

If the $A_t$ span $\operatorname{Sym}(d)$, then $\mathcal A(G_1-G_2)=0$
implies that $G_1-G_2$ is Frobenius-orthogonal to all of
$\operatorname{Sym}(d)$, so $G_1=G_2$.  Therefore the SPD metric is unique.

Conversely, if the $A_t$ do not span, choose a nonzero
$H\in\operatorname{Sym}(d)$ orthogonal to their span.  Let $G_0$ be any SPD
matrix and select

$$
0<\epsilon<\frac{\lambda_{\min}(G_0)}{\lVert H\rVert_2}.
$$

Then $G_+=G_0+\epsilon H$ and $G_-=G_0-\epsilon H$ are distinct SPD matrices,
while for every $t$,

$$
x_t^TG_+x_t-x_t^TG_-x_t
=2\epsilon\langle A_t,H\rangle_F=0.
$$

This constructs the required indistinguishable pair and proves the iff.  It
also implies the necessary count $T\ge d(d+1)/2$, but the count alone is not
sufficient; the rank-one measurement matrices must actually span.

## 8. H1a: noiseless finite spanning bounded-gap convergence

**Theorem H1a.**  Fix $g_*\in\mathbb S_{++}^d$ and a finite family of nonzero
vectors $x_1,\ldots,x_m$ such that $\{x_jx_j^T\}_{j=1}^m$ spans
$\operatorname{Sym}(d)$.  Set $c_j=x_j^Tg_*x_j$.  Let $j_t$ be any schedule for
which there is a finite $B$ such that, for all sufficiently large $t$, every
$j$ occurs at least once among $j_t,\ldots,j_{t+B-1}$.  Starting from any
$g_0\in\mathbb S_{++}^d$, apply (V16.1) with $(x_{j_t},c_{j_t})$ and one fixed
$0<\eta\leq1$.  Then

$$
g_t\longrightarrow g_*.
$$

**Proof, part 1: exact strict Lyapunov decrement.**  Define the Burg/log-det
divergence

$$
\mathcal D(g_*\Vert g)
=\operatorname{tr}(g^{-1}g_*)
-\log\det(g^{-1}g_*)-d.
$$

For one noiseless observation put $z=c/p$.  In the notation of Section 2,
$1+a=z^\eta$.  The matrix determinant lemma and Sherman--Morrison identity give

$$
\det g^+=\det g\,z^\eta,
$$

$$
(g^+)^{-1}
=g^{-1}+\frac{z^{-\eta}-1}{p}xx^T.
$$

Because $x^Tg_*x=c=zp$, their substitution into $\mathcal D$ yields the exact
identity

$$
\mathcal D(g_*\Vert g^+)-\mathcal D(g_*\Vert g)
=\Phi_\eta(z)
:=z^{1-\eta}-z+\eta\log z. \tag{4}
$$

Weighted AM--GM and $\log z\leq z-1$ give

$$
z^{1-\eta}\leq(1-\eta)z+\eta,
$$

$$
\Phi_\eta(z)
\leq\eta(1-z+\log z)\leq0. \tag{5}
$$

For $0<\eta\leq1$, equality in (5) occurs iff $z=1$.  Thus (4) is a strict
decrement for every nonzero selected residual.

**Proof, part 2: the sublevel set is compact inside SPD.**  Let

$$
h=g_*^{-1/2}gg_*^{-1/2}
$$

and let $\lambda_i>0$ be its eigenvalues.  Since $g^{-1}g_*$ is similar to
$h^{-1}$,

$$
\mathcal D(g_*\Vert g)
=\sum_{i=1}^d
\left(\lambda_i^{-1}+\log\lambda_i-1\right). \tag{6}
$$

Each scalar summand is nonnegative, has its unique minimum at $1$, and tends to
$+\infty$ as $\lambda\downarrow0$ or $\lambda\uparrow\infty$.  Hence every
finite sublevel of (6) confines all $\lambda_i$ to one interval
$[\underline\lambda_C,\overline\lambda_C]\subset(0,\infty)$.  The associated
closed and bounded set of symmetric matrices is compact and stays strictly
inside SPD.  By (4)--(5), every $g_t$ lies in the compact sublevel determined by
$g_0$.

**Proof, part 3: the selected residual and step vanish.**  The nonincreasing
nonnegative sequence $\mathcal D_t$ converges, so
$\mathcal D_{t+1}-\mathcal D_t=\Phi_\eta(z_t)\to0$.  Compactness of the metric
sublevel and finiteness of the direction family bound every possible $z_t$
away from zero and infinity.  On that compact scalar interval,
$\Phi_\eta$ is continuous and vanishes only at $1$; therefore

$$
z_t\to1,
\qquad r_t=-\log z_t\to0. \tag{7}
$$

Moreover,

$$
g_{t+1}-g_t
=\frac{z_t^\eta-1}{p_t}(g_tx_{j_t})(g_tx_{j_t})^T.
$$

All factors except $z_t^\eta-1$ are uniformly bounded on the compact sublevel
and finite direction family, with $p_t$ uniformly positive.  Equation (7)
therefore implies

$$
\lVert g_{t+1}-g_t\rVert\to0. \tag{8}
$$

**Proof, part 4: bounded gaps force all residuals at every cluster point.**
Take any convergent subsequence $g_{t_n}\to\bar g$, which exists by compactness.
For each fixed $j$, choose an occurrence $s_n\in[t_n,t_n+B-1]$ of $j$.  From
(8) and the bounded number of intervening steps,

$$
\lVert g_{s_n}-g_{t_n}\rVert
\leq\sum_{k=t_n}^{s_n-1}\lVert g_{k+1}-g_k\rVert\to0.
$$

Thus $g_{s_n}\to\bar g$.  Applying (7) at those selected occurrences and using
continuity gives

$$
x_j^T\bar g x_j=c_j=x_j^Tg_*x_j
\quad\text{for every }j. \tag{9}
$$

The spanning condition and M5 make (9) equivalent to $\bar g=g_*$.  Every
cluster point is therefore $g_*$.  A sequence in a compact set with one cluster
point converges to it, proving $g_t\to g_*$. $\square$

The bounded-gap assumption does real work.  Mere infinite recurrence allows
arbitrarily long intervals in which residual changes from other directions
need not be transferred back to a rarely visited direction by the above
argument.

## 9. H1 boundaries: AIRM distance and continuing noise

H1a does not follow from one-step contraction in AIRM distance.  Indeed, set
$g_*=I$, $\eta=1$,

$$
Q=\frac1{\sqrt{101}}
\begin{pmatrix}1&10\\10&-1\end{pmatrix},
\qquad
g=Q\begin{pmatrix}400&0\\0&1/400\end{pmatrix}Q^T,
\qquad x=(9.5,-1)^T,
$$

and report the noiseless cost $c=x^Tx$.  One valid V16 update increases the
RMS log-generalized-eigenvalue error from approximately $5.99146$ to $8.23318$.
Thus the successful Lyapunov function is the asymmetric Burg divergence, not a
per-observation Fejer contraction in AIRM distance.

Continuing multiplicative noise also gives a complete boundary within the
declared domain.  Take $d=2$, $g_*=I$, and cyclically visit

$$
x_1=e_1,\qquad x_2=e_2,\qquad x_3=e_1+e_2.
$$

Their rank-one matrices span $\operatorname{Sym}(2)$ and the visit gap is
three.  Return $c_t=x_{j_t}^Tg_*x_{j_t}e^{\sigma Z_t}$ for independent standard
Gaussian $Z_t$ and any fixed $0<\eta\leq1$.  Suppose for contradiction that
$g_t\to I$.  At the infinitely many $e_1$ visits, $p_t\to1$ and

$$
g_{t+1}-g_t
=\frac{(e^{\sigma Z_t}/p_t)^\eta-1}{p_t}
(g_te_1)(g_te_1)^T. \tag{10}
$$

The independent event $1\leq Z_t\leq2$ has fixed positive probability on that
subsequence and hence occurs infinitely often almost surely.  Along those
events, the norm of (10) is bounded away from zero once $g_t$ is near $I$.
But convergence of $g_t$ would require
$\lVert g_{t+1}-g_t\rVert\to0$, a contradiction.  Thus fixed-rate noisy point
convergence is false even with a finite spanning family and bounded gaps.

The scalar restriction explains the same mechanism algebraically.  In $d=1$,
with $e_t=\log(g_t/g_*)$, V16.2 becomes

$$
e_{t+1}=(1-\eta)e_t+\eta\sigma Z_t. \tag{11}
$$

For unit-variance noise its stationary variance is
$\eta\sigma^2/(2-\eta)>0$; the persistent innovations prevent almost-sure
convergence to zero.  Hence fixed-rate noisy point convergence to $g_*$ is
false even in one dimension.  Stationary tracking/risk bounds or diminishing-
rate stochastic convergence require separate hypotheses and remain incomplete.

## 10. R1--R5 mathematical boundary

- **R1:** cycle-free reconstruction needs a visited-set/step bound or a
  predecessor-DAG invariant.  SPD edge lengths alone do not ensure a buggy
  predecessor map terminates.
- **R2:** the sufficient DAG invariant $D(u)<D(v)$ on every predecessor edge is
  valid when public edge lengths are strictly positive.  A tie must not replace
  the representative predecessor used for reconstruction.
- **R3:** exact quadratic forms are affine covariant, but IEEE overflow,
  underflow, and cancellation are separate implementation questions.  Stable
  scaling and explicit rejection are required at the declared extremes.
- **R4:** for positive quantities, the gate
  $d_g^2/\ell_0^2>\tau$ is equivalently
  $\log d_g^2-2\log\ell_0>\log\tau$.  The latter can preserve the Boolean
  decision when the displayed ratio overflows or underflows.  This equivalence
  does not certify any particular implementation.
- **R5:** exact arithmetic supplies a positive factor but does not certify a
  binary64 algorithm.  If $g=LL^T$, put
  $v=L^Tx/\sqrt p$ and $z=c/p$.  Then $v^Tv=1$ and

  $$
  g^+=L\left[I+(z^\eta-1)vv^T\right]L^T
  =LCC^TL^T,
  \qquad C=I+(z^{\eta/2}-1)vv^T.
  $$

  Hence an exact positive factor exists and can be retriangularized.  Whether a
  chosen rank-one update/downdate or QR routine avoids cancellation, preserves
  representable positive diagonals, and rejects unrepresentable results is an
  implementation obligation tested by the declared extreme fixtures.

Consequently R1--R5 must be scored by the numeric/implementation lane.  None is
a corollary of M1--M5.

## 11. Reproduction and killing-test results

```powershell
& 'C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe' `
  '_workspace/ce/agi-v16-covariant-metric-flow-20260813/artifacts/verify_v16_math.py'
```

The frozen seed is `160013`.  Across 768 trials in dimensions 2--4, the log
records maximum defects $9.59\times10^{-15}$ for the AIRM identity,
$7.14\times10^{-13}$ for affine covariance, $3.89\times10^{-14}$ for residual
contraction in log space, $3.96\times10^{-14}$ for (2), and
$1.91\times10^{-13}$ for the exact Burg decrement (4).  A cyclic six-direction
spanning fixture with gap six reaches Frobenius error $8.91\times10^{-16}$ and
zero displayed Burg divergence after 12,000 updates.  The nonspanning fixture
produces two SPD metrics $\sqrt2$ apart with exactly zero measurement defect.
These values are regression checks for the algebra above, not proof
substitutes.

Status: COMPLETE
