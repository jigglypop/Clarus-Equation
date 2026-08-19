# Independent-test mathematics for neural Riemannian metrics

Status: COMPLETE

## Claim, chart law, and pre-existing folds

The testable chain is $\Delta W^s\to\Delta g_t(z,c)\to\Delta p(x_{0:T},\tau_B\mid x_0,c)$. $W^s$ is identified-unit structural connectivity; $z$ is a frozen chart; $g_t$ is SPD; $v,Q$ remain distinct drift and noise. The hypotheses are $H_W:\Delta W^s\to\Delta g$, $H_G:\Delta g\to\Delta p$, and mediated $H_C$.

For $z'=Pz+a$, tensors transform as $g'=P^{-\top}gP^{-1}$, $Q'=PQP^\top$, and $v'=Pv$. A unit-to-chart transport for $W^s$ must be declared. Nonlinear charts require a pointwise Jacobian and fixed stochastic-calculus convention.

Three geometries must not be collapsed: anatomical surface metric $h$ (sheet, layer, wiring space), developmental/state baseline $g_0(z)$, and induced field $g_t(z,c)$. E17 constant SPD candidates represented neither $h$ nor a nonconstant $g_0$; it cannot decide inherited folding versus induced deformation. The relative deformation is

$$
A_t=g_0^{-1}g_t,\qquad L_t=\log A_t,
\qquad d_{\rm SPD}=\|\log(g_0^{-1/2}g_tg_0^{-1/2})\|_F.
\tag{M1}
$$

The spectrum of $A_t$ and $d_{\rm SPD}$ are invariant under simultaneous chart transformation. Baseline controls must include physical surface geodesic under $h$, layer, cell type, Euclidean distance, wiring length, baseline edge density, and stimulation spread. Competing nulls are $g_t=g_0$ and a declared pullback $g_t=f_t^*g_0$, which can create apparent deformation without intrinsic change.

Preregister baseline-producer IDs $P_h$ (surface/laminar anatomy), $P_W$
(structural connectivity), $P_D$ (pre-period dynamics), and $P_C$ (combined).
Freeze their maps and budgets using pre-intervention calibration only. Compute
(M1) separately for every retained producer, require a zero-compatible
pseudo-intervention deformation inside the pre-period, and pair each producer
with a complexity-matched flat-pullback null. This prevents $g_0$ from being
chosen post hoc to absorb either the induced deformation or its absence.

## P0 counterexamples

**Nonconstant does not mean curved.** Euclidean space in polar coordinates has $ds^2=dr^2+r^2d\theta^2$. Here $\partial_rg_{\theta\theta}\ne0$ and Christoffel symbols are nonzero, but Riemann curvature is zero. Curvature requires a $C^2$ SPD field, invariant tensor statistic, chart resampling, and a flat-pullback null. A nonconstant $g_0$ can simply be a pre-existing fold.

**Positive rank-one updates lengthen.** For $g_+=g_0+\alpha uu^\top$ and $\alpha>0$, $u^\top g_+u=u^\top g_0u+\alpha\|u\|^4$. Direct shortening requires $g_-=g_0-\alpha uu^\top$ with $0\le\alpha<(u^\top g_0^{-1}u)^{-1}$. A safer construction is $M_\alpha=M_0+\alpha uu^\top$, $g_\alpha=M_\alpha^{-1}$. Strict geodesic shortening still requires relevant paths to traverse altered directions.

**Distance does not determine hitting time.** With $dz=vdt$, start zero and target one, the same Euclidean distance gives $\tau=1/2$ for $v=2$ and $\tau=2$ for $v=1/2$. Thus $d_g\downarrow\not\Rightarrow\tau_B\downarrow$. A time claim needs a bridge such as $dz_t=-g^{-1}\nabla Vdt+B\,dB_t$, $Q=BB^\top$, and must beat matched unconstrained $v,Q$ and gain/noise models.

**Connectivity does not uniquely select a metric.** The same directed $W=\begin{psmallmatrix}0&1\\0&0\end{psmallmatrix}$ admits distinct SPD maps $I+\operatorname{sym}(W)\operatorname{sym}(W)^\top$ and $\operatorname{diag}(2,1)$. This kills unique determination from $W$ alone, not a specific preregistered covariant map $\Phi$.

**E17-style same-source prediction is circular.** $D_\Delta=(J-I)\Sigma_z(J-I)^\top+Q$ mixes drift, occupancy and noise, and $C_{Q,1}=Q$, so S4 at horizon one equals S3 under a common ridge. A metric generated from the same fitted $J,Q$ that determines the scored activity is an algebraic re-expression, not independent evidence. Cross-validation cannot repair this shared-source estimand.

## Decision rules

Estimate $H_W$ from pre/post structural data and an independently estimated later response tensor or metric, with a tensor-aware likelihood or (M1) loss. Estimate $H_G$ from later animal-held-out paths using conditional log likelihood or energy score; score censored first passage by $\delta\log f(\tau)+(1-\delta)\log S(T_{\rm censor})$. $H_C$ needs randomized or validly instrumented structural change, temporality, retained direct $W^s\to x$ path, and both component results.

Animal or randomized experimental unit is the sample unit. Required nulls: sham, negative target, gain-only, matched $Q$, anatomy-fixed context, surface/layer/cell/wiring covariates, target and cell permutation, chart resampling, baseline-null, pullback-null, and direct dynamics. No advantage over direct dynamics, failure under baseline/pullback null, or equal score after cell permutation kills the claimed route.

No $H_C$ result is identified unless $\Phi(W^s,c)$, the explicit
$g\to(v,Q)$ bridge, its parameter budget, the unconstrained $v,Q$ alternative,
and a retained direct $W^s\to x$ path are all frozen before outcome access.
Later-path holdout alone does not remove algebraic dependence when $g$ is a
re-expression of the fitted dynamics.

## Reproducibility

Run `python _workspace/ce/neural-riemannian-metric-independent-tests-20260818/artifacts/check_counterexamples.py`.
