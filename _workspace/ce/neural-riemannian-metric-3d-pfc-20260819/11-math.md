# Mathematical and statistical lane

Status: COMPLETE

## 1. Formal status and typed claims

`NRM3-D1` is a definition: a full cortical-ribbon metric is a smooth section
$g\in\Gamma(\operatorname{Sym}^+T^*M)$ with six independent local components.
`NRM3-H1` is a hypothesis that a frozen producer $\Phi(W^s,h)$ predicts
relative metric deformation. `NRM3-H2` is the separate hypothesis that the
metric constraint improves held-out path prediction under a declared
stochastic generator. Neither hypothesis follows from the definition.

This run can validate a finite synthetic estimator. It cannot establish a
biological law. No released PFC source in `10-sources.md` observes the anatomy,
depth, direct $W^s$, same-unit longitudinal activity, and intervention required
for the full chain.

## 2. Anatomy, folds, and dimensionless chart

Use dimensionless chart coordinates $y=(u,v,w)\in[-1,1]^3$. Physical depth is
$\ell=0.05w$ after the reference-scale normalization in the contract. Define

$$
s(u,v)=\left(u,v,0.25\sin(\pi u)\sin(\pi v)\right),
\qquad
r(u,v,w)=s(u,v)+0.05w\,n(u,v),
\tag{11}
$$

where $n$ is the oriented unit normal with positive third component. Let
$e^a{}_i=\partial_i r^a$ be the coframe matrix. Then

$$
h_{ij}=\delta_{ab}e^a{}_ie^b{}_j=e^\top e.
\tag{12}
$$

Because $h=r^*\delta$, its interior Riemann tensor is zero wherever $r$ is a
local diffeomorphism. Folds enter through $r$, the boundary, thickness, and the
mid-surface first/second fundamental forms, not through nonzero 3D intrinsic
curvature.

Let $a$ be the mid-surface first fundamental form, $\tau=0.05$, and

$$
J_\perp=\frac{|\det Dr|}{\tau\sqrt{\det a}}
=(1-\tau w k_1)(1-\tau w k_2).
$$

Every used atlas point must satisfy

$$
|\det Dr|\ge0.025,
\qquad
\sigma_{\min}(Dr)\ge0.04,
\qquad
0.05\max(|k_1|,|k_2|)\le0.25,
\qquad
J_\perp\ge0.5625.
\tag{13}
$$

These raw determinant/singular-value gates apply only to the frozen normalized
$(u,v,w)$ chart and embedding scale. They are not coordinate invariants. For
the analytic surface in (11),
$0.05\max|k_i|\le\pi^2/80<0.25$, hence the stronger exact bounds are
$|\det Dr|\ge0.05(1-\pi^2/80)^2\approx0.03842$ and
$\sigma_{\min}(Dr)\ge0.05$; the looser registered thresholds leave numerical
margin while $J_\perp$ records the geometric normal-volume condition.

On the 17-by-17-by-9 audit mesh, nonneighboring points with Chebyshev index
distance greater than one must be separated in physical space by more than
0.02. Failure of any condition invalidates the atlas. The ambient Cartesian
axes and $det Dr>0$ fix the coframe orientation. Under an ambient rotation
$Q\in SO(3)$, $e'=Qe$ and every frame tensor rotates by the same $Q$ rule.
Under a chart change and an optional oriented orthonormal-frame gauge rotation,
the combined law is $e'=QeJ^{-1}$. The matrix rule $S'=QSQ^\top$ applies only
to the frame gauge $Q$; the chart factor acts through the coframe. No
Gram-Schmidt sign choice is used.

All $y,h,g,e$, graph weights, tensor coefficients, exponential/logarithm
arguments, and potential $U$ are dimensionless. Time retains one normalized
unit, so contravariant response/diffusion tensors have units of inverse time.

## 3. Intrinsic SPD and relative deformation

Let $\mathcal S$ be an $h$-self-adjoint dimensionless endomorphism. Define

$$
g(a,b)=h(\exp(\mathcal S)a,b).
\tag{14}
$$

If $S=S^\top$ is its matrix in the $h$-orthonormal coframe $e$, production
computes

$$
g_{ij}=e^a{}_i[\exp(S)]_{ab}e^b{}_j.
\tag{15}
$$

Equation (15) is a coframe computation of (14), not a principal square root of
a coordinate metric matrix. Under $e'=Qe$, use $S'=QSQ^\top$; this gives
$g'=J^{-\top}gJ^{-1}$ under a chart change.

For two metrics in the same fiber, define the positive $g_0$-self-adjoint
endomorphism and intrinsic log deformation

$$
A_t=g_0^{-1}\circ g_t,
\qquad
\mathcal L_t=\log A_t,
\qquad
d(g_0,g_t)^2=\operatorname{tr}(\mathcal L_t^2).
\tag{16}
$$

In a $g_0$-orthonormal coframe $\vartheta$, if $L=L^\top$ is the matrix of
$\mathcal L_t$, then

$$
(g_t)_{ij}=\vartheta^a{}_i[\exp(L)]_{ab}\vartheta^b{}_j.
\tag{17}
$$

The Frobenius-orthonormal basis used throughout is

$$
\begin{aligned}
E_0&=I/\sqrt3,&
E_1&=\operatorname{diag}(1,-1,0)/\sqrt2,\\
E_2&=\operatorname{diag}(1,1,-2)/\sqrt6,&
E_3&=(e_{12}+e_{21})/\sqrt2,\\
E_4&=(e_{13}+e_{31})/\sqrt2,&
E_5&=(e_{23}+e_{32})/\sqrt2.
\end{aligned}
\tag{18}
$$

Thus $\operatorname{vec}_6(S)_q=\langle S,E_q\rangle_F$ preserves the
Frobenius norm and explicitly includes all three off-diagonal components.

## 4. Exact baseline metric

The synthetic baseline is observed exactly and is supplied to every candidate;
it is never estimated from post-change paths. Put

$$
\begin{aligned}
S_0(y)=0.04[&\sin(\pi u)E_0+\cos(\pi v)E_1+\sin(\pi w)E_2\\
&+\sin(\pi(u+v))E_3+\cos(\pi(v+w))E_4
+\sin(\pi(u-w))E_5].
\end{aligned}
\tag{19}
$$

Let $R_0=\exp(S_0/2)$ and $\vartheta=R_0e$. Then

$$
g_0=\vartheta^\top\vartheta=e^\top\exp(S_0)e.
\tag{20}
$$

This makes anatomy $h$ and persistence $g_0$ distinct while giving every model
the same baseline information and zero fitted baseline parameters.

## 5. Structural graph generator and tensor features

Each circuit uses the same 4-by-4-by-4 chart grid. For each node, sort every
other node by `(physical_distance, node_index)`, take the first 12, and only
then form the symmetric union. Distinct chart nodes must have
$\lVert r_i-r_j\rVert_2>10^{-12}$; otherwise the circuit fails. For
each retained unordered edge draw independent standard normals $\xi_{ij}$ and
$\eta_{ij}$ from its counter-derived seed and set

$$
W^0_{ij}=\exp\!\left[-\frac{d_{ij}^2}{2(0.75)^2}+0.15\xi_{ij}\right],
\qquad
W^1_{ij}=W^0_{ij}\exp[0.20f_{ij}+0.05\eta_{ij}].
\tag{21}
$$

Let $m=(y_i+y_j)/2$, $q_{ij}=(r_j-r_i)/d_{ij}$, and

$$
f_{ij}=q_{ij}^\top H(m)q_{ij}+0.20\sin(\pi(m_u+m_v)),
\tag{22}
$$

where $m_w$ is the normalized depth and

$$
H(m)=
\begin{pmatrix}
.45\sin\pi m_u&.25\sin\pi(m_u+m_v)&-.20\cos\pi(m_v+m_w)\\
.25\sin\pi(m_u+m_v)&-.35\cos\pi m_v&.18\sin\pi(m_u-m_w)\\
-.20\cos\pi(m_v+m_w)&.18\sin\pi(m_u-m_w)&-H_{11}-H_{22}
\end{pmatrix}.
\tag{23}
$$

$W$ is dimensionless and positive. The estimator receives only node
coordinates and the two edge lists $(W^0,W^1)$, never $H$, random draws,
generator label, true coefficients, or test paths.

For $c\in\{0,1\}$ and $\rho_k\in\{0.45,0.90\}$, define in the ambient
$h$-orthonormal frame

$$
s_k^c(i)=\sum_jW^c_{ij}e^{-d_{ij}^2/(2\rho_k^2)},
\tag{24}
$$

$$
C_k^c(i)=10^{-8}I+
\frac{\sum_jW^c_{ij}e^{-d_{ij}^2/(2\rho_k^2)}q_{ij}q_{ij}^\top}
{s_k^c(i)}.
\tag{25}
$$

Require $s_k^0>10^{-12}$ and $s_k^1>10^{-12}$ at every node and
$C_k^c\succ0$. One invalid node fails the entire circuit; circuits are never
dropped or replaced after the fixed sample count. Normalize before the matrix
log:

$$
A_k^c=\log\!\left(C_k^c/\operatorname{tr}C_k^c\right)
-\frac13\operatorname{tr}\log\!\left(C_k^c/\operatorname{tr}C_k^c\right)I,
\quad
q_k=\log(s_k^1/s_k^0).
\tag{26}
$$

Finite differences of $q_k$ use second-order central differences at interior
nodes and second-order one-sided differences at boundaries. Convert coordinate
covectors to ambient vectors by $\nabla_hq=e^{-\top}dq$. Define six symmetric
ambient-frame covariant matrices

$$
\begin{aligned}
T_1&=q_1I/\sqrt3,&T_2&=A_1^1-A_1^0,&T_3&=A_2^1-A_2^0,\\
T_4&=\operatorname{dev}(\nabla_hq_1\nabla_hq_1^\top),&
T_5&=\operatorname{dev}(\nabla_hq_2\nabla_hq_2^\top),&
T_6&=\operatorname{dev}\operatorname{sym}
(\nabla_hq_1\nabla_hq_2^\top).
\end{aligned}
\tag{27}
$$

Convert each to the $g_0$-orthonormal coframe by

$$
\widetilde T_m=R_0^{-\top}T_mR_0^{-1}.
\tag{28}
$$

The implementation enforces this typed conversion:

```text
dq_coordinate -> e^{-T} dq = ambient vector
A_k and dev(grad q outer grad q) = ambient Euclidean covariant matrices
ambient T -> R0^{-T} T R0^{-1} = g0-coframe covariant matrix
```

The ambient Euclidean metric is the explicit vector/covector identification;
a coordinate matrix is never inserted directly into (28).

On the 16 fit circuits only, compute

$$
r_m=\sqrt{\frac{1}{16\cdot64}
\sum_{i,x}\lVert\widetilde T_m^{(i)}(x)\rVert_F^2}.
\tag{29}
$$

Require every $r_m>10^{-8}$. The same frozen $r_m$ divides fit, validation,
and test fields. The column matrix formed by flattening the six normalized
fields over the 16 fit circuits must have rank six and condition number at most
$10^8$.

The primary six-parameter relative deformation is

$$
L_\theta(x)=\sum_{m=1}^6\theta_m\widetilde T_m(x)/r_m,
\qquad
g_\theta=\vartheta^\top\exp(L_\theta)\vartheta,
\tag{30}
$$

with $\theta_m\in[-0.75,0.75]$. The true vector is

$$
\theta_*=(0.22,-0.18,0.16,0.14,-0.12,0.10).
\tag{31}
$$

## 6. Coordinate-covariant path generator

Use the anatomical Levi-Civita connection $\nabla^h$. For a dimensionless
potential $U$, positive contravariant response $R$, and diffusion $D$, define

$$
\mathscr L f=
[-R^{ij}(\partial_jU-f_j)+\nabla^h_jR^{ij}]\partial_i f
+D^{ij}\nabla^h_i\nabla^h_jf.
\tag{32}
$$

The corresponding coordinate Itô increment has

$$
b^i=-R^{ij}(\partial_jU-f_j)+\nabla^h_jR^{ij}
-D^{jk}\Gamma(h)^i{}_{jk},
\qquad
\operatorname{Cov}(dY)=2D\,dt.
\tag{33}
$$

The metric model imposes the fluctuation-response constraint

$$
R=D=\kappa g^{-1},
\qquad \kappa=0.04.
\tag{34}
$$

The potential is

$$
U(y)=\tfrac12(u^2+1.2v^2+0.7w^2)
+0.08\sin(\pi u)\sin(\pi v)\cos(\pi w/2).
\tag{35}
$$

Training forces are $\{0,\pm0.25e^1,\pm0.25e^2\}$; test forces are
$\{0,\pm0.25e^3\}$. A dataset has 24 training circuits, split 16 fit and 8
validation, plus 40 outer-test circuits. Each circuit/force has 12 independent
paths, 32 Euler steps, and $dt=0.02$. Initial points are uniform on
$[-0.6,0.6]^3$. A path stops at first exit from $[-1,1]^3$; only pre-exit
increments are scored.

The grid fields in (30) are trilinearly interpolated; the lower-index cell owns
a boundary. Derivatives of interpolated tensors are evaluated by centered
coordinate differences of step $10^{-5}$, with second-order inward differences
within $10^{-5}$ of a boundary. $h$ and $\Gamma(h)$ use analytic $Dr$ plus the
same registered derivative rule. Candidate and evaluator share the numerical
integrator but never share hidden truth.

All production paths, fits, and scores use exactly the normalized chart $y$ in
(11). Finite-step Euler paths are not claimed to be chart invariant. The chart
fixture transforms the continuous generator coefficients in (32)-(33), plus
length and curvature tensors, rather than demanding equality of two discrete
Euler sample paths at nonzero $dt$.

For a predicted increment mean $b_tdt$ and covariance $2D_tdt$, the proper
primary score is

$$
\operatorname{NLPD}_t=\frac12[(\Delta y-b_tdt)^\top(2D_tdt)^{-1}
(\Delta y-b_tdt)+\log\det(2D_tdt)+3\log(2\pi)].
\tag{36}
$$

One circuit contributes the mean (36) over its valid test increments. Paths
and steps are estimation replicates, never inferential units.

## 7. Finite candidate universe

Let $E_\vartheta=\vartheta^{-1}$ be the dual $g_0$ frame. For any symmetric
frame field $L$, define

$$
\mathcal P(L)=\kappa E_\vartheta\exp(-L)E_\vartheta^\top.
\tag{37}
$$

The exactly eleven candidates are:

| ID | Parameters | Frozen response and diffusion |
|---|---:|---|
| `MW6` | 6 | $R=D=\mathcal P(L_\theta)$ from (30) |
| `DIRECT-VA` | 12 | $R=\mathcal P(L_\alpha)$ and $D=\mathcal P(L_\delta)$ with independent six-vectors |
| `ANATOMY` | 0 | $R=D=\kappa h^{-1}$ |
| `EUCLIDEAN` | 0 | $R=D=\kappa I$ in the frozen normalized chart |
| `PERSIST` | 0 | $R=D=\kappa g_0^{-1}$ |
| `GAIN` | 3 | $L_R=q_1\operatorname{diag}(\gamma_1,\gamma_2,\gamma_3)/r_1$, $L_D=0$ |
| `NOISE` | 6 | $L_R=0$, $L_D=L_\delta$ |
| `W-DEGREE` | 6 | `MW6` after a separate global edge-weight permutation inside each circuit |
| `W-SPATIAL` | 6 | `MW6` after per-circuit permutation within fit-derived physical-distance bins |
| `W-CIRCUIT` | 6 | `MW6` after circuit derangement of $W^1$ |
| `FLAT-PULLBACK` | 6 | $R=D=\kappa(F_\zeta^*h)^{-1}$ |

All `MW6`, direct, noise, and W-surrogate coefficients use the bounds
$[-0.75,0.75]$. Gain coefficients use $[-0.70,0.70]$. Bounds are implemented
as $a\tanh z$ with unconstrained optimizer coordinate $z$, not by projection.

For `W-DEGREE`, keep graph topology and apply a seeded Sattolo permutation to
the condition-1 edge weights separately inside each circuit, preserving every
unweighted node degree and that circuit's global weight multiset. For
`W-SPATIAL`, derive six equal-count physical-edge distance-bin boundaries from
the pooled edges of the 16 fit circuits, break boundary ties into the lower
bin, and then Sattolo-permute condition-1 weights separately within every
`(circuit, bin)`. Apply the fit-derived boundaries unchanged to validation and
test circuits; weights never move between circuits or splits. Topology, degree,
and each circuit-bin's weight multiset are preserved. A circuit-bin with fewer
than two edges is an explicit circuit failure. For `W-CIRCUIT`,
Sattolo-derange the complete condition-1
graphs separately within fit, validation, and test circuit blocks; no circuit
keeps its own graph. Condition-0 graphs and all paths remain fixed.

For the flat candidate define $F_\zeta(y)=y+\sum_{q=1}^6\zeta_qV_q(y)$ with
$|\zeta_q|\le0.005$ and

$$
\begin{aligned}
V_1&=(s_1(u)s_1(v)s_1(w),0,0),&
V_2&=(s_2(u)s_1(v)s_1(w),0,0),\\
V_3&=(0,s_1(u)s_2(v)s_1(w),0),&
V_4&=(0,s_1(u)s_1(v)s_2(w),0),\\
V_5&=(0,0,s_2(u)s_1(v)s_1(w)),&
V_6&=(0,0,s_1(u)s_2(v)s_1(w)),
\end{aligned}
\tag{38}
$$

where $s_k(x)=\sin(k\pi x)$. The analytic bound is

$$
\sup_y\lVert DF_\zeta-I\rVert_2
\le6(0.005)\pi\sqrt6<0.231<1.
$$

The fields vanish on the boundary, so this bound makes $F_\zeta$ globally
lower-Lipschitz with constant greater than $0.769$ and therefore injective.
Also require $\det DF_\zeta\ge0.25$ and
$\sigma_{\min}(DF_\zeta)\ge0.25$ on the 17-by-17-by-9 mesh. Failure of the
analytic bound or either mesh check invalidates the fit. Set

$$
g_F(y)=DF_\zeta(y)^\top h(F_\zeta(y))DF_\zeta(y).
\tag{39}
$$

It is a nonlinear coordinate pullback of the flat anatomical metric and has
zero intrinsic curvature.

## 8. Fit algorithm

Fitted candidates use ridge values

$$
\lambda\in\{0,10^{-6},10^{-4},10^{-2},1\}.
\tag{40}
$$

For each candidate and $\lambda$, minimize fit-circuit mean NLPD plus
$\lambda\lVert\beta_C\rVert_2^2/2$. Here $\beta_C$ is the complete bounded
physical coefficient vector: $\theta$ for `MW6` and each W surrogate,
$(\alpha,\delta)$ for `DIRECT-VA`, $\gamma$ for `GAIN`, $\delta$ for `NOISE`,
and $\zeta$ for `FLAT-PULLBACK`. Zero-parameter candidates have no penalty.
The unconstrained pre-`tanh` optimizer coordinate is not ridge-penalized. Use
damped BFGS initialized at zero,
two-sided gradient step $10^{-6}\max(1,|z_j|)$, Armijo constant $10^{-4}$,
backtracking factor $1/2$, minimum step $2^{-20}$, gradient infinity-norm
tolerance $10^{-7}$, relative-objective tolerance $10^{-9}$, and at most 300
iterations. Initial inverse Hessian is identity. No altered retry is allowed.

Choose $\lambda$ by validation-circuit NLPD; ties within $10^{-12}$ choose the
larger $\lambda$. Refit on all 24 training circuits using the already frozen
16-circuit preprocessing scales. Every SPD tensor must have minimum eigenvalue
$>10^{-10}$, condition number $\le10^{10}$, and relative antisymmetric norm
$\le10^{-13}$. Invalid or nonconvergent fits receive a failure code.

## 9. Primary decision family

For test circuit $i$ and control $b$, let

$$
d_{ib}=\operatorname{NLPD}_{i,\mathrm{MW6}}
-\operatorname{NLPD}_{i,b}.
\tag{41}
$$

Nine controls (`ANATOMY`, `EUCLIDEAN`, `PERSIST`, `GAIN`, `NOISE`, the three
$W$ surrogates, and `FLAT-PULLBACK`) use one-sided exact sign tests with a win
$d_{ib}<-10^{-10}$. `DIRECT-VA` is the nested 12-parameter alternative and
uses a noninferiority margin

$$
\epsilon_{\rm NI}=0.01\text{ nats per valid increment};
\tag{42}
$$

The direct inferential null is
$H_0:\Pr(d_{i,\mathrm{DIRECT}}<\epsilon_{\rm NI})\le1/2$ against the one-sided
alternative that this paired probability exceeds one half. A direct win is
$d_{i,\mathrm{DIRECT}}<\epsilon_{\rm NI}-10^{-10}$; equality and the tolerance
band are non-wins. Holm correction covers these ten p-values. Predictive
promotion requires adjusted $p\le0.05$, at least 28 wins of 40 using the
unshifted threshold for every superiority comparison and the shifted threshold
for `DIRECT-VA`, negative mean $d_{ib}$ for each superiority control,
descriptive mean $d_{i,\mathrm{DIRECT}}\le\epsilon_{\rm NI}$, finite
outputs, and no budget violation. Thus raw-NLPD equality with the containing
direct model is permitted; a material loss to it rejects the metric coupling.
The mean guard is not a confidence-bound test; the circuit-level sign test is
the registered noninferiority inference.

## 10. Full-field recovery

On an independent 9-by-9-by-5 mesh, compare the intrinsic relative-log matrices
in the common $g_0$ frame:

$$
E_g=\frac{\sum_x\lVert\widehat L-L_*\rVert_F^2}
{\sum_x\lVert L_*\rVert_F^2},
\qquad
C_g=\frac{\sum_x\langle\widehat L,L_*\rangle_F}
{\sqrt{\sum_x\lVert\widehat L\rVert_F^2
\sum_x\lVert L_*\rVert_F^2}}.
\tag{43}
$$

One true dataset recovers the field only if $E_g\le0.10$, $C_g\ge0.95$, all
six coefficient errors are at most 0.05, correlations for the three
off-diagonal entries are each at least 0.90, and every field is SPD. It must
also satisfy Section 9. The true experiment has 200 independent datasets and
passes with at least 180 successes; the exact one-sided 95% binomial lower
bound is then greater than 0.85.

Both denominator sums in (43) must be finite and strictly greater than
$10^{-12}$. A zero, negative, or nonfinite denominator is a recovery failure;
it is never replaced by an epsilon.

## 11. Null families and exact coefficients

The six families are:

1. `N0-EXACT`: $R=D=\kappa g_0^{-1}$ while supplied $W$ changes;
2. `N1-DIRECT`: call the same feature API (21)-(30) on the supplied graph,
   using its 16-fit-circuit scales for every split, then set
   $L_R=L_\alpha$, $L_D=L_\delta$ with
   $\alpha=(.20,-.15,.11,.13,-.10,.08)$ and
   $\delta=(-.12,.18,-.16,.07,.14,-.09)$;
3. `N2-GAIN`: $L_R=q_1\operatorname{diag}(.25,-.18,.12)/r_1$, $L_D=0$;
4. `N3-NOISE`: $L_R=0$ and
   $L_D=L_\delta$ with $\delta=(.18,-.14,.16,-.12,.10,.08)$;
5. `N4-FLAT`: (39) with
   $\zeta=(.004,-.003,.0035,-.0025,.003,-.002)$;
6. `N5-HIDDEN-W`: generate a hidden graph with the same API (21)-(30), a
   disjoint canonical `hidden` seed namespace, and its own hidden
   16-fit-circuit scales; use (30)-(31) from it to generate paths. The fitter
   builds independent fields/scales from an unrelated supplied graph and never
   receives hidden edges, scales, fields, or seeds.

Each family contains 200 independently seeded datasets with its own 24/40
circuit split. A false promotion is any dataset meeting every Section 9 rule.
Each family separately must have at most 4 false promotions of 200. The
validator computes the one-sided 95% Clopper-Pearson upper bound and requires
it to be at most 0.05; it does not trust the count equivalence in prose. The
1,200 datasets are never pooled to conceal a failing family.

Gate B passes only if Section 10 and all six null-family gates pass.

## 12. Numerical geometry fixtures

Fixture `F-H` uses (11)-(12) and must have zero interior curvature. `F-FLAT`
uses the explicit nonconstant pullback (39) and must also be flat. The curved
fixture uses $h=I$ on $[-0.25,0.25]^3$ and

$$
C=\begin{pmatrix}.20&.10&-.08\\.10&-.15&.07\\-.08&.07&.12\end{pmatrix},
\tag{44}
$$

$$
B(x,y,z)=
\begin{pmatrix}
x^3&y^3&z^3\\
y^3&(x+y+z)^3&x^3\\
z^3&x^3&(x-y)^3
\end{pmatrix},
\tag{45}
$$

$$
S_*(x,y,z)=C+0.14(x^2+y^2+z^2)I+0.02B(x,y,z),
\qquad g_*=\exp(S_*).
\tag{46}
$$

All six components are nonzero. At the origin the exact scalar-curvature
oracle is

$$
R_{g_*}(0)=-0.56\operatorname{tr}(e^{-C})\ne0.
\tag{47}
$$

Use fourth-order central differences on a 17-by-17-by-9 mesh, excluding two
cells at each boundary. Frozen Gate A tolerances are:

| Fixture | Requirement |
|---|---|
| symmetry | antisymmetric relative norm $\le10^{-13}$ |
| SPD | minimum eigenvalue $>10^{-10}$, condition $\le10^{10}$ |
| symmetric exp/log round trip | relative Frobenius error $\le10^{-11}$ |
| affine chart law, chart condition $\le100$ | relative tensor error $\le10^{-10}$ |
| sampled-curve length | relative chart error $\le10^{-9}$ |
| `F-H`, `F-FLAT` curvature | invariant norm $\le10^{-5}$ at every audit point |
| `F-CURVED` origin scalar | relative error to (47) $\le10^{-3}$ and absolute value $\ge10^{-2}$ |
| serial/Rayon | byte-identical indexed records and aggregate JSON |
| Rust/reference oracle | `rtol=1e-9`, `atol=1e-11` for registered scalars except curvature's stated tolerance |

## 13. Seeds, truth separation, and PFC seal

Every RNG is ChaCha20 seeded by BLAKE3 of the UTF-8 canonical tuple
`(master_seed, route, generator, dataset, split, circuit, condition, force,
path, candidate)`. Master seed is `0x4e524d3344504643`. No thread-local RNG is
allowed. Rayon writes indexed per-dataset or per-circuit records; aggregation
uses ascending indices.

The result artifact includes raw graph/path traces in a hash-linked file.
Generator labels and truth fields are evaluator-only and are not accepted by
the fitter API. The validator recomputes split counts, per-circuit scores,
sign-test p-values, Holm adjustment, recovery gates, and Clopper-Pearson bounds.

Biological analysis remains sealed until Gate B passes. If it passes, a
schema-derived amendment must freeze the PFC chart, split, estimator, target,
and controls before any neural or behavioral statistic. The maximum allowed
status is `PFC_FEASIBILITY_ONLY` because Wójcik sessions contain new neurons,
only two animals, categorical locations, and no direct $W^s$ or ribbon depth.

## 14. Complete counterexamples and kill conditions

1. $\operatorname{diag}(1,1,e^{-\beta q})$ changes one degree of freedom and
   cannot validate a six-component field.
2. A nonlinear Euclidean pullback has nonconstant components and Christoffel
   symbols but zero Riemann tensor.
3. A folded 3D Euclidean ribbon is interior-flat; folding is not functional
   curvature.
4. Fixed $g$ with different $U$, response, or diffusion gives different path
   and first-passage laws.
5. A path law does not identify $g$ without the response-diffusion coupling;
   `DIRECT-VA` is the explicit countermodel.
6. If $\operatorname{rank}DF<3$, $DF^\top g_zDF$ is singular.
7. Anatomy and activity metrics on different domains cannot be subtracted
   without a frozen rank-three map.
8. Duplicating paths within one circuit can shrink a path-level p-value without
   adding an independent circuit.

Any tensor-law failure, rank/design failure, truth leakage, failed null family,
material loss to `DIRECT-VA`, nonfinite output, or nested pseudoreplication
kills the corresponding claim. No other route rescues it.
