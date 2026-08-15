# V16 route-comparison calculations

This artifact contains the algebra behind the compact comparison in
`../12-routes.md`.  It does not use the confirmation seeds.

## 1. Persistent degrees of freedom

Let

$$
m=\dim \operatorname{Sym}(d)=\frac{d(d+1)}2.
$$

For $d=2,3,4$, respectively, $m=3,6,10$.  A full-matrix one-state learner
therefore has 3, 6, or 10 persistent scalar degrees of freedom.  An RLS learner
that stores both the parameter and a symmetric $m\times m$ inverse information
matrix has

$$
m+\frac{m(m+1)}2=9,27,65
$$

persistent scalars.  Fixed hyperparameters are not counted as learned state.

## 2. Rank-one affine-invariant flow

Put

$$
p=x^Tgx,\qquad q=\frac pc,\qquad
a=q^{-\eta}-1,qquad
u=\frac{g^{1/2}x}{\sqrt p}.
$$

Then $u^Tu=1$ and equation (V16.1) is

$$
g^+=g^{1/2}(I+a uu^T)g^{1/2}.
$$

The eigenvalues of the middle factor are $1$ on $u^\perp$ and
$1+a=q^{-\eta}>0$ on $u$.  Thus real-arithmetic SPD preservation is exact and
does not require a spectral projector.

For $y=Jx$ and $h=J^{-T}gJ^{-1}$,

$$
y^Thy=p,
\qquad
hy=J^{-T}gx,
$$

so direct substitution gives

$$
h^+=J^{-T}g^+J^{-1}.
$$

The selected observation contracts exactly because

$$
x^Tg^+x=p+ap=pq^{-\eta}=p^{1-\eta}c^\eta.
$$

The work is one dense matrix-vector product and one symmetric rank-one update,
namely $O(d^2)$ arithmetic and $O(d^2)$ storage.

## 3. A noiseless Lyapunov route for H1

This is a proof route to audit, not a promotion of the noisy H1 in the
contract.  Let the observation be noiseless, $c=x^Tg_*x$, and define the
nonnegative log-det/Burg divergence

$$
\mathcal D(g_*\Vert g)
=\operatorname{tr}(g^{-1}g_*)
-\log\det(g^{-1}g_*)-d.
$$

The determinant lemma and Sherman--Morrison identity give

$$
\det g^+=\det g\,q^{-\eta},
$$

and

$$
(g^+)^{-1}
=g^{-1}+\frac{q^\eta-1}{p}xx^T.
$$

Consequently, with $z=c/p=q^{-1}$,

$$
\mathcal D(g_*\Vert g^+)-\mathcal D(g_*\Vert g)
=z^{1-\eta}-z+\eta\log z. \tag{A1}
$$

Weighted AM--GM and $\log z\le z-1$ imply

$$
z^{1-\eta}\le(1-\eta)z+\eta,
$$

and hence

$$
\mathcal D(g_*\Vert g^+)-\mathcal D(g_*\Vert g)
\le\eta(1-z+\log z)\le0. \tag{A2}
$$

For $0<\eta\le1$, equality holds only when $p=c$.  The divergence sublevel
sets bound all generalized eigenvalues of $(g_*,g)$ away from zero and
infinity.  The route lane therefore derives the following sharply narrowed
theorem candidate, subject to independent math-lane audit:

> For noiseless observations from a finite nonzero set of directions whose
> outer products span $\operatorname{Sym}(d)$, if every direction occurs at
> least once in every fixed-length window, then fixed-$\eta$ V16.1 converges to
> $g_*$ for every $0<\eta\le1$.

Here is the proof chain.  The nonincreasing nonnegative sequence $\mathcal D_t$
converges, hence its one-step decrement tends to zero.  On its compact sublevel
set, every ratio $z_t$ for the finite direction set stays in a common compact
subset of $(0,\infty)$.  The decrement in (A1) is continuous and vanishes only
at $z=1$, so the actually observed residual tends to zero.  The rank-one update
norm then tends to zero.  The bounded recurrence gap implies that, between an
arbitrary time and the next occurrence of any fixed direction, only a bounded
number of vanishing updates intervene.  Therefore every direction's residual
tends to zero at every sufficiently late time.  Finally the linear measurement
map is injective with a bounded inverse on finite-dimensional
$\operatorname{Sym}(d)$ because the outer products span it, forcing
$\lVert g_t-g_*\rVert_F\to0$.

This proof does not cover a continuum of adaptive directions, unbounded
recurrence gaps, observation noise, or finite precision.

## 4. Killing exact noisy convergence at fixed learning rate

Persistent excitation does not rescue exact convergence under continuing
noise and constant $\eta$.  In $d=1$, take $x_t=1$, hidden $g_*=1$, and
$c_t=\exp(\varepsilon_t)$ with nondegenerate iid noise.  V16.1 becomes

$$
\log g_{t+1}=(1-\eta)\log g_t+\eta\varepsilon_t. \tag{A3}
$$

For $0<\eta<1$, this is an AR(1) process with stationary variance

$$
\operatorname{Var}(\log g_\infty)
=\frac{\eta}{2-\eta}\operatorname{Var}(\varepsilon)>0.
$$

For $\eta=1$, $g_{t+1}=c_t$ and the learner tracks the newest noisy sample.
Thus point convergence to $g_*$ is false even in one dimension.  The noisy
claim must instead concern finite-run risk, convergence to a stationary error
ball, expected drift, or a diminishing learning-rate schedule.

## 5. Additive update: explicit chart defect

Ignore projection by choosing values that remain inside its eigenvalue bounds.
Let

$$
g=I_2,\quad x=e_1,\quad c=e,\quad\eta=0.1,
\quad J=\operatorname{diag}(2,1).
$$

Here $p=1$, $r=-1$, and the original-chart additive update is

$$
g^+=\operatorname{diag}(1.1,1).
$$

Transporting this result yields

$$
J^{-T}g^+J^{-1}=\operatorname{diag}(0.275,1).
$$

In the transformed chart,

$$
h=J^{-T}gJ^{-1}=\operatorname{diag}(0.25,1),
\qquad y=Jx=(2,0)^T,
$$

and the additive rule instead gives

$$
h^+=h+0.1yy^T=\operatorname{diag}(0.65,1).
$$

The first component differs by $0.375$.  No clipping occurred, so the defect
belongs to the Euclidean additive direction itself.  Coordinatewise spectral
clipping introduces a second, independent affine-covariance defect.

## 6. Conformal irreducible ambiguity

Let $d=2$ and

$$
g_*=\operatorname{diag}(0.25,4),\qquad x_1=e_1,\quad x_2=e_2.
$$

The two candidates have equal Euclidean norm, but their true costs are $0.25$
and $4$.  Every conformal state $g=\alpha I$ predicts the same value $\alpha$
for both, so it cannot rank them.  Its best invariant RMS log-eigenvalue error
over $\alpha>0$ occurs at the geometric-mean scale $\alpha=1$ and equals

$$
\sqrt{\frac{(\log 0.25)^2+(\log 4)^2}{2}}=\log4
\approx1.38629436112.
$$

Under a general affine chart $J$, the transported state
$J^{-T}(\alpha I)J^{-1}$ is not conformal unless $J$ is a similarity.  Thus the
model class itself is not closed under the G-CHART transformation.

The contract makes the ambiguity exact throughout the scored protocol: every
training candidate has Euclidean norm one, and every held-out route is the sum
of two such displacements.  A conformal model therefore predicts $\alpha$ for
every training action and $2\alpha$ for every held-out route.  With the declared
tolerance tie and lowest-index policy, its actions are identical to identity
for every learning rate.  Learning $\alpha$ may change invariant metric error,
but cannot change accuracy or regret in this particular protocol.

## 7. Extra-state RLS and log-Euclidean routes

Vectorized quadratic regression uses a feature $\phi(x)$ satisfying
$c=\phi(x)^T\theta$, where $\theta$ contains the $m$ independent entries of
$g$.  RLS additionally stores an $m\times m$ inverse information matrix $P$.
It can be coordinate-covariant only if $P$ is transported as a fourth-order
tensor.  The common initialization $P_0=\delta I_m$ is not invariant under a
general affine reparameterization.  Moreover unconstrained RLS has no SPD
guarantee; adding an eigenvalue projector restores chartwise SPD but loses
general affine covariance.  The extra $P$ alone violates the V16 one-state
contract.

For a log-Euclidean route, write $H=\log g$, make an Euclidean gradient step in
$H$, and return $g^+=\exp H^+$.  This guarantees SPD but requires eigendecomposed
matrix functions, normally $O(d^3)$.  General congruence is not linearized by
the matrix logarithm.  For example,

$$
\log I=0,
$$

while for $J=\operatorname{diag}(2,1)$,

$$
\log(J^{-T}IJ^{-1})=\operatorname{diag}(-\log4,0)\ne0.
$$

Thus an Euclidean additive step in $H$ cannot satisfy the required general
affine metamorphism.  This is distinct from the affine-invariant Riemannian
exponential used by V16.1.

## 8. Information boundary shared by full-matrix routes

For any full-matrix route, the measurement operator is

$$
\mathcal A(g)_t=\langle g,x_tx_t^T\rangle_F.
$$

It is injective on $\operatorname{Sym}(d)$ exactly when
$\{x_tx_t^T\}$ spans that $m$-dimensional space.  This identifiability fact is
algorithm-independent.  It is necessary but not sufficient for convergence of
V16.1, projected SGD, RLS, or a log-Euclidean optimizer.
