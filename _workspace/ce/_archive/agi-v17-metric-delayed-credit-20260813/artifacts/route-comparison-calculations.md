# V17 route-comparison calculations

These calculations are independent route-lane work. They use no confirmation
seed and make no promotion or closure decision. The executable development
check is `verify_v17_route_fixtures.py`; its captured output is
`verify_v17_route_fixtures.log`.

## 1. Strict original-space metric

Take the chart $J=-I$. It leaves a covariant metric fixed and changes the cue
sign:

$$
J^{-T}gJ^{-1}=g,\qquad Jx=-x.
$$

Consequently, a deterministic update satisfying V17.2 is forced algebraically
to obey

$$
U(g,-x,c)=(-I)U(g,x,c)(-I)=U(g,x,c). \tag{C1}
$$

For a randomized update, the final contract requires almost every fixed-seed
map $U_\omega$ itself to satisfy covariance pointwise. The same calculation
therefore gives state equality for the same fixed seed $\omega$, which is
strictly stronger than covariance or equality only after averaging. Averaging
then preserves the equality. The paired branches also start from the same
$g_0$ and coupled pre-cue seed, both independent of $s$. This is only the route
calculation; theorem status belongs to the math and audit lanes.

For the exact strict V16 fixture, $g_0=I$, $x=su$, $\lVert u\rVert=1$,
$c=4$, and $\eta=1$. Thus $p=1$ and the rank-one rule gives

$$
g_1=I+3(su)(su)^T=I+3uu^T. \tag{C2}
$$

Both sign branches therefore have the same state. In addition, an ordinary
quadratic readout on opposite action vectors is even:

$$
(au)^Tg_1(au)=u^Tg_1u=4,\qquad a\in\{-1,+1\}. \tag{C3}
$$

## 2. Homogeneous SPD lift

Write the single lifted metric in blocks:

$$
G=\begin{pmatrix}Q&b\\b^T&\gamma\end{pmatrix}\in\operatorname{SPD}(d+1).
$$

Relative to an original $d$-dimensional SPD metric, the covector cross block
$b$ adds $d$ ambient real state coordinates and $\gamma$ adds one. Hence

$$
\frac{(d+1)(d+2)}2-\frac{d(d+1)}2=d+1. \tag{C4}
$$

For $z_s=(su,1)$, $G_0=I_{d+1}$, $c=4$ and $\eta=1$, the prediction is
$p=z_s^Tz_s=2$. The V16 rule is therefore

$$
G_1=I_{d+1}+\frac12z_sz_s^T. \tag{C5}
$$

For $y_a=(au,-1)$,

$$
\begin{aligned}
y_a^TG_1y_a
&=\lVert y_a\rVert^2+\frac12(z_s^Ty_a)^2\\
&=2+\frac12(sa-1)^2.
\end{aligned} \tag{C6}
$$

Thus the correct action has cost $2$, the wrong action has cost $4$, and the
wrong-minus-correct margin is exactly $2$.

For a spatial chart $J$, set $A=\operatorname{diag}(J,1)$,
$G'=A^{-T}GA^{-1}$, $z'=Az$, and $y'=Ay$. Both the update and readout commute
with transport:

$$
G_1'=A^{-T}G_1A^{-1},\qquad y'^TG_1'y'=y^TG_1y. \tag{C7}
$$

This is linear-chart covariance under the embedded spatial group
$\{\operatorname{diag}(J,1):J\in GL(d)\}$, not unrestricted $GL(d+1)$: the
last coordinate and block decomposition are a declared homogeneous splitting.
Although a Cholesky factor packages all entries in one persistent field, its
new block still contains a covector and scalar and therefore additional memory
content.

## 3. Explicit eligibility covector

Permit a second persistent field $e\in V^*$ and write the cue as

$$
e_s=g(su). \tag{C8}
$$

At $g=I$, use the terminal action vector $v_a=au$ and maximize the invariant
pairing:

$$
e_s(v_a)=as. \tag{C9}
$$

The correct score is $1$, the wrong score is $-1$, and the score margin is
$2$. Under $x'=Jx$, $g'=J^{-T}gJ^{-1}$, the eligibility field must transform
as $e'=J^{-T}e$, so

$$
e'(Jv)=e(v). \tag{C10}
$$

This route adds exactly $d$ persistent real components. Treating $e$ as a
contravariant vector instead is a chart error, except for special orthogonal
charts.

## 4. Randers directional geometry

A Randers candidate stores the same kind of cue-odd information as a one-form,
but installs it in the action geometry:

$$
F(v)=\sqrt{v^Tgv}+\beta(v),\qquad
\lVert\beta\rVert_{g^{-1}}<1. \tag{C11}
$$

Choose $\beta_s=-\kappa g(su)$ with $0<\kappa<1$. At $g=I$,

$$
F(au)=1-\kappa as. \tag{C12}
$$

Minimization selects $a=s$ with cost margin $2\kappa$. The transform
$\beta'=J^{-T}\beta$ preserves both terms in C11 and preserves the norm bound.
The route adds $d$ persistent components, just like explicit eligibility; it
does not manufacture directional information from $g$ alone.

## 5. Signed original-$g$ update with an anchor

Let $\alpha\in V^*$ be a separately declared nonzero oriented anchor. In a
fixture with $g=I$, normalize $\alpha$ to a unit covector $w$ and use

$$
g_s=I+\kappa(w+su)(w+su)^T,\qquad \kappa>0. \tag{C13}
$$

This is SPD and the two signs give distinct matrices. Expanding C13 isolates
the cue-odd cross term:

$$
g_s-I-\kappa(ww^T+uu^T)
=s\kappa(wu^T+uw^T). \tag{C14}
$$

An explicit anchor-aware readout has

$$
w^T\left[g_s-I-\kappa(ww^T+uu^T)\right]u
=s\kappa\left(1+(w^Tu)^2\right), \tag{C15}
$$

which never vanishes. Intrinsically, replace $u$ by $gu$ in the covector sums
and use $g^{-1}\alpha$ in contractions. If $\alpha$ is transported as
$J^{-T}\alpha$, the augmented pair $(g,\alpha)$ is $GL(d)$-covariant. If a
coordinate copy of $\alpha$ is held fixed, covariance is only under its
stabilizer

$$
\{J\in GL(d):J^{-T}\alpha=\alpha\}. \tag{C16}
$$

A normalized anchor has $d-1$ continuous orientation parameters plus a
polarity; the ledger conservatively records all $d$ transported components.
Calling C13 "metric only" while omitting this structure would hide the symmetry
breaker.

There is also a decisive readout boundary. Even after C13,

$$
(-u)^Tg_s(-u)=u^Tg_su. \tag{C17}
$$

Thus the anchor route cannot solve the fixture using only minimum quadratic
cost on the opposite action vectors. It needs the explicit anchor-aware,
baseline-aware readout C15. Multiple writes can also destroy the simple
one-step baseline decomposition unless another rule or state is added.

## 6. Recursive SCC copies

Suppose every component state map is sign-blind, so its paired states obey
$g_i^+=g_i^-$. The oriented terminal reference $u$ is public but identical on
the two sign branches. A deterministic, sign-independent message or
aggregation map $A_N$ therefore receives identical tuples and the same $u$:

$$
A_N(g_1^+,\ldots,g_N^+;u)=A_N(g_1^-,\ldots,g_N^-;u). \tag{C18}
$$

The same statement iterates at every finite nesting depth when the entire joint
seed family, topology and initial states are jointly independent of $s$, and
the paired branches use the same realization of that joint family. Marginal
independence of individual seeds would not suffice. Raw storage grows from $m$
to $Nm$, but cue-odd information remains zero. For countably many components,
equality of every compatible finite cylinder is enough only after the state
space carries the product sigma-algebra and the terminal policy is measurable.
Without those declarations, C18 is a finite-prefix calculation, not an
infinite-system conclusion.

## 7. Conditional information and coordinate-count boundary

Fix the public oriented reference at $U=u$. If one terminal state were shared
by both signs, the common policy could not be exact on both. An exact solver
therefore needs

$$
H(S\mid G_T,U)=0,\qquad I(S;G_T\mid U)=1\ \text{bit} \tag{C19}
$$

for uniform $S$. The conditioning matters: when $U$ is isotropically
randomized, the marginal $I(S;G_T)$ can be zero even though C19 holds. This is
an information separation on the registered task, not a general capacity
bound and not a conversion between one bit and one continuous state
coordinate. Without regularity, finite precision or noise assumptions, exact
real entries can encode arbitrarily long sign-even histories.

## 8. Dimensionless audit

The V17 protocol declares all synthetic coordinates, costs and metrics
dimensionless. Therefore $p$, $c$, $p/c$, the homogeneous coordinate $1$,
$e(v)$, $\beta(v)$, $F(v)$, $\kappa$, losses and regrets are dimensionless.
Every logarithm inherited from V16 receives only $p/c$. This is algebraic
dimensional consistency and supplies no biological or physical evidence.

## 9. Development calculation record

The independent script evaluated both cue signs on all 64 registered
development seeds. It found exact strict-state equality, correct analytic
actions for the lift, eligibility, Randers and explicit-anchor readout, and the
following largest chart defects:

| Quantity | Development value |
|---|---:|
| homogeneous quadratic cost | $2.34\times10^{-15}$ |
| homogeneous transported update | $5.16\times10^{-16}$ |
| eligibility pairing | $6.67\times10^{-16}$ |
| Randers cost | $1.19\times10^{-15}$ |
| anchored transported update | $3.94\times10^{-16}$ |

The smallest computed homogeneous margin was
$1.9999999999999996$. No seed in the confirmation block was opened.
