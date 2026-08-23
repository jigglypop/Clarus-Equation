# BA-TR27 Z3 twisted-content transfer

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

This run retains BA-TR26's 30-coordinate runtime, one-shot delayed packet,
current-column binding, equal-norm positive six-column template, and fresh
episode coordinate remaps.  It changes only the content-composition law so the
global affine/additive arm is false on the eight observed rows.

For an outcome-blind hidden class $\kappa\in\{1,2\}$ and cells
$a,b\in\mathbb Z_3$, the observed dimensionless content is

$$
y_{ab}=A_a+B_{(b+\kappa a)\bmod3}\in\mathbb R^6.
$$

The harness alternates $\kappa=1,2$ by seed parity, but the learner receives
neither seed nor $\kappa$, cell coordinates, factor names, packet roles,
source coordinates, target, decoder, reward, endpoint, store, or query
content.  It receives eight shuffled raw cues with their current content sums
and the ninth raw query cue.

The learner enumerates all additive Cartesian charts of the nine raw cues
using only

$$
r_{ab}-r_{a0}-r_{0b}+r_{00}=0.
$$

For every chart and candidate $k\in\{0,1,2\}$ it builds the signless incidence
matrix $M_k\in\{0,1\}^{8\times6}$,

$$
(M_k)_{(a,b),a}=1,
\qquad
(M_k)_{(a,b),3+(b+ka)\bmod3}=1,
$$

fits

$$
\Theta_k=M_k^+Y,
\qquad
\rho_k=\frac{\|M_k\Theta_k-Y\|_F}{\max(\|Y\|_F,\epsilon)},
$$

and predicts each cue with its corresponding incidence row.  Operational SVD
rank uses the relative cutoff $10^{-10}\sigma_{\max}$.  Every admitted
nonzero-twist candidate must satisfy

$$
\operatorname{rank}M_k
=\operatorname{rank}\begin{pmatrix}M_k\\e_q^\top\end{pmatrix}=5,
\qquad \rho_k\le10^{-10}.
$$

All zero-residual chart/gauge candidates must agree on the query prediction to
relative error at most $10^{-10}$.  The best additive arm $k=0$ must have
$\rho_0\ge10^{-3}$; otherwise the apparatus stops before endpoint.  Current
packet binding uses the same best/second-best relative margin $>10^{-6}$ as
BA-TR26.

The query is cell `22`, omitted from the eight content rows but included as a
raw cue.  Every episode uses a fresh injective six-column map; the query moves
all columns into the second input block.  Recall emits the two true twisted
packets plus one matched nonzero distractor once.  Required receipts remain
`[0,0,0,3,0,0,0]` and `[0,3,0,0,0,0,0]`.

Frozen controls: additive $k=0$ rejection, absent-row joint lookup,
canonical-coordinate memorizer, wrong raw cue, cue/content association shuffle
rejected before endpoint, packet-binding shuffle, and no-context/all-packet
routing.  Row order and orthogonal cue-chart changes must preserve the query
prediction.  Learned recall must match the remapped oracle and the union of
two atomic current-snapshot responses.

Fresh calibration seed: `114001`.  Fresh development seeds: `114101..114116`.
No formula, family, cutoff, threshold, seed, event, runtime weight, remap,
distractor, or control may change after calibration.

Claim ceiling: the family itself is predeclared.  Replacing only the missing
value by $y_{22}+\delta$ leaves all eight observations unchanged, so no finite
run identifies arbitrary interactions.  A pass means only that, inside the
declared $\mathbb Z_3$ twisted-composition family, opaque cue geometry and
eight content observations select a nonadditive coupling class and route the
ninth cell.  It is not unrestricted interaction/factor discovery, biological
routing or memory, curvature, folding, physical energy, or AGI evidence.
