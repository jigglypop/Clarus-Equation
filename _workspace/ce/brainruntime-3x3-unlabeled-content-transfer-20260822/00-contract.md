# BA-TR26 unlabeled 3x3 affine-content transfer

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

Eight opaque training episodes provide only a normalized raw cue
$r_i\in\mathbb R^8$ and the normalized content sum
$y_i\in\mathbb R^6$ of the two packets present in that episode.  The learner
receives no factor name, grid coordinate, semantic role, source coordinate,
target, decoder, reward, endpoint, store, held-out content, or held-out remap.

Let

$$
X=[r_i-\bar r]_{i\in T}\in\mathbb R^{8\times8},\qquad
Y=[y_i-\bar y]_{i\in T}\in\mathbb R^{6\times8}.
$$

For the thin rank-four SVD $X=U_4\Sigma_4V_4^\top$, the frozen compiler is

$$
K=YV_4\Sigma_4^{-1}U_4^\top,
\qquad
\widehat y(r)=\bar y+K(r-\bar r).
$$

Operational rank uses the relative cutoff
$10^{-10}\sigma_{\max}$; both centered cue and content matrices must have rank
four, $\kappa_2(X)\le10^6$, relative affine-fit residual at most $10^{-10}$,
and the held-out raw cue must have relative affine-span residual at most
$10^{-10}$.  All values and cutoffs are dimensionless.

Before held-out execution, the learner enumerates every disjoint pairing of
four raw training cues.  Exactly five train-only cue parallelograms must be
found, their hypergraph must cover all eight rows, and the corresponding
content parallelogram residuals must be at most $10^{-10}$.  This enumeration
uses no grid labels.

At recall, for the actually arrived packet coordinates $J$ and normalized
current columns $z_j=W_{H\leftarrow j}/\lVert W_{H\leftarrow j}\rVert_2$,

$$
A^*(r)=\arg\min_{A\subset J,\ |A|=2}
\left\lVert\widehat y(r)-\sum_{j\in A}z_j\right\rVert_2.
$$

The best/second-best residual gap divided by
$\max(1,\lVert\widehat y\rVert_2)$ must exceed $10^{-6}$; otherwise binding
fails closed.

The synthetic apparatus has 30 coordinates in five blocks of six.  Six
outcome-blind equal-norm packet columns are a seed-specific permutation of one
fixed positive level template; this tests coordinate/content permutation, not
broad transfer across independently varied content geometries.
Every one of the eight training episodes and the held-out episode uses a fresh
injective map from the six packet contents to twelve physical input
coordinates.  The held-out map moves all six columns into the disjoint second
input block.  Two relevant packets plus one matched nonzero distractor are
emitted once; the delay receipts remain `[0,0,0,3,0,0,0]` and
`[0,3,0,0,0,0,0]`.

Frozen controls: absent-row joint lookup, canonical-coordinate memorizer,
wrong raw cue, cue/content association shuffle rejected before endpoint,
packet-content binding shuffle, rank-three map, and no-context/all-packet
routing.  Row-order and orthogonal raw-cue chart changes must preserve the
prediction.  Learned held-out routing must match a remapped oracle and the
union of two atomic current-snapshot responses.

Fresh calibration seed: `113001`.  Fresh development seeds: `113101..113116`.
No equation, rank, cutoff, cue family, dictionary, remap, distractor, control,
or seed may change after calibration.

Claim ceiling: even five observed rectangles cannot establish the global law
at the missing cell.  The alternative
$y_{22}=y^{\rm affine}_{22}+\delta$ has identical training evidence for any
$\delta\ne0$.  A pass therefore means only conditional rank-four affine
cue/content transfer with current-column coordinate binding in this synthetic
apparatus; it is not factor discovery, biological routing or memory,
curvature, cortical folding, physical energy, or AGI evidence.
