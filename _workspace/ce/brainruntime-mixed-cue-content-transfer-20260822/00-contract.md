# BA-TR25 mixed-cue content-coordinate transfer

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

BA-TR24 supplied two factor labels, two separate factor-code tables, and fixed
source coordinates.  BA-TR25 removes those inputs from the learner.  Three
training observations consist only of a raw mixed cue $r_i$ and the sum $y_i$
of the content vectors carried by the two co-occurring packets.

Let

$$
R=[r_1-r_0,\ r_2-r_0],\qquad
D=[y_1-y_0,\ y_2-y_0].
$$

The frozen rank-two affine compiler is

$$
\alpha(r)=(R^\top R)^{-1}R^\top(r-r_0),\qquad
\widehat y(r)=y_0+D\alpha(r).
$$

It is admitted only when $R$ and $D$ have rank two.  The held-out cue must
satisfy the outcome-blind parallelogram receipt
$r_3-r_0=(r_1-r_0)+(r_2-r_0)$.  At recall the compiler sees only the current
arrived packet coordinates and their normalized $H\leftarrow$packet weight
columns.  Among all current packet pairs it selects the unique pair whose
content sum is closest to $\widehat y(r)$.  Ties fail closed.

All four semantic packet columns are moved from their training coordinates to
an unseen permutation of the second input block before the held-out runtime
probe.  A third matched nonzero distractor packet is present.  The learner and
compiler receive no factor tuple, semantic role, source family, target,
decoder, reward, endpoint, store, or expected mask.

Frozen controls are: absent-row joint lookup, absolute-coordinate memorizer,
wrong mixed cue, shuffled packet-content binding, rank-one compiler, and no
context/all-packet routing.  Each must fail while the learned content compiler
matches the remapped oracle bit-for-bit.  The one-shot packet receipt must be
`[0,0,0,3,0,0,0]`, and all stores remain empty.

Fresh calibration seed: `112001`.  Fresh development seeds: `112101..112116`.
No threshold, rank, cue construction, remap, control, or seed may change after
calibration.

This experiment cannot prove distribution-free factor discovery.  With only
three observations an arbitrary fourth mask is nonidentifiable.  A pass means
only conditional recovery of a rank-two additive cue/content subspace and
content-equivariant coordinate transfer in this synthetic apparatus.

