# Mathematics

Status: COMPLETE

The naive route that reused sparse BA-TR3 weights was rejected: their nonzero pattern already names the two correct branch supports, so a weight-only top-four rule can recover the answer. BA-TR6 removes that shortcut by giving all 32 candidate entry edges the same nonzero weight and by denying the compiler access to $W$.

$E,z,C,\Theta,q,W$ and normalized runtime states are dimensionless. Counts and edge cardinalities are unitless. Episode normalization is defined only when the positive eligibility sum exceeds zero; cue normalization requires $n_x>0$. Top-four is discrete and no gradient through the mask is claimed.

The shared hidden block makes the selected support behaviorally identifiable: the correct four-edge matching carries exactly one of the two simultaneous source payloads into $H$ and then through the common trunk to $Y$. There is no alternative hidden branch that can deliver the same endpoint through a different support.

The pooled control must average the two cue columns, not pool raw episodes. Otherwise the unavoidable 8:4 exposure imbalance breaks the intended tie. A fourth/fifth tie is an abstention, never a lexicographic action.
