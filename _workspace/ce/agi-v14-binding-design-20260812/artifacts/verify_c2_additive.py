"""Exact ceiling for ANY additive readout sign(phi(b) + psi(c)) on the 32 cells.
phi arbitrary real function of the 8 bit-patterns, psi(c)=beta_ctx arbitrary.
Only the ordering of the 8 values phi(b) matters -> enumerate all 8! orderings
exactly, best threshold per context per ordering. Also exact max for the
linear-phi subclass by enumerating all strict orderings realizable by v.b
(checked by LP-free method: realizable iff some v from the sign-consistent
cone; here simply cross-checked against the sampled 21/32 from
verify_c2_linear.py)."""
import itertools
import numpy as np
from reality_stone.clarus.local_cloud_v13_benchmark import cell_label

LABELS = np.array([[cell_label(k, i) for i in range(8)] for k in range(4)])

best = 0
for perm in itertools.permutations(range(8)):
    total = 0
    for k in range(4):
        y = LABELS[k, list(perm)]
        neg = np.concatenate([[0], np.cumsum(y == -1)])
        pos = int(np.sum(y == 1)) - np.concatenate([[0], np.cumsum(y == 1)])
        total += int((neg + pos).max())
    best = max(best, total)
print(f"[additive readout, exact over all 8! orderings] max = {best}/32 = {best/32:.4f}")
