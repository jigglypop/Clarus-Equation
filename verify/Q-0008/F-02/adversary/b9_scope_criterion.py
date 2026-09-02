"""Adversary b9 (re-audit of card revision 2): does the NEW scope clause do what it claims?

Card rev2 scope: "gamma_her = 1/d_tree holds for the tree family whose TYPICAL pair has
kappa_vw = |path(v) cap path(w)| of the same order as the depth n^{1/d_tree}
 -- uniform rooted Cayley and chain are IN, star-of-chains and caterpillar are OUT."

Two questions:
  (1) Is caterpillar really OUT?  b2 (rev1 audit) classified caterpillar as a CONFORMING family
      (gamma -> 1/2, the 0.4758 being a finite-size value).  Push k much further.
  (2) Is the stated criterion (E[kappa_vw] over typical pairs ~ depth) a PREDICTOR of gamma,
      i.e. does membership in the criterion class actually imply gamma = 1/d_tree?
      Measure E[kappa_vw] (uniform random pair) / depth for every family.
"""
import math
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_dtree_counterexample import (driver_fast, star_of_chains, caterpillar, chain,
                                     binary, uniform_cayley, slope)


def typical_kappa_and_depth(parent, rng, pairs=20000):
    """E[|path(v) cap path(w)|] over uniform random pairs, and max depth."""
    n = len(parent)
    depth = np.zeros(n, dtype=np.int64)
    order, root = [], -1
    ch = [[] for _ in range(n)]
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    order = [root]; i = 0
    while i < len(order):
        order.extend(ch[order[i]]); i += 1
    for v in order[1:]:
        depth[v] = depth[parent[v]] + 1
    # ancestor sets via bit paths would be heavy; walk up the deeper vertex
    vs = rng.integers(0, n, size=pairs); ws = rng.integers(0, n, size=pairs)
    tot = 0
    for a, b in zip(vs, ws):
        x, y = int(a), int(b)
        while depth[x] > depth[y]:
            x = parent[x]
        while depth[y] > depth[x]:
            y = parent[y]
        while x != y:
            x = parent[x]; y = parent[y]
        tot += depth[x] + 1            # common ancestors incl. root and the meet itself
    return tot / pairs, int(depth.max())


rng = np.random.default_rng(20260902)

print("== (1) caterpillar: local gamma_her over successively larger decades (exact driver) ==")
for ks in ((6, 9, 13, 19, 28, 40), (40, 60, 90, 135, 200), (200, 300, 450, 675, 1000),
           (1000, 1500, 2250, 3375)):
    ns, vs, dep = [], [], []
    for k in ks:
        p = caterpillar(k); d, dm = driver_fast(p)
        ns.append(len(p)); vs.append(math.sqrt(d) / len(p)); dep.append(dm)
    print(f"   k in {str(ks):32s} n in [{ns[0]},{ns[-1]}]  gamma = {slope(ns, vs):.4f}"
          f"   depth exponent = {slope(ns, dep):.4f}   (card law 1/d_tree = 0.5)")

print("\n== star-of-chains, same treatment (the genuine counterexample) ==")
for ks in ((6, 9, 13, 19, 28, 40), (200, 300, 450, 675, 1000)):
    ns, vs, dep = [], [], []
    for k in ks:
        p = star_of_chains(k); d, dm = driver_fast(p)
        ns.append(len(p)); vs.append(math.sqrt(d) / len(p)); dep.append(dm)
    print(f"   k in {str(ks):32s} n in [{ns[0]},{ns[-1]}]  gamma = {slope(ns, vs):.4f}"
          f"   depth exponent = {slope(ns, dep):.4f}")

print("\n== (2) criterion test: E[kappa_vw] (typical pair) vs depth, and gamma vs 1/d_tree ==")
print(f"{'family':22s} {'n':>7s} {'depth':>7s} {'E[kappa]':>9s} {'E[kappa]/depth':>15s}")
fams = {
    "chain":            [chain(n) for n in (64, 256, 1024)],
    "caterpillar":      [caterpillar(k) for k in (8, 16, 32)],
    "star_of_chains":   [star_of_chains(k) for k in (8, 16, 32)],
    "balanced_binary":  [binary(n) for n in (63, 255, 1023)],
    "uniform_Cayley":   [uniform_cayley(n, rng) for n in (64, 256, 1024)],
}
for name, trees in fams.items():
    for p in trees:
        ek, dm = typical_kappa_and_depth(p, rng, pairs=4000)
        print(f"{name:22s} {len(p):7d} {dm:7d} {ek:9.3f} {ek/max(dm,1):15.4f}")

print("\n== criterion vs measured gamma (summary) ==")
S = (16, 32, 64, 128, 256, 512, 1024)
vals = [driver_fast(chain(n))[0] for n in S]
print("   chain            gamma =", round(slope(S, [math.sqrt(v)/n for v, n in zip(vals, S)]), 4), " 1/d_tree = 1.0")
rng2 = np.random.default_rng(20260902)
vals = [float(np.mean([driver_fast(uniform_cayley(n, rng2))[0] for _ in range(200)])) for n in S]
print("   uniform Cayley   gamma =", round(slope(S, [math.sqrt(v)/n for v, n in zip(vals, S)]), 4), " 1/d_tree = 0.5")
vals = [driver_fast(binary(n))[0] for n in S]
print("   balanced binary  gamma =", round(slope(S, [math.sqrt(v)/n for v, n in zip(vals, S)]), 4),
      " 1/d_tree -> 0 (log depth); E[kappa]/depth is O(1) here too -> criterion needs care")
