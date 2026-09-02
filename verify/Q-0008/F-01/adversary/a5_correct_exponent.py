"""Adversary 5: the correct second-order combinatorics gives a DIFFERENT heritable exponent.

Because tl[gram(Y)] = -(n-1) sum_v tl gram(eta_v) + sum_{v!=w} tl G(v,w) and eta_v is
linear in the label, the leading fluctuation is that of sum_v tl[Q(label_v)], a sum of
QUADRATIC forms.  For jointly Gaussian labels Cov(Q(a),Q(b)) ~ Cov(a,b)^2, so

    Var( sum_v tl Q(label_v) )  ~  W2p(n) := sum_{v,w} (depth(v /\ w)+1)^2
                                          =  sum_u (2 depth(u)+1) |sub(u)|^2

replaces the card's W2(n) = sum_u |sub(u)|^2.  Predictions become
    gamma_her = d ln( sqrt(W2p)/n ) / d ln n        (card: uses W2 -> +1/4)
    RMS_her/RMS_iid at n = sqrt( W2p(n) / n )       (card: sqrt(W2(n)/n) = 11.1528)
"""
import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"verify"/"Q-0008"/"F-01"))
import check_modes as CM

def tree_stats(n, rng):
    parent=CM.uniform_rooted_tree(n,rng)
    ch=[[] for _ in range(n)]
    for v,p in enumerate(parent):
        if p>=0: ch[p].append(v)
    root=parent.index(-1); order=[root]; i=0
    while i<len(order):
        x=order[i]; i+=1; order.extend(ch[x])
    depth=np.zeros(n,dtype=np.int64); sub=np.ones(n,dtype=np.int64)
    for v in order:
        p=parent[v]
        if p>=0: depth[v]=depth[p]+1
    for v in reversed(order):
        p=parent[v]
        if p>=0: sub[p]+=sub[v]
    W2=int(np.sum(sub*sub)); W2p=int(np.sum((2*depth+1)*sub*sub))
    return W2,W2p

def slope(sz,v): return float(np.polyfit(np.log(np.asarray(sz,float)),np.log(np.asarray(v,float)),1)[0])

SIZES=(8,16,32,64,128); M=40000
rng=np.random.default_rng(424242)
EW2=[];EW2p=[]
for n in SIZES:
    a=[];b=[]
    for _ in range(M):
        w2,w2p=tree_stats(n,rng); a.append(w2); b.append(w2p)
    EW2.append(float(np.mean(a))); EW2p.append(float(np.mean(b)))
    print(f"  n={n:<4} E W2={EW2[-1]:12.2f}  E W2'={EW2p[-1]:14.2f}  (MC, {M} trees)")
print("\nCARD  (uses W2 ):  gamma_her =", round(slope(SIZES,[math.sqrt(x)/n for x,n in zip(EW2,SIZES)]),4),
      "   ratio(32) =", round(math.sqrt(EW2[2]/32),4), "   [card: 0.2261 / 11.1528]")
print("TRUE  (uses W2'):  gamma_her =", round(slope(SIZES,[math.sqrt(x)/n for x,n in zip(EW2p,SIZES)]),4),
      "   ratio(32) =", round(math.sqrt(EW2p[2]/32),4))
print("   card K2 window [0.126,0.326]; card K3 window [8.92,13.38]")

# asymptotics of W2'
BIG=(64,256,1024,4096); MB=1500
rng=np.random.default_rng(99)
vals=[]
for n in BIG:
    s=[tree_stats(n,rng)[1] for _ in range(MB)]
    vals.append(float(np.mean(s)))
for i in range(len(BIG)-1):
    print(f"  d ln E W2' / d ln n on [{BIG[i]},{BIG[i+1]}] = {math.log(vals[i+1]/vals[i])/math.log(BIG[i+1]/BIG[i]):.4f}  (=> gamma_her = that/2 - 1)")
print("  asymptotic W2' ~ n^3 would give gamma_her = +1/2 -- exactly the 'chain' value the card")
print("  pre-registers as the death of the d_tree=2 universality claim.")
