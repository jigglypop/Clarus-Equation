"""Adversary 8: the EXACT block-residual identity and the correct combinatorial driver.

Writing X_v = Sigma_0 + eta_v (each X_v exactly simple), Y = sum_v X_v, eta_bar = S/n:
      tl[ gram(Y) ]  =  -n * sum_v tl[ gram(eta_v - eta_bar) ]        (EXACT)
so    eps_block(n) = | sum_v tl gram(eta_v - eta_bar) | / ( n * ||gram(Y)||/n^2 ).
Checks: (i) identity numerically; (ii) it reproduces 13.5 coherent no-go exactly;
(iii) since eta_v ~ delta*L(label_v), the driver is a CENTERED quadratic form, so
      RMS ∝ sqrt( E ||H kappa H||_F^2 ) / n ,  kappa_{vw} = depth(v ^ w)+1, H = I - J/n
      -- NOT the card's sqrt(E W2)/n with W2 = sum_u |sub(u)|^2 = ||kappa||_{1,1}-ish.
For iid kappa=I: ||H I H||_F^2 = n-1  ->  sqrt(n-1)/n  (the card's own K1 finite-N note).
"""
import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"verify"/"Q-0008"/"F-01"))
import check_modes as CM
from examples.physics.gravity.causal_face_simplicity import (
    geometric_self_dual_triple, plebanski_gram, wedge_scalar, simplicity_residual)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment
REF=geometric_self_dual_triple(np.eye(4))
def tl(M): return M-np.trace(M)/3.0*np.eye(3)
def G(A,B): return np.array([[wedge_scalar(A[i],B[j]) for j in range(3)] for i in range(3)])

print("== (i) exact identity, n=7, delta=0.15 ==")
rng=np.random.default_rng(21); n,d=7,0.15
Xs=[optimal_internal_alignment(REF,geometric_self_dual_triple(np.eye(4)+d*rng.normal(size=(4,4)))).aligned_candidate for _ in range(n)]
Y=sum(Xs); et=[X-REF for X in Xs]; bar=sum(et)/n
lhs=tl(plebanski_gram(Y)); rhs=-n*sum(tl(G(e-bar,e-bar)) for e in et)
print("   ||lhs-rhs||/||lhs|| =", float(np.linalg.norm(lhs-rhs)/np.linalg.norm(lhs)))
print("== (ii) identity => 13.5 no-go is automatic (all eta equal => residual 0) ==")
Z=sum([Xs[0]]*5); print("   residual of 5 IDENTICAL simple cells =", simplicity_residual(Z))

print("\n== (iii) combinatorial driver: ||H kappa H||_F^2 vs the card's W2 ==")
def kappa_stats(n,rng):
    parent=CM.uniform_rooted_tree(n,rng)
    ch=[[] for _ in range(n)]
    for v,p in enumerate(parent):
        if p>=0: ch[p].append(v)
    root=parent.index(-1); order=[root]; i=0
    while i<len(order):
        x=order[i]; i+=1; order.extend(ch[x])
    depth=np.zeros(n,dtype=np.int64); anc=[None]*n
    for v in order:
        p=parent[v]
        if p>=0:
            depth[v]=depth[p]+1; anc[v]=anc[p]|{v}
        else: anc[v]={v}
    A=np.zeros((n,n))
    for v in range(n):
        for w in range(n):
            A[v,w]=len(anc[v]&anc[w])
    sub=np.ones(n,dtype=np.int64)
    for v in reversed(order):
        if parent[v]>=0: sub[parent[v]]+=sub[v]
    H=np.eye(n)-np.ones((n,n))/n
    K=H@A@H
    return float(np.sum(sub*sub)), float(np.sum(K*K))
def slope(sz,v): return float(np.polyfit(np.log(np.asarray(sz,float)),np.log(np.asarray(v,float)),1)[0])
S=(8,16,32,64,128); M=3000
rng=np.random.default_rng(31337); EW2=[]; EK=[]
for n in S:
    a=[];b=[]
    for _ in range(M):
        w2,kk=kappa_stats(n,rng); a.append(w2); b.append(kk)
    EW2.append(float(np.mean(a))); EK.append(float(np.mean(b)))
    print(f"   n={n:<4} E W2={EW2[-1]:11.1f}   E||H k H||_F^2={EK[-1]:13.1f}   pred RMS-ratio her/iid = {math.sqrt(EK[-1]/(n-1)):8.3f}")
print("\n   CARD    gamma_her =", round(slope(S,[math.sqrt(x)/n for x,n in zip(EW2,S)]),4),
      "  ratio(32) =", round(math.sqrt(EW2[2]/32),3), "   [card 0.2261 / 11.1528, windows 0.126-0.326 / 8.92-13.38]")
print("   CORRECT gamma_her =", round(slope(S,[math.sqrt(x)/n for x,n in zip(EK,S)]),4),
      "  ratio(32) =", round(math.sqrt(EK[2]/31),3))
print("   (iid control: slope of sqrt(n-1)/n over these sizes =", round(slope(S,[math.sqrt(n-1)/n for n in S]),4),")")
# chain / star exact under the CORRECT driver
for name,kap in (("chain",lambda n: np.minimum.outer(np.arange(1,n+1),np.arange(1,n+1)).astype(float)),
                 ("star", lambda n: np.ones((n,n))+np.eye(n)*0)):
    vals=[]
    for n in S:
        A=kap(n)
        if name=="star":
            A=np.ones((n,n)); A[np.arange(1,n),np.arange(1,n)]=2.0; A[0,0]=1.0
        H=np.eye(n)-np.ones((n,n))/n; K=H@A@H; vals.append(float(np.sum(K*K)))
    print(f"   {name:6s} CORRECT gamma = {slope(S,[math.sqrt(v)/n for v,n in zip(vals,S)]):.4f}  ratio(32) = {math.sqrt(vals[2]/31):.3f}")
