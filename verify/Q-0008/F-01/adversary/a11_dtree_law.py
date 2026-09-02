"""Adversary 11: the corrected universal law.  Driver = E||H kappa H||_F^2,
kappa = A A^T with A[v,u]=1 iff u is an ancestor-or-self of v.
gamma_her = d ln( sqrt(E||H kappa H||^2)/n )/d ln n.  Test against families of known d_tree."""
import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"verify"/"Q-0008"/"F-01"))
import check_modes as CM

def driver_from_parent(parent):
    n=len(parent); ch=[[] for _ in range(n)]
    for v,p in enumerate(parent):
        if p>=0: ch[p].append(v)
    root=parent.index(-1); order=[root]; i=0
    while i<len(order):
        x=order[i]; i+=1; order.extend(ch[x])
    A=np.zeros((n,n),dtype=np.float64)
    for v in order:
        p=parent[v]
        if p>=0: A[v]=A[p]
        A[v,v]=1.0
    k=A@A.T
    r=k.mean(axis=1); m=k.mean()
    K=k-r[:,None]-r[None,:]+m
    return float(np.sum(K*K))

def slope(sz,v): return float(np.polyfit(np.log(np.asarray(sz,float)),np.log(np.asarray(v,float)),1)[0])
def chain(n): return [-1]+list(range(n-1))
def star(n):  return [-1]+[0]*(n-1)
def binary(n):
    p=[-1]+[ (i-1)//2 for i in range(1,n)]; return p

print("family          d_tree   sizes                gamma_her (= slope of sqrt(driver)/n)   1/d_tree")
for name,fn,dt in (("chain",chain,"1"),("balanced binary",binary,"inf (log depth)"),("star",star,"0 (depth 1)")):
    S=(16,32,64,128,256,512)
    vals=[driver_from_parent(fn(n)) for n in S]
    print(f"  {name:15s} {dt:15s} {S}  {slope(S,[math.sqrt(v)/n for v,n in zip(vals,S)]):+.4f}")
S=(16,32,64,128,256,512); M=120
rng=np.random.default_rng(2026)
vals=[float(np.mean([driver_from_parent(CM.uniform_rooted_tree(n,rng)) for _ in range(M)])) for n in S]
print(f"  {'uniform Cayley':15s} {'2':15s} {S}  {slope(S,[math.sqrt(v)/n for v,n in zip(vals,S)]):+.4f}   1/2 = 0.5")
for i in range(len(S)-1):
    print(f"      local slope [{S[i]},{S[i+1]}] = {math.log((math.sqrt(vals[i+1])/S[i+1])/(math.sqrt(vals[i])/S[i]))/math.log(S[i+1]/S[i]):+.4f}")
print("\n  card law: gamma_her = 1/(2 d_tree) = 0.25 at d_tree=2.  Corrected law: gamma_her = 1/d_tree.")
