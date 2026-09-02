"""Adversary 1: exact E[W_2(n)] (log-space), card numbers, alternatives, asymptotics."""
import math
from fractions import Fraction
import numpy as np

def w2_frac(n):
    total = Fraction(n*n)
    for k in range(1,n):
        total += Fraction(math.comb(n,k)*k**(k+1)*(n-k)**(n-k), n**(n-1))
    return float(total)

def w2_log(n):
    if n==1: return 1.0
    lg=math.lgamma; ln=math.log
    s=float(n*n)
    for k in range(1,n):
        t=(lg(n+1)-lg(k+1)-lg(n-k+1))+(k+1)*ln(k)+((n-k)*ln(n-k) if n-k>0 else 0.0)-(n-1)*ln(n)
        s+=math.exp(t)
    return s

for n in (1,2,3,8,32):
    print(f"  cross-check n={n}: Fraction={w2_frac(n):.10g}  log-space={w2_log(n):.10g}")

def slope(sz,v): return float(np.polyfit(np.log(np.asarray(sz,float)),np.log(np.asarray(v,float)),1)[0])
S=(8,16,32,64,128)
print("CARD her slope  =", round(slope(S,[math.sqrt(w2_log(n))/n for n in S]),6), "| card claims 0.2261")
print("CARD ratio n=32 =", round(math.sqrt(w2_log(32)/32),6), "| card claims 11.1528")
ch=[math.sqrt(n*(n+1)*(2*n+1)/6)/n for n in S]; st=[math.sqrt(n*n+(n-1))/n for n in S]
print("chain slope =",round(slope(S,ch),6),"| card 0.471   star slope =",round(slope(S,st),6),"| card -0.017")
print("chain ratio32 =",round(math.sqrt(32*33*65/6/32),6),"| card 18.908   star ratio32 =",round(math.sqrt((1024+31)/32),6),"| card 5.742")
print("--- asymptotics of E W2 ---")
for a,b in [(8,128),(128,1024),(1024,8192),(8192,60000)]:
    print(f"  dlnEW2/dlnn [{a},{b}] = {math.log(w2_log(b)/w2_log(a))/math.log(b/a):.5f}  (claim 5/2)")
print("--- local her slope (asymptote claim +0.25) ---")
for a,b in [(8,128),(128,1024),(1024,8192),(8192,60000)]:
    ha,hb=math.sqrt(w2_log(a))/a,math.sqrt(w2_log(b))/b
    print(f"  [{a},{b}] = {math.log(hb/ha)/math.log(b/a):.5f}")
K1=(8,16,32,64,128,256)
print("K1 sqrt(N-1)/N slope",K1,"=",round(slope(K1,[math.sqrt(N-1)/N for N in K1]),6),"| card note -0.483")
print("K1 sqrt(N-1)/N slope (8,16,32,64) =",round(slope((8,16,32,64),[math.sqrt(N-1)/N for N in (8,16,32,64)]),6))
