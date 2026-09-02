"""Adversary 9: low-noise clincher at small n (delta=0.001, seed 4242 - NOT the
pre-registered delta/seed/grid).  Card vs corrected driver, RMS_her/RMS_iid."""
import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"verify"/"Q-0008"/"F-01"))
import check_modes as CM
DELTA=0.001; T=2500
CARD={8:4.131,16:6.738}; CORR={8:1.989,16:4.006}
res={}
for n in (8,16):
    rh=np.random.default_rng(4242+n); ri=np.random.default_rng(8484+n)
    her=np.array([CM.sample_her(n,rh,DELTA) for _ in range(T)])
    iid=np.array([CM.sample_iid(n,ri,DELTA) for _ in range(T)])
    Rh=float(np.sqrt(np.mean(her**2))); Ri=float(np.sqrt(np.mean(iid**2)))
    # jackknife-free bootstrap se of the ratio
    bs=[]
    rr=np.random.default_rng(5)
    for _ in range(400):
        ih=rr.integers(0,T,T); ii=rr.integers(0,T,T)
        bs.append(np.sqrt(np.mean(her[ih]**2))/np.sqrt(np.mean(iid[ii]**2)))
    se=float(np.std(bs))
    res[n]=(Rh,Ri,Rh/Ri,se)
    print(f"  n={n:<3} measured ratio = {Rh/Ri:.3f} +- {se:.3f} (bootstrap)   CARD {CARD[n]:.3f}   CORRECTED {CORR[n]:.3f}")
sl=math.log(res[16][0]/res[8][0])/math.log(2)
print(f"\n  measured her slope(8->16) = {sl:+.3f}")
print("  card exact-comb slope(8->16) =", round(math.log(math.sqrt(726.4)/16/(math.sqrt(136.6)/8))/math.log(2),3))
print("  corrected     slope(8->16) =", round(math.log(math.sqrt(240.7)/16/(math.sqrt(27.7)/8))/math.log(2),3))
