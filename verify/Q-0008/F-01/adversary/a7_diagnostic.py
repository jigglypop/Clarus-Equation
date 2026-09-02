"""Adversary 7: DIAGNOSTIC (deliberately NOT the pre-registered kill statistic:
delta=0.002 not 0.02, seed 777 not 20260902, sizes {8,32} not the 5-size grid).
Distinguishes card prediction (label-sum W2) from corrected prediction (W2').
   card      : ratio(32) = 11.15 , slope(8->32) = +0.218
   corrected : ratio(32) ~ 26.9  , slope(8->32) ~ +0.478
   iid       : slope(8->32) = -0.5 in both
"""
import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"verify"/"Q-0008"/"F-01"))
import check_modes as CM

DELTA=0.002; TRIALS=300
out={}
for n in (8,32):
    rng=np.random.default_rng(777+n)
    her=[CM.sample_her(n,rng,DELTA) for _ in range(TRIALS)]
    rng2=np.random.default_rng(9000+n)
    iid=[CM.sample_iid(n,rng2,DELTA) for _ in range(TRIALS)]
    out[n]=(float(np.sqrt(np.mean(np.square(her)))), float(np.sqrt(np.mean(np.square(iid)))))
    print(f"  n={n:<3} RMS_her={out[n][0]:.6e}  RMS_iid={out[n][1]:.6e}  ratio={out[n][0]/out[n][1]:.3f}")
sl_h=math.log(out[32][0]/out[8][0])/math.log(4); sl_i=math.log(out[32][1]/out[8][1])/math.log(4)
print(f"\n  measured her slope(8->32) = {sl_h:+.4f}   [card exact-comb prediction +0.218 | corrected +0.478]")
print(f"  measured iid slope(8->32) = {sl_i:+.4f}   [both predict -0.500]")
print(f"  measured ratio at n=32    = {out[32][0]/out[32][1]:.3f}  [card 11.15 (window 8.92-13.38) | corrected ~26.9]")
