import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(r"C:/dev/ce/Clarus-Equation"); sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"verify"/"Q-0008"/"F-01"))
import check_modes as CM
D=0.001; T=1200; n=32
rh=np.random.default_rng(4242+n); ri=np.random.default_rng(8484+n)
her=np.array([CM.sample_her(n,rh,D) for _ in range(T)]); iid=np.array([CM.sample_iid(n,ri,D) for _ in range(T)])
Rh=float(np.sqrt(np.mean(her**2))); Ri=float(np.sqrt(np.mean(iid**2)))
rr=np.random.default_rng(5); bs=[np.sqrt(np.mean(her[rr.integers(0,T,T)]**2))/np.sqrt(np.mean(iid[rr.integers(0,T,T)]**2)) for _ in range(400)]
print(f"n=32 measured ratio = {Rh/Ri:.3f} +- {float(np.std(bs)):.3f}   CARD 11.153 (K3 window 8.92-13.38)   CORRECTED ~8.07")
