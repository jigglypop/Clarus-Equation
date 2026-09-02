import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "verify" / "Q-0015" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
import check_theta as C
rng = np.random.default_rng(1)
t0 = time.time()
for _ in range(200):
    C.eps_and_theta(C.block_triple(rng.standard_normal((3, 4, 4))))
print("3-cell per trial ms:", (time.time() - t0) / 200 * 1000)
t0 = time.time()
for _ in range(10):
    C.eps_and_theta(C.block_triple(rng.standard_normal((128, 4, 4))))
print("128-cell per trial ms:", (time.time() - t0) / 10 * 1000)
