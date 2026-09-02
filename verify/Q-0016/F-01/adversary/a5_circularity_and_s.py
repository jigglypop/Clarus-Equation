"""Q-0016 F-01 adversary a5: martingale ambiguity, symmetric circularity, s-tolerance, family ratios."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0016" / "F-01"))
from a1_algebra import A_matrix, C_matrix, D_f02, D_split, cbin  # noqa: E402
from driver_numbers import qspine_block, uniform_rooted_tree  # noqa: E402
from check_modes import heritable_labels  # noqa: E402
from predict_split_kernel import split_labels  # noqa: E402

OUT = HERE / "a5_circularity_and_s.json"
R: dict = {}

p = cbin(2)
n = len(p)
ch = [[] for _ in range(n)]
for v, q in enumerate(p):
    if q >= 0:
        ch[q].append(v)
rng = np.random.default_rng(20260902)
gap_iid = []
gap_split = []
for _ in range(100000):
    xi = rng.normal(size=n)
    li = heritable_labels(p, xi)
    ls = split_labels(p, xi)
    for z, kids in enumerate(ch):
        if len(kids) >= 2:
            gap_iid.append(float(li[kids].mean() - li[z]))
            gap_split.append(float(ls[kids].mean() - ls[z]))
gi = np.asarray(gap_iid)
gs = np.asarray(gap_split)
R["martingale"] = {
    "tree": "complete binary d=2 (n=7)",
    "iid_mean": float(gi.mean()), "iid_sd": float(gi.std()),
    "iid_se_of_mean": float(gi.std() / math.sqrt(gi.size)),
    "split_mean": float(gs.mean()), "split_sd": float(gs.std()),
    "split_max_abs": float(np.max(np.abs(gs))),
    "verdict": ("F-02 iid increments ALREADY give E[child label | parent] = parent label, i.e. the "
                "tree-martingale property in expectation. F-01 strengthens it to an almost-sure pathwise "
                "identity. Section 11.6 lambda = D(1-m) = 1 is first-moment criticality of the offspring "
                "COUNT, so it does not by itself select the pathwise version."),
}
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("martingale:", json.dumps(R["martingale"], indent=1, default=float))
