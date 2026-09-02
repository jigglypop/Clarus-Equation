"""A7: how much of 10/9 (and rho_face = sqrt5/3) is forced, and how much is a modelling choice?

12.1 gives the composition face an ORIENTED boundary  d f = e_um + e_mv - e_uv.
The card instead forms  Y_f = sum_{w in {u,m,v}} R_w Sigma(e_w)  -- an unsigned sum of the three
VERTEX bivectors, using no orientation at all.  Test the sensitivity of the pre-registered kernel
to that unstated choice, and to the 11.9-item-1 attachment variants.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "verify" / "Q-0015" / "F-01"))
import check_theta as C  # noqa: E402

out = {}

def D(k):
    k = np.asarray(k, float); m = k.shape[0]
    H = np.eye(m) - np.ones((m, m)) / m
    K = H @ k @ H
    return float(np.sum(K * K))

kchain = np.array([[1., 1, 1], [1, 2, 2], [1, 2, 3]])
kiid = np.eye(3)
variants = {}
for name, s in (("card: (+,+,+) unsigned sum", (1, 1, 1)),
                ("oriented boundary: (+,+,-)", (1, 1, -1)),
                ("(+,-,+)", (1, -1, 1)),
                ("(-,+,+)", (-1, 1, 1))):
    S = np.diag(s)
    variants[name] = {"D_chain": D(S @ kchain @ S), "D_iid": D(S @ kiid @ S),
                      "rho_face": math.sqrt(D(S @ kchain @ S) / D(S @ kiid @ S))}
out["sign_convention_sensitivity"] = variants
out["sign_note"] = ("H_3 centering is only the right projector when all three cells enter with the "
                    "same sign (the O(delta^0) reference term is n*Sigma_0 with n=3).  With the "
                    "oriented boundary the reference term is 1*Sigma_0 and the whole 3-cell derivation "
                    "would have to be redone.  The card never states which it means.")

# empirical: does the signed face even give the same theta scale?
rng = np.random.default_rng(31337)
res = {}
for name, s in (("(+,+,+)", (1, 1, 1)), ("(+,+,-)", (1, 1, -1))):
    her, iid = [], []
    for _ in range(1024):
        xi = rng.standard_normal((10, 4, 4))
        anc = xi[0]; mid = anc + xi[8]; kid = mid + xi[9]
        lab = np.stack([anc, mid, kid]) * np.asarray(s)[:, None, None]
        her.append(C.eps_and_theta(C.block_triple(lab))[1])
        lab2 = rng.standard_normal((3, 4, 4)) * np.asarray(s)[:, None, None]
        iid.append(C.eps_and_theta(C.block_triple(lab2))[1])
    h = float(np.sqrt(np.mean(np.array(her) ** 2))); i = float(np.sqrt(np.mean(np.array(iid) ** 2)))
    res[name] = {"theta_her": h, "theta_iid": i, "rho": h / i}
out["sign_convention_monte_carlo_1024"] = res
out["sign_mc_note"] = ("flipping the sign of a LABEL is not the same as flipping the sign of the CELL "
                       "in the block sum; this row only shows the label-symmetry of the Gaussian model. "
                       "The real ambiguity is which bivectors are added, and 12.1 says the boundary is "
                       "oriented while the card adds all three with +.")

# 11.9 item-1 attachment variants (already computed in a2, restated as the decisive kill spread)
paths = {
    "chain u<-m<-v (12.1)": [{0}, {0, 10}, {0, 10, 11}],
    "siblings m,v of u": [{0}, {0, 10}, {0, 11}],
    "cousins (different branches)": [{0}, {0, 10, 12}, {0, 11, 13}],
    "u,m siblings; v child of m": [{0, 10}, {0, 11}, {0, 11, 12}],
}
att = {}
for name, ps in paths.items():
    k = np.array([[len(a & b) for b in ps] for a in ps], float)
    att[name] = {"D": D(k), "rho_face_vs_iid": math.sqrt(D(k) / 2.0),
                 "inside_K4_window_[0.685,0.806]": bool(0.685 <= math.sqrt(D(k) / 2.0) <= 0.806)}
out["attachment_variants"] = att
out["attachment_note"] = ("10/9 is NOT a consequence of 3-chain-ness: the sibling face gives 10/9 too. "
                          "It IS destroyed by the cousin/merge variants of 11.9 item 1 (rho = 1.49, "
                          "outside K4).  So K4 tests the 12.1 attachment rule, not the card's "
                          "curvature-angle mapping.")
print(json.dumps(out, indent=2, ensure_ascii=False))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
