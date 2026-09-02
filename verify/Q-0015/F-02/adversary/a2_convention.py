"""Adversary a2: the card's scope claim

  "coordinate-index transition E_u^{-1} E_v carries xi -> xi^T and gives the SAME RMS
   in the isotropic Gaussian ensemble (scope statement, off-ladder)"

is tested three ways:
  (i)  same product order as the card's script (left action, R_f = R_{k-1..0})
  (ii) per-sample identity theta_coord(xi) =? theta_frame(xi^T)
  (iii) coordinate convention combined with the REVERSED (right-action) product order
Also: the leading-order generator of the coordinate convention, and the resulting
c_theta / rho for the face, i.e. what the K3/K4 windows would have to be.
"""
from __future__ import annotations
import json, math, pathlib
import numpy as np

rng = np.random.default_rng(4242)
OUT = {}


def polar(T):
    U, _, Vt = np.linalg.svd(T)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def theta(R):
    a = np.angle(np.linalg.eigvals(R))
    return math.sqrt(0.5 * float(np.sum(a * a)))


def hol(labels, dl, mode="frame", order="left"):
    E = [np.eye(4) + dl * L for L in labels]
    k = len(E)
    H = np.eye(4)
    idx = range(k)
    for i in idx:
        u, v = E[i], E[(i + 1) % k]
        R = polar(v @ np.linalg.inv(u)) if mode == "frame" else polar(np.linalg.inv(u) @ v)
        H = (R @ H) if order == "left" else (H @ R)
    return H


DL = 0.005
# ---- (ii) per-sample map xi -> xi^T
rows = []
for _ in range(200):
    L = np.cumsum(rng.standard_normal((3, 4, 4)), axis=0)
    LT = np.stack([m.T for m in L])
    rows.append(
        (
            theta(hol(L, DL, "coord", "left")),
            theta(hol(LT, DL, "frame", "left")),
            theta(hol(LT, DL, "frame", "right")),
            theta(hol(L, DL, "frame", "left")),
            theta(hol(L, DL, "coord", "right")),
        )
    )
r = np.asarray(rows)
OUT["ii_max_rel_diff_coordleft_vs_frameleft_transposed"] = float(np.max(np.abs(r[:, 0] / r[:, 1] - 1)))
OUT["ii_max_rel_diff_coordleft_vs_frameright_transposed"] = float(np.max(np.abs(r[:, 0] / r[:, 2] - 1)))
OUT["ii_max_rel_diff_coordright_vs_frameleft_transposed"] = float(np.max(np.abs(r[:, 4] / r[:, 1] - 1)))
OUT["ii_note"] = (
    "coord+left equals frame+right on the transposed sample; the card's identification "
    "needs the product order flipped too, which the card does not say"
)

# ---- (i) RMS comparison in the card's own product order, face (n=3) and chain (n=6)
res = {}
for n, sampler in (("face3_her", lambda: np.cumsum(rng.standard_normal((3, 4, 4)), axis=0)),
                   ("face3_iid", lambda: rng.standard_normal((3, 4, 4))),
                   ("chain6_her", lambda: np.cumsum(rng.standard_normal((6, 4, 4)), axis=0))):
    a, b = [], []
    for _ in range(3000):
        L = sampler()
        a.append(theta(hol(L, DL, "frame", "left")))
        b.append(theta(hol(L, DL, "coord", "left")))
    a, b = np.asarray(a), np.asarray(b)
    res[n] = {
        "rms_frame_over_delta2": float(np.sqrt(np.mean(a ** 2)) / DL ** 2),
        "rms_coord_over_delta2": float(np.sqrt(np.mean(b ** 2)) / DL ** 2),
        "rms_ratio_coord_over_frame": float(np.sqrt(np.mean(b ** 2) / np.mean(a ** 2))),
        "trials": 3000,
    }
OUT["i_rms_same_product_order"] = res

# ---- what the face predictions become under coord+left
f_her = res["face3_her"]["rms_coord_over_delta2"]
f_iid = res["face3_iid"]["rms_coord_over_delta2"]
OUT["i_face_rho_hol_coord_left"] = f_her / f_iid
OUT["i_face_c_theta_her_coord_left"] = f_her / (10.0 / 9.0)
OUT["i_card_windows"] = {"rho_face_hol": [0.540, 0.615], "c_theta_face_her": [1.76, 2.06]}

# ---- leading-order generator under the coordinate convention (Richardson at small delta)
def gen_coeff(labels, mode, order):
    """||log R_f|| / delta^2 extrapolated (delta -> 0)."""
    vals = {}
    for dl in (1e-4, 5e-5):
        H = hol(labels, dl, mode, order)
        vals[dl] = theta(H) / dl ** 2
    return 2 * vals[5e-5] - vals[1e-4]


def omega2(labels):
    k = len(labels)
    o = np.zeros((4, 4))
    for i in range(k):
        a_ = 0.5 * (labels[i] + labels[i].T)
        b_ = 0.5 * (labels[(i + 1) % k] + labels[(i + 1) % k].T)
        o += 0.5 * (a_ @ b_ - b_ @ a_)
    return o


L = np.cumsum(rng.standard_normal((5, 4, 4)), axis=0)
card_coeff = float(np.linalg.norm(omega2(L))) / math.sqrt(2)
OUT["gen_frame_left_over_card_formula"] = gen_coeff(L, "frame", "left") / card_coeff
OUT["gen_coord_left_over_card_formula"] = gen_coeff(L, "coord", "left") / card_coeff
OUT["gen_coord_right_over_card_formula"] = gen_coeff(L, "coord", "right") / card_coeff

print(json.dumps(OUT, indent=2))
pathlib.Path(__file__).with_name("a2_convention.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
