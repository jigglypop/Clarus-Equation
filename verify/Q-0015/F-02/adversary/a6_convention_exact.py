"""Adversary a6: EXACT algebra of the transport-convention dependence.

Frame convention   T_i = E_{i+1} E_i^{-1}  ->  log R_f = (d^2/2) sum [sigma_i, sigma_{i+1}]
Coordinate conv.   T_i = E_i^{-1} E_{i+1}  (same left-ordered product) ->
                   log R_f = -(d^2)*( (1/2) sum [sigma_i,sigma_{i+1}] + sum [a_i,a_{i+1}] ),
                   a = asym(xi):  the frame-rotation parts are NOT gauge in this convention.

Consequences: E theta^2 = (9/2) d^4 Theta   vs   (21/2) d^4 Theta,
              every c_theta multiplied by sqrt(7/3) = 1.52753,
              rho^hol (a ratio of two thetas) unchanged.
Everything below is exact Wick contraction plus a machine-precision identity check.
"""
from __future__ import annotations
import json, math, pathlib
import numpy as np

OUT = {}
sym = lambda M: 0.5 * (M + M.T)
asym = lambda M: 0.5 * (M - M.T)
rng = np.random.default_rng(20260903 + 77)


def assembled(xi, coord: bool):
    k = len(xi)
    A, W2 = [], []
    for i in range(k):
        d = xi[(i + 1) % k] - xi[i]
        A.append(asym(d))
        X2 = (-xi[i] @ d) if coord else (-d @ xi[i])
        W2.append(asym(X2) - 0.5 * asym(d @ d))
    G = sum(W2)
    for i in range(k):
        for j in range(i + 1, k):
            G = G + 0.5 * (A[j] @ A[i] - A[i] @ A[j])
    return G


def shape_part(xi):
    k = len(xi)
    return sum(0.5 * (sym(xi[i]) @ sym(xi[(i + 1) % k]) - sym(xi[(i + 1) % k]) @ sym(xi[i])) for i in range(k))


def rot_part(xi):
    k = len(xi)
    return sum(asym(xi[i]) @ asym(xi[(i + 1) % k]) - asym(xi[(i + 1) % k]) @ asym(xi[i]) for i in range(k))


rows = {}
for k in (3, 4, 6):
    xi = rng.standard_normal((k, 4, 4))
    Gf, Gc = assembled(xi, False), assembled(xi, True)
    pred_c = -(shape_part(xi) + rot_part(xi))
    rows[str(k)] = {
        "frame_minus_shape_formula": float(np.linalg.norm(Gf - shape_part(xi))),
        "coord_minus_predicted_formula": float(np.linalg.norm(Gc - pred_c)),
        "norm_ratio_coord_over_frame": float(np.linalg.norm(Gc) / np.linalg.norm(Gf)),
    }
OUT["1_exact_generator_identities"] = rows

# ---- exact Wick: structure constants for the antisymmetric channel
N = 4
d_ = np.eye(N)
Psym = 0.5 * (np.einsum("ac,bd->abcd", d_, d_) + np.einsum("ad,bc->abcd", d_, d_))
Pasym = 0.5 * (np.einsum("ac,bd->abcd", d_, d_) - np.einsum("ad,bc->abcd", d_, d_))


def quad(kappa, P, a, b, c, e):
    t1 = kappa[a, b] * kappa[c, e] * np.einsum("ikkj,illj->", P, P)
    t2 = kappa[a, c] * kappa[b, e] * np.einsum("ikil,kjlj->", P, P)
    t3 = kappa[a, e] * kappa[b, c] * np.einsum("iklj,kjil->", P, P)
    return float(t1 + t2 + t3)


def M(kappa, P, u, v, p, q):
    return quad(kappa, P, u, v, p, q) - quad(kappa, P, u, v, q, p) - quad(kappa, P, v, u, p, q) + quad(kappa, P, v, u, q, p)


def E_norm2(kappa, P):
    n = len(kappa)
    return sum(M(kappa, P, i, (i + 1) % n, j, (j + 1) % n) for i in range(n) for j in range(n))


def Theta(kappa):
    n = len(kappa)
    nx = (np.arange(n) + 1) % n
    k = np.asarray(kappa, float)
    return float(np.sum(k * k[np.ix_(nx, nx)] - k[:, nx] * k[nx, :]))


k2 = np.eye(2)
OUT["2_structure_constants"] = {
    "A_sym_E_comm_norm2": M(k2, Psym, 0, 1, 0, 1),
    "A_asym_E_comm_norm2": M(k2, Pasym, 0, 1, 0, 1),
    "note": "card constant 36 (shape channel); antisymmetric channel constant",
}

cases = {
    "face_her": np.array([[1.0, 1, 1], [1, 2, 2], [1, 2, 3]]),
    "face_iid": np.eye(3),
    "chain_her_6": np.minimum.outer(np.arange(6), np.arange(6)) + 1.0,
    "iid_6": np.eye(6),
}
law = {}
for name, kap in cases.items():
    s = E_norm2(kap, Psym)          # E || sum [sigma,sigma] ||^2
    a = E_norm2(kap, Pasym)         # E || sum [a,a] ||^2
    th = Theta(kap)
    # frame: theta^2 = d^4 ||shape/2||^2 / 2 = d^4 s / 8 ; coord: d^4 (s/4 + a) / 2
    law[name] = {
        "Theta": th,
        "E_theta2_frame_over_delta4": s / 8,
        "E_theta2_coord_over_delta4": (s / 4 + a) / 2,
        "frame_coefficient_over_Theta": (s / 8) / th,
        "coord_coefficient_over_Theta": ((s / 4 + a) / 2) / th,
        "rms_ratio_coord_over_frame": math.sqrt(((s / 4 + a) / 2) / (s / 8)),
    }
OUT["3_two_conventions_exact"] = law
OUT["3_note"] = "frame 9/2 vs coord 21/2 ; ratio sqrt(7/3) = %.6f" % math.sqrt(7 / 3)

OUT["4_card_numbers_under_coord_convention"] = {
    "c_theta_face_her": 27 * math.sqrt(2) / 20 * math.sqrt(7 / 3),
    "card_window_K4": [1.76, 2.06],
    "c_theta_face_iid": math.sqrt(243 / 40) * math.sqrt(7 / 3),
    "card_window_K4_iid": [2.27, 2.66],
    "c_theta_chain_limit": 4.5 * math.sqrt(7 / 3),
    "c_theta_her_16": 4.0560507 * math.sqrt(7 / 3),
    "card_window_K1_16": [3.73, 4.38],
    "rho_face_hol_unchanged": 1 / math.sqrt(3),
    "measured_a2_coord_face_her": 2.9466272911930735,
    "measured_a2_rms_ratio": [1.5252468017916954, 1.5027548556262513, 1.5338622304785838],
}

# ---- rotation-only configuration: pure frame rotations at O(delta^2)
rot_only = np.stack([asym(m) for m in rng.standard_normal((4, 4, 4))])
OUT["5_rotation_only_labels"] = {
    "frame_generator_norm": float(np.linalg.norm(assembled(rot_only, False))),
    "coord_generator_norm": float(np.linalg.norm(assembled(rot_only, True))),
    "card_claim": "asym part is exactly gauge -> frame generator must vanish",
}
print(json.dumps(OUT, indent=2))
pathlib.Path(__file__).with_name("a6_convention_exact.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
