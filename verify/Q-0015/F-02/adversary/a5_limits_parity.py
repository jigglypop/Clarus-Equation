"""Adversary a5: exact assembly of the O(delta^2) generator, the recovers limits executed,
parity (self-dual / anti-self-dual), the corrected Isserlis side identities, and the
pre-registration integrity table (windows vs theory vs pilot).
"""
from __future__ import annotations
import json, math, pathlib
import numpy as np

OUT = {}
rng = np.random.default_rng(20260903 + 11)
sym = lambda M: 0.5 * (M + M.T)
asym = lambda M: 0.5 * (M - M.T)


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


def hol(E):
    H = np.eye(4)
    k = len(E)
    for i in range(k):
        H = polar(E[(i + 1) % k] @ np.linalg.inv(E[i])) @ H
    return H


# ---------------------------------------------------------------- 1. exact O(delta^2) assembly
def assembled_generator(xi):
    k = len(xi)
    A, W2 = [], []
    for i in range(k):
        d = xi[(i + 1) % k] - xi[i]
        A.append(asym(d))
        W2.append(asym(-d @ xi[i]) - 0.5 * asym(d @ d))
    G = sum(W2)
    for i in range(k):
        for j in range(i + 1, k):
            G = G + 0.5 * (A[j] @ A[i] - A[i] @ A[j])
    return G


def card_generator(xi):
    k = len(xi)
    G = np.zeros((4, 4))
    for i in range(k):
        a, b = sym(xi[i]), sym(xi[(i + 1) % k])
        G = G + 0.5 * (a @ b - b @ a)
    return G


rows = {}
for k in (3, 4, 5, 7):
    xi = rng.standard_normal((k, 4, 4))
    G1, G2 = assembled_generator(xi), card_generator(xi)
    rows[str(k)] = {
        "abs_gap": float(np.linalg.norm(G1 - G2)),
        "rel_gap": float(np.linalg.norm(G1 - G2) / np.linalg.norm(G2)),
        "generator_is_antisymmetric": float(np.linalg.norm(G2 + G2.T)),
    }
    # BCH sub-identity: (1/2) sum_{j>i} [A_j, A_i] == -(1/2) sum_i [a_i, a_{i+1}]
    a_ = [asym(m) for m in xi]
    lhs = np.zeros((4, 4))
    A = [asym(xi[(i + 1) % k] - xi[i]) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            lhs = lhs + 0.5 * (A[j] @ A[i] - A[i] @ A[j])
    rhs = -0.5 * sum(a_[i] @ a_[(i + 1) % k] - a_[(i + 1) % k] @ a_[i] for i in range(k))
    rows[str(k)]["bch_collapse_abs_gap"] = float(np.linalg.norm(lhs - rhs))
OUT["1_exact_generator_assembly"] = rows

# ---------------------------------------------------------------- 2. recovers limits
lim = {}
# pure gauge: E_v = s_v * Lambda_v
def rand_so4(r):
    q, rr = np.linalg.qr(r.standard_normal((4, 4)))
    q = q @ np.diag(np.sign(np.diag(rr)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


E = [rng.uniform(0.3, 3.0) * rand_so4(rng) for _ in range(7)]
lim["pure_gauge_theta"] = theta(hol(E))
# identical labels (kappa = c J)
xi = rng.standard_normal((4, 4))
lim["identical_labels_theta"] = theta(hol([np.eye(4) + 0.005 * xi] * 6))
# n = 2 retraced loop
lim["two_cell_loop_theta"] = theta(hol([np.eye(4) + 0.005 * rng.standard_normal((4, 4)) for _ in range(2)]))
# single transport of an exact plane rotation
al = 0.37
rot = np.eye(4)
rot[0, 0] = rot[1, 1] = math.cos(al)
rot[0, 1], rot[1, 0] = -math.sin(al), math.sin(al)
Eu = np.eye(4) + 0.3 * rng.standard_normal((4, 4))
lim["single_transport_angle_minus_alpha"] = theta(polar(rot @ Eu @ np.linalg.inv(Eu))) - al
# exact conformal + rotation per cell (Psi = 0 style): E_v = s_v Lambda_v applied to a common tetrad
common = np.eye(4) + 0.2 * rng.standard_normal((4, 4))
E = [rng.uniform(0.5, 2.0) * rand_so4(rng) @ common for _ in range(5)]
lim["gauge_times_common_tetrad_theta"] = theta(hol(E))
# delta -> 0 scaling on one face
lab = np.cumsum(rng.standard_normal((3, 4, 4)), axis=0)
t1 = theta(hol([np.eye(4) + 0.005 * L for L in lab]))
t2 = theta(hol([np.eye(4) + 0.0025 * L for L in lab]))
lim["delta_ratio_single_sample"] = t2 / t1
OUT["2_recovers_limits"] = lim

# ---------------------------------------------------------------- 3. parity: self-dual split (exact Wick)
N = 4
d_ = np.eye(N)
P = 0.5 * (np.einsum("ac,bd->abcd", d_, d_) + np.einsum("ad,bc->abcd", d_, d_))
EPS = np.zeros((4, 4, 4, 4))
for perm, sgn in (
    ((0, 1, 2, 3), 1), ((0, 2, 3, 1), 1), ((0, 3, 1, 2), 1), ((1, 0, 3, 2), 1), ((1, 2, 0, 3), 1),
    ((1, 3, 2, 0), 1), ((2, 0, 1, 3), 1), ((2, 1, 3, 0), 1), ((2, 3, 0, 1), 1), ((3, 0, 2, 1), 1),
    ((3, 1, 0, 2), 1), ((3, 2, 1, 0), 1), ((0, 1, 3, 2), -1), ((0, 2, 1, 3), -1), ((0, 3, 2, 1), -1),
    ((1, 0, 2, 3), -1), ((1, 2, 3, 0), -1), ((1, 3, 0, 2), -1), ((2, 0, 3, 1), -1), ((2, 1, 0, 3), -1),
    ((2, 3, 1, 0), -1), ((3, 0, 1, 2), -1), ((3, 1, 2, 0), -1), ((3, 2, 0, 1), -1),
):
    EPS[perm] = sgn
mc = np.random.default_rng(31337)
plus, minus, tot = [], [], []
for _ in range(200000):
    z = mc.standard_normal((2, 4, 4))
    s = 0.5 * (z + np.swapaxes(z, -1, -2))
    om = s[0] @ s[1] - s[1] @ s[0]
    du = 0.5 * np.einsum("abcd,cd->ab", EPS, om)
    plus.append(np.sum((0.5 * (om + du)) ** 2))
    minus.append(np.sum((0.5 * (om - du)) ** 2))
    tot.append(np.sum(om * om))
OUT["3_parity"] = {
    "E_norm2_selfdual": float(np.mean(plus)),
    "E_norm2_antiselfdual": float(np.mean(minus)),
    "E_norm2_total": float(np.mean(tot)),
    "ratio_plus_over_minus": float(np.mean(plus) / np.mean(minus)),
    "half_of_total": float(np.mean(tot) / 2),
    "trials": 200000,
}

# ---------------------------------------------------------------- 4. corrected Isserlis side identities
OUT["4_isserlis_side"] = {
    "E_tr_S2T2": float(np.einsum("abbc,cdda->", P, P)),
    "E_tr_STST": float(np.einsum("abcd,bcda->", P, P)),
    "card_verify3": "2*(5/2)^2*4 - 2*7 = 36",
    "assembled": float(2 * np.einsum("abbc,cdda->", P, P) - 2 * np.einsum("abcd,bcda->", P, P)),
}

# ---------------------------------------------------------------- 5. preregistration integrity
pre = {
    "c_theta_her_16": (4.0560507, (3.73, 4.38), None),
    "c_theta_her_32": (4.2832679, (3.94, 4.63), None),
    "c_theta_her_64": (4.3930506, (4.04, 4.74), None),
    "theta_slope_her": (1.0543077, (0.95, 1.15), None),
    "theta_slope_iid": (0.5000000, (0.40, 0.60), None),
    "rho_face_hol": (0.5773503, (0.540, 0.615), 0.5646202445828142),
    "c_theta_face_her": (1.9091883, (1.76, 2.06), 1.9037058512651062),
    "c_theta_face_iid": (2.4647515, (2.27, 2.66), 2.471693541518498),
    "delta_ratio_face_her": (0.2500000, (0.235, 0.265), 0.24851983605588582),
}
tbl = {}
for k, (val, (lo, hi), pilot) in pre.items():
    tbl[k] = {
        "theory": val,
        "window": [lo, hi],
        "window_centre": (lo + hi) / 2,
        "centre_minus_theory": (lo + hi) / 2 - val,
        "half_width_rel": (hi - lo) / 2 / abs(val),
        "pilot": pilot,
        "pilot_inside": (lo <= pilot <= hi) if pilot is not None else None,
        "pilot_dev_rel": ((pilot - val) / val) if pilot is not None else None,
    }
OUT["5_preregistration_table"] = tbl

print(json.dumps(OUT, indent=2))
pathlib.Path(__file__).with_name("a5_limits_parity.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
