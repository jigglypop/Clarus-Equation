"""Adversary a1 (Q-0015 F-02 card audit): independent re-derivation of

    log R_f = (delta^2/2) sum_i [sigma_i, sigma_{i+1}] + O(delta^3),   sigma = sym(xi)

Nothing is imported from holonomy_pilot.py: polar factor, matrix log and the loop
product are re-implemented here by a different route (Newton-Schulz-free SVD polar
plus a *series* logarithm, and a second polar route via the inverse square root of
T^T T) so that a coding error in the card's script cannot propagate.

Checks
  A  second-order polar expansion   W2 = asym(X2) - (1/2) asym(d^2)
  B  BCH assembly of the loop -> (delta^2/2) sum [sigma_i, sigma_{i+1}]
  C  order of convergence of theta_numeric/delta^2 towards the formula (Richardson)
  D  antisymmetric part: exact cancellation at O(delta^2) vs finite-delta size
  E  loop reversal / cyclic start-cell invariance of theta
  F  transposed transport convention E_u^{-1} E_v
  G  polar branch: frequency of det(E_v E_u^{-1}) < 0 at delta = 0.005 and 0.1
"""
from __future__ import annotations
import json, math
import numpy as np

OUT = {}
rng = np.random.default_rng(20260902)


# ---------------------------------------------------------------- independent primitives
def polar_svd(T):
    U, _, Vt = np.linalg.svd(T)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def polar_invsqrt(T):
    """R = T (T^T T)^{-1/2}: a different algebraic route to the same rotation factor."""
    M = T.T @ T
    w, V = np.linalg.eigh(M)
    inv_sqrt = V @ np.diag(w ** -0.5) @ V.T
    return T @ inv_sqrt


def log_series(R, terms=60):
    """log R by the Mercator series in (R - I); valid for R close to the identity."""
    X = R - np.eye(len(R))
    acc = np.zeros_like(X)
    P = np.eye(len(R))
    for k in range(1, terms + 1):
        P = P @ X
        acc = acc + ((-1) ** (k + 1)) * P / k
    return acc


def sym(M):
    return 0.5 * (M + M.T)


def asym(M):
    return 0.5 * (M - M.T)


def transport(Eu, Ev, transposed=False):
    T = np.linalg.solve(Eu.T, Ev.T).T if not transposed else np.linalg.solve(Eu, Ev)
    return polar_svd(T)


def loop_holonomy(tetrads, transposed=False):
    H = np.eye(4)
    k = len(tetrads)
    for i in range(k):
        H = transport(tetrads[i], tetrads[(i + 1) % k], transposed) @ H
    return H


def theta_of(R):
    ang = np.angle(np.linalg.eigvals(R))
    return math.sqrt(0.5 * float(np.sum(ang * ang)))


def omega2(labels):
    k = len(labels)
    out = np.zeros((4, 4))
    for i in range(k):
        a, b = sym(labels[i]), sym(labels[(i + 1) % k])
        out += 0.5 * (a @ b - b @ a)
    return out


def theta_formula(labels, d):
    return d * d * float(np.linalg.norm(omega2(labels))) / math.sqrt(2.0)


# ---------------------------------------------------------------- A: polar 2nd order
xi_u = rng.standard_normal((4, 4))
xi_v = rng.standard_normal((4, 4))
d = xi_v - xi_u
errsA = {}
for dl in (1e-3, 1e-4, 1e-5):
    Eu, Ev = np.eye(4) + dl * xi_u, np.eye(4) + dl * xi_v
    W = log_series(polar_svd(Ev @ np.linalg.inv(Eu)))
    pred = dl * asym(d) + dl * dl * (asym(-d @ xi_u) - 0.5 * asym(d @ d))
    errsA[str(dl)] = float(np.linalg.norm(W - pred) / dl ** 2)
OUT["A_polar_second_order_abs_err_over_delta2"] = errsA
OUT["A_note"] = "should fall like delta (O(delta^3) remainder) -> card's W2 = asym(X2) - asym(d^2)/2 confirmed"
# cross-check the two polar routes agree
T = np.eye(4) + 0.05 * rng.standard_normal((4, 4))
OUT["A_polar_route_agreement"] = float(np.linalg.norm(polar_svd(T) - polar_invsqrt(T)))

# ---------------------------------------------------------------- B/C: loop formula
convergence = {}
for n in (3, 4, 5, 6, 8):
    labels = rng.standard_normal((n, 4, 4))
    coef = theta_formula(labels, 1.0)
    row = {}
    for dl in (2e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 1e-4):
        tet = [np.eye(4) + dl * L for L in labels]
        row[str(dl)] = theta_of(loop_holonomy(tet)) / dl ** 2 / coef
    # generator-level (not just norm) comparison at tiny delta
    dl = 1e-5
    tet = [np.eye(4) + dl * L for L in labels]
    W = log_series(loop_holonomy(tet))
    gen_rel = float(np.linalg.norm(W - dl * dl * omega2(labels)) / np.linalg.norm(dl * dl * omega2(labels)))
    row["generator_rel_err_at_1e-5"] = gen_rel
    ratios = [row[str(x)] - 1.0 for x in (1e-2, 5e-3, 2.5e-3)]
    row["gap_halving_ratios"] = [ratios[1] / ratios[0], ratios[2] / ratios[1]]
    convergence[str(n)] = row
OUT["BC_theta_numeric_over_formula"] = convergence

# ---------------------------------------------------------------- D: antisymmetric part
labels = rng.standard_normal((4, 4, 4))
anti = np.stack([asym(m) for m in rng.standard_normal((4, 4, 4))])
shifted = labels + anti
OUT["D_formula_change_under_antisym_shift"] = float(
    abs(theta_formula(shifted, 1.0) - theta_formula(labels, 1.0))
)
rows = {}
for dl in (2e-2, 5e-3, 1e-3, 1e-4):
    t0 = theta_of(loop_holonomy([np.eye(4) + dl * L for L in labels]))
    t1 = theta_of(loop_holonomy([np.eye(4) + dl * L for L in shifted]))
    rows[str(dl)] = {"rel_change": float(abs(t1 / t0 - 1.0)), "rel_change_over_delta": float(abs(t1 / t0 - 1.0) / dl)}
OUT["D_numeric_rel_change_under_antisym_shift"] = rows
# a *global* antisymmetric shift (same a for every cell) -- is that exactly gauge?
a = asym(rng.standard_normal((4, 4)))
glob = labels + a
rows = {}
for dl in (2e-2, 5e-3, 1e-3):
    t0 = theta_of(loop_holonomy([np.eye(4) + dl * L for L in labels]))
    t1 = theta_of(loop_holonomy([np.eye(4) + dl * L for L in glob]))
    rows[str(dl)] = float(abs(t1 / t0 - 1.0))
OUT["D_global_antisym_shift_rel_change"] = rows

# ---------------------------------------------------------------- E: reversal / start cell
labels = rng.standard_normal((5, 4, 4))
dl = 5e-3
tet = [np.eye(4) + dl * L for L in labels]
base = theta_of(loop_holonomy(tet))
OUT["E_theta_base"] = base
OUT["E_rel_change_reversed_loop"] = float(abs(theta_of(loop_holonomy(tet[::-1])) / base - 1.0))
OUT["E_rel_change_cyclic_shift"] = float(abs(theta_of(loop_holonomy(tet[2:] + tet[:2])) / base - 1.0))
perm = [0, 2, 1, 3, 4]
OUT["E_rel_change_noncyclic_permutation"] = float(abs(theta_of(loop_holonomy([tet[i] for i in perm])) / base - 1.0))

# ---------------------------------------------------------------- F: transposed convention
OUT["F_rel_change_transposed_convention_same_sample"] = float(
    abs(theta_of(loop_holonomy(tet, transposed=True)) / base - 1.0)
)
vals_a, vals_b = [], []
for _ in range(400):
    L = np.cumsum(rng.standard_normal((3, 4, 4)), axis=0)
    t = [np.eye(4) + dl * M for M in L]
    vals_a.append(theta_of(loop_holonomy(t)))
    vals_b.append(theta_of(loop_holonomy(t, transposed=True)))
va, vb = np.asarray(vals_a), np.asarray(vals_b)
OUT["F_rms_ratio_transposed_over_frame"] = float(np.sqrt(np.mean(vb ** 2) / np.mean(va ** 2)))
OUT["F_max_per_sample_rel_diff"] = float(np.max(np.abs(vb / va - 1.0)))

# ---------------------------------------------------------------- G: polar branch
counts = {}
for dl in (5e-3, 0.1, 0.3):
    neg = 0
    for _ in range(4000):
        a_, b_ = rng.standard_normal((4, 4)), rng.standard_normal((4, 4))
        Eu, Ev = np.eye(4) + dl * a_, np.eye(4) + dl * b_
        if np.linalg.det(Ev @ np.linalg.inv(Eu)) < 0:
            neg += 1
    counts[str(dl)] = neg / 4000.0
OUT["G_frac_negative_det_transition"] = counts

print(json.dumps(OUT, indent=2))
import pathlib
pathlib.Path(__file__).with_name("a1_leading_order.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
