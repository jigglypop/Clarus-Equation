"""Adversary a3: exact re-derivation of the loop kernel law

    E || sum_i [sigma_i, sigma_{i+1}] ||_F^2 = 36 * Theta(kappa),
    Theta(kappa) = sum_{i,j} (k_ij k_{i+1,j+1} - k_{i,j+1} k_{i+1,j})   (cyclic)

by EXACT Wick contraction (no Monte Carlo): the covariance of the symmetric parts is
E[sigma_{u,ab} sigma_{v,cd}] = kappa_uv * P_{ab,cd},  P = (d_ac d_bd + d_ad d_bc)/2,
and every 4th moment is the sum of the three Isserlis pairings.  The whole law is then
checked against a *random* kappa, not only the two special kernels of the card.
Also: the structure constants (A, B, C), Theta closed forms, permutation dependence of
Theta, sibling / cousin faces, the F-02 driver D closed form and every closed number.
"""
from __future__ import annotations
import itertools, json, math, pathlib
import numpy as np

N = 4
d_ = np.eye(N)
P = 0.5 * (np.einsum("ac,bd->abcd", d_, d_) + np.einsum("ad,bc->abcd", d_, d_))
OUT = {}


def quad(kappa, a, b, c, e):
    """sum_{i j k l} E[s_{a,ik} s_{b,kj} s_{c,il} s_{e,lj}] with Isserlis (exact)."""
    # pairing 1: (a,b)(c,e):  P_{(ik),(kj)} P_{(il),(lj)}
    t1 = kappa[a, b] * kappa[c, e] * np.einsum("ikkj,illj->", P, P)
    # pairing 2: (a,c)(b,e):  P_{(ik),(il)} P_{(kj),(lj)}
    t2 = kappa[a, c] * kappa[b, e] * np.einsum("ikil,kjlj->", P, P)
    # pairing 3: (a,e)(b,c):  P_{(ik),(lj)} P_{(kj),(il)}
    t3 = kappa[a, e] * kappa[b, c] * np.einsum("iklj,kjil->", P, P)
    return float(t1 + t2 + t3)


def M(kappa, u, v, p, q):
    """E tr([s_u,s_v] [s_p,s_q]^T) exactly."""
    return quad(kappa, u, v, p, q) - quad(kappa, u, v, q, p) - quad(kappa, v, u, p, q) + quad(kappa, v, u, q, p)


def exact_E_norm2(kappa):
    n = len(kappa)
    tot = 0.0
    for i in range(n):
        for j in range(n):
            tot += M(kappa, i, (i + 1) % n, j, (j + 1) % n)
    return tot


def theta_kernel(kappa):
    n = len(kappa)
    nx = (np.arange(n) + 1) % n
    k = np.asarray(kappa, float)
    return float(np.sum(k * k[np.ix_(nx, nx)] - k[:, nx] * k[nx, :]))


def driver(kappa):
    n = len(kappa)
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ np.asarray(kappa, float) @ H
    return float(np.sum(K * K))


# ---- structure constants A, B, C (exact)
k2 = np.eye(2)
OUT["A_E_comm_norm2_independent"] = M(k2, 0, 1, 0, 1)
OUT["B_swapped"] = M(k2, 0, 1, 1, 0)
OUT["C_self_pairing"] = M(k2, 0, 0, 1, 1)
OUT["exact_E_tr_S2T2"] = float(np.einsum("ikkj,jlli->", P, P))
OUT["exact_E_tr_STST"] = float(np.einsum("abcd,bcda->", P, P))
OUT["card_claim_36"] = 36.0

# ---- the law on random / special kappas (exact, no MC)
rows = {}
rng = np.random.default_rng(20260902)
cases = {
    "face_her_depth0": np.array([[1.0, 1, 1], [1, 2, 2], [1, 2, 3]]),
    "face_her_depth7": np.minimum.outer(np.arange(3), np.arange(3)) + 8.0,
    "face_iid": np.eye(3),
    "face_sibling": np.array([[1.0, 1, 1], [1, 2, 1], [1, 1, 2]]),
    "face_cousin": np.array([[3.0, 2, 2], [2, 3, 2], [2, 2, 3]]),
    "chain_her_5": np.minimum.outer(np.arange(5), np.arange(5)) + 1.0,
    "chain_her_6": np.minimum.outer(np.arange(6), np.arange(6)) + 1.0,
    "iid_6": np.eye(6),
    "two_cells": np.array([[2.0, 0.7], [0.7, 1.3]]),
}
for _ in range(4):
    A_ = rng.standard_normal((5, 7))
    cases[f"random_spd_5_{_}"] = A_ @ A_.T
for name, kap in cases.items():
    ex = exact_E_norm2(kap)
    th = theta_kernel(kap)
    rows[name] = {
        "exact_E_norm2": ex,
        "Theta": th,
        "ratio_exact_over_36Theta": (ex / (36 * th)) if abs(th) > 1e-12 else None,
        "abs_gap": ex - 36 * th,
        "driver_D": driver(kap),
    }
OUT["law_check_exact"] = rows

# ---- Theta closed forms
OUT["Theta_closed_forms"] = {
    "iid_n": {str(n): [theta_kernel(np.eye(n)), float(n)] for n in (3, 4, 8, 16, 64)},
    "chain_her": {
        str(n): [theta_kernel(np.minimum.outer(np.arange(n), np.arange(n)) + 1.0), (n - 1) * (n - 2) / 2]
        for n in (3, 4, 8, 16, 32, 64)
    },
    "iid_n2": theta_kernel(np.eye(2)),
    "shift_invariance_chain8": theta_kernel(np.minimum.outer(np.arange(8), np.arange(8)) + 1.0 + 17.0)
    - theta_kernel(np.minimum.outer(np.arange(8), np.arange(8)) + 1.0),
}

# ---- permutation dependence of Theta (does the cell ORDER on the loop matter?)
perm_rows = {}
for n in (3, 4, 5, 6):
    kap = np.minimum.outer(np.arange(n), np.arange(n)) + 1.0
    vals = set()
    for pm in itertools.permutations(range(n)):
        vals.add(round(theta_kernel(kap[np.ix_(pm, pm)]), 9))
    perm_rows[str(n)] = {"distinct_Theta_over_all_orders": sorted(vals), "card_value": (n - 1) * (n - 2) / 2}
OUT["Theta_permutation_dependence"] = perm_rows

# ---- F-02 driver closed form and every closed number of the card
OUT["driver_closed_form"] = {
    str(n): [driver(np.minimum.outer(np.arange(n), np.arange(n)) + 1.0), (n ** 2 - 1) * (2 * n ** 2 + 7) / 180]
    for n in (3, 4, 16, 32, 64)
}
OUT["driver_iid"] = {str(n): [driver(np.eye(n)), n - 1] for n in (3, 16, 64)}


def c_theta(n, Theta, D):
    return n * math.sqrt(9 * Theta / (20 * D))


OUT["closed_numbers"] = {
    "c_theta_face_her": [c_theta(3, 1.0, 10 / 9), 27 * math.sqrt(2) / 20, 1.9091883092],
    "c_theta_face_iid": [c_theta(3, 3.0, 2.0), math.sqrt(243 / 40), 2.4647515088],
    "rho_face_hol": [math.sqrt(1 / 3), 0.5773502692],
    "rho_face_eps_F02": [math.sqrt(5) / 3, 0.7453559925],
    "c_theta_chain": {
        str(n): [
            c_theta(n, (n - 1) * (n - 2) / 2, (n ** 2 - 1) * (2 * n ** 2 + 7) / 180),
            (9 / math.sqrt(2)) * n * math.sqrt((n - 2) / ((n + 1) * (2 * n * n + 7))),
        ]
        for n in (16, 32, 64)
    },
    "c_theta_chain_limit_9over2": 4.5,
    "c_theta_face_her_over_iid_ratio": c_theta(3, 1.0, 10 / 9) / c_theta(3, 3.0, 2.0),
}

# ---- slopes: OLS over three log-equidistant points == endpoint slope?
def fit_slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


sizes = (16, 32, 64)
th_her = [math.sqrt(4.5 * (n - 1) * (n - 2) / 2) for n in sizes]
th_iid = [math.sqrt(4.5 * n) for n in sizes]
eps_her = [math.sqrt(10 * (n ** 2 - 1) * (2 * n ** 2 + 7) / 180) / n for n in sizes]
eps_iid = [math.sqrt(10 * (n - 1)) / n for n in sizes]
OUT["slopes"] = {
    "theta_slope_her_ols": fit_slope(sizes, th_her),
    "theta_slope_her_endpoint": math.log(math.sqrt(63 * 62) / math.sqrt(15 * 14)) / math.log(4),
    "card_1.0543077": 1.0543077,
    "theta_slope_iid_ols": fit_slope(sizes, th_iid),
    "eps_slope_her_ols": fit_slope(sizes, eps_her),
    "card_eps_her_0.9967340": 0.9967340,
    "eps_slope_iid_ols": fit_slope(sizes, eps_iid),
    "card_eps_iid_-0.4824027": -0.4824027,
    "UNNORMALIZED_eps_slope_iid": fit_slope(sizes, [math.sqrt(10 * (n - 1)) for n in sizes]),
    "UNNORMALIZED_eps_slope_her": fit_slope(sizes, [math.sqrt(10 * (n ** 2 - 1) * (2 * n ** 2 + 7) / 180) for n in sizes]),
}
print(json.dumps(OUT, indent=2))
pathlib.Path(__file__).with_name("a3_isserlis_theta.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
