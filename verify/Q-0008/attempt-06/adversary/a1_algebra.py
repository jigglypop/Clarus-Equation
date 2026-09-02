"""a1: independent re-derivation of the Isserlis/Wick contraction (S5.1)-(S5.5),
the vanishing condition (S4.7), the denominator normalisation and the eps_star convention (S6.5).

The four-point function is evaluated from the general Gaussian moment theorem on the full
index set (v,a) and only afterwards compared with the closed form claimed in the derivation.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))

SEED = 20260902
out = {"script": "a1_algebra", "seed": SEED}


def brute_E_norm_phi_sq(K, M):
    n, m = K.shape[0], M.shape[0]
    C = np.einsum("vw,ab->vawb", K, np.eye(m)).reshape(n * m, n * m)
    tot = 0.0
    for v in range(n):
        for w in range(n):
            for a in range(m):
                for b in range(m):
                    i, j = v * m + a, v * m + b
                    for c in range(m):
                        for d in range(m):
                            k, l = w * m + c, w * m + d
                            mom = C[i, j] * C[k, l] + C[i, k] * C[j, l] + C[i, l] * C[j, k]
                            tot += mom * float(np.sum(M[a, b] * M[c, d]))
    return tot


def closed_form(K, M):
    S = sum(M[a, a] for a in range(M.shape[0]))
    T2 = float(np.einsum("abij,abij->", M, M))
    val = float(np.trace(K)) ** 2 * float(np.sum(S * S)) + 2 * T2 * float(np.sum(K * K))
    return val, float(np.sum(S * S)), T2


rng = np.random.default_rng(SEED)
rows = []
for (n, m, traceless) in [(2, 3, True), (3, 4, True), (4, 3, True), (3, 3, False), (2, 4, False)]:
    A = rng.normal(size=(n, n))
    K = A @ A.T
    M = np.zeros((m, m, 3, 3))
    for a in range(m):
        for b in range(a, m):
            B = rng.normal(size=(3, 3))
            B = 0.5 * (B + B.T)
            if traceless:
                B = B - np.trace(B) / 3 * np.eye(3)
            M[a, b] = B
            M[b, a] = B
    brute = brute_E_norm_phi_sq(K, M)
    cf, normS2, T2 = closed_form(K, M)
    rows.append({"n": n, "m": m, "traceless_M": traceless, "brute": brute, "closed": cf,
                 "rel_err": abs(brute - cf) / abs(cf), "normS2": normS2, "T2": T2,
                 "first_term_equals_norm_EPhi_sq": float(np.trace(K)) ** 2 * normS2,
                 "mean_floor_present": bool(normS2 > 1e-12)})
out["isserlis_bruteforce_vs_closed"] = rows
out["isserlis_max_rel_err"] = max(r["rel_err"] for r in rows)

n, m = 3, 4
A = rng.normal(size=(n, n))
K = A @ A.T
M = np.zeros((m, m, 3, 3))
for a in range(m):
    for b in range(a, m):
        B = rng.normal(size=(3, 3))
        B = 0.5 * (B + B.T)
        B = B - np.trace(B) / 3 * np.eye(3)
        M[a, b] = B
        M[b, a] = B
L = np.linalg.cholesky(K + 1e-12 * np.eye(n))
NT = 400000
vals = np.empty(NT)
for s in range(0, NT, 20000):
    z = rng.normal(size=(20000, n, m))
    x = np.einsum("vw,twa->tva", L, z)
    phi = np.einsum("tva,tvb,abij->tij", x, x, M)
    vals[s:s + 20000] = np.einsum("tij,tij->t", phi, phi)
mc, se = float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(NT))
cf, _, _ = closed_form(K, M)
out["mc_cross_check"] = {"trials": NT, "mc": mc, "se": se, "closed": cf, "z": (mc - cf) / se}

from check_cumulant import geometry_constants, linear_map, quadratic_tensor, gram_form, tl, REFERENCE  # noqa: E402

Mtrue = quadratic_tensor(linear_map())
gc = geometry_constants(Mtrue)
S_true = sum(Mtrue[a, a] for a in range(16))
G0 = gram_form(REFERENCE, REFERENCE)
out["structure_constants_recomputed"] = {
    "T2": gc["T2"], "T4": gc["T4"], "max_abs_sum_a_Maa": float(np.abs(S_true).max()),
    "normS2": float(np.sum(S_true * S_true)),
    "normG0_sq": float(np.sum(G0 * G0)), "G0": G0.tolist(),
    "eps_star_sq_over_delta4": 2 * gc["T2"] / float(np.sum(G0 * G0)),
    "M_max_asymmetry_ab": float(np.abs(Mtrue - np.transpose(Mtrue, (1, 0, 2, 3))).max()),
    "M_max_abs_trace": float(max(abs(np.trace(Mtrue[a, b])) for a in range(16) for b in range(16))),
    "M_max_asymmetry_ij": float(np.abs(Mtrue - np.transpose(Mtrue, (0, 1, 3, 2))).max()),
}

eps_shift = 1e-3
Mpert = Mtrue.copy()
D3 = np.diag([1.0, 1.0, -2.0]) / math.sqrt(6.0)
for a in range(16):
    Mpert[a, a] = Mpert[a, a] + eps_shift * D3
Kiid = np.eye(8) - np.ones((8, 8)) / 8
cf_true, s_t, _ = closed_form(Kiid, Mtrue)
cf_pert, s_p, _ = closed_form(Kiid, Mpert)
out["mean_floor_sensitivity"] = {
    "per_direction_shift": eps_shift, "normS2_true": s_t, "normS2_pert": s_p,
    "E_normPhi_sq_true": cf_true, "E_normPhi_sq_pert": cf_pert,
    "rel_change": cf_pert / cf_true - 1.0,
}

d = 0.005
out["eps_star_convention"] = {
    "eps_star_sq_over_delta4_claim": 10.0,
    "from_constants": 2 * gc["T2"] / float(np.sum(G0 * G0)),
    "law_eps_bar_n2_iid": math.sqrt(10.0 * d ** 4 * 1.0 / 4.0),
    "eps_star_over_2": math.sqrt(10.0) * d ** 2 / 2.0,
    "match": bool(abs(math.sqrt(10.0 * d ** 4 / 4.0) - math.sqrt(10.0) * d ** 2 / 2.0) < 1e-18),
}

try:
    import sympy as sp
    nS, mS = 2, 2
    k11, k12, k22 = sp.symbols("k11 k12 k22", real=True)
    Ks = sp.Matrix([[k11, k12], [k12, k22]])
    Ms = {}
    for a in range(mS):
        for b in range(a, mS):
            p, q = sp.symbols("m%d%d_0 m%d%d_1" % (a, b, a, b), real=True)
            mat = sp.Matrix([[p, q, 0], [q, -p, 0], [0, 0, 0]])
            Ms[(a, b)] = mat
            Ms[(b, a)] = mat
    tot = 0
    for v in range(nS):
        for w in range(nS):
            for a in range(mS):
                for b in range(mS):
                    for c in range(mS):
                        for dd in range(mS):
                            mom = (Ks[v, v] * sp.KroneckerDelta(a, b)) * (Ks[w, w] * sp.KroneckerDelta(c, dd)) \
                                + (Ks[v, w] * sp.KroneckerDelta(a, c)) * (Ks[v, w] * sp.KroneckerDelta(b, dd)) \
                                + (Ks[v, w] * sp.KroneckerDelta(a, dd)) * (Ks[v, w] * sp.KroneckerDelta(b, c))
                            tot += mom * sum(Ms[(a, b)][i, j] * Ms[(c, dd)][i, j] for i in range(3) for j in range(3))
    Ssym = sp.zeros(3, 3)
    for a in range(mS):
        Ssym = Ssym + Ms[(a, a)]
    T2s = sum(sum(Ms[(a, b)][i, j] ** 2 for i in range(3) for j in range(3)) for a in range(mS) for b in range(mS))
    claim = sp.trace(Ks) ** 2 * sum(Ssym[i, j] ** 2 for i in range(3) for j in range(3)) \
        + 2 * T2s * sum(Ks[i, j] ** 2 for i in range(2) for j in range(2))
    diff = sp.simplify(sp.expand(tot - claim))
    out["sympy_identity_zero"] = str(diff)
    out["sympy_ok"] = bool(diff == 0)
except Exception as exc:
    out["sympy_ok"] = "unavailable: %s" % exc

(HERE / "a1_algebra.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "isserlis_bruteforce_vs_closed"}, ensure_ascii=False, indent=1, default=float))
print("brute-vs-closed max rel err:", out["isserlis_max_rel_err"])
