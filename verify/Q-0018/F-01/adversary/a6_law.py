"""Adversary a6: Isserlis reduction, surrogate fourth-moment law, sign symmetry."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

M = np.load(HERE / "M_tensor_adversary.npy")
I4 = np.eye(4)
REF = geometric_self_dual_triple(I4)
out = {"script": "a6_law"}
rng = np.random.default_rng(9090909)


def aligned(t):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate


def M_of(X, Y):
    return np.einsum("a,b,abij->ij", np.asarray(X, float).reshape(16),
                     np.asarray(Y, float).reshape(16), M)


T2 = float(np.einsum("abij,abij->", M, M))
T4 = float(np.einsum("aaij,aaij->", M, M))
NG = 200000
z = rng.normal(size=(NG, 16))
phi = np.einsum("ta,tb,abij->tij", z, z, M)
q4g = float(np.mean(np.sum(phi * phi, axis=(1, 2))))
out["c_isserlis"] = {"Q4_gaussian_mc": q4g, "2*T2": 2 * T2, "rel_dev": q4g / (2 * T2) - 1,
                     "T4": T4, "c4_from_kurtosis_T4_over_12": T4 / 12,
                     "q0012_eps_star_times_c4_10_over_60": 10 / 60,
                     "sum_a_M_aa_maxabs": float(np.max(np.abs(np.einsum("aaij->ij", M))))}


def haar_pairs(r, count):
    g = r.normal(size=(count, 4, 2))
    n = g[:, :, 0] / np.linalg.norm(g[:, :, 0], axis=1, keepdims=True)
    m = g[:, :, 1] - np.sum(g[:, :, 1] * n, axis=1, keepdims=True) * n
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return n, m


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


C1 = np.zeros((16, 16))
for i in range(4):
    for k in range(4):
        for j in range(4):
            for l in range(4):
                C1[4 * i + k, 4 * j + l] = ((5 / 72) * (i == j) * (k == l)
                                            - (1 / 72) * ((i == k) * (j == l) + (i == l) * (j == k)))
T2C1 = float(np.einsum("ac,bd,abij,cdij->", C1, C1, M, M))
E01 = np.zeros((4, 4))
E01[0, 1] = 1.0
Q41 = float(np.sum(M_of(E01, E01) ** 2))
sur = {}
NS = 60000
for name, A in (("chain_n16", np.tril(np.ones((16, 16)))),
                ("coh_n16", np.vstack([np.hstack([np.ones((8, 1)), np.zeros((8, 1))]),
                                       np.hstack([np.zeros((8, 1)), np.ones((8, 1))])])),
                ("iid_n16", np.eye(16))):
    n = A.shape[0]
    mi = A.shape[1]
    B = A.T @ centering(n) @ A
    D = float(np.sum(B * B))
    S = float(np.sum(np.diag(B) ** 2))
    law = 256.0 * (2 * T2C1 * D + (Q41 - 2 * T2C1) * S)
    HA = centering(n) @ A
    vals = np.empty(NS)
    bs = 4000
    for s in range(0, NS, bs):
        nn, mm = haar_pairs(rng, bs * mi)
        Z = (4.0 * np.einsum("ti,tk->tik", nn, mm)).reshape(bs, mi, 16)
        xt = np.einsum("vu,tua->tva", HA, Z)
        ph = np.einsum("tva,tvb,abij->tij", xt, xt, M)
        vals[s:s + bs] = np.einsum("tij,tij->t", ph, ph)
    sur[name] = {"D": D, "S_gen": S, "law": law, "mc": float(vals.mean()),
                 "rel_err": float(vals.mean()) / law - 1,
                 "se_rel": float(vals.std(ddof=1) / vals.mean() / math.sqrt(NS)),
                 "c_from_law": law / (12 * D)}
out["d_surrogate_law_adversary_M"] = sur

sign = {}
for dl in (0.02, 0.005):
    r = np.random.default_rng(313131)
    nn, mm = haar_pairs(r, 4)
    Z = 4.0 * np.einsum("ti,tk->tik", nn, mm)
    A = np.tril(np.ones((4, 4)))
    xis = np.einsum("vu,uik->vik", A, Z)

    def eps(sgn):
        Y = sum(aligned(I4 + sgn * dl * xis[v]) for v in range(4))
        g = plebanski_gram(Y)
        return float(np.linalg.norm(g - np.trace(g) / 3 * np.eye(3)) / np.linalg.norm(g))

    ep, em = eps(1.0), eps(-1.0)
    sign[str(dl)] = {"eps_plus": ep, "eps_minus": em, "rel_gap": ep / em - 1,
                     "mean_over_d2": (ep + em) / 2 / dl ** 2}
sign["note"] = "eps(+Xi) != eps(-Xi) at O(delta), but the Haar pair law is invariant under n -> -n, so the O(delta) term cancels in the mean"
out["e_sign_symmetry"] = sign
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "a6_law.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
