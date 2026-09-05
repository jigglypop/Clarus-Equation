"""Adversary a5: limits the card did not test.

(a) non-orthonormal pair family Xi = 4 n m^T with n.m = cos(theta): c2, c4, c_Delta, det.
(b) delta-truncation: paired chain n=16 physical MC at three deltas (common random numbers).
(c) Isserlis: gaussian Q4 = 2 T2 = 120 by surrogate MC with the adversary M tensor.
(d) surrogate check of the fourth-moment law with the adversary M tensor.
(e) sign symmetry Xi -> -Xi.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

M = np.load(HERE / "M_tensor_adversary.npy")
I4 = np.eye(4)
REF = geometric_self_dual_triple(I4)
out = {"script": "a5_limits"}
rng = np.random.default_rng(20260903)


def aligned(t):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate


def M_of(X, Y):
    return np.einsum("a,b,abij->ij", np.asarray(X, float).reshape(16),
                     np.asarray(Y, float).reshape(16), M)


ang = {}
for deg in (90, 75, 60, 45, 30, 15, 0):
    th = math.radians(deg)
    c = math.cos(th)
    beta = (c * c - 0.25) / 18.0
    alpha = (0.25 - 2 * beta) / 4.0
    lam9 = 16 * (alpha + beta)
    lam6 = 16 * (alpha - beta)
    lam1 = 16 * (alpha + 5 * beta)
    n = np.array([1.0, 0, 0, 0])
    m = c * n + math.sin(th) * np.array([0.0, 1, 0, 0])
    Xi = 4 * np.outer(n, m)
    Q4 = float(np.sum(M_of(Xi, Xi) ** 2))
    c2 = 10 * lam9 ** 2
    cd = Q4 / 12
    ang[str(deg)] = {"cos_theta": c, "lambda_P9": lam9, "lambda_P6": lam6, "lambda_P1": lam1,
                     "trace_C_full": lam1 + 9 * lam9 + 6 * lam6, "Q4": Q4, "c2": c2,
                     "c_delta": cd, "c4": cd - c2,
                     "det_I_plus_dXi_at_d0.005": float(np.linalg.det(I4 + 0.005 * Xi)),
                     "det_exact_1_plus_4d_cos": 1 + 4 * 0.005 * c}
ang["note"] = ("only theta=90 gives det=1 exactly; c_delta falls from 32/9 to 0 over the family, "
               "so the card axiom picks an isolated point of the family, not a generic one")
out["a_angle_family"] = ang


def haar_pairs(r, count):
    g = r.normal(size=(count, 4, 2))
    n = g[:, :, 0] / np.linalg.norm(g[:, :, 0], axis=1, keepdims=True)
    m = g[:, :, 1] - np.sum(g[:, :, 1] * n, axis=1, keepdims=True) * n
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return n, m


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


A16 = np.tril(np.ones((16, 16)))
B16 = A16.T @ centering(16) @ A16
D16 = float(np.sum(B16 * B16))
S16 = float(np.sum(np.diag(B16) ** 2))
pred16 = 640 / 81 - 352 / 81 * S16 / D16
trunc = {}
T = 1200
for dl in (0.02, 0.005, 0.00125):
    r = np.random.default_rng(777001)
    vals = np.empty(T)
    mind = 1.0
    for t in range(T):
        nn, mm = haar_pairs(r, 16)
        Z = 4.0 * np.einsum("ti,tk->tik", nn, mm)
        xis = np.einsum("vu,uik->vik", A16, Z)
        for v in range(16):
            mind = min(mind, float(np.linalg.det(I4 + dl * xis[v])))
        Y = sum(aligned(I4 + dl * xis[v]) for v in range(16))
        g = plebanski_gram(Y)
        tlg = g - np.trace(g) / 3 * np.eye(3)
        vals[t] = (np.linalg.norm(tlg) / np.linalg.norm(g)) ** 2
    c = float(vals.mean()) * 256 / (dl ** 4 * D16)
    trunc[str(dl)] = {"c_obs": c, "rel_dev_vs_pred": c / pred16 - 1,
                      "se_rel": float(vals.std(ddof=1) / vals.mean() / math.sqrt(T)),
                      "min_det_tetrad": mind}
trunc["pred"] = pred16
trunc["note"] = "common random numbers across deltas (same seed), so deviations are paired"
out["b_delta_truncation_chain16"] = trunc
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "a5_limits_part1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
