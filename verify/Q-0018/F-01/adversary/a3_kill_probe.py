"""Adversary a3: higher-statistics physical MC probe of the Q-0018 F-01 kill statistics.

NOT the pre-registered run: adversary seed 424243/424244, independent code path, and the
surrogate law is re-evaluated with the adversary M tensor (a1) rather than the card sympy one.
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
DELTA = 0.005
out = {"script": "a3_kill_probe", "delta": DELTA, "seed_base": 424243}
C2, C4 = 640 / 81, -352 / 81
WIN = {"iid_n16": (3.6358, 4.0185), "iid_n8": (3.8938, 4.3037),
       "chain_n16": (6.6180, 7.5705), "coh_n16": (5.4420, 6.0148),
       "chain_n2": (3.3778, 3.7333), "iid_n2": (5.4420, 6.0148), "iid_n4": (4.4099, 4.8741),
       "chain_n4": (4.8068, 5.3128), "chain_n8": (5.9530, 6.7109)}


def aligned(t):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


def generator(name, n):
    if name == "iid":
        return np.eye(n)
    if name == "chain":
        return np.tril(np.ones((n, n)))
    if name == "coh":
        A = np.zeros((n, 2))
        A[:n // 2, 0] = 1.0
        A[n // 2:, 1] = 1.0
        return A
    raise ValueError(name)


def haar_pairs(rng, count):
    g = rng.normal(size=(count, 4, 2))
    n = g[:, :, 0] / np.linalg.norm(g[:, :, 0], axis=1, keepdims=True)
    m = g[:, :, 1] - np.sum(g[:, :, 1] * n, axis=1, keepdims=True) * n
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return n, m


def D_S(A):
    B = A.T @ centering(A.shape[0]) @ A
    return float(np.sum(B * B)), float(np.sum(np.diag(B) ** 2))


def run(mode, n, trials, seed, delta=DELTA):
    rng = np.random.default_rng(seed)
    A = generator(mode, n)
    mi = A.shape[1]
    D, S = D_S(A)
    vals = np.empty(trials)
    tls = np.empty((trials, 3, 3))
    maxlab = 0.0
    for t in range(trials):
        nn, mm = haar_pairs(rng, mi)
        Z = 4.0 * np.einsum("ti,tk->tik", nn, mm)
        xis = np.einsum("vu,uik->vik", A, Z)
        maxlab = max(maxlab, float(np.max(np.linalg.norm(xis, axis=(1, 2)))))
        Y = sum(aligned(I4 + delta * xis[v]) for v in range(n))
        g = plebanski_gram(Y)
        tlg = g - np.trace(g) / 3 * np.eye(3)
        tls[t] = tlg
        vals[t] = (np.linalg.norm(tlg) / np.linalg.norm(g)) ** 2
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(trials))
    scale = n * n / (delta ** 4 * D)
    pred = C2 + C4 * S / D
    key = mode + "_n" + str(n)
    lo, hi = WIN.get(key, (float("nan"), float("nan")))
    c = mean * scale
    return {"mode": mode, "n": n, "trials": trials, "delta": delta, "seed": seed,
            "D": D, "S_gen": S, "c_obs": c, "c_se": se * scale,
            "c_pred_card": pred, "rel_dev": c / pred - 1, "z_vs_pred": (c - pred) / (se * scale),
            "card_window": [lo, hi], "inside_card_window": bool(lo <= c <= hi),
            "c_gauss_F02": 10.0, "c_alt_C_is_I16": 10.0 + (32 / 9 - 10) * S / D,
            "c_alt_kernel_diag_note": "see a4", "cv": float(vals.std(ddof=1) / mean),
            "z_floor": float(trials * np.sum(tls.mean(axis=0) ** 2)
                             / np.mean(np.sum(tls ** 2, axis=(1, 2)))),
            "max_label_frobenius": maxlab, "delta_times_max_label": delta * maxlab}


t0 = time.time()
res = {}
plan = [("chain", 16, 6000, 424243), ("iid", 16, 3000, 424244), ("iid", 8, 3000, 424245),
        ("coh", 16, 3000, 424246), ("chain", 8, 3000, 424247), ("chain", 4, 3000, 424248),
        ("chain", 2, 3000, 424249), ("iid", 4, 3000, 424250), ("iid", 2, 3000, 424251)]
for mode, n, tr, sd in plan:
    res[mode + "_n" + str(n)] = run(mode, n, tr, sd)
    print(mode, n, "done", round(time.time() - t0, 1), "s", flush=True)
out["mc"] = res
scal = {}
for dl in (0.02, 0.005, 0.00125):
    r = run("chain", 2, 1500, 424252, delta=dl)
    scal[str(dl)] = {"cv": r["cv"], "cv_over_delta": r["cv"] / dl, "c_obs": r["c_obs"],
                     "rel_dev": r["rel_dev"]}
out["chain_n2_cv_delta_scaling"] = scal
out["wall_seconds"] = time.time() - t0
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "a3_kill_probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
