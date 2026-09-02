"""Continuum (n -> infinity) spine functional for the LOWER end of the sandwich.

For a tree whose branching is all leaves, the O(n^4) parts of A and A_tilde are

    A     ~ n^4 [ sum_i mu_i^2 (1-w_i)^2 w_i^2 + 2 sum_{j<i} mu_i mu_j (1-w_i)^2 w_j^2 ]
    Atil  ~ n^4 [ sum_i mu_i^2 (1-w_i)^2 w_i^2 + 2 sum_{j<i} mu_i mu_j (1-w_i)^2 w_i w_j ]

where the spine carries n*mu_i vertices of weight w_i = 1 - s/n (w increasing with depth),
and B = O(n^2) is negligible.  So  inf_T c  <=  inf_{mu>=0} A/Atil.  Minimise that ratio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a_core import stats_fast  # noqa: E402
import a_fam as F  # noqa: E402


def forms(w):
    w = np.asarray(w, float)
    m = len(w)
    W = (1 - w) ** 2
    P = np.zeros((m, m))
    Q = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                P[i, i] = W[i] * w[i] ** 2
                Q[i, i] = W[i] * w[i] ** 2
            else:
                deep, shal = (i, j) if w[i] > w[j] else (j, i)
                P[i, j] = W[deep] * w[shal] ** 2
                Q[i, j] = W[deep] * w[deep] * w[shal]
    return P, Q


def ratio(mu, P, Q):
    a = float(mu @ P @ mu)
    b = float(mu @ Q @ mu)
    return a / b if b > 0 else np.inf


def minimise(w, restarts=400, iters=800, seed=1):
    P, Q = forms(w)
    rng = np.random.default_rng(seed)
    m = len(w)
    best = (np.inf, None)
    for r in range(restarts):
        mu = rng.random(m) ** (1 + 4 * rng.random())
        if r == 0:
            mu = np.ones(m)
        mu = mu / mu.sum()
        step = 0.5
        cur = ratio(mu, P, Q)
        for _ in range(iters):
            a = float(mu @ P @ mu)
            b = float(mu @ Q @ mu)
            g = 2 * (P @ mu) / b - 2 * a * (Q @ mu) / b ** 2
            nu = mu - step * g * mu.sum() / (np.abs(g).max() + 1e-300)
            nu = np.maximum(nu, 0.0)
            if nu.sum() <= 0:
                break
            nu = nu / nu.sum()
            rn = ratio(nu, P, Q)
            if rn < cur:
                mu, cur = nu, rn
                step *= 1.15
            else:
                step *= 0.6
                if step < 1e-12:
                    break
        if cur < best[0]:
            best = (cur, mu.copy())
    return best


out = {}
grids = {
    "uniform200": np.linspace(1e-3, 1 - 1e-3, 200),
    "log300": np.unique(np.concatenate([np.geomspace(1e-6, 0.5, 200), np.linspace(0.5, 0.999, 100)])),
    "log500_fine": np.unique(np.concatenate([np.geomspace(1e-9, 0.3, 350), np.linspace(0.3, 0.9999, 150)])),
}
for name, w in grids.items():
    val, mu = minimise(w, restarts=120, iters=600, seed=7)
    idx = np.argsort(-mu)[:12]
    out[name] = {"inf_ratio": val, "m": len(w),
                 "top_atoms": [{"w": float(w[i]), "mu": float(mu[i])} for i in idx if mu[i] > 1e-9]}
    print(name, "inf A/Atilde =", round(val, 6))
    print("   atoms:", [(round(float(w[i]), 5), round(float(mu[i]), 4)) for i in idx if mu[i] > 1e-6])

# --- two-atom analytic scan (mass mu1 at a, mu2 at b, a<b): closed form of the ratio
best2 = (np.inf, None)
for a in np.geomspace(1e-6, 0.9, 400):
    for b in np.linspace(max(a * 1.001, 1e-6), 0.9999, 300):
        if b <= a:
            continue
        for z in np.geomspace(1e-3, 1e3, 120):   # z = mu2/mu1
            X = (1 - a) ** 2 * a ** 2 + z ** 2 * (1 - b) ** 2 * b ** 2
            num = X + 2 * z * (1 - b) ** 2 * a ** 2
            den = X + 2 * z * (1 - b) ** 2 * a * b
            r = num / den
            if r < best2[0]:
                best2 = (r, (float(a), float(b), float(z)))
out["two_atom_scan"] = {"inf": best2[0], "a_b_z": best2[1]}
print("two-atom inf:", best2)

(HERE / "a3_profile.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
