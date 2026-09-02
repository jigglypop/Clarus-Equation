"""Adversary a2: is the K3 amplitude ratio a real test or an identity?

(a) driver formula D = sum((2*dep+1)s^2) - 2 sum(pre^2)/n + (sum s^2)^2/n^2 checked against an
    explicit matrix  ||H kappa H||_F^2  with kappa_vw = |path(v) cap path(w)| on random Q-spine trees.
(b) independent sample (adversary seed) at several depths recording (eps_t, n_t, D_t):
    regression of eps_t^2 on D_t/n_t^2 -> per-depth eps_star estimate, compared with the
    eps_star estimated from the i.i.d. n=36 control.  If eps_star is common the kernel law
    (ladder step 3) carries the ratio prediction; if not, the K3 amplitude window is doing work
    the law does not.
(c) predicted vs observed ratio inside the SAME independent sample.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
import check_modes as cm
from driver_numbers import qspine_block

DELTA = cm.DELTA
ADV = 987654321


def paths(parent):
    n = len(parent)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    order = [root]
    i = 0
    while i < len(order):
        order.extend(ch[order[i]])
        i += 1
    P = [None] * n
    P[root] = [root]
    for v in order[1:]:
        P[v] = P[parent[v]] + [v]
    return order, P


def D_matrix(parent):
    n = len(parent)
    _, P = paths(parent)
    S = [set(p) for p in P]
    K = np.zeros((n, n))
    for v in range(n):
        for w in range(n):
            K[v, w] = len(S[v] & S[w])
    H = np.eye(n) - np.ones((n, n)) / n
    M = H @ K @ H
    return float(np.sum(M * M))


def D_driver(parent):
    n = len(parent)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    order = [root]
    i = 0
    while i < len(order):
        order.extend(ch[order[i]])
        i += 1
    dep = np.zeros(n, np.int64)
    sub = np.ones(n, np.int64)
    pre = np.zeros(n, np.int64)
    for v in order[1:]:
        dep[v] = dep[parent[v]] + 1
    for v in reversed(order):
        if parent[v] >= 0:
            sub[parent[v]] += sub[v]
    for v in order:
        pre[v] = sub[v] + (pre[parent[v]] if parent[v] >= 0 else 0)
    s = sub.astype(float)
    return float(np.sum((2 * dep + 1) * s * s) - 2 * np.sum(pre.astype(float) ** 2) / n + np.sum(s * s) ** 2 / n ** 2)


out = {}
t0 = time.perf_counter()
print("== (a) driver formula vs explicit ||H kappa H||_F^2 ==", flush=True)
rng = np.random.default_rng(ADV)
worst = 0.0
cases = 0
for b in (2, 3, 4, 5, 6):
    for _ in range(60):
        par = qspine_block(b, rng)
        d1, d2 = D_driver(par), D_matrix(par)
        rel = abs(d1 - d2) / max(1e-12, abs(d2))
        worst = max(worst, rel)
        cases += 1
print(json.dumps({"cases": cases, "max_rel_err": worst}), flush=True)
out["driver_vs_matrix"] = {"cases": cases, "max_rel_err": worst}

print("== (b,c) pathwise kernel law on an independent sample ==", flush=True)
rows = []
for b, T in ((4, 384), (8, 384)):
    rng = np.random.default_rng(ADV + b)
    eps, ns, Ds = [], [], []
    while len(eps) < T:
        par = qspine_block(b, rng)
        n = len(par)
        v = cm.block_residual(cm.heritable_labels(par, rng.normal(size=(n, 4, 4))), DELTA)
        if math.isfinite(v):
            eps.append(v)
            ns.append(n)
            Ds.append(D_driver(par))
    eps = np.array(eps)
    ns = np.array(ns, float)
    Ds = np.array(Ds)
    q = Ds / ns ** 2
    e2 = eps ** 2
    eps_star2 = float(np.mean(e2) / np.mean(q))
    r = float(np.corrcoef(e2, q)[0, 1])
    rows.append({"b": b, "trials": T, "eps_star_from_qspine": math.sqrt(eps_star2),
                 "corr_eps2_vs_D_over_n2": r, "mean_D_over_n2": float(np.mean(q)),
                 "rms_eps": float(np.sqrt(np.mean(e2)))})
    print(json.dumps(rows[-1]), flush=True)

rng_i = np.random.default_rng(ADV + 99)
iid = np.array([cm.sample_iid(36, rng_i, DELTA) for _ in range(384)])
rms_iid = float(np.sqrt(np.mean(iid ** 2)))
eps_star_iid = rms_iid / (math.sqrt(35) / 36)
print(json.dumps({"rms_iid_36": rms_iid, "eps_star_from_iid": eps_star_iid}), flush=True)
b8 = [r for r in rows if r["b"] == 8][0]
out["pathwise"] = {"rows": rows, "rms_iid_36": rms_iid, "eps_star_from_iid": eps_star_iid,
                   "eps_star_ratio_qspine_over_iid": b8["eps_star_from_qspine"] / eps_star_iid,
                   "independent_sample_ratio_b8_over_iid36": b8["rms_eps"] / rms_iid,
                   "predicted_from_same_sample_trees": math.sqrt(b8["mean_D_over_n2"]) * 36 / math.sqrt(35)}
print(json.dumps(out["pathwise"], indent=1, default=float), flush=True)

out["runtime_s"] = time.perf_counter() - t0
(HERE / "a2_law_pathwise.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("DONE", out["runtime_s"], flush=True)
