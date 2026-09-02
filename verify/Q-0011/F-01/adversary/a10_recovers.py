"""Audits 3 (recovers) and 4/6 (content): exponent table, weight specificity, prereg numbers."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "verify" / "Q-0008" / "F-02"))
from a_core import tree_arrays  # noqa: E402
import a_fam as F  # noqa: E402
from driver_numbers import cayley_exact, uniform_rooted_tree  # noqa: E402


def mu2_weighted(parent, power):
    """n^2 * E[(sum_{u<=v,w} g(w_u))^2] with g(w) = w**power  (power=1 is the card's kappa_eff;
    power=0 is the raw kappa, i.e. no centering surrogate at all)."""
    n = len(parent)
    order, _, sub = tree_arrays(parent)
    s = sub.astype(np.float64)
    w = (1.0 - s / n) ** power
    P = np.zeros(n)
    for v in order:
        p = parent[v]
        if p >= 0:
            P[v] = P[p] + w[p]
    diag = float(np.sum(s * s * w * w))
    return diag + 2.0 * float(np.sum(s * s * w * P))


def driver(parent):
    n = len(parent)
    order, depth, sub = tree_arrays(parent)
    s = sub.astype(np.float64)
    w = 1.0 - s / n
    P = np.zeros(n)
    Q = np.zeros(n)
    A2 = np.zeros(n)
    for v in order:
        p = parent[v]
        if p >= 0:
            P[v] = P[p] + w[p]
            Q[v] = Q[p] + w[p] ** 2
            A2[v] = A2[p] + s[p] ** 2
    diag = float(np.sum(s * s * w * w))
    A = diag + 2.0 * float(np.sum(s * s * Q))
    tot2 = float(np.sum(s * s))
    B = (tot2 * tot2 - float(np.sum(s ** 4)) - 2.0 * float(np.sum(s * s * A2))) / (n * n)
    return A + B, float(depth.max())


def slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


rng = np.random.default_rng(20260902)
fams = {
    "chain": (lambda n: F.chain(n), (128, 256, 512, 1024)),
    "star": (lambda n: F.star(n), (128, 256, 512, 1024)),
    "caterpillar": (lambda k: F.comb(k * k, 0) if False else _cat(k), (8, 12, 16, 24, 32)),
    "star_of_chains": (lambda k: _soc(k), (6, 8, 10, 12, 14)),
    "two_level_star": (lambda k: _tls(k), (6, 8, 10, 12, 14)),
    "broom": (lambda n: F.broom(n, 0.5), (128, 256, 512, 1024, 2048)),
    "complete_binary": (lambda h: F.kary(2, h), (7, 8, 9, 10, 11)),
    "kary4": (lambda h: F.kary(4, h), (4, 5, 6, 7)),
}


def _cat(k):
    parent = [-1]
    spine = [0]
    prev = 0
    for _ in range(k - 1):
        parent.append(prev)
        prev = len(parent) - 1
        spine.append(prev)
    for sp in spine:
        for _ in range(k - 1):
            parent.append(sp)
    return parent


def _soc(k):
    parent = [-1]
    for _ in range(k):
        prev = 0
        for _ in range(k):
            parent.append(prev)
            prev = len(parent) - 1
    return parent


def _tls(k):
    parent = [-1]
    br = []
    for _ in range(k):
        parent.append(0)
        br.append(len(parent) - 1)
    for b in br:
        for _ in range(k):
            parent.append(b)
    return parent


table = {}
for name, (fn, args) in fams.items():
    ns, dn2, m1, m2, m0, mh, dep = [], [], [], [], [], [], []
    for a in args:
        p = fn(a)
        n = len(p)
        d, mx = driver(p)
        ns.append(n)
        dn2.append(d / n ** 2)
        m1.append(mu2_weighted(p, 1.0) / n ** 2)
        m2.append(mu2_weighted(p, 2.0) / n ** 2)
        m0.append(mu2_weighted(p, 0.0) / n ** 2)
        mh.append(mu2_weighted(p, 0.5) / n ** 2)
        dep.append(max(mx, 1.0))
    table[name] = {
        "n": ns,
        "gamma_from_D": slope(ns, [math.sqrt(x) for x in dn2]),
        "gamma_from_mu2_w1_card": 0.5 * slope(ns, m1),
        "gamma_from_mu2_w2": 0.5 * slope(ns, m2),
        "gamma_from_mu2_w0_raw_kappa": 0.5 * slope(ns, m0),
        "gamma_from_mu2_wsqrt": 0.5 * slope(ns, mh),
        "gamma_depth_law": slope(ns, dep),
        "c_range": [min(x / y for x, y in zip(dn2, m1)), max(x / y for x, y in zip(dn2, m1))],
    }

# uniform rooted Cayley (exact E[D]) and its mu2_eff (MC)
cay_ns = (128, 256, 512, 1024)
ed = [cayley_exact(n)["E_D"] for n in cay_ns]
mu_c, mu_c2, mu_c0, dep_c = [], [], [], []
for n in cay_ns:
    vals, vals2, vals0, dd = [], [], [], []
    for _ in range(300):
        p = uniform_rooted_tree(n, rng)
        vals.append(mu2_weighted(p, 1.0) / n ** 2)
        vals2.append(mu2_weighted(p, 2.0) / n ** 2)
        vals0.append(mu2_weighted(p, 0.0) / n ** 2)
        dd.append(max(tree_arrays(p)[1].max(), 1))
    mu_c.append(float(np.mean(vals)))
    mu_c2.append(float(np.mean(vals2)))
    mu_c0.append(float(np.mean(vals0)))
    dep_c.append(float(np.mean(dd)))
table["cayley"] = {
    "n": list(cay_ns),
    "gamma_from_D": slope(cay_ns, [math.sqrt(x) / n for x, n in zip(ed, cay_ns)]),
    "gamma_from_mu2_w1_card": 0.5 * slope(cay_ns, mu_c),
    "gamma_from_mu2_w2": 0.5 * slope(cay_ns, mu_c2),
    "gamma_from_mu2_w0_raw_kappa": 0.5 * slope(cay_ns, mu_c0),
    "gamma_depth_law": slope(cay_ns, dep_c),
    "c_range": [min(e / (n ** 2 * m) for e, n, m in zip(ed, cay_ns, mu_c)),
                max(e / (n ** 2 * m) for e, n, m in zip(ed, cay_ns, mu_c))],
}

# --- independent recomputation of the card's six pre-registered numbers
BROOM = (8, 16, 32, 64, 128)
SOC_K = tuple(range(4, 13))
TLS_K = tuple(range(4, 13))
d_broom = [driver(F.broom(n, 0.5))[0] for n in BROOM]
d_soc = [driver(_soc(k))[0] for k in SOC_K]
d_tls = [driver(_tls(k))[0] for k in TLS_K]
n_soc = [k * k + 1 for k in SOC_K]
n_tls = [k * k + k + 1 for k in TLS_K]
prereg = {
    "broom_slope": slope(BROOM, [math.sqrt(d) / n for d, n in zip(d_broom, BROOM)]),
    "soc_slope": slope(n_soc, [math.sqrt(d) / n for d, n in zip(d_soc, n_soc)]),
    "tls_slope": slope(n_tls, [math.sqrt(d) / n for d, n in zip(d_tls, n_tls)]),
    "soc_over_broom_65": math.sqrt(driver(_soc(8))[0] / driver(F.broom(65, 0.5))[0]),
    "tls_over_iid_73": math.sqrt(driver(_tls(8))[0] / 72.0),
    "D_broom_grid": d_broom,
    "D_soc_65": driver(_soc(8))[0],
    "D_tls_73": driver(_tls(8))[0],
}
card_nums = {"broom_slope": -0.0349, "soc_slope": 0.2466, "tls_slope": -0.2594,
             "soc_over_broom_65": 2.3408, "tls_over_iid_73": 3.2103}
prereg["max_abs_diff_vs_card"] = max(abs(prereg[k] - v) for k, v in card_nums.items())
out = {"exponent_table": table, "prereg_recompute": prereg, "card_numbers": card_nums}
print(json.dumps({"prereg_recompute": {k: v for k, v in prereg.items() if not isinstance(v, list)},
                  "card_numbers": card_nums}, indent=2))
for k, v in table.items():
    print(f"{k:16s} gD={v['gamma_from_D']:+.4f} w1={v['gamma_from_mu2_w1_card']:+.4f} "
          f"w2={v['gamma_from_mu2_w2']:+.4f} w0(raw)={v['gamma_from_mu2_w0_raw_kappa']:+.4f} "
          f"depth={v['gamma_depth_law']:+.4f} c=[{v['c_range'][0]:.3f},{v['c_range'][1]:.3f}]")
(HERE / "a10_recovers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
