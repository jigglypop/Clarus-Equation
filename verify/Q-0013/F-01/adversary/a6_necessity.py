"""Adversary a6: is axis balance w1=w2=w3 even NECESSARY for a zero floor?

Construct Sigma = u u^T + v v^T with u = e_01 (axis 1, w unbalanced) and v = alpha e_00 + beta e_11
(both e_00 and e_11 are card zero modes, contributing nothing to w).  tl M^(00)(11) turns out to be
parallel to tl M^(01)(01), so a suitable alpha*beta cancels the floor exactly while w stays (1,0,0).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple, simplicity_residual, two_form_from_vectors, wedge_scalar,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

OUT = ROOT / "verify" / "Q-0013" / "F-01" / "adversary"
REF = geometric_self_dual_triple(np.eye(4))
NORM_G0 = 2.0 * math.sqrt(3.0)
DELTA = 0.005
EPS3 = np.zeros((3, 3, 3))
EPS3[0, 1, 2] = EPS3[1, 2, 0] = EPS3[2, 0, 1] = 1.0
EPS3[0, 2, 1] = EPS3[2, 1, 0] = EPS3[1, 0, 2] = -1.0
I4 = np.eye(4)
report = {}


def wf(u, v):
    return two_form_from_vectors(np.asarray(u, float), np.asarray(v, float))


def D_linear(l):
    l = np.asarray(l, float)
    out = []
    for i in range(3):
        f = wf(l[0], I4[i + 1]) + wf(I4[0], l[i + 1])
        for j in range(3):
            for k in range(3):
                if EPS3[i, j, k]:
                    f = f + 0.5 * EPS3[i, j, k] * (wf(l[j + 1], I4[k + 1]) + wf(I4[j + 1], l[k + 1]))
        out.append(f)
    return np.asarray(out)


def L_tilde(l):
    d = D_linear(l)
    c1 = np.array([[wedge_scalar(REF[i], d[j]) for j in range(3)] for i in range(3)])
    return d + ((c1 - c1.T) / 4.0) @ REF


def tl3(m):
    return m - np.trace(m) / 3.0 * np.eye(3)


basis = [np.zeros((4, 4)) for _ in range(16)]
for a in range(16):
    basis[a][a // 4, a % 4] = 1.0
Lt = np.array([L_tilde(e) for e in basis])
M = np.zeros((16, 16, 3, 3))
for a in range(16):
    for b in range(16):
        g = np.array([[wedge_scalar(Lt[a][i], Lt[b][j]) for j in range(3)] for i in range(3)])
        M[a, b] = 0.5 * (g + g.T)
Mt = np.array([[tl3(M[a, b]) for b in range(16)] for a in range(16)])

t0011 = Mt[0, 5]
t0101 = Mt[1, 1]
print("tl M_(00)(11) =\n", np.round(t0011, 6))
print("tl M_(01)(01) =\n", np.round(t0101, 6))
cosang = float(np.sum(t0011 * t0101) / (np.linalg.norm(t0011) * np.linalg.norm(t0101)))
print("cos(angle) between them = %.10f" % cosang)
report["cos_angle_M0011_M0101"] = cosang


def F_master(sigma):
    return float(np.linalg.norm(tl3(np.einsum("ab,abij->ij", sigma, M))))


def T_master(sigma):
    return float(np.einsum("abij,ac,bd,cdij->", Mt, sigma, sigma, Mt))


CLS = {1: [(0, 1), (1, 0), (2, 3), (3, 2)], 2: [(0, 2), (2, 0), (3, 1), (1, 3)],
       3: [(0, 3), (3, 0), (1, 2), (2, 1)]}


def w_of(sigma):
    w = np.zeros(3)
    for k in CLS:
        w[k - 1] = sum(sigma[4 * mu + nu, 4 * mu + nu] for (mu, nu) in CLS[k])
    return w


def F_card(sigma):
    w = w_of(sigma)
    return 0.5 * float(np.linalg.norm(w - w.mean()))


def eps_pred(n, F, T):
    return math.sqrt((n - 1) * ((n - 1) * F * F + 2.0 * T) / (12.0 * n * n))


# solve for alpha*beta that cancels: tl G = tl M_(01)(01) + 2 alpha beta tl M_(00)(11) = 0
s = -float(np.sum(t0101 * t0011) / np.sum(t0011 * t0011)) / 2.0
print("required alpha*beta = %.10f" % s)
alpha = math.sqrt(abs(s)) if s > 0 else math.sqrt(abs(s))
beta = alpha if s > 0 else -alpha
u = np.zeros(16)
u[1] = 1.0
v = np.zeros(16)
v[0] = alpha
v[5] = beta
sig = np.outer(u, u) + np.outer(v, v)
Fm, Fc, Tm = F_master(sig), F_card(sig), T_master(sig)
print("Sigma = e01 (x) e01 + v (x) v with v = %.6f e00 + %.6f e11" % (alpha, beta))
print("  w = %s (UNBALANCED)   card closed-form floor/delta2 = %.8f"
      % (np.round(w_of(sig), 6), Fc / NORM_G0))
print("  master F = %.3e (floor/delta2 = %.3e)   T = %.6f" % (Fm, Fm / NORM_G0, Tm))
print("  eps(64)/delta2 : master %.8f   card closed form %.8f"
      % (eps_pred(64, Fm, Tm), eps_pred(64, Fc, Tm)))
report["necessity_counterexample"] = {
    "alpha": alpha, "beta": beta, "w": [round(float(x), 6) for x in w_of(sig)],
    "F_master": Fm, "F_card": Fc, "T": Tm,
    "eps64_master": eps_pred(64, Fm, Tm), "eps64_card": eps_pred(64, Fc, Tm)}


def cell(lab, d=DELTA):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + d * lab)).aligned_candidate


def block(labels, d=DELTA):
    return simplicity_residual(sum(cell(l, d) for l in labels))


def rms(x):
    a = np.asarray(x, float)
    return float(np.sqrt(np.mean(a * a)))


print("")
print("MONTE CARLO (delta=0.005, 400 trials, seed 246810):")
rng = np.random.default_rng(246810)
curve = {}
for n in (4, 16, 64):
    acc = []
    for _ in range(400):
        g1 = rng.normal(size=n)
        g2 = rng.normal(size=n)
        lab = np.zeros((n, 4, 4))
        lab[:, 0, 1] = g1
        lab[:, 0, 0] = alpha * g2
        lab[:, 1, 1] = beta * g2
        acc.append(block(lab))
    curve[n] = rms(acc) / DELTA ** 2
    print("  n=%-3d observed %.8f   master %.8f   card closed form %.8f"
          % (n, curve[n], eps_pred(n, Fm, Tm), eps_pred(n, Fc, Tm)))
sl = float(np.polyfit(np.log(list(curve)), np.log(list(curve.values())), 1)[0])
print("  observed log-log slope = %.4f  (fluctuation-only -0.4534 ; floor-dominated 0)" % sl)
report["necessity_mc"] = {"observed": {str(k): v for k, v in curve.items()}, "slope": sl,
                          "master": {str(n): eps_pred(n, Fm, Tm) for n in (4, 16, 64)},
                          "card": {str(n): eps_pred(n, Fc, Tm) for n in (4, 16, 64)}}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "a6_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                    encoding="utf-8")
print("")
print("wrote a6_report.json")
