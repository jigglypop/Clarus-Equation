"""Adversary a3 part 1: independent structure constants."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple, simplicity_residual, two_form_from_vectors, wedge_scalar,
)
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

OUT = ROOT / "verify" / "Q-0013" / "F-01" / "adversary"
REF = geometric_self_dual_triple(np.eye(4))
NORM_G0 = 2.0 * math.sqrt(3.0)
DELTA = 0.005
report = {}

EPS3 = np.zeros((3, 3, 3))
EPS3[0, 1, 2] = EPS3[1, 2, 0] = EPS3[2, 0, 1] = 1.0
EPS3[0, 2, 1] = EPS3[2, 1, 0] = EPS3[1, 0, 2] = -1.0
I4 = np.eye(4)


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


def cell(lab, d=DELTA):
    tri = geometric_self_dual_triple(np.eye(4) + d * lab)
    return optimal_internal_alignment(REF, tri).aligned_candidate


def block(labels, d=DELTA):
    return simplicity_residual(sum(cell(l, d) for l in labels))


def rms(v):
    a = np.asarray(v, float)
    return float(np.sqrt(np.mean(a * a)))


def tl3(m):
    return m - np.trace(m) / 3.0 * np.eye(3)


basis = [np.zeros((4, 4)) for _ in range(16)]
for a in range(16):
    basis[a][a // 4, a % 4] = 1.0
Lt = np.array([L_tilde(e) for e in basis])
fd = np.array([(cell(e, 1e-5) - cell(e, -1e-5)) / 2e-5 for e in basis])
print("(C) analytic L vs card finite-difference L: max abs diff = %.3e"
      % float(np.max(np.abs(fd - Lt))))
report["L_tilde_analytic_vs_card_fd"] = float(np.max(np.abs(fd - Lt)))

M = np.zeros((16, 16, 3, 3))
for a in range(16):
    for b in range(16):
        g = np.array([[wedge_scalar(Lt[a][i], Lt[b][j]) for j in range(3)] for i in range(3)])
        M[a, b] = 0.5 * (g + g.T)
Mt = np.array([[tl3(M[a, b]) for b in range(16)] for a in range(16)])
names = ["%d%d" % (a // 4, a % 4) for a in range(16)]

axis_of = {}
for a in range(16):
    t = Mt[a, a]
    if float(np.linalg.norm(t)) < 1e-12:
        axis_of[names[a]] = 0
        continue
    hit = -1
    for k in range(3):
        cand = 0.5 * (np.outer(np.eye(3)[k], np.eye(3)[k]) - np.eye(3) / 3.0)
        if np.linalg.norm(t - cand) < 1e-10:
            hit = k + 1
    axis_of[names[a]] = hit
print("    axis classes (independent):", axis_of)
sum_tl = float(np.linalg.norm(sum(Mt[a, a] for a in range(16))))
T_iso = float(np.einsum("abij,abij->", Mt, Mt))
print("    norm(sum_a tl M_aa) = %.3e ;  T(I16) = %.12f ;  eps_star/delta2 = %.10f ; sqrt10 = %.10f"
      % (sum_tl, T_iso, math.sqrt(2 * T_iso) / NORM_G0, math.sqrt(10.0)))
report["axis_class_independent"] = axis_of
report["sum_a_tl_M_aa"] = sum_tl
report["T_I16"] = T_iso
report["eps_star_over_delta2_iso"] = math.sqrt(2 * T_iso) / NORM_G0

offdiag = {}
mx = 0.0
for a in range(16):
    for b in range(16):
        if a != b:
            v = float(np.linalg.norm(Mt[a, b]))
            mx = max(mx, v)
            if v > 1e-12:
                offdiag["%s_%s" % (names[a], names[b])] = round(v * v, 12)
print("    max over a != b of norm(tl M_ab) = %.6f ; nonzero off-diagonal pairs = %d of 240"
      % (mx, len(offdiag)))
report["max_offdiag_tlM_norm"] = mx
report["n_nonzero_offdiag_pairs"] = len(offdiag)
report["offdiag_examples"] = dict(list(offdiag.items())[:12])


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


def rank1(vec):
    u = np.zeros(16)
    for item in vec:
        u[4 * item[0][0] + item[0][1]] = item[1]
    u = u / np.linalg.norm(u)
    return np.outer(u, u), u


print("")
print("=" * 78)
print("(D) closed form vs master formula on CORRELATED (non diagonal) Sigma")
cases = {}
tests = [
    ("rank1 e01  [card K1, diagonal]", [((0, 1), 1.0)]),
    ("rank1 (e01+e02)/sqrt2", [((0, 1), 1.0), ((0, 2), 1.0)]),
    ("rank1 (e01+e02+e03)/sqrt3  [w BALANCED]", [((0, 1), 1.0), ((0, 2), 1.0), ((0, 3), 1.0)]),
    ("rank1 (e01+e10)/sqrt2", [((0, 1), 1.0), ((1, 0), 1.0)]),
    ("rank1 (e01+e23)/sqrt2", [((0, 1), 1.0), ((2, 3), 1.0)]),
    ("rank1 (e01-e23)/sqrt2", [((0, 1), 1.0), ((2, 3), -1.0)]),
    ("rank1 (e00+e11)/sqrt2  [zero-mode combo]", [((0, 0), 1.0), ((1, 1), 1.0)]),
]
for name, vec in tests:
    sig, u = rank1(vec)
    Fm = F_master(sig)
    Fc = F_card(sig)
    Tm = T_master(sig)
    cases[name] = {"w": [round(float(x), 6) for x in w_of(sig)], "F_master": Fm,
                   "F_card_closed_form": Fc, "T": Tm,
                   "floor_over_delta2_master": Fm / NORM_G0,
                   "floor_over_delta2_card": Fc / NORM_G0,
                   "eps64_master": eps_pred(64, Fm, Tm), "eps64_card": eps_pred(64, Fc, Tm)}
    ratio = ("%.4f" % (Fm / Fc)) if Fc > 1e-12 else "INF (card closed form gives 0)"
    print("  %-42s w=%s" % (name, np.round(w_of(sig), 4)))
    print("      F_master=%.8f  F_card=%.8f  ratio=%s" % (Fm, Fc, ratio))
    print("      eps(64)/delta2 : master %.8f   card closed form %.8f"
          % (eps_pred(64, Fm, Tm), eps_pred(64, Fc, Tm)))
report["closed_form_vs_master"] = cases

print("")
print("  full-rank Sigma on {e01,e02,e03} with pairwise correlation rho (w always balanced):")
for rho in (0.0, 0.5, 0.9, 1.0):
    sig = np.zeros((16, 16))
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            sig[i, j] = 1.0 if i == j else rho
    Fm = F_master(sig)
    Fc = F_card(sig)
    Tm = T_master(sig)
    print("    rho=%.1f w=%s F_master=%.8f F_card=%.8f eps64 master=%.8f card=%.8f"
          % (rho, np.round(w_of(sig), 3), Fm, Fc, eps_pred(64, Fm, Tm), eps_pred(64, Fc, Tm)))
    cases["piso_rho_%.1f" % rho] = {"w": [round(float(x), 6) for x in w_of(sig)],
                                    "F_master": Fm, "F_card_closed_form": Fc, "T": Tm,
                                    "eps64_master": eps_pred(64, Fm, Tm),
                                    "eps64_card": eps_pred(64, Fc, Tm)}

print("")
print("=" * 78)
print("(E) MONTE CARLO check (card convention: delta=0.005, iid cells, trial RMS)")
mc = {}
specs = [("rank1 e01  [card K1]", [((0, 1), 1.0)]),
         ("rank1 (e01+e02+e03)/sqrt3  [w balanced]", [((0, 1), 1.0), ((0, 2), 1.0), ((0, 3), 1.0)]),
         ("piso independent  [card K3]", None)]
for name, vec in specs:
    rng = np.random.default_rng(777001)
    curve = {}
    for n in (4, 16, 64):
        acc = []
        for _ in range(400):
            lab = np.zeros((n, 4, 4))
            if vec is None:
                lab[:, 0, 1] = rng.normal(size=n)
                lab[:, 0, 2] = rng.normal(size=n)
                lab[:, 0, 3] = rng.normal(size=n)
            else:
                g = rng.normal(size=n)
                nrm = math.sqrt(sum(item[1] * item[1] for item in vec))
                for item in vec:
                    lab[:, item[0][0], item[0][1]] = g * item[1] / nrm
            acc.append(block(lab))
        curve[n] = rms(acc) / DELTA ** 2
    if vec is None:
        sig = np.zeros((16, 16))
        for a in (1, 2, 3):
            sig[a, a] = 1.0
    else:
        sig, u = rank1(vec)
    Fm = F_master(sig)
    Fc = F_card(sig)
    Tm = T_master(sig)
    mc[name] = {"observed": {str(k): curve[k] for k in curve},
                "master_pred": {str(n): eps_pred(n, Fm, Tm) for n in (4, 16, 64)},
                "card_closed_form_pred": {str(n): eps_pred(n, Fc, Tm) for n in (4, 16, 64)}}
    print("  %-42s observed  %s" % (name, {k: round(curve[k], 6) for k in curve}))
    print("  %-42s master    %s" % ("", {n: round(eps_pred(n, Fm, Tm), 6) for n in (4, 16, 64)}))
    print("  %-42s cardform  %s" % ("", {n: round(eps_pred(n, Fc, Tm), 6) for n in (4, 16, 64)}))
report["monte_carlo"] = mc

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "a3_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                    encoding="utf-8")
print("")
print("wrote a3_report.json")
