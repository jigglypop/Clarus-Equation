"""Adversary a1 (Q-0013 F-01 card audit): b4 re-examination, exact zero mode, own structure constants."""
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
report = {}


def cell(lab, d):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + d * lab)).aligned_candidate


def block(labels, d):
    return simplicity_residual(sum(cell(l, d) for l in labels))


def rms(v):
    a = np.asarray(v, float)
    return float(np.sqrt(np.mean(a * a)))


def slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


print("=" * 78)
print("(A) b4 part 2 EXACT rerun: only the (0,0) tetrad entry fluctuates")
print("    seed 424242+21, TR=500, delta=0.005, sizes (4,8,16,32)")
sizes = (4, 8, 16, 32)
r = np.random.default_rng(424242 + 21)
vals = []
maxes = []
for n in sizes:
    acc = []
    for _ in range(500):
        lab = np.zeros((n, 4, 4))
        lab[:, 0, 0] = r.normal(size=n)
        acc.append(block(lab, 0.005))
    vals.append(rms(acc))
    maxes.append(float(max(acc)))
    print("    n=%-3d RMS = %.6e   max = %.6e" % (n, vals[-1], maxes[-1]))
b4_slope = slope(sizes, vals)
print("    measured slope = %+.6f   (b4 recorded -0.006)" % b4_slope)
iso_pred = [math.sqrt(10) * 0.005 ** 2 * math.sqrt(n - 1) / n for n in sizes]
print("    isotropic-law scale for comparison: %.3e .. %.3e" % (iso_pred[0], iso_pred[-1]))
report["b4_rerun"] = {"sizes": list(sizes), "rms": vals, "max": maxes, "slope": b4_slope,
                      "isotropic_law_scale": iso_pred}

print()
print("=" * 78)
print("(B) exact zero mode: single diagonal tetrad entry, wide delta, adversarial amplitudes")
zero_rows = []
worst = 0.0
rng = np.random.default_rng(20260902)
for comp in ((0, 0), (1, 1), (2, 2), (3, 3)):
    for d in (0.005, 0.3, 1.0, 3.0):
        for n in (2, 4, 16, 64):
            acc = []
            for _ in range(40):
                lab = np.zeros((n, 4, 4))
                g = rng.normal(size=n)
                g = np.where(np.abs(1.0 + d * g) < 1e-6, 0.0, g)
                lab[:, comp[0], comp[1]] = g
                acc.append(abs(block(lab, d)))
            m = float(max(acc))
            worst = max(worst, m)
            zero_rows.append({"comp": list(comp), "delta": d, "n": n, "max": m})
print("    worst |residual| (4 comps x 4 deltas x 4 sizes x 40 trials) = %.3e" % worst)
report["zero_mode_generic"] = {"worst_max_residual": worst, "rows": zero_rows}

print("    adversarial alignment-flip probe (cells with 1 + delta*g < -1):")
flip_rows = []
for comp in ((0, 0), (1, 1)):
    for d in (0.3, 1.0):
        n = 8
        acc = []
        for _ in range(200):
            lab = np.zeros((n, 4, 4))
            g = rng.normal(size=n) * 4.0 / d
            g = np.where(np.abs(1.0 + d * g) < 1e-6, 0.0, g)
            lab[:, comp[0], comp[1]] = g
            acc.append(abs(block(lab, d)))
        flip_rows.append({"comp": list(comp), "delta": d, "max": float(max(acc)),
                          "median": float(np.median(acc))})
        print("      comp=%s delta=%s: max=%.4e  median=%.4e" % (comp, d, max(acc), np.median(acc)))
report["zero_mode_flip_probe"] = flip_rows

print()
print("=" * 78)
print("(C) independent (analytic) structure constants")

EPS3 = np.zeros((3, 3, 3))
EPS3[0, 1, 2] = EPS3[1, 2, 0] = EPS3[2, 0, 1] = 1.0
EPS3[0, 2, 1] = EPS3[2, 1, 0] = EPS3[1, 0, 2] = -1.0
I4 = np.eye(4)


def wf(u, v):
    return two_form_from_vectors(np.asarray(u, float), np.asarray(v, float))


def sigma_of(e):
    e = np.asarray(e, float)
    out = []
    for i in range(3):
        f = wf(e[0], e[i + 1])
        for j in range(3):
            for k in range(3):
                if EPS3[i, j, k]:
                    f = f + 0.5 * EPS3[i, j, k] * wf(e[j + 1], e[k + 1])
        out.append(f)
    return np.asarray(out)


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


SIG0 = sigma_of(I4)
assert np.allclose(SIG0, REF)


def L_tilde(l):
    d = D_linear(l)
    c1 = np.array([[wedge_scalar(SIG0[i], d[j]) for j in range(3)] for i in range(3)])
    omega = (c1 - c1.T) / 4.0
    return d + omega @ SIG0


basis = []
for a in range(16):
    e = np.zeros((4, 4))
    e[a // 4, a % 4] = 1.0
    basis.append(e)
Lt = np.array([L_tilde(e) for e in basis])
fd = np.array([(cell(e, 1e-5) - cell(e, -1e-5)) / 2e-5 for e in basis])
fd_err = float(np.max(np.abs(fd - Lt)))
print("    analytic vs card finite-difference L~: max abs diff = %.3e" % fd_err)
report["L_tilde_analytic_vs_fd"] = fd_err

M = np.zeros((16, 16, 3, 3))
for a in range(16):
    for b in range(16):
        g = np.array([[wedge_scalar(Lt[a][i], Lt[b][j]) for j in range(3)] for i in range(3)])
        M[a, b] = 0.5 * (g + g.T)
Mt = M - np.trace(M, axis1=2, axis2=3)[:, :, None, None] / 3.0 * np.eye(3)

names = ["%d%d" % (a // 4, a % 4) for a in range(16)]
axis_of = {}
for a in range(16):
    t = Mt[a, a]
    if float(np.linalg.norm(t)) < 1e-12:
        axis_of[names[a]] = 0
        continue
    found = -1
    for k in range(3):
        cand = 0.5 * (np.outer(np.eye(3)[k], np.eye(3)[k]) - np.eye(3) / 3.0)
        if np.linalg.norm(t - cand) < 1e-10:
            found = k + 1
    axis_of[names[a]] = found
print("    axis classes:", axis_of)
report["axis_class_independent"] = axis_of

sum_tl = float(np.linalg.norm(sum(Mt[a, a] for a in range(16))))
T_iso = float(np.einsum("abij,abij->", Mt, Mt))
print("    ||sum_a tl M^aa||_F = %.3e    T(I_16) = %.10f" % (sum_tl, T_iso))
report["sum_a_tl_M_aa"] = sum_tl
report["T_I16"] = T_iso

pair_norms = {}
for (a, b) in ((1, 2), (1, 3), (2, 3), (1, 4), (1, 7), (1, 0), (1, 5), (1, 11), (1, 14), (2, 8)):
    pair_norms["%s-%s" % (names[a], names[b])] = float(np.linalg.norm(Mt[a, b]) ** 2)
print("    ||tl M^ab||_F^2 for selected a != b:", pair_norms)
report["offdiag_tlM_norm2"] = pair_norms

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "a1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
np.save(OUT / "a1_Mt.npy", Mt)
np.save(OUT / "a1_M.npy", M)
print()
print("wrote a1_report.json")
