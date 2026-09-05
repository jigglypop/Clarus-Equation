"""Adversary b5:
(a) isotropy is LOAD-BEARING: per-direction tl G(L_ab,L_ab) table; a single anisotropic direction
    with tl != 0 produces an n-independent floor (gamma -> 0), which would fire K5.
    (The (0,0) 'lapse' direction is degenerate: it leaves the block exactly simple.)
(b) Q-spine step 4: independent sampler, E[n_b] = b(b+1)/2, pathwise bound D/n^2 <= b^2,
    and an independent estimate of E[D/n_b^2] (tree only, adversary seed).
(c) pre-registration integrity: windows == value +- declared uncertainty, and the card numbers
    == predictions.json numbers == check_modes.py PREREGISTERED constants.
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import (
    geometric_self_dual_triple, simplicity_residual, wedge_scalar)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

REF = geometric_self_dual_triple(np.eye(4)); DELTA = 0.005; ADV = 424242
def tl(M): return M - np.trace(M) / 3.0 * np.eye(3)
def G(A, B): return np.array([[wedge_scalar(A[i], B[j]) for j in range(3)] for i in range(3)])
def cell(l, d=DELTA): return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + d * l)).aligned_candidate
def block(ls, d=DELTA): return simplicity_residual(sum(cell(l, d) for l in ls))
def rms(v): a = np.asarray(v, float); return float(np.sqrt(np.mean(a * a)))
def fit(x, y): return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])

print("== (a) per-direction O(delta^2) traceless mean  || tl G(L_ab,L_ab) || ==")
h = 1e-5; tabl = {}
for a in range(4):
    row = []
    for b in range(4):
        E = np.zeros((4, 4)); E[a, b] = 1.0
        L = (cell(E, h) - cell(E, -h)) / (2 * h)
        row.append(float(np.linalg.norm(tl(G(L, L))))); tabl[(a, b)] = L
    print("   ", [f"{x:8.4f}" for x in row])
best = max(tabl, key=lambda k: np.linalg.norm(tl(G(tabl[k], tabl[k]))))
print(f"   strongest anisotropic direction: entry {best}  ||tl G|| = {np.linalg.norm(tl(G(tabl[best],tabl[best]))):.4f}")
sizes = (4, 8, 16, 32, 64)
r = np.random.default_rng(ADV); vals = []
for n in sizes:
    acc = []
    for _ in range(300):
        lab = np.zeros((n, 4, 4)); lab[:, best[0], best[1]] = r.normal(size=n)
        acc.append(block(lab))
    vals.append(rms(acc))
print("   anisotropic RMS:", [f"{v:.3e}" for v in vals])
print(f"   slope = {fit(sizes, vals):+.4f}   isotropic law = {fit(sizes,[math.sqrt(n-1)/n for n in sizes]):+.4f}"
      f"   -> K5 window [-0.58,-0.38] {'FIRES' if not(-0.58<=fit(sizes,vals)<=-0.38) else 'survives'}")
print("   => step 3's isotropy premise is an assumption on P_micro (label covariance kappa (x) I_16),")
print("      exactly true for the declared Gaussian model, false for anisotropic micro-models.")

print("\n== (b) Q-spine: independent sampler, E[n_b], pathwise bound D/n^2 <= b^2, E[D/n^2] ==")
def qspine(b, rng):
    parent = [k - 1 if k > 0 else -1 for k in range(b)]
    depth = list(range(b)); frontier = []
    for k in range(b):
        if k + 1 <= b - 1:
            for _ in range(int(rng.poisson(1.0))):
                parent.append(k); depth.append(k + 1); frontier.append(len(parent) - 1)
    i = 0
    while i < len(frontier):
        v = frontier[i]; i += 1
        if depth[v] + 1 <= b - 1:
            for _ in range(int(rng.poisson(1.0))):
                parent.append(v); depth.append(depth[v] + 1); frontier.append(len(parent) - 1)
    return parent

def driver(parent):
    n = len(parent); ch = [[] for _ in range(n)]; root = -1
    for v, p in enumerate(parent):
        if p >= 0: ch[p].append(v)
        else: root = v
    order = [root]; i = 0
    while i < len(order): order.extend(ch[order[i]]); i += 1
    dep = np.zeros(n, dtype=np.int64); sub = np.ones(n, dtype=np.int64); pre = np.zeros(n, dtype=np.int64)
    for v in order[1:]: dep[v] = dep[parent[v]] + 1
    for v in reversed(order):
        if parent[v] >= 0: sub[parent[v]] += sub[v]
    for v in order: pre[v] = sub[v] + (pre[parent[v]] if parent[v] >= 0 else 0)
    s = sub.astype(float)
    return float(np.sum((2 * dep + 1) * s * s) - 2 * np.sum(pre.astype(float) ** 2) / n + np.sum(s * s) ** 2 / n ** 2), n

rng = np.random.default_rng(20260902 + 555)
card_tab = {2: 0.1017, 3: 0.2126, 4: 0.3558, 5: 0.5327, 6: 0.7411, 7: 0.9842, 8: 1.2607}
print(f"   {'b':>3} {'E[n] obs':>10} {'b(b+1)/2':>9} {'E[D/n^2] adv':>13} {'card MC':>9} {'max D/n^2':>10} {'bound b^2':>10}")
adv = {}
for b in range(2, 9):
    vals = []; ns = []
    for _ in range(40000):
        d, n = driver(qspine(b, rng)); vals.append(d / n ** 2); ns.append(n)
    adv[b] = float(np.mean(vals))
    print(f"   {b:>3} {np.mean(ns):10.4f} {b*(b+1)/2:9.1f} {adv[b]:13.5f} {card_tab[b]:9.4f} {max(vals):10.4f} {b*b:10d}")
bs = list(range(2, 9))
print(f"   slope of sqrt(E[D/n^2]) vs ln E[n]  = {fit([b*(b+1)/2 for b in bs],[math.sqrt(adv[b]) for b in bs]):.4f}"
      f"   (card pre-registers 0.5047, window [0.42,0.59])")
print(f"   ratio_b8_over_iid36 = {math.sqrt(adv[8])*36/math.sqrt(35):.4f}   (card 6.832, window [6.01,7.65])")
print("   NOTE the bound D/n^2 <= b^2 is loose by ~2 orders; it forbids slope > ~0.5 asymptotically,")
print("        so the 'chain class 1.0' baseline of K3 is excluded by the card's OWN lemma.")

print("\n== (c) pre-registration integrity ==")
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
import importlib.util
spec = importlib.util.spec_from_file_location("cm", ROOT / "verify" / "Q-0008" / "F-02" / "check_modes.py")
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
card_pred = {"iid_slope": (-0.4783, 0.10, "abs"), "her_slope": (0.5302, 0.10, "abs"),
             "her_ratio_128": (32.554, 0.20, "rel"), "mix_X_32": (0.7406, 0.25, "abs"),
             "qspine_slope_vs_En": (0.5047, 0.085, "abs"), "qspine_ratio_b8_over_iid36": (6.832, 0.12, "rel"),
             "defect_ratio_64_over_8": (0.140625, 0.12, "rel"), "defect_slope": (-0.9069, 0.05, "abs")}
pj = json.loads((ROOT / "verify" / "Q-0008" / "F-02" / "predictions.json").read_text(encoding="utf-8"))
src = {"iid_slope": pj["gamma_iid_grid_exact"], "her_slope": pj["gamma_her_cayley_grid"],
       "her_ratio_128": pj["cayley"]["128"]["rms_her_over_iid"], "mix_X_32": pj["cayley"]["32"]["mix_excess_X"],
       "qspine_slope_vs_En": pj["qspine_slope_vs_En"], "qspine_ratio_b8_over_iid36": pj["qspine"]["8"]["ratio_to_iid_at_nstar"],
       "defect_ratio_64_over_8": pj["defect"]["ratio_64_over_8"], "defect_slope": pj["defect"]["slope_grid"]}
ok = True
for k, (v, u, kind) in card_pred.items():
    lo, hi = (v - u, v + u) if kind == "abs" else (v * (1 - u), v * (1 + u))
    wlo, whi = cm.WINDOWS[k]
    match_script = abs(cm.PREREGISTERED[k] - v) <= 5e-4 * max(1, abs(v))
    match_json = abs(src[k] - v) <= 1.5e-3 * max(1, abs(v))
    win_ok = abs(lo - wlo) <= 0.006 * max(1, abs(v)) and abs(hi - whi) <= 0.006 * max(1, abs(v))
    ok &= match_script and match_json and win_ok
    print(f"   {k:28s} card={v:<10} script={cm.PREREGISTERED[k]:<10} json={src[k]:<20.5f}"
          f" window={cm.WINDOWS[k]} vs value+-unc=({lo:.4f},{hi:.4f})  {'OK' if (match_script and match_json and win_ok) else 'MISMATCH'}")
print("   ALL CONSISTENT" if ok else "   *** INCONSISTENT ***")
print("   result.json exists (kills already run)?", (ROOT / 'verify' / 'Q-0008' / 'F-02' / 'result.json').is_file())
