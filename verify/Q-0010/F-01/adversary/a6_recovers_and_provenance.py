"""Adversary a6: (a) execute the four `recovers` limits, (b) reproduce every pre-registered number
from its stated source, (c) rho(theta) interpolation between the orbit tangent and a random 4-plane,
(d) how exactly does the card's LINEAR transmission rule reproduce 13.3 (orbit) -- exactly or O(delta^k)?
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual, fit_slope, rms, heritable_labels, REFERENCE  # noqa: E402
from driver_numbers import tree_arrays, uniform_rooted_tree  # noqa: E402
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual  # noqa: E402
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)
NUM = json.loads((ROOT / "verify/Q-0010/F-01/numbers.json").read_text(encoding="utf-8"))
PRED = json.loads((F02 / "predictions.json").read_text(encoding="utf-8"))
CAY = {int(k): v for k, v in PRED["cayley"].items()}


def diag_proj(idx):
    P = np.zeros((16, 16))
    for i in idx:
        P[i, i] = 1.0
    return P


P_ALIGN = diag_proj(GROUPS["scale"] + GROUPS["sd"])


def labels(c):
    return (c @ FLAT).reshape(-1, 4, 4)


def rule(parent, c, P):
    order, *_ = tree_arrays(parent)
    tr = c @ P.T
    acc = np.zeros_like(c)
    for v in order:
        p = parent[v]
        acc[v] = tr[v] + (acc[p] if p >= 0 else 0.0)
    return acc + (c - tr)


# ---------- (b) provenance of every pre-registered number -------------------------------------
def theory_rms(budget, sizes):
    c0, c1, c2, c3 = budget
    return [math.sqrt(c1 * CAY[n]["E_D"] + 2 * c2 * CAY[n]["E_trHk"] + c3 * (n - 1)) / (n * math.sqrt(c0))
            for n in sizes]


SIZES = (8, 16, 32, 64, 128)
ab, rb = NUM["align_budget"], NUM["rand4_budget"]
alignB = (ab["c0"], ab["c1"], ab["c2"], ab["c3"])
randB = (rb["c0"], rb["c1"], rb["c2"], rb["c3"])
ra, rr = theory_rms(alignB, SIZES), theory_rms(randB, SIZES)
iid = [math.sqrt(n - 1) / n for n in SIZES]
print("[b] pre-registered numbers reproduced from (numbers.json budget) x (F-02 exact Cayley table)")
print("    align_slope   = %.6f   (card -0.4783)" % fit_slope(SIZES, ra))
print("    align_ratio_n = %s   (card 1.000 for all n)" % np.round(np.array(ra) / np.array(iid), 8))
print("    rand4_slope   = %.6f   (card  0.2055)  <- NOT present in numbers.json" % fit_slope(SIZES, rr))
print("    rand4_ratio128= %.6f   (card  7.299)" % (rr[-1] / iid[-1]))
print("    S_gap         = %.6f   (card 41.4465)" % (9 * math.log(100.0)))
print("    F-02 her ratio128 = %.4f (card recovers[1] 32.554)" % math.sqrt(CAY[128]["E_D"] / 127))

# ---------- (a) the four recovers limits, executed --------------------------------------------
print("\n[a] recovers, executed")
rng = np.random.default_rng(31337)
n = 12
parent = uniform_rooted_tree(n, rng)
c = rng.normal(size=(n, 16))
lab0 = rule(parent, c, np.zeros((16, 16)))
lab1 = rule(parent, c, np.eye(16))
print("    P->0 : max|label - xi|                      =", float(np.max(np.abs(lab0 - c))), "(exact identity)")
print("    P->I : max|label - F02 heritable path sum|  =",
      float(np.max(np.abs(labels(lab1) - heritable_labels(parent, labels(c))))), "(exact identity)")
print("    n=1  : residual of a single cell            =",
      float(block_residual(labels(rng.normal(size=(1, 16))), 0.005)))

# 13.3 orbit limit: exact group orbit vs the card's LINEAR tangent-sum rule
def orbit_exact(nn, rg):
    blocked = np.zeros_like(REFERENCE)
    for _ in range(nn):
        s = float(np.exp(0.05 * rg.normal()))
        ax = rg.normal(size=3); ax /= np.linalg.norm(ax); th = 0.3 * rg.normal()
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
        blocked += optimal_internal_alignment(REFERENCE, s * (R @ REFERENCE)).aligned_candidate
    return float(simplicity_residual(blocked))


rg = np.random.default_rng(20260902)
print("    13.3 exact-orbit block residual, n=2..64    =",
      max(orbit_exact(k, rg) for k in (2, 4, 8, 16, 32, 64)), "(card: 2.16e-16)")
print("    card's LINEAR rule with fold = 0 (all increments inside the tangent), n=16:")
for delta in (0.02, 0.04, 0.08, 0.16):
    rr2 = np.random.default_rng(4242)
    vals = []
    for _ in range(24):
        par = uniform_rooted_tree(16, rr2)
        cc = rr2.normal(size=(16, 16)) @ P_ALIGN.T
        v = block_residual(labels(rule(par, cc, P_ALIGN)), delta)
        if math.isfinite(v):
            vals.append(v)
    print("        delta=%.3f  RMS residual = %.4e" % (delta, rms(vals)))

# ---------- (c) rho(theta): interpolate orbit tangent -> random 4-plane -------------------------
from examples.physics.causal_face_simplicity import plebanski_gram  # noqa: E402
I4 = np.eye(4)


def dphi(xi, h=0.37):
    return (geometric_self_dual_triple(I4 + h * xi) - geometric_self_dual_triple(I4 - h * xi)) / (2 * h)


def Gb(A, B):
    return (plebanski_gram(A + B) - plebanski_gram(A - B)) / 4.0


def tl(m):
    return m - np.trace(m) / 3 * np.eye(3)


images = np.asarray([dphi(b) for b in BASIS])
Ea = []
for i in range(3):
    for j in range(i + 1, 3):
        e = np.zeros((3, 3)); e[i, j] = e[j, i] = 1 / math.sqrt(2); Ea.append(e)
Ea.append(np.diag([1., -1., 0.]) / math.sqrt(2)); Ea.append(np.diag([1., 1., -2.]) / math.sqrt(6))
M = np.zeros((5, 16, 16))
for a, e in enumerate(Ea):
    for p in range(16):
        for q in range(16):
            M[a, p, q] = float(np.sum(e * tl(Gb(images[p], images[q]))))
Pi = diag_proj(GROUPS["sd"])
Mt = np.asarray([(np.eye(16) - Pi) @ m @ (np.eye(16) - Pi) for m in M])
C0 = float(sum(np.trace(m @ m) for m in Mt))
E_D, E_TR = CAY[128]["E_D"], CAY[128]["E_trHk"]


def rho_of(P):
    Q = np.eye(16) - P
    c1 = float(sum(np.trace(m @ P @ m @ P) for m in Mt))
    c2 = float(sum(np.trace(m @ P @ m @ Q) for m in Mt))
    c3 = float(sum(np.trace(m @ Q @ m @ Q) for m in Mt))
    return math.sqrt(max((c1 * E_D + 2 * c2 * E_TR + c3 * 127) / (C0 * 127), 0.0))


V0 = np.eye(16)[:, GROUPS["scale"] + GROUPS["sd"]]
W, _ = np.linalg.qr(np.random.default_rng(20260902).normal(size=(16, 4)))
print("\n[c] rho(128) as the transmitted 4-plane rotates from the orbit tangent to the card's random plane")
print("    theta/pi:", end=" ")
for t in np.linspace(0, 1, 11):
    Vt = math.cos(t * math.pi / 2) * V0 + math.sin(t * math.pi / 2) * W
    Qt, _ = np.linalg.qr(Vt)
    print("%.2f:%.2f" % (t, rho_of(Qt @ Qt.T)), end="  ")
print()
