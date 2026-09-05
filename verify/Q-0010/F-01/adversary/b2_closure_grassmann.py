"""Adversary b2 (re-audit): does K4 single out the ORBIT TANGENT among the 4-planes of ker M~,
or only a quaternion subalgebra?  Is the sd/asd tie-break by rendering definition needed?

Adversary derivation under test (not the card):
  X = c*one + a_sd + a_asd in ker M~,  e = 1 + d*X = u*1 + d*A,  A = a_sd + a_asd antisymmetric
  e^T e = u^2*1 - d^2 A^2 = (u^2 + d^2(|a_sd|^2+|a_asd|^2))*1 - 2 d^2 a_sd a_asd
  so e in R_+ SO(4) iff a_sd a_asd = 0 iff a_sd = 0 or a_asd = 0.
  Corollary: 4-dim subspaces S of ker M~ with 1+S inside R_+SO(4) are EXACTLY scale+sd and scale+asd.
"""
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from driver_numbers import tree_arrays, uniform_rooted_tree  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple  # noqa: E402
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)
I4 = np.eye(4)
B0 = geometric_self_dual_triple(I4)
SCALE, SD, ASD = GROUPS["scale"], GROUPS["sd"], GROUPS["asd"]
KER = SCALE + SD + ASD
rng = np.random.default_rng(20260902)


def vec(idx):
    v = np.zeros(16); v[idx] = 1.0
    return v


def mat(coeff16):
    return (np.asarray(coeff16, dtype=float) @ FLAT).reshape(4, 4)


def orbit_residual_single(X, delta):
    e = I4 + delta * X
    if float(np.linalg.det(e)) <= 1e-6:
        return math.nan
    return float(optimal_internal_alignment(B0, geometric_self_dual_triple(e)).orbit_residual)


print("[1] algebraic identity for elements of ker M~")
ONE = mat(vec(SCALE[0]))
worst_id, worst_sq, cross_min = 0.0, 0.0, 1e9
for _ in range(200):
    c, d = float(rng.normal()), 0.3
    a_sd = mat(sum(w * vec(i) for w, i in zip(rng.normal(size=3), SD)))
    a_asd = mat(sum(w * vec(i) for w, i in zip(rng.normal(size=3), ASD)))
    A = a_sd + a_asd
    e = I4 + d * (c * ONE + A)
    u = I4 + d * c * ONE
    worst_id = max(worst_id, float(np.abs(e.T @ e + d * d * (A @ A) - u.T @ u).max()))
    worst_sq = max(worst_sq, float(np.abs(a_sd @ a_sd + np.sum(a_sd * a_sd) / 4.0 * I4).max()))
    cross_min = min(cross_min, float(np.linalg.norm(a_sd @ a_asd)))
print("    max | e^T e + d^2 A^2 - u^T u |            = %.3e" % worst_id)
print("    max | a_sd^2 + (|a_sd|_F^2/4) 1 |          = %.3e (nonzero a_sd invertible)" % worst_sq)
print("    min ||a_sd a_asd||_F over 200 pairs        = %.4f (never zero)" % cross_min)

print("")
print("[2] Urbantke direction: Phi(e) in O(B_0) iff e^T e prop. to 1")
for label in ("alpha*SO(4)", "generic e"):
    res, dev = [], []
    for _ in range(60):
        if label == "alpha*SO(4)":
            Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
            Q = Q if float(np.linalg.det(Q)) > 0 else Q @ np.diag([-1.0, 1.0, 1.0, 1.0])
            e = math.exp(0.2 * rng.normal()) * Q
        else:
            e = I4 + 0.3 * rng.normal(size=(4, 4))
        if float(np.linalg.det(e)) <= 1e-6:
            continue
        g = e.T @ e
        dev.append(float(np.linalg.norm(g - np.trace(g) / 4 * I4) / np.linalg.norm(g)))
        res.append(float(optimal_internal_alignment(B0, geometric_self_dual_triple(e)).orbit_residual))
    print("    %-12s max conformal dev = %.3e   max orbit residual = %.3e" % (label, max(dev), max(res)))


print("")
print("[3] which 4-planes of ker M~ (7-dim) close the orbit exactly?  (delta = 0.2)")


def plane_residual(basis_vectors, delta=0.2, draws=40):
    worst = 0.0
    for _ in range(draws):
        w = rng.normal(size=len(basis_vectors))
        X = mat(np.tensordot(w, np.asarray(basis_vectors), axes=1))
        v = orbit_residual_single(X, delta)
        if math.isfinite(v):
            worst = max(worst, v)
    return worst


named = {
    "align  scale+sd = T_{B0}O": [vec(SCALE[0])] + [vec(i) for i in SD],
    "alt1   scale+asd":          [vec(SCALE[0])] + [vec(i) for i in ASD],
    "alt2   sd+asd_1":           [vec(i) for i in SD] + [vec(ASD[0])],
    "null7  whole ker M~ (7d)":  [vec(i) for i in KER],
    "sd only (3d)":              [vec(i) for i in SD],
    "scale+diag so(3) (45 deg)": [vec(SCALE[0])] + [(vec(SD[i]) + vec(ASD[i])) / math.sqrt(2) for i in range(3)],
}
for name, bas in named.items():
    print("    %-28s max single-cell orbit residual = %.3e" % (name, plane_residual(bas)))

hits = 0
for _ in range(400):
    Q, _ = np.linalg.qr(rng.normal(size=(7, 4)))
    bas = []
    for k in range(4):
        v = np.zeros(16)
        for j, idx in enumerate(KER):
            v[idx] = Q[j, k]
        bas.append(v)
    if plane_residual(bas, draws=12) < 1e-12:
        hits += 1
print("    400 Haar-random 4-planes of ker M~: exact-closure count = %d" % hits)

line = []
for theta in np.linspace(0.0, math.pi / 2, 7):
    bas = [vec(SCALE[0])] + [math.cos(theta) * vec(SD[i]) + math.sin(theta) * vec(ASD[i]) for i in range(3)]
    line.append("%2d deg:%.1e" % (round(math.degrees(theta)), plane_residual(bas, draws=12)))
print("    theta-mixed scale + span{cos t sd_i + sin t asd_i}: " + "  ".join(line))


print("")
print("[4] fold = 0: within-block spread of the per-cell polar rotation angle (a Sigma observable)")


def rule(parent, c, P):
    order, *_ = tree_arrays(parent)
    tr = c @ P.T
    acc = np.zeros_like(c)
    for v in order:
        p = parent[v]
        acc[v] = tr[v] + (acc[p] if p >= 0 else 0.0)
    return acc + (c - tr)


def diag_proj(idx):
    P = np.zeros((16, 16))
    for i in idx:
        P[i, i] = 1.0
    return P


def rotation_spread(P, n, seed, delta=0.2, trials=32):
    rr = np.random.default_rng(seed)
    out = []
    while len(out) < trials:
        par = uniform_rooted_tree(n, rr)
        c = rr.normal(size=(len(par), 16)) @ P.T
        lab = (rule(par, c, P) @ FLAT).reshape(-1, 4, 4)
        angles, ok = [], True
        for l in lab:
            e = I4 + delta * l
            if float(np.linalg.det(e)) <= 0.05:
                ok = False
                break
            R = optimal_internal_alignment(B0, geometric_self_dual_triple(e)).rotation
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))))
        if ok:
            out.append(float(np.std(angles)))
    return float(np.mean(out))


for n in (8, 16, 32):
    a = rotation_spread(diag_proj(SCALE + SD), n, 515151 + n)
    b = rotation_spread(diag_proj(SCALE + ASD), n, 515151 + n)
    print("    n=%2d  align(scale+sd) = %7.3f deg   alt1(scale+asd) = %.3e deg" % (n, a, b))

print("")
print("[5] visible transmitted dimension: rank of dPhi restricted to each plane")


def dphi_vec(v, h=0.37):
    return (geometric_self_dual_triple(I4 + h * mat(v)) - geometric_self_dual_triple(I4 - h * mat(v))).ravel() / (2 * h)


for name, bas in named.items():
    s = np.linalg.svd(np.asarray([dphi_vec(v) for v in bas]), compute_uv=False)
    print("    %-28s rank = %d   sv = %s" % (name, int(np.sum(s > 1e-8)), np.round(s, 4)))
