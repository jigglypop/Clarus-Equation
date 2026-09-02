"""Adversary a1 (dimension): independent recount of 16 = 1+3+3+9, rank dPhi = 13, dim T_{B0}O = 4.

Deliberately DIFFERENT route from the card driver:
  * dPhi by EXACT polarization with a weird step h = 0.37 (Phi is quadratic, so this is exact),
    not a central difference with h = 1e-3;
  * matrix of dPhi taken in the STANDARD basis E_ij of gl(4), not in the card's structured basis;
  * T_{B0}O built twice: (a) algebraically span{B0, L_a B0}; (b) numerically as the tangent of the
    curve t -> alpha(t) R(t) B0 for random (alpha', omega);
  * ranks from SVD spectra printed in full so the gap is visible (no hidden tol).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=140)
PHI = lambda e: geometric_self_dual_triple(e).reshape(-1)   # (18,)
I4 = np.eye(4)
B0 = PHI(I4)


def dphi(xi, h=0.37):
    """Exact derivative at I by polarization (Phi quadratic => independent of h)."""
    return (PHI(I4 + h * xi) - PHI(I4 - h * xi)) / (2.0 * h)


# --- 0. exactness of the polarization route (h-independence) -------------------------------
rng = np.random.default_rng(11111)
xi = rng.normal(size=(4, 4))
hs = [0.37, 0.011, 1.7]
d_at = [dphi(xi, h) for h in hs]
print("[0] dPhi h-independence (quadratic Phi):",
      max(float(np.max(np.abs(d_at[0] - d))) for d in d_at[1:]))

# --- 1. rank of dPhi in the STANDARD basis --------------------------------------------------
std = []
for mu in range(4):
    for nu in range(4):
        e = np.zeros((4, 4)); e[mu, nu] = 1.0
        std.append(e)
J = np.asarray([dphi(e) for e in std])          # (16, 18)
sv = np.linalg.svd(J, compute_uv=False)
print("[1] singular values of dPhi (standard basis):", sv)
print("    rank at tol 1e-8 / 1e-6 / 1e-10:",
      [int(np.linalg.matrix_rank(J, tol=t)) for t in (1e-8, 1e-6, 1e-10)],
      " gap sv[12]/sv[13] =", float(sv[12] / max(sv[13], 1e-300)))

# --- 2. kernel of dPhi is the anti-self-dual so(3) -------------------------------------------
u, s, vt = np.linalg.svd(J.T)                    # columns of J.T span image; kernel from J rows
ker = np.linalg.svd(J)[2] if False else None
_, _, vh = np.linalg.svd(J)
kernel = vh[np.sum(sv > 1e-8):]                  # rows spanning ker (in coefficient space? no)
# careful: J maps coefficient vector c -> c @ J ; kernel = left null space of J
uJ, sJ, vJ = np.linalg.svd(J)
kernel_coeffs = uJ[:, np.sum(sJ > 1e-8):]        # (16, k)
print("[2] dim ker dPhi =", kernel_coeffs.shape[1])


def anti(mu, nu):
    out = np.zeros((4, 4)); out[mu, nu] = 1.0; out[nu, mu] = -1.0
    return out / np.sqrt(2.0)


cyc = ((1, 2, 3), (2, 3, 1), (3, 1, 2))
sd = np.asarray([(anti(0, i) + anti(j, k)) / np.sqrt(2.0) for i, j, k in cyc])
asd = np.asarray([(anti(0, i) - anti(j, k)) / np.sqrt(2.0) for i, j, k in cyc])
asd_coeffs = asd.reshape(3, 16)                  # standard-basis coefficients (row-major = std order)
proj_ker = kernel_coeffs @ kernel_coeffs.T
resid = asd_coeffs - asd_coeffs @ proj_ker.T
print("    ||asd - P_ker asd|| =", float(np.linalg.norm(resid)),
      "  ||dPhi(asd)|| =", [float(np.linalg.norm(dphi(a))) for a in asd])
print("    ||dPhi(sd)|| =", [float(np.linalg.norm(dphi(a))) for a in sd])

# --- 3. dim T_{B0}O two ways -----------------------------------------------------------------
B0m = B0.reshape(3, 6)
Lb = []
eps = np.zeros((3, 3, 3)); eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
for a in range(3):
    v = np.zeros_like(B0m)
    for i in range(3):
        for j in range(3):
            if eps[a, i, j]:
                v[i] += eps[a, i, j] * B0m[j]
    Lb.append(v.reshape(-1))
T_alg = np.asarray([B0] + Lb)
svT = np.linalg.svd(T_alg, compute_uv=False)
print("[3a] singular values of {B0, L_a B0}:", svT, " rank =", int(np.linalg.matrix_rank(T_alg, tol=1e-8)))

curve = []
for _ in range(12):
    a1 = float(rng.normal()); om = rng.normal(size=3)
    t = 1e-4
    def act(t):
        cross = np.array([[0, -om[2], om[1]], [om[2], 0, -om[0]], [-om[1], om[0], 0]]) * t
        R = np.eye(3) + cross + 0.5 * cross @ cross + (cross @ cross @ cross) / 6.0
        return (np.exp(a1 * t) * (R @ B0m)).reshape(-1)
    curve.append((act(t) - act(-t)) / (2 * t))
curve = np.asarray(curve)
svC = np.linalg.svd(curve, compute_uv=False)
print("[3b] rank of 12 numeric orbit-curve tangents:", int(np.linalg.matrix_rank(curve, tol=1e-6)),
      " sv:", svC[:6])
both = np.concatenate([T_alg, curve], axis=0)
print("     rank of union(alg, numeric) =", int(np.linalg.matrix_rank(both, tol=1e-6)))

# --- 4. image split: dPhi(scale), dPhi(sd), dPhi(sym) vs T ------------------------------------
scale = I4 / 2.0
sym = []
for mu in range(4):
    for nu in range(mu + 1, 4):
        m = np.zeros((4, 4)); m[mu, nu] = m[nu, mu] = 1 / np.sqrt(2.0); sym.append(m)
for m in (np.diag([1., -1., 0., 0.]) / np.sqrt(2), np.diag([1., 1., -2., 0.]) / np.sqrt(6),
          np.diag([1., 1., 1., -3.]) / np.sqrt(12)):
    sym.append(m)
sym = np.asarray(sym)
print("[4] dim(sym traceless 4x4) =", len(sym), " (10 sym - 1 trace = 9)")
d_scale = dphi(scale); d_sd = np.asarray([dphi(a) for a in sd]); d_sym = np.asarray([dphi(a) for a in sym])
print("    dPhi(scale) parallel to B0?  cos =",
      float(d_scale @ B0 / (np.linalg.norm(d_scale) * np.linalg.norm(B0))))
print("    rank(T) =", int(np.linalg.matrix_rank(T_alg, tol=1e-8)),
      " rank(T + dPhi(sd)) =", int(np.linalg.matrix_rank(np.concatenate([T_alg, d_sd]), tol=1e-8)),
      " rank(dPhi(sym)) =", int(np.linalg.matrix_rank(d_sym, tol=1e-8)),
      " rank(T + dPhi(sym)) =", int(np.linalg.matrix_rank(np.concatenate([T_alg, d_sym]), tol=1e-8)))
Pt = np.linalg.pinv(T_alg) @ T_alg
res_sym = d_sym - d_sym @ Pt
print("    ||(1-P_T) dPhi(sym)||_F / ||dPhi(sym)||_F =",
      float(np.linalg.norm(res_sym) / np.linalg.norm(d_sym)),
      "   overlap of dPhi(sym) with T (Frobenius fraction) =",
      float(np.linalg.norm(d_sym @ Pt) / np.linalg.norm(d_sym)))

# --- 5. the three routes to '9' ---------------------------------------------------------------
r = int(np.linalg.matrix_rank(J, tol=1e-8))
print("[5] rank dPhi - dim T = %d - %d = %d ; 16 - ker(3) - 4 = %d ; dim sym-traceless(4x4) = %d"
      % (r, int(np.linalg.matrix_rank(T_alg, tol=1e-8)), r - 4, 16 - 3 - 4, len(sym)))

# --- 6. is the card's 16 = 1+3+3+9 an orthogonal Frobenius split? -------------------------------
allb = np.concatenate([scale.reshape(1, 16), sd.reshape(3, 16), asd.reshape(3, 16), sym.reshape(9, 16)])
print("[6] label basis orthonormality max err =", float(np.max(np.abs(allb @ allb.T - np.eye(16)))))
