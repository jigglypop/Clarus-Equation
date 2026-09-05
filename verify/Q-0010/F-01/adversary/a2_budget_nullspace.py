"""Adversary a2 (content/dof): what exactly makes c1 = 0?  How big is the family of projectors
that reproduces the card's K1/K3 numbers?

Independent construction of the residual generator:
  G(A,B) := [plebanski_gram(A+B) - plebanski_gram(A-B)] / 4    (polarization, not the card's
            hand-written symmetrized wedge loop)
  M_a[p,q] := <E_a, tl G(dPhi b_p, dPhi b_q)>,  Mt_a := (1-Pi_rot) M_a (1-Pi_rot)
Then:
  (i)  the common null space  N = {xi : Mt_a xi = 0 for all a}   -- dimension and content
  (ii) budgets (c0,c1,c2,c3) and the predicted rho(128) for MANY projectors, including 4-dim
       projectors that are NOT the orbit tangent
  (iii) the raw-M budget (no polar-alignment convention) for the card's P_align
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram  # noqa: E402

I4 = np.eye(4)
PHI = lambda e: geometric_self_dual_triple(e)
B0 = PHI(I4)
E_D_128, E_TRHK_128 = 134587.1450461609, 822.7392437084596     # F-02 exact Cayley table


def dphi(xi, h=0.37):
    return (PHI(I4 + h * xi) - PHI(I4 - h * xi)) / (2.0 * h)


def G(A, B):
    return (plebanski_gram(A + B) - plebanski_gram(A - B)) / 4.0


def tl(m):
    return m - np.trace(m) / 3.0 * np.eye(3)


def anti(mu, nu):
    o = np.zeros((4, 4)); o[mu, nu] = 1.0; o[nu, mu] = -1.0
    return o / np.sqrt(2.0)


cyc = ((1, 2, 3), (2, 3, 1), (3, 1, 2))
basis = [I4 / 2.0]
basis += [(anti(0, i) + anti(j, k)) / np.sqrt(2.0) for i, j, k in cyc]          # sd  1..3
basis += [(anti(0, i) - anti(j, k)) / np.sqrt(2.0) for i, j, k in cyc]          # asd 4..6
for mu in range(4):
    for nu in range(mu + 1, 4):
        m = np.zeros((4, 4)); m[mu, nu] = m[nu, mu] = 1 / np.sqrt(2.0); basis.append(m)
for m in (np.diag([1., -1., 0., 0.]) / np.sqrt(2), np.diag([1., 1., -2., 0.]) / np.sqrt(6),
          np.diag([1., 1., 1., -3.]) / np.sqrt(12)):
    basis.append(m)
basis = np.asarray(basis)
IDX = {"scale": [0], "sd": [1, 2, 3], "asd": [4, 5, 6], "sym": list(range(7, 16))}
images = np.asarray([dphi(b) for b in basis])

Ea = []
for i in range(3):
    for j in range(i + 1, 3):
        e = np.zeros((3, 3)); e[i, j] = e[j, i] = 1 / np.sqrt(2.0); Ea.append(e)
Ea.append(np.diag([1., -1., 0.]) / np.sqrt(2)); Ea.append(np.diag([1., 1., -2.]) / np.sqrt(6))
Ea = np.asarray(Ea)

M = np.zeros((5, 16, 16))
for a, e in enumerate(Ea):
    for p in range(16):
        for q in range(16):
            M[a, p, q] = float(np.sum(e * tl(G(images[p], images[q]))))
Pi_rot = np.zeros((16, 16))
for i in IDX["sd"]:
    Pi_rot[i, i] = 1.0
Mt = np.asarray([(np.eye(16) - Pi_rot) @ m @ (np.eye(16) - Pi_rot) for m in M])

print("[0] independent M vs card driver: ||M||_F =", float(np.sqrt(np.sum(M * M))),
      " ||Mt||_F =", float(np.sqrt(np.sum(Mt * Mt))), " (card: Mtilde_norm 7.745966692414969)")
print("    symmetry max|M - M^T| =", float(max(np.max(np.abs(m - m.T)) for m in M)))

# ---- (i) common null space of the five Mt_a --------------------------------------------------
stack = np.concatenate(list(Mt), axis=0)          # (5*16, 16): xi -> (Mt_a xi)_a
sv = np.linalg.svd(stack, compute_uv=False)
null_dim = int(np.sum(sv <= 1e-8))
print("[1] singular values of stacked Mt:", np.round(sv, 6))
print("    dim null(Mt) =", null_dim, "  (card transmits only 4 of these)")
_, _, vh = np.linalg.svd(stack)
N = vh[16 - null_dim:]                            # rows = orthonormal basis of the null space
for name, idx in IDX.items():
    ov = np.linalg.norm(N[:, idx]) ** 2
    print("    null-space mass on %-5s (dim %d): %.12f" % (name, len(idx), ov))

# ---- (ii) budgets for many projectors ---------------------------------------------------------
def proj_from(idx):
    P = np.zeros((16, 16))
    for i in idx:
        P[i, i] = 1.0
    return P


def proj_from_vectors(V):
    Q, _ = np.linalg.qr(np.asarray(V).T)
    return Q @ Q.T


def budget(P, Mset=Mt):
    Qc = np.eye(16) - P
    c0 = float(sum(np.trace(m @ m) for m in Mset))
    c1 = float(sum(np.trace(m @ P @ m @ P) for m in Mset))
    c2 = float(sum(np.trace(m @ P @ m @ Qc) for m in Mset))
    c3 = float(sum(np.trace(m @ Qc @ m @ Qc) for m in Mset))
    return c0, c1, c2, c3


def rho128(c0, c1, c2, c3):
    return float(np.sqrt(max((c1 * E_D_128 + 2 * c2 * E_TRHK_128 + c3 * 127.0) / (c0 * 127.0), 0.0)))


def folded_visible(P):
    """visible dimensions that are NOT transmitted = rank dPhi - rank dPhi(P)"""
    V = np.asarray([dphi(sum(P[i, j] * basis[j] for j in range(16))) for i in range(16)]).reshape(16, -1)
    return 13 - int(np.linalg.matrix_rank(V, tol=1e-8))


rng = np.random.default_rng(20260902)
qr4, _ = np.linalg.qr(rng.normal(size=(16, 4)))
P_rand4_card = qr4 @ qr4.T
rng2 = np.random.default_rng(7)
inside = rng2.normal(size=(4, null_dim)) @ N          # random 4-plane INSIDE null(Mt)
P_in_null = proj_from_vectors(inside)

cases = {
    "P_align  = scale+sd (CARD, orbit tangent)": proj_from(IDX["scale"] + IDX["sd"]),
    "P_alt1   = scale+asd (4-dim, NOT tangent)": proj_from(IDX["scale"] + IDX["asd"]),
    "P_alt2   = sd+asd_1  (4-dim, NOT tangent)": proj_from(IDX["sd"] + IDX["asd"][:1]),
    "P_alt3   = asd+sym?  (4-dim, asd + 1 sym)": proj_from(IDX["asd"] + IDX["sym"][:1]),
    "P_alt4   = random 4-plane INSIDE null(Mt)": P_in_null,
    "P_scale  = scale only (1-dim)":             proj_from(IDX["scale"]),
    "P_sd     = sd only (3-dim, pure gauge)":    proj_from(IDX["sd"]),
    "P_asd    = asd only (3-dim, invisible)":    proj_from(IDX["asd"]),
    "P_null7  = whole null space (7-dim)":       proj_from(IDX["scale"] + IDX["sd"] + IDX["asd"]),
    "P_rand4  = card seed 20260902 (K2)":        P_rand4_card,
    "P_sym4   = 4 sym-traceless directions":     proj_from(IDX["sym"][:4]),
    "P_zero   = 0 (F-02 iid limit)":             np.zeros((16, 16)),
    "P_I      = identity (F-02 heritable limit)": np.eye(16),
}
print("\n[2] budgets (Mt = polar-aligned convention)")
print("    %-44s %8s %10s %10s %10s %8s %6s" % ("projector", "c1/c0", "2c2/c0", "c3/c0", "rho(128)", "gap", "fold"))
for name, P in cases.items():
    c0, c1, c2, c3 = budget(P)
    print("    %-44s %8.6f %10.6f %10.6f %10.4f %8.1e %6d"
          % (name, c1 / c0, 2 * c2 / c0, c3 / c0, rho128(c0, c1, c2, c3),
             abs(c0 - (c1 + 2 * c2 + c3)), folded_visible(P)))

# ---- (iii) convention dependence: same P_align but WITHOUT the polar-alignment stripping -------
print("\n[3] same projectors with the RAW M (no per-cell polar SO(3) alignment in the pipeline)")
for name in ("P_align  = scale+sd (CARD, orbit tangent)", "P_alt1   = scale+asd (4-dim, NOT tangent)",
             "P_scale  = scale only (1-dim)", "P_asd    = asd only (3-dim, invisible)"):
    c0, c1, c2, c3 = budget(cases[name], Mset=M)
    print("    %-44s c1/c0=%.6f  rho(128)=%.4f" % (name, c1 / c0, rho128(c0, c1, c2, c3)))

# ---- (iv) which single direction is doing the non-trivial work? --------------------------------
print("\n[4] per-direction leak ||Mt_a e_i|| (how much each label direction feeds the residual)")
for name, idx in IDX.items():
    leaks = [float(np.sqrt(sum(np.linalg.norm(m[:, i]) ** 2 for m in Mt))) for i in idx]
    leaksM = [float(np.sqrt(sum(np.linalg.norm(m[:, i]) ** 2 for m in M))) for i in idx]
    print("    %-5s Mt: %s   raw M: %s" % (name, np.round(leaks, 12), np.round(leaksM, 6)))
