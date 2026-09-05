"""Adversary a3 (kill_executable / content): is the K2 control value 7.299 a property of
'a random 4-dim projector', or of THIS random 4-dim projector (seed 20260902)?

Draws N Haar-random 4-planes (QR of 16x4 gaussian), computes the exact budget and the predicted
rho(128), and asks: where does the card's seed sit, how wide is the distribution, and how often
would a random 4-plane land inside the card's own K2 window [5.8, 8.8] / K1 window [0.85, 1.18]?
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram  # noqa: E402

I4 = np.eye(4)
PHI = lambda e: geometric_self_dual_triple(e)
E_D_128, E_TRHK_128 = 134587.1450461609, 822.7392437084596


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
basis += [(anti(0, i) + anti(j, k)) / np.sqrt(2.0) for i, j, k in cyc]
basis += [(anti(0, i) - anti(j, k)) / np.sqrt(2.0) for i, j, k in cyc]
for mu in range(4):
    for nu in range(mu + 1, 4):
        m = np.zeros((4, 4)); m[mu, nu] = m[nu, mu] = 1 / np.sqrt(2.0); basis.append(m)
for m in (np.diag([1., -1., 0., 0.]) / np.sqrt(2), np.diag([1., 1., -2., 0.]) / np.sqrt(6),
          np.diag([1., 1., 1., -3.]) / np.sqrt(12)):
    basis.append(m)
basis = np.asarray(basis)
images = np.asarray([dphi(b) for b in basis])
Ea = []
for i in range(3):
    for j in range(i + 1, 3):
        e = np.zeros((3, 3)); e[i, j] = e[j, i] = 1 / np.sqrt(2.0); Ea.append(e)
Ea.append(np.diag([1., -1., 0.]) / np.sqrt(2)); Ea.append(np.diag([1., 1., -2.]) / np.sqrt(6))
M = np.zeros((5, 16, 16))
for a, e in enumerate(Ea):
    for p in range(16):
        for q in range(16):
            M[a, p, q] = float(np.sum(e * tl(G(images[p], images[q]))))
Pi_rot = np.zeros((16, 16))
for i in (1, 2, 3):
    Pi_rot[i, i] = 1.0
Mt = np.asarray([(np.eye(16) - Pi_rot) @ m @ (np.eye(16) - Pi_rot) for m in M])
C0 = float(sum(np.trace(m @ m) for m in Mt))


def rho_of(P):
    Qc = np.eye(16) - P
    c1 = float(sum(np.trace(m @ P @ m @ P) for m in Mt))
    c2 = float(sum(np.trace(m @ P @ m @ Qc) for m in Mt))
    c3 = float(sum(np.trace(m @ Qc @ m @ Qc) for m in Mt))
    return c1 / C0, float(np.sqrt(max((c1 * E_D_128 + 2 * c2 * E_TRHK_128 + c3 * 127.0) / (C0 * 127.0), 0.0)))


card_rng = np.random.default_rng(20260902)
q, _ = np.linalg.qr(card_rng.normal(size=(16, 4)))
card_c1, card_rho = rho_of(q @ q.T)
print("card seed 20260902:  c1/c0 = %.6f   rho(128) = %.4f" % (card_c1, card_rho))

N = 3000
rng = np.random.default_rng(999333)
c1s, rhos = [], []
for _ in range(N):
    qq, _ = np.linalg.qr(rng.normal(size=(16, 4)))
    a, b = rho_of(qq @ qq.T)
    c1s.append(a); rhos.append(b)
c1s = np.asarray(c1s); rhos = np.asarray(rhos)
qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
print("N =", N, " Haar 4-planes")
print("  c1/c0   quantiles", dict(zip(qs, np.round(np.percentile(c1s, qs), 5))))
print("  rho(128) quantiles", dict(zip(qs, np.round(np.percentile(rhos, qs), 3))))
print("  mean rho = %.3f  sd = %.3f   card seed percentile = %.1f%%"
      % (rhos.mean(), rhos.std(), 100.0 * float(np.mean(rhos < card_rho))))
print("  P[rho in K2 window (5.8, 8.8)]   = %.3f" % float(np.mean((rhos >= 5.8) & (rhos <= 8.8))))
print("  P[rho in K1 window (0.85, 1.18)] = %.3f" % float(np.mean((rhos >= 0.85) & (rhos <= 1.18))))
print("  P[rho > 8.8] = %.3f   P[rho < 5.8] = %.3f" % (float(np.mean(rhos > 8.8)), float(np.mean(rhos < 5.8))))
