"""adv_k1.py -- adversary checks for Q-0002 attempt-01 candidate K1 (numpy only).

Independent re-implementation (does not import check_k1). Generic N1, N2, eps (any length),
kernel mode "exact" (ker H, as in the derivation) or "modN" (Z_N group averaging, k1+k2+eps = 0 mod N).
SEED=20260902. Writes adv_result.json next to this file.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

SEED = 20260902
TOL = 1e-10
SEP = 1e-3


def ks(N):
    M = (N - 1) // 2
    return np.arange(-M, M + 1) if N % 2 == 1 else np.arange(-N // 2, N // 2)


def opnorm(X):
    return float(np.linalg.norm(X, 2))


class Model:
    """H_kin = C^{N1} x C^{N2} x C^{d}, index ((i1*N2)+i2)*d + s."""

    def __init__(self, N1, N2, eps, kernel="exact"):
        self.N1, self.N2, self.eps, self.kernel = N1, N2, np.array(eps, float), kernel
        self.d = len(eps)
        self.K1, self.K2 = ks(N1), ks(N2)
        D = N1 * N2 * self.d
        self.D = D
        diag = np.zeros(D)
        labels = []
        for i1, k1 in enumerate(self.K1):
            for i2, k2 in enumerate(self.K2):
                for s in range(self.d):
                    diag[(i1 * N2 + i2) * self.d + s] = k1 + k2 + self.eps[s]
                    labels.append((int(k1), int(k2), s))
        self.Hdiag = diag
        if kernel == "exact":
            phys = np.where(np.abs(diag) < 1e-12)[0]
        elif kernel == "modN":
            assert N1 == N2
            r = np.mod(diag, N1)
            phys = np.where((np.abs(r) < 1e-12) | (np.abs(r - N1) < 1e-12))[0]
        else:
            raise ValueError(kernel)
        self.phys_idx = phys
        self.labels = [labels[i] for i in phys]
        self.B = np.eye(D, dtype=complex)[:, phys]
        self.dim = len(phys)
        self.Pi = self.B @ self.B.conj().T

    def clock(self, i, tau):
        K = self.K1 if i == 1 else self.K2
        N = self.N1 if i == 1 else self.N2
        return np.exp(-1j * K * tau) / np.sqrt(N)

    def R(self, i, tau=0.0):
        bra = self.clock(i, tau).conj()[None, :]
        Id = np.eye(self.d)
        if i == 1:
            return np.kron(np.kron(bra, np.eye(self.N2)), Id)
        return np.kron(np.kron(np.eye(self.N1), bra), Id)

    def V(self, i, tau=0.0):
        N = self.N1 if i == 1 else self.N2
        return np.sqrt(N) * self.R(i, tau) @ self.B

    def O(self, i, A, tau=0.0):
        """N Pi (|tau><tau|_i x A) Pi, A on (other clock) x S."""
        ket = self.clock(i, tau)[:, None]
        proj = ket @ ket.conj().T
        N = self.N1 if i == 1 else self.N2
        if i == 1:
            K = np.kron(proj, A)
        else:
            A4 = A.reshape(self.N1, self.d, self.N1, self.d)
            K = np.einsum("asbt,kl->aksblt", A4, proj).reshape(self.D, self.D)
        return N * (self.Pi @ K @ self.Pi)

    def rand_phys(self, rng):
        c = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
        return self.B @ (c / np.linalg.norm(c))


def herm(rng, n):
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (X + X.conj().T) / 2


SX = np.array([[0, 1], [1, 0]], complex)


def main():
    out = {}
    rng = np.random.default_rng(SEED)

    # A. overlap failure: eps=(0,5), N=5: s=1 sector absent, both observables vanish
    m = Model(5, 5, (0, 5))
    O1 = m.O(1, np.kron(np.eye(5), SX)); O2 = m.O(2, np.kron(np.eye(5), SX))
    out["A_eps05_N5"] = dict(phys_dim=m.dim, sectors=sorted(set(l[2] for l in m.labels)),
                             norm_O1=opnorm(O1), norm_O2=opnorm(O2), diff=opnorm(O1 - O2),
                             commutator=opnorm(SX @ np.diag([0, 5]) - np.diag([0, 5]) @ SX),
                             claim_i_forward_holds=bool(opnorm(O1 - O2) > SEP))

    # B. non-integer eps: eps=(0,0.5): sector drops; check_k1 kernel test |diag|<0.5 with eps=(0,0.4)
    m = Model(5, 5, (0, 0.5))
    O1 = m.O(1, np.kron(np.eye(5), SX)); O2 = m.O(2, np.kron(np.eye(5), SX))
    out["B_eps_half"] = dict(phys_dim=m.dim, diff=opnorm(O1 - O2), claim_i_forward_holds=bool(opnorm(O1 - O2) > SEP))
    m4 = Model(5, 5, (0, 0.4))
    fake_phys = np.where(np.abs(m4.Hdiag) < 0.5)[0]
    out["B_check_k1_kernel_threshold_bug"] = dict(true_ker_dim=m4.dim, check_k1_would_count=int(len(fake_phys)),
                                                  max_H_on_fake_phys=float(np.max(np.abs(m4.Hdiag[fake_phys]))))

    # C. boundary artifact: spectrum of O1_{1 x sigma_x} on H_phys, eps=(0,1)
    res_C = {}
    for N in (5, 9, 17, 33):
        m = Model(N, N, (0, 1))
        O1 = m.O(1, np.kron(np.eye(N), SX))
        Ored = m.B.conj().T @ O1 @ m.B
        ev = np.linalg.eigvalsh(Ored)
        nzero = int(np.sum(np.abs(ev) < 1e-9))
        res_C[N] = dict(phys_dim=m.dim, n_zero_eigs=nzero, frac=nzero / m.dim,
                        expected_ideal_clock_spectrum="{+1,-1} only")
    out["C_boundary_zero_eigen"] = res_C

    # D. N1 != N2 : isometry + covariance still hold (same-N is not needed)
    m = Model(5, 7, (0, 1))
    V1, V2 = m.V(1), m.V(2)
    Phi = V2 @ V1.conj().T
    iso = max(opnorm(V1.conj().T @ V1 - np.eye(m.dim)), opnorm(V2.conj().T @ V2 - np.eye(m.dim)))
    A = rng.normal(size=(7 * 2, 7 * 2)) + 1j * rng.normal(size=(7 * 2, 7 * 2))
    cov = opnorm(m.O(1, A) - m.O(2, Phi @ A @ Phi.conj().T))
    out["D_N1_5_N2_7"] = dict(phys_dim=m.dim, iso_res=iso, cov_res=cov, shapes=[list(V1.shape), list(V2.shape)])

    # E. Z_N group-averaging projector (mod N) instead of exact kernel
    me = Model(5, 5, (0, 1), kernel="exact"); mm = Model(5, 5, (0, 1), kernel="modN")
    Phi_e = me.V(2) @ me.V(1).conj().T; Phi_m = mm.V(2) @ mm.V(1).conj().T
    v = np.zeros(10, complex); v[(2 + 2) * 2 + 1] = 1  # |k2=2, s=1>
    tgt = int(np.argmax(np.abs(Phi_m @ v)))
    out["E_modN_vs_exact"] = dict(dim_exact=me.dim, dim_modN=mm.dim,
                                  Phi_exact_on_k2eq2_s1=float(np.linalg.norm(Phi_e @ v)),
                                  Phi_modN_on_k2eq2_s1_target_k1_s=[tgt // 2 - 2, tgt % 2],
                                  Phi_diff=opnorm(Phi_e - Phi_m),
                                  Phi_modN_unitary_full=opnorm(Phi_m.conj().T @ Phi_m - np.eye(10)))
    O1 = mm.O(1, np.kron(np.eye(5), SX)); O2 = mm.O(2, np.kron(np.eye(5), SX))
    out["E_modN_claim_i_eps01"] = dict(diff=opnorm(O1 - O2))
    mm5 = Model(5, 5, (0, 5), kernel="modN")
    O1 = mm5.O(1, np.kron(np.eye(5), SX)); O2 = mm5.O(2, np.kron(np.eye(5), SX))
    out["E_modN_claim_i_eps05_wrap"] = dict(phys_dim=mm5.dim, diff=opnorm(O1 - O2), note="eps differ by N: wrap makes them degenerate; forward (i) fails")

    # F. random_sample_20, N=7, eps=(1,-2), seed 20260902: (i) and (ii) survive?
    m = Model(7, 7, (1, -2))
    V1, V2 = m.V(1), m.V(2); Phi = V2 @ V1.conj().T
    worst_cov = 0.0; worst_agree = 0.0; min_diff = 1e9
    for _ in range(20):
        A = rng.normal(size=(14, 14)) + 1j * rng.normal(size=(14, 14))
        worst_cov = max(worst_cov, opnorm(m.O(1, A) - m.O(2, Phi @ A @ Phi.conj().T)))
        OS = herm(rng, 2)
        min_diff = min(min_diff, opnorm(m.O(1, np.kron(np.eye(7), OS)) - m.O(2, np.kron(np.eye(7), OS))))
        OD = np.diag(rng.normal(size=2)).astype(complex)
        worst_agree = max(worst_agree, opnorm(m.O(1, np.kron(np.eye(7), OD)) - m.O(2, np.kron(np.eye(7), OD))))
    out["F_random20_N7_eps1m2"] = dict(phys_dim=m.dim, cov_max=worst_cov, commuting_agree_max=worst_agree,
                                       noncommuting_min_diff=min_diff,
                                       iso=max(opnorm(V1.conj().T @ V1 - np.eye(m.dim)), opnorm(V2.conj().T @ V2 - np.eye(m.dim))))

    # G. tau1 != tau2 != 0 : Phi(tau1,tau2) still partial isometry and covariance holds
    m = Model(5, 5, (0, 1))
    t1, t2 = 0.7, 2 * np.pi * 3 / 5
    V1, V2 = m.V(1, t1), m.V(2, t2); Phi = V2 @ V1.conj().T
    A = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    out["G_tau_nonzero"] = dict(iso=max(opnorm(V1.conj().T @ V1 - np.eye(m.dim)), opnorm(V2.conj().T @ V2 - np.eye(m.dim))),
                                cov=opnorm(m.O(1, A, t1) - m.O(2, Phi @ A @ Phi.conj().T, t2)),
                                Phi_vs_tau0=opnorm(Phi - m.V(2) @ m.V(1).conj().T))

    # H. N -> infinity: omega N-independent, conditioned value ~ 1/N
    res_H = {}
    for N in (5, 9, 17, 33):
        m = Model(N, N, (0, 1))
        a, b = 0.6, 0.8j
        psi = np.zeros(m.D, complex)
        M = (N - 1) // 2
        psi[((0 + M) * N + (0 + M)) * 2 + 0] = a
        psi[((-1 + M) * N + (0 + M)) * 2 + 1] = b
        O1 = m.O(1, np.kron(np.eye(N), SX))
        r1 = m.R(1) @ psi
        res_H[N] = dict(omega=float((psi.conj() @ O1 @ psi).real),
                        conditioned=float((r1.conj() @ np.kron(np.eye(N), SX) @ r1).real),
                        norm_conditioned=float(np.linalg.norm(r1)))
    out["H_N_scaling"] = res_H

    # I. tautology check: covariance identity holds for ANY pair of isometries
    m = Model(5, 5, (0, 1))
    X1 = rng.normal(size=(10, m.dim)) + 1j * rng.normal(size=(10, m.dim)); W1, _ = np.linalg.qr(X1)
    X2 = rng.normal(size=(10, m.dim)) + 1j * rng.normal(size=(10, m.dim)); W2, _ = np.linalg.qr(X2)
    Phi = W2 @ W1.conj().T
    A = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    out["I_covariance_tautology_random_isometries"] = dict(
        res=opnorm(W1.conj().T @ A @ W1 - W2.conj().T @ (Phi @ A @ Phi.conj().T) @ W2))

    # J. 3-level S, eps=(0,1,2), N=5: forward (i) for O_S with only (0,2) element
    m = Model(5, 5, (0, 1, 2))
    OS = np.zeros((3, 3), complex); OS[0, 2] = OS[2, 0] = 1
    O1 = m.O(1, np.kron(np.eye(5), OS)); O2 = m.O(2, np.kron(np.eye(5), OS))
    out["J_3level"] = dict(phys_dim=m.dim, diff=opnorm(O1 - O2))

    text = json.dumps(out, indent=2, default=str)
    print(text)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "adv_result.json"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
