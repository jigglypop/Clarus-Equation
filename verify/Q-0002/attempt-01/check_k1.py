"""check_k1.py -- Q-0002 attempt-01, candidate K1 numeric checks (numpy only).

Pre-declared before running: SEED=20260902, TOL=1e-10 (agreement), SEP=1e-3 (separation).
Prints a JSON {"claim":"K1","checks":{...},"all_pass":bool} to stdout and writes result.json
next to this script.

Model: clocks C_i=C^N with H_i=diag(-M..M), N=2M+1; S=qubit with H_S=diag(eps0,eps1);
constraint H=H_1+H_2+H_S; H_phys=ker H; reduction R_i Psi = <tau=0|_i Psi;
relational observable O^{(i)}_A = N Pi (|0><0|_i (x) A) Pi;
Phi_12 = (sqrt N R_2)(sqrt N R_1)^{-1} : V_1 -> V_2  (energy basis |k2,s> -> |-k2-eps_s, s>).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SEED = 20260902
TOL = 1e-10
SEP = 1e-3
N = 5
M = (N - 1) // 2
KS = np.arange(-M, M + 1)
I2 = np.eye(2, dtype=complex)
IN = np.eye(N, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def clock_state(tau: float) -> np.ndarray:
    """Covariant clock state |tau> = N^{-1/2} sum_k e^{-ik tau}|k>."""
    return np.exp(-1j * KS * tau) / np.sqrt(N)


def opnorm(X: np.ndarray) -> float:
    return float(np.linalg.norm(X, 2))


class Model:
    """Kinematic space C^N (x) C^N (x) C^2 with index ((k1+M)*N + (k2+M))*2 + s."""

    def __init__(self, eps: tuple[int, int]) -> None:
        self.eps = np.array(eps, dtype=float)
        D = N * N * 2
        self.D = D
        diag = np.zeros(D)
        labels = []
        for i1, k1 in enumerate(KS):
            for i2, k2 in enumerate(KS):
                for s in range(2):
                    idx = (i1 * N + i2) * 2 + s
                    diag[idx] = k1 + k2 + self.eps[s]
                    labels.append((int(k1), int(k2), s))
        phys = np.where(np.abs(diag) < 0.5)[0]  # integer spectrum: exact zero
        self.phys_idx = phys
        self.labels = [labels[i] for i in phys]
        self.B = np.eye(D, dtype=complex)[:, phys]  # D x d, orthonormal columns
        self.d = len(phys)
        self.Pi = self.B @ self.B.conj().T
        # H_2 + H_S on C_2 (x) S, index (k2+M)*2+s
        self.H2S = np.kron(np.diag(KS).astype(complex), I2) + np.kron(IN, np.diag(self.eps).astype(complex))

    def index_full(self, k1: int, k2: int, s: int) -> int:
        return ((k1 + M) * N + (k2 + M)) * 2 + s

    @staticmethod
    def index_red(k: int, s: int) -> int:
        return (k + M) * 2 + s

    def R(self, i: int, tau: float = 0.0) -> np.ndarray:
        """Reduction map <tau|_i as a (2N x D) matrix."""
        bra = clock_state(tau).conj()[None, :]
        if i == 1:
            return np.kron(np.kron(bra, IN), I2)
        return np.kron(np.kron(IN, bra), I2)

    def O(self, i: int, A: np.ndarray) -> np.ndarray:
        """Relational observable N Pi (|0><0|_i (x) A) Pi as a full D x D matrix."""
        ket = clock_state(0.0)[:, None]
        proj = ket @ ket.conj().T
        if i == 1:
            K = np.kron(proj, A)
        else:
            A4 = A.reshape(N, 2, N, 2)
            K = np.einsum("asbt,kl->aksblt", A4, proj).reshape(self.D, self.D)
        return N * (self.Pi @ K @ self.Pi)

    def omega(self, psi: np.ndarray, Oi: np.ndarray) -> complex:
        return complex(psi.conj() @ Oi @ psi)

    def random_phys_state(self, rng: np.random.Generator) -> np.ndarray:
        c = rng.normal(size=self.d) + 1j * rng.normal(size=self.d)
        c /= np.linalg.norm(c)
        return self.B @ c

    def isometries(self) -> tuple[np.ndarray, np.ndarray]:
        V1 = np.sqrt(N) * self.R(1) @ self.B
        V2 = np.sqrt(N) * self.R(2) @ self.B
        return V1, V2

    def phi(self) -> np.ndarray:
        V1, V2 = self.isometries()
        return V2 @ V1.conj().T

    def phi_expected(self) -> np.ndarray:
        Phi = np.zeros((2 * N, 2 * N), dtype=complex)
        for k2 in KS:
            for s in range(2):
                k1 = -int(k2) - int(self.eps[s])
                if -M <= k1 <= M:
                    Phi[self.index_red(k1, s), self.index_red(int(k2), s)] = 1.0
        return Phi


def random_hermitian(rng: np.random.Generator, n: int) -> np.ndarray:
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (X + X.conj().T) / 2


def random_complex(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))


def main() -> int:
    checks: dict[str, dict] = {}

    def rec(name: str, ok: bool, **info) -> None:
        clean = {}
        for k, v in info.items():
            if isinstance(v, (np.floating, float)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            else:
                clean[k] = v
        checks[name] = {"pass": bool(ok), **clean}

    rng = np.random.default_rng(SEED)
    m = Model((0, 1))
    rec("phys_dim_eps01", m.d == 9, value=m.d, expected=9)

    # random draws in fixed order
    ab = rng.normal(size=2) + 1j * rng.normal(size=2)
    ab /= np.linalg.norm(ab)
    a, b = ab
    states = [m.random_phys_state(rng) for _ in range(20)]
    A_S_list = [random_complex(rng, 2) for _ in range(20)]
    A_full_list = [random_complex(rng, 2 * N) for _ in range(20)]
    O_S_herm = [random_hermitian(rng, 2) for _ in range(20)]
    O_S_diag = [np.diag(rng.normal(size=2)).astype(complex) for _ in range(20)]

    V1, V2 = m.isometries()
    iso_res = max(opnorm(V1.conj().T @ V1 - np.eye(m.d)), opnorm(V2.conj().T @ V2 - np.eye(m.d)))
    rec("S2_sqrtN_R_i_isometry_on_Hphys", iso_res < TOL, residual=iso_res, tol=TOL)

    # S3: <tau|_1 Psi = exp(-i(H2+HS)tau) <0|_1 Psi
    worst = 0.0
    R10 = m.R(1, 0.0)
    for psi in states:
        for n in range(N):
            tau = 2 * np.pi * n / N
            lhs = m.R(1, tau) @ psi
            rhs = np.diag(np.exp(-1j * np.diag(m.H2S) * tau)) @ (R10 @ psi)
            worst = max(worst, float(np.linalg.norm(lhs - rhs)))
    rec("S3_reduced_state_evolves_with_H2_plus_HS", worst < TOL, residual=worst, tol=TOL)

    # S4: sigma_x differs, sigma_z agrees (20 random physical states)
    O1x, O2x = m.O(1, np.kron(IN, SX)), m.O(2, np.kron(IN, SX))
    O1z, O2z = m.O(1, np.kron(IN, SZ)), m.O(2, np.kron(IN, SZ))
    dx = [abs(m.omega(p, O1x) - m.omega(p, O2x)) for p in states]
    dz = [abs(m.omega(p, O1z) - m.omega(p, O2z)) for p in states]
    rec("S4_sigma_x_observers_differ_20_random_states", min(dx) > SEP, min_diff=min(dx), sep=SEP)
    rec("S4_sigma_z_observers_agree_20_random_states", max(dz) < TOL, max_diff=max(dz), tol=TOL)
    imag_worst = max(abs(m.omega(p, O1x).imag) for p in states)
    rec("S4_omega_is_real", imag_worst < TOL, max_imag=imag_worst, tol=TOL)

    # S4 hand calculation: Psi = a|k1,k2,0> + b|k1',k2,1>, k2=0, k1=-eps0=0, k1'=-eps1=-1
    k2 = 0
    k1 = -k2 - int(m.eps[0])
    k1p = -k2 - int(m.eps[1])
    assert k1p == k1 + int(m.eps[0]) - int(m.eps[1])
    psi_ab = np.zeros(m.D, dtype=complex)
    psi_ab[m.index_full(k1, k2, 0)] = a
    psi_ab[m.index_full(k1p, k2, 1)] = b
    assert np.linalg.norm(m.Pi @ psi_ab - psi_ab) < TOL
    hand = 2 * (np.conj(a) * b).real
    w1x, w2x = m.omega(psi_ab, O1x), m.omega(psi_ab, O2x)
    w1z, w2z = m.omega(psi_ab, O1z), m.omega(psi_ab, O2z)
    r1 = m.R(1) @ psi_ab
    r2 = m.R(2) @ psi_ab
    cond1 = complex(r1.conj() @ np.kron(IN, SX) @ r1)
    cond2 = complex(r2.conj() @ np.kron(IN, SX) @ r2)
    rec("S4_ab_state_sigma_x_differs", abs(w1x - w2x) > SEP, diff=abs(w1x - w2x), sep=SEP)
    rec("S4_ab_state_sigma_z_agrees", abs(w1z - w2z) < TOL, diff=abs(w1z - w2z), tol=TOL)
    rec("S4_hand_obs1_conditioned_equals_2Re(conj(a)b)/N", abs(cond1 - hand / N) < TOL,
        value=cond1.real, expected=hand / N, tol=TOL)
    rec("S4_hand_obs1_omega_equals_2Re(conj(a)b)", abs(w1x - hand) < TOL, value=w1x.real, expected=hand, tol=TOL)
    rec("S4_hand_obs2_conditioned_zero", abs(cond2) < TOL, value=abs(cond2), tol=TOL)
    rec("S4_hand_obs2_omega_zero", abs(w2x) < TOL, value=abs(w2x), tol=TOL)

    # S5: Phi unitary V1 -> V2 and covariance identity
    Phi = m.phi()
    P1, P2 = V1 @ V1.conj().T, V2 @ V2.conj().T
    rec("S5_Phi_matches_energy_basis_map", opnorm(Phi - m.phi_expected()) < TOL,
        residual=opnorm(Phi - m.phi_expected()), tol=TOL)
    rec("S5_PhiDag_Phi_eq_P_V1", opnorm(Phi.conj().T @ Phi - P1) < TOL, residual=opnorm(Phi.conj().T @ Phi - P1), tol=TOL)
    rec("S5_Phi_PhiDag_eq_P_V2", opnorm(Phi @ Phi.conj().T - P2) < TOL, residual=opnorm(Phi @ Phi.conj().T - P2), tol=TOL)
    rank_P1 = int(round(np.trace(P1).real))
    rec("S5_rank_P_V1_equals_phys_dim", rank_P1 == m.d, value=rank_P1, expected=m.d)
    # literal candidate formula N R_2 R_1^{-1} carries an extra factor N (S5.2)
    literal = N * (m.R(2) @ m.B) @ np.linalg.pinv(m.R(1) @ m.B)
    rec("S5_literal_N_R2_R1inv_equals_N_times_Phi", opnorm(literal - N * Phi) < TOL,
        residual=opnorm(literal - N * Phi), note="candidate text has normalisation slip; correct Phi = R2 R1^{-1}")
    # state transport
    wt = max(float(np.linalg.norm(np.sqrt(N) * m.R(2) @ p - Phi @ (np.sqrt(N) * m.R(1) @ p))) for p in states)
    rec("S5_Phi_transports_conditioned_states", wt < TOL, residual=wt, tol=TOL)
    # operator identity O1_A = O2_{Phi A Phi^dag}
    res_S = [opnorm(m.O(1, np.kron(IN, AS)) - m.O(2, Phi @ np.kron(IN, AS) @ Phi.conj().T)) for AS in A_S_list]
    res_F = [opnorm(m.O(1, A) - m.O(2, Phi @ A @ Phi.conj().T)) for A in A_full_list]
    rec("S5_covariance_operator_identity_20_A_S", max(res_S) < TOL, max_residual=max(res_S), tol=TOL)
    rec("S5_covariance_operator_identity_20_A_full", max(res_F) < TOL, max_residual=max(res_F), tol=TOL)
    ws = 0.0
    for A in A_full_list:
        O1A, O2A = m.O(1, A), m.O(2, Phi @ A @ Phi.conj().T)
        for p in states:
            ws = max(ws, abs(m.omega(p, O1A) - m.omega(p, O2A)))
    rec("S5_covariance_expectation_20x20", ws < TOL, max_residual=ws, tol=TOL)

    # claim (i) general: commuting O_S agree; generic Hermitian O_S disagree
    res_c = [opnorm(m.O(1, np.kron(IN, OS)) - m.O(2, np.kron(IN, OS))) for OS in O_S_diag]
    res_n = []
    comm = []
    for OS in O_S_herm:
        res_n.append(opnorm(m.O(1, np.kron(IN, OS)) - m.O(2, np.kron(IN, OS))))
        comm.append(opnorm(OS @ np.diag(m.eps) - np.diag(m.eps) @ OS))
    rec("S4_i_commuting_OS_agree_20", max(res_c) < TOL, max_residual=max(res_c), tol=TOL)
    rec("S4_i_noncommuting_OS_differ_20", min(res_n) > SEP and min(comm) > SEP,
        min_residual=min(res_n), min_commutator_norm=min(comm), sep=SEP)

    # negative control 1: eps=(0,0) -> sigma_x also agrees
    rng0 = np.random.default_rng(SEED)
    m0 = Model((0, 0))
    ab0 = rng0.normal(size=2) + 1j * rng0.normal(size=2)
    ab0 /= np.linalg.norm(ab0)
    states0 = [m0.random_phys_state(rng0) for _ in range(20)]
    O1x0, O2x0 = m0.O(1, np.kron(IN, SX)), m0.O(2, np.kron(IN, SX))
    dx0 = [abs(m0.omega(p, O1x0) - m0.omega(p, O2x0)) for p in states0]
    psi0 = np.zeros(m0.D, dtype=complex)
    psi0[m0.index_full(0, 0, 0)] = ab0[0]
    psi0[m0.index_full(0, 0, 1)] = ab0[1]
    hand0 = 2 * (np.conj(ab0[0]) * ab0[1]).real
    w10, w20 = m0.omega(psi0, O1x0), m0.omega(psi0, O2x0)
    rec("NEG_eps00_sigma_x_agree_20_random_states", max(dx0) < TOL, max_diff=max(dx0), tol=TOL, phys_dim=m0.d)
    rec("NEG_eps00_ab_state_both_observers_2Re(conj(a)b)",
        abs(w10 - hand0) < TOL and abs(w20 - hand0) < TOL, w1=w10.real, w2=w20.real, expected=hand0, tol=TOL)

    # negative control 2: replace Phi by identity -> covariance fails (eps=(0,1))
    res_id_x = opnorm(O1x - O2x)
    res_id_S = [opnorm(m.O(1, np.kron(IN, AS)) - m.O(2, np.kron(IN, AS))) for AS in A_S_list]
    rec("NEG_Phi_identity_fails_sigma_x", res_id_x > SEP, residual=res_id_x, sep=SEP)
    rec("NEG_Phi_identity_fails_20_random_A_S", min(res_id_S) > SEP, min_residual=min(res_id_S), sep=SEP)

    out = {
        "claim": "K1",
        "question": "Q-0002",
        "attempt": 1,
        "seed": SEED,
        "tol": TOL,
        "sep": SEP,
        "N": N,
        "eps": [0, 1],
        "a": [float(a.real), float(a.imag)],
        "b": [float(b.real), float(b.imag)],
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks.values()),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "result.json"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
