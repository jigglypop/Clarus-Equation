"""adv_k1b.py -- supplementary adversary checks (H scaling redo, tau-unitary-difference, S5.7 boundary truncation)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adv_k1 import Model, SX, opnorm

def main():
    out = {}
    # H redo: a=0.6, b=0.8 -> 2Re(ab)=0.96 expected N-independent; conditioned = 0.96/N
    res = {}
    for N in (5, 9, 17, 33):
        m = Model(N, N, (0, 1)); M = (N - 1) // 2
        psi = np.zeros(m.D, complex)
        psi[((0 + M) * N + (0 + M)) * 2 + 0] = 0.6
        psi[((-1 + M) * N + (0 + M)) * 2 + 1] = 0.8
        O1 = m.O(1, np.kron(np.eye(N), SX)); r1 = m.R(1) @ psi
        res[N] = dict(omega=float((psi.conj() @ O1 @ psi).real),
                      conditioned=float((r1.conj() @ np.kron(np.eye(N), SX) @ r1).real),
                      conditioned_times_N=float(N * (r1.conj() @ np.kron(np.eye(N), SX) @ r1).real))
    out["H_N_scaling"] = res

    # G2: Phi(t1,t2) == exp(-i(H1+HS)t2) Phi(0,0) exp(+i(H2+HS)t1) ?  (S3.5 applied to both clocks)
    m = Model(5, 5, (0, 1)); N = 5; K = np.arange(-2, 3)
    t1, t2 = 0.7, 2 * np.pi * 3 / 5
    H2S = np.kron(np.diag(K), np.eye(2)) + np.kron(np.eye(N), np.diag(m.eps))
    U1 = np.diag(np.exp(-1j * np.diag(H2S) * t2))   # acts on C_1 x S (same matrix form)
    U2 = np.diag(np.exp(+1j * np.diag(H2S) * t1))
    Phi_t = m.V(2, t2) @ m.V(1, t1).conj().T
    Phi_0 = m.V(2) @ m.V(1).conj().T
    out["G2_tau_unitary_difference"] = dict(res=opnorm(Phi_t - U1 @ Phi_0 @ U2))

    # S5.7: Phi^dag (1 x O_S) Phi |k2,s> = sum_s' (O_S)_{s's} |k2+eps_s-eps_s', s'>  -- truncated at boundary?
    Phi = Phi_0
    X = Phi.conj().T @ np.kron(np.eye(N), SX) @ Phi
    # |k2=1, s=1> in V_1 (k1=-2). Target: |k2+1-0, s=0> = |2,0> in V_1 -> present
    # |k2=-2, s=0> in V_1 (k1=2). Target: |k2+0-1, s=1> = |-3,1> out of range -> annihilated
    v_in = np.zeros(10, complex); v_in[(1 + 2) * 2 + 1] = 1
    v_bd = np.zeros(10, complex); v_bd[(-2 + 2) * 2 + 0] = 1
    out["S5_7_boundary"] = dict(norm_image_interior=float(np.linalg.norm(X @ v_in)),
                                norm_image_boundary=float(np.linalg.norm(X @ v_bd)),
                                note="ideal clock would give 1 in both; boundary state is annihilated (finite-N artifact)")
    # sigma_x^2 on H_phys: relational observable of sigma_x squared vs relational observable of 1
    O1x = m.O(1, np.kron(np.eye(N), SX)); O1one = m.O(1, np.eye(2 * N))
    out["S2_algebra_not_homomorphism"] = dict(res=opnorm(O1x @ O1x - O1one),
        note="O_A O_B != O_AB on H_phys at boundary: A->O^{(1)}_A is not an algebra homomorphism at finite N")
    text = json.dumps(out, indent=2); print(text)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_result_b.json"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
