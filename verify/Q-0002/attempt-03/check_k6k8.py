"""check_k6k8.py -- Q-0002 attempt-03, candidates K6 + K8 (+ K9 negative lemma), numpy only.

Pre-declared before running: SEED=20260902, TOL=1e-10 (agreement), SEP=1e-3 (separation),
TOL_SPEC=1e-8 (eigenvalue-set membership, as requested by the task), N_STATES=20, N_PAIRS=20.
Prints JSON {"claim":"K6+K8","checks":{...},"all_pass":bool} to stdout and writes result.json
next to this script.

Model (clock-2 <-> flux exchange coupling, Z_N group averaging):
  clocks C_i = C^N, H_i = diag(0..N-1) (labels mod N); T_2 cyclic raising on clock 2;
  S = R (x) Q, register qubit R, flux qubit Q, n_Q = diag(0,1), sigma_-^Q = |0><1|_Q;
  H_S = E* n_Q + g sigma_x^R  (E*=1, g=2);
  C = H_1 + H_2 + H_S + lam (T_2 (x) sigma_-^Q + h.c.) = H_1 (x) 1 + 1 (x) H_rest;
  Pi = (1/N) sum_n exp(2 pi i n C / N)  (projector iff spec C integer);
  integer spectrum iff N/2 +- sqrt((N/2)^2 + lam^2) integer: (N,lam) in {(8,3),(6,4),(24,5)};
  V_i(tau) = sqrt(N) <tau|_i restricted to H_phys; O^{(i)}_A = V_i^dag A V_i; Q_2 = V_2 V_2^dag.

Physical basis is built explicitly: for each eigenvector |mu> of H_rest (eigenvalue mu), the
physical vector is |k_1 = (-mu) mod N> (x) |mu>.  Pi built from this basis is compared with the
group average (check *_Pi_equals_group_average).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

SEED = 20260902
TOL = 1e-10
SEP = 1e-3
TOL_SPEC = 1e-8
N_STATES = 20
N_PAIRS = 20

rng = np.random.default_rng(SEED)

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
NQ = np.diag([0.0, 1.0]).astype(complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma_-^Q = |0><1|
# S index s = 2 r + q  (kron(R, Q))
P_REG = [np.kron(np.diag([1.0, 0.0]).astype(complex), I2), np.kron(np.diag([0.0, 1.0]).astype(complex), I2)]


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def opn(X):
    return float(np.linalg.norm(X, 2))


def tau_ket(N, tau):
    """|tau> = N^{-1/2} sum_k e^{-ik tau} |k>."""
    return np.exp(-1j * np.arange(N) * tau) / np.sqrt(N)


class Model:
    def __init__(self, N, lam, Estar=1.0, g=2.0):
        self.N, self.lam = N, lam
        IN = np.eye(N, dtype=complex)
        self.IN = IN
        Hc = np.diag(np.arange(N).astype(complex))
        T = np.zeros((N, N), complex)
        for k in range(N - 1):
            T[k + 1, k] = 1
        T[0, N - 1] = 1
        self.HS = Estar * np.kron(I2, NQ) + g * np.kron(SX, I2)
        Hint = np.kron(T, np.kron(I2, SM))
        self.Hint = lam * (Hint + Hint.conj().T)
        self.H2f = np.kron(Hc, I4)
        self.HSf = np.kron(IN, self.HS)                       # 1_{C} (x) H_S on C_j (x) S
        self.Hrest = self.H2f + self.HSf + self.Hint          # on C_2 (x) S
        self.H1 = kron3(Hc, IN, I4)
        self.C = self.H1 + np.kron(IN, self.Hrest)
        self.D = 4 * N * N
        w, U = np.linalg.eigh(self.C)
        self.int_res = float(np.max(np.abs(w - np.round(w))))
        # group average of exp(2 pi i n C/N)
        Ug = U @ np.diag(np.exp(2j * np.pi * w / N)) @ U.conj().T
        G = np.zeros_like(Ug)
        P = np.eye(self.D, dtype=complex)
        for _ in range(N):
            G += P
            P = P @ Ug
        self.Pi_ga = G / N
        self.Pi_ga_idem_res = opn(self.Pi_ga @ self.Pi_ga - self.Pi_ga)
        # explicit labelled physical basis from H_rest eigenvectors
        mu, Wr = np.linalg.eigh(self.Hrest)
        self.mu = mu
        self.k1 = (-np.round(mu).astype(int)) % N
        cols = []
        for a in range(4 * N):
            e1 = np.zeros(N, complex)
            e1[self.k1[a]] = 1
            cols.append(np.kron(e1, Wr[:, a]))
        self.B = np.array(cols).T                             # D x 4N isometry
        self.d = 4 * N
        self.Pi = self.B @ self.B.conj().T
        self.Pi_vs_ga = opn(self.Pi - self.Pi_ga)
        self.Wr = Wr

    def V(self, i, tau=0.0):
        bra = tau_ket(self.N, tau).conj()[None, :]
        if i == 1:
            Rd = kron3(bra, self.IN, I4)
        else:
            Rd = kron3(self.IN, bra, I4)
        return np.sqrt(self.N) * Rd @ self.B                   # H_phys -> C_j (x) S

    def O(self, i, A, tau=0.0):
        V = self.V(i, tau)
        return V.conj().T @ A @ V

    def rand(self):
        c = rng.normal(size=self.d) + 1j * rng.normal(size=self.d)
        return c / np.linalg.norm(c)

    def Q2_closed_form(self, tau=0.0):
        """sum_n e^{2 pi i n H_1/N} (x) <tau| e^{2 pi i n H_rest/N} |tau>_2 on C_1 (x) S."""
        N = self.N
        kt = tau_ket(N, tau)
        Hc = np.diag(np.arange(N).astype(complex))
        mu, Wr = self.mu, self.Wr
        out = np.zeros((4 * N, 4 * N), complex)
        for n in range(N):
            U1 = np.diag(np.exp(2j * np.pi * n * np.arange(N) / N))
            Urest = Wr @ np.diag(np.exp(2j * np.pi * n * mu / N)) @ Wr.conj().T
            # partial matrix element <tau|_2 Urest |tau>_2 : (N*4 x N*4) -> (4 x 4)
            Ur4 = Urest.reshape(N, 4, N, 4)
            M = np.einsum('k,kalb,l->ab', kt.conj(), Ur4, kt)
            out += np.kron(U1, M)
        return out


def polarization_defect(gfun, m, pairs):
    worst = 0.0
    vals = []
    for _ in range(pairs):
        a, b = m.rand(), m.rand()
        v = abs(gfun(a + b) + gfun(a - b) - 2 * gfun(a) - 2 * gfun(b))
        vals.append(v)
        worst = max(worst, v)
    return worst, vals


def c_of(N, lam):
    phi = 0.5 * math.atan2(2 * lam, N)
    return math.sqrt((1 + math.sin(2 * phi)) / 2), math.sqrt((1 - math.sin(2 * phi)) / 2), phi


checks: dict[str, dict] = {}


def rec(name, value, ok, note=""):
    checks[name] = {"value": value, "pass": bool(ok), "note": note}


DECLARED_SPEC = {(8, 3): [0.0, 1.0, 2.0], (24, 5): [0.0, 1.0, 2.0]}
# (6,4): {1-c, 1, 1+c} with c = sqrt((1+sin 2phi)/2), tan 2phi = 2 lam/N

for (N, lam) in [(8, 3), (6, 4), (24, 5)]:
    tag = f"N{N}_lam{lam}"
    m = Model(N, lam)
    rec(f"{tag}_integer_spectrum", m.int_res, m.int_res < TOL)
    rec(f"{tag}_Pi_equals_group_average", m.Pi_vs_ga, m.Pi_vs_ga < TOL)
    rec(f"{tag}_phys_dim_equals_4N", m.d, m.d == 4 * N)
    HSk = kron3(m.IN, m.IN, m.HS)
    comm_norm = opn(HSk @ m.C - m.C @ HSk)
    rec(f"{tag}_norm_[HS,C]_equals_lam", comm_norm, abs(comm_norm - lam) < TOL)
    V1, V2 = m.V(1), m.V(2)
    rec(f"{tag}_V1_unitary", max(opn(V1.conj().T @ V1 - np.eye(m.d)), opn(V1 @ V1.conj().T - np.eye(4 * N))),
        max(opn(V1.conj().T @ V1 - np.eye(m.d)), opn(V1 @ V1.conj().T - np.eye(4 * N))) < TOL)
    rec(f"{tag}_V2_not_isometric", opn(V2.conj().T @ V2 - np.eye(m.d)), opn(V2.conj().T @ V2 - np.eye(m.d)) > SEP)

    # ---- K6 (i): spectrum of O^{(2)}_1 = V2^dag V2 on lattice tau
    c, cp, phi = c_of(N, lam)
    declared = DECLARED_SPEC.get((N, lam), [1 - c, 1.0, 1 + c])
    for tau_idx in (0, 1):
        tau = 2 * math.pi * tau_idx / N
        G = m.V(2, tau).conj().T @ m.V(2, tau)
        ev = np.linalg.eigvalsh(G)
        dist = np.array([min(abs(x - y) for y in declared) for x in ev])
        attained = [float(np.min(np.abs(ev - y))) for y in declared]
        rec(f"K6i_{tag}_tau{tau_idx}_V2dV2_spec_in_declared_set", {"declared": declared, "max_dist": float(dist.max()),
                                                                    "unique": sorted(set(np.round(ev, 6).tolist()))},
            dist.max() < TOL_SPEC)
        rec(f"K6i_{tag}_tau{tau_idx}_each_declared_value_attained", attained, max(attained) < TOL_SPEC)
        # block structure in k1 residue classes
        off = G.copy()
        for a in range(m.d):
            for b in range(m.d):
                if m.k1[a] == m.k1[b]:
                    off[a, b] = 0
        rec(f"K6i_{tag}_tau{tau_idx}_V2dV2_block_diagonal_in_k1", opn(off), opn(off) < TOL)
        # Gram of tau-slices w_mu = sqrt(N) <tau|_2 |mu> equals the k1 block
        kt = tau_ket(N, tau)
        w = np.sqrt(N) * np.einsum('k,kab->ab', kt.conj(), m.Wr.reshape(N, 4, 4 * N))   # 4 x 4N
        Gram = w.conj().T @ w
        same_k1 = (m.k1[:, None] == m.k1[None, :])
        Gram_blocks = np.where(same_k1, Gram, 0)        # <k1 mu|V2^dag V2|k1' mu'> = delta_{k1 k1'} <w_mu|w_mu'>
        rec(f"K6i_{tag}_tau{tau_idx}_block_equals_Gram_of_slices", opn(G - Gram_blocks), opn(G - Gram_blocks) < TOL)
    rec(f"K6i_{tag}_c_value", {"c": c, "c_prime": cp, "phi": phi}, True, "informational")

    # ---- K6 (ii): normalized ledger is not a quadratic form (lam != 0)
    HSf = m.HSf
    O2H, O21 = m.O(2, HSf), m.O(2, np.eye(4 * N))
    O1H = m.O(1, HSf)

    def g2(cv):
        return float(np.vdot(cv, cv).real * (cv.conj() @ O2H @ cv).real / (cv.conj() @ O21 @ cv).real)

    def g1(cv):
        return float((cv.conj() @ O1H @ cv).real)  # omega(O1_1) = |c|^2 exactly (V1 unitary)

    worst2, vals2 = polarization_defect(g2, m, N_PAIRS)
    worst1, _ = polarization_defect(g1, m, N_PAIRS)
    rec(f"K6ii_{tag}_E2_normalized_polarization_violation_max_gt_SEP",
        {"max": worst2, "min": float(min(vals2)), "n_pairs_gt_SEP": int(sum(v > SEP for v in vals2)), "n_pairs": N_PAIRS},
        worst2 > SEP)
    rec(f"K6ii_{tag}_E1_exactly_quadratic", worst1, worst1 < TOL)
    # E1 - E2 normalized is therefore not a quadratic form => no operator X with E1-E2 = omega(X)
    worst12, _ = polarization_defect(lambda cv: g1(cv) - g2(cv), m, N_PAIRS)
    rec(f"K6ii_{tag}_E1_minus_E2_polarization_violation_gt_SEP", worst12, worst12 > SEP)

    # ---- K8 (b): observer 2, lam != 0
    Q2 = V2 @ V2.conj().T
    Q2cf = m.Q2_closed_form(0.0)
    rec(f"K8b_{tag}_Q2_closed_form", opn(Q2 - Q2cf), opn(Q2 - Q2cf) < TOL)
    Q2cf1 = m.Q2_closed_form(2 * math.pi / N)
    V2t = m.V(2, 2 * math.pi / N)
    rec(f"K8b_{tag}_Q2_closed_form_tau1", opn(V2t @ V2t.conj().T - Q2cf1), opn(V2t @ V2t.conj().T - Q2cf1) < TOL)
    rec(f"K8b_{tag}_Q2_ne_1", opn(Q2 - np.eye(4 * N)), opn(Q2 - np.eye(4 * N)) > SEP)
    rec(f"K8b_{tag}_Q2_not_a_projector", opn(Q2 @ Q2 - Q2), opn(Q2 @ Q2 - Q2) > SEP,
        "1-Q2 is not a projector: spectrum of Q2 = " + str(sorted(set(np.round(np.linalg.eigvalsh(Q2), 6).tolist()))))
    Rm = np.eye(4 * N) - Q2
    worst_id = 0.0
    worst_R = 0.0
    worst_zero = 0.0
    worst_val = 0.0
    for b in range(2):
        Pb = np.kron(m.IN, P_REG[b])
        lhs = m.O(2, Pb) @ m.O(2, HSf) @ m.O(2, Pb) - m.O(2, Pb @ HSf @ Pb)
        rhs = V2.conj().T @ Pb @ (Q2 @ HSf @ Q2 - HSf) @ Pb @ V2
        rhs_R = V2.conj().T @ Pb @ (Rm @ HSf @ Rm - Rm @ HSf - HSf @ Rm) @ Pb @ V2
        rhs_Q1 = V2.conj().T @ Pb @ (np.eye(4 * N) @ HSf @ np.eye(4 * N) - HSf) @ Pb @ V2
        worst_id = max(worst_id, opn(lhs - rhs))
        worst_R = max(worst_R, opn(rhs - rhs_R))
        worst_zero = max(worst_zero, opn(rhs_Q1))
        for _ in range(N_STATES):
            cv = m.rand()
            den = (cv.conj() @ O21 @ cv).real
            worst_val = max(worst_val, abs((cv.conj() @ lhs @ cv).real) / den)
    rec(f"K8b_{tag}_defect_identity_residual", worst_id, worst_id < TOL)
    rec(f"{'K8b'}_{tag}_defect_depends_only_on_1_minus_Q2", worst_R, worst_R < TOL)
    rec(f"K8b_{tag}_defect_zero_when_Q2_replaced_by_1", worst_zero, worst_zero < TOL)
    rec(f"K8b_{tag}_defect_nontrivial_max_|Erel_b-Ekin_b|/omega(O1)", worst_val, worst_val > SEP)

    # ---- K8 (a): observer 1 homomorphism
    worst_a = 0.0
    for b in range(2):
        Pb = np.kron(m.IN, P_REG[b])
        worst_a = max(worst_a, opn(m.O(1, Pb) @ m.O(1, HSf) @ m.O(1, Pb) - m.O(1, Pb @ HSf @ Pb)))
    rec(f"K8a_{tag}_observer1_projection_homomorphism", worst_a, worst_a < TOL)

# ---- lam = 0 controls: E2 quadratic, observer 2 homomorphism, Q2 = 1
for N in (8, 6, 24):
    tag = f"N{N}_lam0"
    m = Model(N, 0)
    V2 = m.V(2)
    rec(f"{tag}_V2_unitary", opn(V2.conj().T @ V2 - np.eye(m.d)), opn(V2.conj().T @ V2 - np.eye(m.d)) < TOL)
    O2H, O21 = m.O(2, m.HSf), m.O(2, np.eye(4 * N))

    def g2(cv):
        return float(np.vdot(cv, cv).real * (cv.conj() @ O2H @ cv).real / (cv.conj() @ O21 @ cv).real)

    worst2, _ = polarization_defect(g2, m, N_PAIRS)
    rec(f"K6ii_{tag}_E2_normalized_exactly_quadratic", worst2, worst2 < TOL)
    worst_a = 0.0
    for b in range(2):
        Pb = np.kron(m.IN, P_REG[b])
        worst_a = max(worst_a, opn(m.O(2, Pb) @ m.O(2, m.HSf) @ m.O(2, Pb) - m.O(2, Pb @ m.HSf @ Pb)))
    rec(f"K8a_{tag}_observer2_projection_homomorphism", worst_a, worst_a < TOL)
    Q2 = V2 @ V2.conj().T
    rec(f"K8b_{tag}_Q2_equals_1", opn(Q2 - np.eye(4 * N)), opn(Q2 - np.eye(4 * N)) < TOL)
    rec(f"K8b_{tag}_Q2_closed_form", opn(Q2 - m.Q2_closed_form(0.0)), opn(Q2 - m.Q2_closed_form(0.0)) < TOL)

# ---- negative control: non-integer spectrum (8,2): group average is not a projector
m = Model(8, 2)
rec("NEG_N8_lam2_spectrum_not_integer", m.int_res, m.int_res > SEP)
rec("NEG_N8_lam2_group_average_not_projector", m.Pi_ga_idem_res, m.Pi_ga_idem_res > SEP)


# ---- K9 negative lemma: redshift-type coupling H_2 (1 + lam n_Q), H_S Dirac
def rate_model(N, lam, g=2.0):
    IN = np.eye(N, dtype=complex)
    Hc = np.diag(np.arange(N).astype(complex))
    HS = np.kron(I2, NQ) + g * np.kron(SX, I2)
    Hrest = np.kron(Hc, I4 + lam * np.kron(I2, NQ)) + np.kron(IN, HS)
    C = kron3(Hc, IN, I4) + np.kron(IN, Hrest)
    w, U = np.linalg.eigh(C)
    wr = np.round(w).astype(int)
    mask = (wr % N == 0)
    B = U[:, mask]
    d = int(mask.sum())
    bra = np.ones(N) / np.sqrt(N)
    V1 = np.sqrt(N) * kron3(bra[None, :], IN, I4) @ B
    V2 = np.sqrt(N) * kron3(IN, bra[None, :], I4) @ B
    HSf = np.kron(IN, HS)
    HSk = kron3(IN, IN, HS)
    Dl = V1.conj().T @ HSf @ V1 - V2.conj().T @ HSf @ V2
    return {"int_res": float(np.max(np.abs(w - wr))), "phys_dim": d, "V1dV1-1": opn(V1.conj().T @ V1 - np.eye(d)),
            "V2dV2-1": opn(V2.conj().T @ V2 - np.eye(d)), "gcd": math.gcd(1 + lam, N),
            "[HS,C]": opn(HSk @ C - C @ HSk), "Delta_norm": opn(Dl)}


r = rate_model(8, 2)
rec("K9_rate_N8_lam2_gcd1_HS_Dirac", r["[HS,C]"], r["[HS,C]"] < TOL and r["gcd"] == 1)
rec("K9_rate_N8_lam2_gcd1_V2_unitary", r["V2dV2-1"], r["V2dV2-1"] < TOL)
rec("K9_rate_N8_lam2_gcd1_Delta_zero", r["Delta_norm"], r["Delta_norm"] < TOL)
r = rate_model(8, 1)
rec("K9_rate_N8_lam1_gcd2_V2_not_unitary", {"V2dV2-1": r["V2dV2-1"], "gcd": r["gcd"], "Delta_norm": r["Delta_norm"]},
    r["V2dV2-1"] > SEP and r["gcd"] != 1)

all_pass = all(v["pass"] for v in checks.values())
out = {"claim": "K6+K8", "seed": SEED, "tol": TOL, "sep": SEP, "tol_spec": TOL_SPEC,
       "n_checks": len(checks), "n_fail": sum(not v["pass"] for v in checks.values()),
       "checks": checks, "all_pass": all_pass}
text = json.dumps(out, indent=1, default=float)
print(text)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json"), "w", encoding="utf-8") as f:
    f.write(text)
sys.exit(0 if all_pass else 1)
