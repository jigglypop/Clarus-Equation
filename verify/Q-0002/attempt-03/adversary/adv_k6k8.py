"""adv_k6k8.py -- adversary for Q-0002 attempt-03 (K6+K8+K9). numpy only. seed 20260902.
Writes adv_result.json next to this file. Does not modify any prover file.
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np

SEED = 20260902
rng = np.random.default_rng(SEED)
TOL, SEP = 1e-10, 1e-3

I2 = np.eye(2, dtype=complex); I4 = np.eye(4, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
NQ = np.diag([0.0, 1.0]).astype(complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)
P_REG = [np.kron(np.diag([1.0, 0.0]).astype(complex), I2), np.kron(np.diag([0.0, 1.0]).astype(complex), I2)]

def kron3(a, b, c): return np.kron(np.kron(a, b), c)
def opn(X): return float(np.linalg.norm(X, 2))
def tau_ket(N, tau): return np.exp(-1j * np.arange(N) * tau) / np.sqrt(N)
def rnd(d):
    c = rng.normal(size=d) + 1j * rng.normal(size=d); return c / np.linalg.norm(c)

class Model:
    """cyclic=False -> open-boundary raising operator on clock 2 (no wraparound)."""
    def __init__(self, N, lam, Estar=1.0, g=2.0, cyclic=True):
        self.N, self.lam = N, lam
        IN = np.eye(N, dtype=complex); self.IN = IN
        Hc = np.diag(np.arange(N).astype(complex))
        T = np.zeros((N, N), complex)
        for k in range(N - 1): T[k + 1, k] = 1
        if cyclic: T[0, N - 1] = 1
        self.HS = Estar * np.kron(I2, NQ) + g * np.kron(SX, I2)
        Hint = np.kron(T, np.kron(I2, SM)); self.Hint = lam * (Hint + Hint.conj().T)
        self.HSf = np.kron(IN, self.HS)
        self.Hrest = np.kron(Hc, I4) + self.HSf + self.Hint
        self.H1 = kron3(Hc, IN, I4)
        self.C = self.H1 + np.kron(IN, self.Hrest)
        w, U = np.linalg.eigh(self.C)
        self.int_res = float(np.max(np.abs(w - np.round(w))))
        mu, Wr = np.linalg.eigh(self.Hrest); self.mu, self.Wr = mu, Wr
        self.k1 = (-np.round(mu).astype(int)) % N
        cols = []
        for a in range(4 * N):
            e1 = np.zeros(N, complex); e1[self.k1[a]] = 1
            cols.append(np.kron(e1, Wr[:, a]))
        self.B = np.array(cols).T
        self.d = self.B.shape[1]
    def V(self, i, tau=0.0):
        bra = tau_ket(self.N, tau).conj()[None, :]
        Rd = kron3(bra, self.IN, I4) if i == 1 else kron3(self.IN, bra, I4)
        return np.sqrt(self.N) * Rd @ self.B
    def O(self, i, A, tau=0.0):
        V = self.V(i, tau); return V.conj().T @ A @ V

out = {}

# A. K8(b) identity is pure algebra
A = {}
for trial in range(5):
    d, m = 12, 9
    V = rng.normal(size=(m, d)) + 1j * rng.normal(size=(m, d))
    H = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); H = H + H.conj().T
    P = np.zeros((m, m), complex); idx = rng.choice(m, 4, replace=False); P[idx, idx] = 1
    O = lambda X: V.conj().T @ X @ V
    Q = V @ V.conj().T; R = np.eye(m) - Q
    lhs = O(P) @ O(H) @ O(P) - O(P @ H @ P)
    rhs = V.conj().T @ P @ (Q @ H @ Q - H) @ P @ V
    rhsR = V.conj().T @ P @ (R @ H @ R - R @ H - H @ R) @ P @ V
    A[f"random_V_trial{trial}"] = {"S5.1_residual": opn(lhs - rhs), "S5.2_residual": opn(rhs - rhsR)}
A["note"] = "S5.1/S5.2 hold for ANY linear map V and ANY H,P: no model content. Content of K8(b) reduces to Q2 != 1."
out["A_K8b_identity_is_tautology"] = A

# B. integer-spectrum condition = wraparound artifact
B = {}
for N in (6, 8, 10, 12, 24):
    for lam in (1, 2, 3, 4, 5, 7):
        mc = Model(N, lam, cyclic=True); mo = Model(N, lam, cyclic=False)
        rho = math.sqrt((N / 2) ** 2 + lam ** 2)
        B[f"N{N}_lam{lam}"] = {
            "cyclic_int_res": mc.int_res, "open_int_res": mo.int_res,
            "cyclic_Hrest_nonint_count": int(np.sum(np.abs(mc.mu - np.round(mc.mu)) > 1e-8)),
            "rho": rho, "pythagorean": abs(rho - round(rho)) < 1e-9}
B["note"] = ("open-boundary clock-2 raising operator gives integer spectrum for EVERY integer lam; "
             "the cyclic model has exactly 4 non-integer H_rest eigenvalues (j=0 wraparound block) unless (N/2,lam,rho) Pythagorean.")
out["B_integer_condition_is_wraparound_artifact"] = B

Bo = {}
for (N, lam) in [(8, 2), (8, 3), (6, 4), (8, 1), (10, 3)]:
    m = Model(N, lam, cyclic=False)
    V2 = m.V(2); G = V2.conj().T @ V2; ev = np.linalg.eigvalsh(G)
    Q2 = V2 @ V2.conj().T
    HSf = m.HSf; worst = 0.0
    O21 = m.O(2, np.eye(4 * N))
    for b in range(2):
        Pb = np.kron(m.IN, P_REG[b])
        lhs = m.O(2, Pb) @ m.O(2, HSf) @ m.O(2, Pb) - m.O(2, Pb @ HSf @ Pb)
        for _ in range(20):
            c = rnd(m.d); worst = max(worst, abs(c.conj() @ lhs @ c).real / (c.conj() @ O21 @ c).real)
    Bo[f"open_N{N}_lam{lam}"] = {"int_res": m.int_res, "dim_phys": m.d, "spec_V2dV2": sorted(set(np.round(ev, 6).tolist())),
                                 "Q2_ne_1": opn(Q2 - np.eye(4 * N)), "defect_max": worst}
out["B2_open_boundary_generic_lam"] = Bo

# C. E* != 1 and g variations
Cc = {}
for Estar in (0.0, 2.0, 3.0, 0.5):
    row = {}
    for (N, lam) in [(8, 3), (6, 4), (24, 5), (8, 2)]:
        m = Model(N, lam, Estar=Estar); row[f"N{N}_lam{lam}"] = m.int_res
    Cc[f"Estar{Estar}"] = row
Cc["analytic"] = "j>=1 block eigen j+(E*-1)/2 +- sqrt(((E*-1)/2)^2+lam^2). E*=1: always integer. E*=2: lam^2=k(k+1) never a square for lam>0. E*=3: lam^2=(k+1)^2-1 never. Integer admissibility requires E*=1 (exact resonance)."
for g in (1.0, 3.0, 0.5, 1.5):
    row = {}
    for (N, lam) in [(8, 3), (6, 4), (24, 5)]:
        m = Model(N, lam, g=g)
        if m.int_res < 1e-8:
            V2 = m.V(2); ev = np.linalg.eigvalsh(V2.conj().T @ V2)
            row[f"N{N}_lam{lam}"] = {"int_res": m.int_res, "spec_V2dV2": sorted(set(np.round(ev, 6).tolist()))}
        else:
            row[f"N{N}_lam{lam}"] = {"int_res": m.int_res, "spec_V2dV2": "Pi not projector"}
    Cc[f"g{g}"] = row
out["C_Estar_g_dependence"] = Cc

# D. off-lattice tau and POVM reading
Dd = {}
for (N, lam) in [(8, 3), (6, 4), (24, 5)]:
    m = Model(N, lam); rho = math.sqrt((N / 2) ** 2 + lam ** 2); c_ = math.sqrt((1 + lam / rho) / 2)
    declared = [0, 1, 2, 1 - c_, 1 + c_]
    row = {}
    for frac in (0.25, 0.5, 0.31):
        tau = 2 * math.pi * frac / N
        V2 = m.V(2, tau); ev = np.linalg.eigvalsh(V2.conj().T @ V2)
        row[f"tau_frac{frac}"] = {"min_eig": float(ev.min()), "max_eig": float(ev.max()),
                                  "n_distinct": int(len(set(np.round(ev, 6).tolist()))),
                                  "max_dist_to_declared": float(max(min(abs(x - y) for y in declared) for x in ev))}
    S = sum(m.V(2, 2 * math.pi * n / N).conj().T @ m.V(2, 2 * math.pi * n / N) for n in range(N))
    row["sum_lattice_V2dV2_minus_N"] = opn(S - N * np.eye(m.d))
    c = rnd(m.d)
    probs = [float(np.linalg.norm(m.V(2, 2 * math.pi * n / N) @ c) ** 2 / N) for n in range(N)]
    row["clock2_reading_probs_random_state"] = probs
    row["clock2_reading_probs_sum"] = float(sum(probs))
    row["max_reading_prob_times_N"] = float(np.linalg.eigvalsh(m.V(2).conj().T @ m.V(2)).max())
    Dd[f"N{N}_lam{lam}"] = row
Dd["note"] = "Off-lattice tau: spectrum leaves the declared set. V2dV2(tau)/N is the clock-2 reading POVM on H_phys: eigenvalue 2 = reading probability 2/N, 0 = never read at that tau."
out["D_offlattice_and_POVM"] = Dd

# E. K9 gcd!=1 rate model has the same K8(b)-type defect
def rate_model(N, lam, g=2.0):
    IN = np.eye(N, dtype=complex); Hc = np.diag(np.arange(N).astype(complex))
    HS = np.kron(I2, NQ) + g * np.kron(SX, I2)
    Hrest = np.kron(Hc, I4 + lam * np.kron(I2, NQ)) + np.kron(IN, HS)
    C = kron3(Hc, IN, I4) + np.kron(IN, Hrest)
    w, U = np.linalg.eigh(C); wr = np.round(w).astype(int); mask = (wr % N == 0)
    B = U[:, mask]; d = int(mask.sum())
    bra = np.ones(N) / np.sqrt(N)
    V2 = np.sqrt(N) * kron3(IN, bra[None, :], I4) @ B
    HSf = np.kron(IN, HS)
    O = lambda X: V2.conj().T @ X @ V2
    O1 = O(np.eye(4 * N)); Q2 = V2 @ V2.conj().T
    worst = 0.0
    for b in range(2):
        Pb = np.kron(IN, P_REG[b]); lhs = O(Pb) @ O(HSf) @ O(Pb) - O(Pb @ HSf @ Pb)
        for _ in range(20):
            c = rnd(d); den = (c.conj() @ O1 @ c).real
            if den > 1e-9: worst = max(worst, abs(c.conj() @ lhs @ c).real / den)
    return {"dim_phys": d, "V2dV2_minus_1": opn(V2.conj().T @ V2 - np.eye(d)), "Q2_spec": sorted(set(np.round(np.linalg.eigvalsh(Q2), 6).tolist())),
            "K8b_defect_max": worst, "comm_HS_C": opn(kron3(IN, IN, HS) @ C - C @ kron3(IN, IN, HS))}
Ee = {"rate_N8_lam1_gcd2": rate_model(8, 1), "rate_N8_lam3_gcd4": rate_model(8, 3), "rate_N6_lam2_gcd3": rate_model(6, 2), "rate_N8_lam2_gcd1_control": rate_model(8, 2)}
Ee["note"] = "Redshift-type (Dirac H_S) coupling with gcd(1+lam,N)!=1: V2 non-isometric and a K8(b)-type defect appears with H_S Dirac. K9 conclusion that [H_S,C]!=0 is necessary is false in the finite model class as stated."
out["E_K9_gcd_defect"] = Ee

# F. Pythagorean scan
Ff = {}
for N in range(4, 30, 2):
    for lam in range(1, 40):
        rho2 = (N // 2) ** 2 + lam ** 2; rho = int(round(math.sqrt(rho2)))
        if rho * rho != rho2: continue
        m = Model(N, lam)
        if m.int_res > 1e-8: continue
        V2 = m.V(2); ev = np.linalg.eigvalsh(V2.conj().T @ V2)
        c_ = math.sqrt((1 + lam / rho) / 2); cp = math.sqrt((1 - lam / rho) / 2)
        cand = {"0": 0, "1": 1, "2": 2, "1-c": 1 - c_, "1+c": 1 + c_, "1-cp": 1 - cp, "1+cp": 1 + cp}
        labels = set()
        for x in ev:
            k = min(cand, key=lambda kk: abs(cand[kk] - x)); labels.add(k if abs(cand[k] - x) < 1e-6 else "other:%.4f" % x)
        Ff[f"N{N}_lam{lam}_rho{rho}"] = {"spec_labels": sorted(labels), "dim_ker_V2": int(np.sum(ev < 1e-8))}
out["F_pythagorean_scan"] = Ff

# G. defect sign, kernel states
Gg = {}
for (N, lam) in [(8, 3), (6, 4), (24, 5)]:
    m = Model(N, lam); V2 = m.V(2); HSf = m.HSf; O21 = m.O(2, np.eye(4 * N))
    vals = []
    for b in range(2):
        Pb = np.kron(m.IN, P_REG[b]); lhs = m.O(2, Pb) @ m.O(2, HSf) @ m.O(2, Pb) - m.O(2, Pb @ HSf @ Pb)
        for _ in range(40):
            c = rnd(m.d); vals.append(float((c.conj() @ lhs @ c).real / (c.conj() @ O21 @ c).real))
    ev, U = np.linalg.eigh(V2.conj().T @ V2); ker = U[:, ev < 1e-8]
    Gg[f"N{N}_lam{lam}"] = {"defect_min": min(vals), "defect_max": max(vals), "sign_indefinite": bool(min(vals) < -SEP and max(vals) > SEP),
                            "dim_ker_V2_tau0": int(ker.shape[1])}
out["G_defect_sign_and_kernel"] = Gg

# H. no continuum limit
Hh = {}
for N in (8, 16, 32, 64, 128):
    lam = 3; rho = math.sqrt((N / 2) ** 2 + lam ** 2); Hh[f"N{N}_lam3"] = {"rho": rho, "frac": abs(rho - round(rho))}
Hh["note"] = "lam^2 = (rho-N/2)(rho+N/2) with both factors positive integers => rho+N/2 <= lam^2 => N < lam^2. Fixed lam admits finitely many N; no N->inf limit."
out["H_no_continuum_limit"] = Hh

# I. tautology count
res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result.json"), encoding="utf-8"))["checks"]
taut = [k for k in res if any(s in k for s in ("defect_identity_residual", "depends_only_on_1_minus_Q2", "defect_zero_when_Q2_replaced_by_1",
        "block_diagonal_in_k1", "block_equals_Gram_of_slices", "E1_exactly_quadratic", "phys_dim_equals_4N", "_c_value"))]
implied = [k for k in res if any(s in k for s in ("E1_minus_E2_polarization", "K8a_", "Q2_ne_1", "Q2_not_a_projector", "lam0_E2_normalized_exactly_quadratic",
          "lam0_Q2_equals_1", "K9_rate_N8_lam2_gcd1_Delta_zero"))]
out["I_check_discrimination"] = {"n_total": len(res), "n_tautology_or_by_construction": len(taut), "n_implied_by_other_checks": len(implied),
                                 "n_independent": len(res) - len(taut) - len(implied), "tautology": taut, "implied": implied}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_result.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
