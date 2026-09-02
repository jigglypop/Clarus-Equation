"""check_lambda_measure.py -- Q-0005 attempt-01, candidates C1 + C3 + C2 (numpy only).

Pre-declared BEFORE running (not changed afterwards):
  SEED=20260902, TOL=1e-10 (agreement), SEP=1e-3 (separation), RANK_TOL=1e-8 (relative SVD tol),
  mode tie-break = smallest E among the argmax set (the full tie set is also reported).

Shared model (fixed in candidate mode):
  H_kin = C^16_T (x) C^10_g (x) C^7_r, computational basis = Lambda eigenbasis |k>, Lambda = diag(k), k=0..15.
  Clock states |tau_n> = 16^{-1/2} sum_k e^{-i k tau_n}|k>, tau_n = 2 pi n/16 (DFT conjugate).
  H_g = diag(0,1,1,2,2,2,3,3,4,5), H_r = diag(0..6), H_0 = H_g(x)1 + 1(x)H_r, spec H_0 in 0..11 < 16 (no wrap).
  C = Lambda + H_0 (mod 16), Pi = (1/16) sum_t exp(2 pi i t C/16).
  Index: (k*10 + g)*7 + r.

Checks
  C1(a) p(Lambda_k) of Pi(chi(x)phi) equals |chi_hat_k|^2 ||Q_{-k} phi||^2 / Z          (max residual < TOL)
  C1(b) numeric rank of 200 random p-vectors = 12 (reachable sectors k in {0,5..15})       (rank == 12)
  C1(c) ||[Pi(|tau><tau| (x) A (x) 1)Pi, Lambda]|| for 10 random Hermitian A               (all < TOL -> superselection wording)
        diagnostics: A diagonal in H_g basis, and tau-averaged O                            (expected < TOL)
  C2    Haar rho_kin=1/dim -> p(Lambda_k)=d_k/70 ; Lueders with E -> Tr(E P_k)/sum_j Tr(E P_j) (max residual < TOL)
  C3(a) f_k = Tr(E P_k)/d_k constant over reachable sectors?                                (max dev < TOL -> reject C3)
  C3(b) mode -Lambda* nondecreasing in eps in {1,2,3,4}; multiplicity B shifts mode >= 1     (both -> (ii) rejected)
Prints JSON to stdout and writes result.json next to this script.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SEED = 20260902
TOL = 1e-10
SEP = 1e-3
RANK_TOL = 1e-8
N = 16
HG_A = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4, 5], dtype=int)
HG_B = np.array([0, 0, 0, 1, 2, 3, 4, 5, 5, 5], dtype=int)
HR = np.arange(7, dtype=int)
DG, DR = len(HG_A), len(HR)
DS = DG * DR  # 70
DIM = N * DS  # 1120
EPS_LIST = [1, 2, 3, 4]
EXPECTED_D_OF_E = [1, 3, 6, 8, 9, 10, 10, 9, 7, 4, 2, 1]  # hand-computed convolution, E=0..11
REACHABLE = [0] + list(range(5, 16))
UNREACHABLE = [1, 2, 3, 4]

rng = np.random.default_rng(SEED)
checks: dict = {}


def rec(name: str, ok: bool, **kw) -> None:
    d = {"pass": bool(ok)}
    d.update(kw)
    checks[name] = d


def opnorm(X: np.ndarray) -> float:
    return float(np.linalg.norm(X, 2))


def rand_state(n: int) -> np.ndarray:
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    return v / np.linalg.norm(v)


def rand_herm(n: int) -> np.ndarray:
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (M + M.conj().T) / 2


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, hg: np.ndarray) -> None:
        self.hg = hg
        self.E0 = (hg[:, None] + HR[None, :]).reshape(-1)  # H_0 diag on C^70, index g*7+r
        self.r_of = np.tile(HR, DG)  # H_r value per (g,r) index
        self.k_of = np.repeat(np.arange(N), DS)  # Lambda diag on C^1120
        self.E_of = np.tile(self.E0, N)
        self.Cdiag = (self.k_of + self.E_of) % N
        t = np.arange(N)
        self.Pi_ga = np.exp(2j * np.pi * np.outer(t, self.Cdiag) / N).sum(axis=0) / N  # group average
        self.Pi_ind = (self.Cdiag == 0).astype(float)  # shell indicator
        self.d_of_E = np.array([int(np.sum(self.E0 == E)) for E in range(N)])  # E=0..15
        self.d_k = np.array([self.d_of_E[(-k) % N] for k in range(N)])

    def QE(self, E: int) -> np.ndarray:
        return (self.E0 == E).astype(float)

    def Erec(self, eps: int) -> np.ndarray:  # diag of 1(x)1(x)P_{H_r>=eps} on C^70
        return (self.r_of >= eps).astype(float)

    def trEPk(self, eps: int) -> np.ndarray:  # Tr(E P_k), P_k = |k><k| (x) Q_{-k}
        e = self.Erec(eps)
        return np.array([float(np.sum(e * self.QE((-k) % N))) for k in range(N)])


A = Model(HG_A)
B = Model(HG_B)

# ------------------------------------------------------------------ S1: shell, table
rec("S1_no_wrap_max_specH0_lt_N", int(A.E0.max()) < N, max_spec=int(A.E0.max()), N=N)
res = float(np.max(np.abs(A.Pi_ga - A.Pi_ind)))
rec("S1_group_average_equals_shell_indicator", res < TOL, residual=res, tol=TOL)
res = float(np.max(np.abs(A.Pi_ga * A.Pi_ga - A.Pi_ga)))
rec("S1_Pi_is_projector", res < TOL, residual=res, tol=TOL)
rec("S1_dim_phys_70", int(round(A.Pi_ind.sum())) == DS, value=int(round(A.Pi_ind.sum())), expected=DS)
rec("S1_d_of_E_table", A.d_of_E[:12].tolist() == EXPECTED_D_OF_E and int(A.d_of_E[12:].sum()) == 0,
    value=A.d_of_E[:12].tolist(), expected=EXPECTED_D_OF_E)
rec("S1_d_k_unreachable_zero", all(int(A.d_k[k]) == 0 for k in UNREACHABLE), d_k=A.d_k.tolist())
rec("S1_d_of_E_table_B_sums_to_70", int(B.d_of_E.sum()) == DS, value=B.d_of_E[:12].tolist())

# ------------------------------------------------------------------ C1(a): closed form
tau = 2 * np.pi * np.arange(N) / N
ks = np.arange(N)
clock = np.exp(-1j * np.outer(tau, ks)) / np.sqrt(N)  # clock[n] = |tau_n> in k basis
chis = {
    "delta_tau0": clock[0],
    "window4": clock[:4].sum(axis=0) / np.sqrt(4),
    "uniform_tau": clock.sum(axis=0) / np.sqrt(N),
}
res = float(np.linalg.norm(chis["uniform_tau"] - np.eye(N)[0]))
rec("C1a_uniform_tau_equals_Lambda_eigenvector_k0", res < TOL, residual=res, tol=TOL)
res = float(np.max(np.abs(np.abs(chis["delta_tau0"]) ** 2 - 1.0 / N)))
rec("C1a_delta_tau0_has_flat_Lambda_amplitudes", res < TOL, residual=res, tol=TOL)

worst = 0.0
examples = {}
for name, chi in chis.items():
    for i in range(20):
        phi = rand_state(DS)
        psi = np.kron(chi, phi)
        Psi = A.Pi_ga * psi
        Z = float(np.vdot(Psi, Psi).real)
        p_direct = np.array([float(np.vdot(Psi[k * DS:(k + 1) * DS], Psi[k * DS:(k + 1) * DS]).real) for k in range(N)]) / Z
        p_closed = np.array([abs(chi[k]) ** 2 * float(np.sum(A.QE((-k) % N) * np.abs(phi) ** 2)) for k in range(N)])
        p_closed /= p_closed.sum()
        worst = max(worst, float(np.max(np.abs(p_direct - p_closed))))
        if i == 0:
            examples[name] = [round(float(x), 6) for x in p_direct]
rec("C1a_closed_form_p_Lambda_60_states", worst < TOL, max_residual=worst, tol=TOL, examples_first_phi=examples)

# ------------------------------------------------------------------ C1(b): rank of reachable measures
P = np.zeros((200, N))
for i in range(200):
    psi = rand_state(DIM)
    Psi = A.Pi_ga * psi
    Z = float(np.vdot(Psi, Psi).real)
    P[i] = [float(np.vdot(Psi[k * DS:(k + 1) * DS], Psi[k * DS:(k + 1) * DS]).real) / Z for k in range(N)]
s = np.linalg.svd(P, compute_uv=False)
rank = int(np.sum(s > RANK_TOL * s[0]))
rec("C1b_rank_of_200_random_measures_equals_12", rank == len(REACHABLE), rank=rank, expected=len(REACHABLE),
    rank_tol_rel=RANK_TOL, smallest_kept_sv=float(s[rank - 1]), largest_dropped_sv=float(s[rank]) if rank < N else 0.0)
res = float(np.max(np.abs(P[:, UNREACHABLE])))
rec("C1b_unreachable_sectors_zero", res < TOL, max=res, tol=TOL)
res = float(np.max(np.abs(P - A.d_k / DS)))
rec("C1b_measures_not_all_equal_to_dk_over_70", res > SEP, max_dev_from_dk=res, sep=SEP)

# ------------------------------------------------------------------ C1(c): commutator with Lambda
Lam = A.k_of.astype(float)
I7 = np.eye(DR)


def rel_obs(tau_vec: np.ndarray, Amat: np.ndarray) -> np.ndarray:
    K = np.kron(np.outer(tau_vec, tau_vec.conj()), np.kron(Amat, I7))
    return (A.Pi_ga[:, None] * K) * A.Pi_ga[None, :]


def comm_norm(O: np.ndarray) -> float:
    return opnorm(O * (Lam[None, :] - Lam[:, None]))


gen_norms, diag_norms, avg_norms = [], [], []
for i in range(10):
    Am = rand_herm(DG)
    gen_norms.append(comm_norm(rel_obs(clock[0], Am)))
    Ad = np.diag(rng.normal(size=DG)).astype(complex)  # commutes with H_g
    diag_norms.append(comm_norm(rel_obs(clock[0], Ad)))
    Obar = sum(rel_obs(clock[n], Am) for n in range(N)) / N
    avg_norms.append(comm_norm(Obar))
rec("C1c_generic_A_all_commutators_below_TOL", max(gen_norms) < TOL, max_norm=max(gen_norms), min_norm=min(gen_norms), tol=TOL,
    note="pre-declared: pass -> strengthen wording to Lambda superselection; fail -> no strengthening")
rec("C1c_generic_A_commutators_above_SEP", min(gen_norms) > SEP, min_norm=min(gen_norms), sep=SEP)
rec("C1c_diag_A_commutes_with_Lambda", max(diag_norms) < TOL, max_norm=max(diag_norms), tol=TOL)
rec("C1c_tau_averaged_O_commutes_with_Lambda", max(avg_norms) < TOL, max_norm=max(avg_norms), tol=TOL)
Am = rand_herm(DG)
n5 = comm_norm(rel_obs(clock[5], Am))
rec("C1c_tau5_generic_A_commutator_above_SEP", n5 > SEP, norm=n5, sep=SEP)

# ------------------------------------------------------------------ C2: Haar -> counting measure, Lueders
rho = A.Pi_ga.real / A.Pi_ga.real.sum()  # rho_phys diag = Pi/70
p_haar = np.array([rho[k * DS:(k + 1) * DS].sum() for k in range(N)])
res = float(np.max(np.abs(p_haar - A.d_k / DS)))
rec("C2_haar_p_equals_dk_over_70", res < TOL, residual=res, tol=TOL, p=[round(float(x), 6) for x in p_haar])
worst = 0.0
for eps in EPS_LIST:
    e = np.tile(A.Erec(eps), N)
    rho_c = e * rho * e
    rho_c /= rho_c.sum()
    p_c = np.array([rho_c[k * DS:(k + 1) * DS].sum() for k in range(N)])
    t = A.trEPk(eps)
    worst = max(worst, float(np.max(np.abs(p_c - t / t.sum()))))
rec("C2_lueders_equals_TrEPk_over_sum_eps1to4", worst < TOL, max_residual=worst, tol=TOL)


# ------------------------------------------------------------------ C3: selection function and peak
def mode_info(weights_by_E: np.ndarray) -> dict:
    w = weights_by_E[:12]
    m = w.max()
    ties = [int(E) for E in range(12) if abs(w[E] - m) < TOL]
    mean = float(np.sum(np.arange(12) * w) / w.sum())
    return {"mode_E": ties[0], "tie_set": ties, "mean_E": mean, "weights": [float(x) for x in w]}


c3 = {}
for label, M in (("A", A), ("B", B)):
    c3[label] = {}
    for eps in EPS_LIST:
        t = M.trEPk(eps)  # indexed by k
        byE = np.array([t[(-E) % N] for E in range(N)])
        info = mode_info(byE)
        f = np.array([t[k] / M.d_k[k] for k in REACHABLE])
        info["f_k_reachable"] = [round(float(x), 6) for x in f]
        info["f_max_dev"] = float(np.max(np.abs(f - f.mean())))
        info["Lambda_star"] = int((-info["mode_E"]) % N)
        info["Lambda_star_signed"] = int(((-info["mode_E"] + 8) % N) - 8)
        c3[label][eps] = info

f_dev_eps1 = c3["A"][1]["f_max_dev"]
rec("C3a_selection_function_constant_eps1", f_dev_eps1 < TOL, max_dev=f_dev_eps1, tol=TOL,
    note="pre-declared: pass -> reject C3 (no peak); fail -> conditioning moves the peak")
rec("C3a_selection_function_nonconstant_all_eps", all(c3["A"][e]["f_max_dev"] > SEP for e in EPS_LIST),
    devs={str(e): c3["A"][e]["f_max_dev"] for e in EPS_LIST}, sep=SEP)
modes_A = [c3["A"][e]["mode_E"] for e in EPS_LIST]
modes_B = [c3["B"][e]["mode_E"] for e in EPS_LIST]
rec("C3b_mode_nondecreasing_in_eps_A", all(modes_A[i] <= modes_A[i + 1] for i in range(3)), modes_E=modes_A,
    tie_sets=[c3["A"][e]["tie_set"] for e in EPS_LIST])
rec("C3b_mode_shifts_with_eps_A", max(modes_A) - min(modes_A) >= 1, shift=max(modes_A) - min(modes_A))
rec("C3b_mode_shifts_with_multiplicity_B_some_eps", any(modes_A[i] != modes_B[i] for i in range(4)),
    modes_A=modes_A, modes_B=modes_B, tie_sets_B=[c3["B"][e]["tie_set"] for e in EPS_LIST])
mean_shift = max(abs(c3["A"][e]["mean_E"] - c3["B"][e]["mean_E"]) for e in EPS_LIST)
rec("C3b_mean_shifts_with_multiplicity_B_gt_SEP", mean_shift > SEP, max_mean_shift=mean_shift,
    means_A=[c3["A"][e]["mean_E"] for e in EPS_LIST], means_B=[c3["B"][e]["mean_E"] for e in EPS_LIST], sep=SEP)
rec("C3b_sign_Lambda_star_negative_representative", all(c3["A"][e]["Lambda_star_signed"] < 0 for e in EPS_LIST),
    Lambda_star=[c3["A"][e]["Lambda_star"] for e in EPS_LIST], signed=[c3["A"][e]["Lambda_star_signed"] for e in EPS_LIST])
rec("C3b_scale_peak_is_order_input_energies", all(1 <= c3["A"][e]["mode_E"] <= 11 for e in EPS_LIST),
    note="peak at E* inside spec H_0; no hierarchy generated")

CONDITIONAL = ("C1c_generic_A_all_commutators_below_TOL", "C3a_selection_function_constant_eps1")
required = [k for k in checks if k not in CONDITIONAL]
result = {
    "claim": "C1+C3+C2",
    "question": "Q-0005",
    "attempt": 1,
    "seed": SEED,
    "tol": TOL,
    "sep": SEP,
    "rank_tol_rel": RANK_TOL,
    "model": {"N": N, "H_g": HG_A.tolist(), "H_g_B": HG_B.tolist(), "H_r": HR.tolist(), "dim_kin": DIM, "dim_phys": DS},
    "d_k": A.d_k.tolist(),
    "c3": {lab: {str(e): v for e, v in d.items()} for lab, d in c3.items()},
    "checks": checks,
    "conditional_wording": {
        "C1c_superselection_wording_strengthened": checks[CONDITIONAL[0]]["pass"],
        "C3_rejected_by_constant_selection_function": checks[CONDITIONAL[1]]["pass"],
    },
    "required_checks": required,
    "all_pass": all(checks[k]["pass"] for k in required),
}
out = json.dumps(result, ensure_ascii=False, indent=2)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json"), "w", encoding="utf-8") as fh:
    fh.write(out)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
print(out)
