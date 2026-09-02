"""adv_hidden_assumptions.py -- adversary for Q-0005 attempt-01 (numpy only, seed 20260902).

Probes (on the prover shared model unless stated):
  A. random_sample_20: closed form p_k = ||Q_{-k} psi_k||^2/Z on 20 random NON-product states (generalises S2.2).
  B. symmetry: clock shift chi -> e^{-i Lambda tau} chi leaves p(Lambda) invariant.
  C. limit N -> 32, 64 with same H_0: d(E), modes unchanged (finite clock size irrelevant once no wrap).
  D. wrap: N=8 with same H_0 (spec up to 11 > 7): sectors merge, signed representative of Lambda* changes.
  E. eps=0 (no observer conditioning): the peak already exists (mode of d(E)); conditioning shifts it by <=1.
  F. prior dependence: chi=|tau_0>, phi concentrated on one E-shell -> conditioned peak sits wherever phi puts it.
  G. tie-rule / eps-range fragility of C3b checks.
  H. discriminability: random multiplicity tables m_g (200) -> how often the C3b check suite would pass.
  I. interaction (dressing unitary U on C^70 not commuting with H_r): [E,Pi] != 0; Lueders-after-Pi vs Page counting differ;
     coupling -> 0 recovers agreement.
Writes adv_result.json next to this script.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

SEED = 20260902
TOL = 1e-10
rng = np.random.default_rng(SEED)
HG_A = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4, 5])
HG_B = np.array([0, 0, 0, 1, 2, 3, 4, 5, 5, 5])
HR = np.arange(7)
out: dict = {}


def model(hg, N):
    E0 = (hg[:, None] + HR[None, :]).reshape(-1)
    r_of = np.tile(HR, len(hg))
    d_of_E = np.array([int(np.sum(E0 % N == E)) for E in range(N)])  # wrap-aware
    d_k = np.array([d_of_E[(-k) % N] for k in range(N)])
    return E0, r_of, d_of_E, d_k


def shell_indicator(E0, N):
    DS = len(E0)
    k_of = np.repeat(np.arange(N), DS)
    E_of = np.tile(E0, N)
    return ((k_of + E_of) % N == 0).astype(float), DS


def p_of_state(psi, Pi, N, DS):
    Psi = Pi * psi
    Z = np.vdot(Psi, Psi).real
    return np.array([np.vdot(Psi[k*DS:(k+1)*DS], Psi[k*DS:(k+1)*DS]).real for k in range(N)]) / Z


def rs(n):
    v = rng.normal(size=n) + 1j*rng.normal(size=n)
    return v/np.linalg.norm(v)


def n_eps(E0, r_of, N, eps):
    return np.array([float(np.sum((E0 % N == E) & (r_of >= eps))) for E in range(N)])


def mode(w, rule="smallest"):
    m = w.max(); ties = [int(E) for E in range(len(w)) if abs(w[E]-m) < TOL]
    return (ties[0] if rule == "smallest" else ties[-1]), ties


# A
E0, r_of, dE, dk = model(HG_A, 16)
Pi, DS = shell_indicator(E0, 16)
worst = 0.0
for _ in range(20):
    psi = rs(16*DS)
    p_direct = p_of_state(psi, Pi, 16, DS)
    p_closed = np.array([np.sum(((E0 % 16) == (-k) % 16) * np.abs(psi[k*DS:(k+1)*DS])**2) for k in range(16)])
    p_closed /= p_closed.sum()
    worst = max(worst, float(np.max(np.abs(p_direct-p_closed))))
out["A_random20_nonproduct_closed_form_max_residual"] = worst

# B
worst = 0.0
for _ in range(20):
    chi = rs(16); phi = rs(DS)
    p0 = p_of_state(np.kron(chi, phi), Pi, 16, DS)
    for tau in rng.uniform(0, 2*np.pi, size=5):
        chi_s = np.exp(-1j*np.arange(16)*tau)*chi
        worst = max(worst, float(np.max(np.abs(p_of_state(np.kron(chi_s, phi), Pi, 16, DS)-p0))))
out["B_clock_shift_invariance_max_residual"] = worst

# C
resC = {}
for N in (16, 32, 64):
    E0n, r_n, dEn, dkn = model(HG_A, N)
    resC[str(N)] = {"d_of_E_0_11": dEn[:12].tolist(), "modes_eps1to4": [mode(n_eps(E0n, r_n, N, e))[0] for e in (1, 2, 3, 4)],
                    "reachable_sectors": int(np.sum(dkn > 0))}
out["C_large_N"] = resC

# D
E0w, r_w, dEw, dkw = model(HG_A, 8)
modes_w = []
for e in (1, 2, 3, 4):
    m, ties = mode(n_eps(E0w, r_w, 8, e))
    lam = (-m) % 8; signed = ((lam+4) % 8)-4
    modes_w.append({"eps": e, "mode_E_mod8": m, "ties": ties, "Lambda_star_signed": int(signed)})
out["D_wrap_N8"] = {"d_of_E_mod8": dEw.tolist(), "reachable_sectors": int(np.sum(dkw > 0)), "modes": modes_w}

# E
w0 = n_eps(E0, r_of, 16, 0)[:12]
m0, t0 = mode(w0)
resE = {"eps0_mode_E": m0, "eps0_ties": t0, "eps0_mean_E": float(np.sum(np.arange(12)*w0)/w0.sum())}
for e in (1, 2, 3, 4):
    w = n_eps(E0, r_of, 16, e)[:12]
    resE[f"eps{e}_mode_E"] = mode(w)[0]
    resE[f"eps{e}_mean_E"] = float(np.sum(np.arange(12)*w)/w.sum())
out["E_unconditioned_peak"] = resE

# F
resF = {}
chi = np.ones(16, dtype=complex)/4.0  # |tau_0>
def Erec(eps):
    return np.tile((r_of >= eps).astype(float), 16)
def cond_modes(psi, eps_list):
    res = []
    for e in eps_list:
        Psi = Erec(e)*(Pi*psi); Z = np.vdot(Psi, Psi).real
        if Z < 1e-14:
            res.append(None); continue
        p = np.array([np.vdot(Psi[k*DS:(k+1)*DS], Psi[k*DS:(k+1)*DS]).real for k in range(16)])/Z
        byE = np.array([p[(-E) % 16] for E in range(12)])
        res.append(mode(byE)[0])
    return res
for Eshell in (1, 4, 10, 11):
    phi = ((E0 == Eshell).astype(float)) * rs(DS); phi /= np.linalg.norm(phi)
    resF[f"phi_on_shell_E{Eshell}"] = cond_modes(np.kron(chi, phi), (1, 2, 3, 4))
phi = 3.0*((E0 == 4).astype(float))*rs(DS) + 1.0*((E0 == 11).astype(float))*rs(DS); phi /= np.linalg.norm(phi)
resF["phi_mix_E4_E11_modes_eps0to5"] = cond_modes(np.kron(chi, phi), (0, 1, 2, 3, 4, 5))
out["F_prior_dependence"] = resF

# G
resG = {}
for lab, hg in (("A", HG_A), ("B", HG_B)):
    E0x, r_x, _, _ = model(hg, 16)
    resG[lab] = {"smallest": [mode(n_eps(E0x, r_x, 16, e)[:12], "smallest")[0] for e in (1, 2, 3, 4)],
                 "largest": [mode(n_eps(E0x, r_x, 16, e)[:12], "largest")[0] for e in (1, 2, 3, 4)]}
resG["A_shift_eps1to3_only_smallest"] = max(resG["A"]["smallest"][:3])-min(resG["A"]["smallest"][:3])
resG["A_shift_eps1to4_smallest"] = max(resG["A"]["smallest"])-min(resG["A"]["smallest"])
resG["A_shift_eps1to4_largest"] = max(resG["A"]["largest"])-min(resG["A"]["largest"])
resG["B_shift_eps1to4_largest"] = max(resG["B"]["largest"])-min(resG["B"]["largest"])
resG["AB_mode_differs_some_eps_largest_rule"] = bool(any(a != b for a, b in zip(resG["A"]["largest"], resG["B"]["largest"])))
out["G_tie_rule_eps_range"] = resG

# H
cnt = {"nondecreasing": 0, "shift_ge1": 0, "both": 0, "mode_unchanged_eps0_vs_eps4": 0, "constant_selection_fn": 0}
Ntab = 200
for _ in range(Ntab):
    hg = np.sort(rng.integers(0, 6, size=10))
    E0x, r_x, dEx, dkx = model(hg, 16)
    modes = [mode(n_eps(E0x, r_x, 16, e)[:12])[0] for e in (0, 1, 2, 3, 4)]
    nd = all(modes[i] <= modes[i+1] for i in range(1, 4))
    sh = (max(modes[1:])-min(modes[1:])) >= 1
    cnt["nondecreasing"] += int(nd); cnt["shift_ge1"] += int(sh); cnt["both"] += int(nd and sh)
    cnt["mode_unchanged_eps0_vs_eps4"] += int(modes[0] == modes[4])
    w1 = n_eps(E0x, r_x, 16, 1); f = np.array([w1[E]/dEx[E] for E in range(12) if dEx[E] > 0])
    cnt["constant_selection_fn"] += int(np.max(np.abs(f-f.mean())) < TOL)
out["H_discriminability_200_random_mg"] = {k: v/Ntab for k, v in cnt.items()}

# I
def herm(n):
    M = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)); return (M+M.conj().T)/2
G = herm(DS)
QE = {E: np.diag((E0 == E).astype(float)) for E in range(12)}
resI = {}
for theta in (0.0, 0.05, 0.2, 0.5, 1.0):
    w, v = np.linalg.eigh(G); U = (v*np.exp(1j*theta*w)) @ v.conj().T
    Qd = {E: U @ QE[E] @ U.conj().T for E in range(12)}
    dev = {}
    for eps in (1, 2, 3, 4):
        Ediag = np.diag((r_of >= eps).astype(float))
        page = np.array([np.trace(Ediag @ Qd[E]).real for E in range(12)])
        lued = np.array([np.trace(Qd[E] @ Ediag @ Qd[E] @ Ediag).real for E in range(12)])
        page /= page.sum(); lued /= lued.sum()
        comm = float(max(np.linalg.norm(Ediag @ Qd[E]-Qd[E] @ Ediag, 2) for E in range(12)))  # max_E ||[E,Q_E]|| (sum over E is trivially 0)
        dev[str(eps)] = {"max_abs_diff_page_vs_lueders": float(np.max(np.abs(page-lued))),
                         "mode_page": mode(page)[0], "mode_lueders": mode(lued)[0], "norm_E_Pi_commutator": comm}
    resI[str(theta)] = dev
out["I_interaction_dressing"] = resI

out["seed"] = SEED
js = json.dumps(out, ensure_ascii=False, indent=1)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_result.json"), "w", encoding="utf-8").write(js)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
print(js)
