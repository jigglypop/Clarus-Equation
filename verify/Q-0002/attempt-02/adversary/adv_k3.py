"""adv_k3.py -- adversary for Q-0002 attempt-02 (K3). numpy only. seed 20260902.
Reuses Model from check_k3.py (import only; main is guarded).
Tests A..I, see adv_result.json keys.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from check_k3 import Model, P_REG, block_diag_part, off_diag_part, comm, opnorm, I2, I4, SX, SZ, NQ, NR, kron3  # noqa

SEED = 20260902
rng = np.random.default_rng(SEED)
out = {}

def frame_diff(m, c, tau):
    return m.ledger(c, 1, tau), m.ledger(c, 2, tau)

def random_unitary(n):
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(Z)
    return Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))

# ---------------- A: representative dependence of E_unseen ----------------
HS_B = np.kron(1.0 * I2 + 3.0 * SX + 4.0 * SZ, NQ)
N = 13
mB = Model(N, HS_B)
e_hi = int(np.argmax(mB.eps)); e_lo = int(np.argmin(mB.eps))
Pe_hi = mB.W[:, [e_hi]] @ mB.W[:, [e_hi]].conj().T
Pe_lo = mB.W[:, [e_lo]] @ mB.W[:, [e_lo]].conj().T
states = [mB.random_phys(rng) for _ in range(20)]
resA = {}
for label, K in (("K=P_hi", Pe_hi), ("K=P_lo", Pe_lo), ("K=P_hi-P_lo", Pe_hi - Pe_lo), ("K=1_uniform", I4)):
    HSs = HS_B + N * K
    ms = Model(N, HSs)
    dE_un, dFD, dE_tot, frac = [], [], [], []
    for c in states:
        cs = ms.phys_from_kin(mB.B @ c)
        L1, L2 = frame_diff(mB, c, 0.0)
        L1s, L2s = frame_diff(ms, cs, 0.0)
        d = L1s["E_unseen"] - L1["E_unseen"]
        dE_un.append(d)
        dFD.append((L1s["E_unseen"] - L2s["E_unseen"]) - (L1["E_unseen"] - L2["E_unseen"]))
        dE_tot.append(L1s["E_total"] - L1["E_total"])
        frac.append(abs(d / N - round(d / N)))
    resA[label] = {"Pi_residual": opnorm(ms.Pi - mB.Pi), "comm_K_HS": opnorm(comm(K, HS_B)), "eps_new": ms.eps.tolist(),
                   "max_abs_dE_unseen": float(np.max(np.abs(dE_un))), "max_abs_d_frame_diff": float(np.max(np.abs(dFD))),
                   "max_abs_dE_total": float(np.max(np.abs(dE_tot))),
                   "dE_unseen_over_N_fractional_max": float(np.max(frac)),
                   "norm_X_of_K": opnorm(off_diag_part(K))}
out["A_representative_dependence_of_E_unseen"] = resA

# ---------------- H: integer K scan ----------------
resH = {}
for k in range(-3, 4):
    HSs = HS_B + N * k * Pe_hi
    ms = Model(N, HSs)
    Dm = ms.W.conj().T @ block_diag_part(HSs) @ ms.W
    i_hi = int(np.argmax(ms.eps)); i_lo = int(np.argmin(ms.eps))
    resH["m=%d" % k] = {"eps": ms.eps.tolist(), "within_half_N": bool(np.max(np.abs(ms.eps)) < N / 2),
                        "abs_Dprime_hi_lo": float(abs(Dm[i_hi, i_lo])), "norm_comm_Dprime_Hprime": opnorm(comm(block_diag_part(HSs), HSs))}
resH["analytic"] = "Dprime_vw = (gh/r)[-1 + N(m2-m1)/(2r)], zero iff N(m2-m1)=2r; impossible under |eps|<N/2 since 2r<N"
out["H_integer_K_scan"] = resH

# ---------------- B: A2 partition entries frame-dependent ----------------
HS_A2 = np.kron(I2, NQ) + 2.0 * np.kron(SX, NQ)
mA2 = Model(7, HS_A2)
statesA2 = [mA2.random_phys(rng) for _ in range(20)]
wp = wE = wu = ws = 0.0
for c in statesA2:
    for tau in mA2.taus():
        L1, L2 = frame_diff(mA2, c, tau)
        wp = max(wp, abs(L1["p"][0] - L2["p"][0]))
        wE = max(wE, abs(L1["E_b"][0] - L2["E_b"][0]), abs(L1["E_b"][1] - L2["E_b"][1]))
        wu = max(wu, abs(L1["E_unseen"] - L2["E_unseen"]))
        ws = max(ws, abs(L1["E_seen"] - L2["E_seen"]))
out["B_A2_partition_entries"] = {"max_abs_p0_diff": wp, "max_abs_Eb_diff": wE, "max_abs_E_unseen_diff": wu, "max_abs_E_seen_diff": ws,
                                 "comm_Pb_HS": max(opnorm(comm(P, HS_A2)) for P in P_REG),
                                 "comm_D_HS": opnorm(comm(block_diag_part(HS_A2), HS_A2)),
                                 "note": "S7.1 says the partition is frame-independent when [D,H_S]=0; only the sums are; p_b, E_b are not"}

# ---------------- C: N1 != N2 ----------------
def model_N1N2(N1, N2, HS, L):
    eps, W = np.linalg.eigh(HS); eps = np.round(eps).astype(int)
    K1 = np.arange(N1); K2 = np.arange(N2)
    lam = (K1[:, None, None] + K2[None, :, None] + eps[None, None, :]).reshape(-1)
    Wfull = kron3(np.eye(N1), np.eye(N2), W)
    mask = (np.mod(lam, L) == 0)
    B = Wfull[:, mask]
    bra1 = np.ones(N1) / np.sqrt(N1)
    V1 = np.sqrt(N1) * kron3(bra1[None, :], np.eye(N2), I4) @ B
    bra2 = np.ones(N2) / np.sqrt(N2)
    V2 = np.sqrt(N2) * kron3(np.eye(N1), bra2[None, :], I4) @ B
    d = int(mask.sum())
    return {"L": L, "phys_dim": d, "4N1": 4 * N1, "4N2": 4 * N2,
            "V1dagV1_minus_1": opnorm(V1.conj().T @ V1 - np.eye(d)), "V1V1dag_minus_1": opnorm(V1 @ V1.conj().T - np.eye(4 * N2)),
            "V2dagV2_minus_1": opnorm(V2.conj().T @ V2 - np.eye(d)), "V2V2dag_minus_1": opnorm(V2 @ V2.conj().T - np.eye(4 * N1)),
            "lam_range": [int(lam.min()), int(lam.max())]}
out["C_N1_ne_N2"] = {"N1=7,N2=13,L=91": model_N1N2(7, 13, HS_B, 91), "N1=7,N2=13,L=7": model_N1N2(7, 13, HS_B, 7),
                     "N1=7,N2=14,L=14": model_N1N2(7, 14, HS_B, 14), "N1=13,N2=13,L=13_control": model_N1N2(13, 13, HS_B, 13)}

# ---------------- D: 20 random integer-spectrum H_S ----------------
resD = []
Nd = 11
crit_ok = c1_ok = c2_ok = two_ok = True
for t in range(20):
    eps = rng.integers(-5, 6, size=4)
    if t % 4 == 0:
        U = np.zeros((4, 4), dtype=complex)
        U[0:2, 0:2] = random_unitary(2); U[2:4, 2:4] = random_unitary(2)   # register-diagonal blocks: [P_b,H_S]=0
    else:
        U = random_unitary(4)
    HS = U @ np.diag(eps.astype(complex)) @ U.conj().T
    HS = (HS + HS.conj().T) / 2
    m = Model(Nd, HS)
    cs = [m.random_phys(rng) for _ in range(20)]
    HSk = kron3(m.IN, m.IN, m.HS)
    w1 = w2 = maxfd = 0.0
    for c in cs:
        psi = m.B @ c
        Ek = float((psi.conj() @ HSk @ psi).real)
        for tau in m.taus():
            L1, L2 = frame_diff(m, c, tau)
            for L in (L1, L2):
                w1 = max(w1, abs(L["E_total"] - Ek)); w2 = max(w2, abs(L["E_unseen"] - L["E_unseen_formula"]))
            maxfd = max(maxfd, abs(L1["E_unseen"] - L2["E_unseen"]))
    cD = opnorm(comm(block_diag_part(HS), HS))
    Dm = m.W.conj().T @ block_diag_part(HS) @ m.W
    pair = None
    for e in range(4):
        for ep in range(4):
            if m.eps[e] != m.eps[ep] and abs(Dm[e, ep]) > 1e-8:
                pair = (e, ep); break
        if pair:
            break
    tl = None
    if pair:
        e, ep = pair; k2 = 3
        k1 = int((-k2 - m.eps[e]) % Nd); k1p = int((-k2 - m.eps[ep]) % Nd)
        a = 0.6 + 0.3j; b = np.sqrt(1 - abs(a) ** 2)
        idx = lambda kk1, kk2, ee: int(np.where(np.where(m.mask)[0] == (kk1 * Nd + kk2) * 4 + ee)[0][0])
        c2 = np.zeros(m.d, dtype=complex); c2[idx(k1, k2, e)] = a; c2[idx(k1p, k2, ep)] = b
        L1, L2 = frame_diff(m, c2, 0.0)
        pred = float(2 * (np.conj(a) * b * Dm[e, ep]).real)
        tl = {"observed": L1["E_seen"] - L2["E_seen"], "predicted": pred}
        two_ok = two_ok and abs(tl["observed"] - pred) < 1e-10
    frame_dep = maxfd > 1e-3
    consistent = (cD > 1e-8) == frame_dep
    crit_ok = crit_ok and consistent; c1_ok = c1_ok and w1 < 1e-10; c2_ok = c2_ok and w2 < 1e-10
    resD.append({"eps": sorted(m.eps.tolist()), "comm_D_HS": cD, "max_frame_diff": maxfd, "claim1_res": w1, "claim2_res": w2,
                 "criterion_consistent": bool(consistent), "two_level": tl})
out["D_random_HS_20"] = {"all_claim1": c1_ok, "all_claim2": c2_ok, "criterion_iff_consistent_all": crit_ok, "two_level_formula_all": two_ok,
                        "n_with_D_commuting": int(sum(1 for r in resD if r["comm_D_HS"] < 1e-8)), "rows": resD}

# ---------------- E: off-lattice tau ----------------
tau_off = 0.37
c = states[0]
L1 = mB.ledger(c, 1, tau_off); psi = mB.B @ c
Ek = float((psi.conj() @ kron3(mB.IN, mB.IN, mB.HS) @ psi).real)
V = mB.V(1, tau_off)
w, Q = np.linalg.eigh(mB.HjS[1]); Ut = Q @ np.diag(np.exp(-1j * w * tau_off)) @ Q.conj().T
Hred = mB.B.conj().T @ mB.Hfull @ mB.B
out["E_offlattice_tau"] = {"claim1_offlattice_res": abs(L1["E_total"] - Ek), "V_unitary_offlattice_res": opnorm(V.conj().T @ V - np.eye(mB.d)),
                          "S3_2_offlattice_res": float(np.linalg.norm(V @ c - Ut @ (mB.V(1, 0.0) @ c))),
                          "E_unseen_offlattice": L1["E_unseen"], "mean_winding_state0": float((c.conj() @ Hred @ c).real / N)}

# ---------------- F: transport covariance check is tautological ----------------
worst = 0.0
for c in states[:10]:
    Uf = random_unitary(4 * N)
    V1 = mB.V(1, 0.0)
    Vfake = Uf @ V1
    Phi = Vfake @ V1.conj().T
    v1 = V1 @ c; v2 = Vfake @ c
    for X in (off_diag_part(HS_B), P_REG[0], P_REG[0] @ HS_B @ P_REG[0]):
        Y = np.kron(mB.IN, X)
        worst = max(worst, abs((v1.conj() @ Y @ v1).real - (v2.conj() @ (Phi @ Y @ Phi.conj().T) @ v2).real))
out["F_transport_tautology"] = {"random_unitary_as_frame2_transported_residual": worst,
                                "note": "Phi := V2 V1dag gives <V2 c|Phi Y Phidag|V2 c> = <V1 c|Y|V1 c> for ANY unitary V2; K3_4b has no content beyond the definition of Phi"}

# ---------------- G: E39 content ----------------
HS_E39 = 1.0 * (np.kron(I2, NQ) + np.kron(I2 - NR, I2))
out["G_E39"] = {"comm_Pb_HSprime": max(opnorm(comm(P, HS_E39)) for P in P_REG),
                "HSprime_diag_in_rq_basis": bool(opnorm(HS_E39 - np.diag(np.diag(HS_E39))) < 1e-12),
                "eps": sorted(np.round(np.linalg.eigvalsh(HS_E39)).astype(int).tolist())}

# ---------------- I: mirror two-level state (clock-1 definite) ----------------
k1 = 4
k2 = int((-k1 - mB.eps[e_hi]) % N); k2p = int((-k1 - mB.eps[e_lo]) % N)
idxB = lambda kk1, kk2, ee: int(np.where(np.where(mB.mask)[0] == (kk1 * N + kk2) * 4 + ee)[0][0])
a = 0.6 + 0.3j; b = np.sqrt(1 - abs(a) ** 2)
c2 = np.zeros(mB.d, dtype=complex); c2[idxB(k1, k2, e_hi)] = a; c2[idxB(k1, k2p, e_lo)] = b
L1, L2 = frame_diff(mB, c2, 0.0)
out["I_mirror_state"] = {"E_seen1": L1["E_seen"], "E_seen2": L2["E_seen"], "diff_1_minus_2": L1["E_seen"] - L2["E_seen"],
                         "note": "clock-1-definite state: observer 1 decoheres, observer 2 sees the interference; sign flips vs S7.5"}

with open(os.path.join(HERE, "adv_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=float)
print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
