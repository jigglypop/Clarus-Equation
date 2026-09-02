"""adversary b1 (F-02 card audit): independent reconstruction of tl G, its kernel,
spin-2 multiplicity and the T budget.

L~ is rebuilt from the real pipeline by central differences + Richardson (no card formula).
The card's W_k, X_lm rule is re-coded here from the card text (check_floor.card_W_X unused).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import (
    geometric_self_dual_triple, wedge_scalar)
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
REF = geometric_self_dual_triple(np.eye(4))
E3 = np.eye(3)
P = [0.5 * (np.outer(E3[k], E3[k]) - E3 / 3.0) for k in range(3)]
S = {"12": np.outer(E3[0], E3[1]) + np.outer(E3[1], E3[0]),
     "23": np.outer(E3[1], E3[2]) + np.outer(E3[2], E3[1]),
     "31": np.outer(E3[2], E3[0]) + np.outer(E3[0], E3[2])}
NAMES = ["%d%d" % (a // 4, a % 4) for a in range(16)]
CLASS = {1: ["01", "10", "23", "32"], 2: ["02", "20", "31", "13"], 3: ["03", "30", "12", "21"]}
DIAG = ["00", "11", "22", "33"]


def idx(s):
    return 4 * int(s[0]) + int(s[1])


def cell(label, delta):
    triple = geometric_self_dual_triple(np.eye(4) + delta * label)
    return optimal_internal_alignment(REF, triple).aligned_candidate


def l_tilde_fd(label, h):
    return (cell(label, h) - cell(label, -h)) / (2.0 * h)


def l_tilde_richardson(label, h=2.0e-4):
    a = l_tilde_fd(label, h)
    b = l_tilde_fd(label, h / 2.0)
    return (4.0 * b - a) / 3.0


def build_M():
    basis = []
    for a in range(16):
        m = np.zeros((4, 4))
        m[a // 4, a % 4] = 1.0
        basis.append(m)
    L = [l_tilde_richardson(b) for b in basis]
    M = np.zeros((16, 16, 3, 3))
    for a in range(16):
        for b in range(16):
            g = np.array([[wedge_scalar(L[a][i], L[b][j]) for j in range(3)] for i in range(3)])
            M[a, b] = 0.5 * (g + g.T)
    Mt = M - np.einsum("abii->ab", M)[:, :, None, None] * np.eye(3) / 3.0
    return M, Mt, L


def card_rule(sigma):
    """Card step 3 rule, re-coded from the card text."""
    def Sg(a, b):
        return sigma[idx(a), idx(b)]
    cyc = {1: (2, 3), 2: (3, 1), 3: (1, 2)}
    dpairs = {1: [("00", "11"), ("22", "33")], 2: [("00", "22"), ("11", "33")],
              3: [("00", "33"), ("11", "22")]}
    w, p, D, C = {}, {}, {}, {}
    for k in (1, 2, 3):
        ts = [a for a in CLASS[k] if "0" in a]
        ss = [a for a in CLASS[k] if "0" not in a]
        w[k] = sum(Sg(a, a) for a in CLASS[k])
        p[k] = Sg(ts[0], ts[1]) + Sg(ss[0], ss[1])
        D[k] = sum(Sg(x, y) for x, y in dpairs[k])
        C[k] = sum(Sg(a, b) for a in ts for b in ss)
    W = np.zeros(3)
    for k in (1, 2, 3):
        l, m = cyc[k]
        W[k - 1] = w[k] + 2 * p[k] - 4 * D[k] + 2 * (C[m] - C[l])
    X = {}
    for k in (1, 2, 3):
        l, m = cyc[k]
        key = "%d%d" % (l, m) if "%d%d" % (l, m) in S else "%d%d" % (m, l)
        val = 2.0 * sum((1.0 if "0" in a else -1.0) * Sg(a, b) for a in CLASS[l] for b in CLASS[m])
        for mu in range(4):
            t = -1.0 if mu in (0, l) else 1.0
            for b in CLASS[k]:
                if str(mu) in b:
                    continue
                val += 4.0 * t * Sg("%d%d" % (mu, mu), b)
        X[key] = val
    out = sum(W[k] * P[k] for k in range(3))
    for key, val in X.items():
        out = out + 0.25 * val * S[key]
    return W, X, out


def sym_basis():
    B = []
    for a in range(16):
        m = np.zeros((16, 16))
        m[a, a] = 1.0
        B.append(m)
    for a in range(16):
        for b in range(a + 1, 16):
            m = np.zeros((16, 16))
            m[a, b] = m[b, a] = 1.0 / math.sqrt(2.0)
            B.append(m)
    return B


def so3_generators16():
    j = []
    plus = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    minus = [(0, 2, 1), (2, 1, 0), (1, 0, 2)]
    for a in range(3):
        g = np.zeros((4, 4))
        for i in range(3):
            for k in range(3):
                eps = 1.0 if (a, i, k) in plus else (-1.0 if (a, i, k) in minus else 0.0)
                g[i + 1, k + 1] = -eps
        j.append(g)
    return [np.kron(g, np.eye(4)) + np.kron(np.eye(4), g) for g in j], j


def main():
    from fractions import Fraction
    import importlib.util
    M, Mt, L = build_M()
    spec = importlib.util.spec_from_file_location("cf", ROOT / "verify/Q-0013/F-02/check_floor.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    M_card, Mt_card = cf.structure_constants()
    M_vs_card = float(np.max(np.abs(M - M_card)))

    B5 = np.array([P[0].reshape(9), P[1].reshape(9), S["12"].reshape(9),
                   S["23"].reshape(9), S["31"].reshape(9)]).T
    basis = sym_basis()
    Kfull = np.zeros((9, len(basis)))
    for n_, Sig in enumerate(basis):
        Kfull[:, n_] = np.einsum("ab,abij->ij", Sig, Mt).reshape(9)
    sv = np.linalg.svd(Kfull, compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * max(sv)))
    dim_ker = len(basis) - rank

    rng = np.random.default_rng(20260903)
    err_rule, max_trace = 0.0, 0.0
    for t in range(1000):
        Z = rng.normal(size=(16, 16))
        Sig = (Z + Z.T) / 2.0 if t < 500 else Z @ Z.T
        W, X, tl_rule = card_rule(Sig)
        direct = np.einsum("ab,abij->ij", Sig, Mt)
        scale = max(1.0, float(np.linalg.norm(direct)))
        err_rule = max(err_rule, float(np.linalg.norm(tl_rule - direct)) / scale)
        max_trace = max(max_trace, abs(float(np.trace(direct))))

    C5 = np.zeros((5, len(basis)))
    for n_, Sig in enumerate(basis):
        W, X, _ = card_rule(Sig)
        C5[:, n_] = [W[0] - W[1], W[1] - W[2], X["12"], X["23"], X["31"]]
    rank_C = int(np.linalg.matrix_rank(C5, tol=1e-9))
    NK = np.linalg.svd(Kfull)[2][rank:].T
    NC = np.linalg.svd(C5)[2][rank_C:].T
    ang = np.linalg.svd(NK.T @ NC, compute_uv=False)
    same_subspace = bool(NK.shape[1] == NC.shape[1] and abs(float(ang.min()) - 1.0) < 1e-8)

    rho, j4 = so3_generators16()
    A = []
    for r in rho:
        Amat = np.zeros((len(basis), len(basis)))
        for n_, Sig in enumerate(basis):
            img = r @ Sig + Sig @ r.T
            Amat[:, n_] = [float(np.sum(b * img)) for b in basis]
        A.append(Amat)
    cas = sum(a @ a for a in A)
    ev = np.linalg.eigvalsh(cas)
    spins = {}
    for x in ev:
        jj = (-1.0 + math.sqrt(1.0 + 4.0 * max(-float(x), 0.0))) / 2.0
        key = str(int(round(jj)))
        spins[key] = spins.get(key, 0) + 1
    mult = {k: v // (2 * int(k) + 1) for k, v in spins.items()}

    tb = [P[0], P[1], S["12"], S["23"], S["31"]]
    Q_, _ = np.linalg.qr(np.array([t.reshape(9) for t in tb]).T)
    tb = [Q_[:, i].reshape(3, 3) for i in range(5)]
    sig_gen = []
    for a in range(3):
        q = j4[a][1:, 1:]
        g = np.zeros((5, 5))
        for n_, t in enumerate(tb):
            img = q @ t + t @ q.T
            g[:, n_] = [float(np.sum(u * img)) for u in tb]
        sig_gen.append(g)
    rows = [np.kron(np.eye(5), A[a].T) - np.kron(sig_gen[a], np.eye(len(basis))) for a in range(3)]
    svB = np.linalg.svd(np.vstack(rows), compute_uv=False)
    inter_dim = int(np.sum(svB < 1e-8 * max(svB)))

    D_idx = [idx(a) for a in DIAG]
    O_idx = [a for a in range(16) if a not in D_idx]
    T_DD = float(sum(np.sum(Mt[a, b] ** 2) for a in D_idx for b in D_idx))
    T_OO = float(sum(np.sum(Mt[a, b] ** 2) for a in O_idx for b in O_idx))
    T_DO = float(sum(np.sum(Mt[a, b] ** 2) for a in D_idx for b in O_idx))
    T_all = float(np.einsum("abij,abij->", Mt, Mt))
    nonzero = sum(1 for a in range(16) for b in range(a, 16) if np.linalg.norm(Mt[a, b]) > 1e-7)
    max_rat_err = 0.0
    for a in range(16):
        for b in range(a, 16):
            if np.linalg.norm(Mt[a, b]) < 1e-7:
                continue
            c, *_ = np.linalg.lstsq(B5, Mt[a, b].reshape(9), rcond=None)
            for x in c:
                max_rat_err = max(max_rat_err, abs(float(Fraction(float(x)).limit_denominator(4)) - float(x)))

    ns = 2.0 * math.sqrt(3.0)
    res = {
        "L_tilde_source": "independent central difference + Richardson on the real pipeline",
        "M_richardson_vs_card_analytic_maxdiff": M_vs_card,
        "map_rank": rank, "dim_kernel_recomputed": dim_ker, "sym2_dim": len(basis),
        "rule_vs_direct_max_relerr": err_rule, "max_trace_of_tlG": max_trace,
        "card5_rank": rank_C, "kernel_subspace_identical": same_subspace,
        "principal_angle_min_cos": float(ang.min()),
        "spin_content_dims": spins, "spin_multiplicities": mult,
        "spin2_multiplicity": mult.get("2"),
        "intertwiner_dim_Hom_Sym2R16_spin2": inter_dim,
        "nonzero_tlM_unordered_pairs": nonzero,
        "max_rational_err_denom_le_4": max_rat_err,
        "T_I16": T_all, "T_DD": T_DD, "T_OO": T_OO, "twice_T_DO": 2 * T_DO,
        "budget_sum": T_DD + T_OO + 2 * T_DO,
        "eps_star_I16_over_delta2": math.sqrt(2 * T_all) / ns,
        "eps_star_I4_over_delta2": math.sqrt(2 * T_DD) / ns,
        "eps_star_I12_over_delta2": math.sqrt(2 * T_OO) / ns,
        "two_over_sqrt3": 2 / math.sqrt(3), "sqrt_14_over_3": math.sqrt(14 / 3),
        "sqrt10": math.sqrt(10),
        "nonadditivity_delta4": (2 * T_all - 2 * T_DD - 2 * T_OO) / 12.0,
        "sum_a_tlM_aa_norm": float(np.linalg.norm(sum(Mt[a, a] for a in range(16)))),
    }
    (OUT / "b1_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(OUT / "b1_Mt.npy", Mt)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
