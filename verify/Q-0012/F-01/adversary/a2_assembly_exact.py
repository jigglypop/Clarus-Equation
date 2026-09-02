"""a2: independent check of the ASSEMBLY step (ladder 3) and of the lattice rationals.

Claim under test:  E ||Phi||^2 = 2 T2 D + kappa4 T4 S_gen,  Phi_ij = zeta^T [B kron M^ij] zeta,
B = A^T H A,  D = tr(B^2) = ||H kappa H||_F^2,  S_gen = sum_u B_uu^2.

The fourth moment tensor is used RAW (no Mathai-Provost citation):
    E[x_p x_q x_r x_s] = d_pq d_rs + d_pr d_qs + d_ps d_qr + kappa4 d_pqrs   (iid, mean 0, var 1)
    E[(x^T W x)^2] = (tr W)^2 + 2 tr(W^2) + kappa4 sum_p W_pp^2   for symmetric W.
Full 16n x 16n matrices W^ij are built explicitly; nothing is assumed about Kronecker structure.
"""
import json, sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import (linear_map, quadratic_tensor, caterpillar, ancestor_matrix,
                            aligned_cell, uniform_to_label, normal_cdf, KAPPA4)

OUT = Path(__file__).parent


def exact_moment_prediction(B, M, k4):
    total = 0.0
    for i in range(3):
        for j in range(3):
            W = np.kron(B, M[:, :, i, j])
            assert np.max(np.abs(W - W.T)) < 1e-12
            total += np.trace(W) ** 2 + 2 * np.trace(W @ W) + k4 * float(np.sum(np.diag(W) ** 2))
    return total


def card_prediction(B, T2, T4, k4):
    D = float(np.trace(B @ B))
    S = float(np.sum(np.diag(B) ** 2))
    return 2 * T2 * D + k4 * T4 * S


def _anc(parent, u, v):
    w = v
    while w >= 0:
        if w == u:
            return True
        w = parent[w]
    return False


def frac_lattice(parent):
    n = len(parent)
    A = [[F(1) if _anc(parent, u, v) else F(0) for u in range(n)] for v in range(n)]
    H = [[F(1) - F(1, n) if i == j else F(-1, n) for j in range(n)] for i in range(n)]

    def mul(X, Y):
        return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))]
                for i in range(len(X))]
    At = [list(col) for col in zip(*A)]
    B = mul(mul(At, H), A)
    kap = mul(A, At)
    Ker = mul(mul(H, kap), H)
    S_gen = sum(B[u][u] ** 2 for u in range(n))
    S_ker = sum(Ker[v][v] ** 2 for v in range(n))
    D = sum(B[i][j] * B[j][i] for i in range(n) for j in range(n))
    D2 = sum(Ker[i][j] ** 2 for i in range(n) for j in range(n))
    return S_gen, S_ker, D, D2


def main():
    lm = linear_map()
    M = quadratic_tensor(lm)
    T2 = float((M * M).sum())
    T4 = float(sum((M[a, a] * M[a, a]).sum() for a in range(16)))
    rng = np.random.default_rng(20260902 + 4242)
    res = {"T2": T2, "T4": T4}

    worst = 0.0
    for _ in range(20):
        x, y = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        h = 1e-5
        dx = (aligned_cell(x, h) - aligned_cell(x, -h)) / (2 * h)
        dy = (aligned_cell(y, h) - aligned_cell(y, -h)) / (2 * h)
        dxy = (aligned_cell(x + y, h) - aligned_cell(x + y, -h)) / (2 * h)
        worst = max(worst, float(np.max(np.abs(dxy - dx - dy))) / float(np.max(np.abs(dxy))))
    res["L_linearity_max_rel_dev"] = worst
    print("(i) L linear in label: max rel deviation %.3e" % worst)

    rows = []
    gens = {}
    Hs = np.eye(6) - np.ones((6, 6)) / 6
    gens["iid6"] = Hs
    gens["cat3"] = (np.eye(9) - np.ones((9, 9)) / 9) @ ancestor_matrix(caterpillar(3))
    gens["random5"] = (np.eye(5) - np.ones((5, 5)) / 5) @ rng.normal(size=(5, 5))
    worst_rel = 0.0
    for name, HA in gens.items():
        B = HA.T @ HA
        for k4 in (-2.0, 0.0, 3.0, 61.0):
            a = exact_moment_prediction(B, M, k4)
            b = card_prediction(B, T2, T4, k4)
            rel = abs(a - b) / abs(a)
            worst_rel = max(worst_rel, rel)
            rows.append({"gen": name, "k4": k4, "raw_moment": a, "card_formula": b, "rel": rel})
    res["assembly_max_rel_dev"] = worst_rel
    print("(ii) assembly identity: max rel dev over 12 cases %.3e" % worst_rel)

    A0 = ancestor_matrix(caterpillar(3))
    q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    A1 = A0 @ q
    Hn = np.eye(9) - np.ones((9, 9)) / 9
    k0, k1 = A0 @ A0.T, A1 @ A1.T
    B0, B1 = A0.T @ Hn @ A0, A1.T @ Hn @ A1
    res["same_kappa_test"] = {
        "kappa_max_diff": float(np.max(np.abs(k0 - k1))),
        "D_diff": float(abs(np.trace(B0 @ B0) - np.trace(B1 @ B1))),
        "S_gen_A0": float(np.sum(np.diag(B0) ** 2)),
        "S_gen_A1": float(np.sum(np.diag(B1) ** 2)),
    }
    print("(iii) same kappa rotated generator: kappa diff %.2e  D diff %.2e  S_gen %.4f vs %.4f"
          % (res["same_kappa_test"]["kappa_max_diff"], res["same_kappa_test"]["D_diff"],
             res["same_kappa_test"]["S_gen_A0"], res["same_kappa_test"]["S_gen_A1"]))

    lat = {}
    for k in (3, 6):
        S_gen, S_ker, D, D2 = frac_lattice(caterpillar(k))
        lat["cat%d" % k] = {"S_gen": str(S_gen), "S_ker": str(S_ker), "D_trB2": str(D),
                            "D_kernelnorm": str(D2), "S_gen_float": float(S_gen),
                            "S_ker_float": float(S_ker), "D_float": float(D)}
        print("(iv) caterpillar k=%d: S_gen=%s  S_ker=%s  D=tr(B^2)=%s  ||HkH||^2=%s"
              % (k, S_gen, S_ker, D, D2))
    closed = sum(i * i * (6 - i) ** 2 for i in range(1, 6)) + F(30) * (1 - F(1, 36)) ** 2
    lat["cat6_closed_form"] = str(closed)
    lat["cat6_card_values"] = {"S_gen": "62069/216", "S_ker": "54023/432", "D": "23053/36"}
    lat["cat6_card_match"] = bool(closed == F(62069, 216))
    res["lattices"] = lat
    print("     closed form %s == 62069/216 ? %s" % (closed, closed == F(62069, 216)))

    mc = {}
    for name, parent in (("iid6", None), ("cat3", caterpillar(3))):
        nn = 6 if parent is None else 9
        Hn2 = np.eye(nn) - np.ones((nn, nn)) / nn
        HA = Hn2 if parent is None else Hn2 @ ancestor_matrix(parent)
        B = HA.T @ HA
        for dist in ("gauss", "rademacher", "laplace", "spike64"):
            r2 = np.random.default_rng(777)
            acc = np.empty(30000)
            for t in range(30000):
                z = r2.standard_normal((nn, 16))
                zeta = uniform_to_label(normal_cdf(z), z, dist)
                cen = HA @ zeta
                phi = np.einsum("va,vb,abij->ij", cen, cen, M)
                acc[t] = float(np.sum(phi * phi))
            obs = float(acc.mean()); se = float(acc.std(ddof=1) / np.sqrt(len(acc)))
            pred = card_prediction(B, T2, T4, KAPPA4[dist])
            mc["%s_%s" % (name, dist)] = {"obs": obs, "se": se, "pred": pred, "z": (obs - pred) / se}
            print("(v) %-5s %-11s MC %12.5f +- %9.5f   analytic %12.5f   z=%+6.2f"
                  % (name, dist, obs, se, pred, (obs - pred) / se))
    res["phi_mc"] = mc
    res["assembly_rows"] = rows
    (OUT / "a2_assembly_exact.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
