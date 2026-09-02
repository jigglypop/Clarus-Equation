"""adversary b7: exact rational recomputation of the structure constants and of the
card step-3 rule (no floating point).
"""
from __future__ import annotations
import json, sys
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
PAIR = ((0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2))
EPS = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
Z, ONE = Fr(0), Fr(1)


def wf(u, v):
    return [u[i] * v[j] - u[j] * v[i] for i, j in PAIR]


def wedge(f, s):
    return (f[0] * s[3] + f[3] * s[0] + f[1] * s[4] + f[4] * s[1] + f[2] * s[5] + f[5] * s[2])


def triple(e):
    out = []
    for i in range(3):
        f = wf(e[0], e[i + 1])
        for j in range(3):
            for k in range(3):
                s = EPS.get((i, j, k), 0)
                if s:
                    g = wf(e[j + 1], e[k + 1])
                    f = [a + Fr(s, 2) * b for a, b in zip(f, g)]
        out.append(f)
    return out


I4 = [[ONE if i == j else Z for j in range(4)] for i in range(4)]
REF = triple(I4)


def d_linear(l):
    out = []
    for i in range(3):
        a = wf(l[0], I4[i + 1])
        b = wf(I4[0], l[i + 1])
        f = [x + y for x, y in zip(a, b)]
        for j in range(3):
            for k in range(3):
                s = EPS.get((i, j, k), 0)
                if s:
                    g1 = wf(l[j + 1], I4[k + 1])
                    g2 = wf(I4[j + 1], l[k + 1])
                    f = [x + Fr(s, 2) * (y + z) for x, y, z in zip(f, g1, g2)]
        out.append(f)
    return out


def l_tilde(l):
    d = d_linear(l)
    c1 = [[wedge(REF[i], d[j]) for j in range(3)] for i in range(3)]
    anti = [[(c1[i][j] - c1[j][i]) / 4 for j in range(3)] for i in range(3)]
    out = []
    for i in range(3):
        f = list(d[i])
        for j in range(3):
            if anti[i][j]:
                f = [x + anti[i][j] * y for x, y in zip(f, REF[j])]
        out.append(f)
    return out


CLASS = {1: ["01", "10", "23", "32"], 2: ["02", "20", "31", "13"], 3: ["03", "30", "12", "21"]}
CYC = {1: (2, 3), 2: (3, 1), 3: (1, 2)}
DPAIRS = {1: [("00", "11"), ("22", "33")], 2: [("00", "22"), ("11", "33")],
          3: [("00", "33"), ("11", "22")]}


def ix(s):
    return 4 * int(s[0]) + int(s[1])


def rule(sig):
    def Sg(a, b):
        return sig[ix(a)][ix(b)]
    w, p, D, C = {}, {}, {}, {}
    for k in (1, 2, 3):
        ts = [a for a in CLASS[k] if "0" in a]
        ss = [a for a in CLASS[k] if "0" not in a]
        w[k] = sum(Sg(a, a) for a in CLASS[k])
        p[k] = Sg(ts[0], ts[1]) + Sg(ss[0], ss[1])
        D[k] = sum(Sg(x, y) for x, y in DPAIRS[k])
        C[k] = sum(Sg(a, b) for a in ts for b in ss)
    W = {}
    for k in (1, 2, 3):
        l, m = CYC[k]
        W[k] = w[k] + 2 * p[k] - 4 * D[k] + 2 * (C[m] - C[l])
    X = {}
    for k in (1, 2, 3):
        l, m = CYC[k]
        key = (1, 2) if {l, m} == {1, 2} else ((2, 3) if {l, m} == {2, 3} else (3, 1))
        val = 2 * sum((1 if "0" in a else -1) * Sg(a, b) for a in CLASS[l] for b in CLASS[m])
        for mu in range(4):
            t = -1 if mu in (0, l) else 1
            for b in CLASS[k]:
                if str(mu) in b:
                    continue
                val += 4 * t * Sg("%d%d" % (mu, mu), b)
        X[key] = val
    return W, X


def main():
    basis = []
    for a in range(16):
        m = [[Z] * 4 for _ in range(4)]
        m[a // 4][a % 4] = ONE
        basis.append(m)
    L = [l_tilde(b) for b in basis]
    Mt = [[None] * 16 for _ in range(16)]
    for a in range(16):
        for b in range(16):
            g = [[wedge(L[a][i], L[b][j]) for j in range(3)] for i in range(3)]
            sym = [[(g[i][j] + g[j][i]) / 2 for j in range(3)] for i in range(3)]
            tr = sym[0][0] + sym[1][1] + sym[2][2]
            Mt[a][b] = [[sym[i][j] - (tr / 3 if i == j else Z) for j in range(3)] for i in range(3)]
    dens = sorted({Mt[a][b][i][j].denominator for a in range(16) for b in range(16)
                   for i in range(3) for j in range(3)})
    nonzero = sum(1 for a in range(16) for b in range(a, 16)
                  if any(Mt[a][b][i][j] != 0 for i in range(3) for j in range(3)))
    T_all = sum(Mt[a][b][i][j] ** 2 for a in range(16) for b in range(16) for i in range(3) for j in range(3))
    D_idx = [0, 5, 10, 15]
    O_idx = [a for a in range(16) if a not in D_idx]
    T_DD = sum(Mt[a][b][i][j] ** 2 for a in D_idx for b in D_idx for i in range(3) for j in range(3))
    T_OO = sum(Mt[a][b][i][j] ** 2 for a in O_idx for b in O_idx for i in range(3) for j in range(3))
    T_DO = sum(Mt[a][b][i][j] ** 2 for a in D_idx for b in O_idx for i in range(3) for j in range(3))
    sum_diag = [[sum(Mt[a][a][i][j] for a in range(16)) for j in range(3)] for i in range(3)]

    pairs = [(a, a) for a in range(16)] + [(a, b) for a in range(16) for b in range(a + 1, 16)]
    bad, mat = [], []
    for (a, b) in pairs:
        sig = [[Z] * 16 for _ in range(16)]
        sig[a][b] = ONE
        sig[b][a] = ONE
        A = [[sum(sig[x][y] * Mt[x][y][i][j] for x in range(16) for y in range(16))
              for j in range(3)] for i in range(3)]
        W, X = rule(sig)
        wbar = (W[1] + W[2] + W[3]) / 3
        ok = all(W[k + 1] - wbar == 2 * A[k][k] for k in range(3))
        ok = ok and X[(1, 2)] == 4 * A[0][1] and X[(2, 3)] == 4 * A[1][2] and X[(3, 1)] == 4 * A[2][0]
        ok = ok and A[0][1] == A[1][0] and A[1][2] == A[2][1] and A[2][0] == A[0][2]
        ok = ok and A[0][0] + A[1][1] + A[2][2] == 0
        if not ok:
            bad.append([a, b])
        mat.append([A[0][0], A[1][1], A[0][1], A[1][2], A[2][0]])
    rowsM = [list(r) for r in mat]
    r_ = 0
    for c in range(5):
        pr = next((i for i in range(r_, len(rowsM)) if rowsM[i][c] != 0), None)
        if pr is None:
            continue
        rowsM[r_], rowsM[pr] = rowsM[pr], rowsM[r_]
        pv = rowsM[r_][c]
        rowsM[r_] = [x / pv for x in rowsM[r_]]
        for i in range(len(rowsM)):
            if i != r_ and rowsM[i][c] != 0:
                f = rowsM[i][c]
                rowsM[i] = [x - f * y for x, y in zip(rowsM[i], rowsM[r_])]
        r_ += 1
    res = {
        "exact_arithmetic": True,
        "denominators_present": dens, "max_denominator": max(dens),
        "nonzero_unordered_pairs": nonzero,
        "T_I16_exact": str(T_all), "T_DD_exact": str(T_DD), "T_OO_exact": str(T_OO),
        "twice_T_DO_exact": str(2 * T_DO),
        "budget_identity_holds": bool(T_all == T_DD + T_OO + 2 * T_DO),
        "sum_a_tlM_aa_is_zero": bool(all(sum_diag[i][j] == 0 for i in range(3) for j in range(3))),
        "card_rule_exact_failures": bad, "card_rule_exact_ok": bool(not bad),
        "map_rank_exact": r_, "dim_kernel_exact": 136 - r_,
        "eps_star_I16_sq_exact": str(2 * T_all / 12), "eps_star_I4_sq_exact": str(2 * T_DD / 12),
        "eps_star_I12_sq_exact": str(2 * T_OO / 12),
        "nonadditivity_exact": str((2 * T_all - 2 * T_DD - 2 * T_OO) / 12),
    }
    (OUT / "b7_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
