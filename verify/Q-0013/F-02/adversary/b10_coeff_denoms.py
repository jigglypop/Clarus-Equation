"""adversary b10: exact denominators of the 108 coefficient rows in the (P1,P2,S12,S23,S31)
basis (card claims denominator <= 4), plus the exact Sigma_b / Sigma_o coordinates.
"""
from __future__ import annotations
import json, sys
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT / "verify/Q-0013/F-02/adversary"))
import b7_exact_rational as X

OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
Z, ONE = Fr(0), Fr(1)


def main():
    basis = []
    for a in range(16):
        m = [[Z] * 4 for _ in range(4)]
        m[a // 4][a % 4] = ONE
        basis.append(m)
    L = [X.l_tilde(b) for b in basis]
    rows, dens = {}, set()
    for a in range(16):
        for b in range(a, 16):
            g = [[X.wedge(L[a][i], L[b][j]) for j in range(3)] for i in range(3)]
            sym = [[(g[i][j] + g[j][i]) / 2 for j in range(3)] for i in range(3)]
            tr = sym[0][0] + sym[1][1] + sym[2][2]
            A = [[sym[i][j] - (tr / 3 if i == j else Z) for j in range(3)] for i in range(3)]
            if all(A[i][j] == 0 for i in range(3) for j in range(3)):
                continue
            # A = c1 P1 + c2 P2 + s12 S12 + s23 S23 + s31 S31, P3 = -P1-P2
            # diagonal: A_kk = (1/2)(W_k - Wbar) with W = (c1, c2, 0) shifted; solve directly
            c1 = 2 * (A[0][0] - A[2][2])
            c2 = 2 * (A[1][1] - A[2][2])
            s = [A[0][1], A[1][2], A[2][0]]
            coeffs = [c1, c2] + s
            for c in coeffs:
                dens.add(c.denominator)
            rows["%d%d,%d%d" % (a // 4, a % 4, b // 4, b % 4)] = [str(c) for c in coeffs]
    res = {"nonzero_rows": len(rows), "coefficient_denominators": sorted(dens),
           "max_denominator": max(dens), "card_claims_denominator_le_4": bool(max(dens) <= 4),
           "sample_rows": {k: rows[k] for k in list(rows)[:8]}}
    (OUT / "b10_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
