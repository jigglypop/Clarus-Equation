"""a1: independent EXACT recomputation of the Q-0012 F-01 geometry constants.

The card computes L = d/d delta [ polar-aligned Sigma(I + delta e_a) ] by Richardson finite
differences and reports T2 = 60, T4 = 2, c4 = 1/60 numerically.  Here the same object is built
ANALYTICALLY with sympy rationals: for C(delta) = cross_wedge(Sigma_0, Sigma(delta)) with
C(0) = c I (c > 0), the orthogonal polar factor is R = I + delta * skew(C1)/c + O(delta^2), so

    L(xi) = Sigma'(xi) + Omega(xi) Sigma_0 ,   Omega = (C1 - C1^T)/(2c),  C1 = G(Sigma_0, Sigma'(xi)).

No finite differences, no SVD.  Outputs exact rationals for T2, T4, c4, sum_a M_aa, ||G0||^2,
and cross-checks (i) against the card's numerical linear_map, (ii) against Q-0013's
structure_constants.json (sum_ab_K, tl_M_aa_norm).
"""
import json, sys
from pathlib import Path
import numpy as np
import sympy as sp

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))

PAIRS = [(0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2)]   # must match _PAIR_INDEX


def check_pair_index():
    from examples.physics.gravity.causal_face_simplicity import _PAIR_INDEX
    assert list(_PAIR_INDEX) == PAIRS, f"_PAIR_INDEX mismatch: {list(_PAIR_INDEX)}"


def two_form(u, v):
    m = sp.Matrix(4, 4, lambda i, j: u[i] * v[j] - u[j] * v[i])
    return sp.Matrix([m[i, j] for (i, j) in PAIRS])


def wedge(a, b):
    return (a[0] * b[3] + a[3] * b[0] + a[1] * b[4] + a[4] * b[1] + a[2] * b[5] + a[5] * b[2])


EPS = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}


def triple(e):
    """Sigma^i(e), e a sympy 4x4; returns list of three 6-vectors."""
    out = []
    for i in range(3):
        form = two_form(e.row(0).T, e.row(i + 1).T)
        for j in range(3):
            for k in range(3):
                s = EPS.get((i, j, k), 0)
                if s:
                    form = form + sp.Rational(s, 2) * two_form(e.row(j + 1).T, e.row(k + 1).T)
        out.append(sp.simplify(form))
    return out


def gram(A, B):
    return sp.Matrix(3, 3, lambda i, j: wedge(A[i], B[j]))


def traceless(M):
    return M - (M.trace() / 3) * sp.eye(3)


def main():
    check_pair_index()
    d = sp.symbols("d")
    I4 = sp.eye(4)
    S0 = triple(I4)
    G0 = gram(S0, S0)
    c = sp.simplify(G0[0, 0])
    print("G0 =", G0.tolist(), " ||G0||^2 =", sp.simplify(sum(x**2 for x in G0)))
    assert sp.simplify(G0 - c * sp.eye(3)) == sp.zeros(3, 3), "C(0) is not a multiple of identity"

    # exact L_a for the 16 basis directions
    L = []
    for a in range(16):
        xi = sp.zeros(4, 4)
        xi[a // 4, a % 4] = 1
        Sd = triple(I4 + d * xi)
        Sp = [sp.Matrix([sp.diff(comp, d).subs(d, 0) for comp in vec]) for vec in Sd]
        C1 = gram(S0, Sp)
        Om = (C1 - C1.T) / (2 * c)
        La = [sp.zeros(6, 1) for _ in range(3)]
        for i in range(3):
            La[i] = Sp[i] + sum((Om[i, j] * S0[j] for j in range(3)), sp.zeros(6, 1))
        L.append([sp.simplify(v) for v in La])

    M = [[traceless((gram(L[a], L[b]) + gram(L[b], L[a])) / 2) for b in range(16)] for a in range(16)]
    K = sp.Matrix(16, 16, lambda a, b: sp.simplify(sum(x**2 for x in M[a][b])))
    T2 = sp.simplify(sum(K))
    T4 = sp.simplify(sum(K[a, a] for a in range(16)))
    c4 = sp.nsimplify(T4 / (2 * T2))
    Msum = sp.simplify(sum((M[a][a] for a in range(16)), sp.zeros(3, 3)))

    print("EXACT T2 =", T2, "  T4 =", T4, "  c4 =", c4, " = ", float(c4))
    print("sum_a M_aa =", Msum.tolist())
    print("diag K_aa (a=4*mu+nu):", [sp.nsimplify(K[a, a]) for a in range(16)])
    uniq = sorted({sp.nsimplify(K[a, b]) for a in range(16) for b in range(16)}, key=lambda x: float(x))
    print("distinct ||M_ab||^2 values:", uniq)
    from collections import Counter
    cnt = Counter(sp.nsimplify(K[a, b]) for a in range(16) for b in range(16))
    print("multiplicities:", {str(k): v for k, v in sorted(cnt.items(), key=lambda kv: float(kv[0]))})

    # cross-check (i): card's numerical linear_map
    from check_cumulant import linear_map, quadratic_tensor, geometry_constants
    lm = linear_map()
    Lnum = np.array([[[float(L[a][i][j]) for j in range(6)] for i in range(3)] for a in range(16)])
    dev = float(np.max(np.abs(lm - Lnum)))
    mt = quadratic_tensor(lm)
    gc = geometry_constants(mt)
    Mnum = np.array([[[[float(M[a][b][i, j]) for j in range(3)] for i in range(3)] for b in range(16)] for a in range(16)])
    devM = float(np.max(np.abs(mt - Mnum)))
    print(f"max |L_card - L_exact| = {dev:.3e}   max |M_card - M_exact| = {devM:.3e}")
    print("card numeric constants:", {k: v for k, v in gc.items()})

    # cross-check (ii): Q-0013 structure_constants.json
    q13 = json.loads((ROOT / "verify/Q-0013/F-01/structure_constants.json").read_text(encoding="utf-8"))
    tl_norms_exact = {f"{a//4}{a%4}": float(sp.sqrt(K[a, a])) for a in range(16)}
    dev13 = max(abs(q13["tl_M_aa_norm"][k] - v) for k, v in tl_norms_exact.items())
    print(f"Q-0013 sum_ab_K = {q13['sum_ab_K']!r} vs exact T2 = {T2}  (diff {abs(q13['sum_ab_K']-float(T2)):.2e})")
    print(f"Q-0013 tl_M_aa_norm max dev from exact = {dev13:.2e}   -> exact T4 = {T4}")
    print(f"Q-0013 eps_star_iso/delta^2 = {q13['eps_star_isotropic_over_delta2']!r} vs sqrt(2*T2)/||G0|| ="
          f" {float(sp.sqrt(2*T2)/sp.sqrt(sum(x**2 for x in G0)))!r}")

    out = {"T2": str(T2), "T4": str(T4), "c4": str(c4), "c4_float": float(c4),
           "G0": [[str(G0[i, j]) for j in range(3)] for i in range(3)],
           "normG0_sq": str(sp.simplify(sum(x**2 for x in G0))),
           "sum_a_M_aa": [[str(Msum[i, j]) for j in range(3)] for i in range(3)],
           "K_diag": [str(sp.nsimplify(K[a, a])) for a in range(16)],
           "K_multiplicities": {str(k): v for k, v in cnt.items()},
           "max_abs_L_card_minus_exact": dev, "max_abs_M_card_minus_exact": devM,
           "card_numeric_constants": gc,
           "q0013_sum_ab_K": q13["sum_ab_K"], "q0013_tl_M_aa_max_dev": dev13}
    (Path(__file__).parent / "a1_exact_constants.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
