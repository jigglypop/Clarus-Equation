"""b3 (re-audit): how could c4 = 1/60 be wrong?  Which conventions does it depend on?

c4 = T4/(2 T2), T2 = sum_ab ||M_ab||^2 (basis invariant), T4 = sum_a ||M_aa||^2 (NOT basis invariant).
Probed conventions:
  scale      : Plebanski gram G -> lambda G   (residual is normalised by ||gram||, so this must cancel)
  param      : tetrad I + d*xi   vs   exp(d*xi)   (same first derivative -> same M)
  traceless  : drop the tl projection in M_ab
  align      : drop the polar SO(3) alignment in L   (also reports every ||M_aa||)
  basis      : label components i.i.d. in a ROTATED orthonormal basis of the 16-dim tetrad space,
               O(16) Haar and the physical subgroup xi -> R xi S^T (R, S in SO(4)), and the
               symmetric/antisymmetric basis.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import (linear_map, quadratic_tensor, gram_form, REFERENCE, tl, basis_16, DERIV_H)  # noqa
from examples.physics.causal_face_simplicity import geometric_self_dual_triple  # noqa
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa

OUT = Path(__file__).parent
RNG = np.random.default_rng(20260902)


def consts(M):
    t2 = float((M * M).sum())
    t4 = float(sum((M[a, a] * M[a, a]).sum() for a in range(16)))
    return {"T2": t2, "T4": t4, "c4": t4 / (2 * t2), "one_over_c4": 2 * t2 / t4 if t4 else None,
            "max_abs_sum_a_Maa": float(np.abs(sum(M[a, a] for a in range(16))).max())}


def qtensor(lmap, traceless=True):
    out = np.zeros((16, 16, 3, 3))
    for a in range(16):
        for b in range(16):
            s = 0.5 * (gram_form(lmap[a], lmap[b]) + gram_form(lmap[b], lmap[a]))
            out[a, b] = tl(s) if traceless else s
    return out


def expm_series(A, terms=12):
    out, term = np.eye(4), np.eye(4)
    for k in range(1, terms):
        term = term @ A / k
        out = out + term
    return out


def lmap_generic(cellfun, h=DERIV_H):
    def central(step):
        return np.array([(cellfun(e, step) - cellfun(e, -step)) / (2 * step) for e in basis_16()])
    return (4 * central(h) - central(2 * h)) / 3.0


def aligned(e, d):
    return optimal_internal_alignment(REFERENCE, geometric_self_dual_triple(np.eye(4) + d * e)).aligned_candidate


def aligned_exp(e, d):
    return optimal_internal_alignment(REFERENCE, geometric_self_dual_triple(expm_series(d * e))).aligned_candidate


def unaligned(e, d):
    return geometric_self_dual_triple(np.eye(4) + d * e)


def rotate(M, O):
    return np.einsum("ca,db,cdij->abij", O, O, M)


def haar(n):
    q, r = np.linalg.qr(RNG.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def so4():
    q = haar(4)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def symasym_basis_matrix():
    """orthonormal basis: E_ii, (E_ij+E_ji)/sqrt2, (E_ij-E_ji)/sqrt2 expressed in the E_ij basis."""
    cols = []
    for i in range(4):
        v = np.zeros((4, 4)); v[i, i] = 1.0; cols.append(v.ravel())
    for i in range(4):
        for j in range(i + 1, 4):
            v = np.zeros((4, 4)); v[i, j] = v[j, i] = 1 / np.sqrt(2); cols.append(v.ravel())
            w = np.zeros((4, 4)); w[i, j] = 1 / np.sqrt(2); w[j, i] = -1 / np.sqrt(2); cols.append(w.ravel())
    return np.array(cols).T  # columns = new basis vectors in old coordinates


def main():
    M = quadratic_tensor(linear_map())
    res = {"nominal": consts(M)}
    res["nominal"]["diag_norms_sq"] = sorted(round(float((M[a, a] * M[a, a]).sum()), 12) for a in range(16))

    res["scale_lambda_3.7"] = consts(3.7 * M)                       # G -> lambda G
    res["param_exp"] = consts(qtensor(lmap_generic(aligned_exp)))    # exp parametrisation
    res["no_traceless"] = consts(qtensor(linear_map(), traceless=False))
    Mna = qtensor(lmap_generic(unaligned))
    res["no_alignment"] = consts(Mna)
    res["no_alignment"]["diag_norms_sq"] = sorted(round(float((Mna[a, a] * Mna[a, a]).sum()), 14) for a in range(16))
    res["no_alignment"]["offdiag_T2"] = float((Mna * Mna).sum())

    c_haar = np.array([consts(rotate(M, haar(16)))["c4"] for _ in range(200)])
    c_phys = np.array([consts(rotate(M, np.kron(so4(), so4())))["c4"] for _ in range(100)])
    Osa = symasym_basis_matrix()
    res["basis_haar_O16"] = {"n": 200, "min": float(c_haar.min()), "median": float(np.median(c_haar)),
                             "max": float(c_haar.max()), "nominal": 1 / 60,
                             "fraction_within_5pct_of_1_60": float(np.mean(np.abs(c_haar * 60 - 1) < 0.05))}
    res["basis_RxS_so4"] = {"n": 100, "min": float(c_phys.min()), "median": float(np.median(c_phys)),
                            "max": float(c_phys.max()),
                            "max_abs_dev_from_1_60": float(np.max(np.abs(c_phys - 1 / 60)))}
    res["basis_sym_antisym"] = consts(rotate(M, Osa))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "diag_norms_sq"})
                      for k, v in res.items()}, indent=1))
    print("nominal diag norms^2:", res["nominal"]["diag_norms_sq"])
    print("noalign diag norms^2:", res["no_alignment"]["diag_norms_sq"])
    (OUT / "b3_c4_convention.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
