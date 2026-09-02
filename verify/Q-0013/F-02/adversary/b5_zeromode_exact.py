"""adversary b5: step-2 lemma. For DIAGONAL labels the aligned block Gram is exactly
diag(2 SA*SB, 2 SA'*SB', 2 SA''*SB''), so the residual has a closed form for every delta.
Checks: (a) closed form vs the real pipeline, (b) exact zero modes, (c) the (g,g,0,0) breaking,
(d) truncation bias of the master (off-grid n, seed 20260903).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import (
    geometric_self_dual_triple, plebanski_gram, simplicity_residual)
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
Mt = np.load(OUT / "b1_Mt.npy")
REF = geometric_self_dual_triple(np.eye(4))
MIN_DET = 0.05
NS = 2.0 * math.sqrt(3.0)


def closed_form_gram(p):
    """p: (n,4) tetrad diagonals -> exact 3x3 Gram of the summed self-dual triple."""
    A = p[:, 0] * p[:, 1]; B = p[:, 2] * p[:, 3]
    A1 = p[:, 0] * p[:, 2]; B1 = p[:, 3] * p[:, 1]
    A2 = p[:, 0] * p[:, 3]; B2 = p[:, 1] * p[:, 2]
    return 2.0 * np.diag([A.sum() * B.sum(), A1.sum() * B1.sum(), A2.sum() * B2.sum()])


def closed_form_residual(p):
    g = closed_form_gram(p)
    tl = g - np.trace(g) / 3.0 * np.eye(3)
    return float(np.linalg.norm(tl) / np.linalg.norm(g))


def pipeline_gram(p, ):
    tot = sum(optimal_internal_alignment(
        REF, geometric_self_dual_triple(np.diag(row))).aligned_candidate for row in p)
    return plebanski_gram(tot)


def F_T(sigma):
    F = float(np.linalg.norm(np.einsum("ab,abij->ij", sigma, Mt)))
    T = float(np.einsum("abij,ac,bd,cdij->", Mt, sigma, sigma, Mt))
    return F, T


def master(n, F, T):
    return math.sqrt((n - 1) * ((n - 1) * F * F + 2.0 * T) / (12.0 * n * n))


def main():
    rng = np.random.default_rng(20260903)
    out = {}

    # (a) closed form vs real pipeline
    worst_gram, worst_res, worst_offdiag = 0.0, 0.0, 0.0
    for n in (2, 3, 5, 7):
        for delta in (0.005, 0.1, 0.3, 1.0):
            for _ in range(20):
                while True:
                    g = rng.normal(size=(n, 4))
                    p = 1.0 + delta * g
                    if np.all(np.prod(p, axis=1) > MIN_DET) and np.all(p > 0):
                        break
                Gp = pipeline_gram(p)
                Gc = closed_form_gram(p)
                worst_gram = max(worst_gram, float(np.max(np.abs(Gp - Gc)) / np.abs(Gc).max()))
                worst_offdiag = max(worst_offdiag, float(np.max(np.abs(Gp - np.diag(np.diag(Gp)))) / np.abs(Gc).max()))
                rp = simplicity_residual(sum(optimal_internal_alignment(
                    REF, geometric_self_dual_triple(np.diag(row))).aligned_candidate for row in p))
                worst_res = max(worst_res, abs(rp - closed_form_residual(p)))
    out["closed_form_check"] = {"max_rel_gram_diff": worst_gram, "max_rel_offdiag": worst_offdiag,
                                "max_abs_residual_diff": worst_res,
                                "cases": "n in 2,3,5,7 x delta in 0.005,0.1,0.3,1.0 x 20 draws"}

    # (b) exact zero modes: pipeline and closed form
    modes = {"single_e11": np.array([0.0, 1.0, 0.0, 0.0]),
             "3diag_g_g_g_0": np.array([1.0, 1.0, 1.0, 0.0]) / math.sqrt(3.0),
             "3diag_0_g_g_g": np.array([0.0, 1.0, 1.0, 1.0]) / math.sqrt(3.0),
             "trace_g_g_g_g": np.array([1.0, 1.0, 1.0, 1.0]) / 2.0,
             "two_g_g_0_0": np.array([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2.0)}
    zres = {}
    for name, u in modes.items():
        rec = {}
        for delta in (0.005, 0.1, 0.3, 1.0, 3.0):
            worst_p, worst_c = 0.0, 0.0
            for n in ((3, 5, 9, 33) if delta <= 0.3 else (3, 5)):
                for _ in range(6):
                    while True:
                        g = rng.normal(size=(n, 1))
                        p = 1.0 + delta * (g * u[None, :])
                        if np.all(np.prod(p, axis=1) > MIN_DET) and np.all(p > 0):
                            break
                    worst_c = max(worst_c, closed_form_residual(p))
                    if n <= 9:
                        worst_p = max(worst_p, simplicity_residual(sum(optimal_internal_alignment(
                            REF, geometric_self_dual_triple(np.diag(row))).aligned_candidate for row in p)))
            rec[str(delta)] = {"closed_form_max": worst_c, "pipeline_max": worst_p}
        zres[name] = rec
    out["zero_modes"] = zres

    # (c) (g,g,0,0): closed-form violation V = n*sum h^2 - (sum h)^2 vs direct
    viol = []
    for n in (2, 3, 5):
        for _ in range(50):
            g = rng.normal(size=n)
            h = 0.005 * g / math.sqrt(2.0)
            p = np.stack([1 + h, 1 + h, np.ones(n), np.ones(n)], axis=1)
            G = closed_form_gram(p)
            V = n * float(np.sum((1 + h) ** 2)) - float(np.sum(1 + h)) ** 2
            viol.append(abs(float(G[0, 0] / 2.0 - G[1, 1] / 2.0) - V))
    out["ggg00_violation_identity_maxerr"] = max(viol)

    # (d) truncation bias of the master (exact closed form, off-grid n, 200k samples)
    def eN(m, n_):
        v = np.zeros(16); v[4 * m + n_] = 1.0; return v
    bias = {}
    for tag, u, sigma in (
        ("diag4_I4", np.eye(4), sum(np.outer(eN(m, m), eN(m, m)) for m in range(4))),
        ("ce_ii_e00_plus_e11", np.array([[1.0, 1.0, 0.0, 0.0]]) / math.sqrt(2.0),
         np.outer((eN(0, 0) + eN(1, 1)) / math.sqrt(2.0), (eN(0, 0) + eN(1, 1)) / math.sqrt(2.0))),
        ("3diag_zero", np.array([[1.0, 1.0, 1.0, 0.0]]) / math.sqrt(3.0),
         np.outer((eN(0, 0) + eN(1, 1) + eN(2, 2)) / math.sqrt(3.0), (eN(0, 0) + eN(1, 1) + eN(2, 2)) / math.sqrt(3.0))),
    ):
        F, T = F_T(sigma)
        rec = {"F": F, "T": T}
        U = np.atleast_2d(u)
        for delta in (0.005, 0.02, 0.05):
            for n in (3, 5, 9, 33, 65):
                r = U.shape[0]
                gg = rng.normal(size=(25000, n, r))
                p = 1.0 + delta * np.einsum("snr,rm->snm", gg, U)
                ok = np.all(np.prod(p, axis=2) > MIN_DET, axis=1)
                p = p[ok]
                A = p[:, :, 0] * p[:, :, 1]; B = p[:, :, 2] * p[:, :, 3]
                A1 = p[:, :, 0] * p[:, :, 2]; B1 = p[:, :, 3] * p[:, :, 1]
                A2 = p[:, :, 0] * p[:, :, 3]; B2 = p[:, :, 1] * p[:, :, 2]
                x = np.stack([A.sum(1) * B.sum(1), A1.sum(1) * B1.sum(1), A2.sum(1) * B2.sum(1)], axis=1)
                tl = x - x.mean(axis=1, keepdims=True)
                res = np.linalg.norm(tl, axis=1) / np.linalg.norm(x, axis=1)
                rms = float(np.sqrt(np.mean(res ** 2)))
                m = master(n, F, T)
                rec["d%s_n%d" % (delta, n)] = {"exact_rms_over_delta2": rms / delta ** 2,
                                               "master": m, "bias": (rms / delta ** 2) / m - 1.0 if m > 0 else None,
                                               "kept": int(ok.sum()),
                                               "mc_se_rel": 1.0 / math.sqrt(2.0 * int(ok.sum()))}
        bias[tag] = rec
    out["truncation_bias_closed_form"] = bias
    out["_meta"] = {"seed": 20260903, "note": "off-grid n only; closed form is exact in delta"}
    (OUT / "b5_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "truncation_bias_closed_form"}, ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main()
