"""adversary c1 (Q-0008 attempt-03, ladder step 2): is the exact block identity more than
the parallel-axis (Huygens/Koenig) identity, and are its stated premises load-bearing?

Independent re-implementation of wedge/gram/tl (same 6-component convention, checked once
against examples.physics), then:

  H  huygens_general      : for ANY triples X_v (no simplicity, no reference at all)
                            tl gram(sum X) = n sum tl gram(X_v) - n sum tl gram(X_v - Xbar)
  B  basepoint_inert      : (S5.4) with a NON-simple Sigma_0 (premise tl gram(Sigma_0)=0 dropped)
  B2 s42_breaks           : the intermediate step (S4.2) DOES break for that Sigma_0
  W  weighted             : Y = sum w_v X_v (outside the 13.2 no-weight convention)
  S3 three_species        : three deterministic species, n-independence
  L  limits               : n=1, all eta equal (13.5), delta -> 0, pn non-integer
  C  conditioning         : relative error vs delta (the card operates at delta=0.005)
  R  random_sample_20     : seed 20260902, random n/delta/alignment
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    plebanski_gram,
    simplicity_residual,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
TOL = 1.0e-12
PAIRS = ((0, 3), (1, 4), (2, 5))


def wedge(a, b):
    return sum(a[i] * b[j] + a[j] * b[i] for i, j in PAIRS)


def G(A, B):
    return np.array([[wedge(A[i], B[j]) for j in range(3)] for i in range(3)])


def gram(A):
    return G(A, A)


def tl(M):
    return M - np.trace(M) / 3.0 * np.eye(3, dtype=M.dtype)


def T(A, B):
    return tl(0.5 * (G(A, B) + G(B, A)))


def rel(x, y):
    s = float(np.linalg.norm(x))
    d = float(np.linalg.norm(x - y))
    return d / s if s > 0 else d


REF = geometric_self_dual_triple(np.eye(4))


def aligned(tetrad, ref):
    return optimal_internal_alignment(ref, geometric_self_dual_triple(tetrad)).aligned_candidate


def draw_cells(n, delta, rng, min_det=0.2):
    out = []
    while len(out) < n:
        e = np.eye(4) + delta * rng.normal(size=(4, 4))
        if float(np.linalg.det(e)) > min_det:
            out.append(aligned(e, REF))
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED, "tol": TOL}

    worst = 0.0
    for _ in range(20):
        A = rng.normal(size=(3, 6))
        worst = max(worst, rel(plebanski_gram(A), gram(A)))
    res["primitives_match_repo"] = {"max_rel_err": worst, "ok": worst <= 1e-14}

    worst = 0.0
    rows = []
    for n in (1, 2, 3, 5, 17, 64):
        w = 0.0
        for _ in range(20):
            X = [rng.normal(size=(3, 6)) for _ in range(n)]
            xbar = sum(X) / n
            lhs = tl(gram(sum(X)))
            rhs = n * sum(tl(gram(x)) for x in X) - n * sum(tl(gram(x - xbar)) for x in X)
            w = max(w, rel(lhs, rhs))
        rows.append({"n": n, "max_rel_err": w})
        worst = max(worst, w)
    res["huygens_general_no_premise"] = {
        "max_rel_err": worst, "ok": worst <= TOL, "rows": rows,
        "note": "premise-free parallel-axis identity; the claimed theorem is this plus tl gram(X_v)=0",
    }

    worst = 0.0
    worst_ref_resid = 1e9
    for _ in range(20):
        cells = draw_cells(4, 0.3, rng)
        sig0 = rng.normal(size=(3, 6))
        worst_ref_resid = min(worst_ref_resid, float(np.linalg.norm(tl(gram(sig0)))))
        eta = [x - sig0 for x in cells]
        bar = sum(eta) / len(eta)
        lhs = tl(gram(sum(cells)))
        rhs = -len(cells) * sum(tl(gram(e - bar)) for e in eta)
        worst = max(worst, rel(lhs, rhs))
    res["basepoint_inert_nonsimple_reference"] = {
        "max_rel_err": worst, "ok": worst <= TOL,
        "min_tl_gram_sigma0_norm": worst_ref_resid,
        "note": "S5.4 holds verbatim with tl gram(Sigma_0) nonzero, so that assumption is superfluous",
    }

    cells = draw_cells(3, 0.3, rng)
    sig0 = rng.normal(size=(3, 6))
    breaks = []
    for x in cells:
        eta = x - sig0
        breaks.append(rel(2 * T(sig0, eta), -tl(gram(eta))))
    res["s42_breaks_for_nonsimple_reference"] = {
        "rel_errs": breaks, "ok": min(breaks) > 1e-3,
        "note": "S4.2 plus S5.1 is one route, not a necessary one",
    }

    worst = 0.0
    rows = []
    for n in (2, 5, 17):
        w = 0.0
        for _ in range(20):
            cells = draw_cells(n, 0.3, rng)
            wt = rng.normal(size=n)
            W = float(wt.sum())
            if abs(W) < 0.25:
                continue
            eta = [x - REF for x in cells]
            barw = sum(wt[v] * eta[v] for v in range(n)) / W
            Y = sum(wt[v] * cells[v] for v in range(n))
            lhs = tl(gram(Y))
            rhs = -W * sum(wt[v] * tl(gram(eta[v] - barw)) for v in range(n))
            w = max(w, rel(lhs, rhs))
        rows.append({"n": n, "max_rel_err": w})
        worst = max(worst, w)
    res["weighted_generalisation"] = {
        "max_rel_err": worst, "ok": worst <= TOL, "rows": rows,
        "note": "n -> W = sum w_v and mean -> weighted mean; needs W nonzero",
    }

    cells = draw_cells(2, 0.3, rng)
    Y0 = cells[0] - cells[1]
    res["weighted_W_equals_zero"] = {
        "tl_gram_Y_norm": float(np.linalg.norm(tl(gram(Y0)))),
        "note": "W=0: the weighted form is undefined and tl gram(Y) is generally nonzero",
    }

    d1 = draw_cells(1, 0.3, rng)[0] - REF
    d2 = draw_cells(1, 0.25, rng)[0] - REF
    fracs = (0.5, 0.25, 0.25)
    rows = []
    for n in (4, 8, 16, 32, 64):
        counts = [int(round(f * n)) for f in fracs]
        etas = [np.zeros((3, 6))] * counts[0] + [d1] * counts[1] + [d2] * counts[2]
        cells = [REF + e for e in etas]
        eps = simplicity_residual(sum(cells))
        dbar = fracs[1] * d1 + fracs[2] * d2
        pred_num = np.linalg.norm(tl(fracs[1] * gram(d1) + fracs[2] * gram(d2) - gram(dbar)))
        pred = float(pred_num / np.linalg.norm(gram(REF + dbar)))
        rows.append({"n": n, "eps": eps, "eps_pred": pred, "rel_err": abs(eps - pred) / abs(eps)})
    eps_vals = [r["eps"] for r in rows]
    res["three_species_general_corollary"] = {
        "max_rel_err": max(r["rel_err"] for r in rows),
        "n_spread": (max(eps_vals) - min(eps_vals)) / max(eps_vals),
        "ok": max(r["rel_err"] for r in rows) <= TOL, "rows": rows,
        "note": "eps = norm(tl(sum_s f_s gram(D_s) - gram(Dbar))) / norm(gram(Sigma_0+Dbar)), n-free",
    }

    c = draw_cells(1, 0.3, rng)[0]
    res["limit_n1"] = {"eps": simplicity_residual(c), "rhs": 0.0,
                       "ok": simplicity_residual(c) <= 1e-14}

    c = draw_cells(1, 0.3, rng)[0]
    rows = [{"n": n, "eps_block": simplicity_residual(n * c)} for n in (2, 8, 64)]
    res["limit_all_eta_equal"] = {
        "rows": rows, "ok": max(r["eps_block"] for r in rows) <= 1e-14,
        "note": "RHS vanishes identically so the block is exactly simple (13.3 / 13.5)",
    }

    rows = []
    for delta in (0.3, 0.05, 0.005, 0.001):
        w_rel, w_abs, tl_over_gram = 0.0, 0.0, 0.0
        for _ in range(10):
            cells = draw_cells(5, delta, rng, min_det=0.05)
            eta = [x - REF for x in cells]
            bar = sum(eta) / 5
            Y = sum(cells)
            lhs = tl(gram(Y))
            rhs = -5 * sum(tl(gram(e - bar)) for e in eta)
            w_rel = max(w_rel, rel(lhs, rhs))
            scale = float(np.linalg.norm(gram(Y)))
            w_abs = max(w_abs, float(np.linalg.norm(lhs - rhs)) / scale)
            tl_over_gram = max(tl_over_gram, float(np.linalg.norm(lhs)) / scale)
        rows.append({"delta": delta, "max_rel_err_vs_tl": w_rel,
                     "max_err_vs_gram_scale": w_abs, "tl_over_gram": tl_over_gram,
                     "exceeds_TOL_IDENT": bool(w_rel > TOL)})
    res["conditioning_vs_delta"] = {
        "rows": rows,
        "note": "cancellation: norm(tl gram Y) ~ delta^2 norm(gram Y), so the roundoff floor of the "
                "attempt normalisation grows like delta^-2; the card runs at delta=0.005",
    }

    d = draw_cells(1, 0.3, rng)[0] - REF
    p_target = 0.3
    rows = []
    for n in (4, 8, 16, 32, 64):
        n_b = int(round(p_target * n))
        cells = [REF] * n_b + [REF + d] * (n - n_b)
        rows.append({"n": n, "n_B": n_b, "p_eff": n_b / n,
                     "eps": simplicity_residual(sum(cells))})
    eps_vals = [r["eps"] for r in rows]
    res["pn_noninteger_drift"] = {
        "p_target": p_target,
        "relative_spread": (max(eps_vals) - min(eps_vals)) / max(eps_vals),
        "rows": rows,
        "note": "with pn non-integer the n-independence survives only up to O(1/n) rounding of n_B",
    }

    worst_conj, worst_simp, worst_det = 0.0, 0.0, 0.0
    for _ in range(20):
        e = np.eye(4) + 0.3 * rng.normal(size=(4, 4))
        if float(np.linalg.det(e)) <= 0.2:
            continue
        audit = optimal_internal_alignment(REF, geometric_self_dual_triple(e))
        R = audit.rotation
        B = geometric_self_dual_triple(e)
        worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
        worst_conj = max(worst_conj, rel(gram(R @ B), R @ gram(B) @ R.T))
        worst_simp = max(worst_simp, simplicity_residual(audit.aligned_candidate))
    res["alignment_so3"] = {
        "max_det_minus_one": worst_det, "max_rel_err_conjugation": worst_conj,
        "max_cell_simplicity_after_alignment": worst_simp,
        "ok": worst_det <= 1e-12 and worst_conj <= 1e-12 and worst_simp <= 1e-12,
    }

    worst = 0.0
    samples = []
    for k in range(20):
        n = int(rng.integers(2, 31))
        delta = float(np.exp(rng.uniform(np.log(0.01), np.log(0.8))))
        cells = draw_cells(n, delta, rng, min_det=0.05)
        rot = []
        for x in cells:
            Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            rot.append(Q @ x)
        for label, group in (("aligned", cells), ("rotated", rot)):
            eta = [x - REF for x in group]
            bar = sum(eta) / n
            lhs = tl(gram(sum(group)))
            rhs = -n * sum(tl(gram(e - bar)) for e in eta)
            err = rel(lhs, rhs)
            worst = max(worst, err)
            samples.append({"k": k, "n": n, "delta": delta, "group": label, "rel_err": err})
    res["random_sample_20"] = {
        "max_rel_err": worst, "ok": worst <= 1e-9,
        "worst_samples": sorted(samples, key=lambda s: -s["rel_err"])[:5],
        "note": "TOL_IDENT=1e-12 is not attainable at small delta, see conditioning_vs_delta",
    }

    out = Path(__file__).resolve().parent / "c1_result.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    summary = {}
    for k, v in res.items():
        if isinstance(v, dict):
            summary[k] = {kk: vv for kk, vv in v.items() if kk not in ("rows", "worst_samples")}
        else:
            summary[k] = v
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
