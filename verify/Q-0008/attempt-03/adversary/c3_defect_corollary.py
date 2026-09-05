"""adversary c3 (Q-0008 attempt-03): does corollary (b) actually support the card's K4 numbers?

run_defect in verify/Q-0008/F-02/check_modes.py is DETERMINISTIC (one Delta sample, no trials), so
the two K4 statistics are a pure function of the pre-registered seed.  This file replicates that
sampler independently, splits eps(n) into the exact numerator law (n-1)/n^2 and the denominator
drift, and asks how much of the pre-registered window the drift consumes.

K4 is declared in the card as a CONSISTENCY CHECK, not a kill (already_observed = true, r48
disclosed), so evaluating it here changes no kill budget.
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
GRID = (4, 8, 16, 32, 64)
CARD_RATIO = 0.140625
CARD_RATIO_WINDOW = (0.124, 0.158)
CARD_SLOPE = -0.9069
CARD_SLOPE_WINDOW = (-0.96, -0.86)
REF = geometric_self_dual_triple(np.eye(4))


def tl(M):
    return M - np.trace(M) / 3.0 * np.eye(3)


def defect_cell(perturbation, seed, min_det=0.2):
    rng = np.random.default_rng(seed)
    while True:
        e = np.eye(4) + perturbation * rng.normal(size=(4, 4))
        if float(np.linalg.det(e)) > min_det:
            break
    return optimal_internal_alignment(REF, geometric_self_dual_triple(e)).aligned_candidate


def eps_curve(cell_c, grid):
    delta_triple = cell_c - REF
    num = float(np.linalg.norm(tl(plebanski_gram(delta_triple))))
    rows = []
    for n in grid:
        block = (n - 1) * REF + cell_c
        eps = simplicity_residual(block)
        den = float(np.linalg.norm(plebanski_gram(REF + delta_triple / n)))
        pure = (n - 1) / n**2 * num / float(np.linalg.norm(plebanski_gram(REF)))
        rows.append({"n": n, "eps": eps, "eps_from_identity": (n - 1) / n**2 * num / den,
                     "eps_pure_numerator_law": pure,
                     "denominator_over_reference": den / float(np.linalg.norm(plebanski_gram(REF)))})
    return rows


def slope(rows):
    xs = np.log([r["n"] for r in rows])
    ys = np.log([r["eps"] for r in rows])
    return float(np.polyfit(xs, ys, 1)[0])


def main() -> int:
    res: dict = {"seed": SEED, "card_ratio": CARD_RATIO, "card_slope": CARD_SLOPE}

    cell = defect_cell(0.35, SEED)
    rows = eps_curve(cell, GRID)
    eps = {r["n"]: r["eps"] for r in rows}
    ratio = eps[64] / eps[8]
    r48 = eps[8] / eps[4]
    sl = slope(rows)
    res["card_spec_defect"] = {
        "rows": rows,
        "ratio_64_over_8": ratio,
        "slope": sl,
        "r48_eps8_over_eps4": r48,
        "ratio_in_window": CARD_RATIO_WINDOW[0] <= ratio <= CARD_RATIO_WINDOW[1],
        "slope_in_window": CARD_SLOPE_WINDOW[0] <= sl <= CARD_SLOPE_WINDOW[1],
        "ratio_deviation_from_card": ratio / CARD_RATIO - 1.0,
        "slope_deviation_from_card": sl - CARD_SLOPE,
        "max_rel_err_identity_prediction": max(
            abs(r["eps"] - r["eps_from_identity"]) / r["eps"] for r in rows),
        "window_fraction_used_by_denominator_drift":
            abs(ratio - CARD_RATIO) / (CARD_RATIO - CARD_RATIO_WINDOW[0]),
    }

    # attempt-03 used its own Delta (perturbation 0.3, drawn from its own stream): compare
    scan = []
    for pert in (0.05, 0.1, 0.2, 0.3, 0.35, 0.5):
        c = defect_cell(pert, SEED)
        rws = eps_curve(c, GRID)
        e = {r["n"]: r["eps"] for r in rws}
        scan.append({"perturbation": pert, "ratio_64_over_8": e[64] / e[8], "slope": slope(rws),
                     "in_ratio_window": CARD_RATIO_WINDOW[0] <= e[64] / e[8] <= CARD_RATIO_WINDOW[1],
                     "in_slope_window": CARD_SLOPE_WINDOW[0] <= slope(rws) <= CARD_SLOPE_WINDOW[1]})
    res["delta_scale_scan"] = {
        "rows": scan,
        "note": "the pure numerator law 0.140625 is the delta -> 0 limit; the observable moves down "
                "monotonically with the size of Delta",
    }

    out = Path(__file__).resolve().parent / "c3_result.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
