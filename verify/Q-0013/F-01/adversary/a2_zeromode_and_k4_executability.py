"""Adversary a2: (B) exact zero mode with an alignment that does not raise, and
(B2) the executability of the pre-registered K4 run (`--mode zero`, delta=0.3, n=64, 512 trials)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm as _norm

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple, plebanski_gram, simplicity_residual, wedge_scalar,
)
from examples.physics.urbantke_shape_matching_rg import (  # noqa: E402
    cross_wedge_matrix, optimal_internal_alignment,
)

OUT = ROOT / "verify" / "Q-0013" / "F-01" / "adversary"
REF = geometric_self_dual_triple(np.eye(4))
report = {}


def align_only(reference, candidate):
    """The card's alignment WITHOUT the eager Urbantke metric (which raises on det<0)."""
    left, _, right_t = np.linalg.svd(cross_wedge_matrix(reference, candidate))
    rot = left @ right_t
    if float(np.linalg.det(rot)) < 0.0:
        left[:, -1] *= -1.0
        rot = left @ right_t
    return rot @ candidate


def block_safe(labels, d):
    return simplicity_residual(sum(align_only(REF, geometric_self_dual_triple(np.eye(4) + d * l))
                                   for l in labels))


print("=" * 78)
print("(B) exact zero mode, alignment computed directly (no Urbantke metric call)")
worst = 0.0
rows = []
rng = np.random.default_rng(20260902)
for comp in ((0, 0), (1, 1), (2, 2), (3, 3)):
    for d in (0.005, 0.3, 1.0, 3.0):
        for n in (2, 4, 16, 64):
            acc = []
            for _ in range(40):
                lab = np.zeros((n, 4, 4))
                g = rng.normal(size=n)
                g = np.where(np.abs(1.0 + d * g) < 1e-6, 0.0, g)
                lab[:, comp[0], comp[1]] = g
                acc.append(abs(block_safe(lab, d)))
            m = float(max(acc))
            worst = max(worst, m)
            rows.append({"comp": list(comp), "delta": d, "n": n, "max": m})
print("    worst |residual| (4 comps x 4 deltas x 4 sizes x 40 trials) = %.3e" % worst)
report["zero_mode_generic_worst"] = worst

print("    adversarial alignment-flip probe: force some cells to 1 + delta*g < -1")
flip = []
for comp in ((0, 0), (1, 1), (2, 2)):
    for d in (0.3, 1.0):
        n = 8
        acc, nflip = [], 0
        for _ in range(300):
            g = rng.normal(size=n) * 4.0 / d
            g = np.where(np.abs(1.0 + d * g) < 1e-6, 0.0, g)
            nflip += int(np.sum(1.0 + d * g < -1.0))
            lab = np.zeros((n, 4, 4))
            lab[:, comp[0], comp[1]] = g
            acc.append(abs(block_safe(lab, d)))
        flip.append({"comp": list(comp), "delta": d, "max": float(max(acc)),
                     "median": float(np.median(acc)), "n_cells_with_a_below_minus1": nflip})
        print("      comp=%s d=%s  max=%.4e  median=%.4e  (cells with a<-1: %d/2400)"
              % (comp, d, max(acc), np.median(acc), nflip))
report["zero_mode_flip_probe"] = flip

# deliberately construct a block where exactly one cell has a < -1 (flip) and the rest do not
print("    hand-built single-flip block (a_1 = -2, a_rest = +1.0):")
hand = []
for comp in ((0, 0), (1, 1)):
    for n in (2, 4, 8):
        d = 1.0
        lab = np.zeros((n, 4, 4))
        lab[0, comp[0], comp[1]] = -3.0      # a = 1 + 1*(-3) = -2 < -1  -> alignment flips
        lab[1:, comp[0], comp[1]] = 0.5      # a = 1.5
        res = abs(block_safe(lab, d))
        hand.append({"comp": list(comp), "n": n, "residual": res})
        print("      comp=%s n=%d  residual = %.6e" % (comp, n, res))
report["zero_mode_handbuilt_flip"] = hand

print()
print("=" * 78)
print("(B2) K4 executability: the card runs `--mode zero` at delta=0.3 with 1 + 0.3*g as det")
p_neg = float(_norm.cdf(-1.0 / 0.3))
draws = 2 * 512 * (4 + 8 + 16 + 32 + 64)     # 2 comps x TRIALS x sum(SIZES)
print("    P(det = 1 + 0.3 g < 0) = %.4e ; pre-registered draws at delta=0.3 = %d"
      % (p_neg, draws))
print("    expected number of orientation-reversing cells = %.1f  => P(at least one) ~ 1"
      % (p_neg * draws))
report["k4_executability"] = {"p_det_negative": p_neg, "prereg_draws_at_delta_0p3": draws,
                              "expected_negative_dets": p_neg * draws}

print("    direct check: does the card's own cell() raise on a negative-determinant tetrad?")
lab = np.zeros((4, 4))
lab[0, 0] = -4.0                              # a = 1 + 0.3*(-4) = -0.2 < 0
try:
    optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + 0.3 * lab))
    raised = None
except Exception as exc:                      # noqa: BLE001
    raised = "%s: %s" % (type(exc).__name__, exc)
print("      ->", raised)
report["k4_executability"]["card_cell_raises_on_negative_det"] = raised

print("    empirical: run the card's own mode_zero at delta=0.3 with n=64, 512 trials, comp=(0,0)")
sys.path.insert(0, str(ROOT / "verify" / "Q-0013" / "F-01"))
import check_floor as CF                       # noqa: E402
err = None
try:
    r2 = np.random.default_rng(CF.SEED + 101 * 0 + 7 * 0)
    vals = [abs(CF.block_residual(CF.labels_component(r2, 64, [(0, 0)], [1.0]), 0.3))
            for _ in range(512)]
    err = "no exception; max = %.3e" % max(vals)
except Exception as exc:                      # noqa: BLE001
    err = "RAISED %s: %s" % (type(exc).__name__, exc)
print("      ->", err)
report["k4_executability"]["card_mode_zero_n64_delta0p3"] = err

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "a2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("wrote a2_report.json")
