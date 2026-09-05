"""Adversary a5: execute every recovers limit, plus the second counterexample and a determinism check."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple, simplicity_residual,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

OUT = ROOT / "verify" / "Q-0013" / "F-01" / "adversary"
REF = geometric_self_dual_triple(np.eye(4))
report = {}


def cell(lab, d):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + d * lab)).aligned_candidate


def block(labels, d=0.005):
    return simplicity_residual(sum(cell(l, d) for l in labels))


def rms(v):
    a = np.asarray(v, float)
    return float(np.sqrt(np.mean(a * a)))


def slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


print("=" * 78)
print("(I) recovers, executed")
rec = {}

rng = np.random.default_rng(5150)
v = [block(rng.normal(size=(1, 4, 4)), 0.005) for _ in range(200)]
print("  [1] n=1 single cell: max residual = %.3e   (card: 0)" % max(v))
rec["n_eq_1_max"] = float(max(v))

rng = np.random.default_rng(5151)
row = {}
for d in (1e-2, 1e-3, 1e-4, 1e-5):
    rr = np.random.default_rng(5151)
    row[d] = rms([block(rr.normal(size=(8, 4, 4)), d) for _ in range(120)]) / d ** 2
print("  [2] delta -> 0: RMS/delta2 at delta=1e-2,1e-3,1e-4,1e-5 = %s"
      % {k: round(x, 6) for k, x in row.items()})
print("      exact iso law at n=8: %.6f" % (math.sqrt(10) * math.sqrt(7) / 8))
rec["delta_to_zero"] = {str(k): x for k, x in row.items()}
rec["delta_to_zero_predicted"] = math.sqrt(10) * math.sqrt(7) / 8

print("  [3] 13.5 coherent two species (isotropic Sigma), p=1/2, n-independence:")
coh = {}
for n in (4, 8, 16, 32):
    rr = np.random.default_rng(5152)
    acc = []
    for _ in range(200):
        la = rr.normal(size=(4, 4))
        lb = rr.normal(size=(4, 4))
        lab = np.array([la if i < n // 2 else lb for i in range(n)])
        acc.append(block(lab, 0.005))
    coh[n] = rms(acc) / 0.005 ** 2
print("      RMS/delta2 = %s ; F-02 law eps_star*2p(1-p) = sqrt(10)/2 = %.6f"
      % ({k: round(x, 6) for k, x in coh.items()}, math.sqrt(10) / 2))
rec["coherent_p_half"] = {str(k): x for k, x in coh.items()}
rec["coherent_predicted"] = math.sqrt(10) / 2

print("  [4] Sigma = sigma^2 I_16 -> F-02 kernel law (iid kappa=I), amplitude and slope:")
iso = {}
for n in (4, 8, 16, 32, 64):
    rr = np.random.default_rng(5153 + n)
    iso[n] = rms([block(rr.normal(size=(n, 4, 4)), 0.005) for _ in range(200)]) / 0.005 ** 2
pred = {n: math.sqrt(10) * math.sqrt(n - 1) / n for n in iso}
print("      observed %s" % {k: round(x, 5) for k, x in iso.items()})
print("      sqrt(10) sqrt(n-1)/n %s   slope obs %.4f vs exact %.4f"
      % ({k: round(x, 5) for k, x in pred.items()},
         slope(list(iso), list(iso.values())), slope(list(pred), list(pred.values()))))
rec["iso_observed"] = {str(k): x for k, x in iso.items()}
rec["iso_predicted"] = {str(k): x for k, x in pred.items()}
report["recovers"] = rec

print("")
print("=" * 78)
print("(J) SECOND counterexample: label direction (e00 + e11)/sqrt2.")
print("    Both e00 and e11 are exact zero modes; the card w = (0,0,0) -> closed form floor 0.")
mc = {}
for name, comps in (("(e00+e11)/sqrt2", [(0, 0), (1, 1)]), ("(e00-e11)/sqrt2", [(0, 0), (1, 1)])):
    sgn = 1.0 if "+" in name else -1.0
    rr = np.random.default_rng(909090)
    curve = {}
    for n in (4, 16, 64):
        acc = []
        for _ in range(400):
            g = rr.normal(size=n)
            lab = np.zeros((n, 4, 4))
            lab[:, 0, 0] = g / math.sqrt(2.0)
            lab[:, 1, 1] = sgn * g / math.sqrt(2.0)
            acc.append(block(lab, 0.005))
        curve[n] = rms(acc) / 0.005 ** 2
    mc[name] = {str(k): x for k, x in curve.items()}
    print("    %-18s observed eps(n)/delta2 = %s ; log-log slope = %.4f"
          % (name, {k: round(x, 6) for k, x in curve.items()},
             slope(list(curve), list(curve.values()))))
# master prediction for this Sigma: F = 0.81649658, T from a3
Fm, Tm = 0.81649658, None
print("    master formula with F=0.8165 (a3) and T computed there predicts eps(64)/delta2 = 0.23567")
print("    card closed form (w = 0) predicts 0.04134  -> factor 5.7")
report["second_counterexample"] = mc

print("")
print("=" * 78)
print("(K) determinism of the b4 (0,0) rerun (same seed, twice)")
out = []
for rep in range(2):
    r = np.random.default_rng(424242 + 21)
    vals = []
    for n in (4, 8, 16, 32):
        acc = [block(np.array([np.zeros((4, 4)) for _ in range(n)]) + 0.0, 0.005) for _ in range(0)]
        acc = []
        for _ in range(200):
            lab = np.zeros((n, 4, 4))
            lab[:, 0, 0] = r.normal(size=n)
            acc.append(block(lab, 0.005))
        vals.append(rms(acc))
    out.append({"rms": vals, "slope": slope((4, 8, 16, 32), vals)})
    print("    run %d: rms = %s  slope = %+.6f"
          % (rep, ["%.3e" % x for x in vals], out[-1]["slope"]))
report["b4_determinism"] = out

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "a5_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                    encoding="utf-8")
print("")
print("wrote a5_report.json")
