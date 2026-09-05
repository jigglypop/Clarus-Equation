"""Adversary a4: K5 symmetry protection, eps_star convention vs F-02, kill executability."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0013" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
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


print("=" * 78)
print("(F) K5: is axis_ratio_23_over_01 an EXACT identity (per realization), i.e. a safe kill?")
rng = np.random.default_rng(31337)
worst_rel = 0.0
rows = []
for n in (2, 4, 8, 16):
    for d in (0.005, 0.05, 0.3):
        for _ in range(30):
            g = rng.normal(size=n)
            l1 = np.zeros((n, 4, 4))
            l1[:, 0, 1] = g
            l2 = np.zeros((n, 4, 4))
            l2[:, 2, 3] = g
            l3 = np.zeros((n, 4, 4))
            l3[:, 1, 0] = g
            r1, r2, r3 = block(l1, d), block(l2, d), block(l3, d)
            rel23 = abs(r2 - r1) / max(r1, 1e-300)
            rel10 = abs(r3 - r1) / max(r1, 1e-300)
            worst_rel = max(worst_rel, rel23)
            rows.append({"n": n, "delta": d, "rel_23_vs_01": rel23, "rel_10_vs_01": rel10})
print("    worst per-realization relative difference |eps(2,3) - eps(0,1)|/eps(0,1) = %.3e" % worst_rel)
print("    worst |eps(1,0) - eps(0,1)|/eps(0,1) = %.3e" % max(r["rel_10_vs_01"] for r in rows))
report["K5_exact_identity_worst_rel"] = worst_rel
report["K5_10_vs_01_worst_rel"] = max(r["rel_10_vs_01"] for r in rows)

print("")
print("=" * 78)
print("(G) eps_star convention: F-02 defines eps_star by iid n=2 RMS = eps_star/2.")
print("    F-01 iso law predicts eps(n)/delta2 = sqrt(10) sqrt(n-1)/n, so eps(2)/delta2 = sqrt(10)/2.")
DELTA = 0.005
rng = np.random.default_rng(20260902)
vals = [block(rng.normal(size=(2, 4, 4)), DELTA) for _ in range(3000)]
obs2 = rms(vals) / DELTA ** 2
print("    observed iid n=2 RMS/delta2 (3000 trials) = %.6f ; sqrt(10)/2 = %.6f ; ratio = %.4f"
      % (obs2, math.sqrt(10) / 2, obs2 / (math.sqrt(10) / 2)))
report["eps_star_n2_observed_over_delta2"] = obs2
report["eps_star_n2_predicted"] = math.sqrt(10) / 2
report["eps_star_ratio"] = obs2 / (math.sqrt(10) / 2)

print("    F-02 check_modes convention differences vs F-01 check_floor:")
import check_modes as CM  # noqa: E402
import check_floor as CF  # noqa: E402
print("      F-02: DELTA=%s MIN_DET=%s SEED=%s SIZES=%s TRIALS=%s"
      % (CM.DELTA, CM.MIN_DET, CM.SEED, CM.SIZES, CM.TRIALS))
print("      F-01: DELTA=%s MIN_DET=(absent) SEED=%s SIZES=%s TRIALS=%s"
      % (CF.DELTA, CF.SEED, CF.SIZES, CF.TRIALS))
report["convention"] = {"F02_delta": CM.DELTA, "F02_min_det": CM.MIN_DET,
                        "F01_delta": CF.DELTA, "F01_has_min_det": hasattr(CF, "MIN_DET")}

print("")
print("=" * 78)
print("(H) kill executability: run every card mode at SMOKE size (not pre-registered)")
exec_report = {}
for mode in ("rank1", "iso", "piso", "axis", "mix"):
    try:
        out = CF.RUNNERS[mode]((4, 8), 24)
        keys = {k: v for k, v in out.items() if not isinstance(v, dict)}
        exec_report[mode] = {"ok": True, "stats": keys}
        print("    %-6s OK  %s" % (mode, {k: round(float(v), 6) for k, v in keys.items()}))
    except Exception as exc:  # noqa: BLE001
        exec_report[mode] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        print("    %-6s FAILED %s" % (mode, exc))
for label, sizes, trials in (("zero(smoke d=0.005+0.3, n<=8, 16 tr)", (4, 8), 16),
                             ("zero(prereg-ish d=0.3 incl n=64, 512 tr)", (64,), 512)):
    try:
        out = CF.mode_zero(sizes, trials)
        exec_report["zero_" + label] = {"ok": True, "max": out["zero_max_residual"]}
        print("    zero %-42s OK max=%.3e" % (label, out["zero_max_residual"]))
    except Exception as exc:  # noqa: BLE001
        exec_report["zero_" + label] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        print("    zero %-42s FAILED %s" % (label, exc))
report["kill_executability"] = exec_report

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "a4_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                    encoding="utf-8")
print("")
print("wrote a4_report.json")
