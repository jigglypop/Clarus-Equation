"""b1 (re-audit): independent re-estimate of the revision-2 K5 false-fire rate.

The card fixes the RMS window 0.03 from k5_rms_window_design.json, which re-uses a8's seed sequence
500000 + 911k.  A window fitted on one seed stream and validated on the same stream is not a
validation.  Here the SAME statistic is re-estimated on a DISJOINT stream 3000017 + 104729k
(all seeds > 3e6, a8 used 5e5..7.8e5; the pre-registered SEED+777 = 20261679 is not in the stream:
(20261679-3000017)/104729 = 164.83, not an integer).  Nothing is written to the card's result.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import linear_map, quadratic_tensor, gram_form, REFERENCE, run_form, SEED  # noqa

OUT = Path(__file__).parent


def q(a):
    return {"min": float(a.min()), "q25": float(np.quantile(a, .25)), "median": float(np.median(a)),
            "q75": float(np.quantile(a, .75)), "p90": float(np.quantile(a, .9)),
            "p99": float(np.quantile(a, .99)), "max": float(a.max())}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    M = quadratic_tensor(linear_map())
    g0 = float(np.linalg.norm(gram_form(REFERENCE, REFERENCE)))
    a8_seeds = {500000 + 911 * k for k in range(300)}
    rms, mx, med, ratios = [], [], [], []
    t0 = time.time()
    for k in range(reps):
        seed = 3000017 + 104729 * k
        assert seed != SEED + 777 and seed not in a8_seeds, seed
        b = run_form(M, g0, sizes=(3, 5, 8, 12), seed=seed)
        c = np.array([abs(r["delta0005"]["rel_err"]) for r in b["configurations"]])
        rms.append(float(np.sqrt(np.mean(c ** 2))))
        mx.append(float(c.max()))
        med.append(float(np.median(c)))
        ratios.append(b["ratio_delta_scaling"])
    rms, mx, med, ratios = map(np.array, (rms, mx, med, ratios))
    res = {"reps": reps, "seed_sequence": "3000017 + 104729*k (disjoint from a8 and from SEED+777)",
           "wall_s": round(time.time() - t0, 1),
           "rms_rel_err_delta0005": q(rms), "max_rel_err_delta0005": q(mx),
           "median_rel_err_delta0005": q(med), "ratio_delta_scaling": q(ratios),
           "false_fire_rms": {str(w): float(np.mean(rms > w)) for w in (0.01, 0.015, 0.02, 0.03, 0.05)},
           "false_fire_max_window_002_rev1": float(np.mean(mx > 0.02)),
           "ratio_outside_3_7": float(np.mean((ratios < 3) | (ratios > 7))),
           "joint_false_fire_rev2": float(np.mean((rms > 0.03) | (ratios < 3) | (ratios > 7)))}
    print(json.dumps(res, indent=1))
    (OUT / "b1_k5_rms_independent_seeds.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
