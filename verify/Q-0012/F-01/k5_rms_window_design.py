"""K5 revision-2 design: sampling distribution of candidate K5 statistics over independent seeds.

The card's revision-1 K5 used the MAX over 12 random configurations of |rel_err| at delta=0.005 with
window 0.02; adversary a8 (300 seeds 500000+911k, NOT the pre-registered seed SEED+777) showed a
false-fire rate 0.223 when the card is true.  This script re-uses exactly a8's seed sequence and
records the distribution of the RMS and MEDIAN statistics (plus the max for reference) so that the
revision-2 window is fixed from an independent sampling distribution, not from the pre-registered
battery.  The pre-registered seed SEED+777 is never run here.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cumulant import linear_map, quadratic_tensor, gram_form, REFERENCE, run_form, SEED  # noqa: E402

OUT = Path(__file__).resolve().parent / "k5_rms_window_design.json"


def main() -> int:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    M = quadratic_tensor(linear_map())
    g0 = float(np.linalg.norm(gram_form(REFERENCE, REFERENCE)))
    rms, med, mx, ratios = [], [], [], []
    for k in range(reps):
        seed = 500000 + 911 * k
        assert seed != SEED + 777
        b = run_form(M, g0, sizes=(3, 5, 8, 12), seed=seed)
        coarse = np.array([abs(r["delta0005"]["rel_err"]) for r in b["configurations"]])
        rms.append(float(np.sqrt(np.mean(coarse**2))))
        med.append(float(np.median(coarse)))
        mx.append(float(coarse.max()))
        ratios.append(b["ratio_delta_scaling"])
    rms, med, mx, ratios = map(np.array, (rms, med, mx, ratios))

    def q(a):
        return {"min": float(a.min()), "q25": float(np.quantile(a, .25)), "median": float(np.median(a)),
                "q75": float(np.quantile(a, .75)), "p90": float(np.quantile(a, .9)),
                "p99": float(np.quantile(a, .99)), "max": float(a.max())}

    res = {"reps": reps, "seed_sequence": "500000 + 911*k (a8), pre-registered SEED+777 excluded",
           "rms_rel_err_delta0005": q(rms), "median_rel_err_delta0005": q(med),
           "max_rel_err_delta0005": q(mx), "ratio_delta_scaling": q(ratios),
           "false_fire_rms": {str(w): float(np.mean(rms > w)) for w in (0.01, 0.015, 0.02, 0.03, 0.05)},
           "false_fire_median": {str(w): float(np.mean(med > w)) for w in (0.005, 0.01, 0.02)},
           "ratio_outside_3_7": float(np.mean((ratios < 3) | (ratios > 7)))}
    print(json.dumps(res, indent=1))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
