"""Confirmatory GFP-matched killing test for the docs/7_AGI/25 local memory gate.

Preregistration: artifacts/agi/local_memory_gfp_matched_preregistration.json
Statistic: per-target delta' = (R2_L - R2_C) / (1 - R2_C) on the test block,
recording-level median-of-ratios; exact one-sided Mann-Whitney AML32 > AML18.
Imports the frozen local_memory.py (no modification). n_null_shifts=1 because
the implementation requires >= 1; circular-shift nulls are not scored here.
"""
import itertools
import json
import sys
import time as _time
from pathlib import Path

import numpy as np

ROOT = Path(r"c:/Users/dongh/OneDrive/Desktop/Clarus-Equation")
sys.path.insert(0, str(ROOT / "reality_stone" / "python"))

from reality_stone.clarus.cloudcell_dynamics import load_predictioncode_recordings
from reality_stone.clarus.local_memory import evaluate_local_memory_recording

GUARD = 0.01  # exclude targets with (1 - r2_current) < GUARD (preregistered)
DATA = {
    "AML32": ROOT / "data/external/cloudcell/extracted/AML32_moving",
    "AML18": ROOT / "data/external/cloudcell/extracted/AML18_moving",
}


def mw_exact_p_greater(x, y):
    """Exact one-sided Mann-Whitney: P(U >= u_obs) over all C(n+m, n) splits."""
    n, m = len(x), len(y)
    u_obs = sum(1.0 for a in x for b in y if a > b) + 0.5 * sum(
        1 for a in x for b in y if a == b
    )
    allv = list(x) + list(y)
    idx = range(n + m)
    count = 0
    total = 0
    for comb in itertools.combinations(idx, n):
        cset = set(comb)
        xs = [allv[i] for i in comb]
        ys = [allv[i] for i in idx if i not in cset]
        u = sum(1.0 for a in xs for b in ys if a > b) + 0.5 * sum(
            1 for a in xs for b in ys if a == b
        )
        total += 1
        if u >= u_obs:
            count += 1
    return u_obs, count / total


def main():
    recordings = {k: load_predictioncode_recordings(v) for k, v in DATA.items()}
    result = {
        "artifact_type": "clarus_local_memory_gfp_matched_result",
        "artifact_version": 1,
        "phase": "confirmatory",
        "preregistration": "artifacts/agi/local_memory_gfp_matched_preregistration.json",
        "statistic": (
            "per-target delta' = (r2_local_memory - r2_current_nonlinear) / "
            "(1 - r2_current_nonlinear) on test block; recording D_r = median of "
            "included targets (median-of-ratios); guard: exclude targets with "
            "(1 - r2_current_nonlinear) < 0.01"
        ),
        "test": (
            "exact one-sided Mann-Whitney U on recording medians, "
            "AML32 (n=7) > AML18 (n=11), alpha=0.05; primary h=6, secondary h=1"
        ),
        "horizons": {},
    }

    for h in (6, 1):
        block = {"strains": {}}
        for strain, recs in recordings.items():
            strain_rows = []
            for rec in recs:
                t0 = _time.time()
                gate = evaluate_local_memory_recording(
                    rec, horizon_steps=h, n_null_shifts=1
                )
                dprimes = []
                excluded = 0
                for s in gate.scores:
                    headroom = 1.0 - s.r2_current_nonlinear
                    if headroom < GUARD:
                        excluded += 1
                        continue
                    dprimes.append((s.r2_local_memory - s.r2_current_nonlinear) / headroom)
                row = {
                    "recording_id": gate.recording_id,
                    "n_targets_evaluated": len(gate.scores),
                    "n_targets_excluded_headroom_guard": excluded,
                    "n_targets_included": len(dprimes),
                    "median_delta_prime": float(np.median(dprimes)) if dprimes else None,
                    "delta_prime_quartiles": (
                        [float(q) for q in np.percentile(dprimes, [25, 50, 75])]
                        if dprimes
                        else None
                    ),
                }
                strain_rows.append(row)
                print(
                    f"h={h} {strain} {gate.recording_id}: "
                    f"n={len(dprimes)} (excl {excluded}) "
                    f"median delta'={row['median_delta_prime']:.4f} "
                    f"[{_time.time()-t0:.0f}s]",
                    flush=True,
                )
            block["strains"][strain] = strain_rows

        xg = [r["median_delta_prime"] for r in block["strains"]["AML32"]]
        xf = [r["median_delta_prime"] for r in block["strains"]["AML18"]]
        u, p = mw_exact_p_greater(xg, xf)
        med_g, med_f = float(np.median(xg)), float(np.median(xf))
        block["recording_medians_aml32"] = xg
        block["recording_medians_aml18"] = xf
        block["median_aml32"] = med_g
        block["median_aml18"] = med_f
        block["median_difference_aml32_minus_aml18"] = med_g - med_f
        block["mann_whitney_u"] = u
        block["mann_whitney_u_max"] = len(xg) * len(xf)
        block["exact_one_sided_p_aml32_greater"] = p
        result["horizons"][f"h{h}"] = block
        print(
            f"h={h}: AML32 med={med_g:.4f}, AML18 med={med_f:.4f}, "
            f"diff={med_g-med_f:+.4f}, U={u:.1f}/{len(xg)*len(xf)}, exact p={p:.4f}",
            flush=True,
        )

    prim = result["horizons"]["h6"]
    p6 = prim["exact_one_sided_p_aml32_greater"]
    diff6 = prim["median_difference_aml32_minus_aml18"]
    if p6 <= 0.05:
        verdict = "PASS_PENDING_CONFIRMATIONS"
        detail = (
            "primary p <= 0.05; promotion requires eta adjustment and "
            "whitened-GFP replication per preregistration"
        )
    elif diff6 <= 0.02:
        verdict = "KILL"
        detail = (
            "primary p > 0.05 and median difference <= 0.02: the 'neural temporal "
            "memory' interpretation of docs/7_AGI/25 is dead for both horizons; "
            "the gate claim is demoted to non-Markovianity of the measurement process"
        )
    else:
        verdict = "INCONCLUSIVE"
        detail = "primary p > 0.05 but median difference > 0.02 (power-limited)"
    result["verdict"] = verdict
    result["verdict_detail"] = detail
    result["decision_inputs"] = {
        "primary_p_h6": p6,
        "primary_median_difference_h6": diff6,
        "kill_requires": "p > 0.05 AND median difference <= 0.02",
    }

    out = ROOT / "artifacts/agi/local_memory_gfp_matched_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {verdict}\nwritten: {out}", flush=True)


if __name__ == "__main__":
    main()
