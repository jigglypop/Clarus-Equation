"""A4: block mode with INDEPENDENT seeds + OFF-GRID n.

Adjudicates the two remaining smoke-size window escapes reported by prover
(theta_slope_her = 0.654 vs window [0.43,0.63]; cross_32 = 4.969 vs [3.5855,4.3855])
and measures how large eps actually gets on the frozen grid (the card parks the
"large-eps beyond small-angle" question).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "verify" / "Q-0015" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
import check_theta as C  # noqa: E402
from driver_numbers import cayley_exact, uniform_rooted_tree  # noqa: E402


def rms(a): return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))
def slope(xs, ys): return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def block_run(sizes, trials, seed_her, seed_iid):
    her_t, her_e, iid_t, iid_e, eps_max = {}, {}, {}, {}, {}
    for n in sizes:
        rng = np.random.default_rng(seed_her)
        pr = []
        for _ in range(trials):
            parent = uniform_rooted_tree(n, rng)
            labels = C.heritable_labels(parent, rng.standard_normal((n, 4, 4)))
            pr.append(C.eps_and_theta(C.block_triple(labels)))
        her_e[n], her_t[n] = rms([p[0] for p in pr]), rms([p[1] for p in pr])
        eps_max[n] = float(np.max([p[0] for p in pr]))
        rng = np.random.default_rng(seed_iid)
        pr = [C.eps_and_theta(C.block_triple(rng.standard_normal((n, 4, 4)))) for _ in range(trials)]
        iid_e[n], iid_t[n] = rms([p[0] for p in pr]), rms([p[1] for p in pr])
    ratios = [her_t[n] / her_e[n] for n in sizes] + [iid_t[n] / iid_e[n] for n in sizes]
    return {"sizes": list(sizes), "trials": trials, "seed_her": seed_her, "seed_iid": seed_iid,
            "c_theta_ratio": float(np.median(ratios)),
            "c_theta_ratio_max_dev_from_sqrt3_2": float(np.max(np.abs(np.array(ratios) - math.sqrt(3) / 2))),
            "theta_slope_her": slope(sizes, [her_t[n] for n in sizes]),
            "theta_slope_iid": slope(sizes, [iid_t[n] for n in sizes]),
            "eps_slope_her": slope(sizes, [her_e[n] for n in sizes]),
            "eps_slope_iid": slope(sizes, [iid_e[n] for n in sizes]),
            "theta_minus_eps_slope_her": slope(sizes, [her_t[n] for n in sizes]) - slope(sizes, [her_e[n] for n in sizes]),
            "eps_her_rms": {str(n): her_e[n] for n in sizes},
            "eps_her_max": {str(n): eps_max[n] for n in sizes},
            "theta_her_rms": {str(n): her_t[n] for n in sizes}}


out = {}
t0 = time.time()

# ---- (0) exact F-02 combinatorics: what slope does the model itself predict on each grid?
ex = {n: cayley_exact(n)["E_D"] for n in (6, 8, 12, 16, 24, 32, 48, 64, 96, 128)}
pred = {n: math.sqrt(ex[n]) / n for n in ex}
out["exact_model_slopes"] = {
    "full_grid_8_128": slope((8, 16, 32, 64, 128), [pred[n] for n in (8, 16, 32, 64, 128)]),
    "smoke_grid_8_32": slope((8, 16, 32), [pred[n] for n in (8, 16, 32)]),
    "offgrid_6_96": slope((6, 12, 24, 48, 96), [pred[n] for n in (6, 12, 24, 48, 96)]),
    "iid_full_grid": slope((8, 16, 32, 64, 128), [math.sqrt(n - 1) / n for n in (8, 16, 32, 64, 128)]),
    "iid_smoke_grid": slope((8, 16, 32), [math.sqrt(n - 1) / n for n in (8, 16, 32)]),
    "iid_offgrid": slope((6, 12, 24, 48, 96), [math.sqrt(n - 1) / n for n in (6, 12, 24, 48, 96)]),
    "E_D_32": ex[32], "sqrtED32_over_32": pred[32],
    "cross_exact_full": pred[32] / (math.sqrt(10 / 9) / 3),
}

# ---- (1) smoke-size sampling distribution of theta_slope_her and cross (24 trials, {8,16,32})
sm = []
for k in range(24):
    r = block_run((8, 16, 32), 24, 30000 + 13 * k, 70000 + 13 * k)
    rng = np.random.default_rng(31000 + 13 * k)
    face_d0 = rms([C.eps_and_theta(C.block_triple(np.stack([
        (lambda xi: (xi[0], xi[0] + xi[8], xi[0] + xi[8] + xi[9]))(rng.standard_normal((10, 4, 4)))
    ][0])))[1] for _ in range(128)])
    r["cross_32"] = r["theta_her_rms"]["32"] / face_d0
    sm.append(r)
sl = np.array([r["theta_slope_her"] for r in sm]); cr = np.array([r["cross_32"] for r in sm])
sli = np.array([r["theta_slope_iid"] for r in sm])
out["smoke_size_distribution"] = {
    "n_replicates": int(sl.size),
    "theta_slope_her": {"mean": float(sl.mean()), "sd": float(sl.std(ddof=1)),
                        "min": float(sl.min()), "max": float(sl.max()),
                        "frac_at_or_above_0.654": float((sl >= 0.654).mean()),
                        "frac_outside_K2_window": float(((sl < 0.43) | (sl > 0.63)).mean())},
    "theta_slope_iid": {"mean": float(sli.mean()), "sd": float(sli.std(ddof=1)),
                        "frac_outside_K7_window": float(((sli < -0.58) | (sli > -0.38)).mean())},
    "cross_32": {"mean": float(cr.mean()), "sd": float(cr.std(ddof=1)),
                 "min": float(cr.min()), "max": float(cr.max()),
                 "frac_at_or_above_4.969": float((cr >= 4.969).mean()),
                 "frac_outside_P6_window": float(((cr < 3.5855) | (cr > 4.3855)).mean())}}
out["t_after_smoke"] = time.time() - t0

# ---- (2) full-size, INDEPENDENT seeds, ON the frozen grid
full_on = [block_run((8, 16, 32, 64, 128), 256, 101 + k, 901 + k) for k in range(2)]
out["full_ongrid_indep_seeds"] = full_on
# ---- (3) full-size, OFF the frozen grid
full_off = [block_run((6, 12, 24, 48, 96), 256, 201 + k, 801 + k) for k in range(2)]
out["full_offgrid_indep_seeds"] = full_off

out["elapsed_s"] = time.time() - t0
print(json.dumps(out, indent=2, ensure_ascii=False))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
