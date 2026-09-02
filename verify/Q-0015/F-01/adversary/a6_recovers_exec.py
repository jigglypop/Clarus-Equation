"""A6: (1) execute every `recovers` limit with the REAL code (not symbolic substitution);
       (2) test whether theta carries ANY information independent of eps in the kill script.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "verify" / "Q-0015" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
import check_theta as C  # noqa: E402

out = {}
CT = math.sqrt(3) / 2

# ---------- (2) does theta carry information beyond eps?  (run FIRST, it is the load-bearing one)
rng = np.random.default_rng(20260902)
worst_rel = 0.0; pairs = []
for n in (2, 3, 8, 32, 128):
    for _ in range(40):
        lab = rng.standard_normal((n, 4, 4))
        e, t = C.eps_and_theta(C.block_triple(lab))
        pred = CT * e / math.sqrt(1 - e * e)
        worst_rel = max(worst_rel, abs(t - pred) / pred)
        pairs.append((n, e, t))
# also at a DELIBERATELY LARGE delta where eps is O(1) -- still an identity?
big = []
for delta in (0.05, 0.5, 2.0):
    for _ in range(20):
        lab = rng.standard_normal((4, 4, 4))
        tri = C.block_triple(lab, delta=delta)
        if not np.all(np.isfinite(tri)):
            continue
        e, t = C.eps_and_theta(tri)
        big.append({"delta": delta, "eps": e, "theta": t,
                    "pred": CT * e / math.sqrt(1 - e * e) if e < 1 else None,
                    "rel_err": abs(t - CT * e / math.sqrt(1 - e * e)) / (CT * e / math.sqrt(1 - e * e))
                    if 0 < e < 1 else None})
out["theta_is_a_function_of_eps"] = {
    "max_rel_err_over_200_sampled_blocks": worst_rel,
    "large_delta_probe": big[:8],
    "large_delta_max_rel_err": max([b["rel_err"] for b in big if b["rel_err"] is not None] or [None]),
    "eps_range_large_delta": [min(b["eps"] for b in big), max(b["eps"] for b in big)],
    "note": ("check_theta.eps_and_theta computes BOTH numbers from the same 3x3 gram: "
             "eps = ||tlG||/||G||, theta = 1.5||tlG||/trG.  theta is a fixed deterministic "
             "function of eps.  No holonomy, connection or parallel transport is evaluated in "
             "modes blk/face/scale.  Therefore K1,K2,K5,K7 and P1,P4,P5,P6 cannot discriminate "
             "the card's power-1 law from the power-1/2 alternative: the script hard-codes power 1 "
             "at line `theta = 1.5 * tl_norm / trace(gram)`.")}
# counterfactual: what would the 'power 1/2 law' have produced from the SAME data?
sizes = (8, 16, 32, 64, 128)
her_e, her_t = {}, {}
for n in sizes:
    rng = np.random.default_rng(4321)
    from driver_numbers import uniform_rooted_tree  # noqa: E402
    vals = []
    for _ in range(48):
        parent = uniform_rooted_tree(n, rng)
        lab = C.heritable_labels(parent, rng.standard_normal((n, 4, 4)))
        vals.append(C.eps_and_theta(C.block_triple(lab)))
    her_e[n] = float(np.sqrt(np.mean([v[0] ** 2 for v in vals])))
    her_t[n] = float(np.sqrt(np.mean([v[1] ** 2 for v in vals])))
sl = lambda ys: float(np.polyfit(np.log(sizes), np.log([ys[n] for n in sizes]), 1)[0])
half = {n: math.sqrt(her_e[n]) for n in sizes}
out["power_half_counterfactual"] = {
    "slope_eps": sl(her_e), "slope_theta_card": sl(her_t), "slope_sqrt_eps": sl(half),
    "note": ("the '1/2 law' baseline 0.2651 is exactly half the eps slope BY ARITHMETIC; it is a "
             "different DECLARED mapping, not a state of the world the run can select between.")}

# ---------- (1) recovers executed
rec = {}
# R0: exactly simple block -- all cells on the common conformal-metric orbit (13.3)
lab0 = np.zeros((7, 4, 4))
e0, t0 = C.eps_and_theta(C.block_triple(lab0))
same = np.repeat(np.random.default_rng(5).standard_normal((1, 4, 4)), 7, axis=0)
e1, t1 = C.eps_and_theta(C.block_triple(same))
rec["R0_common_orbit"] = {"identical_cells_zero_label": {"eps": e0, "theta": t0},
                          "identical_cells_same_random_label": {"eps": e1, "theta": t1},
                          "card_expects": 0.0, "executed": True}
# R1: 13.5 coherent two-species, ratio p, at several n -- is theta really n-independent?
coh = {}
for n in (8, 16, 32, 64):
    rng2 = np.random.default_rng(99)
    a = rng2.standard_normal((4, 4)); b = rng2.standard_normal((4, 4))
    k = n // 2
    lab = np.stack([a] * k + [b] * (n - k))
    e, t = C.eps_and_theta(C.block_triple(lab))
    coh[n] = {"eps": e, "theta": t, "theta_over_eps": t / e}
rec["R1_coherent_two_species_p=1/2"] = {
    "by_n": coh,
    "theta_ratio_64_over_8": coh[64]["theta"] / coh[8]["theta"],
    "card_expects_n_independent": 1.0, "executed": True}
# R4: tetrad rescale on a real random block
rng3 = np.random.default_rng(17)
lab = rng3.standard_normal((5, 4, 4))
resc = {}
for a in (0.4, 1.0, 2.5, 40.0):
    e, t = C.eps_and_theta(C.block_triple(lab, scale=a))
    resc[str(a)] = {"eps": e, "theta": t, "theta_over_eps": t / e}
rec["R4_tetrad_rescale"] = {"by_alpha": resc,
                            "ratio_2p5_over_1": resc["2.5"]["theta_over_eps"] / resc["1.0"]["theta_over_eps"],
                            "executed": True}
# R3: flat limit of phi_kappa (OUT OF THE CARD'S OWN SCOPE -- isotropic channel)
def phi(kap):
    u = kap / 4.0
    return (kap / 2.0) / math.sqrt((1 + u / 2) * (u / 2)) * math.atan(math.sqrt((u / 2) / (1 + u / 2)))
rec["R3_flat_limit_phi_kappa_over_kappa"] = {
    str(k): phi(k) / k for k in (1.0, 1e-2, 1e-4, 1e-6)}
rec["R3_note"] = ("phi_kappa is the ISOTROPIC channel, which scope[1] declares the card does NOT "
                  "predict.  A limit of a quantity outside the card's own scope cannot constrain it.")
out["recovers_executed"] = rec
print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
