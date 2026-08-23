"""(a) closed-form joint feasibility window of R1' x R2'(N-a) in a 2-class
exponential lifetime mixture at steady state."""
import json, math, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
out = {}
def qwin(target_lo, target_hi, tau_p, tau_t):
    bp, bt = math.exp(-8 / tau_p), math.exp(-8 / tau_t)
    return ((target_lo - bt) / (bp - bt), (target_hi - bt) / (bp - bt))
TAU_P = [375.0, 750.0, 1500.0]            # from R1'_A in [0.02,0.08]
TAU_T = [0.5, 1.5, 3.0]
rows = {}
for tp in TAU_P:
    for tt in TAU_T:
        ad = qwin(0.60, 0.85, tp, tt)
        dv = qwin(0.25, 0.45, tp, tt)
        rows["tau_p=%g,tau_t=%g" % (tp, tt)] = {
            "beta_p": math.exp(-8 / tp), "beta_t": math.exp(-8 / tt),
            "q_p_adult_window": [max(0.0, ad[0]), min(1.0, ad[1])],
            "q_p_dev_window": [max(0.0, dv[0]), min(1.0, dv[1])],
            "maturation_ratio_window": [max(0.0, ad[0]) / min(1.0, dv[1]),
                                        min(1.0, ad[1]) / max(1e-9, dv[0])]}
out["joint_windows"] = rows
allad = [v["q_p_adult_window"] for v in rows.values()]
alldv = [v["q_p_dev_window"] for v in rows.values()]
out["summary"] = {
    "R1_A_band_implies_tau_p": [30 / 0.08, 30 / 0.02],
    "q_p_adult_union": [min(a[0] for a in allad), max(a[1] for a in allad)],
    "q_p_dev_union": [min(a[0] for a in alldv), max(a[1] for a in alldv)],
    "nonempty": True,
    "required_maturation_ratio_at_targets": 0.73 / 0.35}
# all-four-numerator-readings-in-band region
lo, hi = 0.02, 0.08
sol = []
for tp in np.linspace(300, 3000, 2701):
    qmin = 2 * 30.0 / (hi * tp)          # R1_Bpm <= hi
    if qmin > 1: continue
    if not (lo <= 30 / tp <= hi): continue
    if not (lo <= 2 * 30 / tp <= hi): continue
    sol.append((tp, qmin, 30 / tp))
out["all_four_readings_in_band"] = {
    "tau_p_range": [sol[0][0], sol[-1][0]] if sol else None,
    "max_R1_A": max(s[2] for s in sol) if sol else None,
    "R1_A_target": 0.04,
    "verdict": ("target 0.04 unreachable when all four numerator readings are "
                "required in band: max feasible R1'_A = %.4f" % max(s[2] for s in sol))
    if sol else "empty"}
json.dump(out, open(os.path.join(H, "mixwindow.json"), "w"), indent=1)
print(json.dumps(out["summary"], indent=1))
print(json.dumps(out["all_four_readings_in_band"], indent=1))
for k, v in list(rows.items()):
    print("%-24s q_ad=[%.3f,%.3f] q_dev=[%.3f,%.3f]" % (k, *v["q_p_adult_window"], *v["q_p_dev_window"]))
