"""R2'(N) sub-definition: window-length dependence.
(N-a) birth cohort  -> independent of window LENGTH
(N-b) prevalence    -> length-biased, depends on the section-6 window length,
                       which is a declared (fit-forbidden) design constant."""
import json, math, os
H = os.path.dirname(os.path.abspath(__file__))
def n_a(q, tt, tp): return q * math.exp(-8 / tp) + (1 - q) * math.exp(-8 / tt)
def n_b(q, tt, tp, W):
    cP, cT = q * (W + tp), (1 - q) * (W + tt)
    return (cP * math.exp(-8 / tp) + cT * math.exp(-8 / tt)) / (cP + cT)
out = {}
for tt, tp in ((1.5, 750.0), (3.0, 750.0), (1.5, 375.0)):
    q = 0.372 if (tt, tp) == (1.5, 750.0) else None
    # calibrate q so that (N-b) hits 0.73 at W=200, then vary W
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m = (lo + hi) / 2
        if n_b(m, tt, tp, 200.0) < 0.73: lo = m
        else: hi = m
    q = (lo + hi) / 2
    row = {"q_p_calibrated_at_W200": q,
           "N_b_vs_W": {str(int(W)): n_b(q, tt, tp, W) for W in (25, 50, 100, 200, 400)},
           "N_a_any_W": n_a(q, tt, tp)}
    row["band_0.60_0.85_W_range_ok"] = [W for W in (25, 50, 100, 200, 400)
                                        if 0.60 <= n_b(q, tt, tp, W) <= 0.85]
    out["tau_t=%s,tau_p=%s" % (tt, tp)] = row
out["note"] = ("Under (N-b) the section-6 window length moves the gate value by "
               "more than half the band width at fixed dynamics; under (N-a) it "
               "does not enter at all.  The contract leaves the sub-definition open "
               "('[unfinished: original denominator unverified, (N) provisionally adopted]').")
json.dump(out, open(os.path.join(H, "windowdep.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
