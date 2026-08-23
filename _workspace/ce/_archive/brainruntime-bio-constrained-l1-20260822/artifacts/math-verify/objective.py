"""(f) contract section 4-1 objective: is it defined on EVERY row of section 5?
Two-branch rule: ratio rows -> log((x+eps)/(x*+eps))^2 ; inequality rows ->
hinge max(0,violation)^2.  The contract does not assign each row to a branch.
This script prices the ambiguity for the rows whose branch is undetermined."""
import json, os
import numpy as np
EPS = 1e-6
H = os.path.dirname(os.path.abspath(__file__))
# a plausible near-solution operating point (values only, for pricing)
pt = {"R1_A": 0.045, "R2dev_Na": 0.33, "R2ad_Na": 0.71, "R3a": 0.17,
      "R3b": 0.02, "R4": 0.01, "R5": 1.45, "R6": 1.25}
tgt = {"R1_A": 0.04, "R2dev_Na": 0.35, "R2ad_Na": 0.73, "R3a": 0.18,
       "R4": 0.0, "R5": 1.5, "R6": 1.0}
ineq = {"R3b": ("le", 0.05), "R4": ("le", 0.05), "R6": ("ge", 1.3)}


def logterm(x, xs): return float(np.log((x + EPS) / (xs + EPS)) ** 2)
def hinge(x, d, b): return float((max(0.0, x - b) if d == "le" else max(0.0, b - x)) ** 2)


out = {"point": pt, "rows": {}}
for k in pt:
    r = {}
    if k in tgt: r["log_branch"] = logterm(pt[k], tgt[k])
    if k in ineq: r["hinge_branch"] = hinge(pt[k], *ineq[k])
    out["rows"][k] = r
# total under the two readings of the ambiguous rows (R4 target 0, R6 target >1)
readA = sum(logterm(pt[k], tgt[k]) for k in ("R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R5")) \
    + hinge(pt["R3b"], *ineq["R3b"]) + hinge(pt["R4"], *ineq["R4"]) + hinge(pt["R6"], *ineq["R6"])
readB = sum(logterm(pt[k], tgt[k]) for k in ("R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R5", "R4", "R6")) \
    + hinge(pt["R3b"], *ineq["R3b"])
out["total_all_inequality_branch_for_R4_R6"] = readA
out["total_ratio_branch_for_R4_R6"] = readB
out["R4_log_branch_term"] = logterm(pt["R4"], 0.0)
out["R4_share_of_total_under_ratio_branch"] = logterm(pt["R4"], 0.0) / readB
out["R4_log_branch_vs_R4_value"] = {"%.4g" % v: logterm(v, 0.0)
                                    for v in (1e-4, 1e-3, 1e-2, 5e-2, 0.2)}
out["undeclared_rows"] = [
    "R2'-adult: clause 'greater than R2'-dev and monotone maturation' has no band, "
    "no target, no operational definition (how many checkpoints, what tolerance) "
    "and therefore no objective term",
    "R5': clause 'monotone decrease thereafter' likewise has no term",
    "R3b': strict positivity (0,0.05] is not enforceable by hinge (measure zero) "
    "and is structurally guaranteed by section 3.2 lambda(w)>0 -> zero discriminating power",
    "L2 E1/E2: section 5 counts them in 'effective conditions 10' but section 4-1 "
    "must not fit them (L2 is 'not to be imposed'), so they are not objective rows",
]
json.dump(out, open(os.path.join(H, "objective.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
