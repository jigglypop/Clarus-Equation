import json, math, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent
d = json.loads((H / "audit_law_defect.json").read_text(encoding="utf-8"))
f = d["eps_star_fit"]
print("eps_star", f["eps_star"], "chi2", round(f["chi2"], 2), "dof", f["dof"], "chi2/dof", round(f["chi2_over_dof"], 3))
print("worst z:", sorted(((round(p["z"], 2), p["mode"], p["n"]) for p in f["points"]), key=lambda t: -abs(t[0]))[:4])

# K2 seed-to-seed
X = [0.8193510400305117, 0.6972276247279885, 0.7021034213088991, 0.6589481472018541,
     0.7525652711020717, 0.7871119186526794]
m, s = st.mean(X), st.stdev(X)
print("mix X seeds:", [round(x, 4) for x in X])
print("mix mean", round(m, 4), "sd", round(s, 4), "se_mean", round(s / math.sqrt(len(X)), 4))
prereg, hi = 0.7406, 0.99
def tail(center, sd):
    zz = (hi - center) / sd
    return 0.5 * math.erfc(zz / math.sqrt(2))
print("P(X>0.99 | center=prereg 0.7406, sd=seed sd)", "%.2e" % tail(prereg, s))
print("P(X>0.99 | center=seed mean, sd=seed sd)", "%.2e" % tail(m, s))
print("P(X>0.99 | center=observed 0.8194, sd=boot 0.088)", "%.3f" % tail(0.8193510400305117, 0.0880432935980372))
print("z of observed vs prereg using seed sd", round((0.81935 - prereg) / s, 2))
print("z of seed mean vs prereg", round((m - prereg) / (s / math.sqrt(len(X))), 2))

her = [0.5576106551570001, 0.5250482857797666, 0.5374387117450791]
rat = [34.24199061956648, 30.636957301106236, 32.620713939414124]
iidl = [-0.4799658222826968, -0.48194574128909007, -0.4845347675797436]
for name, v, exact in (("her_slope", her, 0.5302004033210348), ("her_ratio_128", rat, 32.55366798958406),
                       ("iid_slope", iidl, -0.478347557133974)):
    print(name, [round(x, 4) for x in v], "mean", round(st.mean(v), 4), "sd", round(st.stdev(v), 4),
          "exact", round(exact, 4), "z_mean", round((st.mean(v) - exact) / (st.stdev(v) / math.sqrt(3)), 2))
