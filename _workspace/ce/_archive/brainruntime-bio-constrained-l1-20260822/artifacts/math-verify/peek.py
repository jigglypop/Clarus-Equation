import json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, "lhs_summary.json")))
pr = d["pairs"]
print("ZERO pairs:", [k for k in pr if pr[k] == 0])
print("--- 16 tightest pairs ---")
for k, v in sorted(pr.items(), key=lambda x: x[1])[:16]:
    print(" %-28s %d" % (k, v))
print("--- best 4 points ---")
for b in d["best"][:4]:
    print(json.dumps(b, indent=0))
print("budget_identity", json.dumps(d["budget_identity_rel_err"]))
