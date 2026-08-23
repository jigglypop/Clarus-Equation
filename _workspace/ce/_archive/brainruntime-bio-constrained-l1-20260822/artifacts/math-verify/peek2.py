import json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, sys.argv[1])))
b = d.get("refine_best") or d.get("best")
for r in b[:3]:
    print(json.dumps({k: r[k] for k in r if k not in ("gates",)}, indent=0))
    print("gates", r.get("gates"))
