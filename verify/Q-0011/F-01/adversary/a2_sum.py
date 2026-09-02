import collections
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
rows = d["all"] if "all" in d else d
s = d.get("summary", {})
print("rows", len(rows), "c_min", min(r["c"] for r in rows), "c_max", max(r["c"] for r in rows))
best = collections.defaultdict(lambda: [9.0, 0.0, None, None])
for r in rows:
    b = best[r["family"]]
    if r["c"] < b[0]:
        b[0], b[2] = r["c"], r["n"]
    if r["c"] > b[1]:
        b[1], b[3] = r["c"], r["n"]
for k, v in sorted(best.items(), key=lambda kv: kv[1][0]):
    print(f"{k:24s} min={v[0]:.4f}(n={v[2]}) max={v[1]:.4f}(n={v[3]})")
