import json
from pathlib import Path
H = Path(__file__).resolve().parent
p1 = json.loads((H / "_part1.json").read_text(encoding="utf-8"))
t = json.loads((H / "audit_trials.json").read_text(encoding="utf-8"))
print("gamma_her_exact_grid", p1["gamma_her_exact_grid"])
print("gamma_iid_exact_grid", p1["gamma_iid_exact_grid"])
print("X32_exact", p1["X32_exact"])
print("ratio128_exact", p1["ratio128_exact"])
print("E_D:", {k: round(v["E_D"], 3) for k, v in p1["cayley_exact"].items()})
print("ratio_pred:", {k: round(v["ratio_pred"], 4) for k, v in p1["cayley_exact"].items()})
print("E_trHk32", p1["cayley_exact"]["32"]["E_trHk"])
print("window slack:", {k: (round(v["low_slack"], 4), round(v["high_slack"], 4)) for k, v in p1["window_vs_value_pm_u"].items()})
print()
for key in ("tails_her", "tails_iid"):
    print(key, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in t[key].items()})
print("tails_mix", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in t["tails_mix"].items()})
print("npz shapes", t["npz_shapes"])
