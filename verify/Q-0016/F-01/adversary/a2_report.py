import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / "a2_sign_and_limits.json").read_text(encoding="utf-8"))
print("census", json.dumps(R["census_by_n"]))
print("total", R["total_shapes"], "viol", R["total_violations"], "min n", R["smallest_n_with_violation"])
print("smallest examples", json.dumps(R["smallest_n_examples"], ensure_ascii=False))
for w, s in zip(R["worst_10"][:6], R["worst_10_shape"][:6]):
    print("worst n=%d parent=%s ratio=%.5f %s" % (w["n"], w["parent"], w["ratio"], json.dumps(s)))
print("terminal-fork rows", [(r["L"], r["n"], round(r["ratio"], 5)) for r in R["chain_with_terminal_fork"]["rows"][:16]])
print("terminal-fork min", R["chain_with_terminal_fork"]["min_ratio"], "at L", R["chain_with_terminal_fork"]["argmin_L"])
print("deep-fork rows", [(r["L"], r["n"], round(r["ratio"], 5)) for r in R["deep_chain_with_late_fork"][:12]])
