import json
from pathlib import Path
d = json.loads((Path(__file__).parent / "rerun_k1_homology.json").read_text(encoding="utf-8"))
for r in d["per_size"]:
    b = r["boundary2"]
    print("n=%d  E=%d F=%d  b1=%d  rank_d2(Q)=%d rank_d2(GF2)=%d  H1=%d  H2=ker_d2=%d"
          % (r["n"], b["E_undirected"], b["F"], r["b1_skeleton"], b["rank_d2_Q"],
             b["rank_d2_GF2"], r["H1_dim"], r["H2_dim_ker_d2"]))
    print("    b1_fine_only=%d  M_hist=%s" % (r["b1_fine_only"], r["M_histogram"]))
    print("    census=%s" % (r["triangle_census"],))
