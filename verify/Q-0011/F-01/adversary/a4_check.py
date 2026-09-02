"""Validate cat_stats against the independent O(n) and matrix implementations, and the card's."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from a_core import stats_fast, stats_matrix  # noqa: E402
from a4_cat import cat_parent, cat_stats  # noqa: E402
from check_families import tree_stats as card_stats  # noqa: E402

rng = np.random.default_rng(20260902)
worst = {"cat_vs_fast_c": 0.0, "cat_vs_card_c": 0.0, "cat_vs_matrix_c": 0.0}
cases = []
for trial in range(40):
    n = int(rng.integers(12, 400))
    k = int(rng.integers(1, min(n - 2, 40)))
    sizes = sorted(set(int(x) for x in rng.integers(1, n, size=k)), reverse=True)
    p = cat_parent(n, sizes)
    a = cat_stats(n, sizes)
    b = stats_fast(p)
    c = card_stats(p)
    worst["cat_vs_fast_c"] = max(worst["cat_vs_fast_c"], abs(a["c"] - b["c"]))
    worst["cat_vs_card_c"] = max(worst["cat_vs_card_c"], abs(a["c"] - c["c"]))
    if n <= 200:
        m = stats_matrix(p)
        worst["cat_vs_matrix_c"] = max(worst["cat_vs_matrix_c"], abs(a["c"] - m["c"]))
    cases.append({"n": n, "k": len(sizes), "c": a["c"]})
print(json.dumps({"worst": worst, "n_cases": len(cases)}, indent=2))
(HERE / "a4_check.json").write_text(json.dumps({"worst": worst, "cases": cases}, indent=2), encoding="utf-8")
