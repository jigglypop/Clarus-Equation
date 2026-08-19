from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
results = json.loads((HERE / "realdata-results.json").read_text(encoding="utf-8"))

assert abs(results["e15"]["epochs"]["0-1"]["official_bootstrap_probability"] - 0.0161057) < 1e-10
assert abs(results["e15"]["epochs"]["5-6"]["official_bootstrap_probability"] - 0.00133728) < 1e-10
assert results["e15"]["epochs"]["0-1"]["direction_sd_minus_nsd"] < 0
assert results["e15"]["same_window_branching_replay"]["status"] == "UNTESTABLE"

assert results["e19"]["participants"] == 34
assert results["e19"]["outcomes"]["item"]["spearman_rho"] < 0
assert results["e19"]["outcomes"]["category"]["spearman_rho"] > 0
assert results["e13"]["fig2d_prediction"]["mean_difference_DCC_minus_Shuffle"] > 0
assert results["e02"]["status"] == "ACCESS_BLOCKED"

with (HERE / "realdata-manifest.csv").open(newline="", encoding="utf-8") as handle:
    manifest = list(csv.DictReader(handle))
assert len(manifest) == results["manifest_file_count"]
assert all(len(row["sha256"]) == 64 and int(row["bytes"]) >= 0 for row in manifest)

print(f"OK realdata: {len(manifest)} hashed files; E15/E19/E13 checks passed")
