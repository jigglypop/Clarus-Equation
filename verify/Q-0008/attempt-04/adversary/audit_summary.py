"""adversary attempt-04 audit: consolidation. Cross-checks stdout logs vs snapshot vs shared result.json
vs attempt result.json, and collects every number produced by this audit."""
from __future__ import annotations
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A04 = HERE.parent
ROOT = HERE.parents[3]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
KEYS = ("her_slope", "her_ratio_128", "mix_X_32", "iid_slope", "defect_ratio_64_over_8", "defect_slope")

snap = json.loads((A04 / "F-02_result_snapshot.json").read_text(encoding="utf-8"))
shared = json.loads((F02 / "result.json").read_text(encoding="utf-8"))
res04 = json.loads((A04 / "result.json").read_text(encoding="utf-8"))
logs = {m: json.loads((A04 / ("log_" + m + ".txt")).read_text(encoding="utf-8")) for m in ("her", "mix", "iid", "defect")}
log_stats = {}
for m, d in logs.items():
    log_stats.update(d["stats"])

out = {"chain_of_custody": {}}
for k in KEYS:
    out["chain_of_custody"][k] = {
        "stdout_log": log_stats[k], "snapshot": snap["stats"][k], "shared_now": shared["stats"][k],
        "attempt_result": res04["results"][k]["value"],
        "all_identical": len({repr(log_stats[k]), repr(snap["stats"][k]), repr(shared["stats"][k]),
                             repr(res04["results"][k]["value"])}) == 1,
        "verdict_log": logs[res04["results"][k]["mode"]]["verdict"][k],
    }
out["snapshot_vs_shared_four_blocks_identical"] = all(
    snap[m] == shared[m] for m in ("her", "mix", "iid", "defect"))
out["shared_has_qspine_not_in_snapshot"] = ("qspine" in shared) and ("qspine" in snap)
out["repro_bit_exact"] = {}
for tag, f, keys in (("her", "repro_her.json", ("her_slope", "her_ratio_128")),
                     ("mix", "repro_mix.json", ("mix_X_32",)), ("iid", "repro_iid.json", ("iid_slope",))):
    d = json.loads((HERE / f).read_text(encoding="utf-8"))
    out["repro_bit_exact"][tag] = {k: (repr(d[k]) == repr(snap["stats"][k])) for k in keys}
    out["repro_bit_exact"][tag]["min_det_audit"] = d["min_det_audit"]
seeds = []
for f in ("seed_100003.json", "seed_200003.json"):
    d = json.loads((HERE / f).read_text(encoding="utf-8"))
    seeds.append({"seed": d["seed"], "her_slope": d["her_slope"], "her_ratio_128": d["her_ratio_128"],
                  "mix_X_32": d["mix_X_32"], "iid_slope": d["iid_slope"],
                  "min_det_rejections": d["min_det_audit"]["nan_rejections"],
                  "min_det": d["min_det_audit"]["min_det"]})
mix_extra = []
for s in (300007, 400009, 500011):
    d = json.loads((HERE / ("mixonly_" + str(s) + ".json")).read_text(encoding="utf-8"))
    mix_extra.append({"seed": d["seed"], "mix_X_32": d["mix_X_32"]})
out["other_seeds"] = seeds
out["mix_extra_seeds"] = mix_extra
W = {"her_slope": (0.43, 0.63), "her_ratio_128": (26.0, 39.1), "mix_X_32": (0.49, 0.99), "iid_slope": (-0.58, -0.38)}
allin = []
for s in seeds:
    allin.append(all(W[k][0] <= s[k] <= W[k][1] for k in ("her_slope", "her_ratio_128", "mix_X_32", "iid_slope")))
out["other_seeds_all_in_window"] = allin
out["mix_extra_all_in_window"] = all(0.49 <= m["mix_X_32"] <= 0.99 for m in mix_extra)
X = [snap["stats"]["mix_X_32"]] + [s["mix_X_32"] for s in seeds] + [m["mix_X_32"] for m in mix_extra]
mean = sum(X) / len(X)
sd = math.sqrt(sum((x - mean) ** 2 for x in X) / (len(X) - 1))
out["k2_seed_ensemble"] = {"values": X, "mean": mean, "sd": sd, "n": len(X),
                           "z_mean_vs_prereg": (mean - 0.7406) / (sd / math.sqrt(len(X))),
                           "P_exceed_0.99_centered_prereg": 0.5 * math.erfc((0.99 - 0.7406) / sd / math.sqrt(2)),
                           "P_exceed_0.99_centered_observed_bootSE": 0.5 * math.erfc((0.99 - X[0]) / 0.0880432935980372 / math.sqrt(2))}
for name in ("_part1", "audit_trials", "tree_surrogate", "audit_law_defect"):
    p = HERE / (name + ".json")
    if p.is_file():
        out[name] = json.loads(p.read_text(encoding="utf-8"))
json.dump(out, open(HERE / "audit_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("chain_of_custody", "snapshot_vs_shared_four_blocks_identical",
                                      "repro_bit_exact", "other_seeds", "other_seeds_all_in_window",
                                      "mix_extra_seeds", "mix_extra_all_in_window", "k2_seed_ensemble")},
                 ensure_ascii=False, indent=1))
