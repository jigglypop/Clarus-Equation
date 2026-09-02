"""Q-0008 attempt-04: assemble verify/Q-0008/attempt-04/result.json from the raw artefacts.

Sources (all produced in this attempt, none edited by hand):
  F-02_result_snapshot.json  copy of verify/Q-0008/F-02/result.json taken after the four runs
  se_bootstrap.json          check_se.py output (bootstrap SE, reproduction check)
  timing.txt                 wall-clock stamps of the four `check_modes.py --mode ...` runs
  log_<mode>.txt             raw stdout of each run

Card windows / pre-registered values / uncertainties are read from the untouched script (WINDOWS,
PREREGISTERED) and the card front-matter is not modified.  qspine data present in the shared
F-02/result.json was written by a concurrent attempt-05 run and is *not* evaluated here.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
import check_modes as cm  # noqa: E402

CARD_UNCERTAINTY = {  # from derivations/Q-0008/F-02.formula.md predicts[].uncertainty (read-only copy)
    "her_slope": 0.10,
    "her_ratio_128": 6.5,
    "mix_X_32": 0.25,
    "iid_slope": 0.10,
    "defect_ratio_64_over_8": 0.017,
    "defect_slope": 0.05,
}
ROLE = {
    "her_slope": "kill K1",
    "her_ratio_128": "kill K1",
    "mix_X_32": "kill K2",
    "iid_slope": "kill K5",
    "defect_ratio_64_over_8": "consistency K4 (not kill)",
    "defect_slope": "consistency K4 (not kill)",
}
MODE_OF = {
    "her_slope": "her",
    "her_ratio_128": "her",
    "mix_X_32": "mix",
    "iid_slope": "iid",
    "defect_ratio_64_over_8": "defect",
    "defect_slope": "defect",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timing(text: str) -> dict:
    stamps = {}
    for line in text.splitlines():
        m = re.match(r"\[(\w+)\] (start|end) (\S+)", line)
        if m:
            stamps.setdefault(m.group(1), {})[m.group(2)] = m.group(3)
        m = re.match(r"(run_start|run_end) (\S+)", line)
        if m:
            stamps[m.group(1)] = m.group(2)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    out = {"run_start_utc": stamps["run_start"], "run_end_utc": stamps["run_end"], "per_mode": {}}
    for mode in ("her", "mix", "iid", "defect"):
        s = datetime.strptime(stamps[mode]["start"], fmt).replace(tzinfo=timezone.utc)
        e = datetime.strptime(stamps[mode]["end"], fmt).replace(tzinfo=timezone.utc)
        out["per_mode"][mode] = {"start_utc": stamps[mode]["start"], "end_utc": stamps[mode]["end"], "elapsed_s": (e - s).total_seconds()}
    s = datetime.strptime(stamps["run_start"], fmt)
    e = datetime.strptime(stamps["run_end"], fmt)
    out["elapsed_s_total"] = (e - s).total_seconds()
    return out


def main() -> int:
    snap = json.loads((HERE / "F-02_result_snapshot.json").read_text(encoding="utf-8"))
    se = json.loads((HERE / "se_bootstrap.json").read_text(encoding="utf-8"))
    timing = parse_timing((HERE / "timing.txt").read_text(encoding="utf-8"))
    logs = {m: (HERE / f"log_{m}.txt").read_text(encoding="utf-8").strip() for m in ("her", "mix", "iid", "defect")}

    # constants of the script vs the card (card values transcribed read-only; compared here, not edited)
    card_constants = {
        "SEED": 20260902, "DELTA": 0.005, "MIN_DET": 0.05, "SIZES": [8, 16, 32, 64, 128], "TRIALS": 256,
        "MIX_N": 32, "MIX_TRIALS": 1024, "DEFECT_GRID": [4, 8, 16, 32, 64], "DEFECT_PERTURBATION": 0.35,
        "DEFECT_MIN_DET": 0.2,
        "PREREGISTERED": {"her_slope": 0.5302, "her_ratio_128": 32.554, "mix_X_32": 0.7406, "iid_slope": -0.4783,
                          "defect_ratio_64_over_8": 0.140625, "defect_slope": -0.9069},
        "WINDOWS": {"her_slope": [0.43, 0.63], "her_ratio_128": [26.0, 39.1], "mix_X_32": [0.49, 0.99],
                    "iid_slope": [-0.58, -0.38], "defect_ratio_64_over_8": [0.124, 0.158], "defect_slope": [-0.96, -0.86]},
    }
    script_constants = {
        "SEED": cm.SEED, "DELTA": cm.DELTA, "MIN_DET": cm.MIN_DET, "SIZES": list(cm.SIZES), "TRIALS": cm.TRIALS,
        "MIX_N": cm.MIX_N, "MIX_TRIALS": cm.MIX_TRIALS, "DEFECT_GRID": list(cm.DEFECT_GRID),
        "DEFECT_PERTURBATION": cm.DEFECT_PERTURBATION, "DEFECT_MIN_DET": cm.DEFECT_MIN_DET,
        "PREREGISTERED": {k: cm.PREREGISTERED[k] for k in card_constants["PREREGISTERED"]},
        "WINDOWS": {k: list(cm.WINDOWS[k]) for k in card_constants["WINDOWS"]},
    }
    constants_match = script_constants == card_constants

    results = {}
    kills_fired = []
    inconsistent = []
    for key in ("her_slope", "her_ratio_128", "mix_X_32", "iid_slope", "defect_ratio_64_over_8", "defect_slope"):
        sig = se["sigma_to_window"][key]
        value = snap["stats"][key]
        assert abs(value - sig["value"]) < 1e-12, key
        lo, hi = cm.WINDOWS[key]
        in_window = lo <= value <= hi
        entry = {
            "mode": MODE_OF[key],
            "role": ROLE[key],
            "value": value,
            "se_bootstrap": sig["se"],
            "ci95_bootstrap": None,
            "window": [lo, hi],
            "preregistered": cm.PREREGISTERED[key],
            "card_uncertainty": CARD_UNCERTAINTY[key],
            "within_card_uncertainty": abs(value - cm.PREREGISTERED[key]) <= CARD_UNCERTAINTY[key],
            "in_window": in_window,
            "script_verdict": snap["verdict"][key],
            "sigma_to_nearest_window_edge": (sig["sigma"] or {}).get("nearest"),
            "sigma_to_low_edge": (sig["sigma"] or {}).get("to_low"),
            "sigma_to_high_edge": (sig["sigma"] or {}).get("to_high"),
            "deviation_from_preregistered_in_se": sig.get("dev_from_prereg_over_se"),
        }
        block = se[MODE_OF[key]]
        if key in block and isinstance(block[key], dict) and "ci95" in block[key]:
            entry["ci95_bootstrap"] = block[key]["ci95"]
        if key.startswith("defect"):
            entry["consistency"] = in_window
            entry["note"] = "deterministic single Delta sample; no trial variance (kill status revoked in card rev.2)"
            if not in_window:
                inconsistent.append(key)
        else:
            entry["pass"] = in_window
            if not in_window:
                kills_fired.append(key)
        results[key] = entry

    out = {
        "question": "Q-0008",
        "card": "F-02",
        "attempt": 4,
        "ladder_step": 6,
        "executed_modes": ["her", "mix", "iid", "defect"],
        "command": ".claude\\hooks\\python.cmd python verify\\Q-0008\\F-02\\check_modes.py --mode <her|mix|iid|defect>",
        "seed": cm.SEED,
        "delta": cm.DELTA,
        "constants_match_card": constants_match,
        "constants_compared": script_constants,
        "results": results,
        "kills_fired": kills_fired,
        "k4_consistency": "consistent" if not inconsistent else "inconsistent: " + ", ".join(inconsistent),
        "timing": timing,
        "se_bootstrap": {"B": se["bootstrap_B"], "seed": se["bootstrap_seed"], "elapsed_s": se["elapsed_s"],
                         "reproduction_ok": se["reproduction_ok"],
                         "reproduction_max_abs_diff": {"her": se["her"]["reproduction_max_abs_diff"],
                                                       "mix": se["mix"]["reproduction_max_abs_diff"],
                                                       "iid": se["iid"]["reproduction_max_abs_diff"]}},
        "runtime_min_total": (timing["elapsed_s_total"] + se["elapsed_s"]) / 60.0,
        "raw": {
            "her": snap["her"], "mix": snap["mix"], "iid": snap["iid"], "defect": snap["defect"],
            "her_local_slopes": se["her"]["local_slopes"],
            "iid_rms_over_exact_shape": se["iid"]["rms_over_sqrt_nm1_over_n"],
            "defect_r48": se["defect"]["r48"],
        },
        "logs": logs,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "integrity": {
            "check_modes.py_sha256": sha256(F02 / "check_modes.py"),
            "driver_numbers.py_sha256": sha256(F02 / "driver_numbers.py"),
            "card_sha256": sha256(ROOT / "derivations" / "Q-0008" / "F-02.formula.md"),
            "note": "script and card not edited in this attempt; hashes recorded for the judge",
        },
        "provenance_notes": [
            "verify/Q-0008/F-02/result.json is shared between attempts; at run time a concurrent attempt-05 process "
            "wrote a qspine block (K3, ladder step 7) into it. That block is not produced or evaluated by attempt-04.",
            "F-02_result_snapshot.json is a verbatim copy of the shared file taken after the four attempt-04 runs.",
            "Standard errors come from check_se.py, which replays the identical seeds/sampling order and bootstraps "
            "trials (B=2000); replayed RMS agree with the script output to < 1e-12.",
            "defect mode is deterministic for the pre-registered seed; the disclosed r48 = eps(8)/eps(4) is reproduced.",
        ],
    }
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"constants_match_card": constants_match, "kills_fired": kills_fired,
                      "k4_consistency": out["k4_consistency"], "runtime_min_total": out["runtime_min_total"],
                      "results": {k: {"value": v["value"], "se": v["se_bootstrap"], "in_window": v["in_window"],
                                      "sigma_nearest": v["sigma_to_nearest_window_edge"]} for k, v in results.items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
