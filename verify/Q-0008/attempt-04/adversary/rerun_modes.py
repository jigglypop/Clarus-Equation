"""adversary (attempt-04 audit): independent re-execution of check_modes run_* at declared seeds.

Does NOT touch verify/Q-0008/F-02/result.json (calls run_her/run_mix/run_iid directly, never main()).
Wraps block_residual to count MIN_DET rejections (the script's own `rejections` field is a dead
constant that is never incremented) and to record the smallest det(I + delta*label) encountered.
Seeds for the robustness runs were fixed before any result was seen: 100003, 200003 (full),
300007, 400009, 500011 (mix only).

Usage: python rerun_modes.py <seed> <modes comma sep> <out.json>
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))

import check_modes as cm  # noqa: E402

_orig_block = cm.block_residual
CNT = {"calls": 0, "nan_rejections": 0, "min_det": float("inf"), "cells": 0}


def counting_block(labels, delta):
    dets = np.linalg.det(np.eye(4) + delta * np.asarray(labels))
    CNT["min_det"] = min(CNT["min_det"], float(np.min(dets)))
    CNT["cells"] += int(np.size(dets))
    v = _orig_block(labels, delta)
    CNT["calls"] += 1
    if not math.isfinite(v):
        CNT["nan_rejections"] += 1
    return v


cm.block_residual = counting_block


def main() -> int:
    seed = int(sys.argv[1])
    modes = sys.argv[2].split(",")
    out_path = HERE / sys.argv[3]
    out = {"seed": seed, "modes": modes, "constants": {"SIZES": list(cm.SIZES), "TRIALS": cm.TRIALS,
           "MIX_N": cm.MIX_N, "MIX_TRIALS": cm.MIX_TRIALS, "DELTA": cm.DELTA, "MIN_DET": cm.MIN_DET}}
    t0 = time.perf_counter()
    for mode in modes:
        m0 = time.perf_counter()
        if mode == "her":
            b = cm.run_her(cm.SIZES, cm.TRIALS, cm.DELTA, seed)
            out["her"] = b
            out["her_slope"] = b["slope"]
            out["her_ratio_128"] = b["ratio_128"]
        elif mode == "mix":
            b = cm.run_mix(cm.MIX_N, cm.MIX_TRIALS, cm.DELTA, seed)
            out["mix"] = b
            out["mix_X_32"] = b["X"]
        elif mode == "iid":
            b = cm.run_iid(cm.SIZES, cm.TRIALS, cm.DELTA, seed)
            out["iid"] = b
            out["iid_slope"] = b["slope"]
        else:
            raise SystemExit("bad mode " + mode)
        out.setdefault("elapsed_s", {})[mode] = time.perf_counter() - m0
    out["min_det_audit"] = dict(CNT)
    out["elapsed_s_total"] = time.perf_counter() - t0
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out.get(k) for k in ("seed", "her_slope", "her_ratio_128", "mix_X_32", "iid_slope",
                                              "min_det_audit", "elapsed_s_total")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
