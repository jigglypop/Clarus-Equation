"""Q-0015 F-02 kill script: polar-transport loop holonomy vs the F-02 block residual.

PRE-REGISTERED 2026-09-03 (card derivations/Q-0015/F-02.formula.md).  Seed, sizes, delta, trial
counts, predictions and windows are frozen; do not edit after seeing results.  The pilot
(holonomy_pilot.py, seed 20260902, n <= 8) produced the predictions; this script uses a different
seed and different sizes and has NOT been run at the time of pre-registration.

Definitions are those of holonomy_pilot.py (imported verbatim):
  transport R(u->v) = polar rotation of E_v E_u^{-1}, E = I + delta*xi;  loop v_0..v_{n-1} closed by
  the coarse edge v_{n-1} -> v_0;  theta = ||log R_f||_F / sqrt(2);  eps = 12.4 residual of the
  polar-aligned UNSIGNED block sum of the same n cells (the Gram never enters theta).

Modes
  chain : heritable (root increment + cumulative increments) and iid labels on a chain loop of
          n in SIZES cells, TRIALS per size -> K1 c_theta(n) her, K2 exponents her/iid
  face  : 3-cell composition face (u, m, v), heritable depth 0 and iid, FACE_TRIALS
          -> K3 rho_face_hol, K4 c_theta face her/iid, K5 delta scaling (common random numbers)
  all   : both.  --selftest runs only the deterministic gates (no seed, no kill statistic).

Usage: python verify/Q-0015/F-02/check_holonomy.py --mode {chain,face,all} | --selftest
Writes verify/Q-0015/F-02/result.json (not in --selftest).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from holonomy_pilot import (  # noqa: E402
    DELTA,
    analytic_angle,
    block_residual,
    chain_her,
    chain_iid,
    fit_slope,
    holonomy_angle,
    loop_holonomy,
    plane_rotation,
    random_so4,
    rms,
    rms_se,
    so4_angle,
    transport,
)

SEED_KILL = 20260903
SIZES = (16, 32, 64)
TRIALS = 512
FACE_TRIALS = 2048
DELTA_SMALL = 0.0025
DELTA_TRIALS = 256

PREREGISTERED = {
    "c_theta_her_16": 4.0560507,
    "c_theta_her_32": 4.2832679,
    "c_theta_her_64": 4.3930506,
    "theta_slope_her": 1.0543077,
    "theta_slope_iid": 0.5000000,
    "rho_face_hol": 0.5773503,
    "c_theta_face_her": 1.9091883,
    "c_theta_face_iid": 2.4647515,
    "delta_ratio_face_her": 0.2500000,
}
WINDOWS = {
    "c_theta_her_16": (3.73, 4.38),
    "c_theta_her_32": (3.94, 4.63),
    "c_theta_her_64": (4.04, 4.74),
    "theta_slope_her": (0.95, 1.15),
    "theta_slope_iid": (0.40, 0.60),
    "rho_face_hol": (0.540, 0.615),
    "c_theta_face_her": (1.76, 2.06),
    "c_theta_face_iid": (2.27, 2.66),
    "delta_ratio_face_her": (0.235, 0.265),
}
ALTERNATIVES = {
    "c_theta_her_*": "F-01 pairing sqrt(3)/2 = 0.866 (any n); iid-kernel values 11.09/21.81/43.27",
    "theta_slope_her": "residual-like 0.997 is INSIDE the window (the claim is equality); sqrt-law 0.53",
    "theta_slope_iid": "residual-like -0.482 (holonomy = residual); perimeter-free 0.0",
    "rho_face_hol": "residual unsigned sqrt(5)/3 = 0.745; residual signed (+,+,-) 3.543; F-01 1.0",
    "c_theta_face_her": "F-01 0.866; signed-boundary pairing 0.273 (pilot); iid-universal 2.465",
    "delta_ratio_face_her": "O(delta) holonomy 0.5",
}


def mode_chain() -> dict:
    stats: dict = {}
    per = {}
    for mode, sampler in (("her", chain_her), ("iid", chain_iid)):
        rng = np.random.default_rng(SEED_KILL)
        rows = {}
        for n in SIZES:
            th, eps = [], []
            for _ in range(TRIALS):
                labels = sampler(n, rng)
                th.append(holonomy_angle(labels, DELTA))
                eps.append(block_residual(labels, DELTA))
            rows[str(n)] = {
                "theta_rms": rms(th),
                "theta_rms_se": rms_se(th),
                "eps_rms": rms(eps),
                "eps_rms_se": rms_se(eps),
                "c_theta": rms(th) / rms(eps),
            }
        per[mode] = rows
        stats[f"theta_slope_{mode}"] = fit_slope(SIZES, [rows[str(n)]["theta_rms"] for n in SIZES])
        stats[f"eps_slope_{mode}"] = fit_slope(SIZES, [rows[str(n)]["eps_rms"] for n in SIZES])
    for n in SIZES:
        stats[f"c_theta_her_{n}"] = per["her"][str(n)]["c_theta"]
        stats[f"c_theta_iid_{n}"] = per["iid"][str(n)]["c_theta"]
    stats["per_size"] = per
    return stats


def mode_face() -> dict:
    rng = np.random.default_rng(SEED_KILL)
    th_h, eps_h = [], []
    for _ in range(FACE_TRIALS):
        labels = chain_her(3, rng)
        th_h.append(holonomy_angle(labels, DELTA))
        eps_h.append(block_residual(labels, DELTA))
    rng = np.random.default_rng(SEED_KILL + 1)
    th_i, eps_i = [], []
    for _ in range(FACE_TRIALS):
        labels = chain_iid(3, rng)
        th_i.append(holonomy_angle(labels, DELTA))
        eps_i.append(block_residual(labels, DELTA))
    rng = np.random.default_rng(SEED_KILL + 2)
    big, small = [], []
    for _ in range(DELTA_TRIALS):
        labels = chain_her(3, rng)
        big.append(holonomy_angle(labels, DELTA))
        small.append(holonomy_angle(labels, DELTA_SMALL))
    return {
        "theta_face_her_rms": rms(th_h),
        "theta_face_her_rms_se": rms_se(th_h),
        "theta_face_iid_rms": rms(th_i),
        "theta_face_iid_rms_se": rms_se(th_i),
        "eps_face_her_rms": rms(eps_h),
        "eps_face_iid_rms": rms(eps_i),
        "rho_face_hol": rms(th_h) / rms(th_i),
        "rho_face_eps": rms(eps_h) / rms(eps_i),
        "c_theta_face_her": rms(th_h) / rms(eps_h),
        "c_theta_face_iid": rms(th_i) / rms(eps_i),
        "delta_ratio_face_her": rms(small) / rms(big),
    }


def selftest() -> dict:
    rng = np.random.default_rng(1)
    frames = [rng.uniform(0.5, 2.0) * random_so4(rng) for _ in range(5)]
    e_u = np.eye(4) + 0.2 * rng.standard_normal((4, 4))
    face = chain_her(3, rng)
    return {
        "pure_gauge_angle": so4_angle(loop_holonomy(frames)),
        "single_transport_angle_minus_0p3": so4_angle(transport(e_u, plane_rotation(0.3) @ e_u)) - 0.3,
        "two_loop_angle": holonomy_angle(rng.standard_normal((2, 4, 4)), DELTA),
        "face_numeric_over_analytic_at_delta_1e-3": holonomy_angle(face, 1e-3) / analytic_angle(face, 1e-3),
    }


def verdicts(stats: dict) -> dict:
    out = {}
    for name, (low, high) in WINDOWS.items():
        if name in stats:
            value = float(stats[name])
            out[name] = {
                "value": value,
                "preregistered": PREREGISTERED[name],
                "window": [low, high],
                "inside": bool(low <= value <= high),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=("chain", "face", "all"))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps({"selftest": selftest()}, indent=2))
        return 0
    t0 = time.time()
    stats: dict = {}
    if args.mode in ("chain", "all"):
        stats.update(mode_chain())
    if args.mode in ("face", "all"):
        stats.update(mode_face())
    payload = {
        "card": "derivations/Q-0015/F-02.formula.md",
        "mode": args.mode,
        "seed": SEED_KILL,
        "delta": DELTA,
        "sizes": list(SIZES),
        "trials": TRIALS,
        "face_trials": FACE_TRIALS,
        "delta_small": DELTA_SMALL,
        "delta_trials": DELTA_TRIALS,
        "stats": stats,
        "verdicts": verdicts(stats),
        "alternatives": ALTERNATIVES,
        "wall_seconds": time.time() - t0,
    }
    (HERE / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
