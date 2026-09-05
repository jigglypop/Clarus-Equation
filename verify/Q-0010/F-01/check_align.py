"""Q-0010 F-01 kill script: the alignment axiom turns the heritable block mode irrelevant.

Pre-registered (card derivations/Q-0010/F-01.formula.md, 2026-09-02).  Seeds, sizes, delta, trial
counts, windows and predicted values are frozen; do not edit after seeing results.

Alignment rule under test (axiom candidate):  each cell v draws a fresh tetrad increment xi_v
(16 orthonormal label directions).  Only the orbit-tangent part P_align xi_v is transmitted to the
children (it accumulates inside the common conformal orbit O(B_0) = {alpha R . B_0}); the folded part
(1 - P_align) xi_v renders locally at v only.  So

    label_v = sum_{u <= v} P_align xi_u  +  (1 - P_align) xi_v .

Modes
  align  : uniform rooted Cayley trees (Pruefer), alignment rule, n in {8,...,128}
           -> align_slope, align_ratio_32, align_ratio_128, align_ratio_spread
  rand4  : same, but P_align replaced by a random 4-dimensional projector (QR of a 16x4 gaussian,
           seed 20260902) -> rand4_slope, rand4_ratio_128       [discriminant: orbit tangent special?]
  qspine : depth-b Q-spine block (11.6) with the alignment rule, RMS_align_Q(8)/RMS_iid(36)
  struct : exact linear algebra (driver_numbers.py) + the 13.3 orbit recovery run
  closure: K4 (pre-registered 2026-09-02, card revision 2).  fold = 0 (every increment projected onto
           the transmitted channel, (1 - P) xi_v == 0), delta = 0.2, n in {8, 16, 32}, 64 trials/size,
           seed 20260902 with common random numbers across projectors.  Statistic per projector =
           max_n RMS(12.4 normalized residual).  Projectors: align = scale+sd (T_{B0}O, left quaternions
           -> exact orbit closure, <= 1e-12), alt1 = scale+asd (right quaternions, also closes; recorded
           only), alt2 = sd+asd_1 and null7 = scale+sd+asd (inside ker M~ but not subalgebras -> >= 1e-3).
           This is the only statistic that singles out the orbit tangent among the 4-planes of ker M~.

Every stochastic mode uses delta = 0.005 and the frozen F-02 residual pipeline
(verify/Q-0008/F-02/check_modes.block_residual: per-cell polar SO(3) alignment to the reference,
MIN_DET = 0.05, 12.4 normalized traceless Plebanski residual, RMS over trials).  The i.i.d. control
uses seed 20260903 (the F-02 convention).

Usage: python verify/Q-0010/F-01/check_align.py --mode {align,rand4,qspine,struct,closure,all} [--smoke]
Writes verify/Q-0010/F-01/result.json (unless --smoke).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))

from check_modes import block_residual, fit_slope, rms  # noqa: E402  (frozen F-02 pipeline)
from driver_numbers import qspine_block, tree_arrays, uniform_rooted_tree  # noqa: E402  (F-02 trees)
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple  # noqa: E402
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

import importlib.util  # noqa: E402

# both cards ship a driver_numbers.py; load this card's driver under its own module name
_spec = importlib.util.spec_from_file_location("q0010_driver", HERE / "driver_numbers.py")
_q0010 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_q0010)
orthonormal_label_basis = _q0010.orthonormal_label_basis

SEED = 20260902
SEED_IID = 20260903
DELTA = 0.005
SIZES = (8, 16, 32, 64, 128)
TRIALS = 256
QSPINE_DEPTH = 8
QSPINE_IID_N = 36
QSPINE_TRIALS = 512
RANK_TOL = 1.0e-8
CLOSURE_SIZES = (8, 16, 32)       # K4 (revision 2): frozen
CLOSURE_TRIALS = 64
CLOSURE_DELTA = 0.2
CLOSURE_UPPER = 1.0e9             # "no upper bound" for the contrast windows (residual is normalized, O(1))

PREREGISTERED = {
    "align_slope": -0.4783,
    "align_ratio_32": 1.000,
    "align_ratio_128": 1.000,
    "align_ratio_spread": 1.000,
    "rand4_slope": 0.2055,
    "rand4_ratio_128": 7.299,
    "qspine_align_ratio": 1.000,
    "dim_folded_visible": 9.0,
    "align_c1_over_c0": 0.0,
    "floor_trace_folded": 0.0,
    "orbit_max_residual": 0.0,
    # K4 (revision 2): exact closure for the quaternion channels; contrasts are lower bounds (1e-3).  The
    # adversary's out-of-configuration reference values (4.1e-3 alt2, 1.5e-2 null7) are NOT the prediction.
    "closure_align": 0.0,
    "closure_alt1": 0.0,
    "closure_alt2": 1.0e-3,
    "closure_null7": 1.0e-3,
}
WINDOWS = {
    "align_slope": (-0.58, -0.38),
    "align_ratio_32": (0.85, 1.18),
    "align_ratio_128": (0.85, 1.18),
    "align_ratio_spread": (1.0, 1.25),
    "rand4_slope": (0.10, 0.31),
    "rand4_ratio_128": (5.8, 8.8),
    "qspine_align_ratio": (0.85, 1.18),
    "dim_folded_visible": (9.0, 9.0),
    "align_c1_over_c0": (-1.0e-12, 1.0e-12),
    "floor_trace_folded": (-1.0e-8, 1.0e-8),
    "orbit_max_residual": (-1.0e-10, 1.0e-10),
    "closure_align": (0.0, 1.0e-12),
    "closure_alt1": (0.0, 1.0e-12),
    "closure_alt2": (1.0e-3, CLOSURE_UPPER),
    "closure_null7": (1.0e-3, CLOSURE_UPPER),
}
KILLS = {
    "K1": ("align_slope", "align_ratio_32", "align_ratio_128", "align_ratio_spread"),
    "K2": ("rand4_slope", "rand4_ratio_128"),
    "K3": ("qspine_align_ratio",),
    "K4": ("closure_align", "closure_alt2", "closure_null7"),
}
CONSISTENCY = ("dim_folded_visible", "align_c1_over_c0", "floor_trace_folded", "orbit_max_residual")
RECORDED_ONLY = ("closure_alt1",)   # right-quaternion channel: closes too; sd/asd is settled by the rendering definition

BASIS, GROUPS = orthonormal_label_basis()
FLAT_BASIS = BASIS.reshape(16, 16)


def align_projector() -> np.ndarray:
    """Diagonal 16x16 projector on span{scale, self-dual so(3)} = the 4-dim orbit tangent directions."""
    projector = np.zeros((16, 16))
    for index in GROUPS["scale"] + GROUPS["sd"]:
        projector[index, index] = 1.0
    return projector


def diagonal_projector(indices) -> np.ndarray:
    projector = np.zeros((16, 16))
    for index in indices:
        projector[index, index] = 1.0
    return projector


def closure_projectors() -> dict[str, np.ndarray]:
    """K4 channels.  All four lie inside ker M~ (identical K1/K2/K3 budgets); only the two quaternion
    subalgebras close the orbit exactly."""
    return {
        "align": diagonal_projector(GROUPS["scale"] + GROUPS["sd"]),                    # T_{B0}O = span{1, so(3)_sd} ~ H (left)
        "alt1": diagonal_projector(GROUPS["scale"] + GROUPS["asd"]),                    # span{1, so(3)_asd} ~ H (right): recorded only
        "alt2": diagonal_projector(GROUPS["sd"] + GROUPS["asd"][:1]),                   # inside ker M~, not a subalgebra
        "null7": diagonal_projector(GROUPS["scale"] + GROUPS["sd"] + GROUPS["asd"]),    # whole ker M~, not a subalgebra
    }


def random_projector(seed: int = SEED, dimension: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(16, dimension)))
    return basis @ basis.T


def labels_from_coefficients(coeff: np.ndarray) -> np.ndarray:
    """(n,16) coefficients in the orthonormal label basis -> (n,4,4) tetrad increments."""
    return (coeff @ FLAT_BASIS).reshape(-1, 4, 4)


def alignment_rule_coefficients(parent, coeff: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """label_v = sum_{u <= v} P xi_u + (1 - P) xi_v (transmitted part accumulates, folded part is local)."""
    order, *_ = tree_arrays(parent)
    transmitted = coeff @ projector.T
    folded = coeff - transmitted
    accumulated = np.zeros_like(coeff)
    for v in order:
        p = parent[v]
        accumulated[v] = transmitted[v] + (accumulated[p] if p >= 0 else 0.0)
    return accumulated + folded


def sample_align(n: int, rng, projector: np.ndarray, delta: float, tree=uniform_rooted_tree) -> float:
    while True:
        parent = tree(n, rng)
        coeff = rng.normal(size=(len(parent), 16))
        value = block_residual(labels_from_coefficients(alignment_rule_coefficients(parent, coeff, projector)), delta)
        if math.isfinite(value):
            return value


def sample_iid(n: int, rng, delta: float) -> float:
    while True:
        value = block_residual(labels_from_coefficients(rng.normal(size=(n, 16))), delta)
        if math.isfinite(value):
            return value


def run_grid(projector: np.ndarray, sizes, trials: int, delta: float, seed: int, seed_iid: int) -> dict:
    rng_a = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed_iid)
    out = {"sizes": list(sizes), "rms_align": [], "rms_iid": [], "trials": trials, "delta": delta,
           "seed": seed, "seed_iid": seed_iid}
    for n in sizes:
        out["rms_align"].append(rms([sample_align(n, rng_a, projector, delta) for _ in range(trials)]))
        out["rms_iid"].append(rms([sample_iid(n, rng_i, delta) for _ in range(trials)]))
    out["slope"] = fit_slope(sizes, out["rms_align"])
    out["ratio"] = [a / i for a, i in zip(out["rms_align"], out["rms_iid"])]
    out["ratio_spread"] = max(out["ratio"]) / min(out["ratio"])
    for n in (32, 128):
        out[f"ratio_{n}"] = out["ratio"][list(sizes).index(n)] if n in sizes else None
    return out


def run_qspine(depth: int, iid_n: int, trials: int, delta: float, seed: int, seed_iid: int) -> dict:
    projector = align_projector()
    rng_a = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed_iid)
    values, sizes = [], []
    while len(values) < trials:
        parent = qspine_block(depth, rng_a)
        coeff = rng_a.normal(size=(len(parent), 16))
        value = block_residual(labels_from_coefficients(alignment_rule_coefficients(parent, coeff, projector)), delta)
        if math.isfinite(value):
            values.append(value)
            sizes.append(len(parent))
    rms_iid = rms([sample_iid(iid_n, rng_i, delta) for _ in range(trials)])
    return {"depth": depth, "trials": trials, "delta": delta, "seed": seed, "mean_n": float(np.mean(sizes)),
            "rms_align_q": rms(values), "rms_iid_36": rms_iid, "ratio": rms(values) / rms_iid}


def sample_closure(n: int, rng, projector: np.ndarray, delta: float) -> float:
    """fold = 0: the increment is projected onto the transmitted channel before the rule, so
    (1 - P) xi_v == 0 and the block is the transmitted channel alone."""
    while True:
        parent = uniform_rooted_tree(n, rng)
        coeff = rng.normal(size=(len(parent), 16)) @ projector.T
        value = block_residual(labels_from_coefficients(alignment_rule_coefficients(parent, coeff, projector)), delta)
        if math.isfinite(value):
            return value


def run_closure(sizes, trials: int, delta: float, seed: int) -> dict:
    out = {"sizes": list(sizes), "trials": trials, "delta": delta, "seed": seed, "fold": 0.0,
           "rms": {}, "max_rms": {}}
    for name, projector in closure_projectors().items():
        rng = np.random.default_rng(seed)          # common random numbers across projectors
        out["rms"][name] = [rms([sample_closure(n, rng, projector, delta) for _ in range(trials)]) for n in sizes]
        out["max_rms"][name] = max(out["rms"][name])
    out["orbit_exact_max_residual"] = run_struct(orbit_sizes=tuple(sizes))["orbit_max_residual"]   # 13.3 reference
    return out


def run_struct(orbit_sizes=(2, 4, 8, 16, 32, 64)) -> dict:
    """Exact linear algebra (driver) + 13.3 recovery: cells inside one orbit sum exactly simple."""
    numbers = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    reference = geometric_self_dual_triple(np.eye(4))
    worst = 0.0
    for n in orbit_sizes:
        blocked = np.zeros_like(reference)
        for _ in range(n):
            scale = float(np.exp(0.05 * rng.normal()))
            axis = rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            angle = 0.3 * rng.normal()
            cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
            candidate = scale * (rotation @ reference)
            blocked += optimal_internal_alignment(reference, candidate).aligned_candidate
        from examples.physics.gravity.causal_face_simplicity import simplicity_residual

        worst = max(worst, float(simplicity_residual(blocked)))
    return {
        "dim_folded_visible": float(numbers["dim_folded_visible"]),
        "rank_dphi": numbers["rank_dphi"],
        "dim_orbit_tangent": numbers["dim_orbit_tangent"],
        "align_c1_over_c0": float(numbers["align_budget"]["c1_over_c0"]),
        "align_c2_over_c0": float(numbers["align_budget"]["c2_over_c0"]),
        "align_c3_over_c0": float(numbers["align_budget"]["c3_over_c0"]),
        "rand4_c1_over_c0": float(numbers["rand4_budget"]["c1_over_c0"]),
        "budget_gap": float(numbers["align_budget"]["budget_gap"]),
        "floor_trace_folded": float(numbers["floor_trace_folded"]),
        "leak_relative": float(numbers["Mtilde_P_align_leak_relative"]),
        "entropy_gap_nats": float(numbers["entropy_gap_nats_at_eps_res_1e-2"]),
        "orbit_max_residual": worst,
        "orbit_sizes": list(orbit_sizes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=("align", "rand4", "qspine", "struct", "closure", "all"))
    parser.add_argument("--smoke", action="store_true", help="execution check only (tiny sizes, no verdict)")
    args = parser.parse_args()

    if args.smoke:
        a = run_grid(align_projector(), (4, 6), 2, DELTA, SEED, SEED_IID)
        b = run_grid(random_projector(), (4, 6), 2, DELTA, SEED, SEED_IID)
        c = run_qspine(3, 6, 2, DELTA, SEED, SEED_IID)
        d = run_struct(orbit_sizes=(2, 4))
        e = run_closure((4, 6), 2, CLOSURE_DELTA, SEED)
        assert all(math.isfinite(x) for x in a["rms_align"] + b["rms_align"] + [c["ratio"], d["orbit_max_residual"]]
                   + list(e["max_rms"].values()))
        print(json.dumps({"smoke": "ok", "align": a["rms_align"], "align_ratio": a["ratio"],
                          "rand4": b["rms_align"], "qspine_ratio": c["ratio"],
                          "orbit_max_residual": d["orbit_max_residual"],
                          "dim_folded_visible": d["dim_folded_visible"],
                          "closure_max_rms_smoke": e["max_rms"],
                          "closure_orbit_exact_smoke": e["orbit_exact_max_residual"]}, ensure_ascii=False))
        return 0

    result = {"card": "F-01", "question": "Q-0010", "seed": SEED, "seed_iid": SEED_IID, "delta": DELTA,
              "preregistered": PREREGISTERED, "windows": WINDOWS, "kills": {k: list(v) for k, v in KILLS.items()}}
    modes = ("align", "rand4", "qspine", "struct", "closure") if args.mode == "all" else (args.mode,)
    stats: dict[str, float] = {}
    for mode in modes:
        if mode == "align":
            block = run_grid(align_projector(), SIZES, TRIALS, DELTA, SEED, SEED_IID)
            stats["align_slope"] = block["slope"]
            stats["align_ratio_32"] = block["ratio_32"]
            stats["align_ratio_128"] = block["ratio_128"]
            stats["align_ratio_spread"] = block["ratio_spread"]
        elif mode == "rand4":
            block = run_grid(random_projector(), SIZES, TRIALS, DELTA, SEED, SEED_IID)
            stats["rand4_slope"] = block["slope"]
            stats["rand4_ratio_128"] = block["ratio_128"]
        elif mode == "qspine":
            block = run_qspine(QSPINE_DEPTH, QSPINE_IID_N, QSPINE_TRIALS, DELTA, SEED, SEED_IID)
            stats["qspine_align_ratio"] = block["ratio"]
        elif mode == "closure":
            block = run_closure(CLOSURE_SIZES, CLOSURE_TRIALS, CLOSURE_DELTA, SEED)
            for name in ("align", "alt1", "alt2", "null7"):
                stats[f"closure_{name}"] = block["max_rms"][name]
        else:
            block = run_struct()
            stats["dim_folded_visible"] = block["dim_folded_visible"]
            stats["align_c1_over_c0"] = block["align_c1_over_c0"]
            stats["floor_trace_folded"] = block["floor_trace_folded"]
            stats["orbit_max_residual"] = block["orbit_max_residual"]
        result[mode] = block

    verdict = {}
    for key, value in stats.items():
        low, high = WINDOWS[key]
        inside = low <= value <= high
        label = "survive" if inside else "KILL"
        if key in CONSISTENCY:
            label = "consistent" if inside else "inconsistent"
        elif key in RECORDED_ONLY:
            label = "as_predicted" if inside else "off_prediction"
        verdict[key] = label
    result["stats"] = stats
    result["verdict"] = verdict
    out = HERE / "result.json"
    existing = {}
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    for key, value in result.items():
        if key in ("stats", "verdict"):
            existing.setdefault(key, {}).update(value)
        else:
            existing[key] = value
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "stats": stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
