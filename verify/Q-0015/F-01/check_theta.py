"""Q-0015 F-01 kill script: composition-face holonomy angle theta = c_theta * eps / sqrt(1 - eps^2).

Pre-registered (card derivations/Q-0015/F-01.formula.md, 2026-09-02).  Seed, sizes, delta, trial
counts and windows are frozen; do not edit after seeing results.

The card maps chapter 14.2's declared contract theta = sqrt(Delta), Delta = (1/4)||Ghat - 1_3||_F^2
(dimensionless shared-face Gram mismatch; zerod_plebanski_closure.py `_typed_history_member`) onto the
12.4 normalized simplicity residual eps = ||tl G||_F / ||G||_F of the polar-aligned block sum.  With
the isotropic (common-orbit, 13.3) rendering as the fine reference, Ghat = 3 G / tr G, hence

    theta = (3/2) ||tl G||_F / tr G  =  c_theta * eps / sqrt(1 - eps^2),
    c_theta = (1/2) tr G_0 / ||G_0||_F = sqrt(3)/2,     G_0 = gram Sigma_0(e) = 2 det(e) 1_3.

Modes
  blk    : uniform rooted Cayley (Pruefer) heritable + iid blocks, n in SIZES
           -> c_theta_ratio (K1), theta_slope_her (K2), theta_slope_iid (K7), cross_32 (P6)
  face   : 3-cell composition face f = (u,m,v) = grandparent-parent-child chain (12.1 attachment)
           heritable at depth d in {0,7} with common random numbers, plus iid
           -> rho_face (K4), face_depth_drift (K3)
  scale  : c_theta under tetrad renormalization e -> alpha e, alpha in {0.4, 1.0, 2.5}
           -> c_theta_alpha_ratio (K5)
  hinge  : chapter 15 exact constant-curvature hinge (kappa = a = 1, Psi = 0)
           -> hinge_mis_angle (K6); phi_kappa is recorded but is the ISOTROPIC channel and is NOT
              predicted by this card.

All residuals are simplicity_residual (12.4, normalized traceless Plebanski gram) of the polar-aligned
block sum; the angle uses the same gram.  delta = 0.005 for every stochastic mode (delta^2 regime).
MIN_DET: a configuration is resampled if any cell has det(I + delta*label) <= MIN_DET (declared; at
delta = 0.005 the expected rejection rate is 0).

Usage: python verify/Q-0015/F-01/check_theta.py --mode {blk,face,scale,hinge,all} [--smoke]
Writes verify/Q-0015/F-01/result.json (unless --smoke; smoke output is NOT a kill verdict).
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))

from driver_numbers import tree_arrays, uniform_rooted_tree  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    plebanski_gram,
)
from examples.physics.gravity.curved_plebanski_hinge import (  # noqa: E402
    constructive_curved_plebanski_hinge_witness,
    de_sitter_plebanski_point_audit,
    exact_primal_triangle_holonomy,
    reference_vertex_coordinates,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

# ------------------------------------------------------------------ frozen constants
SEED_HER = 20260902
SEED_IID = 20260903
DELTA = 0.005
MIN_DET = 0.05
SIZES = (8, 16, 32, 64, 128)
TRIALS = 256
FACE_TRIALS = 2048
FACE_DEPTHS = (0, 7)
FACE_INCREMENT_SLOTS = (8, 9)  # common random numbers: the same two increments for every depth
SCALE_ALPHAS = (0.4, 1.0, 2.5)
HINGE_KAPPA = 1.0
CROSS_SIZE = 32

PREREGISTERED = {
    "c_theta_ratio": 0.8660254038,
    "rho_face": 0.7453560,
    "face_depth_drift": 1.0000,
    "theta_slope_her": 0.5302,
    "theta_slope_iid": -0.4783,
    "cross_32": 3.98550,
    "c_theta_alpha_ratio": 1.0000,
    "hinge_mis_angle": 0.0,
}
WINDOWS = {
    "c_theta_ratio": (0.856, 0.876),
    "rho_face": (0.685, 0.806),
    "face_depth_drift": (0.90, 1.10),
    "theta_slope_her": (0.43, 0.63),
    "theta_slope_iid": (-0.58, -0.38),
    "cross_32": (3.5855, 4.3855),
    "c_theta_alpha_ratio": (0.995, 1.005),
    "hinge_mis_angle": (0.0, 1.0e-12),
}

REFERENCE = geometric_self_dual_triple(np.eye(4))
C_THETA_EXACT = math.sqrt(3.0) / 2.0


# ------------------------------------------------------------------ block -> (eps, theta)
def block_triple(labels: np.ndarray, delta: float = DELTA, scale: float = 1.0) -> np.ndarray:
    reference = geometric_self_dual_triple(scale * np.eye(4))
    blocked = np.zeros_like(reference)
    for lab in labels:
        tetrad = scale * (np.eye(4) + delta * lab)
        if float(np.linalg.det(np.eye(4) + delta * lab)) <= MIN_DET:
            return np.full_like(reference, np.nan)
        candidate = geometric_self_dual_triple(tetrad)
        blocked += optimal_internal_alignment(reference, candidate).aligned_candidate
    return blocked


def eps_and_theta(triple: np.ndarray) -> tuple[float, float]:
    """12.4 normalized residual and the 14.2 mismatch angle of the same gram."""

    gram = plebanski_gram(triple)
    traceless = gram - np.trace(gram) / 3.0 * np.eye(3)
    tl_norm = float(np.linalg.norm(traceless))
    eps = tl_norm / float(np.linalg.norm(gram))
    theta = 1.5 * tl_norm / float(np.trace(gram))
    return eps, theta


def heritable_labels(parent: list[int], xi: np.ndarray) -> np.ndarray:
    order, _, _, _ = tree_arrays(parent)
    labels = np.zeros_like(xi)
    for v in order:
        p = parent[v]
        labels[v] = xi[v] + (labels[p] if p >= 0 else 0.0)
    return labels


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def fit_slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


# ------------------------------------------------------------------ composition face (3-chain)
def face_statistics(trials: int) -> dict[str, float]:
    """Every composition face f = (u,m,v) is a grandparent-parent-child chain (12.1)."""

    depth_rms: dict[int, float] = {}
    max_depth = max(FACE_DEPTHS)
    slot_a, slot_b = FACE_INCREMENT_SLOTS
    draws = max(max_depth + 1, slot_b + 1)
    for depth in FACE_DEPTHS:
        rng = np.random.default_rng(SEED_HER)
        angles = []
        for _ in range(trials):
            xi = rng.standard_normal((draws, 4, 4))
            ancestor = xi[: depth + 1].sum(axis=0)
            middle = ancestor + xi[slot_a]
            child = middle + xi[slot_b]
            angles.append(eps_and_theta(block_triple(np.stack([ancestor, middle, child])))[1])
        depth_rms[depth] = rms(angles)
    rng = np.random.default_rng(SEED_IID)
    iid_angles = [
        eps_and_theta(block_triple(rng.standard_normal((3, 4, 4))))[1] for _ in range(trials)
    ]
    iid_rms = rms(iid_angles)
    deep = depth_rms[max_depth]
    return {
        "theta_face_her_d0": depth_rms[FACE_DEPTHS[0]],
        "theta_face_her_deep": deep,
        "theta_face_iid": iid_rms,
        "rho_face": deep / iid_rms,
        "face_depth_drift": deep / depth_rms[FACE_DEPTHS[0]],
    }


# ------------------------------------------------------------------ modes
def mode_blk(sizes, trials, face_trials) -> dict[str, float]:
    her_theta, her_eps, iid_theta, iid_eps = {}, {}, {}, {}
    for n in sizes:
        rng = np.random.default_rng(SEED_HER)
        pairs = []
        for _ in range(trials):
            parent = uniform_rooted_tree(n, rng)
            labels = heritable_labels(parent, rng.standard_normal((n, 4, 4)))
            pairs.append(eps_and_theta(block_triple(labels)))
        her_eps[n], her_theta[n] = rms([p[0] for p in pairs]), rms([p[1] for p in pairs])
        rng = np.random.default_rng(SEED_IID)
        pairs = [eps_and_theta(block_triple(rng.standard_normal((n, 4, 4)))) for _ in range(trials)]
        iid_eps[n], iid_theta[n] = rms([p[0] for p in pairs]), rms([p[1] for p in pairs])
    ratios = [her_theta[n] / her_eps[n] for n in sizes] + [iid_theta[n] / iid_eps[n] for n in sizes]
    face = face_statistics(face_trials)
    cross_n = CROSS_SIZE if CROSS_SIZE in sizes else max(sizes)
    return {
        "c_theta_ratio": float(np.median(ratios)),
        "c_theta_ratio_spread": float(np.max(ratios) - np.min(ratios)),
        "theta_slope_her": fit_slope(sizes, [her_theta[n] for n in sizes]),
        "theta_slope_iid": fit_slope(sizes, [iid_theta[n] for n in sizes]),
        "eps_slope_her": fit_slope(sizes, [her_eps[n] for n in sizes]),
        "eps_slope_iid": fit_slope(sizes, [iid_eps[n] for n in sizes]),
        "cross_32": her_theta[cross_n] / face["theta_face_her_d0"],
        "cross_size_used": float(cross_n),
    }


def mode_face(face_trials) -> dict[str, float]:
    return face_statistics(face_trials)


def mode_scale() -> dict[str, float]:
    constants = {}
    for alpha in SCALE_ALPHAS:
        gram = plebanski_gram(geometric_self_dual_triple(alpha * np.eye(4)))
        constants[alpha] = 0.5 * float(np.trace(gram)) / float(np.linalg.norm(gram))
    rng = np.random.default_rng(SEED_HER)
    labels = rng.standard_normal((3, 4, 4))
    block_ratio = {
        alpha: (lambda pair: pair[1] / pair[0])(eps_and_theta(block_triple(labels, scale=alpha)))
        for alpha in SCALE_ALPHAS
    }
    return {
        "c_theta_alpha_0p4": constants[0.4],
        "c_theta_alpha_1": constants[1.0],
        "c_theta_alpha_2p5": constants[2.5],
        "c_theta_alpha_ratio": constants[2.5] / constants[1.0],
        "c_theta_block_alpha_ratio": block_ratio[2.5] / block_ratio[1.0],
    }


def mode_hinge() -> dict[str, float]:
    worst = 0.0
    for coordinate in reference_vertex_coordinates().values():
        audit = de_sitter_plebanski_point_audit(
            tuple(float(value) for value in coordinate),
            curvature_times_reference_length_squared=HINGE_KAPPA,
        )
        worst = max(worst, float(audit.simplicity_tracefree_residual))
    witness = constructive_curved_plebanski_hinge_witness(
        curvature_times_reference_length_squared=HINGE_KAPPA
    )
    primal = exact_primal_triangle_holonomy(
        curvature_times_reference_length_squared=HINGE_KAPPA
    )
    return {
        "hinge_simplicity_residual": worst,
        "hinge_mis_angle": C_THETA_EXACT * worst,
        "hinge_max_field_residual": float(witness.maximum_sampled_field_residual),
        "hinge_phi_kappa_isotropic_not_predicted": float(primal.rotation_angle),
    }


# ------------------------------------------------------------------ driver
def verdicts(stats: dict[str, float]) -> dict[str, dict]:
    out = {}
    for name, (low, high) in WINDOWS.items():
        if name not in stats:
            continue
        value = stats[name]
        out[name] = {
            "value": value,
            "preregistered": PREREGISTERED[name],
            "window": [low, high],
            "inside": bool(low <= value <= high),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=("blk", "face", "scale", "hinge", "all"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    sizes = (8, 16, 32) if args.smoke else SIZES
    trials = 24 if args.smoke else TRIALS
    face_trials = 128 if args.smoke else FACE_TRIALS

    stats: dict[str, float] = {}
    if args.mode in ("blk", "all"):
        stats.update(mode_blk(sizes, trials, face_trials))
    if args.mode in ("face", "all"):
        stats.update(mode_face(face_trials))
    if args.mode in ("scale", "all"):
        stats.update(mode_scale())
    if args.mode in ("hinge", "all"):
        stats.update(mode_hinge())

    payload = {
        "card": "derivations/Q-0015/F-01.formula.md",
        "mode": args.mode,
        "smoke": bool(args.smoke),
        "seed_her": SEED_HER,
        "seed_iid": SEED_IID,
        "delta": DELTA,
        "sizes": list(sizes),
        "trials": trials,
        "face_trials": face_trials,
        "c_theta_exact": C_THETA_EXACT,
        "stats": stats,
        "verdicts": verdicts(stats),
    }
    if args.smoke:
        payload["note"] = (
            "SMOKE RUN - reduced sizes/trials; verdicts are indicative only and are NOT the "
            "pre-registered kill decision.  The kill run is the full-size run without --smoke."
        )
    else:
        (HERE / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
