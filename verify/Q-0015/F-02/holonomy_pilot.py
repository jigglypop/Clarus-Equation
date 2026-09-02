"""Q-0015 F-02 pilot: polar-transport face holonomy, computed independently of the Gram residual.

Purpose (card derivations/Q-0015/F-02.formula.md, 2026-09-03): produce the PRE-REGISTERED numbers
of the card and validate the leading-order formula on SMALL configurations only.  This is the
formula's side of the ledger, not the kill.  The kill is check_holonomy.py (sizes 16/32/64,
seed 20260903) and is NOT run here.

Definitions (fixed once, used verbatim by the kill script)
  tetrad          E_v = I + delta * xi_v                     (rows = tetrad vectors, F-02 convention)
  transport       R(u->v) = polar rotation of E_v E_u^{-1}   (SO(4) frame transition, common chart)
  loop            v_0 -> v_1 -> ... -> v_{k-1} -> v_0 ; the closing edge v_{k-1} -> v_0 is the
                  coarse continuation of 12.1 (oriented boundary e_um + e_mv - e_uv for k = 3)
  holonomy        R_f = R(v_{k-1}->v_0) ... R(v_1->v_2) R(v_0->v_1)
  angle           theta_f = ||log R_f||_F / sqrt(2) = sqrt(alpha^2 + beta^2)  (SO(4) angle norm,
                  alpha, beta = the two plane angles; a single-plane rotation by alpha gives alpha)
  residual        eps = 12.4 normalized traceless Plebanski Gram residual of the polar-aligned
                  UNSIGNED block sum of the same cells (F-02 machinery); the Gram never enters theta.

Leading order (declared on the card, verified numerically here):
  log R_f = (delta^2/2) * sum_i [sigma_i, sigma_{i+1}] + O(delta^3),  sigma_v = sym(xi_v),
  E theta_f^2 = (9/2) delta^4 Theta(kappa),
  Theta(kappa) = sum_{i,j} (kappa_ij kappa_{i+1,j+1} - kappa_{i,j+1} kappa_{i+1,j})  (cyclic indices).

Seed 20260902, delta = 0.005, n <= 8, <= 2000 trials per configuration.
Writes verify/Q-0015/F-02/pilot.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from examples.physics.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    simplicity_residual,
)
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
DELTA = 0.005
PILOT_SIZES = (3, 4, 6, 8)
PILOT_TRIALS = 1000
FACE_TRIALS = 2000
FACE_DEPTHS = (0, 7)
REFERENCE = geometric_self_dual_triple(np.eye(4))


# ------------------------------------------------------------------ holonomy (tetrad side, SO(4))
def polar_rotation(matrix: np.ndarray) -> np.ndarray:
    """Proper rotation factor of the left polar decomposition T = R P (P symmetric positive)."""
    left, _, right_t = np.linalg.svd(matrix)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def transport(tetrad_from: np.ndarray, tetrad_to: np.ndarray) -> np.ndarray:
    return polar_rotation(tetrad_to @ np.linalg.inv(tetrad_from))


def loop_holonomy(tetrads: list[np.ndarray]) -> np.ndarray:
    k = len(tetrads)
    holonomy = np.eye(4)
    for i in range(k):
        holonomy = transport(tetrads[i], tetrads[(i + 1) % k]) @ holonomy
    return holonomy


def so4_angle(rotation: np.ndarray) -> float:
    """||log R||_F / sqrt(2) for R in SO(4): eigenvalues e^{+-i alpha}, e^{+-i beta}."""
    angles = np.angle(np.linalg.eigvals(rotation))
    return math.sqrt(0.5 * float(np.sum(angles * angles)))


_EPS4 = np.zeros((4, 4, 4, 4))
for _p in ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2), (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
           (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1), (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0)):
    _EPS4[_p] = 1.0
for _p in ((0, 1, 3, 2), (0, 2, 1, 3), (0, 3, 2, 1), (1, 0, 2, 3), (1, 2, 3, 0), (1, 3, 0, 2),
           (2, 0, 3, 1), (2, 1, 0, 3), (2, 3, 1, 0), (3, 0, 1, 2), (3, 1, 2, 0), (3, 2, 0, 1)):
    _EPS4[_p] = -1.0


def so4_log(rotation: np.ndarray) -> np.ndarray:
    """Real antisymmetric log of R in SO(4) (principal branch) via the complex eigen-decomposition."""
    values, vectors = np.linalg.eig(rotation)
    log = vectors @ np.diag(1j * np.angle(values)) @ np.linalg.inv(vectors)
    log = np.real(log)
    return 0.5 * (log - log.T)


def self_dual_split(rotation: np.ndarray) -> tuple[float, float]:
    """(||omega_+||_F, ||omega_-||_F), omega_+- = (omega +- *omega)/2, (*omega)_ab = (1/2) eps_abcd omega_cd."""
    omega = so4_log(rotation)
    dual = 0.5 * np.einsum("abcd,cd->ab", _EPS4, omega)
    return float(np.linalg.norm(0.5 * (omega + dual))), float(np.linalg.norm(0.5 * (omega - dual)))


def holonomy_angle(labels: np.ndarray, delta: float) -> float:
    tetrads = [np.eye(4) + delta * lab for lab in labels]
    return so4_angle(loop_holonomy(tetrads))


# ------------------------------------------------------------------ leading-order formula
def sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def omega2(labels: np.ndarray) -> np.ndarray:
    """(1/2) sum_i [sigma_i, sigma_{i+1}] (cyclic) -- the O(delta^2) generator without delta^2."""
    k = len(labels)
    out = np.zeros((4, 4))
    for i in range(k):
        s0 = sym(labels[i])
        s1 = sym(labels[(i + 1) % k])
        out += 0.5 * (s0 @ s1 - s1 @ s0)
    return out


def analytic_angle(labels: np.ndarray, delta: float) -> float:
    return delta**2 * float(np.linalg.norm(omega2(labels))) / math.sqrt(2.0)


def loop_kernel(kappa: np.ndarray) -> float:
    """Theta(kappa) = sum_{i,j} (k_ij k_{i+1,j+1} - k_{i,j+1} k_{i+1,j}), cyclic."""
    n = len(kappa)
    nxt = (np.arange(n) + 1) % n
    k = np.asarray(kappa, dtype=float)
    return float(np.sum(k * k[np.ix_(nxt, nxt)] - k[:, nxt] * k[nxt, :]))


def centered_kernel(kappa: np.ndarray) -> float:
    """F-02 driver D = ||H kappa H||_F^2."""
    n = len(kappa)
    h = np.eye(n) - np.ones((n, n)) / n
    k = h @ np.asarray(kappa, dtype=float) @ h
    return float(np.sum(k * k))


def theta_rms_pred(kappa: np.ndarray, delta: float) -> float:
    return delta**2 * math.sqrt(4.5 * loop_kernel(kappa))


def eps_rms_pred(kappa: np.ndarray, delta: float) -> float:
    """F-02 kernel law (E-20260902-018): eps^2 = 10 delta^4 ||H kappa H||_F^2 / n^2."""
    n = len(kappa)
    return delta**2 * math.sqrt(10.0 * centered_kernel(kappa)) / n


# ------------------------------------------------------------------ residual (Gram side, SO(3))
def block_residual(labels: np.ndarray, delta: float, signs=None) -> float:
    blocked = np.zeros_like(REFERENCE)
    if signs is None:
        signs = [1.0] * len(labels)
    for lab, sign in zip(labels, signs):
        candidate = geometric_self_dual_triple(np.eye(4) + delta * lab)
        blocked += sign * optimal_internal_alignment(REFERENCE, candidate).aligned_candidate
    return float(simplicity_residual(blocked))


# ------------------------------------------------------------------ label models
def chain_her(n: int, rng: np.random.Generator) -> np.ndarray:
    """Root has its own increment; child = parent + increment (F-02 heritable_labels on a chain)."""
    return np.cumsum(rng.standard_normal((n, 4, 4)), axis=0)


def chain_iid(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, 4, 4))


def kappa_chain_her(n: int) -> np.ndarray:
    idx = np.arange(n)
    return np.minimum.outer(idx, idx) + 1.0


def kappa_iid(n: int) -> np.ndarray:
    return np.eye(n)


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def rms_se(values) -> float:
    """Standard error of the RMS from the spread of the squares (delta method)."""
    arr = np.asarray(values, dtype=float) ** 2
    m = float(arr.mean())
    return float(arr.std(ddof=1) / math.sqrt(len(arr)) / (2.0 * math.sqrt(m))) if m > 0 else 0.0


def fit_slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def random_so4(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((4, 4)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if float(np.linalg.det(q)) < 0.0:
        q[:, 0] *= -1.0
    return q


def plane_rotation(alpha: float) -> np.ndarray:
    rot = np.eye(4)
    rot[1, 1] = rot[2, 2] = math.cos(alpha)
    rot[1, 2] = -math.sin(alpha)
    rot[2, 1] = math.sin(alpha)
    return rot


# ------------------------------------------------------------------ gates (recovers)
def gates() -> dict:
    rng = np.random.default_rng(SEED + 7)
    out: dict = {}
    # G1 pure gauge: exact SO(4) frames (with scales) -> holonomy exactly identity
    frames = [rng.uniform(0.5, 2.0) * random_so4(rng) for _ in range(6)]
    out["G1_pure_gauge_angle"] = so4_angle(loop_holonomy(frames))
    # G2 identical labels (kappa = c J) -> all transports identity
    xi = rng.standard_normal((4, 4))
    out["G2_identical_labels_angle"] = holonomy_angle(np.stack([xi] * 5), DELTA)
    # G3 single transport of an exact rotation recovers the angle exactly (n = 2 exact rotation)
    alpha = 0.3
    e_u = np.eye(4) + 0.2 * rng.standard_normal((4, 4))
    e_v = plane_rotation(alpha) @ e_u
    out["G3_single_transport_angle"] = so4_angle(transport(e_u, e_v))
    out["G3_expected"] = alpha
    # G4 two-cell loop (retraced edge) is exactly trivial
    labels2 = rng.standard_normal((2, 4, 4))
    out["G4_two_loop_angle"] = holonomy_angle(labels2, DELTA)
    # G5 delta scaling on one composition face: theta/delta^2 vs analytic, Richardson
    face = chain_her(3, rng)
    an = analytic_angle(face, 1.0)  # coefficient of delta^2
    ladder = {}
    for d in (0.02, 0.01, 0.005, 0.0025):
        ladder[str(d)] = holonomy_angle(face, d) / d**2
    r1 = (4 * ladder["0.0025"] - ladder["0.005"]) / 3
    out["G5_theta_over_delta2"] = ladder
    out["G5_analytic_coefficient"] = an
    out["G5_richardson_over_analytic"] = r1 / an
    out["G5_ratio_0025_over_005"] = holonomy_angle(face, 0.0025) / holonomy_angle(face, 0.005)
    # G6 antisymmetric (rotation) parts are exact gauge at leading order: add random antisym parts
    face_rot = face + np.stack([0.5 * (a - a.T) for a in rng.standard_normal((3, 4, 4))])
    out["G6_analytic_invariant_under_antisym_shift"] = abs(
        analytic_angle(face_rot, 1.0) - analytic_angle(face, 1.0)
    )
    out["G6_numeric_rel_change_under_antisym_shift"] = abs(
        holonomy_angle(face_rot, DELTA) / holonomy_angle(face, DELTA) - 1.0
    )
    # G0 Isserlis constant: E ||[S,T]||_F^2 = 36 for independent symmetric parts of N(0,1)^{4x4}
    g = rng.standard_normal((1_000_000, 2, 4, 4))
    s = 0.5 * (g + np.swapaxes(g, -1, -2))
    comm = s[:, 0] @ s[:, 1] - s[:, 1] @ s[:, 0]
    sq = np.sum(comm * comm, axis=(1, 2))
    out["G0_E_comm_norm2_1e6"] = float(sq.mean())
    out["G0_E_comm_norm2_se"] = float(sq.std(ddof=1) / math.sqrt(len(sq)))
    out["G0_expected"] = 36.0
    del g, s, comm, sq
    # G7 loop kernel identities
    out["G7_Theta_face_her"] = loop_kernel(kappa_chain_her(3))
    out["G7_Theta_face_iid"] = loop_kernel(kappa_iid(3))
    out["G7_Theta_two_loop_random"] = loop_kernel(np.array([[2.0, 0.7], [0.7, 1.3]]))
    out["G7_Theta_shift_invariance"] = loop_kernel(kappa_chain_her(5) + 3.0) - loop_kernel(
        kappa_chain_her(5)
    )
    out["G7_Theta_chain_her_closed_form"] = {
        str(n): [loop_kernel(kappa_chain_her(n)), (n - 1) * (n - 2) / 2] for n in (3, 4, 8, 16, 64)
    }
    return out


# ------------------------------------------------------------------ pilot Monte Carlo
def pilot_chains() -> dict:
    out: dict = {}
    for mode, sampler, kernel in (
        ("her", chain_her, kappa_chain_her),
        ("iid", chain_iid, kappa_iid),
    ):
        rng = np.random.default_rng(SEED)
        per_size = {}
        for n in PILOT_SIZES:
            th_num, th_an, eps, om_p, om_m = [], [], [], [], []
            for _ in range(PILOT_TRIALS):
                labels = sampler(n, rng)
                tetrads = [np.eye(4) + DELTA * lab for lab in labels]
                hol = loop_holonomy(tetrads)
                th_num.append(so4_angle(hol))
                p, m = self_dual_split(hol)
                om_p.append(p)
                om_m.append(m)
                th_an.append(analytic_angle(labels, DELTA))
                eps.append(block_residual(labels, DELTA))
            kappa = kernel(n)
            th_num = np.asarray(th_num)
            th_an = np.asarray(th_an)
            eps = np.asarray(eps)
            per_size[str(n)] = {
                "theta_rms_numeric": rms(th_num),
                "theta_rms_numeric_se": rms_se(th_num),
                "theta_rms_analytic": rms(th_an),
                "theta_rms_pred_isserlis": theta_rms_pred(kappa, DELTA),
                "max_rel_dev_numeric_vs_analytic": float(np.max(np.abs(th_num / th_an - 1.0))),
                "eps_rms_numeric": rms(eps),
                "eps_rms_numeric_se": rms_se(eps),
                "eps_rms_pred_F02": eps_rms_pred(kappa, DELTA),
                "c_theta_numeric": rms(th_num) / rms(eps),
                "c_theta_pred": theta_rms_pred(kappa, DELTA) / eps_rms_pred(kappa, DELTA),
                "corr_theta2_eps2": float(np.corrcoef(th_num**2, eps**2)[0, 1]),
                "cv_theta2": float(np.std(th_num**2, ddof=1) / np.mean(th_num**2)),
                "selfdual_over_antiselfdual_rms": rms(om_p) / rms(om_m),
            }
        sizes = list(PILOT_SIZES)
        out[mode] = {
            "per_size": per_size,
            "theta_slope_numeric": fit_slope(sizes, [per_size[str(n)]["theta_rms_numeric"] for n in sizes]),
            "theta_slope_pred": fit_slope(sizes, [per_size[str(n)]["theta_rms_pred_isserlis"] for n in sizes]),
            "eps_slope_numeric": fit_slope(sizes, [per_size[str(n)]["eps_rms_numeric"] for n in sizes]),
            "eps_slope_pred": fit_slope(sizes, [per_size[str(n)]["eps_rms_pred_F02"] for n in sizes]),
        }
    return out


def pilot_face() -> dict:
    """Composition face f = (u, m, v): heritable at depths 0 and 7 (common random numbers), iid."""
    out: dict = {}
    draws = max(FACE_DEPTHS) + 3
    depth_stats = {}
    for depth in FACE_DEPTHS:
        rng = np.random.default_rng(SEED)
        th, eps_u, eps_s = [], [], []
        for _ in range(FACE_TRIALS):
            xi = rng.standard_normal((draws, 4, 4))
            u = xi[: depth + 1].sum(axis=0)
            m = u + xi[draws - 2]
            v = m + xi[draws - 1]
            labels = np.stack([u, m, v])
            th.append(holonomy_angle(labels, DELTA))
            eps_u.append(block_residual(labels, DELTA))
            eps_s.append(block_residual(labels, DELTA, signs=(1.0, 1.0, -1.0)))
        th, eps_u, eps_s = map(np.asarray, (th, eps_u, eps_s))
        depth_stats[str(depth)] = {
            "theta_rms": rms(th),
            "theta_rms_se": rms_se(th),
            "eps_unsigned_rms": rms(eps_u),
            "eps_signed_rms": rms(eps_s),
            "c_theta_unsigned": rms(th) / rms(eps_u),
            "c_theta_signed": rms(th) / rms(eps_s),
        }
    rng = np.random.default_rng(SEED + 1)
    th, eps_u, eps_s = [], [], []
    for _ in range(FACE_TRIALS):
        labels = rng.standard_normal((3, 4, 4))
        th.append(holonomy_angle(labels, DELTA))
        eps_u.append(block_residual(labels, DELTA))
        eps_s.append(block_residual(labels, DELTA, signs=(1.0, 1.0, -1.0)))
    th, eps_u, eps_s = map(np.asarray, (th, eps_u, eps_s))
    iid_stats = {
        "theta_rms": rms(th),
        "theta_rms_se": rms_se(th),
        "eps_unsigned_rms": rms(eps_u),
        "eps_signed_rms": rms(eps_s),
        "c_theta_unsigned": rms(th) / rms(eps_u),
        "c_theta_signed": rms(th) / rms(eps_s),
    }
    d0 = depth_stats[str(FACE_DEPTHS[0])]
    dd = depth_stats[str(max(FACE_DEPTHS))]
    out["her_by_depth"] = depth_stats
    out["iid"] = iid_stats
    out["rho_face_hol_numeric"] = dd["theta_rms"] / iid_stats["theta_rms"]
    out["rho_face_hol_pred"] = 1.0 / math.sqrt(3.0)
    out["rho_face_eps_unsigned_numeric"] = dd["eps_unsigned_rms"] / iid_stats["eps_unsigned_rms"]
    out["rho_face_eps_unsigned_pred_F02"] = math.sqrt(5.0) / 3.0
    out["rho_face_eps_signed_numeric"] = dd["eps_signed_rms"] / iid_stats["eps_signed_rms"]
    out["face_depth_drift_theta"] = dd["theta_rms"] / d0["theta_rms"]
    out["theta_face_her_pred_over_delta2"] = 3.0 / math.sqrt(2.0)
    out["theta_face_iid_pred_over_delta2"] = math.sqrt(27.0 / 2.0)
    out["theta_face_her_numeric_over_delta2"] = dd["theta_rms"] / DELTA**2
    out["theta_face_iid_numeric_over_delta2"] = iid_stats["theta_rms"] / DELTA**2
    out["c_theta_face_her_pred"] = (3.0 / math.sqrt(2.0)) / (10.0 / 9.0)
    out["c_theta_face_iid_pred"] = math.sqrt(27.0 / 2.0) / (math.sqrt(20.0) / 3.0)
    return out


def kill_grid_predictions() -> dict:
    """Numbers the kill script will be judged against (sizes 16/32/64, exact from the two kernel laws)."""
    sizes = (16, 32, 64)
    her = {}
    iid = {}
    for n in sizes:
        kh, ki = kappa_chain_her(n), kappa_iid(n)
        her[str(n)] = {
            "theta_over_delta2": theta_rms_pred(kh, 1.0),
            "eps_over_delta2_F02": eps_rms_pred(kh, 1.0),
            "c_theta": theta_rms_pred(kh, 1.0) / eps_rms_pred(kh, 1.0),
            "closed_form_c_theta": (9 / math.sqrt(2)) * n * math.sqrt((n - 2) / ((n + 1) * (2 * n * n + 7))),
        }
        iid[str(n)] = {
            "theta_over_delta2": theta_rms_pred(ki, 1.0),
            "eps_over_delta2_F02": eps_rms_pred(ki, 1.0),
            "c_theta": theta_rms_pred(ki, 1.0) / eps_rms_pred(ki, 1.0),
        }
    return {
        "sizes": list(sizes),
        "her": her,
        "iid": iid,
        "theta_slope_her": fit_slope(sizes, [her[str(n)]["theta_over_delta2"] for n in sizes]),
        "eps_slope_her_F02": fit_slope(sizes, [her[str(n)]["eps_over_delta2_F02"] for n in sizes]),
        "theta_slope_iid": fit_slope(sizes, [iid[str(n)]["theta_over_delta2"] for n in sizes]),
        "eps_slope_iid_F02": fit_slope(sizes, [iid[str(n)]["eps_over_delta2_F02"] for n in sizes]),
        "c_theta_chain_limit": 4.5,
    }


def main() -> int:
    t0 = time.time()
    out = {
        "card": "derivations/Q-0015/F-02.formula.md",
        "seed": SEED,
        "delta": DELTA,
        "pilot_sizes": list(PILOT_SIZES),
        "pilot_trials": PILOT_TRIALS,
        "face_trials": FACE_TRIALS,
        "gates": gates(),
        "chains": pilot_chains(),
        "face": pilot_face(),
        "kill_grid_predictions": kill_grid_predictions(),
    }
    out["wall_seconds"] = time.time() - t0
    (HERE / "pilot.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
