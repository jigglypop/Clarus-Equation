"""Q-0012 F-01 kill script: fourth-cumulant correction to the block-residual kernel law.

Card: derivations/Q-0012/F-01.formula.md (2026-09-02).  PRE-REGISTERED; seed, sizes, delta, trial
counts, distributions, coupling and windows are frozen and are NOT edited after seeing results.

    eps^2 = eps_star^2 [ ||H kappa H||_F^2 + c4 * kappa4 * S_gen ] / n^2 ,
    S_gen = sum_u [ (A^T H A)_uu ]^2 ,   kappa = A A^T ,   xi = A zeta ,   kappa4 = E zeta^4 - 3 ,
    c4 = T4 / (2 T2) = 2/120 = 1/60   (pure geometry of the traceless Plebanski gram form).

The claim that separates this card from adversary b4's guess: the diagonal that carries the
fourth-cumulant term is the GENERATOR A^T H A, not the KERNEL H kappa H.  Two generators with the
same kappa are indistinguishable for Gaussian labels and split at kappa4 != 0.

Modes
  constants : geometry only (no statistics).  L = d/d delta [ aligned Sigma(I + delta xi) ],
              M_ab = tl sym G(L e_a, L e_b), T2 = 60, T4 = 2, c4 = 1/60, sum_a M_aa = 0,
              ||G(Sigma_0,Sigma_0)|| = 2 sqrt(3); exact lattice rationals for caterpillar k=6.
  form      : LADDER STEP 4 / K5 (card revision 2).  Does the physical block residual equal the
              quadratic form to O(delta)?  12 fixed configurations (one seed), delta in {0.005, 0.001}.
              Statistic: RMS over the 12 configurations of |eps_phys/eps_form - 1| at delta = 0.005
              (window <= 0.03) and the RMS ratio delta=0.005 / delta=0.001 (window [3, 7]).
              Revision 1 used the MAX with window 0.02; adversary a8 showed on 300 independent seeds
              (500000 + 911 k, never the pre-registered SEED + 777) that this fires 22% of the time
              when the card is true.  The RMS sampling distribution on the same 300 seeds
              (k5_rms_window_design.py: median 0.0067, p99 0.0132, max 0.0164) fixes the 0.03 window.
              The 12 configurations are a random draw, so both statistics DO have sampling error.
  surrogate : LADDER STEP 5.  Tetrad-free quadratic-form Monte Carlo -- arithmetic check of the
              assembly and the design calculation behind the pre-registered windows.  NOT a verdict.
  labels    : LADDER STEP 6 / K1-K4.  The physical pre-registered run: n = 36, delta = 0.005,
              8192 trials, five label distributions coupled by the inverse-CDF (probability integral)
              transform of one shared normal stream, two modes (iid, caterpillar k=6).
              Expected wall time ~25 min.

Usage: python verify/Q-0012/F-01/check_cumulant.py --mode {constants,form,surrogate,labels,all} [--smoke]
Writes verify/Q-0012/F-01/result.json (unless --smoke).

Label conventions are inherited verbatim from Q-0008 F-02 (verify/Q-0008/F-02/check_modes.py,
untouched): e_v = I + delta * label_v, residual = simplicity_residual (12.4 normalized traceless
Plebanski gram) of the polar-aligned block sum, MIN_DET = 0.05 resampling, statistic = trial RMS.
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

from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    simplicity_residual,
    wedge_scalar,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
DELTA = 0.005
DELTA_FINE = 0.001
MIN_DET = 0.05
N_CELLS = 36
CAT_K = 6
TRIALS = 8192
SURROGATE_TRIALS = 40000
DERIV_H = 1.0e-4

# excess kurtosis kappa4 = E zeta^4 - 3 of each pre-registered label law (mean 0, variance 1, symmetric)
KAPPA4 = {"gauss": 0.0, "rademacher": -2.0, "uniform": -1.2, "laplace": 3.0, "spike64": 61.0}
DISTS = ("gauss", "rademacher", "uniform", "laplace", "spike64")
SPIKE_P = 1.0 / 64.0

# exact lattice rationals (card verify[5], [6]); D = ||H kappa H||_F^2
EXACT = {
    "cat6_S_gen": 62069 / 216,
    "cat6_S_ker": 54023 / 432,
    "cat6_D": 23053 / 36,
    "iid36_S_gen": 35**2 / 36,
    "iid36_D": 35.0,
    "c4": 1 / 60,
}

PREREGISTERED = {
    "rho_iid36_rademacher": 0.9675925926,
    "rho_iid36_uniform": 0.9805555556,
    "rho_iid36_laplace": 1.0486111111,
    "rho_iid36_spike64": 1.9884259259,
    "rho_cat6_rademacher": 0.9850419565,
    "rho_cat6_uniform": 0.9910251739,
    "rho_cat6_laplace": 1.0224370653,
    "rho_cat6_spike64": 1.4562203280,
    "a_iid36": 0.0162037037,
    "a_cat6": 0.0074790218,
    "slope_ratio": 0.4615624864,
    "form_rms_rel_err_delta0005": 0.0,
    "form_ratio_delta_scaling": 5.0,
}
WINDOWS = {
    "rho_iid36_rademacher": (0.9371, 0.9981),
    "rho_iid36_uniform": (0.9681, 0.9931),
    "rho_iid36_laplace": (1.0352, 1.0620),
    "rho_iid36_spike64": (1.8786, 2.0982),
    "rho_cat6_rademacher": (0.9417, 1.0284),
    "rho_cat6_uniform": (0.9733, 1.0088),
    "rho_cat6_laplace": (1.0026, 1.0422),
    "rho_cat6_spike64": (1.3020, 1.6104),
    "a_iid36": (0.014412, 0.017996),
    "a_cat6": (0.004961, 0.009997),
    "slope_ratio": (0.2974, 0.6257),
    "form_rms_rel_err_delta0005": (0.0, 0.03),   # revision 2: RMS statistic; 300-seed max 0.0164 (a8 seeds)
    "form_ratio_delta_scaling": (3.0, 7.0),
}
# discriminating alternative: the diagonal is the KERNEL (H kappa H)_vv (adversary b4's guess)
ALTERNATIVE_KERNEL_FORM = {
    "rho_cat6_spike64": 1 + 61 * EXACT["c4"] * EXACT["cat6_S_ker"] / EXACT["cat6_D"],
    "a_cat6": EXACT["c4"] * EXACT["cat6_S_ker"] / EXACT["cat6_D"],
    "slope_ratio": (EXACT["cat6_S_ker"] / EXACT["cat6_D"]) / (35 / 36),
}

REFERENCE = geometric_self_dual_triple(np.eye(4))


# ---------------------------------------------------------------- geometry
def tl(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.trace(matrix) / 3.0 * np.eye(3)


def gram_form(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.array([[wedge_scalar(first[i], second[j]) for j in range(3)] for i in range(3)])


def aligned_cell(label: np.ndarray, delta: float) -> np.ndarray:
    tetrad = np.eye(4) + delta * label
    return optimal_internal_alignment(REFERENCE, geometric_self_dual_triple(tetrad)).aligned_candidate


def basis_16() -> list[np.ndarray]:
    out = []
    for i in range(4):
        for j in range(4):
            unit = np.zeros((4, 4))
            unit[i, j] = 1.0
            out.append(unit)
    return out


def linear_map(h: float = DERIV_H) -> np.ndarray:
    """L[a] = d/d delta of the polar-aligned Sigma(I + delta e_a); Richardson from h and 2h."""
    def central(step: float) -> np.ndarray:
        return np.array([(aligned_cell(e, step) - aligned_cell(e, -step)) / (2 * step) for e in basis_16()])

    return (4 * central(h) - central(2 * h)) / 3.0


def quadratic_tensor(lmap: np.ndarray) -> np.ndarray:
    """M[a,b] = tl sym G(L e_a, L e_b), a 16 x 16 array of 3 x 3 traceless symmetric matrices."""
    out = np.zeros((16, 16, 3, 3))
    for a in range(16):
        for b in range(16):
            out[a, b] = tl(0.5 * (gram_form(lmap[a], lmap[b]) + gram_form(lmap[b], lmap[a])))
    return out


def geometry_constants(mtensor: np.ndarray) -> dict:
    t2 = float((mtensor * mtensor).sum())
    t4 = float(sum((mtensor[a, a] * mtensor[a, a]).sum() for a in range(16)))
    iso = float(np.abs(sum(mtensor[a, a] for a in range(16))).max())
    return {
        "T2": t2,
        "T4": t4,
        "c4": t4 / (2 * t2),
        "c4_exact_1_over_60": 1 / 60,
        "isotropy_max_abs_sum_a_Maa": iso,
        "norm_G0": float(np.linalg.norm(gram_form(REFERENCE, REFERENCE))),
        "norm_G0_exact_2sqrt3": 2 * math.sqrt(3.0),
    }


# ---------------------------------------------------------------- lattices
def caterpillar(k: int) -> list[int]:
    """Spine of k vertices (root = spine head) plus k-1 leaves on every spine vertex; n = k^2."""
    parent = [-1] + [i - 1 for i in range(1, k)]
    for spine in range(k):
        parent.extend([spine] * (k - 1))
    return parent


def ancestor_matrix(parent: list[int]) -> np.ndarray:
    n = len(parent)
    a = np.zeros((n, n))
    for v in range(n):
        u = v
        while u >= 0:
            a[v, u] = 1.0
            u = parent[u]
    return a


def lattice_constants(parent: list[int]) -> dict:
    n = len(parent)
    centering = np.eye(n) - np.ones((n, n)) / n
    a = ancestor_matrix(parent)
    atha = a.T @ centering @ a
    kernel = centering @ (a @ a.T) @ centering
    return {
        "n": n,
        "S_gen": float(np.sum(np.diag(atha) ** 2)),
        "S_ker": float(np.sum(np.diag(kernel) ** 2)),
        "D": float(np.sum(kernel * kernel)),
    }


# ---------------------------------------------------------------- labels
def uniform_to_label(u: np.ndarray, z: np.ndarray, dist: str) -> np.ndarray:
    """Inverse-CDF (probability integral) coupling: every law is a fixed function of one normal draw."""
    if dist == "gauss":
        return z
    if dist == "rademacher":
        return np.sign(u - 0.5)
    if dist == "uniform":
        return math.sqrt(3.0) * (2 * u - 1)
    if dist == "laplace":
        clipped = np.clip(u, 1e-15, 1 - 1e-15)
        return -np.sign(clipped - 0.5) * np.log(1 - 2 * np.abs(clipped - 0.5)) / math.sqrt(2.0)
    if dist == "spike64":
        return np.where(u < SPIKE_P / 2, -1.0, np.where(u > 1 - SPIKE_P / 2, 1.0, 0.0)) / math.sqrt(SPIKE_P)
    raise ValueError(f"unknown distribution: {dist}")


_ERF = np.vectorize(math.erf)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(z / math.sqrt(2.0)))


def heritable(parent: list[int], increments: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(increments)
    for v in range(len(parent)):
        chain = []
        u = v
        while u >= 0:
            chain.append(u)
            u = parent[u]
        labels[v] = increments[chain].sum(axis=0)
    return labels


def block_residual(labels: np.ndarray, delta: float) -> float:
    blocked = np.zeros_like(REFERENCE)
    for label in labels:
        if float(np.linalg.det(np.eye(4) + delta * label)) <= MIN_DET:
            return math.nan
        blocked += aligned_cell(label, delta)
    return simplicity_residual(blocked)


def quadratic_residual(labels: np.ndarray, delta: float, mtensor: np.ndarray, norm_g0: float) -> float:
    flat = np.asarray(labels).reshape(len(labels), 16)
    centered = flat - flat.mean(axis=0)
    phi = np.einsum("va,vb,abij->ij", centered, centered, mtensor)
    return delta * delta * float(np.linalg.norm(phi)) / (len(labels) * norm_g0)


# ---------------------------------------------------------------- modes
def run_constants(mtensor: np.ndarray) -> dict:
    out = geometry_constants(mtensor)
    cat = lattice_constants(caterpillar(CAT_K))
    out["caterpillar_k6"] = cat
    out["caterpillar_k6_exact"] = {"S_gen": EXACT["cat6_S_gen"], "S_ker": EXACT["cat6_S_ker"], "D": EXACT["cat6_D"]}
    out["a_cat6_from_constants"] = out["c4"] * cat["S_gen"] / cat["D"]
    out["a_iid36_from_constants"] = out["c4"] * EXACT["iid36_S_gen"] / EXACT["iid36_D"]
    return out


def run_form(mtensor: np.ndarray, norm_g0: float, sizes=(3, 5, 8, 12), seed: int = SEED + 777) -> dict:
    """K5 battery: 12 configurations drawn from one seed -> the RMS / max statistics have sampling error.

    Revision 2 kills on the RMS (window 0.03) and the delta-scaling ratio (window [3, 7]); the max is
    kept in the block for information only.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        z = rng.standard_normal((n, 4, 4))
        parent = [-1] + [(i - 1) // 2 for i in range(1, n)]
        configs = {
            "gauss": z,
            "rademacher": np.sign(z),
            "heritable_gauss": heritable(parent, z),
        }
        for name, labels in configs.items():
            row = {"n": n, "labels": name}
            for tag, delta in (("delta0005", DELTA), ("delta0001", DELTA_FINE)):
                actual = block_residual(labels, delta)
                predicted = quadratic_residual(labels, delta, mtensor, norm_g0)
                row[tag] = {"actual": actual, "quadratic_form": predicted,
                            "rel_err": (actual - predicted) / predicted}
            rows.append(row)
    coarse = np.array([r["delta0005"]["rel_err"] for r in rows])
    fine = np.array([r["delta0001"]["rel_err"] for r in rows])
    return {
        "configurations": rows,
        "max_rel_err_delta0005": float(np.max(np.abs(coarse))),
        "median_rel_err_delta0005": float(np.median(np.abs(coarse))),
        "rms_rel_err_delta0005": float(np.sqrt(np.mean(coarse**2))),
        "rms_rel_err_delta0001": float(np.sqrt(np.mean(fine**2))),
        "ratio_delta_scaling": float(np.sqrt(np.mean(coarse**2)) / np.sqrt(np.mean(fine**2))),
    }


def _ratio_and_se(values: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    n = len(values)
    ma, mb = float(values.mean()), float(reference.mean())
    cov = np.cov(values, reference)
    var = (cov[0, 0] / mb**2 - 2 * ma * cov[0, 1] / mb**3 + ma**2 * cov[1, 1] / mb**4) / n
    return ma / mb, math.sqrt(max(var, 0.0))


def _slope(rhos: dict[str, float]) -> float:
    """a = sum_d kappa4_d (rho_d - 1) / sum_d kappa4_d^2 over the four non-Gaussian laws (weights 1)."""
    num = sum(KAPPA4[d] * (rhos[d] - 1.0) for d in DISTS if d != "gauss")
    den = sum(KAPPA4[d] ** 2 for d in DISTS if d != "gauss")
    return num / den


def run_surrogate(mtensor: np.ndarray, trials: int = SURROGATE_TRIALS, seed: int = SEED + 9000) -> dict:
    """Ladder step 5: tetrad-free quadratic-form MC (arithmetic check + window design)."""
    n = N_CELLS
    centering = np.eye(n) - np.ones((n, n)) / n
    generators = {"iid36": centering, "cat6": centering @ ancestor_matrix(caterpillar(CAT_K))}
    rng = np.random.default_rng(seed)
    samples = {mode: {d: np.empty(trials) for d in DISTS} for mode in generators}
    for t in range(trials):
        z = rng.standard_normal((n, 16))
        u = normal_cdf(z)
        for dist in DISTS:
            zeta = uniform_to_label(u, z, dist)
            for mode, gen in generators.items():
                centered = gen @ zeta
                phi = np.einsum("va,vb,abij->ij", centered, centered, mtensor)
                samples[mode][dist][t] = float(np.sum(phi * phi))
    out: dict = {"trials": trials, "seed": seed}
    for mode in generators:
        rhos, ses = {}, {}
        for dist in DISTS:
            if dist == "gauss":
                rhos[dist], ses[dist] = 1.0, 0.0
                continue
            rhos[dist], ses[dist] = _ratio_and_se(samples[mode][dist], samples[mode]["gauss"])
        out[mode] = {"rho": rhos, "se": ses, "slope": _slope(rhos)}
    return out


def run_labels(trials: int = TRIALS, n: int = N_CELLS, seed: int = SEED) -> dict:
    """K1-K4: the physical pre-registered run."""
    parent = caterpillar(CAT_K)
    assert len(parent) == n, "caterpillar k=6 must have n=36 cells"
    rng = np.random.default_rng(seed)
    samples = {mode: {d: np.empty(trials) for d in DISTS} for mode in ("iid36", "cat6")}
    rejections = 0
    done = 0
    while done < trials:
        z = rng.standard_normal((n, 4, 4))
        u = normal_cdf(z)
        trial: dict[str, dict[str, float]] = {"iid36": {}, "cat6": {}}
        ok = True
        for dist in DISTS:
            zeta = uniform_to_label(u, z, dist)
            for mode, labels in (("iid36", zeta), ("cat6", heritable(parent, zeta))):
                value = block_residual(labels, DELTA)
                if not math.isfinite(value):
                    ok = False
                    break
                trial[mode][dist] = value * value
            if not ok:
                break
        if not ok:
            rejections += 1
            continue
        for mode in samples:
            for dist in DISTS:
                samples[mode][dist][done] = trial[mode][dist]
        done += 1

    out: dict = {"trials": trials, "n": n, "delta": DELTA, "seed": seed, "min_det": MIN_DET,
                 "rejections": rejections, "distributions": {d: KAPPA4[d] for d in DISTS}}
    stats: dict[str, float] = {}
    for mode in ("iid36", "cat6"):
        rhos, ses = {}, {}
        for dist in DISTS:
            if dist == "gauss":
                rhos[dist], ses[dist] = 1.0, 0.0
                continue
            rho, se = _ratio_and_se(samples[mode][dist], samples[mode]["gauss"])
            rhos[dist], ses[dist] = rho, se
            stats[f"rho_{mode}_{dist}"] = rho
        slope = _slope(rhos)
        stats[f"a_{mode}"] = slope
        out[mode] = {"rho": rhos, "se_observed": ses, "slope": slope,
                     "mean_eps2": {d: float(samples[mode][d].mean()) for d in DISTS}}
    stats["slope_ratio"] = stats["a_cat6"] / stats["a_iid36"]
    out["stats"] = stats
    return out


# ---------------------------------------------------------------- main
def verdict_for(stats: dict[str, float]) -> dict[str, str]:
    out = {}
    for key, value in stats.items():
        if key not in WINDOWS:
            continue
        low, high = WINDOWS[key]
        out[key] = "survive" if low <= value <= high else "KILL"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=("constants", "form", "surrogate", "labels", "all"))
    parser.add_argument("--smoke", action="store_true", help="execution check only (tiny sizes, no verdict)")
    args = parser.parse_args()

    mtensor = quadratic_tensor(linear_map())
    norm_g0 = float(np.linalg.norm(gram_form(REFERENCE, REFERENCE)))

    if args.smoke:
        constants = geometry_constants(mtensor)
        cat = lattice_constants(caterpillar(3))
        form = run_form(mtensor, norm_g0, sizes=(3,), seed=SEED + 1)
        surrogate = run_surrogate(mtensor, trials=25, seed=SEED + 2)
        small = run_labels_smoke(mtensor, norm_g0)
        print(json.dumps({"smoke": "ok", "constants": constants, "caterpillar_k3": cat,
                          "form_rms_rel_err": form["rms_rel_err_delta0005"],
                          "form_max_rel_err": form["max_rel_err_delta0005"],
                          "form_ratio_delta_scaling": form["ratio_delta_scaling"],
                          "surrogate_rho_iid36": surrogate["iid36"]["rho"],
                          "labels_smoke": small}, ensure_ascii=False))
        return 0

    result: dict = {"card": "F-01", "question": "Q-0012", "seed": SEED, "delta": DELTA, "n": N_CELLS,
                    "trials": TRIALS, "preregistered": PREREGISTERED, "windows": WINDOWS,
                    "alternative_kernel_form": ALTERNATIVE_KERNEL_FORM}
    modes = ("constants", "form", "surrogate", "labels") if args.mode == "all" else (args.mode,)
    stats: dict[str, float] = {}
    for mode in modes:
        if mode == "constants":
            result["constants"] = run_constants(mtensor)
        elif mode == "form":
            block = run_form(mtensor, norm_g0)
            result["form"] = block
            stats["form_rms_rel_err_delta0005"] = block["rms_rel_err_delta0005"]
            stats["form_ratio_delta_scaling"] = block["ratio_delta_scaling"]
        elif mode == "surrogate":
            result["surrogate"] = run_surrogate(mtensor)
        else:
            block = run_labels()
            result["labels"] = block
            stats.update(block["stats"])

    result["stats"] = stats
    result["verdict"] = verdict_for(stats)
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
    print(json.dumps({"verdict": result["verdict"], "stats": stats}, ensure_ascii=False))
    return 0


def run_labels_smoke(mtensor: np.ndarray, norm_g0: float) -> dict:
    """Tiny end-to-end exercise of the labels path (n=6, 3 trials): execution check only."""
    parent = caterpillar(2)
    n = len(parent)
    rng = np.random.default_rng(SEED + 4)
    values = {mode: {d: [] for d in DISTS} for mode in ("iid", "cat")}
    for _ in range(3):
        z = rng.standard_normal((n, 4, 4))
        u = normal_cdf(z)
        for dist in DISTS:
            zeta = uniform_to_label(u, z, dist)
            values["iid"][dist].append(block_residual(zeta, DELTA))
            values["cat"][dist].append(block_residual(heritable(parent, zeta), DELTA))
    return {mode: {d: float(np.sqrt(np.mean(np.square(v)))) for d, v in inner.items()}
            for mode, inner in values.items()}


if __name__ == "__main__":
    raise SystemExit(main())
