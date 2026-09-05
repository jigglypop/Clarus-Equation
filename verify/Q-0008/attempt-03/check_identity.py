"""Q-0008 attempt-03, ladder step 2: the EXACT block identity

    tl gram(Y) = -n sum_v tl gram(eta_v - etabar),      Y = sum_v X_v,  X_v = Sigma_0 + eta_v,

for polar-aligned cells that are each EXACTLY simple (tl gram(X_v) = 0) around an exactly simple
reference (tl gram(Sigma_0) = 0).  gram(B)_{ij} = B^i wedge B^j (examples/physics.plebanski_gram),
tl M = M - (tr M / 3) I_3.

Three independent layers, in this order:

  (A) SYMBOLIC, model-free.  The claim is a statement about ANY symmetric bilinear form T with
      values in a vector space (tl gram is such a form, componentwise).  Represent T(eta_v,eta_w)
      by free symbols T_vw = T_wv, T(Sigma_0,eta_v) by u_v, T(Sigma_0,Sigma_0) by t0.  Then
      the premises are t0 = 0 and 2 u_v + T_vv = 0, and the theorem is a POLYNOMIAL IDENTITY in
      the free symbols -- verified here for n = 2,3,5,8,17.  The same routine, run WITHOUT the
      cell-simplicity substitution, must leave a nonzero residual (that is where the premise is
      used, and it is the only place).
  (B) NUMERIC, physics.  Random tetrads e = I + delta*N(0,1)^{4x4} (seed 20260902), polar-aligned
      to Sigma(I) with examples.physics.gravity.urbantke_shape_matching_rg.optimal_internal_alignment,
      n in {2,5,17}, delta in {0.3, 0.05}.  Relative error of the identity must be <= TOL_IDENT.
      delta = 0.3 is deliberately large: the identity is not a small-delta statement.
  (C) COROLLARIES.  (a) two deterministic species (13.5): eps = p(1-p) ||tl gram Delta|| /
      ||gram(Sigma_0 + (1-p) Delta)||, exact and n-independent.  (b) one defect cell out of n:
      p(1-p) = (n-1)/n^2.

NOT run here: the pre-registered kill/consistency scripts (verify/Q-0008/F-02/check_modes.py,
ladder steps 6-7, including the K4 defect window).  Layer (C) tests the IDENTITY on its own
Delta samples; no pre-registered window is evaluated or compared against.

Tolerances and seed are declared before any result is produced and are not changed afterwards.
Usage:  .claude\\hooks\\python.cmd python verify\\Q-0008\\attempt-03\\check_identity.py
Output: verify/Q-0008/attempt-03/result.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    plebanski_gram,
    simplicity_residual,
    wedge_scalar,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

# ---------------------------------------------------------------- pre-declared constants
SEED = 20260902
TOL_IDENT = 1.0e-12       # relative error of the exact identity (card ladder step 2 claim)
TOL_CELL = 1.0e-12        # per-cell exact simplicity after polar alignment
TOL_ALG = 1.0e-12         # closed-form algebra
MIN_DET = 0.2             # tetrad acceptance (same rule as adversary b8 at delta = 0.3)
N_GRID = (2, 5, 17)
DELTA_GRID = (0.3, 0.05)
TRIALS = 20
SPECIES_GRID = (4, 8, 16, 32, 64)
P_GRID = (0.5, 0.25)
DEFECT_RATIO_ALGEBRAIC = 0.140625   # ((64-1)/64^2)/((8-1)/8^2), card K4 numerator law

REF = geometric_self_dual_triple(np.eye(4))


def tl(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.trace(matrix) / 3.0 * np.eye(3)


def gram(triple: np.ndarray) -> np.ndarray:
    return plebanski_gram(triple)


def cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """G(B,B')_{ij} = B^i wedge B'^j (bilinear, generally NOT symmetric as a matrix)."""
    return np.array([[wedge_scalar(first[i], second[j]) for j in range(3)] for i in range(3)])


def rel(a: np.ndarray, b: np.ndarray) -> float:
    scale = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / scale) if scale > 0.0 else float(np.linalg.norm(a - b))


def aligned_cell(tetrad: np.ndarray) -> np.ndarray:
    return optimal_internal_alignment(REF, geometric_self_dual_triple(tetrad)).aligned_candidate


def draw_cells(n: int, delta: float, rng: np.random.Generator) -> list[np.ndarray]:
    cells: list[np.ndarray] = []
    while len(cells) < n:
        tetrad = np.eye(4) + delta * rng.normal(size=(4, 4))
        if float(np.linalg.det(tetrad)) > MIN_DET:
            cells.append(aligned_cell(tetrad))
    return cells


# ---------------------------------------------------------------- (A) symbolic layer
def symbolic_layer() -> dict:
    try:
        import sympy as sp  # noqa: PLC0415
    except Exception as error:  # pragma: no cover - environment dependent
        return {"status": "skipped", "reason": f"sympy unavailable: {error}"}

    def free_form(n: int):
        t = [[None] * n for _ in range(n)]
        for v in range(n):
            for w in range(v, n):
                s = sp.Symbol(f"T_{v}_{w}")
                t[v][w] = s
                t[w][v] = s
        return t

    def theorem_residual(n: int, *, use_simplicity: bool):
        """LHS - RHS of  tl gram(Y) = -n sum_v tl gram(eta_v - etabar)  in the free-form model."""
        t = free_form(n)
        total = sum(t[v][w] for v in range(n) for w in range(n))          # T(S,S), S = sum eta
        if use_simplicity:
            u = [-t[v][v] / 2 for v in range(n)]                          # 2 u_v = -T_vv  (S4)
            t0 = sp.Integer(0)                                            # tl gram(Sigma_0) = 0
        else:
            u = [sp.Symbol(f"u_{v}") for v in range(n)]
            t0 = sp.Symbol("t0")
        lhs = n**2 * t0 + 2 * n * sum(u) + total                          # (S5.1)
        rhs = -n * sum(
            t[v][v] - 2 * sum(t[v][w] for w in range(n)) / n + total / n**2 for v in range(n)
        )                                                                 # (S5.2)+(S5.3)
        return sp.simplify(sp.expand(lhs - rhs))

    def centering_residual(n: int):
        x = sp.symbols(f"x0:{n}", real=True)
        bar = sum(x) / n
        lhs = sum((x[v] - bar) ** 2 for v in range(n))
        rhs = sum(x[v] ** 2 for v in range(n)) - sum(x) ** 2 / n
        return sp.simplify(sp.expand(lhs - rhs))

    proved = {}
    for n in (2, 3, 5, 8, 17):
        proved[n] = theorem_residual(n, use_simplicity=True) == 0
    needs = {n: theorem_residual(n, use_simplicity=False) != 0 for n in (2, 3, 5)}
    center = {n: centering_residual(n) == 0 for n in (2, 3, 5, 8, 17)}

    n, p = sp.symbols("n p", positive=True)
    two_species = sp.simplify(n * p * (1 - p) ** 2 + n * (1 - p) * p**2 - n * p * (1 - p)) == 0
    defect = sp.simplify((sp.Integer(1) / n) * (1 - sp.Integer(1) / n) - (n - 1) / n**2) == 0
    v = sp.Symbol("v", integer=True)
    nn = sp.Symbol("n", integer=True, positive=True)
    general_n = sp.simplify(
        sp.expand(sp.Sum(v**2, (v, 1, nn)) - (nn * (nn + 1) / 2) ** 2 / nn - nn * (nn**2 - 1) / 12)
    ) == 0

    ok = all(proved.values()) and all(needs.values()) and all(center.values())
    ok = ok and two_species and defect and general_n
    return {
        "status": "pass" if ok else "fail",
        "theorem_exact_for_n": {str(k): bool(v) for k, v in proved.items()},
        "cell_simplicity_is_necessary": {str(k): bool(v) for k, v in needs.items()},
        "centering_identity_for_n": {str(k): bool(v) for k, v in center.items()},
        "two_species_sum_np(1-p)": bool(two_species),
        "defect_p_eq_1_over_n": bool(defect),
        "general_n_centering_label_family_v": bool(general_n),
    }


# ---------------------------------------------------------------- (B) numeric physics layer
def bilinearity_layer(rng: np.random.Generator) -> dict:
    worst_sym = 0.0
    worst_bil = 0.0
    worst_gram_sym = 0.0
    for _ in range(20):
        a, b, c = (rng.normal(size=(3, 6)) for _ in range(3))
        s, t = float(rng.normal()), float(rng.normal())
        worst_sym = max(worst_sym, rel(cross(a, b), cross(b, a).T))
        worst_bil = max(worst_bil, rel(cross(s * a + t * b, c), s * cross(a, c) + t * cross(b, c)))
        g = cross(a, a)
        worst_gram_sym = max(worst_gram_sym, rel(g, g.T))
    return {
        "status": "pass" if max(worst_sym, worst_bil, worst_gram_sym) <= TOL_ALG else "fail",
        "max_rel_err_G(A,B)=G(B,A)^T": worst_sym,
        "max_rel_err_bilinear": worst_bil,
        "max_rel_err_gram_symmetric": worst_gram_sym,
    }


def identity_layer(rng: np.random.Generator) -> dict:
    rows = []
    worst = 0.0
    worst_cell = 0.0
    for delta in DELTA_GRID:
        for n in N_GRID:
            block_worst = 0.0
            for _ in range(TRIALS):
                cells = draw_cells(n, delta, rng)
                worst_cell = max(worst_cell, max(simplicity_residual(x) for x in cells))
                eta = [x - REF for x in cells]
                bar = sum(eta) / n
                y = sum(cells)
                lhs = tl(gram(y))
                rhs = -n * sum(tl(gram(e - bar)) for e in eta)
                block_worst = max(block_worst, rel(lhs, rhs))
            rows.append({"delta": delta, "n": n, "trials": TRIALS, "max_rel_err": block_worst})
            worst = max(worst, block_worst)
    return {
        "status": "pass" if worst <= TOL_IDENT and worst_cell <= TOL_CELL else "fail",
        "max_rel_err": worst,
        "max_cell_simplicity_residual_after_alignment": worst_cell,
        "rows": rows,
    }


def nonsimple_layer(rng: np.random.Generator) -> dict:
    """(S9.3) If cells are NOT simple, tau_v := tl gram(X_v) enters linearly and the identity
    becomes tl gram(Y) = n sum_v tau_v - n sum_v tl gram(eta_v - etabar).  Checked with generic
    (non-geometric) triples, i.e. cells for which tau_v is O(1)."""
    worst = 0.0
    worst_tau = 0.0
    for n in (2, 5, 17):
        for _ in range(TRIALS):
            eta = [0.3 * rng.normal(size=(3, 6)) for _ in range(n)]
            cells = [REF + e for e in eta]
            tau = [tl(gram(x)) for x in cells]
            worst_tau = max(worst_tau, max(float(np.linalg.norm(s)) for s in tau))
            bar = sum(eta) / n
            y = sum(cells)
            lhs = tl(gram(y))
            rhs = n * sum(tau) - n * sum(tl(gram(e - bar)) for e in eta)
            worst = max(worst, rel(lhs, rhs))
    return {
        "status": "pass" if worst <= TOL_IDENT else "fail",
        "max_rel_err": worst,
        "min_tau_scale_note": "tau_v is O(1) here, so the check is not vacuous",
        "max_tau_norm": worst_tau,
    }


# ---------------------------------------------------------------- (C) corollaries
def corollary_two_species(rng: np.random.Generator) -> dict:
    """(S6) eps = p(1-p) ||tl gram Delta|| / ||gram(Sigma_0 + (1-p) Delta)||, exact, n-independent.
    p = fraction of the label-0 species B; the other (1-p) fraction carries the fixed Delta."""
    cell_c = draw_cells(1, 0.3, rng)[0]
    delta_triple = cell_c - REF
    num = float(np.linalg.norm(tl(gram(delta_triple))))
    rows = []
    worst = 0.0
    spread = {}
    for p in P_GRID:
        eps_values = []
        for n in SPECIES_GRID:
            n_b = int(round(p * n))
            block = n_b * REF + (n - n_b) * cell_c
            observed = simplicity_residual(block)
            den = float(np.linalg.norm(gram(REF + (1.0 - p) * delta_triple)))
            predicted = p * (1.0 - p) * num / den
            err = abs(observed - predicted) / abs(observed)
            worst = max(worst, err)
            eps_values.append(observed)
            rows.append({"p": p, "n": n, "eps": observed, "eps_predicted": predicted, "rel_err": err})
        spread[str(p)] = (max(eps_values) - min(eps_values)) / max(eps_values)
    return {
        "status": "pass" if worst <= TOL_IDENT and max(spread.values()) <= TOL_IDENT else "fail",
        "max_rel_err": worst,
        "n_independence_relative_spread": spread,
        "rows": rows,
    }


def corollary_defect(rng: np.random.Generator) -> dict:
    """(S7) one defect cell out of n: p(1-p) = (n-1)/n^2 exactly.  The pre-registered K4 window is
    NOT evaluated here; only the identity residual and the algebraic numerator law have status."""
    cell_c = draw_cells(1, 0.3, rng)[0]
    delta_triple = cell_c - REF
    num = float(np.linalg.norm(tl(gram(delta_triple))))
    rows = []
    worst = 0.0
    den_ref = float(np.linalg.norm(gram(REF)))
    den_drift = 0.0
    for n in SPECIES_GRID:
        block = (n - 1) * REF + cell_c
        observed = simplicity_residual(block)
        den = float(np.linalg.norm(gram(REF + delta_triple / n)))
        predicted = (n - 1) / n**2 * num / den
        err = abs(observed - predicted) / abs(observed)
        worst = max(worst, err)
        den_drift = max(den_drift, abs(den / den_ref - 1.0))
        rows.append({"n": n, "eps": observed, "eps_predicted": predicted, "rel_err": err})
    ratio_alg = ((64 - 1) / 64**2) / ((8 - 1) / 8**2)
    return {
        "status": "pass"
        if worst <= TOL_IDENT and abs(ratio_alg - DEFECT_RATIO_ALGEBRAIC) <= TOL_ALG
        else "fail",
        "max_rel_err": worst,
        "numerator_law_ratio_64_over_8": ratio_alg,
        "denominator_drift_max_over_grid": den_drift,
        "note": "denominator drift is the size of the 1+O(||Delta||/n) correction to the pure "
        "(n-1)/n^2 numerator law; no pre-registered window is judged here",
        "rows": rows,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng = np.random.default_rng(SEED)
    out = {
        "question": "Q-0008",
        "attempt": 3,
        "ladder_step": 2,
        "seed": SEED,
        "tolerances": {"TOL_IDENT": TOL_IDENT, "TOL_CELL": TOL_CELL, "TOL_ALG": TOL_ALG},
        "grid": {"n": list(N_GRID), "delta": list(DELTA_GRID), "trials": TRIALS,
                 "species_n": list(SPECIES_GRID), "p": list(P_GRID)},
        "not_run": ["verify/Q-0008/F-02/check_modes.py (pre-registered kills K1/K2/K3/K5 and the "
                    "K4 consistency window, ladder steps 6-7)"],
    }
    out["symbolic"] = symbolic_layer()
    out["bilinearity"] = bilinearity_layer(rng)
    out["identity"] = identity_layer(rng)
    out["nonsimple_generalization"] = nonsimple_layer(rng)
    out["corollary_two_species"] = corollary_two_species(rng)
    out["corollary_defect"] = corollary_defect(rng)

    numeric_layers = ("bilinearity", "identity", "nonsimple_generalization",
                      "corollary_two_species", "corollary_defect")
    out["numeric"] = "pass" if all(out[k]["status"] == "pass" for k in numeric_layers) else "fail"
    out["max_rel_err"] = max(
        float(out[k].get("max_rel_err", 0.0)) for k in numeric_layers if "max_rel_err" in out[k]
    )
    out["verdict"] = (
        "identity holds exactly"
        if out["numeric"] == "pass" and out["symbolic"]["status"] == "pass"
        else "CHECK FAILED"
    )
    (Path(__file__).resolve().parent / "result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (dict, list))}, indent=2))
    print("symbolic:", out["symbolic"]["status"], " numeric:", out["numeric"],
          " max_rel_err:", f"{out['max_rel_err']:.3e}")
    for key in numeric_layers:
        print(f"  {key:28s} {out[key]['status']:6s} "
              f"max_rel_err={out[key].get('max_rel_err', float('nan')):.3e}")
    print("  cell simplicity after alignment (max):",
          f"{out['identity']['max_cell_simplicity_residual_after_alignment']:.3e}")
    print("  two-species n-independence spread:", out["corollary_two_species"]["n_independence_relative_spread"])
    print("  defect numerator law 64/8:", out["corollary_defect"]["numerator_law_ratio_64_over_8"])
    return 0 if out["verdict"] != "CHECK FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
