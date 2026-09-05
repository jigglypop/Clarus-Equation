"""adversary b3: recompute every card constant with the independent Mt (b1) and
re-run the card's Sigma_b / Sigma_o / Sigma_d off-grid (n in 6,12,24) with seed 20260903.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
Mt = np.load(OUT / "b1_Mt.npy")
REF = geometric_self_dual_triple(np.eye(4))
MIN_DET = 0.05
NS = 2.0 * math.sqrt(3.0)
SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)


def e(m, n):
    v = np.zeros(16)
    v[4 * m + n] = 1.0
    return v


SPECS = {
    "kernel": [((e(0, 1) + e(2, 3)) / SQ2, 1.0), (e(0, 3), SQ2)],
    "diag4": [(e(m, m), 1.0) for m in range(4)],
    "off12": [(e(m, n), 1.0) for m in range(4) for n in range(4) if m != n],
    "iso16": [(e(m, n), 1.0) for m in range(4) for n in range(4)],
    "univ_o": [((e(0, 1) + e(2, 3)) / SQ2, 1.0), (e(0, 3), SQ3)],
    "univ_d": [(e(0, 3), 1.0)],
    "ce_i": [((e(0, 1) + e(0, 2) + e(0, 3)) / SQ3, 1.0)],
    "ce_ii": [((e(0, 0) + e(1, 1)) / SQ2, 1.0)],
    "ce_iii": [(e(0, 1), 1.0), ((e(0, 0) + e(1, 1)) / 2.0, 1.0)],
    "piso": [(e(0, 1), 1.0), (e(0, 2), 1.0), (e(0, 3), 1.0)],
    "zero_11": [(e(1, 1), 1.0)],
    "zero_3diag": [((e(0, 0) + e(1, 1) + e(2, 2)) / SQ3, 1.0)],
}


def sigma_of(spec):
    A = np.array([s * v for v, s in spec]).T
    return A @ A.T


def F_of(s):
    return float(np.linalg.norm(np.einsum("ab,abij->ij", s, Mt)))


def T_of(s):
    return float(np.einsum("abij,ac,bd,cdij->", Mt, s, s, Mt))


def master(n, F, T):
    return math.sqrt((n - 1) * ((n - 1) * F * F + 2.0 * T) / (12.0 * n * n))


def cell(label, delta):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + delta * label)).aligned_candidate


def mc(sigma, n, trials, delta, seed):
    lam, V = np.linalg.eigh(sigma)
    A = V @ np.diag(np.sqrt(np.clip(lam, 0.0, None)))
    rng = np.random.default_rng(seed)
    vals, res = [], 0
    for _ in range(trials):
        while True:
            g = rng.normal(size=(n, 16))
            lab = (g @ A.T).reshape(n, 4, 4)
            if np.all(np.linalg.det(np.eye(4)[None] + delta * lab) > MIN_DET):
                break
            res += 1
        vals.append(simplicity_residual(sum(cell(l, delta) for l in lab)))
    v = np.asarray(vals)
    return float(np.sqrt(np.mean(v * v))), res


def floor_hat_generic(n1, e1, n2, e2):
    """F^2 = (A_n2 - A_n1)/(n2-n1), A_n = 12 n^2 eps_n^2/(n-1) = (n-1)F^2 + 2T."""
    A1 = 12.0 * n1 * n1 * e1 * e1 / (n1 - 1)
    A2 = 12.0 * n2 * n2 * e2 * e2 / (n2 - 1)
    return math.sqrt(max((A2 - A1) / (n2 - n1), 0.0)) / NS


def main():
    card_pred = {
        "ker_eps64_over_delta2": 0.07733980, "ker_slope": -0.45342619,
        "diag_eps2_over_delta2": 0.57735027, "diag_eps64_over_delta2": 0.14320549,
        "diag_slope": -0.45342619, "off_eps2_over_delta2": 1.08012345,
        "iso_eps2_over_delta2": 1.58113883, "cross_eps2_sq_over_delta4": 1.0,
        "univ_o_eps64_over_delta2": 0.15118752, "univ_o_eps4_over_delta2": 0.34985116,
        "univ_floor_hat_over_delta2": 0.11785113, "univ_d_eps64_over_delta2": 0.11783674,
        "ce_i_eps64": 0.11783674, "ce_ii_eps64": 0.23567349, "ce_iii_eps64": 0.02923170,
        "piso_eps64": 0.05660694,
    }
    exact = {}
    for name, spec in SPECS.items():
        s = sigma_of(spec)
        F, T = F_of(s), T_of(s)
        w = [float(sum(s[4 * int(a[0]) + int(a[1]), 4 * int(a[0]) + int(a[1])]
                       for a in cls)) for cls in (["01", "10", "23", "32"], ["02", "20", "31", "13"], ["03", "30", "12", "21"])]
        exact[name] = {"F2": F * F, "T": T, "w_axis": w, "floor_over_delta2": F / NS,
                       "eps": {str(n): master(n, F, T) for n in (2, 4, 8, 16, 32, 64)}}
    sizes = (4, 8, 16, 32, 64)
    def slope(name):
        v = [exact[name]["eps"][str(n)] for n in sizes]
        return float(np.polyfit(np.log(sizes), np.log(v), 1)[0])
    mine = {
        "ker_eps64_over_delta2": exact["kernel"]["eps"]["64"], "ker_slope": slope("kernel"),
        "diag_eps2_over_delta2": exact["diag4"]["eps"]["2"],
        "diag_eps64_over_delta2": exact["diag4"]["eps"]["64"], "diag_slope": slope("diag4"),
        "off_eps2_over_delta2": exact["off12"]["eps"]["2"],
        "iso_eps2_over_delta2": exact["iso16"]["eps"]["2"],
        "cross_eps2_sq_over_delta4": exact["iso16"]["eps"]["2"] ** 2 - exact["diag4"]["eps"]["2"] ** 2 - exact["off12"]["eps"]["2"] ** 2,
        "univ_o_eps64_over_delta2": exact["univ_o"]["eps"]["64"],
        "univ_o_eps4_over_delta2": exact["univ_o"]["eps"]["4"],
        "univ_floor_hat_over_delta2": floor_hat_generic(4, exact["univ_o"]["eps"]["4"], 64, exact["univ_o"]["eps"]["64"]),
        "univ_d_eps64_over_delta2": exact["univ_d"]["eps"]["64"],
        "ce_i_eps64": exact["ce_i"]["eps"]["64"], "ce_ii_eps64": exact["ce_ii"]["eps"]["64"],
        "ce_iii_eps64": exact["ce_iii"]["eps"]["64"], "piso_eps64": exact["piso"]["eps"]["64"],
    }
    diff = {k: abs(mine[k] - card_pred[k]) for k in card_pred}

    # off-grid geometric MC for the three kill Sigmas
    delta, trials, ns_ = 0.005, 300, (6, 12, 24)
    mcout = {}
    t0 = time.time()
    for name in ("kernel", "univ_o", "univ_d", "diag4", "off12"):
        s = sigma_of(SPECS[name])
        F, T = F_of(s), T_of(s)
        rec = {"F2": F * F, "T": T}
        for n in ns_:
            r, res = mc(s, n, trials, delta, 20260903 + 1000 * len(name) + n)
            rec[str(n)] = {"obs": r / delta ** 2, "master": master(n, F, T),
                           "ratio": (r / delta ** 2) / master(n, F, T), "resampled": res}
        rec["floor_hat_offgrid_6_24"] = floor_hat_generic(6, rec["6"]["obs"], 24, rec["24"]["obs"])
        rec["floor_exact"] = F / NS
        mcout[name] = rec
        print(name, json.dumps(rec)[:300], flush=True)
    payload = {"exact_from_independent_Mt": exact, "card_pred": card_pred,
               "mine": mine, "abs_diff_vs_card": diff,
               "offgrid_mc": mcout,
               "_meta": {"delta": delta, "trials": trials, "sizes": list(ns_), "seed_base": 20260903,
                         "seconds": time.time() - t0}}
    (OUT / "b3_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("maxdiff vs card PRED:", max(diff.values()))


if __name__ == "__main__":
    main()
