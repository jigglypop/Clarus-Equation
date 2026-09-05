"""b2 (re-audit): does the revision-2 K5 actually kill anything?

A window that no wrong model can leave is not a kill.  Here the physical side (block_residual, polar
aligned) is untouched and only the PREDICTION tensor M is replaced by a deliberately wrong one:

  ok        : M as in the card (control)
  noalign   : L computed WITHOUT the polar SO(3) alignment (the card's ladder step 2 says this is what
              K5 is supposed to catch: sum_a M_aa != 0)
  diag_x2c4 : diagonal blocks M_aa scaled by sqrt(2) -> T4 = 4, c4 = T4/(2 T2) ~ 1/31 (c4 doubled)
  diag_zero : diagonal blocks set to 0 -> c4 = 0 (the 'no fourth-cumulant term' alternative)
  global_x2 : M -> 2 M (trivial mis-normalisation, sanity of the statistic)
  jitter_p  : M -> M (1 + p * noise), p in {0.01, 0.03, 0.10} -> smallest distortion K5 can see

Seeds: 12 seeds from the b1 stream (3000017 + 104729 k), never the pre-registered SEED + 777.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import (linear_map, quadratic_tensor, gram_form, REFERENCE, run_form, SEED,
                            tl, basis_16, DERIV_H)  # noqa
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple  # noqa

OUT = Path(__file__).parent
WIN_RMS, WIN_RATIO = 0.03, (3.0, 7.0)


def linear_map_noalign(h: float = DERIV_H) -> np.ndarray:
    def central(step):
        return np.array([(geometric_self_dual_triple(np.eye(4) + step * e)
                          - geometric_self_dual_triple(np.eye(4) - step * e)) / (2 * step)
                         for e in basis_16()])
    return (4 * central(h) - central(2 * h)) / 3.0


def constants(M):
    t2 = float((M * M).sum())
    t4 = float(sum((M[a, a] * M[a, a]).sum() for a in range(16)))
    iso = float(np.abs(sum(M[a, a] for a in range(16))).max())
    return {"T2": t2, "T4": t4, "c4": t4 / (2 * t2), "max_abs_sum_a_Maa": iso}


def scale_diag(M, factor):
    out = M.copy()
    for a in range(16):
        out[a, a] = factor * M[a, a]
    return out


def evaluate(name, M, g0, seeds):
    rms, ratio = [], []
    for s in seeds:
        b = run_form(M, g0, sizes=(3, 5, 8, 12), seed=s)
        rms.append(b["rms_rel_err_delta0005"])
        ratio.append(b["ratio_delta_scaling"])
    rms, ratio = np.array(rms), np.array(ratio)
    fires = (rms > WIN_RMS) | (ratio < WIN_RATIO[0]) | (ratio > WIN_RATIO[1])
    return {"variant": name, "constants": constants(M),
            "rms_median": float(np.median(rms)), "rms_min": float(rms.min()), "rms_max": float(rms.max()),
            "ratio_median": float(np.median(ratio)), "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
            "K5_fires_fraction": float(np.mean(fires)),
            "fires_by_rms": float(np.mean(rms > WIN_RMS)),
            "fires_by_ratio": float(np.mean((ratio < WIN_RATIO[0]) | (ratio > WIN_RATIO[1])))}


def main():
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seeds = [3000017 + 104729 * k for k in range(nseeds)]
    assert SEED + 777 not in seeds
    M = quadratic_tensor(linear_map())
    g0 = float(np.linalg.norm(gram_form(REFERENCE, REFERENCE)))
    M_noalign = quadratic_tensor(linear_map_noalign())
    rng = np.random.default_rng(4242)
    variants = [("ok", M),
                ("noalign", M_noalign),
                ("diag_x2c4", scale_diag(M, np.sqrt(2.0))),
                ("diag_zero", scale_diag(M, 0.0)),
                ("global_x2", 2.0 * M)]
    for p in (0.01, 0.03, 0.10):
        variants.append((f"jitter_{p}", M * (1.0 + p * rng.standard_normal(M.shape))))
    rows = [evaluate(name, Mv, g0, seeds) for name, Mv in variants]
    res = {"seeds": seeds, "window_rms": WIN_RMS, "window_ratio": list(WIN_RATIO), "variants": rows}
    for r in rows:
        print("%-12s c4=%.5f  sum_Maa=%.2e  RMS med %.4f [%.4f,%.4f]  ratio med %.2f  K5 fires %.2f"
              % (r["variant"], r["constants"]["c4"], r["constants"]["max_abs_sum_a_Maa"],
                 r["rms_median"], r["rms_min"], r["rms_max"], r["ratio_median"], r["K5_fires_fraction"]))
    (OUT / "b2_k5_kill_sensitivity.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
