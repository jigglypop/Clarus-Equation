"""A3: independent-seed adjudication of rho_face (K4) and face_depth_drift (K3).

The card's smoke run (128 face trials) gave rho_face = 0.664, OUTSIDE the frozen window
[0.685, 0.806].  Prover called it sampling noise.  This replicates face_statistics with
seeds DISJOINT from the pre-registered pair (20260902/20260903) and reports the sampling
distribution of the RMS-ratio estimator, so the noise claim is testable rather than asserted.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "verify" / "Q-0015" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
import check_theta as C  # noqa: E402

DEPTHS = (0, 7)
SLOT_A, SLOT_B = 8, 9
DRAWS = 10


def face_angles(seed: int, depth: int, trials: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(trials)
    for t in range(trials):
        xi = rng.standard_normal((DRAWS, 4, 4))
        anc = xi[: depth + 1].sum(axis=0)
        mid = anc + xi[SLOT_A]
        kid = mid + xi[SLOT_B]
        out[t] = C.eps_and_theta(C.block_triple(np.stack([anc, mid, kid])))[1]
    return out


def iid_angles(seed: int, trials: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([C.eps_and_theta(C.block_triple(rng.standard_normal((3, 4, 4))))[1]
                     for _ in range(trials)])


def rms(a): return float(np.sqrt(np.mean(a * a)))


def rms_relse(a):
    """relative standard error of the RMS estimator: sd(x^2)/(2 E[x^2] sqrt(N))."""
    x2 = np.asarray(a) ** 2
    return float(np.std(x2, ddof=1) / (2.0 * np.mean(x2) * math.sqrt(len(x2))))


out = {"note": "seeds disjoint from the pre-registered 20260902/20260903"}
t0 = time.time()

# --- (1) 12 independent seed pairs at the pre-registered trial count 2048
TRIALS = 2048
seeds = [(1000 + 7 * k, 5000 + 7 * k) for k in range(12)]
reps = []
for sh, si in seeds:
    a0 = face_angles(sh, 0, TRIALS)
    a7 = face_angles(sh, 7, TRIALS)
    ai = iid_angles(si, TRIALS)
    reps.append({"seed_her": sh, "seed_iid": si,
                 "theta_d0": rms(a0), "theta_d7": rms(a7), "theta_iid": rms(ai),
                 "rho_face": rms(a7) / rms(ai), "face_depth_drift": rms(a7) / rms(a0),
                 "relse_d7": rms_relse(a7), "relse_iid": rms_relse(ai)})
out["replicates_2048"] = reps
rho = np.array([r["rho_face"] for r in reps])
drift = np.array([r["face_depth_drift"] for r in reps])
out["rho_face_stats"] = {"mean": float(rho.mean()), "sd": float(rho.std(ddof=1)),
                         "min": float(rho.min()), "max": float(rho.max()),
                         "prereg": 0.7453560, "window": [0.685, 0.806],
                         "n_outside_window": int(((rho < 0.685) | (rho > 0.806)).sum()),
                         "n_replicates": len(rho)}
out["drift_stats"] = {"mean": float(drift.mean()), "sd": float(drift.std(ddof=1)),
                      "min": float(drift.min()), "max": float(drift.max()),
                      "window": [0.90, 1.10],
                      "n_outside_window": int(((drift < 0.90) | (drift > 1.10)).sum())}

# --- (2) smoke-size (128 trials) sampling distribution: was 0.664 plausible noise?
SM = 128
sm = []
for k in range(40):
    a7 = face_angles(20000 + 11 * k, 7, SM)
    ai = iid_angles(60000 + 11 * k, SM)
    sm.append(rms(a7) / rms(ai))
sm = np.array(sm)
out["smoke128_rho_distribution"] = {
    "mean": float(sm.mean()), "sd": float(sm.std(ddof=1)),
    "min": float(sm.min()), "max": float(sm.max()),
    "frac_at_or_below_0.664": float((sm <= 0.664).mean()),
    "frac_outside_window": float(((sm < 0.685) | (sm > 0.806)).mean()),
    "n": int(sm.size)}

# --- (3) pooled high-precision estimate (24576 heritable + 24576 iid)
pool_h = np.concatenate([face_angles(90000 + k, 7, 2048) for k in range(12)])
pool_i = np.concatenate([iid_angles(95000 + k, 2048) for k in range(12)])
out["pooled"] = {"n_each": int(pool_h.size), "theta_d7": rms(pool_h), "theta_iid": rms(pool_i),
                 "rho_face": rms(pool_h) / rms(pool_i),
                 "relse_d7": rms_relse(pool_h), "relse_iid": rms_relse(pool_i),
                 "sqrt5_over_3": math.sqrt(5) / 3,
                 "alt_sqrt_trHkH_ratio_sqrt(2/3)": math.sqrt(2.0 / 3.0)}
out["elapsed_s"] = time.time() - t0
print(json.dumps(out, indent=2, ensure_ascii=False))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
