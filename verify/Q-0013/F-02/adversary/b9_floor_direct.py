"""adversary b9: measure the FLOOR directly (mean of the normalized traceless Gram),
which the master predicts to be (n-1)/n * ||tl G(Sigma)||/(2 sqrt3) * delta^2 and to be
independent of T -- the norm-universality claim behind K3. Off-grid n, seed 20260903.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
Mt = np.load(OUT / "b1_Mt.npy")
REF = geometric_self_dual_triple(np.eye(4))
NS = 2.0 * math.sqrt(3.0)
SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)


def e(m, n):
    v = np.zeros(16); v[4 * m + n] = 1.0; return v


CASES = {
    "univ_o": [((e(0, 1) + e(2, 3)) / SQ2, 1.0), (e(0, 3), SQ3)],
    "univ_d": [(e(0, 3), 1.0)],
    "kernel_sigma_b": [((e(0, 1) + e(2, 3)) / SQ2, 1.0), (e(0, 3), SQ2)],
    "ce_i": [((e(0, 1) + e(0, 2) + e(0, 3)) / SQ3, 1.0)],
    "ce_ii": [((e(0, 0) + e(1, 1)) / SQ2, 1.0)],
}


def main():
    delta, trials, n = 0.005, 400, 33
    out = {}
    for name, spec in CASES.items():
        A = np.array([s * v for v, s in spec]).T
        S = A @ A.T
        tg = np.einsum("ab,abij->ij", S, Mt)
        F = float(np.linalg.norm(tg))
        T = float(np.einsum("abij,ac,bd,cdij->", Mt, S, S, Mt))
        rng = np.random.default_rng(20260903 + len(name))
        mats = []
        for _ in range(trials):
            while True:
                g = rng.normal(size=(n, A.shape[1]))
                lab = (g @ A.T).reshape(n, 4, 4)
                if np.all(np.linalg.det(np.eye(4)[None] + delta * lab) > 0.05):
                    break
            tot = sum(optimal_internal_alignment(
                REF, geometric_self_dual_triple(np.eye(4) + delta * l)).aligned_candidate for l in lab)
            gr = plebanski_gram(tot)
            mats.append((gr - np.trace(gr) / 3.0 * np.eye(3)) / np.linalg.norm(gr))
        mats = np.asarray(mats)
        mean = mats.mean(axis=0)
        sd = mats.std(axis=0, ddof=1) / math.sqrt(trials)
        pred_mat = -(n - 1) / n * delta ** 2 * tg / (NS * 1.0)
        out[name] = {
            "F": F, "T": T,
            "floor_pred_over_delta2": (n - 1) / n * F / NS,
            "floor_obs_over_delta2": float(np.linalg.norm(mean)) / delta ** 2,
            "mean_matrix_se_over_delta2": float(np.linalg.norm(sd)) / delta ** 2,
            "direction_cosine_vs_pred": float(np.sum(mean * pred_mat) /
                                              (np.linalg.norm(mean) * np.linalg.norm(pred_mat)))
            if F > 1e-12 else None,
        }
        print(name, json.dumps(out[name]), flush=True)
    out["_meta"] = {"n": n, "trials": trials, "delta": delta, "seed_base": 20260903,
                    "note": "direct floor probe, off-grid n, not a preregistered statistic"}
    (OUT / "b9_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
