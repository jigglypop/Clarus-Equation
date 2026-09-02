"""adversary b4: common-random-number delta sweep.
(1) relative O(delta) systematic of the master vs the real geometry (card allows 1% at delta=0.005),
(2) the delta^4 residual floor of KERNEL elements (prover parking counterexample),
(3) n-dependence of the observed/master ratio at fixed delta.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import (
    geometric_self_dual_triple, plebanski_gram)
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

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


CASES = {
    "univ_d_e03": [(e(0, 3), 1.0)],
    "kernel_sigma_b": [((e(0, 1) + e(2, 3)) / SQ2, 1.0), (e(0, 3), SQ2)],
    "antisym": [((e(0, 1) - e(1, 0)) / SQ2, 1.0)],
    "diag4": [(e(m, m), 1.0) for m in range(4)],
}


def factor(spec):
    return np.array([s * v for v, s in spec]).T


def F_T(spec):
    A = factor(spec)
    s = A @ A.T
    F = float(np.linalg.norm(np.einsum("ab,abij->ij", s, Mt)))
    T = float(np.einsum("abij,ac,bd,cdij->", Mt, s, s, Mt))
    return F, T


def master(n, F, T):
    return math.sqrt((n - 1) * ((n - 1) * F * F + 2.0 * T) / (12.0 * n * n))


def residual_matrix(labels, delta):
    tot = sum(optimal_internal_alignment(
        REF, geometric_self_dual_triple(np.eye(4) + delta * l)).aligned_candidate for l in labels)
    gram = plebanski_gram(tot)
    tl = gram - np.trace(gram) / 3.0 * np.eye(3)
    return tl / float(np.linalg.norm(gram))


def sweep(spec, n, trials, deltas, seed):
    A = factor(spec)
    r = A.shape[1]
    rng = np.random.default_rng(seed)
    dmax = max(deltas)
    samples = []
    while len(samples) < trials:
        g = rng.normal(size=(n, r))
        lab = (g @ A.T).reshape(n, 4, 4)
        ok = all(np.all(np.linalg.det(np.eye(4)[None] + d * lab) > MIN_DET) for d in deltas)
        if ok:
            samples.append(lab)
    rms, meanmat = {}, {}
    for d in deltas:
        mats = np.array([residual_matrix(lab, d) for lab in samples])
        norms = np.linalg.norm(mats.reshape(len(mats), 9), axis=1)
        rms[d] = float(np.sqrt(np.mean(norms ** 2)))
        meanmat[d] = float(np.linalg.norm(mats.mean(axis=0)))
    return rms, meanmat, len(samples)


def main():
    deltas = (0.005, 0.02, 0.05, 0.1, 0.2)
    trials, out = 300, {}
    t0 = time.time()
    for name, spec in CASES.items():
        F, T = F_T(spec)
        rec = {"F": F, "T": T, "floor_exact_over_delta2": F / NS}
        for n in (9, 33):
            rms, mm, _ = sweep(spec, n, trials, deltas, 20260903 + 7 * n + len(name))
            pred = master(n, F, T)
            rec["n%d" % n] = {
                "master": pred,
                "rms_over_delta2": {str(d): rms[d] / d ** 2 for d in deltas},
                "ratio_to_master": {str(d): ((rms[d] / d ** 2) / pred if pred > 0 else None) for d in deltas},
                "mean_matrix_norm_over_delta2": {str(d): mm[d] / d ** 2 for d in deltas},
                "mean_matrix_over_delta4": {str(d): mm[d] / d ** 4 for d in deltas},
                "mc_se_rel": 1.0 / math.sqrt(2.0 * trials),
            }
            print(name, n, json.dumps(rec["n%d" % n]["ratio_to_master"]),
                  json.dumps(rec["n%d" % n]["mean_matrix_over_delta4"]), flush=True)
        out[name] = rec
    out["_meta"] = {"deltas": list(deltas), "trials": trials, "seed_base": 20260903,
                    "common_random_numbers": True, "seconds": time.time() - t0,
                    "note": "off-grid n (9,33), different seed; not a preregistered statistic"}
    (OUT / "b4_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("seconds", time.time() - t0)


if __name__ == "__main__":
    main()
