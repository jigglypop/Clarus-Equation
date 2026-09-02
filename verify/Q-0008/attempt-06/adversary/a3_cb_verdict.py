"""a3: verdict on check C-b (2 of 12 physical cases outside the declared 5 percent window).

DECLARED BEFORE RUNNING (constants below are not edited after seeing output):

  (a) null distribution.  The per-trial statistic ||Phi||^2 depends on the kernel only through the
      non-zero eigenvalues mu_p of K = H kappa H:  Phi = sum_p mu_p Psi(y_p), y_p iid N(0, I_16).
      Rank-1 kernels (all coh cases, all n=2 cases) therefore share ONE distribution shape.
      Pools:  RANK1_POOL_REPS = 4000 replicates of 2000 trials (shared by the six rank-1 cases),
      GEN_POOL_REPS = 500 replicates of 2000 trials for each higher-rank case.
      Reported: empirical P(rel <= observed), P(|rel| > window), and the exact Poisson-binomial
      probability that at least 2 of the 12 cases land outside the window.

  (b) independent re-run.  Seed RERUN_SEED = 424242 (independent of 20260902), RERUN_TRIALS = 8000,
      cases iid n=2 and coh n=8 (the two that failed).  Expected relative SE at 8000 trials is
      CV/sqrt(8000) about 1.29 percent.  Decision rule, fixed now:
          abs(rel) < 2.6 percent (2 sigma)                     -> noise
          abs(rel) > 3.9 percent (3 sigma) with the same sign  -> systematic
          otherwise                                            -> undecided
      The 5 percent window of the original run is NOT changed and NOT re-applied.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "verify" / "Q-0012" / "F-01", ROOT / "verify" / "Q-0008" / "F-02"):
    sys.path.insert(0, str(p))

from check_cumulant import linear_map, quadratic_tensor, tl  # noqa: E402
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram  # noqa: E402
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
DELTA = 0.005
MIN_DET = 0.05
BLOCK = 2000
RANK1_POOL_REPS = 4000
GEN_POOL_REPS = 500
RERUN_SEED = 424242
RERUN_TRIALS = 8000
RERUN_NOISE = 0.026
RERUN_SYST = 0.039
T2 = 60.0
WINDOW = 0.05

REF = geometric_self_dual_triple(np.eye(4))
M = quadratic_tensor(linear_map())

_raw = [np.diag([1.0, -1.0, 0.0]), np.diag([1.0, 1.0, -2.0]),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])]
NB = np.array([m / np.linalg.norm(m) for m in _raw])
BM = np.einsum("abij,mij->mab", M, NB)


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


def generator(name, n):
    if name == "iid":
        return np.eye(n)
    if name == "chain":
        return np.tril(np.ones((n, n)))
    nB = n // 2
    A = np.zeros((n, 2))
    A[:nB, 0] = 1.0
    A[nB:, 1] = 1.0
    return A


def spectrum(name, n):
    A = generator(name, n)
    K = centering(n) @ A @ A.T @ centering(n)
    w = np.linalg.eigvalsh(K)
    return w[w > 1e-10 * max(1.0, w.max())]


def pool_values(mu, reps, rng, batch=25000):
    total = reps * BLOCK
    outv = np.empty(total)
    r = len(mu)
    done = 0
    while done < total:
        t = min(batch, total - done)
        y = rng.normal(size=(t, r, 16))
        W = np.einsum("tpa,tpb->tab", y * mu[None, :, None], y, optimize=True)
        phi = W.reshape(t, 256) @ BM.reshape(5, 256).T
        outv[done:done + t] = np.einsum("tm,tm->t", phi, phi)
        done += t
    return outv


result = {"script": "a3_cb_verdict", "declared": {
    "block": BLOCK, "rank1_pool_reps": RANK1_POOL_REPS, "gen_pool_reps": GEN_POOL_REPS,
    "rerun_seed": RERUN_SEED, "rerun_trials": RERUN_TRIALS,
    "rule_noise_below": RERUN_NOISE, "rule_systematic_above": RERUN_SYST, "window": WINDOW}}

obs = json.loads((HERE.parent / "result.json").read_text(encoding="utf-8"))["C_physical"]["cases"]

rng = np.random.default_rng(SEED + 1)
t0 = time.time()
cases = [(k, k.split("_n")[0], int(k.split("_n")[1])) for k in obs]
specs = {k: spectrum(nm, n) for k, nm, n in cases}
rank1 = [k for k in specs if len(specs[k]) == 1]
higher = [k for k in specs if len(specs[k]) > 1]

shared = pool_values(np.array([1.0]), RANK1_POOL_REPS, rng)
shared_wick = 2 * T2 * 1.0
shared_rel = shared.reshape(RANK1_POOL_REPS, BLOCK).mean(axis=1) / shared_wick - 1.0
print("rank-1 shared pool done", time.time() - t0, "pool mean rel", shared.mean() / shared_wick - 1.0, flush=True)

null = {}
for k in rank1:
    null[k] = shared_rel
for k in higher:
    mu = specs[k]
    v = pool_values(mu, GEN_POOL_REPS, rng)
    wick = 2 * T2 * float(np.sum(mu ** 2))
    null[k] = v.reshape(GEN_POOL_REPS, BLOCK).mean(axis=1) / wick - 1.0
    print("pool", k, "r =", len(mu), "done", round(time.time() - t0, 1), flush=True)

table = {}
p_out = []
for k, nm, n in cases:
    rel_obs = obs[k]["rel_err"]
    d = null[k]
    table[k] = {
        "rank": int(len(specs[k])), "rel_obs": rel_obs, "replicates": int(len(d)),
        "null_mean": float(d.mean()), "null_sd": float(d.std(ddof=1)),
        "null_skew": float(((d - d.mean()) ** 3).mean() / d.std() ** 3),
        "P_le_observed": float((d <= rel_obs).mean()),
        "P_ge_observed": float((d >= rel_obs).mean()),
        "P_outside_window": float((np.abs(d) > WINDOW).mean()),
        "P_below_minus_window": float((d < -WINDOW).mean()),
        "normal_theory_P_le_observed": float(0.5 * math.erfc(-(rel_obs - d.mean()) / (d.std(ddof=1) * math.sqrt(2)))),
    }
    p_out.append(table[k]["P_outside_window"])

dist = np.array([1.0])
for p in p_out:
    dist = np.convolve(dist, [1 - p, p])
result["null_distribution"] = {
    "cases": table,
    "P_at_least_1_of_12_outside": float(1 - dist[0]),
    "P_at_least_2_of_12_outside": float(1 - dist[0] - dist[1]),
    "P_at_least_3_of_12_outside": float(1 - dist[0] - dist[1] - dist[2]),
    "expected_number_outside": float(sum(p_out)),
}
print("P at least 2 of 12 outside window =", result["null_distribution"]["P_at_least_2_of_12_outside"], flush=True)


def block_sum(labels, delta):
    Y = np.zeros_like(REF)
    for lab in labels:
        tet = np.eye(4) + delta * lab
        if float(np.linalg.det(tet)) <= MIN_DET:
            return None
        Y = Y + optimal_internal_alignment(REF, geometric_self_dual_triple(tet)).aligned_candidate
    return Y


rerun = {}
rng2 = np.random.default_rng(RERUN_SEED)
for name, n in (("iid", 2), ("coh", 8)):
    A = generator(name, n)
    H = centering(n)
    K = H @ A @ A.T @ H
    D = float(np.sum(K * K))
    wick = n * n * DELTA ** 4 * 2 * T2 * D
    law_eps2 = 10.0 * DELTA ** 4 * D / (n * n)
    vals, eps2, sur = [], [], []
    rej = 0
    t1 = time.time()
    while len(vals) < RERUN_TRIALS:
        z = rng2.normal(size=(A.shape[1], 4, 4))
        labels = np.einsum("vu,uab->vab", A, z)
        Y = block_sum(labels, DELTA)
        if Y is None:
            rej += 1
            continue
        g = plebanski_gram(Y)
        t = tl(g)
        num = float(np.sum(t * t))
        vals.append(num)
        eps2.append(num / float(np.sum(g * g)))
        xt = H @ labels.reshape(n, 16)
        W = np.einsum("va,vb->ab", xt, xt)
        phi = W.reshape(256) @ BM.reshape(5, 256).T
        sur.append(n * n * DELTA ** 4 * float(np.dot(phi, phi)))
    vals, eps2, sur = np.array(vals), np.array(eps2), np.array(sur)
    se = float(vals.std(ddof=1) / math.sqrt(RERUN_TRIALS))
    rel = float(vals.mean() / wick - 1.0)
    key = name + "_n" + str(n)
    rerun[key] = {
        "trials": RERUN_TRIALS, "seed": RERUN_SEED, "D": D, "rejections": rej,
        "wick": wick, "mc_mean": float(vals.mean()), "mc_se": se, "rel_err": rel,
        "z": rel * wick / se, "rel_se": se / wick,
        "law_E_eps2": law_eps2, "mc_E_eps2": float(eps2.mean()),
        "rel_err_eps2": float(eps2.mean() / law_eps2 - 1.0),
        "ratio_of_means_phys_over_surrogate": float(vals.mean() / sur.mean()),
        "crn_mean_ratio": float((vals / sur).mean()),
        "original_rel_err": obs[key]["rel_err"],
        "seconds": time.time() - t1,
    }
    print("rerun", name, n, "rel", rel, "z", rerun[key]["z"], flush=True)

result["rerun"] = rerun
verdicts = {}
for k, v in rerun.items():
    a = abs(v["rel_err"])
    if a < RERUN_NOISE:
        verdicts[k] = "noise"
    elif a > RERUN_SYST and v["rel_err"] * v["original_rel_err"] > 0:
        verdicts[k] = "systematic"
    else:
        verdicts[k] = "undecided"
result["rerun_verdict"] = verdicts
result["conclusion"] = "noise" if set(verdicts.values()) == {"noise"} else "/".join(sorted(set(verdicts.values())))
result["wall_seconds"] = time.time() - t0

(HERE / "a3_cb_verdict.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "null_distribution"}, ensure_ascii=False, indent=1, default=float))
