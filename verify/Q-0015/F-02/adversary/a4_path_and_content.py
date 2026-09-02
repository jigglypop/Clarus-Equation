"""Adversary a4: (1) does epsilon (the Gram residual) enter theta's computation path?
(2) the 'inheritance shares the exponent / iid splits' claim as a normalisation artefact,
(3) independent replication of the already-observed face numbers with a foreign seed,
(4) cost of the pre-registered kill run.  The chain kill statistics (n = 16/32/64,
seed 20260903) are deliberately NOT run here: that is the card's out-of-sample test.
"""
from __future__ import annotations
import json, math, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import holonomy_pilot as HP  # noqa: E402

OUT = {}

# ---------------------------------------------------------------- (1) dynamic path trace
def boom(*a, **k):
    raise AssertionError("Gram machinery touched from the holonomy path")


saved = (HP.geometric_self_dual_triple, HP.simplicity_residual, HP.optimal_internal_alignment)
HP.geometric_self_dual_triple, HP.simplicity_residual, HP.optimal_internal_alignment = boom, boom, boom
rng = np.random.default_rng(999)
lab = np.cumsum(rng.standard_normal((5, 4, 4)), axis=0)
try:
    th = HP.holonomy_angle(lab, HP.DELTA)
    OUT["1_theta_computable_with_gram_disabled"] = {"ok": True, "theta": th}
except AssertionError as exc:
    OUT["1_theta_computable_with_gram_disabled"] = {"ok": False, "error": str(exc)}
try:
    HP.block_residual(lab, HP.DELTA)
    OUT["1_residual_uses_gram"] = False
except AssertionError:
    OUT["1_residual_uses_gram"] = True
HP.geometric_self_dual_triple, HP.simplicity_residual, HP.optimal_internal_alignment = saved

# joint (not merely marginal) behaviour: is theta a deterministic function of eps?
rng = np.random.default_rng(20260905)
th, ep = [], []
for _ in range(1500):
    L = np.cumsum(rng.standard_normal((3, 4, 4)), axis=0)
    th.append(HP.holonomy_angle(L, HP.DELTA))
    ep.append(HP.block_residual(L, HP.DELTA))
th, ep = np.asarray(th), np.asarray(ep)
ratio = th / ep
OUT["1_theta_over_eps_per_sample"] = {
    "mean": float(ratio.mean()),
    "std_over_mean": float(ratio.std() / ratio.mean()),
    "min": float(ratio.min()),
    "max": float(ratio.max()),
    "corr_theta2_eps2": float(np.corrcoef(th ** 2, ep ** 2)[0, 1]),
    "note": "F-01 had a fixed ratio to 4e-16; here the per-sample ratio has O(1) spread",
}

# ---------------------------------------------------------------- (2) normalisation artefact
def fit(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


sizes = (16, 32, 64)
theta_her = [math.sqrt(4.5 * (n - 1) * (n - 2) / 2) for n in sizes]
theta_iid = [math.sqrt(4.5 * n) for n in sizes]
D_her = [(n ** 2 - 1) * (2 * n ** 2 + 7) / 180 for n in sizes]
D_iid = [n - 1 for n in sizes]
eps_her = [math.sqrt(10 * D) / n for D, n in zip(D_her, sizes)]
eps_iid = [math.sqrt(10 * D) / n for D, n in zip(D_iid, sizes)]
OUT["2_exponents"] = {
    "card_pairing_theta_extensive_vs_eps_intensive": {
        "her": [fit(sizes, theta_her), fit(sizes, eps_her)],
        "iid": [fit(sizes, theta_iid), fit(sizes, eps_iid)],
        "card_story": "her: equal (1.054 vs 0.997); iid: split (+0.500 vs -0.482)",
    },
    "both_extensive_theta_vs_n_times_eps": {
        "her": [fit(sizes, theta_her), fit(sizes, [e * n for e, n in zip(eps_her, sizes)])],
        "iid": [fit(sizes, theta_iid), fit(sizes, [e * n for e, n in zip(eps_iid, sizes)])],
    },
    "both_intensive_theta_over_n_vs_eps": {
        "her": [fit(sizes, [t / n for t, n in zip(theta_her, sizes)]), fit(sizes, eps_her)],
        "iid": [fit(sizes, [t / n for t, n in zip(theta_iid, sizes)]), fit(sizes, eps_iid)],
    },
    "note": "under either like-for-like normalisation the card's story inverts: iid agrees, her splits",
}

# ---------------------------------------------------------------- (3) foreign-seed replication (already-observed only)
def face_stats(seed, trials, sampler):
    r = np.random.default_rng(seed)
    t, e = [], []
    for _ in range(trials):
        L = sampler(r)
        t.append(HP.holonomy_angle(L, HP.DELTA))
        e.append(HP.block_residual(L, HP.DELTA))
    t, e = np.asarray(t), np.asarray(e)
    return t, e


t_h, e_h = face_stats(777001, 6000, lambda r: np.cumsum(r.standard_normal((3, 4, 4)), axis=0))
t_i, e_i = face_stats(777002, 6000, lambda r: r.standard_normal((3, 4, 4)))
rmsf = lambda v: float(np.sqrt(np.mean(v ** 2)))
se = lambda v: float(np.std(v ** 2, ddof=1) / math.sqrt(len(v)) / (2 * math.sqrt(np.mean(v ** 2))))
OUT["3_face_replication_seed_777001"] = {
    "trials": 6000,
    "theta_her_over_delta2": rmsf(t_h) / HP.DELTA ** 2,
    "theta_her_se_rel": se(t_h) / rmsf(t_h),
    "theta_iid_over_delta2": rmsf(t_i) / HP.DELTA ** 2,
    "predicted_her_3_over_sqrt2": 3 / math.sqrt(2),
    "predicted_iid_sqrt13p5": math.sqrt(13.5),
    "rho_face_hol": rmsf(t_h) / rmsf(t_i),
    "rho_predicted": 1 / math.sqrt(3),
    "c_theta_face_her": rmsf(t_h) / rmsf(e_h),
    "c_theta_face_her_predicted": 27 * math.sqrt(2) / 20,
    "c_theta_face_iid": rmsf(t_i) / rmsf(e_i),
    "c_theta_face_iid_predicted": math.sqrt(243 / 40),
    "eps_her_over_delta2": rmsf(e_h) / HP.DELTA ** 2,
    "eps_her_predicted_10_over_9": 10 / 9,
    "pilot_values": {"rho": 0.5646202445828142, "c_her": 1.9037058512651062, "c_iid": 2.471693541518498},
}

# ---------------------------------------------------------------- (4) cost of the kill run
timings = {}
r = np.random.default_rng(5)
for n in (16, 64):
    L = np.cumsum(r.standard_normal((n, 4, 4)), axis=0)
    t0 = time.perf_counter()
    for _ in range(20):
        HP.holonomy_angle(L, HP.DELTA)
    t1 = time.perf_counter()
    for _ in range(20):
        HP.block_residual(L, HP.DELTA)
    t2 = time.perf_counter()
    timings[str(n)] = {"holonomy_s": (t1 - t0) / 20, "residual_s": (t2 - t1) / 20}
per_trial = {n: timings[str(n)]["holonomy_s"] + timings[str(n)]["residual_s"] for n in (16, 64)}
est32 = (per_trial[16] + per_trial[64]) / 2
OUT["4_timing"] = {
    "per_call": timings,
    "estimated_chain_mode_seconds": 512 * 2 * (per_trial[16] + est32 + per_trial[64]),
    "estimated_face_mode_seconds": None,
}
t0 = time.perf_counter()
for _ in range(50):
    L = np.cumsum(r.standard_normal((3, 4, 4)), axis=0)
    HP.holonomy_angle(L, HP.DELTA)
    HP.block_residual(L, HP.DELTA)
t1 = time.perf_counter()
OUT["4_timing"]["estimated_face_mode_seconds"] = (t1 - t0) / 50 * (2048 * 2 + 256 * 2)

print(json.dumps(OUT, indent=2))
pathlib.Path(__file__).with_name("a4_path_and_content.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
