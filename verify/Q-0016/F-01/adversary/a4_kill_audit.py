"""Q-0016 F-01 adversary a4: kill executability, sampler correctness, window arithmetic,
tensor structure (kappa (x) I_16) and DISCRIMINATING POWER of K_A1-K_A3."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0016" / "F-01"))
from a1_algebra import A_matrix, C_matrix, D_f02, D_split, cbin  # noqa: E402
from driver_numbers import qspine_block  # noqa: E402
from check_split_modes import PREREGISTERED, WINDOWS, F02_ALTERNATIVE  # noqa: E402
from predict_split_kernel import split_labels  # noqa: E402

OUT = HERE / "a4_kill_audit.json"
R: dict = {}

# ---------------------------------------------------------------- (1) window arithmetic re-derivation
SE = {"qspine_split_slope_vs_En": ("abs", 0.0114), "qspine_split_ratio_b8_over_iid36": ("rel", 0.029),
      "binary_split_ratio_15": ("rel", 0.035), "binary_split_slope_7_63": ("abs", 0.0114),
      "cayley_split_slope": ("abs", 0.019), "cayley_split_ratio_8": ("rel", 0.014)}
wins = {}
for k, (kind, se) in SE.items():
    v = PREREGISTERED[k]
    half = max(3 * se, 0.05 * abs(v)) if kind == "rel" and False else (
        max(3 * se, 0.05 * abs(v)) if kind == "abs" else max(3 * se, 0.05) * abs(v))
    lo, hi = v - half, v + half
    alt = F02_ALTERNATIVE[k]
    wins[k] = {"prereg": v, "recomputed_window": [lo, hi], "card_window": list(WINDOWS[k]),
               "window_matches_card": abs(lo - WINDOWS[k][0]) < 6e-4 and abs(hi - WINDOWS[k][1]) < 6e-4,
               "f02_alternative": alt, "alt_outside_window": not (WINDOWS[k][0] <= alt <= WINDOWS[k][1]),
               "alt_margin_in_halfwidths": (abs(alt - v) / half) if half > 0 else None}
R["windows"] = wins
R["all_windows_match_card"] = all(w["window_matches_card"] for w in wins.values())
R["all_f02_alternatives_outside"] = all(w["alt_outside_window"] for w in wins.values())

# ---------------------------------------------------------------- (2) sampler correctness (card's own split_labels)
rng = np.random.default_rng(20260902)
samp = {}
for name, p in (("qspine_b5", qspine_block(5, np.random.default_rng(7))), ("cbin_d3", cbin(3)),
                ("star8", [-1] + [0] * 7), ("k2", [-1, 0, 0])):
    n = len(p)
    A = A_matrix(p)
    kap = A @ C_matrix(p) @ A.T
    draws = 60000
    X = np.stack([split_labels(p, rng.normal(size=n)) for _ in range(draws)])
    emp = X.T @ X / draws
    # sibling-sum-zero of the INCREMENTS (labels minus parent label)
    ch = [[] for _ in range(n)]
    for v, q in enumerate(p):
        if q >= 0:
            ch[q].append(v)
    lab = split_labels(p, rng.normal(size=n))
    inc = np.array([lab[v] - (lab[p[v]] if p[v] >= 0 else 0.0) for v in range(n)])
    sib = [abs(float(inc[kids].sum())) for kids in ch if len(kids) >= 2]
    mean_eq = [abs(float(lab[kids].mean() - lab[z])) for z, kids in enumerate(ch) if len(kids) >= 2]
    samp[name] = {"n": n, "max_abs_cov_err": float(np.max(np.abs(emp - kap))),
                  "max_abs_kappa_entry": float(np.max(np.abs(kap))), "draws": draws,
                  "max_sibling_increment_sum": max(sib) if sib else 0.0,
                  "max_children_mean_minus_parent": max(mean_eq) if mean_eq else 0.0}
R["sampler"] = samp

# tensor structure: does split_labels on (n,4,4) give kappa (x) I_16 ?
p = cbin(2)
n = len(p)
A = A_matrix(p)
kap = A @ C_matrix(p) @ A.T
draws = 40000
X = np.stack([split_labels(p, rng.normal(size=(n, 4, 4))) for _ in range(draws)]).reshape(draws, n, 16)
full = np.einsum("dva,dwb->vawb", X, X).reshape(n * 16, n * 16) / draws
target = np.kron(kap, np.eye(16))
R["tensor_structure"] = {"n": n, "draws": draws,
                         "max_abs_err_vs_kappa_kron_I16": float(np.max(np.abs(full - target))),
                         "max_abs_target": float(np.max(np.abs(target))),
                         "note": "split_labels applies the same linear map to each of the 16 components independently"}

# ---------------------------------------------------------------- (3) physics MC: does the kill risk anything?
from check_modes import DELTA, block_residual, rms, sample_iid  # noqa: E402

def sample_split(parent, rng, delta):
    n = len(parent)
    while True:
        v = block_residual(split_labels(parent, rng.normal(size=(n, 4, 4))), delta)
        if math.isfinite(v):
            return v

EPS_STAR = math.sqrt(10) * DELTA ** 2
t0 = time.time()
mini = {}
TRIALS = 192
rr = np.random.default_rng(20260902)
ri = np.random.default_rng(20260903)
for n in (7, 15):
    p = cbin(int(round(math.log2(n + 1))) - 1)
    tb = time.time()
    vals = [sample_split(p, rr, DELTA) for _ in range(TRIALS)]
    ivals = [sample_iid(n, ri, DELTA) for _ in range(TRIALS)]
    r_s, r_i = rms(vals), rms(ivals)
    pred_D = D_split(p)
    mini[str(n)] = {"trials": TRIALS, "rms_split": r_s, "rms_iid": r_i, "ratio_obs": r_s / r_i,
                    "ratio_pred_card": math.sqrt(pred_D / (n - 1)),
                    "rel_dev_ratio": r_s / r_i / math.sqrt(pred_D / (n - 1)) - 1,
                    "rms_over_epsstar_obs": r_s / EPS_STAR,
                    "rms_over_epsstar_pred": math.sqrt(pred_D) / n,
                    "rel_dev_amplitude": (r_s / EPS_STAR) / (math.sqrt(pred_D) / n) - 1,
                    "wall_s": time.time() - tb}
    print(n, mini[str(n)], flush=True)
R["physics_mc_mini_binary"] = mini
R["physics_mc_mini_wall_s"] = time.time() - t0
R["eps_star"] = EPS_STAR

# extrapolated cost of the real kills
per = {k: v["wall_s"] / (2 * TRIALS) for k, v in mini.items()}
R["cost_estimate"] = {
    "seconds_per_sample_n7": per["7"], "seconds_per_sample_n15": per["15"],
    "K_A2_binary_512trials_n_7_15_31_63_est_s": 512 * sum(
        per["15"] * (n / 15.0) for n in (7, 15, 31, 63)) * 2,
    "note": "rough linear-in-n extrapolation, split+iid arms",
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("windows match card:", R["all_windows_match_card"], "alts outside:", R["all_f02_alternatives_outside"])
print(json.dumps(R["windows"], indent=1, default=float))
print("sampler:", json.dumps(R["sampler"], indent=1, default=float))
print("tensor:", json.dumps(R["tensor_structure"], indent=1, default=float))
print("cost:", json.dumps(R["cost_estimate"], indent=1, default=float))
