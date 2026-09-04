"""Adversary a7: re-audit of Q-0018 F-01 revision 2.

(1) prereg diff rev1 vs rev2: predicts value/uncertainty, kill windows, seeds/trials/delta.
(2) new verify[26]-[30] recomputed independently.
(3) new step 4 irreducible weights (w1,w9,w6) for shear / gaussian / angle family / uniaxial,
    both per realisation and from the second-moment tensor, plus window discrimination.
(4) gauge invariance of c_Delta/c_2 under (scale of P9 part, arbitrary gauge content).
"""
from __future__ import annotations
import json, math, re, sys
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCRATCH = Path(r"C:/Users/dongh/AppData/Local/Temp/claude/c--dev-ce-Clarus-Equation/6fce6085-2bf0-4802-bf89-9fc3591366be/scratchpad")
M = np.load(HERE / "M_tensor_adversary.npy")
out = {"script": "a7_reaudit"}

rev1 = (SCRATCH / "rev1.md").read_text(encoding="utf-8")
rev2 = (ROOT / "derivations" / "Q-0018" / "F-01.formula.md").read_text(encoding="utf-8")


def prereg_numbers(txt):
    vals = re.findall(r"^\s{4}value:\s*(\S+)\s*$", txt, re.M)
    uncs = re.findall(r"^\s{4}uncertainty:\s*(\S+)\s*$", txt, re.M)
    kills = txt.split("kill:")[1].split("consistency_checks:")[0]
    wins = re.findall(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", kills)
    seeds = sorted(set(re.findall(r"seed[s]?\s*(\d{8})", txt)))
    trials = sorted(set(re.findall(r"(\d+)\s*trials", txt)))
    delta = sorted(set(re.findall(r"δ=([\d.]+)", txt)))
    return {"values": vals, "uncertainties": uncs, "kill_windows": wins,
            "seeds": seeds, "trials": trials, "deltas": delta}


p1, p2 = prereg_numbers(rev1), prereg_numbers(rev2)
out["prereg_diff"] = {
    "rev1": p1, "rev2": p2,
    "values_added": sorted(set(p2["values"]) - set(p1["values"])),
    "values_removed": sorted(set(p1["values"]) - set(p2["values"])),
    "uncertainties_added": sorted(set(p2["uncertainties"]) - set(p1["uncertainties"])),
    "kill_windows_identical": p1["kill_windows"] == p2["kill_windows"],
    "seeds_identical": p1["seeds"] == p2["seeds"],
    "trials_identical": p1["trials"] == p2["trials"],
    "deltas_rev1": p1["deltas"], "deltas_rev2": p2["deltas"],
}
d1 = [x for x in rev1.splitlines() if "P4 c_coh_16" in x][0]
d2 = [x for x in rev2.splitlines() if "P4 c_coh_16" in x][0]
out["P4_text"] = {"rev1": d1, "rev2": d2,
                  "denominator_rev1": re.findall(r"[0-9]+", d1.split("mean")[1])[-1],
                  "denominator_rev2": re.findall(r"[0-9]+", d2.split("mean")[1].split("(D_coh")[0])[-1]}
out["P7b_window_was_preregistered_in_rev1_note"] = ("[4.8068,5.3128]" in rev1.replace(" ", ""))
out["P7b_value_matches_rev1_note"] = abs(5.0598290598 - 592 / 117) < 1e-9
out["P7b_uncertainty_is_5pct"] = abs(0.2529914530 / 5.0598290598 - 0.05) < 1e-8

v = {}
v["26"] = (32 / 9) / (640 / 81) - 9 / 20
v["27"] = (1 / 6) / (2 * 5 / 27) - 9 / 20
v["28_full"] = 10 * (16 / 9) ** 2 - 4 * 640 / 81 + (4 * 32 / 9) / (10 * (16 / 9) ** 2) - 9 / 20
v["28_part_a_c2prime_minus_4c2"] = 10 * (16 / 9) ** 2 - 4 * 640 / 81
v["28_part_b_ratio_minus_9_20"] = (4 * 32 / 9) / (10 * (16 / 9) ** 2) - 9 / 20
v["29"] = 10 * (1 - 2 * 15 / (60 * 16)) - 9.6875
v["30"] = (32 / 9) / 4 - 8 / 9
v["exact_26_Fraction"] = str(Fraction(32, 9) / Fraction(640, 81) - Fraction(9, 20))
out["verify_26_to_30"] = {k: float(x) if not isinstance(x, str) else x for k, x in v.items()}


def irreps(X):
    """(w1, w9, w6) = squared-norm shares of the trace / symmetric-traceless / antisymmetric parts."""
    X = np.asarray(X, float)
    tr = np.trace(X) / 4 * np.eye(4)
    sym = (X + X.T) / 2
    st = sym - tr
    anti = (X - X.T) / 2
    tot = float(np.sum(X * X))
    return (float(np.sum(tr * tr)) / tot, float(np.sum(st * st)) / tot,
            float(np.sum(anti * anti)) / tot)


rng = np.random.default_rng(20260904)
n0 = np.array([1.0, 0, 0, 0])
m0 = np.array([0.0, 1, 0, 0])
w_shear = irreps(4 * np.outer(n0, m0))
g = rng.normal(size=(200000, 4, 4))
sh = np.array([irreps(x) for x in g[:2000]])
w_gauss_mc = sh.mean(axis=0)
w_uni = irreps(np.outer(n0, n0))
ang = {}
for deg in (90, 60, 45, 30, 0):
    th = math.radians(deg)
    m = math.cos(th) * n0 + math.sin(th) * m0
    w = irreps(4 * np.outer(n0, m))
    ang[str(deg)] = {"measured": list(w),
                     "card_closed_form_w1_w6": [math.cos(th) ** 2 / 4, math.sin(th) ** 2 / 2],
                     "w1_ok": abs(w[0] - math.cos(th) ** 2 / 4) < 1e-12,
                     "w6_ok": abs(w[2] - math.sin(th) ** 2 / 2) < 1e-12}
# second-moment tensor route (Haar average) must agree with the per-realisation value
gg = rng.normal(size=(200000, 4, 2))
nn = gg[:, :, 0] / np.linalg.norm(gg[:, :, 0], axis=1, keepdims=True)
mm = gg[:, :, 1] - np.sum(gg[:, :, 1] * nn, axis=1, keepdims=True) * nn
mm = mm / np.linalg.norm(mm, axis=1, keepdims=True)
Xi = 4 * np.einsum("ti,tk->tik", nn, mm)
w_haar = np.array([irreps(x) for x in Xi[:5000]]).mean(axis=0)
WIN = 0.05
out["step4_weights"] = {
    "definition_checked": "w_r = ||P_r(Xi)||_F^2 / ||Xi||_F^2 with R^{4x4} = 1 (trace) + 9 (sym traceless) + 6 (antisym)",
    "shear_axiom_per_realisation": list(w_shear),
    "shear_haar_average": list(w_haar),
    "card_shear": [0.0, 0.5, 0.5],
    "shear_ok": bool(max(abs(a - b) for a, b in zip(w_shear, [0, .5, .5])) < 1e-12),
    "gaussian_mc": list(w_gauss_mc), "card_gaussian": [1 / 16, 9 / 16, 6 / 16],
    "gaussian_ok": bool(max(abs(a - b) for a, b in zip(w_gauss_mc, [1 / 16, 9 / 16, 6 / 16])) < 0.01),
    "uniaxial": list(w_uni), "card_uniaxial": [0.25, 0.75, 0.0],
    "uniaxial_ok": bool(max(abs(a - b) for a, b in zip(w_uni, [.25, .75, 0])) < 1e-12),
    "angle_family": ang,
    "window_pm": WIN,
    "gaussian_excluded_by_w1_window": bool(abs(1 / 16 - 0.0) > WIN) or bool(abs(9 / 16 - 0.5) > WIN),
    "gaussian_w1_margin": 1 / 16 - WIN,
    "gaussian_w9_margin": 9 / 16 - 0.5,
    "uniaxial_excluded": bool(abs(0.25 - 0.0) > WIN),
    "angle_theta_excluded_beyond_deg": [d for d in range(0, 91)
                                        if abs(math.cos(math.radians(d)) ** 2 / 4) > WIN],
    "note_w9_alone_is_weak": "|w9(gauss) - w9(shear)| = 1/16 = 0.0625 is only 1.25x the +-0.05 window; "
                             "w1 (0 vs 1/16) is equally marginal. Only w6 (1/2 vs 3/8) and the size-CV "
                             "and det conditions separate the gaussian branch comfortably.",
}


def M_of(X, Y):
    return np.einsum("a,b,abij->ij", np.asarray(X, float).reshape(16),
                     np.asarray(Y, float).reshape(16), M)


def constants_of(sample):
    C = np.einsum("ta,tb->ab", sample.reshape(len(sample), 16), sample.reshape(len(sample), 16)) / len(sample)
    T2C = float(np.einsum("ac,bd,abij,cdij->", C, C, M, M))
    Q4 = float(np.mean([np.sum(M_of(x, x) ** 2) for x in sample[:400]]))
    c2 = 2 * T2C / 12
    cD = Q4 / 12
    return c2, cD - c2, cD, cD / c2


S = 4000
base = Xi[:S]
gauge_fam = {}
for name, f in (("card_shear", lambda x: x),
                ("gauge_stripped_renorm", lambda x: math.sqrt(2) * ((x + np.swapaxes(x, 1, 2)) / 2)),
                ("P9_scaled_1.5_gauge_x3", lambda x: 1.5 * (x + np.swapaxes(x, 1, 2)) / 2
                 + 3.0 * (x - np.swapaxes(x, 1, 2)) / 2),
                ("P9_same_gauge_zero", lambda x: (x + np.swapaxes(x, 1, 2)) / 2)):
    c2_, c4_, cD_, r_ = constants_of(f(base))
    gauge_fam[name] = {"c2": c2_, "c4": c4_, "c_delta": cD_, "ratio_cD_over_c2": r_,
                       "ratio_minus_9_20": r_ - 0.45}
out["gauge_invariance_9_20"] = {"family": gauge_fam,
                                "claim": "c_Delta/c_2 = Q4/(2 T2(C)) = 9/20 invariant under P9-scaling and arbitrary gauge content",
                                "max_abs_dev": max(abs(v["ratio_minus_9_20"]) for v in gauge_fam.values())}
(HERE / "a7_reaudit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=True))
