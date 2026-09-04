"""Adversary a8: re-audit of Q-0015 F-02 revision 2.

1  K1-K5 constants / windows / seed / trials identical to revision 1.
2  K6 window arithmetic: centres = sqrt(7/3) x frame values; disjointness from K1 / K4.
3  The new justification for the frame convention: same metric, different internal frame
   (E_v = Lambda_v E_0) must be exactly flat.  revision 2 cites a5 plus the 2.80 of a6, but
   that 2.80 was the O(delta^2) generator norm for antisymmetric labels, not this exact
   gauge-orbit test, so the claim is retested directly here (both conventions, both classes).
4  New verify entries 19 / 20 / 21.
5  Card text: line-level compare of the load-bearing lines and ASCII marker counts.
"""
from __future__ import annotations
import importlib.util, json, math, pathlib, sys
import numpy as np

NL = chr(10)
QT = chr(34)
HERE = pathlib.Path(__file__).resolve().parent
CARD2 = HERE.parents[3] / "derivations" / "Q-0015" / "F-02.formula.md"
SCRATCH = pathlib.Path(
    "C:/Users/dongh/AppData/Local/Temp/claude/c--dev-ce-Clarus-Equation/"
    "6fce6085-2bf0-4802-bf89-9fc3591366be/scratchpad"
)
OUT = {}


def load(path, name):
    sys.path.insert(0, str(HERE.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rev1 = load(SCRATCH / "check_holonomy_rev1.py", "chk_rev1")
rev2 = load(HERE.parent / "check_holonomy.py", "chk_rev2")

frozen = {}
for attr in ("SEED_KILL", "SIZES", "TRIALS", "FACE_TRIALS", "DELTA_SMALL", "DELTA_TRIALS", "DELTA",
             "PREREGISTERED", "WINDOWS"):
    a, b = getattr(rev1, attr), getattr(rev2, attr)
    frozen[attr] = {"identical": a == b, "value": b}
OUT["1_frozen_constants"] = frozen
OUT["1_all_identical"] = all(v["identical"] for v in frozen.values())

synthetic = dict(rev1.PREREGISTERED)
v1, v2 = rev1.verdicts(dict(synthetic)), rev2.verdicts(dict(synthetic))
OUT["1_verdicts_K1_K5_identical"] = all(v1[k] == v2[k] for k in v1)
OUT["1_rev2_extra_keys"] = sorted(set(v2) - set(v1))
OUT["1_rev1_keys_lost"] = sorted(set(v1) - set(v2))
OUT["1_K6_triggered_on_frame_theory_values"] = v2["K6_coord_convention"]["triggered"]
coord_stats = dict(rev2.K6_COORD_CONVENTION)
vc = rev2.verdicts(coord_stats)
OUT["1_K6_triggered_on_coord_values"] = vc["K6_coord_convention"]["triggered"]
OUT["1_K1_K4_inside_on_coord_values"] = {k: vc[k]["inside"] for k in coord_stats if k in vc}

s1 = (SCRATCH / "check_holonomy_rev1.py").read_text(encoding="utf-8")
s2 = (HERE.parent / "check_holonomy.py").read_text(encoding="utf-8")
sep = NL * 3
funcs = {}
for fn in ("def mode_chain", "def mode_face", "def selftest", "def main"):
    i1, i2 = s1.index(fn), s2.index(fn)
    j1 = s1.index(sep, i1) if sep in s1[i1:] else len(s1)
    j2 = s2.index(sep, i2) if sep in s2[i2:] else len(s2)
    funcs[fn] = s1[i1:j1] == s2[i2:j2]
OUT["1_function_bodies_identical"] = funcs

R = math.sqrt(7 / 3)
k6 = {}
for k, v in rev2.K6_COORD_CONVENTION.items():
    frame = rev2.PREREGISTERED[k]
    lo, hi = rev2.K6_WINDOWS[k]
    flo, fhi = rev2.WINDOWS[k]
    k6[k] = {
        "frame": frame, "coord_declared": v, "frame_x_sqrt7_3": frame * R,
        "abs_gap": abs(v - frame * R), "k6_window": [lo, hi],
        "centre_over_declared": (lo + hi) / 2 / v, "frame_window": [flo, fhi],
        "disjoint": bool(hi < flo or lo > fhi), "gap_to_frame_window": lo - fhi,
        "half_width_rel": (hi - lo) / 2 / v,
    }
OUT["2_k6"] = k6
OUT["2_centres_match_sqrt7_3"] = all(x["abs_gap"] < 5e-7 for x in k6.values())
OUT["2_all_disjoint"] = all(x["disjoint"] for x in k6.values())
OUT["2_sqrt7_3"] = R

asym = lambda M: 0.5 * (M - M.T)


def polar(T):
    U, _, Vt = np.linalg.svd(T)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    return Rm


def theta(Rm):
    a = np.angle(np.linalg.eigvals(Rm))
    return math.sqrt(0.5 * float(np.sum(a * a)))


def hol(E, coord=False):
    H = np.eye(4)
    k = len(E)
    for i in range(k):
        u, v = E[i], E[(i + 1) % k]
        H = (polar(np.linalg.inv(u) @ v) if coord else polar(v @ np.linalg.inv(u))) @ H
    return H


rng = np.random.default_rng(20260903)


def rand_so4(r):
    q, rr = np.linalg.qr(r.standard_normal((4, 4)))
    q = q @ np.diag(np.sign(np.diag(rr)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


E0 = np.eye(4) + 0.35 * rng.standard_normal((4, 4))
lam = [rand_so4(rng) for _ in range(6)]
left = [L @ E0 for L in lam]
right = [E0 @ L for L in lam]
left_scaled = [rng.uniform(0.4, 2.5) * (L @ E0) for L in lam]
g0 = E0.T @ E0
OUT["3_gauge_orbit"] = {
    "max_metric_deviation_left_class": max(float(np.linalg.norm(x.T @ x - g0)) for x in left),
    "left_class_frame_theta": theta(hol(left)),
    "left_class_coord_theta": theta(hol(left, coord=True)),
    "left_scaled_frame_theta": theta(hol(left_scaled)),
    "left_scaled_coord_theta": theta(hol(left_scaled, coord=True)),
    "dual_right_class_frame_theta": theta(hol(right)),
    "dual_right_class_coord_theta": theta(hol(right, coord=True)),
}
reps = []
for _ in range(40):
    B = np.eye(4) + 0.35 * rng.standard_normal((4, 4))
    cfg = [rand_so4(rng) @ B for _ in range(5)]
    reps.append((theta(hol(cfg)), theta(hol(cfg, coord=True))))
reps = np.asarray(reps)
OUT["3_gauge_orbit_40_configs"] = {
    "frame_max_theta": float(reps[:, 0].max()),
    "coord_min_theta": float(reps[:, 1].min()),
    "coord_median_theta": float(np.median(reps[:, 1])),
}
Es = [np.eye(4) + 0.005 * asym(m) for m in rng.standard_normal((5, 4, 4))]
OUT["3_rotation_labels_delta_0p005"] = {
    "frame_theta": theta(hol(Es)),
    "coord_theta": theta(hol(Es, coord=True)),
}

OUT["4_verify_new"] = {
    "v19_coord_constant": (1 / 2) * ((1 / 4) * 36 + 12),
    "v19_expected": 10.5,
    "v19_ratio_gap": math.sqrt((21 / 2) / (9 / 2)) - math.sqrt(7 / 3),
    "v20_face_her_exact": 27 * math.sqrt(2) / 20 * R,
    "v20_card_number": 2.9163333,
    "v20_chain16_exact": 4.0560507 * R,
    "v20_card_number_chain16": 6.1957197,
    "v20_script_number_chain16": rev2.K6_COORD_CONVENTION["c_theta_her_16"],
    "v21_terms": [(10 / 9) / 2 - 5 / 9, 1 / 3 - 5 / 9 + 2 / 9,
                  (0.9967340 + 1) - 1.9967340, (-0.4824027 + 1) - 0.5175973],
    "v21_term2_is_tautology": abs(1 / 3 + 2 / 9 - 5 / 9) < 1e-15,
}

c1 = (SCRATCH / "F-02_rev1.md").read_text(encoding="utf-8").splitlines()
c2 = CARD2.read_text(encoding="utf-8").splitlines()
t1, t2 = NL.join(c1), NL.join(c2)


def line_with(lines, anchor):
    hits = [x for x in lines if anchor in x]
    return hits[0] if hits else None


box1, box2 = line_with(c1, "operatorname{asym}"), line_with(c2, "operatorname{asym}")
OUT["5_boxed_line"] = {
    "changed": box1 != box2,
    "rev1_len": len(box1) if box1 else None,
    "rev2_len": len(box2) if box2 else None,
    "rev2_has_0p28": "0.28" in (box2 or ""),
    "rev2_has_delta2": "delta^2" in (box2 or ""),
}
markers = {}
for m in ("K6", "P7", "already_observed: true", "already_observed: false", "[0.9996, 1.0014]",
          "0.28", "21/2", "(7/3)", "2.9163333", "[2.683, 3.15]", "3.54", "0.273",
          "- type:", "revision: 2", "extensive", "intensive"):
    markers[m] = [t1.count(m), t2.count(m)]
OUT["5_marker_counts_rev1_rev2"] = markers
win_texts = ["[3.73, 4.38]", "[3.94, 4.63]", "[4.04, 4.74]", "[0.95, 1.15]", "[0.40, 0.60]",
             "[0.540, 0.615]", "[1.76, 2.06]", "[2.27, 2.66]", "[0.235, 0.265]", "seed 20260903",
             "1.9091883092", "0.5773502692", "2.4647515088", "4.0560507", "1.0543077", "0.5000000"]
OUT["5_K1_K5_text_preserved"] = {w: bool(w in t1 and w in t2) for w in win_texts}
OUT["5_scope_item_count"] = {"rev1": sum(1 for x in c1 if x.startswith("  - " + QT)),
                             "rev2": sum(1 for x in c2 if x.startswith("  - " + QT))}
OUT["5_predicts_count"] = {"rev1": t1.count("- observable:"), "rev2": t2.count("- observable:")}
OUT["5_kill_count"] = {"rev1": t1.count("  - " + QT + "K"), "rev2": t2.count("  - " + QT + "K")}

print(json.dumps(OUT, indent=2, default=str))
pathlib.Path(__file__).with_name("a8_reaudit.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
