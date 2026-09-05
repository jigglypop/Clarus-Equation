"""adversary c2 (re-audit):
(1) delta^4 arithmetic of the scope sentence (1.7e-4 vs 1.7e-5) recomputed from b4_report;
(2) the NEW step-4 sentence: tl gram(Y) = -n delta^2 sum_v tl M(xi~_v, xi~_v) + O(delta^3),
    claimed to hold "for an arbitrary per-cell alignment map (not specifically polar)";
(3) cell Gram = 2 det(e_v) I (the stated reason);
(4) floor_hat() at the preregistered grid (4,64) with EXACT eps values (no MC).
No preregistered-size run.
"""
from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

spec = importlib.util.spec_from_file_location("cf", ROOT / "verify/Q-0013/F-02/check_floor.py")
cf = importlib.util.module_from_spec(spec); sys.modules["cf"] = cf; spec.loader.exec_module(cf)

Mt = np.load(OUT / "b1_Mt.npy")            # tl M^{ab}_{ij}, independent reconstruction (b1)
REF = geometric_self_dual_triple(np.eye(4))
rep = {}

# ---------------------------------------------------------------- (1) delta^4 arithmetic
b4 = json.loads((OUT / "b4_report.json").read_text(encoding="utf-8"))
anti = b4["antisym"]
mm4 = {n: anti[n]["mean_matrix_over_delta4"] for n in ("n9", "n33")}
d = 0.005
floor_units_delta2 = {n: mm4[n]["0.005"] * d**4 / d**2 for n in mm4}   # residual in the card's units
half = 0.00775                                                          # K1 window half width
rep["delta4"] = {
    "mean_over_delta4": {n: mm4[n]["0.005"] for n in mm4},
    "constant_in_delta": {n: [mm4[n][k] for k in mm4[n]] for n in mm4},
    "residual_in_delta2_units_at_delta0.005": floor_units_delta2,
    "ratio_to_K1_half_width": {n: floor_units_delta2[n] / half for n in floor_units_delta2},
    "card_says": "1.3e-6 / 0.00775 = 1.7e-4",
    "prior_SUMMARY_said": "1.7e-5",
    "quadrature_effect_on_K1_value": {
        n: (math.hypot(0.07733980, floor_units_delta2[n]) - 0.07733980) / half for n in floor_units_delta2},
}

# ---------------------------------------------------------------- (2)(3) step-4 alignment claim
def triple_of(label, delta):
    return geometric_self_dual_triple(np.eye(4) + delta * label)

def rot_apply(R, tri):
    return np.einsum("ij,ja->ia", R, tri)

def tl(m):
    return m - np.trace(m) / 3.0 * np.eye(3)

def rand_rot(rng, scale=None):
    A = rng.normal(size=(3, 3)); A = A - A.T
    if scale is not None:
        A = A * scale / max(np.linalg.norm(A), 1e-30)
    w, V = np.linalg.eigh(1j * A)
    R = (V @ np.diag(np.exp(-1j * w)) @ V.conj().T).real
    return R

rng = np.random.default_rng(20260903)
n = 3
sig_spec = cf.SPECS["univ_o"]
A = cf.factor(sig_spec)
g = rng.normal(size=(n, A.shape[1]))
labels = (g @ A.T).reshape(n, 4, 4)
xi = labels.reshape(n, 16)
xit = xi - xi.mean(axis=0, keepdims=True)                 # centered
rhs_core = -np.einsum("va,vb,abij->ij", xit, xit, Mt)     # sum_v tl M(xi~,xi~) with the card's minus sign

modes = {}
for delta in (0.02, 0.005):
    tris = [triple_of(l, delta) for l in labels]
    # (a) polar alignment (the pipeline)
    Ya = sum(optimal_internal_alignment(REF, t).aligned_candidate for t in tris)
    # (b) polar alignment composed with an extra O(delta) rotation per cell
    Yb = sum(rot_apply(rand_rot(rng, scale=delta * 3.0),
                       optimal_internal_alignment(REF, t).aligned_candidate) for t in tris)
    # (c) an arbitrary O(1) rotation per cell ("임의의 cell 별 정렬 사상"의 문자 그대로 읽기)
    Yc = sum(rot_apply(rand_rot(rng), optimal_internal_alignment(REF, t).aligned_candidate) for t in tris)
    # (d) no alignment at all
    Yd = sum(tris)
    row = {}
    for tag, Y in (("polar", Ya), ("polar+O(delta)rot", Yb), ("arbitrary O(1) rot", Yc), ("none", Yd)):
        G = plebanski_gram(Y)
        lhs = tl(G)
        rhs = n * delta**2 * rhs_core
        row[tag] = {
            "||tl gram||": float(np.linalg.norm(lhs)),
            "||rhs||": float(np.linalg.norm(rhs)),
            "rel_err": float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-300)),
            "||tl gram||/(n delta^2)": float(np.linalg.norm(lhs) / (n * delta**2)),
        }
    modes[str(delta)] = row
rep["step4_alignment"] = modes

# cell Gram = 2 det(e) I ?
cg = []
for delta in (0.05, 0.005):
    for l in labels:
        e = np.eye(4) + delta * l
        G1 = plebanski_gram(triple_of(l, delta))
        cg.append({"delta": delta, "rel_dev_from_2detE_I":
                   float(np.linalg.norm(G1 - 2.0 * np.linalg.det(e) * np.eye(3)) /
                         np.linalg.norm(G1))})
rep["cell_gram_isotropy"] = cg

# ---------------------------------------------------------------- (4) floor_hat at the prereg grid
def exact_eps(nn, F, T):
    return math.sqrt((nn - 1) * ((nn - 1) * F * F + 2.0 * T) / (12.0 * nn * nn))

checks = {}
for name, F, T in (("univ_o", math.sqrt(1 / 6), 11 / 3), ("univ_d", math.sqrt(1 / 6), 1 / 6),
                   ("kernel", 0.0, 7 / 3), ("diag4", 0.0, 8.0)):
    fh = cf.floor_hat(exact_eps(4, F, T), exact_eps(64, F, T))
    checks[name] = {"floor_hat_at_(4,64)": fh, "exact_floor F/(2sqrt3)": F / (2 * math.sqrt(3)),
                    "abs_err": abs(fh - F / (2 * math.sqrt(3)))}
    fh8 = cf.floor_hat(exact_eps(4, F, T), exact_eps(8, F, T))
    checks[name]["floor_hat_at_(4,8)_smoke_misuse"] = fh8
rep["floor_hat"] = checks

(OUT / "c2_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rep, ensure_ascii=False, indent=2))
