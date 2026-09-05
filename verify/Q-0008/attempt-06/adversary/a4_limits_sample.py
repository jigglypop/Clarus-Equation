"""a4: limits, independent closed forms, structure constants, condition (C), random_sample_20.

DECLARED BEFORE RUNNING (seed 20260902):
  L1 n=1 single cell:            eps <= 1e-12 at delta in {0.005, 0.3}
  L2 all cells same label:       eps <= 1e-12 at delta in {0.005, 0.3}, n in {2,5,16}
  L3 delta -> 0:                 RMS/delta^2 -> eps_star sqrt(D)/n, common random numbers
  L4 kappa = I:                  eps_bar = eps_star sqrt(n-1)/n
  L5 two species p = 1/2:        eps_bar = eps_star/2, independent of n
  S20 random_sample_20:          20 random (kernel, n, delta), SAMPLE_TRIALS = 2000 each,
                                 report z = rel/rel_se; declared criterion max|z| <= 3.5 and
                                 inverse-variance combined |z| <= 3.0
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "verify" / "Q-0012" / "F-01", ROOT / "verify" / "Q-0008" / "F-02"):
    sys.path.insert(0, str(p))

from check_cumulant import gram_form, linear_map, quadratic_tensor, tl  # noqa: E402
from driver_numbers import cayley_exact, tree_arrays, uniform_rooted_tree  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram  # noqa: E402
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
MIN_DET = 0.05
SAMPLE_TRIALS = 2000
SAMPLE_N = 20
TOL_EXACT = 1e-12
MAX_Z = 3.5
MAX_Z_COMB = 3.0
T2 = 60.0
EPS_STAR_SQ = 10.0

REF = geometric_self_dual_triple(np.eye(4))
G0 = plebanski_gram(REF)
out = {"script": "a4_limits_sample", "seed": SEED,
       "declared": {"sample_trials": SAMPLE_TRIALS, "sample_n": SAMPLE_N, "max_z": MAX_Z,
                    "max_z_combined": MAX_Z_COMB, "tol_exact": TOL_EXACT}}


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


def aligned(lab, delta):
    tet = np.eye(4) + delta * lab
    if float(np.linalg.det(tet)) <= MIN_DET:
        return None
    return optimal_internal_alignment(REF, geometric_self_dual_triple(tet)).aligned_candidate


def eps_of(labels, delta):
    Y = np.zeros_like(REF)
    for lab in labels:
        a = aligned(lab, delta)
        if a is None:
            return None
        Y = Y + a
    g = plebanski_gram(Y)
    t = tl(g)
    return math.sqrt(float(np.sum(t * t)) / float(np.sum(g * g)))


def anc_matrix(parent):
    n = len(parent)
    A = np.zeros((n, n))
    for v in range(n):
        u = v
        while u >= 0:
            A[v, u] = 1.0
            u = parent[u]
    return A


def D_of(kappa):
    n = kappa.shape[0]
    K = centering(n) @ kappa @ centering(n)
    return float(np.sum(K * K))


rng = np.random.default_rng(SEED)
t0 = time.time()

# ---------------- L1 / L2 exact zeros
lims = {}
w1 = 0.0
for delta in (0.005, 0.3):
    for _ in range(200):
        e = eps_of(rng.normal(size=(1, 4, 4)), delta)
        if e is not None:
            w1 = max(w1, e)
lims["L1_single_cell_max_eps"] = w1
w2 = 0.0
for delta in (0.005, 0.3):
    for n in (2, 5, 16):
        for _ in range(30):
            lab = rng.normal(size=(4, 4))
            e = eps_of(np.repeat(lab[None, :, :], n, axis=0), delta)
            if e is not None:
                w2 = max(w2, e)
lims["L2_identical_labels_max_eps"] = w2
lims["L1_L2_pass"] = bool(w1 <= TOL_EXACT and w2 <= TOL_EXACT)

# ---------------- L3 delta -> 0, common random numbers, iid n = 8
n = 8
H = centering(n)
D = D_of(np.eye(n))
target = math.sqrt(EPS_STAR_SQ * D) / n
rows = []
labs = [rng.normal(size=(n, 4, 4)) for _ in range(600)]
for delta in (1e-4, 1e-3, 1e-2, 5e-2):
    vals = [eps_of(L, delta) for L in labs]
    vals = np.array([v for v in vals if v is not None])
    rms = math.sqrt(float(np.mean(vals ** 2)))
    rows.append({"delta": delta, "rms_over_delta2": rms / delta ** 2, "target_eps_star_sqrtD_over_n": target,
                 "rel": rms / delta ** 2 / target - 1.0, "trials": int(len(vals))})
lims["L3_delta_to_zero"] = rows

# ---------------- L4 / L5 exact kernel values
l4 = []
for m in (2, 4, 8, 16, 32):
    l4.append({"n": m, "D_iid": D_of(np.eye(m)), "closed_n_minus_1": m - 1,
               "eps_over_eps_star": math.sqrt(D_of(np.eye(m))) / m, "law_sqrt_n_minus_1_over_n": math.sqrt(m - 1) / m})
lims["L4_iid"] = l4
l5 = []
for m in (4, 8, 16, 32):
    k = np.zeros((m, m))
    half = m // 2
    k[:half, :half] = 1.0
    k[half:, half:] = 1.0
    l5.append({"n": m, "D": D_of(k), "closed_4n2p2q2": 4 * m * m * 0.25 * 0.25,
               "eps_over_eps_star": math.sqrt(D_of(k)) / m})
lims["L5_two_species_p_half"] = l5
lims["L5_n_independent_max_dev"] = max(abs(r["eps_over_eps_star"] - 0.5) for r in l5)
out["limits"] = lims
print("limits done", time.time() - t0, lims["L1_single_cell_max_eps"], lims["L2_identical_labels_max_eps"], flush=True)

# ---------------- independent closed forms (kappa built from explicit path sets)
def kappa_from_paths(parent):
    n = len(parent)
    paths = []
    for v in range(n):
        s, u = set(), v
        while u >= 0:
            s.add(u)
            u = parent[u]
        paths.append(s)
    return np.array([[float(len(paths[v] & paths[w])) for w in range(n)] for v in range(n)])


cf = {"worst_rel": 0.0, "rows": []}
for m in range(1, 13):
    chain = [-1] + list(range(m - 1))
    star = [-1] + [0] * (m - 1)
    d_chain = D_of(kappa_from_paths(chain))
    d_star = D_of(kappa_from_paths(star))
    c_chain = (m ** 2 - 1) * (2 * m ** 2 + 7) / 180
    c_star = m - 2 + 1 / m ** 2
    d_iid = D_of(np.eye(m))
    rowset = [(d_chain, c_chain), (d_star, c_star), (d_iid, m - 1)]
    for nB in range(m + 1):
        k = np.zeros((m, m))
        sB = np.arange(m) < nB
        k[np.ix_(sB, sB)] = 1.0
        k[np.ix_(~sB, ~sB)] = 1.0
        pp = nB / m
        rowset.append((D_of(k), 4 * m ** 2 * pp ** 2 * (1 - pp) ** 2))
    for direct, closed in rowset:
        cf["worst_rel"] = max(cf["worst_rel"], abs(direct - closed) / (1 + abs(closed)))
    cf["rows"].append({"n": m, "chain": d_chain, "chain_closed": c_chain, "star": d_star, "star_closed": c_star})
# cross term and mixed additivity
worst_cross = worst_mix = 0.0
for m in range(2, 13):
    for parent in ([-1] + list(range(m - 1)), [-1] + [0] * (m - 1), uniform_rooted_tree(m, rng), uniform_rooted_tree(m, rng)):
        kap = kappa_from_paths(parent)
        _, _, sub, _ = tree_arrays(parent)
        s = sub.astype(float)
        direct = float(np.trace(centering(m) @ kap))
        worst_cross = max(worst_cross, abs(direct - float(np.sum(s * (1 - s / m)))) / (1 + abs(direct)))
        mix = D_of(np.eye(m) + kap)
        add = (m - 1) + D_of(kap) + 2 * direct
        worst_mix = max(worst_mix, abs(mix - add) / (1 + abs(add)))
cf["worst_cross_term_rel"] = worst_cross
cf["worst_mixed_additivity_rel"] = worst_mix
cf["pass"] = bool(max(cf["worst_rel"], worst_cross, worst_mix) <= 1e-10)
out["closed_forms_independent"] = cf
print("closed forms", cf["worst_rel"], worst_cross, worst_mix, flush=True)

# ---------------- structure constants: exact multiplicity arithmetic and step sensitivity
M = quadratic_tensor(linear_map())
Kab = np.einsum("abij,abij->ab", M, M)
mult = {}
for val in Kab.ravel():
    key = round(float(val), 9)
    mult[key] = mult.get(key, 0) + 1
exact_sum = 96 * (1 / 8) + 24 * (1 / 6) + 72 * (1 / 2) + 12 * (2 / 3)
Ma = [M[a, a] for a in range(16)]
sc = {"multiplicities": {str(k): v for k, v in sorted(mult.items())},
      "count_total": int(sum(mult.values())),
      "T2_from_multiplicity_table": exact_sum,
      "T2_numeric": float(Kab.sum()),
      "T4_numeric": float(sum(float(np.sum(m * m)) for m in Ma)),
      "sum_a_Maa_max_abs": float(np.abs(sum(Ma)).max()),
      "normG0_sq": float(np.sum(G0 * G0)),
      "G0_is_2I": float(np.abs(G0 - 2 * np.eye(3)).max())}
for h in (1e-3, 5e-4, 2e-4):
    Mh = quadratic_tensor(linear_map(h))
    sc["T2_step_%g" % h] = float(np.einsum("abij,abij->", Mh, Mh))
    sc["sumMaa_step_%g" % h] = float(np.abs(sum(Mh[a, a] for a in range(16))).max())
out["structure_constants"] = sc
print("structure constants", sc["T2_numeric"], sc["T2_from_multiplicity_table"], flush=True)

# ---------------- condition (C): does the order relation fix the local log slope?
def d_fast(parent):
    _, depth, sub, prefix = tree_arrays(parent)
    n = len(parent)
    s = sub.astype(float)
    w2 = float(np.sum(s * s))
    w2p = float(np.sum((2.0 * depth + 1.0) * s * s))
    s_row = float(np.sum(prefix.astype(float) ** 2))
    return w2p - 2.0 * s_row / n + w2 * w2 / (n * n), int(depth.max())


def caterpillar(k):
    parent = [-1]
    spine = [0]
    for _ in range(1, k):
        parent.append(spine[-1])
        spine.append(len(parent) - 1)
    for v in spine:
        parent.extend([v] * (k - 1))
    return parent


def star_of_chains(k):
    parent = [-1]
    for _ in range(k):
        prev = 0
        for _ in range(k):
            parent.append(prev)
            prev = len(parent) - 1
    return parent


cond = {}
cond["chain"] = [{"n": m, "ratio": d_fast([-1] + list(range(m - 1)))[0] / (m * m * (m - 1) ** 2)} for m in (64, 256, 1024, 4096)]
cond["caterpillar"] = []
for k in (32, 64, 128, 256):
    p = caterpillar(k)
    d, dep = d_fast(p)
    cond["caterpillar"].append({"k": k, "n": len(p), "ratio_depth2": d / (len(p) ** 2 * dep ** 2)})
cond["star_of_chains"] = []
for k in (32, 64, 128, 256):
    p = star_of_chains(k)
    d, dep = d_fast(p)
    cond["star_of_chains"].append({"k": k, "n": len(p), "ratio_depth1": d / (len(p) ** 2 * dep),
                                   "ratio_depth2": d / (len(p) ** 2 * dep ** 2)})
cond["cayley_E_D_over_n3"] = [{"n": m, "v": cayley_exact(m)["E_D"] / m ** 3} for m in (128, 512, 2048, 8192)]
# local log slopes gamma = dln(sqrt(D)/n)/dln n
def local_slope(fn, sizes):
    xs = [math.log(s) for s in sizes]
    ys = [math.log(math.sqrt(fn(s)) / s) for s in sizes]
    return [(sizes[i], (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])) for i in range(len(sizes) - 1)]


cond["local_gamma_chain"] = local_slope(lambda m: d_fast([-1] + list(range(m - 1)))[0], [64, 256, 1024, 4096])
cond["local_gamma_cayley"] = local_slope(lambda m: cayley_exact(m)["E_D"], [64, 256, 1024, 4096])
soc_sizes = [k * k + 1 for k in (32, 64, 128, 256)]
cond["local_gamma_star_of_chains"] = local_slope(
    lambda nn: d_fast(star_of_chains(int(round(math.sqrt(nn - 1)))))[0], soc_sizes)
cond["local_gamma_caterpillar"] = local_slope(
    lambda nn: d_fast(caterpillar(int(round(math.sqrt(nn)))))[0], [k * k for k in (32, 64, 128, 256)])
out["condition_C_scope"] = cond
print("condition C done", time.time() - t0, flush=True)

# ---------------- random_sample_20
def make_kernel(kind, m, rng, p=None, parent=None):
    if kind == "iid":
        return np.eye(m), np.eye(m)
    if kind == "her":
        A = anc_matrix(parent)
        return A @ A.T, A
    if kind == "coh":
        nB = max(1, min(m - 1, int(round(p * m))))
        A = np.zeros((m, 2))
        A[:nB, 0] = 1.0
        A[nB:, 1] = 1.0
        return A @ A.T, A
    A = np.concatenate([np.eye(m), anc_matrix(parent)], axis=1)
    return A @ A.T, A


srows = []
rs = np.random.default_rng(SEED)
kinds = ["iid", "her", "coh", "mix"]
for i in range(SAMPLE_N):
    kind = kinds[i % 4]
    m = int(rs.integers(2, 13))
    delta = float(np.exp(rs.uniform(math.log(0.002), math.log(0.02))))
    parent = uniform_rooted_tree(m, rs) if kind in ("her", "mix") else None
    p = float(rs.uniform(0.2, 0.8)) if kind == "coh" else None
    kap, A = make_kernel(kind, m, rs, p=p, parent=parent)
    Dv = D_of(kap)
    law = EPS_STAR_SQ * delta ** 4 * Dv / (m * m)
    if Dv < 1e-12:
        srows.append({"i": i, "kind": kind, "n": m, "delta": delta, "D": Dv, "degenerate": True})
        continue
    e2 = []
    rej = 0
    while len(e2) < SAMPLE_TRIALS:
        z = rs.normal(size=(A.shape[1], 4, 4))
        labels = np.einsum("vu,uab->vab", A, z)
        e = eps_of(labels, delta)
        if e is None:
            rej += 1
            continue
        e2.append(e * e)
    e2 = np.array(e2)
    mean = float(e2.mean())
    se = float(e2.std(ddof=1) / math.sqrt(SAMPLE_TRIALS))
    srows.append({"i": i, "kind": kind, "n": m, "delta": delta, "D": Dv, "law_E_eps2": law,
                  "mc_E_eps2": mean, "rel": mean / law - 1.0, "rel_se": se / law,
                  "z": (mean - law) / se, "rejections": rej})
    print("sample", i, kind, m, round(delta, 5), "rel", round(srows[-1]["rel"] * 100, 3), "z",
          round(srows[-1]["z"], 2), round(time.time() - t0, 1), flush=True)

zs = [r["z"] for r in srows if "z" in r]
wsum = sum(1.0 / r["rel_se"] ** 2 for r in srows if "z" in r)
comb = sum(r["rel"] / r["rel_se"] ** 2 for r in srows if "z" in r) / wsum
comb_se = 1.0 / math.sqrt(wsum)
out["random_sample_20"] = {"rows": srows, "max_abs_z": max(abs(z) for z in zs),
                           "n_cases": len(zs), "combined_rel": comb, "combined_se": comb_se,
                           "combined_z": comb / comb_se,
                           "pass": bool(max(abs(z) for z in zs) <= MAX_Z and abs(comb / comb_se) <= MAX_Z_COMB)}
out["wall_seconds"] = time.time() - t0
(HERE / "a4_limits_sample.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("closed_forms_independent", "condition_C_scope")},
                 ensure_ascii=False, indent=1, default=float)[:4000])
