"""Adversary a2: physical (tetrad) checks of the Q-0018 F-01 claims."""
from __future__ import annotations
import json, math, sys, time
from itertools import permutations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

M = np.load(HERE / "M_tensor_adversary.npy")
I4 = np.eye(4)
REF = geometric_self_dual_triple(I4)
SEED = 424243
DELTA = 0.005
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
out = {"script": "a2_physical", "seed": SEED, "delta": DELTA, "trials": TRIALS}
EPS4 = np.zeros((4, 4, 4, 4))
for p in permutations(range(4)):
    s = 1
    pl = list(p)
    for i in range(4):
        for j in range(i + 1, 4):
            if pl[i] > pl[j]:
                s = -s
    EPS4[p] = s


def M_of(X, Y):
    return np.einsum("a,b,abij->ij", np.asarray(X, float).reshape(16),
                     np.asarray(Y, float).reshape(16), M)


def aligned(t):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(t)).aligned_candidate


def eps_of(tetrads):
    Y = sum(aligned(t) for t in tetrads)
    g = plebanski_gram(Y)
    return float(np.linalg.norm(g - np.trace(g) / 3 * np.eye(3)) / np.linalg.norm(g))


def act(X, w1, w9, wp, wm):
    sym = (X + X.T) / 2
    anti = (X - X.T) / 2
    tr = np.trace(X) / 4 * np.eye(4)
    st = sym - tr
    du = 0.5 * np.einsum("ijkl,kl->ij", EPS4, anti)
    sd = (anti + du) / 2
    asd = (anti - du) / 2
    return 16 * (w1 / 1 * tr + w9 / 9 * st + wp / 3 * sd + wm / 3 * asd)


def C_full(w1, w9, wp, wm):
    C = np.zeros((16, 16))
    for a in range(16):
        E = np.zeros((4, 4))
        E[a // 4, a % 4] = 1.0
        C[:, a] = act(E, w1, w9, wp, wm).reshape(16)
    return C


cases = {"isotropic_gaussian": (1 / 16, 9 / 16, 3 / 16, 3 / 16),
         "rank1_shear_card": (0.0, 0.5, 0.25, 0.25),
         "symmetric_traceless_only": (0.0, 1.0, 0.0, 0.0),
         "antisymmetric_only": (0.0, 0.0, 0.5, 0.5)}
conv = {"formula": "c2 = 10*lambda9^2, lambda9 = 16*(P9 weight)/9"}
for name, w in cases.items():
    C = C_full(*w)
    T2f = float(np.einsum("ac,bd,abij,cdij->", C, C, M, M))
    conv[name] = {"weights": list(w), "lambda9": 16 * w[1] / 9, "c2": 2 * T2f / 12,
                  "trace_C_full": float(np.trace(C))}
out["conv_c2_vs_P9_weight"] = conv

E01 = np.zeros((4, 4))
E01[0, 1] = 1.0
S01 = (E01 + E01.T) / math.sqrt(2)
A01 = (E01 - E01.T) / math.sqrt(2)
two = {}
for name, D in (("shear_4E01", 4 * E01), ("sym_only_4S", 4 * S01), ("antisym_only_4A", 4 * A01)):
    row = {}
    for dl in (0.02, 0.005, 0.001):
        e = eps_of([I4, I4 + dl * D])
        row[str(dl)] = {"eps": e, "eps_over_d2": e / dl ** 2}
    Phi = 0.5 * M_of(D, D)
    row["predicted_eps_over_d2"] = float(np.linalg.norm(Phi) / (2 * math.sqrt(12)))
    row["norm_D_sq"] = float(np.sum(D * D))
    row["det_tetrad_at_0p02"] = float(np.linalg.det(I4 + 0.02 * D))
    two[name] = row
two["ratio_sym_over_shear_observed"] = (two["sym_only_4S"]["0.005"]["eps_over_d2"]
                                        / two["shear_4E01"]["0.005"]["eps_over_d2"])
two["ratio_sym_over_shear_predicted"] = 2.0
out["two_cell_deterministic"] = two


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


def generator(name, n):
    if name == "iid":
        return np.eye(n)
    if name == "chain":
        return np.tril(np.ones((n, n)))
    if name == "coh":
        A = np.zeros((n, 2))
        A[:n // 2, 0] = 1.0
        A[n // 2:, 1] = 1.0
        return A
    raise ValueError(name)


def haar_pairs(rng, count):
    g = rng.normal(size=(count, 4, 2))
    n = g[:, :, 0] / np.linalg.norm(g[:, :, 0], axis=1, keepdims=True)
    m = g[:, :, 1] - np.sum(g[:, :, 1] * n, axis=1, keepdims=True) * n
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return n, m


def D_S(A):
    n = A.shape[0]
    B = A.T @ centering(n) @ A
    return float(np.sum(B * B)), float(np.sum(np.diag(B) ** 2))


rng = np.random.default_rng(SEED)
ident = {}
for mode in ("iid", "chain", "coh"):
    for n in (3, 4, 8):
        A = generator(mode, n)
        mi = A.shape[1]
        nn, mm = haar_pairs(rng, mi)
        Z = 4.0 * np.einsum("ti,tk->tik", nn, mm)
        xis = np.einsum("vu,uik->vik", A, Z)
        eta = xis - xis.mean(axis=0, keepdims=True)
        Phi = sum(M_of(eta[v], eta[v]) for v in range(n))
        pred = float(np.linalg.norm(Phi) / (n * math.sqrt(12)))
        row = {"predicted_eps_over_d2": pred}
        for dl in (0.01, 0.005, 0.002):
            e = eps_of([I4 + dl * xis[v] for v in range(n)])
            row[str(dl)] = {"eps_over_d2": e / dl ** 2, "rel_err": e / dl ** 2 / pred - 1}
        ident[mode + "_n" + str(n)] = row
out["per_config_master_identity"] = ident

t0 = time.time()
mc = {}
c2e, c4e = 640 / 81, -352 / 81
for mode, n in (("iid", 2), ("iid", 4), ("iid", 8), ("iid", 16), ("chain", 2), ("chain", 16), ("coh", 16)):
    A = generator(mode, n)
    mi = A.shape[1]
    D, S = D_S(A)
    vals = np.empty(TRIALS)
    tls = np.empty((TRIALS, 3, 3))
    for t in range(TRIALS):
        nn, mm = haar_pairs(rng, mi)
        Z = 4.0 * np.einsum("ti,tk->tik", nn, mm)
        xis = np.einsum("vu,uik->vik", A, Z)
        Y = sum(aligned(I4 + DELTA * xis[v]) for v in range(n))
        g = plebanski_gram(Y)
        tlg = g - np.trace(g) / 3 * np.eye(3)
        tls[t] = tlg
        vals[t] = (np.linalg.norm(tlg) / np.linalg.norm(g)) ** 2
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(TRIALS))
    scale = n * n / (DELTA ** 4 * D)
    pred = c2e + c4e * S / D
    mc[mode + "_n" + str(n)] = {"D": D, "S_gen": S, "c_obs": mean * scale, "c_se": se * scale,
                                "c_pred_card": pred, "rel_dev": mean * scale / pred - 1,
                                "c_gauss_F02": 10.0,
                                "c_alt_C_is_I16": 10.0 + (32 / 9 - 10) * S / D,
                                "cv": float(vals.std(ddof=1) / mean),
                                "z_floor": float(TRIALS * np.sum(tls.mean(axis=0) ** 2)
                                                 / np.mean(np.sum(tls ** 2, axis=(1, 2))))}
out["reduced_physical_MC"] = mc
out["mc_wall_seconds"] = time.time() - t0
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "a2_physical.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
