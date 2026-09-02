"""a2: audit of the isotropy lemma (S4.3)-(S4.8).

(1) is xi -> L xi L^T the correct action on a tetrad perturbation e = I + delta xi?
(2) does R_0(L) really run over all of SO(3) (surjectivity needed for Schur)?
(3) is the statement exact order by order in delta (homogeneity)?
(4) independent numeric: E[tl gram Y] = 0 at delta = 0.3, n in {2,5}, 20000 draws.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "verify" / "Q-0012" / "F-01", ROOT / "verify" / "Q-0008" / "F-02"):
    sys.path.insert(0, str(p))

from check_cumulant import gram_form, linear_map, quadratic_tensor, tl  # noqa: E402
from examples.physics.causal_face_simplicity import _PAIR_INDEX, geometric_self_dual_triple, plebanski_gram  # noqa: E402
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
REF = geometric_self_dual_triple(np.eye(4))
G0 = plebanski_gram(REF)
out = {"script": "a2_symmetry", "seed": SEED}


def so4(rng):
    q, r = np.linalg.qr(rng.normal(size=(4, 4)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def mat6(v):
    m = np.zeros((4, 4))
    for comp, (a, b) in zip(v, _PAIR_INDEX):
        m[a, b] = comp
        m[b, a] = -comp
    return m


def vec6(m):
    return np.array([m[a, b] for (a, b) in _PAIR_INDEX])


def rho6(L, triple):
    return np.array([vec6(L @ mat6(row) @ L.T) for row in triple])


def R0_of(L):
    return gram_form(rho6(L.T, REF), REF) @ np.linalg.inv(G0)


def aligned(lab, delta):
    tet = np.eye(4) + delta * lab
    return optimal_internal_alignment(REF, geometric_self_dual_triple(tet)).aligned_candidate


def block(labels, delta):
    Y = np.zeros_like(REF)
    for lab in labels:
        Y = Y + aligned(lab, delta)
    return Y


def safe_block(labels, delta):
    try:
        return block(labels, delta)
    except Exception:
        return None


def rel(gA, gY, R0):
    if gA is None:
        return float("nan")
    return float(np.linalg.norm(gA - R0 @ gY @ R0.T) / np.linalg.norm(gY))


rng = np.random.default_rng(SEED)

# ---- (1) conjugation vs one-sided action, several delta and n
rows = []
for delta in (0.05, 0.2, 0.5):
    for n in (2, 3, 5):
        worst_conj = worst_left = worst_right = 0.0
        for _ in range(6):
            L = so4(rng)
            xi = rng.normal(size=(n, 4, 4))
            if any(np.linalg.det(np.eye(4) + delta * x) <= 0.2 for x in xi):
                continue
            R0 = R0_of(L)
            YY = safe_block(xi, delta)
            if YY is None:
                continue
            gY = plebanski_gram(YY)
            bC = safe_block(np.array([L @ x @ L.T for x in xi]), delta)
            bL = safe_block(np.array([L @ x for x in xi]), delta)
            bR = safe_block(np.array([x @ L.T for x in xi]), delta)
            gC = plebanski_gram(bC) if bC is not None else None
            gL = plebanski_gram(bL) if bL is not None else None
            gR = plebanski_gram(bR) if bR is not None else None
            worst_conj = max(worst_conj, rel(gC, gY, R0))
            wl, wr = rel(gL, gY, R0), rel(gR, gY, R0)
            worst_left = max(worst_left, 0.0 if wl != wl else wl)
            worst_right = max(worst_right, 0.0 if wr != wr else wr)
        rows.append({"delta": delta, "n": n, "conjugation_rel_err": worst_conj,
                     "left_only_rel_err": worst_left, "right_only_rel_err": worst_right})
out["action_check"] = rows
out["conjugation_max_rel_err"] = max(r["conjugation_rel_err"] for r in rows)
out["left_only_min_rel_err"] = min(r["left_only_rel_err"] for r in rows)

# ---- (2) R_0 : SO(4) -> SO(3) homomorphism and surjectivity
hom_err = 0.0
angles = []
for _ in range(400):
    L1, L2 = so4(rng), so4(rng)
    hom_err = max(hom_err, float(np.linalg.norm(R0_of(L1 @ L2) - R0_of(L1) @ R0_of(L2))))
    R = R0_of(L1)
    angles.append(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
angles = np.array(angles)
# Haar SO(3) rotation angle density is (1 - cos t)/pi on [0, pi]
grid = np.linspace(0, math.pi, 9)
emp = np.histogram(angles, bins=grid)[0] / len(angles)
haar = np.array([(grid[i + 1] - math.sin(grid[i + 1]) - (grid[i] - math.sin(grid[i]))) / math.pi for i in range(8)])
out["R0_group"] = {"homomorphism_max_err": hom_err, "angle_hist": emp.tolist(), "haar_so3_hist": haar.tolist(),
                   "max_hist_dev": float(np.abs(emp - haar).max()), "angle_max": float(angles.max()),
                   "note": "surjectivity onto SO(3) is what Schur needs"}

# ---- (3) order-by-order: the delta^2 coefficient tensor M must satisfy the same equivariance
M = quadratic_tensor(linear_map())


def U_conj(L):
    """orthogonal 16x16 matrix of xi -> L xi L^T in the standard basis e_a of R^{4x4}."""
    U = np.zeros((16, 16))
    for a in range(16):
        E = np.zeros((4, 4))
        E[a // 4, a % 4] = 1.0
        U[:, a] = (L @ E @ L.T).reshape(16)
    return U


worst_M = 0.0
for _ in range(20):
    L = so4(rng)
    U, R0 = U_conj(L), R0_of(L)
    xi = rng.normal(size=16)
    phi = np.einsum("a,b,abij->ij", xi, xi, M)
    xi2 = U @ xi
    phi2 = np.einsum("a,b,abij->ij", xi2, xi2, M)
    worst_M = max(worst_M, float(np.linalg.norm(phi2 - R0 @ phi @ R0.T) / np.linalg.norm(phi)))
    worst_M = max(worst_M, float(np.linalg.norm(U @ U.T - np.eye(16))))
out["delta2_order_equivariance_max_rel_err"] = worst_M

# ---- tetrad depends on (delta, xi) only through the product -> homogeneity of the delta series
h_err = 0.0
for _ in range(10):
    xi = rng.normal(size=(2, 4, 4)) * 0.5
    a = safe_block(xi, 0.2)
    b = safe_block(2.0 * xi, 0.1)
    if a is None or b is None:
        continue
    h_err = max(h_err, float(np.linalg.norm(a - b) / np.linalg.norm(a)))
out["homogeneity_delta_xi_product_max_err"] = h_err

# ---- MIN_DET rule: conjugation invariant?  parity invariant?
det_conj, det_par = 0.0, 0.0
for _ in range(50):
    L = so4(rng)
    x = rng.normal(size=(4, 4))
    for d in (0.005, 0.02, 0.3):
        det_conj = max(det_conj, abs(np.linalg.det(np.eye(4) + d * (L @ x @ L.T)) - np.linalg.det(np.eye(4) + d * x)))
        det_par = max(det_par, abs(np.linalg.det(np.eye(4) - d * x) - np.linalg.det(np.eye(4) + d * x)))
out["min_det_rule"] = {"max_det_change_under_conjugation": det_conj,
                       "max_det_change_under_parity": det_par,
                       "note": "MIN_DET conditioning is conjugation invariant (isotropy lemma survives) but NOT parity invariant (odd-order vanishing argument is formally conditional; numerically vacuous at delta<=0.02)"}

# ---- (4) E[tl gram Y] = 0 at delta = 0.3, n in {2,5}, 20000 draws
big = {}
for n in (2, 5):
    N = 20000
    acc = np.zeros((3, 3))
    sq = 0.0
    rej = 0
    got = 0
    rng2 = np.random.default_rng(SEED + 7 * n)
    while got < N:
        xi = rng2.normal(size=(n, 4, 4))
        if any(np.linalg.det(np.eye(4) + 0.3 * x) <= 0.05 for x in xi):
            rej += 1
            continue
        YY = safe_block(xi, 0.3)
        if YY is None:
            rej += 1
            continue
        t = tl(plebanski_gram(YY))
        acc += t
        sq += float(np.sum(t * t))
        got += 1
    mean = acc / N
    rms = math.sqrt(sq / N)
    big[f"n{n}"] = {"delta": 0.3, "draws": N, "rejected": rej,
                    "norm_mean_tl": float(np.linalg.norm(mean)), "rms_tl": rms,
                    "norm_mean_over_rms": float(np.linalg.norm(mean)) / rms,
                    "expected_if_zero_mean_1sigma": 1.0 / math.sqrt(N),
                    "z_like": float(np.linalg.norm(mean)) / rms * math.sqrt(N)}
out["E_tl_gram_Y_zero"] = big

(HERE / "a2_symmetry.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
