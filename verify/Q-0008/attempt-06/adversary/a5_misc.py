"""a5: (i) direct check of the step-2 identity as used in (S2.1); (ii) the delta^2 coefficient
Phi against the physical block; (iii) exact Meir-Moon check of the K2 pre-registered numbers
E tr(H kappa) = 92.3847 and E D_C(32) = 2008.0806 quoted in (S7.7)/verify[20];
(iv) explicit counterexample to (S8.2) *as literally stated* (order relation vs limit).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
from fractions import Fraction
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "verify" / "Q-0012" / "F-01", ROOT / "verify" / "Q-0008" / "F-02"):
    sys.path.insert(0, str(p))

from check_cumulant import linear_map, quadratic_tensor, tl  # noqa: E402
from driver_numbers import cayley_exact  # noqa: E402
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram  # noqa: E402
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
REF = geometric_self_dual_triple(np.eye(4))
M = quadratic_tensor(linear_map())
out = {"script": "a5_misc", "seed": SEED}
rng = np.random.default_rng(SEED)


def gram2(a, b):
    from check_cumulant import gram_form
    return 0.5 * (gram_form(a, b) + gram_form(b, a))


def aligned(lab, delta):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + delta * lab)).aligned_candidate


# (i) step-2 identity, used verbatim in (S2.1)
worst = 0.0
for delta in (0.005, 0.1, 0.3):
    for n in (2, 3, 5, 8):
        xi = rng.normal(size=(n, 4, 4))
        if any(np.linalg.det(np.eye(4) + delta * x) <= 0.2 for x in xi):
            continue
        X = np.array([aligned(x, delta) for x in xi])
        eta = X - REF
        Y = X.sum(axis=0)
        lhs = tl(plebanski_gram(Y))
        zeta = eta - eta.mean(axis=0)
        rhs = -n * sum(tl(plebanski_gram(z)) for z in zeta)
        worst = max(worst, float(np.linalg.norm(lhs - rhs) / np.linalg.norm(plebanski_gram(Y))))
out["step2_identity_max_rel_err"] = worst

# (ii) delta^2 coefficient: tl gram Y / (-n delta^2) -> Phi(H xi)
rows = []
for delta in (1e-4, 1e-3, 1e-2):
    n = 5
    H = np.eye(n) - np.ones((n, n)) / n
    xi = rng.normal(size=(n, 4, 4))
    Y = sum(aligned(x, delta) for x in xi)
    lhs = tl(plebanski_gram(Y)) / (-n * delta ** 2)
    xt = H @ xi.reshape(n, 16)
    phi = np.einsum("va,vb,abij->ij", xt, xt, M)
    rows.append({"delta": delta, "rel_err": float(np.linalg.norm(lhs - phi) / np.linalg.norm(phi))})
out["delta2_coefficient_vs_Phi"] = rows

# (iii) exact Meir-Moon: E[#{u: s_u = k}] = C(n,k) k^{k-1} (n-k)^{n-k} / n^{n-1}
def meir_moon(n):
    from math import comb
    return [Fraction(comb(n, k) * k ** (k - 1) * (n - k) ** (n - k), n ** (n - 1)) if k < n else Fraction(1)
            for k in range(1, n + 1)]


n = 32
Nk = meir_moon(n)
E_sum_s = sum(Fraction(k) * Nk[k - 1] for k in range(1, n + 1))
E_sum_s2 = sum(Fraction(k * k) * Nk[k - 1] for k in range(1, n + 1))
E_cross = E_sum_s - E_sum_s2 / n
ED = cayley_exact(n)["E_D"]
X32 = 2 * float(E_cross) / math.sqrt((n - 1) * ED)
out["K2_numbers"] = {
    "n": n, "E_tr_H_kappa_exact": float(E_cross), "card_note_value": 92.3847,
    "E_D_C_32": ED, "card_note_E_D": 2008.0806,
    "X32_from_exact": X32, "card_prereg_value": 0.7406,
    "sum_Nk_equals_n": float(sum(Nk)),
    "rel_err_cross": abs(float(E_cross) / 92.3847 - 1), "rel_err_ED": abs(ED / 2008.0806 - 1),
    "rel_err_X32": abs(X32 / 0.7406 - 1), "K2_window": [0.49, 0.99],
}

# (iv) (S8.2) as literally stated:  D/n^2 asymp depth^2  and  depth asymp n^{1/d}  =>  gamma = 1/d ?
ns = np.array([2.0 ** k for k in range(6, 25)])
depth = np.sqrt(ns)                                   # depth asymp n^{1/2}, d_tree = 2
ratio = 2.0 + np.sin(np.log(ns))                      # bounded in [1,3]: D/n^2 asymp depth^2 holds
Dn2 = depth ** 2 * ratio
eps = np.sqrt(Dn2) / 1.0
ln_n = np.log(ns)
gam = np.diff(np.log(eps)) / np.diff(ln_n)
out["S8_2_implication_counterexample"] = {
    "construction": "D/n^2 = depth^2 * (2 + sin(ln n)), depth = n^{1/2}",
    "ratio_bounds": [float(ratio.min()), float(ratio.max())],
    "asymp_holds": True,
    "local_gamma_min": float(gam.min()), "local_gamma_max": float(gam.max()),
    "target_1_over_d_tree": 0.5,
    "weak_exponent_lim_ln_eps_over_ln_n": float(np.log(eps[-1]) / ln_n[-1]),
    "note": "the order relation fixes the weak exponent lim ln eps / ln n = 1/d_tree but NOT the local log-derivative gamma; (S8.2) needs D/(n^2 depth^2) -> const (regular variation), which every family the derivation tests does satisfy",
}

# (v) B part: reported tl/gram range
res = json.loads((HERE.parent / "result.json").read_text(encoding="utf-8"))
tg = [r["tl_over_gram"] for r in res["B_equivariance"]["rows"]]
out["B_tl_over_gram_range"] = {"min": min(tg), "max": max(tg), "claimed_in_derivation": [0.027, 0.114]}
out["C_b_rejections"] = res["C_physical"]["rejections"]

(HERE / "a5_misc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
