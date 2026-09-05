"""adversary b8: re-execute the card recovers geometrically (n=1, H=0, delta->0,
coherent two-species, isotropic) at off-grid sizes with seed 20260903, and compare
the w-balance set with the kernel.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment

OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
Mt = np.load(OUT / "b1_Mt.npy")
REF = geometric_self_dual_triple(np.eye(4))
NS = 2.0 * math.sqrt(3.0)


def cell(l, d):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + d * l)).aligned_candidate


def res(labels, d):
    return simplicity_residual(sum(cell(l, d) for l in labels))


def main():
    rng = np.random.default_rng(20260903)
    out = {}
    # n = 1 and H = 0 (all cells identical)
    w1, wH = 0.0, 0.0
    for _ in range(40):
        l = rng.normal(size=(4, 4))
        w1 = max(w1, res(l[None] * 1.0, 0.005))
        for n in (3, 7, 17):
            wH = max(wH, res(np.repeat(l[None], n, axis=0), 0.005))
    out["n1_max_residual"] = w1
    out["H0_identical_labels_max_residual"] = wH

    # delta -> 0 for a generic Sigma (rank-4 random): residual/delta^2 -> master
    Zr = rng.normal(size=(16, 4))
    Sg = Zr @ Zr.T
    F = float(np.linalg.norm(np.einsum("ab,abij->ij", Sg, Mt)))
    T = float(np.einsum("abij,ac,bd,cdij->", Mt, Sg, Sg, Mt))
    lam, V = np.linalg.eigh(Sg)
    A = V @ np.diag(np.sqrt(np.clip(lam, 0, None)))
    conv = {}
    n = 7
    g0 = rng.normal(size=(120, n, 16))
    for d in (1e-2, 1e-3, 1e-4, 1e-5):
        vals = [res((g @ A.T).reshape(n, 4, 4), d) for g in g0]
        conv[str(d)] = float(np.sqrt(np.mean(np.asarray(vals) ** 2))) / d ** 2
    conv["master"] = math.sqrt((n - 1) * ((n - 1) * F * F + 2 * T) / (12 * n * n))
    out["delta_to_zero_generic"] = conv

    # coherent two species (kappa_coh, p = 1/2): eps should be n-independent
    coh = {}
    Ai = np.eye(16)
    for n in (6, 10, 18):
        vals = []
        for _ in range(60):
            base = rng.normal(size=(2, 16))
            pick = np.array([0] * (n // 2) + [1] * (n - n // 2))
            lab = base[pick].reshape(n, 4, 4)
            vals.append(res(lab, 0.005))
        coh[str(n)] = float(np.sqrt(np.mean(np.asarray(vals) ** 2))) / 0.005 ** 2
    coh["predicted_eps_star_2p1mp"] = math.sqrt(10.0) * 2 * 0.5 * 0.5
    out["coherent_two_species_p_half"] = coh

    # isotropic Sigma = I16 at off-grid n
    iso = {}
    for n in (5, 11, 23):
        vals = []
        for _ in range(200):
            lab = rng.normal(size=(n, 4, 4))
            vals.append(res(lab, 0.005))
        iso[str(n)] = {"obs": float(np.sqrt(np.mean(np.asarray(vals) ** 2))) / 0.005 ** 2,
                       "master": math.sqrt(10.0 * (n - 1)) / n}
    out["isotropic_offgrid"] = iso

    # w-balance vs kernel as subspaces of Sym^2(R^16)
    B = []
    for a in range(16):
        m = np.zeros((16, 16)); m[a, a] = 1.0; B.append(m)
    for a in range(16):
        for b in range(a + 1, 16):
            m = np.zeros((16, 16)); m[a, b] = m[b, a] = 1 / math.sqrt(2.0); B.append(m)
    CLS = {1: ["01", "10", "23", "32"], 2: ["02", "20", "31", "13"], 3: ["03", "30", "12", "21"]}
    ix = lambda s: 4 * int(s[0]) + int(s[1])
    K = np.array([np.einsum("ab,abij->ij", S_, Mt).reshape(9) for S_ in B]).T
    Wb = np.array([[sum(S_[ix(a), ix(a)] for a in CLS[k]) for S_ in B] for k in (1, 2, 3)])
    Wb = np.array([Wb[0] - Wb[1], Wb[1] - Wb[2]])
    rk = np.linalg.matrix_rank(K, tol=1e-9)
    rw = np.linalg.matrix_rank(Wb, tol=1e-9)
    both = np.vstack([K, Wb])
    rb = np.linalg.matrix_rank(both, tol=1e-9)
    out["subspaces"] = {"dim_kernel": 136 - int(rk), "dim_w_balanced": 136 - int(rw),
                        "dim_intersection": 136 - int(rb),
                        "kernel_subset_of_w_balanced": bool(rb == rw),
                        "w_balanced_subset_of_kernel": bool(rb == rk)}
    out["_meta"] = {"seed": 20260903, "delta": 0.005, "note": "off-grid sizes"}
    (OUT / "b8_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
