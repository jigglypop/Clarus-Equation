"""adversary b2: geometric MC on random KERNEL Sigma (not the card's Sigma_b),
off-grid n in {6,12,24}, seed 20260903 (different from the card's 20260902).

Kernel construction: I_16 is in ker(tl G); project a random symmetric matrix onto
ker and add lambda*I_16 -> full-rank PSD, non-diagonal, rank 16 kernel element.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
Mt = np.load(OUT / "b1_Mt.npy")
REF = geometric_self_dual_triple(np.eye(4))
MIN_DET = 0.05


def tlG(sigma):
    return np.einsum("ab,abij->ij", sigma, Mt)


def F_of(sigma):
    return float(np.linalg.norm(tlG(sigma)))


def T_of(sigma):
    return float(np.einsum("abij,ac,bd,cdij->", Mt, sigma, sigma, Mt))


def master(n, F, T):
    return math.sqrt((n - 1) * ((n - 1) * F * F + 2.0 * T) / (12.0 * n * n))


def sym_basis():
    B = []
    for a in range(16):
        m = np.zeros((16, 16)); m[a, a] = 1.0; B.append(m)
    for a in range(16):
        for b in range(a + 1, 16):
            m = np.zeros((16, 16)); m[a, b] = m[b, a] = 1.0 / math.sqrt(2.0); B.append(m)
    return B


BAS = sym_basis()
KF = np.zeros((9, 136))
for i, Sg in enumerate(BAS):
    KF[:, i] = tlG(Sg).reshape(9)
U, sv, Vt = np.linalg.svd(KF)
NULL = Vt[5:].T                      # 136 x 131


def project_kernel(sigma):
    v = np.array([float(np.sum(b * sigma)) for b in BAS])
    v = NULL @ (NULL.T @ v)
    return sum(c * b for c, b in zip(v, BAS))


def random_kernel_sigma(rng, jitter=1.0):
    Z = rng.normal(size=(16, 16))
    Ssym = (Z + Z.T) / 2.0
    K = project_kernel(Ssym)
    lam = -float(np.linalg.eigvalsh(K).min()) + jitter
    return K + lam * np.eye(16)      # I_16 in kernel -> stays in kernel, PSD


def cell(label, delta):
    triple = geometric_self_dual_triple(np.eye(4) + delta * label)
    return optimal_internal_alignment(REF, triple).aligned_candidate


def residual(labels, delta):
    return simplicity_residual(sum(cell(l, delta) for l in labels))


def mc(sigma, n, trials, delta, seed):
    lam, V = np.linalg.eigh(sigma)
    A = V @ np.diag(np.sqrt(np.clip(lam, 0.0, None)))
    rng = np.random.default_rng(seed)
    vals, resampled = [], 0
    for _ in range(trials):
        while True:
            g = rng.normal(size=(n, 16))
            lab = (g @ A.T).reshape(n, 4, 4)
            if np.all(np.linalg.det(np.eye(4)[None] + delta * lab) > MIN_DET):
                break
            resampled += 1
        vals.append(residual(lab, delta))
    v = np.asarray(vals)
    r = float(np.sqrt(np.mean(v * v)))
    return r, r / math.sqrt(2.0 * trials), resampled


def main():
    delta, trials = 0.005, 300
    sizes = (6, 12, 24)
    rng = np.random.default_rng(20260903)
    cases = {}
    for i in range(3):
        cases["rand_kernel_%d" % i] = random_kernel_sigma(rng)
    for i in range(2):
        Z = rng.normal(size=(16, 4))
        cases["rand_generic_%d" % i] = Z @ Z.T
    e = lambda m, n_: np.eye(16)[4 * m + n_]
    u = (e(0, 1) + e(2, 3)) / math.sqrt(2.0)
    cases["card_sigma_b"] = np.outer(u, u) / 2.0 * 1.0 + 2.0 * np.outer(e(0, 3), e(0, 3))
    cases["card_sigma_b"] = 0.5 * np.outer(u, u) + 2.0 * np.outer(e(0, 3), e(0, 3))
    a = (e(0, 1) - e(1, 0)) / math.sqrt(2.0)
    cases["antisym_e01_minus_e10"] = np.outer(a, a)
    out = {}
    t0 = time.time()
    for name, Sig in cases.items():
        F, T = F_of(Sig), T_of(Sig)
        rec = {"F": F, "T": T, "in_kernel": bool(F < 1e-9),
               "floor_over_delta2": F / (2 * math.sqrt(3.0)),
               "trF2_check_trace": float(np.trace(Sig))}
        for n in sizes:
            r, se, res = mc(Sig, n, trials, delta, 20260903 + 17 * n + hash(name) % 1000)
            pred = master(n, F, T)
            rec["n%d" % n] = {"observed_over_delta2": r / delta ** 2, "master": pred,
                              "ratio": (r / delta ** 2) / pred if pred > 0 else None,
                              "mc_se_rel": se / r if r > 0 else None, "resampled": res}
        out[name] = rec
        print(name, json.dumps(rec, ensure_ascii=False)[:400], flush=True)
    out["_meta"] = {"delta": delta, "trials": trials, "sizes": list(sizes),
                    "seed_base": 20260903, "min_det": MIN_DET, "seconds": time.time() - t0,
                    "note": "off-grid sizes and a different seed; not the preregistered statistic"}
    (OUT / "b2_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
