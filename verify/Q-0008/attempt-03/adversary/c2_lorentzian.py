"""adversary c2 (Q-0008 attempt-03): the claim "signature (Euclid/Lorentz) irrelevant".

The repository primitives are float-only, so this file re-implements wedge/gram/tl over C in the
same (01,02,03,23,31,12) convention and builds LORENTZIAN self-dual triples

    Sigma^i = e^0 ^ e^i + (i/2) eps^{ijk} e^j ^ e^k      (eigenvalue +i of the Minkowski Hodge star)

from real tetrads, certifies (a) they are genuine eta-self-dual bivectors and (b) they are exactly
simple (tl gram = 0) with a COMPLEX proportionality constant, then tests the step-2 identity:
  - pure Lorentzian blocks
  - mixed Euclidean/Lorentzian blocks (nonsense physically; a pure algebra stress test)
  - orientation-reversed (det e < 0) cells, i.e. the branch the card excludes
  - the degenerate configuration Y = 0 where the corollaries divide by zero
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from examples.physics.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    simplicity_residual,
)
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
PAIR_INDEX = ((0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2))
PAIRS = ((0, 3), (1, 4), (2, 5))
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
EPS3 = np.zeros((3, 3, 3))
EPS3[0, 1, 2] = EPS3[1, 2, 0] = EPS3[2, 0, 1] = 1.0
EPS3[0, 2, 1] = EPS3[2, 1, 0] = EPS3[1, 0, 2] = -1.0


def two_form(u, v):
    m = np.outer(u, v) - np.outer(v, u)
    return np.array([m[i, j] for i, j in PAIR_INDEX], dtype=complex)


def wedge(a, b):
    return sum(a[i] * b[j] + a[j] * b[i] for i, j in PAIRS)


def gram(A):
    return np.array([[wedge(A[i], A[j]) for j in range(3)] for i in range(3)], dtype=complex)


def tl(M):
    return M - np.trace(M) / 3.0 * np.eye(3, dtype=complex)


def rel(x, y):
    s = float(np.linalg.norm(x))
    d = float(np.linalg.norm(x - y))
    return d / s if s > 0 else d


def to_matrix(b):
    m = np.zeros((4, 4), dtype=complex)
    for c, (i, j) in enumerate(PAIR_INDEX):
        m[i, j] = b[c]
        m[j, i] = -b[c]
    return m


def hodge(b):
    """(*B)^{mu nu} = (1/2) eps^{mu nu rho sigma} B_{rho sigma}, indices lowered with eta."""
    m = to_matrix(b)
    lowered = ETA @ m @ ETA
    eps = np.zeros((4, 4, 4, 4))
    from itertools import permutations
    for perm in permutations(range(4)):
        sign = 1
        p = list(perm)
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]:
                    sign = -sign
        eps[perm] = sign
    out = 0.5 * np.einsum("mnrs,rs->mn", eps, lowered)
    return np.array([out[i, j] for i, j in PAIR_INDEX], dtype=complex)


def lorentz_sd_triple(tetrad, sign=1j):
    rows = []
    for i in range(3):
        form = two_form(tetrad[0], tetrad[i + 1])
        for j in range(3):
            for k in range(3):
                if EPS3[i, j, k]:
                    form = form + 0.5 * sign * EPS3[i, j, k] * two_form(tetrad[j + 1], tetrad[k + 1])
        rows.append(form)
    return np.array(rows, dtype=complex)


def rand_so3(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def main() -> int:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED}

    # (a) certification.  For e = I and for genuine Lorentz frames e = L in SO(3,1) the triple is an
    # exact +i eigenvector of the Minkowski Hodge star; for a general tetrad it is self dual with
    # respect to the tetrad metric, not the background eta, so only simplicity is asserted there.
    worst_sd, worst_simple, consts = 0.0, 0.0, []
    B0 = lorentz_sd_triple(np.eye(4))
    for i3 in range(3):
        worst_sd = max(worst_sd, rel(hodge(B0[i3]), 1j * B0[i3]))
    for _ in range(20):
        K = 0.3 * rng.normal(size=(4, 4))
        A = K - ETA @ K.T @ ETA
        L, term = np.eye(4), np.eye(4)
        for k in range(1, 30):
            term = term @ A / k
            L = L + term
        assert float(np.linalg.norm(L.T @ ETA @ L - ETA)) < 1e-10
        B = lorentz_sd_triple(L)
        for i3 in range(3):
            worst_sd = max(worst_sd, rel(hodge(B[i3]), 1j * B[i3]))
        g = gram(B)
        worst_simple = max(worst_simple, float(np.linalg.norm(tl(g))) / float(np.linalg.norm(g)))
        consts.append(complex(g[0, 0]))
    res["lorentzian_triple_is_self_dual_and_simple"] = {
        "max_rel_err_hodge_eq_plus_i_B": worst_sd,
        "max_normalised_tl_gram": worst_simple,
        "example_gram_constants": [[c.real, c.imag] for c in consts[:3]],
        "ok": worst_sd <= 1e-12 and worst_simple <= 1e-12,
        "note": "gram = 2i delta_ij: the Lorentzian branch has an IMAGINARY simplicity constant",
    }

    # (a2) identity on blocks of genuine Lorentz frames (each cell a boosted, exactly simple triple)
    worst = 0.0
    for _ in range(20):
        cells = []
        for _v in range(6):
            K = 0.3 * rng.normal(size=(4, 4))
            A = K - ETA @ K.T @ ETA
            L, term = np.eye(4), np.eye(4)
            for k in range(1, 30):
                term = term @ A / k
                L = L + term
            cells.append(rand_so3(rng) @ lorentz_sd_triple(L))
        n = len(cells)
        base_l = lorentz_sd_triple(np.eye(4))
        eta_l = [x - base_l for x in cells]
        bar = sum(eta_l) / n
        lhs = tl(gram(sum(cells)))
        rhs = -n * sum(tl(gram(x - bar)) for x in eta_l)
        worst = max(worst, rel(lhs, rhs))
    res["identity_lorentz_frames"] = {"max_rel_err": worst, "ok": worst <= 1e-10}

    # (b) the step-2 identity over C, pure Lorentzian blocks, random SO(3) internal alignment
    rows = []
    worst = 0.0
    ref_l = lorentz_sd_triple(np.eye(4))
    for delta in (0.3, 0.05):
        for n in (2, 5, 17):
            w = 0.0
            for _ in range(20):
                cells = []
                while len(cells) < n:
                    e = np.eye(4) + delta * rng.normal(size=(4, 4))
                    if abs(float(np.linalg.det(e))) > 0.2:
                        cells.append(rand_so3(rng) @ lorentz_sd_triple(e))
                eta = [x - ref_l for x in cells]
                bar = sum(eta) / n
                lhs = tl(gram(sum(cells)))
                rhs = -n * sum(tl(gram(x - bar)) for x in eta)
                w = max(w, rel(lhs, rhs))
            rows.append({"delta": delta, "n": n, "max_rel_err": w})
            worst = max(worst, w)
    res["identity_lorentzian"] = {"max_rel_err": worst, "ok": worst <= 1e-10, "rows": rows}

    # (c) mixed Euclidean / Lorentzian block (pure algebra stress: only tl gram(X_v)=0 is used)
    worst = 0.0
    for _ in range(20):
        cells = []
        for k in range(6):
            e = np.eye(4) + 0.3 * rng.normal(size=(4, 4))
            if abs(float(np.linalg.det(e))) < 0.2:
                continue
            if k % 2:
                cells.append(np.asarray(geometric_self_dual_triple(e), dtype=complex))
            else:
                cells.append(lorentz_sd_triple(e))
        n = len(cells)
        if n < 2:
            continue
        base = np.asarray(geometric_self_dual_triple(np.eye(4)), dtype=complex)
        eta = [x - base for x in cells]
        bar = sum(eta) / n
        lhs = tl(gram(sum(cells)))
        rhs = -n * sum(tl(gram(x - bar)) for x in eta)
        worst = max(worst, rel(lhs, rhs))
    res["identity_mixed_signature"] = {"max_rel_err": worst, "ok": worst <= 1e-10}

    # (d) orientation-reversed branch (det e < 0): still exactly simple, identity still exact
    worst, traces, refused = 0.0, [], []
    base = geometric_self_dual_triple(np.eye(4))
    for _ in range(20):
        cells = []
        while len(cells) < 4:
            e = np.eye(4) + 0.3 * rng.normal(size=(4, 4))
            if float(np.linalg.det(e)) < -0.2:
                cand = geometric_self_dual_triple(e)
                try:
                    cells.append(optimal_internal_alignment(base, cand).aligned_candidate)
                    refused.append(False)
                except ValueError:
                    refused.append(True)
                    cells.append(rand_so3(rng) @ cand)
        traces.append(float(np.trace(np.real(gram(np.asarray(cells[0], dtype=complex))))))
        grp = [np.asarray(x, dtype=complex) for x in cells]
        eta = [x - np.asarray(base, dtype=complex) for x in grp]
        bar = sum(eta) / len(grp)
        lhs = tl(gram(sum(grp)))
        rhs = -len(grp) * sum(tl(gram(x - bar)) for x in eta)
        worst = max(worst, rel(lhs, rhs))
    res["identity_reversed_orientation_cells"] = {
        "max_rel_err": worst, "ok": worst <= 1e-10,
        "sample_gram_traces": traces[:5],
        "polar_alignment_refused_fraction": (sum(refused) / len(refused)) if refused else None,
        "note": "cells outside the positive branch stay exactly simple; the identity does not see it",
    }

    # (e) degeneracy: X_v in {Sigma_0, -Sigma_0} with equal counts gives Y = 0, so eps = 0/0
    sig0 = np.asarray(base, dtype=complex)
    cells = [sig0, -sig0, sig0, -sig0]
    Y = sum(cells)
    res["degenerate_block_Y_zero"] = {
        "norm_Y": float(np.linalg.norm(Y)),
        "norm_gram_Y": float(np.linalg.norm(gram(Y))),
        "each_cell_tl_gram": [float(np.linalg.norm(tl(gram(x)))) for x in cells],
        "eps_via_repo": float(simplicity_residual(np.real(sum(cells)))),
        "corollary_denominator": float(np.linalg.norm(gram(sig0 + 0.5 * (-2 * sig0)))),
        "note": "identity holds (0 = 0) but corollaries (a)/(b) are 0/0: nondegeneracy is assumed",
    }

    out = Path(__file__).resolve().parent / "c2_result.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
