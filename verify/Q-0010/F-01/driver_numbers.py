"""Q-0010 F-01 exact numbers: orbit-tangent alignment projector, dimension budget, kernel reduction.

Pure linear algebra (no Monte Carlo, no tetrad simulation).  Every number written by this script is
computed from the derivative of e -> Sigma(e) at e = I and from the Plebanski traceless-gram
quadratic form; the block residual pipeline itself is NOT used here.  Output: numbers.json.

Conventions (frozen with the card derivations/Q-0010/F-01.formula.md, 2026-09-02):
  label space   : 4x4 tetrad increment xi (16 real dimensions), Frobenius inner product
  basis split   : 1 (trace/scale) + 3 (self-dual so(3)) + 3 (anti-self-dual so(3)) + 9 (sym traceless)
  dPhi          : derivative at e = I of the self-dual triple map, exact (Phi is quadratic in e)
  P_align       : orthogonal projector on span{scale, so(3) generators that rotate the triple} (4 dim)
  Pi_rot        : projector on the 3 rotation directions (removed by the per-cell polar alignment
                  in examples/physics/urbantke_shape_matching_rg.optimal_internal_alignment)
  M_a           : matrix of the quadratic form xi -> <E_a, traceless gram(dPhi xi)>, {E_a} an
                  orthonormal basis of traceless symmetric 3x3 (5 elements)
  Mt_a          : (1 - Pi_rot) M_a (1 - Pi_rot), the generator seen by the polar-aligned pipeline
Residual-generator budget:  c0 = sum_a tr(Mt_a Mt_a) = c1 + 2 c2 + c3 with
  c1 = sum_a tr(Mt_a P Mt_a P), c2 = sum_a tr(Mt_a P Mt_a (1-P)), c3 = sum_a tr(Mt_a(1-P) Mt_a(1-P)).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    wedge_scalar,
)

SEED = 20260902
FD_STEP = 1.0e-3
RANK_TOL = 1.0e-8
ZERO_TOL = 1.0e-10
# F-02 exact Cayley table (verify/Q-0008/F-02/predictions.json), used only for the control value
E_D_128 = 134587.1450461609
E_TRHK_128 = 822.7392437084596


def orthonormal_label_basis():
    """16 orthonormal 4x4 matrices: scale(1), self-dual(3), anti-self-dual(3), sym-traceless(9)."""
    basis = []
    groups = {"scale": [], "sd": [], "asd": [], "sym": []}

    basis.append(np.eye(4) / 2.0)
    groups["scale"].append(0)

    def anti(mu, nu):
        out = np.zeros((4, 4))
        out[mu, nu] = 1.0
        out[nu, mu] = -1.0
        return out / np.sqrt(2.0)

    cyc = ((1, 2, 3), (2, 3, 1), (3, 1, 2))
    for i, j, k in cyc:
        groups["sd"].append(len(basis))
        basis.append((anti(0, i) + anti(j, k)) / np.sqrt(2.0))
    for i, j, k in cyc:
        groups["asd"].append(len(basis))
        basis.append((anti(0, i) - anti(j, k)) / np.sqrt(2.0))
    for mu in range(4):
        for nu in range(mu + 1, 4):
            sym = np.zeros((4, 4))
            sym[mu, nu] = 1.0 / np.sqrt(2.0)
            sym[nu, mu] = 1.0 / np.sqrt(2.0)
            groups["sym"].append(len(basis))
            basis.append(sym)
    for matrix in (
        np.diag([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0),
        np.diag([1.0, 1.0, -2.0, 0.0]) / np.sqrt(6.0),
        np.diag([1.0, 1.0, 1.0, -3.0]) / np.sqrt(12.0),
    ):
        groups["sym"].append(len(basis))
        basis.append(matrix)
    return np.asarray(basis), groups


def dphi(direction, step=FD_STEP):
    """Derivative of e -> Sigma(e) at e = I; Phi is quadratic so the central difference is exact."""
    plus = geometric_self_dual_triple(np.eye(4) + step * direction)
    minus = geometric_self_dual_triple(np.eye(4) - step * direction)
    return (plus - minus) / (2.0 * step)


def gram_bilinear(first, second):
    return np.array(
        [
            [0.5 * (wedge_scalar(first[i], second[j]) + wedge_scalar(second[i], first[j])) for j in range(3)]
            for i in range(3)
        ]
    )


def traceless(matrix):
    return matrix - np.trace(matrix) / 3.0 * np.eye(3)


def traceless_sym_basis():
    out = []
    for i in range(3):
        for j in range(i + 1, 3):
            e = np.zeros((3, 3))
            e[i, j] = 1.0 / np.sqrt(2.0)
            e[j, i] = 1.0 / np.sqrt(2.0)
            out.append(e)
    out.append(np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0))
    out.append(np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0))
    return np.asarray(out)


def rotation_tangent(triple):
    """(L_a . B)_i = eps_{aij} B_j: the three internal SO(3) tangent directions at B."""
    plus = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    minus = ((0, 2, 1), (2, 1, 0), (1, 0, 2))
    out = []
    for a in range(3):
        vec = np.zeros_like(triple)
        for i in range(3):
            for j in range(3):
                eps = 1.0 if (a, i, j) in plus else (-1.0 if (a, i, j) in minus else 0.0)
                if eps:
                    vec[i] = vec[i] + eps * triple[j]
        out.append(vec)
    return np.asarray(out)


def flat(arrays):
    return np.asarray([np.asarray(a).reshape(-1) for a in arrays])


def main():
    basis, groups = orthonormal_label_basis()
    gram_basis = flat(basis) @ flat(basis).T
    b0 = geometric_self_dual_triple(np.eye(4))
    images = np.asarray([dphi(b) for b in basis])

    rank_dphi = int(np.linalg.matrix_rank(flat(images), tol=RANK_TOL))
    image_norms = {name: [float(np.linalg.norm(images[i])) for i in idx] for name, idx in groups.items()}

    rot = rotation_tangent(b0)
    orbit_tangent = np.concatenate([flat(b0[None, :, :]), flat(rot)], axis=0)
    dim_orbit = int(np.linalg.matrix_rank(orbit_tangent, tol=RANK_TOL))

    def rank_with(vectors):
        return int(np.linalg.matrix_rank(np.concatenate([orbit_tangent, flat(vectors)], axis=0), tol=RANK_TOL))

    sd_rank = rank_with(images[groups["sd"]])
    asd_rank = rank_with(images[groups["asd"]])
    kernel_group = "asd" if sum(image_norms["asd"]) < sum(image_norms["sd"]) else "sd"
    rot_group = "sd" if kernel_group == "asd" else "asd"

    p_align = np.zeros((16, 16))
    for i in groups["scale"] + groups[rot_group]:
        p_align[i, i] = 1.0
    pi_rot = np.zeros((16, 16))
    for i in groups[rot_group]:
        pi_rot[i, i] = 1.0
    eye16 = np.eye(16)

    forms = []
    for e_a in traceless_sym_basis():
        matrix = np.zeros((16, 16))
        for p in range(16):
            for q in range(16):
                matrix[p, q] = float(np.sum(e_a * traceless(gram_bilinear(images[p], images[q]))))
        forms.append(matrix)
    forms = np.asarray(forms)
    tilde = np.asarray([(eye16 - pi_rot) @ m @ (eye16 - pi_rot) for m in forms])

    scale_col = groups["scale"][0]
    scale_row = float(max(np.linalg.norm(m[:, scale_col]) for m in forms))
    rot_row = float(max(np.linalg.norm(m[:, groups[rot_group]]) for m in forms))
    norm_m = float(np.sqrt(sum(np.sum(m * m) for m in tilde)))
    align_leak = float(np.sqrt(sum(np.sum((m @ p_align) ** 2) for m in tilde)))
    floor_iid = float(max(abs(np.trace(m)) for m in tilde))
    floor_fold = float(max(abs(np.trace(m @ (eye16 - p_align))) for m in tilde))

    def budget(projector):
        q = eye16 - projector
        c0 = float(sum(np.trace(m @ m) for m in tilde))
        c1 = float(sum(np.trace(m @ projector @ m @ projector) for m in tilde))
        c2 = float(sum(np.trace(m @ projector @ m @ q) for m in tilde))
        c3 = float(sum(np.trace(m @ q @ m @ q) for m in tilde))
        return {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "budget_gap": c0 - (c1 + 2 * c2 + c3),
            "c1_over_c0": c1 / c0,
            "c2_over_c0": c2 / c0,
            "c3_over_c0": c3 / c0,
        }

    align_budget = budget(p_align)
    rng = np.random.default_rng(SEED)
    q_rand, _ = np.linalg.qr(rng.normal(size=(16, 4)))
    p_rand = q_rand @ q_rand.T
    rand_budget = budget(p_rand)

    n = 128

    def ratio_128(bud):
        value = (bud["c1"] * E_D_128 + 2 * bud["c2"] * E_TRHK_128 + bud["c3"] * (n - 1)) / (
            align_budget["c0"] * (n - 1)
        )
        return float(np.sqrt(max(value, 0.0)))

    sizes = (8, 16, 32, 64, 128)
    iid_slope = float(np.polyfit(np.log(sizes), np.log([np.sqrt(m - 1) / m for m in sizes]), 1)[0])

    out = {
        "card": "F-01",
        "question": "Q-0010",
        "seed": SEED,
        "basis_orthonormal_max_err": float(np.max(np.abs(gram_basis - np.eye(16)))),
        "rank_dphi": rank_dphi,
        "dim_orbit_tangent": dim_orbit,
        "dim_folded_visible": rank_dphi - dim_orbit,
        "kernel_group": kernel_group,
        "rotation_group": rot_group,
        "image_norms": image_norms,
        "rank_orbit_plus_sd": sd_rank,
        "rank_orbit_plus_asd": asd_rank,
        "M_scale_column_norm": scale_row,
        "M_rot_column_norm": rot_row,
        "Mtilde_norm": norm_m,
        "Mtilde_P_align_leak": align_leak,
        "Mtilde_P_align_leak_relative": align_leak / norm_m,
        "floor_trace_iid": floor_iid,
        "floor_trace_folded": floor_fold,
        "align_budget": align_budget,
        "rand4_budget": rand_budget,
        "align_ratio_128_theory": ratio_128(align_budget),
        "rand4_ratio_128_theory": ratio_128(rand_budget),
        "iid_lattice_slope_8_128": iid_slope,
        "entropy_gap_nats_at_eps_res_1e-2": float((rank_dphi - dim_orbit) * np.log(100.0)),
    }
    (HERE / "numbers.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
