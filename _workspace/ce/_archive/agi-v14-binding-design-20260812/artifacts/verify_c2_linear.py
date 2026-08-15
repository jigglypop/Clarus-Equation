"""C2 verification: linear readout on the concatenation [b; c] cannot realize
y = sign(c^T W b) on all 32 cells, and its balanced held-out ceiling is < 1.

Checks:
 1. Analytic identity behind the impossibility proof (exact, tol 1e-12):
    sum_b sign(w_k . b) b = 4 w_k for all k, and sum_k w_k = 0.
 2. Numerical max accuracy of any affine classifier sign(v.b + beta_ctx)
    over the 32 noiseless cells (c one-hot => the c-part of a linear readout
    is exactly a per-context bias). Direction space covered by 200k random
    directions + integer grid [-4,4]^3 (every strict ordering of the 8 cube
    projections lies in a full-dimensional cone, hit w.h.p.); optimal
    threshold per context per direction computed exactly.
 3. Balanced-split consequence, seeds 9000..9015: max accuracy on the
    24 train cells and the held-out accuracy of the best train-fitting
    (direction, thresholds), reported as the achievable range over ties.
"""
from __future__ import annotations

import itertools
import numpy as np

from reality_stone.clarus.local_cloud_v13_benchmark import (
    cell_label, holdout_cells_balanced,
)

W = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def bits_of(idx: int) -> np.ndarray:
    return np.array([1 if (idx >> s) & 1 else -1 for s in (2, 1, 0)], dtype=float)


CUBE = np.array([bits_of(i) for i in range(8)])          # (8,3) index order 0..7
LABELS = np.array([[cell_label(k, i) for i in range(8)] for k in range(4)], float)  # (4,8)


def check_identity() -> None:
    tol = 1e-12
    for k in range(4):
        s = (np.sign(CUBE @ W[k])[:, None] * CUBE).sum(axis=0)
        assert np.max(np.abs(s - 4.0 * W[k])) <= tol, (k, s)
    assert np.max(np.abs(W.sum(axis=0))) <= tol
    print("[exact] sum_b sign(w_k.b) b = 4 w_k for all k;  sum_k w_k = 0   OK (tol 1e-12)")


def directions(n_random: int = 200_000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=(n_random, 3))
    grid = np.array([v for v in itertools.product(range(-4, 5), repeat=3) if any(v)], float)
    # tiny tie-break jitter on grid so orderings are strict
    grid = grid + rng.normal(scale=1e-9, size=grid.shape)
    return np.vstack([rand, grid])


def best_agreement_matrix(dirs: np.ndarray, pts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """For each direction v: max over threshold beta of
    #{ sign(v.p + beta) == y } over points pts (m,3), labels ys (m,).
    Vectorized over directions. Returns (n_dirs,) ints."""
    S = dirs @ pts.T                       # (n, m)
    order = np.argsort(S, axis=1)          # ascending
    y_sorted = ys[order]                   # (n, m)
    m = pts.shape[0]
    # threshold after position j (0..m): prefix predicted -1, suffix +1
    neg_prefix = np.concatenate(
        [np.zeros((len(dirs), 1)), np.cumsum(y_sorted == -1, axis=1)], axis=1)
    pos_total = np.sum(ys == 1)
    pos_suffix = pos_total - np.concatenate(
        [np.zeros((len(dirs), 1)), np.cumsum(y_sorted == 1, axis=1)], axis=1)
    agree = neg_prefix + pos_suffix        # (n, m+1)
    return agree.max(axis=1).astype(int)


def max_accuracy(cells_by_ctx: dict[int, list[int]], dirs: np.ndarray):
    total = np.zeros(len(dirs), dtype=int)
    for k in range(4):
        idxs = cells_by_ctx[k]
        if not idxs:
            continue
        pts = CUBE[idxs]
        ys = LABELS[k, idxs]
        total += best_agreement_matrix(dirs, pts, ys)
    best = int(total.max())
    arg = np.flatnonzero(total == best)
    return best, dirs[arg[:256]]


def heldout_acc_for_dir(v, train_by_ctx, held_by_ctx) -> float:
    """Optimal per-context thresholds on train; among train-optimal thresholds
    take the best heldout agreement (upper bound for this direction)."""
    correct = 0
    for k in range(4):
        tr_pts, tr_y = CUBE[train_by_ctx[k]], LABELS[k, train_by_ctx[k]]
        he_pts, he_y = CUBE[held_by_ctx[k]], LABELS[k, held_by_ctx[k]]
        s = tr_pts @ v
        cuts = [-np.inf] + list((np.sort(s)[:-1] + np.sort(s)[1:]) / 2.0) + [np.inf]
        best_tr, best_he = -1, 0
        for c in cuts:
            agree = int(np.sum(np.where(s > c, 1, -1) == tr_y))
            hagree = int(np.sum(np.where(he_pts @ v > c, 1, -1) == he_y))
            if agree > best_tr or (agree == best_tr and hagree > best_he):
                best_tr, best_he = agree, hagree
        correct += best_he
    return correct / 8.0


def main() -> None:
    check_identity()
    dirs = directions()
    full = {k: list(range(8)) for k in range(4)}
    best32, argdirs = max_accuracy(full, dirs)
    print(f"[all 32 cells] max affine-concat accuracy = {best32}/32 = {best32/32:.4f}")
    print(f"               example optimal v = {np.round(argdirs[0], 3)}")

    print(f"{'seed':<6}{'train max':>10}   heldout acc range over train-optimal dirs")
    for seed in range(9000, 9016):
        held = holdout_cells_balanced(seed)
        held_by_ctx = {k: [i for (kk, i) in held if kk == k] for k in range(4)}
        train_by_ctx = {k: [i for i in range(8) if i not in held_by_ctx[k]] for k in range(4)}
        best_tr, tie_dirs = max_accuracy(train_by_ctx, dirs)
        h = sorted({round(heldout_acc_for_dir(v, train_by_ctx, held_by_ctx), 4)
                    for v in tie_dirs})
        print(f"{seed:<6}{best_tr:>7}/24   min={h[0]:.4f} max={h[-1]:.4f}")


if __name__ == "__main__":
    main()
