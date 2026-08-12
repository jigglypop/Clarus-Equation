"""C1 verification: lossless-slot + bilinear readout oracle on the frozen V13 task.

Independent path: does NOT reuse any model code from v13 runs. Uses only the
frozen episode generator (reality_stone.clarus.local_cloud_v13_benchmark) as
the task oracle, then implements the C1 structure directly:

  - bits slot  s_b in R^4 : written once at tick 0 (oracle gate), content =
    gain * mean over the 4 repeated local channels of each bit row.
  - ctx  slot  s_c in R^4 : written once at tick 1 (oracle gate), content =
    gain * shared observation.
  - closed gate => s' = s exactly (eigenvalue exactly 1); implemented as
    literal no-op, so T-invariance is structural, not numerical.
  - readout  yhat = sign(s_c^T What s_b) with
      (a) What = true W padded with a zero column for the distractor, and
      (b) What fitted by logistic regression on outer-product features
          s_c (x) s_b from the 24-cell balanced train split.

Also runs exact float checks (rel err <= 1e-9):
  sum_k w_k = 0, ||W b||^2 = 12 for all 8 b, margin |w_c^T b| in {1,3},
and a Monte-Carlo check of the analytic error bound sigma_N^2 = 51 sigma^2 / 4.
"""
from __future__ import annotations

import numpy as np
from math import erf, sqrt

from reality_stone.clarus.local_cloud_benchmark import LocalCloudBenchmarkConfig
from reality_stone.clarus.local_cloud_v13_benchmark import generate_episodes_v2

W = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
GAIN = 0.5  # frozen v13 input gain; cancels in the sign (gain^2 > 0)


def Phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def slots_from_episode(ep) -> tuple[np.ndarray, np.ndarray]:
    """Oracle-gated slots. Gate closed on every tick except 0 (bits) / 1 (ctx);
    closed gate is a literal no-op, so the result is independent of T by
    construction (structural T-invariance)."""
    local0 = np.asarray(ep.observations[0].local, dtype=float)   # (4 rows, 4 copies)
    shared1 = np.asarray(ep.observations[1].shared, dtype=float)  # (4,)
    s_b = GAIN * local0.mean(axis=1)   # R^4  (3 bits + distractor), noise std GAIN*sigma/2
    s_c = GAIN * shared1               # R^4  one-hot context,       noise std GAIN*sigma
    return s_b, s_c


def readout_true(s_b: np.ndarray, s_c: np.ndarray) -> int:
    v = s_c @ W @ s_b[:3]
    return 1 if v > 0 else -1


def fit_bilinear_logistic(feats: np.ndarray, ys: np.ndarray,
                          steps: int = 30000, lr: float = 2.0) -> np.ndarray:
    """Plain full-batch logistic regression on 16 outer-product features."""
    w = np.zeros(feats.shape[1])
    n = len(ys)
    for _ in range(steps):
        m = ys * (feats @ w)
        g = feats.T @ (-ys / (1.0 + np.exp(m))) / n
        w -= lr * g
    return w


def run_panel(seed: int, T: int, sigma: float, split: str) -> dict:
    cfg = LocalCloudBenchmarkConfig(train_episodes=96, evaluation_episodes=256,
                                    episode_steps=T, noise_sigma=sigma)
    train = generate_episodes_v2(seed, 96, cfg, split="train", condition_split=split)
    evl = generate_episodes_v2(seed, 256, cfg, split="evaluation", condition_split=split)

    def feats_and_y(eps):
        F, Y = [], []
        for ep in eps:
            s_b, s_c = slots_from_episode(ep)
            F.append(np.outer(s_c, s_b).ravel())
            Y.append(ep.target)
        return np.asarray(F), np.asarray(Y, dtype=float)

    Ftr, Ytr = feats_and_y(train)
    Fev, Yev = feats_and_y(evl)

    # (a) true-W readout
    acc_true = float(np.mean([readout_true(*slots_from_episode(ep)) == ep.target for ep in evl]))
    # (b) fitted bilinear readout (identification from 24 train cells)
    w = fit_bilinear_logistic(Ftr, Ytr)
    acc_fit = float(np.mean(np.sign(Fev @ w) == Yev))
    return {"acc_true": acc_true, "acc_fit": acc_fit}


def exact_checks() -> None:
    tol = 1e-9
    assert np.max(np.abs(W.sum(axis=0))) <= tol, "sum_k w_k != 0"
    for idx in range(8):
        b = np.array([1 if (idx >> s) & 1 else -1 for s in (2, 1, 0)], dtype=float)
        n2 = float(np.dot(W @ b, W @ b))
        assert abs(n2 - 12.0) <= tol * 12.0, f"||Wb||^2 != 12 at {idx}: {n2}"
        for k in range(4):
            m = abs(float(W[k] @ b))
            assert m in (1.0, 3.0), f"margin not in {{1,3}}: {m}"
    g = float(np.dot(W.T @ np.ones(4), W.T @ np.ones(4)))
    print(f"[exact] sum_k w_k = 0, ||Wb||^2 = 12 (all 8 b), margins in {{1,3}}  OK (tol 1e-9); ||W^T 1||^2 = {g}")


def mc_error_rate(sigma: float, n: int = 2_000_000, seed: int = 1234) -> tuple[float, float]:
    """Monte-Carlo of the exact decision statistic vs analytic Gaussian bound.
    Draws (b, c) uniform, noise as in the task (bits noise std sigma/2 after
    4-copy averaging, ctx noise std sigma)."""
    rng = np.random.default_rng(seed)
    bidx = rng.integers(0, 8, size=n)
    bits = np.stack([(bidx >> s) & 1 for s in (2, 1, 0)], axis=1) * 2.0 - 1.0
    ctx = rng.integers(0, 4, size=n)
    eb = rng.normal(0.0, sigma / 2.0, size=(n, 3))
    ec = rng.normal(0.0, sigma, size=(n, 4))
    ytrue = np.sign(np.einsum('nk,nk->n', W[ctx], bits))
    stat = np.einsum('nk,nk->n', W[ctx] + ec @ W * 0, bits)  # placeholder, replaced below
    # full statistic: (e_c + ec)^T W (b + eb) = w_c.(b+eb) + ec^T W (b+eb)
    stat = np.einsum('nk,nk->n', W[ctx], bits + eb) + np.einsum('nk,nk->n', ec @ W, bits + eb)
    err = float(np.mean(np.sign(stat) != ytrue))
    sigma_n = sigma * sqrt(51.0 / 4.0)
    bound = 1.0 - Phi(1.0 / sigma_n)  # margin >= 1
    return err, bound


def main() -> None:
    exact_checks()
    for sigma in (0.04, 0.08):
        err, bound = mc_error_rate(sigma)
        print(f"[bound] sigma={sigma}: MC err={err:.3e}  Phi(-1/sigma_N)={bound:.3e}  (bound uses worst margin 1)")
    seeds = list(range(9000, 9008))
    print(f"{'panel':<28}{'acc(true W)':>14}{'acc(fitted W)':>16}")
    for T in (4, 8, 16):
        for sigma in (0.04, 0.08):
            for split in ("iid", "balanced"):
                a_t, a_f = [], []
                for s in seeds:
                    r = run_panel(s, T, sigma, split)
                    a_t.append(r["acc_true"]); a_f.append(r["acc_fit"])
                tag = f"T={T:<3} sig={sigma:<5} {split}"
                print(f"{tag:<28}{np.mean(a_t):>14.4f}{np.mean(a_f):>16.4f}")


if __name__ == "__main__":
    main()
