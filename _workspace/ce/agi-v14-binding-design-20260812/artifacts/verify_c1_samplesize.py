"""Supplement: the fitted-bilinear gap on heldout is finite-sample only.
Train-episode count sweep at the hardest point (T=8, sigma=0.08, balanced)."""
import numpy as np
from reality_stone.clarus.local_cloud_benchmark import LocalCloudBenchmarkConfig
from reality_stone.clarus.local_cloud_v13_benchmark import generate_episodes_v2
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "c1", pathlib.Path(__file__).with_name("verify_c1_oracle.py"))
c1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c1)

seeds = list(range(9000, 9008))
for n_train in (96, 480, 1920):
    accs = []
    for s in seeds:
        cfg = LocalCloudBenchmarkConfig(train_episodes=max(96, n_train), evaluation_episodes=256,
                                        episode_steps=8, noise_sigma=0.08)
        tr = generate_episodes_v2(s, n_train, cfg, split="train", condition_split="balanced")
        ev = generate_episodes_v2(s, 256, cfg, split="evaluation", condition_split="balanced")
        F = lambda eps: (np.array([np.outer(*(lambda t: (t[1], t[0]))(c1.slots_from_episode(e))).ravel() for e in eps]),
                         np.array([e.target for e in eps], float))
        Ftr, Ytr = F(tr); Fev, Yev = F(ev)
        w = c1.fit_bilinear_logistic(Ftr, Ytr)
        accs.append(float(np.mean(np.sign(Fev @ w) == Yev)))
    print(f"n_train={n_train:>5}  heldout acc mean={np.mean(accs):.4f} min={min(accs):.4f}")
