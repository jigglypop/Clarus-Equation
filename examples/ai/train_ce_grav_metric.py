"""Amendment A — learn G^(grav) = L L^T via contrastive loss.

H3 in bench_ce_softmax was falsified with identity G. The CE compendium
(6.B.1) specifies G^(grav) as a learnable metric tensor. We train L
such that same-event tokens have small squared Mahalanobis distance
and different-event tokens have large distance.

Loss (InfoNCE-style, per-token):
    L_i = -log( sum_{j in same event} exp(-d_G^2/2s^2)
               / sum_j exp(-d_G^2/2s^2) )

After training, re-run H3 and compare to identity-G baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from clarus.ce_softmax import (
    grav_attention,
    lang_attention,
    mode_gate,
)


def make_event_task(n_events, members_per_event, d, noise, seed):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_events, d, generator=g)
    n = n_events * members_per_event
    event_id = torch.arange(n_events).repeat_interleave(members_per_event)
    tokens = centers[event_id] + noise * torch.randn(n, d, generator=g)
    perm = torch.randperm(n, generator=g)
    return tokens[perm], event_id[perm]


def event_recovery_score(a, event_id):
    same = (event_id.unsqueeze(0) == event_id.unsqueeze(1)).float()
    same.fill_diagonal_(0.0)
    mass = (a * same).sum(dim=-1)
    return float(mass.mean().item())


def contrastive_loss(z, event_id, L, sigma):
    """InfoNCE over (z, event_id) with learned metric G = L L^T."""
    zp = z @ L  # (n, r)
    sq = (zp * zp).sum(dim=-1, keepdim=True)
    d2 = sq + sq.T - 2.0 * zp @ zp.T
    d2 = d2.clamp_min(0.0)
    scores = -d2 / (2.0 * sigma * sigma)

    same = (event_id.unsqueeze(0) == event_id.unsqueeze(1))
    same.fill_diagonal_(False)

    log_probs = torch.log_softmax(scores, dim=-1)
    # for each i, average log-prob of same-event members
    mask = same.float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    loss = -(log_probs * mask).sum(dim=-1) / denom
    return loss.mean()


def train_metric(
    d: int = 32,
    rank: int = 16,
    n_events: int = 8,
    members_per_event: int = 4,
    noise: float = 0.25,
    sigma: float = None,
    epochs: int = 300,
    lr: float = 0.05,
    n_train_tasks: int = 64,
    seed_base: int = 1000,
) -> torch.Tensor:
    if sigma is None:
        sigma = math.sqrt(d)
    L = nn.Parameter(torch.randn(d, rank) * 0.1)
    opt = torch.optim.Adam([L], lr=lr)

    losses = []
    for ep in range(epochs):
        total = 0.0
        for t in range(n_train_tasks):
            tokens, event_id = make_event_task(
                n_events, members_per_event, d, noise, seed=seed_base + ep * n_train_tasks + t
            )
            loss = contrastive_loss(tokens, event_id, L, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        losses.append(total / n_train_tasks)
    return L.detach(), losses


def eval_h3(
    L_learned,
    n_events=8,
    members_per_event=4,
    d=32,
    noise=0.25,
    n_trials=30,
    sigma=None,
):
    if sigma is None:
        sigma = math.sqrt(d)

    scores = {k: [] for k in ("wake_id", "nrem_id", "wake_learned", "nrem_learned",
                              "lang_only", "grav_id", "grav_learned")}
    for trial in range(n_trials):
        tokens, event_id = make_event_task(n_events, members_per_event, d, noise,
                                           seed=10_000 + trial)
        q = tokens.unsqueeze(0)
        k = tokens.unsqueeze(0)

        a_lang = lang_attention(q, k).squeeze(0)
        a_grav_id = grav_attention(k, sigma=sigma).squeeze(0)
        a_grav_lr = grav_attention(k, sigma=sigma, L=L_learned).squeeze(0)

        g_w = mode_gate("wake")
        g_n = mode_gate("nrem")

        scores["wake_id"].append(event_recovery_score(
            g_w.omega_lang * a_lang + g_w.omega_grav * a_grav_id, event_id))
        scores["nrem_id"].append(event_recovery_score(
            g_n.omega_lang * a_lang + g_n.omega_grav * a_grav_id, event_id))
        scores["wake_learned"].append(event_recovery_score(
            g_w.omega_lang * a_lang + g_w.omega_grav * a_grav_lr, event_id))
        scores["nrem_learned"].append(event_recovery_score(
            g_n.omega_lang * a_lang + g_n.omega_grav * a_grav_lr, event_id))
        scores["lang_only"].append(event_recovery_score(a_lang, event_id))
        scores["grav_id"].append(event_recovery_score(a_grav_id, event_id))
        scores["grav_learned"].append(event_recovery_score(a_grav_lr, event_id))

    def stat(xs):
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
        return {"mean": m, "std": var ** 0.5, "n": n}

    return {k: stat(v) for k, v in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--out", type=str,
                        default="examples/ai/results/ce_grav_trained.json")
    args = parser.parse_args()

    torch.manual_seed(0)

    print(f"== training G = L L^T (rank={args.rank}, epochs={args.epochs}) ==")
    L, losses = train_metric(epochs=args.epochs, rank=args.rank)
    print(f"  final loss={losses[-1]:.4f} (start {losses[0]:.4f})")

    print("\n== H3 re-evaluation ==")
    h3 = eval_h3(L, n_trials=args.trials)
    for k, s in h3.items():
        print(f"  {k:16s} mean={s['mean']:.4f}  std={s['std']:.4f}")

    # sigma improvement vs lang_only
    lang = h3["lang_only"]
    nrem_lr = h3["nrem_learned"]
    sigma_diff = (nrem_lr["mean"] - lang["mean"]) / max(lang["std"], 1e-6)
    print(f"\nNREM+learned vs lang_only: {sigma_diff:+.2f} sigma")

    summary = {
        "config": {"rank": args.rank, "epochs": args.epochs, "trials": args.trials},
        "training_loss_first_last": [losses[0], losses[-1]],
        "h3": h3,
        "nrem_learned_vs_lang_sigma": sigma_diff,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
