"""Benchmark: CE Metric-Family Attention vs standard softmax attention.

Tests three hypotheses from CE compendium section 6.B:
  (H1) MFA reproduces standard attention when omega_grav = 0.
  (H2) Mode switching (WAKE <-> NREM) reshapes attention toward event
       clusters rather than linear-sequence neighbors.
  (H3) On a synthetic "event graph + linearization" task, MFA recovers
       the latent event structure better than pure language attention.

This is a toy prototype; it is CPU-friendly and produces JSON output
that downstream scripts (scorecard-style) can aggregate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F

from clarus.ce_softmax import (
    CESoftmaxAttention,
    grav_attention,
    grav_scores,
    lang_attention,
    lang_scores,
    metric_family_attention,
    mode_gate,
)
import torch.nn.functional as _F


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# H1 — equivalence check
# ---------------------------------------------------------------------------


def check_lang_equivalence(b: int = 2, n: int = 16, d: int = 32) -> dict:
    """omega_grav = 0 must reproduce plain softmax attention."""
    q = torch.randn(b, n, d)
    k = torch.randn(b, n, d)
    v = torch.randn(b, n, d)
    a_std = lang_attention(q, k)
    out_std = torch.matmul(a_std, v)

    gate = mode_gate("wake")
    # override: force pure lang
    from clarus.ce_softmax import ModeGate

    pure = ModeGate(omega_lang=1.0, omega_grav=0.0)
    out_mfa = metric_family_attention(q, k, v, gate=pure)
    err = (out_mfa - out_std).abs().max().item()
    return {"max_abs_err": err, "passed": err < 1e-5}


# ---------------------------------------------------------------------------
# H2 — mode-switch attention reshape
# ---------------------------------------------------------------------------


def attention_entropy(a: torch.Tensor) -> float:
    """Average entropy of attention rows (nats)."""
    eps = 1e-12
    h = -(a * (a + eps).log()).sum(dim=-1)
    return float(h.mean().item())


def check_mode_switch(b: int = 4, n: int = 32, d: int = 64) -> dict:
    """Entropy of attention rows should differ across modes."""
    q = torch.randn(b, n, d)
    k = torch.randn(b, n, d)

    a_wake = lang_attention(q, k)

    g_wake = mode_gate("wake")
    g_nrem = mode_gate("nrem")
    a_grav = grav_attention(k, sigma=math.sqrt(d))
    a_mix_wake = g_wake.omega_lang * a_wake + g_wake.omega_grav * a_grav
    a_mix_nrem = g_nrem.omega_lang * a_wake + g_nrem.omega_grav * a_grav

    h_wake = attention_entropy(a_mix_wake)
    h_nrem = attention_entropy(a_mix_nrem)

    return {
        "entropy_wake": h_wake,
        "entropy_nrem": h_nrem,
        "entropy_delta": h_nrem - h_wake,
        "omega_grav_wake": g_wake.omega_grav,
        "omega_grav_nrem": g_nrem.omega_grav,
    }


# ---------------------------------------------------------------------------
# H3 — synthetic event graph recovery
# ---------------------------------------------------------------------------


def make_event_task(
    n_events: int = 8,
    members_per_event: int = 4,
    d: int = 32,
    noise: float = 0.25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build tokens from event clusters, then linearize by permutation.

    Returns:
        tokens:     (n, d) — linearized sequence embeddings
        event_id:   (n,)   — ground-truth event membership
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_events, d, generator=g)
    n = n_events * members_per_event
    event_id = torch.arange(n_events).repeat_interleave(members_per_event)
    tokens = centers[event_id] + noise * torch.randn(n, d, generator=g)

    perm = torch.randperm(n, generator=g)
    return tokens[perm], event_id[perm]


def event_recovery_score(a: torch.Tensor, event_id: torch.Tensor) -> float:
    """Fraction of attention mass that lands on the same-event tokens.

    a: (n, n) attention matrix (rows sum to 1).
    event_id: (n,) ground truth.
    """
    same = (event_id.unsqueeze(0) == event_id.unsqueeze(1)).float()
    same.fill_diagonal_(0.0)
    mass = (a * same).sum(dim=-1)  # per-row same-event mass
    return float(mass.mean().item())


def check_event_recovery(
    n_events: int = 8,
    members_per_event: int = 4,
    d: int = 32,
    noise: float = 0.25,
    n_trials: int = 20,
) -> dict:
    scores = {
        "wake_convex": [], "nrem_convex": [],
        "wake_logit": [], "nrem_logit": [],
        "lang_only": [], "grav_only": [],
    }
    sigma = math.sqrt(d)
    for trial in range(n_trials):
        tokens, event_id = make_event_task(n_events, members_per_event, d, noise, seed=trial)
        q = tokens.unsqueeze(0)
        k = tokens.unsqueeze(0)

        # raw logits
        s_lang = lang_scores(q, k).squeeze(0)
        s_grav = grav_scores(k, sigma=sigma).squeeze(0)

        # individual attentions
        a_lang = _F.softmax(s_lang, dim=-1)
        a_grav = _F.softmax(s_grav, dim=-1)

        g_wake = mode_gate("wake")
        g_nrem = mode_gate("nrem")

        # convex
        a_wake_c = g_wake.omega_lang * a_lang + g_wake.omega_grav * a_grav
        a_nrem_c = g_nrem.omega_lang * a_lang + g_nrem.omega_grav * a_grav
        # logit
        a_wake_l = _F.softmax(g_wake.omega_lang * s_lang + g_wake.omega_grav * s_grav, dim=-1)
        a_nrem_l = _F.softmax(g_nrem.omega_lang * s_lang + g_nrem.omega_grav * s_grav, dim=-1)

        scores["wake_convex"].append(event_recovery_score(a_wake_c, event_id))
        scores["nrem_convex"].append(event_recovery_score(a_nrem_c, event_id))
        scores["wake_logit"].append(event_recovery_score(a_wake_l, event_id))
        scores["nrem_logit"].append(event_recovery_score(a_nrem_l, event_id))
        scores["lang_only"].append(event_recovery_score(a_lang, event_id))
        scores["grav_only"].append(event_recovery_score(a_grav, event_id))

    def stat(xs):
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
        return {"mean": m, "std": var ** 0.5, "n": n}

    return {k: stat(v) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Throughput comparison
# ---------------------------------------------------------------------------


def bench_throughput(b: int = 4, n: int = 128, d: int = 128, n_iters: int = 50) -> dict:
    q = torch.randn(b, n, d)
    k = torch.randn(b, n, d)
    v = torch.randn(b, n, d)

    # warmup
    for _ in range(3):
        lang_attention(q, k)
        metric_family_attention(q, k, v, gate=mode_gate("wake"))

    t0 = time.perf_counter()
    for _ in range(n_iters):
        a = lang_attention(q, k)
        _ = torch.matmul(a, v)
    t_std = (time.perf_counter() - t0) / n_iters

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = metric_family_attention(q, k, v, gate=mode_gate("wake"))
    t_mfa = (time.perf_counter() - t0) / n_iters

    return {
        "std_ms": t_std * 1000.0,
        "mfa_ms": t_mfa * 1000.0,
        "overhead_x": t_mfa / t_std if t_std > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    seed_everything(0)

    print("== H1: lang equivalence (omega_grav=0) ==")
    h1 = check_lang_equivalence()
    print(json.dumps(h1, indent=2))

    print("\n== H2: mode-switch entropy ==")
    h2 = check_mode_switch()
    print(json.dumps(h2, indent=2))

    print("\n== H3: synthetic event recovery (higher = better) ==")
    h3 = check_event_recovery(n_trials=args.trials)
    print(json.dumps(h3, indent=2))

    print("\n== Throughput ==")
    thr = bench_throughput()
    print(json.dumps(thr, indent=2))

    summary = {
        "h1_equivalence": h1,
        "h2_mode_switch": h2,
        "h3_event_recovery": h3,
        "throughput": thr,
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")

    # Decision summary
    def sigma_vs_lang(key):
        return (h3[key]["mean"] - h3["lang_only"]["mean"]) / max(h3["lang_only"]["std"], 1e-6)

    print(f"\nsigma vs lang_only:")
    for key in ("wake_convex", "nrem_convex", "wake_logit", "nrem_logit", "grav_only"):
        print(f"  {key:14s} {sigma_vs_lang(key):+.2f} sigma")
    print(f"\nThroughput overhead: {thr['overhead_x']:.2f}x")


if __name__ == "__main__":
    main()
