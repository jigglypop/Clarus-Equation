"""Combined bench — Riemann rotary (Design 1) + Euler FFN variants, on top
of recursive k=3 self-referential blocks.

All variants share the winning recursion depth k=3 per the compendium
ClarusCell finding (bench_recursive_euler gave -11.5σ vs RoPE at k=3).
We now ablate two orthogonal axes on top:

  attention:  euler_ce  (π-phase + e-decay)          ← previous winner
              riemann   (γ_n = Riemann zeros)         ← Design 1

  ffn:        std, swiglu, euler_decay, euler_full

5 configs (to keep bench time manageable):
  euler_ce_std         baseline (previous winner, k=3)
  euler_ce_swiglu
  euler_ce_euler_full
  riemann_std
  riemann_euler_full
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from clarus.ce_euler import EulerCEAttention
from clarus.ce_ffn import make_ffn
from clarus.ce_riemann_attn import RiemannRotaryAttention


# ---- generic recursive block, attention + ffn pluggable ----


class RecBlock(nn.Module):
    def __init__(self, d_model, n_heads, block, attn_kind, ffn_kind, max_iters=3):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if attn_kind == "euler_ce":
            self.attn = EulerCEAttention(d_model, n_heads, block, learnable_gates=True)
        elif attn_kind == "riemann":
            self.attn = RiemannRotaryAttention(d_model, n_heads, block)
        else:
            raise ValueError(attn_kind)
        self.ffn = make_ffn(ffn_kind, d_model, mult=4)
        self.max_iters = max_iters

    def _step(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x):
        for _ in range(self.max_iters):
            x = self._step(x)
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block,
                 attn_kind, ffn_kind, max_iters):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            RecBlock(d_model, n_heads, block, attn_kind, ffn_kind, max_iters)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def load_docs(root, max_chars):
    text = ""
    paths = sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))
    paths.append(os.path.join(root, "..", "README.md"))
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                text += f.read() + "\n"
        except Exception:
            continue
        if len(text) >= max_chars:
            break
    return text[:max_chars]


def encode(text, stoi):
    return torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)


def batch_iter(data, batch, block, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        idx = torch.randint(0, len(data) - block - 1, (batch,), generator=g)
        x = torch.stack([data[i : i + block] for i in idx])
        y = torch.stack([data[i + 1 : i + 1 + block] for i in idx])
        yield x, y


@torch.no_grad()
def eval_loss(model, data, n_batches, batch, block):
    model.eval()
    total = 0.0
    it = batch_iter(data, batch, block, seed=42)
    for _ in range(n_batches):
        x, y = next(it)
        logits = model(x)
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return total / n_batches


def train_one(data_train, data_val, vocab, attn_kind, ffn_kind, max_iters,
              d_model, n_layers, block, batch, steps, lr, seed):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, 4, n_layers, block,
                   attn_kind=attn_kind, ffn_kind=ffn_kind, max_iters=max_iters)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    it = batch_iter(data_train, batch, block, seed=seed)
    t0 = time.perf_counter()
    for s in range(1, steps + 1):
        x, y = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val_loss = eval_loss(model, data_val, 12, batch, block)
    return {
        "val_ppl": math.exp(val_loss),
        "n_params": sum(p.numel() for p in model.parameters()),
        "time_sec": time.perf_counter() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--k_iters", type=int, default=3)
    p.add_argument("--corpus_chars", type=int, default=400_000)
    p.add_argument("--out", type=str, default="examples/ai/results/riemann_combo.json")
    args = p.parse_args()

    docs_root = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    text = load_docs(docs_root, args.corpus_chars)
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = encode(text, stoi)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}  |  k_iters={args.k_iters}\n")

    configs = [
        ("euler_ce_std",        "euler_ce", "std"),
        ("euler_ce_swiglu",     "euler_ce", "swiglu"),
        ("euler_ce_euler_full", "euler_ce", "euler_full"),
        ("riemann_std",         "riemann",  "std"),
        ("riemann_euler_full",  "riemann",  "euler_full"),
    ]

    results = {}
    for name, attn_kind, ffn_kind in configs:
        print(f"=== {name} ({args.seeds} seeds) ===")
        ppls, times = [], []
        params = None
        for seed in range(args.seeds):
            r = train_one(train_data, val_data, vocab, attn_kind, ffn_kind,
                          max_iters=args.k_iters,
                          d_model=args.d_model, n_layers=args.n_layers,
                          block=args.block, batch=args.batch,
                          steps=args.steps, lr=3e-4, seed=seed)
            ppls.append(r["val_ppl"]); times.append(r["time_sec"])
            params = r["n_params"]
        m = sum(ppls) / len(ppls)
        sd = (sum((x - m) ** 2 for x in ppls) / max(len(ppls) - 1, 1)) ** 0.5
        mean_time = sum(times) / len(times)
        results[name] = {
            "mean_ppl": m, "std_ppl": sd, "ppls": ppls,
            "params": params, "mean_time": mean_time,
            "attn_kind": attn_kind, "ffn_kind": ffn_kind,
        }
        print(f"  params {params/1e3:.1f}K   PPL {m:.3f} ± {sd:.3f}   time {mean_time:.1f}s")

    print("\n=== verdicts (negative z = variant is BETTER than euler_ce_std) ===")
    def zscore(a, b):
        m1, s1 = results[a]["mean_ppl"], results[a]["std_ppl"]
        m2, s2 = results[b]["mean_ppl"], results[b]["std_ppl"]
        se = (s1 ** 2 / args.seeds + s2 ** 2 / args.seeds) ** 0.5
        return (m1 - m2) / max(se, 1e-9)
    for name, _, _ in configs:
        if name == "euler_ce_std":
            continue
        zv = zscore(name, "euler_ce_std")
        d = results[name]["mean_ppl"] - results["euler_ce_std"]["mean_ppl"]
        v = "WIN " if zv < -1.0 else ("TIE " if abs(zv) < 1.0 else "LOSS")
        print(f"  [{v}] {name:22s} vs euler_ce_std  z = {zv:+.2f}  Δ = {d:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
