"""Bench — FFN variants combined with the winning EulerCE attention.

Holds attention fixed to EulerCEAttention (the winning configuration
from bench_euler_rotary.py) and varies the FFN:

  std          GELU-FFN   (baseline)
  swiglu       SwiGLU-FFN (known-good)
  euler_decay  GELU * e-decay
  euler_phase  GELU * (1 + η cos(π h / τ))
  euler_full   GELU * π-phase * e-decay

5 seeds × 500 steps × 400K-char corpus.
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


class EulerCE_Block(nn.Module):
    """Transformer block: EulerCE attention + configurable FFN."""

    def __init__(self, d_model, n_heads, block, ffn_kind="std"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerCEAttention(d_model, n_heads, block, learnable_gates=True)
        self.ffn = make_ffn(ffn_kind, d_model, mult=4)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block, ffn_kind):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [EulerCE_Block(d_model, n_heads, block, ffn_kind) for _ in range(n_layers)]
        )
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


def train_one(data_train, data_val, vocab, ffn_kind, d_model, n_layers, block,
              batch, steps, lr, seed):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, 4, n_layers, block, ffn_kind=ffn_kind)
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

    # collect learned Euler-FFN scalars
    learned = []
    for blk in model.blocks:
        if hasattr(blk.ffn, "log_xi"):
            learned.append({"xi": float(blk.ffn.log_xi.exp().item())})
        if hasattr(blk.ffn, "log_tau"):
            learned[-1 if learned else 0] = {
                **(learned[-1] if learned else {}),
                "tau": float(blk.ffn.log_tau.exp().item()),
                "eta": float(blk.ffn.eta.item()),
            }

    return {
        "val_ppl": math.exp(val_loss),
        "n_params": sum(p.numel() for p in model.parameters()),
        "time_sec": time.perf_counter() - t0,
        "learned": learned,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--corpus_chars", type=int, default=400_000)
    p.add_argument("--out", type=str, default="examples/ai/results/euler_ffn.json")
    args = p.parse_args()

    docs_root = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    text = load_docs(docs_root, args.corpus_chars)
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = encode(text, stoi)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}\n")

    configs = ["std", "swiglu", "euler_decay", "euler_phase", "euler_full"]
    results = {}
    for ffn_kind in configs:
        print(f"=== ffn={ffn_kind} ({args.seeds} seeds) ===")
        ppls, times = [], []
        params = None
        learned_agg = []
        for seed in range(args.seeds):
            r = train_one(train_data, val_data, vocab, ffn_kind,
                          d_model=args.d_model, n_layers=args.n_layers,
                          block=args.block, batch=args.batch,
                          steps=args.steps, lr=3e-4, seed=seed)
            ppls.append(r["val_ppl"]); times.append(r["time_sec"])
            params = r["n_params"]
            if r["learned"]:
                learned_agg.append(r["learned"])
        m = sum(ppls) / len(ppls)
        sd = (sum((x - m) ** 2 for x in ppls) / max(len(ppls) - 1, 1)) ** 0.5
        mean_time = sum(times) / len(times)
        results[ffn_kind] = {
            "mean_ppl": m, "std_ppl": sd, "ppls": ppls,
            "params": params, "mean_time": mean_time,
            "learned_per_seed": learned_agg,
        }
        msg = f"  params {params/1e3:.1f}K  PPL {m:.3f} ± {sd:.3f}  time {mean_time:.1f}s"
        if learned_agg:
            # average each key across layers and seeds
            all_items = [item for seed in learned_agg for item in seed]
            keys = set().union(*[it.keys() for it in all_items])
            avg = {k: sum(it.get(k, 0) for it in all_items) / len(all_items) for k in keys}
            msg += "   learned=" + ", ".join(f"{k}={v:.3f}" for k, v in avg.items())
        print(msg)

    # Verdicts — all vs std
    print("\n=== verdicts (negative z = variant is BETTER than std GELU-FFN) ===")

    def z(a, b):
        m1, s1 = results[a]["mean_ppl"], results[a]["std_ppl"]
        m2, s2 = results[b]["mean_ppl"], results[b]["std_ppl"]
        se = (s1 ** 2 / args.seeds + s2 ** 2 / args.seeds) ** 0.5
        return (m1 - m2) / max(se, 1e-9)

    for v in configs:
        if v == "std":
            continue
        zv = z(v, "std")
        d = results[v]["mean_ppl"] - results["std"]["mean_ppl"]
        verdict = "WIN " if zv < -1.0 else ("TIE " if abs(zv) < 1.0 else "LOSS")
        print(f"  [{verdict}] {v:14s} vs std  z = {zv:+.2f}  Δ = {d:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
