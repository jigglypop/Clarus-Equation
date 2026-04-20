"""Tiny LM with std attention vs attention + dual-Laplacian residual.

Adds a DualLaplacianBlock after each transformer block and compares
validation PPL against baseline std attention. Tests whether the
user's dual-projection Laplacian diffusion proposal (5.2-5.6) helps
on actual language modeling at small scale.
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
import torch.nn as nn
import torch.nn.functional as F

from clarus.ce_laplacian import DualLaplacianBlock


# ---------------------------------------------------------------------------
# Model (baseline + CE-Laplacian variant)
# ---------------------------------------------------------------------------


class StdAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.register_buffer("tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def attn(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        mask = self.tril[:n, :n]
        s = s.masked_fill(~mask, float("-inf"))
        a = F.softmax(s, dim=-1)
        out = (a @ v).transpose(1, 2).contiguous().view(b, n, d)
        return self.o(out)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CELaplacianBlock(nn.Module):
    """Std attention + dual-Laplacian residual diffusion."""

    def __init__(self, d_model, n_heads, block, mode="wake"):
        super().__init__()
        self.std = StdAttnBlock(d_model, n_heads, block)
        self.ln_lap = nn.LayerNorm(d_model)
        self.lap = DualLaplacianBlock(
            d_model=d_model,
            d_lang=d_model // 2,
            d_grav=d_model // 2,
            sigma_grav=math.sqrt(d_model // 2),
            mode=mode,
            max_steps=1,
        )
        # per-block scalar on the laplacian contribution, starts at 0
        # so the model reduces to the std baseline at init.
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def forward(self, x):
        x = self.std(x)
        n = x.shape[1]
        mask = self.tril[:n, :n].unsqueeze(0).expand(x.shape[0], n, n)
        delta = self.lap(self.ln_lap(x), mask=mask) - self.ln_lap(x)
        return x + self.alpha * delta


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block, variant="std"):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block, d_model)
        if variant == "std":
            self.blocks = nn.ModuleList(
                [StdAttnBlock(d_model, n_heads, block) for _ in range(n_layers)]
            )
        elif variant == "ce_lap":
            self.blocks = nn.ModuleList(
                [CELaplacianBlock(d_model, n_heads, block) for _ in range(n_layers)]
            )
        else:
            raise ValueError(variant)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.block = block
        self.variant = variant

    def forward(self, idx):
        b, n = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(n, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Data + training (minimal, reused from bench_tiny_lm_ppl)
# ---------------------------------------------------------------------------


def load_corpus(paths):
    text = ""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text += f.read() + "\n"
    return text


def make_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


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


def train_one(data_train, data_val, vocab, variant, d_model, n_heads, n_layers, block,
              batch, steps, lr, seed):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, n_heads, n_layers, block, variant=variant)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    it = batch_iter(data_train, batch, block, seed=seed)
    history = []
    eval_every = max(steps // 6, 50)
    for s in range(1, steps + 1):
        x, y = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % eval_every == 0:
            vl = eval_loss(model, data_val, 8, batch, block)
            history.append({"step": s, "train": float(loss.item()), "val": vl})
            print(f"  [{variant:6s}] step {s:4d}  train {float(loss.item()):.4f}  val {vl:.4f}")
    return {
        "history": history,
        "time_sec": time.perf_counter() - t0,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--out", type=str,
                        default="examples/ai/results/laplacian_lm.json")
    args = parser.parse_args()

    paths = ["README.md", "docs/상수.md"]
    text = load_corpus(paths)
    stoi = make_vocab(text)
    data = encode(text, stoi)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}")

    results = {}
    for variant in ("std", "ce_lap"):
        print(f"\n=== training variant={variant} ===")
        r = train_one(train_data, val_data, vocab, variant,
                      d_model=args.d_model, n_heads=4, n_layers=args.n_layers,
                      block=args.block, batch=args.batch, steps=args.steps,
                      lr=3e-4, seed=0)
        r["final_val_ppl"] = math.exp(r["history"][-1]["val"])
        results[variant] = r
        print(f"  params: {r['n_params']/1e3:.1f}K  "
              f"time: {r['time_sec']:.1f}s  "
              f"final val PPL: {r['final_val_ppl']:.3f}")

    std_ppl = results["std"]["final_val_ppl"]
    print("\n=== verdict ===")
    for k, r in results.items():
        overhead = r["time_sec"] / results["std"]["time_sec"]
        gain = (std_ppl - r["final_val_ppl"]) / std_ppl * 100
        print(f"  {k:8s}  PPL {r['final_val_ppl']:7.3f}  "
              f"gain {gain:+.2f}%  time {overhead:.2f}x  "
              f"params {r['n_params']/1e3:.1f}K")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "vocab": vocab,
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
