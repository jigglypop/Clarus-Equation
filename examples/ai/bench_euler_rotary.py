"""Bench — Euler-bitfield rotary attention vs std absolute PE and RoPE.

Compares four transformer variants on a 400K-char multi-doc corpus:
  std_abs    d_model=96, absolute learned positional embeddings
  std_rope   d_model=96, RoPE (base=10000, standard)
  euler_hard d_model=96, Euler rotary, fixed all-5-bases bitfield
  euler_soft d_model=96, Euler rotary, learnable sigmoid bitfield

Shared:
  - 4 heads, 2 layers, block=64, batch=16
  - 500 steps, AdamW lr=3e-4
  - 5 seeds

Reports mean PPL, std, converged bitfield values (for euler_soft).
Theory check: does the model prefer certain Euler bases over others?
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

from clarus.ce_euler import (
    EULER_BASIS, EULER_BASIS_NAMES, EulerAttnBlock, EulerCEBlock,
    EulerRotaryAttention,
)


# ---- models ----


class StdAbsBlock(nn.Module):
    def __init__(self, d_model, n_heads, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.register_buffer("tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def attn(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        s = s.masked_fill(~self.tril[:n, :n], float("-inf"))
        a = F.softmax(s, dim=-1)
        return (a @ v).transpose(1, 2).contiguous().view(b, n, d)

    def forward(self, x):
        x = x + self.o(self.attn(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


class RoPEAttnBlock(nn.Module):
    """Standard RoPE (base=10000) rotary attention."""

    def __init__(self, d_model, n_heads, block, base=10000.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.register_buffer("tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("inv_freq", base ** (-k))
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

    def _rotate(self, x, cos, sin):
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    def attn(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        theta = self.pos[:n].view(1, 1, n, 1) * self.inv_freq.view(1, 1, 1, -1)
        cos = theta.cos(); sin = theta.sin()
        q = self._rotate(q, cos, sin); k = self._rotate(k, cos, sin)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        s = s.masked_fill(~self.tril[:n, :n], float("-inf"))
        a = F.softmax(s, dim=-1)
        return (a @ v).transpose(1, 2).contiguous().view(b, n, d)

    def forward(self, x):
        x = x + self.o(self.attn(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block, variant):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.use_abs_pos = (variant == "std_abs")
        if self.use_abs_pos:
            self.pos = nn.Embedding(block, d_model)
        if variant == "std_abs":
            blocks = [StdAbsBlock(d_model, n_heads, block) for _ in range(n_layers)]
        elif variant == "std_rope":
            blocks = [RoPEAttnBlock(d_model, n_heads, block) for _ in range(n_layers)]
        elif variant == "euler_hard":
            init_bits = torch.full((n_heads, 5), 2.0)  # sigmoid(2) ≈ 0.88 (all bases on)
            blocks = [
                type("Blk", (nn.Module,), {})()  # placeholder
                for _ in range(n_layers)
            ]
            # properly construct blocks
            blocks = [EulerAttnBlock(d_model, n_heads, block, softmax_bitfield=False)
                      for _ in range(n_layers)]
            for blk in blocks:
                blk.attn.bit_logits = init_bits.clone()
        elif variant == "euler_soft":
            blocks = [EulerAttnBlock(d_model, n_heads, block, softmax_bitfield=True)
                      for _ in range(n_layers)]
        elif variant == "euler_ce":
            # theory-correct: pi-phase rotary + e-decay bias
            blocks = [EulerCEBlock(d_model, n_heads, block, learnable_gates=True)
                      for _ in range(n_layers)]
        else:
            raise ValueError(variant)
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.variant = variant

    def forward(self, idx):
        b, n = idx.shape
        x = self.tok(idx)
        if self.use_abs_pos:
            x = x + self.pos(torch.arange(n, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---- data / training ----


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


def train_one(data_train, data_val, vocab, variant, d_model, n_layers, block,
              batch, steps, lr, seed):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, 4, n_layers, block, variant=variant)
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

    # For euler variants: extract converged bitfield
    bitfields = []
    for blk in model.blocks:
        if isinstance(blk, EulerAttnBlock):
            bitfields.append(blk.attn.bitfield().detach().cpu().tolist())

    return {
        "val_ppl": math.exp(val_loss),
        "val_loss": val_loss,
        "n_params": sum(p.numel() for p in model.parameters()),
        "time_sec": time.perf_counter() - t0,
        "bitfields": bitfields,
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
    p.add_argument("--out", type=str, default="examples/ai/results/euler_rotary.json")
    args = p.parse_args()

    docs_root = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    text = load_docs(docs_root, args.corpus_chars)
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = encode(text, stoi)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}\n")

    configs = ["std_abs", "std_rope", "euler_hard", "euler_soft", "euler_ce"]
    results = {}
    for variant in configs:
        print(f"=== {variant} ({args.seeds} seeds) ===")
        ppls, times = [], []
        params = None
        bitfield_agg = None
        for seed in range(args.seeds):
            r = train_one(train_data, val_data, vocab, variant,
                          d_model=args.d_model, n_layers=args.n_layers,
                          block=args.block, batch=args.batch,
                          steps=args.steps, lr=3e-4, seed=seed)
            ppls.append(r["val_ppl"])
            times.append(r["time_sec"])
            params = r["n_params"]
            if r["bitfields"]:
                b = torch.tensor(r["bitfields"])  # (n_layers, n_heads, 5)
                bitfield_agg = b if bitfield_agg is None else bitfield_agg + b
        if bitfield_agg is not None:
            bitfield_agg = bitfield_agg / args.seeds
        m = sum(ppls) / len(ppls)
        sd = (sum((x - m) ** 2 for x in ppls) / max(len(ppls) - 1, 1)) ** 0.5
        mean_time = sum(times) / len(times)
        results[variant] = {
            "mean_ppl": m, "std_ppl": sd, "ppls": ppls,
            "params": params, "mean_time": mean_time,
            "bitfield_avg": bitfield_agg.tolist() if bitfield_agg is not None else None,
        }
        print(f"  params {params/1e3:.1f}K   PPL {m:.3f} ± {sd:.3f}   time {mean_time:.1f}s")
        if bitfield_agg is not None:
            # show per-basis usage averaged over layers and heads
            per_basis = bitfield_agg.mean(dim=(0, 1))  # (5,)
            usage = dict(zip(EULER_BASIS_NAMES, per_basis.tolist()))
            print("  bitfield usage (avg over layers/heads/seeds):")
            for name, u in usage.items():
                print(f"     {name:8s} = {u:.4f}")

    # Verdicts
    print("\n=== verdicts (negative z = Euler is BETTER) ===")
    def z(a, b):
        m1, s1 = results[a]["mean_ppl"], results[a]["std_ppl"]
        m2, s2 = results[b]["mean_ppl"], results[b]["std_ppl"]
        se = (s1 ** 2 / args.seeds + s2 ** 2 / args.seeds) ** 0.5
        return (m1 - m2) / max(se, 1e-9)
    for euler in ("euler_hard", "euler_soft", "euler_ce"):
        for base in ("std_abs", "std_rope"):
            zv = z(euler, base)
            d = results[euler]["mean_ppl"] - results[base]["mean_ppl"]
            verdict = "WIN " if zv < -1.0 else ("TIE " if abs(zv) < 1.0 else "LOSS")
            print(f"  [{verdict}] {euler:12s} vs {base:9s}  z = {zv:+.2f}  Δ = {d:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
