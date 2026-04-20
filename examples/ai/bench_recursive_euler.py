"""Bench — self-referential (ClarusCell-style while-loop) EulerCEBlock.

Tests the CE bootstrap-fixed-point idea: same block applied to its own
output k times per forward pass. Compared against single-pass euler_ce
(previous winner) and RoPE baseline.

Configs:
  std_rope           baseline RoPE
  euler_ce_k1        single-pass EulerCEBlock (= previous winner)
  euler_ce_k2        fixed 2-step recursion
  euler_ce_k3        fixed 3-step recursion
  euler_ce_halt      while-loop with tol=5e-3, max 6 iters
  euler_ce_k2_fp     2-step with fixed-point regularizer (lambda=0.1)

Reports PPL mean ± std over seeds, plus for halt variant: mean halt
depth (where the self-recursion actually stopped).
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
    EulerCEBlock, RecursiveEulerCEBlock, fixed_point_loss,
)


# ---- models ----


class RoPEAttnBlock(nn.Module):
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
    def __init__(self, vocab, d_model, n_heads, n_layers, block, variant,
                 max_iters=1, tol=None):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.variant = variant
        self.fp_lambda = 0.0
        if variant == "std_rope":
            blocks = [RoPEAttnBlock(d_model, n_heads, block) for _ in range(n_layers)]
        elif variant == "euler_ce_k1":
            blocks = [EulerCEBlock(d_model, n_heads, block) for _ in range(n_layers)]
        elif variant.startswith("euler_ce_k") or variant == "euler_ce_halt" or variant.endswith("_fp"):
            blocks = [RecursiveEulerCEBlock(d_model, n_heads, block,
                                            max_iters=max_iters, tol=tol)
                      for _ in range(n_layers)]
        else:
            raise ValueError(variant)
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, return_h=False):
        b, n = idx.shape
        x = self.tok(idx)
        h_list = []
        for blk in self.blocks:
            x = blk(x)
            if return_h:
                h_list.append(x)
        return self.head(self.ln_f(x)) if not return_h else (self.head(self.ln_f(x)), h_list)


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
              batch, steps, lr, seed, max_iters=1, tol=None, fp_lambda=0.0):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, 4, n_layers, block, variant=variant,
                   max_iters=max_iters, tol=tol)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    it = batch_iter(data_train, batch, block, seed=seed)
    t0 = time.perf_counter()
    for s in range(1, steps + 1):
        x, y = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        if fp_lambda > 0:
            # only applicable when blocks are RecursiveEulerCEBlock
            with torch.no_grad():
                h = model.tok(x)
            for blk in model.blocks:
                if isinstance(blk, RecursiveEulerCEBlock):
                    loss = loss + fixed_point_loss(blk, h, scale=fp_lambda)
                    with torch.no_grad():
                        h = blk(h)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    val_loss = eval_loss(model, data_val, 12, batch, block)
    mean_depth = None
    # extract halt depths if any
    if any(isinstance(b, RecursiveEulerCEBlock) and b.tol is not None for b in model.blocks):
        # run one more batch to capture depths
        model.eval()
        with torch.no_grad():
            x, _ = next(batch_iter(data_val, batch, block, seed=77))
            _ = model(x)
        depths = []
        for b in model.blocks:
            if isinstance(b, RecursiveEulerCEBlock) and b.last_depths is not None:
                depths.extend(b.last_depths.tolist())
        mean_depth = sum(depths) / len(depths) if depths else None

    return {
        "val_ppl": math.exp(val_loss),
        "val_loss": val_loss,
        "n_params": sum(p.numel() for p in model.parameters()),
        "time_sec": time.perf_counter() - t0,
        "mean_depth": mean_depth,
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
    p.add_argument("--out", type=str, default="examples/ai/results/recursive_euler.json")
    args = p.parse_args()

    docs_root = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    text = load_docs(docs_root, args.corpus_chars)
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = encode(text, stoi)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}\n")

    configs = [
        ("std_rope",         "std_rope",        {}),
        ("euler_ce_k1",      "euler_ce_k1",     {"max_iters": 1}),
        ("euler_ce_k2",      "euler_ce_k2",     {"max_iters": 2}),
        ("euler_ce_k3",      "euler_ce_k3",     {"max_iters": 3}),
        ("euler_ce_halt",    "euler_ce_halt",   {"max_iters": 6, "tol": 5e-3}),
        ("euler_ce_k2_fp",   "euler_ce_k2_fp",  {"max_iters": 2, "fp_lambda": 0.1}),
    ]

    results = {}
    for name, variant, kwargs in configs:
        print(f"=== {name} ({args.seeds} seeds) ===")
        ppls, times, depths = [], [], []
        params = None
        for seed in range(args.seeds):
            r = train_one(train_data, val_data, vocab, variant,
                          d_model=args.d_model, n_layers=args.n_layers,
                          block=args.block, batch=args.batch,
                          steps=args.steps, lr=3e-4, seed=seed, **kwargs)
            ppls.append(r["val_ppl"]); times.append(r["time_sec"])
            params = r["n_params"]
            if r["mean_depth"] is not None:
                depths.append(r["mean_depth"])
        m = sum(ppls) / len(ppls)
        sd = (sum((x - m) ** 2 for x in ppls) / max(len(ppls) - 1, 1)) ** 0.5
        mean_time = sum(times) / len(times)
        mean_d = sum(depths) / len(depths) if depths else None
        results[name] = {
            "mean_ppl": m, "std_ppl": sd, "ppls": ppls,
            "params": params, "mean_time": mean_time,
            "mean_halt_depth": mean_d,
        }
        msg = f"  params {params/1e3:.1f}K   PPL {m:.3f} ± {sd:.3f}   time {mean_time:.1f}s"
        if mean_d is not None:
            msg += f"   halt_depth={mean_d:.2f}"
        print(msg)

    # Verdicts — vs std_rope and vs euler_ce_k1
    print("\n=== verdicts (negative z = variant is BETTER than baseline) ===")

    def z(a, b):
        m1, s1 = results[a]["mean_ppl"], results[a]["std_ppl"]
        m2, s2 = results[b]["mean_ppl"], results[b]["std_ppl"]
        se = (s1 ** 2 / args.seeds + s2 ** 2 / args.seeds) ** 0.5
        return (m1 - m2) / max(se, 1e-9)

    for name, _, _ in configs:
        if name in ("std_rope", "euler_ce_k1"):
            continue
        for base in ("std_rope", "euler_ce_k1"):
            zv = z(name, base)
            d = results[name]["mean_ppl"] - results[base]["mean_ppl"]
            verdict = "WIN " if zv < -1.0 else ("TIE " if abs(zv) < 1.0 else "LOSS")
            print(f"  [{verdict}] {name:16s} vs {base:13s}  z = {zv:+.2f}  Δ = {d:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
