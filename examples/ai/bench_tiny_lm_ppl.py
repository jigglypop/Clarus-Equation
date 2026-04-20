"""Tiny char-level transformer — std attention vs CE MFA.

Offline-friendly proxy for the GPT-2 PPL benchmark. Trains two
identical 2-layer transformers on local repo text (README + docs),
one with standard scaled-dot-product attention and one with CE
Metric-Family Attention (learnable grav metric). Reports
train/val cross-entropy loss and perplexity at each epoch.

The point is not SOTA PPL; it's a like-for-like test of whether
the CE metric-family attention helps on a real (albeit tiny)
language modeling task.
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

from clarus.ce_softmax import grav_attention, lang_attention, mode_gate


# ---------------------------------------------------------------------------
# Data
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
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def batch_iter(data, batch, block, seed=0):
    g = torch.Generator().manual_seed(seed)
    while True:
        idx = torch.randint(0, len(data) - block - 1, (batch,), generator=g)
        x = torch.stack([data[i : i + block] for i in idx])
        y = torch.stack([data[i + 1 : i + 1 + block] for i in idx])
        yield x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Head(nn.Module):
    def __init__(self, d_model, d_head, block, attn_mode):
        super().__init__()
        self.q = nn.Linear(d_model, d_head, bias=False)
        self.k = nn.Linear(d_model, d_head, bias=False)
        self.v = nn.Linear(d_model, d_head, bias=False)
        self.attn_mode = attn_mode  # "std" | "ce"
        self.d_head = d_head
        self.block = block
        self.register_buffer(
            "tril", torch.tril(torch.ones(block, block, dtype=torch.bool))
        )
        if attn_mode == "ce":
            self.L_grav = nn.Parameter(torch.eye(d_head) * 0.3 + torch.randn(d_head, d_head) * 0.05)
            self.sigma = math.sqrt(d_head)
        else:
            self.L_grav = None

    def forward(self, x, mode="wake"):
        b, n, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        mask = self.tril[:n, :n].unsqueeze(0).expand(b, n, n)
        if self.attn_mode == "std":
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
            scores = scores.masked_fill(~mask, float("-inf"))
            a = F.softmax(scores, dim=-1)
            return a @ v
        # CE: metric family
        a_lang = lang_attention(q, k, mask=mask)
        a_grav = grav_attention(k, sigma=self.sigma, mask=mask, L=self.L_grav)
        gate = mode_gate(mode)
        a = gate.omega_lang * a_lang + gate.omega_grav * a_grav
        return a @ v


class MultiHead(nn.Module):
    def __init__(self, n_heads, d_model, block, attn_mode):
        super().__init__()
        d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [Head(d_model, d_head, block, attn_mode) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mode="wake"):
        out = torch.cat([h(x, mode=mode) for h in self.heads], dim=-1)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_heads, d_model, block, attn_mode):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHead(n_heads, d_model, block, attn_mode)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, mode="wake"):
        x = x + self.attn(self.ln1(x), mode=mode)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=2, block=128, attn_mode="std"):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block, d_model)
        self.blocks = nn.ModuleList(
            [Block(n_heads, d_model, block, attn_mode) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.block = block
        self.attn_mode = attn_mode

    def forward(self, idx, mode="wake"):
        b, n = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(n, device=idx.device))
        for blk in self.blocks:
            x = blk(x, mode=mode)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_loss(model, data, n_batches, batch, block, mode="wake"):
    model.eval()
    total = 0.0
    cnt = 0
    it = batch_iter(data, batch, block, seed=42)
    for _ in range(n_batches):
        x, y = next(it)
        logits = model(x, mode=mode)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total += float(loss.item())
        cnt += 1
    model.train()
    return total / cnt


def train_one(
    data_train,
    data_val,
    vocab,
    attn_mode,
    d_model=128,
    n_heads=4,
    n_layers=2,
    block=96,
    batch=24,
    steps=400,
    eval_every=100,
    lr=3e-4,
    seed=0,
    mode="wake",
):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, n_heads, n_layers, block, attn_mode=attn_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.perf_counter()
    it = batch_iter(data_train, batch, block, seed=seed)
    history = []
    for s in range(1, steps + 1):
        x, y = next(it)
        logits = model(x, mode=mode)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % eval_every == 0:
            vl = eval_loss(model, data_val, 8, batch, block, mode=mode)
            history.append({"step": s, "train_loss": float(loss.item()), "val_loss": vl})
            print(f"  [{attn_mode:3s}] step {s:4d} train {float(loss.item()):.4f}  val {vl:.4f}")
    elapsed = time.perf_counter() - t0
    return {"history": history, "time_sec": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--block", type=int, default=96)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--out", type=str,
                        default="examples/ai/results/tiny_lm_ppl.json")
    args = parser.parse_args()

    paths = [
        "README.md",
        "docs/상수.md",
    ]
    text = load_corpus(paths)
    stoi, itos = make_vocab(text)
    data = encode(text, stoi)
    n = len(data)
    split = int(n * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {n} chars, vocab {vocab}, train {len(train_data)}, val {len(val_data)}")

    results = {}
    for attn_mode in ("std", "ce"):
        print(f"\n=== training attn={attn_mode} ===")
        r = train_one(
            train_data, val_data, vocab, attn_mode,
            d_model=args.d_model, n_layers=args.n_layers,
            block=args.block, batch=args.batch, steps=args.steps,
            eval_every=max(args.steps // 6, 50),
        )
        r["final_val_ppl"] = math.exp(r["history"][-1]["val_loss"])
        r["final_train_loss"] = r["history"][-1]["train_loss"]
        print(f"  final val PPL: {r['final_val_ppl']:.2f}  "
              f"(elapsed {r['time_sec']:.1f}s)")
        results[attn_mode] = r

    # Decision
    std_ppl = results["std"]["final_val_ppl"]
    ce_ppl = results["ce"]["final_val_ppl"]
    ratio = ce_ppl / std_ppl
    print("\n=== verdict ===")
    print(f"  std PPL: {std_ppl:.3f}")
    print(f"  ce  PPL: {ce_ppl:.3f}")
    print(f"  ratio (ce/std): {ratio:.3f}  ({'better' if ratio < 1 else 'worse'})")
    print(f"  train time overhead: "
          f"{results['ce']['time_sec'] / results['std']['time_sec']:.2f}x")

    summary = {
        "config": vars(args),
        "vocab_size": vocab,
        "corpus_chars": n,
        "std": results["std"],
        "ce": results["ce"],
        "ce_over_std_ppl": ratio,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
