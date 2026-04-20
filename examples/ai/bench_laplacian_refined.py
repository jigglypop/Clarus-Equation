"""Refined ablation — learnable gate/sigma + larger corpus + more seeds.

Goal: filter the usable configurations from the experimental chaff.

Variants:
  std_96             baseline
  std_matched        params-matched std (d_model bumped so params ≈ ce_*)
  ce_dual_frozen     learnable_gate=False, sigma frozen at sqrt(d)
  ce_dual_gate       learnable_gate=True, sigma frozen
  ce_dual_gs         learnable_gate=True, learnable_sigma=True
  ce_par_gs          parallel std + dual, both learnable

We report mean/std PPL over 5 seeds, plus the converged gate/sigma
values (theory predicts T_WAKE=0.315 as lang weight in wake mode; does
the model actually learn this?).
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

from clarus.ce_laplacian import DualLaplacianBlock
from clarus.constants import T_WAKE


# ---- std baseline block (reused) ----


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
        s = s.masked_fill(~self.tril[:n, :n], float("-inf"))
        a = F.softmax(s, dim=-1)
        out = (a @ v).transpose(1, 2).contiguous().view(b, n, d)
        return self.o(out)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CEDualBlock(nn.Module):
    def __init__(self, d_model, n_heads, block,
                 learnable_gate=False, learnable_sigma=False, mode="wake"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dual = DualLaplacianBlock(
            d_model=d_model, d_lang=d_model, d_grav=d_model,
            sigma_grav=math.sqrt(d_model), mode=mode,
            learnable_gate=learnable_gate, learnable_sigma=learnable_sigma,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.register_buffer("tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def forward(self, x):
        n = x.shape[1]
        mask = self.tril[:n, :n].unsqueeze(0).expand(x.shape[0], n, n)
        x = x + self.dual(self.ln1(x), causal_mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class CEParallelBlock(nn.Module):
    def __init__(self, d_model, n_heads, block,
                 learnable_gate=True, learnable_sigma=True, mode="wake"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.std = StdAttnBlock(d_model, n_heads, block)
        self.dual = DualLaplacianBlock(
            d_model=d_model, d_lang=d_model // 2, d_grav=d_model // 2,
            sigma_grav=math.sqrt(d_model // 2), mode=mode,
            learnable_gate=learnable_gate, learnable_sigma=learnable_sigma,
        )
        self.mix = nn.Parameter(torch.tensor(0.0))  # sigmoid gate over std vs dual
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.register_buffer("tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def forward(self, x):
        n = x.shape[1]
        y = self.ln1(x)
        s_out = self.std.attn(y)
        mask = self.tril[:n, :n].unsqueeze(0).expand(x.shape[0], n, n)
        d_out = self.dual(y, causal_mask=mask)
        g = torch.sigmoid(self.mix)
        x = x + g * s_out + (1 - g) * d_out
        x = x + self.ffn(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block, variant, cfg):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block, d_model)
        if variant == "std":
            self.blocks = nn.ModuleList(
                [StdAttnBlock(d_model, n_heads, block) for _ in range(n_layers)]
            )
        elif variant == "ce_dual":
            self.blocks = nn.ModuleList(
                [CEDualBlock(d_model, n_heads, block, **cfg) for _ in range(n_layers)]
            )
        elif variant == "ce_par":
            self.blocks = nn.ModuleList(
                [CEParallelBlock(d_model, n_heads, block, **cfg) for _ in range(n_layers)]
            )
        else:
            raise ValueError(variant)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        b, n = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(n, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---- data ----


def load_all_docs(root, max_chars=800_000):
    text = ""
    paths = sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))
    paths += [os.path.join(root, "..", "README.md")]
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
              batch, steps, lr, seed, cfg):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, 4, n_layers, block, variant=variant, cfg=cfg)
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
    n_params = sum(p.numel() for p in model.parameters())
    # extract learned params
    learned = []
    for blk in model.blocks:
        d = None
        if isinstance(blk, CEDualBlock):
            d = blk.dual
        elif isinstance(blk, CEParallelBlock):
            d = blk.dual
        if d is not None:
            w_l = torch.sigmoid(d.gate_logit).item()
            sig = torch.exp(d.log_sigma_grav).item()
            learned.append({"w_lang": w_l, "sigma": sig})
    return {
        "val_ppl": math.exp(val_loss),
        "val_loss": val_loss,
        "n_params": n_params,
        "time_sec": time.perf_counter() - t0,
        "learned": learned,
    }


def mean_std(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return m, var ** 0.5


def z_score(m1, s1, m2, s2, n):
    se = (s1 ** 2 / n + s2 ** 2 / n) ** 0.5
    return (m1 - m2) / max(se, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--corpus_chars", type=int, default=400_000)
    parser.add_argument("--out", type=str,
                        default="examples/ai/results/laplacian_refined.json")
    args = parser.parse_args()

    docs_root = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    text = load_all_docs(docs_root, max_chars=args.corpus_chars)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = encode(text, stoi)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}\n")

    configs = [
        ("std_96",         "std",     96,  {}),
        ("std_104",        "std",     104, {}),
        ("ce_dual_frozen", "ce_dual", 96,  dict(learnable_gate=False, learnable_sigma=False)),
        ("ce_dual_gate",   "ce_dual", 96,  dict(learnable_gate=True,  learnable_sigma=False)),
        ("ce_dual_gs",     "ce_dual", 96,  dict(learnable_gate=True,  learnable_sigma=True)),
        ("ce_par_gs",      "ce_par",  96,  dict(learnable_gate=True,  learnable_sigma=True)),
    ]

    results = {}
    for name, variant, d_model, cfg in configs:
        print(f"=== {name} ({args.seeds} seeds) ===")
        ppls, times, learneds = [], [], []
        params = None
        for seed in range(args.seeds):
            r = train_one(train_data, val_data, vocab, variant,
                          d_model=d_model, n_layers=args.n_layers,
                          block=args.block, batch=args.batch,
                          steps=args.steps, lr=3e-4, seed=seed, cfg=cfg)
            ppls.append(r["val_ppl"])
            times.append(r["time_sec"])
            if r["learned"]:
                learneds.append(r["learned"])
            params = r["n_params"]
        m, s = mean_std(ppls)
        mean_time = sum(times) / len(times)
        # aggregate learned params over seeds and layers
        agg = {}
        if learneds:
            w = [item["w_lang"] for ll in learneds for item in ll]
            sg = [item["sigma"] for ll in learneds for item in ll]
            agg = {
                "w_lang_mean": sum(w) / len(w),
                "w_lang_std": mean_std(w)[1],
                "sigma_mean": sum(sg) / len(sg),
                "sigma_std": mean_std(sg)[1],
            }
        results[name] = {
            "mean_ppl": m, "std_ppl": s, "ppls": ppls,
            "params": params, "mean_time": mean_time,
            "learned_aggregate": agg,
        }
        msg = f"  params {params/1e3:.1f}K   PPL {m:.3f} ± {s:.3f}   time {mean_time:.1f}s"
        if agg:
            msg += f"   gate_w_lang={agg['w_lang_mean']:.3f}±{agg['w_lang_std']:.3f}  " \
                   f"sigma={agg['sigma_mean']:.3f}±{agg['sigma_std']:.3f}"
        print(msg)

    # Verdicts
    print("\n=== verdicts (negative z means ce variant is BETTER than baseline) ===")
    baselines = ["std_96", "std_104"]
    for name, _, _, _ in configs:
        if name.startswith("std"):
            continue
        for base in baselines:
            zval = z_score(results[name]["mean_ppl"], results[name]["std_ppl"],
                           results[base]["mean_ppl"], results[base]["std_ppl"],
                           args.seeds)
            d = results[name]["mean_ppl"] - results[base]["mean_ppl"]
            verdict = "WIN " if zval < -1.0 else ("TIE " if abs(zval) < 1.0 else "LOSS")
            print(f"  [{verdict}] {name:16s} vs {base:8s}  z = {zval:+.2f}  Δ = {d:+.3f}")

    # CE theory check — does model keep w_lang near T_WAKE (=0.685 for wake lang)?
    print(f"\nTheory check: T_WAKE-derived init w_lang = {1 - T_WAKE:.4f}")
    for name in ("ce_dual_gate", "ce_dual_gs", "ce_par_gs"):
        if name in results and results[name]["learned_aggregate"]:
            w = results[name]["learned_aggregate"]["w_lang_mean"]
            print(f"  {name:16s} learned w_lang = {w:.4f}  (shift = {w - (1 - T_WAKE):+.4f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
