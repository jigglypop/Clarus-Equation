"""Rigorous ablation for dual-Laplacian block.

Configs:
  std_96          baseline (325K params)
  std_104         params-matched control (~363K params to match ce_lap)
  ce_lap_96_s1    dual-Laplacian, max_steps=1
  ce_lap_96_s2    dual-Laplacian, max_steps=2    (eigenmode-time gate hypothesis)
  ce_lap_96_s3    dual-Laplacian, max_steps=3

Multiple seeds for statistical significance. Reports mean ± std PPL.
Extracts final alpha parameter to see how much the model wants the
Laplacian residual. Also computes eigenvalue spectra of L_lang and
L_grav at end of training on a held-out batch to verify that the two
graphs have statistically distinct spectra.
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

from clarus.ce_laplacian import (
    DualLaplacianBlock,
    _cosine_adjacency,
    _normalized_laplacian,
    _rbf_adjacency,
)


# Reuse the model definitions inline to keep the file self-contained
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
    def __init__(self, d_model, n_heads, block, max_steps=1):
        super().__init__()
        self.std = StdAttnBlock(d_model, n_heads, block)
        self.ln_lap = nn.LayerNorm(d_model)
        self.lap = DualLaplacianBlock(
            d_model=d_model,
            d_lang=d_model // 2,
            d_grav=d_model // 2,
            sigma_grav=math.sqrt(d_model // 2),
            mode="wake",
            max_steps=max_steps,
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def forward(self, x):
        x = self.std(x)
        n = x.shape[1]
        mask = self.tril[:n, :n].unsqueeze(0).expand(x.shape[0], n, n)
        y = self.ln_lap(x)
        delta = self.lap(y, mask=mask) - y
        return x + self.alpha * delta


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block, variant="std", lap_steps=1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block, d_model)
        if variant == "std":
            self.blocks = nn.ModuleList(
                [StdAttnBlock(d_model, n_heads, block) for _ in range(n_layers)]
            )
        elif variant == "ce_lap":
            self.blocks = nn.ModuleList(
                [CELaplacianBlock(d_model, n_heads, block, max_steps=lap_steps) for _ in range(n_layers)]
            )
        else:
            raise ValueError(variant)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.variant = variant

    def forward(self, idx):
        b, n = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(n, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---- data / training helpers ----


def load_corpus(paths):
    text = ""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text += f.read() + "\n"
    return text


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


def train_one(data_train, data_val, vocab, variant, d_model, n_heads, n_layers,
              block, batch, steps, lr, seed, lap_steps=1):
    torch.manual_seed(seed)
    model = TinyLM(vocab, d_model, n_heads, n_layers, block,
                   variant=variant, lap_steps=lap_steps)
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
    val_loss = eval_loss(model, data_val, 8, batch, block)
    n_params = sum(p.numel() for p in model.parameters())
    alphas = []
    if variant == "ce_lap":
        alphas = [float(b.alpha.detach().item()) for b in model.blocks]
    return {
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss),
        "n_params": n_params,
        "time_sec": time.perf_counter() - t0,
        "alphas": alphas,
        "model": model,
    }


def spectral_analysis(model, data_train, batch, block, sigma_factor=0.5):
    """Compute eigenvalues of L_lang, L_grav on a sample batch at the first
    CELaplacianBlock. Returns sorted eigenvalue arrays.
    """
    if not isinstance(model.blocks[0], CELaplacianBlock):
        return None
    model.eval()
    with torch.no_grad():
        it = batch_iter(data_train, 1, block, seed=7)
        x, _ = next(it)
        n = x.shape[1]
        # run through embeddings + first std to get h
        h = model.tok(x) + model.pos(torch.arange(n))
        h = model.blocks[0].std(h)
        h = model.blocks[0].ln_lap(h)
        lap = model.blocks[0].lap
        z_l = lap.P_lang(h)[0]  # (n, d_lang)
        z_g = lap.P_grav(h)[0]
        A_l = _cosine_adjacency(z_l)
        A_g = _rbf_adjacency(z_g, sigma=lap.sigma_grav)
        # apply causal mask
        tril = torch.tril(torch.ones(n, n, dtype=torch.bool))
        A_l = A_l * tril.to(A_l.dtype)
        A_g = A_g * tril.to(A_g.dtype)
        L_l = _normalized_laplacian(A_l)
        L_g = _normalized_laplacian(A_g)
        eigs_l = torch.linalg.eigvalsh(0.5 * (L_l + L_l.T)).sort().values
        eigs_g = torch.linalg.eigvalsh(0.5 * (L_g + L_g.T)).sort().values
    model.train()
    return {
        "eigs_lang": eigs_l.tolist(),
        "eigs_grav": eigs_g.tolist(),
        "lang_spread": float((eigs_l[-1] - eigs_l[0]).item()),
        "grav_spread": float((eigs_g[-1] - eigs_g[0]).item()),
        "lang_mean": float(eigs_l.mean().item()),
        "grav_mean": float(eigs_g.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=str,
                        default="examples/ai/results/laplacian_ablation.json")
    args = parser.parse_args()

    paths = ["README.md", "docs/상수.md"]
    text = load_corpus(paths)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = encode(text, stoi)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}\n")

    # pick d_model for "std matched" so params ≈ ce_lap d_model=96
    # ce_lap adds roughly +37K to std_96 (325K). std at d=104 ~ 382K, close enough.
    configs = [
        ("std_96",       dict(variant="std",    d_model=96,  lap_steps=1)),
        ("std_104",      dict(variant="std",    d_model=104, lap_steps=1)),
        ("ce_lap_s1",    dict(variant="ce_lap", d_model=96,  lap_steps=1)),
        ("ce_lap_s2",    dict(variant="ce_lap", d_model=96,  lap_steps=2)),
        ("ce_lap_s3",    dict(variant="ce_lap", d_model=96,  lap_steps=3)),
    ]

    results = {}
    trained_for_spectrum = None
    for name, cfg in configs:
        print(f"=== {name} ({args.seeds} seeds) ===")
        ppls = []
        params = None
        times = []
        alphas = []
        for seed in range(args.seeds):
            r = train_one(
                train_data, val_data, vocab,
                variant=cfg["variant"], d_model=cfg["d_model"], n_heads=4, n_layers=2,
                block=args.block, batch=args.batch, steps=args.steps, lr=3e-4,
                seed=seed, lap_steps=cfg["lap_steps"],
            )
            ppls.append(r["val_ppl"])
            times.append(r["time_sec"])
            params = r["n_params"]
            if r["alphas"]:
                alphas.append(r["alphas"])
            if name == "ce_lap_s1" and seed == 0:
                trained_for_spectrum = r["model"]
        mean_ppl = sum(ppls) / len(ppls)
        std_ppl = (sum((p - mean_ppl) ** 2 for p in ppls) / max(len(ppls) - 1, 1)) ** 0.5
        mean_time = sum(times) / len(times)
        results[name] = {
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "ppls": ppls,
            "params": params,
            "mean_time": mean_time,
            "alphas": alphas,
        }
        print(f"  params {params/1e3:.1f}K   PPL {mean_ppl:.3f} ± {std_ppl:.3f}   "
              f"time {mean_time:.1f}s")

    # Spectral analysis
    if trained_for_spectrum is not None:
        print("\n=== spectral analysis (trained ce_lap_s1, seed 0, first block) ===")
        spec = spectral_analysis(trained_for_spectrum, train_data, args.batch, args.block)
        # report quantiles
        import statistics as st
        el = spec["eigs_lang"]
        eg = spec["eigs_grav"]
        print(f"  lang  spread {spec['lang_spread']:.3f}  mean {spec['lang_mean']:.3f}  "
              f"q05 {el[len(el)//20]:.3f}  q95 {el[-len(el)//20]:.3f}")
        print(f"  grav  spread {spec['grav_spread']:.3f}  mean {spec['grav_mean']:.3f}  "
              f"q05 {eg[len(eg)//20]:.3f}  q95 {eg[-len(eg)//20]:.3f}")
        results["spectrum"] = spec

    # Statistical conclusion: ce_lap_s1 vs std_104 (matched) and vs std_96 (raw)
    def z(a, b):
        m1, s1 = results[a]["mean_ppl"], results[a]["std_ppl"]
        m2, s2 = results[b]["mean_ppl"], results[b]["std_ppl"]
        # pooled-ish, just a rough sigma
        se = (s1 ** 2 / max(1, args.seeds) + s2 ** 2 / max(1, args.seeds)) ** 0.5
        return (m1 - m2) / max(se, 1e-9)

    print("\n=== verdict (sigma = positive means first is worse) ===")
    print(f"  ce_lap_s1 vs std_96   z = {z('ce_lap_s1','std_96'):+.2f}  "
          f"(Δ={results['ce_lap_s1']['mean_ppl'] - results['std_96']['mean_ppl']:+.3f})")
    print(f"  ce_lap_s1 vs std_104  z = {z('ce_lap_s1','std_104'):+.2f}  "
          f"(Δ={results['ce_lap_s1']['mean_ppl'] - results['std_104']['mean_ppl']:+.3f})")
    print(f"  ce_lap_s2 vs ce_lap_s1 z = {z('ce_lap_s2','ce_lap_s1'):+.2f}  "
          f"(Δ={results['ce_lap_s2']['mean_ppl'] - results['ce_lap_s1']['mean_ppl']:+.3f})")
    print(f"  ce_lap_s3 vs ce_lap_s1 z = {z('ce_lap_s3','ce_lap_s1'):+.2f}  "
          f"(Δ={results['ce_lap_s3']['mean_ppl'] - results['ce_lap_s1']['mean_ppl']:+.3f})")

    # drop non-serializable model references
    for k in list(results.keys()):
        if isinstance(results[k], dict) and "model" in results[k]:
            results[k].pop("model", None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
