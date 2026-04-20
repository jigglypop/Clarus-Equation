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
from tqdm import tqdm

from clarus.ce_euler import (
    EulerCEBlock, RecursiveEulerCEBlock, fixed_point_loss,
)
from clarus.ce_riemann_attn import RiemannAttnBlock, riemann_zero_init
from clarus.ce_mra import MRABlock


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

    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        """Grow positional buffers for length-extrapolation eval.
        Learnable params unchanged; only `pos` and `tril` are re-sized."""
        cur = self.pos.shape[0]
        if new_block <= cur:
            return
        dev = self.pos.device
        self.pos = torch.arange(new_block, dtype=torch.float32, device=dev)
        self.tril = torch.tril(
            torch.ones(new_block, new_block, dtype=torch.bool, device=dev))

    # `attn` here is a method, not a sub-module — extend_to lives on self.

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


class RoPEAlibiAttnBlock(RoPEAttnBlock):
    """RoPE + ALiBi-style linear distance decay (per-head learnable slope).

    Score = (Q' · K') / √d  −  m_h · |i − j|,
    where m_h = exp(log_m_h) is per-head learnable. ALiBi proper uses a
    fixed geometric schedule for m_h; we let it be learnable for fair
    comparison with EulerCE's learnable ξ.
    """

    def __init__(self, d_model, n_heads, block, base=10000.0,
                 m_init: float = 0.05):
        super().__init__(d_model, n_heads, block, base=base)
        self.log_m = nn.Parameter(
            torch.full((n_heads,), math.log(m_init), dtype=torch.float32))
        d_mat = (torch.arange(block).unsqueeze(1)
                 - torch.arange(block).unsqueeze(0)).abs().float()
        self.register_buffer("d_mat", d_mat)

    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        cur = self.pos.shape[0]
        if new_block <= cur:
            return
        super().extend_to(new_block)
        dev = self.d_mat.device
        d = (torch.arange(new_block).unsqueeze(1)
             - torch.arange(new_block).unsqueeze(0)).abs().float().to(dev)
        self.d_mat = d

    def attn(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        theta = self.pos[:n].view(1, 1, n, 1) * self.inv_freq.view(1, 1, 1, -1)
        cos = theta.cos(); sin = theta.sin()
        q = self._rotate(q, cos, sin); k = self._rotate(k, cos, sin)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        m = torch.exp(self.log_m)
        s = s - m.view(1, self.n_heads, 1, 1) \
                * self.d_mat[:n, :n].view(1, 1, n, n)
        s = s.masked_fill(~self.tril[:n, :n], float("-inf"))
        a = F.softmax(s, dim=-1)
        return (a @ v).transpose(1, 2).contiguous().view(b, n, d)


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block, variant,
                 max_iters=1, tol=None,
                 depth_aware_freq=False, depth_aware_iters=False,
                 ffn_init: str = "kaiming"):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.variant = variant
        self.fp_lambda = 0.0
        if variant == "std_rope":
            blocks = [RoPEAttnBlock(d_model, n_heads, block) for _ in range(n_layers)]
        elif variant == "riemann_rope":
            blocks = [RiemannAttnBlock(d_model, n_heads, block) for _ in range(n_layers)]
        elif variant.startswith("mra"):
            # Ablation grid for Mellin-Riemann Attention.
            # mra          = RoPE freq + ζ amplitude (lean; primary claim)
            # mra_noamp    = RoPE freq, amplitude weighting OFF  (= plain RoPE)
            # mra_zeta     = γ_k log(1+p) freq, amplitude ON
            # mra_bias     = lean + additive log decay bias
            # mra_mult     = lean + multiplicative decay (old design)
            # mra_sparse   = lean + ε² = 0.0487 bootstrap sparsity
            # mra_sn       = lean + spectral_norm on W_o
            # mra_h        = lean + Hermitian (bidirectional only — diagnostic)
            mra_cfg = {
                "mra":         dict(freq_mode="rope",     amp_weight=True),
                "mra_noamp":   dict(freq_mode="rope",     amp_weight=False),
                "mra_zeta":    dict(freq_mode="zeta_log", amp_weight=True),
                "mra_bias":    dict(freq_mode="rope",     amp_weight=True,
                                    decay_mode="bias"),
                "mra_mult":    dict(freq_mode="rope",     amp_weight=True,
                                    decay_mode="mult"),
                "mra_sparse":  dict(freq_mode="rope",     amp_weight=True,
                                    sparse_eps2=0.0487),
                "mra_sn":      dict(freq_mode="rope",     amp_weight=True,
                                    spectral_norm_o=True),
                "mra_h":       dict(freq_mode="rope",     amp_weight=True,
                                    hermitian=True),
            }
            if variant not in mra_cfg:
                raise ValueError(f"unknown mra variant: {variant}")
            blocks = [MRABlock(d_model, n_heads, block, **mra_cfg[variant])
                      for _ in range(n_layers)]
        elif variant == "euler_ce_k1":
            blocks = [EulerCEBlock(d_model, n_heads, block,
                                   layer_idx=i, n_layers=n_layers,
                                   depth_aware_freq=depth_aware_freq)
                      for i in range(n_layers)]
        elif variant.startswith("euler_ce") or variant.endswith("_fp"):
            blocks = [RecursiveEulerCEBlock(d_model, n_heads, block,
                                            max_iters=max_iters, tol=tol,
                                            layer_idx=i, n_layers=n_layers,
                                            depth_aware_freq=depth_aware_freq,
                                            depth_aware_iters=depth_aware_iters)
                      for i in range(n_layers)]
        else:
            raise ValueError(variant)
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

        if ffn_init == "riemann":
            self._apply_riemann_ffn_init()
        elif ffn_init != "kaiming":
            raise ValueError(f"unknown ffn_init: {ffn_init}")

    def _apply_riemann_ffn_init(self):
        """Apply riemann_zero_init to the first Linear of every block's FFN.

        Works for both RoPEAttnBlock / RiemannAttnBlock / EulerCEBlock /
        RecursiveEulerCEBlock by walking each block's `ffn` attribute.
        """
        for blk in self.blocks:
            ffn = getattr(blk, "ffn", None) or getattr(blk, "core", None)
            if ffn is None:
                continue
            # RecursiveEulerCEBlock wraps EulerCEBlock as `.core`
            if hasattr(ffn, "ffn"):
                ffn = ffn.ffn
            target = None
            if isinstance(ffn, nn.Sequential):
                for m in ffn:
                    if isinstance(m, nn.Linear):
                        target = m
                        break
            elif isinstance(ffn, nn.Linear):
                target = ffn
            if target is not None:
                riemann_zero_init(target, axis="in")

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
    g = torch.Generator(device=data.device).manual_seed(seed) if data.is_cuda else torch.Generator().manual_seed(seed)
    n = len(data) - block - 1
    while True:
        if data.is_cuda:
            idx = torch.randint(0, n, (batch,), generator=g, device=data.device)
        else:
            idx = torch.randint(0, n, (batch,), generator=g)
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
              batch, steps, lr, seed, max_iters=1, tol=None, fp_lambda=0.0,
              device="cpu", pbar_desc="",
              depth_aware_freq=False, depth_aware_iters=False,
              ffn_init="kaiming"):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyLM(vocab, d_model, 4, n_layers, block, variant=variant,
                   max_iters=max_iters, tol=tol,
                   depth_aware_freq=depth_aware_freq,
                   depth_aware_iters=depth_aware_iters,
                   ffn_init=ffn_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    it = batch_iter(data_train, batch, block, seed=seed)
    t0 = time.perf_counter()
    last_loss = float("nan")
    pbar = tqdm(range(1, steps + 1), desc=pbar_desc, leave=False, dynamic_ncols=True)
    for s in pbar:
        x, y = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        if fp_lambda > 0:
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
        last_loss = float(loss.detach())
        if s % 25 == 0 or s == steps:
            pbar.set_postfix(loss=f"{last_loss:.3f}", ppl=f"{math.exp(min(last_loss, 20)):.2f}")
    pbar.close()

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


def build_configs(mode: str):
    if mode == "full":
        return [
            ("std_rope",       "std_rope",       {}),
            ("riemann_rope",   "riemann_rope",   {}),
            ("euler_ce_k1",    "euler_ce_k1",    {"max_iters": 1}),
            ("euler_ce_k2",    "euler_ce_k2",    {"max_iters": 2}),
            ("euler_ce_k3",    "euler_ce_k3",    {"max_iters": 3}),
            ("euler_ce_k2_fp", "euler_ce_k2_fp", {"max_iters": 2,
                                                  "fp_lambda": 0.1}),
            ("euler_ce_freq",  "euler_ce_freq",  {"max_iters": 1,
                                                  "depth_aware_freq": True}),
            ("euler_ce_depth", "euler_ce_depth", {"max_iters": 1,
                                                  "depth_aware_iters": True}),
            ("euler_ce_both",  "euler_ce_both",  {"max_iters": 1,
                                                  "depth_aware_freq": True,
                                                  "depth_aware_iters": True}),
        ]
    if mode == "block":
        # minimal set to isolate block-aware base effect at varying block.
        return [
            ("std_rope",    "std_rope",    {}),
            ("euler_ce_k1", "euler_ce_k1", {"max_iters": 1}),
        ]
    if mode == "depth":
        # k2_fp vs k2_fp+depth-aware to isolate depth-aware effect at depth.
        return [
            ("std_rope",          "std_rope",          {}),
            ("euler_ce_k2_fp",    "euler_ce_k2_fp",    {"max_iters": 2,
                                                         "fp_lambda": 0.1}),
            ("euler_ce_k2_fp_da", "euler_ce_k2_fp_da", {"max_iters": 2,
                                                         "fp_lambda": 0.1,
                                                         "depth_aware_freq": True,
                                                         "depth_aware_iters": True}),
        ]
    if mode == "ffn_init":
        # Design (4): kaiming vs Riemann-zero-spaced FFN init
        # on top of euler_ce_k1 (the previous winner).
        return [
            ("euler_ce_k1",       "euler_ce_k1", {"max_iters": 1,
                                                  "ffn_init": "kaiming"}),
            ("euler_ce_k1_riffn", "euler_ce_k1", {"max_iters": 1,
                                                  "ffn_init": "riemann"}),
        ]
    if mode == "riemann":
        # Compare RoPE / EulerCE / Riemann-surface PE head-to-head.
        return [
            ("std_rope",     "std_rope",     {}),
            ("euler_ce_k1",  "euler_ce_k1",  {"max_iters": 1}),
            ("riemann_rope", "riemann_rope", {}),
        ]
    if mode == "mra":
        # MRA ablation: test each design component in isolation.
        # Core novel claim is `mra` (RoPE freq + ζ amplitude). Others
        # isolate the contribution (or cost) of each knob.
        return [
            ("std_rope",    "std_rope",   {}),
            ("euler_ce_k1", "euler_ce_k1",{"max_iters": 1}),
            ("mra",         "mra",        {}),   # RoPE freq + ζ amp  (primary)
            ("mra_noamp",   "mra_noamp",  {}),   # ablate: amp off     → ~RoPE
            ("mra_zeta",    "mra_zeta",   {}),   # ablate: γ freq (old bad)
            ("mra_bias",    "mra_bias",   {}),   # + additive log bias
            ("mra_mult",    "mra_mult",   {}),   # + multiplicative decay (old)
            ("mra_sparse",  "mra_sparse", {}),   # + bootstrap ε² sparsity
            ("mra_sn",      "mra_sn",     {}),   # + σ₁(W_o) ≤ 1
        ]
    raise ValueError(f"unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--corpus_chars", type=int, default=400_000)
    p.add_argument("--mode", type=str, default="full",
                   choices=("full", "block", "depth", "ffn_init", "riemann", "mra"))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="examples/ai/results/recursive_euler.json")
    args = p.parse_args()

    docs_root = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    text = load_docs(docs_root, args.corpus_chars)
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = encode(text, stoi).to(args.device)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}, device {args.device}\n")

    configs = build_configs(args.mode)

    results = {}
    for name, variant, kwargs in configs:
        print(f"=== {name} ({args.seeds} seeds) ===")
        ppls, times, depths = [], [], []
        params = None
        for seed in range(args.seeds):
            r = train_one(train_data, val_data, vocab, variant,
                          d_model=args.d_model, n_layers=args.n_layers,
                          block=args.block, batch=args.batch,
                          steps=args.steps, lr=3e-4, seed=seed,
                          device=args.device,
                          pbar_desc=f"{name} seed{seed+1}/{args.seeds}",
                          **kwargs)
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
            if base not in results:
                continue
            zv = z(name, base)
            d = results[name]["mean_ppl"] - results[base]["mean_ppl"]
            verdict = "WIN " if zv < -1.0 else ("TIE " if abs(zv) < 1.0 else "LOSS")
            print(f"  [{verdict}] {name:18s} vs {base:13s}  z = {zv:+.2f}  Δ = {d:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
