"""CE vs Standard Transformer: reproducible head-to-head comparison.

기본 모드는 CE FFN hidden dim을 자동 조정해, 표준 Transformer와
총 파라미터 수가 최대한 가깝도록 맞춘다.

Usage:
    py ce_vs_standard.py
    py ce_vs_standard.py --dim 128 --steps 3000
    py ce_vs_standard.py --data train_data.txt --no-match-params
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from clarus_lm import LBONorm, GaugeLattice


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in s]


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def init_model_weights(module: nn.Module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            weight = layer.weight_orig if hasattr(layer, "weight_orig") else layer.weight
            nn.init.normal_(weight, std=0.02)
        elif isinstance(layer, nn.Embedding):
            nn.init.normal_(layer.weight, std=0.02)


def build_lr_scheduler(opt, steps: int):
    warmup = min(200, steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)


def mean_recent(values, n: int) -> float:
    return sum(values[-n:]) / min(n, len(values))


def loss_to_ppl(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_hidden_dim(dim: int, hidden_dim: int | None, mult: float) -> int:
    if hidden_dim is not None:
        return max(dim, int(hidden_dim))
    return max(dim, int(round(dim * mult)))


def build_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


# ── Standard Transformer ──

class StdAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class StdBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int | None = None, mult: float = 4.0):
        super().__init__()
        hidden_dim = resolve_hidden_dim(dim, hidden_dim, mult)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = StdAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StdLM(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4,
                 max_seq=256, ffn_hidden_dim=None, ffn_mult=4.0):
        super().__init__()
        self.max_seq = max_seq
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList(
            [StdBlock(dim, n_heads, hidden_dim=ffn_hidden_dim, mult=ffn_mult) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        init_model_weights(self)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── CE Transformer ──

class CEAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.utils.spectral_norm(nn.Linear(dim, dim, bias=False))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class CEBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int | None = None,
                 mult: float = 4.0, mix_rank: int | None = None):
        super().__init__()
        self.ln1 = LBONorm(dim)
        self.attn = CEAttention(dim, n_heads)
        self.ln2 = LBONorm(dim)
        self.ffn = GaugeLattice(dim, mult, hidden_dim=hidden_dim, mix_rank=mix_rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    @property
    def curvature(self):
        return (self.ln1._curvature + self.ln2._curvature + self.ffn.phi._curvature) / 3


class CELM(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4,
                 max_seq=256, lambda_curv=0.005, ffn_hidden_dim=None,
                 ffn_mult=4.0, mix_rank=None):
        super().__init__()
        self.max_seq = max_seq
        self.lambda_curv = lambda_curv
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList(
            [
                CEBlock(
                    dim,
                    n_heads,
                    hidden_dim=ffn_hidden_dim,
                    mult=ffn_mult,
                    mix_rank=mix_rank,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = LBONorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        init_model_weights(self)

    def _logits(self, idx):
        _, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def forward(self, idx, targets=None):
        logits = self._logits(idx)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            curv = sum(b.curvature for b in self.blocks) / len(self.blocks)
            loss = ce + self.lambda_curv * curv
        return logits, loss

    def ce_loss_only(self, idx, targets):
        logits = self._logits(idx)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# ── Training ──

def get_batch(data, seq_len, batch_size, device, generator):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, data, seq_len, batch_size, device, generator_seed, n_eval=30):
    model.eval()
    losses = []
    generator = build_generator(generator_seed)
    for _ in range(n_eval):
        x, y = get_batch(data, seq_len, batch_size, device, generator)
        if hasattr(model, "ce_loss_only"):
            loss = model.ce_loss_only(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_model(model, train_data, val_data, steps, lr, seq_len, batch_size,
                device, train_seed, eval_seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = build_lr_scheduler(opt, steps)
    t0 = time.time()
    train_losses = []
    checkpoints = []
    checkpoint_interval = max(1, steps // 5)
    train_generator = build_generator(train_seed)

    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_data, seq_len, batch_size, device, train_generator)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        train_losses.append(loss.item())

        if step % checkpoint_interval == 0 or step == steps:
            vl = eval_loss(
                model,
                val_data,
                seq_len,
                min(batch_size, 8),
                device,
                generator_seed=eval_seed,
            )
            elapsed = time.time() - t0
            tl = mean_recent(train_losses, 50)
            checkpoints.append({"step": step, "train": tl, "val": vl, "time": elapsed})
            print(f"    step {step:5d} | train {tl:.4f} | val {vl:.4f} | {elapsed:.0f}s", flush=True)

    final_val = eval_loss(
        model,
        val_data,
        seq_len,
        min(batch_size, 8),
        device,
        generator_seed=eval_seed,
        n_eval=50,
    )
    elapsed = time.time() - t0
    return final_val, elapsed, checkpoints


def load_text(path: str | None, text_limit: int) -> str:
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    try:
        datasets = importlib.import_module("datasets")
        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])[:text_limit]
    except Exception as exc:
        raise RuntimeError(
            "Pass --data or install the 'datasets' package to load WikiText-2."
        ) from exc
    if not text.strip():
        raise RuntimeError("Loaded text is empty.")
    return text


def build_std_model(vocab_size, args, ffn_hidden_dim):
    return StdLM(
        vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq=args.seq_len,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_mult=args.std_ffn_mult,
    )


def build_ce_model(vocab_size, args, ffn_hidden_dim):
    return CELM(
        vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq=args.seq_len,
        lambda_curv=args.lambda_curv,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_mult=args.std_ffn_mult,
        mix_rank=args.ce_mix_rank,
    )


def match_ce_hidden_dim(vocab_size, args, target_params):
    cache = {}
    std_hidden = resolve_hidden_dim(args.dim, args.std_ffn_hidden_dim, args.std_ffn_mult)
    search_max = max(std_hidden * 8, args.dim * 8)

    def params_for(hidden_dim):
        hidden_dim = max(args.dim, int(hidden_dim))
        if hidden_dim not in cache:
            model = build_ce_model(vocab_size, args, hidden_dim)
            cache[hidden_dim] = count_params(model)
            del model
        return cache[hidden_dim]

    low = args.dim
    high = max(low, std_hidden)
    while params_for(high) < target_params and high < search_max:
        next_high = min(search_max, high * 2)
        if next_high == high:
            break
        high = next_high

    best_hidden = low
    best_diff = abs(params_for(low) - target_params)
    lo, hi = low, high
    while lo <= hi:
        mid = (lo + hi) // 2
        params = params_for(mid)
        diff = abs(params - target_params)
        if diff < best_diff or (diff == best_diff and mid < best_hidden):
            best_hidden = mid
            best_diff = diff
        if params < target_params:
            lo = mid + 1
        else:
            hi = mid - 1

    sweep_start = max(args.dim, best_hidden - 8)
    sweep_end = min(search_max, best_hidden + 8)
    for hidden_dim in range(sweep_start, sweep_end + 1):
        params = params_for(hidden_dim)
        diff = abs(params - target_params)
        if diff < best_diff or (diff == best_diff and hidden_dim < best_hidden):
            best_hidden = hidden_dim
            best_diff = diff

    return best_hidden, params_for(best_hidden)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lambda-curv", type=float, default=0.005)
    p.add_argument("--std-ffn-mult", type=float, default=4.0)
    p.add_argument("--std-ffn-hidden-dim", type=int, default=None)
    p.add_argument("--ce-ffn-hidden-dim", type=int, default=None)
    p.add_argument("--ce-mix-rank", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--text-limit", type=int, default=500_000)
    p.add_argument("--match-params", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("=" * 65)
    print("CE vs Standard Transformer: Head-to-Head")
    print(f"  dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"  steps={args.steps}, batch={args.batch_size}, seq={args.seq_len}")
    print(f"  device={args.device}, seed={args.seed}, match_params={args.match_params}")
    print("=" * 65)

    seed_everything(args.seed, args.deterministic)
    text = load_text(args.data, args.text_limit)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    if len(data) <= args.seq_len + 1:
        raise ValueError("Dataset is too short for the requested seq_len.")
    n_val = max(1000, int(len(data) * 0.05))
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"\nData: {len(data)} chars, vocab={tok.vocab_size}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    std_hidden_dim = resolve_hidden_dim(args.dim, args.std_ffn_hidden_dim, args.std_ffn_mult)

    # ── (A) Standard ──
    print("\n--- (A) Standard Transformer ---")
    seed_everything(args.seed, args.deterministic)
    std = build_std_model(tok.vocab_size, args, std_hidden_dim).to(args.device)
    std_params = count_params(std)
    print(f"  FFN hidden dim: {std_hidden_dim}")
    print(f"  Params: {std_params:,}")
    std_val, std_time, std_ckpts = train_model(
        std,
        train_data,
        val_data,
        args.steps,
        args.lr,
        args.seq_len,
        args.batch_size,
        args.device,
        train_seed=args.seed + 101,
        eval_seed=args.seed + 202,
    )
    std_ppl = loss_to_ppl(std_val)
    print(f"  Final: val_loss={std_val:.4f}, PPL={std_ppl:.2f}, time={std_time:.0f}s")
    del std

    if args.ce_ffn_hidden_dim is not None:
        ce_hidden_dim = max(args.dim, int(args.ce_ffn_hidden_dim))
    elif args.match_params:
        ce_hidden_dim, _ = match_ce_hidden_dim(tok.vocab_size, args, std_params)
    else:
        ce_hidden_dim = std_hidden_dim

    # ── (B) CE ──
    print("\n--- (B) CE Transformer (LBONorm + GaugeLattice + spectral + curv) ---")
    seed_everything(args.seed, args.deterministic)
    ce = build_ce_model(tok.vocab_size, args, ce_hidden_dim).to(args.device)
    ce_params = count_params(ce)
    print(f"  Params: {ce_params:,} (delta: {ce_params - std_params:+,})")
    print(f"  FFN hidden dim: {ce_hidden_dim}")

    lattice = ce.blocks[0].ffn
    print(f"  SU(3) bind:  {lattice.d3:3d} dims")
    print(f"  SU(2) decide:{lattice.d2:3d} dims")
    print(f"  U(1) attend: {lattice.d1:3d} dims")
    print(f"  mix rank:    {lattice.mix_rank:3d}")

    ce_val, ce_time, ce_ckpts = train_model(
        ce,
        train_data,
        val_data,
        args.steps,
        args.lr,
        args.seq_len,
        args.batch_size,
        args.device,
        train_seed=args.seed + 101,
        eval_seed=args.seed + 202,
    )
    ce_ppl = loss_to_ppl(ce_val)
    print(f"  Final: val_loss={ce_val:.4f}, PPL={ce_ppl:.2f}, time={ce_time:.0f}s")
    del ce

    # ── Summary ──
    print("\n" + "=" * 65)
    print("COMPARISON")
    print("=" * 65)
    print(f"  {'':25s} {'Params':>10s} {'Val Loss':>10s} {'PPL':>8s} {'Time':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'Standard Transformer':25s} {std_params:>10,} {std_val:>10.4f} {std_ppl:>8.2f} {std_time:>7.0f}s")
    print(f"  {'CE Transformer':25s} {ce_params:>10,} {ce_val:>10.4f} {ce_ppl:>8.2f} {ce_time:>7.0f}s")
    print()

    delta_loss = ce_val - std_val
    delta_ppl = ce_ppl - std_ppl
    param_ratio = ce_params / std_params
    winner = "CE" if delta_loss < 0 else "Standard"
    print(f"  Winner: {winner}")
    print(f"  Val loss delta: {delta_loss:+.4f}")
    print(f"  PPL delta: {delta_ppl:+.2f}")
    print(f"  Param ratio: {param_ratio:.3f}x")

    if delta_loss < 0 and param_ratio < 1.0:
        print(f"  CE wins on BOTH quality AND efficiency")
    elif delta_loss < 0:
        print(f"  CE wins on quality, slightly more params")
    elif param_ratio < 1.0:
        print(f"  CE wins on efficiency, slightly worse quality")
    else:
        print(f"  Standard wins this round")

    print("\n  Learning curves:")
    print(f"  {'Step':>6s} {'Std val':>10s} {'CE val':>10s} {'delta':>10s}")
    for s_ck, c_ck in zip(std_ckpts, ce_ckpts):
        d = c_ck["val"] - s_ck["val"]
        print(f"  {s_ck['step']:>6d} {s_ck['val']:>10.4f} {c_ck['val']:>10.4f} {d:>+10.4f}")

    out_path = args.out or os.path.join(os.path.dirname(__file__), "ce_vs_standard_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "standard": {
                "params": std_params,
                "ffn_hidden_dim": std_hidden_dim,
                "val_loss": std_val,
                "ppl": std_ppl,
                "time": std_time,
                "checkpoints": std_ckpts,
            },
            "ce": {
                "params": ce_params,
                "ffn_hidden_dim": ce_hidden_dim,
                "mix_rank": args.ce_mix_rank if args.ce_mix_rank is not None else max(0, args.dim // 8),
                "val_loss": ce_val,
                "ppl": ce_ppl,
                "time": ce_time,
                "checkpoints": ce_ckpts,
            },
            "fairness": {
                "same_train_seed": args.seed + 101,
                "same_eval_seed": args.seed + 202,
                "same_lr": args.lr,
                "same_steps": args.steps,
                "same_seq_len": args.seq_len,
                "same_batch_size": args.batch_size,
                "match_params": args.match_params,
                "param_delta": ce_params - std_params,
            },
            "config": vars(args),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
