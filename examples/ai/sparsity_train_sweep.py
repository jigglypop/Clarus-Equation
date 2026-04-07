"""CE Sparsity Training Sweep: train from scratch at different sparsity levels.

Character-level LM을 다양한 TopK 희소율로 처음부터 학습시킨다.
CE 예측: epsilon^2 = 4.87%에서 성능/효율 최적.

동일 파라미터, 동일 데이터, 동일 스텝으로 학습하고 validation loss를 비교한다.

Usage:
    py sparsity_train_sweep.py
    py sparsity_train_sweep.py --dim 256 --steps 3000 --data mytext.txt
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS2 = 0.0487
SPARSITY_RATIOS = [0.02, 0.03, 0.04, 0.0487, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.50, 1.00]


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in s]


class TopKGELU(nn.Module):
    """GELU + TopK: 상위 k개만 유지."""

    def __init__(self, ratio: float, dim: int):
        super().__init__()
        self.k = max(1, math.ceil(ratio * dim))
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(x)
        if self.ratio >= 1.0:
            return h
        _, topk_idx = torch.topk(h.abs(), self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask


class SparseMLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, sparsity: float = 1.0):
        super().__init__()
        h = dim * mult
        self.up = nn.Linear(dim, h, bias=False)
        self.act = TopKGELU(sparsity, h)
        self.down = nn.Linear(h, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class SparseAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class SparseBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mult: int = 4, sparsity: float = 1.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SparseAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = SparseMLP(dim, mult, sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SparseLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 256,
        ffn_mult: int = 4,
        sparsity: float = 1.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [SparseBlock(dim, n_heads, ffn_mult, sparsity) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
                nn.init.normal_(w, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def is_ce_sparsity(sparsity: float) -> bool:
    return math.isclose(sparsity, EPS2, rel_tol=0.0, abs_tol=1e-4)


def get_batch(data: torch.Tensor, seq_len: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model: SparseLM, data: torch.Tensor, seq_len: int, batch_size: int, device: str, n_eval: int = 20):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(data, seq_len, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def load_text(path: str | None) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    try:
        datasets = importlib.import_module("datasets")
        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
        return text[:500_000]
    except Exception:
        pass
    return ("The quick brown fox jumps over the lazy dog. " * 10000)


def prepare_data(text: str):
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = max(1000, int(len(data) * 0.05))
    train_data, val_data = data[:-n_val], data[-n_val:]
    return tok.vocab_size, train_data, val_data


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


def train_one(
    sparsity: float,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    batch_size: int,
    steps: int,
    lr: float,
    device: str,
) -> dict:
    model = SparseLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=seq_len,
        sparsity=sparsity,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    d_ff = dim * 4
    k = max(1, math.ceil(sparsity * d_ff))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = build_lr_scheduler(opt, steps)

    t0 = time.time()
    train_losses = []

    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_data, seq_len, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        train_losses.append(loss.item())

    elapsed = time.time() - t0
    final_train = mean_recent(train_losses, 50)
    final_val = eval_loss(model, val_data, seq_len, min(batch_size, 8), device, n_eval=30)

    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "sparsity": sparsity,
        "k": k,
        "d_ff": d_ff,
        "active_pct": sparsity * 100,
        "params": n_params,
        "train_loss": final_train,
        "val_loss": final_val,
        "val_ppl": loss_to_ppl(final_val),
        "time_sec": elapsed,
    }


def summarize_results(results):
    best = min(results, key=lambda r: r["val_loss"])
    dense = next((r for r in results if r["sparsity"] >= 1.0), results[-1])
    ce = next((r for r in results if is_ce_sparsity(r["sparsity"])), None)
    return best, dense, ce


def main():
    p = argparse.ArgumentParser(description="CE Sparsity Training Sweep")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("=" * 65)
    print("CE Sparsity Training Sweep (train from scratch)")
    print(f"  CE prediction: epsilon^2 = {EPS2:.4f} ({EPS2*100:.2f}%)")
    print(f"  dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"  seq_len={args.seq_len}, batch_size={args.batch_size}, steps={args.steps}")
    print(f"  device={args.device}")
    print(f"  sparsity levels: {[f'{r*100:.2f}%' for r in SPARSITY_RATIOS]}")
    print("=" * 65)

    text = load_text(args.data)
    vocab_size, train_data, val_data = prepare_data(text)
    print(f"\nData: {len(text)} chars, vocab={vocab_size}")

    print(f"\n{'sparsity':>10s} {'k':>5s} {'train':>8s} {'val':>8s} {'val_ppl':>10s} {'time':>6s}")
    print("-" * 55)

    results = []
    for ratio in SPARSITY_RATIOS:
        r = train_one(
            sparsity=ratio,
            train_data=train_data,
            val_data=val_data,
            vocab_size=vocab_size,
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
        )
        results.append(r)

        marker = " <-- CE" if is_ce_sparsity(ratio) else ""
        print(
            f"{r['active_pct']:>9.2f}% {r['k']:>5d} {r['train_loss']:>8.4f} "
            f"{r['val_loss']:>8.4f} {r['val_ppl']:>10.2f} {r['time_sec']:>5.0f}s{marker}",
            flush=True,
        )

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    best, dense, ce = summarize_results(results)

    print(f"  Dense (100%):        val_loss = {dense['val_loss']:.4f}, PPL = {dense['val_ppl']:.2f}")
    print(f"  Best:                {best['active_pct']:.2f}%, val_loss = {best['val_loss']:.4f}, PPL = {best['val_ppl']:.2f}")
    if ce:
        print(f"  CE (4.87%):          val_loss = {ce['val_loss']:.4f}, PPL = {ce['val_ppl']:.2f}")
        print(f"\n  CE vs Dense:         {ce['val_loss'] - dense['val_loss']:+.4f} val_loss")
        print(f"  CE vs Best:          {ce['val_loss'] - best['val_loss']:+.4f} val_loss")

    rank = sorted(results, key=lambda r: r["val_loss"])
    ce_rank = next((i + 1 for i, r in enumerate(rank) if is_ce_sparsity(r["sparsity"])), None)
    if ce_rank is not None:
        print(f"  CE rank: {ce_rank}/{len(rank)}")

    print(f"\n  --- Efficiency (FLOPs vs quality) ---")
    for r in results:
        flop_frac = r["sparsity"]
        loss_delta = r["val_loss"] - dense["val_loss"]
        print(f"    {r['active_pct']:>6.2f}% active: {1/max(flop_frac, 0.01):>5.1f}x fewer MLP FLOPs, val_loss {loss_delta:+.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "sparsity_train_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "eps2": EPS2, "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
