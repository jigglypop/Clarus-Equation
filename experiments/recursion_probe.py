"""First evidence probe for RecursiveEulerCEBlock (ClarusCell fixed-point).

The product claim is that weight-tied self-recursion adds *effective compute*
without adding parameters. This had zero measurements. Here we test it
directly on a tiny char-LM:

  - rec@K : ONE RecursiveEulerCEBlock applied K times (same weights).
            params constant across K; compute = K block-applications.
  - untied-N : N independent blocks (N x params) — the depth ceiling.

Claim under test:
  (1) PPL drops as K grows  -> recursion buys compute, not just iterations.
  (2) rec@K (1x params) approaches untied-K (Kx params) -> param efficiency.
  (3) fixed_point_loss / tol-halting actually converges (report depths).

Small, CPU, single seed -> directional first evidence, not proof.

    python -m experiments.recursion_probe
"""

from __future__ import annotations

import glob
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from reality_stone.clarus.ce_euler import RecursiveEulerCEBlock, fixed_point_loss

SEED = 0
D_MODEL, N_HEADS, BLOCK = 64, 4, 64
STEPS, BATCH, LR = 700, 32, 3e-3
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_corpus() -> str:
    files = sorted(glob.glob(os.path.join(REPO, "reality_stone", "**", "*.py"),
                             recursive=True))
    text = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                text.append(fh.read())
        except Exception:
            pass
        if sum(len(t) for t in text) > 600_000:
            break
    return "\n".join(text)[:600_000]


class RecLM(nn.Module):
    def __init__(self, vocab: int, n_blocks: int, max_iters: int,
                 tol: float | None = None) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.blocks = nn.ModuleList([
            RecursiveEulerCEBlock(D_MODEL, N_HEADS, BLOCK,
                                  max_iters=max_iters, tol=tol)
            for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))

    def mean_depth(self) -> float:
        ds = [b.last_depths.float().mean().item() for b in self.blocks
              if b.last_depths is not None]
        return sum(ds) / len(ds) if ds else 0.0


def batch(data, bs, gen):
    hi = data.numel() - BLOCK - 1
    ix = torch.randint(0, hi, (bs,), generator=gen)
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


def block_params(model: RecLM) -> int:
    return sum(p.numel() for p in model.blocks.parameters())


def train_eval(name, n_blocks, max_iters, train, val,
               tol=None, fp_reg=0.0) -> dict:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    vocab = int(train.max().item()) + 1
    model = RecLM(vocab, n_blocks, max_iters, tol=tol)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        x, y = batch(train, BATCH, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if fp_reg > 0.0:
            h = model.emb(x)
            loss = loss + fixed_point_loss(model.blocks[0], h, scale=fp_reg)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    egen = torch.Generator().manual_seed(SEED + 1)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for _ in range(30):
            x, y = batch(val, BATCH, egen)
            logits = model(x)
            l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            tot += l.item() * x.numel(); cnt += x.numel()
    return {"name": name, "block_params": block_params(model),
            "compute": n_blocks * max_iters, "ppl": math.exp(tot / cnt),
            "depth": model.mean_depth()}


def main() -> None:
    torch.manual_seed(SEED)
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(data.numel() * 0.9)
    train, val = data[:split], data[split:]
    print(f"corpus chars={data.numel():,} vocab={len(chars)} block={BLOCK}\n")

    configs = [
        ("rec@1 (baseline)", 1, 1, None, 0.0),
        ("rec@2",            1, 2, None, 0.0),
        ("rec@4",            1, 4, None, 0.0),
        ("rec@8",            1, 8, None, 0.0),
        ("untied-4 (4x params)", 4, 1, None, 0.0),
        ("rec@8 + fp_loss",  1, 8, None, 0.05),
        ("rec@8 tol=1e-2 (halting)", 1, 8, 1e-2, 0.0),
    ]
    rows = []
    for name, nb, mi, tol, fp in configs:
        r = train_eval(name, nb, mi, train, val, tol=tol, fp_reg=fp)
        rows.append(r)
        print(f"  {name:<26} blkP={r['block_params']//1000}K  "
              f"compute={r['compute']}  PPL={r['ppl']:.3f}  "
              f"mean_depth={r['depth']:.2f}")

    base = rows[0]["ppl"]
    rec4 = next(r for r in rows if r["name"] == "rec@4")["ppl"]
    rec8 = next(r for r in rows if r["name"] == "rec@8")["ppl"]
    untied = next(r for r in rows if r["name"].startswith("untied"))["ppl"]
    print(f"\n  rec@1 PPL = {base:.3f}")
    print(f"  rec@8 PPL = {rec8:.3f}  ({(rec8-base)/base*100:+.1f}% vs rec@1, "
          f"SAME params)")
    print(f"  untied-4  = {untied:.3f}  (4x params)")
    gap = (rec8 - untied) / (base - untied) if base != untied else 0.0
    print(f"\n  claim (1) recursion buys compute : "
          f"{'[YES]' if rec8 < base - 1e-3 else '[NO]'}  "
          f"(rec@8 {'<' if rec8 < base else '>='} rec@1)")
    print(f"  claim (2) approaches untied ceiling: rec@8 closes "
          f"{max(0,(1-gap))*100:.0f}% of the rec@1->untied4 gap at 1/4 the params")


if __name__ == "__main__":
    main()
