"""Reproducible length-OOD experiment for EulerCEMinimal head-types.

Backs (or refutes) the table in docs/7_AGI/19_OOD_Generalization.md, which
claimed a Tier-1/Tier-2 split at 32x length extrapolation but shipped no
runnable script. Here we actually TRAIN a tiny char-level LM at block=64 for
each canonical head-type, then EVALUATE at longer lengths and report PPL
degradation. The thesis under test:

    pure rotation (RoPE) is the only head-type that breaks at length OOD;
    nope / alibi / xpos extrapolate.

Small by design (CPU, ~minutes). Not a SOTA claim — a falsifiable check.

    python -m experiments.ood_length_repro
"""

from __future__ import annotations

import glob
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from reality_stone.clarus.ce_euler import EulerCEMinimalBlock

D_MODEL, N_HEADS, N_LAYERS = 64, 4, 2
TRAIN_BLOCK = 64
STEPS = int(os.environ.get("STEPS", 800))
BATCH, LR = 32, 3e-3
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
EVAL_LENS = [int(x) for x in os.environ.get(
    "EVAL_LENS", "64,128,256,512,1024").split(",")]
HEAD_TYPES = ["nope", "alibi", "rope", "xpos"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_corpus() -> str:
    files = glob.glob(os.path.join(REPO, "reality_stone", "**", "*.py"),
                      recursive=True)
    text = []
    for f in sorted(files):
        try:
            with open(f, encoding="utf-8") as fh:
                text.append(fh.read())
        except Exception:
            pass
        if sum(len(t) for t in text) > 600_000:
            break
    return "\n".join(text)[:600_000]


class TinyLM(nn.Module):
    def __init__(self, vocab: int, head_types: str) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.blocks = nn.ModuleList([
            EulerCEMinimalBlock(D_MODEL, N_HEADS, TRAIN_BLOCK,
                                head_types=head_types)
            for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)

    def extend_to(self, L: int) -> None:
        for b in self.blocks:
            b.attn.extend_to(L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


def batch(data: torch.Tensor, L: int, bs: int, gen: torch.Generator):
    hi = data.numel() - L - 1
    ix = torch.randint(0, hi, (bs,), generator=gen)
    x = torch.stack([data[i:i + L] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + L] for i in ix])
    return x, y


@torch.no_grad()
def eval_ppl(model: TinyLM, data: torch.Tensor, L: int, gen: torch.Generator,
             n_batches: int) -> float:
    model.eval()
    model.extend_to(L)
    tot, cnt = 0.0, 0
    bs = max(2, min(BATCH, 16 if L <= 256 else (4 if L <= 1024 else 2)))
    for _ in range(n_batches):
        x, y = batch(data, L, bs, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += loss.item() * x.numel()
        cnt += x.numel()
    return math.exp(tot / cnt)


def run_one(head_type: str, train, val, seed: int) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    vocab = int(train.max().item()) + 1
    model = TinyLM(vocab, head_type)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        x, y = batch(train, TRAIN_BLOCK, BATCH, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    egen = torch.Generator().manual_seed(seed + 1)
    ppl = {L: eval_ppl(model, val, L, egen,
                       n_batches=(20 if L <= 256 else (8 if L <= 1024 else 3)))
           for L in EVAL_LENS}
    base = ppl[TRAIN_BLOCK]
    deg = {L: (ppl[L] - base) / base * 100.0 for L in EVAL_LENS}
    return {"params": n_params, "ppl": ppl, "deg": deg}


def main() -> None:
    torch.manual_seed(SEEDS[0])
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = data.numel()
    split = int(n * 0.9)
    train, val = data[:split], data[split:]
    print(f"corpus chars={n:,} vocab={len(chars)} "
          f"train_block={TRAIN_BLOCK} eval_lens={EVAL_LENS}\n")

    print(f"  seeds={SEEDS}\n")
    # collect degradation at the far length across seeds
    far_L = EVAL_LENS[-1]
    deg_far = {ht: [] for ht in HEAD_TYPES}
    deg_mean = {ht: {L: [] for L in EVAL_LENS} for ht in HEAD_TYPES}
    for seed in SEEDS:
        for ht in HEAD_TYPES:
            r = run_one(ht, train, val, seed)
            deg_far[ht].append(r["deg"][far_L])
            for L in EVAL_LENS:
                deg_mean[ht][L].append(r["deg"][L])
        print(f"  seed {seed} done")

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs); return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    print(f"\n  degradation vs train-len {TRAIN_BLOCK}, mean over {len(SEEDS)} seed(s)\n")
    hdr = "  head    " + "".join(f"{L:>10}" for L in EVAL_LENS) + "   tier"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for ht in HEAD_TYPES:
        far = mean(deg_mean[ht][far_L])
        tier = 2 if far > 25 else 1
        cells = "".join(f"{mean(deg_mean[ht][L]):>+9.1f}%" for L in EVAL_LENS)
        print(f"  {ht:<6}{cells}   T{tier}")

    far = {ht: mean(deg_far[ht]) for ht in HEAD_TYPES}
    print(f"\n  far-length {far_L} degradation (mean +/- std):")
    for ht in HEAD_TYPES:
        print(f"    {ht:<6} {far[ht]:>+8.1f}% +/- {std(deg_far[ht]):.1f}")
    worst = max(far, key=far.get)
    print(f"\n  worst extrapolator @ {far_L}: {worst} (+{far[worst]:.1f}%)")
    thesis = worst == "rope"
    print("  THESIS (pure rotation = worst extrapolator): "
          + ("[SUPPORTED]" if thesis else "[NOT SUPPORTED]"))


if __name__ == "__main__":
    main()
