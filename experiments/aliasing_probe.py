"""Corollary test: is RoPE's length-OOD failure caused by rotary-phase
aliasing at unseen positions?

Prediction (from euler_ce_formulation.md, corollary): the damage comes from
query/key phases theta = j*omega at positions j > N_train that were never
seen in training. If so, *position interpolation* — rescaling eval positions
back into [0, N_train) so every phase stays in-range — should rescue RoPE.
ALiBi (decay, no phase) should be unaffected by interpolation either way.

This both verifies the mechanism and is the "corrected-rotation baseline"
(the NTK/YaRN family compresses RoPE phases for exactly this reason).

    python -u -m experiments.aliasing_probe
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.ood_length_repro import (
    TinyLM, load_corpus, batch, TRAIN_BLOCK)

SEED = 0
STEPS, BATCH, LR = 600, 32, 3e-3
EVAL_LENS = [64, 512, 2048]


def train(head_type: str, data) -> TinyLM:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    vocab = int(data.max().item()) + 1
    model = TinyLM(vocab, head_type)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        x, y = batch(data, TRAIN_BLOCK, BATCH, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_ppl(model: TinyLM, val, L: int, interpolate: bool) -> float:
    model.eval()
    model.extend_to(L)
    if interpolate and L > TRAIN_BLOCK:
        # position interpolation: squeeze [0, L) into [0, TRAIN_BLOCK)
        scale = (TRAIN_BLOCK - 1) / (L - 1)
        pos = torch.arange(L, dtype=torch.float32) * scale
        for b in model.blocks:
            b.attn.pos = pos.clone()
    else:
        for b in model.blocks:
            b.attn.pos = torch.arange(L, dtype=torch.float32)
    gen = torch.Generator().manual_seed(SEED + 1)
    tot, cnt = 0.0, 0
    bs = 16 if L <= 512 else 3
    nb = 12 if L <= 512 else 4
    for _ in range(nb):
        x, y = batch(val, L, bs, gen)
        logits = model(x)
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += l.item() * x.numel(); cnt += x.numel()
    return math.exp(tot / cnt)


def main() -> None:
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    d = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(d.numel() * 0.9)
    train_d, val = d[:split], d[split:]
    print(f"corpus={d.numel():,} vocab={len(chars)} train_block={TRAIN_BLOCK}\n",
          flush=True)

    far = EVAL_LENS[-1]
    for ht in ("rope", "alibi"):
        model = train(ht, train_d)
        base = eval_ppl(model, val, TRAIN_BLOCK, interpolate=False)
        raw = eval_ppl(model, val, far, interpolate=False)
        itp = eval_ppl(model, val, far, interpolate=True)
        dr = (raw - base) / base * 100
        di = (itp - base) / base * 100
        print(f"  {ht:<6} PPL@64={base:.3f}  "
              f"@{far} raw={raw:.2f} ({dr:+.1f}%)  "
              f"interp={itp:.2f} ({di:+.1f}%)", flush=True)
        if ht == "rope":
            rescue = (dr - di) / dr * 100 if dr > 0 else 0
            print(f"         -> interpolation removes {rescue:.0f}% of RoPE's "
                  f"degradation", flush=True)
            verdict = di < dr * 0.5
            print("         -> COROLLARY (aliasing is the cause): "
                  + ("[CONFIRMED]" if verdict else "[NOT CONFIRMED]"),
                  flush=True)


if __name__ == "__main__":
    main()
