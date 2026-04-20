"""Length extrapolation benchmark — Euler-CE deep-dive.

Models are initialised at `train_block` (so block-aware design choices,
such as EulerCE's `base = π^D_eff · N`, are computed at training scale).
At evaluation time `model.extend_to(N_eval)` regrows positional buffers
so the same parameters can be applied to longer sequences.

Variants
--------
std_rope         : plain RoPE baseline.
rope_alibi       : RoPE + ALiBi linear decay (per-head learnable slope).
mra              : RoPE freq + ζ amplitude (lean MRA).
mra_bias         : lean MRA + log-distance bias.
euler_ce_k1      : full Euler-CE (π-rotation + e-decay + block-aware base).
euler_no_decay   : Euler-CE with e_gate frozen → π-rotation + block-aware
                   base only. Isolates the rotation contribution.
euler_no_pi      : Euler-CE with pi_gate frozen → e-decay only. Isolates
                   the additive distance bias contribution.

The pair (euler_no_decay, euler_no_pi) decomposes Euler-CE's known
extrapolation strength (≈+6 % degrad at 4× vs RoPE +27 %) into its
two components.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from clarus.ce_euler import EulerCEBlock, EulerCEMinimalBlock
from clarus.ce_mra import MRABlock

from examples.ai.bench_recursive_euler import (
    RoPEAttnBlock, RoPEAlibiAttnBlock, NoPEAttnBlock, XPosAttnBlock,
    load_docs, encode,
)


def _freeze_gate(block: EulerCEBlock, which: str) -> None:
    """Force a per-head gate of an EulerCE attn to zero (sigmoid(-inf) = 0)."""
    attn = block.attn
    name = "pi_gate_logit" if which == "pi" else "e_gate_logit"
    p = getattr(attn, name)
    with torch.no_grad():
        p.data.fill_(-1e4)
    p.requires_grad_(False)


class ExtrapLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, train_block, variant):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.variant = variant
        if variant == "std_rope":
            blocks = [RoPEAttnBlock(d_model, n_heads, train_block)
                      for _ in range(n_layers)]
        elif variant == "nope":
            blocks = [NoPEAttnBlock(d_model, n_heads, train_block)
                      for _ in range(n_layers)]
        elif variant == "xpos":
            blocks = [XPosAttnBlock(d_model, n_heads, train_block)
                      for _ in range(n_layers)]
        elif variant == "rope_alibi":
            blocks = [RoPEAlibiAttnBlock(d_model, n_heads, train_block)
                      for _ in range(n_layers)]
        elif variant == "euler_ce_k1":
            blocks = [EulerCEBlock(d_model, n_heads, train_block,
                                   layer_idx=i, n_layers=n_layers)
                      for i in range(n_layers)]
        elif variant == "euler_no_decay":
            blocks = [EulerCEBlock(d_model, n_heads, train_block,
                                   layer_idx=i, n_layers=n_layers)
                      for i in range(n_layers)]
            for blk in blocks:
                _freeze_gate(blk, "e")
        elif variant == "euler_no_pi":
            blocks = [EulerCEBlock(d_model, n_heads, train_block,
                                   layer_idx=i, n_layers=n_layers)
                      for i in range(n_layers)]
            for blk in blocks:
                _freeze_gate(blk, "pi")
        elif variant == "mra":
            blocks = [MRABlock(d_model, n_heads, train_block)
                      for _ in range(n_layers)]
        elif variant == "mra_bias":
            blocks = [MRABlock(d_model, n_heads, train_block,
                               decay_mode="bias")
                      for _ in range(n_layers)]
        elif variant.startswith("min_"):
            # 2-bit minimal Euler-CE: head_types = {"nope","alibi","rope",
            # "xpos","mix","all"} corresponds to literature analogues.
            spec = variant[len("min_"):]
            blocks = [EulerCEMinimalBlock(d_model, n_heads, train_block,
                                          head_types=spec)
                      for _ in range(n_layers)]
        else:
            raise ValueError(variant)
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def extend_to(self, new_block: int) -> None:
        for blk in self.blocks:
            # Sub-module attention (EulerCEBlock, MRABlock): forward call.
            attn = blk._modules.get("attn")
            if attn is not None and hasattr(attn, "extend_to"):
                attn.extend_to(new_block)
                continue
            # Method-based attention (RoPEAttnBlock): block itself owns it.
            if hasattr(blk, "extend_to"):
                blk.extend_to(new_block)

    def forward(self, idx):
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def batch_iter(data, batch, block, seed):
    g = torch.Generator(device=data.device).manual_seed(seed) if data.is_cuda \
        else torch.Generator().manual_seed(seed)
    n = len(data) - block - 1
    while True:
        if data.is_cuda:
            idx = torch.randint(0, n, (batch,), generator=g, device=data.device)
        else:
            idx = torch.randint(0, n, (batch,), generator=g)
        x = torch.stack([data[i:i + block] for i in idx])
        y = torch.stack([data[i + 1:i + 1 + block] for i in idx])
        yield x, y


@torch.no_grad()
def eval_at(model, data, batch, block, n_batches=12):
    model.eval()
    total = 0.0
    it = batch_iter(data, batch, block, seed=42)
    for _ in range(n_batches):
        x, y = next(it)
        logits = model(x)
        total += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return total / n_batches


def train_and_extrap(data_train, data_val, vocab, variant,
                     d_model, n_layers, train_block, eval_blocks,
                     batch, steps, lr, seed, device, pbar_desc=""):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    # Init at train_block — block-aware design choices use training scale.
    model = ExtrapLM(vocab, d_model, 4, n_layers, train_block, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    it = batch_iter(data_train, batch, train_block, seed=seed)
    pbar = tqdm(range(1, steps + 1), desc=pbar_desc, leave=False, dynamic_ncols=True)
    for s in pbar:
        x, y = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 50 == 0 or s == steps:
            pbar.set_postfix(loss=f"{float(loss):.3f}")
    pbar.close()

    # Grow positional buffers to the largest eval block in one shot.
    model.extend_to(max(eval_blocks))

    ppls = {}
    for n in eval_blocks:
        if len(data_val) - n - 1 < batch:
            ppls[n] = None
            continue
        loss = eval_at(model, data_val, batch=batch, block=n)
        ppls[n] = math.exp(loss)
    return {"ppl_by_n": ppls,
            "n_params": sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_block", type=int, default=64)
    p.add_argument("--eval_mults", type=str, default="1,1.5,2,3,4")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--corpus_chars", type=int, default=400_000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str,
                   default="examples/ai/results/mra_extrap.json")
    p.add_argument("--variants", type=str, default="default",
                   help="comma-separated variant names, or 'default' / "
                        "'minimal' / 'all'")
    args = p.parse_args()

    eval_mults = [float(s) for s in args.eval_mults.split(",")]
    eval_blocks = sorted({int(args.train_block * m) for m in eval_mults})

    docs_root = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    text = load_docs(docs_root, args.corpus_chars)
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = encode(text, stoi).to(args.device)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)
    print(f"corpus: {len(data)} chars, vocab {vocab}, device {args.device}")
    print(f"train_block = {args.train_block}, eval_blocks = {eval_blocks}\n")

    DEFAULT_VARIANTS = [
        "nope",
        "std_rope",
        "rope_alibi",
        "xpos",
        "mra",
        "mra_bias",
        "euler_no_pi",
        "euler_no_decay",
        "euler_ce_k1",
    ]
    MINIMAL_VARIANTS = [
        "min_nope", "min_alibi", "min_rope", "min_xpos",
        "min_mix", "min_all",
    ]
    if args.variants == "default":
        variants = DEFAULT_VARIANTS
    elif args.variants == "minimal":
        variants = MINIMAL_VARIANTS
    elif args.variants == "all":
        variants = DEFAULT_VARIANTS + MINIMAL_VARIANTS
    else:
        variants = args.variants.split(",")
    results = {}
    for variant in variants:
        print(f"=== {variant} ({args.seeds} seeds) ===")
        per_seed = {n: [] for n in eval_blocks}
        params = None
        for seed in range(args.seeds):
            r = train_and_extrap(
                train_data, val_data, vocab, variant,
                d_model=args.d_model, n_layers=args.n_layers,
                train_block=args.train_block, eval_blocks=eval_blocks,
                batch=args.batch, steps=args.steps, lr=3e-4, seed=seed,
                device=args.device,
                pbar_desc=f"{variant} seed{seed+1}/{args.seeds}",
            )
            params = r["n_params"]
            for n in eval_blocks:
                if r["ppl_by_n"][n] is not None:
                    per_seed[n].append(r["ppl_by_n"][n])
        ppl_mean = {n: sum(v) / len(v) if v else None for n, v in per_seed.items()}
        ppl_std = {
            n: ((sum((x - ppl_mean[n]) ** 2 for x in v)
                 / max(len(v) - 1, 1)) ** 0.5) if v else None
            for n, v in per_seed.items()
        }
        results[variant] = {
            "params": params,
            "ppl_mean": ppl_mean,
            "ppl_std": ppl_std,
            "ppls": per_seed,
        }
        msg = f"  {variant:14s}"
        for n in eval_blocks:
            if ppl_mean[n] is not None:
                msg += f"  N={n}: {ppl_mean[n]:6.2f}±{ppl_std[n]:4.2f}"
        print(msg)

    print("\n=== relative degradation (PPL(N)/PPL(N_train) - 1) ===")
    for variant in variants:
        r = results[variant]
        base = r["ppl_mean"][args.train_block]
        if base is None:
            continue
        msg = f"  {variant:14s}  base={base:.2f}  "
        for n in eval_blocks:
            if r["ppl_mean"][n] is not None and n != args.train_block:
                deg = r["ppl_mean"][n] / base - 1.0
                msg += f"  ×{n // args.train_block:.0f}: {deg:+.1%}"
        print(msg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args),
                   "eval_blocks": eval_blocks,
                   "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
