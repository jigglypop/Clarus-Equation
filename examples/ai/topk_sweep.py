"""CE TopK Sparsity Sweep: 4.87% is optimal?

GPT-2 MLP activation에 TopK를 적용, 희소율별 perplexity를 측정한다.
CE 예측: epsilon^2 = 4.87%에서 최적 (또는 최적 근방).

Usage:
    py topk_sweep.py
    py topk_sweep.py --max-tokens 20000 --seq-len 256
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

EPS2 = 0.0487

SPARSITY_RATIOS = [
    0.01, 0.02, 0.03, 0.04, 0.0487, 0.06, 0.08,
    0.10, 0.15, 0.20, 0.30, 0.50, 1.00,
]


class TopKActivation(nn.Module):
    """GELU 후 TopK만 유지, 나머지 0."""

    def __init__(self, ratio: float, dim: int):
        super().__init__()
        self.k = max(1, math.ceil(ratio * dim))
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(x)
        if self.ratio >= 1.0:
            return h
        _, topk_idx = torch.topk(h.abs(), self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask


def get_mlp_hidden_dim(model: GPT2LMHeadModel) -> int:
    c_fc = model.transformer.h[0].mlp.c_fc
    if hasattr(c_fc, "out_features"):
        return c_fc.out_features
    if hasattr(c_fc, "nf"):
        return c_fc.nf
    return c_fc.weight.shape[-1]


def is_ce_ratio(ratio: float) -> bool:
    return math.isclose(ratio, EPS2, rel_tol=0.0, abs_tol=1e-4)


def patch_mlp_activation(model: GPT2LMHeadModel, ratio: float):
    """모든 블록의 MLP activation을 TopK로 교체."""
    d_ff = get_mlp_hidden_dim(model)
    for block in model.transformer.h:
        block.mlp.act = TopKActivation(ratio, d_ff)


@contextmanager
def patched_mlp_activation(model: GPT2LMHeadModel, ratio: float):
    """실험 동안만 activation을 바꾸고, 끝나면 원래 모듈로 복원."""
    originals = [block.mlp.act for block in model.transformer.h]
    patch_mlp_activation(model, ratio)
    try:
        yield
    finally:
        for block, original in zip(model.transformer.h, originals):
            block.mlp.act = original


@torch.no_grad()
def measure_perplexity(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    seq_len: int = 128,
    stride: int = 64,
    max_tokens: int = 5000,
) -> float:
    model.eval()
    n = min(len(input_ids), max_tokens)
    nlls = []
    for begin in range(0, max(1, n - seq_len), stride):
        end = min(begin + seq_len, n)
        chunk = input_ids[begin:end].unsqueeze(0)
        target = chunk.clone()
        if begin > 0:
            overlap = seq_len - stride
            target[:, :overlap] = -100
        out = model(chunk, labels=target)
        v = out.loss.item()
        if not math.isfinite(v):
            return float("inf")
        nlls.append(v)
    if not nlls:
        return float("inf")
    return math.exp(sum(nlls) / len(nlls))


@torch.no_grad()
def measure_active_stats(model: GPT2LMHeadModel, input_ids: torch.Tensor, seq_len: int = 128):
    """실제 활성 뉴런 비율 측정."""
    model.eval()
    chunk = input_ids[:seq_len].unsqueeze(0)

    active_fracs = []
    hooks = []

    def hook_fn(module, inputs, output):
        nonzero = (output.abs() > 1e-8).float().mean().item()
        active_fracs.append(nonzero)

    for block in model.transformer.h:
        h = block.mlp.act.register_forward_hook(hook_fn)
        hooks.append(h)

    model(chunk)

    for h in hooks:
        h.remove()

    return sum(active_fracs) / len(active_fracs) if active_fracs else 0.0


def load_eval_data(tokenizer: GPT2Tokenizer, max_tokens: int) -> torch.Tensor:
    """WikiText-2 test set 로드."""
    try:
        datasets = importlib.import_module("datasets")
        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        text = (
            "The tower is 324 metres tall, about the same height as an 81-storey building, "
            "and the tallest structure in Paris. Its base is square, measuring 125 metres on each side. "
        ) * 500
        print("  (fallback: synthetic eval text)")

    ids = tokenizer.encode(text)
    ids = ids[: max_tokens + 256]
    return torch.tensor(ids, dtype=torch.long)


def summarize_results(results):
    valid = [r for r in results if math.isfinite(r["ppl"])]
    if not valid:
        return None, None, None, None
    best = min(valid, key=lambda r: r["ppl"])
    baseline = next((r for r in valid if r["ratio"] >= 1.0), valid[-1])
    ce_result = next((r for r in valid if is_ce_ratio(r["ratio"])), None)
    return valid, best, baseline, ce_result


def run_sweep(args):
    print("=" * 60)
    print("CE TopK Sparsity Sweep")
    print(f"  CE prediction: epsilon^2 = {EPS2:.4f} ({EPS2*100:.2f}%)")
    print(f"  seq_len={args.seq_len}, max_tokens={args.max_tokens}")
    print("=" * 60)

    print("\nLoading GPT-2...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    d_ff = get_mlp_hidden_dim(model)
    n_layers = len(model.transformer.h)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  d_model=768, d_ff={d_ff}, layers={n_layers}, params={n_params/1e6:.1f}M")

    print("\nLoading eval data...", flush=True)
    eval_ids = load_eval_data(tokenizer, args.max_tokens)
    print(f"  {len(eval_ids)} tokens loaded")

    print(f"\n{'ratio':>8s} {'k':>6s} {'active%':>8s} {'PPL':>10s} {'time':>6s}")
    print("-" * 45)

    results = []

    for ratio in SPARSITY_RATIOS:
        k = max(1, math.ceil(ratio * d_ff))
        with patched_mlp_activation(model, ratio):
            t0 = time.time()
            ppl = measure_perplexity(model, eval_ids, args.seq_len, args.stride, args.max_tokens)
            elapsed = time.time() - t0
            actual_active = measure_active_stats(model, eval_ids, args.seq_len)

        marker = " <-- CE" if is_ce_ratio(ratio) else ""
        print(
            f"{ratio*100:>7.2f}% {k:>6d} {actual_active*100:>7.2f}% {ppl:>10.2f} {elapsed:>5.1f}s{marker}",
            flush=True,
        )

        results.append({
            "ratio": ratio,
            "k": k,
            "active_frac": actual_active,
            "ppl": ppl,
            "time_sec": elapsed,
        })

    summary = summarize_results(results)
    valid, best, baseline, ce_result = summary
    if not valid:
        print("\nAll runs diverged.")
        return results

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Baseline (100%):     PPL = {baseline['ppl']:.2f}")
    print(f"  Best sparsity:       {best['ratio']*100:.2f}% (k={best['k']}), PPL = {best['ppl']:.2f}")
    if ce_result:
        print(f"  CE prediction (4.87%): PPL = {ce_result['ppl']:.2f}")

    print(f"\n  Best vs Baseline:    {'%.2f' % (best['ppl'] - baseline['ppl'])} PPL difference")
    if ce_result:
        print(f"  CE vs Baseline:      {'%.2f' % (ce_result['ppl'] - baseline['ppl'])} PPL difference")
        print(f"  CE vs Best:          {'%.2f' % (ce_result['ppl'] - best['ppl'])} PPL difference")

    rank = sorted(valid, key=lambda r: r["ppl"])
    ce_rank = next((i + 1 for i, r in enumerate(rank) if is_ce_ratio(r["ratio"])), None)
    if ce_rank:
        print(f"\n  CE rank: {ce_rank}/{len(rank)}")

    dense_ppl = baseline["ppl"]
    print(f"\n  --- Efficiency frontier ---")
    for r in valid:
        if r["ratio"] >= 1.0:
            continue
        flop_ratio = r["ratio"]
        ppl_overhead = (r["ppl"] / dense_ppl - 1) * 100
        print(f"    {r['ratio']*100:>6.2f}% active: {1/flop_ratio:>5.1f}x fewer MLP FLOPs, +{ppl_overhead:>5.1f}% PPL")

    out_path = os.path.join(os.path.dirname(__file__), "topk_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "d_ff": d_ff, "eps2": EPS2}, f, indent=2)
    print(f"\nSaved: {out_path}")

    return results


def main():
    p = argparse.ArgumentParser(description="CE TopK Sparsity Sweep on GPT-2")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    args = p.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
