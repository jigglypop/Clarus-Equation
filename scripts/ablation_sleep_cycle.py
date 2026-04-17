"""Sleep-cycle ablation on a small ClarusLM (sanity check, not full bench).

비교:
  baseline     : standard AdamW per-batch update (논문 사양 위반)
  sleep        : WAKE-NREM-REM cycle per spec
  sleep+ter    : sleep cycle + 3분배 가중치 분류기

목적: 작은 합성 sequence 데이터에서 spec 구현이 baseline 대비 다른 거동을
보이는지 빠르게 확인 (수치는 실제 LLM과 다를 수 있음, 비교 목적).

Run:
    .venv/Scripts/python.exe scripts/ablation_sleep_cycle.py --device cpu
    .venv/Scripts/python.exe scripts/ablation_sleep_cycle.py --device cuda --steps 200
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def safe_print(*a, **k):
    print(*a, **k, flush=True)


def make_synthetic_ngram_data(vocab_size: int, n_tokens: int, n_repeat: int = 4,
                              seed: int = 0) -> torch.Tensor:
    """가짜 n-gram 데이터: predictable n=2 transitions로 학습 가능성 보장."""
    g = torch.Generator(); g.manual_seed(seed)
    base = torch.randint(0, vocab_size, (n_tokens // n_repeat,), generator=g)
    seq = base.repeat(n_repeat)
    seq = seq[:n_tokens]
    return seq


def make_batch(data, batch, seq_len, gen):
    n = data.shape[0]
    starts = torch.randint(0, n - seq_len - 1, (batch,), generator=gen, device=data.device)
    x = torch.stack([data[s:s + seq_len] for s in starts.tolist()])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts.tolist()])
    return x, y


@torch.no_grad()
def val_perplexity(model, data, batch, seq_len, gen, n=8):
    losses = []
    was_t = model.training; model.eval()
    for _ in range(n):
        x, y = make_batch(data, batch, seq_len, gen)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(float(ce.item()))
    if was_t: model.train()
    return math.exp(min(sum(losses) / len(losses), 20.0))


def train_baseline(model, data, args, seed):
    """Standard AdamW per-batch. 논문 사양 위반 (비교 baseline)."""
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(); gen.manual_seed(seed)
    model.train()
    for step in range(args.steps):
        x, y = make_batch(data, args.batch, args.seq_len, gen)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()


def train_sleep(model, data, args, seed, use_ternary=False):
    """3_Sleep.md spec WAKE-NREM-REM cycle."""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location('sleep_ft',
        str(Path(__file__).parent / 'sleep_finetune_lm.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['sleep_ft'] = mod
    spec.loader.exec_module(mod)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(); gen.manual_seed(seed)
    if hasattr(model, 'set_lambda_schedule'):
        model.set_lambda_schedule(total_steps=args.steps, warmup_steps=args.steps // 10)
    classifier = None
    if use_ternary:
        from clarus.sparsity import TernaryClassifier
        from clarus.constants import ACTIVE_RATIO, STRUCT_RATIO
        classifier = TernaryClassifier(model,
            active_ratio=ACTIVE_RATIO, struct_ratio=STRUCT_RATIO)
    model.train()
    for cyc in range(args.steps):
        lam = model._current_lambda_curv() if hasattr(model, '_current_lambda_curv') else 0.0
        mod.wake_phase(model, optim, data, args.batch, args.seq_len, gen,
                       cycle_len=args.cycle_len, lam_curv=lam)
        if classifier is not None:
            classifier.update_freq()
            classifier.apply_grad_mask(allow_struct=True)
        mod.nrem_phase(model, optim, eta_nrem=args.eta_nrem, top_eps=args.top_eps)
        if classifier is not None and cyc % 10 == 0:
            classifier.reclassify()
        if args.rem_every > 0 and cyc % args.rem_every == 0:
            mod.rem_phase(model, optim, data, args.batch, args.seq_len, gen,
                          n_trials=2, lam_curv=lam,
                          noise_sigma=0.001, top_eps=args.top_eps)


def build_model(vocab, dim, n_layers, n_heads, seq_len, sparsity, mix_rank):
    sys.path.insert(0, str(Path(__file__).parent.parent / 'examples' / 'ai'))
    import clarus_lm
    return clarus_lm.ClarusLM(
        vocab_size=vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
        max_seq_len=seq_len, ffn_mult=4, mix_rank=mix_rank,
        lambda_curv=0.001, lambda_mix=0.001,
        sparsity=sparsity, dense=False,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=80,
                    help="Optimizer steps (baseline) / cycles (sleep).")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--mix-rank", type=int, default=4)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cycle-len", type=int, default=4)
    ap.add_argument("--eta-nrem", type=float, default=0.01)
    ap.add_argument("--top-eps", type=float, default=0.0487)
    ap.add_argument("--rem-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-corpus", type=int, default=4096)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    safe_print("=" * 60)
    safe_print(f" Sleep cycle ablation (sanity)")
    safe_print(f"  device={device}  steps={args.steps}  vocab={args.vocab}  dim={args.dim}")
    safe_print(f"  seq_len={args.seq_len}  cycle_len={args.cycle_len}")
    safe_print("=" * 60)

    # 합성 데이터: 학습 가능한 패턴 (n-gram repeat).
    full_data = make_synthetic_ngram_data(args.vocab, args.n_corpus, n_repeat=4,
                                          seed=args.seed).to(device)
    n_train = int(0.9 * full_data.shape[0])
    train_data = full_data[:n_train]
    val_data = full_data[n_train:]
    val_gen = torch.Generator(); val_gen.manual_seed(args.seed + 1)

    results = {}
    for name, fn in [
        ("baseline (AdamW)", lambda m: train_baseline(m, train_data, args, seed=args.seed + 2)),
        ("sleep (spec)",     lambda m: train_sleep(m, train_data, args, seed=args.seed + 2,
                                                   use_ternary=False)),
        ("sleep + ternary",  lambda m: train_sleep(m, train_data, args, seed=args.seed + 2,
                                                   use_ternary=True)),
    ]:
        torch.manual_seed(args.seed)  # 동일 init
        m = build_model(args.vocab, args.dim, args.n_layers, args.n_heads,
                        args.seq_len, args.sparsity, args.mix_rank).to(device)
        ppl_init = val_perplexity(m, val_data, args.batch, args.seq_len, val_gen)
        t0 = time.perf_counter()
        fn(m)
        elapsed = time.perf_counter() - t0
        ppl_final = val_perplexity(m, val_data, args.batch, args.seq_len, val_gen)
        results[name] = (ppl_init, ppl_final, elapsed)
        safe_print(f"\n[{name}]")
        safe_print(f"  ppl: {ppl_init:.2f}  ->  {ppl_final:.2f}  (delta {ppl_final - ppl_init:+.2f})")
        safe_print(f"  time: {elapsed:.2f}s")

    safe_print("\n" + "=" * 60)
    safe_print(" Summary")
    safe_print("=" * 60)
    safe_print(f"{'method':<22s}  {'ppl_init':>10s}  {'ppl_final':>10s}  {'time_s':>8s}")
    for name, (pi, pf, t) in results.items():
        safe_print(f"{name:<22s}  {pi:>10.2f}  {pf:>10.2f}  {t:>8.2f}")


if __name__ == "__main__":
    main()
