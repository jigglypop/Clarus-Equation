"""Continual learning test: catastrophic forgetting baseline vs sleep cycle.

논문 (3_Sleep.md 7.3, 5_Sparsity.md) 핵심 청구 직접 검증:
  데이터 A 학습 -> 데이터 B 학습 후, A에 대한 perplexity가 얼마나 나빠지는가?

  baseline (standard AdamW): 모든 weight 매 step 업데이트 -> A에 대한 표현
                              덮어쓰기 -> 큰 forgetting 예상.
  sleep (spec):              NREM top-eps^2 mask + ternary BG freeze ->
                              기존 표현 보존 -> 작은 forgetting 예상.

측정 지표:
  forgetting = ppl_A_after_B - ppl_A_after_A   (낮을수록 보존 잘함)
  plasticity = ppl_B_initial - ppl_B_after_B   (높을수록 새거 잘 배움)

Run:
  .venv/Scripts/python.exe scripts/continual_test.py \
      --ckpt clarus/clarus_lm_kogpt2.pt \
      --data examples/ai/train_data.txt \
      --device cuda --steps-per-phase 150
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def safe_print(*a, **k):
    try:
        print(*a, **k, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(((" ".join(map(str, a))) + "\n").encode("utf-8", "replace"))


def make_batch(data, batch, seq_len, gen):
    n = data.shape[0]
    starts = torch.randint(0, n - seq_len - 1, (batch,), generator=gen, device=data.device)
    x = torch.stack([data[s:s + seq_len] for s in starts.tolist()])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts.tolist()])
    return x, y


@torch.no_grad()
def val_perplexity(model, data, batch, seq_len, gen, n_batches=16):
    losses = []
    was_t = model.training
    model.eval()
    for _ in range(n_batches):
        x, y = make_batch(data, batch, seq_len, gen)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(float(ce.item()))
    if was_t:
        model.train()
    return math.exp(min(sum(losses) / len(losses), 20.0))


def train_baseline(model, data, args, seed):
    """Standard AdamW per-batch."""
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=data.device); gen.manual_seed(seed)
    model.train()
    for step in range(args.steps_per_phase):
        x, y = make_batch(data, args.batch, args.seq_len, gen)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if (step + 1) % args.log_every == 0:
            safe_print(f"      [baseline] step {step+1}/{args.steps_per_phase}  loss={loss.item():.3f}")


def train_sleep(model, data, args, seed, use_ternary=True):
    """3_Sleep.md spec WAKE/NREM/REM cycle + 3분배 ternary."""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location('sleep_ft',
        str(Path(__file__).parent / 'sleep_finetune_lm.py'))
    mod = importlib.util.module_from_spec(spec); sys.modules['sleep_ft'] = mod
    spec.loader.exec_module(mod)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=data.device); gen.manual_seed(seed)
    if hasattr(model, 'set_lambda_schedule'):
        model.lambda_curv_base = 0.001
        model.lambda_curv = 0.001
        model.set_lambda_schedule(
            total_steps=args.steps_per_phase * args.cycle_len,
            warmup_steps=10 * args.cycle_len,
        )
    classifier = None
    if use_ternary:
        from clarus.sparsity import TernaryClassifier
        from clarus.constants import ACTIVE_RATIO, STRUCT_RATIO
        classifier = TernaryClassifier(model,
            active_ratio=ACTIVE_RATIO, struct_ratio=STRUCT_RATIO)
    model.train()
    for cyc in range(args.steps_per_phase):
        lam = model._current_lambda_curv() if hasattr(model, '_current_lambda_curv') else 0.0
        wl, _ = mod.wake_phase(model, optim, data, args.batch, args.seq_len, gen,
                               cycle_len=args.cycle_len, lam_curv=lam)
        if classifier is not None:
            classifier.update_freq()
            classifier.apply_grad_mask(allow_struct=True)
        mod.nrem_phase(model, optim, eta_nrem=args.eta_nrem, top_eps=args.top_eps)
        if classifier is not None and (cyc + 1) % 10 == 0:
            classifier.reclassify()
        if (cyc + 1) % args.log_every == 0:
            safe_print(f"      [sleep] cyc {cyc+1}/{args.steps_per_phase}  loss={wl:.3f}  lam={lam:.4f}")


def split_corpus(data, ratio=0.5):
    """Split corpus into A (first half) and B (second half) without overlap."""
    n = data.shape[0]
    cut = int(n * ratio)
    return data[:cut], data[cut:]


def load_model_fresh(ckpt_path, device):
    """Load fresh ClarusLM from checkpoint (deep copy of state)."""
    from clarus import load_clarus_lm_generator
    gen = load_clarus_lm_generator(ckpt_path, device=str(device))
    return gen.model, gen.tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="clarus/clarus_lm_kogpt2.pt")
    ap.add_argument("--data", default="examples/ai/train_data.txt",
                    help="Single corpus to split A|B (when --data-b is omitted).")
    ap.add_argument("--data-b", default=None,
                    help="If set, use --data as A and --data-b as B (강한 domain shift).")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--steps-per-phase", type=int, default=150,
                    help="Training steps (baseline) / cycles (sleep) per phase A and phase B.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--cycle-len", type=int, default=4)
    ap.add_argument("--eta-nrem", type=float, default=0.005)
    ap.add_argument("--top-eps", type=float, default=0.0487)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=30)
    ap.add_argument("--no-ternary", action="store_true",
                    help="Disable ternary classifier (sleep cycle only).")
    return ap.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_one_method(method_name, train_fn, ckpt_path, A_train, B_train,
                   A_val, B_val, args, val_gen):
    """One full continual learning trial: train A → eval, train B → eval."""
    safe_print(f"\n{'='*60}")
    safe_print(f"  Method: {method_name}")
    safe_print(f"{'='*60}")

    device = resolve_device(args.device)
    model, _ = load_model_fresh(ckpt_path, device)
    model.train()

    # Initial perplexity on both.
    ppl_A_init = val_perplexity(model, A_val, args.batch, args.seq_len, val_gen)
    ppl_B_init = val_perplexity(model, B_val, args.batch, args.seq_len, val_gen)
    safe_print(f"  init     ppl_A={ppl_A_init:>8.2f}  ppl_B={ppl_B_init:>8.2f}")

    # Phase 1: train on A.
    safe_print(f"\n  [Phase 1] Training on A ({args.steps_per_phase} steps)...")
    t0 = time.perf_counter()
    train_fn(model, A_train, args, seed=args.seed + 1)
    t1 = time.perf_counter() - t0
    ppl_A_after_A = val_perplexity(model, A_val, args.batch, args.seq_len, val_gen)
    ppl_B_after_A = val_perplexity(model, B_val, args.batch, args.seq_len, val_gen)
    safe_print(f"  phase 1: ppl_A={ppl_A_after_A:>8.2f}  ppl_B={ppl_B_after_A:>8.2f}  ({t1:.1f}s)")

    # Phase 2: train on B (continual).
    safe_print(f"\n  [Phase 2] Training on B ({args.steps_per_phase} steps)...")
    t0 = time.perf_counter()
    train_fn(model, B_train, args, seed=args.seed + 2)
    t2 = time.perf_counter() - t0
    ppl_A_after_B = val_perplexity(model, A_val, args.batch, args.seq_len, val_gen)
    ppl_B_after_B = val_perplexity(model, B_val, args.batch, args.seq_len, val_gen)
    safe_print(f"  phase 2: ppl_A={ppl_A_after_B:>8.2f}  ppl_B={ppl_B_after_B:>8.2f}  ({t2:.1f}s)")

    # Compute metrics.
    forgetting = ppl_A_after_B - ppl_A_after_A
    plasticity = ppl_B_init - ppl_B_after_B
    safe_print(f"\n  forgetting (lower=better): ppl_A drift {forgetting:+.2f}  "
               f"({ppl_A_after_A:.2f} -> {ppl_A_after_B:.2f})")
    safe_print(f"  plasticity (higher=better): ppl_B drop  {plasticity:+.2f}  "
               f"({ppl_B_init:.2f} -> {ppl_B_after_B:.2f})")

    return {
        "method": method_name,
        "ppl_A_init": ppl_A_init,
        "ppl_B_init": ppl_B_init,
        "ppl_A_after_A": ppl_A_after_A,
        "ppl_B_after_A": ppl_B_after_A,
        "ppl_A_after_B": ppl_A_after_B,
        "ppl_B_after_B": ppl_B_after_B,
        "forgetting": forgetting,
        "plasticity": plasticity,
        "time_s": t1 + t2,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    safe_print("=" * 60)
    safe_print(" Continual learning test (catastrophic forgetting)")
    safe_print(f"  ckpt={args.ckpt}  device={device}")
    safe_print(f"  steps_per_phase={args.steps_per_phase}  batch={args.batch}  seq_len={args.seq_len}")
    safe_print(f"  lr={args.lr}  cycle_len={args.cycle_len}  top_eps={args.top_eps}")
    safe_print("=" * 60)

    # Load corpus(es).
    from clarus import load_clarus_lm_generator
    gen_obj = load_clarus_lm_generator(args.ckpt, device=str(device))
    tok = gen_obj.tokenizer

    def encode_file(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return torch.tensor(tok.encode(text), dtype=torch.long, device=device)

    if args.data_b:
        # 강한 domain shift: 두 다른 파일 사용.
        A_full = encode_file(args.data)
        B_full = encode_file(args.data_b)
        safe_print(f"\n[CORPUS] A={args.data} ({A_full.shape[0]} tok)  "
                   f"B={args.data_b} ({B_full.shape[0]} tok)")
    else:
        full = encode_file(args.data)
        safe_print(f"\n[CORPUS] {full.shape[0]} tokens, split 50/50")
        A_full, B_full = split_corpus(full, ratio=0.5)

    A_train, A_val = A_full[: int(0.9 * A_full.shape[0])], A_full[int(0.9 * A_full.shape[0]) :]
    B_train, B_val = B_full[: int(0.9 * B_full.shape[0])], B_full[int(0.9 * B_full.shape[0]) :]
    safe_print(f"  A: train={A_train.shape[0]}  val={A_val.shape[0]}")
    safe_print(f"  B: train={B_train.shape[0]}  val={B_val.shape[0]}")
    # Free generator (we'll reload per-method).
    del gen_obj
    torch.cuda.empty_cache() if device.type == "cuda" else None

    val_gen = torch.Generator(device=device); val_gen.manual_seed(args.seed + 99)

    # Run two methods.
    r_baseline = run_one_method(
        "baseline (standard AdamW)", train_baseline,
        args.ckpt, A_train, B_train, A_val, B_val, args, val_gen,
    )

    val_gen2 = torch.Generator(device=device); val_gen2.manual_seed(args.seed + 99)
    r_sleep = run_one_method(
        "sleep (CE spec)" + ("" if args.no_ternary else " + ternary"),
        lambda m, d, a, seed: train_sleep(m, d, a, seed, use_ternary=not args.no_ternary),
        args.ckpt, A_train, B_train, A_val, B_val, args, val_gen2,
    )

    # Final comparison.
    safe_print("\n" + "=" * 60)
    safe_print(" Summary (catastrophic forgetting comparison)")
    safe_print("=" * 60)
    header = f"{'method':<32s}  {'ppl_A_after_A':>12s}  {'ppl_A_after_B':>12s}  {'forget':>8s}  {'plast':>8s}"
    safe_print(header)
    safe_print("-" * len(header))
    for r in (r_baseline, r_sleep):
        safe_print(
            f"{r['method']:<32s}  {r['ppl_A_after_A']:>12.2f}  {r['ppl_A_after_B']:>12.2f}  "
            f"{r['forgetting']:>+8.2f}  {r['plasticity']:>+8.2f}"
        )
    safe_print("\n  forgetting (lower=better) : 작을수록 A를 잘 보존")
    safe_print("  plasticity (higher=better): 클수록 B를 잘 학습")
    if r_sleep["forgetting"] < r_baseline["forgetting"]:
        delta = r_baseline["forgetting"] - r_sleep["forgetting"]
        safe_print(f"\n  RESULT: sleep cycle reduces forgetting by {delta:.2f} ppl "
                   f"({delta / max(r_baseline['forgetting'], 0.01) * 100:.1f}%)")
    else:
        delta = r_sleep["forgetting"] - r_baseline["forgetting"]
        safe_print(f"\n  RESULT: sleep cycle does NOT reduce forgetting "
                   f"(+{delta:.2f} ppl worse than baseline)")


if __name__ == "__main__":
    main()
