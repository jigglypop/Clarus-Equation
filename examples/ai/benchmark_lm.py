"""CE LLM Benchmark: ClarusLM vs Standard Transformer.

Compares val loss, FLOPs/token, active param ratio, curvature effect,
and C3 self-consistency under identical conditions.

Usage:
    python benchmark_lm.py
    python benchmark_lm.py --data input.txt --steps 3000 --dim 512
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from clarus_lm import ClarusLM

try:
    from clarus.device import auto_device, device_summary
    from clarus.engine import backend_info
    _HAS_CLARUS = True
except ImportError:
    _HAS_CLARUS = False
    def auto_device(pref="auto"):
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(pref)
    def device_summary(d):
        if d.type == "cuda":
            return f"CUDA:{d.index or 0} {torch.cuda.get_device_name(d.index or 0)}"
        return "CPU"
    def backend_info():
        return "standalone"


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids):
        return "".join(self.itos.get(i, "?") for i in ids)


class BaselineBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x, mask=None):
        T = x.size(1)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineLM(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=6, n_heads=8,
                 max_seq_len=512, ffn_mult=4):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [BaselineBlock(dim, n_heads, ffn_mult) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_active_params(model, x, y):
    """Estimate active parameters by checking non-zero gradient fraction."""
    model.train()
    _, loss = model(x, y)
    loss.backward()
    total = 0
    active = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.numel()
            active += (p.grad.abs() > 1e-10).sum().item()
    model.zero_grad(set_to_none=True)
    return active, total


def estimate_flops_per_token(dim, n_layers, n_heads, seq_len, vocab_size):
    """Rough estimate of FLOPs per token (forward pass)."""
    head_dim = dim // n_heads
    attn = 2 * seq_len * dim + 2 * seq_len * dim
    ffn = 2 * dim * (4 * dim)
    per_layer = attn + ffn
    embed = 2 * dim
    total = n_layers * per_layer + embed + dim * vocab_size
    return total


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generator(seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def get_batch(data, seq_len, batch_size, device, generator=None):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, seq_len, batch_size, device, generator_seed, n_eval=10):
    model.eval()
    losses = []
    generator = build_generator(generator_seed)
    for _ in range(n_eval):
        x, y = get_batch(data, seq_len, batch_size, device, generator=generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_model(model, train_data, val_data, args, label):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = min(500, args.steps // 10)
    curv_warmup = int(args.steps * 0.2)
    target_curv = getattr(model, 'lambda_curv', 0.0)

    use_amp = args.device.startswith("cuda") and getattr(args, "amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    train_gen = build_generator(args.seed + 1)
    best_val = float("inf")
    log = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if target_curv > 0 and curv_warmup > 0:
            warmup_f = min(1.0, step / curv_warmup)
            decay_f = 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
            model.lambda_curv = target_curv * warmup_f * decay_f

        model.train()
        x, y = get_batch(
            train_data, args.seq_len, args.batch_size, args.device, generator=train_gen
        )
        with torch.amp.autocast(args.device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.eval_every == 0 or step == 1:
            val_loss = estimate_loss(
                model,
                val_data,
                args.seq_len,
                min(args.batch_size, 8),
                args.device,
                args.seed + 2,
            )
            elapsed = time.time() - t0
            best_val = min(best_val, val_loss)
            entry = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "elapsed": round(elapsed, 1),
            }
            log.append(entry)
            print(
                f"  [{label}] step {step:5d} | train {loss.item():.4f} | "
                f"val {val_loss:.4f} | best {best_val:.4f} | {elapsed:.0f}s"
            )

    return best_val, log, time.time() - t0


def download_shakespeare(path):
    if os.path.exists(path):
        return
    print(f"Downloading Shakespeare to {path}...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, path)


def main():
    p = argparse.ArgumentParser(description="CE LLM Benchmark")
    p.add_argument("--data", type=str, default=None, help="Text file path (auto-downloads Shakespeare if omitted)")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--lambda_curv", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto",
                   help="Device: 'auto', 'cuda', 'cuda:0', 'cpu'")
    p.add_argument("--output", type=str, default="brain_benchmark_results.json")
    p.add_argument("--sparsity", type=float, default=1.0,
                   help="GaugeLattice activation sparsity (0.0487=CE bootstrap, 1.0=dense)")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile (PyTorch 2.0+)")
    p.add_argument("--checkpoint", action="store_true",
                   help="Use gradient checkpointing (save memory)")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed precision (AMP) on CUDA")
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip baseline training, load from previous results")
    args = p.parse_args()

    if args.data is None:
        args.data = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
        download_shakespeare(args.data)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = max(1000, int(len(data) * 0.05))
    train_data, val_data = data[:-n_val], data[-n_val:]

    args.device = str(auto_device(args.device))
    seed_everything(args.seed)

    print("=" * 60)
    print("  CE Brain LLM Benchmark")
    print("=" * 60)
    print(f"  backend: {backend_info()}")
    print(f"  device:  {device_summary(torch.device(args.device))}")
    print(f"  data: {len(data)} chars, vocab={tok.vocab_size}")
    print(f"  dim={args.dim}  layers={args.n_layers}  heads={args.n_heads}")
    print(f"  steps={args.steps}  batch={args.batch_size}  seq_len={args.seq_len}")
    print(f"  sparsity={args.sparsity}")
    print(f"  seed={args.seed}")
    print()

    # --- Build models (auto param matching) ---
    baseline = BaselineLM(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    ).to(args.device)
    n_baseline = count_params(baseline)

    from clarus_lm import ALPHA_S, ALPHA_W, ALPHA_EM
    ta = ALPHA_S + ALPHA_W + ALPHA_EM
    diag_eff = sum((a / ta) ** 2 for a in [ALPHA_S, ALPHA_W, ALPHA_EM])
    est_hidden = int(round(args.dim * 4 / diag_eff))
    best_gap = float("inf")
    best_hidden = est_hidden
    lo = max(args.dim * 4, est_hidden - args.dim)
    hi = est_hidden + args.dim
    for h in range(lo, hi + 1, max(1, (hi - lo) // 64)):
        trial = ClarusLM(
            vocab_size=tok.vocab_size,
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=args.seq_len,
            ffn_hidden_dim=h,
            lambda_curv=args.lambda_curv,
            sparsity=args.sparsity,
        )
        gap = abs(count_params(trial) - n_baseline)
        if gap < best_gap:
            best_gap = gap
            best_hidden = h
        del trial
    print(f"  Auto FFN hidden dim: {best_hidden} (diag_eff={diag_eff:.3f}, gap={best_gap})")

    clarus = ClarusLM(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        ffn_hidden_dim=best_hidden,
        lambda_curv=args.lambda_curv,
        sparsity=args.sparsity,
        use_checkpoint=args.checkpoint,
    ).to(args.device)

    if args.compile and hasattr(torch, "compile"):
        print("  torch.compile enabled")
        clarus = torch.compile(clarus)
        baseline = torch.compile(baseline)

    args.amp = not args.no_amp

    n_clarus = count_params(clarus)
    print(f"  ClarusLM:  {n_clarus / 1e6:.2f}M params (ffn_hidden={best_hidden})")
    print(f"  Baseline:  {n_baseline / 1e6:.2f}M params (ffn_hidden={args.dim * 4})")
    print(f"  Ratio:     {n_clarus / n_baseline:.3f}")
    print()

    # --- Active param estimate ---
    seed_everything(args.seed)
    x_probe, y_probe = get_batch(train_data, args.seq_len, 4, args.device)
    active_c, total_c = count_active_params(clarus, x_probe, y_probe)
    active_ratio_c = active_c / max(1, total_c)

    prev_results_path = os.path.join(os.path.dirname(__file__), args.output)
    if args.skip_baseline and os.path.exists(prev_results_path):
        with open(prev_results_path) as f:
            prev = json.load(f)
        baseline_best = prev["best_val_loss"]["baseline"]
        baseline_log = prev.get("log_baseline", [])
        baseline_time = prev["train_time_s"]["baseline"]
        active_ratio_b = prev["active_param_ratio"]["baseline"]
        flops_per_tok = prev.get("flops_per_token_M", 0.0) * 1e6
        print(f"  [skip] Baseline from previous run: val={baseline_best:.4f}")
        print(f"  Active param ratio  ClarusLM: {active_ratio_c:.4f}")
        print()
    else:
        active_b, total_b = count_active_params(baseline, x_probe, y_probe)
        active_ratio_b = active_b / max(1, total_b)
        print(f"  Active param ratio  ClarusLM: {active_ratio_c:.4f}  Baseline: {active_ratio_b:.4f}")
        print()

        flops_per_tok = estimate_flops_per_token(
            args.dim, args.n_layers, args.n_heads, args.seq_len, tok.vocab_size
        )
        print(f"  FLOPs/token estimate: {flops_per_tok / 1e6:.2f}M")
        print()

        print("[1/2] Training Baseline...")
        seed_everything(args.seed)
        baseline_best, baseline_log, baseline_time = train_model(
            baseline, train_data, val_data, args, "Baseline"
        )
        print()

    del baseline

    # --- Train ClarusLM ---
    print("Training ClarusLM...")
    seed_everything(args.seed)
    clarus_best, clarus_log, clarus_time = train_model(
        clarus, train_data, val_data, args, "ClarusLM"
    )
    print()

    # --- Curvature stats ---
    clarus.eval()
    curv_vals = [b.curvature for b in clarus.blocks]
    avg_curv = sum(curv_vals) / len(curv_vals) if curv_vals else 0.0

    # --- Results ---
    results = {
        "config": {
            "dim": args.dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "lambda_curv": args.lambda_curv,
            "vocab_size": tok.vocab_size,
            "data_chars": len(data),
            "device": args.device,
            "seed": args.seed,
            "sparsity": args.sparsity,
        },
        "params": {
            "clarus_M": round(n_clarus / 1e6, 2),
            "baseline_M": round(n_baseline / 1e6, 2),
            "ratio": round(n_clarus / n_baseline, 4),
        },
        "active_param_ratio": {
            "clarus": round(active_ratio_c, 4),
            "baseline": round(active_ratio_b, 4),
        },
        "flops_per_token_M": round(flops_per_tok / 1e6, 2),
        "best_val_loss": {
            "clarus": round(clarus_best, 4),
            "baseline": round(baseline_best, 4),
            "delta": round(baseline_best - clarus_best, 4),
        },
        "train_time_s": {
            "clarus": round(clarus_time, 1),
            "baseline": round(baseline_time, 1),
        },
        "curvature": {
            "avg": float(f"{avg_curv:.6f}"),
            "per_layer": [float(f"{c:.6f}") for c in curv_vals],
        },
        "log_baseline": baseline_log,
        "log_clarus": clarus_log,
    }

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Best val loss -- Baseline: {baseline_best:.4f}  ClarusLM: {clarus_best:.4f}")
    delta = baseline_best - clarus_best
    if delta > 0:
        print(f"  ClarusLM wins by {delta:.4f}")
    elif delta < 0:
        print(f"  Baseline wins by {-delta:.4f}")
    else:
        print(f"  Tie")
    print(f"  Active ratio -- Baseline: {active_ratio_b:.4f}  ClarusLM: {active_ratio_c:.4f}")
    print(f"  Train time   -- Baseline: {baseline_time:.1f}s  ClarusLM: {clarus_time:.1f}s")
    print(f"  Avg curvature (ClarusLM): {avg_curv:.6f}")
    print()

    out_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
