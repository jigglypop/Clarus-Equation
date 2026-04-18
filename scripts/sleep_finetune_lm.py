"""Sleep-cycle fine-tuner for ClarusLM (docs/7_AGI/3_Sleep.md spec).

학습 한 사이클 = WAKE (gradient 누적) -> NREM (LBO 평탄화 + top-eps^2 mask) -> REM (선택적 탐색).
사양:
  WAKE (3.2-3.3): cycle_len batch 반복하며 gradient 누적, optim.step() 보류.
                  P_sleep(t) = mean curvature 모니터링.
  NREM (4.2):    1) W_l <- W_l - eta_nrem * Delta_g W_l (가중치 라플라시안 평탄화)
                  2) top eps^2=4.87% gradient mask (크기 기준 상위만 통과)
                  3) AdamW step on masked gradient.
  REM  (5.2):    G_rem = (1 - mask) * G + sigma * eps;
                  loss(theta + G_rem) < loss(theta)일 때만 채택.

Run (논문 사양):
    .venv/Scripts/python.exe scripts/sleep_finetune_lm.py \
        --ckpt-in clarus/clarus_lm_kogpt2.pt \
        --ckpt-out clarus/clarus_lm_kogpt2_tuned.pt \
        --data examples/ai/train_data.txt \
        --device cuda --steps 600 --batch 4 --seq-len 128 \
        --cycle-len 4 --eta-nrem 0.01 --top-eps 0.0487

호환:
  --steps     : NREM optimizer step 수 (= 학습 사이클 수)
  --rem-every : REM 위상 주기 (사이클 단위). 0이면 REM 비활성.
  --rem-replay: REM 시도 횟수 (per REM phase).
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


PROBE_PROMPTS = [
    "인공지능의 미래는",
    "오늘 날씨가",
    "한국에서 가장 유명한 음식은",
    "내일은 친구와 함께",
    "클라루스 방정식은",
]


def safe_print(*a, **k):
    try:
        print(*a, **k, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(((" ".join(map(str, a))) + "\n").encode("utf-8", "replace"))


@contextmanager
def measure_peak(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        try:
            import psutil
            baseline = psutil.Process(os.getpid()).memory_info().rss
        except ImportError:
            baseline = 0
    state = {"peak_mb": 0.0}
    try:
        yield state
    finally:
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) - baseline
        else:
            try:
                import psutil
                peak = psutil.Process(os.getpid()).memory_info().rss - baseline
            except ImportError:
                peak = 0
        state["peak_mb"] = max(peak, 0) / 1024 / 1024


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-in", default="clarus/clarus_lm_kogpt2.pt")
    ap.add_argument("--ckpt-out", default="clarus/clarus_lm_kogpt2_tuned.pt")
    ap.add_argument("--data", default="examples/ai/train_data.txt",
                    help="Local text file (used only when --dataset is empty).")
    ap.add_argument("--dataset", default=None,
                    help="HuggingFace dataset id (e.g. lcw99/wikipedia-korean-20221001).")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--doc-limit", type=int, default=1000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=600,
                    help="Total optimizer steps (one per WAKE-NREM pair).")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-curv", type=float, default=0.01,
                    help="CE curvature regularizer weight (lambda_0).")
    ap.add_argument("--curv-warmup", type=int, default=50,
                    help="Curvature schedule warmup steps (2_Architecture.md 5.2).")
    ap.add_argument("--cycle-len", type=int, default=4,
                    help="WAKE batches accumulated per NREM step (3_Sleep.md 3.2).")
    ap.add_argument("--eta-nrem", type=float, default=0.01,
                    help="NREM weight LBO smoothing rate (3_Sleep.md 4.2 step 1).")
    ap.add_argument("--top-eps", type=float, default=0.0487,
                    help="NREM gradient TopK ratio = epsilon^2 (3_Sleep.md 4.2 step 2).")
    ap.add_argument("--rem-every", type=int, default=20,
                    help="REM phase every N cycles (0 = disabled).")
    ap.add_argument("--rem-replay", type=int, default=4,
                    help="REM exploration trials per REM phase (3_Sleep.md 5.2).")
    ap.add_argument("--rem-noise-sigma", type=float, default=0.001,
                    help="REM exploration noise std (3_Sleep.md 5.2).")
    ap.add_argument("--rem-accept-margin", type=float, default=0.01,
                    help="REM relative loss-improvement threshold for acceptance "
                         "(held-out batch). 1%% 미만 개선은 reject (drift 방지).")
    ap.add_argument("--ternary", action="store_true",
                    help="3분배 가중치 분류 활성화 (5_Sparsity.md 4절). "
                         "ACTIVE만 WAKE에서 학습, STRUCT는 NREM에서, BG 동결.")
    ap.add_argument("--reclassify-every", type=int, default=10,
                    help="3분배 재분류 주기 (사이클 단위).")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=20)
    return ap.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_corpus_ids(
    tokenizer,
    path: str,
    device: torch.device,
    *,
    dataset: str | None = None,
    dataset_split: str = "train",
    doc_limit: int = 1000,
) -> torch.Tensor:
    """Tokenize a corpus into a single 1-D tensor of token ids.

    When ``dataset`` is set, pulls the first ``doc_limit`` documents via
    :func:`clarus.sleep.load_corpus_documents` (HF datasets). Otherwise
    reads ``path`` as a local UTF-8 text file.
    """
    if dataset:
        from clarus.sleep import load_corpus_documents
        docs = load_corpus_documents(
            dataset_name=dataset,
            dataset_split=dataset_split,
            doc_limit=doc_limit,
        )
        text = "\n\n".join(docs)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    ids = tokenizer.encode(text)
    if not ids:
        raise RuntimeError("corpus tokenization produced 0 ids")
    return torch.tensor(ids, dtype=torch.long, device=device)


def make_batch(data: torch.Tensor, batch: int, seq_len: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n = data.shape[0]
    if n <= seq_len + 1:
        raise RuntimeError(f"corpus too small: {n} <= seq_len+1 ({seq_len + 1})")
    starts = torch.randint(0, n - seq_len - 1, (batch,), generator=gen, device=data.device)
    x = torch.stack([data[s : s + seq_len] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts.tolist()])
    return x, y


@dataclass
class CycleStat:
    wake_loss: float       # average loss across cycle_len WAKE batches
    p_sleep: float         # average curvature (sleep pressure proxy)
    grad_norm: float       # gradient norm BEFORE topk mask
    nrem_kept_frac: float  # fraction of gradient elements actually applied (~ top_eps)
    rem_accepted: int = 0  # number of accepted REM proposals
    rem_tried: int = 0     # number of attempted REM proposals


def _ce_loss(model, x, y, lambda_curv_eff: float) -> tuple[torch.Tensor, float]:
    """Cross-entropy + curvature reg (one batch). lambda_curv_eff is post-schedule."""
    logits, _ = model(x)
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    curv_val = sum(b.curvature for b in model.blocks) / max(len(model.blocks), 1)
    if lambda_curv_eff > 0:
        loss = ce + lambda_curv_eff * curv_val
    else:
        loss = ce
    return loss, float(curv_val)


def wake_phase(model, optim, train_data, batch, seq_len, gen,
               cycle_len: int, lam_curv: float) -> tuple[float, float]:
    """3_Sleep.md 3.2-3.3: forward + gradient 누적 (cycle_len batch). optim.step() 안 함.

    Returns (avg_loss, avg_curvature=P_sleep proxy).
    """
    optim.zero_grad(set_to_none=True)
    sum_loss = 0.0
    sum_curv = 0.0
    for _ in range(cycle_len):
        x, y = make_batch(train_data, batch, seq_len, gen)
        loss, curv = _ce_loss(model, x, y, lam_curv)
        # Average gradient across cycle_len.
        (loss / float(cycle_len)).backward()
        sum_loss += float(loss.item())
        sum_curv += curv
        # ClarusLM.forward(targets=None) path는 _lambda_step을 증가시키지 않으므로
        # 여기서 수동 증가 (스케줄이 사이클별로 진행되도록).
        if hasattr(model, '_lambda_total_steps') and model._lambda_total_steps > 0:
            model._lambda_step += 1
    return sum_loss / cycle_len, sum_curv / cycle_len


@torch.no_grad()
def _nrem_lbo_smooth(model, eta_nrem: float):
    """3_Sleep.md 4.2 step 1: W_l <- W_l - eta_nrem * Delta_g W_l.

    Δ_g W를 고주파 성분의 finite-difference 근사로 사용:
    smoothed = 0.5*W + 0.25*(roll(W,+1) + roll(W,-1)) (행 방향 평균),
    Δ_g W ≈ W - smoothed.
    spectral_norm wrap된 가중치(weight_orig)는 건너뜀 (정규화 자체가 평탄화 역할).
    Embedding/LayerNorm-style 1D 파라미터는 건너뜀.
    """
    if eta_nrem <= 0.0:
        return
    for module in model.modules():
        # spectral_norm 적용된 Linear는 weight_orig를 통해서만 직접 수정 가능.
        # 평탄화 효과가 SN과 충돌할 수 있어 건너뛴다.
        if hasattr(module, 'weight_orig'):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        W = module.weight
        if W.ndim != 2 or W.shape[0] < 3:
            continue
        rolled_pos = torch.roll(W, shifts=1, dims=0)
        rolled_neg = torch.roll(W, shifts=-1, dims=0)
        smoothed = 0.5 * W + 0.25 * (rolled_pos + rolled_neg)
        # 경계 행은 평탄화에서 제외 (인공적 wrap-around 회피).
        smoothed[0] = W[0]
        smoothed[-1] = W[-1]
        delta_g = W - smoothed
        W.add_(delta_g, alpha=-eta_nrem)


def _grad_global_threshold(parameters, top_eps: float) -> tuple[float, int, int]:
    """Compute the magnitude threshold s.t. only top top_eps fraction passes.

    Returns (threshold, kept_count, total_count). 모든 .grad의 |g|를 합쳐
    하나의 글로벌 임계를 잡는다 (per-tensor topk보다 spec에 부합).
    """
    abs_chunks = []
    total = 0
    for p in parameters:
        if p.grad is None:
            continue
        abs_chunks.append(p.grad.detach().abs().view(-1))
        total += p.grad.numel()
    if total == 0:
        return 0.0, 0, 0
    # Concatenation을 피하기 위해 sample 기반 quantile 사용 (메모리 절감).
    # 큰 모델(수천만 파라미터)에서는 cat 비용이 크므로 stride sampling.
    if total > 1_000_000:
        sample_size = 1_000_000
        sample_chunks = []
        target_per_tensor = max(1, sample_size // max(len(abs_chunks), 1))
        for chunk in abs_chunks:
            if chunk.numel() <= target_per_tensor:
                sample_chunks.append(chunk)
            else:
                stride = chunk.numel() // target_per_tensor
                sample_chunks.append(chunk[::stride])
        flat_sample = torch.cat(sample_chunks)
        # quantile은 1 - top_eps 백분위 (top_eps만 살림).
        threshold = float(torch.quantile(flat_sample, 1.0 - top_eps).item())
    else:
        flat = torch.cat(abs_chunks)
        threshold = float(torch.quantile(flat, 1.0 - top_eps).item())
    kept = 0
    for chunk in abs_chunks:
        kept += int((chunk >= threshold).sum().item())
    return threshold, kept, total


@torch.no_grad()
def _apply_grad_topk_mask(parameters, threshold: float):
    """3_Sleep.md 4.2 step 2: grad <- grad * (|grad| >= threshold)."""
    for p in parameters:
        if p.grad is None:
            continue
        mask = p.grad.abs() >= threshold
        p.grad.mul_(mask.to(p.grad.dtype))


def nrem_phase(model, optim, eta_nrem: float, top_eps: float) -> tuple[float, float]:
    """3_Sleep.md 4.2: NREM 평탄화 + top eps^2 mask + step.

    Returns (kept_fraction, raw_grad_norm).
    """
    # Step 1: weight LBO smoothing.
    _nrem_lbo_smooth(model, eta_nrem)
    # Raw grad norm before mask (clip and metric).
    raw_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    # Step 2: top eps^2 mask.
    threshold, kept, total = _grad_global_threshold(model.parameters(), top_eps)
    _apply_grad_topk_mask(model.parameters(), threshold)
    # Step 3: optimizer step (uses masked .grad).
    optim.step()
    kept_frac = float(kept) / float(total) if total > 0 else 0.0
    return kept_frac, raw_norm


def rem_phase(model, optim, train_data, batch, seq_len, gen,
              n_trials: int, lam_curv: float, noise_sigma: float,
              top_eps: float, accept_margin: float = 0.01) -> tuple[int, int]:
    """3_Sleep.md 5.2: pruned (1 - eps^2) 영역 + 노이즈 탐색.

    각 trial:
      1. trial용 batch로 grad 계산 (gradient batch)
      2. G_pruned = (1 - top_eps_mask) * G  (NREM에서 버려진 영역)
      3. proposal = G_pruned + noise_sigma * randn
      4. 임시로 -lr * proposal 적용
      5. **다른** held-out batch로 loss_pre / loss_post 측정 (overfitting 방지)
      6. (loss_pre - loss_post) > accept_margin*|loss_pre| 이면 채택, 아니면 원복

    accept_margin: relative improvement 요구치 (기본 1%). 같은 batch로 평가하면
    노이즈 자체가 거의 항상 작은 손실 감소를 만들어내는 trivial acceptance가 발생,
    실제 일반화 개선이 아니라 점진적 weight drift만 일으킨다. Held-out batch +
    상대 임계로 이를 방지한다.

    Returns (accepted, tried).
    """
    if n_trials <= 0:
        return 0, 0
    accepted = 0
    lr = float(optim.param_groups[0]['lr'])
    params = [p for p in model.parameters() if p.requires_grad]
    for _ in range(n_trials):
        # Gradient용 batch.
        x_g, y_g = make_batch(train_data, batch, seq_len, gen)
        # Held-out 평가용 batch (서로 다름 보장).
        x_e, y_e = make_batch(train_data, batch, seq_len, gen)
        # Baseline loss on held-out.
        with torch.no_grad():
            logits_pre, _ = model(x_e)
            loss_pre = float(F.cross_entropy(
                logits_pre.view(-1, logits_pre.size(-1)), y_e.view(-1)
            ).item())
        # Compute fresh gradient on grad batch.
        optim.zero_grad(set_to_none=True)
        loss_t, _ = _ce_loss(model, x_g, y_g, lam_curv)
        loss_t.backward()
        threshold, _, _ = _grad_global_threshold(params, top_eps)
        snapshot = []
        for p in params:
            if p.grad is None:
                snapshot.append(None)
                continue
            mask_pruned = (p.grad.abs() < threshold).to(p.grad.dtype)
            g_pruned = p.grad * mask_pruned
            noise = torch.randn_like(p) * noise_sigma
            proposal = g_pruned + noise
            old = p.data.detach().clone()
            p.data.add_(proposal, alpha=-lr)
            snapshot.append((old, proposal))
        with torch.no_grad():
            logits_post, _ = model(x_e)
            loss_post = float(F.cross_entropy(
                logits_post.view(-1, logits_post.size(-1)), y_e.view(-1)
            ).item())
        improvement = loss_pre - loss_post
        threshold_improve = accept_margin * abs(loss_pre)
        if improvement > threshold_improve:
            accepted += 1
        else:
            for p, snap in zip(params, snapshot):
                if snap is None:
                    continue
                old, _ = snap
                p.data.copy_(old)
    optim.zero_grad(set_to_none=True)
    return accepted, n_trials


@torch.no_grad()
def val_perplexity(model, data, batch: int, seq_len: int, gen: torch.Generator, n: int = 8) -> float:
    losses = []
    was_training = model.training
    model.eval()
    for _ in range(n):
        x, y = make_batch(data, batch, seq_len, gen)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(float(ce.item()))
    if was_training:
        model.train()
    mean_loss = sum(losses) / len(losses)
    return math.exp(min(mean_loss, 20.0))


def sample_generations(gen, prompts: list[str], max_tokens: int = 25, seed: int = 42):
    out = []
    for p in prompts:
        text = gen.generate(p, max_tokens=max_tokens, temperature=0.7, top_k=40, seed=seed)
        out.append((p, text))
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    safe_print("=" * 70)
    safe_print(f"  ClarusLM sleep-cycle fine-tune on device={device}")
    safe_print(f"  ckpt-in={args.ckpt_in}  data={args.data}")
    safe_print(f"  steps={args.steps}  batch={args.batch}  seq_len={args.seq_len}  lr={args.lr}")
    safe_print(f"  lambda_curv={args.lambda_curv}  rem_every={args.rem_every}")
    safe_print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    safe_print("\n[LOAD]")
    with measure_peak(device) as load_mem:
        from clarus import load_clarus_lm_generator
        gen_obj = load_clarus_lm_generator(args.ckpt_in, device=str(device))
        model = gen_obj.model
    n_params = sum(p.numel() for p in model.parameters())
    safe_print(f"  load peak: {load_mem['peak_mb']:.1f} MB")
    safe_print(f"  params   : {n_params / 1e6:.2f} M")
    safe_print(f"  block 0 ffn.phi      : {type(model.blocks[0].ffn.phi).__name__}")
    safe_print(f"  block 0 attn.proj SN : {hasattr(model.blocks[0].attn.proj, 'weight_orig')}")

    # ------------------------------------------------------------------
    # 2. Tokenize corpus
    # ------------------------------------------------------------------
    safe_print("\n[CORPUS]")
    src = args.dataset if args.dataset else args.data
    safe_print(f"  source: {src}")
    data = load_corpus_ids(
        gen_obj.tokenizer,
        args.data,
        device,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        doc_limit=args.doc_limit,
    )
    n_train = int(data.shape[0] * 0.95)
    train_data = data[:n_train]
    val_data = data[n_train:]
    safe_print(f"  total tokens : {data.shape[0]}")
    safe_print(f"  train / val  : {train_data.shape[0]} / {val_data.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Pre-train probes
    # ------------------------------------------------------------------
    safe_print("\n[BEFORE FINE-TUNE]")
    val_gen = torch.Generator(device=device); val_gen.manual_seed(args.seed + 1)
    ppl_before = val_perplexity(model, val_data, args.batch, args.seq_len, val_gen)
    safe_print(f"  val_ppl: {ppl_before:.2f}")
    samples_before = sample_generations(gen_obj, PROBE_PROMPTS)
    for p, t in samples_before:
        safe_print(f"  [{p}] -> {t!r}")

    # ------------------------------------------------------------------
    # 4. Sleep-cycle training (3_Sleep.md spec)
    # ------------------------------------------------------------------
    safe_print("\n[TRAIN] (WAKE-NREM-REM cycles)")
    safe_print(f"  cycle_len={args.cycle_len}  eta_nrem={args.eta_nrem}  "
               f"top_eps={args.top_eps}  rem_every={args.rem_every}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_gen = torch.Generator(device=device); train_gen.manual_seed(args.seed + 2)
    cycle_log: list[CycleStat] = []

    # 3분배 가중치 분류기 (5_Sparsity.md 4절). 옵션: --ternary.
    classifier = None
    if args.ternary:
        from clarus.sparsity import TernaryClassifier
        from clarus.constants import ACTIVE_RATIO, STRUCT_RATIO
        classifier = TernaryClassifier(
            model, active_ratio=ACTIVE_RATIO, struct_ratio=STRUCT_RATIO,
        )
        s = classifier.stats()
        safe_print(
            f"  [TERNARY] tracked={s['tracked_params']} weights={s['total_weights']:,}  "
            f"active={s['active_pct']:.2f}%  struct={s['struct_pct']:.2f}%  bg={s['bg_pct']:.2f}%  "
            f"extra_mem={classifier.memory_bytes()/1024/1024:.1f}MB"
        )

    # CLI lambda_curv가 checkpoint config의 값을 override (논문 사양 적용 강제).
    # checkpoint 저장 시 lambda_curv=0.0이 들어가 있어도 CLI 값이 우선.
    if hasattr(model, 'lambda_curv_base'):
        model.lambda_curv_base = float(args.lambda_curv)
        model.lambda_curv = float(args.lambda_curv)
    # Curvature schedule (2_Architecture.md 5.2): warmup + cosine decay.
    # 주의: ClarusLM.forward()가 호출될 때마다 _lambda_step이 +1되므로,
    # 실제 step 단위는 batch (cycle_len * cycles) 임. total_steps도 그에 맞춤.
    if hasattr(model, 'set_lambda_schedule'):
        total_forward_steps = args.steps * args.cycle_len
        warmup_forward_steps = args.curv_warmup * args.cycle_len
        model.set_lambda_schedule(
            total_steps=total_forward_steps,
            warmup_steps=warmup_forward_steps,
        )

    model.train()
    t_start = time.perf_counter()
    with measure_peak(device) as train_mem:
        for cycle_idx in range(1, args.steps + 1):
            # 효과적인 lambda_curv (스케줄 적용 후) 조회.
            if hasattr(model, '_current_lambda_curv'):
                lam_eff = model._current_lambda_curv()
            else:
                lam_eff = float(args.lambda_curv)

            # ---- WAKE: cycle_len 배치 동안 gradient 누적 ----
            wake_loss, p_sleep = wake_phase(
                model, optim, train_data,
                args.batch, args.seq_len, train_gen,
                args.cycle_len, lam_eff,
            )

            # 3분배 ternary: WAKE에서 freq 업데이트, NREM step에서 ACTIVE+STRUCT만 통과.
            if classifier is not None:
                classifier.update_freq()
                classifier.apply_grad_mask(allow_struct=True)  # NREM step이므로 struct 허용

            # ---- NREM: weight LBO smoothing + top-eps mask + step ----
            kept_frac, raw_grad_norm = nrem_phase(
                model, optim, args.eta_nrem, args.top_eps,
            )

            # NREM 후 분류 재계산 (사양 5_Sparsity.md 4.4).
            if classifier is not None and cycle_idx % args.reclassify_every == 0:
                classifier.reclassify()

            # ---- REM (주기적): 비선택 영역 + 노이즈 탐색 ----
            rem_acc, rem_tried = 0, 0
            if args.rem_every > 0 and cycle_idx % args.rem_every == 0:
                rem_acc, rem_tried = rem_phase(
                    model, optim, train_data,
                    args.batch, args.seq_len, train_gen,
                    args.rem_replay, lam_eff,
                    args.rem_noise_sigma, args.top_eps,
                    accept_margin=args.rem_accept_margin,
                )

            stat = CycleStat(
                wake_loss=wake_loss,
                p_sleep=p_sleep,
                grad_norm=raw_grad_norm,
                nrem_kept_frac=kept_frac,
                rem_accepted=rem_acc,
                rem_tried=rem_tried,
            )
            cycle_log.append(stat)

            if cycle_idx % args.log_every == 0 or cycle_idx == 1:
                rem_str = (f"  rem={rem_acc}/{rem_tried}" if rem_tried > 0 else "")
                safe_print(
                    f"  cyc {cycle_idx:4d}  loss={stat.wake_loss:.3f}  "
                    f"P_sleep={stat.p_sleep:.4f}  |g|={stat.grad_norm:.2f}  "
                    f"kept={stat.nrem_kept_frac*100:.2f}%  lam={lam_eff:.4f}{rem_str}"
                )

            if cycle_idx % args.val_every == 0:
                vppl = val_perplexity(model, val_data, args.batch, args.seq_len, val_gen)
                safe_print(f"  cyc {cycle_idx:4d}  [VAL] ppl={vppl:.2f}")

    t_total = time.perf_counter() - t_start
    safe_print(f"\n  total train time : {t_total:.1f}s  ({args.steps / t_total:.1f} cycles/s)")
    safe_print(f"  train peak mem   : {train_mem['peak_mb']:.1f} MB")
    # Aggregate cycle stats.
    if cycle_log:
        avg_kept = sum(s.nrem_kept_frac for s in cycle_log) / len(cycle_log)
        avg_psleep = sum(s.p_sleep for s in cycle_log) / len(cycle_log)
        rem_total_acc = sum(s.rem_accepted for s in cycle_log)
        rem_total_try = sum(s.rem_tried for s in cycle_log)
        safe_print(f"  avg NREM kept fraction: {avg_kept*100:.2f}% (target eps^2={args.top_eps*100:.2f}%)")
        safe_print(f"  avg P_sleep (curv) : {avg_psleep:.4f}")
        safe_print(f"  REM acceptance     : {rem_total_acc}/{rem_total_try}")

    # ------------------------------------------------------------------
    # 5. Post-train probes
    # ------------------------------------------------------------------
    safe_print("\n[AFTER FINE-TUNE]")
    model.eval()
    val_gen2 = torch.Generator(device=device); val_gen2.manual_seed(args.seed + 1)
    ppl_after = val_perplexity(model, val_data, args.batch, args.seq_len, val_gen2)
    safe_print(f"  val_ppl: {ppl_before:.2f}  ->  {ppl_after:.2f}  (delta {ppl_after - ppl_before:+.2f})")

    samples_after = sample_generations(gen_obj, PROBE_PROMPTS)
    samples_path = os.path.splitext(args.ckpt_out)[0] + "_samples.txt"
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(f"=== ClarusLM sleep fine-tune samples ===\n")
        f.write(f"steps={args.steps}  lr={args.lr}  ppl {ppl_before:.2f} -> {ppl_after:.2f}\n\n")
        for (p, t0), (_, t1) in zip(samples_before, samples_after, strict=True):
            f.write(f"[{p}]\n")
            f.write(f"  before -> {t0}\n")
            f.write(f"  after  -> {t1}\n\n")
    safe_print(f"\n[SAMPLES] saved to {samples_path}")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    ckpt = torch.load(args.ckpt_in, map_location="cpu", weights_only=False)
    ckpt["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt.setdefault("config", {})
    ckpt["config"]["fine_tune"] = {
        "steps": args.steps,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "lambda_curv": args.lambda_curv,
        "curv_warmup": args.curv_warmup,
        "cycle_len": args.cycle_len,
        "eta_nrem": args.eta_nrem,
        "top_eps": args.top_eps,
        "rem_every": args.rem_every,
        "rem_replay": args.rem_replay,
        "rem_noise_sigma": args.rem_noise_sigma,
        "val_ppl_before": ppl_before,
        "val_ppl_after": ppl_after,
    }
    torch.save(ckpt, args.ckpt_out)
    sz_mb = os.path.getsize(args.ckpt_out) / 1024 / 1024
    safe_print(f"\n[SAVE] {args.ckpt_out}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
