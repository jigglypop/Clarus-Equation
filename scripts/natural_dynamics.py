"""Natural emergence test: does activation sparsity self-organize to epsilon^2?

논문 재해석:
  - p* = (4.87% / 26.2% / 68.9%)는 자기조직화 시스템의 평형 결과지,
    학습 알고리즘이 강제로 만들어야 하는 구속이 아니다.
  - 따라서 올바른 검증은 "강제 없이 자연 동역학을 제공했을 때 활성 비율이
    epsilon^2 = 4.87% 근방으로 수렴하는가" 이다.

세 가지 학습 변형 비교:
  A. plain         : 평범한 AdamW. CE 사양 메커니즘 일체 없음. (대조군)
  B. heat_flow     : NREM 위상에서 표상에 LBO 흐름 적용 (자연 수축).
                     gradient는 그대로, 강제 mask 없음.
  C. forced_eps    : NREM에서 top-eps^2 gradient mask + ternary BG freeze
                     (이전 잘못된 구현 = 결과 강제).

각 변형에 대해 매 사이클마다 측정:
  - val_ppl                : task 성능
  - active_fraction        : FFN 중간 활성의 |a| > threshold 비율
  - distance_to_p_star     : (active_frac - 0.0487)^2 의 척도

검증:
  - heat_flow가 active_fraction을 4.87% 근방으로 자연스럽게 끌어당기면 -> spec 확인
  - 그렇지 않으면 -> spec falsified (자연 emergence 가설 실패)

Usage:
    .venv/Scripts/python.exe scripts/natural_dynamics.py \
        --ckpt clarus/clarus_lm_kogpt2.pt --device cuda --steps 100
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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
def val_perplexity(model, data, batch, seq_len, gen, n_batches=8):
    losses = []
    was_t = model.training; model.eval()
    for _ in range(n_batches):
        x, y = make_batch(data, batch, seq_len, gen)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(float(ce.item()))
    if was_t: model.train()
    return math.exp(min(sum(losses) / len(losses), 20.0))


# ---------------------------------------------------------------------------
# 활성 비율 측정 후크 (FFN 중간 활성의 sparsity 추적)
# ---------------------------------------------------------------------------
class ActivationProbe:
    """ClarusBlock의 ffn 출력 직전 활성을 후크로 잡아 |a| > threshold 비율 측정.

    threshold는 layer-wise std 기반 적응적: thr = thr_factor * std(|a|).
    self.frac_per_layer: 마지막 forward batch의 layer별 평균 active fraction.
    """
    def __init__(self, model, thr_factor: float = 0.1):
        self.thr_factor = float(thr_factor)
        self.handles = []
        self._collected = []
        self._layer_count = 0
        for name, m in model.named_modules():
            # FFN의 act 모듈 (TopKSiLU 또는 SiLU/GELU) 출력에 후크.
            # GaugeLattice는 su3_act, su2_act, u1_act가 있고, dense면 act 하나.
            if name.endswith('su3_act') or name.endswith('su2_act') or name.endswith('u1_act') \
               or name.endswith('.act'):
                h = m.register_forward_hook(self._hook)
                self.handles.append(h)
                self._layer_count += 1

    def _hook(self, module, inputs, output):
        # output: post-activation tensor.
        a = output.detach()
        std = a.abs().std().clamp_min(1e-8)
        thr = self.thr_factor * std
        active = (a.abs() > thr).float().mean().item()
        self._collected.append(active)

    def reset(self):
        self._collected.clear()

    def avg_active(self) -> float:
        if not self._collected:
            return 0.0
        return sum(self._collected) / len(self._collected)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# 학습 변형 A: plain AdamW (no spec mechanism)
# ---------------------------------------------------------------------------
def train_plain(model, data, args, seed):
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=data.device); gen.manual_seed(seed)
    model.train()
    return optim, gen


def step_plain(model, optim, x, y):
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# 학습 변형 B: heat-kernel flow on representations during periodic NREM
# ---------------------------------------------------------------------------
@torch.no_grad()
def nrem_heat_flow_pass(model, data, batch, seq_len, gen, n_passes: int = 4,
                        eta_boost: float = 5.0):
    """NREM 모사: LBONorm의 h를 일시적으로 boost해서 표상 평탄화 강화 후 forward.

    LBONorm은 forward에서 lerp(x, x@V^T@V, h)를 적용. h가 클수록 평탄화 강함.
    여기서 h를 임시로 곱한 뒤 forward 몇 번 돌려서 표상이 평탄화 영향을 받게 함.
    weights는 안 건드림 (동역학은 표상 공간에).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / 'examples' / 'ai'))
    from clarus_lm import LBONorm
    was_train = model.training
    model.eval()
    saved_h = []
    lbo_modules = [m for m in model.modules() if isinstance(m, LBONorm)]
    for m in lbo_modules:
        saved_h.append(m.h.data.clone())
        # h_max로 clamp되므로 곱하면 그 한계까지 감
        m.h.data = (m.h.data.abs() * eta_boost).sign() * (m.h.data.abs() * eta_boost).clamp(
            max=float(m._h_max.item()) if hasattr(m, '_h_max') else 1.0
        )
    for _ in range(n_passes):
        x, y = make_batch(data, batch, seq_len, gen)
        _ = model(x)  # forward만 (gradient 안 건드림)
    for m, sh in zip(lbo_modules, saved_h):
        m.h.data = sh
    if was_train:
        model.train()


# ---------------------------------------------------------------------------
# 학습 변형 C: forced epsilon^2 (이전 구현, 비교 대조군)
# ---------------------------------------------------------------------------
def setup_forced_eps(model):
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location('sleep_ft',
        str(Path(__file__).parent / 'sleep_finetune_lm.py'))
    mod = importlib.util.module_from_spec(spec); sys.modules['sleep_ft'] = mod
    spec.loader.exec_module(mod)
    from clarus.sparsity import TernaryClassifier
    classifier = TernaryClassifier(model)
    return mod, classifier


# ---------------------------------------------------------------------------
# 메인 비교 실험
# ---------------------------------------------------------------------------
def run_variant(variant_name, model, optim, gen, data, val_data, args, val_gen, probe):
    """한 변형의 학습 + 매 측정 step에서 active fraction & val_ppl 기록."""
    history = {"variant": variant_name, "cycles": []}
    # NREM 보조 (variant C): forced eps 모듈/분류기.
    forced_mod = None; forced_cls = None
    if variant_name == "C_forced_eps":
        forced_mod, forced_cls = setup_forced_eps(model)
    for cyc in range(1, args.steps + 1):
        # WAKE 단계: 모든 변형 공통 — gradient 누적 (cycle_len 배치).
        if variant_name == "C_forced_eps":
            # 사양 정확 적용 (이전 sleep_finetune 함수 사용)
            wake_loss, p_sleep = forced_mod.wake_phase(
                model, optim, data, args.batch, args.seq_len, gen,
                cycle_len=args.cycle_len, lam_curv=0.0,
            )
            forced_cls.update_freq()
            forced_cls.apply_grad_mask(allow_struct=True)
            forced_mod.nrem_phase(model, optim, eta_nrem=args.eta_nrem,
                                  top_eps=args.top_eps)
            if cyc % 10 == 0:
                forced_cls.reclassify()
        else:
            # A & B: 표준 AdamW step (cycle_len batch 평균).
            sum_loss = 0.0
            optim.zero_grad(set_to_none=True)
            for _ in range(args.cycle_len):
                x, y = make_batch(data, args.batch, args.seq_len, gen)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                (loss / float(args.cycle_len)).backward()
                sum_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            wake_loss = sum_loss / args.cycle_len
            # 변형 B에서만: NREM = heat-kernel flow on representations
            if variant_name == "B_heat_flow" and cyc % args.nrem_every == 0:
                nrem_heat_flow_pass(model, data, args.batch, args.seq_len, gen,
                                    n_passes=2, eta_boost=args.eta_boost)
        # 측정 (매 measure_every 사이클).
        if cyc % args.measure_every == 0 or cyc == 1:
            probe.reset()
            with torch.no_grad():
                model.eval()
                for _ in range(4):
                    x, _ = make_batch(val_data, args.batch, args.seq_len, val_gen)
                    _ = model(x)
                model.train()
            active_frac = probe.avg_active()
            ppl = val_perplexity(model, val_data, args.batch, args.seq_len, val_gen,
                                 n_batches=4)
            distance = (active_frac - 0.0487) ** 2
            history["cycles"].append({
                "cyc": cyc,
                "wake_loss": wake_loss,
                "val_ppl": ppl,
                "active_frac": active_frac,
                "distance_to_eps2": distance,
            })
            safe_print(
                f"  [{variant_name}] cyc {cyc:4d}  loss={wake_loss:.3f}  "
                f"ppl={ppl:7.2f}  active={active_frac*100:5.2f}%  "
                f"dist_eps2={distance:.5f}"
            )
    return history


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    safe_print("=" * 70)
    safe_print(" Natural emergence test: does sparsity self-organize to eps^2?")
    safe_print(f"  ckpt={args.ckpt}  device={device}  steps={args.steps}")
    safe_print(f"  cycle_len={args.cycle_len}  measure_every={args.measure_every}")
    safe_print("=" * 70)

    from clarus import load_clarus_lm_generator
    gen_obj = load_clarus_lm_generator(args.ckpt, device=str(device))
    tok = gen_obj.tokenizer
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    full = torch.tensor(tok.encode(text), dtype=torch.long, device=device)
    n_train = int(0.95 * full.shape[0])
    train_data, val_data = full[:n_train], full[n_train:]
    safe_print(f"\n[CORPUS] {full.shape[0]} tokens  train={train_data.shape[0]}  val={val_data.shape[0]}")
    del gen_obj
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_histories = []
    from clarus import load_clarus_lm_generator
    for variant in args.variants.split(","):
        safe_print(f"\n{'='*70}\n  Variant: {variant}\n{'='*70}")
        torch.manual_seed(args.seed)
        gen_obj = load_clarus_lm_generator(args.ckpt, device=str(device))
        model = gen_obj.model
        del gen_obj
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # 학습 가능하게.
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        gen = torch.Generator(device=device); gen.manual_seed(args.seed + 1)
        val_gen = torch.Generator(device=device); val_gen.manual_seed(args.seed + 99)
        probe = ActivationProbe(model, thr_factor=args.thr_factor)
        safe_print(f"  probe layers: {probe._layer_count}")
        # 초기 측정.
        probe.reset()
        with torch.no_grad():
            model.eval()
            for _ in range(4):
                x, _ = make_batch(val_data, args.batch, args.seq_len, val_gen)
                _ = model(x)
            model.train()
        init_active = probe.avg_active()
        init_ppl = val_perplexity(model, val_data, args.batch, args.seq_len, val_gen,
                                  n_batches=4)
        safe_print(f"  init: ppl={init_ppl:.2f}  active={init_active*100:.2f}%  "
                   f"target eps^2={4.87}%")
        t0 = time.perf_counter()
        history = run_variant(variant, model, optim, gen, train_data, val_data,
                              args, val_gen, probe)
        history["init_active"] = init_active
        history["init_ppl"] = init_ppl
        history["time_s"] = time.perf_counter() - t0
        all_histories.append(history)
        probe.remove()
        del model, optim, probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save + summary.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_histories, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    safe_print(f"\n[SAVED] {out_path}")

    safe_print("\n" + "=" * 70)
    safe_print(" Summary: final active fraction vs target 4.87%")
    safe_print("=" * 70)
    safe_print(f"{'variant':<20s}  {'init_act%':>10s}  {'final_act%':>10s}  "
               f"{'final_ppl':>10s}  {'dist_eps2':>12s}")
    for h in all_histories:
        if not h["cycles"]:
            continue
        last = h["cycles"][-1]
        safe_print(
            f"{h['variant']:<20s}  {h['init_active']*100:>9.2f}%  "
            f"{last['active_frac']*100:>9.2f}%  {last['val_ppl']:>10.2f}  "
            f"{last['distance_to_eps2']:>12.5f}"
        )
    safe_print("\n해석:")
    safe_print("  - dist_eps2가 가장 작은 변형이 자연 평형에 가장 가까움.")
    safe_print("  - heat_flow가 plain보다 dist_eps2 작으면 -> spec의 동역학이 작동")
    safe_print("  - forced_eps의 active는 정의상 ~4.87% (강제) -> ppl만 비교")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="clarus/clarus_lm_kogpt2.pt")
    ap.add_argument("--data", default="examples/ai/train_data.txt")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--steps", type=int, default=80,
                    help="Cycles per variant (each cycle = cycle_len batches).")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--cycle-len", type=int, default=4)
    ap.add_argument("--measure-every", type=int, default=10)
    ap.add_argument("--thr-factor", type=float, default=0.1,
                    help="Active threshold = thr_factor * std(|a|).")
    ap.add_argument("--nrem-every", type=int, default=5,
                    help="(B variant) heat-flow every N cycles.")
    ap.add_argument("--eta-boost", type=float, default=5.0,
                    help="(B variant) LBO h temporary boost factor in NREM.")
    ap.add_argument("--eta-nrem", type=float, default=0.005,
                    help="(C variant) weight smoothing rate.")
    ap.add_argument("--top-eps", type=float, default=0.0487,
                    help="(C variant) gradient mask ratio.")
    ap.add_argument("--variants", default="A_plain,B_heat_flow,C_forced_eps",
                    help="Comma-separated list of variants to run.")
    ap.add_argument("--output", default="examples/ai/results/natural_dynamics.json")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
