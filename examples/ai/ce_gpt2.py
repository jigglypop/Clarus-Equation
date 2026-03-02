"""CE-GPT2: GPT-2에 Clarus Equation 모듈 이식.

2단계 접근:
  Phase 1 -- 비파괴 이식 (MLP 유지)
    LayerNorm -> LBONorm (h=0 초기화, scale/bias 복사 -> 원본과 동일 출발)
    c_proj    -> spectral_norm (가중치 보존 + 유니타리 제약)
  Phase 2 -- MLP 압축 (cross-mix GaugeLattice)
    MLP -> GaugeLattice + cross-channel mixing (별도 스텝)

Usage:
    python ce_gpt2.py --data train_data.txt
    python ce_gpt2.py --data train_data.txt --phase 2 --steps 500
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
from clarus_lm import LBONorm, ALPHA_S, ALPHA_W, ALPHA_EM

os.environ['PYTHONIOENCODING'] = 'utf-8'


class GaugeLatticeV2(nn.Module):
    """3x3+1 게이지 격자 + cross-channel mixing.

    V1과 달리 채널간 상호작용을 허용하여 dense MLP 근사 가능.
    mixing은 low-rank로 파라미터 절약.
    """

    def __init__(self, dim, mult=4, mix_rank=64):
        super().__init__()
        total = ALPHA_S + ALPHA_W + ALPHA_EM
        self.d3 = max(1, round(dim * ALPHA_S / total))
        self.d2 = max(1, round(dim * ALPHA_W / total))
        self.d1 = dim - self.d3 - self.d2

        h = dim * mult
        h3 = max(1, round(h * ALPHA_S / total))
        h2 = max(1, round(h * ALPHA_W / total))
        h1 = max(1, h - h3 - h2)

        self.su3 = nn.Sequential(
            nn.Linear(self.d3, h3, bias=False), nn.GELU(),
            nn.Linear(h3, self.d3, bias=False))
        self.su2 = nn.Sequential(
            nn.Linear(self.d2, h2, bias=False), nn.GELU(),
            nn.Linear(h2, self.d2, bias=False))
        self.u1 = nn.Sequential(
            nn.Linear(self.d1, h1, bias=False), nn.GELU(),
            nn.Linear(h1, self.d1, bias=False))

        self.mix_down = nn.Linear(dim, mix_rank, bias=False)
        self.mix_up = nn.Linear(mix_rank, dim, bias=False)
        nn.init.zeros_(self.mix_up.weight)

        self.phi = LBONorm(dim)

    def forward(self, x):
        s = self.d3
        y = torch.cat([
            self.su3(x[..., :s]),
            self.su2(x[..., s:s + self.d2]),
            self.u1(x[..., s + self.d2:]),
        ], dim=-1)
        y = y + self.mix_up(self.mix_down(y))
        return self.phi(y)


def init_lbo_from_ln(lbo, ln):
    """h=0 초기화: LBO(x) = x * scale + bias = LayerNorm과 동일 출발."""
    lbo.scale.data = ln.weight.data.clone()
    lbo.bias.data = ln.bias.data.clone()
    with torch.no_grad():
        lbo.h.data.fill_(0.0)


def transplant_phase1(model, device='cpu'):
    """Phase 1: LayerNorm -> LBONorm + spectral norm. MLP 유지."""
    dim = model.config.n_embd

    old_ln_f = model.transformer.ln_f
    model.transformer.ln_f = LBONorm(dim).to(device)
    init_lbo_from_ln(model.transformer.ln_f, old_ln_f)

    for i, block in enumerate(model.transformer.h):
        new_ln1 = LBONorm(dim).to(device)
        new_ln2 = LBONorm(dim).to(device)
        init_lbo_from_ln(new_ln1, block.ln_1)
        init_lbo_from_ln(new_ln2, block.ln_2)
        block.ln_1 = new_ln1
        block.ln_2 = new_ln2

        old_w = block.attn.c_proj.weight.data.clone()
        old_b = block.attn.c_proj.bias.data.clone()
        proj = nn.Linear(dim, dim).to(device)
        proj.weight.data = old_w.T
        proj.bias.data = old_b
        block.attn.c_proj = nn.utils.spectral_norm(proj)

    return model


def transplant_phase2(model, device='cpu', distill_steps=500):
    """Phase 2: MLP -> GaugeLatticeV2. 증류 초기화."""
    dim = model.config.n_embd

    for i, block in enumerate(model.transformer.h):
        old_mlp = block.mlp
        new_lattice = GaugeLatticeV2(dim, mult=4, mix_rank=64).to(device)

        opt = torch.optim.Adam(new_lattice.parameters(), lr=5e-4)
        old_mlp.eval()
        for p in old_mlp.parameters():
            p.requires_grad = False

        for step in range(distill_steps):
            x = torch.randn(8, 32, dim, device=device) * 0.3
            with torch.no_grad():
                target = old_mlp(x)
            pred = new_lattice(x)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        block.mlp = new_lattice
        print(f'  block {i:2d} distill: {loss.item():.4f}', flush=True)
        del old_mlp

    return model


def freeze_for_phase(model, phase):
    for p in model.parameters():
        p.requires_grad = False

    for block in model.transformer.h:
        for p in block.ln_1.parameters():
            p.requires_grad = True
        for p in block.ln_2.parameters():
            p.requires_grad = True
        for p in block.attn.c_proj.parameters():
            p.requires_grad = True
        if phase >= 2:
            for p in block.mlp.parameters():
                p.requires_grad = True
    for p in model.transformer.ln_f.parameters():
        p.requires_grad = True


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def perplexity(model, input_ids, seq_len=128, device='cpu', max_tokens=10000):
    model.eval()
    n = min(len(input_ids), max_tokens)
    nlls = []
    for i in range(0, max(1, n - seq_len), seq_len):
        chunk = input_ids[i:i + seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        v = out.loss.item()
        if not math.isfinite(v):
            return float('inf')
        nlls.append(v)
    if not nlls:
        return float('inf')
    return math.exp(sum(nlls) / len(nlls))


def fine_tune(model, input_ids, steps, lr, batch_size, seq_len,
              device, lambda_curv=0.005):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    warmup = min(50, steps // 5)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    t0 = time.time()

    for step in range(1, steps + 1):
        ix = torch.randint(len(input_ids) - seq_len - 1, (batch_size,))
        x = torch.stack([input_ids[i:i + seq_len] for i in ix]).to(device)
        y = torch.stack([input_ids[i + 1:i + seq_len + 1] for i in ix]).to(device)

        out = model(x, labels=y)
        loss = out.loss

        curv = 0.0
        n = 0
        for block in model.transformer.h:
            for ln in (block.ln_1, block.ln_2):
                if hasattr(ln, '_curvature'):
                    curv += ln._curvature
                    n += 1
            if hasattr(block.mlp, 'phi') and hasattr(block.mlp.phi, '_curvature'):
                curv += block.mlp.phi._curvature
                n += 1
        if n > 0:
            curv /= n

        cw = lambda_curv * min(1.0, step / max(1, warmup))
        total = loss + cw * curv

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        if step % 25 == 0 or step == 1:
            print(f'step {step:4d} | loss {loss.item():.4f} | curv {curv:.4f} '
                  f'| lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s',
                  flush=True)


def safe_print(text):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'), flush=True)


def generate_sample(model, tokenizer, prompt, max_len=60, device='cpu'):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_len, do_sample=True,
            temperature=0.7, top_k=40,
            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default=None)
    p.add_argument('--phase', type=int, default=1, choices=[1, 2])
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--lambda_curv', type=float, default=0.005)
    p.add_argument('--eval_tokens', type=int, default=10000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    print(f'=== CE-GPT2 Phase {args.phase} ===', flush=True)
    print('Loading GPT-2...', flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    if args.data:
        data_path = args.data
    else:
        data_path = os.path.join(os.path.dirname(__file__), 'train_data.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f'Data: {len(input_ids)} tokens', flush=True)

    # --- Baseline ---
    print('\n--- Baseline GPT-2 ---', flush=True)
    base_model = GPT2LMHeadModel.from_pretrained('gpt2').to(args.device)
    base_total = count_params(base_model)
    print(f'Params: {base_total / 1e6:.1f}M', flush=True)
    ppl_base = perplexity(base_model, input_ids, args.seq_len,
                          args.device, args.eval_tokens)
    print(f'PPL: {ppl_base:.2f}', flush=True)
    for pr in ['The fundamental equation']:
        safe_print(f'  gen: {generate_sample(base_model, tokenizer, pr, device=args.device)[:120]}')
    del base_model

    # --- CE-GPT2 ---
    print('\n--- CE-GPT2 ---', flush=True)
    ce_model = GPT2LMHeadModel.from_pretrained('gpt2').to(args.device)

    print('Phase 1: LBONorm + spectral norm...', flush=True)
    transplant_phase1(ce_model, args.device)

    if args.phase >= 2:
        print('Phase 2: MLP -> GaugeLatticeV2 (distilling)...', flush=True)
        transplant_phase2(ce_model, args.device, distill_steps=500)

    freeze_for_phase(ce_model, args.phase)
    ce_total = count_params(ce_model)
    ce_train = count_params(ce_model, trainable_only=True)
    saved = base_total - ce_total

    print(f'Total: {ce_total / 1e6:.1f}M  Trainable: {ce_train / 1e6:.2f}M  '
          f'Saved: {saved / 1e6:.1f}M ({100 * saved / base_total:.1f}%)', flush=True)

    ppl_pre = perplexity(ce_model, input_ids, args.seq_len,
                         args.device, args.eval_tokens)
    print(f'PPL (before tune): {ppl_pre:.2f}', flush=True)

    print(f'\nFine-tuning {args.steps} steps...', flush=True)
    fine_tune(ce_model, input_ids, args.steps, args.lr,
              args.batch_size, args.seq_len, args.device, args.lambda_curv)

    ppl_post = perplexity(ce_model, input_ids, args.seq_len,
                          args.device, args.eval_tokens)
    print(f'PPL (after tune): {ppl_post:.2f}', flush=True)
    for pr in ['The fundamental equation']:
        safe_print(f'  gen: {generate_sample(ce_model, tokenizer, pr, device=args.device)[:120]}')

    # --- Summary ---
    print('\n' + '=' * 55, flush=True)
    print(f'{"":25s} | {"Params":>8s} | {"PPL":>8s}', flush=True)
    print(f'{"-" * 25}-+-{"-" * 8}-+-{"-" * 8}', flush=True)
    print(f'{"GPT-2 baseline":25s} | {base_total / 1e6:>6.1f}M | {ppl_base:>8.2f}', flush=True)
    print(f'{"CE-GPT2 before tune":25s} | {ce_total / 1e6:>6.1f}M | {ppl_pre:>8.2f}', flush=True)
    print(f'{"CE-GPT2 after tune":25s} | {ce_total / 1e6:>6.1f}M | {ppl_post:>8.2f}', flush=True)
    print('=' * 55, flush=True)

    save_path = os.path.join(os.path.dirname(__file__), 'ce_gpt2.pt')
    torch.save({'model': ce_model.state_dict(), 'phase': args.phase}, save_path)
    print(f'Saved: {save_path}', flush=True)


if __name__ == '__main__':
    main()
