"""Train ClarusLM on text data (character-level).

Usage:
    python train_clarus.py --data input.txt
    python train_clarus.py --data input.txt --dim 512 --n_layers 12 --steps 10000
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from clarus_lm import ClarusLM


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids):
        return ''.join(self.itos.get(i, '?') for i in ids)


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
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
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


def main():
    p = argparse.ArgumentParser(description='Train ClarusLM')
    p.add_argument('--data', type=str, required=True, help='Text file path')
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--lambda_curv', type=float, default=0.01)
    p.add_argument('--ffn_hidden_dim', type=int, default=None)
    p.add_argument('--mix_rank', type=int, default=None)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--c3_passes', type=int, default=1)
    p.add_argument('--c3_candidates', type=int, default=1)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save', type=str, default='clarus_lm.pt')
    args = p.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = max(1000, int(len(data) * 0.05))
    train_data, val_data = data[:-n_val], data[-n_val:]
    seed_everything(args.seed)

    model = ClarusLM(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        ffn_hidden_dim=args.ffn_hidden_dim,
        mix_rank=args.mix_rank,
        lambda_curv=args.lambda_curv,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'ClarusLM  {n_params / 1e6:.2f}M params')
    print(f'  vocab={tok.vocab_size}  dim={args.dim}  layers={args.n_layers}  heads={args.n_heads}')
    print(f'  train={len(train_data)}  val={len(val_data)} chars')
    print(f'  device={args.device}  lambda_curv={args.lambda_curv}  seed={args.seed}')
    print(f'  ffn_hidden_dim={args.ffn_hidden_dim}  mix_rank={args.mix_rank}')
    print(f'  c3_passes={args.c3_passes}  c3_candidates={args.c3_candidates}')
    print(f'\n3x3+1 lattice:')
    for line in model.lattice_summary().splitlines():
        print(f'  {line}')
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    warmup = min(500, args.steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    best_val = float('inf')
    t0 = time.time()
    train_generator = build_generator(args.seed + 1)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(
            train_data,
            args.seq_len,
            args.batch_size,
            args.device,
            generator=train_generator,
        )
        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == 1:
            val_loss = estimate_loss(
                model, val_data, args.seq_len,
                min(args.batch_size, 8), args.device,
                generator_seed=args.seed + 2)
            curv = sum(b.curvature for b in model.blocks) / len(model.blocks)
            elapsed = time.time() - t0
            print(f'step {step:5d} | loss {loss.item():.4f} | val {val_loss:.4f} '
                  f'| curv {curv:.6f} | lr {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s')

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'config': {
                        'vocab_size': tok.vocab_size,
                        'dim': args.dim,
                        'n_layers': args.n_layers,
                        'n_heads': args.n_heads,
                        'max_seq_len': args.seq_len,
                        'ffn_hidden_dim': args.ffn_hidden_dim,
                        'mix_rank': args.mix_rank,
                        'lambda_curv': args.lambda_curv,
                        'seed': args.seed,
                    },
                    'tokenizer': {'stoi': tok.stoi, 'itos': tok.itos},
                }, args.save)

        if step % 1000 == 0:
            model.eval()
            seed = tok.encode('\n')
            ctx = torch.tensor([seed], dtype=torch.long, device=args.device)
            gen = model.generate(
                ctx,
                300,
                c3_passes=max(1, args.c3_passes),
                c3_candidates=max(1, args.c3_candidates),
            )
            print('--- generated sample ---')
            print(tok.decode(gen[0].tolist()))
            print('-' * 50)

    print(f'\nDone. Best val loss: {best_val:.4f}. Saved: {args.save}')


if __name__ == '__main__':
    main()
