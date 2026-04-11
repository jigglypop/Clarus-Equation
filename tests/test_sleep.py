from __future__ import annotations

import torch

from clarus.sleep import SleepBatch, fit_decoder_from_batch, fit_token_head_from_batch


def test_fit_decoder_from_batch_recovers_linear_targets():
    torch.manual_seed(31)
    n = 128
    d = 6
    prev_scale = 0.35

    state_x = torch.randn(n, d)
    prev_x = torch.randn(n, d)
    state_true = torch.randn(d, d) * 0.1
    prev_true = torch.randn(d, d) * 0.1

    target_y = state_x @ state_true + prev_scale * (prev_x @ prev_true)
    soft_y = target_y + 0.01 * torch.randn_like(target_y)
    hard_mask = torch.zeros(n, dtype=torch.bool)
    hard_mask[: n // 4] = True

    batch = SleepBatch(
        state_x=state_x,
        prev_x=prev_x,
        target_y=target_y,
        soft_y=soft_y,
        hard_mask=hard_mask,
        top1_hits=~hard_mask,
        top50_hits=~hard_mask,
        target_ids=torch.zeros(n, dtype=torch.long),
    )

    state_proj, prev_proj = fit_decoder_from_batch(
        batch,
        prev_scale=prev_scale,
        ridge=1e-4,
    )
    pred = state_x @ state_proj + prev_scale * (prev_x @ prev_proj)
    mse = torch.mean((pred - target_y) ** 2).item()
    assert mse < 3e-4


def test_fit_decoder_from_batch_accepts_rem_weighting():
    torch.manual_seed(32)
    n = 64
    d = 4
    batch = SleepBatch(
        state_x=torch.randn(n, d),
        prev_x=torch.randn(n, d),
        target_y=torch.randn(n, d),
        soft_y=torch.randn(n, d),
        hard_mask=torch.arange(n) % 3 == 0,
        top1_hits=torch.zeros(n, dtype=torch.bool),
        top50_hits=torch.zeros(n, dtype=torch.bool),
        target_ids=torch.zeros(n, dtype=torch.long),
    )

    state_proj, prev_proj = fit_decoder_from_batch(
        batch,
        prev_scale=0.35,
        ridge=1e-3,
        rem_weight=2.5,
        rem_mix=0.3,
    )
    assert state_proj.shape == (d, d)
    assert prev_proj.shape == (d, d)
    assert torch.isfinite(state_proj).all()
    assert torch.isfinite(prev_proj).all()


def test_fit_token_head_from_batch_recovers_teacher_topk_scores():
    torch.manual_seed(33)
    n = 96
    d = 5
    k = 4
    prev_scale = 0.4
    token_ids = torch.tensor([7, 11, 19, 23], dtype=torch.long)

    state_x = torch.randn(n, d)
    prev_x = torch.randn(n, d)
    state_true = torch.randn(d, k) * 0.1
    prev_true = torch.randn(d, k) * 0.1
    bias_true = torch.randn(k) * 0.05

    teacher_scores = state_x @ state_true + prev_scale * (prev_x @ prev_true) + bias_true
    teacher_probs = torch.softmax(teacher_scores, dim=1)

    batch = SleepBatch(
        state_x=state_x,
        prev_x=prev_x,
        target_y=torch.randn(n, d),
        soft_y=torch.randn(n, d),
        hard_mask=torch.arange(n) % 4 == 0,
        top1_hits=torch.zeros(n, dtype=torch.bool),
        top50_hits=torch.zeros(n, dtype=torch.bool),
        target_ids=torch.zeros(n, dtype=torch.long),
        teacher_top_ids=token_ids.unsqueeze(0).repeat(n, 1),
        teacher_top_probs=teacher_probs,
    )

    head = fit_token_head_from_batch(
        batch,
        prev_scale=prev_scale,
        ridge=1e-4,
        rem_weight=2.0,
        max_vocab=16,
    )

    assert head is not None
    assert torch.equal(head.token_ids, token_ids)
    pred = state_x @ head.state_proj + prev_scale * (prev_x @ head.prev_proj) + head.bias
    mse = torch.mean((pred - teacher_probs) ** 2).item()
    assert mse < 3e-4
