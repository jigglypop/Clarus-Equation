from __future__ import annotations

import argparse
import pytest
import torch
import torch.nn as nn

from clarus.ce_ops import pack_sparse
from clarus.engine import CEEngine
from tests.bench_gpt2 import build_prompt_weights, select_topical_chunks, sleep_curriculum_stage
from clarus.sleep import (
    PromptReplayBuffer,
    SleepBatch,
    _target_distribution,
    allocate_phase_sample_counts,
    classify_state_dimensions,
    evaluate_guard_set,
    finetune_vocab_head_from_batch,
    fit_decoder_from_batch,
    fit_token_head_from_batch,
    load_corpus_documents,
    offdiag_density,
    prioritize_documents_for_prompts,
    row_topk_mask,
    run_guarded_microsleep_step,
    should_accept_guard_update,
)

PORTAL = 0.031203
BYPASS = 0.489236
T_WAKE = 0.314798


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

    state_proj, prev_proj, bias = fit_decoder_from_batch(
        batch,
        prev_scale=prev_scale,
        ridge=1e-4,
    )
    pred = state_x @ state_proj + prev_scale * (prev_x @ prev_proj) + bias
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

    state_proj, prev_proj, bias = fit_decoder_from_batch(
        batch,
        prev_scale=0.35,
        ridge=1e-3,
        rem_weight=2.5,
        rem_mix=0.3,
    )
    assert state_proj.shape == (d, d)
    assert prev_proj.shape == (d, d)
    assert bias.shape == (d,)
    assert torch.isfinite(state_proj).all()
    assert torch.isfinite(prev_proj).all()
    assert torch.isfinite(bias).all()


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


class IdentityBlock(nn.Module):
    def forward(self, h):
        return h


class FakeModel(nn.Module):
    def __init__(self, vocab: int = 8, dim: int = 4, layers: int = 3):
        super().__init__()
        transformer = nn.Module()
        transformer.wte = nn.Embedding(vocab, dim)
        transformer.wpe = nn.Embedding(32, dim)
        transformer.h = nn.ModuleList(IdentityBlock() for _ in range(layers))
        transformer.ln_f = nn.LayerNorm(dim)
        self.transformer = transformer


def make_runtime_artifact(tmp_path, *, decoder_query_blend=1.0):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<pad>": 0, "<eos>": 1, "alpha": 2, "beta": 3},
            unk_token="<pad>",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    w = torch.eye(4, dtype=torch.float32) * -1.0
    values, col_idx, row_ptr = pack_sparse(w, backend="torch")
    artifact = {
        "artifact_version": 3,
        "model_name": "unit-test",
        "allow_pretrained_fallback": False,
        "d": 4,
        "vocab": 4,
        "n_layer": 1,
        "tau": 1.0,
        "portal": PORTAL,
        "bypass": BYPASS,
        "t_wake": T_WAKE,
        "r_c": 3.141592653589793,
        "active_ratio": 0.25,
        "struct_ratio": 0.25,
        "wake_ratio": 0.6891,
        "nrem_ratio": 0.2623,
        "rem_ratio": 0.0487,
        "target_w_density": 0.0,
        "W": w,
        "W_values": values,
        "W_col_idx": col_idx,
        "W_row_ptr": row_ptr,
        "W_layers": [],
        "emb_weight": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        "pos_weight": torch.zeros(8, 4),
        "ln_f_weight": torch.ones(4),
        "ln_f_bias": torch.zeros(4),
        "decoder_prev_scale": 0.35,
        "decoder_prev_proj": torch.zeros(4, 4),
        "decoder_state_proj": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "decoder_query_bias": torch.zeros(4),
        "decoder_vocab_weight": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        "decoder_vocab_bias": torch.zeros(4),
        "decoder_vocab_scale": 1.0,
        "decoder_query_blend": decoder_query_blend,
        "decoder_candidate_ratio": 0.5,
        "curvature_alpha": 0.0,
        "curvature_lambda": 1.0,
        "curvature_steepness": 8.0,
        "curvature_eval_topk": 4,
        "repeat_window": 4,
        "repeat_ngram": 2,
        "decoder_token_ids": torch.tensor([2, 3], dtype=torch.long),
        "decoder_token_state_proj": torch.zeros(4, 2),
        "decoder_token_prev_proj": torch.zeros(4, 2),
        "decoder_token_bias": torch.tensor([0.0, 1.5]),
        "decoder_token_scale": 1.0,
        "context_first_proj": torch.zeros(4, 4),
        "context_prev_proj": torch.zeros(4, 4),
        "context_last_proj": torch.eye(4),
        "context_mean_proj": torch.zeros(4, 4),
        "context_decay_proj": torch.zeros(4, 4),
        "context_phi_proj": torch.zeros(4, 4),
        "context_len_proj": torch.zeros(4),
        "context_bias": torch.zeros(4),
        "default_init_layer": 0,
        "tokenizer_json": tokenizer.to_str(),
        "tokenizer_specials": {"pad_token": "<pad>", "eos_token": "<eos>"},
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    path = tmp_path / "runtime.pt"
    torch.save(artifact, path)
    return path


def test_runtime_only_artifact_loads_without_clone_state(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    assert eng.model is None
    assert eng.model_source == "runtime"
    ctx = eng.prompt_context("alpha")
    assert ctx.m0.shape == (4,)
    assert torch.isfinite(ctx.m0).all()


def test_standalone_logits_uses_decoder_query_and_token_head(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    ce_hidden = torch.tensor([0.0, 1.0, 0.0, 0.0])
    logits = eng.standalone_logits(ce_hidden, prev_id=2, temperature=1.0)
    assert int(torch.argmax(logits).item()) == 3


def test_standalone_logits_reports_repeat_suppression(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    ce_hidden = torch.tensor([0.0, 1.0, 0.0, 0.0])
    base_logits, _ = eng.standalone_logits(
        ce_hidden,
        prev_id=2,
        temperature=1.0,
        history_ids=[2],
        return_meta=True,
    )
    penalized_logits, meta = eng.standalone_logits(
        ce_hidden,
        prev_id=2,
        temperature=1.0,
        history_ids=[2, 2],
        repeat_ids=[2],
        repeat_penalty=4.0,
        return_meta=True,
    )
    assert penalized_logits[2].item() < base_logits[2].item()
    assert meta["suppressed_count"] >= 1


def test_standalone_logits_biases_sentence_closure_later(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    ce_hidden = torch.tensor([0.0, 1.0, 0.0, 0.0])
    short_logits = eng.standalone_logits(
        ce_hidden,
        prev_id=2,
        temperature=1.0,
        generated_len=0,
    )
    long_logits = eng.standalone_logits(
        ce_hidden,
        prev_id=2,
        temperature=1.0,
        generated_len=12,
    )
    assert eng.eos_token_id is not None
    assert long_logits[int(eng.eos_token_id)].item() > short_logits[int(eng.eos_token_id)].item()


def test_target_distribution_keeps_true_token_as_primary_label(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    target_emb, top_idx, probs, soft_target = _target_distribution(eng, target_id=3, topk=4)
    assert int(top_idx[0].item()) == 3
    assert float(probs[0].item()) >= 0.85
    assert torch.allclose(target_emb, eng.token_embedding([3]).squeeze(0))
    assert torch.isfinite(soft_target).all()


def test_finetune_vocab_head_from_batch_improves_true_token_logit(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    state_x = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(4, 1)
    prev_x = torch.zeros_like(state_x)
    target_y = eng.token_embedding([2]).repeat(4, 1)
    batch = SleepBatch(
        state_x=state_x,
        prev_x=prev_x,
        target_y=target_y,
        soft_y=target_y.clone(),
        hard_mask=torch.zeros(4, dtype=torch.bool),
        top1_hits=torch.zeros(4, dtype=torch.bool),
        top50_hits=torch.zeros(4, dtype=torch.bool),
        target_ids=torch.tensor([2, 2, 2, 2], dtype=torch.long),
    )
    query = eng.decoder_query(state_x[:1], prev_x[:1]).squeeze(0)
    before = float(eng.vocab_logits(query)[2].item())
    stats = finetune_vocab_head_from_batch(eng, batch, steps=12, batch_size=4)
    after = float(eng.vocab_logits(query)[2].item())
    assert after > before
    assert stats["top1_acc"] >= 0.75


def test_finetune_vocab_head_from_batch_uses_soft_topk_targets(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng_hard = CEEngine(str(path), device="cpu", backend="torch")
    eng_soft = CEEngine(str(path), device="cpu", backend="torch")
    state_x = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(8, 1)
    prev_x = torch.zeros_like(state_x)
    target_y = eng_hard.token_embedding([2]).repeat(8, 1)
    teacher_top_ids = torch.tensor([[2, 3]], dtype=torch.long).repeat(8, 1)
    teacher_top_probs = torch.tensor([[0.8, 0.2]], dtype=torch.float32).repeat(8, 1)
    batch = SleepBatch(
        state_x=state_x,
        prev_x=prev_x,
        target_y=target_y,
        soft_y=target_y.clone(),
        hard_mask=torch.zeros(8, dtype=torch.bool),
        top1_hits=torch.zeros(8, dtype=torch.bool),
        top50_hits=torch.zeros(8, dtype=torch.bool),
        target_ids=torch.tensor([2] * 8, dtype=torch.long),
        teacher_top_ids=teacher_top_ids,
        teacher_top_probs=teacher_top_probs,
    )
    query = eng_hard.decoder_query(state_x[:1], prev_x[:1]).squeeze(0)
    hard_stats = finetune_vocab_head_from_batch(
        eng_hard,
        batch,
        steps=24,
        batch_size=4,
        soft_target_weight=0.0,
    )
    soft_stats = finetune_vocab_head_from_batch(
        eng_soft,
        batch,
        steps=24,
        batch_size=4,
        soft_target_weight=0.8,
    )
    hard_alt = float(eng_hard.vocab_logits(query)[3].item())
    soft_alt = float(eng_soft.vocab_logits(query)[3].item())
    assert soft_alt > hard_alt
    assert hard_stats["top1_acc"] >= 0.75
    assert soft_stats["top1_acc"] >= 0.75


def test_evaluate_guard_set_runs_without_teacher_model(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    ce_args = argparse.Namespace(
        dt=0.01,
        cb_weight=None,
        cb_topk=4,
        beta=1.0,
        steps=8,
        backend="torch",
        metric_rank=0,
        lambda0=1.0,
        lambda_phi=0.5,
        lambda_var=0.25,
        noise_scale=0.0,
        seed=0,
        decode_mode="standalone",
        ce_strength=0.3,
        tokens=4,
        temperature=0.8,
        phi_threshold=1.0,
        sleep_threshold=2.0,
        sleep_decay=0.9,
        top_k=4,
        repeat_penalty=2.0,
        multiround_steps=8,
        standalone_refresh_interval=1,
        standalone_refresh_steps=4,
        standalone_refresh_cb_topk=4,
        standalone_refresh_metric_rank=0,
        standalone_refresh_noise_scale=0.0,
    )
    metrics = evaluate_guard_set(
        eng,
        ["alpha beta alpha beta"],
        ce_args,
        max_new_tokens=2,
        refresh_interval=1,
        refresh_steps=4,
        refresh_cb_topk=4,
        refresh_metric_rank=0,
        refresh_noise_scale=0.0,
    )
    assert metrics["samples"] > 0
    assert 0.0 <= metrics["curvature_risk"] <= 1.0


def test_load_corpus_documents_reads_text_file(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("첫 문장입니다.\n둘째 문장입니다.\n\n셋째 문장입니다.", encoding="utf-8")
    docs = load_corpus_documents(str(corpus), doc_limit=8, text_limit=1024)
    assert docs == ["첫 문장입니다. 둘째 문장입니다.", "셋째 문장입니다."]


def test_prioritize_documents_for_prompts_prefers_overlapping_docs():
    docs = [
        "경제 전망과 금융 시장의 흐름을 설명한다.",
        "건강한 식단은 채소와 단백질의 균형이 중요하다.",
        "대한민국의 교육 제도는 입시 구조와 연결된다.",
    ]
    ordered = prioritize_documents_for_prompts(docs, ["건강한 식단을 유지하려면"])
    assert ordered[0] == docs[1]


def test_select_topical_chunks_prefers_overlapping_chunks():
    prompt_weights = build_prompt_weights(["건강한 식단을 유지하려면", "대한민국의 교육 제도는"])
    chunks = [
        "베토벤의 교향곡은 유럽 음악사에서 중요하다.",
        "건강한 식단은 단백질과 채소의 균형이 중요하다.",
        "대한민국의 교육 제도는 입시와 공교육 구조를 포함한다.",
    ]
    picked = select_topical_chunks(chunks, prompt_weights, keep_limit=2)
    assert chunks[1] in picked
    assert chunks[2] in picked


def test_sleep_curriculum_stage_advances_by_cycle():
    assert sleep_curriculum_stage(1)["name"] == "wiki"
    assert sleep_curriculum_stage(3)["name"] == "wiki+alpaca"
    assert sleep_curriculum_stage(5)["name"] == "wiki+alpaca+squad"


def test_analyze_prompt_ids_single_token_stays_finite():
    eng = CEEngine.__new__(CEEngine)
    eng.model = FakeModel()
    eng.n_layer = 3
    eng.device = torch.device("cpu")

    phi, captured, h_true = eng._analyze_prompt_ids(
        torch.tensor([[1]], dtype=torch.long),
        candidate_layers=[0, 2],
        need_teacher=True,
    )
    assert torch.isfinite(phi).all()
    assert torch.equal(phi, torch.zeros_like(phi))
    assert set(captured) == {0, 2}
    assert h_true.shape == (1, 4)


def test_prompt_replay_buffer_keeps_recent_prompts():
    buf = PromptReplayBuffer(capacity=3)
    buf.extend(["a", "b", "c", "d"])
    assert buf.items() == ["b", "c", "d"]


def test_should_accept_guard_update_rejects_regression():
    before = {"top10_acc": 0.25, "top50_acc": 0.50}
    after = {"top10_acc": 0.20, "top50_acc": 0.55}
    assert not should_accept_guard_update(before, after)
    assert should_accept_guard_update(before, {"top10_acc": 0.25, "top50_acc": 0.55})


def test_classify_state_dimensions_matches_target_partition():
    batch = SleepBatch(
        state_x=torch.tensor(
            [
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            ]
        ),
        prev_x=torch.zeros(2, 6),
        target_y=torch.zeros(2, 6),
        soft_y=torch.zeros(2, 6),
        hard_mask=torch.zeros(2, dtype=torch.bool),
        top1_hits=torch.zeros(2, dtype=torch.bool),
        top50_hits=torch.zeros(2, dtype=torch.bool),
        target_ids=torch.zeros(2, dtype=torch.long),
    )
    part = classify_state_dimensions(batch, active_ratio=0.2, struct_ratio=0.3)
    assert int(part["active_mask"].sum().item()) == 1
    assert int(part["struct_mask"].sum().item()) == 3
    assert int(part["background_mask"].sum().item()) == 3


def test_allocate_phase_sample_counts_matches_profile():
    counts = allocate_phase_sample_counts(
        12,
        {"wake": 0.6891, "nrem": 0.2623, "rem": 0.0487},
    )
    assert counts == {"wake": 8, "nrem": 3, "rem": 1}


def test_row_topk_mask_hits_target_offdiag_density():
    matrix = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    matrix = 0.5 * (matrix + matrix.T)
    mask = row_topk_mask(matrix, keep_ratio=0.25)
    assert not torch.diagonal(mask).any()
    assert offdiag_density(mask) == 0.25


def test_engine_resparsify_relax_matrix_hits_target_density():
    eng = CEEngine.__new__(CEEngine)
    eng.d = 8
    eng.device = torch.device("cpu")
    eng.target_w_density = 0.25
    eng.sparsity_radius = 10.0
    eng._state_coords = None
    eng._state_graph_laplacian = None
    w = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    w = 0.5 * (w + w.T)
    sparse = eng.resparsify_relax_matrix(w)
    off = sparse.clone()
    off.fill_diagonal_(0)
    density = float((off != 0).sum().item()) / float(8 * 7)
    assert density == 0.25


def test_run_guarded_microsleep_step_waits_for_trigger():
    buf = PromptReplayBuffer(capacity=2)
    event = run_guarded_microsleep_step(
        None,
        buf,
        "alpha",
        [],
        None,
        step_index=1,
        sleep_every=2,
        max_new_tokens=1,
        teacher_topk=1,
        ridge=1e-3,
        rem_weight=1.0,
        rem_mix=0.0,
        token_head_max_vocab=8,
        token_head_scale=1.0,
        refresh_interval=0,
        refresh_steps=1,
        refresh_cb_topk=1,
        refresh_metric_rank=0,
        refresh_noise_scale=0.0,
        refresh_pq=False,
        pq_subdim=1,
        pq_bits=1,
        pq_iters=1,
        pq_batch_size=1,
        pq_sample_size=1,
    )
    assert event is None
    assert buf.items() == ["alpha"]
