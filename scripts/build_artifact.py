"""Build a CE runtime-only artifact from a Korean GPT-2 checkpoint.

Usage:
    python scripts/build_artifact.py
    python scripts/build_artifact.py --model skt/kogpt2-base-v2 --device cpu
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from clarus.ce_ops import pack_sparse

PORTAL = 0.031203
BYPASS = 0.489236
T_WAKE = 0.314798


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((str(msg) + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()


def extract_hidden_states(model, tok, prompts: list[str], device: str) -> torch.Tensor:
    """Run prompts through the model and collect final hidden states."""
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=128)
            ids = {k: v.to(device) for k, v in ids.items()}
            out = model(**ids, output_hidden_states=True)
            hidden = out.hidden_states
            for layer_h in hidden:
                h = layer_h[0].float().cpu()
                all_hidden.append(h)
    return torch.cat(all_hidden, dim=0)


def build_hopfield_w(hidden: torch.Tensor, dim: int) -> torch.Tensor:
    """Build the Hopfield coupling matrix from hidden-state covariance."""
    h = hidden.float()
    h = h - h.mean(dim=0, keepdim=True)
    n = h.shape[0]
    cov = (h.T @ h) / max(n - 1, 1)
    cov = 0.5 * (cov + cov.T)
    cov.fill_diagonal_(0)
    eigvals = torch.linalg.eigvalsh(cov)
    lam_max = float(eigvals[-1].item())
    if lam_max >= -1e-4:
        cov = cov - (lam_max + 1e-3) * torch.eye(dim)
    return cov


def build_decoder_projections(hidden: torch.Tensor, dim: int, ridge: float = 1e-3):
    """Build simple decoder projections from hidden-state statistics."""
    state_proj = torch.eye(dim)
    prev_proj = torch.zeros(dim, dim)
    bias = torch.zeros(dim)

    if hidden.shape[0] > 2:
        h = hidden.float()
        mean = h.mean(dim=0)
        std = h.std(dim=0).clamp_min(1e-6)
        h_norm = (h - mean) / std

        cov = (h_norm.T @ h_norm) / max(h_norm.shape[0] - 1, 1)
        cov = 0.5 * (cov + cov.T) + ridge * torch.eye(dim)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        top_k = min(dim, 32)
        proj_basis = eigvecs[:, -top_k:]
        state_proj = proj_basis @ proj_basis.T

        if h.shape[0] > 3:
            shift = torch.roll(h_norm, 1, dims=0)
            shift[0] = 0
            prev_cov = (shift.T @ h_norm) / max(h_norm.shape[0] - 1, 1)
            prev_proj = 0.5 * (prev_cov + prev_cov.T)

        bias = mean * 0.01

    return state_proj, prev_proj, bias


def build_vocab_head(emb: torch.Tensor, dim: int, top_k: int = 2048):
    """Build a compact vocab head from embedding similarity."""
    norms = emb.norm(dim=1)
    valid = norms > 1e-6
    if valid.sum() < 4:
        return None, None, None, None

    scores = norms[valid]
    k = min(top_k, int(scores.numel()))
    _, top_idx = torch.topk(scores, k)
    all_valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    token_ids = all_valid_idx[top_idx]

    vocab_emb = emb[token_ids]
    vocab_weight = vocab_emb[:, :dim].contiguous()
    vocab_bias = torch.zeros(k)
    return token_ids, vocab_weight, vocab_bias, 1.0


def build_token_head(emb: torch.Tensor, dim: int, top_k: int = 256):
    """Build a smaller token-level scoring head."""
    norms = emb.norm(dim=1)
    valid = norms > 1e-6
    if valid.sum() < 4:
        return None, None, None, None, None

    scores = norms[valid]
    k = min(top_k, int(scores.numel()))
    _, top_idx = torch.topk(scores, k)
    all_valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    token_ids = all_valid_idx[top_idx]

    state_proj = torch.zeros(dim, k)
    prev_proj = torch.zeros(dim, k)
    bias = torch.zeros(k)
    return token_ids, state_proj, prev_proj, bias, 1.0


def build_context_projections(dim: int):
    """Build context prompt-state projections."""
    return {
        "context_first_proj": torch.eye(dim) * 0.1,
        "context_prev_proj": torch.eye(dim) * 0.05,
        "context_last_proj": torch.eye(dim),
        "context_mean_proj": torch.eye(dim) * 0.2,
        "context_decay_proj": torch.eye(dim) * 0.15,
        "context_phi_proj": torch.eye(dim) * 0.1,
        "context_len_proj": torch.zeros(dim),
        "context_bias": torch.zeros(dim),
    }


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Build CE runtime artifact from Korean GPT-2")
    ap.add_argument("--model", default="skt/kogpt2-base-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    t0 = time.time()
    log("=" * 60)
    log("  CE Runtime Artifact Builder")
    log("=" * 60)
    log(f"  model: {args.model}")
    log(f"  device: {args.device}")

    log("\n[1/6] Loading pretrained model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    dim = model.config.n_embd
    vocab = model.config.vocab_size
    n_layer = model.config.n_layer
    log(f"  dim={dim}  vocab={vocab}  n_layer={n_layer}")

    log("\n[2/6] Collecting hidden states...")
    prompts = [
        "인공지능의 미래는 우리가 생각하는 것보다",
        "오늘 날씨가 좋아서 밖에 나가면",
        "한국어를 배우는 가장 좋은 방법은",
        "좋은 모델의 조건은 정확성과",
        "대한민국의 교육 제도는",
        "건강한 식단을 유지하려면",
        "세계 경제의 흐름은 최근",
        "과학 기술의 발전은 인류의 삶을",
        "한국의 전통 음식 중 대표적인 것은",
        "효율적인 학습을 위해서는 반복과",
        "한글의 우수성은 과학적 원리에 기반한",
        "도시와 농촌의 격차를 줄이기 위해서는",
    ]
    hidden = extract_hidden_states(model, tok, prompts, args.device)
    log(f"  collected {hidden.shape[0]} hidden vectors (dim={hidden.shape[1]})")

    log("\n[3/6] Building Hopfield W matrix...")
    W = build_hopfield_w(hidden, dim)
    values, col_idx, row_ptr = pack_sparse(W, backend="torch")
    log(f"  W shape={W.shape}  nnz={values.numel()}")

    log("\n[4/6] Building decoder projections...")
    state_proj, prev_proj, query_bias = build_decoder_projections(hidden, dim)

    emb_weight = model.get_input_embeddings().weight.detach().cpu().float()
    pos_weight = None
    if hasattr(model.config, "n_positions"):
        for name, param in model.named_parameters():
            if "wpe" in name:
                pos_weight = param.detach().cpu().float()
                break

    th_ids, th_state, th_prev, th_bias, th_scale = build_token_head(emb_weight, dim)
    context_projs = build_context_projections(dim)

    log("\n[5/6] Extracting tokenizer...")
    from tokenizers import Tokenizer as HfTokenizer
    backend_tokenizer = tok.backend_tokenizer
    tokenizer_json = backend_tokenizer.to_str()
    tokenizer_specials = {}
    if tok.pad_token:
        tokenizer_specials["pad_token"] = tok.pad_token
    if tok.eos_token:
        tokenizer_specials["eos_token"] = tok.eos_token
    if tok.bos_token:
        tokenizer_specials["bos_token"] = tok.bos_token
    if tok.unk_token:
        tokenizer_specials["unk_token"] = tok.unk_token

    ln_f = None
    for name, module in model.named_modules():
        if "ln_f" in name and hasattr(module, "weight"):
            ln_f = module
            break
    ln_f_weight = ln_f.weight.detach().cpu().float() if ln_f else torch.ones(dim)
    ln_f_bias = ln_f.bias.detach().cpu().float() if ln_f and ln_f.bias is not None else torch.zeros(dim)

    log("\n[6/6] Assembling runtime artifact...")
    artifact = {
        "artifact_version": 3,
        "model_name": args.model,
        "allow_pretrained_fallback": False,
        "d": dim,
        "vocab": vocab,
        "n_layer": n_layer,
        "tau": 1.0,
        "portal": PORTAL,
        "bypass": BYPASS,
        "t_wake": T_WAKE,
        "hidden_norm_ref": float(hidden.norm(dim=1).mean().item()),
        "r_c": math.pi,
        "active_ratio": 0.0487,
        "struct_ratio": 0.2623,
        "wake_ratio": 0.6891,
        "nrem_ratio": 0.2623,
        "rem_ratio": 0.0487,
        "target_w_density": 0.0316,
        "W": W,
        "W_values": values,
        "W_col_idx": col_idx,
        "W_row_ptr": row_ptr,
        "W_layers": [],
        "emb_weight": emb_weight,
        "pos_weight": pos_weight,
        "ln_f_weight": ln_f_weight,
        "ln_f_bias": ln_f_bias,
        "decoder_prev_scale": 0.35,
        "decoder_prev_proj": prev_proj,
        "decoder_state_proj": state_proj,
        "decoder_query_bias": query_bias,
        "decoder_vocab_weight": None,
        "decoder_vocab_bias": None,
        "decoder_vocab_scale": 1.0,
        "decoder_query_blend": 0.7,
        "decoder_candidate_ratio": 0.04865,
        "curvature_alpha": 1.5,
        "curvature_lambda": 1.25,
        "curvature_steepness": 8.0,
        "curvature_eval_topk": 256,
        "repeat_window": 16,
        "repeat_ngram": 3,
        "decoder_token_ids": th_ids,
        "decoder_token_state_proj": th_state,
        "decoder_token_prev_proj": th_prev,
        "decoder_token_bias": th_bias,
        "decoder_token_scale": th_scale or 1.0,
        "tokenizer_json": tokenizer_json,
        "tokenizer_specials": tokenizer_specials,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        **context_projs,
    }

    safe_model = args.model.replace("/", "_")
    out_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "clarus", f"{safe_model}.ce.pt"
    )
    torch.save(artifact, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    elapsed = time.time() - t0

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log(f"\n  Artifact saved: {out_path}")
    log(f"  Size: {size_mb:.2f} MB")
    log(f"  Time: {elapsed:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
