"""CE-AGI Model Converter: Transformer -> standalone Hopfield engine.

Pipeline:
  1. Extract W_hop from transformer (sum, NOT averaged)
  2. Spectral conditioning (negative definite)
  3. Optional 3D sparsify
  4. Save: W + embedding (codebook) + constants
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from math import e, pi

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .ce_ops import pack_sparse as ce_pack_sparse, pq_build_codebook
except ImportError:
    from clarus.ce_ops import pack_sparse as ce_pack_sparse, pq_build_codebook

_ad = 4 / (e ** (4 / 3) * pi ** (4 / 3))
PORTAL = (_ad * (1 - _ad)) ** 2
BYPASS = 1 / (e ** (1 / 3) * pi ** (1 / 3))
T_WAKE = 1 / (3 + _ad * (1 - _ad))
R_C = pi
CALIBRATION_PROMPTS = (
    "오늘 날씨가",
    "인공지능의 미래는",
    "맛있는 음식을",
    "한국의 역사에서",
    "대한민국의 수도는",
    "경제 전망은",
    "The future of AI is",
    "In the history of science,",
)


def log(msg):
    print(msg, flush=True)


def extract_hopfield(mdl, d, n_layer, d_head):
    device = next(mdl.parameters()).device
    W = torch.zeros(d, d, device=device)
    for layer in mdl.transformer.h:
        w_attn = layer.attn.c_attn.weight.data
        Wq, Wk, Wv = w_attn[:, :d], w_attn[:, d:2*d], w_attn[:, 2*d:]
        Wo = layer.attn.c_proj.weight.data
        qk = Wq @ Wk.T
        W += (qk + qk.T) / (2 * d_head ** 0.5)
        vo = Wv @ Wo
        W += (vo + vo.T) / 2
        Wf = layer.mlp.c_fc.weight.data @ layer.mlp.c_proj.weight.data
        W += (Wf + Wf.T) / 4
    return (W + W.T) / 2


def extract_per_layer(mdl, d, n_layer, d_head):
    """Extract per-layer transition matrices (NOT symmetrized)."""
    device = next(mdl.parameters()).device
    layers = []
    for layer in mdl.transformer.h:
        w_attn = layer.attn.c_attn.weight.data
        Wq, Wk, Wv = w_attn[:, :d], w_attn[:, d:2*d], w_attn[:, 2*d:]
        Wo = layer.attn.c_proj.weight.data
        W_layer = (Wv @ Wo + layer.mlp.c_fc.weight.data @ layer.mlp.c_proj.weight.data)
        W_layer = W_layer / W_layer.norm() * (d ** 0.5)
        layers.append(W_layer.cpu())
    return layers


def make_negative_definite(W):
    eigvals, V = torch.linalg.eigh(W)
    lam_min, lam_max = eigvals[0].item(), eigvals[-1].item()
    log(f"  raw eig: [{lam_min:.4f}, {lam_max:.4f}]")
    margin = 0.01 * abs(lam_min) if abs(lam_min) > 1e-6 else 0.01
    if lam_max <= -margin:
        log(f"  already negative-definite")
        return W, eigvals
    shift = lam_max + margin
    eigvals_shifted = eigvals - shift
    W_safe = V @ torch.diag(eigvals_shifted) @ V.T
    W_safe = (W_safe + W_safe.T) / 2
    log(f"  shift={shift:.4f}  eig: [{eigvals_shifted[0]:.4f}, {eigvals_shifted[-1]:.4f}]")
    return W_safe, eigvals_shifted


def build_lattice(N):
    s = int(np.ceil(N ** (1 / 3)))
    coords = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        coords[i] = [i // (s * s), (i // s) % s, i % s]
    return coords, s


def sparsify_3d(W, N, r_c=R_C):
    coords, side = build_lattice(N)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)
    mask = torch.from_numpy((dist_sq < r_c ** 2).astype(np.float32)).to(W.device)
    mask.fill_diagonal_(0)
    W_sp = W * mask
    nnz = int(mask.sum().item())
    K_avg = nnz / N
    log(f"  lattice {side}^3, r_c=pi")
    log(f"  K_avg={K_avg:.1f}  density={nnz / (N * (N - 1)) * 100:.2f}%")
    return W_sp


def fit_decoder_prev_proj(mdl, tok, emb_weight, device, max_new_tokens=48, ridge=1e-3):
    prev_ids: list[int] = []
    next_ids: list[int] = []
    pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id

    for prompt in CALIBRATION_PROMPTS:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(ids)
        with torch.no_grad():
            out = mdl.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        seq = out[0].tolist()
        prev_ids.extend(seq[:-1])
        next_ids.extend(seq[1:])

    x = emb_weight[prev_ids]
    y = emb_weight[next_ids]
    xtx = x.T @ x
    xty = x.T @ y
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype)
    proj = torch.linalg.solve(xtx + ridge * eye, xty)
    return proj.cpu().float(), len(prev_ids)


def fit_decoder_state_proj(mdl, tok, emb_weight, device, max_new_tokens=48, ridge=1e-3):
    xs = []
    ys = []
    pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id

    for prompt in CALIBRATION_PROMPTS:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(ids)
        with torch.no_grad():
            out = mdl.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
            seq = out[0]
            if seq.numel() < 2:
                continue
            seq_in = seq[:-1].unsqueeze(0)
            mask_in = torch.ones_like(seq_in)
            h = mdl.transformer(seq_in, attention_mask=mask_in).last_hidden_state.squeeze(0)
            h = mdl.transformer.ln_f(h)
        xs.append(h.cpu().float())
        ys.append(emb_weight[seq[1:].cpu()].float())

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    xtx = x.T @ x
    xty = x.T @ y
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype)
    proj = torch.linalg.solve(xtx + ridge * eye, xty)
    return proj.cpu().float(), int(x.shape[0])


def extract_clone_state(mdl):
    clone_state = {}
    for name, param in mdl.named_parameters():
        if torch.is_floating_point(param):
            clone_state[name] = param.detach().cpu().to(dtype=torch.float16)
        else:
            clone_state[name] = param.detach().cpu()
    return clone_state


def convert(
    model_name: str,
    out_path: str,
    device_name: str = "cpu",
    sparse: bool = False,
    *,
    save_pq: bool = False,
    pq_only: bool = False,
    pq_subdim: int = 64,
    pq_bits: int = 8,
    pq_iters: int = 16,
    pq_batch_size: int = 4096,
    pq_sample_size: int = 16384,
    decoder_prev_scale: float = 0.35,
    save_clone: bool = False,
):
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    device = torch.device(device_name)
    log(f"Loading {model_name} ...")
    tok = PreTrainedTokenizerFast.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    mdl.to(device).eval()

    cfg = mdl.config
    d = cfg.n_embd if hasattr(cfg, "n_embd") else cfg.hidden_size
    n_layer = cfg.n_layer if hasattr(cfg, "n_layer") else cfg.num_hidden_layers
    n_head = cfg.n_head if hasattr(cfg, "n_head") else cfg.num_attention_heads
    d_head = d // n_head
    vocab = cfg.vocab_size
    log(f"  d={d}  layers={n_layer}  heads={n_head}  vocab={vocab}")

    log("\n[1/5] Extracting Hopfield matrix (SUM) ...")
    t0 = time.time()
    W_hop = extract_hopfield(mdl, d, n_layer, d_head)
    log(f"  ||W_sum|| = {W_hop.norm():.2f}  time={time.time()-t0:.1f}s")

    log("\n[2/5] Extracting per-layer transition matrices ...")
    W_layers = extract_per_layer(mdl, d, n_layer, d_head)
    for i, wl in enumerate(W_layers):
        log(f"  layer {i:2d}: ||W||={wl.norm():.2f}")

    if sparse:
        log("\n[3/5] 3D sparsification ...")
        W_hop = sparsify_3d(W_hop, d)
    else:
        log("\n[3/5] Dense mode ...")
        log(f"  W_sum size: {d*d*4/1024:.1f} KB")
        log(f"  W_layers size: {n_layer*d*d*4/1024:.1f} KB ({n_layer*d*d*4/1024/1024:.2f} MB)")

    log("\n[4/5] Spectral conditioning (W_sum) ...")
    W_cond, eigvals = make_negative_definite(W_hop)
    tau = 1.0 / abs(eigvals[-1].item()) if abs(eigvals[-1].item()) > 1e-8 else 1.0
    log(f"  tau = {tau:.6f}")

    log("\n[4.5/5] Packing sparse runtime state ...")
    w_values, w_col_idx, w_row_ptr = ce_pack_sparse(W_cond.cpu(), backend="torch")
    _, eigvecs_runtime = torch.linalg.eigh(W_cond.cpu())
    hess_rank = min(8, eigvecs_runtime.shape[1])
    w_eigvecs = eigvecs_runtime[:, :hess_rank].T.contiguous()
    packed_kb = (
        w_values.numel() * w_values.element_size()
        + w_col_idx.numel() * w_col_idx.element_size()
        + w_row_ptr.numel() * w_row_ptr.element_size()
    ) / 1024
    log(f"  nnz={w_values.numel()}  packed={packed_kb:.1f} KB  hess_rank={hess_rank}")

    log("\n[5/5] Extracting embeddings ...")
    emb_weight = mdl.transformer.wte.weight.data.cpu().float()
    emb_norms = emb_weight.norm(dim=1)
    log(f"  emb norms: mean={emb_norms.mean():.2f}  min={emb_norms.min():.2f}  max={emb_norms.max():.2f}")

    pq_payload = None
    if save_pq:
        log("\n[5.5/5] Building PQ lexical memory ...")
        t_pq = time.time()
        pq_payload = pq_build_codebook(
            emb_weight,
            subdim=pq_subdim,
            bits=pq_bits,
            iters=pq_iters,
            batch_size=pq_batch_size,
            sample_size=pq_sample_size,
        )
        pq_centroids = pq_payload["centroids"]
        pq_codes = pq_payload["codes"]
        pq_bytes = (
            pq_centroids.numel() * pq_centroids.element_size()
            + pq_codes.numel() * pq_codes.element_size()
        )
        log(
            f"  pq subdim={pq_subdim} bits={pq_bits} "
            f"centroids={pq_centroids.shape[0]}x{pq_centroids.shape[1]} "
            f"time={time.time()-t_pq:.1f}s size={pq_bytes/1024/1024:.2f} MB"
        )

    with torch.no_grad():
        test_ids = tok.encode("오늘 날씨가")
        input_ids = torch.tensor([test_ids], device=device)
        out = mdl(input_ids, output_hidden_states=True)
        h_last = out.hidden_states[-1][0, -1]
        log(f"  GPT2 hidden norm: {h_last.norm():.2f}")

    ln_f_w = mdl.transformer.ln_f.weight.data.cpu().float()
    ln_f_b = mdl.transformer.ln_f.bias.data.cpu().float()
    log("\n[5.6/5] Fitting standalone prev-token decoder ...")
    prev_proj, pair_count = fit_decoder_prev_proj(mdl, tok, emb_weight, device)
    log(f"  prev-pairs={pair_count}  prev-proj={tuple(prev_proj.shape)}")
    log("\n[5.7/5] Fitting standalone state decoder ...")
    state_proj, state_count = fit_decoder_state_proj(mdl, tok, emb_weight, device)
    log(f"  state-pairs={state_count}  state-proj={tuple(state_proj.shape)}")

    clone_state = None
    tokenizer_json = None
    tokenizer_specials = None
    if save_clone:
        log("\n[5.8/5] Copying standalone model + tokenizer ...")
        t_clone = time.time()
        clone_state = extract_clone_state(mdl)
        tokenizer_json = tok._tokenizer.to_str()
        tokenizer_specials = {
            "bos_token": tok.bos_token,
            "eos_token": tok.eos_token,
            "unk_token": tok.unk_token,
            "sep_token": tok.sep_token,
            "pad_token": tok.pad_token,
            "cls_token": tok.cls_token,
            "mask_token": tok.mask_token,
        }
        clone_bytes = sum(v.numel() * v.element_size() for v in clone_state.values())
        log(f"  clone params: {clone_bytes/1024/1024:.2f} MB  tokenizer_json={len(tokenizer_json)/1024:.1f} KB  time={time.time()-t_clone:.1f}s")

    engine = {
        "artifact_version": 2,
        "model_name": model_name,
        "d": d,
        "vocab": vocab,
        "n_layer": n_layer,
        "tau": tau,
        "portal": PORTAL,
        "bypass": BYPASS,
        "t_wake": T_WAKE,
        "r_c": R_C,
        "W": W_cond.cpu(),
        "W_values": w_values.cpu(),
        "W_col_idx": w_col_idx.cpu(),
        "W_row_ptr": w_row_ptr.cpu(),
        "W_eigvecs": w_eigvecs.cpu(),
        "W_layers": W_layers,
        "sparse": sparse,
        "emb_weight": None if pq_only else emb_weight,
        "ln_f_weight": ln_f_w,
        "ln_f_bias": ln_f_b,
        "hidden_norm_ref": h_last.norm().item(),
        "decoder_prev_scale": float(decoder_prev_scale),
        "decoder_prev_proj": prev_proj,
        "decoder_state_proj": state_proj,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
    }
    if pq_payload is not None:
        engine["pq_centroids"] = pq_payload["centroids"]
        engine["pq_codes"] = pq_payload["codes"]
        engine["pq_subdim"] = int(pq_payload["subdim"])
        engine["pq_bits"] = int(pq_payload["bits"])
    if save_clone and clone_state is not None:
        engine["clone_config"] = mdl.config.to_dict()
        engine["clone_state"] = clone_state
        engine["tokenizer_json"] = tokenizer_json
        engine["tokenizer_specials"] = tokenizer_specials

    torch.save(engine, out_path)
    file_size = os.path.getsize(out_path)
    gpt2_size = sum(p.numel() * p.element_size() for p in mdl.parameters())

    w_sum_kb = d * d * 4 / 1024
    w_layers_kb = n_layer * d * d * 4 / 1024
    emb_kb = 0.0 if pq_only else emb_weight.numel() * 4 / 1024
    pq_kb = 0.0
    if pq_payload is not None:
        pq_kb = (
            pq_payload["centroids"].numel() * pq_payload["centroids"].element_size()
            + pq_payload["codes"].numel() * pq_payload["codes"].element_size()
        ) / 1024
    ln_kb = (ln_f_w.numel() + ln_f_b.numel()) * 4 / 1024
    prev_proj_kb = prev_proj.numel() * prev_proj.element_size() / 1024
    state_proj_kb = state_proj.numel() * state_proj.element_size() / 1024
    clone_kb = 0.0
    if clone_state is not None:
        clone_kb = sum(v.numel() * v.element_size() for v in clone_state.values()) / 1024
    core_kb = w_sum_kb + w_layers_kb + ln_kb + prev_proj_kb + state_proj_kb

    log(f"\n=== Conversion Complete ===")
    log(f"  Output:   {out_path}")
    log(f"  File:     {file_size / 1024 / 1024:.2f} MB")
    log(f"  GPT2:     {gpt2_size / 1024 / 1024:.2f} MB")
    log(f"  Ratio:    {file_size / gpt2_size * 100:.1f}%")
    log(f"")
    log(f"  W_sum:      {w_sum_kb/1024:.2f} MB")
    log(f"  W_layers:   {w_layers_kb/1024:.2f} MB ({n_layer} layers)")
    log(f"  ln_f:       {ln_kb:.1f} KB")
    log(f"  Embedding:  {emb_kb/1024:.2f} MB")
    if pq_payload is not None:
        log(f"  PQ memory:  {pq_kb/1024:.2f} MB")
    log(f"  Prev proj:  {prev_proj_kb/1024:.2f} MB")
    log(f"  State proj: {state_proj_kb/1024:.2f} MB")
    if clone_state is not None:
        log(f"  Clone copy: {clone_kb/1024:.2f} MB")
    log(f"  Core total: {core_kb/1024:.2f} MB")
    log(f"  Full total: {(core_kb+emb_kb+pq_kb+clone_kb)/1024:.2f} MB")
    log(f"  vs GPT2:    {(core_kb+emb_kb+pq_kb+clone_kb)*1024/gpt2_size*100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Convert transformer to CE engine")
    ap.add_argument("--model", default="skt/kogpt2-base-v2")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--sparse", action="store_true", help="Apply 3D sparsification")
    ap.add_argument("--save-pq", action="store_true", help="Save product-quantized lexical memory")
    ap.add_argument("--pq-only", action="store_true", help="Drop full embedding table when PQ is saved")
    ap.add_argument("--pq-subdim", type=int, default=64)
    ap.add_argument("--pq-bits", type=int, default=8)
    ap.add_argument("--pq-iters", type=int, default=16)
    ap.add_argument("--pq-batch-size", type=int, default=4096)
    ap.add_argument("--pq-sample-size", type=int, default=16384)
    ap.add_argument("--decoder-prev-scale", type=float, default=0.35)
    ap.add_argument("--save-clone", action="store_true", help="Copy full model weights and tokenizer into artifact")
    args = ap.parse_args()

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        args.output = os.path.join(os.path.dirname(__file__), f"{safe_name}.ce.pt")

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    convert(
        args.model,
        args.output,
        args.device,
        sparse=args.sparse,
        save_pq=args.save_pq,
        pq_only=args.pq_only,
        pq_subdim=args.pq_subdim,
        pq_bits=args.pq_bits,
        pq_iters=args.pq_iters,
        pq_batch_size=args.pq_batch_size,
        pq_sample_size=args.pq_sample_size,
        decoder_prev_scale=args.decoder_prev_scale,
        save_clone=args.save_clone,
    )


if __name__ == "__main__":
    main()
