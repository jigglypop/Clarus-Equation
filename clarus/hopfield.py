"""CE-AGI Hopfield runtime.

Q,K only         -> Q,K + V,O + FFN (full model weights)
tau=1            -> tau=1/|eig_min(H_E)| (physical timescale)
phi=0            -> phi from embedding variance (bypass activates)
no codebook      -> Modern Hopfield codebook (3.4)
direction inject -> layer injection decode
single prompt    -> multi-prompt benchmark
dt=0.005 fixed   -> adaptive dt from spectrum

12_Equation.md sections: 1.3, 1.5, 3.1, 3.4, 4.2, 4.3, 14.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import e, pi

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .ce_ops import (
        build_metric_basis as ce_build_metric_basis,
        pack_sparse as ce_pack_sparse,
        relax_packed as ce_relax_packed,
    )
except ImportError:
    from clarus.ce_ops import (
        build_metric_basis as ce_build_metric_basis,
        pack_sparse as ce_pack_sparse,
        relax_packed as ce_relax_packed,
    )

os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ad = 4 / (e ** (4 / 3) * pi ** (4 / 3))
PORTAL = (_ad * (1 - _ad)) ** 2
BYPASS = 1 / (e ** (1 / 3) * pi ** (1 / 3))
T_WAKE = 1 / (3 + _ad * (1 - _ad))
R_C = pi


def safe_print(text):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        data = (str(text) + "\n").encode("utf-8", errors="replace")
        sys.stdout.buffer.write(data)
        sys.stdout.flush()


def block_hidden(block, h):
    out = block(h)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def load_model(name, device):
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    safe_print(f"Loading {name} ...")
    tok = PreTrainedTokenizerFast.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    mdl.to(device)
    mdl.eval()
    cfg = mdl.config
    d = cfg.n_embd if hasattr(cfg, "n_embd") else cfg.hidden_size
    n_layer = cfg.n_layer if hasattr(cfg, "n_layer") else cfg.num_hidden_layers
    n_head = cfg.n_head if hasattr(cfg, "n_head") else cfg.num_attention_heads
    d_head = d // n_head
    total_p = sum(p.numel() for p in mdl.parameters())
    safe_print(f"  d={d}  layers={n_layer}  heads={n_head}  d_head={d_head}")
    safe_print(f"  vocab={cfg.vocab_size}  params={total_p/1e6:.1f}M  device={device}")
    return mdl, tok, d, n_layer, n_head, d_head


def extract_hopfield_full(mdl, d, n_layer, d_head):
    """Attention(Q,K) + Attention(V,O) + FFN -> symmetric Hopfield matrix."""
    device = next(mdl.parameters()).device
    W = torch.zeros(d, d, device=device)
    norms = {"qk": 0.0, "vo": 0.0, "ffn": 0.0}

    for layer in mdl.transformer.h:
        w_attn = layer.attn.c_attn.weight.data
        Wq, Wk, Wv = w_attn[:, :d], w_attn[:, d : 2 * d], w_attn[:, 2 * d :]
        Wo = layer.attn.c_proj.weight.data

        qk = Wq @ Wk.T
        W_qk = (qk + qk.T) / (2 * d_head ** 0.5)
        W += W_qk
        norms["qk"] += W_qk.norm().item()

        vo = Wv @ Wo
        W_vo = (vo + vo.T) / 2
        W += W_vo
        norms["vo"] += W_vo.norm().item()

        W_up = layer.mlp.c_fc.weight.data
        W_down = layer.mlp.c_proj.weight.data
        Wf = W_up @ W_down
        W_ffn = (Wf + Wf.T) / 4
        W += W_ffn
        norms["ffn"] += W_ffn.norm().item()

    W /= n_layer
    W = (W + W.T) / 2

    for k, v in norms.items():
        safe_print(f"  ||W_{k}|| avg = {v/n_layer:.2f}")
    return W


def build_lattice(N):
    s = int(np.ceil(N ** (1 / 3)))
    coords = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        coords[i] = [i // (s * s), (i // s) % s, i % s]
    return coords, s


def sparsify(W, N, r_c=R_C):
    coords, side = build_lattice(N)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = (diff**2).sum(axis=-1)
    mask = torch.from_numpy((dist_sq < r_c**2).astype(np.float32)).to(W.device)
    mask.fill_diagonal_(0)
    W_sp = W * mask
    nnz = int(mask.sum().item())
    K_avg = nnz / N
    safe_print(f"  lattice {side}^3, r_c=pi")
    safe_print(f"  K_avg={K_avg:.1f}  density={nnz/(N*(N-1))*100:.1f}%")
    return W_sp, mask, nnz


def make_negative_definite(W):
    """Spectral shift: W -> W - (lambda_max + margin)*I.
    margin = 0.1 * |lambda_min| to keep tau in a reasonable range."""
    eigvals, V = torch.linalg.eigh(W)
    lam_min = eigvals[0].item()
    lam_max = eigvals[-1].item()
    margin = 0.1 * abs(lam_min) if abs(lam_min) > 1e-6 else 0.1
    if lam_max <= -margin:
        safe_print(f"  already negative-definite, eig: [{lam_min:.4f}, {lam_max:.4f}]")
        return W
    shift = lam_max + margin
    eigvals_shifted = eigvals - shift
    W_safe = V @ torch.diag(eigvals_shifted) @ V.T
    W_safe = (W_safe + W_safe.T) / 2
    safe_print(f"  spectral shift: {shift:.4f} (margin={margin:.4f})")
    safe_print(f"  eig: [{eigvals_shifted[0]:.4f}, {eigvals_shifted[-1]:.4f}]")
    return W_safe


def build_codebook(mdl, m_ref, top_k=1024, verbose=True):
    emb = mdl.transformer.wte.weight.data
    scores = emb @ m_ref
    _, idx = scores.topk(min(top_k, emb.shape[0]))
    cb = emb[idx].clone()
    if verbose:
        safe_print(f"  codebook: {cb.shape[0]} patterns, avg norm={cb.norm(dim=1).mean().item():.2f}")
    return cb


@torch.no_grad()
def codebook_grad(m, codebook, beta):
    logits = beta * (codebook @ m)
    w = F.softmax(logits, dim=0)
    return -(w @ codebook)


def energy_full(m, W, b, phi, codebook=None, beta=1.0, cb_w=0.5, bypass_c=0.0):
    e_hop = -0.5 * (m @ W @ m)
    e_bias = -(m @ b)
    phi_hat = F.normalize(phi, dim=0)
    e_portal = -PORTAL * (m @ phi_hat)
    e_bypass = -BYPASS * bypass_c * (m @ phi_hat)
    e_cb = torch.tensor(0.0, device=m.device)
    if codebook is not None:
        logits = beta * (codebook @ m)
        e_cb = -(cb_w / beta) * torch.logsumexp(logits, dim=0)
    return e_hop + e_bias + e_portal + e_cb + e_bypass, (e_hop, e_bias, e_portal, e_cb)


@torch.no_grad()
def relax(
    W,
    b,
    phi,
    m0,
    codebook=None,
    beta=1.0,
    cb_w=0.5,
    tau=1.0,
    dt=0.01,
    max_steps=500,
    tol=1e-4,
    backend="auto",
    metric_rank=8,
    lambda0=1.0,
    lambda_phi=0.5,
    lambda_var=0.25,
    noise_scale=1.0,
    seed=0,
    w_eigvecs=None,
    dense_w=None,
):
    if isinstance(W, tuple):
        values, col_idx, row_ptr = W
    else:
        values, col_idx, row_ptr = ce_pack_sparse(W, backend=backend)

    if codebook is None:
        codebook = m0.new_empty((0, m0.numel()))
    metric_basis = ce_build_metric_basis(
        codebook, m0, metric_rank, w_eigvecs=w_eigvecs, backend=backend,
    )

    return ce_relax_packed(
        values,
        col_idx,
        row_ptr,
        b,
        phi,
        m0,
        codebook,
        metric_basis,
        portal=PORTAL,
        bypass=BYPASS,
        t_wake=T_WAKE,
        beta=beta,
        cb_w=cb_w,
        lambda0=lambda0,
        lambda_phi=lambda_phi,
        lambda_var=lambda_var,
        tau=tau,
        dt=dt,
        max_steps=max_steps,
        tol=tol,
        noise_scale=noise_scale,
        metric_rank=metric_rank,
        backend=backend,
        seed=seed,
        dense_w=dense_w,
    )


def update_phi(phi, m_star, phi_var=None):
    alpha = BYPASS
    if phi_var is None:
        v = m_star.detach().pow(2).clamp(max=1.0)
    else:
        v = phi_var.detach().clamp(min=0.0)
    return (1 - alpha) * phi + alpha * v


def get_initial_state(mdl, prompt_ids, init_layer=0):
    with torch.no_grad():
        seq_len = prompt_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=prompt_ids.device).unsqueeze(0)
        emb = mdl.transformer.wte(prompt_ids) + mdl.transformer.wpe(pos_ids)
        h = emb
        for i in range(init_layer + 1):
            h = block_hidden(mdl.transformer.h[i], h)
        m0 = h[:, -1, :].squeeze(0)
        phi = emb.squeeze(0).var(dim=0)
    return m0, phi


def decode_direct(m_star, mdl, tok, prompt_ids, max_tok=40, temperature=0.8):
    with torch.no_grad():
        h = mdl.transformer.ln_f(m_star.unsqueeze(0).unsqueeze(0))
        logits = mdl.lm_head(h).squeeze() / temperature
        probs = F.softmax(logits, dim=-1)
        first_tok = torch.multinomial(probs, 1).item()

    if first_tok == tok.eos_token_id or max_tok <= 1:
        return tok.decode(
            [first_tok] if first_tok != tok.eos_token_id else [],
            skip_special_tokens=True,
        )

    gen_ids = torch.cat(
        [prompt_ids, torch.tensor([[first_tok]], device=prompt_ids.device)],
        dim=1,
    )
    attention_mask = torch.ones_like(gen_ids)
    pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    with torch.no_grad():
        out = mdl.generate(
            gen_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tok - 1,
            do_sample=True,
            temperature=temperature,
            top_k=40,
            pad_token_id=pad_token_id,
        )
    return tok.decode(out[0, prompt_ids.shape[1] :].tolist(), skip_special_tokens=True)


def decode_inject(mdl, tok, prompt_ids, m_star, inject_layer=None, ce_strength=0.3, max_tok=40, temperature=0.8):
    n_layer = len(mdl.transformer.h)
    if inject_layer is None:
        inject_layer = n_layer // 2
    gen_ids = prompt_ids.clone()
    out_ids = []

    for _ in range(max_tok):
        with torch.no_grad():
            seq_len = gen_ids.shape[1]
            pos_ids = torch.arange(seq_len, device=gen_ids.device).unsqueeze(0)
            h = mdl.transformer.wte(gen_ids) + mdl.transformer.wpe(pos_ids)

            for i in range(inject_layer):
                h = block_hidden(mdl.transformer.h[i], h)

            h_last = h[:, -1:, :]
            m_proj = m_star.view(1, 1, -1)
            m_proj = m_proj * (h_last.norm() / (m_proj.norm() + 1e-8))
            h[:, -1:, :] = (1 - ce_strength) * h_last + ce_strength * m_proj

            for i in range(inject_layer, n_layer):
                h = block_hidden(mdl.transformer.h[i], h)

            h = mdl.transformer.ln_f(h)
            logits = mdl.lm_head(h[:, -1, :]) / temperature

        probs = F.softmax(logits, dim=-1)
        t = torch.multinomial(probs, 1).item()
        if t == tok.eos_token_id:
            break
        out_ids.append(t)
        gen_ids = torch.cat([gen_ids, torch.tensor([[t]], device=gen_ids.device)], dim=1)

    return tok.decode(out_ids, skip_special_tokens=True)


def generate_multiround(
    mdl,
    tok,
    W,
    prompt_ids,
    n_layer,
    cb_topk=1024,
    beta=1.0,
    cb_w=0.031,
    tau=1.0,
    dt=0.05,
    relax_steps=100,
    max_tok=40,
    temperature=0.8,
    backend="auto",
    metric_rank=8,
    lambda0=1.0,
    lambda_phi=0.5,
    lambda_var=0.25,
    noise_scale=1.0,
    seed=0,
    w_eigvecs=None,
    phi_init=None,
    phi_threshold=0.0,
    sleep_decay=1.0,
    top_k=40,
    repeat_penalty=3.0,
):
    gen_ids = prompt_ids.clone()
    out_ids = []
    energies = []
    phi_norms = []
    phi_state = phi_init.clone() if phi_init is not None else None

    for step in range(max_tok):
        with torch.no_grad():
            seq_len = gen_ids.shape[1]
            pos_ids = torch.arange(seq_len, device=gen_ids.device).unsqueeze(0)
            emb = mdl.transformer.wte(gen_ids) + mdl.transformer.wpe(pos_ids)
            h = emb
            best_l = min(n_layer - 2, 9)
            for i in range(best_l + 1):
                h = block_hidden(mdl.transformer.h[i], h)
            m_context = h[:, -1, :].squeeze(0)
            if phi_state is None:
                phi_state = emb.squeeze(0).var(dim=0)

        codebook = build_codebook(mdl, m_context, top_k=cb_topk, verbose=False)
        phi_norms.append(phi_state.norm().item())

        b = m_context.clone()
        m_star, hist, _ = relax(
            W,
            b,
            phi_state,
            m_context,
            codebook,
            beta,
            cb_w,
            tau=tau,
            dt=dt,
            max_steps=relax_steps,
            backend=backend,
            metric_rank=metric_rank,
            lambda0=lambda0,
            lambda_phi=lambda_phi,
            lambda_var=lambda_var,
            noise_scale=noise_scale * max(0.3, 1.0 - step / max_tok),
            seed=seed + step,
            w_eigvecs=w_eigvecs,
        )
        energies.append(hist["E"][-1])
        phi_var = hist.get("phi_var")
        if phi_var:
            phi_tensor = m_star.new_tensor(phi_var)
            phi_state = update_phi(phi_state, m_star, phi_var=phi_tensor)
        else:
            phi_state = update_phi(phi_state, m_star)
        if sleep_decay < 1.0 and phi_threshold > 0.0 and phi_state.norm().item() > phi_threshold:
            phi_state = phi_state * sleep_decay

        with torch.no_grad():
            h_out = mdl.transformer.ln_f(m_star.unsqueeze(0).unsqueeze(0))
            logits = mdl.lm_head(h_out).squeeze() / temperature
            for prev_id in set(out_ids[-6:]):
                logits[prev_id] -= repeat_penalty
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            t = torch.multinomial(probs, 1).item()

        if t == tok.eos_token_id:
            break
        out_ids.append(t)
        gen_ids = torch.cat([gen_ids, torch.tensor([[t]], device=gen_ids.device)], dim=1)

    text = tok.decode(out_ids, skip_special_tokens=True)
    return text, energies, phi_norms


def top_token(m_star, mdl, tok):
    with torch.no_grad():
        h = mdl.transformer.ln_f(m_star.unsqueeze(0).unsqueeze(0))
        tok_id = mdl.lm_head(h).squeeze().argmax().item()
    return tok_id, tok.decode([tok_id], skip_special_tokens=True)


def standard_gen(mdl, tok, prompt, max_tok=40, temperature=0.8):
    device = next(mdl.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(
            ids,
            max_new_tokens=max_tok,
            do_sample=True,
            temperature=temperature,
            top_k=40,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0].tolist(), skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="skt/kogpt2-base-v2")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "rust", "cuda"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--ce-strength", type=float, default=0.3)
    ap.add_argument("--cb-topk", type=int, default=1024)
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--lambda-phi", dest="lambda_phi", type=float, default=0.5)
    ap.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["인공지능이란", "대한민국의 수도는", "오늘 날씨가"],
    )
    args = ap.parse_args()
    device = resolve_device(args.device)

    safe_print("=== CE Constants (12_Equation.md 1.3) ===")
    safe_print(f"  portal  = [ad(1-ad)]^2     = {PORTAL:.6f}")
    safe_print(f"  bypass  = 1/(e^1/3 pi^1/3) = {BYPASS:.6f}")
    safe_print(f"  T_wake  = [3+ad(1-ad)]^-1  = {T_WAKE:.6f}")
    safe_print(f"  backend = {args.backend}  device = {device}")

    mdl, tok, d, n_layer, n_head, d_head = load_model(args.model, device)

    safe_print("\n=== Hopfield Matrix (Q,K + V,O + FFN) ===")
    t0 = time.time()
    W_hop = extract_hopfield_full(mdl, d, n_layer, d_head)
    safe_print(f"  ||W_hop|| = {W_hop.norm():.2f}")
    eigvals_raw = torch.linalg.eigvalsh(W_hop)
    safe_print(f"  raw eig: [{eigvals_raw[0]:.2f}, {eigvals_raw[-1]:.2f}]")

    safe_print("\n=== Conditioning ===")
    if d <= 1024:
        safe_print(f"  d={d} <= 1024: skipping lattice sparsification (dense)")
        W_sp = W_hop
        nnz = d * d
    else:
        safe_print("  3D Sparsification:")
        W_sp, _, nnz = sparsify(W_hop, d)

    spec_norm = max(abs(torch.linalg.eigvalsh(W_sp)[0].item()), abs(torch.linalg.eigvalsh(W_sp)[-1].item()))
    if spec_norm > 0:
        W_sp = W_sp / spec_norm
        safe_print(f"  spectral norm = {spec_norm:.2f}")

    W_sp = make_negative_definite(W_sp)
    eigvals_final, eigvecs_final = torch.linalg.eigh(W_sp)

    hess_rank = min(args.metric_rank // 2, 8)
    w_eigvecs = eigvecs_final[:, :hess_rank].T.contiguous()
    safe_print(f"  Hessian eigvecs for metric: {hess_rank} directions")

    eig_H_min = abs(eigvals_final[-1].item())
    tau = 1.0 / eig_H_min if eig_H_min > 1e-8 else 1.0
    safe_print(f"  tau = 1/|eig_H_min| = {tau:.4f}")

    dt_eff = min(args.dt, 0.9 * tau)
    safe_print(f"  dt = {dt_eff:.4f}")

    dense_kb = d * d * 4 / 1024
    sparse_kb = nnz * 4 / 1024
    safe_print(f"  mem: {dense_kb:.0f}KB -> {sparse_kb:.0f}KB ({sparse_kb/dense_kb*100:.1f}%)")
    safe_print(f"  build time: {time.time()-t0:.1f}s")
    pack_backend = "rust" if device.type == "cpu" and args.backend in ("auto", "rust") else "torch"
    W_pack = ce_pack_sparse(W_sp, backend=pack_backend)

    beta = 1.0
    cb_w = PORTAL
    inject_layer = n_layer // 2
    results = []

    for prompt in args.prompts:
        safe_print(f'\n{"="*60}')
        safe_print(f'Prompt: "{prompt}"')
        safe_print(f'{"="*60}')

        prompt_ids = tok.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            h_true = mdl.transformer(prompt_ids).last_hidden_state[:, -1, :]

        safe_print("\n  -- Init layer sweep --")
        layer_results = {}
        for init_l in [0, n_layer // 4, n_layer // 2, 3 * n_layer // 4, n_layer - 1]:
            m0_l, phi_l = get_initial_state(mdl, prompt_ids, init_layer=init_l)
            cos_l = F.cosine_similarity(m0_l.unsqueeze(0), h_true).item()
            layer_results[init_l] = (m0_l, phi_l, cos_l)
            safe_print(f"    layer {init_l:2d}: ||m0||={m0_l.norm():.2f}  cos(m0,h_true)={cos_l:.4f}")

        best_layer = max(layer_results, key=lambda l: layer_results[l][2])
        m0, phi, cos_m0_best = layer_results[best_layer]
        safe_print(f"    -> best init: layer {best_layer} (cos={cos_m0_best:.4f})")

        codebook = build_codebook(mdl, m0, top_k=args.cb_topk)

        t0 = time.time()
        m_star, hist, n_steps = relax(
            W_pack,
            m0,
            phi,
            m0,
            codebook,
            beta,
            cb_w,
            tau=tau,
            dt=dt_eff,
            max_steps=args.steps,
            backend=args.backend,
            metric_rank=args.metric_rank,
            lambda0=args.lambda0,
            lambda_phi=args.lambda_phi,
            lambda_var=args.lambda_var,
            noise_scale=args.noise_scale,
            seed=args.seed,
            w_eigvecs=w_eigvecs,
        )
        dt_relax = time.time() - t0

        cos_ms = F.cosine_similarity(m_star.unsqueeze(0), h_true).item()

        safe_print(f"\n  -- Relaxation (from layer {best_layer}) --")
        safe_print(f"  steps={n_steps}  time={dt_relax:.2f}s")
        safe_print(f'  E: {hist["E"][0]:.4f} -> {hist["E"][-1]:.4f} (best={min(hist["E"]):.4f})')
        safe_print(f'    E_hop:    {hist["E_hop"][-1]:.4f}')
        safe_print(f'    E_bias:   {hist["E_bias"][-1]:.4f}')
        safe_print(f'    E_portal: {hist["E_portal"][-1]:.4f}')
        safe_print(f'    E_cb:     {hist["E_cb"][-1]:.4f}')
        safe_print(f"  ||m*|| = {m_star.norm():.4f}  ||m*-m0|| = {(m_star-m0).norm():.4f}")

        max_bypass = max(hist["bypass_C"])
        safe_print(f"  max bypass C_k = {max_bypass:.6f}")

        if len(hist["delta"]) > 10:
            d_first = np.mean(hist["delta"][:10])
            d_last = np.mean(hist["delta"][-10:])
            safe_print(f"  delta: {d_first:.6f} -> {d_last:.6f}")

        safe_print(f"  cos(m0, h_true)={cos_m0_best:.4f}  cos(m*, h_true)={cos_ms:.4f}")

        safe_print("\n  -- Geometry benchmark --")
        m_riem_det, _, _ = relax(
            W_pack,
            m0,
            phi,
            m0,
            codebook,
            beta,
            cb_w,
            tau=tau,
            dt=dt_eff,
            max_steps=args.steps,
            backend=args.backend,
            metric_rank=args.metric_rank,
            lambda0=args.lambda0,
            lambda_phi=args.lambda_phi,
            lambda_var=args.lambda_var,
            noise_scale=0.0,
            seed=args.seed,
            w_eigvecs=w_eigvecs,
        )
        m_euc_det, _, _ = relax(
            W_pack,
            m0,
            phi,
            m0,
            codebook,
            beta,
            cb_w,
            tau=tau,
            dt=dt_eff,
            max_steps=args.steps,
            backend=args.backend,
            metric_rank=0,
            lambda0=1.0,
            lambda_phi=0.0,
            lambda_var=0.0,
            noise_scale=0.0,
            seed=args.seed,
        )
        cos_riem = F.cosine_similarity(m_riem_det.unsqueeze(0), h_true).item()
        cos_euc = F.cosine_similarity(m_euc_det.unsqueeze(0), h_true).item()
        top_riem_id, top_riem_tok = top_token(m_riem_det, mdl, tok)
        top_euc_id, top_euc_tok = top_token(m_euc_det, mdl, tok)
        safe_print(f"  cos(euclid)={cos_euc:.4f}  cos(riemann)={cos_riem:.4f}  gain={cos_riem-cos_euc:+.4f}")
        safe_print(f"  top1(euclid)={top_euc_id}:{top_euc_tok!r}")
        safe_print(f"  top1(riemann)={top_riem_id}:{top_riem_tok!r}")

        safe_print("\n  -- Oracle test (m0 = h_true) --")
        h_true_sq = h_true.squeeze(0)
        cb_oracle = build_codebook(mdl, h_true_sq, top_k=args.cb_topk)
        m_oracle, hist_o, _ = relax(
            W_pack,
            h_true_sq,
            phi,
            h_true_sq,
            cb_oracle,
            beta,
            cb_w,
            tau=tau,
            dt=dt_eff,
            max_steps=200,
            backend=args.backend,
            metric_rank=args.metric_rank,
            lambda0=args.lambda0,
            lambda_phi=args.lambda_phi,
            lambda_var=args.lambda_var,
            noise_scale=0.0,
            seed=args.seed,
            w_eigvecs=w_eigvecs,
        )
        cos_oracle = F.cosine_similarity(m_oracle.unsqueeze(0), h_true).item()
        drift = (m_oracle - h_true_sq).norm().item() / h_true_sq.norm().item()
        safe_print(f"  cos(m*_oracle, h_true) = {cos_oracle:.4f}")
        safe_print(f"  drift = ||m*-h_true||/||h_true|| = {drift:.4f}")
        safe_print(f'  E: {hist_o["E"][0]:.4f} -> {hist_o["E"][-1]:.4f}')

        m_oracle_euc, _, _ = relax(
            W_pack,
            h_true_sq,
            phi,
            h_true_sq,
            cb_oracle,
            beta,
            cb_w,
            tau=tau,
            dt=dt_eff,
            max_steps=200,
            backend=args.backend,
            metric_rank=0,
            lambda0=1.0,
            lambda_phi=0.0,
            lambda_var=0.0,
            noise_scale=0.0,
            seed=args.seed,
        )
        oracle_drift_euc = (m_oracle_euc - h_true_sq).norm().item() / h_true_sq.norm().item()
        safe_print(f"  oracle drift (euclid) = {oracle_drift_euc:.4f}  gain={oracle_drift_euc-drift:+.4f}")

        phi_var = hist.get("phi_var")
        if phi_var:
            phi_updated = update_phi(phi, m_star, phi_var=m_star.new_tensor(phi_var))
        else:
            phi_updated = update_phi(phi, m_star)

        safe_print("\n  -- Decode --")
        ce_direct = decode_direct(m_star, mdl, tok, prompt_ids, max_tok=args.tokens)
        safe_print(f"  [1] Direct (lm_head(ln_f(m*)) + AR):")
        safe_print(f"      {prompt}{ce_direct}")

        ce_inject = decode_inject(
            mdl,
            tok,
            prompt_ids,
            m_star,
            inject_layer=inject_layer,
            ce_strength=args.ce_strength,
            max_tok=args.tokens,
        )
        safe_print(f"  [2] Layer inject (layer {inject_layer}/{n_layer}):")
        safe_print(f"      {prompt}{ce_inject}")

        safe_print("  [3] Multi-round CE (relax per token, phi context):")
        t0_mr = time.time()
        ce_multi, mr_energies, mr_phi = generate_multiround(
            mdl,
            tok,
            W_pack,
            prompt_ids,
            n_layer,
            cb_topk=args.cb_topk,
            beta=beta,
            cb_w=cb_w,
            tau=tau,
            dt=0.05,
            relax_steps=100,
            max_tok=args.tokens,
            backend=args.backend,
            metric_rank=args.metric_rank,
            lambda0=args.lambda0,
            lambda_phi=args.lambda_phi,
            lambda_var=args.lambda_var,
            noise_scale=args.noise_scale,
            seed=args.seed,
            w_eigvecs=w_eigvecs,
        )
        dt_mr = time.time() - t0_mr
        safe_print(f"      {prompt}{ce_multi}")
        safe_print(f"      time={dt_mr:.2f}s  tokens={len(mr_energies)}")
        if mr_phi:
            safe_print(f"      phi: {mr_phi[0]:.2f} -> {mr_phi[-1]:.2f}")

        std_out = standard_gen(mdl, tok, prompt, max_tok=args.tokens)
        safe_print("  [4] Standard AR:")
        safe_print(f"      {std_out}")

        results.append(
            {
                "prompt": prompt,
                "best_init_layer": best_layer,
                "init_layer_sweep": {str(k): round(v[2], 4) for k, v in layer_results.items()},
                "n_steps": n_steps,
                "relax_time_s": round(dt_relax, 3),
                "energy_start": round(hist["E"][0], 4),
                "energy_end": round(hist["E"][-1], 4),
                "energy_best": round(min(hist["E"]), 4),
                "energy_decomp": {
                    "hopfield": round(hist["E_hop"][-1], 4),
                    "bias": round(hist["E_bias"][-1], 4),
                    "portal": round(hist["E_portal"][-1], 4),
                    "codebook": round(hist["E_cb"][-1], 4),
                },
                "max_bypass_C": round(max_bypass, 6),
                "m_star_norm": round(m_star.norm().item(), 4),
                "cos_m0_h": round(cos_m0_best, 4),
                "cos_ms_h": round(cos_ms, 4),
                "oracle_cos": round(cos_oracle, 4),
                "oracle_drift": round(drift, 4),
                "geometry_compare": {
                    "cos_euclidean": round(cos_euc, 4),
                    "cos_riemann": round(cos_riem, 4),
                    "cos_gain": round(cos_riem - cos_euc, 4),
                    "oracle_drift_euclidean": round(oracle_drift_euc, 4),
                    "oracle_drift_riemann": round(drift, 4),
                    "oracle_drift_gain": round(oracle_drift_euc - drift, 4),
                    "top1_euclidean": {"id": top_euc_id, "token": top_euc_tok},
                    "top1_riemann": {"id": top_riem_id, "token": top_riem_tok},
                },
                "phi_norm": round(phi_updated.norm().item(), 4),
                "decode_direct": prompt + ce_direct,
                "decode_inject": prompt + ce_inject,
                "decode_multiround": prompt + ce_multi,
                "multiround_time_s": round(dt_mr, 3),
                "multiround_phi_evolution": [round(x, 2) for x in mr_phi[:10]],
                "decode_standard": std_out,
                "energy_sample": [round(x, 4) for x in hist["E"][:: max(1, len(hist["E"]) // 20)]],
            }
        )

    summary = {
        "model": args.model,
        "device": str(device),
        "backend": args.backend,
        "d_model": d,
        "n_layers": n_layer,
        "tau": round(tau, 6),
        "dt": round(dt_eff, 6),
        "codebook_k": args.cb_topk,
        "metric_rank": args.metric_rank,
        "metric_params": {
            "lambda0": args.lambda0,
            "lambda_phi": args.lambda_phi,
            "lambda_var": args.lambda_var,
            "noise_scale": args.noise_scale,
        },
        "inject_layer": inject_layer,
        "ce_strength": args.ce_strength,
        "memory_dense_kb": round(dense_kb, 1),
        "memory_sparse_kb": round(sparse_kb, 1),
        "constants": {
            "portal": round(PORTAL, 6),
            "bypass": round(BYPASS, 6),
            "T_wake": round(T_WAKE, 6),
        },
        "w_hop_sources": [
            "Q@K^T (attention routing)",
            "V@O (output pathway)",
            "FFN (linearized GELU)",
        ],
        "prompts": results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "hopfield_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    safe_print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
