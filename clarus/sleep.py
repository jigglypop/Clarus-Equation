"""Wake/NREM/REM refinement for standalone CE artifacts."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from .engine import CEEngine, DEFAULT_PROMPTS
    from .ce_ops import pq_build_codebook
except ImportError:
    from clarus.engine import CEEngine, DEFAULT_PROMPTS
    from clarus.ce_ops import pq_build_codebook


@dataclass
class SleepBatch:
    state_x: torch.Tensor
    prev_x: torch.Tensor
    target_y: torch.Tensor
    soft_y: torch.Tensor
    hard_mask: torch.Tensor
    top1_hits: torch.Tensor
    top50_hits: torch.Tensor
    target_ids: torch.Tensor
    teacher_top_ids: torch.Tensor | None = None
    teacher_top_probs: torch.Tensor | None = None


@dataclass
class DecoderTokenHead:
    token_ids: torch.Tensor
    state_proj: torch.Tensor | None
    prev_proj: torch.Tensor | None
    bias: torch.Tensor | None
    scale: float = 1.0


def safe_print(text):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(str(text).encode("utf-8", errors="replace").decode("utf-8"), flush=True)


def ridge_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x.float()
    y = y.float()
    if weights is not None:
        w = weights.float().clamp_min(1e-6).sqrt().unsqueeze(1)
        x = x * w
        y = y * w
    xtx = x.T @ x
    xty = x.T @ y
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
    return torch.linalg.solve(xtx + float(ridge) * eye, xty)


def batch_weights(batch: SleepBatch, rem_weight: float) -> torch.Tensor:
    weights = torch.ones(
        batch.target_y.shape[0],
        dtype=batch.target_y.dtype,
        device=batch.target_y.device,
    )
    if batch.hard_mask.numel() and rem_weight > 1.0:
        weights[batch.hard_mask.bool()] = float(rem_weight)
    return weights


def fit_decoder_from_batch(
    batch: SleepBatch,
    *,
    prev_scale: float,
    ridge: float,
    rem_weight: float = 1.0,
    rem_mix: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = batch.target_y.clone()
    weights = batch_weights(batch, rem_weight)

    if batch.hard_mask.numel() and rem_mix > 0.0:
        hard = batch.hard_mask.bool()
        y[hard] = (1.0 - rem_mix) * y[hard] + rem_mix * batch.soft_y[hard]

    state_proj = ridge_solve(batch.state_x, y, ridge=ridge, weights=weights)
    residual = y - batch.state_x @ state_proj
    prev_target = residual / max(float(prev_scale), 1e-6)
    prev_proj = ridge_solve(batch.prev_x, prev_target, ridge=ridge, weights=weights)
    return state_proj, prev_proj


def fit_token_head_from_batch(
    batch: SleepBatch,
    *,
    prev_scale: float,
    ridge: float,
    rem_weight: float = 1.0,
    max_vocab: int = 2048,
    scale: float = 1.0,
) -> DecoderTokenHead | None:
    if batch.teacher_top_ids is None or batch.teacher_top_probs is None:
        return None

    top_ids = batch.teacher_top_ids.long()
    top_probs = batch.teacher_top_probs.float()
    if top_ids.numel() == 0 or top_probs.numel() == 0:
        return None

    flat_ids = top_ids.reshape(-1)
    flat_probs = top_probs.reshape(-1)
    uniq_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    mass = torch.zeros(uniq_ids.shape[0], dtype=flat_probs.dtype, device=flat_probs.device)
    mass.scatter_add_(0, inverse, flat_probs)

    if max_vocab > 0 and uniq_ids.numel() > max_vocab:
        keep = torch.topk(mass, max_vocab).indices
        uniq_ids = uniq_ids.index_select(0, keep)
        uniq_ids, _ = torch.sort(uniq_ids)

    if uniq_ids.numel() == 0:
        return None

    token_map = {int(token_id): col for col, token_id in enumerate(uniq_ids.tolist())}
    y = torch.zeros(
        (top_ids.shape[0], uniq_ids.shape[0]),
        dtype=torch.float32,
        device=batch.state_x.device,
    )
    for row_idx, (row_ids, row_probs) in enumerate(zip(top_ids.tolist(), top_probs.tolist(), strict=False)):
        for token_id, prob in zip(row_ids, row_probs, strict=False):
            col_idx = token_map.get(int(token_id))
            if col_idx is not None:
                y[row_idx, col_idx] += float(prob)

    weights = batch_weights(batch, rem_weight).float()
    denom = weights.sum().clamp_min(1e-6)
    bias = (weights.unsqueeze(1) * y).sum(dim=0) / denom
    y_center = y - bias
    state_proj = ridge_solve(batch.state_x, y_center, ridge=ridge, weights=weights)
    residual = y_center - batch.state_x @ state_proj
    prev_target = residual / max(float(prev_scale), 1e-6)
    prev_proj = ridge_solve(batch.prev_x, prev_target, ridge=ridge, weights=weights)
    return DecoderTokenHead(
        token_ids=uniq_ids.long().cpu(),
        state_proj=state_proj.cpu(),
        prev_proj=prev_proj.cpu(),
        bias=bias.cpu(),
        scale=float(scale),
    )


def collect_sleep_batch(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    teacher_topk: int,
) -> SleepBatch:
    state_rows: list[torch.Tensor] = []
    prev_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    soft_rows: list[torch.Tensor] = []
    hard_rows: list[bool] = []
    top1_hits: list[bool] = []
    top50_hits: list[bool] = []
    target_ids: list[int] = []
    teacher_top_ids_rows: list[torch.Tensor] = []
    teacher_top_prob_rows: list[torch.Tensor] = []

    for prompt in prompts:
        ids = eng.tok.encode(prompt, return_tensors="pt").to(eng.device)
        for _ in range(max_new_tokens):
            ctx = eng.context_from_ids(ids)
            relax_result = eng.relax_context(ctx, ce_args)
            ce_hidden = eng.ce_hidden(relax_result["m_star"]).detach()
            teacher_hidden = ctx.h_true.squeeze(0).detach()

            teacher_logits = eng.teacher_next_logits(ids)
            top_vals, top_idx = torch.topk(teacher_logits, min(teacher_topk, teacher_logits.numel()))
            probs = F.softmax(top_vals, dim=0)
            target_id = int(top_idx[0].item())
            prev_id = int(ids[0, -1].item())

            prev_emb = eng.token_embedding([prev_id]).squeeze(0).detach()
            soft_emb = (probs.unsqueeze(1) * eng.teacher_embedding(top_idx).detach()).sum(dim=0)

            standalone_logits = eng.standalone_logits(
                ce_hidden,
                prev_id,
                temperature=1.0,
            )
            stand_top1 = int(torch.argmax(standalone_logits).item())
            top50 = torch.topk(standalone_logits, min(50, standalone_logits.numel())).indices.tolist()
            hit1 = stand_top1 == target_id
            hit50 = target_id in top50

            state_rows.append(ce_hidden.cpu())
            prev_rows.append(prev_emb.cpu())
            target_rows.append(teacher_hidden.cpu())
            soft_rows.append(soft_emb.cpu())
            hard_rows.append(not hit50)
            top1_hits.append(hit1)
            top50_hits.append(hit50)
            target_ids.append(target_id)
            teacher_top_ids_rows.append(top_idx.cpu())
            teacher_top_prob_rows.append(probs.cpu())

            ids = torch.cat([ids, torch.tensor([[target_id]], device=eng.device)], dim=1)

    return SleepBatch(
        state_x=torch.stack(state_rows, dim=0),
        prev_x=torch.stack(prev_rows, dim=0),
        target_y=torch.stack(target_rows, dim=0),
        soft_y=torch.stack(soft_rows, dim=0),
        hard_mask=torch.tensor(hard_rows, dtype=torch.bool),
        top1_hits=torch.tensor(top1_hits, dtype=torch.bool),
        top50_hits=torch.tensor(top50_hits, dtype=torch.bool),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        teacher_top_ids=torch.stack(teacher_top_ids_rows, dim=0),
        teacher_top_probs=torch.stack(teacher_top_prob_rows, dim=0),
    )


def batch_stats(batch: SleepBatch) -> dict[str, float]:
    top1 = batch.top1_hits.float().mean().item() if batch.top1_hits.numel() else 0.0
    top50 = batch.top50_hits.float().mean().item() if batch.top50_hits.numel() else 0.0
    hard = batch.hard_mask.float().mean().item() if batch.hard_mask.numel() else 0.0
    return {
        "top1_acc": top1,
        "top50_acc": top50,
        "hard_ratio": hard,
        "samples": int(batch.state_x.shape[0]),
    }


def maybe_refresh_pq(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    subdim: int,
    bits: int,
    iters: int,
    batch_size: int,
    sample_size: int,
):
    if eng.data.get("clone_state") is None:
        return None

    emb = eng.model.transformer.wte.weight.detach().cpu().float()
    freq = torch.bincount(batch.target_ids.cpu(), minlength=emb.shape[0]).float()
    hot = torch.nonzero(freq > 0, as_tuple=False).squeeze(1)
    if hot.numel() == 0:
        return None

    hot_emb = emb.index_select(0, hot)
    pool = torch.cat([emb, hot_emb, hot_emb], dim=0)
    pq = pq_build_codebook(
        pool,
        subdim=subdim,
        bits=bits,
        iters=iters,
        batch_size=batch_size,
        sample_size=min(sample_size, pool.shape[0]),
        seed=0,
    )
    centroids = pq["centroids"].cpu()

    codes = torch.empty((emb.shape[0], centroids.shape[0]), dtype=torch.uint8)
    for sub_idx in range(centroids.shape[0]):
        start = sub_idx * subdim
        stop = start + subdim
        dist = torch.cdist(emb[:, start:stop], centroids[sub_idx].float())
        codes[:, sub_idx] = dist.argmin(dim=1).to(torch.uint8)

    eng.pq_centroids = centroids.to(eng.device)
    eng.pq_codes = codes.to(eng.device)
    eng.data["pq_centroids"] = centroids
    eng.data["pq_codes"] = codes
    return {
        "pq_centroids_mb": centroids.numel() * centroids.element_size() / 1024 / 1024,
        "pq_codes_mb": codes.numel() * codes.element_size() / 1024 / 1024,
    }


def run_sleep_cycle(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    teacher_topk: int,
    ridge: float,
    rem_weight: float,
    rem_mix: float,
    token_head_max_vocab: int,
    token_head_scale: float,
    refresh_pq: bool,
    pq_subdim: int,
    pq_bits: int,
    pq_iters: int,
    pq_batch_size: int,
    pq_sample_size: int,
) -> dict[str, object]:
    wake = collect_sleep_batch(
        eng,
        prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
    )
    wake_stats = batch_stats(wake)

    state_nrem, prev_nrem = fit_decoder_from_batch(
        wake,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
    )
    eng.apply_decoder_refine(prev_nrem.cpu(), state_nrem.cpu())
    token_nrem = fit_token_head_from_batch(
        wake,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        max_vocab=token_head_max_vocab,
        scale=token_head_scale,
    )
    if token_nrem is not None:
        eng.apply_token_head(
            token_nrem.token_ids,
            state_proj=token_nrem.state_proj,
            prev_proj=token_nrem.prev_proj,
            bias=token_nrem.bias,
            scale=token_nrem.scale,
        )

    nrem = collect_sleep_batch(
        eng,
        prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
    )
    nrem_stats = batch_stats(nrem)

    state_rem, prev_rem = fit_decoder_from_batch(
        nrem,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        rem_weight=rem_weight,
        rem_mix=rem_mix,
    )
    eng.apply_decoder_refine(prev_rem.cpu(), state_rem.cpu())
    token_rem = fit_token_head_from_batch(
        nrem,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        rem_weight=rem_weight,
        max_vocab=token_head_max_vocab,
        scale=token_head_scale,
    )
    if token_rem is not None:
        eng.apply_token_head(
            token_rem.token_ids,
            state_proj=token_rem.state_proj,
            prev_proj=token_rem.prev_proj,
            bias=token_rem.bias,
            scale=token_rem.scale,
        )

    pq_stats = None
    if refresh_pq:
        pq_stats = maybe_refresh_pq(
            eng,
            nrem,
            subdim=pq_subdim,
            bits=pq_bits,
            iters=pq_iters,
            batch_size=pq_batch_size,
            sample_size=pq_sample_size,
        )

    rem = collect_sleep_batch(
        eng,
        prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
    )
    rem_stats = batch_stats(rem)

    return {
        "wake": wake_stats,
        "nrem": nrem_stats,
        "rem": rem_stats,
        "pq": pq_stats,
        "token_head_vocab": 0 if token_rem is None else int(token_rem.token_ids.numel()),
    }


def build_prompts(args) -> list[str]:
    prompts = list(args.prompts) if args.prompts else list(DEFAULT_PROMPTS)
    if args.prompt:
        prompts = [args.prompt] + prompts
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt and prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def main():
    ap = argparse.ArgumentParser(description="Sleep refinement for standalone CE artifacts")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--teacher-topk", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--rem-weight", type=float, default=2.5)
    ap.add_argument("--rem-mix", type=float, default=0.35)
    ap.add_argument("--token-head-max-vocab", type=int, default=2048)
    ap.add_argument("--token-head-scale", type=float, default=1.0)
    ap.add_argument("--refresh-pq", action="store_true")
    ap.add_argument("--pq-subdim", type=int, default=64)
    ap.add_argument("--pq-bits", type=int, default=8)
    ap.add_argument("--pq-iters", type=int, default=8)
    ap.add_argument("--pq-batch-size", type=int, default=4096)
    ap.add_argument("--pq-sample-size", type=int, default=16384)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cb-topk", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--backend", default="torch", choices=["auto", "torch", "rust", "cuda"])
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--lambda-phi", dest="lambda_phi", type=float, default=0.5)
    ap.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    eng = CEEngine(args.engine, device=args.device, backend=args.backend)
    prompts = build_prompts(args)

    ce_args = argparse.Namespace(
        dt=args.dt,
        cb_weight=None,
        cb_topk=args.cb_topk,
        beta=args.beta,
        steps=args.steps,
        backend=args.backend,
        metric_rank=args.metric_rank,
        lambda0=args.lambda0,
        lambda_phi=args.lambda_phi,
        lambda_var=args.lambda_var,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    safe_print("=== CE Sleep Refinement ===")
    safe_print(f"  engine={args.engine}")
    safe_print(f"  prompts={len(prompts)}  cycles={args.cycles}  tokens={args.tokens}")
    safe_print(f"  model_source={eng.model_source}")

    reports = []
    for cycle in range(1, args.cycles + 1):
        report = run_sleep_cycle(
            eng,
            prompts,
            ce_args,
            max_new_tokens=args.tokens,
            teacher_topk=args.teacher_topk,
            ridge=args.ridge,
            rem_weight=args.rem_weight,
            rem_mix=args.rem_mix,
            token_head_max_vocab=args.token_head_max_vocab,
            token_head_scale=args.token_head_scale,
            refresh_pq=args.refresh_pq,
            pq_subdim=args.pq_subdim,
            pq_bits=args.pq_bits,
            pq_iters=args.pq_iters,
            pq_batch_size=args.pq_batch_size,
            pq_sample_size=args.pq_sample_size,
        )
        reports.append(report)
        safe_print(
            f"  cycle {cycle}: "
            f"wake top50={report['wake']['top50_acc']:.3f} -> "
            f"nrem {report['nrem']['top50_acc']:.3f} -> "
            f"rem {report['rem']['top50_acc']:.3f}  "
            f"token_vocab={report['token_head_vocab']}"
        )

    out_path = args.output or args.engine
    eng.save_artifact(out_path)
    result_path = os.path.join(os.path.dirname(out_path), "sleep_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "engine": out_path,
                "prompts": prompts,
                "cycles": args.cycles,
                "reports": reports,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    safe_print(f"  saved_engine={out_path}")
    safe_print(f"  saved_report={result_path}")


if __name__ == "__main__":
    main()
