"""Runtime-only standalone benchmark: relax -> evaluate."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from tqdm import tqdm


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode("utf-8", errors="replace").decode("utf-8"), flush=True)


def fmt_metric(value):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def mean_metric(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not values else sum(values) / len(values)


def repeated_ngram_rate(token_ids, n):
    if len(token_ids) < n:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for idx in range(len(token_ids) - n + 1):
        ngram = tuple(int(v) for v in token_ids[idx : idx + n])
        repeats += int(ngram in seen)
        seen.add(ngram)
        total += 1
    return repeats / max(total, 1)


def repeated_token_rate(token_ids, window=8):
    if not token_ids:
        return 0.0
    repeats = 0
    for idx, token_id in enumerate(token_ids):
        start = max(0, idx - max(int(window), 1))
        repeats += int(token_id in token_ids[start:idx])
    return repeats / max(len(token_ids), 1)


def unfinished_sentence_rate(text):
    stripped = text.strip()
    if not stripped:
        return 1.0
    if stripped[-1] in ".?!…'\")]}":
        return 0.0
    if stripped.endswith(("다", "요", "죠", "네", "까", "니다")):
        return 0.0
    return 1.0


def sentence_collapse_rate(text, token_ids):
    if not token_ids:
        return 1.0
    repeat3 = repeated_ngram_rate(token_ids, 3)
    unique_ratio = len(set(token_ids)) / max(len(token_ids), 1)
    collapsed = repeat3 >= 0.20 or (unfinished_sentence_rate(text) > 0 and unique_ratio < 0.55)
    return float(collapsed)


def generation_metrics(text, token_ids):
    return {
        "repeated_token_rate": repeated_token_rate(token_ids),
        "repeated_bigram_rate": repeated_ngram_rate(token_ids, 2),
        "repeated_trigram_rate": repeated_ngram_rate(token_ids, 3),
        "unfinished_sentence_rate": unfinished_sentence_rate(text),
        "sentence_collapse_rate": sentence_collapse_rate(text, token_ids),
        "unique_token_ratio": len(set(token_ids)) / max(len(token_ids), 1),
        "generated_tokens": len(token_ids),
    }


DEFAULT_WIKI_DATASET = "lcw99/wikipedia-korean-20221001"
DEFAULT_TRAIN_CORPUS = "mixed-ko-train"
DEFAULT_EVAL_CORPUS = "mixed-ko-eval"
WIKI_ALPACA_TRAIN_CORPUS = "mixed-ko-wiki-alpaca"

KOREAN_CORPUS_MIXES = {
    WIKI_ALPACA_TRAIN_CORPUS: [
        {
            "dataset_name": DEFAULT_WIKI_DATASET,
            "dataset_split": "train[:85%]",
            "format": "wiki",
            "weight": 5.0,
        },
        {
            "dataset_name": "Bingsu/ko_alpaca_data",
            "dataset_split": "train[:90%]",
            "format": "alpaca",
            "weight": 1.0,
        },
    ],
    DEFAULT_TRAIN_CORPUS: [
        {
            "dataset_name": DEFAULT_WIKI_DATASET,
            "dataset_split": "train[:85%]",
            "format": "wiki",
            "weight": 4.0,
        },
        {
            "dataset_name": "Bingsu/ko_alpaca_data",
            "dataset_split": "train[:90%]",
            "format": "alpaca",
            "weight": 1.0,
        },
        {
            "dataset_name": "squad_kor_v1",
            "dataset_split": "train[:90%]",
            "format": "squad",
            "weight": 1.0,
        },
    ],
    DEFAULT_EVAL_CORPUS: [
        {
            "dataset_name": DEFAULT_WIKI_DATASET,
            "dataset_split": "train[85%:90%]",
            "format": "wiki",
            "weight": 2.0,
        },
        {
            "dataset_name": "Bingsu/ko_alpaca_data",
            "dataset_split": "train[90%:95%]",
            "format": "alpaca",
            "weight": 2.0,
        },
        {
            "dataset_name": "squad_kor_v1",
            "dataset_split": "validation",
            "format": "squad",
            "weight": 2.0,
        },
    ],
}


def allocate_budgets(total, weights):
    total = max(int(total), 0)
    if total == 0:
        return [0 for _ in weights]
    weights = [max(float(weight), 0.0) for weight in weights]
    weight_sum = sum(weights)
    if weight_sum <= 1e-8:
        base = total // max(len(weights), 1)
        counts = [base for _ in weights]
        for idx in range(total - base * len(weights)):
            counts[idx] += 1
        return counts
    raw = [total * weight / weight_sum for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda idx: (raw[idx] - counts[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def interleave_document_groups(groups):
    ordered = []
    max_len = max((len(group) for group in groups), default=0)
    for row_idx in range(max_len):
        for group in groups:
            if row_idx < len(group):
                ordered.append(group[row_idx])
    return ordered


def content_terms(text):
    return {match.group(0).lower() for match in re.finditer(r"[0-9A-Za-z가-힣]{2,}", text)}


def build_prompt_weights(prompts):
    weights = {}
    for prompt in prompts or []:
        for token in content_terms(prompt):
            weights[token] = weights.get(token, 0) + 1
    return weights


def topical_document_score(text, prompt_weights):
    if not prompt_weights:
        return 0.0
    tokens = content_terms(text)
    if not tokens:
        return 0.0
    overlaps = [prompt_weights[token] for token in tokens if token in prompt_weights]
    if not overlaps:
        return 0.0
    overlap_mass = float(sum(overlaps))
    coverage = float(len(overlaps)) / max(len(prompt_weights), 1)
    density = float(len(overlaps)) / max(len(tokens), 1)
    return overlap_mass + 2.0 * coverage + density


def select_topical_chunks(chunks, prompt_weights, keep_limit):
    if not chunks:
        return []
    if not prompt_weights:
        return chunks[:keep_limit]
    scored = [
        (topical_document_score(chunk, prompt_weights), idx, chunk)
        for idx, chunk in enumerate(chunks)
    ]
    positive = [item for item in scored if item[0] > 0.0]
    chosen = positive if positive else scored
    chosen.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [chunk for _, _, chunk in chosen[:keep_limit]]


def format_mixed_corpus_row(row, row_format):
    if row_format == "wiki":
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        if title and text:
            return f"제목: {title}\n본문: {text}"
        return text or title

    if row_format == "alpaca":
        instruction = str(row.get("instruction", "")).strip()
        user_input = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            return ""
        if user_input:
            return f"질문: {instruction}\n입력: {user_input}\n답변: {output}"
        return f"질문: {instruction}\n답변: {output}"

    if row_format == "squad":
        question = str(row.get("question", "")).strip()
        context = str(row.get("context", "")).strip()
        answers = row.get("answers") or {}
        answer_texts = answers.get("text") or []
        answer = str(answer_texts[0]).strip() if answer_texts else ""
        if not question or not context:
            return ""
        answer_start = 0
        answer_starts = answers.get("answer_start") or []
        if answer_starts:
            try:
                answer_start = max(int(answer_starts[0]), 0)
            except (TypeError, ValueError):
                answer_start = 0
        if answer:
            span_start = max(0, answer_start - 96)
            span_end = min(len(context), answer_start + max(len(answer), 64) + 96)
            context = context[span_start:span_end].strip()
            return f"질문: {question}\n답변: {answer}\n근거: {context}"
        return f"질문: {question}\n근거: {context}"

    return ""


def load_mixed_runtime_docs(
    mix_name,
    *,
    doc_limit,
    text_limit,
    topical_prompts=None,
):
    from datasets import load_dataset
    from clarus.sleep import _chunk_document

    sources = KOREAN_CORPUS_MIXES[mix_name]
    weights = [source.get("weight", 1.0) for source in sources]
    doc_budgets = allocate_budgets(doc_limit, weights)
    text_budgets = allocate_budgets(text_limit, weights)
    prompt_weights = build_prompt_weights(topical_prompts)
    groups = []
    for source, source_doc_limit, source_text_limit in zip(sources, doc_budgets, text_budgets, strict=False):
        if source_doc_limit <= 0 or source_text_limit <= 0:
            groups.append([])
            continue
        ds = load_dataset(
            source["dataset_name"],
            source.get("dataset_config"),
            split=source.get("dataset_split", "train"),
        )
        source_chunks = []
        total_chars = 0
        row_scan_limit = max(source_doc_limit * 24, 256) if prompt_weights else None
        scanned_rows = 0
        for row in ds:
            scanned_rows += 1
            text = format_mixed_corpus_row(row, source["format"])
            if not text:
                continue
            for chunk in _chunk_document(text):
                source_chunks.append(chunk)
                total_chars += len(chunk)
                if (
                    not prompt_weights
                    and (len(source_chunks) >= source_doc_limit or total_chars >= source_text_limit)
                ):
                    break
            if (
                not prompt_weights
                and (len(source_chunks) >= source_doc_limit or total_chars >= source_text_limit)
            ):
                break
            if prompt_weights and scanned_rows >= row_scan_limit and source_chunks:
                break
        groups.append(select_topical_chunks(source_chunks, prompt_weights, source_doc_limit))
    return interleave_document_groups(groups)[: max(int(doc_limit), 1)]


def load_runtime_docs(
    *,
    data_path=None,
    dataset_name=None,
    dataset_config=None,
    dataset_split="train",
    text_column="text",
    doc_limit=64,
    text_limit=200000,
    fallback_docs=None,
    topical_prompts=None,
):
    from clarus.sleep import load_corpus_documents

    if data_path:
        return load_corpus_documents(
            data_path,
            dataset_name=None,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            text_column=text_column,
            doc_limit=doc_limit,
            text_limit=text_limit,
        )
    if dataset_name in KOREAN_CORPUS_MIXES:
        return load_mixed_runtime_docs(
            dataset_name,
            doc_limit=doc_limit,
            text_limit=text_limit,
            topical_prompts=topical_prompts,
        )
    if dataset_name:
        return load_corpus_documents(
            None,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            text_column=text_column,
            doc_limit=doc_limit,
            text_limit=text_limit,
        )
    return list(fallback_docs or [])


def sleep_curriculum_stage(cycle_idx):
    if cycle_idx <= 2:
        return {"name": "wiki", "dataset_name": DEFAULT_WIKI_DATASET}
    if cycle_idx <= 4:
        return {"name": "wiki+alpaca", "dataset_name": WIKI_ALPACA_TRAIN_CORPUS}
    return {"name": "wiki+alpaca+squad", "dataset_name": DEFAULT_TRAIN_CORPUS}


def build_sleep_curriculum_docs(
    *,
    n_cycles,
    dataset_config,
    dataset_split,
    text_column,
    doc_limit,
    text_limit,
    fallback_docs,
    topical_prompts,
):
    cache = {DEFAULT_WIKI_DATASET: list(fallback_docs)}
    schedule = []
    for cycle_idx in range(1, n_cycles + 1):
        stage = sleep_curriculum_stage(cycle_idx)
        dataset_name = stage["dataset_name"]
        docs = cache.get(dataset_name)
        if docs is None:
            try:
                docs = load_runtime_docs(
                    dataset_name=dataset_name,
                    dataset_config=dataset_config,
                    dataset_split=dataset_split,
                    text_column=text_column,
                    doc_limit=doc_limit,
                    text_limit=text_limit,
                    fallback_docs=fallback_docs,
                    topical_prompts=topical_prompts,
                )
            except Exception as exc:
                log(f"  [WARN] curriculum stage '{stage['name']}' fallback: {exc}")
                docs = list(fallback_docs)
            cache[dataset_name] = docs
        schedule.append(
            {
                "cycle": cycle_idx,
                "name": stage["name"],
                "dataset_name": dataset_name,
                "docs": docs,
            }
        )
    return schedule


def measure_time(fn, label=""):
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    if label:
        log(f"  [{label}] {elapsed:.2f}s")
    return result, elapsed


def section(title):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")




def phase_engine_bench(
    artifact_path,
    device,
    prompts,
    eval_docs,
    *,
    context_window,
    seed_tokens,
):
    section("PHASE 2: Runtime-Only Benchmark (Memory / Speed / Stability)")
    from clarus.engine import CEEngine
    from clarus.sleep import evaluate_guard_set

    eng, load_time = measure_time(
        lambda: CEEngine(artifact_path, device=device, backend="torch"),
        "load engine",
    )

    mem = eng.memory_usage()
    artifact_mb = os.path.getsize(artifact_path) / 1024 / 1024
    log(f"\n  --- Memory ---")
    log(f"  Artifact:    {artifact_mb:.2f} MB")
    log(f"  W_dense:     {mem['W_dense_MB']:.2f} MB")
    log(f"  W_packed:    {mem['W_packed_MB']:.2f} MB")
    log(f"  Embedding:   {mem['Embedding_MB']:.2f} MB")
    log(f"  PQ:          {mem['PQ_MB']:.2f} MB")
    log(f"  Runtime:     {mem['runtime_total_MB']:.2f} MB")

    ce_args = argparse.Namespace(
        dt=0.01, cb_weight=None, cb_topk=256, beta=1.0, steps=100,
        backend="torch", metric_rank=8, lambda0=1.0, lambda_phi=0.5,
        lambda_var=0.25, noise_scale=0.3, seed=42,
        decode_mode="standalone", ce_strength=0.3, tokens=24,
        temperature=0.8, phi_threshold=1.0, sleep_threshold=2.0,
        sleep_decay=0.9, top_k=48, repeat_penalty=3.0,
        multiround_steps=64,
        standalone_refresh_interval=1,
        standalone_refresh_steps=32,
        standalone_refresh_cb_topk=128,
        standalone_refresh_metric_rank=0,
        standalone_refresh_noise_scale=0.0,
    )

    holdout = evaluate_guard_set(
        eng,
        eval_docs,
        ce_args,
        max_new_tokens=16,
        refresh_interval=1,
        refresh_steps=32,
        refresh_cb_topk=128,
        refresh_metric_rank=0,
        refresh_noise_scale=0.0,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    log(f"\n  --- Held-out Next Token ---")
    log(
        f"  top1={holdout['top1_acc']:.3f}  top10={holdout['top10_acc']:.3f}  "
        f"top50={holdout['top50_acc']:.3f}  curvature={holdout['curvature_risk']:.3f}"
    )

    results = []
    total_relax_time = 0.0
    total_steps = 0
    total_gen_time = 0.0

    log(f"\n  --- Generation ({len(prompts)} prompts) ---")
    for prompt in tqdm(prompts, desc="gen prompts", unit="p", ncols=80):
        ctx = eng.prompt_context(prompt)
        relax_result, relax_time = measure_time(
            lambda: eng.relax_context(ctx, ce_args),
        )
        total_relax_time += relax_time
        total_steps += relax_result["steps"]

        decode_result, gen_time = measure_time(
            lambda: eng.decode_outputs(ctx, relax_result, ce_args),
        )
        _, outputs, meta = decode_result
        total_gen_time += gen_time

        log(f"\n  [{prompt}]")
        log(
            f"    cos(m0,h)={fmt_metric(relax_result.get('cos_m0_h'))}  "
            f"cos(m*,h)={fmt_metric(relax_result.get('cos_ms_h'))}"
        )
        log(f"    steps={relax_result['steps']}  relax={relax_time:.3f}s  decode={gen_time:.3f}s")
        e_hist = relax_result["hist"]["E"]
        if e_hist:
            log(f"    energy: {e_hist[0]:.4f} -> {e_hist[-1]:.4f}")

        for mode, text in outputs.items():
            log(f"    [{mode}] {text}")

        token_ids = meta.get("standalone_token_ids", [])
        gen_metrics = generation_metrics(outputs.get("standalone", ""), token_ids)
        log(
            f"    repeat={gen_metrics['repeated_token_rate']:.3f}  "
            f"repeat3={gen_metrics['repeated_trigram_rate']:.3f}  "
            f"collapse={gen_metrics['sentence_collapse_rate']:.3f}  "
            f"unfinished={gen_metrics['unfinished_sentence_rate']:.3f}"
        )
        log(
            f"    curvature={fmt_metric(meta.get('standalone_curvature_risk'))}  "
            f"suppression_hits={int(meta.get('standalone_suppression_hits', 0))}"
        )

        convergence_ok = True
        if len(e_hist) > 10:
            tail_var = torch.tensor(e_hist[-10:]).var().item()
            convergence_ok = tail_var < 1.0
            if not convergence_ok:
                log(f"    [WARN] energy tail variance={tail_var:.4f} (unstable)")

        results.append({
            "prompt": prompt,
            "cos_m0_h": relax_result["cos_m0_h"],
            "cos_ms_h": relax_result["cos_ms_h"],
            "steps": relax_result["steps"],
            "relax_time_s": round(relax_time, 4),
            "decode_time_s": round(gen_time, 4),
            "energy_start": e_hist[0] if e_hist else None,
            "energy_end": e_hist[-1] if e_hist else None,
            "convergence_stable": convergence_ok,
            "outputs": outputs,
            "standalone_refresh_count": meta.get("standalone_refresh_count", 0),
            "standalone_curvature_risk": meta.get("standalone_curvature_risk"),
            "standalone_suppression_hits": meta.get("standalone_suppression_hits", 0),
            "generation": gen_metrics,
        })

    avg_time = total_relax_time / max(len(prompts), 1)
    avg_steps = total_steps / max(len(prompts), 1)
    avg_decode = total_gen_time / max(len(prompts), 1)
    generation_summary = {
        "repeated_token_rate": mean_metric([r["generation"]["repeated_token_rate"] for r in results]),
        "repeated_bigram_rate": mean_metric([r["generation"]["repeated_bigram_rate"] for r in results]),
        "repeated_trigram_rate": mean_metric([r["generation"]["repeated_trigram_rate"] for r in results]),
        "unfinished_sentence_rate": mean_metric([r["generation"]["unfinished_sentence_rate"] for r in results]),
        "sentence_collapse_rate": mean_metric([r["generation"]["sentence_collapse_rate"] for r in results]),
        "unique_token_ratio": mean_metric([r["generation"]["unique_token_ratio"] for r in results]),
        "curvature_risk_frequency": mean_metric([r.get("standalone_curvature_risk") for r in results]),
        "suppression_hits": sum(int(r.get("standalone_suppression_hits", 0)) for r in results),
    }
    log(f"\n  --- Speed ---")
    log(f"  avg relax: {avg_time:.3f}s  ({avg_steps:.0f} steps)")
    log(f"  avg decode: {avg_decode:.3f}s")
    log(f"  total relax: {total_relax_time:.2f}s for {len(prompts)} prompts")
    log(f"\n  --- Degeneration ---")
    log(
        f"  repeat={fmt_metric(generation_summary['repeated_token_rate'])}  "
        f"repeat3={fmt_metric(generation_summary['repeated_trigram_rate'])}  "
        f"collapse={fmt_metric(generation_summary['sentence_collapse_rate'])}  "
        f"unfinished={fmt_metric(generation_summary['unfinished_sentence_rate'])}"
    )

    return eng, {
        "load_time_s": load_time,
        "artifact_mb": artifact_mb,
        "memory": mem,
        "avg_relax_time_s": avg_time,
        "avg_relax_steps": avg_steps,
        "avg_decode_time_s": avg_decode,
        "heldout": holdout,
        "generation_summary": generation_summary,
        "prompts": results,
    }


def phase_sleep(eng, train_docs, guard_docs, n_cycles, *, context_window, seed_tokens, train_schedule=None):
    section(f"PHASE 3: Corpus Sleep ({n_cycles} cycles)")
    from clarus.sleep import evaluate_guard_set, run_sleep_cycle

    ce_args = argparse.Namespace(
        dt=0.01, cb_weight=None, cb_topk=256, beta=1.0, steps=64,
        backend="torch", metric_rank=8, lambda0=1.0, lambda_phi=0.5,
        lambda_var=0.25, noise_scale=0.3, seed=42,
        decode_mode="standalone", ce_strength=0.3, tokens=8,
        temperature=0.8, phi_threshold=1.0, sleep_threshold=2.0,
        sleep_decay=0.9, top_k=48, repeat_penalty=3.0,
        multiround_steps=64,
        standalone_refresh_interval=1, standalone_refresh_steps=32,
        standalone_refresh_cb_topk=128, standalone_refresh_metric_rank=0,
        standalone_refresh_noise_scale=0.0,
    )

    before = evaluate_guard_set(
        eng,
        guard_docs,
        ce_args,
        max_new_tokens=16,
        refresh_interval=1,
        refresh_steps=32,
        refresh_cb_topk=128,
        refresh_metric_rank=0,
        refresh_noise_scale=0.0,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    log(
        f"  heldout BEFORE: top1={before['top1_acc']:.3f}  top10={before['top10_acc']:.3f}  "
        f"top50={before['top50_acc']:.3f}  curvature={before['curvature_risk']:.3f}"
    )

    cycles = []
    cycle_bar = tqdm(range(1, n_cycles + 1), desc="sleep cycles", unit="cyc", ncols=80)
    for cycle_idx in cycle_bar:
        cycle_entry = None if train_schedule is None else train_schedule[cycle_idx - 1]
        cycle_docs = train_docs if cycle_entry is None else cycle_entry["docs"]
        cycle_stage = "static" if cycle_entry is None else cycle_entry["name"]
        cycle_bar.set_postfix_str(f"cycle {cycle_idx}/{n_cycles} {cycle_stage}")
        report, elapsed = measure_time(
            lambda: run_sleep_cycle(
                eng, cycle_docs, ce_args,
                max_new_tokens=16, teacher_topk=16,
                ridge=1e-3, rem_weight=2.5, rem_mix=0.35,
                token_head_max_vocab=4096, token_head_scale=1.0,
                refresh_interval=2, refresh_steps=32,
                refresh_cb_topk=128, refresh_metric_rank=0,
                refresh_noise_scale=0.0, refresh_pq=False,
                pq_subdim=64, pq_bits=8, pq_iters=8,
                pq_batch_size=2048, pq_sample_size=8192,
                guard_prompts=guard_docs,
                context_window=context_window,
                seed_tokens=seed_tokens,
                vocab_finetune_lr=2e-3,
                vocab_finetune_steps=128,
                vocab_finetune_batch_size=256,
                vocab_finetune_soft_target_weight=0.45,
            ),
            f"cycle {cycle_idx}",
        )
        log(
            f"    stage={cycle_stage} docs={len(cycle_docs)}  "
            f"wake top10={report['wake']['top10_acc']:.3f} top50={report['wake']['top50_acc']:.3f} -> "
            f"rem top10={report['rem']['top10_acc']:.3f} top50={report['rem']['top50_acc']:.3f}  "
            f"token_vocab={report['token_head_vocab']}"
        )
        rem_weight = report.get("rem_weight") or {}
        cycles.append({
            "cycle": cycle_idx,
            "corpus_stage": cycle_stage,
            "corpus_docs": len(cycle_docs),
            "elapsed_s": elapsed,
            "wake_top10": report["wake"]["top10_acc"],
            "wake_top50": report["wake"]["top50_acc"],
            "nrem_top10": report["nrem"]["top10_acc"],
            "nrem_top50": report["nrem"]["top50_acc"],
            "rem_top10": report["rem"]["top10_acc"],
            "rem_top50": report["rem"]["top50_acc"],
            "rem_curvature": report["rem"]["curvature_risk"],
            "rem_accepted": rem_weight.get("accepted", report.get("cycle_applied", True)),
            "token_head_vocab": report["token_head_vocab"],
        })

    after = evaluate_guard_set(
        eng,
        guard_docs,
        ce_args,
        max_new_tokens=16,
        refresh_interval=1,
        refresh_steps=32,
        refresh_cb_topk=128,
        refresh_metric_rank=0,
        refresh_noise_scale=0.0,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    log(
        f"  heldout AFTER:  top1={after['top1_acc']:.3f}  top10={after['top10_acc']:.3f}  "
        f"top50={after['top50_acc']:.3f}  curvature={after['curvature_risk']:.3f}"
    )
    return {
        "before": before,
        "after": after,
        "top10_delta": after["top10_acc"] - before["top10_acc"],
        "top50_delta": after["top50_acc"] - before["top50_acc"],
        "curvature_delta": after["curvature_risk"] - before["curvature_risk"],
        "cycles": cycles,
    }


def phase_microsleep(eng, prompts, guard_prompts):
    section("PHASE 4: Online Microsleep (Long-term Memory)")
    from clarus.sleep import (
        PromptReplayBuffer,
        evaluate_guard_set,
        run_guarded_microsleep_step,
    )

    ce_args = argparse.Namespace(
        dt=0.01, cb_weight=None, cb_topk=256, beta=1.0, steps=64,
        backend="torch", metric_rank=8, lambda0=1.0, lambda_phi=0.5,
        lambda_var=0.25, noise_scale=0.3, seed=42,
        tokens=10, temperature=0.8, top_k=40, repeat_penalty=3.0,
        standalone_refresh_interval=1, standalone_refresh_steps=32,
        standalone_refresh_cb_topk=128, standalone_refresh_metric_rank=0,
        standalone_refresh_noise_scale=0.0, decode_mode="standalone",
        ce_strength=0.3, phi_threshold=1.0, sleep_threshold=2.0,
        sleep_decay=0.9, multiround_steps=64,
    )

    initial_guard = evaluate_guard_set(
        eng, guard_prompts, ce_args,
        max_new_tokens=16, refresh_interval=1, refresh_steps=32,
        refresh_cb_topk=128, refresh_metric_rank=0, refresh_noise_scale=0.0,
    )
    log(f"  guard BEFORE: top1={initial_guard['top1_acc']:.3f}  top10={initial_guard['top10_acc']:.3f}  top50={initial_guard['top50_acc']:.3f}")

    buffer = PromptReplayBuffer(capacity=16)
    accepted = 0
    rejected = 0

    for idx, prompt in enumerate(prompts, start=1):
        event = run_guarded_microsleep_step(
            eng, buffer, prompt, guard_prompts, ce_args,
            step_index=idx, sleep_every=2, max_new_tokens=16,
            teacher_topk=16, ridge=1e-3, rem_weight=2.5, rem_mix=0.35,
            token_head_max_vocab=4096, token_head_scale=1.0,
            refresh_interval=1, refresh_steps=32, refresh_cb_topk=128,
            refresh_metric_rank=0, refresh_noise_scale=0.0, refresh_pq=False,
            pq_subdim=64, pq_bits=8, pq_iters=8,
            pq_batch_size=2048, pq_sample_size=8192,
        )
        if event is not None:
            status = "accepted" if event["accepted"] else "rejected"
            if event["accepted"]:
                accepted += 1
            else:
                rejected += 1
            log(f"    step {idx}: {status}  buffer={event['buffer_size']}")

    final_guard = evaluate_guard_set(
        eng, guard_prompts, ce_args,
        max_new_tokens=16, refresh_interval=1, refresh_steps=32,
        refresh_cb_topk=128, refresh_metric_rank=0, refresh_noise_scale=0.0,
    )
    log(f"  guard AFTER:  top1={final_guard['top1_acc']:.3f}  top10={final_guard['top10_acc']:.3f}  top50={final_guard['top50_acc']:.3f}")
    log(f"  accepted={accepted}  rejected={rejected}")

    return {
        "initial_guard": initial_guard,
        "final_guard": final_guard,
        "accepted": accepted,
        "rejected": rejected,
        "top10_delta": final_guard["top10_acc"] - initial_guard["top10_acc"],
        "top50_delta": final_guard["top50_acc"] - initial_guard["top50_acc"],
    }


def phase_creativity(eng, prompts):
    section("PHASE 4: Standalone Creativity Evaluation")
    ce_args = argparse.Namespace(
        dt=0.01, cb_weight=None, cb_topk=256, beta=1.0, steps=100,
        backend="torch", metric_rank=8, lambda0=1.0, lambda_phi=0.5,
        lambda_var=0.25, noise_scale=0.5, seed=42,
        decode_mode="standalone", ce_strength=0.3, tokens=20,
        temperature=1.0, phi_threshold=1.0, sleep_threshold=2.0,
        sleep_decay=0.9, top_k=50, repeat_penalty=2.5,
        multiround_steps=64,
        standalone_refresh_interval=1,
        standalone_refresh_steps=48,
        standalone_refresh_cb_topk=128,
        standalone_refresh_metric_rank=0,
        standalone_refresh_noise_scale=0.0,
    )

    results = []
    for prompt in prompts:
        ctx = eng.prompt_context(prompt)
        relax_result = eng.relax_context(ctx, ce_args)
        _, outputs, meta = eng.decode_outputs(ctx, relax_result, ce_args)

        text = outputs.get("standalone", "")
        token_ids = meta.get("standalone_token_ids", [])
        gen_metrics = generation_metrics(text, token_ids)

        log(f"\n  [{prompt}]")
        log(f"    CE: {text}")
        log(
            f"    unique={gen_metrics['unique_token_ratio']:.3f}  "
            f"repeat3={gen_metrics['repeated_trigram_rate']:.3f}  "
            f"collapse={gen_metrics['sentence_collapse_rate']:.3f}  "
            f"curvature={fmt_metric(meta.get('standalone_curvature_risk'))}"
        )

        results.append({
            "prompt": prompt,
            "ce_output": text,
            "unique_ratio": gen_metrics["unique_token_ratio"],
            "ce_tokens": gen_metrics["generated_tokens"],
            "repeated_trigram_rate": gen_metrics["repeated_trigram_rate"],
            "sentence_collapse_rate": gen_metrics["sentence_collapse_rate"],
            "curvature_risk": meta.get("standalone_curvature_risk"),
        })

    avg_unique = sum(r["unique_ratio"] for r in results) / max(len(results), 1)
    avg_collapse = mean_metric([r["sentence_collapse_rate"] for r in results]) or 0.0
    log(f"\n  avg unique ratio: {avg_unique:.3f}  avg collapse: {avg_collapse:.3f}")
    return {
        "prompts": results,
        "avg_unique_ratio": avg_unique,
        "avg_collapse_rate": avg_collapse,
        "avg_curvature_risk": mean_metric([r["curvature_risk"] for r in results]),
    }


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="GPT-2 CE full-pipeline benchmark")
    ap.add_argument("--model", default="skt/kogpt2-base-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2])
    ap.add_argument("--sleep-cycles", type=int, default=8)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--train-data", default=None)
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--train-dataset", default="lcw99/wikipedia-korean-20221001")
    ap.add_argument("--eval-dataset", default="lcw99/wikipedia-korean-20221001")
    ap.add_argument("--sleep-curriculum", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--dataset-text-column", default="text")
    ap.add_argument("--train-doc-limit", type=int, default=64)
    ap.add_argument("--eval-doc-limit", type=int, default=32)
    ap.add_argument("--text-limit", type=int, default=200000)
    ap.add_argument("--context-window", type=int, default=64)
    ap.add_argument("--seed-tokens", type=int, default=8)
    args = ap.parse_args()

    bench_prompts = [
        "인공지능의 미래는 우리가 생각하는 것보다",
        "오늘 날씨가 좋아서 밖에 나가면",
        "한국어를 배우는 가장 좋은 방법은",
        "좋은 모델의 조건은 정확성과",
        "더 나은 시스템을 만들려면 먼저",
        "대한민국의 교육 제도는",
        "건강한 식단을 유지하려면",
        "세계 경제의 흐름은 최근",
    ]
    train_prompts = [
        "과학 기술의 발전은 인류의 삶을",
        "사회 문제를 해결하려면 우선적으로",
        "창의성이 중요한 이유는 새로운 가치를",
        "언어와 사고의 관계는 밀접하게",
        "실용적인 인공지능의 예시는 자연어 처리와",
        "한국의 전통 음식 중 대표적인 것은",
        "효율적인 학습을 위해서는 반복과",
        "환경 문제를 해결하기 위한 방안으로는",
        "디지털 전환이 기업에 미치는 영향은",
        "한글의 우수성은 과학적 원리에 기반한",
        "미래 사회에서 필요한 역량은 비판적",
        "도시와 농촌의 격차를 줄이기 위해서는",
        "한국 영화가 세계적으로 주목받는 이유는",
        "올바른 독서 습관을 기르려면 매일",
        "기술 혁신이 가져올 미래의 변화는",
        "한국의 사계절은 각각 고유한 아름다움을",
        "좋은 글을 쓰기 위해서는 많이 읽고",
        "스마트폰이 일상생활에 미친 영향은",
        "전통과 현대의 조화가 중요한 이유는",
        "자연 생태계를 보전하기 위한 노력으로",
    ]
    guard_prompts = [
        "인공지능의 미래는 우리가 생각하는 것보다",
        "오늘 날씨가 좋아서 밖에 나가면",
        "한국어를 배우는 가장 좋은 방법은",
        "대한민국의 교육 제도는",
        "건강한 식단을 유지하려면",
    ]
    creativity_prompts = [
        "꿈속에서 기계가 발견한 것은",
        "시간이 거꾸로 흐른다면 우리는",
        "우주의 끝에서 우리가 본 것은",
        "바다 밑 도시에서 살게 된다면",
    ]

    artifact_dir = os.path.join(os.path.dirname(__file__), "..", "clarus")
    safe_model = args.model.replace("/", "_")
    artifact_path = args.artifact or os.path.join(artifact_dir, f"{safe_model}.ce.pt")

    log("=" * 60)
    log("  CE-AGI Runtime-Only Korean Benchmark")
    log("=" * 60)
    log(f"  model: {args.model}")
    log(f"  device: {args.device}")
    log(f"  phase: {args.phase}")
    log(f"  sleep_cycles: {args.sleep_cycles}")

    report = {"model": args.model, "device": args.device, "phase": args.phase}
    t_total = time.time()

    train_docs = load_runtime_docs(
        data_path=args.train_data,
        dataset_name=args.train_dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.dataset_text_column,
        doc_limit=args.train_doc_limit,
        text_limit=args.text_limit,
        fallback_docs=train_prompts,
        topical_prompts=[*bench_prompts, *guard_prompts, *train_prompts],
    )
    train_schedule = None
    if (
        args.sleep_curriculum
        and not args.train_data
        and args.train_dataset == DEFAULT_WIKI_DATASET
    ):
        train_schedule = build_sleep_curriculum_docs(
            n_cycles=args.sleep_cycles,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_column=args.dataset_text_column,
            doc_limit=args.train_doc_limit,
            text_limit=args.text_limit,
            fallback_docs=train_docs,
            topical_prompts=[*bench_prompts, *guard_prompts, *train_prompts],
        )
    eval_docs = load_runtime_docs(
        data_path=args.eval_data,
        dataset_name=args.eval_dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.dataset_text_column,
        doc_limit=args.eval_doc_limit,
        text_limit=args.text_limit,
        fallback_docs=guard_prompts,
        topical_prompts=[*bench_prompts, *guard_prompts],
    )
    report["corpus"] = {
        "train_docs": len(train_docs),
        "eval_docs": len(eval_docs),
        "sleep_curriculum": None if train_schedule is None else [entry["name"] for entry in train_schedule],
    }

    eng, engine_report = phase_engine_bench(
        artifact_path,
        args.device,
        bench_prompts,
        eval_docs,
        context_window=args.context_window,
        seed_tokens=args.seed_tokens,
    )
    report["engine"] = engine_report

    if args.sleep_cycles > 0:
        report["sleep"] = phase_sleep(
            eng,
            train_docs,
            eval_docs,
            args.sleep_cycles,
            context_window=args.context_window,
            seed_tokens=args.seed_tokens,
            train_schedule=train_schedule,
        )

    report["creativity"] = phase_creativity(eng, creativity_prompts)

    report["total_time_s"] = time.time() - t_total

    section("SUMMARY")
    log(f"  Total time: {report['total_time_s']:.1f}s")
    log(f"  Avg relax: {engine_report['avg_relax_time_s']:.3f}s")

    log(f"  Runtime artifact: {engine_report['artifact_mb']:.2f} MB")
    log(
        f"  Held-out top10/top50: "
        f"{engine_report['heldout']['top10_acc']:.3f}/{engine_report['heldout']['top50_acc']:.3f}"
    )
    log(
        f"  Repeat/collapse: "
        f"{fmt_metric(engine_report['generation_summary']['repeated_trigram_rate'])}/"
        f"{fmt_metric(engine_report['generation_summary']['sentence_collapse_rate'])}"
    )
    log(f"  Creativity unique ratio: {report.get('creativity', {}).get('avg_unique_ratio', 0):.3f}")
    if "sleep" in report:
        log(
            f"  Sleep gain top10/top50: "
            f"{report['sleep']['top10_delta']:.3f}/{report['sleep']['top50_delta']:.3f}"
        )

    all_stable = all(r.get("convergence_stable", True) for r in engine_report["prompts"])
    log(f"  Stability: {'PASS' if all_stable else 'FAIL'}")

    out_path = os.path.join(os.path.dirname(__file__), "bench_gpt2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    log(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()
