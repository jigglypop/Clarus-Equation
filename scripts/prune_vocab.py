"""Vocab row pruning for the CE runtime artifact (V1, fp32-only).

Goal
----
The fp32 emb_weight (51200 x 768 = 150 MB) dominates the runtime artifact and
breaks the agi-artifact memory rule (artifact must be smaller than the base
model). Quantization is forbidden by agi-artifact section 4, so the only
allowed lossless reduction is **row deletion**: keep top-K most frequent
Korean tokens at full fp32 precision, drop the rest.

Pipeline
--------
1. Tokenize a Korean corpus (the same one used by scripts/distill_decoder.py
   plus an optional --extra-corpus file with one sentence per line).
2. Count BPE token frequencies.
3. Pick the top-K tokens by frequency and unconditionally include:
   - eos / pad / unk / bos
   - any token id referenced by `decoder_token_ids` in the artifact
4. Build a compact emb (K x 768) from the original emb_weight and the
   matching id maps:
   - kept_token_ids[k] -> original tokenizer id (long, K)
   - vocab_id_map[g]   -> compact id k or -1 if pruned (long, vocab)
5. Compute pruned_unk_emb = mean of pruned rows (fallback embedding for
   any prompt token that turns out to be pruned).
6. Save the artifact in-place.

The runtime decoder projections (state_proj, prev_proj, query_bias) are
unaffected because they map state_hidden -> hidden_dim query, independent of
vocab. Only `lexical_scores` and `token_embedding` change behaviour.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from scripts.distill_decoder import CORPUS


def load_corpus(extra_path: str | None) -> list[str]:
    sentences = list(CORPUS)
    if extra_path:
        with open(extra_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    sentences.append(s)
    return sentences


def count_token_frequencies(sentences: list[str], tokenizer) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        counts.update(int(i) for i in ids)
    return counts


def select_kept_ids(
    counts: Counter[int],
    top_k: int,
    *,
    vocab_size: int,
    always_keep: list[int],
) -> torch.Tensor:
    forced = sorted({int(i) for i in always_keep if 0 <= int(i) < vocab_size})
    forced_set = set(forced)
    sorted_pairs = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept: list[int] = list(forced)
    for tok_id, _ in sorted_pairs:
        if len(kept) >= top_k:
            break
        if tok_id in forced_set or tok_id < 0 or tok_id >= vocab_size:
            continue
        kept.append(tok_id)
        forced_set.add(tok_id)
    if len(kept) < top_k:
        for tok_id in range(vocab_size):
            if len(kept) >= top_k:
                break
            if tok_id in forced_set:
                continue
            kept.append(tok_id)
            forced_set.add(tok_id)
    return torch.tensor(sorted(kept), dtype=torch.long)


def coverage(counts: Counter[int], kept_set: set[int]) -> tuple[int, int, float]:
    total = sum(counts.values())
    covered = sum(c for tid, c in counts.items() if tid in kept_set)
    ratio = covered / total if total else 1.0
    return covered, total, ratio


def main():
    ap = argparse.ArgumentParser(description="Prune CE runtime emb to top-K Korean tokens (fp32, no quantization)")
    ap.add_argument("--artifact", default="clarus/skt_kogpt2-base-v2.ce.pt")
    ap.add_argument("--top-k", type=int, default=16384)
    ap.add_argument("--extra-corpus", default=None,
                    help="optional path to extra Korean sentences (one per line)")
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  CE Vocab Pruning (fp32 row deletion, no quantization)")
    print("=" * 60)
    print(f"  artifact : {args.artifact}")
    print(f"  top-K    : {args.top_k}")
    print(f"  extra    : {args.extra_corpus}")

    print("\n[1/5] Loading artifact...")
    art = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if art.get("emb_weight") is None:
        raise RuntimeError("artifact has no emb_weight; nothing to prune")
    if art.get("kept_token_ids") is not None:
        raise RuntimeError("artifact already pruned; rebuild from a fresh fp32 baseline first")
    if art.get("pq_centroids") is not None or art.get("pq_codes") is not None:
        raise RuntimeError("artifact contains PQ payload; quantization is forbidden")

    emb_full: torch.Tensor = art["emb_weight"].float()
    vocab_size = int(art["vocab"])
    if emb_full.shape[0] != vocab_size:
        raise RuntimeError(f"emb_weight rows {emb_full.shape[0]} != vocab {vocab_size}")
    print(f"  emb_full: {tuple(emb_full.shape)}  bytes={emb_full.numel()*4/1024/1024:.2f} MB")

    print("\n[2/5] Loading tokenizer (from artifact, no remote download)...")
    from tokenizers import Tokenizer
    backend_tok = Tokenizer.from_str(art["tokenizer_json"])
    encode = lambda s: [int(i) for i in backend_tok.encode(s).ids]
    class _TokAdapter:
        def encode(self, s, add_special_tokens=False):
            return encode(s)
    tok = _TokAdapter()

    print("\n[3/5] Counting token frequencies on Korean corpus...")
    sentences = load_corpus(args.extra_corpus)
    print(f"  sentences: {len(sentences)}")
    t0 = time.perf_counter()
    counts = count_token_frequencies(sentences, tok)
    print(f"  unique tokens observed: {len(counts)}  total: {sum(counts.values())}  took {time.perf_counter()-t0:.2f}s")

    always_keep: list[int] = []
    for key in ("eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id"):
        if art.get(key) is not None:
            always_keep.append(int(art[key]))
    if art.get("decoder_token_ids") is not None:
        always_keep.extend(int(i) for i in art["decoder_token_ids"].tolist())
    print(f"  always-keep ids: {len(set(always_keep))}")

    print(f"\n[4/5] Selecting top-{args.top_k} ids...")
    kept_ids = select_kept_ids(counts, args.top_k, vocab_size=vocab_size, always_keep=always_keep)
    kept_set = set(int(x) for x in kept_ids.tolist())
    cov_n, cov_total, cov_ratio = coverage(counts, kept_set)
    print(f"  kept: {kept_ids.numel()}  corpus coverage: {cov_n}/{cov_total} = {cov_ratio*100:.2f}%")

    vocab_map = torch.full((vocab_size,), -1, dtype=torch.long)
    vocab_map[kept_ids] = torch.arange(kept_ids.numel(), dtype=torch.long)

    pruned_mask = vocab_map < 0
    if int(pruned_mask.sum().item()) > 0:
        unk_emb = emb_full[pruned_mask].mean(dim=0).contiguous().clone()
    else:
        unk_emb = torch.zeros(emb_full.shape[1], dtype=emb_full.dtype)

    emb_compact = emb_full.index_select(0, kept_ids).contiguous().clone()
    print(f"  emb_compact: {tuple(emb_compact.shape)}  bytes={emb_compact.numel()*4/1024/1024:.2f} MB")

    print("\n[5/5] Writing back...")
    art["emb_weight"] = emb_compact
    art["kept_token_ids"] = kept_ids
    art["vocab_id_map"] = vocab_map
    art["pruned_unk_emb"] = unk_emb
    art["vocab_prune_meta"] = {
        "top_k": int(args.top_k),
        "kept": int(kept_ids.numel()),
        "vocab": int(vocab_size),
        "corpus_sentences": int(len(sentences)),
        "coverage_ratio": float(cov_ratio),
        "extra_corpus": args.extra_corpus or "",
    }
    if "decoder_vocab_weight" in art and art["decoder_vocab_weight"] is not None:
        del art["decoder_vocab_weight"]
        art["decoder_vocab_weight"] = None
    if "decoder_vocab_bias" in art and art["decoder_vocab_bias"] is not None:
        del art["decoder_vocab_bias"]
        art["decoder_vocab_bias"] = None
    del emb_full
    gc.collect()
    torch.save(art, args.artifact)
    size_mb = os.path.getsize(args.artifact) / 1024 / 1024
    print(f"  saved: {args.artifact}  size={size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
