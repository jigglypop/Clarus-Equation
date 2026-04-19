"""Side-by-side benchmark: HuggingFace KoGPT2 vs ClarusLM (KoGPT2 transferred).

Measures memory, latency, throughput, stability (no NaN/Inf), accuracy
(token-identity vs HF baseline + per-layer cosine), and on-disk size.

Run:
    .venv/Scripts/python.exe scripts/bench_clarus_lm.py
    .venv/Scripts/python.exe scripts/bench_clarus_lm.py --device cuda
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F


PROMPTS = [
    "인공지능의 미래는",
    "오늘 날씨가",
    "한국에서 가장 유명한 음식은",
    "내일은 친구와 함께",
    "서울의 봄은",
    "가장 좋아하는 책은",
    "클라루스 방정식은 우주의",
    "대한민국 대통령은",
]


def safe_print(*a, **k):
    try:
        print(*a, **k, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(((" ".join(map(str, a))) + "\n").encode("utf-8", "replace"))


@contextmanager
def measure_peak(device: torch.device):
    """Yield a callable returning peak resident memory delta in MB."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        try:
            import psutil
            baseline = psutil.Process(os.getpid()).memory_info().rss
        except ImportError:
            baseline = 0
    state = {"peak_mb": 0.0}
    try:
        yield state
    finally:
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) - baseline
        else:
            try:
                import psutil
                peak = psutil.Process(os.getpid()).memory_info().rss - baseline
            except ImportError:
                peak = 0
        state["peak_mb"] = max(peak, 0) / 1024 / 1024


def time_calls(fn, n_warmup: int = 2, n_iter: int = 5):
    """Return median latency in ms."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load_hf(device: torch.device):
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>",
    )
    m = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2").to(device)
    m.eval()
    return m, tok


def load_clm(device: torch.device, path: str):
    from clarus import load_clarus_lm_generator
    gen = load_clarus_lm_generator(path, device=str(device))
    gen.model.eval()
    return gen


def hf_generate(hf, tok, device, prompt, n=25, seed=42, T=0.7, k=40):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    out = hf.generate(
        ids, max_new_tokens=n, do_sample=True,
        temperature=T, top_k=k, pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def hidden_cosines(hf, clm, tok, device, prompt: str):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    cosines = []
    with torch.no_grad():
        hf_out = hf(ids, output_hidden_states=True)
        hf_hs = hf_out.hidden_states
        x = clm.tok_emb(ids) + clm.pos_emb(clm._pos_idx[: ids.shape[1]])
        for i, blk in enumerate(clm.blocks):
            x = blk(x)
            cos = F.cosine_similarity(x[0, -1], hf_hs[i + 1][0, -1], dim=0).item()
            cosines.append(cos)
        x_norm = clm.norm(x)
        cos_final = F.cosine_similarity(
            x_norm[0, -1],
            hf.transformer.ln_f(hf_hs[-1])[0, -1],
            dim=0,
        ).item()
    return cosines, cos_final


def state_finiteness(model: torch.nn.Module) -> tuple[int, int]:
    """Return (count_finite, count_total) over all params."""
    finite = total = 0
    for p in model.parameters():
        finite += int(torch.isfinite(p).sum().item())
        total += int(p.numel())
    return finite, total


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--clm", default="clarus/clarus_lm_kogpt2.pt")
    ap.add_argument("--seq-len", type=int, default=64,
                    help="Forward latency probe sequence length.")
    ap.add_argument("--n-tokens", type=int, default=25,
                    help="Generation length per prompt.")
    return ap.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    safe_print("=" * 70)
    safe_print(f"  ClarusLM benchmark on device={device}")
    safe_print("=" * 70)

    # ------------------------------------------------------------------
    # Disk size
    # ------------------------------------------------------------------
    clm_size = os.path.getsize(args.clm) / 1024 / 1024
    safe_print(f"\n[DISK] ClarusLM artifact   : {clm_size:.1f} MB ({args.clm})")

    # ------------------------------------------------------------------
    # Load both models with peak-memory measurement
    # ------------------------------------------------------------------
    safe_print("\n[LOAD]")
    with measure_peak(device) as hf_mem:
        hf, tok = load_hf(device)
    safe_print(f"  HF KoGPT2     load mem: {hf_mem['peak_mb']:.1f} MB")
    with measure_peak(device) as clm_mem:
        gen = load_clm(device, args.clm)
        clm = gen.model
    safe_print(f"  ClarusLM      load mem: {clm_mem['peak_mb']:.1f} MB")

    n_params_hf = sum(p.numel() for p in hf.parameters())
    n_params_clm = sum(p.numel() for p in clm.parameters())
    safe_print(f"\n[PARAMS]")
    safe_print(f"  HF KoGPT2 parameters : {n_params_hf / 1e6:.2f} M")
    safe_print(f"  ClarusLM  parameters : {n_params_clm / 1e6:.2f} M")
    safe_print(f"  ratio                : {n_params_clm / n_params_hf:.3f}x")

    # ------------------------------------------------------------------
    # Stability: NaN/Inf scan
    # ------------------------------------------------------------------
    safe_print("\n[STABILITY] (parameter finiteness)")
    f_hf, t_hf = state_finiteness(hf)
    f_clm, t_clm = state_finiteness(clm)
    safe_print(f"  HF KoGPT2  finite params: {f_hf}/{t_hf}  ({pct(f_hf / t_hf)})")
    safe_print(f"  ClarusLM   finite params: {f_clm}/{t_clm}  ({pct(f_clm / t_clm)})")

    # ------------------------------------------------------------------
    # Forward latency (single fixed-seq pass)
    # ------------------------------------------------------------------
    safe_print(f"\n[FORWARD LATENCY] seq_len={args.seq_len}")
    ids = tok.encode("인공지능의 미래는 새로운 시대의 개막을 의미한다", return_tensors="pt").to(device)
    if ids.shape[1] < args.seq_len:
        ids = torch.cat([ids] * ((args.seq_len // ids.shape[1]) + 1), dim=1)
    ids = ids[:, : args.seq_len]

    def fwd_hf():
        with torch.no_grad():
            hf(ids)
            if device.type == "cuda":
                torch.cuda.synchronize()

    def fwd_clm():
        with torch.no_grad():
            clm(ids)
            if device.type == "cuda":
                torch.cuda.synchronize()

    hf_ms = time_calls(fwd_hf)
    clm_ms = time_calls(fwd_clm)
    safe_print(f"  HF KoGPT2  median latency: {hf_ms:.1f} ms  ({args.seq_len / (hf_ms / 1000):.0f} tok/s)")
    safe_print(f"  ClarusLM   median latency: {clm_ms:.1f} ms  ({args.seq_len / (clm_ms / 1000):.0f} tok/s)")
    safe_print(f"  ClarusLM / HF                : {clm_ms / hf_ms:.2f}x")

    # ------------------------------------------------------------------
    # Per-layer accuracy: cosine similarity vs HF
    # ------------------------------------------------------------------
    safe_print("\n[ACCURACY] per-layer cosine vs HF KoGPT2 (avg over prompts)")
    n_layers = len(clm.blocks)
    layer_avg = [0.0] * n_layers
    final_avg = 0.0
    for p in PROMPTS:
        cs, fc = hidden_cosines(hf, clm, tok, device, p)
        for i, c in enumerate(cs):
            layer_avg[i] += c
        final_avg += fc
    for i in range(n_layers):
        layer_avg[i] /= len(PROMPTS)
    final_avg /= len(PROMPTS)
    for i, c in enumerate(layer_avg):
        bar = "#" * int(c * 40)
        safe_print(f"  block {i:2d}: cos={c:.4f}  {bar}")
    safe_print(f"  ln_f   : cos={final_avg:.4f}")

    # ------------------------------------------------------------------
    # Token-identity check (full sweep)
    # ------------------------------------------------------------------
    safe_print(f"\n[ACCURACY] token-identity vs HF (sample: temp=0.7 top_k=40 seed=42 max={args.n_tokens})")
    matches = 0
    for p in PROMPTS:
        h = hf_generate(hf, tok, device, p, n=args.n_tokens)
        c = gen.generate(p, max_tokens=args.n_tokens, temperature=0.7, top_k=40, seed=42)
        ok = h == c
        matches += int(ok)
        mark = "MATCH" if ok else "DIFF "
        safe_print(f"  [{mark}] {p}")
        if not ok:
            safe_print(f"      HF  -> {h!r}")
            safe_print(f"      CLM -> {c!r}")
    safe_print(f"  -> {matches}/{len(PROMPTS)} prompts byte-identical to HF baseline")

    # ------------------------------------------------------------------
    # End-to-end generation throughput
    # ------------------------------------------------------------------
    safe_print(f"\n[GENERATION THROUGHPUT] {args.n_tokens} tokens, {len(PROMPTS)} prompts")
    t0 = time.perf_counter()
    for p in PROMPTS:
        gen.generate(p, max_tokens=args.n_tokens, temperature=0.7, top_k=40, seed=42)
        if device.type == "cuda":
            torch.cuda.synchronize()
    clm_total = time.perf_counter() - t0
    t0 = time.perf_counter()
    for p in PROMPTS:
        hf_generate(hf, tok, device, p, n=args.n_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
    hf_total = time.perf_counter() - t0
    total_tokens = args.n_tokens * len(PROMPTS)
    safe_print(f"  HF KoGPT2  : {hf_total:.2f}s  ({total_tokens / hf_total:.0f} tok/s, {hf_total / total_tokens * 1000:.1f} ms/tok)")
    safe_print(f"  ClarusLM   : {clm_total:.2f}s  ({total_tokens / clm_total:.0f} tok/s, {clm_total / total_tokens * 1000:.1f} ms/tok)")
    safe_print(f"  ratio      : ClarusLM/HF = {clm_total / hf_total:.2f}x")


if __name__ == "__main__":
    main()
