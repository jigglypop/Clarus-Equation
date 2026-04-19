"""Korean-base AGI bridge gate benchmark (docs/7_AGI/12_Equation.md A).

Drives the Clarus Hopfield BrainRuntime + ce_ops.relax with a Korean
KoGPT2 hidden-state covariance, then reports gate F1 / F2 / F3 measurements
side-by-side. F4 is regression-only and reported separately when an external
PCI series is supplied via --pci-csv.

Run:
    .venv/Scripts/python.exe scripts/bench_gates.py
    .venv/Scripts/python.exe scripts/bench_gates.py --steps 400
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager

import torch

from clarus.ce_ops import relax
from clarus.constants import ACTIVE_RATIO, BACKGROUND_RATIO, STRUCT_RATIO
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


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


def safe_print(*a, **k) -> None:
    try:
        print(*a, **k, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((" ".join(map(str, a)) + "\n").encode("utf-8", "replace"))


@contextmanager
def measure_peak(device: torch.device):
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


def load_kogpt2(device: torch.device):
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>",
    )
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2").to(device)
    model.eval()
    return model, tok


def hidden_pool(model, tok, prompts, device: torch.device) -> torch.Tensor:
    pool = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=64)
            ids = {k: v.to(device) for k, v in ids.items()}
            out = model(**ids, output_hidden_states=True)
            for h in out.hidden_states:
                pool.append(h[0].float().cpu())
    return torch.cat(pool, dim=0)


def hopfield_from_hidden(hidden: torch.Tensor) -> torch.Tensor:
    h = hidden - hidden.mean(dim=0, keepdim=True)
    n, d = h.shape
    cov = (h.T @ h) / max(n - 1, 1)
    cov = 0.5 * (cov + cov.T)
    cov.fill_diagonal_(0.0)
    lam_max = float(torch.linalg.eigvalsh(cov)[-1].item())
    if lam_max >= -1e-4:
        cov = cov - (lam_max + 1e-3) * torch.eye(d)
    return cov


def driving_stream(hidden: torch.Tensor, n_steps: int, *, gain: float = 0.4) -> torch.Tensor:
    """Cycle through pooled hidden states to feed BrainRuntime.step external_input."""
    n = hidden.shape[0]
    idx = torch.arange(n_steps) % n
    drive = hidden[idx]
    drive = drive / (drive.norm(dim=1, keepdim=True) + 1e-9)
    return drive * gain


def _runtime_with_flag(W: torch.Tensor, *, f1_on: bool, dim: int, active_ratio: float) -> BrainRuntime:
    cfg = BrainRuntimeConfig(
        dim=dim,
        active_ratio=active_ratio,
        f1_self_measure=f1_on,
        f1_pull_strength=0.5,
        f1_ema_alpha=0.05,
    )
    return BrainRuntime(W, config=cfg, backend="torch")


def run_runtime_session(
    W: torch.Tensor,
    drive: torch.Tensor,
    *,
    f1_on: bool,
    active_ratio: float,
    mode_schedule: list[RuntimeMode] | None = None,
) -> dict:
    """Drive a BrainRuntime for `drive.shape[0]` steps.

    When `mode_schedule` is supplied, each step's mode is forced from that
    list (cycled if shorter than the run). Otherwise the runtime's auto-mode
    policy decides. Forced cycles validate the F3 KL meter independently of
    the auto-mode policy.
    """
    rt = _runtime_with_flag(W, f1_on=f1_on, dim=W.shape[0], active_ratio=active_ratio)
    n = int(drive.shape[0])
    t0 = time.perf_counter()
    for k in range(n):
        forced = mode_schedule[k % len(mode_schedule)] if mode_schedule else None
        rt.step(external_input=drive[k], force_mode=forced)
    elapsed = time.perf_counter() - t0
    report = rt.bridge_gate_report()
    return {
        "f1_on": f1_on,
        "steps": n,
        "step_latency_ms": elapsed / n * 1000.0,
        "F1": report["F1_self_organization"],
        "F3": report["F3_ergodic_kl"],
    }


def p_star_schedule(n: int) -> list[RuntimeMode]:
    """Build a mode schedule whose long-run ratio matches p*.

    Allocates round(n * w) and round(n * s) to WAKE / NREM and the remainder
    to REM. Order is irrelevant for mode_occupancy_kl (it only sums counts),
    but we interleave WAKE blocks with sleep blocks so transitions can be
    visualized over time when needed.
    """
    nw = int(round(n * BACKGROUND_RATIO))
    ns = int(round(n * STRUCT_RATIO))
    nr = max(0, n - nw - ns)
    return [RuntimeMode.WAKE] * nw + [RuntimeMode.NREM] * ns + [RuntimeMode.REM] * nr


def run_relax_session(W: torch.Tensor, hidden: torch.Tensor) -> dict:
    d = W.shape[0]
    phi = hidden.std(dim=0).clamp_min(1e-8)
    m0 = hidden.mean(dim=0)
    b = torch.zeros(d)
    t0 = time.perf_counter()
    _, hist, steps = relax(
        W, b, phi, m0,
        portal=0.03120, bypass=0.4892, t_wake=0.3148,
        beta=1.0, cb_w=0.0,
        lambda0=0.1, lambda_phi=0.0, lambda_var=0.0,
        tau=10.0, dt=0.5, max_steps=300, tol=1e-5,
        anneal_ratio=0.5, noise_scale=0.0, seed=0,
        backend="torch",
    )
    elapsed = time.perf_counter() - t0
    return {
        "steps": int(steps),
        "elapsed_s": elapsed,
        "iss": hist["iss"],
        "final_E": hist["E"][-1],
    }


def fmt_kl(report: dict) -> str:
    return (
        f"pi=({report['pi_wake']:.4f}, {report['pi_nrem']:.4f}, {report['pi_rem']:.4f}) "
        f"KL={report['kl_to_p_star']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--active-ratio", type=float, default=0.30,
                    help="Initial active_ratio. F1 self-measure pulls toward 0.0487.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    safe_print("=" * 72)
    safe_print(f"  Korean AGI bridge gate benchmark  device={device}  steps={args.steps}")
    safe_print("=" * 72)

    safe_print("\n[LOAD] skt/kogpt2-base-v2")
    with measure_peak(device) as hf_mem:
        model, tok = load_kogpt2(device)
    safe_print(f"  HF KoGPT2 load mem: {hf_mem['peak_mb']:.1f} MB")

    safe_print("\n[HIDDEN] pooling 13 layers x 8 prompts")
    hidden = hidden_pool(model, tok, PROMPTS, device)
    safe_print(f"  pooled hidden: {tuple(hidden.shape)}  dim={hidden.shape[1]}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    safe_print("\n[BUILD W] hopfield from Korean hidden covariance")
    W = hopfield_from_hidden(hidden)
    eig = torch.linalg.eigvalsh(W)
    safe_print(f"  W shape: {tuple(W.shape)}  lambda in [{eig[0]:.4f}, {eig[-1]:.4f}]")

    safe_print("\n[F2] ce_ops.relax ISS report")
    relax_out = run_relax_session(W, hidden)
    iss = relax_out["iss"]
    safe_print(f"  steps={relax_out['steps']}  elapsed={relax_out['elapsed_s']:.3f}s  E_final={relax_out['final_E']:.4f}")
    safe_print(
        f"  c_k_max={iss['c_k_max']:.4e}  phi_inf={iss['phi_inf_norm']:.4e}  "
        f"mu={iss['mu']:.4f}  R_ball={iss['iss_ball_radius']:.4e}"
    )

    drive = driving_stream(hidden, args.steps, gain=0.4)

    safe_print("\n[F1/F3] BrainRuntime sessions  active_ratio_init={:.4f}  ACTIVE_RATIO*={:.4f}".format(
        args.active_ratio, ACTIVE_RATIO
    ))

    sess_off = run_runtime_session(W, drive, f1_on=False, active_ratio=args.active_ratio)
    sess_on = run_runtime_session(W, drive, f1_on=True, active_ratio=args.active_ratio)
    sess_pstar = run_runtime_session(
        W, drive,
        f1_on=True, active_ratio=args.active_ratio,
        mode_schedule=p_star_schedule(args.steps),
    )

    safe_print("  F1 self-measure OFF (auto-mode):")
    safe_print(f"    EMA={sess_off['F1']['active_ratio_ema']:.4f}  "
               f"deviation={sess_off['F1']['deviation']:+.4f}  "
               f"latency/step={sess_off['step_latency_ms']:.2f} ms")
    safe_print(f"    F3 {fmt_kl(sess_off['F3'])}")

    safe_print("  F1 self-measure ON  (auto-mode):")
    safe_print(f"    EMA={sess_on['F1']['active_ratio_ema']:.4f}  "
               f"deviation={sess_on['F1']['deviation']:+.4f}  "
               f"latency/step={sess_on['step_latency_ms']:.2f} ms")
    safe_print(f"    F3 {fmt_kl(sess_on['F3'])}")

    safe_print("  F1 self-measure ON  (forced p* schedule, F3 meter validation):")
    safe_print(f"    EMA={sess_pstar['F1']['active_ratio_ema']:.4f}  "
               f"deviation={sess_pstar['F1']['deviation']:+.4f}  "
               f"latency/step={sess_pstar['step_latency_ms']:.2f} ms")
    safe_print(f"    F3 {fmt_kl(sess_pstar['F3'])}")

    safe_print("\n[GATE TABLE] (Korean KoGPT2 covariance, dim={})".format(W.shape[0]))
    safe_print("  gate         metric                value")
    safe_print(f"  F2 ISS ball  R_ball                {iss['iss_ball_radius']:.4e}")
    safe_print(f"  F2 ISS ball  mu (Hessian floor)   {iss['mu']:.4f}")
    safe_print(f"  F2 ISS ball  C_k_max              {iss['c_k_max']:.4e}")
    safe_print(f"  F1 EMA off   active_ratio_ema     {sess_off['F1']['active_ratio_ema']:.4f}")
    safe_print(f"  F1 EMA on    active_ratio_ema     {sess_on['F1']['active_ratio_ema']:.4f}")
    safe_print(f"  F1 target    ACTIVE_RATIO         {ACTIVE_RATIO:.4f}")
    safe_print(f"  F1 closure   |EMA_on - target|    {abs(sess_on['F1']['deviation']):.4f}")
    safe_print(f"  F3 KL off    auto-mode WAKE-only  {sess_off['F3']['kl_to_p_star']:.4f}")
    safe_print(f"  F3 KL on     auto-mode WAKE-only  {sess_on['F3']['kl_to_p_star']:.4f}")
    safe_print(f"  F3 KL on     forced p* schedule   {sess_pstar['F3']['kl_to_p_star']:.4f}")
    safe_print(f"  p* target    (Lambda, DM, b)      ({BACKGROUND_RATIO:.4f}, {STRUCT_RATIO:.4f}, {ACTIVE_RATIO:.4f})")


if __name__ == "__main__":
    main()
