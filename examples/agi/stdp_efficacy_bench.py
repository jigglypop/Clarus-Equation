"""STDP on/off efficacy A/B bench for the Layer-F closed loop (F.14.2).

Question this bench answers honestly: does the STDP learning gate that is now
wired into ``BrainRuntime`` (critic_score -> compute_learning_gate ->
apply_stdp_update) actually *improve* anything measurable, or is it inert /
harmful? ``stdp_enabled`` defaults to ``False`` precisely because that question
was open. This script measures it; it does NOT change the default and it does
NOT modify runtime code (measurement only, per canonical reduction invariant).

Design
------
For each seed we build ONE shared init weight and ONE shared input sequence,
then run the *identical* RuntimeAgent loop twice: stdp_enabled=False (control)
and stdp_enabled=True (treatment). Same seed, same inputs, same forced WAKE
schedule (STDP fires in WAKE), so any difference is attributable to the
learning gate alone.

Objective (2026-07 revision): W-CONTROLLABLE next-step prediction error
----------------------------------------------------------------------
The previous verdict used agent critic.score, which is NOT a valid efficacy
target for a weight-learning rule: 40% of it (c_pred) is a cerebellum forward
model independent of the runtime weight W, and c_nov is identically 0 in this
bench. A diagnostic showed critic.score is only ~1.9% controllable by W, so any
STDP result on it is uninformative. We therefore measure a quantity STDP can
actually move: how well the runtime's own recurrent operator predicts its next
activation.

  pred_t   = tanh(W_{start of step t} @ a_{t-1})     # runtime's 1-step forward model
  perr_t   = || a_t - pred_t ||                       # a_t is the realized activation
                                                      # LOWER is better

W enters both the predictor and the realized trajectory, so this is the natural
objective a Hebbian/STDP rule shapes. c_pred (cerebellum) and c_nov (dead) are
excluded from the verdict entirely.

Metrics
-------
(a) perr_improvement = mean(perr[first window]) - mean(perr[last window]).
    POSITIVE = the evolving W predicts its own trajectory better over the run.
(b) perr_slope = OLS slope of perr vs step. NEGATIVE = improving.
(c) weight_drift = ||W_final - W_init||_F. Sanity that STDP moved weights
    (control should be ~0).
(d) guard_regression = guard-probe mean perr evaluated on a FROZEN (stdp-off)
    runtime seeded with the trained weight, on a held-out probe set. Compared on
    vs off; POSITIVE (on - off) means STDP damaged forward-model quality on
    inputs it was not trained toward.

PASS/FAIL (fixed BEFORE running; not tuned to results)
------------------------------------------------------
  EFFICACY: PASS iff mean_seeds(perr_improvement_on - perr_improvement_off) > 0
            AND that mean exceeds its own std across seeds (signal > 1 sigma of
            noise). Otherwise NO-EFFECT (|mean| <= std) or WORSE (mean < -std).
  GUARD:    PASS iff mean_seeds(guard_on - guard_off) <= GUARD_TOL. Else FAIL
            (learning regressed held-out forward-model quality).

Run:  python examples/agi/stdp_efficacy_bench.py
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from reality_stone.clarus.agent import RuntimeAgent, RuntimeAgentConfig
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


GUARD_TOL = 0.02  # allowed guard-probe critic increase before we call it regression


def _make_runtime(weight: torch.Tensor, *, stdp: bool, dim: int) -> BrainRuntime:
    return BrainRuntime(
        weight.clone(),
        config=BrainRuntimeConfig(
            dim=dim,
            active_ratio=0.25,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            memory_capacity=16,
            stdp_enabled=stdp,
            stdp_interval=1,
            stdp_apply_interval=4,
            stdp_lr=0.01,
            stdp_density=0.25,
            stdp_gate_threshold=0.0,
            stdp_spike_threshold=0.1,
        ),
        backend="torch",
        device="cpu",
    )


def _init_weight(gen: torch.Generator, dim: int) -> torch.Tensor:
    w = torch.randn(dim, dim, generator=gen) * 0.05
    w = 0.5 * (w + w.T)
    w.fill_diagonal_(0.0)
    return w


def _input_sequence(gen: torch.Generator, dim: int, n: int) -> list[torch.Tensor]:
    # Structured, mildly autocorrelated stream so a forward model has something
    # learnable rather than white noise.
    seq = []
    base = torch.zeros(dim)
    for _ in range(n):
        step = torch.randn(dim, generator=gen) * 0.3
        base = 0.7 * base + 0.3 * step
        seq.append(base.clone())
    return seq


def _ols_slope(ys: list[float]) -> float:
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    return num / den


@dataclass
class RunResult:
    scores: list[float]
    weight_final: torch.Tensor
    weight_init: torch.Tensor
    stdp_updates: int
    kl_to_p_star: float


def _next_step_perr(runtime: BrainRuntime, prev_act: torch.Tensor) -> float:
    """|| a_t - tanh(W_start @ a_{t-1}) || : W-controllable forward-model error.

    Must be called AFTER runtime.step, passing the activation captured BEFORE it
    and using the weight captured BEFORE it (STDP may have updated W in-step).
    """
    a_t = runtime.activation.detach()
    return float((a_t - torch.tanh(runtime._last_pred_weight @ prev_act)).norm().item())


def _run(weight: torch.Tensor, inputs: list[torch.Tensor], *, stdp: bool, dim: int) -> RunResult:
    runtime = _make_runtime(weight, stdp=stdp, dim=dim)
    w_init = runtime.weight.detach().clone()
    agent = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=4))
    scores: list[float] = []
    for x in inputs:
        prev_act = runtime.activation.detach().clone()
        runtime._last_pred_weight = runtime.weight.detach().clone()  # W at start of step
        out = agent.step(external_input=x, observation=x, force_mode=RuntimeMode.WAKE)
        scores.append(_next_step_perr(runtime, prev_act))
    return RunResult(
        scores=scores,
        weight_final=runtime.weight.detach().clone(),
        weight_init=w_init,
        stdp_updates=int(runtime._stdp_updates),
        kl_to_p_star=float(runtime.mode_occupancy_kl().get("kl_to_p_star", float("nan"))),
    )


def _guard_probe(weight: torch.Tensor, probes: list[torch.Tensor], dim: int) -> float:
    """Frozen (STDP-off) eval of mean next-step perr on a held-out probe set."""
    runtime = _make_runtime(weight, stdp=False, dim=dim)
    agent = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=4))
    vals = []
    for x in probes:
        prev_act = runtime.activation.detach().clone()
        runtime._last_pred_weight = runtime.weight.detach().clone()
        agent.step(external_input=x, observation=x, force_mode=RuntimeMode.WAKE)
        vals.append(_next_step_perr(runtime, prev_act))
    return sum(vals) / max(len(vals), 1)


@dataclass
class SeedMetrics:
    seed: int
    improvement_off: float
    improvement_on: float
    slope_off: float
    slope_on: float
    drift_off: float
    drift_on: float
    guard_off: float
    guard_on: float
    updates_on: int


def run_seed(seed: int, dim: int, n_steps: int, window: int, n_probes: int) -> SeedMetrics:
    gen = torch.Generator().manual_seed(seed)
    weight = _init_weight(gen, dim)
    inputs = _input_sequence(gen, dim, n_steps)
    probes = _input_sequence(gen, dim, n_probes)  # held-out, disjoint from training

    off = _run(weight, inputs, stdp=False, dim=dim)
    on = _run(weight, inputs, stdp=True, dim=dim)

    def improvement(scores: list[float]) -> float:
        return (sum(scores[:window]) / window) - (sum(scores[-window:]) / window)

    guard_off = _guard_probe(off.weight_final, probes, dim)
    guard_on = _guard_probe(on.weight_final, probes, dim)

    return SeedMetrics(
        seed=seed,
        improvement_off=improvement(off.scores),
        improvement_on=improvement(on.scores),
        slope_off=_ols_slope(off.scores),
        slope_on=_ols_slope(on.scores),
        drift_off=float((off.weight_final - off.weight_init).norm().item()),
        drift_on=float((on.weight_final - on.weight_init).norm().item()),
        guard_off=guard_off,
        guard_on=guard_on,
        updates_on=on.stdp_updates,
    )


def _fmt(mean: float, std: float) -> str:
    return f"{mean:+.5f} +/- {std:.5f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--probes", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    args = ap.parse_args()

    torch.set_num_threads(1)
    results = [run_seed(s, args.dim, args.steps, args.window, args.probes) for s in args.seeds]

    print("=" * 78)
    print("STDP on/off efficacy A/B bench  (F.14.2 critic -> learning gate)")
    print(f"dim={args.dim} steps={args.steps} window={args.window} "
          f"probes={args.probes} seeds={args.seeds}")
    print("=" * 78)
    hdr = (f"{'seed':>4} {'impr_off':>10} {'impr_on':>10} {'d_impr':>10} "
           f"{'slope_off':>10} {'slope_on':>10} {'drift_off':>9} {'drift_on':>9} "
           f"{'guard_d':>9} {'upd':>4}")
    print(hdr)
    print("-" * len(hdr))
    d_impr, d_guard, d_slope = [], [], []
    for r in results:
        di = r.improvement_on - r.improvement_off
        dg = r.guard_on - r.guard_off
        ds = r.slope_on - r.slope_off
        d_impr.append(di)
        d_guard.append(dg)
        d_slope.append(ds)
        print(f"{r.seed:>4} {r.improvement_off:>+10.5f} {r.improvement_on:>+10.5f} "
              f"{di:>+10.5f} {r.slope_off:>+10.6f} {r.slope_on:>+10.6f} "
              f"{r.drift_off:>9.4f} {r.drift_on:>9.4f} {dg:>+9.5f} {r.updates_on:>4}")

    def ms(xs: list[float]) -> tuple[float, float]:
        m = statistics.mean(xs)
        s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return m, s

    print("-" * len(hdr))
    mi, si = ms(d_impr)
    mg, sg = ms(d_guard)
    msl, ssl = ms(d_slope)
    print(f"delta improvement (on - off), higher=better : {_fmt(mi, si)}")
    print(f"delta slope       (on - off), lower =better : {_fmt(msl, ssl)}")
    print(f"delta guard score (on - off), lower =better : {_fmt(mg, sg)}")
    print("=" * 78)

    # Verdicts (thresholds fixed before running; not tuned to output).
    if mi > si and mi > 0:
        efficacy = "PASS (improves surprise beyond 1 sigma)"
    elif mi < -si:
        efficacy = "WORSE (degrades surprise beyond 1 sigma)"
    else:
        efficacy = "NO-EFFECT (within 1 sigma of zero)"

    guard = "PASS (no regression)" if mg <= GUARD_TOL else "FAIL (regressed held-out capability)"

    print(f"EFFICACY verdict : {efficacy}")
    print(f"GUARD    verdict : {guard}  (tol={GUARD_TOL})")
    print("NOTE: mode-occupancy KL to p* is degenerate under forced WAKE and is")
    print("      intentionally excluded from the verdict (needs realistic sleep")
    print("      timescales tau_w~65k steps to exercise NREM/REM).")
    print("=" * 78)


if __name__ == "__main__":
    main()
