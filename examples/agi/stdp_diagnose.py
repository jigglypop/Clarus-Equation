"""Diagnostic for STDP no-effect result (H1-H4). Measurement only; no runtime edits."""
from __future__ import annotations
import os, sys, math, statistics, warnings
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import torch
from reality_stone.clarus.agent import RuntimeAgent, RuntimeAgentConfig
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from reality_stone.clarus import stdp as stdp_mod
from reality_stone.clarus.stdp import structural_projection

DIM = 48
STEPS = 240


def make_runtime(weight, *, stdp, dim=DIM, **over):
    cfg = dict(dim=dim, active_ratio=0.25, active_threshold=0.0, noise_sigma=0.0,
               dale_law=False, axon_delay=False, memory_capacity=16,
               stdp_enabled=stdp, stdp_interval=1, stdp_apply_interval=4,
               stdp_lr=0.01, stdp_density=0.25, stdp_gate_threshold=0.0,
               stdp_spike_threshold=0.1)
    cfg.update(over)
    return BrainRuntime(weight.clone(), config=BrainRuntimeConfig(**cfg),
                        backend="torch", device="cpu")


def init_weight(gen, dim=DIM):
    w = torch.randn(dim, dim, generator=gen) * 0.05
    w = 0.5 * (w + w.T); w.fill_diagonal_(0.0); return w


def input_seq(gen, n, dim=DIM):
    seq = []; base = torch.zeros(dim)
    for _ in range(n):
        step = torch.randn(dim, generator=gen) * 0.3
        base = 0.7 * base + 0.3 * step; seq.append(base.clone())
    return seq


# --- H1: gate time series ---------------------------------------------------
def h1(seed=1):
    gen = torch.Generator().manual_seed(seed)
    w = init_weight(gen); inputs = input_seq(gen, STEPS)
    rt = make_runtime(w, stdp=True)
    agent = RuntimeAgent(rt, config=RuntimeAgentConfig(action_count=4))
    gates, drives, derivs, boot = [], [], [], []
    orig = stdp_mod.compute_learning_gate
    def spy(critic_score, prev_critic_score, active_ratio, **kw):
        d = (critic_score - prev_critic_score) / 1.0
        alpha = kw.get("alpha_g", 0.7)
        bdev = (active_ratio - 0.0487) ** 2 + (0.26 - 0.2623) ** 2 + (0.69 - 0.6891) ** 2
        g = orig(critic_score, prev_critic_score, active_ratio, **kw)
        drives.append(critic_score); derivs.append(d); boot.append(bdev); gates.append(g)
        return g
    # patch on both module and runtime's imported ref
    stdp_mod.compute_learning_gate = spy
    import reality_stone.clarus.runtime as rtm
    rtm.compute_learning_gate = spy
    for x in inputs:
        agent.step(external_input=x, observation=x, force_mode=RuntimeMode.WAKE)
    stdp_mod.compute_learning_gate = orig; rtm.compute_learning_gate = orig
    def stats(a):
        return (statistics.mean(a), statistics.pstdev(a), min(a), max(a))
    npos = sum(1 for g in gates if g > 0); nneg = sum(1 for g in gates if g < 0)
    print("=== H1 gate time series (seed=%d, n_gate_evals=%d) ===" % (seed, len(gates)))
    print("gate       mean/std/min/max: %+.5f %.5f %+.5f %+.5f" % stats(gates))
    print("drive(crit)mean/std/min/max: %+.5f %.5f %+.5f %+.5f" % stats(drives))
    print("deriv term mean/std/min/max: %+.5f %.5f %+.5f %+.5f" % stats(derivs))
    print("boot term  mean/std/min/max: %+.5f %.5f %+.5f %+.5f" % stats(boot))
    print("gate sign: +%d / -%d / 0=%d" % (npos, nneg, len(gates) - npos - nneg))
    print("0.7*deriv contrib mean: %+.6f | 0.3*boot contrib mean: %+.6f"
          % (0.7 * statistics.mean(derivs), 0.3 * statistics.mean(boot)))
    return gates


# --- H3: critic sensitivity to weight perturbation --------------------------
def h3(seed=1, n_pert=12):
    gen = torch.Generator().manual_seed(seed)
    w = init_weight(gen); inputs = input_seq(gen, 80)

    def run_mean_critic(weight):
        rt = make_runtime(weight, stdp=False)
        ag = RuntimeAgent(rt, config=RuntimeAgentConfig(action_count=4))
        vals = []
        for x in inputs:
            out = ag.step(external_input=x, observation=x, force_mode=RuntimeMode.WAKE)
            vals.append(float(out.critic.score))
        return statistics.mean(vals)

    base = run_mean_critic(w)
    diffs = []
    for k in range(n_pert):
        g2 = torch.Generator().manual_seed(1000 + k)
        pert = torch.randn(DIM, DIM, generator=g2)
        pert = 0.5 * (pert + pert.T); pert.fill_diagonal_(0.0)
        pert = pert / pert.norm() * w.norm()  # same magnitude as W
        for scale in [0.3]:
            wp = structural_projection(w + scale * pert, density=0.25)
            diffs.append(run_mean_critic(wp) - base)
    print("\n=== H3 critic sensitivity to weight perturbation (seed=%d) ===" % seed)
    print("base mean critic: %.5f" % base)
    print("perturbed deltas mean/std/absmax: %+.5f %.5f %.5f"
          % (statistics.mean(diffs), statistics.pstdev(diffs), max(abs(d) for d in diffs)))
    print("relative sensitivity (absmax/base): %.4f" % (max(abs(d) for d in diffs) / max(base, 1e-9)))
    return diffs


# --- H4: projection annihilation of the update ------------------------------
def h4(seed=1):
    gen = torch.Generator().manual_seed(seed)
    w = init_weight(gen); inputs = input_seq(gen, STEPS)
    rt = make_runtime(w, stdp=True)
    agent = RuntimeAgent(rt, config=RuntimeAgentConfig(action_count=4))
    ratios = []
    orig_apply = stdp_mod.apply_stdp_update
    import reality_stone.clarus.runtime as rtm
    def spy(weight, tracker, gate, lr=0.01, density=0.25):
        dw = lr * gate * tracker.eligibility
        pre_norm = dw.norm().item()
        new_w = structural_projection(weight + dw, density=density)
        post_norm = (new_w - structural_projection(weight, density=density)).norm().item()
        el = tracker.eligibility.norm().item()
        ratios.append((el, pre_norm, post_norm))
        return new_w
    stdp_mod.apply_stdp_update = spy; rtm.apply_stdp_update = spy
    for x in inputs:
        agent.step(external_input=x, observation=x, force_mode=RuntimeMode.WAKE)
    stdp_mod.apply_stdp_update = orig_apply; rtm.apply_stdp_update = orig_apply
    if not ratios:
        print("\n=== H4: no updates applied ==="); return
    els = [r[0] for r in ratios]; pre = [r[1] for r in ratios]; post = [r[2] for r in ratios]
    print("\n=== H4 projection effect on update (seed=%d, n_apply=%d) ===" % (seed, len(ratios)))
    print("||eligibility||     mean/max: %.5f %.5f" % (statistics.mean(els), max(els)))
    print("||dw=lr*g*e|| (pre) mean/max: %.6f %.6f" % (statistics.mean(pre), max(pre)))
    print("||Proj(W+dw)-Proj(W)|| (post) mean/max: %.6f %.6f" % (statistics.mean(post), max(post)))
    keep = [po / pr for po, pr in zip(post, pre) if pr > 1e-12]
    if keep:
        print("post/pre retained-fraction mean: %.3f" % statistics.mean(keep))


# --- H2: hyperparameter sweep ----------------------------------------------
def h2(seeds=(1, 2, 3)):
    print("\n=== H2 hyperparameter sweep (delta improvement on-off, mean over %d seeds) ===" % len(seeds))
    grids = {
        "baseline": {},
        "lr x3": dict(stdp_lr=0.03),
        "lr x0.3": dict(stdp_lr=0.003),
        "A+/A- x3 (a_plus/a_minus via spike thr lower)": dict(stdp_spike_threshold=0.03),
        "density 0.5": dict(stdp_density=0.5),
        "density 0.1": dict(stdp_density=0.1),
        "apply_interval 1": dict(stdp_apply_interval=1),
        "apply_interval 16": dict(stdp_apply_interval=16),
    }
    def improvement(scores, window=40):
        return sum(scores[:window]) / window - sum(scores[-window:]) / window
    def run(weight, inputs, stdp, over):
        rt = make_runtime(weight, stdp=stdp, **over)
        ag = RuntimeAgent(rt, config=RuntimeAgentConfig(action_count=4))
        return [float(ag.step(external_input=x, observation=x, force_mode=RuntimeMode.WAKE).critic.score) for x in inputs]
    for name, over in grids.items():
        deltas = []
        for s in seeds:
            gen = torch.Generator().manual_seed(s)
            w = init_weight(gen); inputs = input_seq(gen, STEPS)
            off = run(w, inputs, False, over); on = run(w, inputs, True, over)
            deltas.append(improvement(on) - improvement(off))
        print("%-45s d_impr = %+.5f +/- %.5f" % (name, statistics.mean(deltas), statistics.pstdev(deltas)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    h1(1)
    h3(1)
    h4(1)
    h2((1, 2, 3))
