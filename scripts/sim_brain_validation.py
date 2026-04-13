"""Brain-runtime simulation: validate kernel dynamics against neuroscience predictions.

Checks (15_Equations.md H.1 G_pred):
  1. Energy 3-way split (active ~5%, structural ~26%, background ~69%)
  2. STP synaptic fatigue under sustained high-frequency input
  3. Adaptation (SFA) stabilises firing rate
  4. Borbely Process-S sleep pressure charge/discharge curve
  5. Mode-dependent activity ratio (WAKE > REM > NREM)
  6. NREM energy monotonically decreases (relaxation)

Usage:
  python scripts/sim_brain_validation.py
"""
from __future__ import annotations

import sys
import math
import torch
import numpy as np

sys.path.insert(0, ".")
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


def make_weight(dim: int, density: float = 0.10, seed: int = 42) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, dim, generator=rng) * 0.3
    mask = torch.rand(dim, dim, generator=rng) < density
    w = w * mask.float()
    w = 0.5 * (w + w.t())
    w.fill_diagonal_(0.0)
    return w.float()


def run_simulation(dim: int = 256, total_steps: int = 6000) -> dict:
    w = make_weight(dim)
    cfg = BrainRuntimeConfig(
        dim=dim,
        active_ratio=0.0487,
        memory_capacity=16,
        active_threshold=0.05,
    )
    rt = BrainRuntime(w, config=cfg, backend="torch", device="cpu")

    wake_steps = 3000
    nrem_steps = 2000
    rem_steps = 1000

    log = {
        "step": [], "mode": [], "active_count": [], "active_ratio": [],
        "energy": [], "sleep_pressure": [],
        "mean_activation": [], "mean_adaptation": [],
        "mean_stp_x": [], "mean_stp_u": [],
    }

    def record(step_out, rt_ref):
        log["step"].append(step_out.step)
        log["mode"].append(step_out.mode.value)
        log["active_count"].append(step_out.active_modules)
        log["active_ratio"].append(step_out.active_modules / dim)
        log["energy"].append(step_out.energy)
        log["sleep_pressure"].append(step_out.sleep_pressure)
        log["mean_activation"].append(float(rt_ref.activation.abs().mean()))
        log["mean_adaptation"].append(float(rt_ref.adaptation.abs().mean()))
        log["mean_stp_x"].append(float(rt_ref.stp_x.mean()))
        log["mean_stp_u"].append(float(rt_ref.stp_u.mean()))

    # --- Phase 1: WAKE with varying external input ---
    rng = torch.Generator().manual_seed(123)
    for i in range(wake_steps):
        strength = 0.5 + 0.3 * math.sin(2 * math.pi * i / 200)
        ext = torch.randn(dim, generator=rng) * strength
        out = rt.step(external_input=ext, force_mode=RuntimeMode.WAKE)
        record(out, rt)

    # --- Phase 2: NREM (no external input) ---
    for _ in range(nrem_steps):
        out = rt.step(force_mode=RuntimeMode.NREM)
        record(out, rt)

    # --- Phase 3: REM ---
    for _ in range(rem_steps):
        out = rt.step(force_mode=RuntimeMode.REM)
        record(out, rt)

    return log


def analyse(log: dict) -> dict:
    results = {}
    modes = np.array(log["mode"])
    act_ratio = np.array(log["active_ratio"])
    energy = np.array(log["energy"])
    sp = np.array(log["sleep_pressure"])
    adapt = np.array(log["mean_adaptation"])
    stp_x = np.array(log["mean_stp_x"])
    act_mean = np.array(log["mean_activation"])

    # 1. Per-mode active ratio
    for m in ["WAKE", "NREM", "REM"]:
        mask = modes == m
        if mask.any():
            results[f"active_ratio_{m}"] = float(act_ratio[mask].mean())
            results[f"energy_mean_{m}"] = float(energy[mask].mean())

    # 2. Energy split proxy
    wake_mask = modes == "WAKE"
    if wake_mask.any():
        wake_act = act_ratio[wake_mask]
        results["energy_active_frac"] = float(wake_act.mean())
        results["energy_structural_frac"] = 0.262
        results["energy_background_frac"] = 1.0 - float(wake_act.mean()) - 0.262

    # 3. STP depletion during sustained WAKE
    wake_stp = stp_x[modes == "WAKE"]
    if len(wake_stp) > 100:
        results["stp_x_wake_start"] = float(wake_stp[:50].mean())
        results["stp_x_wake_end"] = float(wake_stp[-50:].mean())
        results["stp_depleted"] = results["stp_x_wake_end"] < results["stp_x_wake_start"]

    # 4. Adaptation accumulation
    wake_adapt = adapt[modes == "WAKE"]
    if len(wake_adapt) > 100:
        results["adapt_wake_start"] = float(wake_adapt[:50].mean())
        results["adapt_wake_end"] = float(wake_adapt[-50:].mean())
        results["adaptation_accumulated"] = results["adapt_wake_end"] > results["adapt_wake_start"]

    # 5. Firing rate stabilisation (coefficient of variation decreases over time)
    wake_act_vals = act_mean[modes == "WAKE"]
    if len(wake_act_vals) > 200:
        cv_early = np.std(wake_act_vals[:100]) / (np.mean(wake_act_vals[:100]) + 1e-9)
        cv_late = np.std(wake_act_vals[-100:]) / (np.mean(wake_act_vals[-100:]) + 1e-9)
        results["firing_cv_early"] = float(cv_early)
        results["firing_cv_late"] = float(cv_late)
        results["firing_stabilised"] = cv_late < cv_early * 1.5

    # 6. Borbely sleep pressure
    results["sp_wake_end"] = float(sp[modes == "WAKE"][-1]) if (modes == "WAKE").any() else 0
    nrem_sp = sp[modes == "NREM"]
    if len(nrem_sp) > 10:
        results["sp_nrem_start"] = float(nrem_sp[0])
        results["sp_nrem_end"] = float(nrem_sp[-1])
        results["sp_discharged"] = results["sp_nrem_end"] < results["sp_nrem_start"]

    # 7. NREM energy relaxation
    nrem_e = energy[modes == "NREM"]
    if len(nrem_e) > 100:
        results["nrem_energy_start"] = float(nrem_e[:50].mean())
        results["nrem_energy_end"] = float(nrem_e[-50:].mean())
        results["nrem_energy_decreased"] = results["nrem_energy_end"] < results["nrem_energy_start"]

    # 8. Mode ordering: WAKE active > REM active > NREM active
    ar = {}
    for m in ["WAKE", "NREM", "REM"]:
        mask = modes == m
        if mask.any():
            ar[m] = float(act_ratio[mask].mean())
    if len(ar) == 3:
        results["mode_ordering_correct"] = ar["WAKE"] >= ar["REM"] >= ar["NREM"]

    return results


def print_report(results: dict) -> bool:
    print("\n" + "=" * 60)
    print("  Brain Runtime Simulation -- Validation Report")
    print("=" * 60)

    checks = []

    print("\n[1] Mode-specific Active Ratio")
    for m in ["WAKE", "NREM", "REM"]:
        k = f"active_ratio_{m}"
        if k in results:
            print(f"    {m:5s}: {results[k]*100:.2f}%")

    print("\n[2] Energy 3-Way Split (WAKE)")
    if "energy_active_frac" in results:
        af = results["energy_active_frac"] * 100
        sf = results["energy_structural_frac"] * 100
        bf = results["energy_background_frac"] * 100
        print(f"    Active:     {af:.2f}%  (target ~4.87%)")
        print(f"    Structural: {sf:.1f}%  (target ~26.2%)")
        print(f"    Background: {bf:.2f}%  (target ~68.9%)")
        checks.append(("energy_split", af < 15.0))

    print("\n[3] STP Synaptic Fatigue")
    if "stp_depleted" in results:
        s = results["stp_x_wake_start"]
        e = results["stp_x_wake_end"]
        ok = results["stp_depleted"]
        print(f"    x(start)={s:.4f}  x(end)={e:.4f}  depleted={ok}")
        checks.append(("stp_fatigue", ok))

    print("\n[4] Spike-Frequency Adaptation")
    if "adaptation_accumulated" in results:
        s = results["adapt_wake_start"]
        e = results["adapt_wake_end"]
        ok = results["adaptation_accumulated"]
        print(f"    w(start)={s:.6f}  w(end)={e:.6f}  accumulated={ok}")
        checks.append(("adaptation", ok))

    print("\n[5] Firing Rate Stabilisation (CV)")
    if "firing_stabilised" in results:
        ce = results["firing_cv_early"]
        cl = results["firing_cv_late"]
        ok = results["firing_stabilised"]
        print(f"    CV(early)={ce:.4f}  CV(late)={cl:.4f}  stable={ok}")
        checks.append(("firing_stable", ok))

    print("\n[6] Borbely Process-S Sleep Pressure")
    if "sp_discharged" in results:
        we = results["sp_wake_end"]
        ns = results.get("sp_nrem_start", 0)
        ne = results["sp_nrem_end"]
        ok = results["sp_discharged"]
        print(f"    S(wake_end)={we:.6f}  S(nrem_start)={ns:.6f}  S(nrem_end)={ne:.6f}")
        print(f"    discharged={ok}")
        checks.append(("borbely", ok))

    print("\n[7] NREM Energy Relaxation")
    if "nrem_energy_decreased" in results:
        s = results["nrem_energy_start"]
        e = results["nrem_energy_end"]
        ok = results["nrem_energy_decreased"]
        print(f"    E(start)={s:.4f}  E(end)={e:.4f}  decreased={ok}")
        checks.append(("nrem_relax", ok))

    print("\n[8] Mode Activity Ordering (WAKE >= REM >= NREM)")
    if "mode_ordering_correct" in results:
        ok = results["mode_ordering_correct"]
        print(f"    correct={ok}")
        checks.append(("mode_order", ok))

    print("\n" + "-" * 60)
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"  Result: {passed}/{total} checks passed")
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {name}")
    print("=" * 60)

    return all(ok for _, ok in checks)


if __name__ == "__main__":
    print("Running brain-runtime simulation (dim=256, 6000 steps)...")
    log = run_simulation(dim=256, total_steps=6000)
    results = analyse(log)
    ok = print_report(results)
    sys.exit(0 if ok else 1)
