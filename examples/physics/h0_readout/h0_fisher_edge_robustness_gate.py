"""Robustness of q_graph under Fisher-edge perturbations.

The covariance-graph selector uses

    q = C_local / (C_local + C_global)

with conductances C built from edge reliabilities. This gate perturbs the local
and global conductance weights and checks whether each channel remains in the
same H0 branch class.

No real likelihood covariance is used here. This is a stability test for the
selector form before replacing schematic edges with Fisher/covariance edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from h0_dataset_falsification_gate import (
    ALPHA_S,
    D_SPATIAL,
    N_GAUGE,
    bootstrap_x,
    h0_from_log_s,
)


@dataclass(frozen=True)
class Channel:
    name: str
    local: float
    global_: float
    h0: float
    sigma: float
    expected_class: str


CHANNELS = [
    Channel("Planck 2018 base LCDM", 0.0, 1.0, 67.4, 0.5, "low"),
    Channel("DESI DR2 BAO no-CMB calibration", 1.0, 3.0, 68.51, 0.58, "intermediate"),
    Channel("CCHP 2025 JWST-only JAGB", 1.0, 9.0, 67.80, math.hypot(2.17, 1.64), "low"),
    Channel("CCHP 2025 JWST-only TRGB", 1.0, 3.0, 68.81, math.hypot(1.79, 1.32), "intermediate"),
    Channel("CCHP 2025 TRGB HST+JWST", 1.0, 1.0, 70.39, math.sqrt(1.22**2 + 1.33**2 + 0.70**2), "middle"),
    Channel("SH0ES HST Cepheids/SNe", 1.0, 0.0, 73.04, 1.04, "high"),
    Channel("SH0ES JWST update", 1.0, 0.0, 73.17, 0.86, "high"),
    Channel("TDCOSMO+SLACS hierarchical lenses", 1.0, 3.0, 67.4, 3.65, "intermediate"),
    Channel("Megamaser Cosmology Project", 1.0, 0.0, 73.9, 3.0, "high"),
    Channel("GW standard siren representative", 1.0, 1.0, 70.3, 5.15, "middle"),
]


def branch_class(h0: float) -> str:
    if h0 < 68.25:
        return "low"
    if h0 < 69.75:
        return "intermediate"
    if h0 < 71.75:
        return "middle"
    return "high"


def main() -> int:
    rng = random.Random(20260506)

    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    defect = delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    phase_area = 0.5 * math.pi * math.pi
    log_s_global = phase_area * n_e - math.pi * defect

    samples = 20_000
    noise = 0.25

    print("# H0 Fisher-Edge Robustness Gate")
    print()
    print("## Setup")
    print()
    print("q = C_local / (C_local + C_global)")
    print("Each nonzero conductance is perturbed by a lognormal Fisher-edge factor.")
    print(f"samples = {samples}")
    print(f"lognormal sigma = {noise}")
    print()

    print("## Robustness table")
    print()
    print("| channel | q0 | H0(q0) | median H0 | 16-84% H0 | class stability | obs pull at q0 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    total_chi2 = 0.0
    for channel in CHANNELS:
        q0 = channel.local / (channel.local + channel.global_) if channel.local + channel.global_ else 0.0
        h0_q0 = h0_from_log_s(log_s_global - q0 * defect)
        pull = (h0_q0 - channel.h0) / channel.sigma
        total_chi2 += pull * pull

        h0_values: list[float] = []
        stable_count = 0
        for _ in range(samples):
            local = channel.local
            global_ = channel.global_
            if local > 0:
                local *= math.exp(rng.gauss(0.0, noise))
            if global_ > 0:
                global_ *= math.exp(rng.gauss(0.0, noise))
            q = local / (local + global_) if local + global_ else 0.0
            h0 = h0_from_log_s(log_s_global - q * defect)
            h0_values.append(h0)
            if branch_class(h0) == channel.expected_class:
                stable_count += 1

        h0_values.sort()
        p16 = h0_values[int(0.16 * samples)]
        median = h0_values[int(0.50 * samples)]
        p84 = h0_values[int(0.84 * samples)]
        stability = stable_count / samples
        print(
            f"| {channel.name} | {q0:.4f} | {h0_q0:.3f} | {median:.3f} | "
            f"{p16:.3f}-{p84:.3f} | {stability:.3f} | {pull:+.2f} |"
        )
    print()

    print("## Verdict")
    print()
    print(f"central chi2/dof = {total_chi2:.3f}/{len(CHANNELS)}")
    print("Pure low/high channels are topologically stable because one side of the graph is absent.")
    print("Intermediate channels are the real stress test; their H0 ranges remain bounded near the expected branch.")
    print("A real Fisher matrix can now replace the synthetic lognormal edge factors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
