"""Small-scale boost requirement gate for high-z halo closure.

The previous high-z proxy showed that background-only CE suppresses rare halos.
This gate asks the inverse question: how large a scale-dependent small-scale
amplitude boost would be required to recover or exceed LCDM rare-tail abundance?

This is still not a full halo mass function.  It is a requirement card: it
turns a desired rare-tail abundance ratio into the required multiplicative
boost B on the CE linear amplitude at fixed mass scale.
"""

from __future__ import annotations

import math


CE_STATIC_AMP_RATIO = 0.97125559
CE_H0_AMP_RATIO = 0.93208112


def tail_ratio(total_amp_ratio: float, nu_lcdm: float) -> float:
    nu_ce = nu_lcdm / total_amp_ratio
    lcdm_tail = math.erfc(nu_lcdm / math.sqrt(2.0))
    ce_tail = math.erfc(nu_ce / math.sqrt(2.0))
    return ce_tail / lcdm_tail


def required_total_amp_ratio(nu_lcdm: float, target_tail_ratio: float) -> float:
    """Find total amplitude ratio R such that tail_ratio(R, nu)=target."""
    lo = 0.1
    hi = 3.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        value = tail_ratio(mid, nu_lcdm)
        if value < target_tail_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    rarity_bins = [2.0, 3.0, 4.0]
    target_tail_ratios = [1.0, 2.0, 5.0, 10.0]
    branches = [
        ("CE static", CE_STATIC_AMP_RATIO),
        ("CE H0 branch", CE_H0_AMP_RATIO),
    ]

    print("# Small-scale Boost Requirement Gate")
    print()
    print("## Baseline amplitude ratios")
    print()
    print(f"CE static background amplitude ratio = {CE_STATIC_AMP_RATIO:.8f}")
    print(f"CE H0-branch background amplitude ratio = {CE_H0_AMP_RATIO:.8f}")
    print()

    print("## Requirement formula")
    print()
    print("Rare-tail proxy: F(>nu) proportional to erfc(nu/sqrt(2))")
    print("If total amplitude ratio is R_tot, then nu_CE = nu_LCDM/R_tot")
    print("B_required = R_tot_required / R_background")
    print()

    print("## Required total amplitude ratio")
    print()
    print("| nu_LCDM | target tail ratio | R_tot required |")
    print("|---:|---:|---:|")
    required = {}
    for nu in rarity_bins:
        for target in target_tail_ratios:
            r_tot = required_total_amp_ratio(nu, target)
            required[(nu, target)] = r_tot
            print(f"| {nu:.0f} | {target:.1f} | {r_tot:.8f} |")
    print()

    print("## Required extra boost over CE background")
    print()
    print("| branch | nu_LCDM | target tail ratio | B_required | percent boost |")
    print("|---|---:|---:|---:|---:|")
    for label, r_bg in branches:
        for nu in rarity_bins:
            for target in target_tail_ratios:
                r_tot = required[(nu, target)]
                boost = r_tot / r_bg
                print(
                    f"| {label} | {nu:.0f} | {target:.1f} | "
                    f"{boost:.8f} | {100.0 * (boost - 1.0):+.2f}% |"
                )
    print()

    print("## Candidate CE-sized handles")
    print()
    delta = 0.17775842340997383
    x = 0.04864671964402835
    sigma = 1.0 - x
    q_source = x * sigma
    q_a3c = (2.0 / math.pi) * sigma ** (3.1777584234099736 / 4.177758423409974) * q_source
    print(f"delta = {delta:.8f}")
    print(f"x = {x:.8f}")
    print(f"q_source = x(1-x) = {q_source:.8f}")
    print(f"q_A3c = {q_a3c:.8f}")
    print(f"1 + delta = {1.0 + delta:.8f}")
    print(f"1 + q_source = {1.0 + q_source:.8f}")
    print(f"1 + q_A3c = {1.0 + q_a3c:.8f}")
    print()

    candidates = [
        ("none", 1.0),
        ("1+q_A3c", 1.0 + q_a3c),
        ("1+q_source", 1.0 + q_source),
        ("1+delta", 1.0 + delta),
    ]
    print("## Candidate boost outcomes")
    print()
    print("Tail ratios below use nu=3 as a representative rare high-z object.")
    print()
    print("| branch | candidate boost | total amp ratio | tail ratio at nu=3 |")
    print("|---|---:|---:|---:|")
    for label, r_bg in branches:
        for cname, boost in candidates:
            r_tot = r_bg * boost
            ratio = tail_ratio(r_tot, 3.0)
            print(f"| {label} / {cname} | {boost:.8f} | {r_tot:.8f} | {ratio:.6f} |")
    print()

    print("## Verdict")
    print()
    print("To merely match LCDM rare-tail abundance, CE needs only a modest small-scale boost:")
    print("about +3% for the static branch and +7% for the H0 branch.")
    print("For genuinely rare nu=3-4 tails, factors of 2-10 require O(10-45%) boosts.")
    print("For less rare nu=2 objects, a 10x tail enhancement would require a much larger boost.")
    print("q_A3c alone roughly restores the static branch but does not strongly enhance it.")
    print("delta-sized boosts are large enough in principle, especially for nu=3-4 tails.")
    print("This remains an Open candidate until a scale-dependent transfer law is derived.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
