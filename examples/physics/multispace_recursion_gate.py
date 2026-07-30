"""Audit self- and cross-space recursion in the CE zero-trigger model."""

from __future__ import annotations

import numpy as np

from reality_stone.clarus.multispace_bootstrap import (
    branching_regime,
    identity_branch_radius,
    minimal_multispace_fixed_point,
    nearest_neighbor_coupling,
    normalized_transfer_coupling,
    symmetric_reduction_depth,
)


def report(name: str, coupling: np.ndarray) -> None:
    result = minimal_multispace_fixed_point(coupling)
    print(name)
    print(f"  coupling rows       {coupling.tolist()}")
    print(f"  Perron radius       {identity_branch_radius(coupling):.12g}")
    print(f"  regime              {branching_regime(coupling)}")
    print(f"  minimal fixed point {result.survival}")
    print(f"  fixed-point radius  {result.stability_radius:.12g}")
    print(f"  residual            {result.residual:.3e}")
    try:
        depth = symmetric_reduction_depth(coupling)
    except ValueError:
        print("  homogeneous sector none (unequal row sums)")
    else:
        print(f"  homogeneous sector numerical gate passed, D={depth:.12g}")


def main() -> None:
    print("CE MULTISPACE RECURSION GATE")

    report(
        "cross-only reciprocal pair",
        np.array([[0.0, 1.8], [1.8, 0.0]]),
    )
    report(
        "self + asymmetric neighboring spaces",
        np.array([[1.6, 0.9], [0.3, 1.2]]),
    )
    report(
        "large but one-way, non-recursive influence",
        np.array([[0.0, 5.0], [0.0, 0.0]]),
    )
    report(
        "periodic nearest-neighbor ring",
        nearest_neighbor_coupling(
            5,
            self_depth=1.2,
            neighbor_depth=0.6,
            periodic=True,
        ),
    )
    report(
        "open nearest-neighbor chain",
        nearest_neighbor_coupling(
            5,
            self_depth=1.2,
            neighbor_depth=0.6,
            periodic=False,
        ),
    )
    report(
        "normalized two-space transfer (D_eff=d+delta)",
        normalized_transfer_coupling(
            3.0,
            0.17776,
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ),
    )

    print(
        "scope: row i is the source and column j is the next-generation type; "
        "the scalar CE equation is the homogeneous equal-row-sum sector"
    )


if __name__ == "__main__":
    main()
