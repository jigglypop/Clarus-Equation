"""Run the first CE core-strengthening loop.

This is an audit, not an observable-validation script.  It checks which
candidate laws survive composition, ensemble-mixture affinity, branch
stability, Hodge degree closure, and normalized electroweak coherence.
"""

from __future__ import annotations

from functools import partial

from reality_stone.clarus.core_axioms import (
    ElectroweakCoherence,
    bivector_vector_closure_dimensions,
    bootstrap_stability_multiplier,
    complement_feedback,
    composition_residual,
    electroweak_effective_depth,
    exponential_survival,
    low_bootstrap_fixed_point,
    mixture_affinity_residual,
    power_survival,
    powered_feedback,
    stretched_exponential_survival,
    weak_mixing_from_fixed_point,
)


def main() -> None:
    depths = (0.0, 0.25, 0.75, 1.5, 3.0)
    values = (0.0, 0.1, 0.5, 0.9, 1.0)
    weights = (0.0, 0.2, 0.5, 0.8, 1.0)

    survival_candidates = {
        "exponential": exponential_survival,
        "stretched_p2": partial(stretched_exponential_survival, power=2.0),
        "power": power_survival,
    }
    feedback_candidates = {
        "complement": complement_feedback,
        "powered_p2": partial(powered_feedback, power=2.0),
    }

    print("CE CORE AXIOM LOOP")
    print("composition residuals")
    for name, law in survival_candidates.items():
        print(f"  {name:16s} {composition_residual(law, depths):.12g}")

    print("mixture-affinity residuals")
    for name, law in feedback_candidates.items():
        print(
            f"  {name:16s} "
            f"{mixture_affinity_residual(law, values, weights):.12g}"
        )

    sin2_theta = 0.23122
    g = (1.0 - sin2_theta) ** 0.5
    g_prime = sin2_theta**0.5
    coherence = ElectroweakCoherence(g=g, g_prime=g_prime)
    depth = electroweak_effective_depth(3, g=g, g_prime=g_prime)
    low = low_bootstrap_fixed_point(depth)

    print("conditional core chain")
    print(f"  EW coherence intensity  {coherence.intensity:.12g}")
    print(f"  effective depth         {depth:.12g}")
    print(f"  low fixed point         {low:.12g}")
    print(f"  low stability           {bootstrap_stability_multiplier(low, depth):.12g}")
    print(f"  identity stability      {bootstrap_stability_multiplier(1.0, depth):.12g}")
    print(f"  inverse sin^2(theta)    {weak_mixing_from_fixed_point(low):.12g}")
    print(
        "  bivector->vector d     "
        f"{bivector_vector_closure_dimensions(max_dimension=12)}"
    )
    print("scope: structural/conditional checks only; no observable bridge is scored")


if __name__ == "__main__":
    main()
