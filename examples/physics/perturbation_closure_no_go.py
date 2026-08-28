"""No-go theorem: a closed FLRW background does not fix perturbations.

Let X=-g^{mu nu}partial_mu T partial_nu T/2 and suppose a monotonic clock
background follows X=X_b(T).  Compare the covariant k-essence Lagrangians

    P0(T,X) = X - V(T),
    P_lambda(T,X) = P0(T,X) + lambda(T) [X-X_b(T)]^2.

On the full background trajectory the added operator and its first X and T
derivatives vanish.  Hence P, rho=2 X P_X-P, the clock equation, H(a), and
the background Ward identity are identical.  The second X derivative is not:

    c_s^2 = P_X/(P_X+2 X P_XX)
           = 1/(1+4 lambda X_b).

Thus a free dimensionless coefficient lambda X_b changes linear scalar
perturbations while leaving every background receipt unchanged.  This exact
counterexample deletes any inference from background closure alone to a
unique growth function, CMB spectrum, lensing spectrum, or f sigma_8.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PerturbationClosureNoGoAudit:
    background_x_over_reference_density: float
    potential_over_reference_density: float
    lambda_times_background_x: float
    background_energy_density_over_reference_density: float
    background_pressure_over_reference_density: float
    canonical_sound_speed_squared: float
    deformed_sound_speed_squared: float
    same_background_lagrangian: bool
    same_background_first_variation: bool
    same_background_ward_identity: bool
    unique_linear_perturbations_follow: bool
    status: str = "BACKGROUND_TO_UNIQUE_PERTURBATIONS_IMPLICATION_DISPROVED"
    claim_ceiling: str = "COMPLETE_COVARIANT_K_ESSENCE_COUNTEREXAMPLE"


def perturbation_closure_no_go(
    *,
    background_x_over_reference_density: float,
    potential_over_reference_density: float,
    lambda_times_background_x: float,
) -> PerturbationClosureNoGoAudit:
    """Return two covariant actions with one background and different c_s^2."""

    x = float(background_x_over_reference_density)
    potential = float(potential_over_reference_density)
    lambda_x = float(lambda_times_background_x)
    if not all(math.isfinite(value) for value in (x, potential, lambda_x)):
        raise ValueError("all dimensionless inputs must be finite")
    if x <= 0.0:
        raise ValueError("background_x_over_reference_density must be positive")
    if lambda_x <= 0.0:
        raise ValueError("lambda_times_background_x must be positive")
    sound_speed_squared = 1.0 / (1.0 + 4.0 * lambda_x)
    return PerturbationClosureNoGoAudit(
        background_x_over_reference_density=x,
        potential_over_reference_density=potential,
        lambda_times_background_x=lambda_x,
        background_energy_density_over_reference_density=x + potential,
        background_pressure_over_reference_density=x - potential,
        canonical_sound_speed_squared=1.0,
        deformed_sound_speed_squared=sound_speed_squared,
        same_background_lagrangian=True,
        same_background_first_variation=True,
        same_background_ward_identity=True,
        unique_linear_perturbations_follow=False,
    )


def subhorizon_acceleration_difference(
    audit: PerturbationClosureNoGoAudit,
    *,
    wavenumber_over_a_h: float,
    density_contrast: float,
) -> float:
    """Return delta_NN(lambda)-delta_NN(0) from the pressure-gradient term."""

    scale = float(wavenumber_over_a_h)
    contrast = float(density_contrast)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("wavenumber_over_a_h must be finite and non-negative")
    if not math.isfinite(contrast):
        raise ValueError("density_contrast must be finite")
    return (
        audit.canonical_sound_speed_squared
        - audit.deformed_sound_speed_squared
    ) * scale * scale * contrast
