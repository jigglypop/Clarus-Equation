"""No-go theorem: finite face geometry does not fix continuum GR or two DOF.

The finite Lorentzian face reconstruction modules determine intrinsic metric
data and a declared transport on a fixed complex.  They contain no functional
derivative with respect to a continuum metric and therefore cannot determine
a continuum action.

An explicit covariant counterexample is enough.  On the same flat background,

    S_EH       = integral sqrt(-g) R,
    S_R2       = integral sqrt(-g) (R + alpha R^2),  alpha > 0,

have the identical zero-curvature stationary geometry and hence identical
finite flat face/transport data.  Their linearized spectra differ.  The trace
of the R+alpha R^2 field equation is

    -R + 6 alpha Box R = 0,

so R obeys a Klein--Gordon equation with m_scalar^2=1/(6 alpha).  Numerically
we use alpha_bar=alpha/L_ref^2 and report (m_scalar L_ref)^2, so every API
input and output is dimensionless.  Einstein
gravity has only the two massless transverse-traceless polarizations, whereas
the second action has those two plus the scalaron.  Thus no finite background
reconstruction theorem can imply a unique Einstein--Hilbert action or exactly
two propagating degrees of freedom.

This is a complete counterexample to that implication, not a claim that the
R^2 action is selected by CE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MINKOWSKI_METRIC = np.diag((-1.0, 1.0, 1.0, 1.0))


def massless_spin_two_polarization_count(dimension: int) -> int:
    """Little-group count d(d-3)/2 for a massless spin-2 field."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 4:
        raise ValueError("dimension must be an integer of at least four")
    return dimension * (dimension - 3) // 2


def massive_spin_two_polarization_count(dimension: int) -> int:
    """Symmetric tensor minus d transversality and one trace constraint."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 3:
        raise ValueError("dimension must be an integer of at least three")
    return dimension * (dimension + 1) // 2 - dimension - 1


def massless_tt_basis_4d() -> tuple[np.ndarray, np.ndarray]:
    """Return plus/cross tensors for a null wave along the positive z axis."""

    plus = np.zeros((4, 4))
    plus[1, 1] = 1.0
    plus[2, 2] = -1.0
    cross = np.zeros((4, 4))
    cross[1, 2] = cross[2, 1] = 1.0
    return plus, cross


def massive_traceless_transverse_basis_4d() -> tuple[np.ndarray, ...]:
    """Return five rest-frame polarizations of a massive spin-2 field."""

    spatial: list[np.ndarray] = []
    first = np.diag((1.0, -1.0, 0.0))
    second = np.diag((1.0, 1.0, -2.0))
    spatial.extend((first, second))
    for left, right in ((0, 1), (0, 2), (1, 2)):
        item = np.zeros((3, 3))
        item[left, right] = item[right, left] = 1.0
        spatial.append(item)
    result: list[np.ndarray] = []
    for item in spatial:
        tensor = np.zeros((4, 4))
        tensor[1:, 1:] = item
        result.append(tensor)
    return tuple(result)


@dataclass(frozen=True)
class ContinuumGRNoGoAudit:
    alpha_over_reference_length_squared: float
    scalaron_mass_squared_times_reference_length_squared: float
    einstein_hilbert_polarizations: int
    r_plus_r_squared_polarizations: int
    shared_flat_stationary_background: bool
    shared_finite_flat_face_data: bool
    both_actions_diffeomorphism_invariant: bool
    unique_continuum_action_follows: bool
    exactly_two_dof_follow: bool
    status: str = "FINITE_FACE_TO_UNIQUE_CONTINUUM_GR_IMPLICATION_DISPROVED"
    claim_ceiling: str = "COMPLETE_COUNTEREXAMPLE_TO_BACKGROUND_ONLY_GR_CLOSURE"


def continuum_gr_dof_no_go(
    alpha_over_reference_length_squared: float = 1.0,
) -> ContinuumGRNoGoAudit:
    """Return the exact R versus R+alpha R^2 counterexample."""

    alpha_bar = float(alpha_over_reference_length_squared)
    if not math.isfinite(alpha_bar) or alpha_bar <= 0.0:
        raise ValueError(
            "alpha_over_reference_length_squared must be finite and positive"
        )
    eh_count = massless_spin_two_polarization_count(4)
    scalaron_mass_squared = 1.0 / (6.0 * alpha_bar)
    return ContinuumGRNoGoAudit(
        alpha_over_reference_length_squared=alpha_bar,
        scalaron_mass_squared_times_reference_length_squared=(
            scalaron_mass_squared
        ),
        einstein_hilbert_polarizations=eh_count,
        r_plus_r_squared_polarizations=eh_count + 1,
        shared_flat_stationary_background=True,
        shared_finite_flat_face_data=True,
        both_actions_diffeomorphism_invariant=True,
        unique_continuum_action_follows=False,
        exactly_two_dof_follow=False,
    )
