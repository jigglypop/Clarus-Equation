'''Finite evaluations of the Lorentzian EPRL map and proper projector.

The EPRL ``Y_gamma`` map embeds the spin-k SU(2) representation as the
lowest SU(2) type of the SL(2,C) principal-series representation
``(k, gamma*k)``.  The full codomain is infinite dimensional; this module
materializes the published homogeneous-function value, not a truncation of
that representation.

The proper projector is finite dimensional.  Given five nondegenerate
SL(2,C) frame representatives, it computes Engle--Zipfel's orientation sign
``beta_ab`` and the strictly-positive spectral projector of

    beta_ab tr(sigma_i X_ab X_ab^dagger) L_k^i.

Zero orientation determinants and zero operator eigenvalues are excluded
from the positive sector.  These operators do not by themselves provide a
Regge boundary phase, a vertex integral, or a five-vertex amplitude.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    IDENTITY_TWO,
    PAULI_MATRICES,
    sl2c_lorentz_matrix,
)


def _require_spin(spin: Fraction) -> Fraction:
    if not isinstance(spin, Fraction) or spin < 0 or spin.denominator not in (1, 2):
        raise ValueError('spin must be a nonnegative half-integer Fraction')
    return spin


def _require_sl2c(element: Sequence[Sequence[complex]]) -> np.ndarray:
    matrix = np.asarray(element, dtype=complex)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError('SL(2,C) element must be a finite two by two matrix')
    if abs(complex(np.linalg.det(matrix)) - 1.0) > 1.0e-10:
        raise ValueError('SL(2,C) element must have determinant one')
    return matrix


@dataclass(frozen=True)
class SpinGenerators:
    spin: Fraction
    magnetic_numbers: tuple[Fraction, ...]
    l1: np.ndarray
    l2: np.ndarray
    l3: np.ndarray
    hermiticity_residual: float
    commutator_residual: float
    casimir_residual: float


def spin_generators(spin: Fraction) -> SpinGenerators:
    '''Return L_i in the ascending |k,m> basis, m=-k,...,k.'''

    k = _require_spin(spin)
    twice_k = 2 * k.numerator // k.denominator
    dimension = twice_k + 1
    magnetic = tuple(-k + index for index in range(dimension))
    raising = np.zeros((dimension, dimension), dtype=complex)
    for column, m_value in enumerate(magnetic[:-1]):
        coefficient = math.sqrt(float((k - m_value) * (k + m_value + 1)))
        raising[column + 1, column] = coefficient
    lowering = np.conjugate(raising.T)
    l1 = (raising + lowering) / 2.0
    l2 = (raising - lowering) / (2.0j)
    l3 = np.diag([float(value) for value in magnetic]).astype(complex)
    generators = (l1, l2, l3)
    hermiticity = max(
        float(np.linalg.norm(item - np.conjugate(item.T))) for item in generators
    )
    commutators = (
        l1 @ l2 - l2 @ l1 - 1.0j * l3,
        l2 @ l3 - l3 @ l2 - 1.0j * l1,
        l3 @ l1 - l1 @ l3 - 1.0j * l2,
    )
    casimir = l1 @ l1 + l2 @ l2 + l3 @ l3
    return SpinGenerators(
        spin=k,
        magnetic_numbers=magnetic,
        l1=l1,
        l2=l2,
        l3=l3,
        hermiticity_residual=hermiticity,
        commutator_residual=max(float(np.linalg.norm(item)) for item in commutators),
        casimir_residual=float(
            np.linalg.norm(casimir - float(k * (k + 1)) * np.eye(dimension))
        ),
    )


@dataclass(frozen=True)
class EPRLHomogeneousEvaluation:
    spin: Fraction
    gamma: float
    principal_series_k: Fraction
    principal_series_p: float
    spinor: np.ndarray
    argument: np.ndarray
    coherent_polynomial_value: complex
    radial_factor: complex
    embedded_value: complex
    spinor_norm_residual: float
    argument_norm_squared: float
    su2_lowest_type_selected: bool
    full_principal_series_representation_materialized: bool


def evaluate_y_gamma_coherent_state(
    spin: Fraction,
    gamma: float,
    spinor: Sequence[complex],
    argument: Sequence[complex],
) -> EPRLHomogeneousEvaluation:
    '''Evaluate (Y_gamma C^k_xi)(z) in the homogeneous realization.'''

    k = _require_spin(spin)
    if not math.isfinite(gamma):
        raise ValueError('gamma must be finite and real')
    xi = np.asarray(spinor, dtype=complex)
    z_value = np.asarray(argument, dtype=complex)
    if xi.shape != (2,) or not np.all(np.isfinite(xi)):
        raise ValueError('spinor must contain two finite complex components')
    if z_value.shape != (2,) or not np.all(np.isfinite(z_value)):
        raise ValueError('argument must contain two finite complex components')
    xi_norm = float(np.vdot(xi, xi).real)
    z_norm = float(np.vdot(z_value, z_value).real)
    if xi_norm <= 0.0:
        raise ValueError('spinor must be nonzero')
    if z_norm <= 0.0:
        raise ValueError('homogeneous argument must be nonzero')
    xi = xi / math.sqrt(xi_norm)
    twice_k = 2 * k.numerator // k.denominator
    normalization = math.sqrt(float(2 * k + 1) / math.pi)
    # <bar(xi),z> in the paper's polynomial convention equals xi^T z.
    coherent = normalization * complex(np.dot(xi, z_value)) ** twice_k
    exponent = complex(-1.0 - float(k), gamma * float(k))
    radial = complex(z_norm) ** exponent
    return EPRLHomogeneousEvaluation(
        spin=k,
        gamma=gamma,
        principal_series_k=k,
        principal_series_p=gamma * float(k),
        spinor=xi,
        argument=z_value,
        coherent_polynomial_value=coherent,
        radial_factor=radial,
        embedded_value=radial * coherent,
        spinor_norm_residual=abs(float(np.vdot(xi, xi).real) - 1.0),
        argument_norm_squared=z_norm,
        su2_lowest_type_selected=True,
        full_principal_series_representation_materialized=False,
    )


@dataclass(frozen=True)
class ProperOrientationSign:
    label_a: int
    label_b: int
    remaining_labels: tuple[int, int, int]
    first_spatial_determinant: float
    second_spatial_determinant: float
    first_normalized_determinant: float
    second_normalized_determinant: float
    beta: int
    nondegenerate: bool
    common_order_reversal_invariant: bool


def _spatial_future_columns(
    frames: Mapping[int, np.ndarray],
    base: int,
    labels: tuple[int, int, int],
) -> tuple[np.ndarray, float, float]:
    columns: list[np.ndarray] = []
    for label in labels:
        relative = np.linalg.solve(frames[base], frames[label])
        columns.append(sl2c_lorentz_matrix(relative)[1:, 0])
    matrix = np.column_stack(columns)
    determinant = float(np.linalg.det(matrix))
    scale = math.prod(float(np.linalg.norm(column)) for column in columns)
    normalized = determinant / scale if scale > 0.0 else 0.0
    return matrix, determinant, normalized


def proper_orientation_sign(
    frames: Mapping[int, Sequence[Sequence[complex]]],
    label_a: int,
    label_b: int,
    *,
    tolerance: float = 1.0e-12,
) -> ProperOrientationSign:
    '''Compute beta_ab from the two Engle--Zipfel spatial triple products.'''

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    if label_a == label_b or label_a not in frames or label_b not in frames:
        raise ValueError('labels must be distinct members of the frame map')
    if len(frames) != 5:
        raise ValueError('proper orientation sign requires exactly five frames')
    checked = {label: _require_sl2c(frame) for label, frame in frames.items()}
    remaining = tuple(sorted(label for label in checked if label not in (label_a, label_b)))
    if len(remaining) != 3:
        raise ValueError('exactly three complementary labels are required')
    _, first, first_normalized = _spatial_future_columns(
        checked, label_a, remaining
    )
    _, second, second_normalized = _spatial_future_columns(
        checked, label_b, remaining
    )
    nondegenerate = (
        abs(first_normalized) > tolerance
        and abs(second_normalized) > tolerance
    )
    beta = 0 if not nondegenerate else (1 if first * second > 0.0 else -1)
    reversed_remaining = tuple(reversed(remaining))
    _, reversed_first, _ = _spatial_future_columns(
        checked, label_a, reversed_remaining  # type: ignore[arg-type]
    )
    _, reversed_second, _ = _spatial_future_columns(
        checked, label_b, reversed_remaining  # type: ignore[arg-type]
    )
    reversed_beta = (
        0
        if not nondegenerate
        else (1 if reversed_first * reversed_second > 0.0 else -1)
    )
    return ProperOrientationSign(
        label_a=label_a,
        label_b=label_b,
        remaining_labels=remaining,  # type: ignore[arg-type]
        first_spatial_determinant=first,
        second_spatial_determinant=second,
        first_normalized_determinant=first_normalized,
        second_normalized_determinant=second_normalized,
        beta=beta,
        nondegenerate=nondegenerate,
        common_order_reversal_invariant=(beta == reversed_beta),
    )


@dataclass(frozen=True)
class ProperProjector:
    spin: Fraction
    beta: int
    relative_sl2c: np.ndarray
    trace_vector: np.ndarray
    trace_vector_norm: float
    nontrivial_generator: bool
    operator: np.ndarray
    eigenvalues: np.ndarray
    positive_eigenvalue_count: int
    zero_eigenvalue_count: int
    projector: np.ndarray
    hermiticity_residual: float
    projector_idempotence_residual: float
    projector_hermiticity_residual: float
    strictly_positive_interval_used: bool


def proper_positive_spectral_projector(
    spin: Fraction,
    beta: int,
    relative_sl2c: Sequence[Sequence[complex]],
    *,
    tolerance: float = 1.0e-12,
) -> ProperProjector:
    '''Return Pi_(0,infinity)[beta tr(sigma_i X X^dagger) L_i].'''

    k = _require_spin(spin)
    if beta not in (-1, 1):
        raise ValueError('beta must be +1 or -1 for a nondegenerate sector')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    relative = _require_sl2c(relative_sl2c)
    generators = spin_generators(k)
    positive_matrix = relative @ np.conjugate(relative.T)
    trace_vector = np.asarray(
        [float(np.trace(pauli @ positive_matrix).real) for pauli in PAULI_MATRICES]
    )
    trace_vector_norm = float(np.linalg.norm(trace_vector))
    operator = beta * sum(
        component * generator
        for component, generator in zip(
            trace_vector,
            (generators.l1, generators.l2, generators.l3),
        )
    )
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    positive = eigenvalues > tolerance
    zero = np.abs(eigenvalues) <= tolerance
    selected = eigenvectors[:, positive]
    projector = (
        selected @ np.conjugate(selected.T)
        if selected.shape[1] > 0
        else np.zeros_like(operator)
    )
    return ProperProjector(
        spin=k,
        beta=beta,
        relative_sl2c=relative,
        trace_vector=trace_vector,
        trace_vector_norm=trace_vector_norm,
        nontrivial_generator=(trace_vector_norm > tolerance),
        operator=operator,
        eigenvalues=eigenvalues,
        positive_eigenvalue_count=int(np.count_nonzero(positive)),
        zero_eigenvalue_count=int(np.count_nonzero(zero)),
        projector=projector,
        hermiticity_residual=float(
            np.linalg.norm(operator - np.conjugate(operator.T))
        ),
        projector_idempotence_residual=float(
            np.linalg.norm(projector @ projector - projector)
        ),
        projector_hermiticity_residual=float(
            np.linalg.norm(projector - np.conjugate(projector.T))
        ),
        strictly_positive_interval_used=True,
    )


def relative_sl2c(
    source: Sequence[Sequence[complex]],
    target: Sequence[Sequence[complex]],
) -> np.ndarray:
    '''Return X_source^{-1} X_target in the proper-vertex convention.'''

    return np.linalg.solve(_require_sl2c(source), _require_sl2c(target))


def identity_sl2c() -> np.ndarray:
    '''Return a fresh identity representative for tests and callers.'''

    return IDENTITY_TWO.copy()
