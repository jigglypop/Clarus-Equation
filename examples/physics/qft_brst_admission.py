'''Finite BRST admission gates for the QFT redesign programme.

The module deliberately separates four statements that are often conflated:

* nilpotency of a graded differential;
* triviality of a ghost-number-one breaking;
* nonzero ghost-number-zero cohomology;
* positivity of the form induced on that cohomology.

It also evaluates the one-momentum, free, linearized Einstein Ward complex on a
flat background.  That second calculation is an exact tree-level sector of the
M1 action when Lambda=m=0 and the scalar backgrounds are constant.  Constant
reference scalars do not define a relational chart, so the result is not an M2
admission and is not a claim about an interacting or continuum BRST charge.
All matrices use dimensionless momentum k/k_ref.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TOL = 1.0e-10


@dataclass(frozen=True)
class PerturbativeBrstContract:
    '''Frozen scope data for the first, tree-level BRST calculation.'''

    signature: str
    background: str
    background_on_shell: bool
    reference_background: str
    reference_patch_nondegenerate: bool
    loop_order: int
    regulator: str
    gauge_fixing_fermion: str
    eft_operator_dimension_max: int
    renormalization_scale_over_planck: float
    counterterm_basis: str
    linearized_tree_ward_identity_computed: bool
    loop_anomaly_cohomology_computed: bool
    nonperturbative_m2_passed: bool


def flat_tree_contract() -> PerturbativeBrstContract:
    '''Return the exact scope of the linearized calculation in this module.'''

    return PerturbativeBrstContract(
        signature='Lorentzian (-,+,+,+)',
        background='Minkowski; Lambda=0; chi=0; m=0',
        background_on_shell=True,
        reference_background='constant X^A',
        reference_patch_nondegenerate=False,
        loop_order=0,
        regulator='none: finite one-momentum algebraic audit',
        gauge_fixing_fermion='de Donder family, used only as admission data',
        eft_operator_dimension_max=2,
        renormalization_scale_over_planck=1.0,
        counterterm_basis='none at tree level',
        linearized_tree_ward_identity_computed=True,
        loop_anomaly_cohomology_computed=False,
        nonperturbative_m2_passed=False,
    )


def validate_contract(contract: PerturbativeBrstContract) -> None:
    '''Fail closed when required scope data are absent or dimensionally invalid.'''

    text_fields = (
        contract.signature,
        contract.background,
        contract.reference_background,
        contract.regulator,
        contract.gauge_fixing_fermion,
        contract.counterterm_basis,
    )
    if any(not value.strip() for value in text_fields):
        raise ValueError('all BRST admission scope fields must be nonempty')
    if contract.loop_order < 0:
        raise ValueError('loop order must be nonnegative')
    if contract.eft_operator_dimension_max < 2:
        raise ValueError('the Einstein two-derivative sector must be retained')
    if not np.isfinite(contract.renormalization_scale_over_planck):
        raise ValueError('mu/M_P must be finite and dimensionless')
    if contract.renormalization_scale_over_planck <= 0.0:
        raise ValueError('mu/M_P must be positive')
    if contract.linearized_tree_ward_identity_computed and not contract.background_on_shell:
        raise ValueError('the physical tree-sector audit requires an on-shell background')
    if contract.loop_order > 0 and contract.regulator.startswith('none'):
        raise ValueError('positive loop order requires an explicit regulator')
    if contract.loop_order > 0 and contract.counterterm_basis.startswith('none'):
        raise ValueError('positive loop order requires an explicit counterterm basis')
    if contract.nonperturbative_m2_passed and (
        not contract.reference_patch_nondegenerate
        or not contract.loop_anomaly_cohomology_computed
    ):
        raise ValueError('a degenerate or unaudited sector cannot pass full M2')


@dataclass(frozen=True)
class FiniteBrstComplex:
    '''Three-step cochain data V_-1 -> V_0 -> V_1 -> V_2.'''

    d_minus_one: np.ndarray
    d_zero: np.ndarray
    d_one: np.ndarray
    names_minus_one: tuple[str, ...]
    names_zero: tuple[str, ...]
    names_one: tuple[str, ...]


def quartet_complex(*, include_anomaly: bool = False) -> FiniteBrstComplex:
    '''Build s q=c, s c=0, s cbar=B, s B=0, s x=0.

    With ``include_anomaly=True`` an additional closed ghost-one generator a is
    supplied without a ghost-zero preimage.  It is therefore a deliberately
    nontrivial H^1 obstruction in the declared counterterm space.
    '''

    d_minus_one = np.array([[0.0], [0.0], [1.0]])
    if include_anomaly:
        d_zero = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        names_one = ('c', 'a')
    else:
        d_zero = np.array([[0.0, 1.0, 0.0]])
        names_one = ('c',)
    d_one = np.zeros((0, d_zero.shape[0]))
    return FiniteBrstComplex(
        d_minus_one=d_minus_one,
        d_zero=d_zero,
        d_one=d_one,
        names_minus_one=('cbar',),
        names_zero=('x', 'q', 'B'),
        names_one=names_one,
    )


def _rank(matrix: np.ndarray, *, tol: float = TOL) -> int:
    matrix = np.asarray(matrix, dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not singular.size or singular[0] == 0.0:
        return 0
    threshold = tol * max(matrix.shape) * singular[0]
    return int(np.count_nonzero(singular > threshold))


def _nullspace(matrix: np.ndarray, *, tol: float = TOL) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    if not singular.size or singular[0] == 0.0:
        rank = 0
    else:
        threshold = tol * max(matrix.shape) * singular[0]
        rank = int(np.count_nonzero(singular > threshold))
    return vh[rank:].T.copy()


def cohomology_dimensions(
    complex_: FiniteBrstComplex, *, tol: float = TOL
) -> tuple[int, int, int]:
    '''Return dimensions of H^-1, H^0 and H^1 after checking s^2=0.'''

    residual = np.linalg.norm(complex_.d_zero @ complex_.d_minus_one)
    residual += np.linalg.norm(complex_.d_one @ complex_.d_zero)
    if residual > tol:
        raise ValueError('the supplied differential is not nilpotent')
    h_minus_one = complex_.d_minus_one.shape[1] - _rank(
        complex_.d_minus_one, tol=tol
    )
    h_zero = (
        complex_.d_zero.shape[1]
        - _rank(complex_.d_zero, tol=tol)
        - _rank(complex_.d_minus_one, tol=tol)
    )
    h_one = (
        complex_.d_one.shape[1]
        - _rank(complex_.d_one, tol=tol)
        - _rank(complex_.d_zero, tol=tol)
    )
    return h_minus_one, h_zero, h_one


def ghost_zero_representatives(
    complex_: FiniteBrstComplex, *, tol: float = TOL
) -> np.ndarray:
    '''Return an orthonormal complement of exact states inside ker d_0.'''

    closed = _nullspace(complex_.d_zero, tol=tol)
    exact = complex_.d_minus_one
    if exact.shape[1]:
        exact_u, singular, _ = np.linalg.svd(exact, full_matrices=False)
        exact_u = exact_u[:, singular > tol]
        candidates = closed - exact_u @ (exact_u.T @ closed)
    else:
        candidates = closed
    u, singular, _ = np.linalg.svd(candidates, full_matrices=False)
    return u[:, singular > tol]


@dataclass(frozen=True)
class BreakingAudit:
    closed: bool
    removable: bool
    closure_residual: float
    counterterm_residual: float
    counterterm: tuple[float, ...]


def audit_breaking(
    complex_: FiniteBrstComplex,
    breaking: np.ndarray,
    *,
    tol: float = TOL,
) -> BreakingAudit:
    '''Test Wess--Zumino closure and exactness in the declared counterterm space.'''

    breaking = np.asarray(breaking, dtype=float)
    if breaking.shape != (complex_.d_zero.shape[0],):
        raise ValueError('breaking must be a ghost-number-one vector')
    closure_residual = float(np.linalg.norm(complex_.d_one @ breaking))
    counterterm, *_ = np.linalg.lstsq(complex_.d_zero, breaking, rcond=None)
    counterterm_residual = float(
        np.linalg.norm(complex_.d_zero @ counterterm - breaking)
    )
    closed = closure_residual <= tol
    return BreakingAudit(
        closed=closed,
        removable=closed and counterterm_residual <= tol,
        closure_residual=closure_residual,
        counterterm_residual=counterterm_residual,
        counterterm=tuple(float(value) for value in counterterm),
    )


@dataclass(frozen=True)
class PhysicalFormAudit:
    descends_to_cohomology: bool
    positive: bool
    exact_closed_pairing_residual: float
    eigenvalues: tuple[float, ...]


def audit_physical_form(
    complex_: FiniteBrstComplex,
    gram: np.ndarray,
    *,
    tol: float = TOL,
) -> PhysicalFormAudit:
    '''Check that a form descends to H^0 and is positive there.'''

    gram = np.asarray(gram, dtype=float)
    n_zero = complex_.d_zero.shape[1]
    if gram.shape != (n_zero, n_zero):
        raise ValueError('gram matrix has the wrong ghost-zero dimension')
    if np.linalg.norm(gram - gram.T) > tol:
        raise ValueError('gram matrix must be Hermitian in this real toy')
    cohomology_dimensions(complex_, tol=tol)
    closed = _nullspace(complex_.d_zero, tol=tol)
    exact = complex_.d_minus_one
    descent_residual = float(np.linalg.norm(exact.T @ gram @ closed))
    representatives = ghost_zero_representatives(complex_, tol=tol)
    physical_gram = representatives.T @ gram @ representatives
    eigenvalues = np.linalg.eigvalsh(physical_gram)
    descends = descent_residual <= tol
    positive = descends and bool(np.all(eigenvalues > tol))
    return PhysicalFormAudit(
        descends_to_cohomology=descends,
        positive=positive,
        exact_closed_pairing_residual=descent_residual,
        eigenvalues=tuple(float(value) for value in eigenvalues),
    )


def gauge_fermion_deformation_class_residual(
    complex_: FiniteBrstComplex, coefficient: float
) -> float:
    '''Project the s-exact B deformation onto the chosen H^0 representatives.'''

    deformation = complex_.d_minus_one @ np.array([float(coefficient)])
    representatives = ghost_zero_representatives(complex_)
    return float(np.linalg.norm(representatives.T @ deformation))


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
SYMMETRIC_INDICES = tuple(
    (mu, nu) for mu in range(4) for nu in range(mu, 4)
)


def pack_symmetric(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (4, 4):
        raise ValueError('a symmetric rank-two tensor must be 4 by 4')
    if np.linalg.norm(tensor - tensor.T) > TOL:
        raise ValueError('tensor must be symmetric')
    return np.array([tensor[mu, nu] for mu, nu in SYMMETRIC_INDICES])


def unpack_symmetric(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (len(SYMMETRIC_INDICES),):
        raise ValueError('packed symmetric tensor must have ten components')
    tensor = np.zeros((4, 4))
    for value, (mu, nu) in zip(vector, SYMMETRIC_INDICES, strict=True):
        tensor[mu, nu] = value
        tensor[nu, mu] = value
    return tensor


def linearized_diffeomorphism_map(k_contravariant: np.ndarray) -> np.ndarray:
    '''Map a covector gauge parameter to delta h_mn=k_m xi_n+k_n xi_m.'''

    k_contravariant = np.asarray(k_contravariant, dtype=float)
    if k_contravariant.shape != (4,):
        raise ValueError('momentum must have four components')
    k_covariant = ETA @ k_contravariant
    columns = []
    for index in range(4):
        xi_covariant = np.zeros(4)
        xi_covariant[index] = 1.0
        variation = np.outer(k_covariant, xi_covariant)
        variation += np.outer(xi_covariant, k_covariant)
        columns.append(pack_symmetric(variation))
    return np.column_stack(columns)


def _linearized_einstein_tensor(
    h_covariant: np.ndarray, k_contravariant: np.ndarray
) -> np.ndarray:
    k_covariant = ETA @ k_contravariant
    k_squared = float(k_contravariant @ k_covariant)
    trace = float(np.einsum('mn,mn->', ETA, h_covariant))
    k_h = np.einsum('r,rn->n', k_contravariant, h_covariant)
    k_h_k = float(np.einsum('r,s,rs->', k_contravariant, k_contravariant, h_covariant))
    result = k_squared * h_covariant
    result += np.outer(k_covariant, k_covariant) * trace
    result -= np.outer(k_covariant, k_h)
    result -= np.outer(k_h, k_covariant)
    result -= ETA * (k_squared * trace - k_h_k)
    return 0.5 * result


def linearized_einstein_operator(k_contravariant: np.ndarray) -> np.ndarray:
    '''Return the ten-by-ten Fierz--Pauli equation map at momentum k/k_ref.'''

    k_contravariant = np.asarray(k_contravariant, dtype=float)
    columns = []
    for index in range(len(SYMMETRIC_INDICES)):
        basis = np.zeros(len(SYMMETRIC_INDICES))
        basis[index] = 1.0
        h_covariant = unpack_symmetric(basis)
        columns.append(
            pack_symmetric(_linearized_einstein_tensor(h_covariant, k_contravariant))
        )
    return np.column_stack(columns)


def transverse_traceless_basis(
    k_contravariant: np.ndarray | None = None,
) -> np.ndarray:
    '''Return normalized plus/cross tensors transverse to a nonzero null k.'''

    if k_contravariant is None:
        k_contravariant = np.array([1.0, 0.0, 0.0, 1.0])
    momentum = np.asarray(k_contravariant, dtype=float)
    if momentum.shape != (4,):
        raise ValueError('momentum must have four components')
    momentum_norm = float(np.linalg.norm(momentum))
    if momentum_norm <= TOL:
        raise ValueError('the TT frame requires nonzero momentum')
    if abs(float(momentum @ ETA @ momentum)) > TOL * momentum_norm**2:
        raise ValueError('the TT frame requires null momentum')
    spatial = momentum[1:]
    spatial_norm = float(np.linalg.norm(spatial))
    if spatial_norm <= TOL:
        raise ValueError('a null TT frame requires nonzero spatial momentum')
    direction = spatial / spatial_norm
    reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
    transverse_one = np.cross(direction, reference)
    transverse_one /= np.linalg.norm(transverse_one)
    transverse_two = np.cross(direction, transverse_one)

    plus = np.zeros((4, 4))
    plus[1:, 1:] = (
        np.outer(transverse_one, transverse_one)
        - np.outer(transverse_two, transverse_two)
    ) / np.sqrt(2.0)
    cross = np.zeros((4, 4))
    cross[1:, 1:] = (
        np.outer(transverse_one, transverse_two)
        + np.outer(transverse_two, transverse_one)
    ) / np.sqrt(2.0)
    return np.column_stack((pack_symmetric(plus), pack_symmetric(cross)))


def scalar_kinetic_gram(mu_x_over_k_ref: float = 1.0) -> np.ndarray:
    '''Return the M1 kinetic Gram for chi and four unrescaled X^A modes.'''

    ratio = float(mu_x_over_k_ref)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError('mu_X/k_ref must be finite, positive and dimensionless')
    return np.diag([1.0, ratio**2, ratio**2, ratio**2, ratio**2])


def _symmetric_frobenius_weight() -> np.ndarray:
    weights = [1.0 if mu == nu else 2.0 for mu, nu in SYMMETRIC_INDICES]
    return np.diag(weights)


@dataclass(frozen=True)
class LinearizedGravityBrstAudit:
    k_squared: float
    ward_residual: float
    equation_rank: int
    solution_dimension: int
    gauge_rank: int
    quotient_dimension: int
    tt_equation_residual: float
    tt_gauge_overlap_residual: float
    tt_gram_eigenvalues: tuple[float, ...]
    scalar_gram_eigenvalues: tuple[float, ...]
    five_scalar_modes_positive: bool
    total_free_physical_mode_count: int
    reference_patch_nondegenerate: bool
    tree_gate_passed: bool
    loop_anomaly_cohomology_computed: bool
    nonperturbative_m2_passed: bool


def audit_flat_null_gravity_sector(
    k_contravariant: np.ndarray | None = None,
    *,
    mu_x_over_k_ref: float = 1.0,
    tol: float = TOL,
) -> LinearizedGravityBrstAudit:
    '''Audit the free null spin-two quotient and its two positive TT modes.'''

    if k_contravariant is None:
        k_contravariant = np.array([1.0, 0.0, 0.0, 1.0])
    k_contravariant = np.asarray(k_contravariant, dtype=float)
    momentum_norm = float(np.linalg.norm(k_contravariant))
    if momentum_norm <= tol:
        raise ValueError('the momentum must be nonzero')
    raw_k_squared = float(k_contravariant @ ETA @ k_contravariant)
    if abs(raw_k_squared) > tol * momentum_norm**2:
        raise ValueError('the massless one-particle audit requires null momentum')
    k_contravariant = k_contravariant / momentum_norm
    k_squared = float(k_contravariant @ ETA @ k_contravariant)

    equation = linearized_einstein_operator(k_contravariant)
    gauge = linearized_diffeomorphism_map(k_contravariant)
    ward_residual = float(np.linalg.norm(equation @ gauge))
    equation_rank = _rank(equation, tol=tol)
    solution_dimension = equation.shape[1] - equation_rank
    gauge_rank = _rank(gauge, tol=tol)
    quotient_dimension = solution_dimension - gauge_rank

    tt = transverse_traceless_basis(k_contravariant)
    weight = _symmetric_frobenius_weight()
    tt_equation_residual = float(np.linalg.norm(equation @ tt))
    tt_gauge_overlap_residual = float(np.linalg.norm(tt.T @ weight @ gauge))
    tt_gram = tt.T @ weight @ tt
    tt_eigenvalues = np.linalg.eigvalsh(tt_gram)

    scalar_gram = scalar_kinetic_gram(mu_x_over_k_ref)
    scalar_eigenvalues = np.linalg.eigvalsh(scalar_gram)
    scalar_modes_positive = bool(np.all(scalar_eigenvalues > tol))
    scalar_mode_count = int(np.count_nonzero(scalar_eigenvalues > tol))
    total_modes = quotient_dimension + scalar_mode_count
    tree_gate_passed = (
        ward_residual <= tol
        and equation_rank == 4
        and gauge_rank == 4
        and quotient_dimension == 2
        and tt_equation_residual <= tol
        and tt_gauge_overlap_residual <= tol
        and bool(np.all(tt_eigenvalues > tol))
        and total_modes == 7
    )
    return LinearizedGravityBrstAudit(
        k_squared=k_squared,
        ward_residual=ward_residual,
        equation_rank=equation_rank,
        solution_dimension=solution_dimension,
        gauge_rank=gauge_rank,
        quotient_dimension=quotient_dimension,
        tt_equation_residual=tt_equation_residual,
        tt_gauge_overlap_residual=tt_gauge_overlap_residual,
        tt_gram_eigenvalues=tuple(float(value) for value in tt_eigenvalues),
        scalar_gram_eigenvalues=tuple(float(value) for value in scalar_eigenvalues),
        five_scalar_modes_positive=scalar_modes_positive,
        total_free_physical_mode_count=total_modes,
        reference_patch_nondegenerate=False,
        tree_gate_passed=tree_gate_passed,
        loop_anomaly_cohomology_computed=False,
        nonperturbative_m2_passed=False,
    )


def deformed_ward_residual(
    epsilon: float,
    k_contravariant: np.ndarray | None = None,
) -> float:
    '''Return the Ward failure caused by adding epsilon times the identity.'''

    if k_contravariant is None:
        k_contravariant = np.array([1.0, 0.0, 0.0, 1.0])
    equation = linearized_einstein_operator(k_contravariant)
    gauge = linearized_diffeomorphism_map(k_contravariant)
    deformed = equation + float(epsilon) * np.eye(equation.shape[0])
    return float(np.linalg.norm(deformed @ gauge))
