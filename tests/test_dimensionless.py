from __future__ import annotations

import importlib.util
import math
from fractions import Fraction
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLARUS_DIR = ROOT / "reality_stone" / "python" / "reality_stone" / "clarus"


def _load_standalone_module(name: str, filename: str):
    """Load a math gate without importing the torch-backed package facade."""

    spec = importlib.util.spec_from_file_location(name, CLARUS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


_dimensionless = _load_standalone_module("ce_dimensionless_math", "dimensionless.py")
HAS_SYMPY = importlib.util.find_spec("sympy") is not None
_checker = (
    _load_standalone_module("ce_dimensionless_checker", "dimensionless_checker.py")
    if HAS_SYMPY
    else None
)
requires_sympy = pytest.mark.skipif(
    not HAS_SYMPY,
    reason="dimensionless formula registry validation requires sympy",
)

CURVATURE = _dimensionless.CURVATURE
DIMENSIONLESS = _dimensionless.DIMENSIONLESS
LENGTH = _dimensionless.LENGTH
Quantity = _dimensionless.Quantity
audit_dimensionless = _dimensionless.audit_dimensionless
buckingham_pi_groups = _dimensionless.buckingham_pi_groups
check_dimensionless = _dimensionless.check_dimensionless
compensate_beta_for_affine_defect = _dimensionless.compensate_beta_for_affine_defect
compensate_beta_for_reference_scale = _dimensionless.compensate_beta_for_reference_scale
dim = _dimensionless.dim
evaluate_group = _dimensionless.evaluate_group
exp_argument = _dimensionless.exp_argument
exp_arguments = _dimensionless.exp_arguments
group_dimension = _dimensionless.group_dimension
linear_equality_defect = _dimensionless.linear_equality_defect
log_equality_defect = _dimensionless.log_equality_defect
mahalanobis_equality_defect = _dimensionless.mahalanobis_equality_defect
nondimensionalize = _dimensionless.nondimensionalize
require_dimensionless = _dimensionless.require_dimensionless
typed_equal = _dimensionless.typed_equal
typed_zero = _dimensionless.typed_zero

Dimension = _checker.Dimension if _checker is not None else None
DimensionVector = _checker.DimensionVector if _checker is not None else None
DimensionlessChecker = _checker.DimensionlessChecker if _checker is not None else None


def test_curvature_must_be_scaled_before_exponential() -> None:
    ricci = Quantity("R", 2.5, CURVATURE)
    length_c = Quantity("L_c", 3.0, LENGTH)

    with pytest.raises(ValueError, match="exponential"):
        exp_argument(ricci)

    r_tilde = nondimensionalize(ricci, [length_c])

    assert r_tilde.dims == DIMENSIONLESS
    assert math.isclose(r_tilde.value, 22.5)
    assert math.isclose(exp_argument(r_tilde), 22.5)


def test_mass_scale_lift_closes_as_ratio() -> None:
    m_phi = Quantity("m_phi", 29.65, dim(1, 0, 0, 0))
    m_p = Quantity("m_p", 938.2720813, dim(1, 0, 0, 0))

    ratio = nondimensionalize(m_phi, [m_p])

    assert ratio.dims == DIMENSIONLESS
    assert math.isclose(ratio.value, 29.65 / 938.2720813)


def test_ba_srm1_synapse_chart_and_kernel_arguments_are_dimensionless() -> None:
    # The checker tracks equality of dimensions; the electrical-current base is
    # not part of CE's four-axis registry, so voltage/resistance use distinct
    # nonzero representatives and are divided only by like-dimension scales.
    voltage = dim(1, 2, -3, 0)
    resistance = dim(1, 2, -2, 0)
    time = dim(0, 0, 1, 0)
    ratios = [
        nondimensionalize(
            Quantity("resting_psp", 8e-4, voltage),
            [Quantity("resting_psp_reference", 1e-3, voltage)],
        ),
        nondimensionalize(
            Quantity("soma_distance", 120e-6, LENGTH),
            [Quantity("metre", 1.0, LENGTH)],
        ),
        nondimensionalize(
            Quantity("input_resistance", 80e6, resistance),
            [Quantity("ohm", 1.0, resistance)],
        ),
        nondimensionalize(
            Quantity("membrane_tau", 18e-3, time),
            [Quantity("second", 1.0, time)],
        ),
        nondimensionalize(
            Quantity("late_pulse_psp", 6e-4, voltage),
            [Quantity("resting_psp_reference", 1e-3, voltage)],
        ),
        Quantity("log_variability", -0.3),
        Quantity("geodesic_over_bandwidth", 1.7),
    ]

    result = audit_dimensionless(ratios, context="BA-SRM1 log/Fisher/kernel core")

    assert result.passed
    assert all(quantity.dimensionless for quantity in result.unwrap())


def test_ba_srm2_event_history_target_and_kernel_are_dimensionless() -> None:
    voltage = dim(1, 2, -3, 0)
    current = dim(1, 2, -3, 0)
    resistance = dim(1, 2, -2, 0)
    capacitance = dim(-1, -2, 4, 0)
    time = dim(0, 0, 1, 0)
    temperature = dim(0, 0, 0, 1)

    ratios = [
        nondimensionalize(
            Quantity("pulse_interval", 20e-3, time),
            [Quantity("T0", 1e-3, time)],
        ),
        nondimensionalize(
            Quantity("ic_response", 8e-4, voltage),
            [Quantity("V0", 1e-3, voltage)],
        ),
        nondimensionalize(
            Quantity("stimulus_current", 50e-12, current),
            [Quantity("I0", 1e-12, current)],
        ),
        nondimensionalize(
            Quantity("input_resistance", 80e6, resistance),
            [Quantity("R0", 1e6, resistance)],
        ),
        nondimensionalize(
            Quantity("capacitance", 35e-12, capacitance),
            [Quantity("C0", 1e-12, capacitance)],
        ),
        nondimensionalize(
            Quantity("soma_distance", 120e-6, LENGTH),
            [Quantity("L0", 100e-6, LENGTH)],
        ),
        nondimensionalize(
            Quantity("bath_temperature", 307.0, temperature),
            [Quantity("Theta0", 310.0, temperature)],
        ),
        Quantity("pulse_count", 8.0),
        Quantity("pullback_line_element_sq", 1.4),
        Quantity("kernel_bandwidth_sq", 2.0),
    ]

    result = audit_dimensionless(ratios, context="BA-SRM2 event/Fisher/kernel core")

    assert result.passed
    assert all(quantity.dimensionless for quantity in result.unwrap())


def test_buckingham_pi_finds_reynolds_number_shape() -> None:
    rho = Quantity("rho", 1.2, dim(1, -3, 0, 0))
    velocity = Quantity("v", 3.0, dim(0, 1, -1, 0))
    length = Quantity("L", 2.0, dim(0, 1, 0, 0))
    viscosity = Quantity("mu", 1.8e-5, dim(1, -1, -1, 0))

    groups = buckingham_pi_groups([rho, velocity, length, viscosity])

    assert groups == [
        {"rho": Fraction(-1, 1), "v": Fraction(-1, 1), "L": Fraction(-1, 1), "mu": Fraction(1, 1)}
    ]
    assert group_dimension([rho, velocity, length, viscosity], groups[0]) == DIMENSIONLESS
    assert math.isclose(
        evaluate_group([rho, velocity, length, viscosity], groups[0]),
        1 / (1.2 * 3.0 * 2.0 / 1.8e-5),
    )


def test_dimensionless_guard_accepts_ce_core_ratio() -> None:
    epsilon2 = Quantity("epsilon^2", 0.04865)

    assert require_dimensionless(epsilon2).value == 0.04865


def test_dimensionless_gate_result_composes_value_transform() -> None:
    epsilon2 = Quantity("epsilon^2", 0.25)

    result = (
        check_dimensionless(epsilon2)
        .map(lambda q: q.value)
        .bind(lambda value: check_dimensionless(Quantity("sqrt_epsilon2", math.sqrt(value))))
    )

    assert result.passed
    assert math.isclose(result.unwrap().value, 0.5)


def test_dimensionless_audit_accumulates_all_failures() -> None:
    checks = [
        Quantity("epsilon^2", 0.04865),
        Quantity("R", 2.5, CURVATURE),
        Quantity("L", 3.0, LENGTH),
    ]

    result = audit_dimensionless(checks, context="CE selection gate")

    assert not result.passed
    assert len(result.errors) == 2
    assert "R must be dimensionless for CE selection gate" in result.errors[0]
    assert "L must be dimensionless for CE selection gate" in result.errors[1]
    with pytest.raises(ValueError, match="CE selection gate"):
        result.unwrap()


def test_exp_arguments_validates_batch_before_kernel_use() -> None:
    args = exp_arguments(
        [
            Quantity("D_eff", 0.31),
            Quantity("phi", 1.7),
        ]
    )

    assert args.passed
    assert args.unwrap() == (0.31, 1.7)


def test_typed_equality_rejects_metre_second_and_preserves_typed_zero() -> None:
    metre = Quantity("one metre", 1.0, LENGTH)
    second = Quantity("one second", 1.0, dim(0, 0, 1, 0))

    with pytest.raises(ValueError, match="same dimensions"):
        typed_equal(metre, second)

    zero_length = typed_zero("zero length", LENGTH)
    assert typed_equal(zero_length, Quantity("origin", 0.0, LENGTH))
    with pytest.raises(ValueError, match="same dimensions"):
        typed_equal(zero_length, Quantity("bare zero", 0.0))
    with pytest.raises(ValueError, match="finite"):
        typed_equal(metre, Quantity("infinite length", math.inf, LENGTH))


def test_equality_defects_enforce_domains_and_common_unit_rescaling() -> None:
    left = Quantity("left", 2.0, LENGTH)
    right = Quantity("right", 1.0, LENGTH)
    scale = Quantity("scale", 4.0, LENGTH)

    assert math.isclose(linear_equality_defect(left, right, scale), 0.25)
    assert math.isclose(
        linear_equality_defect(
            Quantity("left cm", 200.0, LENGTH),
            Quantity("right cm", 100.0, LENGTH),
            Quantity("scale cm", 400.0, LENGTH),
        ),
        0.25,
    )
    assert math.isclose(log_equality_defect(left, right), math.log(2.0))
    assert math.isclose(
        log_equality_defect(Quantity("large", 1e308, LENGTH), Quantity("small", 1e-308, LENGTH)),
        1418.3924172843322,
    )
    adjacent_large = math.nextafter(1e308, math.inf)
    adjacent_forward = log_equality_defect(Quantity("large", 1e308, LENGTH), Quantity("adjacent", adjacent_large, LENGTH))
    adjacent_reverse = log_equality_defect(Quantity("adjacent", adjacent_large, LENGTH), Quantity("large", 1e308, LENGTH))
    assert adjacent_forward > 0.0
    assert adjacent_forward == adjacent_reverse
    with pytest.raises(ValueError, match="positive"):
        log_equality_defect(Quantity("zero", 0.0, LENGTH), right)
    with pytest.raises(ValueError, match="positive reference scale"):
        linear_equality_defect(left, right, Quantity("bad scale", 0.0, LENGTH))
    with pytest.raises(ValueError, match="finite"):
        linear_equality_defect(Quantity("nan", math.nan, LENGTH), right, scale)
    with pytest.raises(ValueError, match="finite"):
        linear_equality_defect(Quantity("large", 1e308, LENGTH), Quantity("negative", -1e308, LENGTH), scale)


def test_mahalanobis_equality_defect_requires_spd_covariance() -> None:
    left = [Quantity("x", 3.0, LENGTH), Quantity("t", 5.0, dim(0, 0, 1, 0))]
    right = [Quantity("x0", 1.0, LENGTH), Quantity("t0", 1.0, dim(0, 0, 1, 0))]
    scales = [Quantity("L", 2.0, LENGTH), Quantity("T", 4.0, dim(0, 0, 1, 0))]

    assert math.isclose(mahalanobis_equality_defect(left, right, scales, [[1.0, 0.0], [0.0, 1.0]]), 2.0)
    for covariance, message in (
        ([[1.0, 0.0], [0.0, -1.0]], "positive definite"),
        ([[1.0, 1.0], [1.0, 1.0]], "positive definite"),
        ([[1.0, 2.0], [0.0, 1.0]], "symmetric"),
    ):
        with pytest.raises(ValueError, match=message):
            mahalanobis_equality_defect(left, right, scales, covariance)
    with pytest.raises(ValueError, match="finite"):
        mahalanobis_equality_defect(
            [Quantity("infinite x", math.inf, LENGTH), left[1]], right, scales, [[1.0, 0.0], [0.0, 1.0]]
        )
    with pytest.raises(ValueError, match="finite"):
        mahalanobis_equality_defect(
            [Quantity("large x", 1e308, LENGTH), left[1]],
            [Quantity("zero x", 0.0, LENGTH), right[1]],
            [Quantity("unit L", 1.0, LENGTH), scales[1]],
            [[1.0, 0.0], [0.0, 1.0]],
        )


def test_beta_compensation_preserves_affine_gibbs_weight_ratios() -> None:
    beta = 1.7
    offset = 3.2
    multiplier = 2.5
    delta_i, delta_j = 0.4, 1.1
    beta_prime = compensate_beta_for_affine_defect(beta, multiplier)

    original_ratio = math.exp(-beta * (delta_i - delta_j))
    transformed_ratio = math.exp(
        -beta_prime * ((offset + multiplier * delta_i) - (offset + multiplier * delta_j))
    )
    assert math.isclose(transformed_ratio, original_ratio)
    assert math.isclose(compensate_beta_for_reference_scale(beta, 4.0), 4.0 * beta)
    with pytest.raises(ValueError, match="finite"):
        compensate_beta_for_affine_defect(1e308, 1e-308)
    with pytest.raises(ValueError, match="finite"):
        compensate_beta_for_reference_scale(1e308, 1e308)


@requires_sympy
def test_checker_preserves_unnamed_inverse_time_dimension() -> None:
    inverse_time = Dimension.TIME**-1

    assert isinstance(inverse_time, DimensionVector)
    assert inverse_time.exponents == tuple(map(Fraction, (0, 0, -1, 0)))
    assert not inverse_time.is_dimensionless()


@requires_sympy
def test_checker_preserves_mass_squared_and_composes_back_to_mass() -> None:
    mass_squared = Dimension.MASS**2

    assert isinstance(mass_squared, DimensionVector)
    assert mass_squared.exponents == tuple(map(Fraction, (2, 0, 0, 0)))
    assert not mass_squared.is_dimensionless()
    assert mass_squared / Dimension.MASS == Dimension.MASS


@requires_sympy
def test_registered_rate_and_magnetic_field_have_nontrivial_dimensions() -> None:
    formulas = {formula.name: formula for formula in DimensionlessChecker().formulas}

    rate = formulas["STDP learning rate upper bound"].expected_dim
    magnetic_field = formulas["Critical magnetic field"].expected_dim

    assert rate == Dimension.TIME**-1
    assert not rate.is_dimensionless()
    assert magnetic_field == Dimension.MASS**2
    assert not magnetic_field.is_dimensionless()


@requires_sympy
def test_clarus_field_gate_and_phase_score_are_registered_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["g_CF"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["chi_CF"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["g_CF"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["chi_CF"])["status"].startswith("PASS")


@requires_sympy
def test_unified_metric_surprise_and_condition_ratio_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["chi_UM"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["kappa_UM"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["chi_UM"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["kappa_UM"])["status"].startswith("PASS")


@requires_sympy
def test_v16_metric_flow_residual_and_regret_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["r_V16"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["rho_V16"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["r_V16"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["rho_V16"])["status"].startswith("PASS")


@requires_sympy
def test_v17_conditional_information_and_lift_margin_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["I_V17"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["delta_V17"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["I_V17"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["delta_V17"])["status"].startswith("PASS")


@requires_sympy
def test_v18b_reward_decoder_and_classifier_increment_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["y_tilde_V18b"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["delta_w_V18b"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["y_tilde_V18b"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["delta_w_V18b"])["status"].startswith("PASS")


@requires_sympy
def test_a4_a5_graph_metric_core_arguments_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    for symbol in ("a_A4", "r_w_A4", "chi_A5"):
        assert formulas[symbol].expected_dim == Dimension.DIMENSIONLESS
        assert checker.check_formula(formulas[symbol])["status"].startswith("PASS")


@requires_sympy
def test_a6_pullback_and_reachability_ratios_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    for symbol in ("s_A6", "Lambda_A6", "delta_logV_A6", "rho_E_A6"):
        assert formulas[symbol].expected_dim == Dimension.DIMENSIONLESS
        assert checker.check_formula(formulas[symbol])["status"].startswith("PASS")
