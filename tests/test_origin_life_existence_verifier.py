from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

from reality_stone.clarus.origin_life_existence_verifier import (
    independently_verified,
    verify_existence_certificate,
)


def _q(value: Fraction | int) -> dict[str, str | float]:
    value = Fraction(value)
    return {"exact": str(value), "decimal": float(value)}


def _certificate() -> dict[str, object]:
    witness_rows = []
    for q in (Fraction(1, 4), Fraction(3, 4)):
        state = [_q(Fraction(1, 2)), _q(Fraction(1, 2)), _q(q)]
        witness_rows.append(
            {
                "state": state,
                "predivision_mass": _q(1),
                "division_triggered": True,
                "next_state": deepcopy(state),
                "exact_fixed_point": True,
            }
        )
    return {
        "artifact_type": "clarus_primitive_lineage_exact_existence_certificate",
        "artifact_version": 5,
        "parameters": {
            "growth": "9/2",
            "leak": "5/2",
            "boundary_production": "1/5",
            "boundary_decay": "1/10",
            "copy_selection": "1/2",
            "mutation": "3/32",
            "inheritance_gain": "1",
            "division_threshold": "3/4",
            "capacity": "1",
        },
        "witnesses": witness_rows,
        "invariant_box": {
            "passed": True,
            "predivision_mass_upper": _q(Fraction(121, 72)),
            "mass_after_step_upper": _q(Fraction(121, 144)),
            "boundary_after_step_upper": _q(Fraction(9, 10)),
            "heredity_derivative_minimum": _q(Fraction(5, 16)),
            "heredity_image_endpoints": [_q(Fraction(3, 32)), _q(Fraction(29, 32))],
        },
        "local_stability": {
            "passed": True,
            "mass_boundary_jacobian": [
                [_q(Fraction(-1, 8)), _q(Fraction(5, 8))],
                [_q(Fraction(1, 10)), _q(Fraction(4, 5))],
            ],
            "trace": _q(Fraction(27, 40)),
            "determinant": _q(Fraction(-13, 80)),
            "jury_schur_conditions": {
                "1_minus_trace_plus_determinant": _q(Fraction(13, 80)),
                "1_plus_trace_plus_determinant": _q(Fraction(121, 80)),
                "1_minus_determinant": _q(Fraction(93, 80)),
            },
            "lineage_fixed_point_multiplier": _q(Fraction(7, 8)),
            "central_fixed_point_multiplier": _q(Fraction(17, 16)),
            "det_I_minus_full_jacobian": _q(Fraction(13, 640)),
        },
        "global_heredity_dynamics": {
            "passed": True,
            "fixed_points": [_q(Fraction(1, 4)), _q(Fraction(1, 2)), _q(Fraction(3, 4))],
            "stable_lineage_count": 2,
        },
        "fixed_point_classification": {
            "passed": True,
            "divided_mass_polynomial_coefficients": [
                _q(Fraction(-9, 10)),
                _q(Fraction(1, 4)),
                _q(Fraction(1, 10)),
            ],
            "divided_mass_roots": [_q(Fraction(-2, 9)), _q(Fraction(1, 2))],
            "no_positive_nondivided_fixed_point_below_threshold": True,
            "positive_reproductive_fixed_states": 3,
            "locally_stable_reproductive_fixed_states": 2,
            "full_fixed_state_count": 6,
        },
        "certified_reproductive_basin": {
            "passed": True,
            "mass_boundary_rectangle": [_q(Fraction(49, 100)), _q(Fraction(51, 100))],
            "division_every_generation": True,
            "predivision_mass_bounds": [
                _q(Fraction(9843, 10000)),
                _q(Fraction(10143, 10000)),
            ],
            "postdivision_mass_bounds": [
                _q(Fraction(9843, 20000)),
                _q(Fraction(10143, 20000)),
            ],
            "boundary_image_bounds": [
                _q(Fraction(24549, 50000)),
                _q(Fraction(25449, 50000)),
            ],
            "infinity_norm_contraction_bound": _q(Fraction(113, 125)),
        },
        "single_term_ablation_lemmas": {
            "no_autocatalysis": {"passed": True},
            "no_boundary_production": {
                "passed": True,
                "generations_until_boundary_at_most_one_quarter": 14,
                "post_decay_maximum_predivision_mass": _q(Fraction(841, 1152)),
            },
            "no_inheritance": {"passed": True, "one_step_heredity": _q(Fraction(1, 2))},
            "no_bistabilizing_selection": {"passed": True},
        },
        "conditional_core_ablation_suite_passed": True,
        "branching_reproduction": {
            "passed": True,
            "daughter_count": 2,
            "descendants_after_n_generations": "2^n",
        },
        "base_existence_theorem_proven": True,
        "universal_necessity_proven": False,
        "empirical_autonomous_protocell_proven": False,
        "genotype_phenotype_coupling_proven": False,
        "endogenous_evolution_proven": False,
    }


def test_independent_verifier_recomputes_every_exact_obligation() -> None:
    report = verify_existence_certificate(_certificate())

    assert report.verified, report.errors
    assert len(report.checks) == 11


def test_independent_verifier_rejects_derived_field_tampering() -> None:
    certificate = _certificate()
    certificate["local_stability"]["jury_schur_conditions"][
        "1_minus_trace_plus_determinant"
    ] = _q(Fraction(1, 5))

    report = verify_existence_certificate(certificate)
    assert not report.verified
    assert any("Jury" in error or "jury" in error for error in report.errors)


def test_independent_verifier_rejects_flags_that_hide_wrong_equations() -> None:
    certificate = _certificate()
    certificate["parameters"]["inheritance_gain"] = "0"

    assert not independently_verified(certificate)


def test_independent_verifier_fails_closed_on_missing_scope_guard() -> None:
    certificate = _certificate()
    del certificate["endogenous_evolution_proven"]

    report = verify_existence_certificate(certificate)
    assert not report.verified
    assert any("endogenous_evolution_proven" in error for error in report.errors)
