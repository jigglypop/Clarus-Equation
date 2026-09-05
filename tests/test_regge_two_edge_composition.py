"""실제 일곱 단체의 결합 영역과 두 내부 적분의 전달을 독립 대조한다."""

from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad


SOURCE = Path(__file__).resolve().parents[1] / "verify/Q-0020/regge_two_edge_composition.py"
SPEC = importlib.util.spec_from_file_location("ce_two_edge_checks", SOURCE)
model = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(model)
    from regge_pachner_constraints import boundary_facets, local_increment
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def domain():
    return model.Domain(model.WHOLE.lengths(model.reference_points())[model.BOUNDARY_IDS])


@pytest.fixture(scope="module")
def forward(domain):
    return model.integrate(domain, (0, 1, 5), order=64)


def test_actual_gluing_moves_and_two_internal_edges():
    before = model.OLD
    for cell, glued_count in zip(model.ADDED, (1, 2, 3)):
        facets = {tuple(sorted(f)) for f in combinations(cell, 4)}
        assert len(facets & boundary_facets(before)) == glued_count
        after = model.ReggeComplex(before.cells+(cell,))
        assert boundary_facets(after) == boundary_facets(before) ^ facets
        before = after
    facets = boundary_facets(model.WHOLE)
    edges = {tuple(e) for facet in facets for e in combinations(facet, 2)}
    assert len(model.WHOLE.edges) == 20
    assert len(edges) == 18
    assert set(model.WHOLE.edges)-edges == set(model.INTERNAL)
    assert len(facets) == 11
    for edge in model.INTERNAL:
        link = [tuple(sorted(set(cell)-set(edge))) for cell in model.WHOLE.cells
                if set(edge).issubset(cell)]
        link_edges = Counter(e for face in link for e in combinations(face, 2))
        assert set(link_edges.values()) == {2}
        vertices = set().union(*map(set, link))
        assert len(vertices)-len(link_edges)+len(link) == 2


@pytest.mark.parametrize("case", [0, 1, 2])
def test_domain_fibers_match_independent_simplex_gram_geometry(case, report):
    d = model.Domain(report["cases"][case]["boundary"])
    for f in np.linspace(*np.sqrt(d.projections[1]), 6)[1:-1]:
        a, b = d.fiber(f, 1)
        for e in (a+.15*(b-a), a+.8*(b-a)):
            assert d.contains(e, f)
            values = d.lengths(e, f)
            actual = model.WHOLE.evaluate(values)
            assert actual["minimum_gram_eigenvalue"] > 0
            assert model.WHOLE_ACTION(values) == pytest.approx(actual["action"], abs=1e-11)
        for e in (max(a*.9, 1e-8), b*1.01):
            if e < a or e > b:
                assert not d.contains(e, f)
                with pytest.raises(ValueError):
                    model.WHOLE.evaluate(d.lengths(e, f))


def test_support_counterexample_has_exact_positive_fine_minors(domain, report):
    certificate = report["exact_certificate"]["support_witness"]
    for minors in certificate["fine_leading_minors"]:
        assert len(minors) == 4
        assert all(Fraction(x) > 0 for x in minors)
    assert Fraction(certificate["coarse_determinant"]) == Fraction(-1132, 5625)
    assert domain.contains(2/3, 7/10)
    values = domain.lengths(2/3, 7/10)
    assert model.WHOLE.evaluate(values)["minimum_gram_eigenvalue"] > .02
    coarse_values = values[model.WHOLE.indices(model.COARSE.edges)]
    with pytest.raises(ValueError):
        model.COARSE.evaluate(coarse_values)
    assert 7/10 < (8-math.sqrt(13))/6


def test_corner_correction_and_sequential_local_actions(domain):
    expected = {(0, 3, 4), (0, 3, 5), (0, 4, 5), (2, 3, 4), (2, 3, 5), (2, 4, 5)}
    assert {t for t, c in zip(model.SHELL.triangles, model.CORNER) if c} == expected
    assert all(c in (-1, 0) for c in model.CORNER)
    values = domain.lengths(2/3, 7/10)
    before, total = model.OLD, 0
    for cell in model.ADDED:
        after = model.ReggeComplex(before.cells+(cell,))
        total += local_increment(before, after, cell, values[model.WHOLE.indices(after.edges)])["action"]
        before = after
    assert total == pytest.approx(model.increment(domain, .7), abs=1e-12)
    for e in (.64, .68):
        lengths = domain.lengths(e, .7)
        assert model.WHOLE_ACTION(lengths)-model.OLD_ACTION(lengths) == pytest.approx(total, abs=1e-12)


def test_breakpoints_split_tangent_length_boundary_without_changing_domain(domain):
    breaks = domain.breakpoints(1)
    assert np.all(np.diff(breaks) > 0)
    assert min(abs(breaks-math.sqrt(5)/2)) < 1e-10
    assert breaks[[0, -1]] == pytest.approx(np.sqrt(domain.projections[1]))
    for a, b in zip(breaks, breaks[1:]):
        f = (a+b)/2
        left, right = domain.fiber(f, 1)
        assert domain.contains((left+right)/2, f)


@pytest.mark.parametrize("kind", ["length", "squared"])
def test_transmitted_conditional_kernel_and_marginal_reproduce_full_integral(kind, forward):
    result = forward[kind]
    transport = result["transport"]
    assert result["kernel"][0] == pytest.approx(1, abs=2e-13)
    assert transport["marginal_weights"].sum() == pytest.approx(1, abs=2e-13)
    assert np.max(abs(transport["old_kernel"])) <= 1+2e-13
    transmitted = np.sum(transport["marginal_weights"][:, None]
                         *transport["old_kernel"]*transport["increment_phase"], axis=0)
    np.testing.assert_allclose(transmitted, result["kernel"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(result["staged_kernel"], result["kernel"], atol=2e-12, rtol=0)
    assert abs(result["kernel"][2]-result["wrong_reset_kernel"][2]) > .01


def test_reverse_integration_preserves_joint_measure_and_keeps_increment_inside(domain, forward):
    reverse = model.integrate(domain, (0, 1, 5), order=64, outer_axis=0)
    for name in ("length", "squared"):
        np.testing.assert_allclose(forward[name]["kernel"], reverse[name]["kernel"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(reverse[name]["staged_kernel"], reverse[name]["kernel"], atol=2e-12, rtol=0)
        assert abs(forward[name]["wrong_reset_kernel"][2]-reverse[name]["wrong_reset_kernel"][2]) > .01


def test_squared_coordinate_jacobian_and_two_distinct_supplied_measures(domain, forward):
    breaks = domain.breakpoints(1)**2
    def width(F):
        lower, upper = domain.fiber_squared(F, 1)
        return float(upper-lower)
    squared_volume = sum(quad(width, a, b, epsabs=1e-11, epsrel=1e-11)[0]
                         for a, b in zip(breaks, breaks[1:]))
    assert forward["squared"]["volume"] == pytest.approx(squared_volume, abs=1e-9)
    assert abs(forward["squared"]["kernel"][2]-forward["length"]["kernel"][2]) > .1


def test_saved_general_boundary_quadrature_and_actual_source(report):
    assert report["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, digest in report["dependencies"].items():
        assert digest == hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()
    assert report["exact_certificate"] == model.exact_certificate()
    assert len(report["cases"]) == 3
    for case in report["cases"]:
        assert case["batch_scalar_error"] < 1e-9
        assert case["action_identity_error"] < 1e-9
        assert all(v > 0 for v in case["volumes"].values())
        for row in case["rows"]:
            assert row["order_difference"] < 1e-6
            assert row["elimination_order_difference"] < 1e-6
            assert row["staged_direct_difference"] < 1e-10
            assert abs(complex(*row["kernel"])) <= 1+1e-12
        for transport in case["transport"].values():
            assert transport["marginal_mass"] == pytest.approx(1, abs=1e-12)
            assert transport["maximum_conditional_modulus"] <= 1+1e-12
    assert "공통 계량" in " ".join(report["unfinished"])


@pytest.mark.parametrize("boundary", [np.ones(17), np.zeros(18), np.full(18, np.nan)])
def test_invalid_boundary_is_rejected(boundary):
    with pytest.raises(ValueError):
        model.Domain(boundary)


@pytest.mark.parametrize("betas", [[], [math.nan], [[1]]])
def test_invalid_phase_sequence_is_rejected(domain, betas):
    with pytest.raises(ValueError):
        model.integrate(domain, betas, order=16)


def test_outside_projection_and_invalid_quadrature_are_rejected(domain):
    with pytest.raises(ValueError):
        domain.fiber(10, 1)
    for order in (True, 15, 16.5):
        with pytest.raises(ValueError):
            model.integrate(domain, order=order)
    with pytest.raises(ValueError):
        model.integrate(domain, outer_axis=2)


def test_negative_real_phase_is_the_conjugate(domain):
    result = model.integrate(domain, (-1, 1), order=16, staged=False)
    for kind in ("length", "squared"):
        assert result[kind]["kernel"][0] == pytest.approx(result[kind]["kernel"][1].conjugate(), abs=1e-13)
