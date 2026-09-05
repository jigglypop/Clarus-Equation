"""경계 쌍의 실제 재사용, 제외 지지와 원뿔 충전의 범위를 검산한다."""

import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "verify/Q-0020/regge_boundary_pushforward.py"
SPEC = importlib.util.spec_from_file_location("ce_boundary_pair_checks", SOURCE)
model = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(model)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def domain(report):
    return model.base.Domain(report["cases"][0]["boundary"])


def unpack(values):
    a = np.asarray(values)
    return a[..., 0]+1j*a[..., 1]


def test_no_new_vertex_fillings_without_edge_01_are_exhausted():
    possible = [c for c in combinations(range(6), 5) if not {0, 1}.issubset(c)]
    assert possible == [(0, 2, 3, 4, 5), (1, 2, 3, 4, 5)]
    boundary_edges = {e for t in model.FACETS for e in combinations(t, 2)}
    assert boundary_edges == set(model.EDGES)
    assert len(model.FACETS) == 8 and len(model.EDGES) == 14
    for face in model.FACETS:
        fillings = [c for c in possible if set(face).issubset(c)]
        assert len(fillings) == 1
    assert model.boundary_facets(model.base.COARSE) == set(model.FACETS)
    assert len(model.CONE.cells) == 8
    assert model.boundary_facets(model.CONE) == set(model.FACETS)


@pytest.mark.parametrize("kind", ["length", "squared"])
def test_boundary_pair_is_nonzero_where_two_simplex_fill_is_invalid(domain, kind):
    f = .7
    lower, upper = model.coarse_interval(domain)
    assert f < lower < upper
    values = domain.lengths(2/3, f)
    assert model.base.WHOLE.evaluate(values)["minimum_gram_eigenvalue"] > 0
    with pytest.raises(ValueError):
        model.base.COARSE.evaluate(values[model.base.WHOLE.indices(model.base.COARSE.edges)])
    result = model.pair_at(domain, f, (0, 5), order=128)[kind]
    assert result["mass"] > .3
    assert result["amplitude"][0] == pytest.approx(result["mass"], abs=1e-13)
    assert abs(result["amplitude"][1]) > .25
    assert np.max(abs(result["amplitude"])) <= result["mass"]+1e-13


def test_serialized_pair_reuses_old_amplitude_without_recomputing_it(report, monkeypatch):
    table = model.PairTable.from_json(json.dumps(report["cases"][0]["table"]))
    def forbidden(*args, **kwargs):
        raise AssertionError("저장한 경계 쌍을 읽은 뒤 기존 내부 작용을 재계산했다")
    monkeypatch.setattr(model.base, "OLD_ACTION", forbidden)
    current = model.PairTable.from_json(table.to_json()).integrate()
    for name in ("length", "squared"):
        saved = report["cases"][0]["results"][name]
        np.testing.assert_allclose(current[name]["kernel"], unpack(saved["kernel"]), atol=1e-12, rtol=0)
        assert current[name]["mass"] == pytest.approx(saved["mass"], abs=1e-12)


def test_interpolated_pair_agrees_at_independent_interior_points(report, domain):
    table = model.PairTable.from_json(json.dumps(report["cases"][0]["table"]))
    for index, segment in enumerate(table.segments):
        x = np.array([-.77, -.13, .41, .86])
        angle = (x+1)*np.pi/4
        f = segment["lower"]+(segment["upper"]-segment["lower"])*np.sin(angle)**2
        direct = model.pair_at(domain, f, table.betas, order=128)
        for name in ("length", "squared"):
            mass, amplitude = table.evaluate_segment(index, x, name)
            np.testing.assert_allclose(mass, direct[name]["mass"], atol=1e-11, rtol=0)
            np.testing.assert_allclose(amplitude, direct[name]["amplitude"], atol=1e-8, rtol=0)
            assert np.max(abs(amplitude)-mass[:, None]) < 1e-11


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_removed_support_has_positive_mass_and_changes_normalized_kernel(report, case_index):
    case = report["cases"][case_index]
    for name in ("length", "squared"):
        row = case["results"][name]
        p = row["coarse_cut_fraction"]
        kernel, kept = unpack(row["kernel"]), unpack(row["coarse_cut_kernel"])
        excluded = unpack(row["coarse_cut_amplitude"])
        assert .02 < p < .06
        np.testing.assert_allclose(kernel, (1-p)*kept+excluded, atol=1e-12, rtol=0)
        assert kernel[0] == pytest.approx(1, abs=1e-12)
        assert kept[0] == pytest.approx(1, abs=1e-12)
        assert excluded[0] == pytest.approx(p, abs=1e-12)
        assert abs(kernel[2]-kept[2]) > .007
        assert np.max(abs(kernel-kept)) <= 2*p+1e-12
        assert np.max(abs(excluded)) <= p+1e-12


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_same_boundary_metric_cones_exist_with_distinct_actions(report, case_index):
    d = model.base.Domain(report["cases"][case_index]["boundary"])
    cones = [model.cone_completion(d, .7, c) for c in (1, 4)]
    for current, saved in zip(cones, report["cases"][case_index]["invalid_coarse_sample"]["cones"]):
        assert current["boundary_preserved"]
        assert current["minimum_gram_eigenvalue"] > .003
        assert current["radius_squared"]-current["maximum_circumradius_squared"] == pytest.approx(current["clearance"])
        assert current["action"] == pytest.approx(saved["action"], abs=1e-11)
    assert abs(cones[0]["action"]-cones[1]["action"]) > 10


def test_cone_construction_handles_other_admissible_boundary_slices(domain):
    a, b = np.sqrt(domain.projections[1])
    for f in (a+.17*(b-a), a+.82*(b-a)):
        current = model.cone_completion(domain, f)
        assert current["minimum_gram_eigenvalue"] > 0
        assert current["boundary_preserved"]


def test_artifact_tracks_source_and_separates_numerical_error_sources(report):
    assert report["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, expected in report["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == expected
    for case in report["cases"]:
        for row in case["results"].values():
            assert row["direct_error"] < 1e-6
            assert row["table_order_error"] < 1e-6
            assert row["interpolation_order_error"] < 1e-6
            assert row["inner_order_error"] < 1e-6
            assert row["mass_error"] < 1e-10
            assert row["dominance_error"] < 1e-10
    assert "공통 계량" in " ".join(report["unfinished"])


@pytest.mark.parametrize("clearance", [0, -1, float("nan")])
def test_cone_rejects_nonpositive_or_nonfinite_clearance(domain, clearance):
    with pytest.raises(ValueError):
        model.cone_completion(domain, .7, clearance)


def test_pair_rejects_outside_support_and_invalid_phase(domain):
    with pytest.raises(ValueError):
        model.pair_at(domain, .1)
    with pytest.raises(ValueError):
        model.pair_at(domain, .7, [])
    with pytest.raises(ValueError):
        model.pair_at(domain, .7, [float("nan")])
    with pytest.raises(ValueError):
        model.PairTable.build(domain, degree=True)
