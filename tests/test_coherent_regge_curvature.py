"""정확한 LS 커널과 실제 Regge 위상을 검사한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"
original = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("coherent_regge_under_test", HERE/"coherent_regge_curvature.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = original


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_reduced_ls_formula_matches_all_complex_cg_components(n):
    for c in (.13, 1/math.sqrt(3), .83):
        value, log_norm = module.symmetric_state(n, c, normalize=False)
        expected, norm = module.coherent.intertwiner(
            n, module.coherent.shape_normals(math.acos(c), math.pi/2), normalize=False)
        np.testing.assert_allclose(value, expected, atol=2e-13, rtol=2e-12)
        assert math.exp(log_norm) == pytest.approx(norm, abs=2e-13)


def test_logarithmic_high_spin_states_are_normalized_and_keep_relative_phases():
    value, log_norm = module.symmetric_state(1999, .74)
    assert math.isfinite(log_norm)
    assert np.vdot(value, value).real == pytest.approx(1., abs=1e-10)
    support = np.flatnonzero(abs(value[:-1])*abs(value[1:]) > 1e-12)
    assert len(support) > 1
    for k in support[:5]:
        ratio = value[k+1]/value[k]
        assert abs(ratio.real) < 1e-10
        assert ratio.imag < 0


@pytest.mark.parametrize("a", [.6, 1., 1.28, 1.49])
def test_actual_simplex_assembly_and_schlafli_identity(a):
    row = module.direct_geometry(a, 1.)
    assert (row["triangles"], row["triangle_incidences"], row["boundary_tetrahedra"]) == (19, 30, 9)
    assert row["minimum_cell_gram_eigenvalue"] > 0
    for key in ("action_residual", "deficit_residual", "schlafli_chain_rule_residual", "boundary_phi_residual"):
        assert row[key] < 1e-9
    h = 1e-5
    numerical = (module.geometric_functions(a+h, 1.)[0]-module.geometric_functions(a-h, 1.)[0])/(2*h)
    assert numerical == pytest.approx(module.geometric_functions(a, 1.)[1], abs=1e-7)


def test_regular_incidence_action_detects_the_missing_angular_term():
    theta = math.acos(.25)
    correct = module.geometric_functions(1., 1.)[0]
    assert correct == pytest.approx(20*math.pi-30*theta, abs=1e-12)
    printed_reduction = 20*math.pi-21*theta
    assert printed_reduction-correct == pytest.approx(9*theta, abs=1e-12)


@pytest.mark.parametrize("a,b", [(0., 1.), (1.5, 1.), (1.6, 1.), (1., 0.), (math.nan, 1.)])
def test_invalid_four_simplex_branch_is_rejected(a, b):
    with pytest.raises(ValueError):
        module.geometric_functions(a, b)


@pytest.mark.parametrize("n", [19, 199])
def test_bulk_labels_obey_su2_parity_and_complete_geometric_domain(n):
    data = module.kernel_data(n)
    labels = data["labels"]
    assert np.all((labels+3*n) % 2 == 0)
    assert np.all(labels <= 3*n)
    assert np.all(np.diff(labels) == 2)
    assert np.all(data["areas"] < 1.5*data["boundary_area"])
    next_area = module.area_spectrum(labels[-1]+2)
    assert next_area >= 1.5*data["boundary_area"]
    assert data["log_gluing"][data["reference_index"]] == pytest.approx(0., abs=1e-10)
    assert max(data["log_gluing"]) <= 0


def test_kernel_uses_nine_overlap_amplitudes_not_nine_squared_fidelities():
    data = module.kernel_data(19)
    indices = [data["reference_index"], data["reference_index"]-1]
    c = data["areas"][indices]/(math.sqrt(3)*data["boundary_area"])
    states = [module.symmetric_state(19, value)[0] for value in c]
    overlap = np.vdot(states[0], states[1])
    assert abs(overlap.imag) < 1e-12
    assert 0 < overlap.real < 1
    assert data["log_gluing"][indices[1]] == pytest.approx(9*math.log(overlap.real), abs=1e-10)


def test_phase_off_control_and_complex_ratio_are_separate_from_squared_moduli():
    off = module.amplitude_sum(199, 0.)
    on = module.amplitude_sum(199, .1)
    assert off["cancellation_ratio"] == pytest.approx(1., abs=1e-12)
    assert off["complex_curvature_expectation"][0] == pytest.approx(off["phase_off_envelope_curvature"], abs=1e-12)
    assert abs(off["complex_curvature_expectation"][1]) < 1e-12
    assert on["denominator_resolved"]
    z = complex(*on["scaled_partition"])
    numerator = complex(*on["scaled_curvature_numerator"])
    assert complex(*on["complex_curvature_expectation"]) == pytest.approx(numerator/z, abs=1e-12)
    assert on["cancellation_ratio"] < 1.
    assert abs(on["complex_curvature_expectation"][1]) > 1e-6
    assert on["labelwise_squared_modulus_curvature"] == pytest.approx(
        off["labelwise_squared_modulus_curvature"], abs=1e-12)


def test_measure_and_spectrum_are_explicit_model_choices():
    base = module.amplitude_sum(199, .1)
    weighted = module.amplitude_sum(199, .1, measure="dimension")
    linear = module.amplitude_sum(199, .1, spectrum="linear")
    assert weighted["measure"] == "dimension"
    assert linear["spectrum"] == "linear"
    assert weighted["phase_off_envelope_curvature"] != base["phase_off_envelope_curvature"]
    assert linear["actual_reference_area_ratio"] != base["actual_reference_area_ratio"]


def test_zero_spin_is_filtered_by_declared_area_spectrum_after_su2_admissibility():
    linear = module.kernel_data(20, spectrum="linear")
    casimir = module.kernel_data(20, spectrum="casimir")
    odd = module.kernel_data(19, spectrum="linear")
    assert linear["labels"][0] == 0
    assert linear["areas"][0] == .5
    assert casimir["labels"][0] == 2
    assert 0 not in odd["labels"]


@pytest.mark.parametrize("label", [-1, .5, math.nan, math.inf])
def test_invalid_twice_spin_area_labels_are_rejected(label):
    with pytest.raises(ValueError):
        module.area_spectrum(label)


def test_reference_snap_is_visible_in_reported_inputs():
    row = module.amplitude_sum(19, 0., reference_ratio=1.28)
    assert row["requested_reference_area_ratio"] == 1.28
    assert row["reference_area_ratio_snap_error"] == pytest.approx(
        row["actual_reference_area_ratio"]-1.28, abs=1e-15)



def test_local_saddle_width_tracks_the_exact_high_spin_envelope():
    middle = module.amplitude_sum(199, 0.)
    fine = module.amplitude_sum(1999, 0.)
    middle_error = abs(middle["envelope_width_relative_to_leading"]-1)
    fine_error = abs(fine["envelope_width_relative_to_leading"]-1)
    assert fine_error < middle_error
    assert fine_error < 1e-3


def test_joint_scaling_keeps_finite_phase_cancellation_and_curved_signal():
    coupling = 3.
    middle = module.amplitude_sum(199, coupling/math.sqrt(199/2))
    fine = module.amplitude_sum(1999, coupling/math.sqrt(1999/2))
    for row in (middle, fine):
        assert row["gamma_sqrt_boundary_spin"] == pytest.approx(coupling, abs=1e-12)
        assert row["denominator_resolved"]
        assert .85 < row["cancellation_ratio"] < .95
        assert row["complex_curvature_expectation"][0] > .5
    assert abs(fine["complex_curvature_expectation"][1]) < abs(middle["complex_curvature_expectation"][1])


@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("gamma", [0., .5])
@pytest.mark.parametrize("measure", ["unit", "dimension"])
def test_two_bulk_sum_matches_independent_complex_intertwiner_double_sum(n, gamma, measure):
    data = module.pair_kernel_data(n)
    areas, deficits = data["areas"], data["deficit"]
    states = [module.coherent.intertwiner(
        n, module.coherent.shape_normals(math.acos(a/(math.sqrt(3)*data["boundary_area"])), math.pi/2))[0]
        for a in areas]
    mu = np.ones(len(areas)) if measure == "unit" else data["labels"]+1.
    mu /= mu.max()
    phase = gamma*(data["action"]-data["action"][np.argmin(abs(deficits))])
    total = numerator = mismatch = opposite = 0j
    envelope = envelope_mismatch = 0.
    for i, left in enumerate(states):
        for k, right in enumerate(states):
            weight = mu[i]*mu[k]*np.vdot(left, right)**9
            amplitude = weight*np.exp(1j*(phase[i]+phase[k]))
            delta2 = ((areas[i]-areas[k])/data["boundary_area"])**2
            total += amplitude
            numerator += amplitude*(deficits[i]+deficits[k])/2
            mismatch += amplitude*delta2
            opposite += weight*np.exp(1j*(phase[k]-phase[i]))
            envelope += weight.real
            envelope_mismatch += weight.real*delta2
    row = module.pair_sum(n, gamma, measure=measure)
    assert complex(*row["scaled_partition"]) == pytest.approx(total, abs=2e-12)
    assert complex(*row["scaled_curvature_numerator"]) == pytest.approx(numerator, abs=3e-12)
    assert complex(*row["complex_mean_curvature"]) == pytest.approx(numerator/total, abs=2e-12)
    assert complex(*row["complex_relative_area_mismatch"]) == pytest.approx(mismatch/total, abs=2e-12)
    assert row["phase_off_relative_area_mismatch"] == pytest.approx(envelope_mismatch/envelope, abs=1e-12)
    assert complex(*row["opposite_orientation_scaled_partition"]) == pytest.approx(opposite, abs=2e-12)
    assert opposite.real >= -1e-12
    assert abs(opposite.imag) < 1e-12
    if gamma:
        assert abs(total-opposite) > 1e-3


def test_two_bulk_kernel_contains_the_previous_conditional_columns():
    pair = module.pair_kernel_data(19)
    for ratio in (.9, 1.28):
        conditional = module.kernel_data(19, ratio)
        np.testing.assert_allclose(pair["kernel"][:, conditional["reference_index"]],
                                   np.exp(conditional["log_gluing"]), atol=1e-12)
    np.testing.assert_allclose(pair["kernel"], pair["kernel"].T, atol=1e-14)
    assert np.linalg.eigvalsh(pair["kernel"])[0] >= -1e-12


def test_phase_off_pair_separates_shape_agreement_from_common_shape_selection():
    row = module.pair_sum(199, 0.)
    assert row["cancellation_ratio"] == pytest.approx(1., abs=1e-12)
    assert row["complex_mean_curvature"][0] == pytest.approx(row["phase_off_mean_curvature"], abs=1e-12)
    assert abs(row["complex_mean_curvature"][1]) < 1e-12
    assert row["phase_off_relative_area_mismatch"] > 0
    assert row["phase_off_common_area_ratio_variance"] > 0
    assert row["common_face_areas_fixed"]
    assert not row["second_bulk_label_fixed"]
    assert row["bulk_label_pairs"] == row["bulk_label_count_per_side"]**2


@pytest.mark.parametrize("gamma", [-1., math.inf, math.nan])
def test_invalid_pair_phase_is_rejected(gamma):
    with pytest.raises(ValueError):
        module.pair_sum(19, gamma)


def test_saved_sources_and_scope_are_current():
    report = json.loads((HERE/"coherent_regge_curvature.json").read_text(encoding="utf-8"))
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((HERE/path).read_bytes()).hexdigest() == digest
    for key in ("general_area_to_length_branches_exhausted", "complex_expectation_is_probability_average",
                "source_gaussian_tables_used_as_validation_target", "microscopic_ce_dynamics_derived",
                "uniform_exact_sum_joint_limit_proven", "gamma_running_law_physically_derived",
                "all_common_face_spins_summed", "full_orientation_sum_implemented",
                "two_bulk_sum_is_physical_refinement",
                "autonomous_split_merge_energy_budget_derived", "common_metric_continuum_sector_proven",
                "continuum_einstein_limit_derived"):
        assert report["scope"][key] is False
