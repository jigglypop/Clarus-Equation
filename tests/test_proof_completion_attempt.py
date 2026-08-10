from examples.physics.proof_completion_attempt import attempts
from examples.physics.primordial_spectrum_readout_gate import (
    OBS_AS_1E9,
    OBS_AS_SIGMA_1E9,
)


def by_name(name: str):
    return next(item for item in attempts() if item.name == name)


def test_vcb_lo_remains_obstruction_under_strict_average():
    vcb = by_name("|V_cb| LO")
    assert vcb.status == "obstruction"
    assert vcb.sigma_offset is not None
    assert abs(vcb.sigma_offset) > 6


def test_vcb_nlo_candidate_is_numerically_viable_but_not_a_derivation():
    vcb = by_name("|V_cb| NLO candidate")
    assert vcb.status == "candidate_pass"
    assert vcb.sigma_offset is not None
    assert abs(vcb.sigma_offset) < 1
    assert "electroweak projector" in vcb.proof_status


def test_vus_tree_fails_but_one_loop_candidate_passes():
    tree = by_name("|V_us| tree")
    one_loop = by_name("|V_us| one-loop candidate")
    assert tree.status == "obstruction"
    assert tree.sigma_offset is not None
    assert abs(tree.sigma_offset) > 3
    assert one_loop.status == "candidate_pass"
    assert one_loop.sigma_offset is not None
    assert abs(one_loop.sigma_offset) < 1


def test_as_raw_fails_but_readout_remains_a_candidate():
    raw = by_name("A_s raw")
    readout = by_name("A_s readout candidate")
    assert raw.status == "obstruction"
    assert raw.sigma_offset is not None
    assert abs(raw.sigma_offset) > 100
    assert readout.status == "candidate_pass"
    assert "projected residual-drive" in readout.proof_status
    assert readout.observed == OBS_AS_1E9 * 1e-9
    assert readout.sigma == OBS_AS_SIGMA_1E9 * 1e-9


def test_ns_candidate_depends_on_transition_count():
    ns = by_name("n_s transition-count candidate")
    assert ns.status == "candidate_pass"
    assert ns.sigma_offset is not None
    assert abs(ns.sigma_offset) < 1
    assert "transition count 12" in ns.proof_status
