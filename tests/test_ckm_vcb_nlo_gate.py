from examples.physics.ckm_vcb_nlo_gate import (
    vcb_lo,
    vcb_nlo_electroweak_projector,
    vcb_wrong_phase_controls,
)


def test_vcb_lo_is_rejected_by_strict_average():
    item = vcb_lo()
    assert item.status == "reject"
    assert item.sigma_offset < -6


def test_vcb_nlo_projector_passes_strict_average():
    item = vcb_nlo_electroweak_projector()
    assert item.status == "pass"
    assert abs(item.sigma_offset) < 1


def test_vcb_nlo_projector_is_not_trivially_any_phase_factor():
    controls = {item.name: item for item in vcb_wrong_phase_controls()}
    assert controls["half phase"].sigma_offset > 7
    assert controls["quarter phase"].sigma_offset < -2
    assert abs(controls["QCD phase"].sigma_offset) > 1
    assert abs(controls["dimension averaged QCD"].sigma_offset) > 1
