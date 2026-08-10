from __future__ import annotations

import math

from reality_stone.clarus.dimensionless import GateResult
from tests import run_validation
from tests.scorecard import ConstantsScorecard


def test_combined_validator_reuses_every_canonical_scorecard_row() -> None:
    scorecard = ConstantsScorecard()
    canonical = scorecard.constants
    combined = run_validation.ConstantsValidator().validate_all()
    rows = combined["results"]

    assert len(rows) == len(canonical)
    for row, constant in zip(rows, canonical):
        assert row["name"] == constant.name
        assert row["symbol"] == constant.symbol
        assert row["ce"] == constant.ce_value
        assert row["obs"] == constant.obs_value
        assert row["obs_sigma"] == constant.obs_sigma
        if math.isnan(constant.sigma_offset):
            assert math.isnan(row["sigma_offset"])
        else:
            assert row["sigma_offset"] == constant.sigma_offset
        assert row["grade"] == constant.grade
        assert row["role"] == constant.role
        assert row["status"] == constant.status
        assert row["scoreable"] == constant.scoreable
        assert row["is_scored"] == constant.is_scored
    assert combined["summary"] == scorecard.summary()


def test_combined_validator_uses_scored_denominator_and_signed_sigma() -> None:
    combined = run_validation.ConstantsValidator().validate_all()
    rows = {row["symbol"]: row for row in combined["results"]}
    summary = combined["summary"]

    assert summary["total"] == 23
    assert summary["scored_total"] == 12
    assert summary["passed"] == 11
    assert summary["caution"] == 1
    assert summary["input"] == 1
    assert math.isclose(summary["pass_rate"], 100 * 11 / 12)
    assert summary["status"] == "CAUTION"

    assert rows["alpha_s(M_Z)"]["grade"] == "Selection"
    assert rows["alpha_s(M_Z)"]["role"] == "Input"
    assert rows["alpha_s(M_Z)"]["status"] == "INPUT"
    assert not rows["alpha_s(M_Z)"]["scoreable"]
    assert not rows["alpha_s(M_Z)"]["is_scored"]
    assert rows["alpha_s(M_Z)"]["obs"] == 0.1180
    assert rows["alpha_s(M_Z)"]["obs_sigma"] == 0.0009
    assert rows["sin^2(theta_W)"]["obs"] == 0.23122
    assert rows["sin^2(theta_W)"]["obs_sigma"] == 0.00006
    assert rows["Omega_b h^2"]["sigma_offset"] < 0
    assert rows["Omega_b h^2"]["status"] == "CAUTION"
    assert rows["A_s x 10^9"]["obs"] == 2.099
    assert rows["A_s x 10^9"]["obs_sigma"] == 0.029
    assert rows["A_s x 10^9"]["sigma_offset"] > 0


def test_weak_angle_derivatives_use_ce_value_and_propagated_pdg_uncertainty() -> None:
    rows = {
        constant.symbol: constant
        for constant in ConstantsScorecard().constants
    }
    sin2_ce = 4 * (0.11789 ** (4 / 3))
    lambda_ce = sin2_ce * (1 - sin2_ce)
    lambda_obs_sigma = abs(1 - 2 * 0.23122) * 0.00006

    assert rows["lambda_W"].ce_value == lambda_ce
    assert rows["lambda_W"].obs_sigma == lambda_obs_sigma
    assert rows["sin^2(theta_13)"].ce_value == lambda_ce / (3**2 - 1)
    n_eff = 3 * (3 + lambda_ce) * 12 / 2
    assert rows["n_s"].ce_value == 1 - 2 / n_eff


def test_combined_validator_has_no_stale_constant_count_claim() -> None:
    assert "45" not in (run_validation.ConstantsValidator.__doc__ or "")


def test_dimensional_validator_runs_live_dimensionless_gates() -> None:
    result = run_validation.DimensionalValidator().validate_all()

    assert result["total"] == 7
    assert result["passed"] == 7
    assert result["all_pass"]


def test_dimensional_validator_propagates_checker_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        run_validation,
        "exp_arguments",
        lambda quantities: GateResult.fail("forced dimensional failure"),
    )

    result = run_validation.DimensionalValidator().validate_all()

    assert not result["all_pass"]
    assert result["passed"] < result["total"]
    assert any(
        "forced dimensional failure" in info["errors"]
        for info in result["formulas"].values()
    )


def test_overall_status_preserves_canonical_caution() -> None:
    result = run_validation.main()

    assert result["overall_status"] == "CAUTION"
    assert result["constants"]["summary"]["scored_total"] == 12
