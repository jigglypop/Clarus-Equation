import json

from reality_stone.clarus.runtime import BrainRuntimeConfig
from reality_stone.clarus.experiments.runtime_factorized_competition_composition import (
    analyze_factorized_artifact,
)
from reality_stone.clarus.experiments.runtime_experience_delayed_binding import _blocks


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-factorized-competition-composition-20260822") / "artifacts" / "calibration-input.json"


def test_factorized_config_rejects_jitter_and_nonunit_gain() -> None:
    source, hidden, _target = _blocks()
    common = dict(
        dim=20,
        axon_delay=True,
        competition_indices=hidden,
        competition_input_indices=source,
        competition_factorize_delayed_input=True,
    )
    for extra in (
        {"competition_lateral_gain": 0.5},
        {"competition_lateral_gain": 1.0, "competition_jitter_sigma": 0.1},
    ):
        try:
            BrainRuntimeConfig(**common, **extra)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid factorized competition config was admitted")


def test_frozen_calibration_composes_by_source_provenance() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_factorized_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "FACTORIZED_COMPETITION_CALIBRATION_PASS"
    assert result["atomic_success_total"] == 4
    assert result["factorized_pair_success_total"] == 4
    assert result["legacy_pair_success_total"] == 0
    assert result["misaligned_pair_success_total"] == 0

