from __future__ import annotations

import pickle

import numpy as np

from reality_stone.clarus.evidence import (
    assess_manifest,
    celegans_elife_66135_manifest,
    celegans_locomotion_gate,
    celegans_locomotion_gate_from_pickle,
    linear_decoder_gate,
    main,
    validate_locomotion_artifact,
    validate_locomotion_artifact_file,
)


def test_celegans_manifest_reaches_gate_ready_from_predictioncode_mapping():
    check = assess_manifest(celegans_elife_66135_manifest())

    assert check.source_id == "celegans_elife_66135_locomotion"
    assert check.readiness == "gate-ready"
    assert check.is_gate_ready
    assert check.missing == ("local_gate_command", "local_artifacts")


def test_manifest_reaches_gate_ready_when_fields_and_gate_are_defined():
    manifest = celegans_elife_66135_manifest()
    manifest["neural_fields"] = ("activity",)
    manifest["behavior_fields"] = ("velocity", "curvature")
    manifest["stimulus_fields"] = ("condition",)
    manifest["timebase_fields"] = ("time",)
    manifest["subject_id_fields"] = ("recording_id",)

    check = assess_manifest(manifest)

    assert check.readiness == "gate-ready"
    assert check.is_gate_ready
    assert not check.is_reproducible
    assert check.missing == ("local_gate_command", "local_artifacts")


def test_manifest_requires_field_mapping_before_gate_design():
    manifest = {
        "source_id": "zebrafish_example",
        "species": "Danio rerio",
        "paper_or_dataset_url": "https://example.org",
        "license_or_access": "public",
        "raw_files": ("https://example.org/data.zip",),
    }

    check = assess_manifest(manifest)

    assert check.readiness == "download-ready"
    assert "neural_fields" in check.missing
    assert "timebase_fields" in check.missing


def test_linear_decoder_gate_detects_heldout_signal():
    rng = np.random.default_rng(7)
    time = np.linspace(0.0, 8.0, 120)
    signal = np.sin(time)
    features = np.column_stack(
        [
            signal + rng.normal(scale=0.03, size=time.size),
            rng.normal(scale=0.2, size=time.size),
        ]
    )
    target = 1.5 * signal + rng.normal(scale=0.04, size=time.size)

    gate = linear_decoder_gate(features, target, n_permutations=20, seed=11)

    assert gate.n_train == 84
    assert gate.n_test == 36
    assert gate.delta_r2 > 0.5
    assert gate.p_value is not None
    assert gate.passed


def test_celegans_locomotion_gate_summarizes_recording_panel():
    recordings = make_recordings()

    panel = celegans_locomotion_gate(recordings, seed=17)

    assert panel.recording_count == 3
    assert panel.pass_count("velocity") == 3
    assert panel.pass_count("curvature") == 3
    assert panel.pass_rate("velocity") == 1.0
    assert panel.summary()["targets"]["velocity"]["pass_rate"] == 1.0


def test_celegans_locomotion_gate_loads_predictioncode_pickle(tmp_path):
    path = tmp_path / "gcamp_recordings.dat"
    with path.open("wb") as handle:
        pickle.dump(make_recordings(), handle)

    panel = celegans_locomotion_gate_from_pickle(path)

    assert panel.recording_count == 3
    assert panel.to_dict()["summary"]["targets"]["curvature"]["pass_count"] == 3


def test_celegans_cli_writes_artifact_and_enforces_pass_rate(tmp_path, capsys):
    pickle_path = tmp_path / "gcamp_recordings.dat"
    output_path = tmp_path / "celegans_gate.json"
    with pickle_path.open("wb") as handle:
        pickle.dump(make_recordings(), handle)

    exit_code = main(
        [
            str(pickle_path),
            "--output",
            str(output_path),
            "--min-pass-rate",
            "1.0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()
    assert '"recording_count": 3' in captured.out
    assert '"gate_passed": true' in captured.out
    assert '"pass_rate": 1.0' in output_path.read_text(encoding="utf-8")

    check = validate_locomotion_artifact_file(output_path)
    assert check.is_reproducible
    assert check.source_id == "celegans_elife_66135_locomotion"
    assert check.artifact_type == "clarus_locomotion_gate"


def test_celegans_cli_compares_treatment_against_control(tmp_path, capsys):
    treatment_path = tmp_path / "gcamp_recordings.dat"
    control_path = tmp_path / "gfp_recordings.dat"
    with treatment_path.open("wb") as handle:
        pickle.dump(make_recordings(), handle)
    with control_path.open("wb") as handle:
        pickle.dump(make_control_recordings(), handle)

    exit_code = main(
        [
            str(treatment_path),
            "--control-pickle",
            str(control_path),
            "--min-pass-rate",
            "1.0",
            "--min-control-delta",
            "0.5",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"artifact_type": "clarus_locomotion_control_gate"' in captured.out
    assert '"gate_passed": true' in captured.out
    assert '"control_pass_rate": 0.0' in captured.out
    assert '"pass_rate_delta": 1.0' in captured.out
    assert '"passed": true' in captured.out


def test_artifact_validator_rejects_failed_artifact():
    artifact = {
        "artifact_type": "clarus_locomotion_gate",
        "artifact_version": 1,
        "source_id": "celegans_elife_66135_locomotion",
        "criteria": {"min_pass_rate": 1.0},
        "gate_passed": False,
        "result": {"summary": {"recording_count": 1}},
    }

    direct = validate_locomotion_artifact(artifact)
    assert not direct.is_reproducible
    assert direct.next_action == "gate artifact is complete but did not pass"


def test_cli_validates_existing_artifact(tmp_path, capsys):
    pickle_path = tmp_path / "gcamp_recordings.dat"
    output_path = tmp_path / "celegans_gate.json"
    with pickle_path.open("wb") as handle:
        pickle.dump(make_recordings(), handle)

    assert main([str(pickle_path), "--output", str(output_path), "--min-pass-rate", "1.0"]) == 0
    capsys.readouterr()

    exit_code = main(["--validate-artifact", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"is_reproducible": true' in captured.out
    assert '"artifact_type": "clarus_locomotion_gate"' in captured.out


def make_recordings():
    rng = np.random.default_rng(13)
    recordings = {}
    for idx in range(3):
        time = np.linspace(0.0, 7.0, 100)
        velocity_driver = np.sin(time + idx * 0.2)
        curvature_driver = np.cos(time * 0.8 + idx * 0.1)
        neurons = np.vstack(
            [
                velocity_driver + rng.normal(scale=0.02, size=time.size),
                curvature_driver + rng.normal(scale=0.02, size=time.size),
                rng.normal(scale=0.1, size=time.size),
            ]
        )
        recordings[f"recording_{idx}"] = {
            "time": time,
            "neurons": neurons,
            "neuron_derivatives": np.gradient(neurons, axis=1),
            "velocity": velocity_driver + rng.normal(scale=0.03, size=time.size),
            "curvature": curvature_driver + rng.normal(scale=0.03, size=time.size),
        }
    return recordings


def make_control_recordings():
    recordings = make_recordings()
    for recording in recordings.values():
        n_time = np.asarray(recording["velocity"]).shape[0]
        recording["velocity"] = np.zeros(n_time)
        recording["curvature"] = np.zeros(n_time)
    return recordings
