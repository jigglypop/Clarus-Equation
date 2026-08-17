from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reality_stone.clarus.connectome_replay import (
    Observation,
    ReplayValidationError,
    artifact_sha256,
    build_artifact,
    canonical_bytes,
    load_manifest,
    parse_source_bytes,
    validate_observations,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "c_elegans_connectome_tiny.csv"
REGISTERED_MANIFEST = (
    REPOSITORY_ROOT
    / "experiments"
    / "preregistration"
    / "c_elegans_connectome_replay_v1.json"
)
CLI = REPOSITORY_ROOT / "examples" / "brain" / "c_elegans_connectome_replay.py"


def manifest_for(raw: bytes) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "openworm_celegansneuroml_herm_full_edgelist",
        "scope": "adult_hermaphrodite_structural_graph_only",
        "source": {
            "url": "https://example.invalid/frozen.csv",
            "commit": "synthetic-fixture",
            "path": "tiny.csv",
            "byte_length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "redistribution_permission": "not_established",
            "repository_handling": "manifest_only_raw_and_full_output_run_local",
        },
        "population": (
            "all_normalized_endpoint_identifiers_in_frozen_file_including_non_neuron_cells"
        ),
        "parser": {
            "header": ["Source", "Target", "Weight", "Type"],
            "utf8": "strict",
            "bom": "forbidden",
            "classes": ["chemical", "electrical"],
            "endpoint_identifier": (
                "ascii_alphanumeric_after_leading_trailing_ascii_space_removal"
            ),
            "weight_grammar": "0|[1-9][0-9]*",
            "self_loops": "accepted_and_preserved",
            "ordinal_policy": (
                "r1_zero_based_csv_data_record_position_preserved_and_complete"
            ),
            "duplicate_identity": "source_record_ordinal",
        },
        "electrical_weight_semantics": (
            "sum_of_released_row_weights_not_physical_gap_junction_count"
        ),
        "expected_source_metrics": {
            "source_byte_length": len(raw),
            "source_observation_count": 7,
            "normalized_endpoint_identifier_count": 7,
            "self_loop_observation_count": 1,
            "chemical_observation_count": 3,
            "chemical_released_weight_sum": 7,
            "electrical_observation_count": 4,
            "electrical_released_weight_sum": 19,
            "total_released_weight_sum": 26,
            "electrical_unordered_pair_count": 2,
            "electrical_reciprocal_two_row_pair_count": 2,
            "electrical_unequal_reciprocal_weight_pair_count": 1,
            "electrical_max_observations_per_pair": 2,
            "exact_duplicate_released_row_count": 0,
        },
    }


def parsed_fixture() -> tuple[bytes, dict[str, object], list[Observation]]:
    raw = FIXTURE.read_bytes()
    manifest = manifest_for(raw)
    return raw, manifest, parse_source_bytes(raw, manifest)


def test_manifest_and_source_rejections_are_fail_closed(tmp_path: Path) -> None:
    raw, manifest, observations = parsed_fixture()

    tampered = bytes([raw[0] ^ 1]) + raw[1:]
    with pytest.raises(ReplayValidationError, match="sha256 mismatch"):
        parse_source_bytes(tampered, manifest)

    malformed_utf8 = b"\xff,not,csv"
    with pytest.raises(ReplayValidationError, match="strict UTF-8"):
        parse_source_bytes(malformed_utf8, manifest_for(malformed_utf8))

    invalid_sources = (
        b"Source,Target,Weight,Type\nA,B,1,unknown\n",
        b"Source,Target,Weight,Type\nA B,C,1,chemical\n",
        "Source,Target,Weight,Type\nÅ,B,1,chemical\n".encode(),
        b"Source,Target,Weight,Type\nA\tB,C,1,chemical\n",
        b"Source,Target,Weight,Type\n ,B,1,chemical\n",
        b"Source,Target,Weight,Type\nA,,1,chemical\n",
        b"Source,Target,Weight,Type\nA,B,01,chemical\n",
        b"Source,Target,Weight,Type\nA,B,-1,chemical\n",
        b"Source,Target,Weight,Type\nA,B,1.5,chemical\n",
        b"Source,Target,Weight,Type\nA,B,1,chemical,extra\n",
        b"Source,Target,Weight\nA,B,1\n",
        b'Source,Target,Weight,Type\nA,B,1,"chemical\n',
    )
    for body in invalid_sources:
        with pytest.raises(ReplayValidationError):
            parse_source_bytes(body, manifest_for(body))

    with pytest.raises(ReplayValidationError):
        Observation("chemical", "A1", "B2", True, 0)
    with pytest.raises(ReplayValidationError):
        Observation("chemical", "A1", "B2", 1.0, 0)  # type: ignore[arg-type]
    with pytest.raises(ReplayValidationError):
        Observation("chemical", "A1", "B2", 1, False)
    with pytest.raises(ReplayValidationError):
        validate_observations([observations[0], observations[0]])
    with pytest.raises(ReplayValidationError):
        validate_observations(observations[:3] + observations[4:])

    invalid_manifest = deepcopy(manifest)
    invalid_manifest["schema_version"] = True
    with pytest.raises(ReplayValidationError, match="exact integer 1"):
        build_artifact(observations, invalid_manifest)
    invalid_manifest = deepcopy(manifest)
    invalid_manifest["unknown"] = "forbidden"
    with pytest.raises(ReplayValidationError, match="keys must be exactly"):
        build_artifact(observations, invalid_manifest)

    registered = load_manifest(REGISTERED_MANIFEST)
    assert registered["source"]["sha256"] == (
        "0ab9baab5f404895b8dbeb8daa453c86e8f342961bc458cd19bf1b5f6a38d859"
    )


def test_structural_replay_preserves_semantics_and_r1_invariance() -> None:
    _, manifest, observations = parsed_fixture()
    artifact = build_artifact(observations, manifest)
    permuted = build_artifact(list(reversed(observations)), manifest)

    assert set(artifact) == {
        "format",
        "metadata",
        "nodes",
        "observations",
        "connections",
        "summary",
    }
    assert artifact["metadata"]["scope"] == "structural_graph_only"
    assert artifact["metadata"]["electrical_weight_semantics"] == (
        "sum_of_released_row_weights_not_physical_gap_junction_count"
    )
    assert [item["id"] for item in artifact["nodes"]] == [
        "A1",
        "B2",
        "C3",
        "D4",
        "E5",
        "F6",
        "G7",
    ]

    chemical = [
        item
        for item in artifact["observations"]
        if item["connection_class"] == "chemical"
    ]
    assert [(item["source_id"], item["target_id"]) for item in chemical] == [
        ("A1", "B2"),
        ("B2", "A1"),
        ("G7", "G7"),
    ]
    electrical_observations = [
        item
        for item in artifact["observations"]
        if item["connection_class"] == "electrical"
    ]
    assert [
        (item["source_id"], item["target_id"], item["released_weight"])
        for item in electrical_observations
    ] == [
        ("C3", "D4", 5),
        ("D4", "C3", 5),
        ("E5", "F6", 2),
        ("F6", "E5", 7),
    ]
    electrical_connections = [
        item
        for item in artifact["connections"]
        if item["connection_class"] == "electrical"
    ]
    assert [
        (item["endpoint_a"], item["endpoint_b"], item["released_weight_sum"])
        for item in electrical_connections
    ] == [("C3", "D4", 10), ("E5", "F6", 9)]
    assert all(
        set(item) == {
            "connection_class",
            "endpoint_a",
            "endpoint_b",
            "released_observation_count",
            "released_weight_sum",
            "source_record_ordinals",
        }
        for item in artifact["connections"]
    )

    by_ordinal = {
        item["source_record_ordinal"]: item for item in artifact["observations"]
    }
    assert set(by_ordinal) == set(range(len(observations)))
    for connection in artifact["connections"]:
        assert connection["source_record_ordinals"] == sorted(
            connection["source_record_ordinals"]
        )
        for ordinal in connection["source_record_ordinals"]:
            observation = by_ordinal[ordinal]
            assert observation["connection_class"] == connection["connection_class"]
            assert observation["endpoint_a"] == connection["endpoint_a"]
            assert observation["endpoint_b"] == connection["endpoint_b"]

    assert sum(
        item["released_observation_count"] for item in artifact["connections"]
    ) == len(observations)
    assert artifact["summary"]["total_released_weight_sum"] == sum(
        item["released_weight_sum"] for item in artifact["connections"]
    )
    assert artifact["summary"]["self_loop_observation_count"] == 1

    first = canonical_bytes(artifact)
    second = canonical_bytes(artifact)
    shuffled = canonical_bytes(permuted)
    assert first == second == shuffled
    assert first.endswith(b"\n") and not first.endswith(b"\n\n")
    assert artifact_sha256(artifact) == hashlib.sha256(first).hexdigest()
    assert artifact_sha256(artifact) == artifact_sha256(permuted)


def test_offline_cli_writes_atomically_and_rejects_input_aliases(tmp_path: Path) -> None:
    raw = FIXTURE.read_bytes()
    source = tmp_path / "source.csv"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "artifact.json"
    source.write_bytes(raw)
    manifest_path.write_text(json.dumps(manifest_for(raw)), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    command = [
        sys.executable,
        "-B",
        str(CLI),
        "--manifest",
        str(manifest_path),
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    artifact_bytes = output.read_bytes()
    assert completed.stdout.strip() == hashlib.sha256(artifact_bytes).hexdigest()
    assert artifact_bytes.endswith(b"\n")
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))

    source_before = source.read_bytes()
    alias_source = subprocess.run(
        [*command[:-1], str(source)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert alias_source.returncode != 0
    assert "must not alias" in alias_source.stderr
    assert source.read_bytes() == source_before

    manifest_before = manifest_path.read_bytes()
    alias_manifest = subprocess.run(
        [*command[:-1], str(manifest_path)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert alias_manifest.returncode != 0
    assert "must not alias" in alias_manifest.stderr
    assert manifest_path.read_bytes() == manifest_before
