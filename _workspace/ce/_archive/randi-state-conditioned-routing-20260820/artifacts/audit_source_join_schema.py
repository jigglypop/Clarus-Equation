"""Audit source-identity/assignment admission from a frozen NWB schema inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_NWB_BYTES = 1_273_970
EXPECTED_NWB_SHA256 = "40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e"
EXPECTED_SCHEMA_SHA256 = "45e53bb20739b3e1bbe61e9108422ad5e0f85cce9516af37f0b1df631433e54f"
EVENT_TABLE = "/intervals/OptogeneticStimulusTable"
TARGET_TABLE = "/processing/ophys/TargetedImageSegmentation/TargetPlaneSegmentation"
PUMP_GREEN_TABLE = (
    "/processing/ophys/PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation"
)
PROHIBITED = {
    "/processing/ophys/GreenSignals/GreenSignal/data",
    "/processing/ophys/RedSignals/RedSignal/data",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("attributes", {}).get("selected_values", {})


def inspect(nwb_path: Path, schema_path: Path) -> dict[str, Any]:
    nwb_bytes = nwb_path.stat().st_size
    nwb_hash = sha256(nwb_path)
    schema_hash = sha256(schema_path)
    if nwb_bytes != EXPECTED_NWB_BYTES or nwb_hash != EXPECTED_NWB_SHA256:
        raise ValueError(f"frozen NWB mismatch: bytes={nwb_bytes}, sha256={nwb_hash}")
    if schema_hash != EXPECTED_SCHEMA_SHA256:
        raise ValueError(f"frozen schema mismatch: sha256={schema_hash}")

    inventory = json.loads(schema_path.read_text(encoding="utf-8"))
    records = {record["path"]: record for record in inventory["objects"]}
    target_group = records[TARGET_TABLE]
    event_group = records[EVENT_TABLE]
    event_columns = sorted(event_group.get("children", []))
    target_columns = sorted(target_group.get("children", []))

    target_regions = sorted(
        path
        for path, record in records.items()
        if path.startswith("/general/OptogeneticStimulusTarget")
        and path.endswith("/targeted_rois")
        and selected(record).get("table") == TARGET_TABLE
    )
    source_identity_tokens = ("neuropal", "label", "identity", "confidence")
    assignment_tokens = (
        "assignment",
        "random",
        "manual",
        "control",
        "sham",
        "failed",
        "autoresponse",
        "condition",
    )
    source_identity_columns = [
        name for name in target_columns if any(token in name.lower() for token in source_identity_tokens)
    ]
    assignment_columns = [
        name for name in event_columns if any(token in name.lower() for token in assignment_tokens)
    ]
    target_table_refs = sorted(
        {
            selected(record).get("table")
            for path, record in records.items()
            if path.startswith(f"{TARGET_TABLE}/") and selected(record).get("table")
        }
    )
    response_identity_path = f"{PUMP_GREEN_TABLE}/neuropal_ids"
    response_identity_present = response_identity_path in records
    response_identity_description = selected(records[response_identity_path]).get("description")

    checks = {
        "event_columns": event_columns,
        "target_columns": target_columns,
        "target_region_reference_count": len(target_regions),
        "target_region_reference_table": TARGET_TABLE,
        "source_identity_columns": source_identity_columns,
        "target_external_table_references": target_table_refs,
        "response_side_neuropal_mapping_present": response_identity_present,
        "event_assignment_columns": assignment_columns,
        "explicit_source_identity_join": bool(source_identity_columns or target_table_refs),
        "event_assignment_receipt": bool(assignment_columns),
        "response_dataset_values_read": False,
        "geometric_matching_performed": False,
    }
    statuses = []
    if not checks["explicit_source_identity_join"]:
        statuses.append("BLOCKED_EXPLICIT_SOURCE_JOIN")
    if not checks["event_assignment_receipt"]:
        statuses.append("BLOCKED_ASSIGNMENT_RECEIPT")
    if not statuses:
        statuses.append("PASS_APPARATUS_SCHEMA")

    return {
        "schema": "clarus.randi.state-conditioned-routing-admission.v1",
        "status": " / ".join(statuses),
        "inputs": {
            "nwb": {"path": nwb_path.as_posix(), "bytes": nwb_bytes, "sha256": nwb_hash},
            "schema_inventory": {"path": schema_path.as_posix(), "sha256": schema_hash},
        },
        "audit_boundary": {
            "schema_inventory_only": True,
            "dataset_values_read": False,
            "neural_effect_computed": False,
            "endpoint_or_threshold_selected": False,
            "prohibited_response_paths": sorted(PROHIBITED),
        },
        "evidence": {
            "target_region_paths": target_regions,
            "target_table_colnames": selected(target_group).get("colnames", []),
            "response_identity_path": response_identity_path,
            "response_identity_description": response_identity_description,
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.nwb.resolve(), args.schema.resolve())
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["checks"], sort_keys=True))
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
