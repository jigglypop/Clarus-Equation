"""Build the target-blind BA-SRM2 Allen-SynPhys train sequence manifest.

This stage is deliberately unable to read response-fit values, response QC,
stimulus QC, BLOBs, or target completeness.  It sees structural/protocol
metadata only, applies the frozen slice split, and selects at most 1,500
sequences per E/I stratum by the preregistered slice-round-robin cap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
from typing import Any, Iterable


SPLIT_SALT = "BA-SRM2-MEDIUM-R21-20260823-V1"
CAP_SALT = "BA-SRM2-MEDIUM-SEQUENCE-CAP-V1:"
MANIFEST_VERSION = "BA-SRM2-TRAIN-MANIFEST-V2-ZERO-BASED"
MAX_PER_STRATUM = 1_500
PROJECTS = ("mouse V1 coarse matrix", "mouse V1 pre-production")

FORBIDDEN_SQL_TOKENS = (
    "pulse_response_fit",
    "dec_fit_",
    "baseline_dec_fit_",
    "ex_qc_pass",
    "in_qc_pass",
    "sp.qc_pass",
    "stim_pulse.data",
    "pulse_response.data",
)


STRUCTURAL_SEQUENCE_SQL = """
WITH events AS (
    SELECT
        sl.id AS slice_id,
        sl.ext_id AS slice_ext_id,
        sy.synapse_type AS synapse_type,
        pa.id AS pair_id,
        post_r.id AS post_recording_id,
        pre_r.id AS pre_recording_id,
        post_r.stim_name AS post_stim_name,
        mpp.induction_frequency AS induction_frequency,
        mpp.recovery_delay AS recovery_delay,
        sp.id AS stim_pulse_id,
        sp.pulse_number AS pulse_number,
        sp.onset_time AS onset_time
    FROM allowed_train_slice ats
    JOIN slice sl ON sl.id = ats.slice_id
    JOIN experiment ex ON ex.slice_id = sl.id
    JOIN pair pa ON pa.experiment_id = ex.id
    JOIN synapse sy ON sy.pair_id = pa.id
    JOIN cell pre_c ON pre_c.id = pa.pre_cell_id
    JOIN cell post_c ON post_c.id = pa.post_cell_id
    JOIN pulse_response pr ON pr.pair_id = pa.id
    JOIN stim_pulse sp ON sp.id = pr.stim_pulse_id
    JOIN recording post_r ON post_r.id = pr.recording_id
    JOIN recording pre_r ON pre_r.id = sp.recording_id
    JOIN sync_rec post_sr ON post_sr.id = post_r.sync_rec_id
    JOIN sync_rec pre_sr ON pre_sr.id = pre_r.sync_rec_id
    JOIN patch_clamp_recording post_pcr
      ON post_pcr.recording_id = post_r.id
    JOIN multi_patch_probe mpp
      ON mpp.patch_clamp_recording_id = post_pcr.id
    WHERE sl.species = 'mouse'
      AND ex.project_name IN (?, ?)
      AND pa.has_synapse = 1
      AND sy.synapse_type IN ('ex', 'in')
      AND post_pcr.clamp_mode = 'ic'
      AND mpp.induction_frequency > 0
      AND mpp.recovery_delay > 0
      AND post_r.id <> pre_r.id
      AND post_r.sync_rec_id = pre_r.sync_rec_id
      AND post_sr.experiment_id = pa.experiment_id
      AND pre_sr.experiment_id = pa.experiment_id
      AND (sp.cell_id IS NULL OR sp.cell_id = pa.pre_cell_id)
      AND pre_r.electrode_id = pre_c.electrode_id
      AND post_r.electrode_id = post_c.electrode_id
), ordered AS (
    SELECT *,
        lag(pulse_number) OVER (
            PARTITION BY pair_id, post_recording_id, pre_recording_id,
                         post_stim_name, induction_frequency, recovery_delay
            ORDER BY onset_time, stim_pulse_id
        ) AS previous_number,
        lag(onset_time) OVER (
            PARTITION BY pair_id, post_recording_id, pre_recording_id,
                         post_stim_name, induction_frequency, recovery_delay
            ORDER BY onset_time, stim_pulse_id
        ) AS previous_onset
    FROM events
), sequences AS (
    SELECT
        slice_id, slice_ext_id, synapse_type, pair_id,
        post_recording_id, pre_recording_id, post_stim_name,
        induction_frequency, recovery_delay,
        count(*) AS event_rows,
        count(DISTINCT pulse_number) AS distinct_pulses,
        min(pulse_number) AS min_pulse,
        max(pulse_number) AS max_pulse,
        sum(CASE WHEN previous_number IS NOT NULL
                   AND pulse_number <= previous_number THEN 1 ELSE 0 END)
            AS bad_number_order,
        sum(CASE WHEN previous_onset IS NOT NULL
                   AND onset_time <= previous_onset THEN 1 ELSE 0 END)
            AS bad_time_order
    FROM ordered
    GROUP BY slice_id, slice_ext_id, synapse_type, pair_id,
             post_recording_id, pre_recording_id, post_stim_name,
             induction_frequency, recovery_delay
)
SELECT slice_id, slice_ext_id, synapse_type, pair_id,
       post_recording_id, pre_recording_id, post_stim_name,
       induction_frequency, recovery_delay
FROM sequences
WHERE event_rows = 12
  AND distinct_pulses = 12
  AND min_pulse = 0
  AND max_pulse = 11
  AND bad_number_order = 0
  AND bad_time_order = 0
ORDER BY synapse_type, slice_ext_id, pair_id, post_recording_id,
         pre_recording_id, induction_frequency, recovery_delay
"""


class ManifestFailure(RuntimeError):
    """Raised when the structural manifest cannot be frozen safely."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def split_bucket(group_id: str) -> int:
    material = f"{SPLIT_SALT}:{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10


def _f64be(value: Any) -> str:
    number = float(value)
    return struct.pack(">d", number).hex()


def canonical_sequence_key(row: sqlite3.Row | dict[str, Any]) -> str:
    stim_name = row["post_stim_name"]
    if stim_name is None:
        encoded_stim = "NULL"
    else:
        encoded_stim = "UTF8HEX:" + str(stim_name).encode("utf-8").hex()
    return (
        f"pair_id={int(row['pair_id'])}"
        f"|post_recording_id={int(row['post_recording_id'])}"
        f"|pre_recording_id={int(row['pre_recording_id'])}"
        f"|post_stim_name={encoded_stim}"
        f"|induction_frequency_f64be={_f64be(row['induction_frequency'])}"
        f"|recovery_delay_f64be={_f64be(row['recovery_delay'])}"
    )


def cap_hash(sequence_key: str) -> str:
    return sha256_bytes((CAP_SALT + sequence_key).encode("utf-8"))


def assert_target_blind_sql(sql: str) -> None:
    normalized = " ".join(sql.lower().split())
    for token in FORBIDDEN_SQL_TOKENS:
        if token.lower() in normalized:
            raise ManifestFailure(f"locked outcome/QC token in manifest SQL: {token}")


def train_slices(con: sqlite3.Connection) -> list[tuple[int, str]]:
    rows = list(con.execute("SELECT id, ext_id FROM slice ORDER BY id"))
    observed: set[str] = set()
    selected: list[tuple[int, str]] = []
    for slice_id, ext_id in rows:
        if ext_id is None or str(ext_id) == "":
            raise ManifestFailure("slice.ext_id is NULL/empty")
        group_id = str(ext_id)
        if group_id in observed:
            raise ManifestFailure(f"duplicate slice.ext_id: {group_id}")
        observed.add(group_id)
        if split_bucket(group_id) <= 5:
            selected.append((int(slice_id), group_id))
    return selected


def select_round_robin(
    candidates: Iterable[dict[str, Any]], cap: int = MAX_PER_STRATUM
) -> list[dict[str, Any]]:
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_slice[candidate["slice_ext_id"]].append(candidate)
    for rows in by_slice.values():
        rows.sort(key=lambda row: (row["cap_hash"], row["sequence_key"]))

    selected: list[dict[str, Any]] = []
    ordered_slices = sorted(by_slice, key=lambda value: value.encode("utf-8"))
    round_index = 0
    while len(selected) < cap:
        added = 0
        for group_id in ordered_slices:
            rows = by_slice[group_id]
            if round_index < len(rows):
                item = dict(rows[round_index])
                item["round_robin_index"] = round_index
                selected.append(item)
                added += 1
                if len(selected) == cap:
                    break
        if added == 0:
            break
        round_index += 1
    return selected


def build_manifest(database: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = database.resolve(strict=True)
    uri = f"file:{resolved.as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        slices = train_slices(con)
        con.execute("CREATE TEMP TABLE allowed_train_slice(slice_id INTEGER PRIMARY KEY)")
        con.executemany(
            "INSERT INTO allowed_train_slice(slice_id) VALUES (?)",
            ((slice_id,) for slice_id, _ in slices),
        )
        assert_target_blind_sql(STRUCTURAL_SEQUENCE_SQL)
        raw_rows = list(con.execute(STRUCTURAL_SEQUENCE_SQL, PROJECTS))
    finally:
        con.close()

    candidates_by_type: dict[str, list[dict[str, Any]]] = {
        "ex": [],
        "in": [],
    }
    seen_keys: set[str] = set()
    for raw in raw_rows:
        row = dict(raw)
        row["slice_ext_id"] = str(row["slice_ext_id"])
        key = canonical_sequence_key(row)
        if key in seen_keys:
            raise ManifestFailure(f"duplicate canonical sequence key: {key}")
        seen_keys.add(key)
        item = {
            "version": MANIFEST_VERSION,
            "split": "train",
            "synapse_type": str(row["synapse_type"]),
            "slice_id": int(row["slice_id"]),
            "slice_ext_id": row["slice_ext_id"],
            "pair_id": int(row["pair_id"]),
            "post_recording_id": int(row["post_recording_id"]),
            "pre_recording_id": int(row["pre_recording_id"]),
            "post_stim_name": row["post_stim_name"],
            "induction_frequency": float(row["induction_frequency"]),
            "recovery_delay": float(row["recovery_delay"]),
            "sequence_key": key,
            "cap_hash": cap_hash(key),
        }
        if item["synapse_type"] not in candidates_by_type:
            raise ManifestFailure("unexpected E/I label")
        if split_bucket(item["slice_ext_id"]) > 5:
            raise ManifestFailure("non-train slice reached structural manifest")
        candidates_by_type[item["synapse_type"]].append(item)

    empty_strata = [
        label for label in ("ex", "in") if not candidates_by_type[label]
    ]
    if empty_strata:
        raise ManifestFailure(
            "no target-blind structural candidates for strata: "
            + ",".join(empty_strata)
        )

    selected: list[dict[str, Any]] = []
    for synapse_type in ("ex", "in"):
        selected.extend(select_round_robin(candidates_by_type[synapse_type]))
    selected.sort(
        key=lambda row: (
            row["synapse_type"],
            row["round_robin_index"],
            row["slice_ext_id"].encode("utf-8"),
            row["cap_hash"],
        )
    )

    receipt = {
        "status": "TRAIN_MANIFEST_FROZEN",
        "version": MANIFEST_VERSION,
        "database": str(resolved),
        "database_sha256": sha256_file(resolved),
        "split_salt": SPLIT_SALT,
        "cap_salt": CAP_SALT,
        "cap_per_stratum": MAX_PER_STRATUM,
        "project_names": list(PROJECTS),
        "outcome_values_read": False,
        "response_qc_values_read": False,
        "target_availability_read": False,
        "confirmation_outcomes_read": False,
        "train_slice_count": len(slices),
        "candidate_counts": {
            label: len(candidates_by_type[label]) for label in ("ex", "in")
        },
        "candidate_slice_counts": {
            label: len({row["slice_ext_id"] for row in candidates_by_type[label]})
            for label in ("ex", "in")
        },
        "selected_counts": {
            label: sum(row["synapse_type"] == label for row in selected)
            for label in ("ex", "in")
        },
        "selected_slice_counts": {
            label: len(
                {
                    row["slice_ext_id"]
                    for row in selected
                    if row["synapse_type"] == label
                }
            )
            for label in ("ex", "in")
        },
    }
    return selected, receipt


def render_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    try:
        selected, receipt = build_manifest(args.database)
        manifest_bytes = render_jsonl(selected)
        receipt["manifest_sha256"] = sha256_bytes(manifest_bytes)
        receipt["manifest_rows"] = len(selected)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_bytes(manifest_bytes)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ManifestFailure, OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_TRAIN_MANIFEST", "error": str(exc)}))
        return 2

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
