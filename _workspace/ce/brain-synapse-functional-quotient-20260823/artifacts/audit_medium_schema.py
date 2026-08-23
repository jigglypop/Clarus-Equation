"""Outcome-blind receipt and schema audit for Allen-SynPhys r2.1 medium.

This program deliberately never selects fitted response values, fit-success/null
patterns, or response QC values.  It verifies the downloaded object, relational
schema, foreign keys, event identity, and pulse-order metadata only.  Train
outcomes remain locked after this program completes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


EXPECTED_BYTES = 11_125_997_568
SPLIT_SALT = "BA-SRM2-MEDIUM-R21-20260823-V1"
AUDITOR_VERSION = "BA-SRM2-MEDIUM-SCHEMA-AUDITOR-V3"

TARGET_VALUE_COLUMNS = (
    "dec_fit_reconv_amp",
    "dec_fit_latency",
    "dec_fit_rise_time",
    "dec_fit_decay_tau",
)

# QC values and target availability are also locked at this stage.  Column
# names may be inspected via PRAGMA table_info, but these tokens may not appear
# in SELECT/WHERE expressions executed by this script.
FORBIDDEN_VALUE_SQL_TOKENS = TARGET_VALUE_COLUMNS + (
    "baseline_dec_fit_reconv_amp",
    "ex_qc_pass",
    "in_qc_pass",
    "sp.qc_pass",
)

FIT_RELATION_SQL = """
    SELECT count(*) AS fit_rows,
           count(DISTINCT pulse_response_id) AS distinct_pulse_response_ids,
           sum(CASE WHEN pulse_response_id IS NULL THEN 1 ELSE 0 END)
               AS null_pulse_response_ids
    FROM pulse_response_fit
"""

REQUIRED_COLUMNS = {
    "slice": {"id", "ext_id", "species"},
    "experiment": {"id", "slice_id", "project_name", "target_region"},
    "cell": {"id", "experiment_id", "electrode_id", "ext_id"},
    "pair": {
        "id",
        "experiment_id",
        "pre_cell_id",
        "post_cell_id",
        "has_synapse",
        "distance",
    },
    "synapse": {"id", "pair_id", "synapse_type"},
    "sync_rec": {"id", "experiment_id", "ext_id", "temperature"},
    "recording": {"id", "sync_rec_id", "electrode_id", "stim_name"},
    "patch_clamp_recording": {
        "id",
        "recording_id",
        "clamp_mode",
        "baseline_potential",
        "baseline_current",
        "baseline_noise_stdev",
        "nearest_test_pulse_id",
    },
    "multi_patch_probe": {
        "id",
        "patch_clamp_recording_id",
        "induction_frequency",
        "recovery_delay",
        "n_spikes_evoked",
    },
    "stim_pulse": {
        "id",
        "recording_id",
        "pulse_number",
        "cell_id",
        "onset_time",
        "amplitude",
        "duration",
        "n_spikes",
        "first_spike_time",
        "previous_pulse_dt",
        "qc_pass",
        "data",
    },
    "pulse_response": {
        "id",
        "recording_id",
        "stim_pulse_id",
        "pair_id",
        "ex_qc_pass",
        "in_qc_pass",
        "data",
    },
    "pulse_response_fit": {
        "id",
        "pulse_response_id",
        "dec_fit_reconv_amp",
        "baseline_dec_fit_reconv_amp",
        "dec_fit_latency",
        "dec_fit_rise_time",
        "dec_fit_decay_tau",
        "dec_fit_nrmse",
    },
    "intrinsic": {"id", "cell_id", "input_resistance", "tau"},
    "test_pulse": {
        "id",
        "recording_id",
        "input_resistance",
        "capacitance",
        "time_constant",
    },
}

REQUIRED_FOREIGN_KEYS = {
    ("experiment", "slice_id", "slice", "id"),
    ("pair", "experiment_id", "experiment", "id"),
    ("pair", "pre_cell_id", "cell", "id"),
    ("pair", "post_cell_id", "cell", "id"),
    ("synapse", "pair_id", "pair", "id"),
    ("recording", "sync_rec_id", "sync_rec", "id"),
    ("patch_clamp_recording", "recording_id", "recording", "id"),
    (
        "multi_patch_probe",
        "patch_clamp_recording_id",
        "patch_clamp_recording",
        "id",
    ),
    ("stim_pulse", "recording_id", "recording", "id"),
    ("stim_pulse", "cell_id", "cell", "id"),
    ("pulse_response", "recording_id", "recording", "id"),
    ("pulse_response", "stim_pulse_id", "stim_pulse", "id"),
    ("pulse_response", "pair_id", "pair", "id"),
    (
        "pulse_response_fit",
        "pulse_response_id",
        "pulse_response",
        "id",
    ),
}


class AuditFailure(RuntimeError):
    """Raised when an acquisition or schema gate fails."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def split_bucket(group_id: str) -> int:
    material = f"{SPLIT_SALT}:{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10


def split_name(bucket: int) -> str:
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "development"
    return "confirmation"


def normalize_sql(sql: str) -> str:
    return " ".join(sql.lower().split())


def assert_schema_only_sql(sql: str) -> None:
    normalized = normalize_sql(sql)
    for token in FORBIDDEN_VALUE_SQL_TOKENS:
        if token.lower() in normalized:
            raise AuditFailure(
                f"schema-only SQL attempted locked value/QC column: {token}"
            )
    if "pulse_response_fit" in normalized:
        allowed_fit_queries = {
            'select count(*) from "pulse_response_fit"',
            normalize_sql(FIT_RELATION_SQL),
        }
        if normalized not in allowed_fit_queries:
            raise AuditFailure(
                "schema-only SQL may access pulse_response_fit only through "
                "the frozen identity/count allowlist"
            )


def fetchall_schema_only(
    con: sqlite3.Connection, sql: str, params: Iterable[Any] = ()
) -> list[sqlite3.Row]:
    assert_schema_only_sql(sql)
    return list(con.execute(sql, tuple(params)))


def scalar_schema_only(
    con: sqlite3.Connection, sql: str, params: Iterable[Any] = ()
) -> Any:
    rows = fetchall_schema_only(con, sql, params)
    if len(rows) != 1 or len(rows[0]) != 1:
        raise AuditFailure("scalar query returned unexpected shape")
    return rows[0][0]


def table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    # PRAGMA reveals only schema declarations, never row values.
    escaped = table.replace('"', '""')
    return {row[1] for row in con.execute(f'PRAGMA table_info("{escaped}")')}


def foreign_keys(con: sqlite3.Connection, table: str) -> set[tuple[str, str, str, str]]:
    escaped = table.replace('"', '""')
    return {
        (table, row[3], row[2], row[4])
        for row in con.execute(f'PRAGMA foreign_key_list("{escaped}")')
    }


def inspect_schema(con: sqlite3.Connection) -> dict[str, Any]:
    tables = {
        row[0]
        for row in fetchall_schema_only(
            con,
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        )
    }
    missing_tables = sorted(set(REQUIRED_COLUMNS) - tables)
    columns = {
        table: sorted(table_columns(con, table))
        for table in REQUIRED_COLUMNS
        if table in tables
    }
    missing_columns = {
        table: sorted(required - set(columns.get(table, ())))
        for table, required in REQUIRED_COLUMNS.items()
        if required - set(columns.get(table, ()))
    }

    observed_fks: set[tuple[str, str, str, str]] = set()
    for table in REQUIRED_COLUMNS:
        if table in tables:
            observed_fks.update(foreign_keys(con, table))
    missing_fks = sorted(REQUIRED_FOREIGN_KEYS - observed_fks)

    return {
        "table_count": len(tables),
        "required_tables": sorted(REQUIRED_COLUMNS),
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "missing_foreign_keys": [list(item) for item in missing_fks],
        "target_field_schema_provenance": {
            field: (
                field in set(columns.get("pulse_response_fit", ()))
                and (
                    "pulse_response_fit",
                    "pulse_response_id",
                    "pulse_response",
                    "id",
                )
                in observed_fks
            )
            for field in TARGET_VALUE_COLUMNS
        },
        "schema_pass": not missing_tables and not missing_columns and not missing_fks,
    }


def inspect_table_counts(con: sqlite3.Connection) -> dict[str, int]:
    # Whole-table counts contain no fitted values, null patterns, QC values, or
    # split-specific target availability information.
    return {
        table: int(scalar_schema_only(con, f'SELECT count(*) FROM "{table}"'))
        for table in (
            "slice",
            "experiment",
            "pair",
            "synapse",
            "recording",
            "stim_pulse",
            "pulse_response",
            "pulse_response_fit",
        )
    }


def inspect_split_groups(con: sqlite3.Connection) -> dict[str, Any]:
    group_ids = [
        str(row[0])
        for row in fetchall_schema_only(
            con,
            "SELECT ext_id FROM slice WHERE ext_id IS NOT NULL ORDER BY ext_id",
        )
    ]
    counts = Counter(split_name(split_bucket(group_id)) for group_id in group_ids)
    manifest_lines = [f"{group_id}\t{split_bucket(group_id)}" for group_id in group_ids]
    manifest_sha = hashlib.sha256("\n".join(manifest_lines).encode("utf-8")).hexdigest()
    return {
        "salt": SPLIT_SALT,
        "group_column": "slice.ext_id",
        "total_groups": len(group_ids),
        "split_counts": dict(sorted(counts.items())),
        "manifest_sha256": manifest_sha,
        "ids_embedded_in_receipt": False,
    }


def inspect_relations_and_order(con: sqlite3.Connection) -> dict[str, Any]:
    relation_sql = """
        SELECT
            count(*) AS linked_rows,
            sum(CASE WHEN post_r.id = pre_r.id THEN 1 ELSE 0 END) AS same_recording,
            sum(CASE WHEN post_r.sync_rec_id IS NULL
                       OR pre_r.sync_rec_id IS NULL
                       OR post_r.sync_rec_id != pre_r.sync_rec_id
                     THEN 1 ELSE 0 END) AS sync_rec_mismatch,
            sum(CASE WHEN post_sr.experiment_id != pa.experiment_id
                       OR pre_sr.experiment_id != pa.experiment_id
                     THEN 1 ELSE 0 END) AS experiment_mismatch,
            sum(CASE WHEN sp.cell_id IS NULL THEN 1 ELSE 0 END)
                AS stim_cell_id_null,
            sum(CASE WHEN sp.cell_id IS NOT NULL AND sp.cell_id != pa.pre_cell_id
                     THEN 1 ELSE 0 END) AS nonnull_pre_cell_mismatch,
            sum(CASE WHEN pre_r.electrode_id IS NULL
                       OR pre_cell.electrode_id IS NULL
                       OR pre_r.electrode_id != pre_cell.electrode_id
                     THEN 1 ELSE 0 END) AS pre_electrode_mismatch,
            sum(CASE WHEN post_r.electrode_id IS NULL
                       OR post_cell.electrode_id IS NULL
                       OR post_r.electrode_id != post_cell.electrode_id
                     THEN 1 ELSE 0 END) AS post_electrode_mismatch
        FROM pulse_response pr
        JOIN pair pa ON pa.id = pr.pair_id
        JOIN cell pre_cell ON pre_cell.id = pa.pre_cell_id
        JOIN cell post_cell ON post_cell.id = pa.post_cell_id
        JOIN stim_pulse sp ON sp.id = pr.stim_pulse_id
        JOIN recording post_r ON post_r.id = pr.recording_id
        JOIN recording pre_r ON pre_r.id = sp.recording_id
        JOIN sync_rec post_sr ON post_sr.id = post_r.sync_rec_id
        JOIN sync_rec pre_sr ON pre_sr.id = pre_r.sync_rec_id
    """
    relation = dict(fetchall_schema_only(con, relation_sql)[0])

    base_event_sql = """
        SELECT
            pa.id AS pair_id,
            post_r.id AS post_recording_id,
            pre_r.id AS pre_recording_id,
            coalesce(post_r.stim_name, '') AS stim_name,
            mpp.induction_frequency AS induction_frequency,
            mpp.recovery_delay AS recovery_delay,
            sp.id AS stim_pulse_id,
            sp.pulse_number AS pulse_number,
            sp.onset_time AS onset_time,
            CASE WHEN post_r.id != pre_r.id
                       AND post_r.sync_rec_id = pre_r.sync_rec_id
                       AND post_sr.experiment_id = pa.experiment_id
                       AND pre_sr.experiment_id = pa.experiment_id
                       AND (sp.cell_id IS NULL OR sp.cell_id = pa.pre_cell_id)
                       AND pre_r.electrode_id = pre_cell.electrode_id
                       AND post_r.electrode_id = post_cell.electrode_id
                 THEN 1 ELSE 0 END AS valid_identity
        FROM pulse_response pr
        JOIN pair pa ON pa.id = pr.pair_id
        JOIN cell pre_cell ON pre_cell.id = pa.pre_cell_id
        JOIN cell post_cell ON post_cell.id = pa.post_cell_id
        JOIN stim_pulse sp ON sp.id = pr.stim_pulse_id
        JOIN recording post_r ON post_r.id = pr.recording_id
        JOIN recording pre_r ON pre_r.id = sp.recording_id
        JOIN sync_rec post_sr ON post_sr.id = post_r.sync_rec_id
        JOIN sync_rec pre_sr ON pre_sr.id = pre_r.sync_rec_id
        JOIN patch_clamp_recording post_pcr
          ON post_pcr.recording_id = post_r.id
        JOIN multi_patch_probe mpp
          ON mpp.patch_clamp_recording_id = post_pcr.id
    """

    order_sql = f"""
        WITH events AS ({base_event_sql}), ordered AS (
            SELECT *,
                lag(pulse_number) OVER (
                    PARTITION BY pair_id, post_recording_id, pre_recording_id,
                                 stim_name, induction_frequency, recovery_delay
                    ORDER BY onset_time, stim_pulse_id
                ) AS previous_number,
                lag(onset_time) OVER (
                    PARTITION BY pair_id, post_recording_id, pre_recording_id,
                                 stim_name, induction_frequency, recovery_delay
                    ORDER BY onset_time, stim_pulse_id
                ) AS previous_onset
            FROM events
        )
        SELECT
            count(*) AS event_rows,
            sum(CASE WHEN previous_number IS NOT NULL
                       AND pulse_number <= previous_number THEN 1 ELSE 0 END)
                AS nonincreasing_pulse_number,
            sum(CASE WHEN previous_onset IS NOT NULL
                       AND onset_time <= previous_onset THEN 1 ELSE 0 END)
                AS nonincreasing_onset_time
        FROM ordered
    """
    ordering = dict(fetchall_schema_only(con, order_sql)[0])

    duplicate_sql = f"""
        WITH events AS ({base_event_sql}), duplicate_groups AS (
            SELECT pair_id, post_recording_id, pre_recording_id, stim_name,
                   induction_frequency, recovery_delay, pulse_number, count(*) AS n
            FROM events
            GROUP BY pair_id, post_recording_id, pre_recording_id, stim_name,
                     induction_frequency, recovery_delay, pulse_number
            HAVING count(*) > 1
        )
        SELECT count(*) FROM duplicate_groups
    """
    ordering["duplicate_sequence_pulse_groups"] = int(
        scalar_schema_only(con, duplicate_sql)
    )

    sequence_sql = f"""
        WITH events AS ({base_event_sql}), ordered AS (
            SELECT *,
                lag(pulse_number) OVER (
                    PARTITION BY pair_id, post_recording_id, pre_recording_id,
                                 stim_name, induction_frequency, recovery_delay
                    ORDER BY onset_time, stim_pulse_id
                ) AS previous_number,
                lag(onset_time) OVER (
                    PARTITION BY pair_id, post_recording_id, pre_recording_id,
                                 stim_name, induction_frequency, recovery_delay
                    ORDER BY onset_time, stim_pulse_id
                ) AS previous_onset
            FROM events
        ), sequences AS (
            SELECT pair_id, post_recording_id, pre_recording_id, stim_name,
                   induction_frequency, recovery_delay,
                   count(*) AS event_rows,
                   count(DISTINCT pulse_number) AS distinct_pulses,
                   min(pulse_number) AS min_pulse,
                   max(pulse_number) AS max_pulse,
                   min(valid_identity) AS all_identity_valid,
                   sum(CASE WHEN previous_number IS NOT NULL
                              AND pulse_number <= previous_number THEN 1 ELSE 0 END)
                       AS bad_number_order,
                   sum(CASE WHEN previous_onset IS NOT NULL
                              AND onset_time <= previous_onset THEN 1 ELSE 0 END)
                       AS bad_time_order
            FROM ordered
            GROUP BY pair_id, post_recording_id, pre_recording_id, stim_name,
                     induction_frequency, recovery_delay
        )
        SELECT count(*) AS sequences,
               sum(CASE WHEN distinct_pulses = 12
                          AND min_pulse = 1 AND max_pulse = 12
                        THEN 1 ELSE 0 END) AS structural_12_pulse_sequences,
               sum(CASE WHEN event_rows = 12 AND distinct_pulses = 12
                          AND min_pulse = 1 AND max_pulse = 12
                          AND all_identity_valid = 1
                          AND bad_number_order = 0 AND bad_time_order = 0
                        THEN 1 ELSE 0 END) AS valid_ordered_12_pulse_sequences
        FROM sequences
    """
    sequences = dict(fetchall_schema_only(con, sequence_sql)[0])

    fit_relation = dict(fetchall_schema_only(con, FIT_RELATION_SQL)[0])
    fit_relation["duplicate_fit_rows"] = (
        int(fit_relation["fit_rows"] or 0)
        - int(fit_relation["distinct_pulse_response_ids"] or 0)
        - int(fit_relation["null_pulse_response_ids"] or 0)
    )

    exclusions = {
        key: int(value or 0)
        for key, value in {**relation, **ordering}.items()
        if key.endswith("mismatch")
        or key.startswith("nonincreasing")
        or key == "same_recording"
        or key == "duplicate_sequence_pulse_groups"
    }
    hard_violations = {
        "duplicate_fit_rows": int(fit_relation["duplicate_fit_rows"]),
        "null_fit_relation_ids": int(
        fit_relation["null_pulse_response_ids"] or 0
        ),
    }

    valid_sequences = int(sequences["valid_ordered_12_pulse_sequences"] or 0)

    return {
        "relation_counts": relation,
        "ordering_counts": ordering,
        "sequence_counts": sequences,
        "fit_relation_counts": fit_relation,
        "sequence_exclusion_counts": exclusions,
        "hard_relation_violations": hard_violations,
        "relation_order_pass": (
            valid_sequences > 0
            and all(value == 0 for value in hard_violations.values())
        ),
    }


def sqlite_checks(con: sqlite3.Connection, full_integrity: bool) -> dict[str, Any]:
    quick = [row[0] for row in con.execute("PRAGMA quick_check")]
    foreign_key_rows = [list(row) for row in con.execute("PRAGMA foreign_key_check")]
    result: dict[str, Any] = {
        "quick_check": quick,
        "quick_check_pass": quick == ["ok"],
        "foreign_key_check_rows": foreign_key_rows,
        "foreign_key_check_pass": not foreign_key_rows,
        "full_integrity_requested": full_integrity,
    }
    if full_integrity:
        integrity = [row[0] for row in con.execute("PRAGMA integrity_check")]
        result["integrity_check"] = integrity
        result["integrity_check_pass"] = integrity == ["ok"]
    else:
        result["integrity_check"] = None
        result["integrity_check_pass"] = False
    return result


def audit_database(path: Path, full_integrity: bool) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    auditor_path = Path(__file__).resolve(strict=True)
    actual_bytes = resolved.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise AuditFailure(
            f"byte count mismatch: expected {EXPECTED_BYTES}, observed {actual_bytes}"
        )

    file_hash = sha256_file(resolved)
    uri = f"file:{resolved.as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        integrity = sqlite_checks(con, full_integrity=full_integrity)
        schema = inspect_schema(con)
        if not schema["schema_pass"]:
            raise AuditFailure("required schema or foreign-key provenance is missing")
        counts = inspect_table_counts(con)
        splits = inspect_split_groups(con)
        relations = inspect_relations_and_order(con)
    finally:
        con.close()

    passed = (
        integrity["quick_check_pass"]
        and integrity["foreign_key_check_pass"]
        and integrity["integrity_check_pass"]
        and schema["schema_pass"]
        and relations["relation_order_pass"]
    )
    return {
        "status": "PASS_SCHEMA_ONLY" if passed else "BLOCKED_SCHEMA_ONLY",
        "auditor": {
            "version": AUDITOR_VERSION,
            "path": str(auditor_path),
            "sha256": sha256_file(auditor_path),
            "stim_cell_identity_rule": (
                "stim_pulse.cell_id IS NULL OR "
                "stim_pulse.cell_id = pair.pre_cell_id"
            ),
            "identity_source": (
                "pre/post recording electrode, shared sync_rec, and "
                "pair experiment"
            ),
        },
        "outcome_values_read": False,
        "confirmation_outcomes_read": False,
        "database": str(resolved),
        "expected_bytes": EXPECTED_BYTES,
        "observed_bytes": actual_bytes,
        "sha256": file_hash,
        "integrity": integrity,
        "schema": schema,
        "table_counts": counts,
        "split_groups": splits,
        "relations_and_order": relations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-integrity",
        action="store_true",
        help="Run full PRAGMA integrity_check; required for a PASS receipt.",
    )
    args = parser.parse_args()

    try:
        receipt = audit_database(args.database, full_integrity=args.full_integrity)
    except (AuditFailure, OSError, sqlite3.Error) as exc:
        print(json.dumps({"status": "BLOCKED_SCHEMA_ONLY", "error": str(exc)}))
        return 2

    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if receipt["status"] == "PASS_SCHEMA_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
