"""Outcome-free schema and support audit for Allen-Synphys r2.1 small DB."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {
    "synapse": [
        "id",
        "pair_id",
        "synapse_type",
        "latency",
        "psp_amplitude",
        "psp_rise_time",
        "psp_decay_tau",
    ],
    "dynamics": [
        "pair_id",
        "qc_pass",
        "pulse_amp_stp_initial_50hz",
        "pulse_amp_stp_induction_50hz",
        "pulse_amp_stp_recovery_250ms",
        "variability_stp_induced_state_50hz",
    ],
    "pair": ["id", "experiment_id", "pre_cell_id", "post_cell_id", "has_synapse"],
    "experiment": ["id", "ext_id", "slice_id", "project_name", "target_region"],
    "slice": ["id", "ext_id", "species"],
    "resting_state_fit": ["synapse_id", "ic_pulse_ids"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for(group_id: str) -> str:
    bucket = hashlib.sha256(group_id.encode("utf-8")).digest()[0] % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "development"
    return "confirmation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    path = args.database.resolve(strict=True)

    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    tables = [row[0] for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    columns = {
        table: [row[1] for row in con.execute(f'PRAGMA table_info("{table}")')]
        for table in REQUIRED_COLUMNS
        if table in tables
    }
    required = {
        table: {
            col: table in columns and col in columns[table]
            for col in cols
        }
        for table, cols in REQUIRED_COLUMNS.items()
    }

    def scalar(sql: str, params: tuple = ()):
        return con.execute(sql, params).fetchone()[0]

    counts = {
        "tables": len(tables),
        "slice_rows": scalar("SELECT count(*) FROM slice"),
        "experiment_rows": scalar("SELECT count(*) FROM experiment"),
        "cell_rows": scalar("SELECT count(*) FROM cell"),
        "synapse_rows": scalar("SELECT count(*) FROM synapse"),
        "dynamics_rows": scalar("SELECT count(*) FROM dynamics"),
        "dynamics_qc_rows": scalar("SELECT count(*) FROM dynamics WHERE qc_pass=1"),
        "stim_pulse_rows": scalar("SELECT count(*) FROM stim_pulse"),
        "pulse_response_rows": scalar("SELECT count(*) FROM pulse_response"),
        "pulse_response_fit_rows": scalar("SELECT count(*) FROM pulse_response_fit"),
        "intrinsic_rows": scalar("SELECT count(*) FROM intrinsic"),
        "resting_state_fit_rows": scalar("SELECT count(*) FROM resting_state_fit"),
        "resting_state_ic_pulse_ids_nonnull": scalar(
            "SELECT count(*) FROM resting_state_fit WHERE ic_pulse_ids IS NOT NULL"
        ),
        "resting_state_ic_pulse_ids_nonempty": scalar(
            "SELECT count(*) FROM resting_state_fit WHERE length(ic_pulse_ids) > 0"
        ),
    }

    base_from = """
        FROM synapse AS sy
        JOIN pair AS pa ON pa.id = sy.pair_id
        JOIN experiment AS ex ON ex.id = pa.experiment_id
        JOIN slice AS sl ON sl.id = ex.slice_id
        JOIN dynamics AS dy ON dy.pair_id = pa.id
    """
    target_domain = """
        dy.qc_pass = 1
        AND sy.psp_amplitude IS NOT NULL AND abs(sy.psp_amplitude) > 0
        AND dy.pulse_amp_stp_initial_50hz IS NOT NULL
        AND dy.pulse_amp_stp_induction_50hz IS NOT NULL
        AND dy.pulse_amp_stp_recovery_250ms IS NOT NULL
        AND dy.variability_stp_induced_state_50hz IS NOT NULL
    """
    shared_summary_domain = target_domain + """
        AND sy.latency IS NOT NULL AND sy.latency > 0
        AND sy.psp_rise_time IS NOT NULL AND sy.psp_rise_time > 0
        AND sy.psp_decay_tau IS NOT NULL AND sy.psp_decay_tau > 0
    """
    counts["joined_synapse_dynamics_rows"] = scalar("SELECT count(*) " + base_from)
    counts["target_complete_rows_all"] = scalar(
        "SELECT count(*) " + base_from + " WHERE " + target_domain
    )
    counts["target_complete_slice_groups_all"] = scalar(
        "SELECT count(DISTINCT sl.ext_id) " + base_from + " WHERE " + target_domain
    )
    counts["shared_summary_complete_rows_all"] = scalar(
        "SELECT count(*) " + base_from + " WHERE " + shared_summary_domain
    )
    counts["shared_summary_complete_slice_groups_all"] = scalar(
        "SELECT count(DISTINCT sl.ext_id) " + base_from + " WHERE " + shared_summary_domain
    )

    strict_from = base_from + " JOIN intrinsic AS ipost ON ipost.cell_id = pa.post_cell_id\n"
    strict_domain = target_domain + """
        AND pa.distance IS NOT NULL AND pa.distance > 0
        AND ipost.input_resistance IS NOT NULL AND ipost.input_resistance > 0
        AND ipost.tau IS NOT NULL AND ipost.tau > 0
    """
    counts["strict_complete_rows_all"] = scalar(
        "SELECT count(*) " + strict_from + " WHERE " + strict_domain
    )
    counts["strict_complete_slice_groups_all"] = scalar(
        "SELECT count(DISTINCT sl.ext_id) " + strict_from + " WHERE " + strict_domain
    )

    strata_rows = con.execute(
        """
        SELECT sl.species, ex.project_name, sy.synapse_type,
               count(*) AS pairs, count(DISTINCT sl.ext_id) AS slices
        """
        + base_from
        + " WHERE "
        + target_domain
        + " GROUP BY sl.species, ex.project_name, sy.synapse_type"
        + " ORDER BY pairs DESC"
    ).fetchall()
    strata = [dict(row) for row in strata_rows]

    strict_strata_rows = con.execute(
        """
        SELECT sl.species, ex.project_name, sy.synapse_type,
               count(*) AS pairs, count(DISTINCT sl.ext_id) AS slices
        """
        + strict_from
        + " WHERE "
        + strict_domain
        + " GROUP BY sl.species, ex.project_name, sy.synapse_type"
        + " ORDER BY pairs DESC"
    ).fetchall()
    strict_strata = [dict(row) for row in strict_strata_rows]

    primary_support = {}
    for synapse_type in ("ex", "in"):
        rows = con.execute(
            "SELECT sl.ext_id, count(*) AS pairs "
            + base_from
            + " WHERE "
        + target_domain
            + " AND sl.species='mouse'"
            + " AND ex.project_name IN ('mouse V1 coarse matrix', 'mouse V1 pre-production')"
            + " AND sy.synapse_type=?"
            + " GROUP BY sl.ext_id ORDER BY sl.ext_id",
            (synapse_type,),
        ).fetchall()
        split_groups = Counter(split_for(str(row[0])) for row in rows)
        split_pairs = Counter()
        for row in rows:
            split_pairs[split_for(str(row[0]))] += int(row[1])
        primary_support[synapse_type] = {
            "pairs": sum(int(row[1]) for row in rows),
            "slice_groups": len(rows),
            "slice_split_counts": dict(sorted(split_groups.items())),
            "pair_split_counts": dict(sorted(split_pairs.items())),
        }

    strict_primary_support = {}
    for synapse_type in ("ex", "in"):
        rows = con.execute(
            "SELECT sl.ext_id, count(*) AS pairs "
            + strict_from
            + " WHERE "
            + strict_domain
            + " AND sl.species='mouse'"
            + " AND ex.project_name IN ('mouse V1 coarse matrix', 'mouse V1 pre-production')"
            + " AND sy.synapse_type=?"
            + " GROUP BY sl.ext_id ORDER BY sl.ext_id",
            (synapse_type,),
        ).fetchall()
        split_groups = Counter(split_for(str(row[0])) for row in rows)
        split_pairs = Counter()
        for row in rows:
            split_pairs[split_for(str(row[0]))] += int(row[1])
        strict_primary_support[synapse_type] = {
            "pairs": sum(int(row[1]) for row in rows),
            "slice_groups": len(rows),
            "slice_split_counts": dict(sorted(split_groups.items())),
            "pair_split_counts": dict(sorted(split_pairs.items())),
        }

    pulse_types = [
        dict(row)
        for row in con.execute(
            """
            SELECT typeof(ic_pulse_ids) AS storage_type,
                   count(*) AS rows,
                   min(length(ic_pulse_ids)) AS min_bytes,
                   max(length(ic_pulse_ids)) AS max_bytes
            FROM resting_state_fit
            WHERE ic_pulse_ids IS NOT NULL
            GROUP BY typeof(ic_pulse_ids)
            ORDER BY storage_type
            """
        )
    ]

    metadata = []
    if "metadata" in tables:
        metadata = [dict(row) for row in con.execute("SELECT * FROM metadata")]

    report = {
        "status": "SCHEMA_ONLY_NO_RESPONSE_VALUES_REPORTED",
        "database": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "integrity_check": scalar("PRAGMA integrity_check"),
        "tables": tables,
        "required_columns": required,
        "counts": counts,
        "complete_strata": strata,
        "strict_complete_strata": strict_strata,
        "primary_support_by_synapse_type": primary_support,
        "strict_primary_support_by_synapse_type": strict_primary_support,
        "resting_state_ic_pulse_id_storage": pulse_types,
        "metadata": metadata,
        "nonfinite_policy": "SQLite NULL is counted as missing; no epsilon or imputation used.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    con.close()


if __name__ == "__main__":
    main()
