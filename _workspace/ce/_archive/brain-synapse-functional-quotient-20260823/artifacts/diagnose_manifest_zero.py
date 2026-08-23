"""Outcome-blind structural diagnostics for an empty BA-SRM2 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


FORBIDDEN = (
    "pulse_response_fit",
    "dec_fit_",
    "ex_qc_pass",
    "in_qc_pass",
    "qc_pass",
    "stim_pulse.data",
    "pulse_response.data",
)

QUERIES = {
    "slice_species": """
        SELECT species, count(*) AS n
        FROM slice GROUP BY species ORDER BY n DESC
    """,
    "mouse_projects": """
        SELECT ex.project_name, count(DISTINCT ex.id) AS experiments,
               count(DISTINCT sl.id) AS slices
        FROM experiment ex JOIN slice sl ON sl.id = ex.slice_id
        WHERE sl.species = 'mouse'
        GROUP BY ex.project_name ORDER BY slices DESC
    """,
    "mouse_synapse_projects": """
        SELECT ex.project_name, sy.synapse_type,
               count(DISTINCT pa.id) AS pairs,
               count(DISTINCT sl.id) AS slices
        FROM slice sl
        JOIN experiment ex ON ex.slice_id = sl.id
        JOIN pair pa ON pa.experiment_id = ex.id
        JOIN synapse sy ON sy.pair_id = pa.id
        WHERE sl.species = 'mouse' AND pa.has_synapse = 1
        GROUP BY ex.project_name, sy.synapse_type
        ORDER BY slices DESC
    """,
    "mouse_ic_protocol_projects": """
        SELECT ex.project_name, sy.synapse_type,
               count(*) AS event_rows,
               count(DISTINCT sl.id) AS slices,
               count(DISTINCT pa.id) AS pairs
        FROM slice sl
        JOIN experiment ex ON ex.slice_id = sl.id
        JOIN pair pa ON pa.experiment_id = ex.id
        JOIN synapse sy ON sy.pair_id = pa.id
        JOIN pulse_response pr ON pr.pair_id = pa.id
        JOIN recording post_r ON post_r.id = pr.recording_id
        JOIN patch_clamp_recording pcr ON pcr.recording_id = post_r.id
        JOIN multi_patch_probe mpp ON mpp.patch_clamp_recording_id = pcr.id
        WHERE sl.species = 'mouse' AND pa.has_synapse = 1
          AND sy.synapse_type IN ('ex', 'in')
          AND pcr.clamp_mode = 'ic'
          AND mpp.induction_frequency > 0 AND mpp.recovery_delay > 0
        GROUP BY ex.project_name, sy.synapse_type
        ORDER BY slices DESC
    """,
    "v1_ic_sequence_shapes": """
        WITH events AS (
            SELECT sy.synapse_type, pa.id AS pair_id,
                   post_r.id AS post_recording_id,
                   pre_r.id AS pre_recording_id,
                   post_r.stim_name,
                   mpp.induction_frequency, mpp.recovery_delay,
                   sp.pulse_number
            FROM slice sl
            JOIN experiment ex ON ex.slice_id = sl.id
            JOIN pair pa ON pa.experiment_id = ex.id
            JOIN synapse sy ON sy.pair_id = pa.id
            JOIN pulse_response pr ON pr.pair_id = pa.id
            JOIN stim_pulse sp ON sp.id = pr.stim_pulse_id
            JOIN recording post_r ON post_r.id = pr.recording_id
            JOIN recording pre_r ON pre_r.id = sp.recording_id
            JOIN patch_clamp_recording pcr ON pcr.recording_id = post_r.id
            JOIN multi_patch_probe mpp ON mpp.patch_clamp_recording_id = pcr.id
            WHERE sl.species = 'mouse'
              AND ex.project_name IN (
                  'mouse V1 coarse matrix', 'mouse V1 pre-production'
              )
              AND pa.has_synapse = 1
              AND sy.synapse_type IN ('ex', 'in')
              AND pcr.clamp_mode = 'ic'
              AND mpp.induction_frequency > 0
              AND mpp.recovery_delay > 0
        ), sequences AS (
            SELECT synapse_type, pair_id, post_recording_id, pre_recording_id,
                   stim_name, induction_frequency, recovery_delay,
                   count(*) AS event_rows,
                   count(DISTINCT pulse_number) AS distinct_pulses,
                   min(pulse_number) AS min_pulse,
                   max(pulse_number) AS max_pulse
            FROM events
            GROUP BY synapse_type, pair_id, post_recording_id, pre_recording_id,
                     stim_name, induction_frequency, recovery_delay
        )
        SELECT synapse_type, event_rows, distinct_pulses, min_pulse, max_pulse,
               count(*) AS sequences
        FROM sequences
        GROUP BY synapse_type, event_rows, distinct_pulses, min_pulse, max_pulse
        ORDER BY sequences DESC LIMIT 40
    """,
}


def assert_safe(sql: str) -> None:
    normalized = " ".join(sql.lower().split())
    for token in FORBIDDEN:
        if token in normalized:
            raise RuntimeError(f"locked token in structural diagnostic: {token}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = args.database.resolve(strict=True)
    con = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro&immutable=1", uri=True
    )
    con.row_factory = sqlite3.Row
    try:
        result = {}
        for name, sql in QUERIES.items():
            assert_safe(sql)
            result[name] = [dict(row) for row in con.execute(sql)]
    finally:
        con.close()
    receipt = {
        "status": "STRUCTURAL_DIAGNOSTIC_ONLY",
        "outcome_values_read": False,
        "response_qc_values_read": False,
        "confirmation_outcomes_read": False,
        "database": str(resolved),
        "queries": result,
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
