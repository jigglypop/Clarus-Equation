from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import sys

import pytest


MODULE_PATH = Path(__file__).with_name("audit_medium_schema.py")
SPEC = importlib.util.spec_from_file_location("ba_srm2_medium_schema", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_split_is_deterministic_and_exhaustive() -> None:
    ids = ["slice-a", "slice-b", "slice-c", "slice-d"]
    first = [(AUDIT.split_bucket(item), AUDIT.split_name(AUDIT.split_bucket(item))) for item in ids]
    second = [(AUDIT.split_bucket(item), AUDIT.split_name(AUDIT.split_bucket(item))) for item in ids]

    assert first == second
    assert all(0 <= bucket <= 9 for bucket, _ in first)
    assert all(split in {"train", "development", "confirmation"} for _, split in first)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT dec_fit_reconv_amp FROM pulse_response_fit",
        "SELECT * FROM pulse_response pr WHERE pr.ex_qc_pass = 1",
        "SELECT sp.id FROM stim_pulse sp WHERE sp.qc_pass = 1",
    ],
)
def test_schema_only_guard_rejects_locked_value_or_qc_columns(sql: str) -> None:
    with pytest.raises(AUDIT.AuditFailure, match="locked value/QC column"):
        AUDIT.assert_schema_only_sql(sql)


def test_schema_only_guard_allows_identity_and_order_metadata() -> None:
    AUDIT.assert_schema_only_sql(
        "SELECT pr.id, pr.pair_id, sp.pulse_number, sp.onset_time "
        "FROM pulse_response pr JOIN stim_pulse sp ON sp.id=pr.stim_pulse_id"
    )


def test_schema_only_guard_rejects_wildcard_fit_table_access() -> None:
    with pytest.raises(AUDIT.AuditFailure, match="identity/count allowlist"):
        AUDIT.assert_schema_only_sql("SELECT * FROM pulse_response_fit")


def test_schema_only_guard_allows_only_frozen_fit_identity_queries() -> None:
    AUDIT.assert_schema_only_sql('SELECT count(*) FROM "pulse_response_fit"')
    AUDIT.assert_schema_only_sql(AUDIT.FIT_RELATION_SQL)


def test_all_future_output_fields_are_schema_requirements() -> None:
    required = AUDIT.REQUIRED_COLUMNS["pulse_response_fit"]

    assert set(AUDIT.TARGET_VALUE_COLUMNS) <= required


def test_locked_small_release_has_the_required_relational_schema() -> None:
    repository = Path(__file__).resolve().parents[4]
    database = repository / "data/external/allen-synphys/raw/synphys_r2.1_small.sqlite"
    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        schema = AUDIT.inspect_schema(con)
    finally:
        con.close()

    assert schema["schema_pass"], schema


def test_locked_small_release_exercises_structural_sql_without_outcomes() -> None:
    repository = Path(__file__).resolve().parents[4]
    database = repository / "data/external/allen-synphys/raw/synphys_r2.1_small.sqlite"
    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        relations = AUDIT.inspect_relations_and_order(con)
    finally:
        con.close()

    assert relations["sequence_counts"]["sequences"] == 0
    assert relations["relation_order_pass"] is False
