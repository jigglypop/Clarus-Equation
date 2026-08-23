"""Audit clamp-dependent units in the frozen BA-SRM3 eligible train cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np


VERSION = "BA-SRM3-CLAMP-UNIT-AUDIT-V1"
EXPECTED_DATABASE_SHA256 = (
    "dbf19786f9e0d0d73c26351dc29d69ef8c10a2e67e32e19ac73034a5624d48c5"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "4ddb4a52294a55b011c5118a02432ca28c057ca5b5ebb63d8d7c945923aa62c2"
)
EXPECTED_ELIGIBLE_MANIFEST_SHA256 = (
    "74d6d3b142e48d7906305e133983b91cc8227a40748335985414e674dd1fd81c"
)
INVALID_EXTRACTOR_SHA256 = (
    "bc5602af2c9e5a9351fcf1ae74e0c64f87d91d3196759dd2d20e0e6c839f23ab"
)


class AuditFailure(RuntimeError):
    pass


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise AuditFailure(f"SHA-256 mismatch: {path.name}")
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"n": 0, "q00": None, "q50": None, "q99": None, "q100": None}
    q = np.quantile(finite, [0.0, 0.5, 0.99, 1.0])
    return {
        "n": int(finite.size),
        "q00": float(q[0]),
        "q50": float(q[1]),
        "q99": float(q[2]),
        "q100": float(q[3]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise AuditFailure("refusing to overwrite clamp-unit receipt")
        database = args.database.resolve(strict=True)
        if sha256_file(database) != EXPECTED_DATABASE_SHA256:
            raise AuditFailure("database SHA-256 mismatch")
        if sha256_file(args.extractor) != INVALID_EXTRACTOR_SHA256:
            raise AuditFailure("extractor SHA-256 mismatch")
        source = load_jsonl(args.source_manifest, EXPECTED_SOURCE_MANIFEST_SHA256)
        eligible = load_jsonl(args.eligible_manifest, EXPECTED_ELIGIBLE_MANIFEST_SHA256)
        source_by_key = {str(row["sequence_key"]): row for row in source}
        selected = []
        for item in eligible:
            row = source_by_key.get(str(item["sequence_key"]))
            if row is None:
                raise AuditFailure("eligible sequence absent from source manifest")
            selected.append(row)

        con = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True
        )
        con.row_factory = sqlite3.Row
        try:
            con.execute(
                """
                CREATE TEMP TABLE selected_unit_sequence (
                    sequence_key TEXT PRIMARY KEY,
                    synapse_type TEXT NOT NULL,
                    pair_id INTEGER NOT NULL,
                    pre_recording_id INTEGER NOT NULL,
                    post_recording_id INTEGER NOT NULL,
                    post_stim_name TEXT,
                    induction_frequency REAL,
                    recovery_delay REAL
                )
                """
            )
            con.executemany(
                "INSERT INTO selected_unit_sequence VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["sequence_key"],
                        row["synapse_type"],
                        row["pair_id"],
                        row["pre_recording_id"],
                        row["post_recording_id"],
                        row["post_stim_name"],
                        row["induction_frequency"],
                        row["recovery_delay"],
                    )
                    for row in selected
                ],
            )
            rows = list(
                con.execute(
                    """
                    SELECT
                        ss.sequence_key,
                        ss.synapse_type,
                        pre_pcr.clamp_mode AS pre_clamp_mode,
                        post_pcr.clamp_mode AS post_clamp_mode,
                        sp.pulse_number,
                        sp.amplitude AS stim_amplitude_raw,
                        prf.dec_fit_reconv_amp AS response_amplitude_raw,
                        post_pcr.baseline_noise_stdev AS baseline_noise_raw
                    FROM selected_unit_sequence ss
                    JOIN pulse_response pr
                      ON pr.pair_id = ss.pair_id
                     AND pr.recording_id = ss.post_recording_id
                    JOIN stim_pulse sp
                      ON sp.id = pr.stim_pulse_id
                     AND sp.recording_id = ss.pre_recording_id
                    JOIN recording pre_r ON pre_r.id = ss.pre_recording_id
                    JOIN recording post_r ON post_r.id = ss.post_recording_id
                    JOIN patch_clamp_recording pre_pcr
                      ON pre_pcr.recording_id = pre_r.id
                    JOIN patch_clamp_recording post_pcr
                      ON post_pcr.recording_id = post_r.id
                    JOIN multi_patch_probe mpp
                      ON mpp.patch_clamp_recording_id = post_pcr.id
                    LEFT JOIN pulse_response_fit prf
                      ON prf.pulse_response_id = pr.id
                    WHERE post_r.stim_name IS ss.post_stim_name
                      AND mpp.induction_frequency = ss.induction_frequency
                      AND mpp.recovery_delay = ss.recovery_delay
                    ORDER BY ss.sequence_key, sp.pulse_number, sp.id
                    """
                )
            )
        finally:
            con.close()

        by_sequence: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            by_sequence[str(row["sequence_key"])].append(row)
        if set(by_sequence) != {str(row["sequence_key"]) for row in selected}:
            raise AuditFailure("unit audit sequence set mismatch")
        if any(
            len(items) != 12
            or sorted(int(row["pulse_number"]) for row in items) != list(range(12))
            for items in by_sequence.values()
        ):
            raise AuditFailure("unit audit did not recover exact pulse 0..11")

        mode_counts: Counter[str] = Counter()
        stim_by_pre_mode: dict[str, list[float]] = defaultdict(list)
        response_by_post_mode: dict[str, list[float]] = defaultdict(list)
        noise_by_post_mode: dict[str, list[float]] = defaultdict(list)
        for key, items in by_sequence.items():
            first = items[0]
            label = str(first["synapse_type"])
            pre_mode = str(first["pre_clamp_mode"])
            post_mode = str(first["post_clamp_mode"])
            mode_counts[f"{label}|pre={pre_mode}|post={post_mode}"] += 1
            stim_by_pre_mode[pre_mode].extend(
                float(row["stim_amplitude_raw"])
                for row in items
                if row["stim_amplitude_raw"] is not None
            )
            response_by_post_mode[post_mode].extend(
                float(row["response_amplitude_raw"])
                for row in items
                if int(row["pulse_number"]) in range(8, 12)
                and row["response_amplitude_raw"] is not None
            )
            if first["baseline_noise_raw"] is not None:
                noise_by_post_mode[post_mode].append(float(first["baseline_noise_raw"]))

        post_modes = sorted({str(items[0]["post_clamp_mode"]) for items in by_sequence.values()})
        pre_modes = sorted({str(items[0]["pre_clamp_mode"]) for items in by_sequence.values()})
        invalid = len(post_modes) > 1 or post_modes != ["ic"] or pre_modes != ["ic"]
        receipt = {
            "status": "INVALIDATED_CLAMP_UNIT_CONTRACT" if invalid else "PASS_CLAMP_UNIT_AUDIT",
            "version": VERSION,
            "database_sha256": EXPECTED_DATABASE_SHA256,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "eligible_manifest_sha256": EXPECTED_ELIGIBLE_MANIFEST_SHA256,
            "audited_extractor_sha256": INVALID_EXTRACTOR_SHA256,
            "eligible_sequences": len(selected),
            "event_rows": len(rows),
            "mode_sequence_counts": dict(sorted(mode_counts.items())),
            "pre_clamp_modes": pre_modes,
            "post_clamp_modes": post_modes,
            "raw_stim_amplitude_by_pre_mode": {
                mode: quantiles(values) for mode, values in sorted(stim_by_pre_mode.items())
            },
            "raw_response_amplitude_pulses_8_11_by_post_mode": {
                mode: quantiles(values)
                for mode, values in sorted(response_by_post_mode.items())
            },
            "raw_baseline_noise_by_post_mode": {
                mode: quantiles(values) for mode, values in sorted(noise_by_post_mode.items())
            },
            "unit_finding": (
                "stimulus command amplitude is clamp-mode dependent; postsynaptic response "
                "amplitude and baseline noise are volts in IC and amperes in VC"
            ),
            "extractor_error": (
                "extractor treated every stimulus amplitude as amperes and every response/noise "
                "amplitude as volts while omitting pre/post clamp mode"
            ),
            "prior_train_operator_valid": False,
            "prior_rank_stop_is_biological_evidence": False,
            "development_outcomes_read": False,
            "confirmation_outcomes_read": False,
            "waveform_blobs_read": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_name(args.output.name + ".partial")
        if partial.exists():
            raise AuditFailure("stale partial clamp-unit receipt")
        text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        partial.write_text(text, encoding="utf-8")
        if partial.read_text(encoding="utf-8") != text:
            raise AuditFailure("partial clamp-unit receipt verification failed")
        partial.replace(args.output)
    except (AuditFailure, OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_CLAMP_UNIT_AUDIT", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS_CLAMP_UNIT_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
