# Publication-native event recovery validation

Status: COMPLETE

## Commands

The initial online invocation used the repository-approved Windows interpreter path and froze the seven-page OSF metadata response, downloaded the bounded selected native files, and downloaded the pinned Zenodo archive:

```powershell
.\.codex\hooks\python.cmd python _workspace\ce\randi-native-event-recovery-20260820\artifacts\native_event_recovery_audit.py --run-dir _workspace\ce\randi-native-event-recovery-20260820
```

The deterministic local audit was then rerun without any network access:

```powershell
.\.codex\hooks\python.cmd python _workspace\ce\randi-native-event-recovery-20260820\artifacts\native_event_recovery_audit.py --run-dir _workspace\ce\randi-native-event-recovery-20260820 --offline
.\.codex\hooks\python.cmd python -m py_compile _workspace\ce\randi-native-event-recovery-20260820\artifacts\native_event_recovery_audit.py
git diff --check -- _workspace/ce/randi-native-event-recovery-20260820
```

All three local validations completed successfully. The offline machine output was:

```json
{"files": 538, "overall": "BLOCKED_SOURCE_INDEX_JOIN", "sessions": 113, "source_index_join": false}
```

## Receipts and results

- Full OSF manifest: 68 metadata pages; SHA-256 `962140bba5863602273a2cf9c32a97c7c1a40456e3620f3fa53ecc522686a8b3`.
- Exact eligible files present: 538: `ds_name` 108, `labels` 108, `stim_neurons` 108, `stim_volume_i` 108, and `t` 106. Downloaded native byte total: 9,346,585.
- The manifest contains 113 numeric-prefix sessions, but only 87 have all five permitted families; 26 are incomplete. The 87 complete sessions have equal stimulation/volume row counts, valid nonnegative stimulation indices against the blank-preserving label table, and a nonempty monotone clock. Their source-index joins contain 827 blank local-label entries, so they do not constitute canonical identities.
- Zenodo `pumpprobe-1.1.zip`: 1,287,278 bytes; MD5 `40d87e790193d38528b4ba0cecf23e8c`; local SHA-256 `e6a52ec8fbaa2cdb8da2b72549495d5c91f459786bcf61549b0c771e803a9378`.
- ZIP static audit: 230 members, 6,109,093 uncompressed bytes, no unsafe member; 91 source files scanned. It found 458 native-field-loading-path hits and static terminology hits (`automatic` 9, `manual` 59, `fail` 2, `exclude` 188). These code hits are schema evidence only, not event-level assignment receipts.
- R1b converter: the codeload URL and archive root both carry frozen commit `3544c9bb59f90d5630fa1871850d990db9cafc18`; 47,983 bytes; SHA-256 `9d6d01e74243f5be17204d0a086a33a74a98c8e98ec2f7befa57089eee0bc603`. The combined acquisition receipt is 540 files and 10,681,846 bytes, within the 602-file/32,000,000-byte cap.
- Converter static mapping finds `TargetPlaneSegmentation`, NeuroPAL, and the complementary label/confidence/comment schema. It sees `targets_manually_located.txt`, a manually targeted ID output, and a static conversion of “not manually located or failed targeting” to `NaN`. It does not find `stim_neurons`/`stim_volume_i` or `-1/-2/-3` as preserved native-event mappings. This is evidence of a lossy/insufficient converter path for a complete assignment or canonical-provenance receipt, not evidence that a missing source event never occurred.

The authoritative structured evidence is [`artifacts/native_event_audit.json`](artifacts/native_event_audit.json). Its gates are `PASS_SOURCE_INDEX_JOIN: false`, `PASS_SOURCE_JOIN: false`, `PASS_ASSIGNMENT_RECEIPT: false`, `PASS_APPARATUS_INPUT: false`, and overall `BLOCKED_SOURCE_INDEX_JOIN`.

No fluorescence, response, autoresponse amplitude, fitted kernel, effect, state, p/q value, or outcome-derived exclusion data were read. `git diff --check` passed.
