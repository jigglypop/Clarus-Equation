# Publication-native event recovery implementation

Status: COMPLETE

The implementation is [`artifacts/native_event_recovery_audit.py`](artifacts/native_event_recovery_audit.py), a standard-library-only, noninteractive Python program. It never opens a file whose name contains `gcamp`, `response`, `autoresponse`, `fluorescence`, `deltaf`, `dff`, `kernel`, `pvalue`, `qvalue`, `effect`, `state`, or `fit`.

## Acquisition and safety contract implemented

The program paginates the official OSF child relation for folder `671a5286badd54a2128707e3`, normalizes and freezes the full provider manifest in [`artifacts/osf_raw_extracted_manifest.json`](artifacts/osf_raw_extracted_manifest.json), then selects only exact case-insensitive suffixes `ds_name.txt`, `labels.txt`, `stim_neurons.txt`, `stim_volume_i.txt`, and `t.txt`. Selection order is family priority, declared byte size, and canonical path. It enforces 600/25,000,000-byte global and 120/15,000,000-byte family bounds before downloading; each request has a 60-second timeout, three attempts, exponential backoff, and an eight-worker upper bound.

The frozen selection receipt is [`artifacts/selected_native_files.json`](artifacts/selected_native_files.json). It records provider-checksum presence and verification independently from the locally calculated SHA-256. Native content is saved in `artifacts/native_files/` with deterministic collision-safe canonical names.

Only Zenodo record 8312985 file `pumpprobe-1.1.zip` is acquired. The program requires exactly 1,287,278 bytes and MD5 `40d87e790193d38528b4ba0cecf23e8c`, then records its local SHA-256. It uses `zipfile` metadata and bounded UTF-8 source-text reads without extraction, import, execution, or pickle loading. Absolute/traversal paths, encrypted members, symbolic links, excessive member count, and excessive decompressed size stop the run.

R1b additionally permits exactly `https://codeload.github.com/catalystneuro/leifer_lab_to_nwb/zip/3544c9bb59f90d5630fa1871850d990db9cafc18`, bounded to 5,000,000 bytes. Its commit-root path, local SHA-256, ZIP safety metadata, and source text are checked without extraction. The aggregate cap is 602 files and 32,000,000 bytes, including the existing OSF selection and both archives.

## Native parsing

The parser treats `labels.txt` as an index-addressed table and retains blank placeholder rows. It checks numeric prefixes and five-family presence per session; primitive numeric parsing of stimulation and clock files; event/cardinality agreement; integer/sentinel/index domains; nonblank local-label joins; and a nonempty monotone first clock column. Sentinels are reported rather than silently discarded. Static source scans separately report paths that load native fields and hits for `automatic`, `manual`, `fail`, `assignment`, and `exclude`.

The machine result has distinct gates: `PASS_SOURCE_INDEX_JOIN`, `PASS_SOURCE_JOIN`, and `PASS_ASSIGNMENT_RECEIPT`. Static code terminology cannot turn either of the last two into a pass.

The R1b static mapping audit separately identifies native stimulation-field presence/loss, `-1/-2/-3` event-sentinel mapping, complementary NeuroPAL label/confidence/comment schema, manual target input/output, failed-target representation, and `TargetPlaneSegmentation`/NeuroPAL paths. It reports schema evidence but never upgrades it to a rowwise receipt.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| OSF normalized manifest | `962140bba5863602273a2cf9c32a97c7c1a40456e3620f3fa53ecc522686a8b3` |
| Zenodo archive | `e6a52ec8fbaa2cdb8da2b72549495d5c91f459786bcf61549b0c771e803a9378` |
| Converter archive | `9d6d01e74243f5be17204d0a086a33a74a98c8e98ec2f7befa57089eee0bc603` |
| Audit script | `b6e39c19295c25049efdcee5e4de04c6e1f623ab621c6a7009d20ec23cb1e528` |
