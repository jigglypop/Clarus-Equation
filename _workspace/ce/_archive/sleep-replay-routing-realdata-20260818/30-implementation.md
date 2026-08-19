# Real-data implementation

Status: COMPLETE

## Acquired objects

The run acquired official E15 processed replay data and code, official E19 participant-level sleep/RSA matrices and MATLAB figure code, and official E13 criticality source tables and code. E02 metadata were acquired, but its file streams required credentials or returned HTTP 403. See `artifacts/acquisition-receipt.md` and `artifacts/realdata-manifest.csv`.

## Analysis

`artifacts/analyze_real_brain_data.py` performs three independent reproductions.

1. E15: loads official one-hour replay counts and bootstrap distributions, translates `get_bootstrap_prob` directly, and compares NSD with SD at registered epochs. The 13 released session labels are retained, but the processed table does not establish that they are independent animals or sessions.
2. E19: applies the official participant exclusion, MATLAB condition indices, encoding baseline adjustment, and released cluster coordinates, then recomputes participant-level REM/SWS Spearman associations.
3. E13: reads the released Figure 2 source workbook and recomputes DCC prediction means and the DCC-plus-baseline increment.

The script also hashes every acquired non-Git file and writes `artifacts/realdata-results.json`. It deliberately does not estimate E15 branching from firing-rate summaries or turn E13 aggregate/source data into raw event data.

## Canonical update

`docs/7_AGI/3_Sleep.md` now records the reproduced empirical values and the same-window data limitation. `docs/7_AGI/1_AGI.md` carries the high-level boundary: biological effects were reproduced, but branching-to-replay, dream generation, integrated mechanism, and AGI superiority were not.
