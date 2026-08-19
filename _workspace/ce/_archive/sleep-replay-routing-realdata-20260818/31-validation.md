# Validation

Status: COMPLETE

## Commands

The analysis ran in an isolated temporary `uv` environment with SciPy and openpyxl:

```powershell
$env:UV_CACHE_DIR="$env:TEMP\uv-cache-ce"
uv run --no-project --with scipy --with openpyxl --python C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe python _workspace\ce\sleep-replay-routing-realdata-20260818\artifacts\analyze_real_brain_data.py
```

Result: exit 0. It wrote `artifacts/realdata-results.json` and a SHA-256 manifest for 574 acquired non-Git files.

Focused verification:

```text
python -B ...\artifacts\verify_realdata_analysis.py
OK realdata: 574 hashed files; E15/E19/E13 checks passed
```

Source-only compile:

```text
OK compile: 2 files
```

## Reproduction checks

| Check | Result |
|---|---|
| E15 official 0--1 h bootstrap probability | `0.0161057`, matches notebook `0.01611` |
| E15 official 5--6 h bootstrap probability | `0.00133728`, matches notebook `0.001337` |
| E15 direction | SD lower than NSD at 0--1 h and 5--6 h |
| E19 official participant count | 34 after participant 5 exclusion |
| E19 item/category directions | item negative, category positive |
| E13 source-table direction | DCC prediction above shuffle; DCC+Base above Base |
| E02 payload | access error only; rejected as neural data |

## Interpretation boundary

This validation establishes deterministic file acquisition, schema handling and numerical reproduction. It does not turn group bootstrap draws into independent animals, source tables into raw events, or cross-study agreement into a biological or AGI theorem. The full repository test suite was not run because the implementation is a self-contained research artifact and the focused source/data checks passed.
