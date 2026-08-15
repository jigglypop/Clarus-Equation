# Validation command ledger

Snapshot: `5414336ae2ff20197efe3bf8a92ec5183ad079aa` on `main`, dirty worktree  
Environment: Python 3.11.9, uv 0.11.9, Windows/PowerShell  
Run date: 2026-08-15 (Asia/Seoul)

No product source was changed for this audit. This ledger records the machine
checks; it does not promote any physical bridge to a theorem or prediction.

## Canonical CE harness

| Command | Exit | Essential result |
|---|---:|---|
| `python reality_stone\python\reality_stone\clarus\bootstrap_solver.py` | 0 | low fixed point `0.0486466333`; residual `2.08e-17`; Newton/Brent difference `1.25e-13`; SciPy cross-check skipped because direct interpreter lacked SciPy |
| `python tests\scorecard.py` | 0 | 23 total; 12 scored; 11 `<1 sigma`; one CAUTION (`Omega_b h^2`, `-1.80 sigma`); 9 exact/reference; 1 external input; 1 open test; aggregate CAUTION |
| `python tests\run_validation.py` | 0 | bootstrap PASS; scorecard CAUTION; registered dimension checks 7/7; overall CAUTION |
| `python examples\physics\proof_completion_attempt.py` | 0 | raw `A_s` route `+197.80 sigma`; effective projected readout `+0.17 sigma`; output itself calls the latter a candidate/bridge |
| `uv run --extra dev python -m pytest tests\test_bootstrap_solver.py tests\test_dimensionless.py tests\test_layer_a.py tests\test_bridge_gates.py -q` | 0 | 58 passed, 2 PyTorch sparse warnings |

## Focused cosmology and document gates

| Command | Exit | Essential result |
|---|---:|---|
| focused eight-file cosmology pytest set | 0 | 85 passed in 9.42 s |
| `python -m pytest tests\test_dimensionless.py -q` | 0 | 15 passed; one cache-permission warning |
| `uv run --extra dev python -m pytest tests\test_canonical_document_policy.py -q` | 0 | 5 passed |
| `python docs\2_경로적분과_응용\validate_manuscript.py` | 0 | 47/47 implemented document/arithmetic/routing checks; validator explicitly reports zero active `[예측]` and zero CE-specific physical closures |
| `python experiments\preregistration\validate_holdout_manifest.py experiments\preregistration\cosmology_future_holdout_v2.json` | 0 | manifest valid; future holdout unassigned; evaluation `NOT_READY` |

The focused 85-test set was:

```text
tests/test_cosmology_ratio_audit.py
tests/test_ce_residual_forward_model.py
tests/test_recombination_drag_adapter.py
tests/test_primordial_spectrum_readout_gate.py
tests/test_holdout_preregistration.py
tests/test_proof_completion_attempt.py
tests/test_core_model_selection.py
tests/test_run_validation_consistency.py
```

There is no test module for `examples/physics/hubble_tension.py` or
`examples/physics/cosmological_constant_holographic_gate.py`.

## DESI DR2 embedded-vector diagnostics

`python examples\physics\ce_residual_forward_model.py --bao-dataset desi-dr2-all`
returned, for the external `r_d=147.09 Mpc` candidate,

- fixed model: `chi2=37.100260857`, `dof=13`, `p=0.000399573259824`;
- one-scale diagnostic: `q=0.986476933470`, `chi2=12.608346862`,
  `dof=12`, `p=0.398138192515`;
- equivalent scale at fixed `r_d`: `H0=68.323949312 km/s/Mpc`.

`python examples\physics\ce_residual_forward_model.py --rd-mode early-universe
--bao-dataset desi-dr2-all` returned

- Eisenstein--Hu hybrid `z_drag=1020.020419907`, `r_d=151.318753028 Mpc`;
- fixed model: `chi2=40.468225544`, `dof=13`, `p=0.000116176098098`;
- the same one-scale shape diagnostic `chi2=12.608346862`, `p=0.398138192515`.

The embedded DESI DR2 vector was inspected before preregistration and is marked
exploratory in the v2 manifest. These p-values are therefore not confirmatory
holdout verdicts.

## Executable research branches

| Script | Exit | Output that requires theory-status audit |
|---|---:|---|
| `cosmology_ratio_audit.py` | 0 | all four hard-coded central-value baselines within the arbitrary 4% relative window; no covariance or uncertainty used |
| `cosmology.py --print-h0t0 --extended --z-list 0,0.5,1,2` | 0 | LO default `(Omega_m,Omega_Lambda)=(0.307918429,0.692081571)` and `H0 t0=0.957087030` |
| `ce_residual_forward_model.py` | 0 | different 3-layer/default-constant background `(0.311000,0.689100)` |
| `primordial_spectrum_readout_gate.py` | 0 | raw route rejected; selected effective projection gives `A_s*1e9=2.1038087`, `+0.17 sigma` on reused Planck snapshot |
| `cosmological_constant_holographic_gate.py` | 0 | conditional phase-area readout gives `2.2412 meV`, `+0.054%`; executable says “zero free parameters” although canonical prose classifies the bridge as empirical/conditional |
| `hubble_tension.py` | 0 | prints `Delta H0=+5.5595`; independent math audit found a radiation-era Ricci-term error and an unused `omega_b h^2` argument, so this number is not a valid conditional cosmology result |

## Full repository regression

`uv run --extra dev python -m pytest -q` exited 1 after 280.48 s:

```text
49 failed, 2500 passed, 14 skipped, 2 warnings, 41 errors
```

The dominant errors are absent ScienceDB fusion payloads and missing/changed
AGI/Q0 benchmark manifests or sealed artifacts in the dirty checkout. No test in
the focused cosmology set failed. Compared with the same-day predecessor run
(`32 failed, 2517 passed, 14 skipped, 41 errors`), 17 nodes moved from pass to
fail; this is a repository-level regression, not evidence about the cosmology
equations.

## Static check

A focused Ruff invocation exited 1 with nine findings. Eight are mechanical
unused-import/f-string findings. The material one is
`hubble_tension.py:285`: a background `h2` value is computed but never used,
consistent with the independent finding that the purported LCDM acoustic-scale
calculation does not use all declared inputs.

