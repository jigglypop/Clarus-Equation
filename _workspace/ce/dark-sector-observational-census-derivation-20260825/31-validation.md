# 31-validation — dark-sector census corrections

Status: COMPLETE

## 1. Focused test scope

The stable staged snapshot was tested with the repository's policy-allowed
system-Python harness, pytest cache disabled and a unique harness-owned
temporary base directory. Because the cosmology repository is split from its
core dependency, the sibling `ce-core/src` path was declared explicitly.

Command:

```powershell
$env:PYTHONPATH='C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\.tmp\ce-cosmo-dso-20260825\src;C:\dev\ce\ce-core\src'
& 'C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\.codex\hooks\python.cmd' pytest `
  tests\test_cosmology_registry.py `
  tests\test_cosmology_ratio_audit.py `
  tests\test_ce_residual_forward_model.py `
  tests\test_cosmology_closure_gate.py `
  tests\test_recombination_drag_adapter.py -q
```

Result: `58 passed in 0.99s`, exit code `0`.

After the exact reviewed manifest was applied, the same focused suite was run
again in the canonical repository with
`PYTHONPATH=C:\dev\ce\ce-cosmo\src;C:\dev\ce\ce-core\src`. Result:
`58 passed in 0.90s`, exit code `0`. The canonical worktree remained at base
HEAD `f78accbdd075454437e57ff39b6b6b0154088c10` with exactly the intended
sixteen modified or untracked paths; `git diff --check` passed and all sixteen
canonical files matched the stable staging hashes.

The first harness attempt omitted the split sibling `ce-core/src` dependency
and stopped during collection with five `ModuleNotFoundError: clarus_core`
errors. This was an unavailable dependency path, not a test failure. No package
was installed and no security policy was bypassed; the corrected command above
used the same interpreter and harness with the exact existing sibling source.

The focused tests cover zero-argument constructor rejection, explicit density
provenance, historical opt-in quarantine, CLI missing/partial/mixed failures,
registry status, ratio-audit status, raw DESI integrity/order/covariance,
historical full-covariance rejection, scale-fit nonprediction wording, closure
gate callsites and recombination adapter callsites.

## 2. Raw-asset reproduction

Independent filesystem hashing reproduced:

| Asset | Bytes | SHA-256 | Match |
|---|---:|---|---|
| mean | 472 | `9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585` | yes |
| covariance | 2547 | `252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509` | yes |

Both values match the pinned manifest and the audited upstream Cobaya
`bao_data` v2.6 assets.

## 3. Static quarantine check

`rg` was run across `src`, `tests` and `README.md` for:

```text
DESI_DR2_ALL_(DATA|COVARIANCE)
ACTIVE_RATIO|STRUCT_RATIO|BACKGROUND_RATIO|BOOTSTRAP_CONTRACTION
RUNTIME_COMPATIBILITY_DEFAULT
^def CEForwardParams
```

Result: no matches. Exit code `1` is ripgrep's expected no-match status.

This confirms removal of the embedded DESI bypass, unsafe public default
aliases and test shadow helpers. The explicitly named
`LEGACY_ROUNDED_RUNTIME_V1` and `historical_runtime_boundary_params()` remain as
the intended quarantined witness and opt-in negative control.

## 4. Independent post-implementation audit

The status auditor inspected the stable sixteen-path snapshot against the
approved manifest. The first post-audit returned three P1 findings: embedded
DESI duplicates, historical test-helper shadowing and public runtime-default
aliases. After their removal and the added failure-mode tests, the second audit
returned `POST-AUDIT: PASS` with no remaining P0 or P1 issue.

The audit also confirmed that the named loader is the sole DESI path, density
provenance never says `ce_prediction`, $q$ and the historical ratios are not
promoted to $\Omega$, and canonical prose does not claim a dark-sector
ontology proof.

A final read-only audit of the applied canonical snapshot also returned
`PASS` with no P0/P1 issue. Its only non-blocking P2 note is that the production
loader validates assets against the manifest, while the manifest's expected
hashes are independently pinned in tests rather than in signed provenance.

## 5. Validation boundary

The successful tests and audit establish source/provenance integrity, explicit
configuration, deterministic historical-diagnostic reproduction and document
status alignment. They do not validate the residual physical-map axiom,
identify real dark matter or dark energy, close the energy/stress junction, or
turn the fitted DESI scale into a prediction.
