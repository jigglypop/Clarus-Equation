# 30-implementation — dark-sector census corrections

Status: COMPLETE

Implementation was performed in the isolated staging tree
`C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\.tmp\ce-cosmo-dso-20260825`.
The external canonical repository `C:\dev\ce\ce-cosmo` remained unchanged
during implementation and pre-application validation. After the stable-snapshot
audits passed, the exact sixteen-path manifest below was copied into the clean
canonical worktree at base HEAD
`f78accbdd075454437e57ff39b6b6b0154088c10`. No staging, commit, fetch, pull or
push was performed. Post-copy hashes matched the reviewed staging files 16/16.

## 1. Changed-path manifest

The stable implementation contains exactly these sixteen approved paths:

1. `README.md`
2. `src/ce_cosmo/registry.py`
3. `src/ce_cosmo/gates/cosmology_ratio_audit.py`
4. `src/ce_cosmo/gates/ce_residual_forward_model.py`
5. `src/ce_cosmo/gates/cosmology_closure_gate.py`
6. `tests/test_cosmology_registry.py`
7. `tests/test_cosmology_ratio_audit.py`
8. `tests/test_ce_residual_forward_model.py`
9. `tests/test_cosmology_closure_gate.py`
10. `tests/test_recombination_drag_adapter.py`
11. `benchmarks/cosmology/desi_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt`
12. `benchmarks/cosmology/desi_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt`
13. `benchmarks/cosmology/desi_dr2/manifest.json`
14. `docs/검증_원장/상수_우주론_원장.md`
15. `docs/5_유도/00_선택과_접힘.md`
16. `docs/5_유도/04_Dark_Energy_Derivation.md`

No AGI, brain, guard, unrelated split file or Git state was changed.

## 2. Density and provenance correction

`CEForwardParams` now requires explicit `omega_b0`, `omega_dm0` and
`omega_lambda0`. A zero-argument call fails. Ordinary explicit inputs carry the
role `adopted_or_external_boundary` and never qualify as CE physical
predictions.

The rounded historical runtime tuple remains available only through the
explicitly named `historical_runtime_boundary_params()` negative-control
factory and CLI flag `--historical-runtime-boundary`. Partial density input or
mixing the historical flag with explicit density values fails closed.

The public aliases `ACTIVE_RATIO`, `STRUCT_RATIO`, `BACKGROUND_RATIO`,
`BOOTSTRAP_CONTRACTION` and `RUNTIME_COMPATIBILITY_DEFAULT` were removed. The
explicitly named `LEGACY_ROUNDED_RUNTIME_V1` record remains only as quarantined
historical provenance. Historical $q$/ratio-to-density registry entries and the
flat runtime boundary are marked excluded/historical and are not eligible for
active scientific selection.

`cosmology_ratio_audit.py` now exposes a historical runtime negative control,
not `CE_RATIOS` as a scientific score. Tests forbid `ce_prediction` on density
inputs and no longer shadow the production `CEForwardParams` class with a
helper that silently injected historical values.

## 3. DESI raw-data gate

The upstream raw compressed DESI DR2 files were pinned with their original
names and bytes. `manifest.json` records source URLs, Cobaya `bao_data` v2.6
commit `b7b8a36e9bccb063081f811f323cada21ab5fbdd`, observable order, units, byte
lengths and SHA-256 values.

The named DESI loader now checks, before use:

- exact byte length and SHA-256 of both raw assets;
- exactly thirteen mean-vector rows and the declared observable order;
- a finite $13\times13$ covariance matrix;
- covariance symmetry; and
- positive definiteness by Cholesky decomposition.

The old module-global embedded DESI vector and covariance were removed, so an
importable unverified duplicate cannot bypass the raw loader. Tests pin the
expected lengths and hashes directly rather than merely checking a mutable
manifest against itself.

## 4. Canonical documentation

The ledger removes active $q\to\Omega$ parents, records the residual map as an
axiom and keeps absolute abundance non-identifiability active. The two narrative
documents present the mechanism in the order

$$
\text{static external 0D boundary}
\to\text{one-way open channel}
\to\text{directed neighbour bootstrap}
\to\text{residual-map axiom}
\to\text{conditional DM/DE-like EFT}.
$$

They preserve the empirical DESI rejection and do not present observational
closeness, code success or the fixed-point probabilities as evidence for dark
matter/dark energy identity.

## 5. Implementation boundary

No microscopic residual-to-gravity map, energy/stress junction, absolute
abundance law, perturbation transfer model or new observation fit was added.
The implementation removes false defaults and makes existing diagnostics
reproducible; it does not complete the physical theory.
