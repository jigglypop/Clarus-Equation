# 20-audit — dark-sector observational census

Status: COMPLETE  
Gate: PASS  
Scope: narrowed removal/quarantine/correction implementation only; mandatory
post-implementation audit required.

Audit scope is the stable, cosmology-only snapshot of `00-contract.md`,
`10-sources.md`, `11-math.md`, and `12-routes.md`.  AGI, brain, and guard
trees are excluded.  No source, math, or route file was modified.  This gate
separates permission to implement an honest bounded diagnostic from proof that
unselected quantum paths are dark matter or dark energy; the latter remains
unproved.

## P0 findings

### P0-1 — superseded source table is still machine-readable evidence

`10-sources.md:133–156` labels §5 “superseded”, but still presents `0.0486`,
`Omega_m=0.315`, and the rounded runtime values in claim rows.  The prose at
`10-sources.md:42` also retains the superseded LiteBIRD wording before the
correction at `:44–46`.  A reader, parser, or implementation that consumes the
first matching row can therefore resurrect a false legacy baseline.  §6a
(`:160–176`) must be the sole evidentiary CE-internal section; the tombstone
must be removed from active source tables or made non-consumable, and the
LiteBIRD text must be corrected at its point of use.

### P0-2 — unresolved runtime tuple is on the production import path

In standalone `C:/dev/ce/ce-cosmo`, `src/ce_cosmo/gates/cosmology_ratio_audit.py:16,31–35,125`
imports `LEGACY_ROUNDED_RUNTIME_V1` and constructs `CE_RATIOS` from
`(0.0487, 0.2623, 0.6891)`.  `src/ce_cosmo/gates/ce_residual_forward_model.py:28,49–51`
then imports those ratios as forward-model defaults.  This violates the
contract's quarantine of the `UNRESOLVED` AGI-owned tuple, even if comments
call it compatibility-only.  The runtime tuple must be quarantined to an
explicit historical fixture/negative-control module and removed from all
ratio-audit and forward-default imports.

### P0-3 — provenance labels promote inputs to predictions

`C:/dev/ce/ce-cosmo/src/ce_cosmo/gates/ce_residual_forward_model.py:849–879`
labels `omega_b0`, `omega_dm0`, and `omega_lambda0` with role
`ce_prediction`, while their values are runtime/legacy ratios.  The matching
tests at `tests/test_ce_residual_forward_model.py:497–499` assert the false
status, and `tests/test_cosmology_registry.py:164–182` asserts the same legacy
ratio-audit behaviour.  These must become external/adopted or quarantined
roles, with tests changed to forbid `ce_prediction` unless a no-input bridge
is actually derived.

### P0-4 — rejected abundance parents remain in the split ledger

The standalone ledger `C:/dev/ce/ce-cosmo/docs/검증_원장/상수_우주론_원장.md:32,65–85,104–108`
retains `q_ext -> Omega_b`, `Omega_m=1/D`, and the runtime three-way split as
active historical formulae.  `src/ce_cosmo/registry.py:189,242,359–370` also
keeps direct-readout and ratio-to-density parents in the executable registry.
The math lane's DSO-3/DSO-7 P0 counterexamples require these parents to be
deleted from active claims (or explicitly tombstoned and unreachable), not
merely accompanied by a warning.  Flat closure may remain only as a conditional
identity after component completeness and flatness are adopted.

## P1 findings

* The DESI 13-vector and covariance are numerically consistent with Cobaya
  `bao_data` v2.6, commit `b7b8a36e9bccb063081f811f323cada21ab5fbdd`; the stated
  mean SHA is
  `9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585` and
  covariance SHA is
  `252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509`.
  However, raw assets are not pinned and validated in the repository.  The
  chi-square is therefore a conditional diagnostic, not a locally reproducible
  primary-likelihood result.  Add a source manifest, byte hashes, and a focused
  validation before implementation claims reproduction.
* `C:/dev/ce/ce-cosmo/docs/5_유도/00_선택과_접힘.md` is missing.  The split
  repository therefore lacks the required narrative spine
  **끼임 → 접힘 → 암흑 표현**.  Add it before narrative promotion; do not let
  ledger tables substitute for the first-reader derivation.
* DSO-1/2/4/5/6 remain conditional theorems or new axioms exactly as stated in
  `11-math.md`; their P1 missing premises (bath/drive energy, branching limit,
  covariant history-to-stress matching, scalar mass/amplitude/vacuum scale)
  must not be silently promoted by implementation.

## Claim-status ledger

| Claim | Audited status | Decision |
|---|---|---|
| DSO-1 neighbour activation | `[정리]` only for declared diagonal/decohered Lindblad/CTMC; finite positive-decay systems absorb | retain conditional; delete SCC/energy parent |
| DSO-2 Lambert-W fixed point | `[정리]` inside the separately declared Poisson branching limit | retain; `D_eff` origin is `[미완성]` |
| DSO-3 standard conditioning | `[정리]` counterexample: unrecorded branch is not added to local stress | delete automatic cross-branch-gravity parent |
| DSO-4 history → stress map | `[공리]` candidate C1; locality, covariance, current, matching and no-double-counting `[미완성]` | do not call derived |
| DSO-5 scalar dust limit | conditional `[정리]`/WKB `[산출]`; mass, fraction, transfer `[미완성]` | bounded EFT statement only |
| DSO-6 constant offset | conditional `[정리]` with exact `w=-1`; magnitude/origin `[미완성]` | comparator only |
| DSO-7 absolute abundance/split | `[미완성]` with P0 non-identifiability/no-go | delete all direct q/survival/ratio parents |
| DSO-8 forward tests | `[경험식]` conditional diagnostic; fitted scale is not prediction; raw covariance unpinned P1 | no promotion until manifest validation |
| DSO-9 strongest statement | `[공리]` + conditional EFT motif; “are” statement `[미완성]` | retain only narrowed wording |

## Gate decision and remediation boundary

The P0 findings are real defects in the current consumers, but each has an
unambiguous bounded correction: remove or tombstone the named parent rows,
quarantine the named tuple, change the named provenance roles, and preserve
§6a/current correction as the sole evidentiary subset.  Therefore the gate
passes **only for that removal/quarantine/correction implementation**.  It does
not pass a new physical-map derivation, an abundance fit, or any promotion of
the DSO-9 headline.  The implementation owner must not copy §5 into canonical
code or prose; after the changes, a post-implementation audit of the same
claims is mandatory.

If the repairs below are completed without adding a new physical claim, the
approved manifest is ledger-first, narrative-second, implementation-third:

1. **Ledger first:**
   `C:/dev/ce/ce-cosmo/docs/검증_원장/상수_우주론_원장.md` and, only where
   needed for the same status correction,
   `docs/검증_원장/참조_우주론GR_정리_증명.md`.
   Remove/tombstone the four rejected parents, mark C1 as `[공리]`, DSO-5/6
   as conditional `[정리]`, DSO-7 as `[미완성]`, and every density value as
   external/adopted or quarantined rather than `ce_prediction`.
2. **Narrative second:**
   `C:/dev/ce/ce-cosmo/docs/5_유도/00_선택과_접힘.md` (new central spine) and
   `docs/5_유도/04_Dark_Energy_Derivation.md`, written from the frozen ledger.
   The narrative must state **끼임 → 접힘 → 암흑 표현**, and must not copy the
   §5 tombstone or assert observational agreement as proof.
3. **Implementation third:**
   `src/ce_cosmo/registry.py`,
   `src/ce_cosmo/gates/cosmology_ratio_audit.py`,
   `src/ce_cosmo/gates/ce_residual_forward_model.py`,
   `tests/test_cosmology_registry.py`,
   `tests/test_cosmology_ratio_audit.py`, and
   `tests/test_ce_residual_forward_model.py`.
   Quarantine the runtime tuple, remove all automatic q/survival/ratio→Omega
   consumers, change provenance roles, and add the DESI manifest/hash gate.

No AGI/brain path, source snapshot, Git state, or unrelated split deletion may
be changed.  A post-implementation status audit of the same claim IDs is
mandatory before the narrowed implementation is considered complete; the
PASS here is not a theory-proof PASS.

### Manifest expansion required by zero-argument default removal

Status-auditor revision 2/2 found that removing the implicit runtime density
boundary has necessary live callsites outside the first manifest. The approved
scope expands only to these dependency corrections; they add no scientific
claim:

- `src/ce_cosmo/gates/ce_residual_forward_model.py` must remove `CE_RATIOS`
  density defaults, require explicit `omega_b0`, `omega_dm0` and
  `omega_lambda0`, and make the CLI require those values or an explicitly
  named, quarantined `--historical-runtime-boundary` opt-in that is never the
  default.
- `src/ce_cosmo/gates/cosmology_closure_gate.py` must replace its zero-argument
  construction with an explicitly named historical/adopted fixture.
- `tests/test_cosmology_closure_gate.py` must reject `ce_prediction` for the
  density boundary.
- `tests/test_recombination_drag_adapter.py` must pass an explicit declared
  density fixture to every `CEForwardParams` construction.
- `README.md` must remove any wording that presents `CE_RATIOS` or the
  AGI-owned rounded tuple as a normal cosmology boundary while retaining its
  compatibility/quarantine record.
- `benchmarks/cosmology/desi_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt` and
  `desi_gaussian_bao_ALL_GCcomb_cov.txt` must preserve the upstream raw bytes.
- `benchmarks/cosmology/desi_dr2/manifest.json` must record the two URLs,
  Cobaya `bao_data` v2.6 commit
  `b7b8a36e9bccb063081f811f323cada21ab5fbdd`, byte hashes
  `9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585` and
  `252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509`,
  observable order, dimensionless distance-ratio convention and validation
  command.

The original registry, ratio-audit and forward-model tests remain approved.
No H0 or xi legacy script is required for this bounded correction; those files
remain out of scope absent a later live-consumer finding.

## Final referee verdict

The snapshot proves only the declared conditional neighbour/branching results,
the standard-QM counterexample, the scalar WKB dust limit, and the constant
offset `w=-1` theorem.  It does not prove that dark matter or dark energy are
unselected quantum paths, does not identify their absolute abundances, and
does not establish indefinite neighbour self-execution.  The implementation
Gate is PASS only for repairing P0-1 through P0-4 within the approved manifest;
the resulting snapshot must pass the mandatory post-implementation audit,
including the listed P1 reproducibility/narrative checks, before any broader
scientific status is reported.
