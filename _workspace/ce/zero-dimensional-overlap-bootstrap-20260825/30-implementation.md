# Implementation: one-way zero-dimensional boundary integration

Status: COMPLETE

No production simulator or cosmology likelihood was added by this ZDO run. The
Gate authorized a narrow documentation correction plus preservation of the
revised deterministic certificate.

## 1. Canonical staging target

Integration was performed in the isolated cosmology staging tree

`C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\.tmp\ce-cosmo-dso-20260825`.

The external canonical repository `C:\dev\ce\ce-cosmo` was not modified.

## 2. Changed canonical documents

- `docs/검증_원장/상수_우주론_원장.md`
  - replaced the reciprocal common-bus-centred ZDO ledger with the audited
    `Z -> M` statuses;
  - recorded ZDO-1 through ZDO-5 as definitions/conditional results;
  - retained ZDO-6 as incomplete, ZDO-7 as a physical-map axiom and ZDO-8 as
    an abundance non-identifiability no-go;
  - added the reverse-only `M -> Z` causal no-go;
  - retained the old low-rank reciprocal bus only as a rejected comparison.
- `docs/5_유도/00_선택과_접힘.md`
  - rewrote the central narrative as static 0D boundary, one-way open channel,
    directed in-spacetime bootstrap, residual-map axiom and conditional EFT;
  - separated one-time preparation from continuous injection;
  - separated finite-DAG termination/absorption from infinite Poisson
    genealogy;
  - retained the numerical fixed point only as a genealogy probability.
- `docs/5_유도/04_Dark_Energy_Derivation.md`
  - aligned the scalar FLRW, scalar--tensor and growth derivations with the
    one-way boundary definition;
  - made the source-current/junction requirement explicit;
  - retained DM-like/DE-like behavior as conditional EFT readouts, not an
    identity or abundance proof.

## 3. Research artifacts

- `artifacts/verify_zero_dimensional_overlap.py`
  - replaced reciprocal-bus-centred checks with a one-dimensional-input
    preparation/instrument certificate, cascaded-GKSL directionality check,
    directed CTMC sample, finite-DAG adverse control, Poisson fixed point,
    abundance non-identifiability and dimension checks.
- `artifacts/dimensionless-audit.md`
  - audited the preparation channel, rate-normalized cascade operators,
    directed jump/branching core, residual EFT and bulk junction-current
    dimensions.

## 4. Deliberately unimplemented claims

No code was added that treats the external 0D boundary as observed, maps
nonselected histories to gravity without an explicit axiom, supplies free
excitation energy, identifies $q$ with a density fraction, or predicts
$\Omega_{\rm DM}$ and $\Omega_{\rm DE}$. Those boundaries are mathematical and
physical status decisions, not missing software features.
