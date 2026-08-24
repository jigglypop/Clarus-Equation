# Implementation record: quantum-neighbor bootstrap

Status: COMPLETE

## Canonical changes

Implementation followed the frozen ledger-first order.

1. `docs/검증_원장/참조_양자_보존_원장.md`
   - registered `QNB-E1`--`QNB-E7`;
   - added the facilitated jumps, exact diagonal-sector CTMC and correlation
     hierarchy;
   - recorded finite-vacuum absorption, seed/energy bookkeeping, the SCC
     limitation, and the instrument/locality/residual-map boundaries;
   - retained the predecessor's `QD-M3` counterexample and `QD-M4`--`QD-M5`
     physical-map/incomplete statuses.
2. `docs/3_상수/3_부트스트랩.md`
   - fixed the parent-row/child-column convention
     `A_{ji}=E[type-i children | type-j parent]`;
   - restricted `A_{ji}=kappa_{ij} tau`, independent Poisson offspring, and
     the Perron threshold to their declared fixed-window/fresh-target limit;
   - recorded exponential-lifetime, collision, exclusion, coherent, and
     collective-jump counterexamples;
   - retained `D_eff=d+delta` as a CE readout axiom rather than a microscopic
     output.
3. `docs/5_유도/00_선택과_접힘.md`
   - inserted the derivation from local facilitated Lindblad jumps through the
     exact CTMC, first-moment nonclosure, finite absorption, and the conditional
     branching reduction;
   - preserved the narrative order `끼임 -> 접힘 -> 암흑 표현` and the
     predecessor's explicit nonselected-outcome P0 boundary;
   - kept the residual map as a physical axiom and derived only the conditional
     DM-like oscillatory and DE-like constant-offset EFT limits;
   - repaired equation numbering, the verified `x_0 in [0,1/D]` iteration
     domain, and the residual-measure term;
   - added a 25-row revision map preserving every prior HEAD section, reader
     path, and canonical link while tombstoning the rejected direct-abundance
     chain as `C-B-LEGACY-01`.
4. `docs/코어_독자_가이드.md`
   - added a short entry route explaining the exact meaning and limits of
     “neighboring quanta execute one another.”

No production code was added. Existing
`quantum_jump_bridge.py`, `multispace_bootstrap.py`, and their focused tests
already cover the conditional algebraic bridge and multitype fixed point. The
new two-node exact certificate is run-local so that rejected P0 claims are not
encoded as production behavior.

## Revision record

The first post-audit returned `Gate: REVISE` for documentation governance: a
predecessor edit had replaced a large part of the earlier narrative. Revision
1 added the explicit 25/25 correspondence map instead of restoring the false
legacy abundance claim. The revised post-audit closed P1-01 with `Gate: PASS`.

