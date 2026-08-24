# Post-audit revision 1: quantum-neighbor bootstrap

Status: COMPLETE

## Finding first

P1-01 is closed. The revised `docs/5_유도/00_선택과_접힘.md` now contains
§0.9 “정본 개정 대응표와 보존 지도”. The table has 25 entries, matching the
25 prior HEAD sections, and maps each prior section to a current derivation,
ledger, reader-guide path, or explicit tombstone. It preserves the reader
route through §0.1--§0.10 and identifies where the former derivations now live.

The former §0.6.2 abundance mapping is not silently omitted: its false
`q_ext\to\Omega_b`, `R=\alpha_sD_eff`, and 95.14% abundance interpretation are
explicitly tombstoned as `C-B-LEGACY-01`, while current §0.7 states the
non-identifiability result. This does not reintroduce the rejected claim.

## Closure checks

- The 25-row correspondence table is bounded between §0.9 and §0.10 and was
  counted directly from the latest file.
- The revised document retains QNB §0.2.3, exact CTMC and first-moment
  hierarchy, finite absorption/energy no-go, SCC limitation, conditional
  Poisson/Perron bridge, instrument boundary, residual-map axiom, and
  DM-like/DE-like EFT conditions.
- The current equation flow remains `(1)--(17)` with local `Q1--Q3` labels;
  the repaired `x_0\in[0,1/D]` iteration domain and residual measure remain.
- No P0 wording has returned: finite perpetual survival, SCC⇒survival,
  automatic Poisson offspring, neighbor-supplied energy, automatic
  cross-branch gravity, and microscopic `D_eff` derivation remain excluded.
- `git diff --check` remains clean. No Git state was changed by this audit.

## Residual status

No P1 residual issue remains for the approved QNB implementation scope.
Notation in the reader guide still offers both row and incoming forms of `A`,
but it explicitly states `A^{incoming}_{i\leftarrow j}=A^{row}_{ji}`; this is
at most a P2 readability preference, not a correctness blocker.

## Gate

Gate: PASS

PASS is limited to the narrowed conditional result: declared local facilitated
transitions, exact diagonal-sector CTMC, and an explicitly declared/proven
branching limit with seed-reachable supercritical components. It does not claim
universal quantum ontology, finite eternal bootstrap, automatic dark-sector
gravity, observed abundance derivation, or microscopic closure of `D_eff`.

Referee readiness: internal conditional result; separate instrument,
locality, conservation, branching-error and abundance-closure work is still
needed for a stronger quantitative or arXiv-ready claim.
