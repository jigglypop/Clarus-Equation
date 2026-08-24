# Audit gate: nonselected quantum paths as the dark sector

Status: COMPLETE

## Snapshot and scope

This is a read-only audit of the stable snapshot consisting of
`00-contract.md`, `10-sources.md`, `11-math.md`, and `12-routes.md` in this
run directory. No canonical document, source file, test, or Git state was
changed by this audit. The implementation scope is deliberately narrowed to
the user's central idea:

> Nonselected quantum histories are retained by an explicit physical map;
> the matter-like and vacuum-like regimes of the resulting common residual
> sector are read out as dark matter and dark energy.

The phrase does **not** mean that ordinary discarded alternatives, a path
integral sum, or a decohered branch automatically gravitates in the selected
branch. M5 is a complete counterexample to that stronger parent claim.

## Normalized claim ledger

| ID | Normalized claim | Status | Canonical action |
|---|---|---|---|
| QD-M1 | C1/C2 dimensions and on-shell residual stress conservation hold conditionally, with locality, covariance, regularity, and transition matching assumptions. | `[정리·조건부]` | Retain; attach assumptions. |
| QD-M2 | A quadratic massive scalar is matter-like only in the stated adiabatic, mode and perturbative regime; a constant offset is vacuum-like inside C2. | `[정리·조건부]` | Retain; do not call it an unconditional CDM prediction. |
| QD-M3 | Standard conditional quantum mechanics does not add the nonselected outcome to the selected branch's local stress expectation. | `[반례·P0]` | Delete the stronger parent consequence wherever active. |
| QD-M4 | A common dark-sector origin is the physical-map axiom C1 plus C2, not a consequence of ordinary measurement theory. | `[공리: 물리 사상]` | Promote as the central surviving claim. |
| QD-M5 | The map must specify instrument/history/kernel/scale, total-stress conservation, locality/covariance, and no double counting with visible/environment sectors. | `[미완성]` | Keep as closure obligations. |
| QD-M6 | Absolute \(\Omega_{\rm DM}\), \(\Omega_\Lambda\), and the DM/DE partition are not fixed by the Poisson root or C1 without independent physical inputs and a frozen forward model. | `[미완성·비식별]` | Retain; block prediction language. |
| QD-M7 | A covariant residual EFT can be tested by R1--R4, including perturbations, transition matching, and observational falsifiers. | `[프로그램·미완성]` | Retain as route, not result. |

The old M1--M6 labels in `11-math.md` are internally coherent, but the
normalized QD IDs above are the IDs to use in the next ledger edit. In
particular, old M5 must not be copied as a positive claim: it is the P0
counterexample, while old M6 is the surviving physical-map statement.

## Closure-gate treatment of M5

The complete two-outcome conditioning control in `11-math.md`, section
“M5: fatal counterexample to automatic cross-branch gravity”, removes the
stronger parent proposition from active canonical prose. The following forms
are prohibited as theorem, derivation, or ordinary consequence:

* “nonselected/unselected quantum paths automatically gravitate in the
  selected branch”;
* “the folded probability is already a gravitational energy source”; and
* “decoherence or the path-integral sum supplies the dark-sector stress tensor.”

The permitted replacement is exactly QD-M4: an explicitly adopted
local-covariant physical map from nonselected history data to a new residual
sector, followed by the conditional EFT theorems QD-M1/QD-M2. The map is a
new physical assumption until a microscopic instrument and gravity/source
rule derive it.

## Existing active-prose violations

At least one P0 parent wording remains active and must be revised before
canonical closure:

1. `docs/5_유도/00_선택과_접힘.md`, opening three-step narrative (the
   “암흑 표현” step): it states that the energy of the folded component
   appears as dark matter and dark energy. This is stronger than QD-M4
   unless it explicitly says “under the C1 physical-map axiom” and
   “conditional C2 regimes”.
2. The same file's section `6.3`/the `1-q_ext` readout paragraph (the
   paragraph containing “95.14%” and the DM/DE split) presents the split as
   a consequence of the fixed-point/readout chain. It must be marked as a
   model/readout axiom or `[미완성]`, with QD-M6's non-identifiability note.
3. The same file's closure/status table (the row mapping
   `1-q_ext` to `Omega_DM, Omega_Lambda`) must not label the mapping as a
   theorem or derivation. It may remain only as an explicitly tagged physical
   readout ansatz pending the covariant map and stress/abundance closure.

The surrounding canonical material already contains useful caveats (for
example `docs/경로적분.md`, section 3.3, and
`docs/2_경로적분과_응용/10_공리_정당화.md`, the residual/readout status
tables). Those passages are not blockers, but they must remain consistent
with the stronger correction above. `docs/CE_통합_논문.md` also describes the
idea as a motivation and notes missing stress/bridge closure; it should be
checked during narrative editing, but this audit found no need to broaden the
approved ledger scope to that whole paper.

## Approved edit order and exact scope

Ledger-first edits are approved for only these files and sections:

1. `docs/검증_원장/상수_우주론_원장.md`: the entries for the
   `1-q_ext -> Omega_DM, Omega_Lambda` readout and any absolute-abundance or
   Poisson-root prediction status. Normalize them to QD-M6.
2. `docs/검증_원장/참조_양자_보존_원장.md` and/or
   `docs/검증_원장/참조_양자_정리_증명.md`: the smallest existing quantum
   bridge entries for nonselected conditioning, pushforward, and the
   physical-map status. Add QD-M3/QD-M4/QD-M5 only if those entries are
   already the designated home; do not duplicate a new broad ledger.

After those ledger entries are frozen, narrative edits are approved for:

1. `docs/5_유도/00_선택과_접힘.md`: opening “암흑 표현” paragraph, the
   `1-q_ext`/95.14% readout paragraph, and its status table row. Preserve the
   three-stage story, but qualify the DM/DE interpretation by QD-M4 and mark
   QD-M6 as unresolved for absolute abundance/partition.
2. Only the smallest linked cosmology narrative paragraph that repeats the
   same parent implication; do not rewrite unrelated derivations or the full
   integrated paper in this scope.

No code or test change is required by this audit. The focused EFT verifier
remains evidence for QD-M1/QD-M2 only and cannot promote QD-M3--QD-M6.

## Stable ledger implementation set

The ledger writer must implement the following stable IDs and statuses before
the narrative writer edits the canonical prose:

| ID | Required ledger status | Required meaning |
|---|---|---|
| QD-M1 | `[정리·조건부]` | C1/C2 dimensionality and on-shell stress conservation, with locality, covariance, regularity and transition-matching assumptions. |
| QD-M2 | `[정리·조건부]` | Quadratic oscillatory matter-like and constant-offset vacuum-like limits, only in their stated validity regimes. |
| QD-M3 | `[반례·P0]` | Standard conditional QM does not make nonselected paths gravitate in the selected branch. |
| QD-M4 | `[공리: 물리 사상]` | A declared physical map assigns the retained nonselected-history data to a common residual sector whose conditional EFT regimes may be read as DM-like and DE-like. |
| QD-M5 | `[미완성]` | Microscopic instrument/kernel/scale, locality-covariance, total conservation, transition matching, and visible-sector no-double-counting remain to be closed. |
| QD-M6 | `[미완성·비식별]` | Absolute abundances and the DM/DE partition are not fixed by C1 or the Poisson root without independent inputs and a frozen forward model. |
| QD-M7 | `[프로그램·미완성]` | R1--R4 provide the required microscopic, EFT, perturbation and abundance closure routes; they are not completed results. |

The ledger writer must delete or replace any active entry that promotes the
M5 parent wording to `[공리]`, `[정리]`, `[산출]`, or `[예측]`. QD-M4 is the
only permitted replacement for the user's central interpretation, and QD-M1
through QD-M2 may be cited only as conditional theorems. QD-M6 must remain
incomplete; the `1-q_ext` value may remain a composition/readout diagnostic,
not an abundance prediction.

## Gate decision

Gate: PASS
Scope: narrowed implementation only

The gate judges whether the exact approved implementation can proceed, not
whether all canonical prose has already been corrected. The stable lanes
provide a coherent surviving claim, conditional E1--E3/C2 theorems, a
complete M5 counterexample, and a bounded ledger-first/narrative edit
manifest. No additional P0 finding blocks that narrowed implementation.

M5 remains the mandatory-removal reason: the stronger parent wording must be
deleted or replaced before the canonical edit is considered complete. The
required implementation is therefore (i) remove the automatic
cross-branch-gravity consequence, (ii) register QD-M4 as an explicit physical
map axiom, (iii) register QD-M1/QD-M2 as conditional theorems, and (iv) keep
QD-M5--QD-M7 incomplete where their closure inputs are absent. Reopening
automatic cross-branch gravity would require a new microscopic,
empirically-testable theory passing R1--R3 and is outside this PASS.

## Post-implementation audit of the stable canonical diff

Audit target is limited to the three completed files named in the handoff:

- `docs/검증_원장/상수_우주론_원장.md`
- `docs/검증_원장/참조_양자_보존_원장.md`
- `docs/5_유도/00_선택과_접힘.md`

### Claim tags and ledger/narrative consistency

The canonical status assignments are consistent with the approved set:

| Claim | Canonical location | Status found | Audit result |
|---|---|---|---|
| QD-M1 | `상수_우주론_원장.md` §3.5 | `[정리]` with explicit `(조건부)` qualifier | Pass: conditional theorem, not an unconditional derivation. |
| QD-M2 | `상수_우주론_원장.md` §3.5 | `[정리]` with explicit `(조건부)` qualifier | Pass: adiabatic/mode conditions and non-CDM caveat are present. |
| QD-M3 | `참조_양자_보존_원장.md` §5.1 | `[정리]` `(P0 반례)` | Pass: correctly records the counterexample and rejects the parent claim. |
| QD-M4 | `참조_양자_보존_원장.md` §5.1; narrative §0.3 | `[공리: 물리 사상]` | Pass: common residual origin is explicitly adopted, not derived. |
| QD-M5 | `참조_양자_보존_원장.md` §5.1 | `[미완성]` | Pass: microscopic map, conservation and double-counting remain open. |
| QD-M6 | `상수_우주론_원장.md` §3.5; narrative §0.7 | `[미완성]` `(비식별)` | Pass: abundance and DM/DE partition remain unfixed. |
| QD-M7 | `상수_우주론_원장.md` §3.5 | `[미완성]` `(프로그램)` | Pass: R1--R4 are correctly presented as closure routes. |

The narrative summary table (`00_선택과_접힘.md` §0.8) reproduces the same
status classes: history-to-field is an axiom, EFT and regime statements are
conditional theorems, and the abundance readout is incomplete. The absence of
literal QD IDs in that summary table is not a status inconsistency because
the prose links the authoritative ledgers and preserves the same tags.

### Symbols and action definitions

The symbols needed by the new bridge are defined before use: the instrument
and $\rho_0$ appear in narrative §0.2; $\Gamma_{\rm ns}$,
$\nu_{{\rm ns},\beta}$, $\widehat K$, $M_*$ and $\phi$ are defined with the
pushforward in §0.3; $m$, $V_\Lambda$, signature and natural units are fixed
before the residual action in §0.4. The same action and dimensions are
recorded in the cosmology ledger §3.5. The stress tensor, scalar equation,
FLRW density/pressure, adiabatic conditions, sound speed, and Jeans-scale
conditions are all stated before their corresponding status claims. No
undefined symbol was found that creates a P0/P1 implementation defect.

### M5 closure-gate check

The stronger parent is removed from active assertion. The quantum ledger
explicitly says that an automatic cross-branch source does not follow, and the
narrative states that “비선택 경로가 선택된 branch에서 저절로 중력을 낸다” is
not a standard measurement-theory result. The remaining positive statement is
the QD-M4 physical-map axiom. This is the required M5 closure treatment.

### Preservation and first-reader gate

The electroweak mixing readout ($\delta=s_W^2(1-s_W^2)$), the Hodge dimension
condition $d=d(d-1)/2$, the additive fold trace, and the Poisson generating
function/fixed-point/Lambert-$W$ chain remain present. They are explicitly
separated from the quantum-to-residual bridge and are not promoted to energy
or abundance derivations. The opening of `00_선택과_접힘.md` now gives the
three-step story (끼임 → 접힘 → 암흑 표현), defines the reader's required
background, explains the status tags, and points to the two authoritative
ledgers before technical detail. This satisfies the first-reader narrative
gate.

### Post-implementation verdict

No remaining P0 or P1 implementation defect was found in the stable diff.
The unresolved QD-M5--QD-M7 items are intentional `[미완성]` closure claims,
not defects in this implementation. The existing gate therefore remains:

Gate: PASS
Scope: narrowed implementation only
