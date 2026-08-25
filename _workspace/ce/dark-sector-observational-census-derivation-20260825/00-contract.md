# Dark-sector observational census and first-principles CE derivation

Status: COMPLETE

PREDECESSOR: `_workspace/ce/_archive/nonselected-quantum-dark-sector-20260825`

## 1. Question, interpretation, and claim ceiling

The user's target hypothesis is fixed in its strongest auditable form:

> Quantum neighbours can conditionally activate one another in a bootstrap
> network.  The alternatives not selected in the visible outcome are retained
> by a new physical map, and the matter-like and vacuum-like regimes of the
> resulting common residual sector are read cosmologically as dark matter and
> dark energy.

This run reconstructs that chain from the beginning, inventories the current
published values relevant to every observable the chain presently claims or
needs, and decides exactly which equalities are proved, conditional, adopted,
empirical, excluded, or still non-identifiable.  “Prove with observations”
means a preregistered empirical compatibility or exclusion test under a named
model and data combination.  It never means that agreement of central values
is a mathematical proof or that a fitted parameter is a no-input prediction.

The maximal authorized positive result is therefore:

1. exact mathematics for a declared quantum-neighbour process and its stated
   limits;
2. an explicit, dimensionally consistent and conserved residual-sector EFT;
3. conditional dark-matter-like and dark-energy-like limits of that EFT;
4. a source-frozen observational census and honest likelihood/posterior tests;
5. a CE prediction only where every required physical scale, initial datum,
   nuisance choice and bridge was fixed independently of the tested data.

If the abundance bridge is absent, a rigorous non-identifiability or no-go
result completes that claim audit; the run must not invent a bridge to satisfy
the requested conclusion.

## 2. Frozen domain and repository snapshot

The run is cosmology plus quantum mechanics only.  Brain, biology, AGI runtime,
agent-guard and their documents, constants, tests and run artifacts are out of
scope.  In particular, the tuple `(0.0487, 0.2623, 0.6891)` owned by the AGI
runtime is quarantined: `ce-meta/interfaces.md` records its provenance as
`UNRESOLVED` and forbids a cosmological lineage.  It may be inspected only as
a historical counterexample; it cannot be observational evidence or a CE
prediction.

Frozen code/document snapshots:

- monorepo working snapshot at `ecf05d16efa100c6b22607905949de128cccc107`,
  with the user's existing split-related deletions preserved and excluded;
- standalone `C:/dev/ce/ce-cosmo` at
  `f78accbdd075454437e57ff39b6b6b0154088c10`;
- standalone `C:/dev/ce/ce-runs` at
  `a51b08773322510adb03d6c1c56639cd86745860`;
- `C:/dev/ce/ce-meta/interfaces.md`, revision dated 2026-08-23;
- the active canonical dark-sector narrative beginning at
  `docs/5_유도/00_선택과_접힘.md`, read as
  **끼임 → 접힘 → 암흑 표현**;
- the cosmology ledgers and forward-model contract named by the source and
  mathematics lanes.

Mandatory predecessors, whose P0/P1 conclusions cannot be silently reopened:

- `nonselected-quantum-dark-sector-20260825`;
- `quantum-neighbor-bootstrap-dark-sector-20260825`;
- `cosmology-quantum-audit-20260824`;
- `cosmology-density-bridge-derivation-20260815`;
- `cosmology-full-closure-unification-20260815`;
- `cosmology-theory-repository-audit-20260815`;
- `pstar-br8-adjudication-20260823`.

All external observational information is frozen at access date 2026-08-25.
Only primary papers, official collaboration releases, official likelihoods or
official data repositories may support numerical claims.

## 3. Frozen definitions and the derivation chain

### 3.1 Quantum selection

A selected outcome is the recorded outcome of an explicitly declared quantum
instrument.  Nonselected alternatives are the complementary coarse-grained
history class before any physical identification.  They are not automatically
particles, energy, an Everett world, or a branch-local gravitational source.

### 3.2 Neighbour bootstrap

The minimum quantum-neighbour candidate is the declared open-system jump

$$
L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\,\sigma_i^+n_j,
$$

supplemented by an explicit Hamiltonian/bath/drive/decay and energy-current
ledger.  In the occupation-diagonal or justified decohered sector this may be
read as “active neighbour (j) gates activation of (i).”  The run must keep
separate:

- exact finite-network Lindblad or classical population dynamics;
- the finite absorbing process with positive decay;
- any independently declared infinite branching limit in which a spectral
  radius criterion is meaningful.

An SCC, a supercritical mean, or a fixed-point survival probability is not by
itself proof of perpetual activity in a finite system, and a neighbour is not
an energy source unless the current ledger says so.

### 3.3 Nonselected-history physical map

Let the nonselected history subprobability be
`nu_ns,beta`, a dimensionless measure on `Gamma_ns`.  The candidate map is

$$
\phi(x)=M_*\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\,\nu_{{\rm ns},\beta}(d\gamma).
\tag{C1}
$$

Here (M_*) has mass dimension one and the candidate kernel is dimensionless.
Equation (C1) is `[공리: 물리 사상]` unless a microscopic instrument,
history measure, local-covariant kernel, normalization and source rule derive
it.  Probability normalization alone is not energy normalization.

### 3.4 Minimal residual EFT

The candidate common residual sector is

$$
S_{\rm res}=\int d^4x\sqrt{-g}\left[
-\frac12 g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi
-\frac12m^2\phi^2-V_\Lambda\right],
\tag{C2}
$$

with a declared transition/matching prescription and no visible-sector energy
exchange after establishment in the minimal branch.  The oscillating quadratic
piece is the dark-matter-like candidate; the constant offset is the
dark-energy-like candidate.  The absolute present densities require, at
minimum, (M_*,m), initial data, (V_\Lambda), (M_{\rm Pl}), expansion
history and transfer/matching information.

### 3.5 Cosmological forward map

Any observational test must proceed through a declared Einstein--Boltzmann or
explicitly limited background model:

$$
(S_{\rm res},T_{\mu\nu},\text{initial data},\text{other species})
\longrightarrow H(z),\,D(z),\,P(k,z)
\longrightarrow \text{CMB/BAO/SN/lensing/clustering observables}.
\tag{C3}
$$

Substituting a published posterior mean for a missing arrow is an external
input, not a derivation.

## 4. Closed observational census

“Existing values” is operationally closed as follows.  The source lane must
record every value already used or claimed by the CE cosmology corpus and, for
the current comparison, every published parameter below from the selected
authoritative probe families for which a primary result exists by the cutoff:

| Family | Required parameters or limits |
|---|---|
| Composition | `omega_b = Omega_b h^2`, `omega_c = Omega_c h^2`, `Omega_m`, `Omega_DE` or the flat-derived `1-Omega_m`, and `Omega_k` where varied |
| Expansion/calibration | `H0`, `r_d` or the actually constrained `H0*r_d`, with calibration provenance |
| Dark-energy dynamics | constant `w`, and CPL `w0`, `wa`, including covariance/correlation when published |
| Structure | `sigma8`, `S8`, and the relevant residual-scalar fraction/mass or sound-speed limits |
| CE-internal claims | every numerical tuple, scorecard row, BAO chi-square, tolerance and data version present in the frozen CE corpus |

The authoritative probe census is:

1. final Planck 2018 primary CMB parameter results;
2. current DESI DR2 BAO and DESI DR2 dark-energy combinations, plus the
   official covariance/likelihood products used by the existing CE forward
   model;
3. the named supernova families actually combined with DESI (Pantheon+,
   Union3 and DES-SN five-year where applicable);
4. at least one current primary weak-lensing/large-scale-structure result that
   reports `S8`, with its model and probe combination fixed;
5. the current primary local-distance-ladder `H0` result only as an external
   calibration/tension comparator;
6. primary scalar/fuzzy-dark-matter structure constraints sufficient to test
   the mass/fraction regime claimed by (C2).

For every row the census must store: source DOI/arXiv/official URL, release and
access date, exact model, exact data combination, parameter convention,
central statistic, interval type/confidence, units, covariance availability,
whether a value is directly sampled or derived, and whether CE used it before
or after model construction.  Rows from different likelihoods may not be
assembled into a synthetic tuple.  Where no public covariance exists, only a
marginal comparison is authorized and must be labelled as such.

The census is complete when all five parameter families and all six probe
families above are either populated or carry an explicit primary-source
`NOT_AVAILABLE` reason.  It does not claim to enumerate every dark-matter
particle search, every cosmological paper, or every model outside the CE
candidate's observable scope.

## 5. Registered claims to prove, test, narrow, or reject

### DSO-1 — quantum-neighbour gate

Derive the population-level action of the declared jump operator and specify
the conditions under which neighbour activation is exact.  Prove or disprove
finite-system survival and identify the additional assumptions needed for a
branching spectral-radius theorem.  Close the energy-source ledger or retain
it as incomplete.

### DSO-2 — fixed-point status

Reproduce the declared Poisson/Lambert-W fixed point and its low-extinction
root, including uniqueness/domain, numerical residual, sensitivity and
dimensionless audit.  Determine which inputs are measured, selected, fitted or
derived.  The numbers `q_ext` and `1-q_ext` remain genealogical probabilities
unless DSO-4 derives an energy map.

### DSO-3 — standard-conditioning counterexample

Recheck whether standard conditional quantum mechanics adds nonselected
alternatives to the selected branch's local stress source.  A complete
counterexample deletes the automatic-cross-branch-gravity parent claim while
leaving C1 available as a new axiom.

### DSO-4 — probability/history to stress-energy bridge

Attempt the strongest local, covariant, dimensionally valid map from the
neighbour/history process to C1/C2.  It must state the conversion scale,
energy current, transition hypersurface, total-stress matching, locality,
instrument dependence and no-double-counting rule.  Prove identifiability or
give an explicit degeneracy/counterexample.

### DSO-5 — dark-matter-like theorem

From C2 derive the homogeneous oscillatory solution, averaged equation of
state, density scaling and perturbative sound/Jeans regime.  Separate exact
equalities from WKB averages.  Test whether CE predicts the scalar mass,
initial amplitude, present fraction or transfer function before seeing the
data.

### DSO-6 — dark-energy-like theorem

From C2 derive the constant-offset stress tensor and `w=-1`.  Test the
minimal prediction against the frozen `w`, `w0`, `wa` census.  Do not tune a
potential or interaction after inspecting those values and call the result a
prediction.

### DSO-7 — absolute abundance and split

Derive, if possible, `Omega_DM`, `Omega_DE` and their ratio from CE-only inputs
through C1--C3.  Audit separately the historical routes
`q_ext -> Omega_b`, `R = alpha_s D_eff`, the rounded runtime tuple, and flat
closure.  A continuous family of amplitudes/vacuum offsets yielding identical
fixed-point data is a proof of non-identifiability, not a failed calculation.

### DSO-8 — observational forward tests

Reproduce the current CE BAO full-covariance result and every other existing
CE dark-sector score using its frozen inputs.  Distinguish:

- genuine fixed prediction with no tested-data calibration;
- conditional output using external `H0`, `r_d`, `sigma8` or other inputs;
- fitted/nuisance-optimized result;
- marginal central-value comparison without covariance;
- unsupported or provenance-conflicted comparison.

No result may be promoted from one category to another.

### DSO-9 — strongest surviving central statement

Decide the exact status of “dark matter and dark energy are unselected quantum
paths activated through neighbouring quantum bootstraps.”  The anticipated
narrow candidate is: C1 is a new physical-map axiom; DSO-1 supplies at most a
conditional microscopic motif; and DSO-5/6 are conditional EFT theorems.  Any
stronger wording requires DSO-4, DSO-7 and DSO-8 to close.

## 6. Predeclared tests and decision rules

1. **Algebra/dimensions:** all exponential, logarithmic, fixed-point,
   probability and density-fraction arguments must be dimensionless; every
   probability-to-energy step must introduce and identify the required scale.
2. **Conservation:** the residual stress tensor must be on-shell conserved, and
   any transition or exchange must close only for the total stress tensor.
3. **Identifiability:** construct two physically distinct parameter choices
   with the same CE fixed-point outputs.  If both yield different present dark
   abundances, the abundance claim is non-identifiable.
4. **Finite bootstrap:** a finite graph with positive decay and no sustaining
   drive is tested for absorbing-vacuum behaviour.  SCC membership alone is
   never accepted as a survival proof.
5. **Observation:** use a published joint likelihood/covariance when available.
   Otherwise report standardized marginal residuals only, never a joint
   significance.  The existing DESI DR2 13-vector result is reproduced before
   any new model variation.
6. **Prediction hygiene:** a quantity entering C3 from a published posterior,
   a legacy tuple, or a post-data scale fit is counted as an external input.
7. **Model comparison:** the constant-offset branch is evaluated at its fixed
   `w=-1`; CPL values are observational comparators, not parameters that may be
   retrofitted into C2 during this run.
8. **Counterexample removal:** a complete counterexample removes its stronger
   parent from active canonical prose and ledgers; it is not retained beside
   the correction as an equal-status claim.

Observational compatibility is reported with the statistic provided by the
primary likelihood.  A conventional `p < 0.05` may be labelled rejection for
the frozen frequentist chi-square gate already implemented by CE, but no
universal discovery threshold is invented for posterior tables.  Tension
metrics based only on marginal Gaussians are explicitly diagnostic.

## 7. Independent lanes and ownership

- `ce_physics_sourcer` owns only `10-sources.md` and freezes the primary-source
  census.  It does not change equations, statuses, code or narrative.
- `ce_math_verifier` owns `11-math.md`, `12-routes.md` and run-local verifier
  artifacts.  It treats `10-sources.md` as empirical input and does not write
  canonical files.
- after both lanes stop, `ce_status_auditor` owns only `20-audit.md` on the
  stable snapshot.
- if and only if the audit gate passes a bounded implementation, a single
  `ce_ledger_writer` updates the approved cosmology ledger first.  The frozen
  ledger is then read-only input to one `ce_paper_writer` for the approved
  derivation/narrative files.  They never edit the same file or work
  concurrently.
- one implementation owner may add a source manifest, focused audit module and
  focused tests in `ce-cosmo` only after the claim ledger freezes.  No AGI or
  brain repository is touched.

## 8. Completion requirements

The run is complete only when:

1. the source census satisfies the closure rule in section 4;
2. DSO-1 through DSO-9 each have a derivation, counterexample, empirical test,
   or explicit missing premise;
3. the status audit supplies a claim ledger and a `Gate: PASS` or `FAIL`;
4. any approved canonical changes follow ledger-first ownership and pass
   focused source/tests plus the dimensionless and cosmology gates;
5. `31-validation.md` records exact commands, interpreter, repository/commit,
   pass/fail counts and limitations;
6. `40-final-report.md` answers in Korean, first stating what is proved now,
   then what observations say, then exactly what remains to prove;
7. the run root contains exactly the eight numbered stage files required by
   the CE research harness, with all auxiliary material under `artifacts/` or
   `revisions/`.

No full test suite, release, commit, push or modification of the user's split
deletions is authorized by this contract.
