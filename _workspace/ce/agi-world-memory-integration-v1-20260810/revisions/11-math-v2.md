# CE AGI-world-memory V2 final mathematical audit

Status: COMPLETE

Gate: REVISE

Audit completion and execution authorization are different decisions. The V2 design is
salvageable and it resolves several V1 defects, so the design-level gate is REVISE rather
than BLOCKED. The registered implementation and every registered seed remain BLOCKED
until all P0 replacements below are incorporated, hash-locked, and independently
re-audited.

No seed, benchmark, calibration program, or implementation code was run in this audit.

## 1. Bottom line

V2 correctly freezes the benefit-oriented factorial signs, ratio-of-means convention,
positive-denominator rule, paired-CI critical values, strict win/tie rule, hard-zero
provenance gates, and most scalar metric denominators. It does not yet define one
executable mathematical object. The remaining problems are specification problems, not
empirical questions:

1. the stated 20-parameter core contains four intercepts that disappear from both the
   codec and rollout equations;
2. raw and standardized codec coordinates are mixed at the schema and dream boundaries;
3. the inherited dream join acts on entity-fingerprint columns that must not participate
   in schema recombination;
4. the learner-visible action/continuity graph is not constructed;
5. the unconstrained lesion has 240 eligible non-observed pairs but only 24 update
   slots, with no deterministic selection or storage semantics;
6. the common persistent-memory cap is not closed under all required arrays and the
   ownership/accounting boundary is not defined;
7. inherited G7-M APIs are described as directly reusable even where V2 changes scope,
   dimensionality, capacity, or table semantics.

These defects prevent A2, A3, A6, A7, and A9 from being executable as written. A4 is
also only conditionally closed because the learner return schema and evaluation-prefix
construction are incomplete.

## 2. A1--A10 / P0 resolution

| Item | V2 audit | Reason |
|---|---|---|
| A1 / P0-1, factorial estimand | PASS | C1 is the marginal LTM effect and both simple effects are separately reported; M11 versus M00 is only RR_joint. |
| A2 / P0-3, metrics | REVISE | Main denominators are recoverable, but recall-target coordinates and raw/standardized boundaries are ambiguous. Numeric denominator assertions must be frozen. |
| A3 / P0-2, planning | REVISE | The action set and cost are numeric, but the evaluation prefix and learner-validity graph are incomplete. |
| A4 / P0-4, leakage | REVISE | Evaluator-after-hash and poison gates are strong, but the learner return schema does not yet contain enough typed information to compute its declared validity rule. |
| A5 / P0-5, provenance | PASS | Attempts, successful writes, recalls, and provenance violations are separate integer hard-zero gates. Keep them exactly zero; no tolerance is permitted. |
| A6 / P0-6, thresholds | REVISE | Threshold selection is train-only, but it must be calibrated through the exact scoped recall wrapper and the exact residual-only dream join used at evaluation. |
| A7 / P0-7, common budget | REVISE | Several counts are correct, but proposal/check counts are conflated and the byte-cap ledger omits required arrays. |
| A8 / P0-8, comparisons | PASS | Benefit signs, ratios, zero-denominator failure, strict wins/ties, multiplicity, and relative 2 percent are now explicit and correct. |
| A9 / P1, controls | REVISE | Shuffled and zero-q controls are identifiable, but the unconstrained lesion is not an executable equal-budget treatment. |
| A10, source boundary | PASS | Literature is motivational only and does not enter a mathematical or biological gate. |

## 3. Remaining P0 replacements

The following text is intended as exact contract replacement language.

### P0-1: retain all 20 fitted core parameters

Problem: each target regression has five coefficients, but the codec and rollout omit
the fitted intercept. Four targets times five coefficients is 20; an intercept-free
model would instead have 16 parameters.

Replace every learned-core prediction, codec residual, and rollout transition by:

\[
\widehat f_r(x,a)
=\widehat c_r+\widehat d_r x_r
 \widehat b_r\tanh(x_{\operatorname{src}(r)})
 \widehat g_{r0}a_0+\widehat g_{r1}a_1,\qquad r=0,1,2,3.
\]

\[
\widehat x_{t+1,r}
=\widehat f_r(\widehat x_t,a_t)
 \widehat q_r+\widehat s_{t,r}.
\]

The codec residual must be computed against the same expression including
\(\widehat c_r\). Freeze:

\[
P_{\rm core}=4(1+1+1+2)=20,\qquad N_{\rm solve}=4,\qquad
N_{\rm core\ transitions}=40\cdot96\cdot12=46{,}080.
\]

All cells must hash the same ordered 20-vector
\((\widehat c,\widehat d,\widehat b,\widehat G)\).

Alternative, not recommended: delete the intercept column everywhere and change the
registered parameter count to 16. Mixing the two conventions is forbidden.

### P0-2: freeze the raw/standardized coordinate boundary

All state, action, transition, codec-residual, schema-rollout, and cost coordinates are
dimensionless. Standardization therefore does not repair a physical-unit problem; it
defines a different numerical coordinate system and must be explicit.

Insert:

1. \(T^{\rm raw}\in\mathbb R^{12\times8}\) is the codec produced from dimensionless
   raw trajectories. Only storage, addressing, thresholding, schema joins, and recall
   internals may use
   \[
   T^{\rm std}=(T^{\rm raw}-\mu_{\rm codec})/\sigma_{\rm codec}.
   \]
2. The inherited hard-recall wrapper accepts a raw cue, standardizes internally, and
   returns \(T_{\rm completed}^{\rm raw}\). No caller subtracts a standardized schema
   entry from this raw result.
3. On rejection or in a no-LTM cell, visible raw cue entries remain clamped and every
   hidden entry is filled with the raw codec mean \(\mu_{\rm codec}\), not standardized
   zero unless the conversion is explicitly performed.
4. Slow-schema and dream payloads are stored in standardized codec coordinates.
   Before a payload contributes to a state rollout,
   \[
   S_{\rm raw}=\mu_{\rm codec}+\sigma_{\rm codec}\odot S_{\rm std}.
   \]
5. The observed anchor used to estimate episode drift is
   \[
   A_{\rm raw}
   =\operatorname{inverse\_standardize}
     (\operatorname{lookup}_{\rm observed}(k))_{:,0:4},
   \]
   and
   \[
   \widehat q
   ={1\over12}\sum_{t=0}^{11}
     \left(T_{{\rm completed},t,0:4}^{\rm raw}
     -A_{t,0:4}^{\rm raw}\right).
   \]
6. State normalizers used by NRMSE and planning are named
   \((\mu_x,\sigma_x)\), not \(q\), and never enter the dynamics.
7. The recall reconstruction target is the raw codec of the selected ledger record,
   compared after both target and reconstruction are transformed by the same frozen
   codec standardizer. Hidden-coordinate count is asserted as
   \[
   24(96-24)=1{,}728.
   \]

### P0-3: remove identity fingerprints from dream/schema joining

Problem: codec columns 4:8 are independent entity fingerprints intended for episodic
addressing. The inherited dream proposer and join operate on all eight columns. A
prefix, connector, and suffix from different bindings therefore compare unrelated
fingerprints. This can reject otherwise valid missing transitions and makes the dream
factor partly an identity-matching factor.

Insert:

1. Episodic hard recall continues to address all eight codec columns.
2. Slow schema, dream proposal, join calibration, dream acceptance, and rollout use
   only the residual view \(T_{:,0:4}\).
3. For a proposed residual-only join,
   \[
   J_L=\sqrt{{1\over4}\sum_{r=0}^3
   (P_{11,r}-C_{0,r})^2},\qquad
   J_R=\sqrt{{1\over4}\sum_{r=0}^3
   (C_{11,r}-S_{0,r})^2}.
   \]
   Acceptance requires the registered strict/non-strict inequalities used by the
   calibration algorithm, applied to these exact four-dimensional values.
4. The implementation must hash-lock a V2 residual-only proposer/join adapter. It must
   not claim direct equivalence to an inherited eight-column function.
5. The join-threshold calibration population must be generated by this same
   residual-only adapter. No eight-column calibration value may select a
   four-column evaluation threshold.

A neutral-fingerprint eight-column snapshot is acceptable only if columns 4:8 are fixed
to standardized zero for every schema/dream item, are excluded from every join statistic,
and are never decoded into rollout state. Residual-only storage is simpler.

### P0-4: define the evaluation prefix and learner-validity graph

Replace “separately generated 12-step prefix” by:

For evaluation origin \((c,p,i)\), use the correct opaque action token and the numeric
action \(A_i\) for all 12 prefix transitions, the observed anchor key \((i,i)\), the
origin's frozen episode drift, a fresh initial state and 12 fresh innovations from named
evaluation-only streams, and the same transition law as wake generation. Record states
\(x_0,\ldots,x_{12}\); prediction begins at \(x_{12}\). Because 12 is one complete
phase cycle, the H20 future begins at phase zero. Evaluation innovations are independent
of training and wake innovations, are unavailable to the learner, and are shared across
the four cells for the same origin and candidate.

Replace the inferred-validity prose by:

From wake records only, construct an immutable action index mapping each observed opaque
action token to exactly one tuple
\[
(\text{context token},\text{component id},\text{numeric action},
 \text{suffix token}).
\]
Construction hard-fails on zero or multiple matches. Infer the requested component from
the prefix/suffix co-occurrence graph. A candidate step is learner-valid iff its opaque
token resolves uniquely, its context and component equal the requested context and
component, its numeric action equals the registered action vector, and the required
schema key resolves. Candidate generation returns exactly the 20 numeric actions and one
20-element Boolean inferred-valid vector. The evaluator hashes this return before
consulting generator truth.

Delete undeclared “continuity tokens” and “emitted tokens.” If they are retained, their
types, construction, and return fields must be added to the API and budget before train.
The present numeric-state-plus-Boolean API cannot be validated against undefined tokens.

### P0-5: freeze dream counts and the 240-to-24 lesion selection

The active constrained enumeration has:

\[
\begin{aligned}
N_{\rm ordered\ pair\ enumerations}&=2\cdot12\cdot12=288,\\
N_{\rm component/port\ checks}&=288,\\
N_{\rm same\ component\ pairs}&=2\cdot4\cdot3\cdot3=72,\\
N_{\rm observed\ keys}&=48,\\
N_{\rm valid\ missing\ keys}&=24,\\
N_{\rm join\ candidates}&=24,\\
N_{\rm scalar\ endpoint\ join\ values}&=48,\\
N_{\rm output\ slots}&=24,\qquad N_{\rm update\ slots}=24.
\end{aligned}
\]

The contract must not call all 288 enumerations “proposals.” Freeze each counter
separately. Inactive cells and attribution controls execute the same enumerations and
checks, compute the same number of join statistics, and pad all rejected output/update
slots without changing treatment-visible state.

For the unconstrained lesion, removal of component/port/join restrictions leaves
\(288-48=240\) non-observed ordered pairs. Insert this deterministic rule:

1. Enumerate the 288 ordered pairs in lexicographic order over the learner-visible tuple
   \((\text{context token},\text{prefix slot},\text{suffix slot})\), where token order is
   the frozen first-occurrence order in the wake ledger.
2. Reject the 48 observed keys.
3. The first 24 remaining pairs are the lesion's accepted splice audit set; the other
   216 are checked and rejected with reason capacity_padding.
4. Store the 24 accepted objects only in a preallocated lesion audit buffer of the same
   shape and dtype allocated in every compared cell. They never overwrite the 72-key
   valid slow-schema table and never become legal action-graph edges.
5. Define invalid_splice_rate with denominator exactly 24. Hard-fail unless exactly 24
   accepted lesion objects exist.

If the scientific intent is instead to let all 240 lesion pairs affect rollout, the
contract must allocate the same 288-key lesion-capable table in every compared cell and
define collision, lookup, and rollout semantics. That is a different, more expensive
registered treatment; it cannot be inferred after calibration.

### P0-6: close the common-budget ownership ledger

The following arithmetic is fixed and should be asserted:

\[
N_{\rm wake/seed}=96\cdot12=1{,}152,\quad
N_{\rm rollout}=24\cdot8=192,\quad
N_{\rm predicted\ transitions}=192\cdot20=3{,}840.
\]

Minimum persistent numeric payload under the currently described representation is:

| Object | Shape / dtype | Bytes |
|---|---:|---:|
| LTM traces | \(96\times12\times8\), float64 | 73,728 |
| schema payload | \(72\times12\times8\), float64 | 55,296 |
| all predictions | \(24\times8\times20\times4\), float64 | 122,880 |
| inferred-valid flags | \(24\times8\times20\), bool8 | 3,840 |
| per-origin drift | \(24\times4\), float64 | 768 |
| candidate costs | \(24\times8\), float64 | 1,536 |
| selected indices | \(24\), int64 | 192 |
| learned core | \(20\), float64 | 160 |
| Minimum subtotal |  | 258,400 |

The stated 262,144-byte cap leaves only 3,744 bytes before LTM occupancy/provenance,
scope indices, schema occupancy/provenance, normalizers, recall audit arrays, dream
buffers, lesion buffers, hashes, and padding. Thus the cap is not demonstrated and is
likely false.

Replace the cap paragraph by:

Before train, enumerate every owned persistent array by semantic name, shape, dtype,
byte count, and owning subsystem. Immutable generator inputs may be excluded only when
all cells receive the identical read-only object by reference. Views must be zero-copy;
copies count against the owner. Python/container overhead is either counted by a frozen
measurement method or excluded uniformly while all numeric/index/mask payloads are
counted. Every cell preallocates identical shapes, including inactive stores, schema,
dream, lesion, prediction, audit, and padding arrays. Equality means exact equality of
the registered call/update/counter vector and the ordered allocation-ledger hash, not
equality of incidental peak process memory.

Raise the persistent numeric-payload cap to 524,288 bytes, or choose a smaller
representation and prove an exact total below 262,144 bytes before train. Add at least
schema bytes, scope-index bytes, occupancy/provenance masks, residual-only dream buffers,
lesion audit buffers, and standardizer arrays to the ledger. Apply the identical ledger
to M00/M01/M10/M11 and every shuffled, zero-q, and lesion comparator.

### P0-7: freeze inherited API semantics and calibration populations

Insert:

1. The physical episodic store has capacity 96. A scoped \((c,p)\) recall has exactly
   12 candidates, not 96. Thresholds are selected by the exact scoped wrapper used at
   evaluation and are named scoped-bank-96 thresholds.
2. The eight scoped views are prebuilt zero-copy index views in frozen storage order.
   Cross-port or invalid-scope cues reject before any view or distance call.
3. The 72-slot V2 schema adapter is preallocated. It implements observed and synthetic
   occupancy/provenance masks explicitly; it does not rely on dynamic inherited
   dictionaries or claim type identity with the inherited table.
4. Threshold calibration populations are asserted as:
   \[
   N_{\rm core}=46{,}080\ {\rm transitions},\quad
   N_{\rm state}=40\cdot96\cdot13=49{,}920\ {\rm rows},
   \]
   \[
   N_{\rm codec}=40\cdot96=3{,}840\ {\rm trajectories},\quad
   N_{\rm recall,+}=N_{\rm recall,lure}=40\cdot24=960.
   \]
   Each recall item passes through the scoped wrapper.
5. After P0-3, join calibration uses
   \(40\cdot48\cdot2=3{,}840\) scalar endpoint values computed in exactly four residual
   dimensions by the V2 adapter.
6. Every threshold selector hard-fails on empty input, nonfinite input/output, a
   non-unique selection rule, or any population-count mismatch.

## 4. Verified metric denominators and cost bound

Freeze these numeric denominator assertions, not just symbolic set expressions:

| Quantity | Exact scalar denominator |
|---|---:|
| \(E_{\rm all}(20)\) | \(24\cdot6\cdot20\cdot4=11{,}520\) |
| \(E_{\rm all}(5)\) | \(24\cdot6\cdot5\cdot4=2{,}880\) |
| \(E_{\rm uv}(20)\) | \(24\cdot40\cdot4=3{,}840\) |
| \(E_{\rm uv}(5)\) | \(24\cdot10\cdot4=960\) |
| recall hidden coordinates | \(24\cdot72=1{,}728\) |
| valid-candidate transition rate | \(24\cdot6\cdot20=2{,}880\) |
| selected-candidate metrics | 24 |
| false-positive and false-lure origin metrics | 24 each |
| constrained dream coverage | 24 |
| lesion invalid-splice rate after P0-5 | 24 |

Every empty, zero, negative, mismatched, NaN, or infinite denominator/result is a hard
run failure. No epsilon substitution, dropped origin, finite-only averaging, or
zero-denominator convention is permitted.

The invalid cost bound is safe only after the public goal is explicitly covered by the
same coordinate envelope as the state. Insert:

\[
|x_r|\le2,\quad |g_r|\le2,\quad \sigma_{x,r}\ge0.05.
\]

Then each standardized squared state difference is at most \(80^2=6{,}400\).
Registered actions have \(\|a\|_2^2=1\), so their action term is exactly 0.01, and

\[
J_{\rm valid}\le6{,}400.01<10{,}000=P_{\rm invalid}.
\]

The generator feasibility gate “at least one valid \(J\le25\) and one valid \(J>25\)
per origin” is an executable preregistered acceptance test, not an analytic consequence
of the current constants. It must run once on the frozen generator before any training,
without resampling failed origins; any failure stops the whole registered run. The
declared goal RNG stream is currently unused by the deterministic goal formula: delete
that stream from the registry or define its exact contribution before hashing.

## 5. Factor identification after the fixes

After the P0 replacements:

- the LTM factor changes only scoped episodic completion and the resulting
  \(\widehat q\);
- the dream factor changes only the 24 missing residual-schema slots;
- observed anchors and the 20-parameter core are immutable and identical across cells;
- M01 destroys queryable episodic contents while retaining identical allocation,
  calls, and counters;
- no entity fingerprint enters dream acceptance or schema rollout;
- attribution controls use the same fixed allocation and operation ledger.

Under those conditions, A1's marginal and simple effects identify conditional software
treatment effects for this frozen generator and implementation. They do not identify a
general biological memory mechanism. Before the fixes, raw/std mixing and fingerprint
joining contaminate the treatment paths, while the underdefined lesion does not identify
an unconstrained-recombination effect.

The registered benefit signs are correct:

\[
d_s^{\rm LTM}
=\frac{E_{00,s}+E_{01,s}}2-\frac{E_{10,s}+E_{11,s}}2,
\]
\[
d_s^{\rm dream}
=\frac{E_{00,s}+E_{10,s}}2-\frac{E_{01,s}+E_{11,s}}2,
\]
\[
I_s=(E_{10,s}-E_{11,s})-(E_{00,s}-E_{01,s}).
\]

Positive values mean benefit under these definitions. Relative gates remain
ratio-of-means with a strictly positive registered denominator; a nonpositive or
nonfinite denominator is a hard failure. A paired difference of exactly zero is a tie,
not a win. Provenance remains hard-zero: any positive attempt, successful forbidden
write, recalled synthetic identity, or scope/provenance violation fails immediately.

## 6. Dimensionless gate

The generator's state, action, drift, transition, residual, standardized error, cost,
NRMSE, regret ratio, factorial effect, and CI quantities are dimensionless. Arguments of
\(\tanh\) are dimensionless. Square-root mean-square distances use a fixed coordinate
count, and ratios compare like quantities. The mathematical dimensionless gate therefore
passes after the raw/std coordinate boundary in P0-2 is enforced.

This is a symbolic audit only. No dimension checker was executed because this round
forbids running code.

## 7. P1 fixes

1. Hash-lock the complete typed learner return schema, not only numeric predictions.
2. Report the exact recall acceptance count in addition to conditional recall error.
3. Report dream rejection reasons separately: observed_key, component/port,
   left_join, right_join, provenance, and capacity_padding.
4. Report both \(\operatorname{upper}_{95}\)(false_lure) and the maximum per-seed lure
   rate; keep cross-context accepted recall exactly zero.
5. Remove unused RNG streams and duplicate symbols before freezing the registry.
6. Call the 288 quantity pair enumerations, not proposals, in every ledger and table.
7. Record the public-goal envelope and the valid-cost bound as pre-seed assertions.

## 8. Reproduction / scratch record

Read-only inputs:

- revisions/00-contract-v2-draft.md
- revisions/v2-redteam.md
- 11-math.md
- 20-audit.md
- .codex/agents/ce-math-verifier.md
- .codex/skills/ce-dimensionless/SKILL.md
- docs/참조/무차원_감사_수학.md

Input SHA-256 values observed during this audit:

- V2 draft: be19f0fd23ac883786adb8a92e7e3cb1c13d335515cee2a3e67bcdff167c75d4
- V2 red-team: beb6a2dda8f00539a4bcaf542882288ef31bfa3090892e6066be715df9665801
- V1 math: ce9c1991922475ba92b661b8f520900735d449ba87c6f65261b07923f72e8290
- V1 audit: fd48d16d4aa5f142db6d05894dcbd118303b5b82c3da6d77a9e0572252e8b7fe

Scratch calculations were algebraic and count-based only. No registered seed, code path,
test, calibration selector, or benchmark was executed.

Final decision: REVISE. Incorporate every P0 replacement above, freeze and hash the
result, then perform a final pre-seed executability audit. Until then, do not train and
do not run registered seeds.
