# Research contract — G9-CBM V2 executable draft

Status: LOCKED_PRE_IMPLEMENTATION

## 0. Disposition and version boundary

G9-CBM V1 remains `Gate: BLOCKED`. No V1 train, validation, or test seed was
opened. This document is a replacement preregistration draft, not an amendment
that changes the interpretation of V1. It resolves V1 findings P0-1 through
P0-8 and audit actions A1 through A10 before implementation.

V2 uses only route R1: **early prefix completion plus missing slow-schema
completion**. R2 residual recursion, R3 graph-state rollout, R4 posterior
conditioning, V6 consensus, and the V7 AR replacement are outside the search
space. No G9-CBM V2 registered seed or development seed was run while drafting
this contract.

The independent red-team requirements are adopted: a new generator rather than
the G7-M generator, an in-component recall wrapper, independent entity
fingerprint and drift, a nonreconstructible missing-pair interaction, opaque
same-value validity lures, and an integer budget. This contract deliberately
uses `d=4,m=2`, 24 origins, trajectory cost, and a larger padded ledger instead
of the red-team's scalar-action/one-origin minimal variant; those fixed choices
score every missing pair in every seed and give recall rates nontrivial
within-seed denominators. The `d=4,m=2`, 24-origin design below is the sole V2
design. The scalar-action/one-origin red-team sketch is audit input only and is
not implemented, piloted, or selected after train.

## 1. Question and claim boundary

In the exact CPU-only synthetic family below, conditional on one frozen
train-only calibration artifact, do (i) a real-wake-only episodic lookup and
(ii) a provenance-separated constrained completion of preallocated slow-schema
slots improve H20 autonomous action-conditioned rollout and fixed-set action
selection, while every registered leakage, provenance, constraint, and resource
gate remains satisfied?

A PASS supports only a conditional software-component statement for this one
generator, codec, bank size, horizon, action set, and calibration sample. It
does not establish a general world model, semantic consolidation, recurrent
attractor dynamics, biological hippocampal memory, human dreaming or sleep,
consciousness, open-world planning, continual learning, or AGI. G9-CB V5–V7
remain active failed long-horizon counterevidence. G7-M V2 remains evidence for
deterministic hard exemplar completion and constrained known-slot
recombination only.

## 2. Fixed spaces, tokens, and valid compositions

All numeric generator variables are dimensionless.

- Observed state: `x_t in R^4`; therefore `d=4`.
- Action: `a_t in R^2`; therefore `m=2`, with each coordinate in `[-1,1]`.
- Horizon: `H=20`. H5 is rows `0:5` of the same H20 array.
- Contexts: `c in {0,1}`; ports: `p in {0,1,2,3}`.
- Prefix locals and suffix locals: `i,j in {0,1,2}`.
- Each seed independently permutes opaque strings for `(c,p,i)` prefix tokens,
  `(c,p,j)` suffix tokens, the context `c`, and the public port `p`. The
  `EpisodicRecord.context_token` represents `c`; co-occurrence components of
  its prefix/suffix tokens separate the four ports. Tokens expose
  equality only and never contain an integer index, seed, split, validity bit,
  goal identity, or outcome rank.

For every `(c,p)`, the wake-observed binding set is

```text
O = {(0,0),(0,1),(1,1),(1,2),(2,2),(2,0)}.
```

The valid but wake-unobserved set is

```text
M = {(0,2),(1,0),(2,1)}.
```

Thus `O union M = {0,1,2}^2`. A fragment splice is generator-invalid if it
crosses context or port, uses a token not assigned to that seed, overwrites an
`O` binding, changes the 12-row phase order, or uses an invalid action.

## 3. Exact generator

### 3.1 Stable action-conditioned mechanism

The mechanism is fixed across every seed and split:

```text
D = diag(0.55, 0.50, 0.45, 0.40)

B = [[ 0.00,  0.00,  0.00,  0.05],
     [ 0.08,  0.00,  0.00,  0.00],
     [ 0.00, -0.07,  0.00,  0.00],
     [ 0.00,  0.00,  0.06,  0.00]]

G = [[ 0.28,  0.00],
     [ 0.00,  0.28],
     [-0.14,  0.10],
     [ 0.10, -0.14]].
```

For lead phase `phi_t = t mod 12`, define the schema residual

```text
b(c,p,i,j,phi) = P[c,p,i]                 for phi in {0,1,2,3,4}
                 C[c,p]                   for phi in {5,6}
                 S[c,p,j] + E[c,p,i,j]    for phi in {7,8,9,10,11}.
```

The structural equation under intervention is

```text
x[t+1] = D x[t] + B tanh(x[t]) + G do(a[t])
         + q_episode + b(c,p,i,j(a[t]),phi_t) + eta[t+1].
```

Actions are exogenous. `eta` is additive and independent of state. Wake uses
antithetic paired arrays; evaluation uses one array shared at the same
`(seed,origin,lead)` across all action candidates and all cells. This common
noise coupling is the counterfactual coupling. No candidate receives `eta`.

### 3.2 Seed-specific bounded primitives

All streams are separate `numpy.random.Generator(numpy.random.PCG64(
numpy.random.SeedSequence([master_seed, stream_id])))` children. The exact NumPy
version is frozen in the implementation lock and must match calibration,
validation, and test. Stream IDs are frozen as: token `0`, initial state `1`, prefix
primitive `2`, connector `3`, suffix primitive `4`, irreducible interaction
`5`, episode drift `6`, entity fingerprint `7`, wake noise `8`, evaluation
prefix `9`, evaluation noise `10`, cue mask `11`, cue noise `12`, and lure
construction `13`. Candidate-presentation permutation is a handcrafted off-range
unit fixture and has no registered RNG stream.
Candidate code receives neither master seed nor stream IDs.

For each vector family, draw iid `Uniform[-1,1]`, subtract its local mean over
the three locals where applicable, and divide the whole local family by its
maximum absolute value; a value `<=1e-12` hard-fails. Then set

```text
P[c,p,i] = 0.025 * normalized_prefix[c,p,i]
C[c,p]   = 0.010 * normalized_connector[c,p]
S[c,p,j] = 0.025 * normalized_suffix[c,p,j].
```

The array/draw convention is exact and C-order: prefix stream 2 draws raw shape
`(2,4,3,4)` and, separately for each `(c,p)`, subtracts the mean over the local
axis of length 3 and divides that `(3,4)` block by its maximum absolute value;
suffix stream 4 does the same with raw shape `(2,4,3,4)`. Connector stream 3
draws shape `(2,4,4)` and normalizes each single `(4,)` `(c,p)` vector by its
own maximum absolute value without mean subtraction. Any normalization
denominator `<=1e-12` hard-fails rather than resampling. Context is the outermost
axis, then port, local, and state coordinate.

Draw `E[c,p,i,j]` independently from `Uniform[-0.010,0.010]^4` for all nine
valid pairs, including the three missing pairs. `E` is never exposed to the
learner and is not algebraically reconstructible from prefix/connector/suffix
means. Therefore constrained recombination estimates a missing schema slot; it
does not receive an oracle-exact concatenation target.

Interaction stream 5 draws one C-order array `(2,4,3,3,4)`. Drift stream 6
draws `(2,4,6,4)` in displayed-`O` order and normalizes each final axis.
Fingerprint stream 7 draws `(2,4,6,2,4)` with sign order `h=-1,+1` and
normalizes each final axis. A norm `<=1e-12` in either family hard-fails.

Independently draw one drift direction for every `(c,p,i,j) in O`, divide by its
L2 norm, and call it `u_theta[c,p,i,j]`; zero norm hard-fails. The two real wake
instances have `q_episode=h*0.040*u_theta`, `h in {-1,+1}`. Independently draw
and normalize a different fingerprint `f[c,p,i,j,h]` for each real entity. The
public signature sensor at row `r` is
`y[r]=f+0.005*xi[r]`, `xi[r]~Uniform[-1,1]^4`, with fresh independent arrays
for wake and re-encounter prefixes.
Fingerprint and drift streams are independent; no token or fingerprint encodes
`q`, schema, goal, validity, future, or candidate rank. Initial states are iid
`Uniform[-0.25,0.25]^4`.
Wake innovations are iid `Uniform[-0.002,0.002]^4`; the `h=-1` record uses the
negative of the corresponding `h=+1` innovation array. Evaluation innovations
are iid `Uniform[-0.002,0.002]^4` and common across candidates/cells.

Every generated true or predicted valid state must satisfy `max(abs(x))<=2`.
Generator violation is a hard implementation failure; a candidate violation is
counted by the validity metric. The registered coefficients and bounds imply a
finite invariant envelope, but the executable assertion is still mandatory.

### 3.3 Learned shared core

The candidate does not receive `D,B,G`. On the 40 train seeds only, pool every
real wake transition and fit one common model. For target coordinate `r`, use
the design columns

```text
[1, x[r], tanh(x[source(r)]), a[0], a[1]],
source = (3,0,1,2) for r=(0,1,2,3).
```

Solve `(X.T X + Lambda)^-1 X.T y` with float64 `numpy.linalg.solve`, ridge
`1e-6` on the four non-intercept coefficients and zero ridge on the intercept.
No coefficient clipping, edge selection, validation refit, or seed-specific
core adaptation is allowed. The resulting 20 float64 parameters and canonical
SHA are common to all cells and later splits.

For target coordinate `r`, retain all five fitted coefficients in the exact
ordered expression

```text
f_hat_r(x,a) = c_hat[r] + d_hat[r]*x[r]
               + b_hat[r]*tanh(x[source(r)])
               + g_hat[r,0]*a[0] + g_hat[r,1]*a[1].
```

The ordered core vector is `(c_hat,d_hat,b_hat,G_hat)` with respective sizes
`4,4,4,8`; therefore `P_core=20` and `N_solve=4`. The same expression, including
`c_hat`, is used for every wake/evaluation codec residualization, every candidate
rollout, and every learned-core control. It is never used to generate evaluator-
truth prefix or future states. An intercept-free 16-parameter interpretation is
forbidden.

## 4. Wake ledger and G7-M V2 12x8 codec

### 4.1 Exact real wake records

Each seed has exactly

```text
2 contexts * 4 ports * 6 observed bindings * 2 signs = 96 records.
```

Each record contains 13 observed states, 12 executed valid actions, 12 public
signature observations, immutable opaque tokens, and a unique real ledger ID.
For a record with suffix local `j`, its action is `A[j]` at all 12 rows, so the
suffix selected by the structural equation is exactly the record's suffix.
Across `O`, all three actions occur equally often. All 96
records end before every evaluation origin. Canonical append order is
`c,p,i,j,h`, with `O` in its displayed order and `h=-1,+1`.

After the shared core is frozen, `TrajectoryCodecV2` maps a completed real wake
record to exactly one float64 `(12,8)` array:

```text
r_hat[r] = x[r+1] - f_hat(x[r],a[r])
trajectory_raw[r,0:4] = r_hat[r]
trajectory_raw[r,4:8] = y[r]
```

Here `f_hat` is exactly the five-coefficient expression in Section 3.3, including
`c_hat`. The raw codec array is denoted `T_raw`; the episodic store owns this raw
payload. Define `vec_C` as C-order flattening. Its derived read-only standardized
view is exactly

```text
T_std = reshape((vec_C(T_raw)-mu_codec)/sigma_codec, (12,8), order="C"),
```

with a coordinatewise population-scale floor of `1e-8`. This same `vec_C` and
reshape operation is used by inherited masked recall, calibration hashes, schema
construction, inverse-standardization, and recall-error scoring. `T_std` is not
stored as an episodic payload.

Rows `0:5`, `5:7`, and `7:12` are respectively the G7-M prefix, connector, and
suffix slots. The codec reads only completed wake transitions. It never encodes
an evaluation future, hidden generator parameter, RNG/seed, split, validity
label, evaluator cost, or oracle action. Float order is C-contiguous float64.

The implementation must import and hash-lock the G7-M V2
`EpisodicRecord`, `PartialCue`, `PersistentEpisodicStore`, and
`hard_cue_anchored_recall`. It also hash-locks the provenance/immutability
semantics of `SlowSchemaTable`, `constrained_missing_binding_dream`, and
`update_missing_slow_binding`, but implements the separately locked residual-only
72-slot V2 adapter below rather than calling their eight-column join path.
Capacity is 96 and trace payload is exactly
`96*12*8*8 = 73,728` bytes. Hard recall keeps its strict `confidence > tau`,
masked-coordinate non-read, first-storage-index tie, component precheck, zero
rejection reconstruction, and single-exemplar completion. It is not called an
attractor. The inherited routine's all-bank argmax after its component precheck
is not sufficient here: a new `scoped_hard_recall_v2` wrapper must first form a
read-only, storage-order-preserving facade containing only records in the cue's
learner-inferred context and co-occurrence component, then call the inherited
hard routine. This is a registered remediation, not a claim of byte-equivalence;
tests must prove equivalence on already in-scope inputs and zero possible
cross-context/component return on mixed banks.

### 4.2 Evaluation origins and cues

There are exactly `N_origin=24` origins per seed, one for each `(c,p,i)` in
lexicographic order. Its anchor suffix is `j_anchor=i`, which is in `O`. Its
sign is `h=-1` when the origin index is even and `h=+1` otherwise. The current
evaluation episode has a fresh unique ID but shares the selected wake record's
`q_episode`, opaque context/component tokens, and entity fingerprint.

Its observed prefix is constructed exactly as follows. Use numeric action `A[i]`
and the correct learner-visible opaque action token for all 12 transitions, use
the observed anchor key `(i,i)`, draw a fresh initial state and 12 fresh
innovations from the registered evaluation streams, and apply the same structural
equation as wake generation with the same `q_episode`. Record `x[0]..x[12]`;
prediction begins at `x[12]`. Because the prefix is one complete 12-row cycle,
future lead one begins at phase zero. These innovations are independent of all
train/wake innovations and are never candidate inputs. Evaluation futures are
separate arrays and remain disjoint from the wake ledger.

The `PartialCue` is the current prefix codec view. It has exactly 24 visible
cells: 10 uniformly permuted cells among rows `0:5`, 4 among rows `5:7`, and 10
among rows `7:12`. The mask is fixed before cells execute. For masked cell
`(r,h)`, visible noise is exactly
`0.01*reshape(sigma_codec,(12,8))[r,h]*Normal(0,1)`; values outside the mask are
poison and must never be read. The positive target identity is the selected real wake
record, not the new evaluation episode.

For each origin construct one unstored near-neighbour lure with identical valid
tokens. Let normalized target fingerprint be `f`. For each attempt draw
`v~Uniform[-1,1]^4` from lure stream 13, form
`v_perp=v-dot(v,f)*f`, reject the attempt when `norm(v_perp)<=1e-12`, otherwise
normalize it and set

```text
f_lure = 0.85*f + sqrt(1-0.85^2)*v_perp.
```

Accept the first attempt with
`max_other(abs(dot(f_lure,f_other)))<0.95`; its target cosine is exactly 0.85.
Draw in canonical origin order with at most 10,000 attempts per origin. Then draw
one fresh `Uniform[-1,1]^4` drift vector from stream 13, hard-fail if its norm is
`<=1e-12`, normalize and scale it by `0.040`; use further stream-13 draws for the
lure's fresh initial state, innovations, and `0.005*Uniform[-1,1]^4` signature
noise. Generate its 12-step prefix from the same anchor schema. Neither
fingerprint nor drift is stored. Exhaustion hard-fails the generator. Also construct one invalid
cross-port cue by retaining the prefix token and replacing the suffix token by
the next port's suffix token. Thus each seed has `|P_s|=24`, `|L_s|=24`, and 24
cross-port diagnostics.

### 4.3 R1 completion and slow schema

All states, actions, residuals, codec coordinates, rollout errors, and costs are
dimensionless. Raw versus standardized coordinates are nevertheless distinct
numerical representations and are never mixed implicitly:

1. `T_raw in R^(12x8)` is the codec generated from a raw dimensionless wake
   trajectory and is the episodic store payload. Its derived `T_std` view is used
   only for masked addressing/thresholding and within the V2 schema/dream adapter.
2. `scoped_hard_recall_v2` accepts a raw cue, standardizes internally, and returns
   a completed raw codec view. It never returns a standardized view to rollout.
3. Every cell clamps visible raw cue cells. On acceptance, hidden raw cells come
   from the selected real exemplar. On rejection or with LTM disabled, hidden raw
   cells are filled by raw `mu_codec`; standardized zero is not substituted by
   an implicit conversion. Episode ID is discarded before rollout.
4. Slow-schema and dream payloads are stored in standardized coordinates. Before
   rollout they are converted exactly by
   `S_raw=reshape(mu_codec+sigma_codec*vec_C(S_std),(12,8),order="C")`, and only
   columns `0:4` are read.

Let `schema_anchor_raw[12,4]` be columns `0:4` of the inverse-standardized real
observed slow-schema mean for the current anchor key, available equally in every
cell. The only prefix condition is

```text
q_hat = mean(completed_view_raw[:,0:4] - schema_anchor_raw, axis=0).
```

There is no learned blend, persistent residual, confidence weighting, or
candidate-specific refit. Disabled LTM sets `q_hat` through the same fallback
completion path. Fingerprint columns are used to address an entity but are not
decoded as drift; the drift estimate comes only from that entity's observed
pre-origin transition residuals.

All cells have a 72-key preallocated V2 schema adapter for
`2*4*3*3` `(context,port,prefix,suffix)` keys, with explicit observed/synthetic
occupancy and provenance masks. It does not rely on a dynamic inherited
dictionary or claim type identity with the inherited table. The two antithetic
real records make the observed binding mean cancel episodic `q` and wake noise.
Schema decode inverse-standardizes an entry and reads only residual columns
`0:4` as its 12-phase template. Fingerprint columns `4:8` are never used by
slow-schema construction, dream joins, dream acceptance, or rollout. Observed
keys are immutable. To keep the inherited storage shape and the common allocation
ledger, schema columns `4:8` are stored as standardized zero in every key and
condition; they are excluded from every join statistic and never decoded.

The adapter's aggregation is unique. First average the two standardized real
records for each of the 48 observed `(context,component,prefix,suffix)` keys and
retain residual columns `0:4`. For each context/component, define each prefix
fragment as the arithmetic mean of rows `0:5` over every observed key having that
prefix; define the connector as the arithmetic mean of rows `5:7` over all six
observed keys; and define each suffix fragment as the arithmetic mean of rows
`7:12` over every observed key having that suffix. Concatenating those exact
`(5,4),(2,4),(5,4)` arrays forms a missing-key proposal. An unfilled missing-key
fallback is the arithmetic mean of the six complete observed residual templates
in the same context/component. All means use float64 and canonical wake order;
empty or mismatched groups hard-fail.

Dream candidate order is canonical `(context ordinal, component ordinal,
prefix ordinal, suffix ordinal)` and includes exactly the 24 keys in `M`; no
other key reaches join acceptance. An unfilled `M` key still resolves to its
preallocated `0..71` index, remains learner-valid, and uses the registered
component fallback. Hypothetical occupancy changes only its schema source and
payload, never action validity.

Dream-enabled cells take one immutable 96-real-record snapshot and execute
exactly one hash-locked V2 residual-only constrained recombination pass. For a
candidate residual prefix `P`, connector `C`, and suffix `S`, the two join values
are

```text
J_left  = sqrt(mean((P[-1,0:4]-C[0,0:4])^2))
J_right = sqrt(mean((C[-1,0:4]-S[0,0:4])^2)).
```

Both comparisons use `<=` against the registered residual-only threshold; equality
is accepted. The adapter
rechecks same context, same port/component, token, phase, numeric action, and
empty-key compatibility, then writes with unit weight only to an empty `M` slot.
It must not claim direct equivalence to the inherited eight-column proposer/join.
Outputs remain
`synthetic/hypothetical/observed=false/recalled=false/episode_id=null`.

The `(288,) uint8` pair-reason codebook is frozen as:
`0=unexamined`, `1=same_component_candidate`,
`2=constrained_component_port_rejection`, `3=observed_key_rejection`,
`4=join_accepted`, `5=left_join_rejection`, `6=right_join_rejection`,
`7=lesion_accepted_valid_missing`, `8=lesion_accepted_invalid_cross_port`, and
`9=lesion_capacity_padding`; values `10..255` are invalid. If both joins fail,
left rejection has priority. All 288 check flags must be true after a pass.
No-dream cells execute an equal-shaped shadow pass and discard every write. For
`M01`, the real snapshot is destroyed before evaluation and its queryable LTM
has zero records.

At rollout lead `t`, a valid action selects `j(a_t)`, the schema table supplies
row `(t mod 12)`, and the candidate computes

```text
x_hat[t+1] = f_hat(x_hat[t],a[t])
             + q_hat + s_hat_raw[c,p,i,j(a_t),t mod 12].
```

The anchor `x_hat[0]=x_origin` is not part of the returned array. Returned row
`predictions[k,ell,:]` stores mathematical state `x_hat[ell+1]` for
`ell=0..19`; evaluator future and public-goal arrays use the identical lead-row
convention.

An observed key uses its real mean. A missing unfilled key uses the residual-only
V2 adapter's component-local mean fallback. A dream-filled key uses the hypothetical
entry. Predictions recurse; no true state is reread.

For a token-incompatible step, the rollout still executes the same arithmetic
using the numeric action and component-local fallback schema row, marks the step
inferred invalid, returns schema key `-1` and source code `0`, and continues. The evaluator does not need an invalid-action
counterfactual state: its cost is the registered penalty, but the padded
predicted array must remain finite and length 20.

## 5. Fixed actions, goal, and planning

### 5.1 Alphabet and canonical candidate order

```text
A0 = (-1, 0), A1 = (0, 1), A2 = (1, 0).
j(A0)=0, j(A1)=1, j(A2)=2.
```

Each numeric action also carries an opaque `(context,port,action-local)` token.
A step is valid only when that token belongs to the request context/port, its
public numeric value equals the numeric action, and the required schema key
resolves in the learner's wake-derived index.

`K=8` sequences, in immutable canonical order, are:

```text
0: A0 repeated 20 times
1: A1 repeated 20 times
2: A2 repeated 20 times
3: repeat (A0,A1,A2) and truncate to 20
4: repeat (A1,A2,A0) and truncate to 20
5: repeat (A2,A0,A1) and truncate to 20
6: numeric copy of sequence 3; only one-based lead 7 (`ell=6`) uses the same-valued token from
   port `(p+1) mod 4`
7: numeric copy of sequence 4; only one-based lead 13 (`ell=12`) uses the same-valued token from
   context `(c+1) mod 2`.
```

Indices 0–5 are generator-valid; 6–7 are deliberate opaque-token lures. Their
numeric arrays are byte-identical to valid templates, so magnitude cannot reveal
invalidity. Every cell
rolls all eight for all 20 leads. No inferred-invalid early exit or candidate
removal is allowed. The canonical order is formulaic and independent of goals
and outcomes. A handcrafted permutation test must reorder candidates, map back
to canonical indices, and obtain identical selections except exact ties.

From wake records only, construct one immutable action index that maps every
observed opaque action token to exactly one tuple

```text
(context_token, component_id, numeric_action, suffix_token).
```

The component ID is inferred only from the prefix/suffix co-occurrence graph and
frozen first-occurrence token order. Construction hard-fails if a token has zero
or multiple mappings. A candidate step is learner-valid iff its token resolves
uniquely, the resolved context/component equals the request context/component,
the numeric action is exactly the registered action vector, and the corresponding
schema key resolves. There are no undeclared continuity or emitted tokens.

Context, component, prefix, and suffix ordinals are each their frozen
first-occurrence order in the canonical real-wake ledger. A schema key is

```text
key_index = (((context_ordinal*4 + component_ordinal)*3
              + prefix_ordinal)*3 + suffix_ordinal),
```

so valid `int16` values are exactly `0..71` and unresolved is `-1`. The returned
`schema_source` codebook is `uint8`: `0=unresolved`, `1=observed_real`,
`2=synthetic_hypothetical`, `3=component_fallback`; values `4..255` are invalid.
For each candidate `k`,
`inferred_sequence_valid[k]=all(inferred_valid[k,0:20])`. Its inferred cost is
exactly 10000 when false and otherwise uses the registered state/action cost.

### 5.2 Public goal and cost

For origin `(c,p,i)`, `goal_id=(c+2*p+i) mod 3`. Define
`x_origin:=x_prefix[12]` and `x_goal[0]=x_origin`. For lead `ell=0..19`, advance
the public reference with the registered literal `D,B,G`, constant action
`A[goal_id]`, and `q=b=eta=0`, then store `g[ell,:]=x_goal[ell+1]`. Thus `g`
has exact shape `(20,4)` and is not a view of any evaluator outcome.

Fit state normalizers on train wake states only:

```text
mu_x[j]    = population mean, ddof=0
sigma_x[j] = max(population std, 0.05)
z_j(v) = (v_j-mu_x[j])/sigma_x[j].
```

With `lambda=0.02`, all coordinate/action weights one, define for a valid
sequence

```text
J(o,k) = sum[ell=0..19,j]
           (z_j(x_star[o,k,ell])-z_j(g[o,ell]))^2 / (20*4)
         + 0.02 * sum[ell=0..19,h] a[k,ell,h]^2 / (20*2).
```

For an invalid sequence, `J(o,k)=P_invalid=10000`. When a split is legally
opened, the generator must assert both `max(abs(x_star))<=2` and `max(abs(g))<=2`
for every origin/candidate in that split without resampling. Given
`sigma_x>=0.05`, every valid state
term is at most 6400. Every registered action has squared norm one, so the action
term is exactly `0.01`; hence `J_valid<=6400.01<10000`.
The candidate uses the identical formula on `x_hat`; its inferred graph assigns
the same finite penalty to incompatible opaque tokens without consulting
generator truth.

```text
k_hat(o,c) = argmin_k (J_hat(o,c,k), canonical_index(k))
k_opt(o)   = argmin_k (J(o,k),       canonical_index(k))
r(o,c)     = J(o,k_hat)-J(o,k_opt)
s(o,c)     = 1[J(o,k_hat)<=25 and selected sequence is generator-valid]
R(seed,c)  = mean_o r(o,c)
S(seed,c)  = mean_o s(o,c).
```

Ties use the lower canonical index. Require `r>=-1e-12`; a smaller value is an
evaluator hard failure and values in `[-1e-12,0)` are serialized as zero. The
generator asserts before candidate execution that every origin has at least one
valid candidate with `J<=25` and at least one valid candidate with `J>25`; a
failure invalidates the seed and the whole registered run rather than resampling.

## 6. Candidate/evaluator chronology and capability boundary

The exact per-seed order is:

1. Privately generate streams, truth, 96 complete wake records, 24 evaluation
   prefixes, all action arrays, public goals, and evaluator-only futures.
2. Seal evaluator futures, true schema/q, validity labels, costs, and stream
   metadata behind `EvaluatorOnlyHandle`; candidate modules cannot import or
   receive the handle type.
3. Append and hash the immutable real wake ledger. Build isolated, byte-equal
   core/schema/capacity copies for all cells.
4. Run the one offline real-snapshot/shadow pass. Destroy the M01 snapshot and
   hash evaluation-time stores.
5. Give a candidate only the following frozen dataclasses and exact field order:

   ```text
   CoreModelV2(
     intercept:float64[4], diagonal:float64[4],
     bridge:float64[4], action:float64[4,2])

   CostSpecV2(
     mu_x:float64[4], sigma_x:float64[4],
     action_cost_weight:float64, success_threshold:float64,
     invalid_penalty:float64)

   CodecSpecV2(mu_codec:float64[96], sigma_codec:float64[96])

   CandidateRequestV2(
     cue:PartialCue, anchor_state:float64[4],
     numeric_actions:float64[8,20,2], action_tokens:OpaqueTokenV2[8,20],
     public_goal:float64[20,4], cost_spec:CostSpecV2,
     codec_spec:CodecSpecV2, core:CoreModelV2,
     action_index:WakeActionIndexV2, schema:ResidualSchemaTableV2,
     episodic_store:ScopedEpisodicFacadeV2|null)
   ```

   `PartialCue` itself owns the observed raw `(12,8)` prefix codec, `(12,8)`
   Boolean mask, and opaque context/prefix/anchor-suffix tokens. `CodecSpecV2`
   is one read-only object shared by episodic/schema adapters and counted once.
   The request contains no cell label or evaluator handle.
6. Return one typed `CandidateResultV2` containing exactly `(8,20,4)` float64
   predictions, `(8,20)` Boolean inferred-valid values, `(8,20)` `int16`
   resolved-schema-key indices (`-1` when unresolved), `(8,20)` `uint8`
   schema-source codes, `(8,)` float64 inferred costs, one canonical `int64`
   selected index, one origin-local `OriginRecallAuditV2`, and no token emissions.
   The exact result types are

   ```text
   OriginRecallAuditV2(
     accepted:bool8, identity:int16, confidence:float64, scope:uint8)

   CandidateResultV2(
     predictions:float64[8,20,4], inferred_valid:bool8[8,20],
     resolved_schema_keys:int16[8,20], schema_sources:uint8[8,20],
     inferred_costs:float64[8], selected_index:int64,
     origin_recall_audit:OriginRecallAuditV2)
   ```

   Identity
   is the physical ledger index `0..95` or `-1`, and scope is
   `0=no_queryable_scope_disabled_ltm`, `1=valid_12-row_scope`, and
   `2=invalid_context_or_component_precheck` (values `3..255` invalid).
   When scope is 0 or 2, accepted is false, identity is -1, and serialized
   confidence is the finite sentinel `-2.0`. Invalid context/component is checked
   before store availability, so a cross-port diagnostic always uses scope 2.
   An inherited internal `-inf` may exist only inside the helper and is never
   serialized, thresholded, averaged, or exposed to candidate arithmetic. When
   scope is 1, confidence is finite and lies in `[-1,1]`.
   The schema-key/source arrays are the typed local provenance audit.
7. The seed harness, not a per-origin request/result, owns
   `SeedRecallAuditV2(accepted:bool8[72], identity:int16[72],
   confidence:float64[72], scope:uint8[72])` in that field order. It copies the 24
   positive origin audits in canonical order, then invokes the same scoped
   facade once for each of the 24 lures and 24 cross-port diagnostics and appends
   those scalar audits in canonical origin order. These diagnostic cues are
   never passed to `execute_candidate_v2` and never affect rollout or selection.
   The harness also owns and appends the 17 integer hard counts in exact Section
   11 order. Hash every 24-origin result sequence and both harness audits
   separately for all cells and non-oracle controls before evaluator access.
8. Only after every hash is frozen, unseal the evaluator and score all fixed
   futures and selected actions. The evaluation prefix/suffix is never inserted
   into LTM during the run.

The candidate API contains no `World`, `Episode`, master seed, stream ID, split
result, target episode ID, evaluator handle, future state, true q/schema,
generator graph/validity label, outcome, realized cost, reward, or oracle rank.

## 7. Factorial cells and controls

The four isolated cells differ only in queryable LTM content and accepted
hypothetical slot writes:

| cell | queryable real records | hypothetical M-slot writes |
|---|---:|---:|
| M00 | 0 | 0 |
| M10 | 96 | 0 |
| M01 | 0 after offline snapshot destruction | at most 24 |
| M11 | 96 | at most 24 |

All cells retain identical observed slow-schema entries, core bytes, inferred
graph, array capacities, inputs, calls, and padding.

Required non-oracle controls use identical requests and the same padded budget:

- persistence: repeat the observed anchor;
- frozen causal core: recursive `f_hat`, including `c_hat`, with `q=s=0`;
- schema-only fallback: R1 with empty LTM and no hypothetical writes (identical
  to M00, checked byte-for-byte);
- shuffled episodic binding: preserve every M10 recall acceptance/confidence,
  but cyclically assign accepted reconstruction to the next query within the
  same `(context,port)` stratum before projecting q;
- zero-q ablation: preserve recall calls/audits but replace the projected q by
  zero immediately before rollout;
- unconstrained recombination lesion: execute the same 288 ordered-pair
  enumerations (`2` contexts times `12` prefix tokens times `12` suffix tokens)
  and 288 checks, but remove component/port/join acceptance. Enumerate by
  learner-visible `(context token, prefix slot, suffix slot)` using frozen
  first-occurrence token order, reject the 48 observed keys, accept the first 24
  of the remaining 240 into a preallocated lesion-audit buffer, and reject the
  other 216 with reason `capacity_padding`. These 24 objects never enter the
  72-key valid schema table and never become action-index edges. Provenance and
  observed-overwrite rejection remain active; exactly 24 lesion objects and 24
  padded update slots are required. Constrained component/port rejection count
  is exactly `288-72=216`; it has a distinct reason code from the lesion's
  `240-24=216` capacity-padding count. With the displayed `O` order and frozen
  traversal, the lesion's first 24 contain exactly 3 same-port valid-missing and
  21 cross-port invalid objects, so its invalid-splice rate is exactly
  `21/24=0.875`. Every condition computes all 48 residual endpoint values; the
  lesion records but ignores their threshold for acceptance.
- zero-synthetic-slot ablation: preserve dream calls/audits but replace all
  hypothetical missing-slot payloads by the ordinary fallback before rollout.

Every condition, including M00/M10/M01/M11 and every non-lesion control, executes
one active constrained-or-shadow dream enumeration into the shared 288 pair
check/reason arrays, then performs the same deterministic `240 -> 24` lesion
post-classification from those already enumerated indices into the separate
preallocated lesion audit buffer. This post-classification does not enumerate a
pair again, compute an extra join, propose/update a schema item, or increment
`N_dream_passes`; the registered counts remain 288 enumerations, 48 endpoint
values, and one dream pass. In a non-lesion condition this classification is
shadow-only: its bytes are hashed but
never enter the 72-key schema/action index, never affect rollout/selection, and
are excluded from constrained provenance counts and the condition's scientific
`invalid_splice_rate`. The unconstrained-lesion control exposes exactly that same
audit only to its registered diagnostic metric. Thus every condition records
actual lesion counts `240,24,216` without overwriting the constrained pair-reason
array. The `(24,)` lesion occupancy array is all true; its `uint8` provenance
codes are `1=valid_missing` for exactly 3 slots and `2=invalid_cross_port` for
exactly 21 slots (`0=empty`, values `3..255` invalid). The remaining 216
capacity-padding decisions are represented by the actual 29-vector count and
canonical traversal hash, not a second 288-row reason array. Dream-output
provenance uses `0=empty/rejected`, `1=synthetic_hypothetical` only.

The evaluator-truth oracle uses true core, q, schema, and common noise. It is
diagnostic only, runs after candidate hashes, is outside selection, and cannot
satisfy any treatment gate.

Mechanism attribution requires both:

```text
lower95(E_all(shuffled,20)-E_all(M10,20)) > 0
lower95(E_all(zero_q,20)-E_all(M10,20)) > 0
lower95(invalid_splice_rate(unconstrained)
        - invalid_splice_rate(constrained_M01)) > 0
lower95(E_uv(zero_synthetic,20)-E_uv(M01,20)) > 0.
```

The zero-q ablation must remove the registered LTM benefit and the
zero-synthetic-slot ablation must remove the registered dream benefit under the
same positive paired-difference orientation; otherwise the corresponding path
attribution fails.

If either fails, the all-of V2 claim fails; a favorable M11/M00 result may be
reported only as an unattributed bundle contrast.

## 8. Metrics

### 8.1 Train-standardized rollout errors

For the six valid sequences only (`K_eval=6`):

```text
E_all(s,c,H) = sqrt(
  sum[o,k<6,ell=0..H-1,j] (z_j(x_hat[o,k,ell])
                             -z_j(x_star[o,k,ell]))^2
  / (24*6*H*4)).
```

H5 uses exactly the first five rows of each stored H20 prediction.

The numeric scalar denominators are asserted, not inferred at report time:

```text
E_all(H20): 24*6*20*4 = 11,520
E_all(H5):  24*6*5*4  =  2,880.
```

Before cells execute, the evaluator constructs the H-dependent scalar set

```text
U_s(H) = {(o,k,ell,r): k<6, ell=0..H-1, r=0..3,
          (i(o),j(a[k,ell])) in M}.
```

Membership is defined against the common real wake ledger, is nonempty,
identical across cells, and is never passed to candidates:

```text
E_uv(s,c,H) = sqrt(sum[index in U_s(H)] error_z^2 / count[U_s(H)]).
```

No abstention, missing key, invalid output, or fallback removes an index.
Nonfinite output hard-fails and increments `nonfinite_prediction_count`.
For every origin, exactly two of the six valid candidates use its missing suffix
at each lead. Therefore the asserted scalar denominators are
`24*40*4=3,840` for H20 and `24*10*4=960` for H5; any mismatch hard-fails.

A valid-candidate predicted transition is invalid exactly when it is nonfinite,
has `max(abs(x_hat))>2`, returns `inferred_valid=false`, returns an unresolved or
out-of-range schema-key index, or its schema-source/key audit is inconsistent
with the request context/component, `ell mod 12`, registered numeric action, or
the 72-slot table. These booleans are computed from the already-hashed candidate
return and define the fixed validity numerator; no future/outcome-dependent
geometric tolerance is introduced. Its denominator is exactly
`24*6*20=2,880`.

Recall hidden error is computed on all positive queries, treating rejection as
the coordinate-mean fallback:

```text
E_recall(s,c) = sqrt(sum[positive hidden cells] standardized_error^2
                            / number_of_positive_hidden_cells).
```

The target is the raw codec of the selected immutable ledger record. Target and
raw reconstruction are transformed by the same frozen codec standardizer before
comparison. The asserted hidden-coordinate denominator is
`24*(96-24)=1,728`; rejection does not remove an item.

### 8.2 Recall, dream, and validity denominators

For M10 and M11:

```text
coverage = accepted positives / 24
identity_accuracy = accepted correct identities / 24
wrong_all = accepted wrong identities / 24
wrong_given_accept = accepted wrong / max(accepted positives,1)
false_lure = accepted unstored lures / 24.
```

Report cross-port accepts over 24 separately. Dream valid-binding coverage is
accepted valid missing keys divided by 24. Report all 288 ordered-pair
enumerations, 288 component/port checks, 72 same-component pairs, 48 observed-key
rejections, 24 join candidates, 48 scalar endpoint values, accepted count, and
every rejection reason.

Three invalidity rates are never pooled:

```text
invalid_splice_rate = invalid accepted synthetic-splice audit flags / 24
invalid_predicted_transition_rate = invalid state/key/source/action-audit
                      predictions / 2,880
invalid_selected_action_rate = invalid selected sequences / 24.
```

Action-lure rollouts are excluded from the predicted-transition denominator,
because their action is invalid by construction.

The selected-action, false-positive, false-lure, cross-port, and constrained
dream-coverage denominators are each exactly 24. The unconstrained-lesion
invalid-splice denominator is exactly 24. Empty, zero, negative, nonfinite, or
mismatched denominators/results hard-fail; no epsilon, dropped origin, or
finite-only average is permitted. The sole registered totalization exception is
`wrong_given_accept = accepted_wrong/max(accepted_positives,1)`; mandatory
coverage and identity gates prevent abstention from passing.

### 8.3 Factorial signs and intervals

For lower-is-better error `E`:

```text
B_L^E(s)=0.5*((E00-E10)+(E01-E11))
B_D^E(s)=0.5*((E00-E01)+(E10-E11))
I^E(s)=E10+E01-E00-E11.
```

For higher-is-better `Q`:

```text
B_L^Q(s)=0.5*((Q10-Q00)+(Q11-Q01))
B_D^Q(s)=0.5*((Q01-Q00)+(Q11-Q10))
I^Q(s)=Q11-Q10-Q01+Q00.
```

For seed vector `v`, use arithmetic mean and two-sided Student-t 95% CI with
`ddof=1`. Critical values are exactly `2.022690911734728` for validation `n=40`
and `2.0009953780882674` for test `n=60`. A benefit win is strictly `>0`; zero
is a tie. Relative reductions are ratios of cell means, never means of seed
ratios. Every denominator must be finite and strictly positive or the gate
hard-fails. Every seed/cell value, denominator, mean, SD, CI, strict win, and tie
count is serialized.

## 9. Frozen all-of gates

Validation and, if unlocked, test must independently pass every item.

### 9.1 Prediction and integration

```text
RR_L = 1-(mean(E_all10,20)+mean(E_all11,20))
          /(mean(E_all00,20)+mean(E_all01,20)) >= 0.10
lower95(B_L^E_all)>0; strict-win fraction(B_L^E_all)>=0.65.

For both 00->01 and 10->11 independently:
1-mean(E_uv,dream,20)/mean(E_uv,no_dream,20) >= 0.10,
lower95(E_uv,no_dream-E_uv,dream)>0,
strict-win fraction>=0.65.

RR_joint=1-mean(E_all11,20)/mean(E_all00,20) >= 0.10,
lower95(E_all00-E_all11)>0; strict-win fraction>=0.65.
```

Absolute adequacy requires `mean(E_all11,20)<=1.00`,
`mean(E_uv01,20)<=1.00`, `mean(E_uv11,20)<=1.00`, and
`mean(E_all11,20)/mean(E_all11,5)<=2.00`. M11 must reduce H20 error versus
persistence by at least 10%, with paired lower95 benefit `>0` and strict win
fraction `>=0.65`.

Report the two simple LTM effects `E_all00-E_all10` and
`E_all01-E_all11`, the marginal dream effect, and all joint cell means even
when they are not additional gates.

### 9.2 Planning

```text
RR_regret=1-mean(R11)/mean(R00) >= 0.20
lower95(R00-R11)>0
mean(S11-S00)>=0.10
lower95(S11-S00)>0
mean(S11)>=0.75.
```

`mean(R00)` must be finite and positive. M11 invalid selected-action rate must
be exactly zero.

### 9.3 Absolute component and no-antagonism gates

For both M10 and M11: mean coverage and identity accuracy `>=0.80`; mean
`wrong_all` and `wrong_given_accept <=0.05`; mean false lure `<=0.05` and
`upper95(false_lure)<=0.05`; cross-port accepts `=0`. Report, but do not hide,
the maximum seed lure rate.

For both M01 and M11: valid missing-binding coverage `>=0.80`, accepted invalid
splice count `=0`, observed overwrite count `=0`, and the absolute E_uv cap in
9.1. An always-abstaining component therefore cannot pass.

No antagonism requires

```text
upper95(E_recall11-1.02*E_recall10)<=0
upper95(E_uv11,20-1.02*E_uv01,20)<=0.
```

The 2% margin is relative. All benefit-oriented interactions for E_all, E_uv,
regret, and success are reported regardless of sign. No synergy claim is
registered, so no outcome may be described generically as synergy.

### 9.4 Stability and constraints

All outputs and metrics are finite; `max(abs(predicted_state))<=2`;
`max_seed invalid_predicted_transition_rate<=0.01`; constrained accepted-splice
violations are zero; M11 invalid selected-action count is zero; H5/H20 identity
is bit-exact. Any hard-zero integrity count overrides performance.

## 10. Train-only calibration

Before opening train, freeze raw preregistration bytes and SHAs for generator,
model, runner, tests, inherited G7-M V2 and V1 dependency, and codec stripping
functions. Then run train seeds exactly once.

The sole calibration artifact contains:

1. the 20 shared core coefficients and core SHA;
2. state `mu_x,sigma_x` and 96-coordinate codec mean/population scale with floor
   `1e-8`;
3. one `scoped-bank-96` hard-recall threshold selected through the exact
   evaluation wrapper. The physical bank has 96 records, while each `(context,
   component)` scope has exactly 12 storage-order-preserving candidates. The
   eight scope views are prebuilt zero-copy index views. A cross-port/invalid
   scope rejects before any view or distance call. Candidates are sorted unique train
   positive/lure initial confidences plus the symbolic `REJECT_ALL`; accept iff
   `confidence>tau`; among candidates with pooled false lure and wrong_all each
   `<=0.025`, lexicographically maximize correct identity accuracy with
   abstentions wrong, minimize lure acceptance, then choose largest tau;
4. one join threshold equal to the `0.99` NumPy linear quantile of all train
   real observed prefix-to-connector and connector-to-suffix standardized RMS
   joins computed only in residual columns `0:4` by the exact V2 adapter. For
   every observed binding mean `R`, these are
   `sqrt(mean((R[4]-R[5])^2))` and `sqrt(mean((R[6]-R[7])^2))`;
5. exact pools, tie decisions, arrays, source SHAs, raw registration SHA, and
   the calibration artifact's canonical SHA.

There is no pre-48 threshold, rollout blend, dream weight, planner tuning,
success tuning, route selection, or cell-specific calibration. Unit dream
weight, one pass, state scale floor `0.05`, `lambda=0.02`, success threshold
`25`, and invalid penalty `10000` are preregistered literals. If threshold
feasibility fails, preserve the calibration failure and do not open validation;
changing a literal or algorithm requires V3 and fresh seeds.

Calibration asserts the exact populations `46,080` core transitions,
`49,920=40*96*13` state rows, `3,840=40*96` codec trajectories, `960=40*24`
positive recall queries, `960` lure queries, and `3,840=40*48*2`
four-dimensional residual endpoint join values. Every recall calibration item
passes through the scoped wrapper. Empty/nonfinite pools, count mismatch,
nonfinite output, or a non-unique selector hard-fails.

The selector's `+infinity` option is a symbolic `REJECT_ALL` candidate, not a
member of either finite confidence pool and never a persisted numeric value. If
`REJECT_ALL` would win, calibration is infeasible and validation remains closed.
The operational threshold written to calibration must be finite.

## 11. Hard provenance and leakage gates

The exact semantic tuples are:

| object | source | epistemic status | observed | recalled | episode ID |
|---|---|---|---:|---:|---|
| wake LTM record | real | observed | true | false | unique ledger ID |
| accepted recall | real | recalled | false | true | existing ledger ID |
| dream/schema completion | synthetic | hypothetical | false | false | null |
| fallback | schema_fallback | inferred | false | false | null |

Each of the following integer counts must equal zero in every seed and cell:

```text
synthetic_with_episode_id
synthetic_tagged_observed
synthetic_tagged_recalled
synthetic_to_ltm_insert_attempts
synthetic_to_ltm_successful_inserts
nonledger_real_record_in_ltm
observed_record_overwrite_or_hash_change
cross_context_or_cross_port_or_cross_component_accepted_splice
accepted_context_component_phase_action_or_key_constraint_violation
accepted_cross_context_recall
heldout_future_reads
evaluator_latent_or_truth_reads
generator_validity_or_outcome_reads
masked_cue_coordinate_reads
test_path_reads_before_unlock
cell_cross_write_or_shared_mutation
nonfinite_outputs.
```

This displayed order is the canonical `(17,) int64` serialization order.

Counts are capability-instrumented/taint-derived, not candidate self-reports.
Required handcrafted pre-registration tests, using no registered or development
seed, must show:

- independently poison futures, true q/schema, outcomes/costs, validity labels,
  masked cue cells, and seed/stream metadata while holding candidate inputs;
  candidate bytes and selected index remain bit-identical;
- all 24 cell execution orders give bit-identical per-cell bytes;
- world, prefix, actions, goals, masks, wake-ledger, core, and observed-schema
  hashes are equal across cells;
- H5 equals the H20 prefix bit-for-bit;
- candidate presentation permutation maps back to the same canonical selection;
- pre-unlock reads of the test artifact/path are denied;
- synthetic insertion attempts and successful inserts are separately counted;
- rejected recall's zero reconstruction cannot influence fallback output.

## 12. Numeric common-budget ledger

Closed-form core fitting occurs once in train: `46,080` real wake transitions,
four float64 ridge solves, 20 fitted parameters. Validation/test load the
byte-identical core and perform zero updates.

For every registered seed, every factorial cell, and every non-oracle control,
require the exact vector

```text
b_registered = (
  N_wake_records=96,
  N_wake_transitions=1152,
  U_core=0,
  P_core=20,
  N_origins=24,
  K=8,
  H=20,
  N_rollout_calls=192,
  N_predicted_transitions=3840,
  N_planner_score_calls=192,
  B_ltm_trace_bytes=73728,
  Q_ltm_call_slots=72,
  N_scoped_distance_rows=576,
  N_schema_key_slots=72,
  N_ordered_pair_enumerations=288,
  N_component_port_checks=288,
  N_same_component_pairs=72,
  N_observed_keys=48,
  N_join_candidates=24,
  N_scalar_endpoint_join_values=48,
  N_dream_output_slots=24,
  U_dream_update_slots=24,
  N_lesion_nonobserved_pairs=240,
  N_lesion_accepted_slots=24,
  N_lesion_capacity_padding=216,
  N_dream_passes=1,
  persistent_numeric_payload_bytes=393216,
  persistent_byte_cap=524288,
  temporary_workspace_byte_cap=1048576).
```

The 72 LTM call slots are 24 positives, 24 unstored lures, and 24 cross-port
diagnostics. Primary recall is cached once per origin and shared across that
origin's eight rollouts. Positive and lure calls each examine exactly 12 scoped
rows (`48*12=576` distance rows); cross-port calls reject before any view or
distance access. Inactive LTM cells allocate an unqueryable 73,728-byte buffer
and execute identical shadow facade/distance slots whose results are discarded.
Inactive dream cells allocate identical schema/pair/join/output arrays, execute
288 shadow enumerations/checks, 48 shadow endpoint values, and 24 shadow update
slots, and discard writes. Every candidate is rolled for 20 steps even when
inferred invalid. No early stopping or confidence-dependent extra work is
allowed.

Before train, each condition preallocates the following ordered persistent
numeric/index/mask ledger. All shapes are C-contiguous and views are zero-copy:

| owned semantic array | shape / dtype | bytes |
|---|---:|---:|
| LTM trace payload | `(96,12,8)` float64 | 73,728 |
| LTM occupancy / provenance | `(96)` bool8 + `(96)` uint8 | 192 |
| eight scoped storage-index views | `(8,12)` int16 | 192 |
| schema payload | `(72,12,8)` float64 | 55,296 |
| schema occupancy / provenance | `(72)` bool8 + `(72)` uint8 | 144 |
| all candidate predictions | `(24,8,20,4)` float64 | 122,880 |
| inferred-valid flags | `(24,8,20)` bool8 | 3,840 |
| resolved schema-key indices | `(24,8,20)` int16 | 7,680 |
| schema-source codes | `(24,8,20)` uint8 | 3,840 |
| per-origin `q_hat` | `(24,4)` float64 | 768 |
| inferred candidate costs | `(24,8)` float64 | 1,536 |
| selected canonical indices | `(24)` int64 | 192 |
| learned core | `(20)` float64 | 160 |
| codec mean and scale | `2*(96)` float64 | 1,536 |
| state mean and scale | `2*(4)` float64 | 64 |
| completed positive cue views | `(24,12,8)` float64 | 18,432 |
| dream residual output buffer | `(24,12,4)` float64 | 9,216 |
| lesion residual audit buffer | `(24,12,4)` float64 | 9,216 |
| dream/lesion occupancy+provenance | `4*(24)` bool8/uint8 | 96 |
| recall accepted/identity/confidence/scope | `(72)` bool8, int16, float64, uint8 | 864 |
| ordered pair indices | `(288,2)` int16 | 1,152 |
| pair check flags and reason codes | `2*(288)` bool8/uint8 | 576 |
| endpoint join values | `(24,2)` float64 | 384 |
| actual budget vector | `(29,)` int64 | 232 |
| registered budget vector | `(29,)` int64 | 232 |
| hard provenance/leak counts | `(17)` int64 | 136 |
| fixed inactive/padding payload | `(80632,)` uint8 | 80,632 |
| **exact persistent numeric payload** |  | **393,216** |

The ledger's shape/dtype/name order and SHA are frozen in the implementation
lock. Immutable generator inputs may be excluded only when every condition gets
the same read-only object by reference; any copy is charged to its owner. Python
container/class overhead is excluded uniformly, while every numeric, index,
mask, audit, inactive, and padding payload above is counted. Controls run in
separate identical condition allocations rather than borrowing a treatment
cell's state.

Serialize actual integer counts, exact owned-array bytes and ordered-ledger SHA,
canonical input/core hashes, and the vector for every condition. Require exact
equality to `b_registered`; overage, allocation-copy, or inequality is a hard
resource failure. CPU only,
NumPy only, one process, no network/download, no GPU, no external trajectory
file, and target wall time 120 seconds per complete split are resource gates;
wall time is reported but is not the equality proof.

Byte caps count C-contiguous NumPy payloads by `array.nbytes`; Python interpreter
and immutable class metadata are excluded as stated above. Temporary arrays are
measured by the same owner method and must remain at or below 1,048,576 bytes.
Canonical serialized token/audit metadata has a separate `32,768`-byte UTF-8 cap
in every condition.

## 13. Fresh splits, locks, and state machine

Envelope and task-feasibility assertions run split-locally only when that split
is legally opened: train inside the one train/calibration opening, validation
inside its one registered run, and test only after unlock. No validation/test
origin, future, cost, or seed is generated or read during the pre-train lock or
train opening. A split failure hard-fails that opened split without resampling
and never authorizes the next split. Handcrafted off-range fixtures may test the
assertion code but may not open a registered seed.

Registered seeds are exactly:

```text
train/calibration: 86100..86139 (40)
validation:        87100..87139 (40), exactly one registered run
locked test:       88100..88159 (60), exactly one run after unlock
```

These ranges are disjoint from V1, G7-M, and G9-CB registered ranges. Tests and
development must use handcrafted fixtures or seeds outside all registered
ranges. This drafting phase used neither.

Exact future lock paths are:

```text
experiments/preregistration/agi_world_memory_integration_v2.json
reality_stone/python/reality_stone/clarus/agi_world_memory_integration_v2.py
examples/agi/agi_world_memory_integration_v2_gate.py
tests/test_agi_world_memory_integration_v2.py
tests/test_agi_world_memory_integration_integrity_v2.py
artifacts/agi/agi_world_memory_integration_implementation_lock_v2.json
artifacts/agi/agi_world_memory_integration_train_calibration_v2.json
artifacts/agi/agi_world_memory_integration_validation_v2.json
artifacts/agi/agi_world_memory_integration_test_v2.json
artifacts/agi/agi_world_memory_integration_integrity_v2.json
```

Implementation lock precedes train. Train calibration is written once.
Validation must reference exact raw-registration, implementation-lock, core,
calibration, inherited-module, runner, and tests SHAs. Test remains inaccessible
until a committed validation artifact passes every performance, safety,
integrity, control, and resource gate. There is no separate mutable unlock file.
At test entry, the runner requires the validation path to be Git-tracked, clean
at that path, and byte-identical to `git show HEAD:<validation-path>`; it reloads
the artifact, recomputes `performance_passed=all(checks)`,
`resource_passed=all(resource_checks)`, their conjunction, and every dependency
SHA. Only then does it construct an in-memory `UnlockRecordV2` containing the
validation raw SHA, registration/implementation/calibration/dependency SHAs, and
`test_unlocked=true`. The test artifact serializes that exact record and names
the validation raw SHA. Test reuses calibration bytes exactly. A validation FAIL,
untracked/dirty/tampered validation path, or inconsistent pass flag is preserved
and test remains unopened.

Any post-lock change to generator equations or streams, codec/mask, actions or
order, cost/success, model, controls, thresholds, metrics/gates, budgets,
provenance, chronology, seed ranges, or access rules creates V3 with fresh
seeds. Audit-only tests may become post-unlock state-aware only if no scientific
file/artifact changes; the change and both hashes must be added to integrity.

## 14. Resolution matrix and stop conditions

| audit item | V2 resolution |
|---|---|
| A1 / P0-1 | C1 is the exact marginal LTM estimand; M11/M00 is separately named `RR_joint`. |
| A2 / P0-3 | E_all, E_uv, U_s, H5 slice, normalizers, denominators, and nonfinite rules are explicit. |
| A3 / P0-2 | d=4, m=2, do-equation, common noise, K/order, goal, cost, regret, success, lures, and tie rule are fixed. |
| A4 / P0-4 | Append-only chronology, minimal API, evaluator-after-hash, taint/poison, cell/order, and pre-unlock tests are hard gates. |
| A5 / P0-5 | Exact provenance tuples and attempts/successes are separate hard-zero integer invariants. |
| A6 / P0-6 | Recall/dream absolute gates and the only train-derived thresholds have unique algorithms; all other values are literals. |
| A7 / P0-7 | Every capacity, call, update, byte cap, inactive padding rule, and equality hash is numeric. |
| A8 / P0-8 | Benefit signs, ratio-of-means, positive denominators, t values, strict wins/ties, relative 2%, and interaction language are frozen. |
| A9 / P1 | Shuffled/lesion decisions, absolute error caps, persistence comparator, lure upper CI/max report, and cross-context recall zero are included. |
| A10 | Literature is motivational only. In particular, Schapiro et al. supports the reported shared-property/unique-property findings, not a general dream or consolidation mechanism. No biological claim enters a gate. |

The final mathematical revision additionally freezes: all four fitted
intercepts; the raw/standardized codec boundary; residual-only four-coordinate
dream joining; the exact 12-transition evaluation prefix; a wake-only typed
action index and typed candidate return; deterministic `240 -> 24` lesion
selection; numeric denominator assertions; scoped 12-row recall calibration;
all calibration population counts; and an exact 393,216-byte allocation ledger
under a 524,288-byte cap. The dimensionless audit is
`revisions/13-dimensionless-v2.md` and is a symbolic/unit-consistency gate, not
empirical support.

Stop before train on any unresolved implementation ambiguity, inherited-boundary
hash/equivalence failure, poison/cell-order failure, unequal budget, or missing
lock. Stop before validation if calibration is infeasible or any integrity gate
fails. Stop after validation FAIL and do not open test. Preserve every failure;
do not retune, drop seeds, average ratios, substitute epsilons, relabel controls,
or weaken a hard-zero invariant.
