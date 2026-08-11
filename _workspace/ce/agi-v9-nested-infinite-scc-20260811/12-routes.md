# V9 nested recurrent-tower route audit

Status: COMPLETE

## 1. Route decision

Four distinct constructions are admissible, but only one should receive the
first development-only test.

| Rank | Route | Mathematical object | Development role | Verdict |
|---:|---|---|---|---|
| 1 | C: lazy generated adaptive-depth recurrent tower | a finite, causally selected prefix of a generated nested SCC tower | primary V9 development candidate | **SELECT** |
| 2 | A: nested direct-limit predictive-state tower | inclusions `G_n -> G_(n+1)` and an ideal direct-limit dynamics | theorem target and generator contract | formal parent, not first empirical arm |
| 3 | B: inverse/projective multiresolution quotient | surjections `Z_(n+1) -> Z_n` and a compatible inverse-limit state | later compression route | defer until lumpability is demonstrated |
| 4 | D: SCC wrapper around V8 or ACBSM | old output with unused recurrent metadata/state | mandatory negative control | **REJECT as a mechanism** |

The primary route is C because it is finite, causal, measurable, and capable of
failing. It realizes generated prefixes of A without pretending to instantiate
the infinite direct limit. It is genuinely new relative to V5--V8 and ACBSM
only if decisions read the tower state, multiple levels retain history, and
real level/cross-scale interventions change held-out predictions. Until all of
those checks pass, it remains a `THEORY/DESIGN` candidate rather than an
empirical V9 result.

No confirmatory V9 registration, seed block, or execution is authorized by this
route report. The only next empirical object would be one separately hashed
**development-only mechanism registration**. It must not be named or reported
as V9 confirmation.

## 2. What V1--V8 and ACBSM actually leave open

The tower must solve a state-representation problem rather than rename a closed
readout family.

| Checkpoint | Preserved result | Constraint on the tower |
|---|---|---|
| V1 | programmed graph was recovered, but prediction and lesion gates failed; one control was defective | graph recovery or an SCC diagram is not predictive evidence |
| V2 | passed only in a narrowed, no-target-confounding, ideal paired-do family; same-probe dense was essentially tied | sparse support must be compared with equal-information dense controls |
| V3 | rank-one loading was recovered, but per-seed scalar AR estimation was unstable | do not fit a free tower timescale separately on each evaluation episode |
| V4 | pooled scalar residual AR passed a true-state-reentry one-step gate | V9 must be judged on prefix-only closed H20 rollout, not one-step reentry |
| V5 | first genuine prefix-only H20 rollout was finite but failed four robustness/comparator clauses | frozen V5 is the primary failed parent and an obligatory comparator |
| V6 | registration and pilot disclosure only; no implementation/result | V6 supplies no empirical evidence and cannot be promoted as a parent |
| V7 | sparse contribution survived, but parent repair, persistence, and stability failed; test stayed closed | retain prefix/future-read/stability discipline and the real no-sparse control |
| V8 | output shrinkage passed every clause except reliable superiority to unshrunk V5; structured gains, prefix gains, and multi-origin blends also collapsed to the same solution | no `P+g(S-P)`, parent-path interpolation, horizon gain, or output ensemble may enter C |
| ACBSM | rank-one posterior observer improved eight training folds on average, but its interval crossed zero; rank two collapsed; fresh `82100..82355` remained unopened | rank-one ACBSM is a comparator, not evidence; the tower must add causal scale structure rather than another unsupported residual mode |
| unified breakthrough map | correctly moved the target from output shrinkage to internal predictive state, but rejected more than two independent residual modes | C continues that direction with generated nested recurrent state, adaptive truncation, and bidirectional cross-scale messages; shells may not be separately fitted AR modes |
| Indra orbit quotient/cone | exact fixed three-orbit closure, delayed local-cone reconstruction, certificates, and one frozen synthetic sidecar earned a narrow GO | this is a valid positive construction for fixed equivariant/equitable kernels, not evidence for learned multiresolution quotients, arbitrary graphs, the V1--V8 H20 task, or AGI |

The surviving shared asset is the frozen sparse structural mechanism and its
training-only normalization. It may be supplied as an identical parameterized
primitive to every matched arm. The candidate may not consume the V5 forecast,
the V8 blended forecast, an ACBSM posterior, a hidden simulator state, or a
future target as an input.

## 3. Common causal stream and information boundary

The general online event is

```text
e_t = (
    observation_t,
    previous_action_t?,
    arrived_reward_t?,
    terminal_t,
    availability_mask_t,
    causal_tick_t,
)
```

Only fields available at `causal_tick_t` may be read. Actions and rewards are
optional typed channels. In the inherited four-chart forecasting family they
are absent/masked for every arm; no candidate gets an extra input because the
general API supports them.

Every dimensional value is divided by a named positive training-only reference
scale. The normalizer is frozen before development. Evaluation-prefix statistics
may update a declared causal state, but may not redefine a scale using the H20
target.

For the historical benchmark, every arm receives exactly `x[0]..x[80]`, makes
one uninterrupted H20 rollout, and never rereads `x[81]..x[100]`. H5 is the
first five rows of that same H20 object. The hidden simulator state, OOD loading
label, future noise, and oracle chart target are scorer-only fields.

## 4. Route A -- nested direct-limit SCC predictive-state tower

### 4.1 Graph and state construction

Start with a finite nonempty strongly connected shell `S_0`. At level `n+1`,
add a new finite shell `S_(n+1)` that is internally strongly connected and add
at least one path from the old graph to the new shell and one return path. Set

```text
V_n = union_(k=0)^n S_k,
E_n = all within-shell and registered cross-shell edges up to n.
```

Then `G_n=(V_n,E_n)` is strongly connected, `G_n` embeds in `G_(n+1)`, and the
direct union is one SCC. `G_n` is a maximal SCC only in its level-`n` graph
view. It is a proper strongly connected subgraph, not a second maximal SCC,
inside a larger level.

Let `X_n` be the normalized level state and `F_n:X_n->X_n`. Exact direct-limit
dynamics require state embeddings `J_n` satisfying

```text
J_n(F_n(x)) = F_(n+1)(J_n(x))
```

on the declared invariant image. Readouts must also be compatible or converge
under a separately stated truncation bound. Strong connectivity supplies none
of these conditions.

A finite parameter generator may return a shell, edge, operator block, initial
state, or message map at `(level,index)` without constructing every level. The
generator is a representation of the tower; it is not by itself a proof that
`F_n` is compatible or that a limit update exists.

### 4.2 Exact data path

At a finite prefix depth `d`, causal events enter shell zero. Previous-tick
messages propagate upward and downward through registered bridges:

```text
u_t^0 = encode(e_t)
u_t^(k+1) = Up_k(h_t^k)
d_t^k = Down_k(h_t^(k+1))

h_(t+1)^k = phi_k(
    A_k h_t^k
    + B_k u_t^k
    + C_k d_t^k
)
```

All right-hand sides read tick `t`; there is no instantaneous algebraic loop.
At the top finite boundary, the absent downward message is a declared boundary
condition. A forecast or policy reads only the state token produced after this
update:

```text
y_t = Readout_d(h_t^0, ..., h_t^d).
```

Raw input, an analytic posterior, persistence, V5, V8, and ACBSM outputs have no
readout bypass.

### 4.3 Scaling, choices, and failure modes

For shell width `m`, `d+1` active shells, `s` within-shell nonzeros per shell,
and cross-message rank `r`, a sparse step costs

```text
O((d+1) * (s + 2mr)) MACs,
O((d+1)m) live state,
O((d+1)(s+mr)) explicit coefficients,
```

or a fixed generator-parameter cost if coefficients are shared/generated. A
dense shell changes the within-shell term to `m^2`. SCC auditing of a generated
prefix is linear in its generated vertices and edges. A block-gain spectral or
resolvent certificate can cost `O(d^3)` for a dense `(d+1)x(d+1)` gain matrix.
The infinite union is never materialized.

Route A chooses the shell width schedule, within-shell topology, bridge pattern,
message rank, generator family, embedding, boundary condition, nonlinearity,
update schedule, norm, gain cap, and compatible readout: at least 12 structural
degrees of freedom before learned coefficients. Varying the number of levels or
timescale schedule after reading a development gap is a route change.

Route A fails as a useful computational model if embeddings are incompatible,
readouts do not converge, uniform gains are unavailable, finite-prefix error
does not shrink, upper shells are output-inert, or the tower stabilizes after a
finite level while being advertised as properly infinite. Oscillatory or
divergent strongly connected shells are complete counterexamples to the claim
that topology provides stability.

### 4.4 Why A is not the first empirical route

An ideal direct limit is not physically executable, and exact inclusion
compatibility can make new levels too restrictive to add a predictive effect.
Testing a hand-picked finite depth would not validate infinity. Route A should
therefore provide the generator, nesting, and truncation contract for C, while
C carries the falsifiable finite mechanism claim.

## 5. Route B -- inverse/projective multiresolution quotient

### 5.1 Construction

Route B does not add subgraphs. It represents the same dynamics at progressively
finer resolutions with surjections

```text
pi_(n+1,n): Z_(n+1) -> Z_n
```

and demands exact quotient compatibility

```text
pi_(n+1,n)(F_(n+1)(z)) = F_n(pi_(n+1,n)(z)).
```

The inverse-limit state is a compatible sequence

```text
(z_0,z_1,...) with pi_(n+1,n)(z_(n+1)) = z_n.
```

SCCs are computed in each scale's coordinate-dependency graph. They are not
nested maximal SCCs of one graph, and the inverse limit is not the direct union
from Route A.

There is a narrow positive local precedent: the existing Indra sidecar proved
exact closure for a fixed three-orbit equivariant/equitable kernel, reproduced
delayed homogeneous covers and a finite local causal cone to numerical
precision, and passed its frozen synthetic compression/parity gate. That result
is retained as a positive-control fixture for exact quotient code. It does not
identify a hierarchy of learned predictive quotients and cannot be imported as
an H20 or V9 performance result.

### 5.2 Causal data path

A causal encoder forms the finest currently available state from the event
prefix. Locked projection maps produce coarser states. Each scale updates its
own recurrent state; consistency residuals measure commutation:

```text
z_t^D = Encoder_D(e_0:t)
z_t^k = pi_(k+1,k)(z_t^(k+1))
z_(t+1)^k = F_k(z_t^k)
r_t^k = ||pi_(k+1,k)(z_(t+1)^(k+1)) - z_(t+1)^k||.
```

The policy reads a registered compatible aggregate of the projective state.
If exact readout compatibility is imposed as
`R_(k+1)=R_k o pi_(k+1,k)`, fine detail is output-inert by definition. If fine
detail may change output, convergence of the scale-dependent readouts must be
proved or bounded separately.

### 5.3 Scaling, choices, predictions, and STOP conditions

At finite depth `D`, state and update cost are the sums of scale dimensions and
operator costs. A dense projection from dimension `m_(k+1)` to `m_k` costs
`O(m_k m_(k+1))`; sparse aggregation may be linear in its registered support.
Exact lumpability tests on a fully enumerated finite transition system can be
far larger than the original state-variable graph and may be exponential in
the number of variables.

Choices include scale count, node partitions, projection maps, scale-specific
dynamics, commutation norm/tolerance, encoder, finest available resolution,
readout compatibility, and aggregation: at least nine structural degrees of
freedom before fitting. Choosing a parcellation because it makes commutation
look good is target-aware.

Route B predicts small held-out commutation residuals, indistinguishable macro
futures for microstates in one quotient class, and lesion effects localized to
the information removed by a projection. It is stopped if exact lumpability has
a counterexample, approximate residuals accumulate beyond their locked bound,
fine-scale state is either readout-inert or acts through an undeclared bypass,
or a same-dimension flat encoder matches it. Because current V1--V8 data do not
identify multiresolution lumpability, B is deferred.

## 6. Route C -- lazy generated adaptive-depth recurrent controller

### 6.1 Primary mechanism

Route C fixes a maximum finite development depth `D_max` but stores a generator
that can describe the next shell. At each causal tick it runs only an active
prefix `G_d`, where `0 <= d <= D_max`. Every prefix is audited as strongly
connected using the construction in Route A.

The candidate uses a shared, level-conditioned recurrent cell and registered
up/down message maps. One valid implementation family is

```text
p_t^0 = Encode(normalize(e_t))
p_t^(k+1) = U_k h_t^k
q_t^k = D_k h_t^(k+1)                  # zero only at active boundary

h_(t+1)^k = tanh(
    W_k h_t^k + B_k p_t^k + C_k q_t^k + b_k
),  k=0..d,
```

using previous-tick/Jacobi reads. `W_k`, `U_k`, `D_k`, and topology are returned
by the finite generator. Delay buffers become part of the state if a different
schedule is used; a Jacobi gain certificate may not be reused for Gauss--Seidel
or multirate updates.

The first candidate must use weight-tied or finitely generated shell operators.
It may not fit one independent residual pole/loading per shell: the unified map
already rejected more than two independent residual modes, and ACBSM's second
mode collapsed under its frozen stability rule. Tower depth is generated
recurrent computation over shared parameters, not a license to reopen a
three-, four-, or many-mode AR search.

The depth decision uses no target or hidden state. It queries only the next
generated boundary block and a registered truncation certificate. There are two
admissible certificate forms.

First, prove a uniform defect on the declared forward-invariant reachable
domain `R_d` in the common weighted metric:

```text
eta_bar_d >= sup_(x in R_d)
    ||F_(d+1)(J_d x) - J_d F_d(x)||_w
B_bar_d = eta_bar_d / (1-q_d).
```

Then `B_bar_d` is the geometric/resolvent truncation bound. Alternatively,
measure the defect only along the realized lower-level trajectory but propagate
a certified online error envelope from a registered initial bound:

```text
eta_(d,t) = ||F_(d+1)(J_d h_t^d) - J_d F_d(h_t^d)||_w
E_(d,t+1) <= q_d E_(d,t) + eta_(d,t).
```

Activation may be conservative, but deactivation requires `H` consecutive
ticks for which the applicable `B_bar_d` or recursively propagated `E_(d,t)` is
at most `epsilon_depth`. A defect sampled at only the current state is a
diagnostic; by itself it cannot be divided by `1-q_d`, authorize depth
deactivation, or support route promotion.

`q_d<1`, `epsilon_depth`, hysteresis `H`, the invariant/reachable domain,
initial online bound, boundary state, and extra boundary-probe MACs are locked.
If the bound is not valid for the actual update schedule, the controller is
marked uncertified and cannot pass the development gate.

### 6.2 Exclusive state-to-output path

The causal path is intentionally narrow:

```text
raw event available at t
    -> frozen training-only normalizer
    -> shell-0 encoder
    -> generated recurrent prefix update
    -> cross-scale previous-tick messages
    -> immutable TowerStateToken(t, active_depth, state_hash)
    -> forecast/policy readout(TowerStateToken, action_mask?)
    -> action or next-state prediction
```

The readout function accepts no observation, reward, hidden state, posterior,
parent forecast, persistence path, or completed trajectory. An action mask is
permitted only as a feasibility mask and is given identically to all policies.
For forecasting, the H20 rollout repeatedly advances the tower with a declared
`observation_missing` mask and its own previous prediction; no true future
observation reenters.

The frozen sparse mechanism may be compiled into the generated shell-zero
operator or provided as an identical fixed parameter block to all structural
arms. It may not produce a parallel V5 path that is later blended at the output.
Poisoning any cached V5/V8/ACBSM prediction must be exactly inert because no such
value is in the candidate API.

### 6.3 Why C is genuinely new, conditionally

C differs from V8 because it changes recurrent state evolution before readout;
there is no output gain or path interpolation. It differs from ACBSM because it
does not assume a fitted rank-one/two Gaussian residual posterior as its state.
It maintains multiple recurrent scale shells, reciprocal cross-scale messages,
and a causal computation-depth decision based on a truncation defect.

Those architectural differences do not establish a mechanism. The phrase
"genuinely new" is earned only if all of these occur on the one fresh
development block:

1. same-current-input/different-history pairs produce different tower states
   and outputs;
2. resetting or swapping the state removes or transfers that history effect;
3. every retained upper level and both cross-scale directions have a measurable
   lesion effect;
4. adaptive depth is nondegenerate and beats a compute-matched fixed-depth
   controller;
5. the candidate beats V5, ACBSM rank one, a finite flat recurrent control, and
   a matched monolithic SCC at preregistered floors.

If any clause fails, C is a new code shape at most, not a new causal mechanism.

### 6.4 Scaling

Let active depth be `d`, shell width `m`, within-shell nonzeros `s`, message rank
`r`, and output width `o`. The registered implementation must count rather than
estimate:

```text
live state scalars       = (d+1)m + delay/controller state,
recurrent MACs/tick      = (d+1)(s + 2mr),
readout MACs/tick        = O((d+1)mo),
boundary-probe MACs      = one generated shell/update block when queried,
topology audit           = O(|V_d|+|E_d|),
gain certificate         = O((d+1)^3) conservatively.
```

With level-shared generator parameters, parameter count can remain finite while
state/MAC grows with queried depth. The implementation manifest reports both
generator parameters and every active generated coefficient. It may not call
an infinite representation `O(1)` while hiding linear query work.

## 7. Route D -- cosmetic SCC wrapper negative control

Route D deliberately constructs a recurrent tower and then bypasses it:

```text
dummy_state_t = Tower(dummy_state_(t-1), event_t)

output_D1 = P + g(S-P)                  # frozen V8 R1
output_D2 = ACBSM_rank1(prefix)          # frozen ACBSM path
```

The wrapper may emit SCC counts, direct-limit diagrams, state norms, and
contraction certificates. None of those values enters the output. Real state
reset, cross-scale cut, time shift, sign flip, and shuffle must therefore leave
the output bitwise identical to the parent.

This is the mandatory topology-only negative control. It demonstrates that a
strongly connected tower can be causally irrelevant. D is allowed to execute
the same state updates as C so its state and MAC overhead are visible, but dummy
parameters and padded operations are not counted as active capacity. D1 and D2
are intentional aliases to their named historical parents and are excluded
from claims about matched independent controls.

If C is bitwise or numerically identical to D, if C's lesions are inert like D's,
or if output-dependency tracing finds the parent bypass, V9 development stops
immediately regardless of RMSE.

## 8. Matched arm panel

All arms use the same raw prefix, training episodes, normalized metric, forecast
origin, H20 target, numerical precision, optimizer budget where learned, and
frozen sparse/dense mechanisms where applicable.

1. **C-intact:** adaptive-depth generated tower.
2. **V5 parent:** frozen prefix-only sparse H20 parent.
3. **ACBSM-rank1:** the preserved rank-one posterior observer, with no newly
   tuned output gain.
4. **flat finite-depth recurrent:** same live state and active parameter count as
   C but no cross-scale nesting; all recurrent state is in one timescale.
5. **matched monolithic SCC:** one strongly connected recurrent graph with the
   same maximum state, trainable coefficient count, input/readout family, and
   mean MAC budget.
6. **fixed-depth tower:** the same cells/messages as C at one training-selected
   fixed depth whose estimated mean MAC matches C within 1 percent.
7. **maximum-depth tower:** always uses `D_max`; it is an accuracy ceiling, not a
   compute-matched superiority target.
8. **upward-cut, downward-cut, one-tick-shift, sign-flip, level-reset, and
   state-shuffle lesions:** distinct executions, no returned-array aliases.
9. **zero-bridge and symmetric-dense tower controls:** identical tower observer
   with only the structural support changed, preserving the V2/V7 fairness
   requirement.
10. **D1/D2 cosmetic wrappers:** explicit topology-only negative controls.
11. **oracle diagnostic:** scorer-only and excluded from every promotion test.

The flat and monolithic controls are capacity-favored if an exact equality is
impossible: they receive no fewer trainable parameters, maximum live-state
scalars, or training steps than C. No untrained dummy parameter or no-op MAC is
counted toward equality.

The manifest records for every arm:

```text
trainable/nontrainable parameters,
active recurrent coefficients,
maximum and mean live state,
serialized bytes,
maximum and mean MACs per observed and rollout tick,
boundary probes,
optimizer updates,
peak memory,
latency distribution,
source/config hash.
```

Mean compute is paired per episode because C uses adaptive depth. Peak compute
is reported separately. The fixed-depth matched arm is selected from training
folds before the development block is opened.

## 9. Real lesions and mediation tests

### 9.1 No aliases

A lesion changes the tensor consumed by the next recurrent update, not a label
in the result table. All lesion arms construct independent state objects and
return arrays with distinct storage. The implementation must prove that:

- `level_reset(k)` zeros or restores the registered baseline state before the
  next update;
- `cut_up(k)` replaces only `U_k h^k` with zero while preserving call order;
- `cut_down(k)` replaces only `D_k h^(k+1)` with zero;
- `time_shift(k)` consumes the preregistered one-tick older message;
- `sign_flip(k)` multiplies the actual message tensor by `-1`;
- `state_shuffle(k)` uses a frozen cross-episode permutation within the same
  evaluation role and current-input stratum;
- no lesion changes raw information, parameter count, forecast target, or
  action availability.

Compute-preserving zero multiplication may be measured diagnostically, but it
does not turn a cut coefficient into an active coefficient.

### 9.2 Same-input/different-history construction

Before development, generate paired histories `(H_a,H_b)` that end in the same
normalized current event within an exact synthetic fixture or a locked
tolerance in the stochastic benchmark. Neither history may use a future target.
Let

```text
y_a = Readout(State(H_a), current_mask)
y_b = Readout(State(H_b), current_mask)

Delta_hist = ||y_a-y_b||_normalized.
```

Then run two interventions:

1. reset both states to the same registered neutral state before the common
   current input;
2. swap the full tower states between histories while holding the current input
   fixed.

Define donor transfer

```text
T_swap = <y_swap-y_a, y_b-y_a> / (||y_b-y_a||^2 + epsilon).
```

Development requires:

- mean `Delta_hist >= 0.02` and paired 95 percent LCB above zero;
- reset removes at least 80 percent of `Delta_hist`;
- mean `T_swap` lies in `[0.5,1.5]` with LCB above zero;
- the cosmetic D wrappers have zero tower-mediated effect to numerical
  tolerance;
- poisoning the event after a state token has been issued does not change the
  readout, while a registered state mutation does.

These tests establish finite state mediation in the synthetic model, not memory
or cognition in a brain.

### 9.3 Lesion effect floors

For normalized H20 RMSE, define positive intact benefit

```text
L_arm = mean_seed(RMSE_arm - RMSE_intact).
```

Every retained upper level must have individual reset `L_arm >= 0.002` with a
paired 95 percent LCB above zero. Each upward cut, downward cut, time shift,
sign flip, and state shuffle must have mean degradation at least `0.003` and LCB
above zero. The depth-one truncation must lose at least `0.005` with LCB above
zero. Slow/upper-level lesions must damage H6--H20 more than H1--H5 by at least
`0.002` in mean normalized RMSE; otherwise the proposed scale interpretation is
not supported.

A violent sign flip can degrade any recurrent network. Therefore a sign effect
alone is never sufficient; reset, directional cut, time shift, state shuffle,
and history transfer must all pass their independent clauses.

## 10. Development-only preregistration

### 10.1 Data roles

The development lock is created only after a complete repository seed-role
scan. It must record exact integer seeds and raw-role hashes.

- inherited observational training episodes may be used identically by all
  arms;
- historical V1--V8 validations and disclosed pilots are route-selection
  context only, not new evidence;
- V7 and V8 locked tests remain unopened;
- V8 test `81100..81355` is forbidden;
- the unused ACBSM block `82100..82355` remains reserved and unopened; it is not
  reassigned to the tower after ACBSM's HOLD;
- choose exactly 256 new, collision-free development episodes from a new raw
  role after the seed scan;
- do not allocate a V9 confirmation/test block in this run.

The 256-seed block is opened once after the development registration,
implementation, unit tests, comparison identities, configuration, and hashes
are frozen. After execution it is permanently development data. A failure does
not authorize another tower depth, threshold, or route on the same block.

### 10.2 Primary performance floors

For candidate `C` and baseline `B`, define per-seed improvement

```text
I_B = RMSE_B - RMSE_C
```

and use a preregistered paired Student-t 95 percent lower endpoint over the 256
independent seed summaries. The full development conjunction requires:

1. `LCB(I_V5) >= +0.005`;
2. `LCB(I_ACBSM_rank1) >= +0.003`;
3. `LCB(I_flat_finite) >= +0.003`;
4. `LCB(I_monolithic) >= +0.003`;
5. `LCB(I_fixed_depth) >= +0.002`;
6. positive LCB versus persistence and zero bridge;
7. symmetric-dense noninferiority: paired log-RMSE-ratio upper endpoint no
   greater than `log(1.02)`;
8. versus the always-maximum-depth tower, LCB at least `-0.002` and mean dynamic
   MAC reduction at least 20 percent;
9. all mediation and lesion floors in Section 9;
10. all stability, causal-integrity, capacity, and hash clauses below.

The `+0.005` V5 floor is inherited as the already disclosed route-selection
safety buffer. The other floors are fixed here before a tower result exists.
They may be made stricter, but not relaxed, in a later development registration.

### 10.3 Stability and integrity conjunction

The development candidate also requires:

- every generated active prefix has the declared nested topology and is one
  SCC;
- the actual scheduled finite update is a self-map and has a registered common
  contraction or equivalent Lyapunov certificate with factor at most `0.95`;
- every emitted truncation/residual bound is finite and encloses a
  high-precision deeper-prefix reference in unit/property tests;
- finite states and outputs, zero future/hidden reads, maximum true-state index
  80, one origin, exact H5/H20 identity, and no evaluation probes;
- the candidate API has no parent-output, analytic-posterior, or target field;
- stale tokens, duplicate rewards, future timestamps, unknown masks, and
  nonfinite inputs fail closed;
- at least two depths each occur on at least 10 percent of development episodes,
  and the deepest registered level occurs on at least 5 percent; otherwise
  adaptive depth has collapsed;
- source, configuration, registration, parent, normalizer, test, and result
  hashes match;
- no historical registration, implementation, result, or seed role changes.

One false conjunct yields `DEVELOPMENT STOP`. A partial score, attractive SCC
diagram, low average error, or passed direct-limit theorem cannot override it.

## 11. STOP rules and interpretation

Stop before any development run if:

- a readout bypass from raw input, V5/V8, ACBSM, persistence, or an analytic
  posterior is present;
- comparison arms share returned predictions or lesion arrays except the
  explicitly labeled D aliases;
- parameter/MAC/state manifests cannot be made comparable;
- the actual schedule lacks the registered stability certificate;
- seed-role collision or future-read poisoning is detected;
- a route/configuration was selected using the proposed fresh block.

Stop after the single development run if any Section 10 conjunct fails. In
particular, treat the following as valid negative outcomes:

- one level is always chosen;
- upper-level resets are inert;
- only a sign flip matters;
- the flat or monolithic recurrent control matches C;
- maximum depth gives no accuracy/compute tradeoff;
- current input explains the output after history is reset;
- the tower is a stateful but nonpredictive ornament;
- a finite level is sufficient and the claimed proper infinity adds nothing.

A full development pass authorizes only a separate status audit to consider a
future V9 confirmatory registration on a newly allocated block. It is not
itself V9 confirmation.

## 12. Look-elsewhere and degree-of-freedom audit

There are four disclosed route families. A, B, and C are not three models to
try on one development block. C is selected now; A supplies its formal
contract, B is deferred, and D is a mandatory negative control.

The primary C family has at least the following researcher choices:

1. maximum depth;
2. shell width or width schedule;
3. within-shell strongly connected mask;
4. cross-shell bridge pattern;
5. message rank;
6. level/timescale encoding;
7. generator/weight-sharing family;
8. activation and state domain;
9. Jacobi, sequential, or multirate schedule;
10. gain norm and certificate cap;
11. boundary condition;
12. depth residual and tolerance;
13. hysteresis/deactivation rule;
14. readout aggregation;
15. training objective and horizon weights;
16. regularization and optimizer budget;
17. matched-control mask construction;
18. history-pairing and shuffle strata.

This is at least 18 manual/categorical selection coordinates before learned
coefficients. Trying `a_i` values of each coordinate creates up to
`product_i a_i` configurations, not `sum_i a_i`. Training-only nested selection
may reduce the active set but does not erase the search. The development
registration must disclose every attempted configuration, crash, viewed fold
metric, and change.

Target-aware prohibited moves include increasing cross-message gain until cuts
matter, lowering the depth tolerance until multiple levels activate, deleting a
level whose lesion is weak, choosing history pairs after inspecting output
separation, selecting only long-horizon metrics after H1--H5 fails, or replacing
the monolithic control after it wins.

Because ACBSM already showed an unstable second mode in only four of eight
folds, the tower may not claim a discovered timescale merely from pole ordering.
Scale semantics require the preregistered horizon-selective lesion result.

## 13. Minimal typed API and isolated files

The proposed public surface is:

```python
spec = TowerSpec(
    shell_width=...,
    maximum_depth=...,
    message_rank=...,
    update_schedule="previous_tick_jacobi",
    normalized_reference_scales=...,
    contraction_cap=...,
    depth_error_tolerance=...,
    hysteresis_ticks=...,
)

generator = NestedTowerGenerator(spec, frozen_parameters)
audit = generator.audit_prefix(depth)       # nesting, SCC, edge manifest
certificate = generator.certify_prefix(depth)

controller = AdaptiveTowerController(generator)
controller.reset_episode()
token = controller.observe(CausalEvent(...))
decision = controller.read_policy(token, action_mask)
# or
prediction = controller.read_forecast(token)

snapshot = controller.state_dict()
controller.load_state_dict(snapshot)

lesioned = controller.with_intervention(
    LevelReset(k) | CutUp(k) | CutDown(k) |
    TimeShift(k, ticks=1) | SignFlip(k) | StateShuffle(permutation_hash)
)
```

`read_policy` and `read_forecast` accept a current immutable state token, not a
raw event. Tokens bind controller identity, tick, active depth, and state hash;
stale or foreign tokens fail closed.

Suggested files, if a later implementation phase is authorized:

- `reality_stone/python/reality_stone/clarus/nested_scc_tower.py`;
- `reality_stone/python/reality_stone/clarus/adaptive_scc_tower_controller.py`;
- `reality_stone/python/reality_stone/clarus/nested_scc_tower_development.py`;
- `examples/agi/nested_scc_tower_development.py`;
- `tests/test_nested_scc_tower.py`;
- `tests/test_adaptive_scc_tower_controller.py`;
- `tests/test_nested_scc_tower_development.py`;
- a development-only preregistration, implementation lock, seed-role manifest,
  and result under distinct `development` names.

Do not create `sparse_causal_bridge_v9.json`, a V9 validation artifact, or a V9
test artifact during this research/design run.

## 14. Minimum tests before any fresh development seed

1. Generate prefixes through a registered finite depth and verify injective
   graph nesting, strong connectivity at every level, and reciprocal-union
   maximality by an independent reachability implementation.
2. Verify that two nested maximal SCCs are never reported in one fixed graph.
3. Verify that a positive-delay forward event unroll is acyclic with singleton
   SCCs, while the recurrent template remains strongly connected.
4. Check generator determinism: identical `(level,index,parameter_hash)` queries
   return identical blocks without materializing deeper levels.
5. Verify exact compatibility on constructed fixtures and reject incompatible
   level maps whose direct-limit update is ill defined.
6. Test a strongly connected tower with divergent/oscillatory dynamics and
   ensure topology does not issue a stability certificate.
7. Compare analytic block-gain bounds with exhaustive/interval Jacobian bounds
   on small domains; reject schedule mismatch.
8. Confirm truncation and residual certificates enclose a much deeper reference
   on deterministic contractive fixtures.
9. Poison every true state after index 80, hidden generator field, target,
   cached V5/V8 output, and ACBSM posterior; candidate H20 must remain bitwise
   unchanged.
10. Poison the tower state and verify the output changes; poison the raw event
    after token creation and verify the readout does not.
11. Build exact same-current-input/different-history fixtures and verify reset
    removal plus state-swap transfer.
12. Prove level reset, upward/downward cuts, time shift, sign flip, and shuffle
    mutate the actual next-update tensors and have distinct array storage.
13. Verify the D wrappers remain bitwise identical to their parents under all
    tower lesions.
14. Verify snapshot/restore, active depth, delay buffers, hysteresis counter,
    RNG/permutation state, and pending causal tokens continue exactly.
15. Emit and independently recompute parameter, active-coefficient, state,
    serialized-byte, MAC, memory, and latency manifests for every arm.
16. Scan every historical raw seed role, reject `81100..81355` and
    `82100..82355`, and prove the proposed development block is disjoint before
    its values can be generated.
17. Verify finite normalized states, positive reference scales, invalid input
    rejection, and exact H5-as-H20-prefix behavior.
18. Confirm no default runtime adapter or historical V1--V8/ACBSM file changes.

These tests validate implementation and causal instrumentation. They do not
turn finite tests into a proof of an infinite limit or into biological evidence.

## 15. Cross-route predictions

The four routes make distinguishable predictions:

| Observation | A direct-limit tower | B quotient tower | C adaptive finite tower | D cosmetic wrapper |
|---|---|---|---|---|
| upper level added | compatible extension; finite readout change must converge | finer state projects consistently to coarse state | activated only when truncation bound requires it | state changes, output does not |
| level reset | can change finite-prefix approximation | removes scale-specific information | must cross lesion floor | exactly inert |
| projection/embedding violation | limit update ill defined | quotient/lumpability false | certificate fails closed | parent output unaffected |
| same input, different history | possible through recurrent state | possible through projective state | mandatory mediated difference | only parent history effect; dummy tower swap inert |
| compute with depth | grows with generated prefix | grows with represented scale dimensions | varies causally by episode/tick | extra wasted overhead |
| matched monolithic tie | does not refute direct-limit theorem | weakens quotient utility | **stops V9 mechanism route** | expected |

This panel prevents a successful graph proof from substituting for the finite
behavioral question.

## 16. Claim boundary and final recommendation

Route A can establish that a properly nested sequence of finite strongly
connected graphs has one countably infinite direct-limit SCC. Route B can, with
much stronger premises, establish an inverse/projective quotient dynamics.
Route C can engineer finite physical computation with a lazy generator,
adaptive truncation, and measurable causal state. None implies that a physical
brain has infinitely many neurons or computational modules, that its
connectome realizes these embeddings/projections, or that the resulting agent
is intelligent.

Proceed only with Route C's isolated unit implementation and a separately
hashed development-only registration. Keep A as the formal specification, B
deferred, and D mandatory. Do not open the V8 test, the reserved ACBSM block, or
any V9 confirmation block. If C cannot beat the matched recurrent controls and
survive state mediation plus real lesions, the correct endpoint is:

```text
NESTED-SCC MATHEMATICS: survives
V9 PREDICTIVE-STATE MECHANISM: DEVELOPMENT STOP
BIOLOGICAL/AGI CLAIM: untested
```

CE_RUN=_workspace/ce/agi-v9-nested-infinite-scc-20260811
