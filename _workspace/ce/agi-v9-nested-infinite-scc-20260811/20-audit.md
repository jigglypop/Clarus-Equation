# Formal status audit: V9 nested infinite-SCC tower

Status: COMPLETE

Gate: PASS (formal theory and isolated unit-mechanism scope only)

Sub-gates:

```text
formal graph/direct-limit mathematics       PASS
isolated tower/generator unit implementation GO
isolated controller/real-lesion unit tests   GO
development registration                    HOLD
256-seed development execution               BLOCKED NOW
V9 confirmation/test                         BLOCKED
biological/AGI promotion                      BLOCKED
```

The audit is complete even though execution is blocked: the block is a deliberate
phase boundary, not an unresolved mathematical defect. No development values may
be generated or opened until the pre-run conditions in Section 9 are frozen and
independently re-audited.

## 1. Executive verdict

The central construction is coherent:

```text
G_0 subset G_1 subset ... , every G_n strongly connected
                         |
                         +--> union_n G_n is strongly connected

finite levels + infinitely many genuine vertex births
                         +--> countably infinite standalone limit graph
                         +--> exactly one infinite maximal SCC
```

This does not produce nested distinct maximal SCCs inside one fixed graph. Each
finite level is a strongly connected subgraph (or the maximal SCC of its own
indexed graph view); the fixed direct-limit graph has one maximal SCC. The source
and mathematical lanes preserve this distinction consistently.

The dynamics require additional structure. Exact compatible inclusions induce an
algebraic direct-limit update; one uniform finite Lipschitz constant extends it to
the metric completion; one uniform strict contraction gives a unique fixed point
and geometric truncation bounds. Strong connectivity, parameter sharing, and
finite-level stability do not imply any of those analytic premises.

`V9-1` is not a result. It remains an untested causal-development hypothesis.
`BRAIN-N1` is a valid ideal mathematical model, `BRAIN-N2` is a conditional
engineering design, and `BRAIN-N3` remains untested only under a nonliteral
finite-generator/virtual-tower reading. The literal branch “the physical brain
has countably infinitely many neurons/vertices” is rejected.

## 2. Claim-by-claim decision

| Claim | Actual status | Exact domain and ceiling |
|---|---|---|
| `NISCC-1` | **THEOREM / PROVED** | An inclusion-nested sequence of nonempty strongly connected digraphs has a strongly connected direct union. No dynamics or finiteness is implied. |
| `NISCC-1C` | **ADDITIONAL THEOREM / PROVED** | A countable strongly connected digraph admits an increasing finite strongly connected exhaustion; conversely such an exhaustion has a countable strongly connected union. Explicit enumeration or ZFC is declared. |
| `NISCC-2` | **THEOREM / PROVED WITH FOUNDATION PREMISE** | Finite levels with infinitely many genuine vertex additions have a countably infinite union in ZFC or under explicit per-level enumeration. The standalone union is one maximal SCC; an ambient graph may merge it into a larger SCC. |
| `NISCC-3` | **NO-GO THEOREM / PROVED** | Distinct maximal SCCs of one fixed graph are disjoint equivalence classes and cannot be properly nested or overlap. Indexed strongly connected subgraphs or SCCs of different graphs may be nested. |
| `NISCC-4` | **NO-GO THEOREM / PROVED** | If every event edge strictly increases time, the finite- or infinite-horizon forward unroll is a DAG with singleton SCCs. Recurrence belongs to the template or a separately declared quotient. |
| `NISCC-5` | **CONDITIONAL THEOREM / PROVED** | Injective isometric state embeddings plus exact `J_n F_n = F_(n+1) J_n` give a well-defined algebraic limit map. A single level-independent finite Lipschitz constant gives its unique completion extension. |
| `NISCC-6` | **CONDITIONAL THEOREM / PROVED** | A completed self-map with one uniform `q<1` has a unique fixed point. Lifted truncations require a common metric/domain and uniform or recursively certified step defects for geometric/resolvent bounds. |
| `NISCC-7` | **BOUNDARY THEOREM / PROVED BY COUNTEREXAMPLES** | Nested strong connectivity entails no compatible limit dynamics, convergence, stability, memory, hierarchy, prediction, intelligence, or biological identity. |
| `NISCC-8` | **CONSTRUCTION THEOREM / PROVED CONDITIONALLY** | Certified generators answer finite-prefix queries; compatible finite-support trajectories/readouts can be exact; finite in-degree plus local one-edge-per-tick causality and complete predecessor enumeration give exact finite causal cones. Arbitrary completed states still require quotient/lumpability or truncation bounds. |
| `V9-1` | **UNTESTED EMPIRICAL/CAUSAL HYPOTHESIS** | Novelty requires exclusive state-to-output mediation, persistent multi-level history, genuine cross-scale lesions, and matched-compute held-out benefit. No such run exists. |
| `BRAIN-N1` | **VALID MATHEMATICAL MODEL** | A countably infinite one-SCC ideal template is constructible as a proper direct union. It is an idealization, not a neuron-count claim. |
| `BRAIN-N2` | **VALID CONDITIONAL ENGINEERING DESIGN** | A finite generator can evaluate certified prefixes, exact finite causal cones, or bounded approximations while every actual execution remains finite. |
| `BRAIN-N3` | **NARROWED EMPIRICAL HYPOTHESIS** | A finite biological system implementing an analogous multiscale recurrent representation is untested. Literal infinite physical neurons/agents are **REJECTED**, and the chosen geometry is unverified. |

The status table and proofs are at `11-math.md:33-45,81-690`. The mathematical
language agrees with the source distinction at `10-sources.md:250-270,292-320`.

## 3. Dependency DAG

```text
finite-path reachability + injective graph inclusions + strong levels
    |
    +--> NISCC-1 --> NISCC-2 [finite levels + properness + enumeration/ZFC]
    |       |
    |       +--> BRAIN-N1 [standalone ideal model only]
    |
    +--> NISCC-1C [enumeration + finite witness paths]

SCC equivalence/maximality ------------------------------> NISCC-3
strictly increasing event time --------------------------> NISCC-4

isometric J_n + exact inclusion compatibility ----------> NISCC-5A
NISCC-5A + one uniform finite L -------------------------> NISCC-5B completion
NISCC-5B + complete self-map + one uniform q<1 ---------> NISCC-6A fixed point
NISCC-6A + common lift/domain + uniform defect ----------> NISCC-6B/C bounds

certified total generator -------------------------------> NISCC-8A prefixes
exact compatibility + compatible readout ---------------> NISCC-8B finite support
finite indegree + local ticks + complete predecessors ---> NISCC-8C causal cone
surjective Q + fiber invariance -------------------------> exact quotient (separate)

NISCC-1/8 + finite controller implementation -----------> BRAIN-N2 design
BRAIN-N2 + no bypass + causal lesions + matched controls
             + fresh preregistered data -----------------> V9-1 (not yet tested)
direct biological measurements/interventions -----------> BRAIN-N3 (open)
```

`NISCC-7` consists of counterexamples that cut every unsupported arrow from
topology to dynamics, computation, V9 novelty, or biology.

## 4. Mandatory object distinctions

### 4.1 Maximal SCC versus nested strong exhaustion

At each indexed level, `G_n` may be the one maximal SCC of that level's graph.
Inside the fixed union, proper finite levels are nonmaximal strongly connected
subgraphs. Calling all levels distinct maximal SCCs of the union violates
equivalence-class maximality. This is a typing no-go, not a rejection of a
scale-indexed exhaustion.

### 4.2 Direct versus projective limits

- The direct construction uses injective maps `J_n:X_n -> X_(n+1)` and a
  colimit/union. Its compatibility equation is `J_n F_n = F_(n+1) J_n`.
- The quotient/projective construction uses surjections
  `pi_(n+1,n):Z_(n+1)->Z_n` and compatible sequences in an inverse limit. Its
  equation is `pi F_(n+1) = F_n pi`.

Neither construction implies the other. Route B is therefore deferred until
lumpability/projective compatibility is demonstrated; it may not inherit Route
A's theorem.

### 4.3 Recurrent template versus forward event unroll

Unbounded recurrence in time creates an infinite event stream, not a nontrivial
event-graph SCC. A positive-delay unroll is acyclic. A time-translation quotient
or finite recurrent template is a different graph object.

### 4.4 Finite causal cone versus infinite horizon

Finite in-degree and local one-edge-per-tick updates make the backward cone of a
finite query at finite horizon finite. They do not make an infinite-horizon,
stationary, or fixed-point query finite. Those require spatial decay, a uniform
contraction/tail defect, a bounded resolvent, or an exact quotient.

### 4.5 Uniform versus pointwise defect

A bound `epsilon_n/(1-q)` requires a uniform defect on the declared invariant
domain. A defect measured at one current state is only a diagnostic. A sequence
of pointwise defects can support a finite realized-trajectory envelope only when
the initial error is certified, the same contracting comparison map is used, and
every visited step is included:

```text
E_(t+1) <= q E_t + eta_t.
```

It cannot be silently promoted to a fixed-point or future infinite-horizon bound.
Route C must identify whether `q` certifies the deeper lifted update, preserve the
common metric/boundary, and charge every boundary probe used to obtain `eta_t`.

### 4.6 Weight tying versus compatibility

Repeated parameter names or a finite generator do not prove
`J_n F_n = F_(n+1) J_n`. New coordinates, boundary states, biases, nonlinearities,
and down-messages must leave the embedded image invariant and satisfy the equation.
Otherwise the controller is an approximate finite tower and must use a valid
defect envelope; it may not claim the exact direct-limit theorem.

## 5. V1--V8 and ACBSM provenance audit

The final source lane hash was independently confirmed as
`3747ab3957f7143d1119b6fa2713c4d8eaa9a47b29d13b1dc01e0c8f71cfe341`.
All eight current preregistration SHA-256 values independently match the ledger at
`10-sources.md:111-127`. In particular V8 remains
`a175a3d722f031e4878741a3c7136c75b1af229287e8e90442cecbc9591cdafc`.

The clean V8 provenance chain was independently checked:

```text
7c11c04d... historical V1--V7/R1 checkpoint
  -> 6baeaf17... V8 registration
  -> b0abbaa1... V8 implementation lock
  -> 5f20748f... failed V8 validation
```

All three ancestor tests returned true. The historical interpretation survives:

- V1, V3, V5, and V7 failed their conjunctive gates;
- V2 and V4 passed only their narrowed synthetic tasks, with V4 re-reading true
  state at each scored step rather than performing H20 free rollout;
- V6 was registered but never implemented or run;
- V8 was prospectively registered and implemented, then failed reliable
  superiority to V5; its locked test stayed closed;
- ACBSM remained a training-only `HOLD`; its rank-two proposal collapsed to rank
  one, and `82100..82355` remained unopened.

The following were independently confirmed absent at this audit:

```text
artifacts/agi/sparse_causal_bridge_test_v7.json
artifacts/agi/sparse_causal_bridge_test_v8.json
artifacts/agi/integrated_latent_state_bridge_development.json
experiments/preregistration/sparse_causal_bridge_v9.json
```

Therefore no V7/V8 locked test, ACBSM fresh block, or new V9 evidence block was
opened. No failed parent was reclassified. V1--V5 prospective-timing limitations
also remain disclosed; only V8 has the clean audited registration ancestry above.

## 6. Source-to-claim fit and biological ceiling

Primary sources support finite recurrent anatomy, feedback loops, communities,
hierarchies, and graph-specific giant SCCs. They do not support the V9 direct-limit
causal-state object.

- FlyWire `v630` has a 93.3% giant SCC under its exact confidence and five-synapse
  construction; this favors permitting a giant-core result, not nested maximal
  SCCs.
- The larval-fly “nested recurrent architecture” uses hierarchical connection
  clustering and bounded-hop return cascades, not a direct-limit SCC theorem.
- BANC's 13 networks are undirected spectral clusters, not SCCs.
- Mouse hierarchy and loop studies support finite feedback organization, not an
  infinite tower or V9 state mediation.
- Current physical connectomes and human cell counts are finite. “Infinite” may
  describe an ideal limit or indefinitely queryable virtual rule only.

Thus no source licenses biological identity, infinite physical neurons, stable
dynamics, cognition, consciousness, or AGI.

## 7. Severity findings

### P0 — claim-killing if violated

1. **Nested maximal SCCs are impossible in one fixed graph.** Use indexed
   strongly connected subgraphs or SCCs of explicitly different graph views.
2. **Direct and projective limits cannot be interchanged.** Injection compatibility
   does not prove quotient lumpability, and surjective projections do not build a
   direct union.
3. **The forward unroll is a DAG.** Infinite event count is not an infinite SCC.
4. **Topology does not define limit dynamics.** Incompatible alternating maps make
   the canonical limit update representative-dependent.
5. **Finite certificates do not imply a uniform limit certificate.** Compatible
   nilpotent finite contractions with `q_n -> 1` yield a completed shift with norm
   and spectral radius one.
6. **Generator/weight sharing is not computation preservation.** Exact arbitrary-
   state compression requires fiber invariance; exact direct evolution requires
   the compatibility equation.
7. **Finite causal locality is not infinite-horizon computability.** Missing this
   distinction invalidates a fixed-point or asymptotic claim.
8. **Lineage and biological locks remain closed.** No V9 result, brain identity,
   AGI, or literal physical infinity may be inferred from the formal tower.

### P1 — mandatory implementation and pre-run locks

1. Declare ZFC or deterministic per-level enumeration, genuine vertex births,
   and whether the limit is standalone or embedded in an ambient graph.
2. Freeze isometric embeddings, boundary conditions, update schedule, invariant
   domain, and readout compatibility. A Jacobi certificate cannot be reused for
   sequential, delayed, switched, or multirate code.
3. Report `L_n`, `sup_n L_n`, `q_n`, the uniform cap, and completion/operator
   certificate separately. Finite principal spectral radii are insufficient.
4. Keep uniform defects distinct from realized online defect envelopes; a single
   inspected point cannot authorize deactivation or division by `1-q`.
5. Exact causal-cone queries require complete incoming-adjacency certificates.
   “Not yet generated” is not evidence that a predecessor is absent.
6. Enforce the exclusive immutable state-token-to-readout path. Raw events,
   analytic posteriors, V5/V8/ACBSM outputs, persistence, hidden simulator state,
   and targets are forbidden bypasses.
7. Lesions must mutate the actual next-update tensors with distinct storage;
   capacity/MAC/state matching cannot count dummy parameters or no-op work.
8. Freeze all 18 disclosed model-selection coordinates before development data is
   opened. The many 95% endpoints are development criteria, not simultaneous
   confirmatory confidence statements.

### P2 — reporting and falsifier requirements

1. Emit birth levels, prefix and parameter hashes, witness paths/SCC certificates,
   predecessor-closure certificates, and generator determinism records.
2. Report generator parameters, every active generated coefficient, live state,
   serialized bytes, mean/peak MACs, boundary probes, memory, and latency.
3. Report exact compatibility residuals, truncation defects, readout defects, and
   deeper-prefix enclosure tests separately.
4. Preserve one giant SCC, finite tower stabilization, single-depth collapse,
   inert upper levels, failed mediation/lesions, and matched-control ties as valid
   negative outcomes.
5. Keep structural, thresholded, colored, effective, unrolled, direct-limit, and
   projective graphs in separate namespaces and artifacts.

## 8. Dimensionless and type gate

The pure graph statements are combinatorial. Dynamic and development quantities
pass the dimensionless gate only under the following declarations:

- physical time and delay are divided by a positive training-only reference tick;
- rates, energy, rewards, costs, and dimensional edge strengths are divided by
  named positive reference scales before fixed-point, probability, or threshold
  kernels;
- every `d_n` is a declared normalized metric and every isometric embedding uses
  that same scale convention;
- `L`, `q`, spectral radii, normalized block gains, relative residuals, and
  probability-kernel arguments are dimensionless;
- `epsilon_n`, online `E_t`, and residuals live in the same normalized comparison
  metric, so `epsilon_n/(1-q)` and resolvent bounds are type-correct;
- normalization is frozen from training data before development; an H20 target may
  not redefine a scale.

Reject nonpositive/nonfinite scales, states, gains, defects, timestamps, masks,
or inputs. Changing scales after a failed certificate defines a new model. The
18-premise ledger at `11-math.md:695-722` is sufficient if implemented literally.

## 9. Exact implementation and execution authorization

### Authorized now: isolated formal/unit mechanism

The following may be created and tested without opening any evidence seed:

1. `reality_stone/python/reality_stone/clarus/nested_scc_tower.py`
   - finite prefix generator and deterministic query API;
   - injective nesting, strong-connectivity, birth-level, and hash certificates;
   - exact inclusion-compatibility fixtures and explicit rejection paths;
   - finite causal-cone discovery with complete-predecessor certificates;
   - uniform/online error-envelope and finite resolvent helpers.
2. `reality_stone/python/reality_stone/clarus/adaptive_scc_tower_controller.py`
   - finite `D_max` controller with previous-tick schedule;
   - immutable state tokens and no raw-input/parent-output readout bypass;
   - genuine reset/up-cut/down-cut/time-shift/sign/shuffle interventions;
   - fail-closed snapshots, masks, timestamps, nonfinite inputs, and certificates.
3. `tests/test_nested_scc_tower.py` and
   `tests/test_adaptive_scc_tower_controller.py`, covering the 18 pre-seed
   obligations in `12-routes.md:767-807` insofar as they are unit/property tests.
4. A non-evidence example/demo that uses deterministic fixtures only.

The cosmetic wrapper may exist only as an explicit negative-control fixture. No
default runtime adapter is authorized. Unit fixtures do not validate an infinite
physical system or `V9-1`.

### Held now: development registration and runner

A development registration/template may be drafted with `Status: DRAFT`, but it
may not be frozen as executable and no development runner/result artifact may be
used until all of the following exist:

1. complete implementation, source/config/normalizer/comparator hashes, and an
   implementation lock;
2. all pre-seed unit/property/poisoning/alias/snapshot/budget tests passing;
3. schedule-specific contraction or registered Lyapunov certificate for the
   actual finite controller;
4. exact definitions and frozen values for all architecture, optimizer, depth,
   lesion, history-pair, effect-floor, and compute-matching choices;
5. a read-only exhaustive historical seed-role scan and a manifest selecting
   exactly 256 new collision-free development seeds;
6. explicit rejection of `81100..81355` and `82100..82355` and proof that no V7,
   V8, ACBSM, or confirmation role is reused;
7. an independent pre-run audit that recomputes every hash and confirms the block
   is still unopened.

### Explicit seed decision

**The 256-seed development run is not authorized now.** Exact seeds are not yet
registered, the implementation and tests do not yet exist, comparator identities
and budgets are not locked, and no pre-run hash audit has occurred. Generating
the raw 256 development episodes, viewing their values, training on them, or
executing any registered score is forbidden at this stage.

After the seven conditions above pass, a later audit may authorize one
development-only execution. Even a full pass would authorize only consideration
of a new confirmatory registration; it would not be V9 confirmation. Creation of
`sparse_causal_bridge_v9.json`, a V9 test block/artifact, reuse of the locked V8 or
ACBSM blocks, or any biological/AGI result remains forbidden.

## 10. Audit counts and final gate

| Item | Count/status |
|---|---:|
| Registered claims audited | 12 |
| Additional converse theorem audited (`NISCC-1C`) | 1 |
| Formal theorems/no-go/construction results surviving | 9 |
| Valid ideal mathematical models | 1 (`BRAIN-N1`) |
| Conditional engineering designs | 1 (`BRAIN-N2`) |
| Untested empirical/causal hypotheses | 2 (`V9-1`, narrowed `BRAIN-N3`) |
| Literal physical-infinity branch rejected | 1 |
| Explicit premise families | 18 |
| Complete counterexample/no-go families retained | 7 |
| Historical registration hashes independently matched | 8/8 |
| V8 ancestry edges independently confirmed | 3/3 |
| Locked test/fresh evidence blocks opened in this run | 0 |
| Development seeds authorized now | 0/256 |

Final disposition:

```text
NESTED-SCC MATHEMATICS       SURVIVES
FORMAL GENERATOR/UNIT CODE   AUTHORIZED
V9-1 CAUSAL MECHANISM        UNTESTED
256-SEED DEVELOPMENT RUN     BLOCKED PENDING PRE-RUN GATE
V9 CONFIRMATION              NOT REGISTERED / BLOCKED
BIOLOGICAL OR AGI CLAIM      UNTESTED / NOT AUTHORIZED
```

CE_RUN=_workspace/ce/agi-v9-nested-infinite-scc-20260811
