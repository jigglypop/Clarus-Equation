# V9 nested infinite-SCC tower contract

Status: COMPLETE

## Question

Can the brain/agent architecture be modeled not as several disjoint maximal SCCs
of one fixed graph, but as a nested sequence of recurrent cores whose direct
limit is one infinite SCC; and is that a legitimate V9 continuation of the
preserved V1--V8 sparse-causal-bridge research line?

## Frozen lineage and non-negotiable boundaries

1. V1--V8 registrations, code, results, seed partitions, hashes, and failure
   artifacts are immutable historical evidence. This run does not reopen a V7
   or V8 locked split and does not reinterpret a failed gate as a pass.
2. `V9` in this run initially means a **theory/design candidate**, not a
   confirmatory result. A V9 validation may be registered only if the lineage
   audit shows a genuinely new causal-state mechanism and a development-only
   falsifier passes without using a locked V8 test block.
3. The prerequisite SCC audit remains active: maximal SCCs of one fixed graph
   are disjoint equivalence classes; a positive-delay forward time-unroll is a
   DAG with singleton SCCs; SCC topology alone does not imply convergence.
4. No statement about an infinite physical neuron count, biological identity,
   AGI, cognition, or consciousness may be promoted from graph theory.

## Typed graph objects

1. A **nested SCC tower** is a sequence
   `(G_n, i_n)_{n in N}` with `G_n=(V_n,E_n)`, injective graph embeddings
   `i_n:G_n->G_{n+1}`, and each `G_n` nonempty and strongly connected. After
   replacing each level by its image, write `V_n subset V_{n+1}` and
   `E_n subset E_{n+1}`.
2. The direct-limit graph is
   `G_inf=(union_n V_n, union_n E_n)`. A finite-level `G_n` may be the unique
   maximal SCC of its own graph view; inside `G_inf` it is generally only a
   strongly connected subgraph, not a maximal SCC.
3. The tower is **properly infinite** when infinitely many inclusions add at
   least one new vertex. If every level is finite, its direct-limit vertex set
   is then countably infinite.
4. A **quotient tower** uses surjective scale maps
   `pi_{n+1,n}:Z_{n+1}->Z_n` instead of set inclusion. Quotient nesting and
   subgraph nesting are distinct constructions and must not be conflated.
5. A forward event unroll has vertices `(v,t)` and edges whose time coordinate
   strictly increases. It is distinct from both the recurrent template and a
   nested spatial/representational tower.

## Typed dynamics and normalization

1. Level `n` has a normalized metric state space `(X_n,d_n)` and update
   `F_n:X_n->X_n`. Physical time, rates, energy, rewards, costs, and edge
   strengths entering fixed-point/probability kernels are divided by named
   positive reference scales.
2. Exact inclusion compatibility means an isometric state embedding `J_n`
   satisfies `J_n F_n = F_{n+1} J_n` on the declared invariant image.
   Exact quotient compatibility instead means
   `pi_{n+1,n} F_{n+1} = F_n pi_{n+1,n}`.
3. Approximate finite truncations are compared in one declared complete metric
   space. Their uniform update defect, contraction factor, boundary condition,
   and observation/readout maps are frozen before a rollout bound is claimed.
4. A generator/query representation is a finite object that returns a requested
   node, edge, state, or operator block at `(level,index,time)` without
   instantiating every element. It is not automatically an exact quotient.

## Claims to prove, refute, or delimit

- `NISCC-1 [theorem candidate]`: the direct union of a nested sequence of
  nonempty strongly connected graphs is strongly connected.
- `NISCC-2 [theorem candidate]`: a proper tower of finite levels has a
  countably infinite direct limit, hence the entire limit graph is one infinite
  SCC when no outside vertices are present.
- `NISCC-3 [no-go candidate]`: two distinct maximal SCCs of one fixed graph
  cannot be nested. The phrase "nested SCCs" is valid only with graph views,
  induced domains, colors, thresholds, resolutions, or quotient maps indexed.
- `NISCC-4 [no-go candidate]`: unbounded recurrent time does not make the
  standard forward event unroll an SCC; it remains acyclic. The recurrent SCC
  lives in the template/quotient, while the infinite event stream is a separate
  object.
- `NISCC-5 [theorem candidate]`: exact compatible level maps induce a
  well-defined update on the algebraic direct limit; a uniform Lipschitz bound
  extends that update uniquely to the metric completion.
- `NISCC-6 [conditional theorem candidate]`: if the completed limit update is
  a self-map and a uniform contraction `q<1`, it has a unique fixed point. If a
  finite truncation has uniform one-step defect `epsilon_n`, its fixed-point or
  rollout error is bounded by the appropriate geometric/resolvent bound.
- `NISCC-7 [boundary theorem candidate]`: nested strong connectivity alone
  implies neither existence of a limit dynamics nor stability, memory,
  intelligence, or useful hierarchy.
- `NISCC-8 [construction candidate]`: finite prefixes can be generated and
  audited without materializing the infinite limit, but exact computation
  preservation requires lumpability/semiconjugacy in addition to topology.
- `V9-1 [lineage hypothesis]`: a nested multiscale recurrent predictive state
  is a genuinely new mechanism relative to V5--V8 output-shrinkage/readout
  variants only if policy outputs causally depend on the internal tower state,
  history survives under the declared update, and level/cross-scale lesions
  change held-out predictions under matched information and compute.
- `BRAIN-N1 [mathematical model]`: a whole-system nested SCC tower is
  constructible and its idealized direct limit can be one infinite SCC.
- `BRAIN-N2 [engineering design]`: a finite physical implementation may use a
  lazy generator plus adaptive finite truncations and certified error bounds.
- `BRAIN-N3 [empirical hypothesis]`: a biological brain instantiates this
  tower, its limit, or its chosen geometry. This remains untested.

## Mandatory counterexamples and falsifiers

1. Show that two nested maximal SCCs cannot occur in one fixed graph.
2. Show that a recurrent self-loop unrolled forward in time has only singleton
   SCCs.
3. Give a nested tower with divergent or oscillatory dynamics despite every
   level being strongly connected.
4. Give incompatible level maps for which no limit update is well-defined.
5. Treat one giant SCC, stabilization to a finite level, loss of held-out
   predictive benefit, state-readout bypass, or collapse under level lesions as
   valid negative outcomes, not thresholds to tune away.

## V9 development gate

Before any confirmatory V9 registration, an isolated development mechanism must:

1. consume only raw observations/actions/rewards available to all matched arms;
2. expose actual per-level recurrent state and cross-scale messages;
3. read decisions exclusively from that state rather than an analytic posterior
   or frozen parent output;
4. pass same-current-input/different-history mediation tests;
5. pass real level reset, cross-scale cut, time shift, sign, and state-shuffle
   interventions with no arm aliases;
6. beat V5 and matched finite-depth/monolithic/recurrent controls on a fresh
   development block with a preregistered effect floor;
7. report parameter, state, latency, and MAC budgets and leave any held-out V9
   confirmation block unopened unless the full conjunction passes.

The exact benchmark, thresholds, seeds, and model family are not authorized by
this contract alone; the independent route and status audits must first select
them. Failure leaves V9 at `THEORY/DESIGN` or `DEVELOPMENT STOP`.

## Acceptance for this research run

- Independent lineage/source, mathematical, and alternative-route reports are
  `Status: COMPLETE`.
- A status audit assigns formal provenance and deletes or narrows every parent
  claim defeated by a complete counterexample.
- Approved graph/dynamics helpers have exhaustive finite and property tests.
- Any development benchmark is preregistered before its result and is explicitly
  separated from confirmation.
- The canonical document states precisely: fixed-graph SCC partition,
  scale-indexed nesting, direct-limit infinity, forward time-unroll, dynamics,
  V1--V8 lineage, and biological evidence are different layers.

CE_RUN=_workspace/ce/agi-v9-nested-infinite-scc-20260811
