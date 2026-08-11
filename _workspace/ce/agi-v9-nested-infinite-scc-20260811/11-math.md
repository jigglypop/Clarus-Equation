# Mathematical verification: nested infinite-SCC towers

Status: COMPLETE

## Verdict

The nested-tower construction is mathematically coherent after four objects are
kept separate:

1. finite strongly connected template graphs at indexed levels;
2. their algebraic direct-union graph;
3. the forward event unroll in time;
4. the direct-limit and completed metric state dynamics.

The direct union of nested nonempty strongly connected graphs is strongly
connected.  If its levels are finite and add vertices infinitely often, its
vertex set is countably infinite (under the declared enumeration/ZFC convention),
and the standalone limit graph is one infinite maximal SCC.  Conversely, every
countable strongly connected digraph admits an increasing exhaustion by finite
strongly connected subgraphs.  This gives an exact graph-theoretic equivalence,
but the finite levels are not distinct maximal SCCs of the fixed limit graph.

Compatible level updates induce an algebraic direct-limit update.  A common finite
Lipschitz constant is sufficient for a unique extension to the metric completion,
and a common strict contraction factor gives a unique fixed point and geometric
truncation bounds.  None of these metric premises follows from strong
connectivity.  In particular, compatible finite truncations can all have spectral
radius zero and individual contraction factors below one while their completed
limit has spectral radius one.

| Claim | Status | Exact scope |
|---|---|---|
| `NISCC-1` | **PROVED** | Direct union of an inclusion-nested sequence of nonempty strongly connected graphs is strongly connected |
| `NISCC-1C` | **PROVED** | A digraph is countable and strongly connected iff it has an increasing finite strongly connected exhaustion; properness corresponds to an infinite vertex set |
| `NISCC-2` | **PROVED, with foundation declaration** | Finite levels plus infinitely many vertex additions give a countably infinite limit in ZFC or with explicit finite-level enumerations |
| `NISCC-3` | **PROVED no-go** | Distinct maximal SCCs of one fixed graph cannot be nested or overlap |
| `NISCC-4` | **PROVED no-go** | A strictly forward time-unroll is a DAG with singleton SCCs, even at infinite horizon |
| `NISCC-5` | **PROVED, conditional** | Exact compatible isometric embeddings give a direct-limit update; a uniform finite Lipschitz bound gives a unique completion extension |
| `NISCC-6` | **PROVED, conditional** | A completed self-map with one uniform `q<1` has a unique fixed point and geometric/resolvent error bounds |
| `NISCC-7` | **PROVED by counterexamples** | Nested strong connectivity gives neither compatible dynamics nor stability, memory, hierarchy, intelligence, or useful prediction |
| `NISCC-8` | **PROVED as a construction theorem** | Certified finite prefixes and finite causal cones can be evaluated lazily; exact arbitrary-state compression additionally requires semiconjugacy/lumpability |
| `V9-1` | **UNTESTED EMPIRICAL/CAUSAL HYPOTHESIS** | Mathematics does not establish novelty, mediation, held-out benefit, or lesion sensitivity |
| `BRAIN-N1` | **VALID MATHEMATICAL MODEL** | A countably infinite one-SCC idealization is constructible |
| `BRAIN-N2` | **VALID ENGINEERING DESIGN** | A finite generator can query exact finite causal cones and certified approximations |
| `BRAIN-N3` | **UNTESTED EMPIRICAL HYPOTHESIS** | No biological identity or infinite physical neuron count follows |

## 1. Typed graph objects and conventions

Let

\[
G_n=(V_n,E_n),\qquad n\in\mathbb N,
\]

be nonempty directed graphs.  The general definition uses injective graph
embeddings `i_n:G_n -> G_(n+1)`.  Since all maps are injective, their images may
be used as representatives; thereafter write

\[
V_n\subseteq V_{n+1},\qquad E_n\subseteq E_{n+1}.
\]

This replacement preserves graph structure.  The direct-union graph is

\[
G_\infty=(V_\infty,E_\infty),\qquad
V_\infty=\bigcup_{n=0}^{\infty}V_n,
\quad E_\infty=\bigcup_{n=0}^{\infty}E_n.
\]

Every directed path in this report is finite; a length-zero path is admitted.
An infinite walk does not create reachability by a limiting convention.  An SCC
is a maximal mutual-reachability equivalence class, not an arbitrary strongly
connected subgraph.

The tower is **properly infinite** if infinitely many inclusions add at least one
new vertex.  Properness is about vertices, not merely adding parallel metadata or
edges.  A tower that eventually stops adding vertices has a finite vertex union,
although its edge annotations or dynamics may continue changing.

## 2. `NISCC-1`: direct unions preserve strong connectivity

### Theorem 1

The direct union of an inclusion-nested sequence of nonempty strongly connected
directed graphs is strongly connected.

### Proof

Take arbitrary `u,v in V_inf`.  There are indices `r,s` with `u in V_r` and
`v in V_s`.  Let `k=max(r,s)`.  Nesting puts both vertices in `V_k`.  Since
`G_k` is strongly connected, there is a finite path from `u` to `v` whose edges
all lie in `E_k`.  Because `E_k` is a subset of `E_inf`, this is also a path in
`G_inf`.  Reversing the roles of `u,v` gives a path from `v` to `u`.  Thus every
pair is mutually reachable.  QED.

No finiteness, properness, metric, dynamics, or choice premise is used in this
proof.  Injectivity and nesting are essential: a noninjective scale map is a
quotient tower, not this direct-union construction.

### Theorem `NISCC-1C` (converse finite exhaustion)

Under an explicit enumeration, or in ordinary ZFC, a nonempty directed graph is countable
and strongly connected if and only if it is the union of an increasing sequence
of finite nonempty strongly connected subgraphs.  If the graph is countably
infinite, the exhaustion can be chosen properly infinite.

### Proof: reverse implication

If `G=union_n H_n` for nested finite strongly connected `H_n`, Theorem 1 makes
`G` strongly connected.  A countable union of explicitly enumerated finite sets
is countable.  If vertices are added infinitely often, the union is infinite and
hence countably infinite.

### Proof: forward implication

Let `G=(V,E)` be countable and strongly connected.  Enumerate its vertices as
`v_0,v_1,...` and its edges as `e_0,e_1,...`; `E` is countable because it is a
subset of the countable set `V x V`.  Fix root `r=v_0`.  For every vertex `v`,
choose one finite witness path `P_v:r ->* v` and one `Q_v:v ->* r`.  With an
enumerated graph, choose the least-coded shortest witnesses, so no unrecorded
choice is needed.

At stage `n`, let `S_n` contain the first `n+1` vertices and both endpoints of
the first `n+1` edges.  Let `H_n` contain those first edges and all edges and
vertices on `P_v,Q_v` for `v in S_n`.  Each `H_n` is finite and the construction
is cumulative, hence nested.

Every vertex `z` of `H_n` is mutually reachable with `r` inside `H_n`.  If `z`
lies on `P_v`, use the prefix `r ->* z` in `P_v`, and use the suffix
`z ->* v` followed by `Q_v` for the reverse path.  If `z` lies on `Q_v`, use
`P_v` followed by the prefix `v ->* z`, and use the suffix `z ->* r` for the
reverse.  Endpoints of explicitly added edges are in `S_n` and already have both
witness paths.  Therefore any `z,z'` connect via `z ->* r ->* z'`; `H_n` is
strongly connected.

Every vertex occurs in some `S_n` and every edge is explicitly added at some
stage, so `union_n H_n=G` as a graph, not only as a vertex set.  If `V` is
infinite, pass to the subsequence of stages at which a new vertex first appears;
the resulting exhaustion is proper.  QED.

This equivalence is graph-theoretic.  In a proper exhaustion, every `H_n` is a
strongly connected subgraph of the fixed limit graph, but none is a maximal SCC
of that limit because it is properly contained in a later strongly connected
subgraph.

## 3. `NISCC-2`: countability and the unique infinite SCC

### Theorem 2

Assume every `V_n` is finite and the tower is properly infinite.  Then `V_inf`
is countably infinite.  As a standalone graph, `G_inf` has exactly one SCC,
namely all of `V_inf`.

### Proof

In ZFC, a countable union of finite sets is countable.  More constructively, if
each finite difference `V_n minus V_(n-1)` is returned in a declared order, map
each vertex to the pair consisting of its first/birth level and its within-level
rank; pairs of natural numbers are countable.  Properness gives infinitely many
distinct new vertices, so the union is not finite.  It is therefore countably
infinite.  Theorem 1 makes the whole union strongly connected, and a strongly
connected graph has one mutual-reachability class.  QED.

In bare ZF, the unrestricted statement that every countable union of arbitrary
finite sets is countable uses a weak countable-choice principle.  This report
uses either ordinary ZFC or the explicit per-level enumeration supplied by a
generator.  The foundation convention is therefore visible rather than hidden.

If `G_inf` is embedded in a larger ambient graph, `V_inf` is still strongly
connected but need not remain maximal: reciprocal paths through outside vertices
can merge it into a larger SCC.  The unique-SCC conclusion above concerns the
standalone direct-limit graph whose vertex set is exactly `V_inf`.

## 4. `NISCC-3`: nested distinct maximal SCCs are impossible

### Theorem 3

Two maximal SCCs of one fixed directed graph are either equal or disjoint.  In
particular, distinct maximal SCCs cannot satisfy `C subset D` and cannot overlap.

### Proof

Mutual reachability is an equivalence relation.  SCCs are its equivalence classes,
and two equivalence classes with a common vertex are equal.  If nonempty `C` is a
subset of `D`, they intersect and hence are equal.  QED.

Thus the phrase **nested SCC** is well typed only when the graph view changes with
the index:

- `G_n` may be the unique maximal SCC of its own finite graph view;
- the same vertex set may define different SCCs under colors, thresholds, delays,
  resolutions, or induced domains;
- a quotient tower may use surjective scale maps;
- in the fixed direct-limit graph, finite levels are merely nested strongly
  connected subgraphs and the union is the one maximal SCC.

If two proposed modules of a fixed uncolored graph have cross paths in both
directions, internal strong connectivity concatenates those paths and merges the
modules into one SCC.  Relabeling them as levels does not evade maximality.

## 5. `NISCC-4`: infinite recurrent time is not an SCC unroll

Let a forward event unroll have vertices `(v,t)` and edges whose time coordinate
strictly increases.  The common one-tick form is

\[
(u,t)\longrightarrow(v,t+1)
\quad\text{whenever}\quad u\longrightarrow v
\]

in the recurrent template.

### Theorem 4

Every strictly forward time-unroll is acyclic and has singleton SCCs, even when
the time horizon is infinite.

### Proof

Along every positive-length directed path, time increases strictly at each edge.
Such a path cannot return to its starting time and hence cannot form a directed
cycle.  If two event vertices were mutually reachable, the first path would make
the target time no smaller and the reverse path would make it no larger.  Equality
of times forces both paths to have length zero, so the event vertices are equal.
QED.

A recurrent self-loop `v -> v` therefore unrolls as

\[
(v,0)\to(v,1)\to(v,2)\to\cdots,
\]

whose SCCs are singletons.  The recurrent template is recovered by the separately
declared time-translation quotient `(v,t) ~ (v,s)`, not by SCC decomposition of
the event graph.  Zero-delay within-slice edges violate the strict-time premise
and must be analyzed in their own slice graph.

## 6. `NISCC-5`: algebraic direct-limit dynamics

Let `(X_n,d_n)` be metric spaces with injective isometric embeddings

\[
J_n:X_n\longrightarrow X_{n+1}.
\]

Write `J_(m,n)=J_(m-1)...J_n` for `m>n`.  The algebraic direct limit consists of
pairs `(n,x)` modulo

\[
(n,x)\sim(m,y)
\quad\Longleftrightarrow\quad
J_{k,n}x=J_{k,m}y
\text{ for some }k\ge\max(m,n).
\]

Denote a class by `[n,x]` and the canonical injection by `j_n(x)=[n,x]`.

### Theorem 5A (well-defined update)

Suppose updates `F_n:X_n -> X_n` satisfy exact inclusion compatibility

\[
J_nF_n=F_{n+1}J_n.
\tag{C}
\]

Then

\[
F_{alg}[n,x]=[n,F_nx]
\]

is a well-defined update on the algebraic direct limit.  Conversely, the canonical
prescription is well defined only if (C) holds, because the canonical injections
are injective.

### Proof

Compatibility iterates to `J_(k,n)F_n=F_kJ_(k,n)`.  If `[n,x]=[m,y]`, choose a
common level `k` at which `J_(k,n)x=J_(k,m)y`.  Applying `F_k` and compatibility
gives

\[
J_{k,n}F_nx=F_kJ_{k,n}x
=F_kJ_{k,m}y=J_{k,m}F_my.
\]

Hence `[n,F_nx]=[m,F_my]`; the result is independent of representative.

For necessity, `[n,x]=[n+1,J_nx]`.  Well-definedness makes the images of these
representatives equal:

\[
j_{n+1}(J_nF_nx)=j_{n+1}(F_{n+1}J_nx).
\]

Injectivity of `j_(n+1)`, inherited from all injective `J_n`, yields (C).  QED.

### Direct-limit metric

Isometry makes

\[
d_{alg}([n,x],[m,y])
=d_k(J_{k,n}x,J_{k,m}y),
\qquad k\ge\max(m,n),
\]

independent of the chosen common level and representatives.  It is a genuine
metric.  Let `X_bar` be its metric completion.  The algebraic limit is dense in
`X_bar` by definition and need not itself be complete.

### Theorem 5B (uniform Lipschitz extension)

If one finite constant `L` satisfies

\[
d_n(F_nx,F_ny)\le Ld_n(x,y)
\quad\text{for every }n,x,y,
\tag{UL}
\]

then `F_alg` is `L`-Lipschitz and extends uniquely to an `L`-Lipschitz self-map
`F_bar:X_bar -> X_bar`.

### Proof

Move any two algebraic representatives to one common level and apply (UL) there;
this gives

\[
d_{alg}(F_{alg}a,F_{alg}b)\le Ld_{alg}(a,b).
\]

For `x in X_bar`, choose algebraic `x_r -> x`.  The sequence `F_alg(x_r)` is
Cauchy because of the Lipschitz bound, so define `F_bar(x)` as its limit.  The
same bound makes this independent of the approximating sequence, preserves `L`,
and forces uniqueness on the completion because the algebraic subspace is dense.
QED.

Uniformity cannot be replaced by `every level has some finite Lipschitz constant`.
For example, take `X_n=R^n` with Euclidean norm, append-zero embeddings, and

\[
F_n(x_1,\ldots,x_n)=(x_1,2x_2,\ldots,nx_n).
\]

These maps are exactly compatible but have `L_n=n`.  Their algebraic limit on
finite-support sequences sends basis vector `e_k` to `k e_k`.  Since
`e_k/k -> 0` in `l2` while its image has norm one, the map is not continuous at
zero and has no continuous everywhere extension to `l2`.  Indeed the `l2`
sequence `(1/k)` would be sent to the non-`l2` all-ones sequence.

## 7. `NISCC-6`: contraction and finite-truncation errors

### Theorem 6A (completed fixed point)

If `F_bar:X_bar -> X_bar` is a self-map of the complete metric space and is a
uniform contraction

\[
d(F_bar x,F_bar y)\le qd(x,y),\qquad 0\le q<1,
\]

then it has one fixed point `x_star`, and every iteration converges to it with

\[
d(F_bar^t x,x_star)\le q^t d(x,x_star).
\]

### Proof

This is Banach's contraction theorem.  Completeness, self-mapping, and the single
strict constant `q` are all indispensable premises.  QED.

### Theorem 6B (lifted finite-truncation rollout)

Compare a finite-prefix approximation only after embedding/lifting it into the
same complete metric space.  Let `Fhat_n:X_bar -> X_bar` be the declared lifted
prefix map and suppose on the relevant invariant domain

\[
\sup_z d(F_bar z,Fhat_n z)\le\epsilon_n.
\tag{D}
\]

For trajectories `x_(t+1)=F_bar x_t` and `y_(t+1)=Fhat_n y_t`,

\[
d(x_t,y_t)
\le q^td(x_0,y_0)
+\frac{1-q^t}{1-q}\epsilon_n.
\tag{R}
\]

### Proof

With `e_t=d(x_t,y_t)`, the triangle inequality, contraction, and (D) give

\[
e_{t+1}
\le d(F_bar x_t,F_bar y_t)+d(F_bar y_t,Fhat_n y_t)
\le qe_t+\epsilon_n.
\]

Induction sums the geometric series and yields (R).  QED.

A defect measured only at the current inspected state is not a recursive
certificate: after one update the approximate trajectory generally leaves that
point.  Bound (R) needs the displayed uniform bound over a trajectory-invariant
domain, or a separately certified time-indexed defect at every visited step.

For time-dependent defects, replace the last term by

\[
\sum_{s=0}^{t-1}q^{t-1-s}\epsilon_{n,s}.
\]

If only an `L>=1` Lipschitz bound is available, the analogous finite-horizon
bound grows as `L^t`; it gives no uniform infinite-horizon certificate.

### Theorem 6C (fixed-point and residual bounds)

If `y_n` is a fixed point of `Fhat_n`, then

\[
d(y_n,x_star)\le\frac{\epsilon_n}{1-q}.
\]

More generally, any candidate `y` with residual
`r=d(y,F_bar y)` satisfies

\[
d(y,x_star)\le\frac{r}{1-q}.
\]

For the first result, insert `Fhat_n y_n=y_n` and `F_bar x_star=x_star` into
the same one-step inequality; for the second, use
`d(y,x_star)<=r+q d(y,x_star)` and rearrange.  Existence of `y_n` is a separate
premise, supplied for example if `Fhat_n` is itself a self-map contraction.

### Block resolvent form

For a finite nonnegative block-gain matrix `M` with `rho(M)<1`, componentwise
errors satisfying

\[
e_{t+1}\le Me_t+\delta_n
\]

obey

\[
e_t\le M^te_0+\sum_{j=0}^{t-1}M^j\delta_n,
\qquad
\limsup_{t\to\infty}e_t\le(I-M)^{-1}\delta_n.
\]

For infinitely many blocks, these formulas require `M` to be a bounded positive
operator on a declared Banach sequence space, `M^t -> 0` in the required sense,
and a bounded positive resolvent.  Finite principal submatrices passing a spectral
test do not supply those properties.

### Infinite-dimensional small-gain counterexample

Let `X_n=R^n` with append-zero embeddings and define the compatible weighted
backward shift

\[
(B_nx)_i=\frac{i}{i+1}x_{i+1}\quad(1\le i<n),
\qquad (B_nx)_n=0.
\]

Every `B_n` is nilpotent, so `rho(B_n)=0`, and its Euclidean operator norm is

\[
q_n=\frac{n-1}{n}<1.
\]

Thus every finite level is a contraction.  But `q_n -> 1`.  The completed limit
on `l2` is the infinite weighted backward shift `B` with `||B||=1`.  For every
power `m`, products of `m` consecutive weights can be made arbitrarily close to
one by starting sufficiently far in the tail, so `||B^m||=1` and

\[
\rho(B)=\lim_{m\to\infty}\|B^m\|^{1/m}=1.
\]

There is no uniform strict contraction or finite resolvent bound.  This is a
complete counterexample to inferring limit small-gain stability from all finite
truncations separately.

## 8. `NISCC-7`: topology-only boundary and counterexamples

Use the proper graph tower `V_n={0,...,n}` with both directed edges between every
adjacent pair.  Every level is finite and strongly connected, and the union is
one countably infinite SCC.  On `X_n=R^(n+1)` with append-zero embeddings, both
of the following dynamic towers are exactly compatible:

1. `F_n(x)=2x`.  The completed `l2` update diverges in norm for every nonzero
   state under iteration.
2. `F_n(x)=-x`.  Every nonzero state is period two and does not converge.

The graph tower is identical in both cases.  Hence nested strong connectivity
does not imply convergence, stability, a fixed point attractor, or a useful time
scale.

### Incompatible-map counterexample

Keep identity/append-zero state embeddings but let level maps alternate between
`F_n=identity` and `F_n=-identity`.  The direct-limit representatives
`[n,x]=[n+1,J_nx]` would be sent to `x` under one representative and `-x` under
the other.  For `x != 0` these are different direct-limit states.  Therefore no
canonical limit update exists.  Strong connectivity of every graph level cannot
repair a failure of dynamic compatibility.

The same SCC topology can host maps that erase state, preserve every state,
diverge, oscillate, become multistable, or behave chaotically.  It consequently
implies neither memory nor intelligence.  A level hierarchy is also not visible in
the maximal-SCC partition of the fixed limit: that partition contains one giant
class.  The level index, embeddings, geometry, update, readout, and interventions
are extra structure.

## 9. `NISCC-8`: lazy finite-prefix generation and exact causal cones

### Theorem 8A (certified prefix generation)

Suppose a total generator `Gen(n)` returns finite `G_n`, its embedding into
`G_(n+1)`, and finite state/operator data, together with a proof or inductive
certificate that every level is nonempty, strongly connected, and nested.  Then
any query naming finitely many levels and indices can be answered by generating
only the largest requested prefix.  Theorems 1 and 2 certify the ideal union
without materializing it.

Testing finitely many outputs of `Gen` does not prove a universal invariant over
all levels.  The generator needs a proof, a verified inductive construction, or
a separately audited rule.  Likewise, deciding that an edge or predecessor is
absent from the entire union needs a completeness certificate; failure to see it
in the current prefix is insufficient.

### Theorem 8B (compatible finite-support evolution)

For `m>=n`, exact inclusion compatibility implies, for every finite time `t`,

\[
F_m^tJ_{m,n}=J_{m,n}F_n^t.
\tag{E}
\]

This follows by induction on `t`.  Consequently, if readouts are compatible,

\[
R_mJ_{m,n}=R_n,
\]

then an initial state originating at level `n` has exactly the same finite-time
readout whether computed at level `n` or embedded into any later level.  This is
an exact lazy computation theorem for the invariant finite-support image.  It
does not cover an arbitrary completed state with a nonzero infinite tail.

Weight tying is a possible implementation of this premise, but repeated parameter
names alone are not a proof.  The newly added coordinates and boundary/cross-scale
messages must make `J_(m,n)(X_n)` invariant and must satisfy equation (E) exactly.

### Theorem 8C (finite causal-cone evaluation)

Let the limit graph have finite in-degree at every vertex.  Consider a synchronous
local update in which information crosses at most one directed edge per tick:

\[
x_v(t+1)=f_v\left(x_v(t),
\{x_u(t):u\in\operatorname{Pred}(v)\},
\text{declared local input at }t\right).
\tag{L}
\]

For a finite query set `S` and finite horizon `T`, define its directed backward
causal cone

\[
\mathcal C_T(S)=
\{u:\text{there is a directed path }u\to^*s,
s\in S,\text{ of length at most }T\}.
\]

Then `C_T(S)` is finite, and the states of `S` at time `T` depend only on initial
states and time-indexed inputs inside that cone.  If in-degree is bounded by
`Delta`, then

\[
|\mathcal C_T(S)|
\le |S|\sum_{k=0}^{T}\Delta^k.
\]

### Proof

Finite in-degree makes the one-step predecessor set of a finite set finite.
Induction for `T` steps makes the entire backward cone finite.  Formula (L) says
the state at one tick depends only on distance-one predecessors at the previous
tick.  Induction on time therefore shows that no vertex at backward distance
greater than `T` can influence the query by time `T`.  The displayed size bound
counts at most `Delta^k` predecessors per query vertex at depth `k`; overlaps only
reduce it.  QED.

Because the tower exhausts the limit, the finite cone is contained in some
prefix.  A lazy generator can instantiate that prefix, or only the cone, and return
the exact finite-horizon answer without instantiating the infinite SCC.  It must
provide complete incoming adjacency for every cone vertex; otherwise a future
prefix could reveal a missing predecessor and invalidate exactness.

This theorem is the precise form of `finite substrate/generator, infinitely
queryable recurrent field`.  It requires local one-edge-per-tick causality.
Instantaneous global fixed-point coupling, unbounded in-degree, or a readout that
directly scans the infinite tail violates its premises.

An infinite-horizon or exact fixed-point query can have an infinite backward cone.
It therefore needs a separate contraction plus truncation/tail defect bound such
as Theorem 6, spatial decay, or an exact quotient.  Finite-horizon locality alone
does not make an infinite fixed point finitely computable.

### Exact quotient/lumpability theorem

An inclusion-compatible generator is not automatically an exact quotient for
arbitrary states.  Let `F:X -> X` and let `Q:X -> Z` be a surjective proposed
aggregation.  There exists a unique macro update `Phi:Z -> Z` with

\[
QF=\Phi Q
\]

if and only if

\[
Qx=Qy\quad\Longrightarrow\quad QFx=QFy.
\tag{fiber invariance}
\]

Necessity follows by applying `Phi` to equal aggregates.  For sufficiency, define
`Phi(z)=QF(x)` for any `x` in the fiber of `z`; fiber invariance makes the result
independent of representative, and surjectivity gives uniqueness.  This is exact
deterministic lumpability/semiconjugacy.

A quotient tower uses surjections
`pi_(n+1,n):Z_(n+1) -> Z_n` satisfying

\[
\pi_{n+1,n}F_{n+1}=F_n\pi_{n+1,n}.
\]

It is a projective/quotient construction, not the injective direct union proved
in Theorem 1.  To compute a low-level output from an arbitrary high-level or
completed state exactly, a retraction/aggregation must satisfy this equation and
the readout must factor through it.  A finite generator alone provides neither
fiber invariance nor exact reconstruction.

If semiconjugacy holds only up to a uniform defect, Theorem 6 supplies a bounded
rollout error only after a contraction and a common comparison metric are also
declared.

## 10. Exact meanings of `the brain itself is an infinite SCC`

The phrase has one valid mathematical reading in this run:

> An idealized, countably infinite directed template is presented as the proper
> direct union of finite strongly connected graph views.  Its declared standalone
> vertex set is the union, so every pair of vertices is mutually reachable and the
> entire template is one infinite maximal SCC.

The converse exhaustion theorem shows that every countable strongly connected
template can be presented this way.  The presentation is not unique: roots,
enumerations, witness paths, and level boundaries are degrees of freedom.

The phrase does **not** mean any of the following:

1. infinitely many physical neurons exist;
2. finite levels are nested distinct maximal SCCs of the same fixed graph;
3. the forward event unroll is strongly connected;
4. topology supplies a compatible or computable limit update;
5. the completed dynamics contract or have a fixed point;
6. a lazy generator is an exact quotient of arbitrary infinite states;
7. the level hierarchy is unique, biologically identified, intelligent, conscious,
   or an AGI.

A finite physical system can implement the **description and query rule** for an
unbounded virtual tower and can evaluate every finite causal cone on demand.  At
every actual time it still performs a finite computation on finite storage.  This
is an engineering representation of an ideal limit, not literal material infinity.

`V9-1` therefore remains a theory/design candidate.  Causal dependence of policy
outputs on tower state, history mediation, matched-compute benefit, and real
level/cross-scale lesions are empirical gates; no graph theorem can mark them
passed.

## 11. Assumptions ledger

| ID | Premise | Used for | If absent |
|---|---|---|---|
| `A1` | paths are finite; length zero admitted | SCC equivalence and union proof | reachability semantics change |
| `A2` | injective graph embeddings, represented as inclusions | `NISCC-1/2` | quotient collapse may identify vertices |
| `A3` | every level nonempty and strongly connected | direct-union strong connectivity | limit need not be one SCC |
| `A4` | infinitely many genuine vertex additions | countably **infinite** conclusion | tower may stabilize finitely |
| `A5` | finite levels have explicit enumerations, or ZFC is used | countable union | hidden countable-choice issue |
| `A6` | no ambient outside vertices for maximality claim | unique limit SCC | outside reciprocal paths may merge it |
| `A7` | event edges strictly increase time | unroll DAG theorem | within-slice SCCs may occur |
| `A8` | isometric injective state embeddings | direct-limit metric | distances can depend on representative |
| `A9` | exact equation `J_n F_n=F_(n+1) J_n` | well-defined algebraic update | representative-dependent update |
| `A10` | one finite Lipschitz `L` for all levels | completion extension | compatible map may be discontinuous/unextendable |
| `A11` | completed update is a self-map with one `q<1` | Banach fixed point | finite factors may approach one |
| `A12` | truncations compared in one metric with a uniform defect | rollout/fixed-point bound | `epsilon_n` is not typed |
| `A13` | fixed normalization scales and norms | all gain/defect statements | numerical constants change with units |
| `A14` | total certified prefix generator | lazy graph queries | finite samples do not prove the tower |
| `A15` | finite in-degree, local synchronous one-edge-per-tick update | exact finite causal cone | a finite query may need infinite data immediately |
| `A16` | complete predecessor enumeration for the cone | exact lazy result | later prefixes can reveal omitted causes |
| `A17` | compatible readout or exact quotient | computation preservation | topology alone preserves no output |
| `A18` | bounded positive infinite gain operator/resolvent when used | infinite block small gain | finite matrix certificates do not pass to limit |

All physical quantities entering these state maps must first be normalized:
physical time by a positive time scale, rates by reference rates, energy by a
positive energy scale, rewards/costs by a fixed utility scale, and edge strengths
by declared block/state scales.  `q`, Lipschitz constants, normalized residuals,
and probability-kernel arguments are dimensionless.  Changing scales after seeing
a failed limit certificate is a new model, not a proof repair.

## 12. Findings by severity

### P0

1. **Nested maximal SCCs in one graph are impossible.**  The finite tower levels
   are strongly connected views/subgraphs; the fixed direct limit has one maximal
   SCC.
2. **The infinite time-unroll is a DAG.**  Its event infinity must not be cited as
   an infinite SCC.
3. **Topology does not define limit dynamics.**  Exact compatibility is necessary;
   equivalent representatives under alternating incompatible level maps give a
   complete no-limit counterexample.
4. **Finite stability certificates do not automatically survive completion.**  A
   compatible sequence of nilpotent strict finite contractions can have a limit
   operator with spectral radius one.
5. **A generator is not a quotient.**  Exact arbitrary-state compression needs
   fiber invariance/semiconjugacy; finite-horizon locality is a different theorem.
6. **The biological claim is unproved.**  Mathematical construction neither counts
   physical neurons nor identifies a brain hierarchy, cognition, consciousness,
   or AGI.

### P1

1. Countability requires ZFC or an explicit enumeration of finite additions.
2. Uniform Lipschitz and contraction constants must be strict and level-independent;
   `sup_n q_n=1` is not a contraction certificate.
3. A finite-prefix defect is meaningful only after lift/embedding into one common
   complete metric space with frozen boundary conditions.
4. Exact causal-cone evaluation needs finite in-degree, local tick semantics, and
   certified complete incoming adjacency.
5. Infinite-horizon and fixed-point queries need tail/resolvent bounds even when
   every finite-horizon causal cone is finite.
6. Structural, colored, thresholded, effective-gain, time-unrolled, and quotient
   graphs must report separate SCC statements.

### P2

1. Emit birth level, prefix hash, witness paths/strong-connectivity certificate,
   and predecessor-closure certificate for every lazy query.
2. Report `L_n`, `sup L_n`, finite `rho(M_n)`, an infinite-operator norm/resolvent
   certificate, truncation defect, and readout defect separately.
3. When using an approximate quotient, report both the one-step semiconjugacy
   defect and its geometric/resolvent amplification.
4. Treat tower stabilization, a giant unstructured SCC, state-readout bypass,
   failed lesions, or lost held-out benefit as negative outcomes rather than
   redefining the levels after inspection.

## 13. Independent scratch verification

The proofs do not depend on computation.  A read-only in-memory Python check
verified 64 bidirected-path prefixes, all 89,440 ordered reachability pairs within
them, compatible weighted-shift factors approaching one, and the sharp scalar
geometric defect bound.  It also instantiated the incompatible representative
images `1` and `-1`.

Reproduction command shape:

```powershell
@'
# For n=0..63 and every u,v in {0,...,n}, verify the adjacent bidirectional path.
# For n in (2,4,8,32,256,4096), record q_n=(n-1)/n for the nilpotent
# compatible weighted backward shift.
# With q=.6 and eps=.02, iterate y_(t+1)=q*y_t+eps and compare to
# eps*(1-q**t)/(1-q); compare the fixed point to eps/(1-q).
# Under identity embeddings, compare alternating-map images of x=1: 1 and -1.
'@ | python -
```

Observed output:

```text
{'nested_prefixes': 64,
 'reachable_ordered_pairs': 89440,
 'weighted_shift_q': [(2, 0.5), (4, 0.75), (8, 0.875),
                      (32, 0.96875), (256, 0.99609375),
                      (4096, 0.999755859375)],
 'finite_shift_spectral_radius': '0 (nilpotent)',
 'limit_shift_spectral_radius': '1',
 'sharp_fixed_point_error': 0.049999999999999996,
 'rollout_first_6': [0.0, 0.02, 0.032, 0.0392, 0.04352, 0.046112],
 'incompatible_representative_images': (1.0, -1.0)}
```

Scratch path: none; calculations were read-only and in memory.  This lane modified
only this report.

CE_RUN=_workspace/ce/agi-v9-nested-infinite-scc-20260811
