# Math and graph-verification lane — C. elegans connectome graph replay MVP

Status: COMPLETE

## Definitions and exact domain

Let a parsed source observation be

$$
r=(c,p,q,w,o),
$$

where `c` is one of the closed strings `chemical` and `electrical`, `p,q` are nonempty stable source-identifier strings, `w` is a nonnegative integer, and `o` is a unique nonnegative stable source-record ordinal.  Exact integer means a JSON integer whose host value has type `int`, not `bool`; IEEE floating values, numeric strings, negative values, and missing values are outside the domain.  Input bytes must first be decoded as strict UTF-8 and the declared schema must supply every field.  These are predicates for C2, not numerical estimates.

Define endpoint normalization

$$
N(c,p,q)=
\begin{cases}
(p,q),&c=\mathrm{chemical},\\
(\min(p,q),\max(p,q)),&c=\mathrm{electrical},
\end{cases}
$$

where the ordering is an explicitly implemented total ordering of the accepted identifier strings (Unicode code-point order is sufficient if identifiers are retained as exact decoded strings).  Thus `chemical(AVA,AVB)` and `chemical(AVB,AVA)` remain distinct, whereas both orientations of an electrical observation have the same normalized pair.  A self-pair is neither accepted nor rejected by this algebra; the frozen source schema must state that policy and validation must apply it consistently.

The observation table is

$$
T=\operatorname{sort}\{(c,a,b,w,o):(a,b)=N(c,p,q)\}
$$

under the lexicographic key `(connection_class, endpoint_a, endpoint_b, source_record_ordinal)`.  It is an audit table: no accepted observation is removed.  The distinct aggregate graph is

$$
G_{c,a,b}=\sum_{(c,a,b,w,o)\in T}w.
$$

The sum is over precisely the same normalized class/endpoints, so it cannot merge a chemical edge with its reverse or with an electrical pair.  `o` and the original released row representation/provenance remain attached to each entry of `T`; an aggregate must contain either the ordered provenance list or a stable reference to it.  A reciprocal electrical export therefore contributes two released observations and weight `w_1+w_2`, not an invented pair of directed edges and not a silently doubled single observation.

For an accepted row set `R`, required exact invariants are

$$
|T|=|R|,\qquad \sum_{r\in R}w(r)=\sum_{c,a,b}G_{c,a,b},\qquad
\{a,b:(c,a,b)\in G\}\subseteq V,
$$

and every provenance reference in an aggregate resolves to exactly one table observation.  With unique `o`, the sort key makes `T` unique for a fixed multiset of observations.  Addition of nonnegative mathematical integers is associative and commutative, hence `G` is independent of collection iteration order.

Canonical output must be a value built only from strings, exact integers, lists, and objects, serialized as UTF-8 JSON with recursively sorted object keys, compact separators, no non-finite values, and exactly one terminal LF.  Array order is the prescribed node/table/aggregate sort order.  Let `B` be these bytes and `H=SHA256(B)`.  Equality is byte equality (and 64-character lowercase hexadecimal equality for `H`), never a tolerance comparison.  Repeated serialization of the same canonical value therefore yields the same `B`; permutations of an in-memory collection yield the same `B` only when each observation retains the same stable `o`.

## Count comparison

For every primary-source statistic registered by the source lane, construct a named integer vector

$$
K=(|V|,|T|,|\{(c,a,b):G_{c,a,b}\ne0\}|,\sum_Tw,\sum_GG,…).
$$

The manifest must also name the source’s population filter, edge-layer convention, aggregation rule, and whether its reported count is observations, distinct pairs, or multiplicity.  C6 passes only componentwise exact equality against the corresponding vector from the frozen bytes.  A published `279`-style context value with a different population or layer is not a substitute.  No floating summary can discharge this comparison.

## Claim dispositions

| Claim | Math/graph disposition | Priority | Evidence boundary |
|---|---|---:|---|
| C1 | Exact raw-byte hash comparison is a well-defined precondition; no graph derivation may run before it. | P1 | Frozen URL/release/license/hash are source-lane evidence, absent here. |
| C2 | The closed domain above gives deterministic, fail-closed acceptance predicates. | P1 | Implementation must exercise all rejection fixtures. |
| C3 | The table and aggregate invariants prove conditional preservation and conservation for accepted rows. | P1 | Requires replay output and provenance check on frozen/fixture bytes. |
| C4 | Directional chemical normalization and unordered electrical normalization are derived above. | P1 | Source lane must verify class semantics; implementation must run hand fixtures. |
| C5 | Conditional theorem holds only with an immutable, stable `o`; see P0 below. | P0 | Contract must fix the meaning of ordinal before a permutation assertion is valid. |
| C6 | Exact integer comparison is defined, but no selected release/count vector is available in this lane. | P1 | Source lane and offline replay must supply it. |
| C7 | Structural-table outputs contain no state, dynamics, signs beyond supplied multiplicity, or behavior variable; the stated limitation follows. | P2 | Metadata/final report must carry the limitation. |

## P0 / P1 / P2 findings

- **P0-MATH-01 (C5):** `source_record_ordinal` cannot be assigned from the current parser iteration order while also claiming canonical bytes are invariant under a row-permutation fixture.  Reversing such input changes `o`, the prescribed table key, provenance, and bytes.  The parent claim is C5; it is not discharged until the contract/implementation selects a stable-ordinal rule and scopes the permutation accordingly.
- **P1-MATH-01 (C1/C6):** No frozen primary release, manifest, or registered integer summary vector exists yet, so source-hash and published-count equality cannot be independently checked.
- **P1-MATH-02 (C2--C4):** The formulas require focused fixtures for malformed UTF-8/schema/types, chemical reversal, electrical reciprocal observations, aggregation, and referential provenance.
- **P2-MATH-01 (C7):** Output metadata must state `structural_graph_only` (or equivalent) so a graph artifact is not relabeled as a functional simulation.

## Reproducible scratch witness

`artifacts/verify_graph_canonicalization.py` independently checks chemical direction, electrical unordered aggregation, row-order invariance when stable ordinals travel with observations, canonical byte/digest equality, and Boolean-count rejection.  It is illustrative only: it neither parses a release nor establishes C1/C6.

Run:

```powershell
python _workspace/ce/connectome-graph-replay-20260818/artifacts/verify_graph_canonicalization.py
```
