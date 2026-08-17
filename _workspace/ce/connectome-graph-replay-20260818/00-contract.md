# Research contract: C. elegans connectome graph replay MVP

Status: COMPLETE

PREDECESSOR: none

## Question

Can Clarus-Equation deterministically replay a frozen public adult-hermaphrodite *C. elegans* structural connectome from source bytes into a canonical graph, and independently verify transport, schema, graph integrity, and published-scale summary statistics?

## Scope

This run covers structural graph replay only. It does not claim to reproduce membrane dynamics, synaptic signs or weights beyond the released contact counts, neural activity, behavior, learning, an individual animal, or a human connectome.

The source lane shall select one publicly accessible primary or project-authoritative adult-hermaphrodite release and freeze its URL, release/materialization identifier, license or reuse terms, raw SHA-256, and row/column semantics. The exact included-cell population is defined by that frozen release and must be stated explicitly; the familiar “302 neurons” figure is context, not an unchecked hard-coded acceptance value.

## Definitions

- A source row is a released connection observation with identified pre/post cells, connection class, and a nonnegative integral multiplicity where supplied.
- A chemical edge is directed from presynaptic to postsynaptic cell.
- An electrical/gap-junction pair is canonicalized as an unordered cell pair while retaining its released multiplicity and source provenance. The implementation must not silently double reciprocal exports.
- The canonical node table is sorted by stable source identifier.
- The canonical connection table is sorted by `(connection_class, endpoint_a, endpoint_b, source_record_ordinal)` after endpoint normalization appropriate to the class.
- The aggregate neuron graph sums multiplicities only for rows with the same normalized endpoints and connection class. Individual source observations remain auditable.
- Canonical bytes are UTF-8 JSON with sorted object keys, compact separators, finite JSON values, and a single terminal LF.

## Registered claims

| ID | Claim | Required evidence |
|---|---|---|
| C1 | Frozen source bytes are the only authority for replay. | URL/release/license/raw SHA-256 manifest and hash check before parse. |
| C2 | Parsing is deterministic and fail-closed. | Exact schema/type/domain checks; duplicate, missing endpoint, negative/nonintegral count, unknown class, and malformed encoding rejection fixtures. |
| C3 | Canonical replay preserves every accepted source observation and its provenance. | Source-row count, canonical-row count, multiplicity sums, endpoint referential integrity, and canonical digest. |
| C4 | Chemical direction and electrical pair normalization match the frozen source semantics. | Small hand-computed fixtures plus source documentation. |
| C5 | Canonical output is independent of input row order and repeated execution. | Permutation fixture and repeated byte-equality/digest assertions. |
| C6 | Full-release replay matches the registered exact dataset counts and primary-source summary values selected by the source lane. | Offline replay evidence from frozen bytes; exact integer equality. |
| C7 | The result is a structural graph artifact, not a functional brain simulation. | Explicit output metadata and final-report limitation. |

## Implementation boundary

- Prefer the Python standard library for the MVP; no NetworkX or pandas dependency is required for this dataset scale.
- Separate download/acquisition from replay. Tests and verification never access the network.
- Raw third-party bytes are not silently modified. If licensing or repository size prevents committing them, the manifest and an explicit acquisition command are committed while full replay uses a user-supplied cached file.
- Repository outputs shall include a machine-readable manifest, a deterministic replay CLI/library boundary, a tiny redistributable fixture, and focused tests. Exact paths are selected after repository mapping and frozen by the audit before implementation.

## Acceptance criteria

1. All C1–C7 claims have source, math, and audit dispositions with no open P0.
2. Manifest and canonical-output SHA-256 comparisons are exact lowercase hexadecimal equality; all graph counts are exact integers with booleans rejected as integers.
3. The replay rejects any source hash mismatch before parsing.
4. The focused offline test covers schema rejection, directionality, gap-junction normalization, aggregation, row-order invariance, and canonical byte stability.
5. A full-release command is documented and either executed against frozen bytes or reported BLOCKED with the missing external byte object and exact resume condition. A fixture-only pass cannot be described as full-connectome reproduction.
6. Validation remains scoped to the new replay test and source-only checks unless a shared dependency boundary is changed.

## Allowed error

No numerical tolerance is used for transport, counts, multiplicities, identifiers, or digests. Any optional floating summary is informational and cannot satisfy an acceptance criterion.
