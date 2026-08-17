# Routes — C. elegans connectome graph replay MVP

Status: COMPLETE

## Trigger

Route work is required because `P0-MATH-01` in `11-math.md` is open.  Target: preserve every released observation and its source provenance while producing byte-identical canonical output for permitted permutations.  The target is structural replay only; no route supplies functional or behavioral semantics.

## Candidate routes

| Route | Mechanism | Free choices / dof | Target-aware? | Independent verification / rejection condition |
|---|---|---:|---|---|
| R1 | Define `source_record_ordinal` as the zero-based ordinal in the frozen raw source byte stream.  A permutation fixture permutes already parsed records while retaining that field. | 0 after byte format is frozen | No | Parse fixture, permute record container, and require table bytes/digest equal. Reject on lost/duplicate/out-of-range ordinal. |
| R2 | Require an authoritative source-provided immutable record identifier and use it as `o` (or deterministically map it to an ordinal). | 0 if identifier is unique; otherwise undefined | No | Validate uniqueness and stable mapping from frozen bytes; reject absent/nonunique identifiers. |
| R3 | Replace ordinal provenance with a canonical raw-record digest plus a multiplicity/provenance multiset, and change the contract’s table-key wording accordingly. | 1 design choice: exact raw record projection | No | Independently permute records and compare bytes; reject digest collisions unless complete raw bytes provide a deterministic tie-breaker. |

R1 is the minimal route because the contract already names `source_record_ordinal`, preserves source position as provenance, and permits an in-memory permutation test without redefining source rows.  It is not a claim that reordering the frozen raw file itself preserves identity: that operation changes the source byte authority and, under R1, the record ordinals.

## Required resolution

Before implementation, audit must select R1, R2, or R3 and state the exact fixture scope.  For R1 the acceptance sentence is: “Permuting the parsed observation container while retaining each frozen-source ordinal leaves canonical bytes and SHA-256 unchanged.”  The implementation must reject duplicate ordinals, preventing a hidden sort tie.  If the intended claim instead permits raw-file row permutations with no retained identity, R1 is invalid and C5 stays open until R2 or a revised R3 contract is adopted.

No additional routes are warranted: the P0 is an identity/provenance definition conflict, not a numerical fit or a target-selection problem.  C1/C6 remain P1 evidence dependencies for the source and replay lanes, and C7 remains the P2 metadata task recorded in `11-math.md`.
