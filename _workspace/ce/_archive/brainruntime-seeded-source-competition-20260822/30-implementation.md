# Implementation

Status: COMPLETE

The isolated module `runtime_source_seeded_competition.py` builds a zero-output delayed Torch runtime with exactly sixteen positive $H\leftarrow S$ candidates. Each candidate column receives a seed-only balanced edge code. Four source-only episodes produce first-arrival exact-delay eligibility, after which the pure `allocate_source_bindings` function applies hard WTA and an occupied-capacity state.

The implementation never imports or calls a payload book, learned output trunk, decoder, reward, task mapping, or endpoint. Uniform, source-independent bias, no-capacity, hidden-row permutation, and changed-order controls use the same observation and allocation code. A read-only stable-snapshot audit found no P0/P1 mismatch within this endpoint-closed scope.
