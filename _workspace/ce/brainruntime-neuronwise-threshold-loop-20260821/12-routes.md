# Routes

Status: COMPLETE

## R1 — additive optional threshold vectors (selected)

- keep all scalar fields unchanged;
- add three optional finite length-`dim` tuples;
- resolve effective device tensors from the current mutable config at each use;
- use them only in bit hysteresis and outer eligibility;
- rely on snapshot config deepcopy and re-resolve/validate effective vectors at
  each use after restore, with no cached tensor snapshot state.

This is the smallest route that implements the requested heterogeneity without changing the
continuous dynamics or existing scalar consumers.

## R2 — replace scalar fields by scalar-or-sequence union (rejected)

This would break callers that cast the scalar config field with `float(...)`, complicate
serialization and make legacy equality behavior ambiguous. It offers no mathematical benefit
over R1.

## R3 — silently reduce vectors to a scalar for Rust (rejected)

Mean/min/max projection changes which bit guard fires. It violates (T4) and would create false
backend parity. The selected behavior is Torch fallback for `auto` and explicit-Rust rejection.

## R4 — Rust ABI vector expansion (deferred A8-D/Rust route)

This requires binding/kernel signature changes, extension rebuild, and separate delay semantics.
It is outside A8-T and must use the A7-H backend counterexample as an unchanged regression gate.

## R5 — empirical threshold estimation (blocked)

Requires calibrated per-neuron observations, identity/provenance, units, intervention or a
prespecified observational estimand. Synthetic config tests cannot supply those data.

## Stop/revision order

1. audit the frozen API/equations/backend contract;
2. implement R1 only after PASS;
3. run the new focused test file and the existing scalar Rust parity node;
4. preserve any first failure before a source-only or formula revision;
5. stop on silent Rust vector-bit execution or scalar incompatibility;
6. only after independent audit update the ledger, then the narrative paper.
