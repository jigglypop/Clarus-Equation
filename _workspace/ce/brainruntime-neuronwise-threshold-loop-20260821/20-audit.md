# Pre-implementation audit

Status: COMPLETE

Gate: PASS

## 1. Independent mathematics audit

The audit verified the type separation

$$
W\in\mathbb R^{q\times q},\qquad
\boldsymbol\vartheta,\boldsymbol\theta^-,\boldsymbol\theta^+\in\mathbb R^q,
$$

the elementwise selection/hysteresis equations, and the three-neuron frozen
outcomes. It initially required the scalar-broadcast theorem to be restricted
to the admissible $\theta^-<\theta^+$ domain, because legacy scalar configs can
still represent overlapping guards while the new vector path rejects them.

Revision 1 closed that issue, froze both mixed vector/scalar bit cases, required
finite effective counterparts, and added nonzero delay-buffer/index snapshot
continuation. The final mathematics audit returned PASS with no P0/P1.

## 2. Independent source-feasibility audit

The source audit confirmed:

- additive optional fields preserve existing scalar attributes and callers;
- `asdict(config)` necessarily gains three deterministic optional keys, so
  digest/schema growth is acknowledged rather than called byte-compatible;
- snapshot config deepcopy plus constructor restore needs no extra tensor state;
- vector-bit is not representable in the frozen Rust ABI;
- active-vector-only selection is valid on no-delay Rust because final Python
  selection overwrites the private Rust count.

It then found that initialization caching would break the current mutable-config
behavior. Revision 2 therefore requires use-time validation/materialization in
`_select_active`, `_step_torch`, and `_use_rust`, including post-construction
mutation gates. The final source audit returned PASS with no remaining P0/P1.

## 3. Implementation authorization

Only R1 is authorized, and only in:

- `reality_stone/python/reality_stone/clarus/runtime.py`;
- one new focused threshold test file.

The implementation must not alter Rust source/ABI, continuous cell equations,
weight semantics, A7 delay behavior, test values, tolerances, or the claim
ceiling. Any first failing test must be retained and classified before a
minimal correction.
