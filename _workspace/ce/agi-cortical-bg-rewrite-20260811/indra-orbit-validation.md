# Indra orbit-quotient validation

The deterministic finite-cover validation expanded a fixed three-orbit kernel to
full networks of 6, 9, 15, 27, and 51 nodes.

Results:

- quotient dimension stayed exactly `3` at every full size;
- maximum nonlinear closure error in (F_A L=L F_{\bar A}):
  `1.1102230246251565e-16`;
- maximum full-versus-lifted fixed-point error:
  `2.220446049250313e-16`;
- an explicit unequal row-block perturbation was rejected as non-equitable;
- a radius-12 bidirectional causal cone contained exactly `25` nodes;
- an absolute active budget capped the same cone at exactly `10` nodes and
  reported exhaustion.

The infinite-chain counterexample was also reproduced:

- finite open chains of sizes 4, 8, 16, and 32 all returned extinction `1.0`;
- the translation quotient ̄A=`[2]` returned extinction
  `0.2031878699799794`;
- periodic translation-equivariant covers matched the quotient with maximum
  error `0.0`.

All seven registered algebraic/mechanistic gates passed. Verdict: `GO`.

This GO validates quotient closure, bounded finite-horizon work, and the failure
of naive finite-SCC reasoning on an infinite escaping chain. It is not yet a
dynamic neural-runtime or AGI performance result. The next implementation must
add tied spatial shifts, per-edge causal delays, a symmetric quotient background,
and a sparse localized-deviation cone, then compare full covers of increasing
size against the fixed quotient trajectory.
