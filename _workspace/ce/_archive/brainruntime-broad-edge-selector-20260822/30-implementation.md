# Implementation

Status: COMPLETE

BA-TR6 is isolated in a new module and leaves BA-TR3--TR5 unchanged. Each factor starts from the inherited delayed runtime and common $H\to Y$ payload trunk, then replaces its entry substrate by all 32 edges from $S_0\cup S_1$ to one shared hidden block, each with frozen weight one.

The edge field has shape $32\times2$. Every episode normalizes positive exact-delay eligibility over the 32 candidates, and each cue column is then divided by its own exposure count. The compiler accepts only `(gate_snapshot, factor_cue, candidate_support, trunk_support)`, rejects a fourth/fifth tie, selects four candidates, and adds the four-edge trunk. It cannot read recurrent weights or either old branch family.

The endpoint uses two independent factor snapshots and a compact hash receipt for the exact $12\times12=144$ Cartesian conjunction. `WEIGHT_ONLY` and cue-column-mean `POOLED_STATIC` remain pre-endpoint abstention receipts; neither receives an arbitrary tie break.
