# Confirmation audit

Status: COMPLETE

- Route A mode: `confirmation`; Route B mode: `route_b_confirmation`.
- Both contain 32 unique seeds, exactly 98101--98132.
- Route A: Loop 6 32/32 GO, Loop 7 32/32 GO, Loop 8 0/32 GO, Loop 9 0/32 GO,
  Loop 10 32/32 GO.
- Route B: Loop 8B 32/32 GO, Loop 9B 32/32 GO.
- Loop 8B: clean mean 1.0, corrupt mean 0.9765625 and minimum 0.75, attractor-gain mean
  0.2995469252 and minimum 0.2391505018, control-advantage minimum 1.0.
- Loop 9B: held-out accuracy mean 1.0, shuffled accuracy mean 0, control-advantage minimum 1.0.
- Every audited temporal/hippocampal store count after cutoff and rollout is zero.
- Every Route A source-order, arrival-last negative-control, finite-weight, and snapshot-parity
  audit passed.
- Every Loop 7 context-precedence/zero-read and Loop 10 frozen-model/leakage audit passed.
- Maximum installed norm: Loop 8B 3.7462695; Loop 9B 2.7093623; both below bound 5.0.

Artifact hashes:

- Route A file: `2fd40c7e32f2ed8b143701bc517393b7df279d36a293483a6846279863726633`.
- Route A embedded results: `c3ddbe2bffc80f27690aa1fbddfb2bc713862fe790002184ab1b7971ff5df45c`.
- Route B file: `a4bc4821b19a735f7fd5451816934e8ad788f7a38ff79a34e28cdc0641919dc9`.
- Route B embedded results: `496c614d18b03a8898fd93156704c630d99b33da7d290c94d25b9f6d88a1b111`.
