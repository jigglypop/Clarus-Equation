# Validation

Status: COMPLETE

Overall result: `STRUCTURED_SPARSITY_OBSERVED / CUE_ROUTING_STOP / TOPOLOGY_SPECIFIC_STOP`.

The repaired full and topology-masked pairwise binding baselines both passed
`16/16`. All finite-state, snapshot-immutability, store-cutoff, and exact
budget receipts passed. Binding budgets ranged from `105` to `138`, derived
from minimum admissible supports `420..552`.

Factor-transfer development results (`16` seeds):

| Arm | Success | Mean separation | Mean runtime-energy proxy |
|---|---:|---:|---:|
| `FULL` | 10/16 | -0.002245 | 0.663411 |
| `WEIGHT` | 14/16 | 0.137221 | 0.318104 |
| `CLUSTER` | 16/16 | 0.250592 | 0.480375 |
| `PATH_ONLY` | 16/16 | 0.251433 | 0.494000 |
| `TOPOLOGY` | 16/16 | 0.253374 | 0.488300 |
| `RETURN_SHUFFLED` | 16/16 | 0.249756 | 0.491119 |
| `RANDOM_MATCHED` | 0/16 | -0.021526 | 0.180964 |
| `WRONG_CONTEXT` | 16/16 | 0.251200 | 0.478446 |

Every sparse arm retained a mean `0.080986` of learned edges. `TOPOLOGY`
used about `26.4%` less simulator energy than `FULL`, but this is not physical
energy. Its mask differed from `PATH_ONLY` in all `16` circuits (mean
normalized Hamming `0.006162`), yet success tied `PATH_ONLY` and
`RETURN_SHUFFLED`. `WRONG_CONTEXT` also tied at `16/16`; therefore neither cue
alignment nor return/cycle placement was necessary for the observed gain.

Both registered GO gates are false. Confirmation was not opened. Machine
result SHA-256 `c68eb749c88d896b605fa4ab10a2c4cef52e485e872f6ebfa06eda54c7553d13`;
source freeze `928ba7a031246fe8f07bd26644fcc15dec2e45f3df7088ceba3f2d41cc490467`;
focused JUnit `acdda0b2ff1c8871980611835d45f2bcbb112029d5dac9c974f2539ef6df7860`.
