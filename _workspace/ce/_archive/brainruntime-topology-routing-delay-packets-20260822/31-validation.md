# Validation

Status: COMPLETE

Overall result: `DELAY_PACKET_FIX_PASS / ROUTING_BUDGET_APPARATUS_INVALID / TOPOLOGY_UNTESTED`.

Focused delay tests passed `4/4`; five adjacent snapshot, threshold-backend,
and no-delay Rust parity tests passed `5/5`. The repaired delayed full M1
baseline then passed clean and corrupt binding in `16/16`, with snapshot,
finite-state, and store-cutoff receipts intact.

The registered topology binding arm passed seed `97201` (`clean=1.0`,
`corrupt=1.0`, `426/426` retained) and stopped before scoring seed `97202`:
its restricted cluster/path support contained fewer edges than the frozen
budget `ceil(0.25*nnz(W))`. The constructor correctly returned
`APPARATUS_INVALID`; no factor-transfer endpoint and no confirmation seed was
opened.

Machine result `artifacts/development-results.json`, SHA-256
`a419d9e68c9f031895a1eae860704a2bc343e284091b74aaa4b977e34f2dec00`.
Original freeze `fd6d88fcb75a3d92c9ff94f556934732d727425e7bc9448d8e864c8dbe77cd08`;
outcome-blind runner revision receipt
`822e9a9c68b14c777205307d1c12a762c86c35ff37c042510fb4be4ecd0b9c3d`.
