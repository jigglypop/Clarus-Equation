# Validation

Status: COMPLETE

The focused command

` .codex\hooks\python.cmd pytest -q tests/test_realdata_transport_composition.py -p no:cacheprovider `

returned `3 passed in 4.32s`. The synthetic exact-semigroup witness passed,
and an independent-phase null exposed a pre-endpoint mean-regression loophole.
The frozen decision was therefore strengthened to require positive held-out
skill against the train-fold mean before opening the real result.

The real-data command

` .codex\hooks\python.cmd python -m reality_stone.clarus.realdata_transport_composition --output _workspace/ce/brain-realdata-transport-composition-20260822/artifacts/e17-transport-composition-results.json `

completed in `8.9s` over 11 sessions, 22 condition blocks, 1,532 trials, and
both the primary `dff` and sensitivity `branch` fields. The primary result was
`OBSERVATIONAL_TRANSPORT_COMPOSITION_STOP`, with one of three animal
aggregates satisfying all frozen criteria. `git diff --check` passed for the
module, focused test, and run directory.

SHA-256 receipts:

| item | SHA-256 |
|---|---|
| implementation | `1ca4633e74760bfcb647910783484d79777304f5cde22ad69fcd0fc8f015da97` |
| focused test | `bfd7fcc6f7894706cabb6630e4d94ca29913aaf913b11b68b4647c52e66ca438` |
| contract | `0808b13617b25086dd71508595fe5338813f9169e0498fa4a3b1d2c7ac4da3af` |
| machine result | `f03851f8e76404137fcd9d51a5e3a9ec051fa8c25302dcc4c5d6764c7f3238ab` |

The machine result itself records every source MAT path and source hash. No
full test suite was run because the change is isolated and the focused test
exercises its complete computational boundary.
