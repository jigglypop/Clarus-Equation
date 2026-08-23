# Validation

Status: COMPLETE

Overall development result:

`GO / SYNTHETIC_CONTEXT_ENTRY_BRANCH_IDENTIFIED`

All `16/16` frozen development seeds passed every pre-endpoint gate and the
per-seed decision rule.  Confirmation remained sealed.

| Route | Mean accuracy | Opposite-payload delivery | Mean runtime-energy proxy | Mean active fraction | Edges |
|---|---:|---:|---:|---:|---:|
| `CORRECT` | 1.000000 | 0.000000 | 0.200115 | 0.096429 | 12 |
| `WRONG` | 0.000000 | 1.000000 | 0.200115 | 0.096429 | 12 |
| `SWAPPED_AFTER_CUE` | 0.000000 | 1.000000 | 0.200115 | 0.096429 | 12 |
| `STATIC_0` | 0.500000 | 0.500000 | 0.200115 | 0.096429 | 12 |
| `STATIC_1` | 0.500000 | 0.500000 | 0.200115 | 0.096429 | 12 |
| `RANDOM_MATCHED` | 0.296875 | 0.296875 | 0.225246 | 0.110268 | 12 |
| `FULL` | 0.000000 | 0.000000 | 0.253479 | 0.125000 | 16 |
| `STATIC_UNION` | 0.000000 | 0.000000 | 0.253479 | 0.125000 | 16 |

Correct and wrong routes have exactly the same edge count, delay histogram,
threshold profile, STP rule, decoder hash, mean activity, and simulator-energy
proxy.  Wrong routing therefore did not fail by silencing the network: it
delivered the other simultaneously present payload on every trial.  Each
static branch succeeded on exactly the matching half of contexts.  Opening
both branches delivered an unresolved two-payload mixture and failed the
frozen unique-output margin.

Every learned block had four edges and operational rank four.  Both two-edge
products had minimum singular value at least `0.7071068287`, above the frozen
`0.25` floor.  Actual weight change outside the declared support was zero;
hippocampal and temporal rows were zero at cutoff; the delay ring and activation
were zero before recall; dense/sparse parity and snapshot immutability passed.

Revision 2 scored fields are identical to Revision 1 for every trial.  The
only numerical reporting change is an explicit two-hidden-block receipt.

Validation commands:

```powershell
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests\test_runtime_context_branch_routing.py
.codex\hooks\python.cmd python -m reality_stone.clarus.runtime_context_branch_routing_benchmark --stage development --output _workspace\ce\brainruntime-context-branch-routing-20260822\artifacts\development-results-r2.json
```

Focused tests: `3 passed`.  `git diff --check` passed.  PyTorch emitted only
its existing sparse-CSR beta/invariant warnings.

Evidence hashes:

- development result: `bc8dd6f1f884e500f691fa4cddd125a11f442201bfd422ccd576d231e0abe2c5`
- focused JUnit: `5410690155999b93b41a54f395117be2ae75626f479c6359339aab9dc5001bdb`
- source freeze: `9b68715a724c06cf51e78364cf5a4cd83462e1b18ffdbcaf5a7bfd5e184ff302`

The energy column is a dimensionless runtime proxy, not a physical energy
measurement.

