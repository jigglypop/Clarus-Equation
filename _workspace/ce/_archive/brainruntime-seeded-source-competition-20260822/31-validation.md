# Validation

Status: COMPLETE

Focused command:

```text
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests\test_runtime_source_seeded_competition.py
```

Result: `2 passed` in 3.42 seconds. The two warnings were PyTorch sparse-CSR beta/invariant warnings already emitted by the runtime constructor; no test failed.

Development command:

```text
.codex\hooks\python.cmd python -m reality_stone.clarus.runtime_source_seeded_competition_benchmark --stage development --output _workspace\ce\brainruntime-seeded-source-competition-20260822\artifacts\development-results.json
```

Result: `DEVELOPMENT_GO`. All 16 seeds passed every apparatus gate; the capacity arm formed 16/16 bijections and reduced mean collision fraction from `0.328125` without capacity to `0.0`. Uniform raw and uniform competition controls abstained in 16/16 seeds. Reversing source presentation order changed the particular binding in 14/16 seeds while preserving a bijection, demonstrating path dependence rather than invariant output meaning. Every endpoint remained closed and all 32 confirmation seeds remained sealed.

Frozen hashes:

- module: `e67287834004bada9e4e8850198032e338003573698c58c6a92cdf76ba6303f5`
- benchmark: `ac897cf12b4db0e74c0f97dd53ba3a4dbefd629730c4c381ae6748987ab1831f`
- focused test: `1e2a3275a515901d6757bc69ee861e1e2814b38fe37a2224a1e4f78fac3e1222`
- development result: `a323d9927d39229c41094db32b49bbb2a73a9cb7e6c730865fd9020f3c8fa9fe`

Interpreter and dependency versions are frozen in `artifacts/source-freeze.json`.
