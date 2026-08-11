# Validation

Status: COMPLETE

## Mechanical result

- `uv run --extra dev python -m pytest tests/test_local_cloud_kernel.py tests/test_local_cloud_benchmark.py tests/test_dimensionless.py -q -W error`
  - `36 passed in 3.58s`;
- Ruff check: `All checks passed!`;
- Ruff format check: `4 files already formatted`;
- CE dimensionless checker: exit `0`, no output;
- default live certificate: `q = 0.9355555555555556 < 0.95` in
  `weighted_local_cloud_sup`.

## Frozen implementation hashes

| file | SHA-256 |
|---|---|
| `clarus/__init__.py` | `B0B0E9D99AB35B28017216728E115C8366B064E842FDBC9480F9E0519609A9D8` |
| `local_cloud_kernel.py` | `1F157E0CB9C4B41EFAD3FAEB934AF5502ED39C281F9AD7BEEB64ADEA75756ADD` |
| `local_cloud_benchmark.py` | `62BCFCDA99395F8DC2A2590FCD0224D5CF29D66BD1D9706541BA46305234A5BE` |
| `test_local_cloud_kernel.py` | `9D75CD55AAA2952C143B0DE6BBE58E5DD929A686AB5E041E4C89610E36C8BA34` |
| `test_local_cloud_benchmark.py` | `D5C3F6A685A1F5A15002BF7165ECC709A0F3C77DE63E611136BAAC7BEDE3318C` |

Public API import smoke test returned the same live contraction factor,
`0.9355555555555556`.

## Mathematical status

The small-gain certificate is a conditional theorem for the declared bounded synchronous map.
The composition test checks deterministic kernel composition. Neither establishes task utility,
whole-brain identity, SCC necessity, or AGI.

## Empirical status

All viewed engineering probes are target-aware and burned. Registered development remains
`0 seeds`; confirmation remains unopened.
