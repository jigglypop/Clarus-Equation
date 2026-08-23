# BA-TR10 validation

Focused command:

`.codex\hooks\python.cmd pytest tests/test_runtime_local_stochastic_binding.py tests/test_runtime_endogenous_competition_homeostasis.py -q -p no:cacheprovider`

Result: `8 passed` in 6.49 s; two pre-existing Torch sparse warnings.

Source compile:

`.codex\hooks\python.cmd python -m py_compile reality_stone/python/reality_stone/clarus/runtime.py reality_stone/python/reality_stone/clarus/runtime_local_stochastic_binding.py reality_stone/python/reality_stone/clarus/runtime_local_stochastic_binding_benchmark.py tests/test_runtime_local_stochastic_binding.py`

Result: PASS.

Calibration seed 98301: `CALIBRATION_PASS`; fresh noise-off mapping
`[0,1,3,2]`; minimum normalized column distance `.0259974431`.

Development seeds 98501..98516: `DEVELOPMENT_GO`, 16/16 all gates. There were
14 distinct learned permutations; source 0 selected every hidden coordinate
across the block. Mean minimum normalized column distance was `.0270839323`;
minimum strict activation margin was `.0007511477`. Deterministic/no-jitter and
no-learning controls abstained for every source in every seed. Homeostasis-off
was a bijection in only 4/16 seeds and had mean collision fraction `.21875`.
Confirmation seeds remain unopened.

