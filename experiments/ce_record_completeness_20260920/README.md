# CE-RC1: record completeness and independent-selection audit

Read `../../paper/후속연구_기록과_상태선택/16_기록_완비성과_독립적인_자연값_선택조건.md`.

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_completeness.py --output results.json
```

The 34 checks cover complete position/time records, general local kinetic metrics,
angular-sector probes in the existing CE model, and explicit nonuniqueness of
self-generated-record selection. The maximal-O(7) alternative does not preserve
the nonzero-splitting CE branch. No observational data or natural-constant
prediction is claimed. `as1_reference.py` is an unmodified copy of the previously
published AS1 numerical helper. `results.json` and `run.log` are actual execution
outputs. The derivations' continuum conditions and numerical scope are stated
in the chapter; finite-basis convergence is not a rigorous continuum error bound.
