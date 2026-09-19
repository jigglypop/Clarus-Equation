# CE-AS1 — conditional action selection

This study determines the kinetic and local-potential coefficients compatible with specified joint-vacuum records in the inherited finite CE model. It does not derive natural constants without independent records.

See [Chapter 15](../../paper/후속연구_기록과_상태선택/15_기록에_의한_작용계수의_유일성과_진공부문.md) for definitions, proofs, assumptions and negative controls.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_selection.py --output results.json
```

The script is standalone. It constructs synthetic vacuum records, does not receive target coefficients in its identification functions, and runs 35 checks. It also checks excluded-sector predictions, missing interactions, finite-basis leakage and record-noise sensitivity.

The stored results were generated from the inherited diagnostic inputs, not experimental measurements. A clean-directory rerun produced the identical JSON in the recorded environment.
