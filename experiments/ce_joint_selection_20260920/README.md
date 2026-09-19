# CE-JS1 — joint Higgs/holonomy/radius gates

Read REPORT_ko.md before using these formulas. This is an explicit product-circle bulk one-loop completion, not the full CE parent and not an observational fit.

Run from the repository root:

```bash
python -m pip install -r experiments/ce_joint_selection_20260920/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/ce_joint_selection_20260920/verify_joint_selection.py --output results_rerun.json
```

The sibling `ce_holonomy_selection_20260920/verify_holonomy.py` is required for the independent inherited-kernel comparison. There are 39 checks, including negative controls. `results.json` contains the actual run, not a natural-constant prediction. The time integration is local finite mechanics; its Higgs-frequency derivative-expansion gate fails, so it is not promoted to a real-time QFT or cosmological prediction.
