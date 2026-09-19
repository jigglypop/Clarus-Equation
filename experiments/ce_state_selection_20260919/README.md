# CE-CS1: microscopic preparation of the shared qutrit state

Conditional model of identical isotropic environmental collisions. It derives
an existing CE symmetric ensemble as a dynamically attained stationary state
within a conserved SWAP sector, with a spectral error bound. It does not select
the numerical initial sector weight or derive PMNS, baryon abundance or vacuum
energy. The physical bath counting law is not the legacy Poisson(3+delta) law.

- [Full Korean proof and assumptions](REPORT_ko.md)
- [Standalone calculation and 20 tests](state_selection.py)
- [Computed results](results.json)

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

No observational parameter optimization is performed. gamma sets a diagnostic
time unit, not a fitted cosmic timescale. Original input alpha_s=0.11789 is
inherited, not predicted. The optional provenance directory in the conversation
ZIP contains original source extracts, not new reruns of CMB calculations.

This directory is additive to the existing main branch and does not replace
its separate Higgs–Clarus Gaussian portal research.
