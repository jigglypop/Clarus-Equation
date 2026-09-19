# CE-CR1: continuum records and finite-work preparation

Follow-up to CE-RS1 at main `ac06cc8f1a8508d36d550b2ed8709c02c84d397c`.

- REPORT_ko.md: derivations, diagnostic inputs, results and limits.
- verify_continuum.py: standalone verifier and 27 checks.
- results.json: selected publication-time diagnostics; not observational accuracy.
- provenance.json: source identity and scope.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_continuum.py --output rerun.json
```

Conditional vacua / adiabatic preparations are a separately specified branch,
not a silent replacement of RS1's finite recurrent environment. The tanh mass
history is prescribed. Constants, spatial volume, outcome selection, local record
redundancy, total quantum Higgs backreaction and cosmological history remain open.
The standalone code regenerates the complete output including per-mode work checks.
