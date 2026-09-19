# CE-RV1 — Records to conditional values

A continuation of CE-RH2 / CE-RS1 / CE-CR1. This package calculates a finite-resolution record of the existing portal operator O, the conditioned **joint quantum** state, its Higgs-proxy magnitude distribution and subsequent unitary evolution. It does not prescribe a Higgs time history or factorize the Higgs/environment correlation.

Read **REPORT_ko.md** for the derivation, input ledger, numerical scope and counterexamples. **verify_record_values.py** is standalone. **results.json** combines the two executed sections (40 checks); **static_results.json** and **time_results.json** retain the actual execution records. **provenance.json** records source hashes and publication status.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_record_values.py --section all --output rerun.json
```

This is a finite-mode signed real Higgs proxy, not the SM Higgs doublet or a renormalized continuum QFT. The initial ground-state boundary condition, Hamiltonian parameters and Gaussian detector instrument are inputs. The reported values are conditional means; their nonzero conditional variances are reported, not replaced by numerical error bars. Born conditioning is used rather than derived. No observed constants are fitted, and fundamental constants are not predicted. New changes were not pushed to main during this execution.
