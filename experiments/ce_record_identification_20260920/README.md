# CE-RI1: Spectral records, response reconstruction and selection gates

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_identification.py`.

The package is standalone with NumPy, SciPy and mpmath. `inherited/ce_rv1.py`
is an unchanged copy of the supplied CE-RV1 implementation. `REPORT_ko.md`
contains the derivations, scope distinctions and negative controls.

No observed data, no observational fitting, no selection of fundamental
constants. Synthetic spectral records recover parameters of a prescribed
free-field bath. Separately, a finite quantum Higgs-proxy model recovers local
coefficients from conditional moments and independently computed acceleration.
The 4D field and finite-mode normalizations are not silently identified.
36 checks were executed. Prior test totals are not added to the new total.
The remote repository was read but not modified.
