# Scoped implementation record

Status: COMPLETE

No production cosmology code was changed. The approved implementation is the reproducible audit program `artifacts/verify_measurement_wall.py`.

It checks the exact qubit dephasing solution, idempotence and Kraus completeness of $\mathcal D_P$, Kraus completeness and positivity of $\Phi_\eta$, trace preservation, the finite-strength bound $\eta<1$, the arbitrary-POVM sandwich counterexample, and two opportunity-cost identities. The implementation deliberately contains no conversion from information cost to energy or cosmological density.

The accompanying `artifacts/dimensionless-audit.md` records the dimensions of every exponential, logarithm, rate, and proposed wall-weighted functional.
