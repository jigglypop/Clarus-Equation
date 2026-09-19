# CE-TH1: microscopic thermal origin and macroscopic response

Additive follow-up to CE-CS1 at main commit `cbbc1cd930eb6c7ff42e0437aca261334f2a271d`.

- [Full Korean proof, assumptions, and limits](REPORT_ko.md)
- [Standalone calculation and 24 tests](thermal_origin.py)
- [Computed results](results.json)
- [Source provenance](provenance.json)

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python thermal_origin.py --output results.json
```

Exact finite energy-preserving system/bath collisions lead to conditional thermal
stationary states. The same partition functions determine purity, event-energy
bias, free energy and pressure. beta=0 recovers CS1 exactly. A separately labeled
local-bath extension selects Gibbs purity instead of conserving the initial weight.

No observational fitting. Temperature and diagnostic masses are supplied, not
predicted; no physical PMNS, baryon yield, cosmological vacuum or CMB fit is claimed.
The numerical precision is finite-model implementation precision, not natural-law
accuracy. Previous raw ZIP archives are not duplicated into Git by this commit.
