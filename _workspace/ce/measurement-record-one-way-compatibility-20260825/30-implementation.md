# Scoped implementation

Status: COMPLETE

No production channel, cosmology simulator, or physical $Z$ sector was implemented. The scoped artifact `artifacts/verify_one_way_record.py` checks the informative-projector counterexample, scalar-effect no-signalling control, exact finite-duration system--apparatus unitary, induced pointer Kraus operators and completeness.

The implementation deliberately keeps $Z_{\rm phys}$, record $R$, and persistent carrier measure $\mu_F$ as separate typed objects. It does not implement a retention map, energy scale, stress tensor or dark-sector source.
