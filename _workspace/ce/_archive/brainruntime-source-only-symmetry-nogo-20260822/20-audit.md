# Pre-implementation audit

Status: COMPLETE

Gate: PASS

The probe distinguishes two failure modes: no packet has arrived by tick $L$, while a real but permutation-symmetric hidden response arrives at $L+1$. It ends before decoding, so a negative endpoint cannot be confused with the mathematical selection failure.
