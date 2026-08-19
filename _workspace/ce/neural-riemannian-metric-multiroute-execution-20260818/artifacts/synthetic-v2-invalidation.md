# Synthetic v2 invalidation ledger

Status: INVALID_IMPLEMENTATION

The replaced `run_synthetic_suite.py` source has SHA-256
`AEF26F421E8BFC4E0544FCBB2473AD9049ECDD64EE059CD7515A8649FA6D7144`.
Its unexecuted v2 design is incomplete and must not be run or promoted.

Reasons: its metric baseline used an arbitrary scaled inverse covariance
rather than a fixed likelihood family; G2--G6 had no fixed 100-case
any-Holm false-positive numerator; fitting collapsed the direct comparator
into an intervention-biased common estimate; and no C2 finite-difference
curvature calculation was implemented. The retained v1 invalid result and
hash are unaffected by this ledger.
