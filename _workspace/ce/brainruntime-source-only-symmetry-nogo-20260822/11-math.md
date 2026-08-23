# Mathematics

Status: COMPLETE

The runtime computes activation before neuronwise active and bit thresholds. Exact-delay eligibility observes activation, not bitfield. Therefore the heterogeneous threshold vectors cannot break hidden activation symmetry at the first arrival tick. They may alter later lifecycle or bit states, but that is a fixed coordinate bias independent of source identity.

With uniform $W_{hs}=1$, zero initial hidden state, and shared scalar source drive, hidden permutations commute with the first-arrival map. The candidate eligibility field is invariant under every permutation of the four hidden rows. A deterministic top-four ordering would insert coordinate information not present in the experience.

All runtime states, weights, eligibility values, and score gaps are dimensionless. Tick and support counts are unitless. No endpoint statistic enters the result.
