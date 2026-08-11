# Validation lane

Status: COMPLETE

No experiment was executed. Required first gates:

1. Different actions produce measurably different predicted next-state distributions.
2. Future-state poison cannot affect current belief or action.
3. Posterior covariance changes correction trust in the predicted direction.
4. H2/H3 planning beats the current cosine-reactive action rule on held-out seeds.
5. Signed TD eligibility improves return while a separate signed homeostatic rule keeps activity bounded.
6. Full augmented belief-state stability is checked, not component spectral radii alone.

