# Loop 2c — matched structural manifold preregistration

Status: COMPLETE

Loop 2b is preserved as `STOP`: every plastic mode incurred an approximately
`4.77` weight jump because the first nonzero update applied RowNorm+TopK to a
dense initialization, while OFF did not.

Question: after every arm starts from the exact same projected weight
`Proj(W0, density=0.25)`, does signed causal improvement credit outperform OFF,
legacy critic derivative, sign flip, absolute signal, trace-off, and shuffled
signal while passing the frozen held-out guard?

Only registered change from Loop 2b:

\[
W_0^{all\ arms}=\operatorname{Proj}(W_{dense},0.25).
\]

Seeds, stream, learning rate, interval, density, windows, comparisons, and hard
gates remain unchanged. Loop 2b artifact is not overwritten. Failure remains a
failure and no further learning-rate or threshold sweep is allowed in this
route.

