# Evidence sources

Status: COMPLETE

No external source or new experimental result is used. The route map is based
on the committed V1–V8 implementation lineage, the preserved V8 failure, the
post-failure development route searches, and direct inspection of the current
residual-filter and free-rollout code.

The key implementation bottleneck is the current one-direction residual
filter with one pooled autoregressive persistence value and a deterministic
free rollout. The exhausted family changes the final readout of that same
predictive state; it does not improve the represented state itself.
