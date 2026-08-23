# Pre-implementation audit

Status: COMPLETE

Gate: PASS

The contract preserves M1 delayed binding and the prior T1 11/16 transfer
STOP, while treating delay plus heterogeneous thresholds as a new apparatus.
The new mechanism is structural routing; it does not retune the predecessor's
decoder, seeds, or endpoint.

Route construction is restricted to sealed $W$, cue, block indices, public
seed, route name, and budget.  Targets, labels, decoder output, post-rollout
state, endpoint score, and stores are forbidden.  Exact budget, off-diagonal
support, and fail-closed degenerate behavior are defined before execution.

`PATH_ONLY` and `RETURN_SHUFFLED` isolate the return-support term.  A
topology-specific result requires a nonzero mask difference and superiority
over both.  Otherwise the strongest permitted positive classification is
cue-conditioned cluster/path routing.

The raw normalized-runtime tensors are dimensionless without fitted
normalization.  Runtime energy and active edge/node counts remain simulator
proxies, not physical energy.  Biological, clinical, anatomical, AGI, and
complete motif-dictionary claims are excluded.

Implementation is authorized only with all named `V-INPUT`, `V-BUDGET`,
`V-DEGENERATE`, `V-SNAPSHOT`, `V-CUTOFF`, and `V-FINITE` checks reported in
`31-validation.md`.  Failure of any such check invalidates the apparatus and
does not authorize tuning.
