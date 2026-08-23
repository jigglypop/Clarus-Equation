# Final

Status: COMPLETE

Decision: `DEVELOPMENT_GO / ENDPOINT_CLOSED / CONFIRMATION_SEALED`

BA-TR7 showed that a uniform source-to-hidden substrate supplies no coordinate information: its first-arrival eligibility remains tied. BA-TR8 therefore tested an explicit new mechanism rather than retuning a threshold. It added a seed-only balanced perturbation to each source-to-hidden column, extracted only source-driven local eligibility, and used a hard winner-take-all rule with a one-use hidden-capacity state. The uniform controls still abstained, while all sixteen development seeds formed four-edge bijections. Removing capacity left a mean hidden collision fraction of `0.328125`; restoring it reduced the fraction to zero. Reversing experience order changed fourteen of sixteen allocations, so the result is path-dependent circuit formation rather than recovery of an invariant semantic code.

The mathematical result is conditional: distinct edge-level perturbations create local margins, and a penalty larger than the normalized evidence bound prevents reuse of an occupied hidden coordinate. The simulator result confirms that this construction is implemented with the actual delayed runtime and exact-delay eligibility. It does not show that the runtime contains biological lateral inhibition; the occupancy variable is an external hard local-capacity proxy.

No output or decoder endpoint was opened, because source-only observations cannot identify which hidden coordinate should mean which output. The seed supplies the missing symmetry-breaking information. Accordingly, this run supports only synthetic, outcome-blind, path-dependent source allocation. It does not establish semantic memory, graph morphology discovery, curvature-as-memory, cortical folding, biology, disease intervention, physical energy, or AGI.

Reproduction uses `runtime_source_seeded_competition.py`, its benchmark, the focused test, and the frozen hashes in `artifacts/source-freeze.json`. Confirmation seeds `100801..100832` remain sealed. The next falsifier must replace the hard occupancy proxy with an endogenous delayed competition/homeostasis state and test whether the same allocation remains identifiable without embedding a coordinate convention in an output codebook.
