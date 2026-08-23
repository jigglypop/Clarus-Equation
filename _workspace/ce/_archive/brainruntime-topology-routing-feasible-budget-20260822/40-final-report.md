# What survived the topology test

Status: COMPLETE

The strong hypothesis did not survive. A return-aware topology mask reached
`16/16`, but so did a simple cluster mask, a path-only mask, a return-shuffled
mask, and the wrong-context mask. Consequently the result does not show that
the runtime chooses a task-specific graph morphology, uses cycle/return
structure, or dynamically routes by the present cue.

A narrower simulator result did survive: retaining roughly `8.1%` of learned
edges inside the declared two-block structural support removed enough
interference to improve factor transfer from full `10/16` and random `0/16`
to structured `16/16`, while preserving pairwise binding `16/16`. This is best
described as structured sparsity or block-support pruning. It is not yet a
motif dictionary, routing algorithm, memory geometry, or biological energy
law.

The next admissible mechanism, if pursued, must distinguish context alignment
at construction time—for example, independently varying source/destination
block roles so `WRONG_CONTEXT` produces a genuinely different required route.
Changing the current budget, thresholds, decoder, horizon, or seeds is not an
admissible rescue. Confirmation remains sealed.
