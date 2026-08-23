# BA-TR22 three-event relevance no-go

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

BA-TR21 sums every locally valid arriving packet route. BA-TR22 asks whether
that rule can identify a desired two-event subset when a third event is
locally indistinguishable. The runtime factorized sum is generalized from two
to every active delayed packet coordinate without changing the per-column
competition rule.

For each frozen pair, one fixed distractor coordinate receives an exact copy
of a learned source-to-H column whose source is absent from that pair. The
three events arrive once at the same tick and have identical delay, gain, and
local support quality. The desired readout remains the original two-event
union; no context or relevance label is supplied to the runtime.

Fresh calibration seed: `109001`. Fresh development seeds: `109101..109116`,
opened only after calibration passes.

A valid no-go witness requires, per seed: pair-only route 4/4; copied columns
bit-exact; packet receipt `[0,0,0,3,0,0,0]`; exactly three H routes and the
three corresponding targets 4/4; desired pair alone 0/4; zero stores. Any
other outcome is apparatus/bound mismatch, not evidence for relevance.

Expected conclusion: without context, goal, or another relevance variable,
the desired pair and the matched distractor are locally exchangeable. Packet
factorization solves superposition but cannot decide which valid event matters.

Claim ceiling: a synthetic information/identifiability boundary. It does not
show how biological attention, inhibition, or context gating solves the
problem.

