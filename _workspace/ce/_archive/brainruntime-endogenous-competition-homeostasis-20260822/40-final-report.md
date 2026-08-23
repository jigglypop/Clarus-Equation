# Endogenous delayed competition and homeostasis

Status: COMPLETE

Decision: `DEVELOPMENT_GO / ENDPOINT_CLOSED / CONFIRMATION_SEALED`

BA-TR8 could allocate four sources to four hidden coordinates, but it did so
with an external Boolean capacity vector and a hard winner function. This run
removed both objects from the state transition. A single persistent
`BrainRuntime` now receives the four experiences in sequence. The only supplied
symmetry break is BA-TR8's seed-only, balanced microscopic edge perturbation.

The new runtime mechanism acts on a packet only after the existing axonal delay
has actually delivered it. A used hidden coordinate attenuates its next packet
by $e^{-r_h}$. Each hidden coordinate then subtracts the strongest competing
coordinate and passes the remainder through a positive threshold. Equal inputs
therefore produce no winner, while a unique largest input leaves only its
positive margin. The runtime stores normalized squared activation one tick
later in a floating homeostasis vector. A decaying packet envelope prevents the
long tail of one source pulse from being counted as several independent
experiences. No source-to-hidden binding is fed back into this calculation;
the strict winner is read only after the first arrival for evaluation.

The fixed equations passed calibration seed `97091` and all 16 fresh
development seeds. Persistent homeostasis produced 16/16 four-source
bijections with zero collision. Removing only its attenuation by setting
$\lambda=0$ raised mean collision to `0.28125`. The uniform-weight control
abstained in every seed, as required by permutation symmetry. Hidden-row
permutation moved the complete arrival and homeostasis trajectory by the same
permutation, and restoring the midpoint snapshot twice produced identical
remaining trajectories. Each source pulse used the same persistent runtime and
required 53 zero-input ticks to meet the frozen washout criterion.

The strongest adverse result is equally important. A source-independent row
bias also generated a positional bijection in 16/16 seeds once homeostasis was
present. Because its four pre-history source columns are identical, that result
contains no source identity and is labelled `SOURCE_UNIDENTIFIED`. Thus the
evidence is not “homeostasis makes meaning.” It is narrower: source-specific
microscopic variation plus endogenous continuous competition can resolve
collisions without an external occupied mask.

This is a synthetic runtime result, not a biological or semantic theorem. The
seed still supplies coordinate-level information; the experiment does not
learn the candidate support, identify the meaning of hidden coordinates, or
open an output decoder. It therefore does not establish memory content,
topological motif discovery, curvature-as-memory, cortical folding, a disease
mechanism, physical energy, or AGI. Confirmation seeds `101801..101832` remain
sealed.

Reproduction uses the implementation, benchmark, focused test, exact commands,
interpreter versions, and SHA-256 values recorded in `30-implementation.md`,
`31-validation.md`, and `artifacts/source-freeze.json`.
