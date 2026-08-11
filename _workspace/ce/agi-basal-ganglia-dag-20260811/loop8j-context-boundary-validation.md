# Loop 8J validation — surprise-gated directional context update

Status: LOCKED VALIDATION RUN ONCE — 50/100 STOP

The candidate satisfied its exact algebraic identities, was nonexpansive,
preserved the eligibility-orthogonal state component to floating-point error,
remained finite, and retained the large NLL improvement over the hard parent.

ID post-switch accuracy improved from no-reset `0.4560` to `0.5980`; paired LCB
was `+0.1080`, above the locked `+0.08`. OOD post-switch accuracy instead fell
from `0.2977` to `0.2908`; LCB was `-0.0230`.

The candidate also reduced overall accuracy versus the hard parent and damaged
the matched stationary condition by `0.1006`. Generic forgetting and full
negative reset had higher post-switch accuracy, so selective directional reset
was not uniquely superior.

Diagnosis: a negative outcome is not a unique context-boundary observation.
Especially under OOD sensory noise, many errors originate in uncertain content
bits rather than a stale context route. Confidence-gated reset assigns these
errors to context memory and deletes useful state. The next admissible route is
cause-specific credit assignment using the already available factorization
`P(context) P(base|evidence)`, not another reset threshold or gain sweep.
