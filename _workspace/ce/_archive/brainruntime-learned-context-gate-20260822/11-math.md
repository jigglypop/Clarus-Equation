# Mathematics

Status: COMPLETE

The context vectors $q_c$, exact-delay eligibility $E$, branch-use vector $u$, gate matrix $\Theta$, logits $\ell$, runtime activations, and recurrent weights are dimensionless. The clip bounds and logit margin are therefore dimensionless; the argmax and mask cardinality are discrete and unitless. No physical-time or joule quantity is introduced.

The update $\Delta\Theta=uq^{\mathsf T}$ has shape $2\times4$. It is a local coactivity association between a context afferent and one of two fixed gate actuators. Because $q_0^{\mathsf T}q_1=0$, balanced experience gives, before clipping,

$$
\Theta q_c=\sum_{n:c_n=c}u^{(n)},
$$

so the score for one context is not contaminated by the other context code in exact arithmetic. This proves only the algebraic separation of the two stored associations. It does not guarantee that the physical experience produces a larger correct-branch $u_b$; that is a pre-endpoint empirical receipt.

The map $\arg\max(\Theta q_c)$ is discontinuous at a tie. No derivative through the mask is claimed. Ties fail closed rather than being resolved by an outcome-selected rule. The fixed branch actuators are an architectural prior, so a successful test is association learning over two declared actions, not topology discovery.

Agreement with the environmental bijection $\sigma_s$ does not by itself identify this computation: a hidden lookup, including the known seed-parity rule, could produce the same mask. Identifiability therefore requires a separate frozen-state witness. Recomputing the branch from serialized $\Theta,q$, changing the seed and every schedule-derived value at fixed $\Theta,q$, prohibiting those values from the compiler signature or closure, swapping only the two cue vectors, and row-swapping $\Theta$ at fixed cues distinguish the declared map $q\mapsto\arg\max(\Theta q)$ from captured $\sigma_s$, cue-only, or seed-only lookups.

Counterexamples retained as controls: $\Theta=0$ cannot identify a branch; pairing each physical experience with $q_{1-c}$ learns the reversed association; a context-independent branch succeeds on only one of two balanced contexts; opening both branches produces the BA-TR3 interference failure.
