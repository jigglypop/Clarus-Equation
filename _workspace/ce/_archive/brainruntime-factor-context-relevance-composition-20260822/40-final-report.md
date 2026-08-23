# BA-TR24 final

Status: COMPLETE

Verdict: `SYNTHETIC_HELDOUT_FACTOR_CONTEXT_RELEVANCE_COMPOSITION /
DEVELOPMENT_GO / CONFIRMATION_SEALED`.

The relevance gate composed a context never presented as a joint case. After
training on `00,01,10`, it reconstructed the `11` packet mask and matched the
oracle on all 16 fresh seeds. A monolithic lookup with no `11` entry and either
single-factor cue shuffle failed on every seed.

This improves BA-TR23 from memorized context lookup to a narrow heldout
factor-composition statement:

\[
\text{seen }A_1 + \text{seen }B_1
\longrightarrow
\text{unseen joint relevance mask }g(1,1).
\]

The factor split and circuit support are still declared by the experiment.
The result does not show that the brain discovers those factors, infers context
from its own state, or transfers to a new graph.

Next falsifier: remove the supplied factor labels by learning a low-rank
context/event representation from mixed cues, then hold out both a context
combination and a remapped source coordinate. This must beat a direct joint
lookup and a coordinate-memorization control.

