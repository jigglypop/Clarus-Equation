# BA-TR22 final

Status: COMPLETE

Verdict: `THREE_EVENT_RELEVANCE_NO_GO_CONFIRMED / ENDPOINT_CLOSED /
CONFIRMATION_SEALED`.

Packet factorization solves superposition but not relevance. When three
locally valid routes arrive, the equation correctly preserves all three. It
has no information with which to select the externally desired pair:

\[
\{p_i,p_j,p_d\}\;\xrightarrow{\text{local factorization}}\;
\{y_i,y_j,y_d\},
\]

while the task asks for only ({y_i,y_j}). Because the distractor has the
same delay, amplitude, and copied local weight column, the desired subset is
not identifiable from packet and synapse-local variables alone.

The next equation needs a relevance variable, not a stronger WTA. A minimal
test is an experience-learned context-to-packet gate trained only from context
and event co-occurrence, then frozen before these three-event probes.

Claim ceiling: a synthetic identifiability boundary. It neither proves a
biological attention mechanism nor says that all distractors in real brains
are locally exchangeable.

