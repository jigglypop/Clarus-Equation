# BA-TR23 implementation

BA-TR23 adds the relevance variable required by the BA-TR22 no-go. Four
seed-specific orthonormal context codes are paired during experience with the
two source events relevant in that context. The only gate update is

\[
A\leftarrow A+e_cq_c^\top,
\qquad
g_j(q)=\mathbf1[(Aq)_j>1/2].
\]

The two selected packet coordinates are then supplied to the unchanged
source-factorized competition. Context is not injected into runtime
activation or the target decoder. Gate training reads only the context code
and co-occurring event coordinates; it reads no target, output, reward,
decoder, endpoint, or target-side weight.

The gate is a separate immutable snapshot whose hash is checked before and
after every probe batch.

