# BA-TR24 implementation

BA-TR24 factorizes the relevance context itself. Context (c=(a,b)) selects
one event from source factor A (`S0/S1`) and one from factor B (`S2/S3`). Only
`00,01,10` are used for training. Count-normalized event use is projected onto
independent seed-specific orthonormal factor codes:

\[
\Theta^A=\sum_a \frac{U^A_{:a}}{n^A_a}(q^A_a)^\top,
\qquad
\Theta^B=\sum_b \frac{U^B_{:b}}{n^B_b}(q^B_b)^\top,
\]

\[
g_j(a,b)=\mathbf1[(\Theta^Aq^A_a+\Theta^Bq^B_b)_j>1/2].
\]

The heldout `11` compiler is constructed only from the separately observed A1
and B1 factor values. A joint lookup has no `11` entry and uses a frozen `00`
fallback. Gate training consumes no target, decoder, reward, output, endpoint,
or heldout row.

