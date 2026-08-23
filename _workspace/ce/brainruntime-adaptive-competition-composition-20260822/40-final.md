# BA-TR17 final

Status: `ADAPTIVE_COMPETITION_STOP / DEVELOPMENT_8_OF_16 /
PAIR_47_OF_64 / CONFIRMATION_SEALED`.

Adaptive k-WTA repairs the hard capacity-one obstruction but does not preserve
source attribution after recurrent contributions have already been summed:

\[
W(p_i+p_j)=Wp_i+Wp_j.
\]

Taking global top-2 after this sum can select a different pair from the union
of the two singleton winners. Threshold or gain retuning cannot reconstruct
which source generated which contribution.

The next equation must compete each delayed source contribution separately and
only then add the selected routes:

\[
c^{(s)}_h=[W_{hs}p_s-\max_{k\ne h}W_{ks}p_s]_+,
\qquad c_h=\sum_{s\in A_t}c^{(s)}_h.
\]

This source-factorized route uses packet provenance, not targets or decoder
feedback. It is the next falsifier; BA-TR17 itself remains STOP.

