# BA-TR23 context-to-packet relevance gate

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

BA-TR22 proved that packet-local variables alone cannot choose a desired pair
from three equally valid routes. BA-TR23 supplies context but learns its
relevance relation only from context/event co-occurrence.

For orthonormal seed-specific context codes (q_c\in\mathbb R^4) and binary
event-coordinate vector (e_c\in\{0,1\}^{20}), training is

\[
A\leftarrow A+e_cq_c^\top,
\qquad
g_j(q)=\mathbf1[(Aq)_j>1/2].
\]

The gate receives no target, decoder, reward, output, learned target mapping,
or endpoint. It is frozen before probes. At test the same BA-TR22 three-event
packet arrives; only coordinates selected by (g(q_c)) enter factorized
competition. The context code is not injected into runtime activation or the
decoder.

Fresh calibration seed: `110001`. Fresh development seeds: `110101..110116`,
opened only after calibration passes.

Per seed gates: orthonormal codes; exact learned two-coordinate compiler;
learned and oracle 4/4 with bit-exact targets; context-shuffle 0/4; fixed static
gate exactly 1/4; no-context all-input 0/4; exact three-event packet receipt;
gate hash immutable; zero stores. Any failure is STOP without changing the
threshold, schedule, context dimension, or event pairs.

Claim ceiling: a synthetic four-context lookup learned from supplied
context/event co-occurrence. It is not autonomous context inference, OOD
generalization, biological attention, or support discovery.

