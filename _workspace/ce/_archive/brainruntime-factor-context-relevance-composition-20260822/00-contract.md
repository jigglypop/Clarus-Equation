# BA-TR24 held-out factor-context relevance composition

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

The four relevance contexts form a Cartesian product:

\[
(a,b)\in\{0,1\}^2,
\qquad
P_{ab}=\{S_a,S_{2+b}\}.
\]

Only contexts `00,01,10` are presented during gate learning; `11` is sealed.
Each factor value is nevertheless observed at least once. For factor-specific
event count matrices (U^A,U^B), exposure counts (n^A,n^B), and orthonormal
factor codes (q^A_a,q^B_b),

\[
\Theta^A=\sum_a \frac{U^A_{:a}}{n^A_a}(q^A_a)^\top,
\qquad
\Theta^B=\sum_b \frac{U^B_{:b}}{n^B_b}(q^B_b)^\top,
\]

\[
g_j(a,b)=\mathbf1[(\Theta^Aq^A_a+\Theta^Bq^B_b)_j>1/2].
\]

Training reads context factors and co-occurring event coordinates only. It
does not read targets, outputs, rewards, decoder, endpoint, or the heldout
context row.

Fresh calibration seed: `111001`. Fresh development seeds: `111101..111116`,
opened only after calibration passes.

Per seed gates: training contexts exactly `00,01,10`; counts `[2,1]` for both
factors; compilers exact for all four; training contexts recalled; heldout
`11` equals oracle; joint unseen lookup with frozen `00` fallback fails;
factor-A and factor-B cue shuffles fail; no context fails; gate immutable and
stores zero. Any failure is STOP without changing factors, threshold, fallback,
or context schedule.

Claim ceiling: synthetic held-out composition of two declared independent
context factors on fixed circuit support. It is not discovery of the factor
decomposition or general OOD reasoning.

