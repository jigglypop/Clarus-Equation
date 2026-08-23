# Mathematical verification

Status: COMPLETE

## Runtime direction

The runtime recurrent term is $Wq$ with row-post/column-pre indexing. Every learnable candidate is
therefore a $d\times d$ matrix whose association term has the form

$$
\mathrm{post}\;\mathrm{pre}^{\mathsf T}.
$$

All raw collectors are external audit objects. They observe detached native activations, cannot
mutate runtime weight mid-phase, and apply one bounded matrix only at block end. Automatic STDP is
disabled so it cannot create a second unlogged write.

## T1 factor transfer

The factorized code is

$$
c_{ab}=c^A_a\oplus c^B_b\oplus0\oplus0,
\qquad
v_{ab}=0\oplus0\oplus v^A_a\oplus v^B_b.
$$

Training set $\{(0,0),(0,1),(1,0)\}$ leaves $(1,1)$ absent. The intervention changes the B cue
block from 0 to 1 while keeping A fixed at 1. Because the decoder codebook is fixed before learning,
success is a compositional held-out test under the declared factor bases. Factor value frequencies are
imbalanced in the three observations, so a pass is transfer under that schedule, not general causal
composition.

## M2 lagged contrast

M2 starts from a verified fixed point $W_0$ of the exact projection:

$$
\left\|\Pi(W_0)-W_0\right\|_F\le10^{-7},
$$

using density 1 and frozen hysteresis thresholds $10^{-6}$ and $5\times10^{-7}$. Therefore
$C^+=C^-$ implies raw, proposed, and applied delta zero rather than an update caused by projection.

For each external collector phase, store a previous vector $p$ and add

$$
C\leftarrow C+a_t p^{\mathsf T},
\qquad p\leftarrow a_t.
$$

The cue is observed before the native transient reset, and the collector alone retains it. Define
$\widetilde a_0$ as cached cue and $\widetilde a_{1:3}$ as the three post-reset replay states.
Therefore $T=3$ and the first term is target-from-cached-cue, not a literal adjacent runtime
transition. Later terms are adjacent replay transitions. Positive and negative phases fork the same
pre-block weight snapshot and have equal length. The proposed weight is

$$
W'=\Pi\left(W+\eta(C^+-C^-)\right),
$$

with $\eta=0.8$. Implementation forms the raw term, projects $W+$ raw, zeros the diagonal,
subtracts $W$ to obtain an additive delta, clips that delta once, and sends only the delta to the
native bounded install. The primary audit uses actual $W'-W$, not a raw nonzero correlation.
Identical phases must yield a zero update within $10^{-7}$; target shuffle alters only positive target
assignment.

## M3 predictor and replay-residual update

Let the frozen pre-transition feature use one schema in fitting, scoring, and replay:

$$
\phi(z_t,e_t,m_t,r_t)\in\mathbb R^p.
$$

It contains ten native $d$-vectors: activation, refractory, memory trace, adaptation, STP-u, STP-x,
bitfield, lifecycle, inactive steps, and goal. It then appends the exact external-drive $d$-vector,
the exact effective replay-drive $d$-vector, forced-mode one-hot, replay-present scalar, and bias.
Hence $p=12d+5$. Ridge fitting gives

$$
\Theta=(X^{\mathsf T}X+10^{-4}I)^{-1}X^{\mathsf T}Y,
\qquad
\widehat a_{t+1}=\phi(z_t,e_t,m_t,r_t)^{\mathsf T}\Theta,
$$

where $\Theta\in\mathbb R^{p\times d}$. Training and held-out scoring use actual adjacent runtime
transitions with independent replay calibration vectors unrelated to task codebooks, so replay-drive
coefficients are identified before freezing. Association replay resets native transients, so the predictor receives the true
immediate reset predecessor. Its first error is paired with a separately cached cue-credit vector;
that cache is exactly the unit-normalized detached cue activation and is never described as the
predictor state. For later replay ticks, the presynaptic term is
the actual STP/mask-gated recurrent vector. With

$$
e_t=a_t-\widehat a_t,
$$

the exact block total is

$$
\Delta W=\frac{0.8}{3}\left[e_1q_{\mathrm{credit}}^{\mathsf T}
+e_2p_1^{\mathsf T}+e_3p_2^{\mathsf T}\right]
\in\mathbb R^{d\times d}.
$$

The replayed target state is a continuous teacher signal. Thus this is a teacher-forced replay-state
residual with cached credit, not autonomous prediction-error learning and not a proven gradient for
the nonlinear runtime. `Theta` is never refit after held-out scoring starts and receives no symbolic
target, codebook similarity, decoder result, or condition label.

Prediction and memory are logically separate. The predictor gate tests

$$
\operatorname{MSE}_{\mathrm{model}}
\le0.90\operatorname{MSE}_{\mathrm{persistence}}.
$$

Even if this holds, binding requires the independent post-cutoff recall conjunction. An error update
can fail binding despite an accurate predictor, or bind despite a weak predictor; both outcomes are
reported rather than merged.

## Controls and units

All controls preserve circuit snapshot, cue order, number of runtime steps, phase length, install
count, decoder, and RNG stream. They change exactly one declared relationship. The independent unit
is a seeded circuit. A per-cue accuracy or per-tick error is summarized within circuit before any
cross-seed decision.

No nonzero $\Delta W$, predictor MSE, or held-out decode can satisfy a route alone. Zero-store,
snapshot, sparse/dense, finite, adverse-control, exact-update reconstruction, and codebook-exclusion
predicates are conjunctive.
