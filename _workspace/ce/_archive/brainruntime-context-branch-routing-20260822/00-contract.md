# Context-only branch routing in a shared delayed trunk

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brainruntime-topology-routing-feasible-budget-20260822`

## PREDECESSOR_EVIDENCE

| Evidence | Frozen result | Narrow surviving claim | Reuse prohibition |
|---|---|---|---|
| TR2 development | `STRUCTURED_SPARSITY_OBSERVED / CUE_ROUTING_STOP / TOPOLOGY_SPECIFIC_STOP` | A declared block support at about 8.1% of learned edges reduced interference in the synthetic factor task. | Do not change the TR2 budget, decoder, horizon, thresholds, or seeds and relabel it as routing. |
| Machine result | SHA-256 `c68eb749c88d896b605fa4ab10a2c4cef52e485e872f6ebfa06eda54c7553d13` | `WRONG_CONTEXT=16/16` is the direct falsifier motivating this successor. | The successor must make correct and wrong masks structurally and functionally distinct before opening endpoints. |

## Question and claim ceiling

Can a context value, supplied only to a mask compiler, select one of two
simultaneous payloads through a context-invariant downstream trunk in the
actual delayed Torch `BrainRuntime`?

The strongest admissible positive statement is:

> In this synthetic runtime task, a context-only entry-branch mask selected
> one of two simultaneous payloads while the relay-to-output trunk and decoder
> remained invariant.

This is not evidence for general graph morphology, biological cortical
routing, curvature-memory identity, disease treatment, physical energy, or
AGI.

## Frozen apparatus

Revision 0 used an extra shared relay $R$.  The first frozen seed passed every
support/rank/cutoff receipt, reached the selected hidden block, but produced
zero $Y$ activity: the projected $R\leftarrow H$ row split its norm across two
alternative inputs and the extra hop fell below the already-frozen emission
threshold.  No endpoint threshold, decoder, or seed was changed.  Revision 1
removes only that unnecessary relay and retains a context-invariant output
trunk.

Let the normalized runtime state have five equal-width coordinate blocks

$$
\mathbb R^{5m}=S_0\oplus S_1\oplus H_0\oplus H_1\oplus Y,
\qquad m=4,
$$

with row-post/column-pre weights $W_{ij}:j\to i$.  The only learnable support
is

$$
U=(H_0\times S_0)\cup(H_1\times S_1)
 \cup(Y\times H_0)\cup(Y\times H_1).
$$

For payload $k$ and context $c$, training supplies the experienced pulse
trajectory

$$
S_c(k)\xrightarrow{L}H_c(k)\xrightarrow{L}Y(k),
\qquad L=2\ \text{ticks}.
$$

At recall, two distinct payloads are injected simultaneously:

$$
u_0=S_0(k_0)+S_1(k_1),\qquad k_0\ne k_1,\qquad u_{t>0}=0.
$$

Context never enters the runtime input, state, common trunk, output codebook,
or decoder.  It enters only the pure mask compiler.

The context mask is

$$
M_c=Q_{H_cS_c}\cup Q_{YH_0}\cup Q_{YH_1},
\qquad \lVert M_c\rVert_0=3m=:B.
$$

The wrong intervention is $M_{1-c}$.  `STATIC_0` and `STATIC_1` use one mask
for both contexts.  `RANDOM_MATCHED` uses exactly $B$ edges without context.
`FULL`/`STATIC_UNION` opens all $4m$ learned edges and is an interference
control, not an energy-matched control.

## Frozen local learning rule

For post-step normalized activity $a_t$, the episode-local exact-delay trace
is

$$
E_{t+1}=\rho E_t+a_{t+1}a_{t+1-L}^{\mathsf T}
-\mu a_{t+1-L}a_{t+1}^{\mathsf T},
$$

with $\rho=0.99$, $\mu=0.20$.  Each episode contributes only inside its
declared experienced corridor $P_c\subset U$.  Episode traces are summed for
one epoch and the sole recurrent write is

$$
\Delta W_{\rm actual}
=\mathbf 1_U\odot
\left[\operatorname{Proj}(W+0.8E_{\rm epoch})-W\right],
$$

followed by the existing Frobenius bound 5.0.  Projection is performed once;
no target matrix, least-squares solution, SVD target projection, decoder, or
rollout result enters the write.

## Frozen timing, thresholds, and endpoint

- `dim=20`, `m=4`, one learning epoch, Torch backend, zero noise.
- Event-time ring length `L=2`; pulses occur at ticks `0,2,4` without an
  inter-pulse reset.
- Recall is read after call index $2(L+1)=6$ (seven calls including call zero).
- External cue gain is 5.0.  Heterogeneous active and bit thresholds repeat
  the same four-value profile in every block, so homologous neurons are
  matched while neuron thresholds are not assumed equal.
- The frozen decoder reads only $Y$.  A trial succeeds only when target cosine
  is at least 0.50 and exceeds every other payload cosine by at least 0.15.
- Development seeds are `97501..97516`.  Confirmation seeds `99501..99532`
  remain sealed.

## Pre-endpoint apparatus gates

Before any recall score is computed, every seed must establish:

1. `supp(W) subset U` and actual installed delta is exactly zero outside $U$;
2. no direct $S\to R$, $S\to Y$, context-to-state, or context-to-decoder path;
3. all five learned block maps contain exactly $m$ edges and operational rank
   $m$ at SVD tolerance $10^{-6}$;
4. each two-edge context product has minimum singular value at least 0.25;
5. $|M_0|=|M_1|=B=3m$, their shared trunk is bit-identical, and their symmetric
   difference is exactly $2m$ entry-branch edges;
6. correct and wrong masks have equal depth, edge count, delay histogram,
   threshold profile, STP rule, and decoder hash;
7. the mask compiler receives only `(weight, context, blocks, seed, route)`;
8. hippocampal rows, temporal rows, activation, and the delay ring are zero at
   the sealed recall boundary.

Any failure is `APPARATUS_INVALID` and endpoints remain unopened.

Revision 0's first-seed endpoint is retained as a failed formula witness.  It
is not counted in Revision 1 development and it does not authorize changing
the frozen thresholds or decoder.

## Frozen decision rule

For a seed, `CONTEXT_BRANCH_PASS` requires all apparatus gates plus:

- `CORRECT` accuracy at least 0.95;
- `WRONG` accuracy at most 0.05 and opposite-payload delivery at least 0.95;
- `STATIC_0`, `STATIC_1`, and `RANDOM_MATCHED` accuracy at most 0.55;
- `FULL`/`STATIC_UNION` accuracy at most 0.55 under the strict unique decoder;
- correct accuracy exceeds every exact-$B$ static/wrong control by at least
  0.40;
- swapping $M_c\mapsto M_{1-c}$ after the cue reproduces `WRONG` without
  changing the mask budget.

Development GO requires at least `15/16` seed passes and all receipts.  A
negative result cannot be repaired by changing payload count, thresholds,
decoder, margin, delay, horizon, seed list, or budgets on these seeds.
