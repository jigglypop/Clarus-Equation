# BrainRuntime topology-aware routing contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brain-mechanism-alternative-routes-20260819`; `_workspace/ce/brain-memory-contrastive-predictive-routes-20260819`

## Question and scope

Test whether a cue-conditioned graph mask computed from the learned recurrent
weight alone can preserve M1 binding and improve the previously failed T1
held-out factor transfer at a fixed edge budget.  This is a synthetic
`BrainRuntime` mechanism test.  It is not evidence for a biological router,
clinical efficacy, physical brain energy, or anatomical graph morphology.

The mask is selected before rollout.  No target vector, decoder score,
held-out label, post-rollout state, or endpoint may enter mask construction.
All scored probes physically remove temporal and hippocampal stores.

## Predecessor evidence

| Result | Evidence/status | Preserved claim | Never retry / new seam |
|---|---|---|---|
| M1 delayed signed eligibility | `brain-mechanism-alternative-routes-20260819/31-validation.md`; confirmation artifact SHA-256 `c9bc90b9172f3f0915615665eaca64212eaf5dd12ef86ac19324846f81bfa155`; PASS 32/32 | local block-delayed synthetic cue/value binding | do not retune per-tick native STDP; routing must leave the M1 learner unchanged |
| T1 frozen M1 transfer | `brain-memory-contrastive-predictive-routes-20260819/31-validation.md`; artifact `t1-development-results-v2-audited.json`; STOP 11/16 | pairwise binding does not guarantee composition; five failures selected frequent `(1,0)` | no threshold/decoder/seed retune; only a new structural routing mechanism and matched controls may reopen transfer |
| A7-H actual runtime | `brain-algorithm-route-ledger.md`; fixed-branch runtime property PASS, Rust delay parity blocked | Torch runtime exposes STP, refractory, lifecycle and delay state | use Torch only; no Rust parity claim with delay enabled |
| A8-T threshold implementation | current worktree focused tests | optional neuronwise active/bit thresholds execute in runtime | the frozen threshold profile below is a simulator fixture, not a biological distribution |

The user-supplied 512-seed toy graph results are motivating input only.  Their
CSV assets are not present in this workspace, so none of their numerical
claims are treated as reproduced evidence here.

## Frozen apparatus

- Development seeds: T1 `97301..97316`; M1 binding `97201..97216`.
- Confirmation seeds remain unopened.
- Dimension 48; M1 defaults otherwise unchanged.
- Backend Torch; axonal delay enabled with ring length 2.
- For coordinate $i=0,\ldots,47$, neuronwise float64 literals are generated
  once as $\vartheta_i=0.18+0.08i/47$,
  $\theta_i^-=0.06+0.08i/47$, and
  $\theta_i^+=0.24+0.12i/47$, then passed to the float32 Torch runtime.
- Four contiguous equal coordinate blocks are the declared architectural
  partition.  Their labels are not target identities.
- Runtime orientation is $W_{ij}:j\to i$.  Define
  $A=|W|\odot(\mathbf 1-I)$ so self-edges enter neither path scores nor the
  budget.  For non-full routes, the exact edge budget is
  $B=\lceil0.25\,\operatorname{nnz}_{i\ne j}(W)\rceil$.  Diagonal edges are
  excluded.  Every route must have at least $B$ admissible learned edges or
  the circuit is `APPARATUS_INVALID`; no route may fill from outside its
  admissible support.
- Block relevance ties choose the smaller block index; downstream block ties
  choose the smaller non-source block index; edge ties sort by descending
  score, then ascending row, then ascending column.
- A route mask is applied to a cloned sealed snapshot as $W_g=W\odot g$;
  the original learned snapshot is immutable.
- Zero learned off-diagonal support, $B=0$, a zero cue, or no cue-active
  source block is `APPARATUS_INVALID`.  The return normalization uses the
  ordinary maximum only after these admission checks; $R\equiv0$ on a
  nonempty support is valid because the denominator remains $\epsilon$.

## Frozen routes

Let $A=|W|\odot(\mathbf 1-I)$, $q=|c|$, and let the four blocks be $V_b$.
Both are raw normalized-runtime tensors and are dimensionless; no additional
data-fitted normalization is performed.

1. `FULL`: $g_{ij}=1$ for every nonzero off-diagonal learned edge.
2. `WEIGHT`: retain the $B$ largest $A_{ij}$ globally.  It is cue-blind.
3. `CLUSTER`: cue-active source blocks are exactly
   $\{b:\sum_{j\in V_b}q_j>10^{-8}\}$.  For each active source block $s$,
   select the non-source downstream block $d(s)$ maximizing
   $C_{ds}=\sum_{i\in V_d,j\in V_s}A_{ij}$.  Score by $A_{ij}$ only on
   source-to-destination and selected source/destination intra-block edges,
   then retain exactly $B$.
4. `TOPOLOGY`: use the same admissible cluster-path edges, forward relevance
   $f=q+Aq+A^2q$, and local return support
   $R_{ij}=A_{ji}+(A^2)_{ji}$.  Rank admissible edges by
   $$S_{ij}=A_{ij}(\epsilon+f_j)\left(1+\frac{R_{ij}}{\max R+\epsilon}\right),
   \qquad \epsilon=10^{-8},$$
   then retain exactly $B$.  This is the first path-plus-cycle candidate; it
   does not claim a complete motif dictionary.
5. `PATH_ONLY`: use the exact `TOPOLOGY` admissible support and forward
   relevance but set $R_{ij}=0$ before ranking.
6. `RETURN_SHUFFLED`: use the exact `TOPOLOGY` admissible support, $A$, $q$,
   and $f$, but permute the $R_{ij}$ values over admissible edges with a
   seed-fixed permutation before ranking.
7. `RANDOM_MATCHED`: retain exactly $B$ learned off-diagonal edges using a
   seed-fixed permutation.
8. `WRONG_CONTEXT`: construct `TOPOLOGY` with the cyclic next codebook cue
   `(k+1) mod 4` while evaluating cue $k$.

No route may inspect a target codebook, decoded identity, accuracy, or final
activation during construction.

## Measurements

- task: clean/corrupt M1 accuracy and T1 held-out accuracy;
- separation: correct-target cosine minus the largest wrong-target cosine;
- simulator observables: sum of returned `RuntimeStep.energy`, mean active
  lifecycle fraction, and dynamic exposed-edge fraction
  $|\{(i,j):g_{ij}=1,\ j\text{ active}}|/\operatorname{nnz}(W)$;
- structural cost: retained-edge fraction and normalized Hamming switch cost
  between consecutive cue masks in codebook order `(0,0),(0,1),(1,0),(1,1)`;
- topology separation: normalized Hamming distance between `TOPOLOGY` and
  `PATH_ONLY` masks.

The edge/node counts are dimensionless compute proxies, not joules.  No
weighted composite objective is primary; task and cost coordinates are
reported separately.

## Development decision

Apparatus is invalid if full delayed/heterogeneous-threshold M1 binding does
not pass at least 15/16 circuits, any sparse arm misses the exact budget, a
mask reads a forbidden target/decoder object, stores survive cutoff, or the
source snapshot changes.

`PATH_ROUTING_DEVELOPMENT_GO` requires all of:

1. T1 held-out success at least 13/16 (the prior 80% gate rounded up);
2. at least two more successful circuits than `FULL` and strictly more than
   each same-budget `WEIGHT`, `CLUSTER`, `RANDOM_MATCHED`, and
   `WRONG_CONTEXT` arm;
3. positive mean separation advantage over every same-budget control;
4. M1 clean accuracy at least 15/16 and no more than one loss relative to
   `FULL`;
5. exact cutoff, snapshot immutability, finite state, and edge-budget checks.

The stronger `TOPOLOGY_SPECIFIC_DEVELOPMENT_GO` additionally requires a
nonzero `TOPOLOGY`/`PATH_ONLY` mask distance in at least 8/16 T1 circuits and
strictly greater success count plus positive mean separation advantage over
both `PATH_ONLY` and `RETURN_SHUFFLED`.  Without this, any positive result is
classified only as cue-conditioned cluster/path routing.

Failure is a mechanism STOP, not permission to tune budget, threshold profile,
partition count, horizon, decoder, route score, or seeds.  Usage-based slow
consolidation is a later gated experiment and is not tested in this run.

## Named implementation obligations

`31-validation.md` may classify a scientific result only if all six focused
checks pass:

- `V-INPUT`: route-constructor signature allowlists only weight, cue, blocks,
  public seed, route, and budget; source scan finds no target/decoder/endpoint
  access;
- `V-BUDGET`: every sparse mask has exactly $B$ learned off-diagonal edges;
- `V-DEGENERATE`: $M<B$, zero learned support, zero cue, and absent source
  block all fail closed as `APPARATUS_INVALID`;
- `V-SNAPSHOT`: the unmasked sealed snapshot hash is unchanged by every arm;
- `V-CUTOFF`: temporal and hippocampal rows equal zero during every scored
  rollout;
- `V-FINITE`: all mask scores, runtime states, and reported measurements are
  finite.
