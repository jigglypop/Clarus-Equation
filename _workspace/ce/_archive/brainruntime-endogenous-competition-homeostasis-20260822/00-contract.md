# BA-TR9: endogenous delayed competition and homeostasis

Status: COMPLETE

Mode: light

PREDECESSOR: `_workspace/ce/brainruntime-seeded-source-competition-20260822`

## Question

Can BA-TR8's external Boolean occupied-vector and hard `argmax` allocator be
removed, while a persistent `BrainRuntime` still forms four distinct
source-to-hidden allocations using only delayed source packets, continuous
local competition, and runtime-owned homeostasis?

## PREDECESSOR_EVIDENCE

| predecessor result | frozen evidence | status | retained boundary |
|---|---|---|---|
| Seed-only balanced edge heterogeneity plus an external hard capacity proxy formed 16/16 four-edge bijections and reduced mean collision from `0.328125` to `0.0`. | BA-TR8 development result SHA-256 `a323d9927d39229c41094db32b49bbb2a73a9cb7e6c730865fd9020f3c8fa9fe`; module SHA-256 `e67287834004bada9e4e8850198032e338003573698c58c6a92cdf76ba6303f5`. | PASS | The microscopic seed breaks symmetry, but the hard external allocator is not a runtime mechanism and output meaning is nonidentified. |
| Uniform source-only weights preserve hidden-row symmetry and force a strict top-boundary tie. | `_workspace/ce/brainruntime-source-only-symmetry-nogo-20260822/40-final-report.md` | STOP for uniform support | Competition/homeostasis alone may not manufacture a first winner from an exactly symmetric state. |

## Frozen mechanism

Use one four-coordinate source block $S$, one four-coordinate hidden block
$H$, delay $L=2$, and the BA-TR8 outcome-blind weights

$$
B_{hs}=1+0.2\xi_{hs},
\qquad
\xi_{:s}\in\operatorname{Perm}
\frac{(-3,-1,1,3)}{\sqrt{20}}.
$$

Only the sixteen entries $H\leftarrow S$ are nonzero. All neuron thresholds
are scalar broadcasts; no coordinate-specific threshold vector is allowed.
There is no hidden/output pulse, output weight, decoder, reward, target, goal,
replay, or endpoint access.

For the actually delivered delayed recurrent packet $d_t$, runtime-owned
$r_t\in[0,1]^4$, and the positive hidden packet $d^+_{i,t}=[d_{i,t}]_+$, define

$$
v_{i,t}=e^{-\lambda r_{i,t}}d^+_{i,t},
\qquad
c_{i,t}=\left[v_{i,t}-\gamma\max_{j\ne i}v_{j,t}\right]_+.
$$

Only $c_t$ replaces the hidden coordinates of the recurrent drive. Frozen
$\gamma=1$ makes a tie map to zero and leaves only a unique maximum's positive
margin. It is a continuous piecewise-linear competition operator, not a
winner index or mask.

Let $p_t=\sum_i d^+_{i,t}$ and let $m_t\ge0$ be a decaying packet envelope.
With $\alpha=0.8$ and $\varepsilon=10^{-8}$,

$$
\nu_t=\frac{[p_t-m_t]_+}{\varepsilon+p_t},
\qquad
m_{t+1}=\max\{\alpha m_t,p_t\},
$$

$$
q_{i,t}=\nu_t\frac{[a_{i,t}]_+^2}
{\varepsilon+\sum_j[a_{j,t}]_+^2},
\qquad
r_{t+1}=\operatorname{clip}_{[0,1]}
\left((1-\delta)r_t+\rho q_{t-D_c}\right).
$$

Freeze $D_c=1$, $\delta=0$, $\rho=1$, and $\lambda=1$. The envelope admits
the rising edge of a packet burst and suppresses repeated commits from its
decaying tail. Both $r$ and the $q$ delay ring belong to `BrainRuntime`, are
included in snapshots, and are reset only at a run boundary. Between source
pulses the same runtime receives zeros until activation, axon packets, usage
packets, and envelope fall below `1e-5`; no fast-state reset is permitted.

The tuple above is fixed analytically. Since
$\lambda>\log(B_{\max}/B_{\min})\approx0.270$, a fully recorded used row is
weaker than even the smallest unused row on the next pulse. Calibration seed
`97091` checks only apparatus timing, abstention, washout, and snapshot
continuation; it may not select or tune a coefficient. Development seeds are
`98201..98216`; confirmation seeds `101801..101832` remain sealed.

## Controls and decision

- uniform $B=1$: the first pulse must tie, produce zero competed activation,
  and leave $r=0$;
- seeded with $\lambda=0$: same runtime and schedule, but no homeostatic
  attenuation; report collision fraction;
- source-independent row bias: even a four-row positional sequence is
  `SOURCE_UNIDENTIFIED`, because every pre-history source column is identical;
- hidden-row permutation: arrival traces, $r$ trajectory, and observational
  bindings must transform by the same permutation;
- midpoint snapshot: two independent restores must produce identical remaining
  trajectories and final state;
- the observational strict-margin winner is computed only after each first
  arrival and is never read by `BrainRuntime.step`.

Development is `GO` only if all 16 positive seeds form strict four-source
bijections, every uniform arm abstains, snapshot continuation and row
covariance pass, and mean collision is at least `0.20` lower than the
$\lambda=0$ control. Otherwise this fixed mechanism is `NO_GO`; no coefficient,
threshold, delay, washout rule, seed, or margin may be retuned on those seeds.

## Claim ceiling

A pass establishes only path-dependent synthetic source allocation in a
declared four-by-four substrate under an outcome-blind seed and a runtime-owned
continuous competition/homeostasis state. It does not identify output
semantics, memory content, graph morphology, curvature, cortical development,
biology, disease treatment, physical energy, or AGI.
