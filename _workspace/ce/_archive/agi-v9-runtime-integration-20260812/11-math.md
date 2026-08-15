# Mathematical integration audit

Status: COMPLETE

## M1. Dimensionless action-evidence map

For finite $o\in\mathbb{R}^d$ and action embedding $a_i\in\mathbb{R}^d$, define

$$
e_i(o)=
\begin{cases}
\dfrac{\langle a_i,o\rangle}{\lVert a_i\rVert_2\lVert o\rVert_2},
&\lVert a_i\rVert_2\lVert o\rVert_2>0,\\
0,&\text{otherwise}.
\end{cases}
$$

By Cauchy--Schwarz, $|e_i(o)|\le 1$. The units of numerator and denominator cancel,
so $e_i$ is dimensionless. A finite-input check before normalization and a finite-output
check after normalization are required implementation guards.

## M2. State-mediation property

Let the only V9 action branch be

$$
o_t\longmapsto e_t\longmapsto
\tau_t=C.\operatorname{observe}(t,e_t)\longmapsto
\pi_t=C.\operatorname{read\_policy}(\tau_t,m_t)
\longmapsto u_t=\pi_t.\operatorname{selected\_action}.
$$

If implementation assigns `action_index` only from the last expression, then all effects of
$o_t$ on $u_t$ in this branch are mediated by the sealed recurrent state named by $\tau_t$.
This is a code-path property, not a performance or necessity theorem. A regression must fail
if `action_index` is replaced by legacy similarity argmax or an external posterior.

## M3. Causality and history dependence

The inherited controller accepts exactly tick $t=C.tick+1$, updates from previous-tick state,
builds the token before commit, and rejects stale/future tokens. Therefore the adapter may use
the runtime step index only as a consistency check; it must derive the controller event tick
from the controller itself and must not read future runtime state.

For recurrence gain or cross-level gain nonzero, two legal histories can yield different
tower states under the same current $e_t$. This establishes representational statefulness.
It does not establish that the history is useful for any task; that remains a later registered
development claim.

## M4. Policy simplex

The inherited policy applies a stable softmax only over allowed coordinates and writes zero to
masked coordinates. Hence for any mask with at least one `True`, $\pi_i\ge0$ and
$\sum_i\pi_i=1$. The integration must preserve shell width $A$ equal to action count and must
pass the mask unchanged to this readout.

## M5. Composition boundary

The V9 tower has a finite schedule-specific contraction certificate in
`global_coordinate_sup`. The cosine encoder is bounded but is not itself a contraction claim,
and the surrounding `BrainRuntime` has no joined contraction proof with the tower. Therefore
the composed runtime agent is an executable finite causal system only; global convergence,
adaptive truncation, predictive superiority, AGI, and biological identity do not follow.

## M6. P2 cleanup judgment

`depth_error_tolerance` and `hysteresis_ticks` do not enter the grow-only update or depth
decision. Retaining them creates false degrees of freedom, so deleting them is semantics-
preserving for the implemented controller. The old `generated_parameter_count` expression is
neither a trainable-parameter count nor a MAC/capacity count; it may be retained only under a
name such as `serialized_template_scalar_count` that states its narrow metadata meaning.

## Verdict

The integration is mathematically admissible as a finite, opt-in, state-mediated action path.
It is not an AGI theorem and must remain outside evidence claims until a separately registered
task with matched controls tests V9-1.
