# Mathematical verification

Status: COMPLETE

## Native state transition

Let the executable runtime state be

$$
z_t=(a_t,r_t,m_t,u_t,x_t,\ell_t,d_t),
$$

where the terms denote activation, refractory state, memory trace, short-term facilitation and
depression, lifecycle mask, and optional delay state. With delay disabled for this experiment,
the recurrent presynaptic vector is

$$
q_t=u_t\odot x_t\odot a_t\odot \mathbf 1_{\mathrm{active},t}.
$$

The PyTorch runtime uses the row-output convention

$$
a_{t+1}=\operatorname{clip}\left((1-d_m)a_t+g_m\tanh\left(Wq_t+
g_{\mathrm{ext}}e_t+g_{\mathrm{goal}}h_t+g_{\mathrm{rep}}p_t-r_f r_t-
\lambda A_t+\xi_t\right)\right).
$$

Thus a causal association from cue coordinate $j$ to target coordinate $i$ requires a positive
increment to $W_{ij}$.

## P0: eligibility orientation

The current eligibility code forms `outer(pre_trace, spike)`. Under `W @ pre`, a cue spike at
$j$ followed by a target spike at $i$ therefore increments $W_{ji}$, the reverse of the required
$W_{ij}$. Existing tests check only nonzero eligibility and generic weight drift; they do not test
asymmetric causal direction. Native replay requires an opt-in causal orientation
`outer(post_spike, pre_trace)` with the LTD term aligned to the same row-output convention. The
legacy orientation must remain the default until separately migrated.

The implementation gate therefore includes a two-coordinate test: cue $j$ at $t$, target $i$ at
$t+1$, and a positive scalar gate must yield $\Delta W_{ij}>0$; cue-only free rollout must then
increase target-$i$ activation relative to the matched initial runtime.

## Independent recall

For a fixed cue $c$, evaluation starts from a declared deterministic reset state, drives only
$c$, and then performs $H=6$ calls with zero external input. A fixed pre-training value codebook
decodes only the final activation $a_{t+H}$. It may not receive a target, goal, old activation,
temporal row, hippocampal row, or trained cue/value lookup. The runtime must satisfy

$$
|M_{\mathrm{temporal}}|=|M_{\mathrm{hippocampus}}|=0
$$

before the cue is applied. Clearing only one store is insufficient because every runtime step
otherwise calls hippocampal recall.

Weight drift is necessary but not sufficient. Native and controls must start from identical
post-constructor $W_0$, use identical cue order, timing, counts, scalar learning signals, and RNG
streams, and differ only in the preregistered ablation. The reported causal quantity is paired
recall advantage, not merely $\lVert W-W_0\rVert_F$.

## Intervention transfer

Cue coordinates are partitioned into base and intervention blocks before data generation. The
intervention changes only the intervention block, leaves $W$ and the decoder fixed, and is scored
on held-out base/intervention combinations absent from replay. Without a factorized generator and
held-out combinations, the measurement is interpolation rather than transfer.

## Bounded self-prediction

The Loop 10 predictor receives a fixed function of current runtime observables and the committed
action only. It predicts the next summary before the transition and is compared with the
persistence predictor. Normalization and predictor parameters freeze before confirmation. A
correction chosen after seeing the next state demonstrates error monitoring only; predictive
metacontrol additionally requires that the policy selecting a correction cannot read that state.

## Findings

- P0: correct causal eligibility orientation behind an explicit opt-in switch.
- P0: physically detach both temporal and hippocampal stores before recall.
- P0: fixed decoder and zero-input rollout must be auditable.
- P1: use Torch CPU, zero noise, no axon delay, and an explicit reset policy.
- P1: treat Loop 10 as bounded self-prediction/metacognitive monitoring, not consciousness.

