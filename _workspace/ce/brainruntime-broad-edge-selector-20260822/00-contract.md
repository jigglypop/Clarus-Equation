# BA-TR6: broad-substrate edge-field selector

Status: COMPLETE

Mode: light

PREDECESSOR: `_workspace/ce/brainruntime-factor-compositional-routing-20260822`

## Question

Can a factor cue select the four useful source-to-hidden edges from a uniformly weighted broad substrate using only local exact-delay eligibility, rather than receiving either of the two branch masks used by BA-TR3--TR5?

## Apparatus

Use two independent 20-dimensional delayed runtimes. In each factor, only the blocks $S_0,S_1,H,Y$ participate. The declared candidate entry substrate is

$$
\mathcal C=H\times(S_0\cup S_1),\qquad |\mathcal C|=4\times8=32,
$$

and every candidate has the same frozen recurrent weight $W_e=1$. The context-independent trunk $\mathcal T\subset Y\times H$ contains exactly the four inherited payload-matching edges. All other recurrent weights are zero. Thus weight magnitude and zero support cannot reveal which four entry edges should be opened.

For development seed $s$, factor $F$ uses a balanced source map $r_s^F(x)=x\oplus p_s^F$. Gate experience contains exactly four payload repetitions of joint contexts `00`, `01`, and `10`; `11` is absent. An experience pulses $S_{r_s^F(x)}(k)$ and, after the exact delay, the shared $H(k)$. It never pulses $Y$ and never reads a target, decoder, reward, or endpoint.

For each episode, positive local eligibility is normalized over the broad substrate:

$$
z_e^{F,\ell}=\frac{[E_e^{F,\ell}]_+}
{\varepsilon+\sum_{e'\in\mathcal C}[E_{e'}^{F,\ell}]_+}.
$$

The unequal factor exposure is removed by the same count normalization established in BA-TR5:

$$
C^F\leftarrow C^F+z^{F,\ell}(q_x^F)^{\mathsf T},\qquad
n^F\leftarrow n^F+q_x^F,\qquad
\Theta^F_{:,x}=C^F_{:,x}/n_x^F.
$$

The frozen compiler receives only $(\Theta^F,q_x^F,\mathcal C,\mathcal T)$. It selects the four largest entry scores and adds the fixed trunk:

$$
M^F(x)=\operatorname{Top}_4(\Theta^Fq_x^F)\cup\mathcal T.
$$

It must fail closed unless the fourth/fifth score gap is at least $10^{-6}$. Each factor mask has eight edges and the direct-product pair has sixteen.

## Identifiability gates

- All 32 candidate weights are exactly one; `WEIGHT_ONLY` has a fourth/fifth tie and abstains.
- The cue-column mean $\bar\Theta=(\Theta_{:,0}+\Theta_{:,1})/2$ has an exact fourth/fifth tie and `POOLED_STATIC` abstains. Raw pooled sums are forbidden because the schedule has an 8:4 marginal imbalance.
- Compiler signature/AST excludes weight, branch family, source mapping, factor name, joint context, seed, schedule, payload, expected answer, target, decoder, endpoint, reward, and oracle.
- Each cue has four selected candidates, a strict boundary margin, and a distinct edge set. Cue shuffle exchanges only that factor's edge set.
- `11` occurs zero times before endpoint; all source, gate, and codebook snapshots remain frozen; all stores and delay packets are empty at cutoff.

## Endpoint controls and decision

Development seeds are `97801..97816`; confirmation `99801..99832` remains sealed. Endpoint routes are `ORACLE`, `EDGE_FIELD_LEARNED`, factor-A/B cue-shuffle, four fixed source-pair controls, `RANDOM_MATCHED_16`, and `FULL_72`. All routes except `FULL_72` use exactly sixteen edges.

A seed passes when preflight passes, learned and oracle held-out `11` joint accuracy are at least `.95`, each factor shuffle has joint accuracy at most `.05` while delivering the manipulated factor's opposite payload and preserving the other factor at least `.95`, and every source/gate/store freeze holds.

Batch GO requires at least 15/16 seed passes, exact four-way seed parity balance, every fixed source-pair mean at most `.30`, random matched at most `.55`, full at most `.55`, and learned advantage over the strongest matched non-oracle endpoint control at least `.40`.

No threshold, delay, decoder, horizon, candidate support, edge budget, tie rule, control, held-out pair, or seed may be changed after endpoint opening. At most one mechanism-only revision is allowed.

## Claim ceiling

A pass establishes only context-conditioned recovery of externally demonstrated four-edge actuator sets inside a declared, uniformly weighted 32-edge synthetic source-to-hidden block, followed by held-out direct-product composition. It does not establish autonomous support discovery, factor discovery, motif/topology discovery, biology, cortical folding, curvature-as-memory, physical energy, disease intervention, or AGI.
