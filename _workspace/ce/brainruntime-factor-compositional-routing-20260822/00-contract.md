# BA-TR5: held-out factor-compositional routing

Status: COMPLETE

Mode: light

PREDECESSOR: `_workspace/ce/brainruntime-learned-context-gate-20260822`

## PREDECESSOR_EVIDENCE

- BA-TR4 machine result: `artifacts/development-results.json`, SHA-256 `afc9d0aba4606f9dcc7a0370894c5b5682ceabfd9128b083fed5f46522d5f064`, development `16/16`, `LEARNED=ORACLE=1.0`, context-shuffle and wrong-cue `0.0` with opposite payload `1.0`.
- BA-TR4 source freeze: `artifacts/source-freeze.json`, SHA-256 `540086950be848437511cf1eed94547af6219bac89c296a4e637a4ebdb271116`.
- Admitted predecessor claim: two trained context cues can be associated with two fixed synthetic branch actuators by local branch-use learning. It did not test an unseen context combination. BA-TR4 confirmation remains sealed.

## Question

Can two independently learned local gates compose a route for a completely unobserved context pair, when each factor value was observed separately but the pair $(1,1)$ was absent from every gate-training experience?

## Frozen direct-product apparatus

Use two independent, unchanged BA-TR4 20-dimensional delayed payload runtimes, factors $F\in\{A,B\}$. Formally their joint state and recurrent operator are the direct sum

$$
x=x^A\oplus x^B,\qquad W=W^A\oplus W^B.
$$

Each factor retains the five BA-TR3 blocks $S^F_0,S^F_1,H^F_0,H^F_1,Y^F$, width $m=4$, exact delay $L=2$, neuronwise threshold profile, STP, decoder, zero-store cutoff, and recall horizon. No cross-factor state, weight, output block, or decoder exists. A factor mask contains 12 edges; a composed pair contains exactly 24. `FULL` contains 32 and is only a capacity-favourable interference control.

The two task factors are $a,b\in\{0,1\}$ with one-hot cues

$$
q^A_a=e_a\in\mathbb R^2,\qquad q^B_b=e_b\in\mathbb R^2,
\qquad q_{ab}=q^A_a\oplus q^B_b.
$$

For development seed $s$, independently balanced branch bijections are

$$
\sigma_s^A(x)=x\oplus p_s^A,\qquad
\sigma_s^B(x)=x\oplus p_s^B,
$$

where $p_s^A$ is the low seed bit and $p_s^B$ is the next seed bit. Across 16 consecutive development seeds, the four $(p^A,p^B)$ pairs must each occur four times.

## Training exclusion and local gate equation

The gate-training schedule is exactly

$$
\mathcal T=\{(0,0),(0,1),(1,0)\}.
$$

The pair $(1,1)$ must occur zero times in gate experience, gate update, joint-lookup update, source selection, schedule metadata used by a compiler, and pre-endpoint scoring. Each individual factor value still occurs in $\mathcal T$.

For each factor, exact-delay entry eligibility produces only its own local branch-use vector

$$
u_r^F=\frac1m\sum_{(i,j)\in Q_r^F}[E^F_{ij}]_+.
$$

Because deleting $(1,1)$ makes factor marginals unavoidably unequal, raw sums are forbidden. Each factor gate stores a local accumulator $C^F\in\mathbb R^{2\times2}$ and cue count $n^F\in\mathbb R^2$:

$$
C^F\leftarrow C^F+u^F(q_x^F)^{\mathsf T},
\qquad n^F\leftarrow n^F+q_x^F,
$$

$$
\Theta^F_{:,x}=\frac{C^F_{:,x}}{n_x^F},
\qquad n_x^F>0.
$$

At recall, both gates are frozen and receive only their factor cue:

$$
\widehat r_F(x)=\arg\max_r(\Theta^Fq_x^F)_r,
$$

$$
\widehat M_{ab}=\widehat M^A_a\oplus\widehat M^B_b.
$$

The primary endpoint is only $(a,b)=(1,1)$. Each factor receives two simultaneous distinct payloads and independently decodes the selected one. Joint success requires both output decoders to succeed. All ordered distinct payload pairs are evaluated in each factor and combined by the exact direct-product Cartesian readout.

## Leakage boundary

`CountNormalizedFactorGate.observe` may read only one factor cue and that factor's local branch-use vector. Its compiler may read only a frozen factor gate, one factor cue, that factor's frozen recurrent weight, and its blocks. Neither may read factor name, joint context, seed, $\sigma$, schedule, payload identity, expected answer, target, $Y$, decoder, endpoint, reward, route, the other factor, or oracle mask. Context never enters either BrainRuntime state or decoder.

## Controls

Endpoint arms are `ORACLE`, `FACTORWISE_LEARNED`, `A_FACTOR_SHUFFLE_TRAIN`, `B_FACTOR_SHUFFLE_TRAIN`, `A_LESION_STATIC_0`, `B_LESION_STATIC_0`, `STATIC_00`, `STATIC_01`, `STATIC_10`, `STATIC_11`, `RANDOM_MATCHED_24`, and `FULL_32`. All except `FULL_32` use exactly 24 recurrent edges. A four-action `JOINT_LOOKUP_HOLDOUT_ABSTAIN` is a pre-endpoint receipt: its one-hot joint cue column for `11` receives zero updates and must remain an exact tie, so no held-out endpoint may be opened for that arm.

The A-shuffle arm reverses only the A cue/experience pairing and must deliver A's opposite payload while B remains correct. The B-shuffle arm is symmetric. Lesion and static arms are evaluated only over the preregistered mapping-balanced seed batch because a fixed branch can match an individual seed by chance.

## Pre-endpoint gates

- Exact training multiset: four payload repetitions of `00`, `01`, `10`; zero `11` rows.
- Per-factor counts are positive and unequal as predicted; frozen $\Theta^F$ exactly equals an independently recomputed $C^F/n^F$.
- Each experience has correct local branch-use separation, no $Y$ pulse, and no decoder or endpoint read.
- Factor compiler signature and AST exclude all forbidden inputs and have no closure.
- Independent $\Theta q$ reference, cue swap, finite row-swapped-$\Theta$ counterfactual, and seed/$\sigma$/schedule invariance pass separately for A and B.
- Learned actions match both factor bijections; A/B shuffled gates reverse only their respective action.
- Four composed masks have 24 edges, a common 16-edge output trunk, and Hamming distances 8 when one factor changes and 16 when both change.
- The two runtime snapshots and two decoders are independent; all BA-TR3 rank/path, threshold, delay, STP, dense/sparse, zero-store, and cutoff receipts pass.
- `JOINT_LOOKUP_HOLDOUT_ABSTAIN` has a zero `11` count, zero `11` column, and exact four-way tie before any endpoint.

Any failed receipt is `APPARATUS_INVALID` and keeps the held-out endpoint closed.

## Development decision

Use seeds `97701..97716`; reserve `99701..99732` unopened for confirmation. A seed passes only when apparatus gates pass, held-out `FACTORWISE_LEARNED>=0.95`, `ORACLE>=0.95`, oracle gap is at most `0.05`, both factor-shuffle joint accuracies are at most `0.05`, each shuffled factor delivers its opposite payload at least `0.95` while the untouched factor remains correct at least `0.95`, all snapshots remain frozen, and no store survives.

Development GO additionally requires at least 15/16 seed passes, exact four-way mapping balance, mean lesion accuracy at most `0.55`, every static-pair mean accuracy at most `0.30`, `RANDOM_MATCHED_24<=0.55`, `FULL_32<=0.55`, and learned advantage over the strongest 24-edge non-oracle endpoint control of at least `0.40`.

One mechanism-only revision is allowed. Threshold, decoder, delay, horizon, mask budget, held-out pair, controls, or seeds may not be tuned. A negative result after the allowed revision is `FACTOR_COMPOSITION_NOT_IDENTIFIED`; confirmation remains sealed.

## Claim ceiling

A pass establishes only that two local factor gates over two declared independent synthetic branch families compose an unobserved fourth context pair. It does not establish candidate-support discovery, interacting-factor routing, arbitrary graph morphology, biological gating, cortical folding, curvature-as-memory, physical energy, disease intervention, or AGI.
