# BA-TR8: seeded source allocation under local competition and homeostasis

Status: COMPLETE

Mode: light

PREDECESSOR: `_workspace/ce/brainruntime-source-only-symmetry-nogo-20260822`

## Question

Can an outcome-blind edge-level heterogeneity break BA-TR7's exact source-only symmetry, and can a local capacity/homeostasis state turn the resulting winners into four distinct source-to-hidden bindings without a hidden pulse, output, decoder, reward, or endpoint?

## Frozen apparatus

Use one four-coordinate source block $S$ and one four-coordinate hidden block $H$ in the existing delayed Torch runtime. The only nonzero recurrent weights are the sixteen candidates $H\leftarrow S$. For every physical source coordinate $s$, draw an independent seed-frozen permutation of

$$
\left(-3,-1,1,3\right)/\sqrt{20}
$$

and define

$$
B_{hs}=1+\epsilon\xi_{hs},\qquad \epsilon=0.2.
$$

The heterogeneity generator receives only its dedicated seed and shape. It cannot read a task mapping, payload label, hidden/output codebook, decoder, reward, source presentation order, or endpoint. Every column has zero mean, equal norm, and four distinct values; all $B_{hs}$ remain positive.

Pulse one physical source coordinate at tick zero. For delay $L=2$, observe through the true first hidden arrival at tick $L+1=3$. From exact-delay local eligibility define the per-source evidence

$$
z_{hs}=\frac{[E_{hs}]_+}{\varepsilon_0+\sum_{k\in H}[E_{ks}]_+}.
$$

Process the four source coordinates in a separately seeded order. With local occupied-capacity state $r_h\in\{0,1\}$,

$$
u_{hs}=z_{hs}-\lambda_r r_h,\qquad
h^*(s)=\operatorname*{argmax}_{h\in H}u_{hs},
$$

$$
\mathcal A_{hs}\leftarrow \mathbf 1[h=h^*(s)],\qquad
r_h\leftarrow\min\{1,r_h+\mathbf 1[h=h^*(s)]\},
$$

with $\lambda_r=1.1$. Any winner gap below $10^{-6}$ is an abstention; coordinate-order tie-breaking is forbidden. Because $0<z_{hs}<1$ and $\lambda_r>1$, an occupied hidden coordinate cannot defeat an unused coordinate. This is a hard local WTA/capacity proxy, not a claim that BrainRuntime already implements lateral inhibition.

## Controls and decision

Run development seeds `98001..98016`; keep confirmation seeds `100801..100832` sealed.

- Uniform $B=1$ with or without the capacity state must reproduce BA-TR7 and abstain on the first source.
- Seeded $B$ without homeostasis may form source winners but may collide; report its collision fraction.
- A source-independent neuron bias must collapse the raw winners to one hidden coordinate, showing that neuronwise bias alone is not a source code.
- Seeded $B$ plus capacity/homeostasis must create a four-edge bijection with positive winner margins on every development seed.
- Permuting hidden rows of $\xi$ must permute the learned bindings covariantly.
- Changing the presentation order may change the binding while preserving bijection; this is recorded as developmental path dependence, not invariant semantic identity.
- No hidden/output pulse, output/trunk weight, decoder, reward, or endpoint may be read. `endpoint_opened` remains false even on success.

The development result is `SEEDED_SOURCE_ALLOCATION_PASS` only if all apparatus gates pass for 16/16 seeds, the combined mechanism yields 16/16 bijections, the uniform controls abstain, and the source snapshots remain immutable during observation.

## Claim ceiling

A pass means only that an explicitly supplied outcome-blind microscopic edge code plus a local WTA/capacity rule can form path-dependent source-to-hidden allocations in this normalized simulator. The seed supplies symmetry-breaking information. Source-only data cannot identify the meaning of hidden/output coordinates, so memory semantics, output routing, curvature-as-memory, cortical development, biology, disease treatment, physical energy, and AGI remain untested.
