# Loop 8B preregistration — MD-modulated attractor geometry

Status: LOCKED BEFORE IMPLEMENTATION

## 1. Question

Can a continuous mediodorsal-thalamus-like modulation of a shared recurrent
attractor protect the context-relevant memory dimension better than pure
diffusion or a context-independent attractor, without giving the candidate
privileged information at readout?

This is a bounded synthetic mechanism test. It is not evidence that biological
MD literally implements a Riemannian metric and is not an AGI claim.

## 2. Domain and state

- Normalized time step: `h = 0.05`.
- PFC population state: dimensionless `z = (z1, z2)`.
- Each trial contains two opposite signed features, `f2 = -f1`, so loss of the
  individual spatial mode cannot be hidden by the conserved mean mode.
- A latent context `c in {0, 1}` selects which feature must be reported after a
  delay.
- Context arrives only through a noisy scalar cue. Every arm receives the same
  causal cue filter and the same final context readout.
- Context switches in blocks; no arm reads the latent context or future data.

## 3. Fixed dynamics

The attractor arms use Euler–Maruyama on

`dz_i = [a_i z_i (1-z_i^2) - 2 gamma z_i z_j^2 + u_i] dt + sqrt(2 D dt) dW_i`.

The cross term is a fixed shared-capacity constraint. Parameters are locked:

- `a = 1.5`, `gamma = 1.0`, `D = 0.08`, `D_heat = 0.80`;
- `encoding_steps = 10`, `input_gain = 1.3`;
- three delay distractor pulses with gain `0.80` at one-quarter, one-half, and
  three-quarters of the delay;
- ID delay `40`, OOD delay `70`;
- ID context noise `0.55`, OOD context noise `0.85`;
- context-filter retention `0.85`, cue gain `0.50`, logit gain `2.0`;
- MD relevance floor `0.40`, relevance gain `1.20`.

The fixed-attractor arm has `a1 = a2 = a`. The MD arm uses the same filtered
context probability `theta` as the readout but additionally sets
`a_i = a * (0.40 + 1.20 * relevance_i(theta))`. This is continuous and contains
no rule-specific branch.

The pure-diffusion arm evolves under the two-node positive Laplacian
`L = [[1,-1],[-1,1]]` as `z <- z - h D_heat L z`, with the same encoding input
and stochastic increments. Its nonconstant mode must decay.

## 4. Arms

1. `pure_diffusion` — no attractor drift.
2. `fixed_attractor` — context-independent recurrent wells.
3. `md_attractor` — continuous context-dependent well depth.
4. `md_context_shuffle` — identical MD dynamics but context cues are permuted
   across trials within seed before entering the filter.
5. `oracle_context_md` — latent context supplied to theta; ceiling only.

Common random numbers are used across arms for features, encoding noise,
distractors, and SDE increments.

## 5. Metrics

- report accuracy;
- opposite-feature memory margin in the selected dimension;
- context accuracy from the common causal cue filter;
- post-switch trials 3–8 accuracy;
- state boundedness and non-finite count;
- empirical pure-diffusion nonconstant-mode decay versus the registered
  exponential coefficient.

ID uses 32 seeds x 192 trials. OOD changes only delay and context noise.
Bootstrap lower confidence bounds use 3,000 resamples and a fixed seed.

## 6. Locked gates

All gates are conjunctive.

1. `fixed_attractor - pure_diffusion` accuracy LCB is at least `+0.10` in ID
   and OOD.
2. `md_attractor - fixed_attractor` accuracy LCB is at least `+0.03` ID and
   `+0.02` OOD.
3. `md_attractor - md_context_shuffle` accuracy LCB is at least `+0.05` ID
   and OOD.
4. MD post-switch accuracy is no worse than fixed attractor by more than
   `0.01` in either domain.
5. Every state is finite and `max(abs(z)) <= 4` in every arm and seed.
6. The deterministic pure-diffusion mode check matches
   `exp(-2 D_heat h n)` within absolute error `1e-10`.
7. Oracle MD is not worse than inferred-context MD.

Score: `100` only if all gates pass; otherwise `0 STOP`. A passing result is a
go decision for adding the residual-replay ablation, not for runtime merge.

## 7. Prohibited after-result actions

- no parameter or threshold sweep;
- no seed deletion;
- no replacement of the fixed-attractor comparator;
- no residual or STN term until this gate is scored;
- failed outcomes remain in the report and validation artifact.
