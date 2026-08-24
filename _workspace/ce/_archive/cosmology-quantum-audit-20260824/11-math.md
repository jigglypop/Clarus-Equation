# Mathematics lane: cosmology--quantum mechanics

Status: COMPLETE

## Finding first

No arithmetic or dimensional counterexample was found for the conditional
Poisson fixed point or for the named LO density-partition arithmetic.  The
physical chain is nevertheless open at two indispensable seams: a supplied
open-system generator is not a derivation of a quantum instrument/unravelling
or independent Poisson genealogy, and a path-space pushforward is not a local
covariant field or a conserved stress tensor.  Consequently no fixed-point
result derives any of the cosmological density parameters.

## Reconstructed objects and assumptions

For an independent offspring count with mean $D>0$, extinction is the least
fixed point of

$$
q=\exp[-D(1-q)].
$$

For $D>1$, putting $z=-D e^{-D}$ gives

$$
q_{\rm low}=-\frac{W_0(z)}{D},\qquad q=1=-\frac{W_{-1}(z)}{D}.
$$

The first equality needs the principal real branch; the second is the
identity fixed point.  The low root is unique in $(0,1/D)$ and is locally
attracting under fixed-point iteration because $Dq_{\rm low}<1$.  These are
conditional branching-process facts, not statements about a quantum action
or cosmological matter.

The frozen exact chain uses the dimensionless input

$$
\alpha_s=0.11789,\quad \sin^2\theta_W=4\alpha_s^{4/3},\quad
\delta=\sin^2\theta_W(1-\sin^2\theta_W),\quad D_{\rm eff}=3+\delta.
$$

It yields $D_{\rm eff}=3.1777584234099736$ and
$q_{\rm low}=0.048646719644028225$.  Direct bisection and an independent
Newton evaluation of the principal Lambert branch agree to
$6.94\times10^{-18}$; the equation residual rounds to zero and
$Dq=0.15458752312007412<1$.  Rounding $\delta$ to five decimal places changes
the low root by $-8.63\times10^{-8}$, so the exact and legacy routes are not
interchangeable.

All arguments of the exponential and Lambert $W$, and all quantities
$\alpha_s,\sin^2\theta_W,\delta,D,q,\Omega_i$, are dimensionless.  This
passes the narrow dimensional check.  It does not provide a physical
normalization: the scale/scheme for $\alpha_s$ and the formula
$4\alpha_s^{4/3}$ are declared model inputs, and a density fraction requires
critical-density and stress-energy definitions not contained in the fixed
point.

## Quantum and measure-theoretic seams

`quantum_jump_bridge.py` correctly labels its result conditional.  A positive
Kossakowski matrix plus a chosen Hamiltonian, jump operators, population
sector, and constant-hazard projector can establish a supplied Lindblad model
has a classical closed block.  It cannot select the physical instrument,
system--bath split, unravelling, outcome rule, or Born probabilities.  The
repository's coherent-Hamiltonian and collective-jump tests give explicit
counterexamples to population closure.  Even a closed Markov jump chain is
not an offspring process without an additional birth interpretation, reset,
independent increments, and genealogical independence.

For a measurable nonselected subprobability measure $\nu_{\rm ns}$ and a
measurable integrable kernel, the expression

$$
\phi_\beta(x)=\int_{\Gamma_{\rm ns}}K_\phi(x,\gamma)
\nu_{{\rm ns},\beta}(d\gamma)
$$

is a well-defined scalar-valued pushforward candidate.  It supplies neither
locality nor diffeomorphism covariance.  In particular a kernel depending on
an arbitrary global path functional is a counterexample to locality while the
integral still exists.  No CE action $S[g,\phi]$, metric variation
$T_{\mu\nu}=-2(-g)^{-1/2}\delta S/\delta g^{\mu\nu}$, Ward identity, or
conserved species current follows from the pushforward definition.  Thus
there is no derivation of $\nabla_\mu T^{\mu\nu}=0$.

## Cosmological mapping and status audit

The registry's named LO partition is arithmetically consistent after it
*adopts* $\Omega_b=q$ and $R=\alpha_sD$:

$$
\Omega_c=(1-q)\frac{R}{1+R}=0.25927170943410105,\qquad
\Omega_\Lambda=\frac{1-q}{1+R}=0.6920815709218708,
$$

and the three fractions sum to one.  This normalization is an algebraic
partition, not a derivation of baryon, cold-dark-matter, or vacuum species.
Changing the dark ratio gives a continuum of partitions with the same $q$;
the fixed point alone therefore underdetermines both $\Omega_c$ and
$\Omega_\Lambda$.  The old direct $q\mapsto\Omega_b$ readout is correctly
kept as an excluded/historical axiom, while its target remains incomplete.
No status inconsistency was found in the sampled core registry, cosmology
registry, and integrated manuscript: each retains the physical bridge as
axiom/incomplete rather than theorem.

## Severity ledger

| ID | Result | Severity | Scope and required closure condition |
|---|---|---|---|
| M1 | Low Poisson root/Lambert-$W$ branch and numerical certificate pass. | P2 | Valid only for independent Poisson offspring with finite dimensionless $D$. |
| M2 | $D_{\rm eff}$ chain is dimensionally valid but its physical input map is not derived. | P1 | Specify scale/scheme and a microscopic derivation of the positive rate matrix. |
| M3 | Quantum instrument, Born outcome rule, unravelling, and Poisson genealogy are absent. | P1 | Derive them from a declared microscopic action and state the approximation limit. |
| M4 | Pushforward does not imply a local covariant field, stress tensor, or conservation law. | P1 | Provide a covariant action and prove the Ward/conservation identities. |
| M5 | $q\mapsto\Omega_b$ and the DM/Lambda ratio are nonunique physical readouts. | P1 | Supply conserved currents, transition hypersurface, Einstein--Boltzmann evolution, and a fixed forward likelihood. |
| M6 | No sampled status inflation was found. | P2 | This is only a frozen-corpus consistency result. |

## Reproduction

Independent calculation: `artifacts/verify_math_lane.py`.

Command: `.codex/hooks/python.cmd python _workspace/ce/cosmology-quantum-audit-20260824/artifacts/verify_math_lane.py`.

Focused regression: `.codex/hooks/python.cmd pytest tests/test_cosmology_registry.py tests/test_quantum_jump_bridge.py -q`.

Result: `19 passed in 4.56s`.
