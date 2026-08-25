# Source lane: one-way zero-dimensional boundary and quantum cascade

Status: COMPLETE

Revision: 1 — `Z -> M` 단방향 경계 해석으로 재조사. 기존 common-bus 자료는
비교용 경로로만 유지한다.

Accessed: 2026-08-25

## 1. Primary-source ledger

| ID | Primary source | Supported statement | Boundary for CE |
|---|---|---|---|
| ZS-0D-01 | Rivasseau, *Constructive Field Theory in Zero Dimension*, [DOI 10.1155/2009/180159](https://doi.org/10.1155/2009/180159) | A zero-dimensional field theory is represented by an ordinary integral rather than a spacetime functional integral; there are no spacetime derivative terms. | Supports strict 0D as a static measure/toy interface. It does not supply time evolution, causal propagation or bootstrap execution. |
| ZS-CAS-01 | Gardiner, *Driving a Quantum System with the Output Field From Another Driven Quantum System*, [DOI 10.1103/PhysRevLett.70.2269](https://doi.org/10.1103/PhysRevLett.70.2269) | Derives quantum Langevin and master equations in which the emitted output field of a first driven atom drives a second atom. | Establishes a source-to-target open-system precedent. It does not identify the source with a strict 0D cosmological boundary. |
| ZS-CAS-02 | Carmichael, *Quantum trajectory theory for cascaded open systems*, [DOI 10.1103/PhysRevLett.70.2273](https://doi.org/10.1103/PhysRevLett.70.2273) | Formulates quantum trajectories for an open quantum system driven by a photoemissive source. | Supports a one-way cascade with a propagating output/reservoir channel; it does not supply the CE residual map. |
| ZS-SLH-01 | Combes, Kerckhoff and Sarovar, *The SLH framework for modeling quantum input-output networks*, [arXiv:1611.00375](https://arxiv.org/abs/1611.00375), [DOI 10.1080/23746149.2017.1343097](https://doi.org/10.1080/23746149.2017.1343097) | Represents open nodes by $(S,L,H)$ triples and composes output fields into downstream inputs; feedback is a separate network operation. | Supplies a modern conditional implementation of feed-forward directionality, with Markov/input-output approximations. It is not evidence for a literal external 0D ontology. |
| ZS-CHI-01 | Pichler et al., *Quantum Optics of Chiral Spin Networks*, [arXiv:1411.2963](https://arxiv.org/abs/1411.2963) | Derives driven-dissipative spin-network dynamics mediated by asymmetric/chiral one-dimensional bosonic waveguides. | Shows that spatially directed propagation requires channel chirality/asymmetry; a bare point does not create directionality. |
| ZS-INS-01 | Davies and Lewis, *An operational approach to quantum probability*, [DOI 10.1007/BF01647093](https://doi.org/10.1007/BF01647093) | Introduces the operational instrument framework that combines outcome probabilities with state transformations. | Supports selected/nonselected outcome bookkeeping. This run separately imposes the finite-dimensional CP/TNI conditions; identifying the nonselected outcome with a gravitating residual field remains a CE axiom. |
| ZS-TH-01 | Strasberg and Winter, *First and Second Law of Quantum Thermodynamics: A Consistent Derivation Based on a Microscopic Definition of Entropy*, [DOI 10.1103/PRXQuantum.2.030202](https://doi.org/10.1103/PRXQuantum.2.030202) | Gives microscopic open-system definitions and first/second-law accounting for internal energy, work, heat and entropy. | Requires source/reservoir/work currents to accompany a physical one-way channel; a neighbour gate is not automatically the energy source. |
| ZS-OQS-01 | de Vega and Alonso, *Dynamics of non-Markovian open quantum systems*, [DOI 10.1103/RevModPhys.89.015001](https://doi.org/10.1103/RevModPhys.89.015001) | Reviews reduced dynamics obtained by coupling a system to an environment and the assumptions behind Markov and non-Markov treatments. | Supports keeping the propagating environment and approximation domain explicit; it does not derive CE cosmology. |
| ZS-AE-01 | Warszawski and Wiseman, *Adiabatic Elimination in Compound Quantum Systems with Feedback*, [arXiv:quant-ph/0005127](https://arxiv.org/abs/quant-ph/0005127) | A fast ancillary quantum system may be eliminated in a controlled response-time regime, leaving an effective master equation for the retained system. | Supports a $0+1$D ancilla/bus as a conditional mediator. It does not derive an arbitrary target jump. |
| ZS-AE-02 | Azouit, Sarlette and Rouchon, *Adiabatic elimination for open quantum systems with effective Lindblad master equations*, [arXiv:1603.04630](https://arxiv.org/abs/1603.04630) | Open-system adiabatic elimination can preserve an effective Lindblad structure under its stated scale separation. | Supports a completely positive reduced model only after fast-mode, Markov and perturbative assumptions are stated. |
| ZS-CAV-01 | Hagenmüller et al., cavity-mediated coherent and dissipative couplings, [arXiv:1912.12703](https://arxiv.org/abs/1912.12703) | Eliminating a lossy cavity mode can generate both coherent and dissipative effective emitter couplings in a bounded parameter regime. | Supports a response-kernel picture; it does not make the result local or occupation-facilitated automatically. |
| ZS-COL-01 | Damanet, Braun and Martin, *Master equation for collective spontaneous emission with quantized atomic motion*, [arXiv:1512.06676](https://arxiv.org/abs/1512.06676) | A shared electromagnetic reservoir produces collective decay described by a dissipative coupling matrix and collective jump modes. | Supports a positive-semidefinite Kossakowski matrix and linear collective channels, not the specific raising jump $\sigma_i^+n_j$. |
| ZS-OP-01 | Reiter and Sørensen, *Effective operator formalism for open quantum systems*, [DOI 10.1103/PhysRevA.85.032111](https://doi.org/10.1103/PhysRevA.85.032111) | Eliminating excited ancillary states can yield engineered effective Hamiltonians and jump operators. | Shows that a target jump may be engineered from additional microscopic structure; the CE jump still requires an explicit derivation. |
| ZS-NL-01 | Leghtas et al., *Single-photon resolved cross-Kerr interaction for autonomous stabilization of photon-number states*, [arXiv:1504.03382](https://arxiv.org/abs/1504.03382) | Nonlinear cross-Kerr coupling permits photon-number-dependent driven-dissipative control. | Supports the feasibility of number-conditioned channels, not universality or the exact CE neighbour operator. |
| ZS-LR-01 | Bravyi, Hastings and Verstraete, *Lieb-Robinson bounds and the generation of correlations and topological quantum order*, [arXiv:quant-ph/0603121](https://arxiv.org/abs/quant-ph/0603121) | Local finite-range Hamiltonians have an effective light cone and finite correlation/information propagation speed. | A bare instantaneous global bus has no such guarantee. CE must provide geometry, retardation or an explicitly nonlocal axiom. |
| ZS-EN-01 | Manzano et al., *Quantum stochastic thermodynamics: principles and perspectives*, [DOI 10.1103/PhysRevE.100.022127](https://doi.org/10.1103/PhysRevE.100.022127) | Driven open-system energy accounting separates internal-energy change, work and heat, including ancillary and measurement/control contributions. | A neighbour can gate a transition but cannot be silently designated its energy source. Seed, drive, bath, bus and currents must be recorded. |

## 2. Source-supported chain

The revised sources support the conditional architecture

$$
\underbrace{Z=\{\star\}}_{\text{static boundary datum}}
\xrightarrow{\text{preparation/instrument}}
\underbrace{\text{source output field}}_{\text{open channel}}
\xrightarrow{\text{cascade/chiral propagation}}
\underbrace{M\text{ target dynamics}}_{\text{no upstream feedback}}.
$$

The arrow belongs to the channel, not to an intrinsic coordinate of the 0D
point. Gardiner and Carmichael establish a quantum-optical source-to-target
cascade; the SLH and chiral-network literature supplies explicit feed-forward
network realizations. These constructions require a propagating field,
reservoir, chirality/isolator, or equivalent feed-forward operation. They do
not show that a bare 0D object itself propagates, stores histories or supplies
energy.

The older ancillary-elimination/common-bus sources below remain useful only for
the rejected alternative in which all nodes couple reciprocally to a shared
mode. Their low-rank and locality restrictions do not refute the revised
one-way cascade, because that cascade is not being derived from one reciprocal
mode.

## 3. Unsupported claims retained as hypotheses

No cited source establishes that

1. a strict 0D point has an intrinsic arrow, clock or self-updating dynamics;
2. the cosmological boundary $Z$ physically exists or emits the initial state;
3. cascaded quantum optics automatically produces the nonlinear neighbour jump
   $L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\sigma_i^+n_j$;
4. a channel called zero-dimensional is automatically relativistically local;
5. nonselected quantum histories become a local covariant stress tensor; or
6. the resulting sector equals the observed dark matter/dark energy or fixes
   either absolute abundance.

These items remain CE axioms, open derivations or falsified parent claims as
classified by the math and status lanes.
