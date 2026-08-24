# Physics-source lane: cosmology--quantum seam

Status: COMPLETE

Scope: source facts only. The implications below are explicitly labelled inference; no CE mathematical claim is evaluated or promoted.

## 1. Quantum measurement and open systems

| Input | Primary/authoritative source and version | Source fact | Audit relevance |
|---|---|---|---|
| Quantum instrument | [Blume-Kohout et al., PR A 2021, “Characterizing mid-circuit measurements…”](https://arxiv.org/abs/2103.03008) (2021) | An instrument is a collection of completely positive maps $\{\mathcal I_i\}$ whose sum is trace-preserving; outcome probability is $p_i=\mathrm{Tr}[\mathcal I_i(\rho)]$ and conditional state is $\mathcal I_i(\rho)/p_i$. A POVM has classical output and cannot by itself represent a general mid-circuit measurement. | Inference: a proposed environmental-selection link must specify its instrument/outcome maps (or an equivalent construction), not only a generator or scalar branching probability. |
| Markovian open dynamics | [Lindblad, Commun. Math. Phys. 48, 119 (1976)](https://doi.org/10.1007/BF01608499) | The general norm-continuous quantum dynamical semigroup preserving complete positivity has the Lindblad/GKSL structure. | Inference: writing a Lindblad-like equation is a conditional dynamical model; it does not by itself derive the system--environment coupling, measurement instrument, or Born rule. |
| Microscopic origin/validity | [Nathan & Rudner, PRB 102, 115109 (2020)](https://arxiv.org/abs/2004.01469) | A Lindblad master equation is derived under bath/coupling and Markov-approximation assumptions; approximation error is controlled by a dimensionless ratio of bath correlation and relaxation times. | Inference: a CE open-system equation needs stated bath, coupling, and approximation regime before it can support a physical measurement claim. |

## 2. GR/FLRW conservation and perturbations

| Input | Primary/authoritative source and version | Source fact | Audit relevance |
|---|---|---|---|
| FLRW background | [Einstein Online, “Friedmann–Lemaître–Robertson–Walker universe”](https://www.einstein-online.info/en/explandict/friedmann-lemaitre-robertson-walker-universe/) (accessed 2026-08-24) | FLRW is the family of homogeneous and isotropic solutions of Einstein’s equations; these are restrictive symmetry assumptions. | Inference: a homogeneous density split is not automatically a covariant local sector or a perturbation-level model. |
| Stress-energy conservation | [Einstein Online, conservation laws](https://www.einstein-online.info/en/explandict/conservation-laws/) (accessed 2026-08-24); see also [Noether overview](https://www.einstein-online.info/en/spotlight/A-guiding-light-through-Natures-darkness-symmetry-in-physics/) | Conservation is a statement about transfer within a complete system; Noether links conserved quantities to continuous symmetries. In GR, the matter equation is normally imposed consistently with diffeomorphism invariance and Einstein equations. | Inference: mapping a folded sector into dark matter/energy requires a covariant stress tensor and explicit exchange currents; a numerical fixed point for fractions is insufficient to establish total conservation. |
| Perturbations/observables | [Mukhanov, Feldman & Brandenberger, Phys. Rept. 215, 203 (1992)](https://doi.org/10.1016/0370-1573(92)90044-Z) | Cosmological perturbation theory evolves gauge-invariant perturbations and relates them to observable spectra; background FLRW equations alone do not supply those predictions. | Inference: a cosmological readout claim requires a forward perturbation/observable model (CMB, BAO, lensing, growth), not only $\Omega$-level background matching. |

## 3. Current observational reference values

Values below are model-dependent posterior constraints, not direct theory-free measurements.

| Dataset/model combination | Value (68% unless noted) | Source/version | Use and limitation |
|---|---:|---|---|
| Planck PR3 TT,TE,EE+lowE+lensing, base flat $\Lambda$CDM | $H_0=67.4\pm0.5$ km s$^{-1}$ Mpc$^{-1}$; $\Omega_m=0.315\pm0.007$; $\sigma_8=0.811\pm0.006$ | [Planck 2018 VI, A&A 641 A6 (2020)](https://doi.org/10.1051/0004-6361/201833910) | Standard reference baseline; posterior assumes base $\Lambda$CDM and Planck likelihood/model choices. |
| Planck base-$\Lambda$CDM physical densities | $\omega_b=\Omega_bh^2\simeq0.0224$ and $\omega_c=\Omega_ch^2\simeq0.120$ (table values depend on exact likelihood combination) | [Planck 2018 parameter paper](https://arxiv.org/abs/1807.06209) | Use physical densities with $h$ stated; do not compare $\omega_i$ directly to dimensionless $\Omega_i$. |
| DESI DR1 BAO alone, flat $\Lambda$CDM | $\Omega_m=0.295\pm0.015$ | [DESI 2024 VI, arXiv:2404.03002](https://arxiv.org/abs/2404.03002) | More independent late-time reference; BAO constraints are relative to sound-horizon calibration and model assumptions. |
| DESI full-shape + BAO; with CMB | $\Omega_m=0.3056\pm0.0049$, $\sigma_8=0.8121\pm0.0053$; with DES-Y3 also $H_0=68.40\pm0.27$ km s$^{-1}$ Mpc$^{-1}$ | [DESI 2024 VII, arXiv:2411.12022](https://arxiv.org/abs/2411.12022) | Illustrates dataset-combination dependence; not an independently fixed CE prediction test. |

## 4. Source-lane conclusion

The sources establish the external requirements: (i) a quantum instrument or equivalent CP outcome maps and a specified open-system derivation/approximation; (ii) covariant stress-energy/current conservation for any sector exchange; (iii) perturbation and observable forward modelling beyond an FLRW background; and (iv) explicit dataset/likelihood/model provenance for $\Omega$ comparisons. They do not establish any CE bridge. Numerical proximity to Planck or DESI posteriors is therefore, at most, a conditional consistency comparison unless the CE parameters and observable likelihood were fixed independently.

