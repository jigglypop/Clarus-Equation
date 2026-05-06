# Recursive Cosmology Research Audit

## Core recursive state

| quantity | value |
|---|---:|
| sin2_theta_w | 0.23122207 |
| delta | 0.17775842 |
| D_eff | 3.17775842 |
| x | 0.04864672 |
| sigma | 0.95135328 |
| fixed-point residual | -1.041e-16 |
| contraction D_eff*x | 0.15458752 |
| N_e | 57.19965162 |
| phase area pi^2/2 | 4.93480220 |
| integrated boundary defect pi*delta*sigma | 0.53127806 |
| endpoint defect delta*sigma | 0.16911106 |
| H0 global q=0 | 67.247245 |
| H0 endpoint q=1 | 73.180689 |
| branch gap | 5.933444 |

## Remaining recursive leverage

| priority | name | layer | current status | safe extension | guardrail | next gate |
|---:|---|---|---|---|---|---|
| 1 | H0 q-selector | selector | strongest open recursive lever | predict q from source/covariance graph before reading H0 | do not patch high H0 branch without selector derivation | prospective_covariance_graph_selector_gate |
| 1 | residual contraction cascade | cross-observable residual | open | ask whether failed raw terms share a contraction/projection rule | do not tune a separate recursion for each residual | residual_cascade_invariant_gate |
| 2 | A_s / A3c residual readout | readout | selection candidate; raw total sensitivity rejected | reuse the same projected residual in spectrum and anomaly handles | do not mark A3c exact before likelihood or n_i derivation | primitive_spectrum_common_readout_gate |
| 2 | fixed-point kernel | kernel | closed mathematical fixed point | test constrained deformations of K(x) before adding observables | do not change the kernel per observable | kernel_deformation_no_free_parameter_gate |
| 3 | d0 zero-residual boundary | boundary | candidate boundary condition | derive measure map from d0 identity to d3 contracted branch | do not call d0 a physical place or reachable state | d0_measure_transport_gate |
| 3 | horizon phase-area lift | bridge | conditional bridge | derive why late horizon entropy reads primordial phase area | do not treat pi^2/2 lift as local slow-roll entropy growth | early_late_measure_preservation_gate |

## Verdict

More self-recursion remains usable, but mainly as selector/readout recursion rather than a new core fixed point. Top targets: H0 q-selector, residual contraction cascade.

The next useful calculation is not another unconstrained correction term. It is a prospective selector/residual audit: predict the readout layer first, then compare the observable.
