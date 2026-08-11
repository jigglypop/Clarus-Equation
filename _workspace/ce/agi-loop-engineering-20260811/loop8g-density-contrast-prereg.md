# Loop 8G preregistration — prior-relative density contrast

Status: LOCKED BEFORE IMPLEMENTATION

Loop 8F used total action-source density, which leaves two wells even when
`m_plus=m_minus=0.5`. Loop 8G changes only the source definition:

`delta_rho = sum_a (m_a - pi_a) K_a`, with uniform prior `pi_a=0.5`.

Thus equal-prior evidence produces exactly zero directional field. All Loop 8F
grid, PDE, integration, evidence, capture, noise, and comparison coefficients
remain fixed.

Arms: fixed DDM, contrast dynamic gravity, contrast mass shuffle, contrast sign
flip. All receive identical frozen memory traces and evidence increments.

Locked gates:

1. Equal-prior source and field remain <= `1e-12` for 100 steps.
2. Contrast gravity minus fixed-DDM accuracy LCB >= `-0.01` ID/OOD.
3. Contrast gravity minus fixed-DDM utility LCB >= `0` ID/OOD.
4. Contrast gravity minus shuffle accuracy LCB >= `+0.10` ID/OOD.
5. Contrast gravity minus sign-flip accuracy LCB >= `+0.20` ID/OOD.
6. Low-coherence capture time exceeds high-coherence by >= `10` steps ID/OOD.
7. Capture rate >= `0.90`, flip rate <= `0.02`, all states finite, field energy
   <= `1e4`.
8. Memory traces bit-identical, no future reads, no environment clones.

All conjunctive: `100 GO` or `0 STOP`. No parameter, prior, kernel, capture, or
threshold change after results.
