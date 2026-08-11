# Loop 8F preregistration — finite-response gravitational decision field

Status: LOCKED BEFORE IMPLEMENTATION

## 1. Frozen predecessor

Loop 8E grid, Gaussian sources, screening, coupling, evidence streams, DDM,
motion friction/noise, time cost, and Loop 8C memory trace remain fixed. The
quasi-static arm is retained unchanged as a failed comparator.

## 2. Dynamic field

The online field obeys

`tau_phi^2 Phi_tt + 2 zeta tau_phi Phi_t
 + c_phi^2 L Phi + mu^2 Phi = -kappa (rho-mean(rho))`.

Locked additions:

- `tau_phi = 1.0`, `zeta = 0.80`, `c_phi = 1.0`;
- the existing `dt = 0.02`, grid spacing `dx = 0.025`, hence CFL
  `c_phi dt / (tau_phi dx) = 0.80`;
- zero initial field and field velocity;
- semi-implicit velocity-first field integration;
- no minimum capture step;
- capture requires mechanical energy below the instantaneous saddle, basin
  barrier `Delta Phi / T_d >= log(20)`, and the same side for `3` consecutive
  steps;
- freeze the field at capture and retain the existing 50-step flip check.

All quantities are normalized and dimensionless.

## 3. Arms

1. frozen fixed DDM;
2. frozen quasi-static gravity;
3. dynamic gravity;
4. dynamic mass shuffle;
5. dynamic source-sign flip.

Every arm receives identical evidence increments and frozen memory traces.

## 4. Gates

1. CFL <= `1`, zero-source/zero-initial field remains exactly zero, and an
   equal-mass source keeps central force <= `1e-10` for 100 steps.
2. Dynamic minus quasi-static accuracy LCB >= `+0.05` ID/OOD.
3. Dynamic minus fixed-DDM accuracy LCB >= `-0.01` ID/OOD.
4. Dynamic minus fixed-DDM utility LCB >= `0` ID/OOD.
5. Dynamic minus mass-shuffle accuracy LCB >= `+0.10` ID/OOD.
6. Dynamic minus sign-flip accuracy LCB >= `+0.20` ID/OOD.
7. Low-coherence mean capture time exceeds high-coherence mean by at least
   `10` steps ID/OOD, and overall mean capture time exceeds `10`.
8. Capture rate >= `0.90`, flip rate <= `0.02`, field/particle states finite,
   and maximum absolute field energy <= `1e4`.
9. Memory trace bit identity, no future reads, no environment clones.

All gates are conjunctive: `100 GO` or `0 STOP`. No dynamic-field coefficient,
capture confidence, persistence, or numerical scheme change after results.
