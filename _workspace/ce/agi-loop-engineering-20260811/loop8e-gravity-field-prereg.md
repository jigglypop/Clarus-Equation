# Loop 8E preregistration — numerical gravitational decision field

Status: LOCKED BEFORE IMPLEMENTATION

## 1. Frozen inputs

Loop 8C generates one memory signal and target per trial. That trace is frozen
and shared by all decision arms. The field solver and decision layer cannot
write to PFC, MD, or residual state.

## 2. Field discretization

- one-dimensional action space `[-3, 3]`;
- `241` equally spaced grid points;
- action sources at `-1` and `+1`;
- Gaussian source width `sigma = 0.30`, normalized on the grid;
- symmetric finite-volume Neumann Laplacian;
- screening `mu = 0.60`, coupling `kappa = 1.00`;
- solve `(L + mu^2 I) Phi = -kappa (rho - mean(rho))`;
- force is the centered finite-difference `-dPhi/dx`.

The two unit-source basis fields are solved once. Trial fields are their linear
combination, so no arm receives a different numerical solver.

## 3. Common evidence stream

At normalized step `dt = 0.02`:

`e_(n+1) = e_n + q kappa_e dt + sigma_e sqrt(dt) xi_n`.

- initial `e_0 = 0.25 * memory_signal`;
- ID coherence levels `(0.10, 0.20, 0.40, 0.70)`, `sigma_e = 0.35`;
- OOD levels `(0.05, 0.15, 0.30, 0.60)`, `sigma_e = 0.45`;
- mass `m_plus = sigmoid(2 e)`, `m_minus = 1-m_plus`;
- maximum `500` steps.

Every arm receives the identical evidence increments.

## 4. Gravity motion and capture

`x_(n+1) = x_n + v_n dt`

`v_(n+1) = v_n + (-gamma v_n - grad Phi_n(x_n)) dt
             + sqrt(2 gamma T_d dt) zeta_n`

with `gamma = 1.0`, `T_d = 0.005`, `x_0=v_0=0`. Capture requires all of:

Motion uses reflection at `x = -3,+3`, matching the field's zero-flux boundary.

1. at least `5` integration steps;
2. the state lies on one side of the instantaneous inter-well saddle;
3. `H = v^2/2 + Phi(x) < Phi_s - 1e-6`.

At capture the source is frozen and the chosen basin is the captured side.
Fifty additional frozen-source steps measure escape/side flip.

## 5. Arms

1. `fixed_ddm`: first `abs(e) >= 1.0`, else deadline sign.
2. `linear_stn`: first `abs(e) >= 0.70 + C`, with
   `C = clamp(1-kappa_e/0.70,0,1)`.
3. `gravity_capture`: equations above.
4. `gravity_mass_shuffle`: evidence paths permuted across trials before sourcing
   the field; destructive causality control only.
5. `gravity_sign_flip`: swaps the two source masses.

## 6. Locked gates

All are conjunctive.

1. Discrete field residual infinity norm <= `1e-10` for both basis solves.
2. Equal-mass central force absolute value <= `1e-12`.
3. For mass pairs `0.7/0.3` and `0.3/0.7`, central-force sign matches mass
   difference and magnitudes agree within `1e-12`.
4. Gravity accuracy LCB minus fixed DDM >= `+0.02` ID and `0` OOD.
5. Gravity utility LCB minus linear STN >= `0` ID and OOD, with utility
   `(+1 correct,-1 wrong)-0.002*decision_steps`.
6. Gravity minus mass-shuffle accuracy LCB >= `+0.10` ID/OOD.
7. Gravity minus sign-flip accuracy LCB >= `+0.20` ID/OOD.
8. Mean capture time is strictly ordered from the two lowest coherence levels
   to the two highest levels in ID and OOD.
9. Capture rate >= `0.95`, post-capture side-flip rate <= `0.02`, and all
   positions/velocities/potentials are finite.
10. Frozen memory traces are bit-identical; no future reads or environment
    clones.

Score is `100 GO` only if every gate passes, else `0 STOP`. No coefficient or
grid sweep is allowed after results. A pass is a mechanism checkpoint only,
not a physical gravity, biological STN, or runtime claim.
