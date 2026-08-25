# Mathematics lane — first-principles dark-sector derivation audit

Status: COMPLETE

Scope: this lane independently checks DSO-1--DSO-9. It neither supplies current observational numbers nor converts a fit into a proof; those belong to `10-sources.md` and the audit.

## Finding first

The strongest surviving result is conditional and three-step. A declared decohered neighbour Lindbladian supplies a finite-state facilitation process, but neither energy nor indefinite activity. A new local/covariant map from instrument-defined nonselected history to a residual scalar is needed to form a gravitational source. Given the stipulated scalar EFT, fast quadratic oscillations are dust-like and a constant offset has exactly `w=-1`.

The CE fixed-point probability does not contain field amplitude, mass, vacuum offset, transition time, or Planck normalization. It therefore cannot determine `Omega_DM`, `Omega_DE`, their ratio, or a transfer function. The historical direct routes `q_ext -> Omega_b`, `1-q_ext -> dark fraction`, and `R=alpha_s D_eff -> dark ratio` are excluded as derivations.

## DSO-1 — neighbour gate, finite extinction, energy

For occupation states `x in {0,1}^V`, declared jumps `L_(i<-j)=sqrt(kappa_ij) sigma_i^+ n_j`, and `R_i=sqrt(gamma_i)sigma_i^-`, assume a Hamiltonian diagonal in the occupation basis. Diagonal density matrices are then invariant and have exact CTMC rates

\[
b_i(x)=(1-x_i)\sum_j\kappa_{ij}x_j,\qquad d_i(x)=\gamma_i x_i. \tag{1}
\]

Thus an active neighbour exactly gates the listed birth jump. It does not provide its energy. For diagonal `H`,

\[
{d\over dt}\langle H\rangle=\sum_{ij}{\rm tr}[H\mathcal D[L_{i\leftarrow j}]\rho]+\sum_i{\rm tr}[H\mathcal D[R_i]\rho]. \tag{2}
\]

Any upward jump needs a bath/pump/drive current; arbitrary rates do not specify it. First moments do not close:

\[
\dot p_i=\sum_j\kappa_{ij}(p_j-C_{ij})-\gamma_i p_i,\qquad C_{ij}=\langle n_i n_j\rangle. \tag{3}
\]

At `11` in a two-node mutual graph, the exact node-1 gain is zero, so `dot n1=-gamma1`; the false linear equation gives `kappa12-gamma1`. A coherent term such as `Omega sigma_x` also breaks the diagonal closure.

The zero state is absorbing. On every finite graph with all `gamma_i>0`, successive positive-rate decays reach it from any state; it is the unique closed class and absorption occurs almost surely. An SCC is only a support graph for mutual facilitation, not survival.

**Status:** conditional theorem for the declared diagonal/decohered model. P0 counterexamples remove unconditional population closure, SCC-implies-survival, and energy-from-neighbour wording. Bath/drive and total-energy completion remain P1.

## DSO-2 — Poisson/Lambert-W root

Only a separately declared branching limit (fresh targets, negligible exclusion/collisions, independent offspring, decorrelated bath records, fixed parent window) has

\[
q_j=\exp\!\left[\sum_iA_{ji}(q_i-1)\right]. \tag{4}
\]

For irreducible nonsingular branching, survival requires `rho(A)>1`; the seed must reach the class. This is not a theorem about (1). Uniform row sum `A 1=D 1` gives

\[
q=e^{-D(1-q)},\qquad q_{ext}=-D^{-1}W_0(-De^{-D}). \tag{5}
\]

For `D>1`, `h(q)=log q+D(1-q)` is strictly concave, has exactly one root in `(0,1/D)` plus `q=1`; `W_0` selects the low root and `W_-1` selects `q=1`. `D`, `q`, and the Lambert-W argument are dimensionless; in the rate construction `A_ji=kappa_ij tau` is dimensionless.

For CE's supplied readout `D=3.1777584234099736`, independent iteration yields `q_ext=0.04864671964402820`, residual `-2.78e-17`, and

\[
{dq\over dD}=-{q(1-q)\over1-Dq}=-0.0547427647184780. \tag{6}
\]

This is only the stipulated scalar branching result. `D_eff` and its cosmological identification are CE readout assumptions. With exponential parent lifetime, `E[N]=kappa/gamma` and `Var[N]=kappa/gamma+(kappa/gamma)^2`; offspring are mixed Poisson, not Poisson, unless fixed-window or age-structured hypotheses are added.

**Status:** theorem inside the Poisson model; P1 for Lindblad-to-branching and microscopic `D_eff` derivation.

## DSO-3 — standard-conditioning counterexample

For a quantum instrument `{I_r}` and recorded outcome zero,

\[
\rho_0={I_0(\rho)\over {\rm tr}\,I_0(\rho)}. \tag{7}
\]

Local observables and any branch stress source are evaluated in `rho_0`. The complementary `I_1(rho)` is not added merely because it was unrecorded. This two-outcome counterexample removes automatic cross-branch gravity. A Lindblad equation admits multiple unravellings, so jump notation does not select records.

**Status:** P0 deletion of automatic nonselected-path gravity. A residual map is possible only as new physics.

## DSO-4 — probability/history to stress bridge

The strongest dimensionally admissible candidate is

\[
\phi(x)=M_*\int_{\Gamma_{ns}}\widehat K(x,\gamma)\nu_{ns,\beta}(d\gamma). \tag{8}
\]

With dimensionless `nu,K-hat` and `[M_*]=mass`, `[phi]=mass`. This does not provide an instrument/history measure, covariant local kernel, matching surface, source current, or no-double-counting rule.

If event `E` has probability `q` and energy weight `W`, then

\[
\Omega_E={E[W1_E]\over E[W]},\quad \Omega_E-q={q(1-q)(E[W|E]-E[W|E^c])\over E[W]}. \tag{9}
\]

Equality requires equal conditional mean energy, so probability normalization never implies energy normalization. The certificate takes `q=0.0486467` and weights 9,1, obtaining energy fraction `0.3151661`, not `q`. A transition must also have `nabla T_res=Q` and compensating `-Q`, conserving only total stress.

**Status:** (8) is a CE physical-map axiom. Locality, covariance, current/matching and no-double-counting are P1 open premises.

## DSO-5 — scalar dust limit and perturbations

For the stipulated action

\[
S_{res}=\int\sqrt{-g}\left[-\tfrac12(\nabla\phi)^2-\tfrac12m^2\phi^2-V_\Lambda\right]d^4x, \tag{10}
\]

metric variation yields canonical conserved stress on `Box phi-m^2phi=0`. In FLRW,

\[
\ddot\phi+3H\dot\phi+m^2\phi=0,\quad \rho=\tfrac12\dot\phi^2+\tfrac12m^2\phi^2+V_\Lambda,\quad p=\tfrac12\dot\phi^2-\tfrac12m^2\phi^2-V_\Lambda. \tag{11}
\]

For `psi=a^(3/2)phi`,

\[
\ddot\psi+[m^2-\tfrac32\dot H-\tfrac94H^2]\psi=0. \tag{12}
\]

When `H/m << 1` and `|dot H|/m^2 << 1`, WKB gives `phi=a^(-3/2)[A cos(mt+delta)+O(H/m)]`; averaging gives

\[
\langle w_{osc}\rangle=O(H^2/m^2,\dot H/m^2),\qquad \langle\rho_{osc}\rangle\propto a^{-3}[1+O(H^2/m^2,\dot H/m^2)]. \tag{13}
\]

This is not exact and not automatically CDM at all scales. Nonrelativistic scalar modes have `c_s^2 approximately k^2/(4m^2a^2)`; CDM-like growth needs `k/a << (mH)^(1/2)` and `k/a << m`. Thus mass/fraction/transfer have structure and lensing falsifiers.

**Status:** conditional EFT theorem; P1 for predicting mass, amplitude, initial data, fraction or transfer function.

## DSO-6 — constant offset

For constant `V_Lambda`,

\[
T^{(\Lambda)}_{\mu\nu}=-V_\Lambda g_{\mu\nu},\quad \rho_\Lambda=V_\Lambda,\quad p_\Lambda=-V_\Lambda,\quad w=-1. \tag{14}
\]

This exact theorem neither sets the observed vacuum magnitude nor establishes radiative stability. Current `w,w0,wa` are source-lane comparators; post-data potentials/interactions are a different fitted model.

**Status:** exact conditional theorem; P1 for magnitude and microscopic origin.

## DSO-7 — abundance no-go and historical readouts

The same `D,q` permits arbitrary `m,A,V_Lambda`. In the oscillatory limit density contains `m^2 A^2/2+V_Lambda`; the certificate keeps the same `q,m=7` but changes `(A,V)` from `(1,0)` to `(2,5)`, changing density from `24.5` to `103`. This continuous degeneracy proves that the fixed point alone cannot identify `Omega_DM` or `Omega_DE`.

Flatness only imposes `Omega_b+Omega_DM+Omega_DE+Omega_r=1`; it never supplies a split. For conserved baryon dust,

\[
{d\log\Omega_b\over d\log a}=3w_{tot}, \tag{15}
\]

so a nonzero constant `q` cannot equal it throughout an accelerating era. `R=alpha_s D_eff` is dimensionless but has no stress tensor, species map, current or abundance normalization. Rounded runtime tuples are contract-quarantined.

**Status:** P0 no-go for direct absolute-abundance/split claims. Required inputs are map normalization/current/transition, `m,A,V_Lambda,M_Pl`, initial conditions and an Einstein--Boltzmann forward model.

## DSO-8 — existing forward tests

The required map is

\[
(S_{res},T_{\mu\nu},{\rm initial\ data},{\rm other\ species})\to H(z),D(z),P(k,z)\to {\rm observables}. \tag{16}
\]

The frozen predecessor reproduced a DESI-DR2 13-vector conditional calculation: external `r_d=147.09 Mpc` gave `chi2=37.100260857`/13 dof (`p=3.996e-4`); fitting one scale gave `chi2=12.608346862`/12 dof (`p=0.3981`). The former is an external-input rejection test and the latter a fitted-nuisance result, not CE abundance predictions. This lane does not alter those figures before the source/covariance manifest freezes.

No active CE calculation supplies all inputs of (16) without external cosmological parameters. Central-value-only comparisons are marginal diagnostics, never joint significance.

## DSO-9 — strongest surviving statement

CE adopts a new physical-map axiom that may send instrument-defined nonselected history data into a residual scalar sector. A neighbour-bootstrap model gives a conditional microscopic motif. In the minimal scalar EFT, rapid quadratic and constant-offset limits behave respectively like dust and a cosmological constant. It is not established that dark matter/dark energy are unselected paths, that neighbouring quanta self-execute indefinitely, or that the mechanism predicts their observed abundances.

## Reproduction and severity

```powershell
.codex/hooks/python.cmd python _workspace/ce/dark-sector-observational-census-derivation-20260825/artifacts/verify_dso_math.py
```

The certificate reports `q_ext=0.048646719644028197`, residual `-2.776e-17`, `dq/dD=-0.054742764718478013`, mixed-Poisson variance `0.56 > 0.4`, and explicit probability/energy and abundance-degeneracy counterexamples.

| Finding | Status |
|---|---|
| finite positive-decay network absorbs; SCC insufficient | P0 counterexample |
| conditioning does not add unselected stress | P0 counterexample |
| probability does not fix energy | P0 counterexample |
| fixed point leaves continuous density family | P0 no-go |
| Poisson limit and `D_eff` origin | P1 |
| C1 covariant conserved physical map | P1 |
| scalar mass/fraction/transfer prediction | P1 |
