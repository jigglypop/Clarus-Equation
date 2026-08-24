# Mathematics lane: nonselected quantum paths as a unified dark sector

Status: COMPLETE

## Finding first

The candidate action (C2) is a dimensionally consistent, ghost-free (for its
displayed sign) conditional residual-sector EFT.  It exactly supplies a
conserved metric stress tensor on shell, a cosmological-constant component,
and an adiabatic massive-scalar limit that behaves as dust.  It does **not**
turn a nonselected quantum-history probability into gravitating energy, nor
does it determine an absolute dark abundance.  The literal claim that the
unselected alternatives of a selected outcome gravitate in that selected
branch has a complete counterexample under ordinary conditional quantum
mechanics (M5 below).  It must be removed as a claimed consequence of standard
conditioning and retained, at most, as the new physical-map axiom stated in
M6.

## Definitions and dimensional conditions

Use natural units, metric signature $(-,+,+,+)$, and coordinates of mass
dimension $-1$.  A subprobability measure is dimensionless.  Thus (C1) gives
$[\phi]=M$ exactly if $M_*$ has mass dimension one, $\widehat K$ is
dimensionless, and the integral exists.  Its label ``local-covariant'' needs
more than these dimensions: the history space and measure must transform
covariantly, $\widehat K(x,\gamma)$ must transform as a scalar in $x$, and its
value must be determined by the local covariant data stipulated by the EFT;
an arbitrary global-history functional is a counterexample to locality.
Reality and regularity of the integral are also assumptions.  None follows
from a subprobability alone.

With $[\phi]=[m]=M$ and $[V_\Lambda]=M^4$, every term in (C2)'s Lagrangian has
dimension $M^4$, $d^4x$ has $M^{-4}$, and $S_{\rm res}$ is dimensionless.
The constant $M_*$ is a conversion scale, not a derived energy normalization.

## E1: stress tensor and conservation

Metric variation of (C2) gives

$$
T^{\rm res}_{\mu\nu}
=\partial_\mu\phi\partial_\nu\phi-g_{\mu\nu}
\left(\frac12\partial_\alpha\phi\partial^\alpha\phi+
V_\Lambda+\frac12m^2\phi^2\right).
\tag{M1}
$$

The field equation is

$$
\Box\phi-m^2\phi=0.
\tag{M2}
$$

Metric compatibility and commutation of covariant derivatives on a scalar
then yield

$$
\nabla_\mu T^{{\rm res}\,\mu}{}_{\nu}
=(\Box\phi-m^2\phi)\partial_\nu\phi=0.
\tag{M3}
$$

This is exact, not an FLRW approximation.  If an interaction
$-U_{\rm int}(\phi,\psi)$ is restored, then
$\nabla_\mu T_{\phi}^{\mu}{}_{\nu}=-\partial_\phi U_{\rm int}\,
\partial_\nu\phi=:Q_\nu$ on the scalar equation; separate conservation
fails unless $Q_\nu=0$, while conservation of the sum requires the visible
sector to carry $-Q_\nu$.  A map that injects residual energy at a transition
also needs a matching current or boundary/hypersurface stress term.  Otherwise
M3 cannot be asserted through the transition.

## E2--E3: homogeneous FLRW limits

For $ds^2=-dt^2+a^2(t)d\mathbf x^2$, $H=\dot a/a$, a homogeneous field obeys

$$
\ddot\phi+3H\dot\phi+m^2\phi=0,
\quad
\rho_\phi=\frac12\dot\phi^2+\frac12m^2\phi^2+V_\Lambda,
\quad
p_\phi=\frac12\dot\phi^2-\frac12m^2\phi^2-V_\Lambda.
\tag{M4}
$$

Hence $\dot\rho_\phi+3H(\rho_\phi+p_\phi)=0$.  In a model including other
sectors, the background equation is

$$
3M_{\rm Pl}^2H^2=\rho_{\rm other}+\rho_\phi,
\tag{M5}
$$

so (C2) alone neither fixes $H$ nor supplies a baryon sector.

Put $\psi=a^{3/2}\phi$.  The exact equation is

$$
\ddot\psi+\left[m^2-\frac32\dot H-\frac94H^2\right]\psi=0.
\tag{M6}
$$

If the frequency and background vary little in one period, for example
$H/m\ll1$, $|\dot H|/m^2\ll1$ (and the corresponding slow variation of these
ratios), WKB gives

$$
\phi=a^{-3/2}\left[A\cos(mt+\delta)+O(H/m)\right].
\tag{M7}
$$

Period averaging therefore gives

$$
\langle p_{\rm osc}\rangle/\langle\rho_{\rm osc}\rangle
=O(H^2/m^2,\dot H/m^2),
\qquad
\langle\rho_{\rm osc}\rangle\propto a^{-3}
\left[1+O(H^2/m^2,\dot H/m^2)\right].
\tag{M8}
$$

The leading equality of kinetic and quadratic-potential averages is checked
independently in `artifacts/verify_residual_eft.py`.  M8 fails or changes form
for $m\lesssim H$ (a slowly rolling/frozen field can instead have $w\simeq-1$),
for appreciable decay or visible exchange, nonadiabatic production, or a
nonquadratic potential.  For $V\propto|\phi|^n$, the standard oscillatory
average is $w=(n-2)/(n+2)$, displaying why quadratic curvature is essential.

The constant piece has

$$
T^{(\Lambda)}_{\mu\nu}=-V_\Lambda g_{\mu\nu},
\qquad\rho_\Lambda=V_\Lambda,
\qquad p_\Lambda=-V_\Lambda,
\qquad w_\Lambda=-1.
\tag{M9}
$$

It is separately conserved when $V_\Lambda$ is constant.  This is a theorem
of C2, not a prediction of the observed vacuum scale.

## Perturbative cold-dark-matter conditions

The homogeneous dust average is insufficient.  After averaging a canonical
massive scalar, modes have a scale-dependent effective pressure; in the
nonrelativistic regime its leading sound speed is

$$
c_s^2\simeq\frac{k^2}{4m^2a^2}.
\tag{M10}
$$

Thus CDM-like growth on a tested comoving mode requires $m\gg H$, physical
wavenumber $k/a\ll m$, and, more strongly, that it lie well below the scalar
Jeans scale $k_J/a\sim(mH)^{1/2}$ (order-one convention factors depend on the
background).  Initial adiabaticity, negligible isocurvature/decay, and no
large self-interaction or gradient instability are additional requirements.
For modes near or above $k_J$, wave pressure suppresses clustering; calling
the component CDM without this mode-by-mode limit is a P1 overstatement.

## M5: fatal counterexample to automatic cross-branch gravity

Take a measured qubit initially
$|+\rangle=(|0\rangle+|1\rangle)/\sqrt2$ and an ideal apparatus, whose
nonselective postmeasurement state is

$$
\rho'=\tfrac12|0,A_0\rangle\langle0,A_0|
+\tfrac12|1,A_1\rangle\langle1,A_1|.
$$

Conditioning on the recorded outcome $0$ gives
$\rho_0=|0,A_0\rangle\langle0,A_0|$.  For every branch-local observable $O$,
including any supplied visible stress operator, its selected-branch source is
$\operatorname{tr}(\rho_0O)$; the complementary history has coefficient zero.
The nonselected outcome has probability $1/2$, but no rule of conditional
quantum mechanics converts it into an added term in this expectation value.
Using instead $\operatorname{tr}(\rho'O)$ is an *unconditioned ensemble*
source, not a hidden energy component of outcome 0.  This counterexample is
independent of the numerical value of the nonselected probability.  Standard
quantum theory also has no settled derivation of classical gravity from this
conditioning rule; it cannot supply the desired extra gravitational source.

Therefore the parent statement ``unselected paths gravitate in the selected
branch'' is P0 false if presented as a consequence of standard measurement,
decoherence, or a path-integral sum.  It is not refuted as a new theory with
new degrees of freedom, but that theory must state its map and conservation
law explicitly.

## M6: strongest honest unified-origin statement and non-identifiability

The strongest surviving statement is the following physical-map axiom:

> A declared local-covariant map (C1), together with a declared transfer
> prescription conserving total stress energy, produces the scalar field of
> (C2).  Its oscillatory and constant-potential regimes are then respectively
> matter-like and vacuum-like by M1--M9.

This makes ``a common residual-sector origin'' meaningful without asserting
that ordinary discarded alternatives are already energy.  It additionally
requires an instrument, covariant history/kernel construction, a physical
$M_*$, initial data, the origin of $V_\Lambda$, and a no-double-counting rule
with the visible/environment sector.

Absolute abundances remain non-identifiable: at fixed $m$, changing the
oscillation amplitude changes $\rho_{{\rm osc},0}\simeq m^2A_0^2/2$ continuously,
and $V_\Lambda$ is an independent continuous input.  C1 adds the independent
normalization $M_*$ and kernel/measure data.  The dimensionless Poisson root
and any fraction partition contain none of $m$, $A_0$, $M_*$, $V_\Lambda$, or
$M_{\rm Pl}$, so they fix no parameter of C2.  Any claim that they predict
absolute $\Omega_{\rm DM}$ or $\Omega_\Lambda$ is P1 unsupported unless these
inputs are derived and propagated through a fixed Einstein--Boltzmann
likelihood.

## Severity ledger and reproduction

| ID | Finding | Severity | Required closure |
|---|---|---|---|
| M1 | C1/C2 dimensions and on-shell stress conservation pass conditionally. | P2 | Specify covariant/local kernel and transition matching. |
| M2 | Quadratic adiabatic scalar is dust-like; constant offset is vacuum-like. | P2 | Keep regime and perturbation conditions attached. |
| M3 | CDM label is scale dependent because of scalar wave pressure. | P1 | Test a fixed perturbation/structure forward model. |
| M4 | C1 does not normalize energy or absolute cosmic abundances. | P1 | Derive physical scale, initial data, and vacuum term. |
| M5 | Standard conditional QM does not make nonselected paths gravitate in the selected branch. | P0 for the stronger parent claim | Replace consequence language with M6 axiom or derive a new gravitational theory. |
| M6 | A common residual origin is viable only as an explicit physical-map axiom. | P1 | Supply microscopic map, total conservation, and no-double-counting test. |

Reproduction: `.codex/hooks/python.cmd python _workspace/ce/nonselected-quantum-dark-sector-20260825/artifacts/verify_residual_eft.py`.
