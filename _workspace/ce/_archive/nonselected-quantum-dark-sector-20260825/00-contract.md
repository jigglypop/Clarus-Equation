# Nonselected quantum paths as a unified dark sector

Status: COMPLETE

PREDECESSOR: `_workspace/ce/_archive/cosmology-quantum-audit-20260824`

## 1. Question and claim ceiling

The user's core physical proposal is fixed as follows: quantum alternatives
that are not selected in the visible outcome are not deleted; they are mapped
to a residual physical sector, and the matter-like and vacuum-like regimes of
that same sector appear cosmologically as dark matter and dark energy.

This run asks how much of that proposal can be made into a covariant,
conserved, dimensionally consistent and falsifiable effective theory without
calling an adopted physical map a derivation. The maximal authorized result is
a minimal conditional EFT with explicit axioms, stress tensor, FLRW limits,
perturbative validity conditions and observational falsifiers. The run may not
claim that standard quantum mechanics already implies the residual sector, or
that CE predicts the absolute dark abundances without new input.

## 2. Frozen meanings

`Selected` means an outcome recorded by a declared quantum instrument in the
visible algebra. `Nonselected paths` means the complementary coarse-grained
history class before any physical identification; it is not automatically a
particle species, energy reservoir or Everett branch visible through gravity.
`Residual sector` means the new physical sector obtained only after the
following explicit map is adopted and shown well-defined.

Let $\nu_{{\rm ns},\beta}$ be a dimensionless nonselected subprobability and
let $\widehat K(x,\gamma)$ be a dimensionless local-covariant kernel candidate.
For a positive mass scale $M_*$ define

$$
\phi(x)=M_*
\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\,
\nu_{{\rm ns},\beta}(d\gamma).
\tag{C1}
$$

Equation (C1) is a physical-map axiom unless the instrument, history space,
kernel and scale are derived from a microscopic model.

## 3. Minimal residual-sector candidate

The candidate to test is a real minimally coupled scalar with a constant
vacuum offset,

$$
S_{\rm res}
=\int d^4x\sqrt{-g}
\left[
-\frac12 g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi
-V_\Lambda-\frac12m^2\phi^2
\right],
\qquad V_\Lambda\geq0,
\tag{C2}
$$

on the spatially flat FLRW branch. In units $c=\hbar=1$, $[\phi]=[m]=M$,
$[V_\Lambda]=M^4$ and $S_{\rm res}$ is dimensionless. The minimal branch has
no visible-sector energy exchange after the residual sector is established.

The proposed interpretation is:

- the rapidly oscillating massive contribution is the dark-matter-like
  component;
- the constant offset $V_\Lambda$ is the dark-energy-like component;
- both inherit the common residual-sector origin in (C1).

This interpretation does not identify a normalized discarded probability
with energy density. Initial amplitude, $m$, $M_*$ and $V_\Lambda$ remain
physical inputs unless independently derived.

## 4. Claims to prove, narrow or reject

### E1. Covariant conservation

For (C2), derive the metric stress tensor and prove its on-shell covariant
conservation. State exactly when separate residual-sector conservation fails
if interactions are restored.

### E2. Matter-like limit

For a homogeneous solution in the adiabatic regime $m\gg H$, prove that the
quadratic oscillatory contribution has period-averaged pressure
$\langle p_{\rm osc}\rangle\simeq0$ and density
$\langle\rho_{\rm osc}\rangle\propto a^{-3}$, with the approximation scale and
failure regimes stated. Determine the additional wavelength/mass conditions
needed to call its perturbations cold-dark-matter-like.

### E3. Vacuum-like limit

Prove that the additive constant has
$T^{(\Lambda)}_{\mu\nu}=-V_\Lambda g_{\mu\nu}$,
$p_\Lambda=-\rho_\Lambda$ and constant density. This is a conditional theorem
inside (C2), not a prediction of the observed value.

### E4. Unified residual origin

Decide the formal status of “dark matter and dark energy are nonselected
quantum paths.” The intended narrow form is an `[공리: 물리 사상]` saying that
(C1) supplies the field used in (C2), followed by the conditional E1--E3
theorems. Reject any stronger wording that treats path-integral alternatives,
decohered outcomes or a subprobability as automatically gravitating energy.

### E5. Numerical and observational scope

Retain the existing dimensionless Poisson fixed point only as an optional
composition/readout diagnostic. Test whether it fixes any parameter of (C2);
if not, record the non-identifiability. Give falsifiers using background,
structure growth, lensing, CMB/BAO and dark-matter clustering constraints,
without post-hoc promotion to prediction.

## 5. Falsifiers and revision rule

The candidate is rejected or narrowed if its stress tensor is not conserved,
the scalar contains a ghost or gradient instability, the matter-like average
fails in its declared regime, perturbations erase observed structure, the
vacuum term is counted twice, or the nonselected-path map violates locality,
covariance, instrument normalization or branch-conditioning consistency.

If the literal cross-branch gravitational interpretation conflicts with
standard quantum conditioning, the surviving claim must be phrased as a new
residual-sector physical-map axiom rather than as a consequence of ordinary
decoherence. A complete counterexample removes the stronger parent claim from
active canonical prose.

## 6. Authorized canonical scope

After the independent lanes and status gate, the ledger writer may update
`docs/검증_원장/상수_우주론_원장.md` and, if required for the new physical-map
claim, the smallest relevant quantum/bridge ledger. Once the ledger is frozen,
the paper writer may update `docs/5_유도/00_선택과_접힘.md` and the smallest
necessary cosmology narrative document. Code or tests may be changed only if
the approved equations need a focused regression; full-suite or release work
is not authorized.
