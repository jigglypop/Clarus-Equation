# 11. Independent mathematics lane — equality, dimensions, and special dimensions

Status: COMPLETE

Scope: E1--E5 and E7 of `00-contract.md`.  This is an independent derivation
from the stated rescaling action, rather than a confirmation from numerical
coincidences.  `d` is spatial/Riemannian dimension, `D` spacetime dimension,
and `p` a form degree unless stated otherwise.

## E1 — typed equality

Let nonzero scalar representatives have characters `u,v in Q^r`.  If `F=G !=
0` remains a numerical equality after every `lambda in (R_{>0})^r`, then
`chi_u(lambda)=chi_v(lambda)` for every lambda.  Put every component but `a`
equal to one and write `lambda_a=e^t`; then `e^{u_a t}=e^{v_a t}` for all real
`t`, hence `u_a=v_a`.  Conversely `u=v` multiplies both sides by the same
character.  Thus numerical equality of nonzero quantities is unit-covariant
iff their dimension characters agree.  This is only the dimensional gate:
well-typed equality can be stricter and require the same target bundle or
physical kind (equal characters alone do not identify, for example, energy
with torque).

`0` is an exception only when it is the zero *section of that same target*:
`F=0_u` is covariant for any fixed `u`.  A bare numeral zero has no type and
does not license comparing `F` to a scalar zero.  The concrete counterexample
`1 m = 1 s` loses equality after replacing the metre by `100 cm` but retaining
the second: the numerical representatives become `100` and `1`.

## E2 — defects and Buckingham--Pi

For positive scalar `F,G,S` of the same character, a unit rescaling multiplies
all three representatives by one positive factor.  Therefore

`|F-G|/S` and `|log(F/G)|`

are invariant.  The linear defect has zero set exactly `F=G` provided `S>0`
is finite and nonzero.  The logarithmic defect has the same zero set only on
the declared domain `F,G>0`; it is undefined at zero and for sign-changing
real quantities.  For a vector residual `r`, under a scalar unit change
`r -> a r`, `Sigma -> a^2 Sigma`, so `r^T Sigma^{-1}r` is invariant.  More
generally the correct transformation is `r -> A r`, `Sigma -> A Sigma A^T`.
Its zero set is `r=0` only when `Sigma` is symmetric positive definite.  An
invertible indefinite metric supplies a fatal counterexample: `diag(1,-1)`
gives `(1,1) Sigma^{-1} (1,1)^T=0` with nonzero residual.

For quantities restricted to the positive torus and dimension matrix `M`,
the monomial `Q^a` is invariant exactly when `M a=0`.  If `M` has rational
entries, its nullspace has a rational basis, so the registered Pi groups span
all rational-exponent monomial invariants.  Locally, all smooth invariants
are functions of independent Pi groups when the rescaling action has constant
rank.  This is not a global theorem over zeros or signs: sign sectors and
zero strata add invariants, and non-monomial invariants are functions of Pi
groups rather than themselves a single Pi monomial.

## E3 — finite-beta PreEq normalization

Fix a state space and a base measure, with weights `w_beta(x)=exp[-beta
delta(x)]`.  Replacing `delta` by `delta'=a+c delta`, `c>0`, yields exactly the
same normalized Gibbs distribution after the reparameterization
`beta'=beta/c`; the additive `a` cancels in normalization.  At fixed beta it
is unchanged only for `c=1` (up to an additive constant) when `beta != 0` and
the support realizes at least two distinct defect levels.  If `beta=0`, or if
`delta` is constant on the support, every normalized weight is already
degenerate and this uniqueness conclusion is false.  A reference-scale change
`S' = k S` in the linear defect gives `delta'=delta/k` and therefore requires
`beta'=k beta` for equality of finite weights.

A strictly monotone nonlinear map preserves the zero set and ordering, but
not generally the finite-beta family.  Equality of all probability ratios
requires `beta'[h(delta_i)-h(delta_j)]=beta(delta_i-delta_j)` for every pair
of realized defect values.  For nonzero beta and a nondegenerate support,
`h` must therefore be affine on the realized levels (except for one- or
two-level supports, where accidental agreement is possible).  At `beta=0`,
all normalized weights are uniform and impose no affine constraint.  Example:
defects `(0,1/2,2)` and `h(t)=t^2` cannot be compensated by one nonzero beta
because the two independent gaps give inconsistent ratios.
Consequently a finite-beta CE rule must record defect, reference scale, beta,
and base measure as structure; common zero sets alone are insufficient.

## E4 — Hodge, binomial, and cross-product classifications

On an oriented pseudo-Riemannian `n`-space of signature with `s` negative
directions (the convention used here),

`star^2|Lambda^p = (-1)^{p(n-p)+s}`.

Hodge duality maps `Lambda^p` to `Lambda^{n-p}`.  It is degree-preserving iff
`n=2p`; real self/anti-self-dual subspaces exist only if the displayed square
is `+1`.  If it is `-1`, the eigenvalues are `+/- i` after complexification,
not real self-duality.  Thus Euclidean `n=4,p=2` has real self-duality,
whereas Lorentzian 3+1 with one negative direction has `star^2=-1` on real
2-forms.  In `n=2p+1`, star maps p-forms to (p+1)-forms of equal dimension,
but this is an isomorphism between distinct degrees, not self-duality.

For integers `0<=p,q<=n`, binomial coefficients strictly increase from zero
through `floor(n/2)` and obey reflection symmetry.  Hence

`binom(n,p)=binom(n,q) iff q=p or q=n-p`.

First nontrivial cases: `n=2`: 1-forms self degree; `n=3`: 1- and 2-forms
pair; `n=4`: 2-forms self degree; `n=5`: 2- and 3-forms pair.  The CE equation
`dim Lambda^1=dim Lambda^2` is the particular case `(p,q)=(1,2)`.  The form
degree domain itself requires `n>=2`; within it the integer solution is
`n=3` only.  The formal root `n=0` of a polynomially extended equation lies
outside that form-degree domain and is not a solution.  Reading `n=3` as
physical space is an extra CE axiom, not a uniqueness theorem for nature.

A bilinear vector cross product satisfying orthogonality and the usual norm
identity exists on real Euclidean vector spaces only in dimensions `0,1,3,7`;
the nontrivial familiar cases are 3 and 7.  Normed real division algebras have
dimensions `1,2,4,8`, with imaginary-part dimensions `0,1,3,7`.  These are
algebraic classifications, not cosmological observations.

## E5 — field-theory dimensions

Use natural units, `[d^D x]=-D`, and dimensionless action.  A canonical scalar
has `[phi]=(D-2)/2`, so `[lambda_n]=D-n(D-2)/2` for `lambda_n phi^n`.  In
particular phi^4 is marginal in D=4 and phi^3 in D=6.  With canonical
normalization, `[g_YM]=(4-D)/2`; Yang--Mills is power-counting marginal in
D=4.  From `(16 pi G_D)^{-1} integral sqrt(-g) R`, `[G_D]=2-D`, so Newton's
constant is dimensionless in D=2.  This does not make D=2 Einstein gravity a
propagating UV-complete gravitational theory: the Einstein--Hilbert term is
topological there.

For a p-form potential `A_p`, a free Maxwell-type action `|F_{p+1}|^2` is
classically conformal when `D=2(p+1)`.  If `p` instead labels the field
strength degree, the same statement reads `D=2p`; the two conventions must
not be mixed.  The local massless-graviton polarization count is
`D(D-3)/2` for ordinary `D>=3` linearized Einstein gravity: zero in D=3 and
two in D=4.  D=2 is exceptional/degenerate (not a negative degree count), and
the usual local Einstein graviton begins only at D>=4.  Engineering dimensions
alone establish none of UV completion, compactification, or physical
realization.

## E7 — CE taxonomy and falsifiers

* `d=3` is the active spatial EFT input; `D=4` is the active spacetime EFT
  input.  Neither can silently be replaced by an algebraic `n`.
* `D_eff` in CE is a declared dimensionless control readout, not automatically
  Hausdorff, spectral, compact, or spacetime dimension.
* Internal fiber dimension, configuration/path-space dimension (often
  infinite), compact dimensions, and spectral/effective dimensions require
  their own type labels and cannot be counted as additional macroscopic axes.

Falsifiers are a unit-dependent equality conclusion; unlike-character
addition/equality; a dimensional exponent with dimensionless beta; unrecorded
reference scale/base measure; use of Hodge degree outside its domain; or
identification of internal/effective dimension with spacetime dimension.

## Findings requiring canonical narrowing

* **P0:** Any untyped `0`, or bare numerical equality of unlike dimensions,
  is invalid.  C3 has the asserted equality zero set only with positive
  definite covariance; invertibility alone is insufficient.
* **P0:** Same equality zero set does not imply the same finite-beta selection.
  Nonlinear monotone redefinitions generally alter the Gibbs family.
* **P0:** A Hodge/binomial special dimension or cross-product classification
  does not infer observed spacetime dimension or extra-dimensional existence.
* **P1:** Buckingham--Pi exhaustion needs positive nonzero variables and a
  constant-rank rescaling action; signs/zero strata require a stratified
  extension.
* **P1:** The D=2 Newton marginality and D=4 Yang--Mills marginality are power
  counting only, not UV-completion claims.

## Reproducibility

`.codex/hooks/python.cmd python _workspace/ce/equality-dimensionless-alternative-dimensions-20260825/artifacts/verify_equality_dimension_math.py`

The standard-library certificate checks a separating character, exact Gibbs
rescaling and nonlinear failure, binomial classification through n=14, Hodge
signature cases, and coupling-dimension special cases.
