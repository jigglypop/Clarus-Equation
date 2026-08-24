# Equality, dimensionless mathematics, and alternative dimensions

Status: COMPLETE

PREDECESSOR: none. Canonical starting points are
`docs/9_등호이전/04c_PreEq_보편스킴.md`,
`docs/검증_원장/참조_무차원_감사_수학.md`, and
`docs/2_경로적분과_응용/01_차원의_유일성.md`.

## 1. Question and claim ceiling

Develop the mathematics that makes a physical equality admissible before it
is used by the CE pre-equality/Gibbs machinery. In particular, separate
equality of dimensions from equality of numerical values, construct
unit-invariant dimensionless defects, and determine how the choice of
normalization affects finite-selection measures even when the zero set is
unchanged.

Systematically locate dimensions other than ordinary $3+1$ in which a
mathematical or physical structure becomes possible or special. The result
must distinguish spatial dimension $d$, spacetime dimension $D$, form degree
$p$, internal/configuration-space dimension, compact extra dimensions, and
effective or spectral dimension.

The maximal result is a collection of exact conditional classification
theorems plus a source-grounded map of established extra-dimensional model
classes and current observational status. It may not claim that a dimension
count proves which spacetime nature realizes, that CE has detected an extra
dimension, or that equal dimensionless numbers have a common physical cause.

## 2. Equality typing model

Let the base-unit rescaling group be

$$
G=(\mathbb R_{>0})^r,
$$

and let a quantity $Q$ with rational dimension vector
$u=(u_1,\ldots,u_r)\in\mathbb Q^r$ transform by the character

$$
Q\longmapsto \chi_u(\lambda)Q,
\qquad
\chi_u(\lambda)=\prod_{a=1}^r\lambda_a^{u_a}.
\tag{C1}
$$

An equality $F(x)=G(x)$ is `dimensionally typed` only when both sides are
sections of the same dimension character or the same dimensioned vector
bundle. The symbol $0$ is the typed zero of that target, not an automatically
dimensionless scalar.

For positive scalar quantities $F,G$ of the same dimension and a positive
reference scale $S$ with that dimension, register the candidate defects

$$
\delta_{\rm lin}=\frac{|F-G|}{S},
\qquad
\delta_{\log}=\left|\log\frac{F}{G}\right|.
\tag{C2}
$$

For vector residual $r=F-G$ with covariance or metric scale $\Sigma$ having
the corresponding product dimensions, register

$$
\delta_{\Sigma}=r^{\mathsf T}\Sigma^{-1}r.
\tag{C3}
$$

Every exponent in a CE Gibbs/pre-equality kernel must use either a
dimensionless defect with dimensionless $\beta$, or the more general typed
combination $[\beta]=[\delta]^{-1}$. The active CE branch uses the first
convention.

## 3. Dimension notions and candidate classifications

Use $d$ for spatial/Riemannian dimension, $D$ for spacetime dimension, and
$p$ for differential-form degree. Audit at least the following candidates.

1. Hodge duality $\star:\Lambda^p\to\Lambda^{d-p}$:
   self-degree sectors $d=2p$ and adjacent-degree sectors
   $\Lambda^p\simeq\Lambda^{p+1}$ at $d=2p+1$, with signature and reality
   qualifications.
2. General binomial equality
   $\binom d p=\binom d q$ and whether it has solutions beyond
   $q=p$ or $q=d-p$ in the registered integer domain.
3. Vector cross products and normed-division-algebra dimensions, including
   the special mathematical roles of $3$ and $7$, without converting them to
   cosmological evidence.
4. Canonical field dimensions in $D$ spacetime dimensions: scalar
   interactions, Yang--Mills coupling, Newton coupling, conformal $p$-form
   sectors, and local graviton degrees of freedom.
5. Physical model locations: Kaluza--Klein, compactification, large or warped
   extra dimensions, string/supergravity critical dimensions, and lower-
   dimensional defect/boundary theories.
6. CE-native non-spacetime locations: arbitrary-dimensional candidate spaces,
   infinite-dimensional path spaces, internal fibers, latent spaces, and
   effective/spectral dimensions.

## 4. Claims to prove, narrow, or reject

### E1. Typed equality and unit covariance

Prove that equality of two nonzero physical quantities is invariant under all
base-unit rescalings only when both sides carry the same dimension character.
Specify the typed-zero exception correctly and reject comparison by bare
numerical value across unlike dimensions.

### E2. Dimensionless equality defects

Prove the unit invariance and zero-set properties of (C2)--(C3), list their
domains, and determine how Buckingham--Pi nullspaces exhaust monomial
dimensionless invariants under (C1).

### E3. PreEq normalization dependence

Determine which changes of defect preserve only the equality zero set and
which also preserve the finite-$\beta$ Gibbs family. In particular test
$\delta\mapsto c\delta$, nonlinear monotone transforms, reference-scale
changes, and simultaneous $\beta$ compensation. Record normalization as
additional structure whenever finite residual weights change.

### E4. Hodge and algebraic dimension classification

Prove the general Hodge/form-degree classifications and enumerate the first
nontrivial dimensions. Separate algebraic isomorphism, real self-duality,
signature conditions, and any cross-product theorem.

### E5. Field-theory special dimensions

Derive by power counting the dimensions of scalar, Yang--Mills, and Newton
couplings in general $D$. Identify conditional marginal/conformal dimensions
and lower-dimensional gravity boundaries. Do not infer ultraviolet completion
from engineering dimensions alone.

### E6. Physical extra-dimensional routes

Verify primary or authoritative sources for major extra-dimensional theories
and current searches/constraints. Classify each route as mathematical
consistency, effective model, experimental constraint, or confirmed
observation.

### E7. CE integration and falsifiers

Produce a dimension taxonomy showing exactly where $d\ne3$ or $D\ne4$ can
enter CE without silently changing the active $3+1$ EFT branch. Register
falsifiers: unit-dependent conclusions, mismatched dimensions across an
equality, missing normalization scales, form degree outside its domain,
confusion of internal/effective dimension with spacetime, or unsupported
existence claims.

### E8. Focused implementation certificate

Audit the existing dimensionless checker. After the formal gate, add only a
focused equality-typing/dimensionless-defect certificate if the current API
cannot test E1--E3. Production code must not encode a physical claim about the
existence of extra dimensions.

## 5. Negative controls

Registered counterexamples are: $1\,{\rm m}=1\,{\rm s}$ by numerical value;
adding quantities of unlike dimension; inserting a dimensional defect into
$e^{-\beta\delta}$ with dimensionless $\beta$; changing $S$ in (C2) while
calling the finite Gibbs weights invariant; interpreting $d=2p+1$ Hodge
adjacency as observed spacetime; treating compact/internal or spectral
dimension as an additional macroscopic coordinate; and reading an absence of
experimental signal as a mathematical no-go for all compactification scales.

## 6. Authorized implementation scope

The source and mathematics lanes may create only the canonical numbered lane
files and run-local artifacts. After the status gate, the ledger writer may
update the smallest relevant dimensionless/formal-math ledger. After that
freezes, the paper writer may update the equality/pre-equality reader path and
the dimension-uniqueness derivation. A focused extension of
`reality_stone.clarus.dimensionless` and `tests/test_dimensionless.py` is
authorized only if E1--E3 lack an executable certificate. No cosmological
constant or dark-abundance file is in scope.
