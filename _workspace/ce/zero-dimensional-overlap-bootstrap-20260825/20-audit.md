# Status audit: one-way zero-dimensional boundary and quantum bootstrap

Status: COMPLETE

Gate: PASS

Scope: revised `external strict 0D Z -> present 3+1D M` definition, source lane,
mathematics, alternatives, dimension audit and deterministic certificate. Narrow
canonical ledger/narrative integration is authorized; a post-integration audit
remains mandatory.

The stable snapshot consists of `00-contract.md`, `10-sources.md`, `11-math.md`,
`12-routes.md`, `artifacts/verify_zero_dimensional_overlap.py`, and
`artifacts/dimensionless-audit.md`.

## 1. Gate finding

No unresolved P0 or P1 defect remains in the audited snapshot. One P1 found in
the first revised audit was corrected: the instrument artifact had named a
fixed-output positivity check too broadly. For the declared
$\mathcal H_Z\cong\mathbb C$ domain, each output $\mathcal E_a(1)$ is now
explicitly treated as the complete Choi matrix. The checker verifies both Choi
minimum eigenvalues, each map's trace-nonincreasing condition, and the summed
map's trace-preservation residual.

The resulting certificate reports

$$
\lambda_{\min}(J_{\rm sel})=0,
\qquad
\lambda_{\min}(J_{\rm ns})=0.08686291501015238,
\qquad
r_{\rm TP}=0.
$$

This certifies the displayed finite-dimensional instrument, not every possible
cosmological boundary channel.

## 2. Final ZDO status ledger

| Claim | Final status | Decision |
|---|---|---|
| ZDO-1: “0D에서 일방향”은 $Z\to M$ sector/channel orientation이다 | definition, consistent | retain; never describe it as an intrinsic direction inside a point |
| ZDO-2: $\mathcal H_Z\cong\mathbb C$ CPTP map is fixed-state preparation | conditional theorem | retain only for the declared one-dimensional input; a fixed output may already encode correlations |
| ZDO-3: exact upstream no-feedback exists in an open cascaded GKSL model | conditional construction theorem | retain with propagating field/reservoir, approximation domain, noise and energy current explicit |
| ZDO-4: $\sigma_i^+n_j$ jumps induce the directed birth/death CTMC | conditional theorem | retain only in the diagonal/decohered closed population sector |
| ZDO-5: infinite independent Poisson genealogy obeys $q=e^{-D(1-q)}$ | conditional theorem | retain with fresh targets, independent clocks and negligible collisions; it needs neither reciprocity nor an SCC |
| ZDO-6: boundary injection and cosmological stress close | incomplete | require $J_Z^\nu$, channel/source stress, junction or initial-data matching and noise kernel |
| ZDO-7: nonselected records map to residual stress | CE physical-map axiom | retain as a new axiom, not a standard-conditioning theorem |
| ZDO-8: bootstrap/genealogy fixes absolute dark abundance | non-identifiability no-go | remove every direct $q\leftrightarrow\Omega$ identification |
| $M\to Z$-only sink generates a dark source back in $M$ | causal no-go | opposite orientation cannot produce an $M$-side co-output without an added return channel or modified gravity |

## 3. Exact surviving derivation

The admissible architecture is

$$
\underbrace{Z=\{\star\}}_{\text{static boundary datum}}
\xrightarrow{\text{preparation/instrument}}
\underbrace{\text{open source channel}}_{\text{no }M\to Z\text{ feedback}}
\longrightarrow
\underbrace{\text{directed cascade in }M}_{\text{conditional jumps}}.
$$

A normalized representative cascade is

$$
\dot\rho=-i\left[H_A+H_B+\frac{b^\dagger a-a^\dagger b}{2i},\rho\right]
+\mathcal D[a+b]\rho.
$$

The source partial trace of the cross generator vanishes exactly, while the
target partial trace is generally nonzero. The artifact reports zero algebraic
expansion residual, zero upstream-feedback residual and downstream-drive norm
$0.5$ for the declared sample.

In the diagonal/decohered sector,

$$
b_i(x)=(1-x_i)\sum_{j:j\to i}\kappa_{ij}x_j,
\qquad
d_i(x)=\gamma_i x_i.
$$

A finite DAG with no sustaining input does not establish infinite survival. In
the distinct infinite fresh-target Poisson limit,

$$
D=3.1777584234099736,
\qquad
q=0.048646719644028225,
\qquad
1-q=0.9513532803559718,
\qquad
Dq=0.15458752312007412.
$$

These are genealogy probabilities and cannot be relabelled as cosmological
density fractions.

## 4. Required boundaries and counterexamples

The following parent claims are closed or prohibited in canonical prose.

1. A strict point has an intrinsic spatial arrow or bare Hamiltonian clock.
2. A one-dimensional input state selects and updates different histories by
   itself. It may only prepare one fixed output, possibly with pre-encoded
   correlations.
3. A simple closed Hermitian exchange pair is an exact no-feedback cascade.
4. A finite DAG plus finite seed proves indefinite self-execution.
5. Complete positivity, neighbour occupation or a source label supplies free
   excitation energy.
6. Standard conditioning adds the stress of other outcomes to the selected
   outcome.
7. The reverse-only arrow $M\to Z$ produces a positive residual source in $M$.
8. The values $q$ or $1-q$ equal $\Omega_{\rm DM}$ or $\Omega_{\rm DE}$.

The earlier reciprocal common-bus rank theorem
$K=G\mathcal G G^\dagger$, $\operatorname{rank}K\le r$ remains correct inside
that separate linear model. It is no longer the centre of the user's model and
must appear only as a rejected comparison.

## 5. Residual and conservation boundary

The audited residual statement is only the declared physical map

$$
\phi(x)=M_*\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\nu_{\rm ns}(d\gamma).
$$

$\nu_{\rm ns}$ remains subnormalized so its total outcome weight is not erased
by conditioning. The map still needs a local/covariant kernel, matching rule,
no-double-counting condition and physical origin.

For a one-time boundary preparation, subsequent $M$ dynamics must satisfy
conservation of the total visible+residual+channel stress. For continuous
injection, $\nabla_\mu T_M^{\mu\nu}=J_Z^\nu$ must be completed by source/channel
stress or a junction/modification such that the full generally covariant system
is conserved. No such $J_Z^\nu$ has yet been derived.

## 6. Approved integration manifest

This ZDO revision authorizes only the following staged canonical documentation
changes:

- `docs/검증_원장/상수_우주론_원장.md`;
- `docs/5_유도/00_선택과_접힘.md`;
- `docs/5_유도/04_Dark_Energy_Derivation.md`.

The revised certificate and dimension audit remain research artifacts:

- `artifacts/verify_zero_dimensional_overlap.py`;
- `artifacts/dimensionless-audit.md`.

The integrated wording must preserve

$$
\text{static 0D boundary}
\to\text{open one-way channel}
\to\text{directed in-$M$ bootstrap}
\to\text{independent residual-map axiom}
\to\text{conditional DM/DE-like EFT}.
$$

It must identify ZDO-1--5 as definitions or conditional results, ZDO-6 as
incomplete, ZDO-7 as an axiom, and ZDO-8 as a no-go. This PASS is not evidence
that the external 0D boundary exists or that real dark matter/dark energy are
nonselected quantum histories.
