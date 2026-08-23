# Mathematical audit

Status: COMPLETE

With runtime convention $W_{ij}:j\to i$, define

$$
A=|W|\odot(\mathbf 1-I),\qquad q=|c|\in\mathbb R_{\ge0}^d.
$$

Then $f=q+Aq+A^2q$ is the nonnegative length-zero through length-two forward
relevance, while $R_{ij}=A_{ji}+(A^2)_{ji}$ is length-one/two return support
for edge $j\to i$.  The registered score

$$
S_{ij}=A_{ij}(\epsilon+f_j)
\left(1+\frac{R_{ij}}{\max_{k\ne l}R_{kl}+\epsilon}\right)
$$

is finite, nonnegative, and dimensionless for the contract's raw
normalized-runtime $W$ and cue, with no additional fitted normalization.  The
diagonal must be removed before computing $f$ and $R$;
removing it only from the final mask would let self-edges alter topology ranks.

An exact sparse budget is not automatic.  If a route has $M<B$ admissible
learned edges, exact matching is impossible.  The contract therefore makes
$M\ge B$ an apparatus precondition rather than silently filling from outside
the route.  Block and edge tie rules are total and deterministic.

Zero learned off-diagonal support, $B=0$, zero cue, and absence of a cue-active
source block are explicit apparatus failures.  For a nonempty admitted graph,
$R\equiv0$ is well-defined because the registered denominator is $\epsilon$.

`RuntimeStep.energy`, active-node fraction, and exposed-edge fraction are
simulator observables or dimensionless compute proxies.  None is a calibrated
physical energy in joules.

The new apparatus changes delay and threshold heterogeneity relative to the
M1 predecessor.  Consequently the full arm must re-establish binding before
masked comparisons; predecessor M1 success alone cannot certify this fixture.

No theorem makes a path/cycle mask improve recall.  The claim is empirically
falsifiable on the frozen development seeds.

## Leakage boundary

Every route constructor receives only sealed $W$, cue $c$, declared block
indices, public route seed, and budget.  Target codebooks, decoded identities,
post-rollout activation, and endpoint scores are excluded.  A focused
function-signature/source test must enforce this boundary.

## Counterexamples closed before implementation

1. Sparse admissible support can contain fewer than $B$ edges; this is now an
   explicit apparatus failure.
2. `TOPOLOGY > CLUSTER` alone cannot identify the return/cycle term because
   cue-path relevance changes simultaneously.  `PATH_ONLY` and
   `RETURN_SHUFFLED` now isolate that term.
3. A zero `TOPOLOGY`/`PATH_ONLY` mask distance carries no topology-specific
   information and cannot count toward the stronger claim.
