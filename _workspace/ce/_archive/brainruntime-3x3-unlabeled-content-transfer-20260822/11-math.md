# BA-TR26 mathematics

## Definition

Eight shuffled observations contain only a dimensionless cue row
$r_i\in\mathbb R^8$ and a dimensionless current-packet content sum
$y_i\in\mathbb R^6$.  With centered matrices

$$
X=[r_i-\bar r]\in\mathbb R^{8\times8},\qquad
Y=[y_i-\bar y]\in\mathbb R^{6\times8},
$$

the operational rank is four.  If
$X=U_4\Sigma_4V_4^\top$, define

$$
K=YV_4\Sigma_4^{-1}U_4^\top,
\qquad
\widehat y(r)=\bar y+K(r-\bar r).
$$

The query is admitted only inside the certified affine span.  Current packet
coordinates are selected by the unique minimum

$$
A^*(r)=\arg\min_{A\subset J,\ |A|=2}
\left\|\widehat y(r)-\sum_{j\in A}
\frac{W_{H\leftarrow j}}{\|W_{H\leftarrow j}\|_2}\right\|_2.
$$

## Conditional theorem

If both cue and content rows obey one global rank-four affine law and the
training rows span that affine subspace, then $K$ reconstructs every query in
the span, including the missing ninth cell.  The implementation verifies this
by the SVD identity $Y=KX$, not by supplied grid coordinates.

The eight training cues contain exactly five unlabeled parallelograms in the
frozen generic fixture.  Their detected hypergraph covers every training row,
and the corresponding content sums obey the same equalities.

## Complete counterexample

The observed rows cannot prove the global law.  The alternative

$$
y_{22}=y_{22}^{\rm affine}+\delta,\qquad \delta\ne0,
$$

has identical eight-row evidence.  Hence BA-TR26 cannot establish arbitrary
factor discovery or distribution-free completion.  A $GL(4)$ change of latent
basis also leaves the fitted map on the observed cue span unchanged, so no
latent axis receives a unique semantic name.

## Dimensionless audit

$r,y,X,Y,U,V,K$, residuals, singular-value ratios, condition number, packet
weights, activations, and binding margins are normalized simulator quantities
with dimension vector $(0,0,0,0)$.  No dimensional value enters an exponential,
logarithm, probability kernel, or fixed point.  The numerical cutoffs are
dimensionless certification conventions, not physical constants.
