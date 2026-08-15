# N-E2 preregistered set (written before any T=32 trajectory)

This file names the compact set and the scoring grid used for claim N-E2.
It does not contain occupancy numbers, division counts, or any image of
\(F_{1/4}^{32}\). Those numbers are produced later by
`verify_l3_algebra.py` after this file exists.

## Compact set

The registered compact is the source recursive rectangle already named
in the predecessor final report (cite only; not recomputed here):

$$
R := R_0
=
\left[\frac25,\frac35\right]
\times
\left[\frac49,\frac6{11}\right]
\subset[0,1]^2.
$$

Path cited: `_workspace/ce/agi-universe-sim-life-climb-20260814/40-final-report.md`
(definition block of \(R_0\)). Contract N-D3 allows this rectangle.

No \(\kappa=1/4\)-adapted replacement \(R_{1/4}\) is registered. The
concentric box below is a geometric subset of \(R_0\), not a
target-aware enlargement.

## Geometric center box (a priori subset)

Let \(m\)-width \(1/5\) and \(b\)-width \(10/99\). The center of \(R_0\)
is \(\bigl(1/2,\,49/99\bigr)\). The closed box of one-third linear scale
about that center is

$$
B_{\mathrm{c}}
=
\left[\frac{13}{30},\frac{17}{30}\right]
\times
\left[\frac{137}{297},\frac{157}{297}\right]
\subset R_0.
$$

This box is named from the geometry of \(R_0\) only.

## Grid

Let \(N=5\). The registered grid \(G\subset R_0\) is the product of
endpoint-inclusive uniform rational nodes

$$
m_i=\frac25+\frac{i}{4}\cdot\frac15,
\qquad
b_j=\frac49+\frac{j}{4}\cdot\frac{10}{99},
\qquad
i,j\in\{0,1,2,3,4\}.
$$

So \(|G|=25\). The center \(\bigl(1/2,\,49/99\bigr)\) is the node
\((i,j)=(2,2)\). Corners of \(R_0\) are included.

## Scoring (defined before evaluation)

Horizon \(T=32\). Pair each \((m,b)\in R_0\) with labels
\(q_0\in\{1/4,3/4\}\). Growth uses \(\kappa=1/4\) and (N.1) only.

- Occupancy indicator:
  \(\mathbf 1\bigl[(m_{32},b_{32})\in R_0\bigr]\).
- Division count: number of steps \(t=0,\ldots,31\) with
  \(\widetilde m_t\ge 3/4\).

N-E2 asks for a nonempty open subset of \(R_0\) on which one of these
two observables differs between the pair. A single pair, or a finite
grid by itself, is not that open set. The grid is a witness sample.
An open-set claim needs an interval enclosure on \(R_0\) or on
\(B_{\mathrm{c}}\).

## What is not registered

- No point chosen after seeing a trajectory.
- No enlargement of \(R_0\).
- No second compact defined from \(T=32\) images.
- LAS boxes around \(Z_\pm\) (claim N-E1) are a different object and
  are not used to score N-E2.
