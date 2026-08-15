# O-E2 preregistered open sets (written before any new T=32 evaluation)

This file names the open sets used for O-E1, O-E2, and O-H1.
It contains only geometry of the already named rectangles \(R_0\) and
\(B_{\mathrm{c}}\). It does not contain occupancy numbers, division
counts, images of \(F_{1/4}^{32}\), or any scored trajectory.

Inherited names (cite only):

$$
R_0
=
\left[\frac25,\frac35\right]
\times
\left[\frac49,\frac6{11}\right],
\qquad
B_{\mathrm{c}}
=
\left[\frac{13}{30},\frac{17}{30}\right]
\times
\left[\frac{137}{297},\frac{157}{297}\right]
\subset R_0.
$$

Paths:
`_workspace/ce/agi-l3-nonlinear-las-t32-20260814/artifacts/n_e2_preregister.md`,
contract O-D2.

The common center of \(R_0\) and \(B_{\mathrm{c}}\) is
\(\bigl(1/2,\,49/99\bigr)\). Half-widths of \(B_{\mathrm{c}}\) are
\(1/15\) in \(m\) and \(10/297\) in \(b\).

## Registered open sets

All sets below are open axis-aligned boxes, concentric about
\(\bigl(1/2,\,49/99\bigr)\), and geometric subsets of \(B_{\mathrm{c}}\)
(hence of \(R_0\)). Scale factors are taken from the linear dimensions
of \(B_{\mathrm{c}}\) only. No set is enlarged after seeing an image.

### \(U_0=\operatorname{int}(B_{\mathrm{c}})\) (O-H1 first candidate)

$$
U_0
=
\left(\frac{13}{30},\frac{17}{30}\right)
\times
\left(\frac{137}{297},\frac{157}{297}\right).
$$

### \(U_1\) — concentric open box, one-third linear scale of \(B_{\mathrm{c}}\) (O-H1 second candidate)

Half-widths \((1/15)/3=1/45\) and \((10/297)/3=10/891\):

$$
U_1
=
\left(\frac{43}{90},\frac{47}{90}\right)
\times
\left(\frac{431}{891},\frac{451}{891}\right).
$$

### \(U_2\) — concentric open box, one-ninth linear scale of \(B_{\mathrm{c}}\)

Half-widths \((1/15)/9=1/135\) and \((10/297)/9=10/2673\):

$$
U_2
=
\left(\frac{133}{270},\frac{137}{270}\right)
\times
\left(\frac{1313}{2673},\frac{1333}{2673}\right).
$$

### \(U_3\) — concentric open box, one-twenty-seventh linear scale of \(B_{\mathrm{c}}\)

Half-widths \((1/15)/27=1/405\) and \((10/297)/27=10/8019\):

$$
U_3
=
\left(\frac{403}{810},\frac{407}{810}\right)
\times
\left(\frac{3959}{8019},\frac{3979}{8019}\right).
$$

The contract example “open box around \((1/2,49/99)\)” is \(U_1\)
(and the nested \(U_2,U_3\)).

## Scoring (defined before evaluation)

Horizon \(T=32\). Pair each \((m,b)\) in a registered \(U\) with
\(q_0\in\{1/4,3/4\}\). Growth uses \(\kappa=1/4\) and (N.1) only.

- Occupancy: \(\mathbf 1\bigl[(m_{32},b_{32})\in R_0\bigr]\).
- Division count: number of steps \(t=0,\ldots,31\) with
  \(\widetilde m_t\ge 3/4\).

O-E1 asks that occupancy differ for every pair in some nonempty
registered \(U\). O-E2 asks the same for the count. O-H1 claims that
\(U_0\) or \(U_1\) works for occupancy.

A finite grid is not an open set. A proof that assumes global
continuity of \(F_{1/4}^{32}\) is not authorized. An axis-aligned hull
that wraps both hybrid images after a mixed branch is not an enclosure.

## What is not registered

- No point chosen after seeing a trajectory.
- No enlargement of \(R_0\) or \(B_{\mathrm{c}}\).
- No second compact defined from \(T=32\) images.
- LAS boxes \(Q_\pm\) (predecessor N-E1) are a different object and
  are not used to *name* these \(U\).
