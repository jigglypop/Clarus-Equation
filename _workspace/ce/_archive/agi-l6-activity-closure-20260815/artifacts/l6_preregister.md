# L6 preregister (written before any one-step readout)

This file names the registered activity pair, the common bit, the
common drive, and the one-step readout used for \(L6\)-\(E1\),
\(L6\)-\(E2\), \(L6\)-\(E3\), and \(L6\)-\(H1\).
It contains no \((m',b')\) values, no \(T=32\) images, and no scored
trajectory.

Inherited geometry (cite only; predecessor
_workspace/ce/agi-l5-role-split-20260814/artifacts/l5_preregister.md
and L3 open-set
_workspace/ce/agi-l3-ne2-open-set-20260814/artifacts/o_e2_preregister.md):


$$
R_0
=
\left[\frac25,\frac35\right]
\times
\left[\frac49,\frac6{11}\right],
\qquad
U_0=\operatorname{int}(B_{\mathrm{c}})
=
\left(\frac{13}{30},\frac{17}{30}\right)
\times
\left(\frac{137}{297},\frac{157}{297}\right).
$$


The one-step map is the predecessor \(F_{1/4}\) with drive (L4.1) at
\(u=1\), \(\kappa=1/4\). No new channel from \(E\) or \(\sigma\) into
\((m,b,q)\) is authorized. \(q\)-maps stay uncoupled from drive.
Label \(q'\) is not an observable of this run.

## Registered points

Both points are written before evaluation. Both are required to lie
in \(U_0\times\{3/4\}\).


$$
P_{\star}
=
\Bigl(\tfrac12,\tfrac{49}{99},\tfrac34\Bigr),
\qquad
P_{\circ}
=
\Bigl(\tfrac{7}{15},\tfrac{49}{99},\tfrac34\Bigr).
\tag{L6.1}
$$


Membership in the open box (geometry only; no image):


$$
\frac{13}{30}
<
\frac{7}{15}
=
\frac{14}{30}
<
\frac12
=
\frac{15}{30}
<
\frac{17}{30},
$$



$$
\frac{137}{297}
<
\frac{49}{99}
=
\frac{147}{297}
<
\frac{157}{297}.
$$


Hence \((m,b)\) of both points lies in \(U_0\), and both labels are
\(q=3/4\). The two mass coordinates are distinct. The common
boundary coordinate is the inherited center abscissa of \(B_{\mathrm{c}}\).

## Registered bit and drive

Both points carry the same registered bit and the same drive:


$$
\sigma(P_{\star})=\sigma(P_{\circ})=1,
\qquad
u=1.
$$


The bit is an external label of this pair. It is not computed from a
wash in this run. Drive \(u=1\) recovers the predecessor \(F_{1/4}\)
growth bracket. No other \(u\) is scored.

## Readout

The activity readout of a point \(P=(m,b,q)\) is the one-step pair
\((m',b')\) of \(F_{1/4}(P)\) at the registered drive. The pair is
an element of \([0,1]^2\). This file does not record those numbers.

A bit predictor is any map \(\{0,1\}\to[0,1]^2\) that ignores \(P\)
and returns one pair from \(\sigma\) alone.

Occupancy of a copy remains the predecessor indicator
\(\mathbf 1[(m_{32},b_{32})\in R_0]\). This file does not score it.
\(L6\)-\(H1\) cites \(O\)-\(E1\) on \(U_0\times\{3/4\}\) at \(u=1\)
and does not name a new hull.

## What is not registered

- No one-step \((m',b')\) number.
- No \(T=32\) image, no occupancy bit, no recurrent drive
  \(u_t=\sigma m_t\).
- No second compact defined from images.
- No \(q_0=1/4\) pair, no zebrafish / calcium identity, no autonomy
  \(A\), no AGI.
- LAS boxes \(Q_{\pm}\) are not used to name these initials.
- No new channel from \(E\) or \(\sigma\) into the cube.
