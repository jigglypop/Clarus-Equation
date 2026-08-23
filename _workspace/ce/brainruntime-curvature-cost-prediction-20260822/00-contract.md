# BA-TR12 direct curvature-cost contract

Date: 2026-08-22

The curvature-sufficiency claim is tested first by the exact flat nonlinear
pair `A1=I2`, `A2=R(pi/4)`. Both have `g(0)=I` and intrinsic `K=0` everywhere,
but their finite-amplitude tanh saturation along `e1` differs. This is a
mandatory killing control.

The restricted learned-family diagnostic uses each frozen BA-TR10 matrix `B`,
its once-frozen top-two source plane `P`, and six predeclared hidden orthogonal
rotations `Q_r`. Every route has the exact same origin metric
`(Q_r B P)^T(Q_r B P)`. Eight fixed unit directions and calibration radii
`0,.1,...,.5` are used. The endpoint amplitude `A=1` is not used in either
cost.

Curvature cost:

\[
C^K_{r,u}=L_{r,u}\int_0^{.5}|K_r(su)|
\sqrt{u^Tg_r(su)u}\,ds.
\]

Metric-strain cost:

\[
C^g_{r,u}=L_{r,u}^{-1}\int_0^{.5}
\|g_0^{-1/2}g_r(su)g_0^{-1/2}-I\|_F,d\ell.
\]

Held-out finite-amplitude distortion:

\[
D_{r,u}(1)=
\frac{\|\tanh(Q_rBPu)-Q_rBPu\|_2}{\|Q_rBPu\|_2}.
\]

All quantities are dimensionless. Every geometry point must be rank-valid; no
ridge is allowed. A signed hidden permutation is an equality null. No output,
decoder, target, reward, semantic label, or runtime endpoint is read.

The exact flat pair necessarily rejects `K` as a sufficient route cost. The
learned-family result is reported as restricted association only. If metric
strain has lower mean held-out regret than curvature, classify
`K_INSUFFICIENT_METRIC_STRAIN_REQUIRED`; otherwise retain at most
`CURVATURE_ASSOCIATION_NOT_SUFFICIENT`.

