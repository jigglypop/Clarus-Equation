# Math spot check

Date: 2026-08-18

The calculation was independently evaluated in IEEE-754 double precision from the
declared input `alpha_s(M_Z) = 0.11789` and the displayed definitions:

$$
s_W^2=4\alpha_s^{4/3},\quad
\delta=s_W^2(1-s_W^2),\quad
D=3+\delta,\quad
q=e^{-D(1-q)}.
$$

| quantity | independently recomputed value | document value | result |
|---|---:|---:|---|
| $s_W^2$ | 0.231222068260755 | 0.2312220683 | agrees |
| $\delta$ | 0.177758423409974 | 0.1777584234 | agrees |
| $D$ | 3.17775842340997 | 3.1777584234 | agrees |
| $q_{\rm ext}$ | 0.0486467196440282 | 0.0486467196 | agrees |
| fixed-point residual | 0 at displayed double precision | — | passes |
| $Dq_{\rm ext}$ | 0.154587523120074 | 0.1545875231 | agrees |
| $R=\alpha_sD$ | 0.374625940535802 | 0.3746259405 | agrees |
| $\Omega_{\rm DM}$ | 0.259271709434101 | 0.2592717094 | agrees |
| $\Omega_\Lambda$ | 0.692081570921871 | 0.6920815709 | agrees |
| closure | 1.00000000000000 | 1 | agrees |

The repository's focused pytest command could not be collected in the available
interpreter because `torch` is not installed; this is an environment dependency
failure, not a numerical discrepancy. The direct spot check above has no package
dependency.
