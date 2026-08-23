# BA-TR15 local attenuation-compensated binding contract

Date: 2026-08-22

BA-TR14 is frozen as `STOP` and is not retuned. BA-TR15 uses fresh BA-TR10
source-code seeds: calibration `102001`, development `102101..102116`. These
are disjoint from BA-TR10 development and sealed confirmation seeds and from
all BA-TR14 inputs.

The source pulse is at call 0, the true hidden arrival is at call 3, and the
sensory target experience is at call 6. The local trace and actual terminal
packet are

\[
z_h=\mathbf 1[a_h(3)>10^{-6}],
\qquad
p_h=[p_h^{\mathrm{axon}}(6)]_+.
\]

Only a synapse with both a retained event and a real terminal packet may write.
Its bounded dimensionless compensation is

\[
c_h=\mathbf 1[p_h>10^{-12}]
\operatorname{clip}\!\left(
\frac{10^{-4}}{10^{-12}+p_h},1,16
\right),
\]

\[
E_{yh}\leftarrow E_{yh}+[a_y(6)]_+z_hc_h,
\qquad
\Delta W_{yh}=\operatorname{clip}(E_{yh},0,13).
\]

Exactly one additive install occurs after four experiences. Its Frobenius norm
must be below 26 before the runtime boundary; global rescaling is forbidden.
There is no structural projection, top-k write, decoder, reward, endpoint,
target label, or store read in the learner.

Calibration may stop the route but may not change `p_ref`, epsilon, cap,
edge cap, learning rate, delay, threshold, or horizon. Development opens only
if calibration passes exactly as frozen. Development GO requires all 16 fresh
rows to recall 4/4 with minimum absolute margin at least `2e-5`, all timing,
support, raw/install, store, snapshot, and target-shuffle gates to pass, and
the compensated cohort to beat the attenuation-off cohort in pass count and
minimum margin. Packet-amplitude shuffle is logged as a matched adverse
control; eta-zero, time reversal, no target, uniform source code, and random
same-norm writes must fail their declared identity readout.

Claim ceiling: a bounded training-local attenuation compensation can support
synthetic delayed association in this runtime. It is not semantic memory,
biological homeostasis, curvature, graph morphology, or AGI.

