# BA-TR15 implementation

BA-TR15 leaves BA-TR14 frozen and generates fresh source-code matrices with
the unchanged BA-TR10 producer. Calibration seed `102001` and development
seeds `102101..102116` are disjoint from BA-TR10 confirmation and all BA-TR14
inputs.

At the true hidden arrival, a binary local event trace is retained. At target
experience time, the learner reads the actual axon-ring terminal packet on the
same hidden coordinate and applies

\[
c_h=\mathbf 1[p_h>10^{-12}]
\operatorname{clip}\left(\frac{10^{-4}}{10^{-12}+p_h},1,16\right),
\]

\[
E_{yh}\leftarrow E_{yh}+[a_y(6)]_+z_hc_h,
\qquad
\Delta W_{yh}=\operatorname{clip}(E_{yh},0,13).
\]

The compensation is coordinate-local and dimensionless. A zero packet cannot
write. All four experiences accumulate before exactly one runtime install;
the implementation rejects a raw Frobenius norm above 26 rather than allowing
the mutation boundary to rescale it. No structural projection or endpoint
signal enters learning.

Controls preserve schedule and support: compensation off, terminal-packet
amplitude shuffle, eta zero, target-experience shuffle, temporal reversal, no
target, uniform source-code reset, and random same-norm output write.

