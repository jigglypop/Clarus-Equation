# BA-TR10 direct synthetic contract

Date: 2026-08-22

## Question

Can a source-specific hidden weight code form from an exactly uniform
`H <- S` substrate without a supplied microscopic edge code, using only an
exchangeable packet-local fluctuation and a local pre/post/weight update?

This is a synthetic mechanism test. Output weights, hidden/output pulses,
decoder, target, reward, endpoint score, biological identity, curvature, and
AGI claims remain closed.

## Frozen equations (Revision 1)

All variables are normalized and dimensionless. Initially

\[
B_{hs}(0)=1.
\]

Only when the true delayed recurrent packet reaches the hidden group,

\[
\zeta_h(t)=1+\sigma\tanh \xi_h(t),\qquad
\xi_h\overset{iid}{\sim}\mathcal N(0,1),\qquad 0\le\sigma<1,
\]

\[
u_h=\zeta_h[d_h]_+,
\qquad
v_h=e^{-\lambda r_h}u_h,
\qquad
c_h=[v_h-\max_{k\ne h}v_k]_+.
\]

Thus no packet gives no jitter-induced drive. The bounded factor obeys
`1-sigma < zeta < 1+sigma`. The frozen values are `sigma=.35`, `lambda=1`,
which satisfy

\[
\lambda>\log\frac{1+\sigma}{1-\sigma}=0.730887\ldots .
\]

The actual delivered presynaptic packet `p` and actual hidden activation `a`
write only the declared 4x4 support:

\[
\Delta B_{hj}=\eta a_h(p_j-a_hB_{hj}),\qquad
B^+=\operatorname{clip}_{[.2,2]}(B+\Delta B),\qquad \eta=4.
\]

Evaluation freezes `B`, sets `sigma=0`, resets fast and homeostatic state, and
reverses the training presentation order. A strict tie abstains.

## Revision receipt

The first calibration formula used unbounded log-normal jitter. It failed:
one draw produced factor `2.109756...`, overcame `exp(-1)` reuse suppression,
and yielded collision fraction `.25`. No development seed was opened. Revision
1 replaced only the jitter support by the bounded mean-one symmetric factor
above; thresholds, delay, support, local update, and endpoint closure stayed
fixed.

## Frozen stages and decision

- Calibration: seed `98301`.
- Development: seeds `98501..98516`.
- Confirmation: seeds `101901..101932`, sealed.
- Interpreter: `C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe`, Python 3.11.9; numpy 2.4.6; torch 2.12.1+cpu; pytest 9.1.1.

Every admitted seed must pass: exact uniform initial candidate weights; no
outside support; true first arrival at `L+1=3`; exactly one positive delivered
source component; local-only installed delta; complete washout; learned
jitter-off/fresh-state four-source bijection; normalized learned-column minimum
distance at least `1e-4`; `sigma=0` all-abstain/no-write; `eta=0` all-abstain;
source-independent row bias rejected; evaluation snapshot immutable; no
forbidden read.

Development is `GO` only if all 16 seeds pass, there are at least four distinct
learned permutations, and source slot 0 maps to every hidden coordinate across
the seed block. Otherwise stop; do not change the formula on these seeds.

## Claim ceiling

A pass identifies stochastic, path-dependent formation of a durable
source-column code within a declared synthetic support. It does not identify a
semantic binding, output computation, biological mechanism, manifold
curvature, or AGI.

