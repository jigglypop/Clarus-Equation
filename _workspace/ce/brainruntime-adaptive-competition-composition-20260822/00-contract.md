# BA-TR17 delay-aligned adaptive competition contract

Date: 2026-08-22

Fresh BA-TR10 source-code seeds are calibration `104001` and development
`104101..104116`. Atomic memories use BA-TR15's frozen local attenuation rule
and cyclic target map `pi=(1,2,3,0)`. No joint pair is experienced.

Let `v_[1] >= ... >= v_[4]` be the attenuated hidden candidate values. The
number `K` is read only from the explicit source coordinates in the same axon
packet that is now arriving at H. For one arriving source, the existing
max-relative branch is executed byte-for-byte. For exactly two sources,

\[
\tau_2=v_{[3]},\qquad
c_h=\mathbf 1[v_{[2]}>v_{[3]}][v_h-\tau_2]_+.
\]

Thus the second winner is retained and a tie at the 2/3 boundary fails closed.
More than two arriving sources also fail closed because that capacity has not
been tested. The adaptive option requires `competition_lateral_gain=1`, true
axon delay, and explicit source indices. It reads no target, decoder, reward,
route label, or endpoint.

Required calibration and development gates:

- exact singleton parity with the legacy branch and 4/4 atomic recall;
- adaptive simultaneous recall 4/4 with exactly two positive first-arrival H
  units and the exact two expected Y components;
- count-blind legacy control 0/4;
- one-tick/source-coordinate-misaligned count control 0/4;
- independent-union capability control 4/4;
- tie and invalid-config tests fail closed;
- zero stores, immutable source snapshot, and no confirmation opening.

Development GO requires all 16 fresh rows, hence 64/64 simultaneous pairs.
No thresholds, gains, write caps, source producer, or horizon may be tuned after
calibration.

Claim ceiling: a deterministic two-packet synthetic routing-capacity rule. It
does not discover source groups, modules, support, biological circuitry,
curvature, or AGI.

