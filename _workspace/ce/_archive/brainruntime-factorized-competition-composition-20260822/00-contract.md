# BA-TR18 source-factorized competition contract

Date: 2026-08-22

Fresh BA-TR10 source-code seeds are calibration `105001` and development
`105101..105116`. Four atomic cyclic associations are learned with frozen
BA-TR15 local attenuation compensation; none of the four two-source pairs is
experienced.

For the explicit source coordinates whose axon packets arrive together, route
selection occurs before recurrent contributions are summed:

\[
r_h^{(s)}=[W_{hs}p_s]_+,
\qquad
c_h^{(s)}=\left[r_h^{(s)}-\max_{k\ne h}r_k^{(s)}\right]_+,
\qquad
c_h=\sum_{s\in A_t}c_h^{(s)}.
\]

The option is Torch-only, uses actual delayed packet provenance, and requires
explicit input coordinates, true delay, lateral gain 1, and jitter 0. One
source executes the original branch exactly. More than two simultaneous
sources remains fail-closed. Target identity, decoder, reward, label, and
endpoint are absent from routing.

Required gates: bitwise singleton parity and atomic 4/4; factorized pair 4/4
with exactly two positive H units and exact two-target set; legacy global WTA
0/4; source-coordinate-misaligned factorization 0/4; global adaptive top-2
reported as an interference control; independent-union 4/4; stores zero and
snapshots immutable. Development GO requires all 16 fresh rows and 64/64 pair
recall. No retuning after calibration.

Claim ceiling: synthetic two-packet source-provenance routing over an explicit
source group. It does not discover graph modules, biological axon labels,
curvature, or AGI.

