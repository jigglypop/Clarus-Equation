# Implementation

Status: COMPLETE

PREDECESSOR: `20-audit.md`

`BrainRuntimeConfig` now has a default-off local competition group. When the
group is present, the Torch path owns a floating homeostasis vector, a delayed
usage ring, and a scalar packet envelope. The actual delayed recurrent packet
is attenuated by homeostasis and passed through max-relative competition before
the ordinary drive update. After activation, a normalized squared-activation
usage vector enters the one-tick usage ring. No winner index, binding, occupied
mask, decoder, target, or reward enters these methods.

The new state is included in snapshot/restore and cleared by the existing
run-boundary reset. Structural mutation of group indices or usage-delay length
after construction fails closed. Explicit Rust rejects the feature, and
`backend="auto"` selects Torch because the Rust ABI has no matching state.
With `competition_indices=None`, the old recurrent path is unchanged.

The isolated experiment module keeps one runtime alive across four source
pulses and waits for a measured positive-packet washout rather than resetting
fast state between sources. It compares persistent homeostasis against
$\lambda=0$, uniform weights, source-independent row bias, a hidden-row
permutation, and midpoint snapshot continuation. Output weights and every
semantic endpoint remain absent.

Source and result hashes are frozen in `artifacts/source-freeze.json`.
