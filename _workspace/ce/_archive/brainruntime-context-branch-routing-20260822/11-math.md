# Mathematical check

Status: COMPLETE

The five-block Revision 1 removes the principal output-gating counterexample.
Only $H_c\times S_c$ changes with context; $Y\times H_0$ and $Y\times H_1$
are identical in both masks.  Therefore a correct-versus-wrong
difference cannot be attributed to a context-specific edge adjacent to $Y$.

Revision 0's extra relay was a genuine formula failure: after row-normalized
projection, activity reached $H_c$ but not the relay's emission event, so
$Y$ remained exactly zero.  Removing the redundant hop changes the mechanism,
not an endpoint threshold, and preserves the same context-identifiability
argument with a two-edge product.

For each block map, exactly $m$ selected edges do not alone imply a usable
$m$-dimensional channel.  The contract therefore requires rank $m$ and a
fixed lower singular-value gate before endpoints.  The wrong mask must decode
the opposite simultaneous payload, not merely produce silence; this rules out
a generic activity-loss explanation.

All dynamic quantities are normalized and dimensionless: $a,W,E,\rho,\mu,$
the learning rate, thresholds, gains, cosines, and margins.  $L$ is an integer
tick count.  The runtime energy readout, if reported, is a dimensionless
simulator proxy and cannot be interpreted as joules.

The rule is experience-supervised local eligibility: the experienced output
pulse is allowed during learning, but no output pulse, target, decoder, or
answer-derived projection is allowed at recall or mask construction.  A pass
identifies only context-dependent entry-branch selection through a shared
trunk.
