# Mathematical verification

Status: COMPLETE

R1 is the convex trajectory interpolation

`prediction = P + g(S-P)`, with `0 <= g <= 1`.

The training-only scalar least-squares projection reproduced
`g = 0.7868543064870357` over 176 H20 windows. Same-probe dense and zero-bridge
controls independently reproduced `0.7835668486813699` and
`0.882857758971467`.

Inference used the independent seed as the paired unit and the registered
two-sided Student-t endpoint `1.9693105698498752` for 255 degrees of freedom.
The only failed endpoint was the V5-parent lower improvement bound
`-0.003191881598197708`; strict positivity was required.

Component dynamics, rather than the nonrecursive output blend, were audited.
The retained maximum pathwise radius was `0.8216411318037443`, retained AR
maximum was `0.9640458798007217`, and the frozen sparse common-norm bound was
`0.96786`, all below `0.98`.
