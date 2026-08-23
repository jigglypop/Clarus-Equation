# BA-TR18 final

Status: `FACTORIZED_ROUTE_64_OF_64 / CONTRACT_CONTROL_STOP_15_OF_16 /
CONFIRMATION_SEALED`.

Source-factorized pre-sum competition eliminated the identity interference
seen in BA-TR17: all 64 unseen pair routes produced the exact two-target set,
while global top-2 reached only 48 and global WTA reached none. This directly
supports the computational distinction

\[
\operatorname{Select}(Wp_i+Wp_j)
\ne
\operatorname{Select}(Wp_i)+\operatorname{Select}(Wp_j).
\]

The route cannot be promoted to a full contract GO because an auxiliary
independent-union control used a stricter `2e-5` component threshold than the
atomic decoder's `1e-5` threshold and missed two pairs in one seed. The main
mechanism did not fail, but the frozen all-gate decision remains STOP.

The next test must align the atomic and union decision rule before opening new
fresh seeds and retain the stronger factorized-vs-global controls. It must also
report that persistent source activity produces a packet stream rather than a
single biological spike.

Claim ceiling: synthetic source-provenance routing over an explicitly declared
source group; no learned discovery of source modules or biological claim.

