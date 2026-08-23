# BA-TR18 implementation

The runtime receives a second default-off input-aware competition mode. It
decomposes the actual delayed presynaptic packet over explicitly declared
source coordinates, applies the legacy singleton competition to each source's
recurrent contribution, and sums only the selected contributions. One source
continues through the exact legacy branch. Jitter is disallowed in this mode;
true delay and lateral gain 1 are required.

The experiment trains only four atomic cyclic associations with BA-TR15 and
evaluates four unseen two-source combinations. Controls are legacy global WTA,
BA-TR17 global adaptive top-2, source-index-misaligned factorization, and two
independent singleton clones.

