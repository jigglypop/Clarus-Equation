# Implementation

Status: COMPLETE

`BrainRuntimeConfig` retains the three scalar threshold fields and adds
optional immutable neuronwise active, lower-bit, and upper-bit tuples. Runtime
selection and Torch hysteresis resolve effective vectors at use time, so
post-construction scalar/vector mutation cannot leave stale caches. Vector-bit
configs force `auto` to Torch and explicit Rust fails closed; active-vector-only
selection remains allowed on the no-delay Rust cell path because final
selection occurs in Python.

The later A8-D packet-delay repair is orthogonal and is not promoted by this
run. Validation below rechecks the threshold contract on the current combined
source snapshot.
