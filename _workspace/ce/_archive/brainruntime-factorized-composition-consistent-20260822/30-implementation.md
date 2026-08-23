# BA-TR19 implementation

BA-TR19 leaves the BA-TR18 source-factorized competition law unchanged and
repairs only the inconsistent auxiliary decision threshold.  Atomic probes
retain the existing single-label decoder, while pair and independent-union
probes use the same pre-existing activation floor
`MIN_DECODE_ACTIVATION = 1e-5` to form a multi-label target set.

For an arriving delayed source packet (p_s(t)), the active rule is

\[
c_h(t)=\sum_{s:p_s(t)\ne0}
\left[
[W_{hs}p_s(t)]_+-\max_{k\ne h}[W_{ks}p_s(t)]_+
\right]_+ .
\]

The source coordinates are declared apparatus inputs.  No target, decoder,
reward, endpoint, or external store enters this competition.  The probe also
records the number of source-coordinate packets in the actual delay-ring slot
before every runtime step.

The current BrainRuntime cue is not a one-shot spike: residual source
activation emits a short packet stream.  BA-TR19 therefore tests packet-stream
composition and does not reinterpret it as a biological single event.

