# G3-D development v1 apparatus status

Status: APPARATUS_INVALID — not scientific evidence and never confirmation-eligible.

The immutable JSON artifact is `g3-diagnostic-development-results-v1.json`, with canonical result
SHA-256 `ce2384031827f36e206c1e5bc81fb7d9875109e6c9cdb4b4003cc72ecb0ebc85`.

All 16 circuits passed training, structural, calibration, recall, source, and frozen-protocol
checks, but all 16 failed the lesion-bank integrity gate. Across 256 candidate installs, the
float32 applied-delta reconstruction residual was
`[1.6486802678628e-7,1.88206996654117e-7]`, while the frozen limit was `1e-7`.

Although the scientific fields were printed (`route_verdict=STOP`), they are quarantined because
`integrity_all=false`. Development seeds `97701..97716` are retired after inspection. Confirmation
seeds were not opened.
