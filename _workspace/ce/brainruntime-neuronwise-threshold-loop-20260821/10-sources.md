# Sources

Status: COMPLETE

## Code sources

| source | admitted fact | forbidden inference |
|---|---|---|
| frozen `runtime.py` | scalar config fields, Torch comparisons, Python outer TopK, snapshot deepcopy/restore, backend dispatch | scalar implementation does not establish neuronwise thresholds |
| frozen Rust `lib.rs` and `kernel.rs` | bit thresholds cross the ABI as scalar `f32`; internal active threshold is `.22` | no vector-bit Rust claim |
| frozen `tests/test_runtime_contracts.py` | existing scalar no-delay Torch/Rust parity | it does not test vectors or delay-on parity |
| completed A7-H run | full discrete hybrid equation and exact Rust delay blocker | A7 smooth/branch evidence is not threshold implementation evidence |

## API compatibility evidence

Existing metric and benchmark consumers read `float(runtime.config.active_threshold)`.
Therefore replacing the scalar field with a sequence is not admitted. The additive optional
tuple API is selected specifically to preserve these callers and scalar dynamics. Dataclass
`asdict(config)` intentionally gains three deterministic optional keys even when their values
are `None`, so serialized config hashes/digests change; that schema change is receipted rather
than mislabeled as byte-for-byte serialization compatibility.

Snapshot state already deep-copies the entire config and `from_snapshot` invokes the runtime
constructor. Use-time threshold resolution therefore needs no mutable tensor fields in the
snapshot schema and also preserves post-construction scalar config mutation semantics.

## Input boundary

- all test values are synthetic and frozen in `00-contract.md`;
- no response, anatomy, training, task-score, or confirmation asset is opened;
- no package install, Rust rebuild, external network, or backend ABI change is authorized;
- circuit strength heterogeneity is read from the existing arbitrary signed weight matrix;
- threshold vectors supplied by a test are apparatus parameters, not measured neuron biology.
