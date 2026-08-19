# Rust 3D metric kernel implementation

Status: COMPLETE

The run-local crate is at `artifacts/rust/nrm3d-core`. It implements full
six-component `f64` SPD tensors, spectral exp/log, strict Cholesky validation,
coframe and pullback laws, relative log deformation, finite-difference
Ricci/Riemann invariants, folded-ribbon atlas checks, and deterministic Rayon
fixtures. The NumPy oracle is `artifacts/reference_oracle.py`.

Only Gate A was implemented. The Gate B generator/fitter and all PFC adapters
remain sealed. Generated `target/` trees, copied `.exe` files, and historical
fixture lineages are disposable and excluded from version control.
