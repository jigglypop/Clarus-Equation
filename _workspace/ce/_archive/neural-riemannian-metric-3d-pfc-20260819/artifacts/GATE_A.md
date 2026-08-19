# Gate A kernel record

Status: `KERNEL_PASS`

Scope: numerical and differential-geometric fixtures only. No true/null Gate B
dataset and no PFC outcome was run.

## Required fast gate

```powershell
cargo test --locked --manifest-path artifacts/rust/nrm3d-core/Cargo.toml --lib
cargo run --locked --manifest-path artifacts/rust/nrm3d-core/Cargo.toml --bin nrm3d-core -- --fixtures
python -B artifacts/reference_oracle.py
```

The final source produced 39 passing fixtures and six passing Rust tests.
Important values were:

| Fixture | Value | Gate |
|---|---:|---:|
| six-component spatial exp/coframe/log error | `1.2406742300186124e-14` | `<= 1e-11` |
| signed ribbon Jacobian minimum | `0.038424003022341446` | `> 0` |
| folded-anatomy Riemann norm | `1.3694212555634001e-7` | `<= 1e-5` |
| flat-pullback Riemann norm | `1.0843952538107194e-6` | `<= 1e-5` |
| curved-field Riemann norm | `0.9383799130476164` | `> 1e-2` |
| relative-log chart/gauge residual | `3.0359345942773024e-15` | `<= 1e-10` |
| serial versus Rayon difference | `0` | `= 0` |

The compact machine records are
`gate-a-fixtures-r6-release-final6.json` and
`oracle-r6-release-final6.json`. Build trees, executable copies, intermediate
lineages, and release manifests are intentionally not versioned.

Gate A-LOCK remains optional until immediately before a one-shot Gate B run.
Gate B is currently unimplemented and sealed.
