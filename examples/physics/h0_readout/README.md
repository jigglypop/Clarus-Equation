# H0 readout gates

This directory contains the H0 readout-law gates, TDCOSMO covariance adapters,
and source/provenance checks.

Main entry point:

```bash
python examples/physics/h0_readout/h0_fisher_io_full_suite.py
```

Key subdirectories:

- `h0_fisher_io_examples/`: Fisher/covariance JSON examples and TDCOSMO-derived covariance payloads.
- `h0_real_data/`: downloaded public TDCOSMO HDF5 chains and notebook cache. This
  directory is ignored by git; regenerate it with the downloader/provenance gates.

The current provenance chain is:

```text
AST(MCMCSampler first argument)
-> generated likelihood factor graph G_L
-> role map R_L(G_L)
-> q_F(F, R_L)
-> H0(q_F)
```
