# Acquisition and schema implementation

Status: COMPLETE

## Implemented scope

1. `artifacts/fetch_e2syt_public_manifest.py` fetches only the fixed DANDI
   version and asset metadata, checks the frozen aggregate counts and writes a
   canonical 223-asset receipt.
2. Exactly the manifest-selected minimum segmentation asset was downloaded to
   the ignored local data tree.  A `.download` staging file was promoted only
   after exact byte-count and SHA-256 verification.
3. `artifacts/inspect_e2syt_nwb_schema.py` verifies the input identity, opens it
   read-only and records HDF5 groups, datasets, shapes, dtypes, selected schema
   attributes and reference targets.  It never indexes a dataset value.
4. The generated schema artifact is
   `artifacts/e2syt-exemplar-schema.json`; its interpretation is frozen in
   `artifacts/e2syt-exemplar-schema-audit.md`.

## Environment

Windows application control prevents use of the repository uv environment.
The allowed system CPython 3.11.9 was used with bytecode disabled.  A temporary,
isolated read-only HDF5 dependency was placed outside the repository at
`C:/tmp/clarus-e2syt-h5-reader` (`h5py 3.15.1`, HDF5 1.14.6).  No project
environment, lockfile or package metadata was changed.

## Data hygiene

The NWB payload remains only under gitignored `data/external/randi_e2syt/`.
No raw or binary neural payload was added to the CE run.  The versioned run
contains only compact source code, JSON inventories and Markdown findings.

The implementation did not inspect neural values, compute pair responses,
choose a horizon, fit a model, select an endpoint or evaluate an empirical
effect.
