# Clarus cell JUMP morphology operator gate

- passed: `True`
- claim level: `empirical_morphology_operator_activity_branch`
- source: [JUMP Cell Painting datasets](https://github.com/jump-cellpainting/datasets)
- profile index: [v0.11.0 manifest](https://raw.githubusercontent.com/jump-cellpainting/datasets/v0.11.0/manifests/profile_index.json)
- local profile: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_profiles_pca_corrected.parquet`
- local subset: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_clarus_operator_morphology_subset.csv`
- source note: JUMP CRISPR assembled well-position, cell-count, variance/MAD, outlier, feature-selected, sphered, Harmony, PCA-corrected profiles.
- rows/features: `51185` / `259`
- unique JCP ids: `7977`
- mapped JCP ids: `7977`
- treatment genes: `7974`
- active threshold: negative-control q95 = `0.449`
- active rule: gene median RMS > negative-control q95 or active profile fraction >= 0.5
- morphology control ok: `True`
- passed operators: `5/6`
- operators supported: `B,U,A,I,D,Q`
- failed or weak operators: `E_energy_mitochondria_morphology`

## profile controls

| group | n | median RMS | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|
| `negative_control` | 7478 | 0.304 | 0.340 | 0.394 | 0.449 | 0.914 |
| `treatment` | 43138 | 0.317 | 0.373 | 0.482 | 0.605 | 1.091 |

## operator summaries

| operator | vars | observed | active | median gene RMS | median active frac | passed |
|---|---|---:|---:|---:|---:|---|
| `B_boundary_morphology` | `B` | 9 | 4 | 0.413 | 0.200 | `True` |
| `U_traffic_morphology` | `U` | 7 | 2 | 0.413 | 0.400 | `True` |
| `E_energy_mitochondria_morphology` | `E` | 15 | 0 | 0.328 | 0.000 | `False` |
| `A_metabolic_core_morphology` | `A` | 18 | 3 | 0.317 | 0.000 | `True` |
| `I_identity_template_morphology` | `I` | 21 | 8 | 0.366 | 0.200 | `True` |
| `D_Q_repair_quality_morphology` | `D,Q` | 19 | 6 | 0.384 | 0.200 | `True` |

## strongest morphology-active genes

### `B_boundary_morphology`

| gene | profiles | median RMS | q90 RMS | active fraction | active |
|---|---:|---:|---:|---:|---|
| `TSG101` | 10 | 1.004 | 1.191 | 1.000 | `True` |
| `ATP6V0D1` | 5 | 0.701 | 0.712 | 1.000 | `True` |
| `ATP2A2` | 7 | 0.500 | 0.746 | 0.714 | `True` |
| `ATP6V1B2` | 5 | 0.482 | 0.729 | 0.800 | `True` |
| `RAB7A` | 5 | 0.413 | 0.794 | 0.200 | `False` |
| `ATP6V1E1` | 5 | 0.410 | 0.538 | 0.200 | `False` |
| `ATP1A1` | 5 | 0.403 | 0.443 | 0.000 | `False` |
| `ATP6V1A` | 5 | 0.383 | 0.442 | 0.200 | `False` |
| `ATP6V1H` | 5 | 0.370 | 0.413 | 0.000 | `False` |

### `U_traffic_morphology`

| gene | profiles | median RMS | q90 RMS | active fraction | active |
|---|---:|---:|---:|---:|---|
| `TSG101` | 10 | 1.004 | 1.191 | 1.000 | `True` |
| `VPS18` | 5 | 0.486 | 0.622 | 0.800 | `True` |
| `VPS11` | 5 | 0.424 | 0.533 | 0.400 | `False` |
| `RAB7A` | 5 | 0.413 | 0.794 | 0.200 | `False` |
| `VPS29` | 5 | 0.394 | 0.415 | 0.000 | `False` |
| `RAB5A` | 5 | 0.365 | 0.498 | 0.400 | `False` |
| `RAB11A` | 5 | 0.306 | 0.375 | 0.000 | `False` |

### `E_energy_mitochondria_morphology`

| gene | profiles | median RMS | q90 RMS | active fraction | active |
|---|---:|---:|---:|---:|---|
| `VDAC1` | 5 | 0.395 | 0.476 | 0.200 | `False` |
| `SDHA` | 5 | 0.362 | 0.389 | 0.000 | `False` |
| `UQCRC1` | 7 | 0.353 | 0.415 | 0.000 | `False` |
| `NDUFB9` | 5 | 0.352 | 0.398 | 0.000 | `False` |
| `SDHD` | 6 | 0.338 | 0.373 | 0.000 | `False` |
| `COX5A` | 5 | 0.337 | 1.388 | 0.200 | `False` |
| `NDUFS8` | 5 | 0.329 | 0.414 | 0.000 | `False` |
| `TFAM` | 5 | 0.328 | 0.383 | 0.000 | `False` |
| `NDUFS2` | 5 | 0.328 | 0.372 | 0.000 | `False` |
| `UQCRC2` | 5 | 0.317 | 0.347 | 0.000 | `False` |

### `A_metabolic_core_morphology`

| gene | profiles | median RMS | q90 RMS | active fraction | active |
|---|---:|---:|---:|---:|---|
| `RRM1` | 5 | 1.172 | 1.452 | 1.000 | `True` |
| `RRM2` | 6 | 0.965 | 1.393 | 1.000 | `True` |
| `GAPDH` | 5 | 0.563 | 0.689 | 1.000 | `True` |
| `ENO1` | 5 | 0.400 | 0.693 | 0.200 | `False` |
| `LDHA` | 5 | 0.387 | 0.393 | 0.000 | `False` |
| `TPI1` | 5 | 0.384 | 0.388 | 0.000 | `False` |
| `ACLY` | 5 | 0.370 | 0.420 | 0.000 | `False` |
| `ACACA` | 5 | 0.337 | 0.346 | 0.000 | `False` |
| `MTHFD1` | 5 | 0.317 | 0.371 | 0.000 | `False` |
| `ATIC` | 5 | 0.316 | 0.345 | 0.000 | `False` |

### `I_identity_template_morphology`

| gene | profiles | median RMS | q90 RMS | active fraction | active |
|---|---:|---:|---:|---:|---|
| `POLR2A` | 7 | 1.280 | 1.471 | 1.000 | `True` |
| `PCNA` | 5 | 0.966 | 1.172 | 1.000 | `True` |
| `RPA1` | 6 | 0.960 | 1.416 | 1.000 | `True` |
| `POLR2B` | 5 | 0.719 | 0.822 | 1.000 | `True` |
| `POLR2C` | 5 | 0.658 | 0.865 | 1.000 | `True` |
| `TOP2A` | 7 | 0.488 | 0.590 | 0.714 | `True` |
| `RFC1` | 5 | 0.475 | 0.530 | 0.800 | `True` |
| `MCM4` | 5 | 0.469 | 0.515 | 0.600 | `True` |
| `SUZ12` | 10 | 0.414 | 0.486 | 0.200 | `False` |
| `MCM7` | 5 | 0.395 | 0.439 | 0.000 | `False` |

### `D_Q_repair_quality_morphology`

| gene | profiles | median RMS | q90 RMS | active fraction | active |
|---|---:|---:|---:|---:|---|
| `VCP` | 5 | 2.179 | 2.708 | 1.000 | `True` |
| `PSMC1` | 5 | 0.904 | 1.011 | 1.000 | `True` |
| `RAD51` | 5 | 0.574 | 0.624 | 1.000 | `True` |
| `BECN1` | 5 | 0.517 | 1.031 | 1.000 | `True` |
| `XRCC6` | 5 | 0.468 | 1.157 | 0.600 | `True` |
| `BRCA1` | 10 | 0.465 | 0.539 | 0.700 | `True` |
| `ATG3` | 5 | 0.447 | 0.466 | 0.400 | `False` |
| `BARD1` | 10 | 0.402 | 0.521 | 0.200 | `False` |
| `ATM` | 10 | 0.389 | 0.423 | 0.000 | `False` |
| `CHEK1` | 5 | 0.384 | 0.610 | 0.200 | `False` |

## claim boundary

This gate supports image-based morphology activity of shared human cell operators.  It does not prove channel-specific organelle causality, cell recurrence, primitive-cell origin, or the full human brain mechanism. The weak E result should be read as a limitation of this PCA morphology branch, not as evidence against DepMap/HPA mitochondrial support.
