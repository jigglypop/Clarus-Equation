# Clarus cell Perturb-seq state reconstruction gate

- passed: `True`
- claim level: `empirical_perturbseq_operator_state_branch`
- primary paper: [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013)
- source: [Figshare processed datasets](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387)
- summary csv: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\replogle_perturbseq_clarus_state_summary.csv`
- source note: Processed normalized pseudo-bulk AnnData files. Rows are perturbation pseudobulk populations; columns are normalized transcript features.
- operators supported: `E,A,I,D,Q,R`
- broad state operators: `I,D,Q`
- single-dataset broad state operators: `A,R`
- module-local-only operators: `E`
- passed operators: `5/5`
- broad core ok: `True`

## datasets

| dataset | rows | genes | controls | control global q95 | control AD q95 |
|---|---:|---:|---:|---:|---:|
| `K562_essential` | 2285 | 8563 | 97 | 0.142 | 3.000 |
| `RPE1_essential` | 2679 | 8749 | 113 | 0.154 | 10.800 |

## aggregate operators

| operator | vars | observed rows | module replicated | broad replicated | passed | note |
|---|---|---:|---|---|---|---|
| `E_energy_mitochondria_state` | `E` | 21 | `True` | `False` | `True` | module-local replicated; broad state weak |
| `A_metabolic_core_state` | `A` | 25 | `True` | `False` | `True` | module-local replicated; broad state in one dataset |
| `I_identity_template_state` | `I` | 43 | `True` | `True` | `True` | module-local and broad state replicated |
| `D_Q_repair_quality_state` | `D,Q` | 24 | `True` | `True` | `True` | module-local and broad state replicated |
| `R_recurrence_cell_cycle_state` | `R` | 34 | `True` | `False` | `True` | module-local replicated; broad state in one dataset |

## dataset operator summaries

### `K562_essential`

| operator | observed | program genes | broad active | module active | median global RMS | median module RMS | broad pass | module pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `E_energy_mitochondria_state` | 9 | 24 | 2 (0.222) | 8 (0.889) | 0.130 | 0.295 | `False` | `True` |
| `A_metabolic_core_state` | 11 | 18 | 4 (0.364) | 8 (0.727) | 0.139 | 0.415 | `True` | `True` |
| `I_identity_template_state` | 21 | 26 | 10 (0.476) | 18 (0.857) | 0.149 | 0.225 | `True` | `True` |
| `D_Q_repair_quality_state` | 12 | 25 | 5 (0.417) | 9 (0.750) | 0.117 | 0.213 | `True` | `True` |
| `R_recurrence_cell_cycle_state` | 17 | 22 | 5 (0.294) | 15 (0.882) | 0.126 | 0.216 | `False` | `True` |

### `RPE1_essential`

| operator | observed | program genes | broad active | module active | median global RMS | median module RMS | broad pass | module pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `E_energy_mitochondria_state` | 12 | 24 | 2 (0.167) | 10 (0.833) | 0.117 | 0.281 | `False` | `True` |
| `A_metabolic_core_state` | 14 | 18 | 4 (0.286) | 12 (0.857) | 0.137 | 0.469 | `False` | `True` |
| `I_identity_template_state` | 22 | 26 | 20 (0.909) | 22 (1.000) | 0.315 | 0.444 | `True` | `True` |
| `D_Q_repair_quality_state` | 12 | 26 | 12 (1.000) | 12 (1.000) | 0.333 | 0.472 | `True` | `True` |
| `R_recurrence_cell_cycle_state` | 17 | 22 | 13 (0.765) | 17 (1.000) | 0.258 | 0.474 | `True` | `True` |

## strongest examples

### `K562_essential`

#### `E_energy_mitochondria_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `SDHD` | 0.161 | 0.198 | 3.000 | 0.471 | 45.000 |
| `TFAM` | 0.154 | 0.313 | 35.000 | 0.741 | 82.000 |
| `UQCRC2` | 0.148 | 0.347 | 57.000 | 0.231 | 68.000 |
| `SDHA` | 0.139 | 0.229 | 8.000 | 0.047 | 62.000 |
| `COX5A` | 0.130 | 0.483 | 39.000 | 0.365 | 85.000 |

#### `A_metabolic_core_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `RRM1` | 0.371 | 0.383 | 915.000 | 1.344 | 103.000 |
| `PKM` | 0.212 | 0.535 | 62.000 | 0.806 | 34.000 |
| `PGK1` | 0.190 | 0.458 | 1.000 | 0.362 | 33.000 |
| `TPI1` | 0.182 | 0.415 | 9.000 | 0.968 | 40.000 |
| `IMPDH2` | 0.170 | 0.641 | 278.000 | 0.509 | 77.000 |

#### `I_identity_template_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `POLR2C` | 0.285 | 0.362 | 5202.000 | 1.729 | 185.000 |
| `PCNA` | 0.284 | 0.271 | 1805.000 | 1.416 | 132.000 |
| `POLR2B` | 0.237 | 0.325 | 5682.000 | 1.506 | 320.000 |
| `RPA2` | 0.207 | 0.234 | 2265.000 | 1.345 | 225.000 |
| `TOP2A` | 0.176 | 0.225 | 1.000 | 0.622 | 38.000 |

#### `D_Q_repair_quality_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `PSMD2` | 0.427 | 1.461 | 3865.000 | 2.283 | 63.000 |
| `PSMC1` | 0.289 | 0.940 | 5206.000 | 1.724 | 316.000 |
| `XRCC6` | 0.254 | 0.432 | 180.000 | 0.838 | 28.000 |
| `XRCC5` | 0.180 | 0.407 | 494.000 | 0.921 | 88.000 |
| `CHEK1` | 0.145 | 0.209 | 2390.000 | 0.949 | 295.000 |

#### `R_recurrence_cell_cycle_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `PCNA` | 0.284 | 0.266 | 1805.000 | 1.416 | 132.000 |
| `CCNA2` | 0.206 | 0.259 | 6.000 | 0.085 | 24.000 |
| `AURKB` | 0.187 | 0.208 | 4.000 | 0.995 | 33.000 |
| `AURKA` | 0.158 | 0.225 | 41.000 | 1.076 | 58.000 |
| `MCM4` | 0.154 | 0.275 | 147.000 | 0.568 | 87.000 |

### `RPE1_essential`

#### `E_energy_mitochondria_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `ATP5F1C` | 0.369 | 0.523 | 1712.000 | 2.738 | 44.000 |
| `SDHC` | 0.186 | 0.273 | 161.000 | 0.799 | 49.000 |
| `SDHD` | 0.146 | 0.159 | 1.000 | 0.267 | 51.000 |
| `UQCRC2` | 0.126 | 0.228 | 6.000 | 0.519 | 68.000 |
| `UQCRC1` | 0.126 | 0.288 | 12.000 | 0.366 | 82.000 |

#### `A_metabolic_core_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `RRM1` | 0.226 | 0.399 | 747.000 | 1.533 | 67.000 |
| `RRM2` | 0.209 | 0.468 | 2522.000 | 1.414 | 229.000 |
| `ATIC` | 0.200 | 0.469 | 1055.000 | 1.130 | 86.000 |
| `FASN` | 0.196 | 0.527 | 540.000 | 1.255 | 73.000 |
| `GAPDH` | 0.176 | 0.168 | 0.000 | 0.228 | 35.000 |

#### `I_identity_template_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `TOP2A` | 0.459 | 0.426 | 2.000 | 2.161 | 6.000 |
| `RPA2` | 0.387 | 0.494 | 2279.000 | 2.584 | 75.000 |
| `POLR2B` | 0.387 | 0.573 | 1029.000 | 3.041 | 33.000 |
| `TOP1` | 0.370 | 0.451 | 1390.000 | 1.852 | 48.000 |
| `PCNA` | 0.367 | 0.514 | 1759.000 | 1.994 | 56.000 |

#### `D_Q_repair_quality_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `PSMC1` | 0.921 | 4.065 | 3271.000 | 5.413 | 82.000 |
| `PSMD2` | 0.881 | 4.458 | 2055.000 | 5.750 | 24.000 |
| `PSMD1` | 0.711 | 3.835 | 1819.000 | 4.892 | 32.000 |
| `VCP` | 0.417 | 0.900 | 458.000 | 2.149 | 30.000 |
| `BRCA1` | 0.354 | 0.415 | 131.000 | 1.789 | 14.000 |

#### `R_recurrence_cell_cycle_state`

| gene | global RMS | module RMS | AD counts | leverage | cells |
|---|---:|---:|---:|---:|---:|
| `AURKB` | 0.539 | 0.663 | 39.000 | 2.165 | 5.000 |
| `CCNA2` | 0.431 | 0.274 | 4.000 | 1.589 | 13.000 |
| `PCNA` | 0.367 | 0.590 | 1759.000 | 1.994 | 56.000 |
| `AURKA` | 0.314 | 0.537 | 744.000 | 1.628 | 35.000 |
| `MCM2` | 0.313 | 0.526 | 1004.000 | 1.659 | 42.000 |

## claim boundary

This is a proliferative K562/RPE1 pseudo-bulk transcriptomic state gate. It supports operator-level state reconstruction, especially I/D/Q/R broad state shifts. E is promoted only as a replicated module-local transcriptomic response here, not as a broad transcriptome or direct mitochondrial morphology proof.
