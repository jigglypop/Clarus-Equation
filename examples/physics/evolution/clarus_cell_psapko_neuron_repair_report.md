# Clarus cell PSAP-KO neuron repair gate

- passed: `False`
- claim level: `parsed_no_promotion`
- dataset: [GSE152988](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152988)
- primary paper: [CRISPRbrain human neuron screen](https://www.nature.com/articles/s41593-021-00862-0)
- local data: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\GSE152988_WT_vs_PSAPKO.csv.gz`
- contrast note: File name is WT_vs_PSAPKO. Positive log2FC is treated as WT-enriched/PSAPKO-reduced; negative log2FC is treated as PSAPKO-induced.

## background

- genes after baseMean filter: `9600`
- significant/effect genes: `155`
- background top fraction: `0.016`
- background median |log2FC|: `0.172`

## target control

- PSAP observed: `True`
- PSAP log2FC: `1.643`
- PSAP padj: `3.678e-73`
- direction/control ok: `True`

## decision

- target control ok: `True`
- neural identity context ok: `True`
- repair signal ok: `False`
- stress signal ok: `False`
- D/Q branch signal ok: `False`

## operator summaries

| operator | vars | observed | top | enrichment | median |log2FC| ratio | KO-induced | WT-enriched |
|---|---|---:|---:|---:|---:|---:|---:|
| `Q_repair_lysosome_autophagy` | `Q,D,R` | 33 | 0 | 0.000 | 1.134 | 0 | 0 |
| `D_damage_stress_response` | `D,Q,R` | 20 | 0 | 0.000 | 1.204 | 0 | 0 |
| `E_mito_energy` | `E,A,R` | 169 | 2 | 0.733 | 0.993 | 0 | 2 |
| `U_traffic_boundary` | `B,U,Q` | 25 | 0 | 0.000 | 0.690 | 0 | 0 |
| `I_neural_identity` | `I,R` | 15 | 1 | 4.129 | 0.462 | 0 | 1 |
| `S_glia_support_context` | `S,Q,R` | 1 | 0 | 0.000 | 3.493 | 0 | 0 |

## strongest mapped genes

### `Q_repair_lysosome_autophagy`

| gene | log2FC | padj | baseMean |
|---|---:|---:|---:|
| `CLN3` | -1.097 | 0.118 | 20.160 |
| `NPC1` | -0.955 | 0.257 | 19.938 |
| `HEXB` | -0.846 | 0.407 | 15.193 |
| `GNPTG` | 0.568 | 0.477 | 28.374 |
| `GALC` | -0.312 | 0.569 | 73.470 |
| `SMPD1` | -0.818 | 0.640 | 10.976 |
| `CTSB` | 0.206 | 0.646 | 143.241 |
| `PINK1` | 0.190 | 0.674 | 128.191 |

### `D_damage_stress_response`

| gene | log2FC | padj | baseMean |
|---|---:|---:|---:|
| `HSP90B1` | 0.335 | 0.002 | 533.485 |
| `ATF4` | -0.380 | 0.003 | 386.118 |
| `HSPA5` | 0.241 | 0.052 | 564.318 |
| `TXNIP` | 0.558 | 0.239 | 55.525 |
| `SOD1` | 0.200 | 0.401 | 263.809 |
| `GPX4` | 0.306 | 0.596 | 70.305 |
| `JUN` | -0.184 | 0.800 | 98.683 |
| `GCLM` | -0.337 | 0.803 | 27.749 |

### `E_mito_energy`

| gene | log2FC | padj | baseMean |
|---|---:|---:|---:|
| `VDAC1` | 0.458 | 1.215e-06 | 513.603 |
| `MT-ND1` | -0.264 | 8.076e-06 | 6652.485 |
| `MT-CO1` | 0.190 | 1.899e-05 | 7739.759 |
| `ATP5G1` | 0.538 | 4.669e-04 | 205.328 |
| `SLC25A3` | 0.302 | 4.876e-04 | 808.568 |
| `NDUFAB1` | 0.340 | 0.006 | 430.551 |
| `ATP5A1` | 0.247 | 0.012 | 625.996 |
| `NDUFS2` | 0.490 | 0.014 | 183.120 |

### `U_traffic_boundary`

| gene | log2FC | padj | baseMean |
|---|---:|---:|---:|
| `AP2M1` | 0.301 | 0.221 | 239.109 |
| `RAB3A` | 0.585 | 0.300 | 36.000 |
| `COPA` | 0.171 | 0.613 | 214.318 |
| `RAB7A` | -0.137 | 0.637 | 318.268 |
| `RAB11A` | -0.097 | 0.731 | 546.586 |
| `VPS16` | 0.386 | 0.766 | 23.645 |
| `VPS33A` | -0.185 | 0.766 | 97.873 |
| `TSG101` | -0.153 | 0.795 | 120.626 |

### `I_neural_identity`

| gene | log2FC | padj | baseMean |
|---|---:|---:|---:|
| `SNAP25` | 0.521 | 5.591e-05 | 325.638 |
| `DLG4` | 0.379 | 0.024 | 407.170 |
| `DCX` | -0.113 | 0.052 | 4533.757 |
| `NCAM1` | -0.079 | 0.474 | 3512.595 |
| `NEFL` | 0.060 | 0.602 | 4985.486 |
| `NEFH` | 0.352 | 0.808 | 24.209 |
| `SLC17A7` | -0.252 | 0.861 | 29.728 |
| `RBFOX3` | -0.110 | 0.878 | 127.683 |

### `S_glia_support_context`

| gene | log2FC | padj | baseMean |
|---|---:|---:|---:|
| `CLU` | 0.600 | 0.257 | 39.285 |

## claim boundary

This is a narrow empirical pilot for the postmitotic D/Q repair branch. It does not validate the full Clarus-cell loop, cell origin model, or human-brain mechanism by itself.
