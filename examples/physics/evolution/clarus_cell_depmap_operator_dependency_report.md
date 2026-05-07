# Clarus cell DepMap operator dependency gate

- passed: `True`
- claim level: `empirical_proliferative_recurrence_branch`
- release: [DepMap 24Q4 Public](https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248)
- local subset: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\depmap_24q4_clarus_operator_dependency_subset.csv`
- models: `1178`
- genes in subset: `2038`
- dependent threshold: `-0.5`
- control ok: `True`
- passed operators: `6/7`
- core ok: `True`

## controls

| control | genes | median gene median effect | median dependent fraction |
|---|---:|---:|---:|
| `common_essential` | 1242 | -0.999 | 0.970 |
| `nonessential` | 726 | -0.001 | 0.001 |

## operator summaries

| operator | vars | genes | median effect | dependent frac | passed |
|---|---|---:|---:|---:|---|
| `B_boundary_membrane` | `B,U,R` | 16 | -1.372 | 0.985 | `True` |
| `U_regulated_ports_traffic` | `U,B,Q,R` | 19 | -0.581 | 0.649 | `True` |
| `E_energy_mitochondria` | `E,A,R` | 24 | -0.478 | 0.468 | `True` |
| `A_metabolic_autocatalytic_core` | `A,E,R` | 18 | -0.373 | 0.357 | `True` |
| `I_identity_template` | `I,R` | 24 | -1.339 | 0.994 | `True` |
| `D_Q_repair_quality_control` | `D,Q,R` | 25 | -0.262 | 0.118 | `False` |
| `R_proliferative_recurrence` | `R,I,A` | 22 | -1.225 | 0.992 | `True` |

## strongest dependencies by operator

### `B_boundary_membrane`

| gene | median effect | dependent fraction | strong fraction |
|---|---:|---:|---:|
| `ATP6V0C` | -2.227 | 0.996 | 0.984 |
| `COPB1` | -2.196 | 1.000 | 1.000 |
| `COPA` | -2.051 | 1.000 | 0.998 |
| `ATP6V1B2` | -1.893 | 0.970 | 0.922 |
| `ATP6V1A` | -1.887 | 0.998 | 0.986 |
| `ATP2A2` | -1.709 | 0.997 | 0.976 |
| `CLTC` | -1.425 | 0.992 | 0.870 |
| `ATP6V1E1` | -1.407 | 0.993 | 0.941 |
| `ATP6V0D1` | -1.336 | 0.985 | 0.940 |
| `DNM2` | -1.194 | 0.898 | 0.664 |

### `U_regulated_ports_traffic`

| gene | median effect | dependent fraction | strong fraction |
|---|---:|---:|---:|
| `SRP54` | -2.245 | 1.000 | 0.999 |
| `COPB1` | -2.196 | 1.000 | 1.000 |
| `COPA` | -2.051 | 1.000 | 0.998 |
| `SEC61A1` | -1.863 | 0.998 | 0.996 |
| `CLTC` | -1.425 | 0.992 | 0.870 |
| `TSG101` | -1.133 | 0.985 | 0.696 |
| `CHMP4B` | -1.034 | 0.786 | 0.520 |
| `VPS18` | -0.885 | 0.930 | 0.319 |
| `AP2M1` | -0.666 | 0.680 | 0.170 |
| `VPS35` | -0.581 | 0.649 | 0.057 |

### `E_energy_mitochondria`

| gene | median effect | dependent fraction | strong fraction |
|---|---:|---:|---:|
| `ATP5F1B` | -0.928 | 0.922 | 0.398 |
| `ATP5F1E` | -0.890 | 0.921 | 0.351 |
| `SDHC` | -0.811 | 0.842 | 0.288 |
| `VDAC1` | -0.749 | 0.908 | 0.115 |
| `ATP5F1D` | -0.682 | 0.740 | 0.141 |
| `ATP5F1A` | -0.653 | 0.737 | 0.092 |
| `TFAM` | -0.572 | 0.614 | 0.063 |
| `ATP5F1C` | -0.550 | 0.581 | 0.070 |
| `NDUFB9` | -0.531 | 0.553 | 0.063 |
| `NDUFB4` | -0.503 | 0.504 | 0.066 |

### `A_metabolic_autocatalytic_core`

| gene | median effect | dependent fraction | strong fraction |
|---|---:|---:|---:|
| `RRM1` | -2.920 | 1.000 | 0.998 |
| `RRM2` | -2.392 | 0.998 | 0.990 |
| `GAPDH` | -1.265 | 0.987 | 0.800 |
| `TPI1` | -0.802 | 0.782 | 0.289 |
| `DHFR` | -0.795 | 0.748 | 0.372 |
| `PKM` | -0.769 | 0.800 | 0.196 |
| `PGK1` | -0.766 | 0.727 | 0.266 |
| `IMPDH2` | -0.542 | 0.551 | 0.121 |
| `LDHA` | -0.382 | 0.198 | 0.008 |
| `ACLY` | -0.364 | 0.352 | 0.041 |

### `I_identity_template`

| gene | median effect | dependent fraction | strong fraction |
|---|---:|---:|---:|
| `PCNA` | -2.750 | 0.999 | 0.998 |
| `POLR2B` | -2.411 | 1.000 | 0.999 |
| `RPA1` | -2.340 | 1.000 | 0.999 |
| `TOP2A` | -2.274 | 0.997 | 0.986 |
| `RPA3` | -2.142 | 1.000 | 0.998 |
| `POLR2C` | -2.085 | 1.000 | 1.000 |
| `MCM7` | -1.898 | 1.000 | 0.997 |
| `RPA2` | -1.801 | 1.000 | 0.997 |
| `MCM2` | -1.561 | 0.998 | 0.977 |
| `RFC2` | -1.499 | 0.990 | 0.879 |

### `D_Q_repair_quality_control`

| gene | median effect | dependent fraction | strong fraction |
|---|---:|---:|---:|
| `VCP` | -2.533 | 1.000 | 0.999 |
| `CHEK1` | -1.867 | 1.000 | 0.993 |
| `XRCC6` | -1.749 | 0.999 | 0.986 |
| `RAD51` | -1.442 | 0.994 | 0.951 |
| `PSMC1` | -1.367 | 1.000 | 0.919 |
| `PSMD2` | -1.348 | 0.999 | 0.937 |
| `ATR` | -1.345 | 0.992 | 0.932 |
| `PSMD1` | -1.258 | 0.999 | 0.860 |
| `XRCC5` | -1.223 | 0.997 | 0.871 |
| `BARD1` | -0.691 | 0.797 | 0.110 |

### `R_proliferative_recurrence`

| gene | median effect | dependent fraction | strong fraction |
|---|---:|---:|---:|
| `PCNA` | -2.750 | 0.999 | 0.998 |
| `PLK1` | -2.686 | 0.999 | 0.999 |
| `CDK1` | -2.388 | 1.000 | 0.995 |
| `AURKB` | -2.298 | 1.000 | 0.997 |
| `CDC20` | -2.002 | 1.000 | 1.000 |
| `MCM7` | -1.898 | 1.000 | 0.997 |
| `MCM2` | -1.561 | 0.998 | 0.977 |
| `MCM5` | -1.399 | 0.995 | 0.910 |
| `CCNA2` | -1.292 | 0.992 | 0.849 |
| `MCM4` | -1.280 | 0.998 | 0.880 |

## claim boundary

This is a cancer-cell-line proliferative dependency gate. It supports Clarus recurrence as survival/proliferation for shared human cell operators, but it is not a normal tissue, developmental, or brain-wide proof.
