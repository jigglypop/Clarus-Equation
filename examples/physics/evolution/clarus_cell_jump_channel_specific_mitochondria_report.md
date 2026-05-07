# Clarus cell JUMP direct mitochondrial-channel gate

- passed: `False`
- claim level: `parsed_no_promotion`
- specific claim: `mitochondrial_E_not_promoted`
- source: [JUMP Cell Painting datasets](https://github.com/jump-cellpainting/datasets)
- profile index: [v0.11.0 manifest](https://raw.githubusercontent.com/jump-cellpainting/datasets/v0.11.0/manifests/profile_index.json)
- source parquet: `https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/CRISPR/v1.0a/profiles_wellpos_cc_var_mad_outlier.parquet`
- local subset: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_mito_direct_features.parquet`
- gene summary: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_mito_direct_gene_summary.csv`
- rows/direct Mito features: `51185` / `129`
- direct Mito modules: `Intensity,Granularity,RadialDistribution`
- unique JCP ids: `7977`
- treatment genes: `7974`
- active threshold: negative-control q95 = `1.664`
- active rule: gene median direct-Mito robust-z RMS > negative-control q95 or active profile fraction >= 0.5
- data ok: `True`
- positive controls ok: `True`
- E promoted: `False`

## profile controls

| group | n | median direct-Mito RMS | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|
| `negative_control` | 7478 | 0.901 | 1.197 | 1.488 | 1.664 | 2.745 |
| `treatment` | 43138 | 0.855 | 1.074 | 1.353 | 1.600 | 2.733 |

## operator summaries

| operator | vars | observed | active | median gene direct-Mito RMS | median active frac | passed |
|---|---|---:|---:|---:|---:|---|
| `E_energy_mitochondria_direct_mito` | `E` | 15 | 0 | 0.918 | 0.000 | `False` |
| `I_identity_template_direct_mito_control` | `I` | 14 | 5 | 1.046 | 0.000 | `True` |
| `D_Q_repair_quality_direct_mito_control` | `D,Q` | 12 | 3 | 1.041 | 0.150 | `True` |

## strongest direct-Mito genes

### `E_energy_mitochondria_direct_mito`

| gene | profiles | median direct-Mito RMS | q90 | active fraction | active |
|---|---:|---:|---:|---:|---|
| `SDHB` | 5 | 1.227 | 1.358 | 0.000 | `False` |
| `UQCRC1` | 7 | 1.160 | 1.424 | 0.000 | `False` |
| `NDUFS8` | 5 | 1.150 | 1.313 | 0.000 | `False` |
| `COX6B1` | 6 | 0.963 | 1.100 | 0.000 | `False` |
| `NDUFS2` | 5 | 0.919 | 1.035 | 0.000 | `False` |
| `COX5A` | 5 | 0.919 | 1.349 | 0.000 | `False` |
| `VDAC1` | 5 | 0.919 | 1.203 | 0.000 | `False` |
| `NDUFB9` | 5 | 0.918 | 1.077 | 0.000 | `False` |
| `SDHA` | 5 | 0.888 | 1.020 | 0.000 | `False` |
| `SDHD` | 6 | 0.780 | 0.976 | 0.000 | `False` |

### `I_identity_template_direct_mito_control`

| gene | profiles | median direct-Mito RMS | q90 | active fraction | active |
|---|---:|---:|---:|---:|---|
| `POLR2A` | 7 | 5.265 | 5.838 | 1.000 | `True` |
| `RPA1` | 6 | 2.763 | 3.394 | 1.000 | `True` |
| `PCNA` | 5 | 2.282 | 2.342 | 1.000 | `True` |
| `POLR2C` | 5 | 2.189 | 2.354 | 0.800 | `True` |
| `POLR2B` | 5 | 2.123 | 2.288 | 1.000 | `True` |
| `RFC1` | 5 | 1.482 | 1.723 | 0.200 | `False` |
| `MCM7` | 5 | 1.119 | 1.221 | 0.000 | `False` |
| `MCM4` | 5 | 0.973 | 1.198 | 0.000 | `False` |
| `MCM6` | 7 | 0.882 | 1.255 | 0.000 | `False` |
| `MCM5` | 5 | 0.870 | 1.089 | 0.000 | `False` |

### `D_Q_repair_quality_direct_mito_control`

| gene | profiles | median direct-Mito RMS | q90 | active fraction | active |
|---|---:|---:|---:|---:|---|
| `VCP` | 5 | 6.568 | 7.142 | 1.000 | `True` |
| `PSMC1` | 5 | 2.533 | 2.559 | 1.000 | `True` |
| `RAD51` | 5 | 1.838 | 2.042 | 0.800 | `True` |
| `BECN1` | 5 | 1.608 | 1.784 | 0.200 | `False` |
| `ATG3` | 5 | 1.333 | 1.913 | 0.200 | `False` |
| `BRCA1` | 10 | 1.113 | 1.353 | 0.000 | `False` |
| `BARD1` | 10 | 0.968 | 1.282 | 0.100 | `False` |
| `XRCC6` | 5 | 0.965 | 1.331 | 0.000 | `False` |
| `CHEK1` | 5 | 0.872 | 1.770 | 0.200 | `False` |
| `ATM` | 10 | 0.783 | 1.401 | 0.100 | `False` |

## claim boundary

This gate tests only direct Mito-channel interpretable CellProfiler features under CRISPR perturbation. A non-promotion does not override DepMap fitness, HPA localization, or CRISPRbrain stress evidence for E; it marks direct JUMP Mito-channel morphology as an unresolved E branch.
