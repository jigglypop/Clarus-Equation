# Clarus cell JUMP chemical mitochondrial positive-control gate

- passed: `False`
- claim level: `parsed_no_promotion`
- specific claim: `direct_mito_uncoupler_partial_sensitivity`
- source: [JUMP Cell Painting datasets](https://github.com/jump-cellpainting/datasets)
- profile index: [v0.11.0 manifest](https://raw.githubusercontent.com/jump-cellpainting/datasets/v0.11.0/manifests/profile_index.json)
- source parquet: `https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int.parquet`
- compound identity source: [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
- local subset: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_compound_mito_direct_features.parquet`
- compound summary: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_compound_mito_positive_control_summary.csv`
- rows/direct Mito features: `803853` / `125`
- active threshold: negative-control q95 = `1.414`
- active rule: compound median direct-Mito robust-z RMS > negative-control q95 or active profile fraction >= 0.5
- data ok: `True`
- observed compounds: `6/6`
- active compounds: `2`
- uncoupler active: `True`
- partial sensitivity: `True`
- positive control ok: `False`

## profile controls

| group | n | median direct-Mito RMS | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|
| `negative_control` | 93552 | 0.861 | 1.048 | 1.257 | 1.414 | 1.907 |
| `treatment` | 649876 | 1.010 | 1.267 | 1.592 | 1.920 | 2.776 |

## compound summaries

| compound | mode | profiles | median direct-Mito RMS | q90 | active frac | active |
|---|---|---:|---:|---:|---:|---|
| `FCCP` | oxidative-phosphorylation uncoupler | 8 | 1.080 | 2.208 | 0.375 | `False` |
| `CCCP` | oxidative-phosphorylation uncoupler | 10 | 1.471 | 2.461 | 0.500 | `True` |
| `phenformin` | biguanide mitochondrial complex-I stress | 8 | 0.929 | 1.356 | 0.125 | `False` |
| `metformin` | biguanide mitochondrial complex-I stress | 10 | 1.012 | 1.284 | 0.000 | `False` |
| `menadione` | redox/mitochondrial oxidative stress | 150 | 0.905 | 1.236 | 0.047 | `False` |
| `niclosamide` | mitochondrial uncoupling/stress | 10 | 1.752 | 2.669 | 0.500 | `True` |

## compound identifiers

| compound | PubChem CID | InChIKey | JCP ids |
|---|---:|---|---|
| `FCCP` | 3330 | `BMZRVOVNUMQTIN-UHFFFAOYSA-N` | `JCP2022_007370` |
| `CCCP` | 2603 | `UGTJLJZQQFGTJD-UHFFFAOYSA-N` | `JCP2022_089117` |
| `phenformin` | 8249 | `ICFJFFQQTFMIBG-UHFFFAOYSA-N` | `JCP2022_034151` |
| `metformin` | 4091 | `XZWYZXLIPXDOLR-UHFFFAOYSA-N` | `JCP2022_107128` |
| `menadione` | 4055 | `MJVAVZPDRWSRRC-UHFFFAOYSA-N` | `JCP2022_054618` |
| `niclosamide` | 4477 | `RJMUSRYZPJIFPJ-UHFFFAOYSA-N` | `JCP2022_078761` |

## claim boundary

This is an assay positive-control gate. It shows whether direct Mito-channel JUMP compound profiles respond to known mitochondrial perturbagens. It does not by itself validate the genetic Clarus E operator, cell recurrence, or brain mechanism.
