# Clarus cell CRISPRbrain neuron maintenance gate

- passed: `True`
- claim level: `empirical_DQ_neuron_maintenance_branch`
- primary paper: [Tian et al. 2021](https://www.nature.com/articles/s41593-021-00862-0)
- supplement table: [Supplementary Table 2](https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-021-00862-0/MediaObjects/41593_2021_862_MOESM4_ESM.csv)
- local data: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\41593_2021_862_MOESM4_hit_class.csv`
- source note: Supplementary Table 2 reports hit class values for Fig. 2g. Values 1, -1, and 0 denote positive phenotype-score hits, negative phenotype-score hits, and non-hits.

## readouts

- `CellRox_CRISPRi`: D: ROS load
- `Liperfluo_CRISPRi`: D: lipid peroxidation
- `Lysotracker_CRISPRi`: Q/U: lysosome state
- `FeRhoNox-1_CRISPRi`: D/E: labile iron

## background

- genes in hit-class table: `118`
- multi-channel genes: `79`
- multi-channel fraction: `0.669492`
- all-four-channel genes: `36`

## PSAP control

- observed: `True`
- vector CellROX,Liperfluo,LysoTracker,FeRhoNox: `1,1,1,1`
- all four active: `True`
- same-sign active: `True`
- passed: `True`

## decision

- PSAP control ok: `True`
- Q core ok: `True`
- Q readout coverage ok: `True`
- D coupling ok: `True`
- E/U support ok: `True`

## operator summaries

| operator | vars | observed | multi | multi frac | all four | same-sign multi | mean channels |
|---|---|---:|---:|---:|---:|---:|---:|
| `Q_lysosome_autophagy_repair` | `Q,U,D,R` | 24 | 21 | 0.875 | 7 | 21 | 3.166667 |
| `D_redox_iron_lipid_damage` | `D,Q,E,R` | 13 | 11 | 0.846154 | 6 | 10 | 3.307692 |
| `E_mito_energy` | `E,A,D,R` | 11 | 9 | 0.818182 | 5 | 6 | 3.272727 |
| `U_boundary_traffic` | `B,U,Q,R` | 12 | 9 | 0.75 | 3 | 8 | 2.916667 |
| `A_metabolic_core` | `A,E,R` | 10 | 7 | 0.7 | 3 | 3 | 3.0 |

## readout coverage by operator

### `Q_lysosome_autophagy_repair`

- `CellRox_CRISPRi`: `12` genes
- `Liperfluo_CRISPRi`: `20` genes
- `Lysotracker_CRISPRi`: `22` genes
- `FeRhoNox-1_CRISPRi`: `22` genes

### `D_redox_iron_lipid_damage`

- `CellRox_CRISPRi`: `12` genes
- `Liperfluo_CRISPRi`: `11` genes
- `Lysotracker_CRISPRi`: `8` genes
- `FeRhoNox-1_CRISPRi`: `12` genes

### `E_mito_energy`

- `CellRox_CRISPRi`: `10` genes
- `Liperfluo_CRISPRi`: `10` genes
- `Lysotracker_CRISPRi`: `7` genes
- `FeRhoNox-1_CRISPRi`: `9` genes

### `U_boundary_traffic`

- `CellRox_CRISPRi`: `6` genes
- `Liperfluo_CRISPRi`: `9` genes
- `Lysotracker_CRISPRi`: `10` genes
- `FeRhoNox-1_CRISPRi`: `10` genes

### `A_metabolic_core`

- `CellRox_CRISPRi`: `6` genes
- `Liperfluo_CRISPRi`: `8` genes
- `Lysotracker_CRISPRi`: `9` genes
- `FeRhoNox-1_CRISPRi`: `7` genes

## strongest coupled genes

### `Q_lysosome_autophagy_repair`

| gene | active channels | vector |
|---|---:|---|
| `AP3S1` | 4 | `1,1,1,1` |
| `ATG13` | 4 | `1,1,1,1` |
| `ATG14` | 4 | `1,1,1,1` |
| `ATG9A` | 4 | `1,1,1,1` |
| `BECN1` | 4 | `1,1,1,1` |
| `PSAP` | 4 | `1,1,1,1` |
| `WIPI2` | 4 | `1,1,1,1` |
| `ATG3` | 3 | `0,1,1,1` |
| `CTSD` | 3 | `1,0,1,1` |
| `GM2A` | 3 | `0,1,1,1` |

### `D_redox_iron_lipid_damage`

| gene | active channels | vector |
|---|---:|---|
| `AKR7A2` | 4 | `1,1,1,1` |
| `CYB561D2` | 4 | `1,1,1,1` |
| `NDUFB9` | 4 | `1,1,-1,1` |
| `PSAP` | 4 | `1,1,1,1` |
| `SDHC` | 4 | `1,1,1,1` |
| `SYVN1` | 4 | `1,1,1,1` |
| `ATF4` | 3 | `0,-1,-1,-1` |
| `NDUFA9` | 3 | `1,1,0,1` |
| `NDUFB4` | 3 | `1,1,0,1` |
| `NDUFS8` | 3 | `1,1,0,1` |

### `E_mito_energy`

| gene | active channels | vector |
|---|---:|---|
| `ADSL` | 4 | `-1,-1,-1,-1` |
| `COASY` | 4 | `-1,-1,1,1` |
| `NDUFB9` | 4 | `1,1,-1,1` |
| `PPCS` | 4 | `-1,-1,1,1` |
| `SDHC` | 4 | `1,1,1,1` |
| `AHCY` | 3 | `0,-1,-1,-1` |
| `NDUFA9` | 3 | `1,1,0,1` |
| `NDUFB4` | 3 | `1,1,0,1` |
| `NDUFS8` | 3 | `1,1,0,1` |
| `FH` | 2 | `1,1,0,0` |

### `U_boundary_traffic`

| gene | active channels | vector |
|---|---:|---|
| `AP3S1` | 4 | `1,1,1,1` |
| `CTAGE5` | 4 | `1,1,1,1` |
| `DYNC1I2` | 4 | `1,-1,-1,-1` |
| `MON2` | 3 | `0,1,1,1` |
| `PQLC2` | 3 | `1,0,1,1` |
| `TMED10` | 3 | `0,1,1,1` |
| `TMED2` | 3 | `0,1,1,1` |
| `VPS39` | 3 | `0,1,1,1` |
| `VPS41` | 3 | `0,1,1,1` |
| `TFG` | 2 | `0,-1,0,-1` |

### `A_metabolic_core`

| gene | active channels | vector |
|---|---:|---|
| `ADSL` | 4 | `-1,-1,-1,-1` |
| `COASY` | 4 | `-1,-1,1,1` |
| `PPCS` | 4 | `-1,-1,1,1` |
| `AGPAT6` | 3 | `0,-1,1,1` |
| `AHCY` | 3 | `0,-1,-1,-1` |
| `FDPS` | 3 | `0,-1,1,-1` |
| `GFPT1` | 3 | `0,-1,-1,-1` |
| `EBP` | 2 | `1,0,-1,0` |
| `FH` | 2 | `1,1,0,0` |
| `TK2` | 2 | `1,0,-1,0` |

## claim boundary

This promotes only the postmitotic neuron D/Q maintenance branch: lysosome/autophagy/repair genes are coupled to ROS, lipid peroxidation, lysosome state, and labile iron readouts.  It does not prove the whole Clarus-cell mechanism.
