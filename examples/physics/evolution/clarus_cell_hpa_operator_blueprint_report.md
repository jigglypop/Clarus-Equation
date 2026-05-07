# Clarus cell HPA operator blueprint gate

- passed: `True`
- claim level: `empirical_subcellular_operator_blueprint`
- source: [Human Protein Atlas subcellular data](https://www.proteinatlas.org/humanproteome/subcellular/data)
- local data: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\hpa_subcellular_location.tsv.zip`
- HPA genes loaded: `12805`
- passed blueprints: `7/7`
- operators supported: `B,U,E,A,I,Q,D,S,R`
- distinct expected location classes: `12`

## operator summaries

| operator | vars | observed | matched | fraction | passed |
|---|---|---:|---:|---:|---|
| `B_boundary_membrane` | `B,U,R` | 12 | 10 | 0.833 | `True` |
| `U_regulated_ports_traffic` | `U,B,Q,R` | 17 | 14 | 0.824 | `True` |
| `E_energy_mitochondria` | `E,A,R` | 15 | 15 | 1.000 | `True` |
| `A_metabolic_autocatalytic_core` | `A,E,R` | 18 | 17 | 0.944 | `True` |
| `I_identity_template` | `I,R` | 25 | 25 | 1.000 | `True` |
| `D_Q_repair_quality_control` | `D,Q,R` | 25 | 25 | 1.000 | `True` |
| `S_support_context` | `S,D,Q,U,R` | 13 | 12 | 0.923 | `True` |

## location coverage

### `B_boundary_membrane`

| expected location | genes |
|---|---:|
| `Plasma membrane` | 3 |
| `Cell Junctions` | 0 |
| `Vesicles` | 6 |
| `Endosomes` | 1 |
| `Lysosomes` | 2 |

- unmatched observed genes: `DNM2,COPA`

### `U_regulated_ports_traffic`

| expected location | genes |
|---|---:|
| `Vesicles` | 6 |
| `Endosomes` | 3 |
| `Golgi apparatus` | 3 |
| `Endoplasmic reticulum` | 2 |
| `Plasma membrane` | 2 |

- unmatched observed genes: `SRP54,RAB7A,VPS11`

### `E_energy_mitochondria`

| expected location | genes |
|---|---:|
| `Mitochondria` | 15 |

### `A_metabolic_autocatalytic_core`

| expected location | genes |
|---|---:|
| `Cytosol` | 14 |
| `Mitochondria` | 2 |
| `Nucleoplasm` | 3 |

- unmatched observed genes: `PGK1`

### `I_identity_template`

| expected location | genes |
|---|---:|
| `Nucleoplasm` | 25 |
| `Nucleoli` | 3 |
| `Nuclear bodies` | 2 |
| `Nuclear speckles` | 0 |
| `Mitotic chromosome` | 0 |

### `D_Q_repair_quality_control`

| expected location | genes |
|---|---:|
| `Lysosomes` | 0 |
| `Vesicles` | 7 |
| `Aggresome` | 0 |
| `Cytosol` | 11 |
| `Nucleoplasm` | 18 |
| `Nuclear bodies` | 4 |
| `Endosomes` | 0 |

### `S_support_context`

| expected location | genes |
|---|---:|
| `Predicted to be secreted` | 8 |
| `Secreted` | 0 |
| `Plasma membrane` | 5 |
| `Vesicles` | 6 |
| `Endosomes` | 0 |
| `Lysosomes` | 0 |
| `Extracellular` | 0 |

- unmatched observed genes: `SOD2`

## examples

### `B_boundary_membrane`

- `CLTC` (Enhanced): `Centriolar satellite;Cytosol;Endosomes;Lysosomes;Mid piece;Mitotic spindle`
- `COPB1` (Enhanced): `Cytosol;Golgi apparatus;Vesicles`
- `AP2M1` (Supported): `Plasma membrane`
- `AP3S1` (Approved): `Vesicles`
- `ATP1A1` (Approved): `Plasma membrane;Vesicles`
- `ATP6V0C` (Approved): `Vesicles`
- `ATP6V1A` (Supported): `Cytosol;Nucleoplasm;Vesicles`
- `ATP6V1B2` (Supported): `Vesicles`

### `U_regulated_ports_traffic`

- `AP1G1` (Enhanced): `Cytosol;Golgi apparatus;Vesicles`
- `CLTC` (Enhanced): `Centriolar satellite;Cytosol;Endosomes;Lysosomes;Mid piece;Mitotic spindle`
- `COPB1` (Enhanced): `Cytosol;Golgi apparatus;Vesicles`
- `SNX1` (Enhanced): `Endosomes;Lysosomes`
- `AP2M1` (Supported): `Plasma membrane`
- `AP3S1` (Approved): `Vesicles`
- `COPA` (Approved): `Cytosol;Golgi apparatus;Nucleoplasm;Predicted to be secreted`
- `RAB11A` (Supported): `Basal body;Centriolar satellite;Cytosol;Equatorial segment;Mid piece;Primary cilium;Principal piece;Vesicles`

### `E_energy_mitochondria`

- `ATP5F1B` (Enhanced): `Mitochondria`
- `TFAM` (Enhanced): `Mitochondria`
- `UQCRC2` (Enhanced): `Mitochondria`
- `ATP5F1A` (Supported): `End piece;Mitochondria`
- `ATP5F1D` (Supported): `Connecting piece;Flagellar centriole;Mitochondria;Principal piece`
- `COX4I1` (Supported): `Mitochondria`
- `COX6B1` (Supported): `Mitochondria;Principal piece`
- `NDUFA9` (Approved): `Mitochondria;Nucleoplasm`

### `A_metabolic_autocatalytic_core`

- `ENO1` (Enhanced): `Cytosol;Plasma membrane`
- `FASN` (Enhanced): `Cytosol;Plasma membrane`
- `GAPDH` (Enhanced): `Cytosol;Nuclear membrane;Plasma membrane;Vesicles`
- `LDHA` (Enhanced): `Cytosol`
- `MTHFD1` (Enhanced): `Cytosol`
- `PKM` (Enhanced): `Cytosol;Principal piece;Vesicles`
- `ACACA` (Supported): `Actin filaments;Cytosol;Nucleoli fibrillar center`
- `ACLY` (Supported): `Basal body;Cytosol;Nucleoplasm;Primary cilium transition zone;Principal piece`

### `I_identity_template`

- `EZH2` (Enhanced): `Nucleoplasm`
- `HDAC1` (Enhanced): `Nucleoplasm`
- `MCM4` (Enhanced): `Nucleoplasm`
- `MCM5` (Enhanced): `Nucleoplasm`
- `PCNA` (Enhanced): `Nucleoplasm`
- `POLR2A` (Enhanced): `Cytosol;Nucleoplasm`
- `RFC1` (Enhanced): `Nucleoplasm`
- `RPA1` (Enhanced): `Nucleoplasm`

### `D_Q_repair_quality_control`

- `PARP1` (Enhanced): `Nuclear bodies;Nucleoli;Nucleoli fibrillar center;Nucleoplasm`
- `PRKDC` (Enhanced): `Nucleoplasm`
- `XRCC6` (Enhanced): `Nucleoplasm`
- `ATG12` (Approved): `Nucleoplasm;Vesicles`
- `ATG13` (Supported): `Cytosol;Nucleoplasm;Plasma membrane`
- `ATG14` (Approved): `Plasma membrane;Vesicles`
- `ATG3` (Supported): `Cytosol;Plasma membrane`
- `ATG5` (Approved): `Basal body;Centrosome;Primary cilium transition zone;Vesicles`

### `S_support_context`

- `APOE` (Approved): `Predicted to be secreted;Vesicles`
- `AXL` (Supported): `Actin filaments;Plasma membrane;Vesicles`
- `CCL2` (Approved): `Golgi apparatus;Predicted to be secreted;Vesicles`
- `CLU` (Supported): `Cytosol;Predicted to be secreted`
- `CSF1` (Supported): `Nuclear bodies;Plasma membrane;Predicted to be secreted`
- `CSF1R` (Supported): `Plasma membrane;Vesicles`
- `CXCL8` (Approved): `Golgi apparatus;Predicted to be secreted`
- `GAS6` (Approved): `Centriolar satellite;Cytosol;Predicted to be secreted`

## claim boundary

This is a static subcellular localization blueprint. It supports the physical compartment map of Clarus-cell operators, but it is not a perturbational recurrence or dynamics proof.
